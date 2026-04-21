// rkYolov5s.cc �?RK3568 optimized zero-copy YOLOv5 inference engine
// Changes vs original RK3588 version:
//   1. Removed rknn_set_core_mask (RK3568 has single NPU)
//   2. RGA does BGR to RGB conversion during resize
//   3. Zero-copy: rknn_create_mem + rknn_set_io_mem + wrapbuffer_fd_t
//   4. Output read directly from output_mems[]->virt_addr (no rknn_outputs_get)

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "rknn_api.h"
#include "im2d.h"
#include "rga.h"
#include "postprocess.h"
#include "preprocess.h"
#include "rkYolov5s.hpp"

static std::string build_label_path_from_model(const std::string &model_path)
{
    size_t sep = model_path.find_last_of("/\\");
    if (sep == std::string::npos) {
        return "coco_80_labels_list.txt";
    }
    return model_path.substr(0, sep + 1) + "coco_80_labels_list.txt";
}

// ---------------------------------------------------------------------------
// File helpers (unchanged from original)
// ---------------------------------------------------------------------------

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data = NULL;
    if (!fp) return NULL;
    if (fseek(fp, ofst, SEEK_SET) != 0) {
        printf("blob seek failure.\n");
        return NULL;
    }
    data = (unsigned char *)malloc(sz);
    if (!data) {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    unsigned char *data = load_data(fp, 0, size);
    fclose(fp);
    *model_size = size;
    return data;
}

// ---------------------------------------------------------------------------
// rkYolov5s implementation
// ---------------------------------------------------------------------------

rkYolov5s::rkYolov5s(const std::string &model_path)
{
    this->ret = 0;
    this->model_path = model_path;
    this->model_data = nullptr;
    this->ctx = 0;
    memset(&this->io_num, 0, sizeof(this->io_num));
    this->input_attrs = nullptr;
    this->output_attrs = nullptr;
    this->channel = 0;
    this->width = 0;
    this->height = 0;
    this->nms_threshold = NMS_THRESH;
    this->box_conf_threshold = BOX_THRESH;
    memset(input_mems, 0, sizeof(input_mems));
    memset(output_mems, 0, sizeof(output_mems));
}

int rkYolov5s::init(rknn_context *ctx_in, bool share_weight)
{
    std::string label_path = build_label_path_from_model(model_path);
    if (setLabelNamePath(label_path.c_str()) == 0) {
        printf("Label path: %s\n", label_path.c_str());
    }

    printf("Loading model...\n");
    int model_data_size = 0;
    model_data = load_model(model_path.c_str(), &model_data_size);
    if (!model_data) return -1;

    // Init or duplicate context
    if (share_weight && ctx_in)
        ret = rknn_dup_context(ctx_in, &ctx);
    else
        ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);

    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    // *** Step 1: NO rknn_set_core_mask ***
    // RK3568 has a single NPU core. The Linux Galcore driver manages
    // the queue automatically. Calling rknn_set_core_mask would fail.

    // SDK version
    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        printf("rknn_query SDK_VERSION error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    // I/O count
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("rknn_query IN_OUT_NUM error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Input attributes
    input_attrs = (rknn_tensor_attr *)calloc(io_num.n_input, sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret < 0) { printf("rknn_query INPUT_ATTR error\n"); return -1; }
    }

    // Output attributes
    output_attrs = (rknn_tensor_attr *)calloc(io_num.n_output, sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret < 0) { printf("rknn_query OUTPUT_ATTR error\n"); return -1; }
    }

    // Extract model input dimensions
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        channel = input_attrs[0].dims[1];
        height  = input_attrs[0].dims[2];
        width   = input_attrs[0].dims[3];
    } else if (input_attrs[0].fmt == RKNN_TENSOR_NHWC) {
        height  = input_attrs[0].dims[1];
        width   = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    } else {
        printf("Unsupported tensor format: %d\n", input_attrs[0].fmt);
        return -1;
    }
    printf("model input: %dx%dx%d\n", width, height, channel);

    // Cache quantization params (avoids rebuilding vectors every frame)
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }

    // *** Step 3: Zero-copy memory setup ***

    // Input memory: UINT8 NHWC (NPU only supports NHWC in zero-copy mode)
    input_attrs[0].type = RKNN_TENSOR_UINT8;
    input_attrs[0].fmt  = RKNN_TENSOR_NHWC;
    input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);
    if (!input_mems[0]) {
        printf("rknn_create_mem for input failed\n");
        return -1;
    }

    ret = rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]);
    if (ret < 0) {
        printf("rknn_set_io_mem input error ret=%d\n", ret);
        return -1;
    }

    // Output memory: keep native INT8 type (post-process handles dequantization)
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        output_mems[i] = rknn_create_mem(ctx, output_attrs[i].size_with_stride);
        if (!output_mems[i]) {
            printf("rknn_create_mem for output[%d] failed\n", i);
            return -1;
        }
        ret = rknn_set_io_mem(ctx, output_mems[i], &output_attrs[i]);
        if (ret < 0) {
            printf("rknn_set_io_mem output[%d] error ret=%d\n", i, ret);
            return -1;
        }
    }

    printf("Zero-copy I/O memory bound (input fd=%d)\n", input_mems[0]->fd);
    return 0;
}

rknn_context *rkYolov5s::get_pctx()
{
    return &ctx;
}

int rkYolov5s::infer(void *frame_data, int img_w, int img_h, int src_format,
                     detect_result_group_t *det_group, float *out_scale_w, float *out_scale_h)
{
    // *** Step 2: RGA does format conversion + resize in one hardware pass ***
    ret = rga_resize_convert(frame_data, img_w, img_h, src_format,
                             input_mems[0]->fd, width, height, RK_FORMAT_RGB_888);
    if (ret != 0) {
        fprintf(stderr, "RGA resize+convert failed\n");
        return -1;
    }

    // *** Step 3 continued: NPU inference — zero memcpy ***
    ret = rknn_run(ctx, NULL);
    if (ret < 0) {
        fprintf(stderr, "rknn_run error ret=%d\n", ret);
        return -1;
    }

    // *** Step 5: Post-process directly from zero-copy output pointers ***
    float scale_w = (float)width  / img_w;
    float scale_h = (float)height / img_h;

    BOX_RECT pads;
    memset(&pads, 0, sizeof(pads));

    ret = post_process((int8_t *)output_mems[0]->virt_addr,
                       (int8_t *)output_mems[1]->virt_addr,
                       (int8_t *)output_mems[2]->virt_addr,
                       height, width,
                       box_conf_threshold, nms_threshold,
                       pads, scale_w, scale_h,
                       out_zps, out_scales,
                       det_group);
    if (ret != 0) {
        fprintf(stderr, "post_process failed, check label file path\n");
        return -1;
    }

    *out_scale_w = scale_w;
    *out_scale_h = scale_h;
    return 0;
}

rkYolov5s::~rkYolov5s()
{
    deinitPostProcess();

    // Destroy zero-copy memory
    if (ctx && input_mems[0])
        rknn_destroy_mem(ctx, input_mems[0]);
    uint32_t out_num = io_num.n_output;
    if (out_num > 3) out_num = 3;
    for (uint32_t i = 0; i < out_num; i++) {
        if (ctx && output_mems[i])
            rknn_destroy_mem(ctx, output_mems[i]);
    }

    if (ctx)
        rknn_destroy(ctx);

    if (model_data) free(model_data);
    if (input_attrs) free(input_attrs);
    if (output_attrs) free(output_attrs);
}



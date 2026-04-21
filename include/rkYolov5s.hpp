#ifndef RKYOLOV5S_H
#define RKYOLOV5S_H

#include <string>
#include <vector>
#include "rknn_api.h"
#include "rga.h"
#include "postprocess.h"

class rkYolov5s {
private:
    int ret;
    std::string model_path;
    unsigned char *model_data;

    rknn_context ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;

    // Zero-copy DMA memory
    rknn_tensor_mem *input_mems[1];
    rknn_tensor_mem *output_mems[3];  // YOLOv5 has 3 output heads

    int channel, width, height;  // model input dimensions
    float nms_threshold, box_conf_threshold;

    // Quantization params (cached at init, reused every infer)
    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;

public:
    rkYolov5s(const std::string &model_path);
    ~rkYolov5s();

    int init(rknn_context *ctx_in = nullptr, bool share_weight = false);
    rknn_context *get_pctx();

    // Zero-copy inference: raw frame data in (any RGA-supported format), detection results out
    // src_format: RK_FORMAT_BGR_888, RK_FORMAT_YUYV_422, RK_FORMAT_YCbCr_420_SP (NV12), etc.
    int infer(void *frame_data, int img_w, int img_h, int src_format,
              detect_result_group_t *det_group, float *scale_w, float *scale_h);
};

#endif

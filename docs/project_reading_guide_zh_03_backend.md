# 推理共享后段

这份文档讲 `rkYolov5s`、`preprocess`、`postprocess`。

它们合在一起，完成这条链路：

`原始帧 -> RGA 预处理 -> rknn_run() -> 后处理 -> detect_result_group_t`

## 1. `rkYolov5s` 的接口

先看头文件里的公共接口。

```cpp
public:
    rkYolov5s(const std::string &model_path);
    ~rkYolov5s();

    int init(rknn_context *ctx_in = nullptr, bool share_weight = false);
    rknn_context *get_pctx();

    int infer(void *frame_data, int img_w, int img_h, int src_format,
              detect_result_group_t *det_group, float *scale_w, float *scale_h);
```

这三件事分别是：

- `init()`：加载模型、创建 RKNN 上下文、绑定输入输出内存。
- `get_pctx()`：把内部 `rknn_context` 暴露出去，给池里其他模型复用。
- `infer()`：一帧数据进来，完成预处理、推理和后处理。

这个类不要只看成“一个模型包装器”。它实际上把模型文件、RKNN 上下文、输入输出内存和后处理状态绑成了一个可复用的执行单元；`rknnPool` 复制/共享的也是这个单元，而不是一张孤立的推理函数调用。

再往深一点看，`rkYolov5s` 把模型文件、RKNN 上下文、输入输出张量属性、量化参数和 zero-copy 内存都收在一起，是因为 RK3568 上的推理链路不是孤立的 `rknn_run()` 调用，而是要先经 RGA、再进 NPU、最后回到后处理和显示。

## 2. `rkYolov5s::init()`

初始化流程的第一步是确定标签文件路径，再加载模型，然后初始化或者复用上下文。

```cpp
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

    if (share_weight && ctx_in)
        ret = rknn_dup_context(ctx_in, &ctx);
    else
        ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
```

这段说明：

- 模型文件不是直接拿来用，还要先读成内存。
- `share_weight` 为真时，会复用第一个模型的上下文。
- 这个设计是给 `rknnPool` 用的，多个 worker 可以共享权重，减少重复初始化成本。

继续往下，`init()` 会查询 SDK 版本、输入输出数量和属性。

```cpp
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
```

这里的重点是：

- 输入输出张量信息不是硬编码的，要从模型里问出来。
- 后面的前处理和后处理都依赖这些属性。
- 输出的量化参数也会被缓存起来。

这些 `rknn_query()` 的结果不是调试信息，而是后面每一层的“运行时契约”。输入输出数量、tensor 的格式、尺寸和量化参数都缓存下来以后，`infer()` 才能知道该把图像变成什么形状，`post_process()` 才能知道该怎么把 int8 输出还原回可读坐标。

这里还要记住 RK3568 的约束：代码里明确不调用 `rknn_set_core_mask`，因为只有单 NPU，调度交给 Linux 侧驱动队列即可。`rknn_query` 取出来的输入输出属性不是调试信息，而是后面 `infer()` 和 `post_process()` 的运行时契约。

### zero-copy 绑定

`init()` 最关键的一段是把输入输出内存直接绑定到 RKNN。

注：这里申请的是按 stride 对齐后的 NPU IO 缓冲，不是简单的 `width * height * channel` 连续字节数。

```cpp
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
```

这表示：

- 输入帧经过 RGA 后，直接写进 NPU 可读的 DMA 内存。
- 输出结果也直接从 `virt_addr` 里读，不再做额外拷贝。

这是这个工程的性能核心之一。

这里的性能核心不只是“不拷贝”，而是“把 RGA 直接接到 RKNN 申请的 DMA 内存上”。输入帧先经过 RGA 变成模型要的 RGB888/NHWC，再直接写到 `input_mems[0]` 的 fd；输出结果也直接落在 `output_mems[i]->virt_addr`，后面不再多做一层搬运。对嵌入式板子来说，这种设计通常比微调算法更能省时间，因为少一次内存搬运就少一次带宽压力。

也就是说，RGA 负责把用户态输入变成 NPU 能吃的 DMA 内存，RKNN 只负责推理。这里不是单纯少了一次 copy，而是把“图像缩放和格式转换”与“模型执行”明确拆成两个硬件职责。

## 3. `rkYolov5s::infer()`

推理的顺序非常固定：

1. RGA 做缩放和颜色转换。
2. `rknn_run()` 执行推理。
3. `post_process()` 解码输出。

```cpp
int rkYolov5s::infer(void *frame_data, int img_w, int img_h, int src_format,
                     detect_result_group_t *det_group, float *out_scale_w, float *out_scale_h)
{
    ret = rga_resize_convert(frame_data, img_w, img_h, src_format,
                             input_mems[0]->fd, width, height, RK_FORMAT_RGB_888);
    if (ret != 0) {
        fprintf(stderr, "RGA resize+convert failed\n");
        return -1;
    }

    ret = rknn_run(ctx, NULL);
    if (ret < 0) {
        fprintf(stderr, "rknn_run error ret=%d\n", ret);
        return -1;
    }

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
```

这段的语义很清楚：

- `rga_resize_convert()` 负责把输入变成模型要的尺寸和格式。
- `rknn_run()` 只负责真正推理。
- `post_process()` 负责把三个输出头变成可读的检测结果。

最后会把缩放比例回填出去：

```cpp
    *out_scale_w = scale_w;
    *out_scale_h = scale_h;
    return 0;
}
```

这些比例后面会被显示路径和结果映射用到。

`infer()` 的三个步骤其实对应三种不同职责：`rga_resize_convert()` 负责把输入变成模型尺寸和格式，`rknn_run()` 负责纯推理，`post_process()` 负责把三个输出头解释成可读的检测框。`scale_w` 和 `scale_h` 不是多余数据，它们保证后面画框时仍然能回到原始图像坐标。

## 4. `preprocess`

预处理有两个接口：

```cpp
int rga_resize_convert(void *src_data, int src_w, int src_h, int src_fmt,
                       int dst_fd, int dst_w, int dst_h, int dst_fmt);

int rga_resize_convert_vaddr(void *src_data, int src_w, int src_h, int src_fmt,
                             void *dst_data, int dst_w, int dst_h, int dst_fmt);
```

前者是 zero-copy 版，目标是 `rknn_create_mem()` 返回的 DMA fd。
后者是 fallback 版，目标是普通虚拟地址。

前者走 fd 是为了和 RKNN 的 DMA 内存直连，后者走虚拟地址则更适合调试或没有 fd 的场景。两者的算法路径其实一样，只是目标内存的接入方式不同。

这也是为什么文档里要区分 fd 版和 vaddr 版：前者贴着主路径走，后者只在没有 fd 或调试时使用。

### zero-copy 版

注：目标宽度要先按 RGA 对齐要求补成 stride，否则后面的 DMA 访问步长会错。

```cpp
int rga_resize_convert(void *src_data, int src_w, int src_h, int src_fmt,
                       int dst_fd, int dst_w, int dst_h, int dst_fmt)
{
    rga_buffer_t src;
    rga_buffer_t dst;
    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));

    // Source: virtual address from user-space buffer
    src = wrapbuffer_virtualaddr(src_data, src_w, src_h, src_fmt);

    // Destination: DMA fd from rknn_create_mem (zero-copy to NPU)
    int wstride = dst_w + (RGA_ALIGN - dst_w % RGA_ALIGN) % RGA_ALIGN;
    int hstride = dst_h;
    dst = wrapbuffer_fd_t(dst_fd, dst_w, dst_h, wstride, hstride, dst_fmt);

    int ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret) {
        fprintf(stderr, "RGA imcheck error: %s\n", imStrError((IM_STATUS)ret));
        return -1;
    }

    IM_STATUS status = imresize(src, dst);
    if (status != IM_STATUS_SUCCESS) {
        fprintf(stderr, "RGA imresize error: %s\n", imStrError(status));
        return -1;
    }
    return 0;
}
```

你要抓的重点是：

- 源端是虚拟地址。
- 目标端是 fd。
- 这就是“源缓冲 -> NPU DMA 内存”的零拷贝路径。

`imcheck()` 的意义是提前把尺寸、格式和裁剪区间校验掉，避免把错误拖到 `imresize()` 里才发现。这个函数不是只帮你“转图”，它还在保护后面的推理输入不被错误尺寸污染。

### vaddr fallback 版

```cpp
int rga_resize_convert_vaddr(void *src_data, int src_w, int src_h, int src_fmt,
                             void *dst_data, int dst_w, int dst_h, int dst_fmt)
{
    rga_buffer_t src;
    rga_buffer_t dst;
    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));

    src = wrapbuffer_virtualaddr(src_data, src_w, src_h, src_fmt);
    dst = wrapbuffer_virtualaddr(dst_data, dst_w, dst_h, dst_fmt);
 
    int ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret) {
        fprintf(stderr, "RGA imcheck error: %s\n", imStrError((IM_STATUS)ret));
        return -1;
    }
    IM_STATUS status = imresize(src, dst);
    if (status != IM_STATUS_SUCCESS) {
        fprintf(stderr, "RGA imresize error: %s\n", imStrError(status));
        return -1;
    }
 
    return 0;
}
```

这条路只是为了对比理解：

- 一个写 fd，一个写普通内存。
- 逻辑一样，但性能和接入方式不同。

## 5. `postprocess`

后处理把三个输出头解码成最终检测结果。

先看结果结构和接口。

```cpp
typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct __detect_result_t
{
    char name[OBJ_NAME_MAX_SIZE];
    BOX_RECT box;
    float prop;
} detect_result_t;

typedef struct _detect_result_group_t
{
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, BOX_RECT pads, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);
```

这三个结构把“框、类名、概率”打包成了一个最小的输出协议。`BOX_RECT` 只管坐标，`detect_result_t` 把单个目标的名字和置信度挂上去，`detect_result_group_t` 则给整帧留一个容器。后面的函数都不是单独输出，而是围绕这个协议填数据。

### 标签初始化

```cpp
int setLabelNamePath(const char *path)
{
    if (path == nullptr || path[0] == '\0') {
        return -1;
    }
    snprintf(label_file_path, sizeof(label_file_path), "%s", path);
    return 0;
}

int readLines(const char *fileName, char *lines[], int max_line)
{
    FILE *file = fopen(fileName, "r");
    char *s = nullptr;
    int i = 0;
    int n = 0;

    if (file == NULL) { printf("Open %s fail!\n", fileName); return -1; }

    while ((s = readLine(file, s, &n)) != NULL) {
        lines[i++] = s;
        if (i >= max_line) break;
    }
    fclose(file);
    return i;
}

int loadLabelName(const char *locationFilename, char *label[])
{
    printf("loadLabelName %s\n", locationFilename);
    int count = readLines(locationFilename, label, OBJ_CLASS_NUM);
    if (count <= 0) {
        return -1;
    }
    for (int i = count; i < OBJ_CLASS_NUM; i++) {
        label[i] = nullptr;
    }
    return 0;
}
```

这里的作用是：

- 先把标签文件路径保存起来。
- 再把类别名逐行读进内存。
- `post_process()` 在第一次调用时只初始化一次。

`setLabelNamePath()` / `readLines()` / `loadLabelName()` 的职责分得很清楚：先保存路径，再逐行读文件，最后把标签缓存到全局数组。这样做的好处是，`post_process()` 不必每帧都去碰磁盘，而是在第一次调用时惰性初始化，后续直接复用。

### 单个输出头的解析

注：这里先把浮点阈值映射到 int8 量化域再比较，避免先把整张输出特征图反量化成 float。

```cpp
static int process(int8_t *input, int *anchor, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId,
                   float threshold, int32_t zp, float scale)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);

    for (int a = 0; a < 3; a++) {
        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                int8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= thres_i8) {
                    int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                    int8_t *in_ptr = input + offset;

                    float box_x = (deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                    float box_y = (deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                    float box_w = (deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                    float box_h = (deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    int8_t maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
                        int8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs) {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    if (maxClassProbs > thres_i8) {
                        objProbs.push_back(deqnt_affine_to_f32(maxClassProbs, zp, scale) *
                                           deqnt_affine_to_f32(box_confidence, zp, scale));
                        classId.push_back(maxClassId);
                        validCount++;
                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                    }
                }
            }
        }
    }
    return validCount;
}
```

这段负责：

- 遍历每个 grid。
- 按量化参数反解。
- 初筛出候选框。
- 把候选框、置信度和类别 id 收集起来。

`process()` 里的三层循环其实是在扫 YOLO 头的每个 anchor、每个 grid cell。阈值先通过 `qnt_f32_to_affine()` 转成 int8 域，再去和输出张量里的 `box_confidence` 比较，这样就不用先把整块 int8 张量转成浮点。满足阈值的候选框会先进入临时 vector，等三个输出头都扫完之后再统一做排序和 NMS。

### 三个输出头 + NMS

注：标签文件只在第一次进入 `post_process()` 时加载一次，后续帧直接复用内存中的结果。

```cpp
int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, BOX_RECT pads, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group)
{
    std::call_once(labels_init_once, [](){
        labels_init_ret = loadLabelName(label_file_path, labels);
    });
    if (labels_init_ret < 0) {
        return -1;
    }

    memset(group, 0, sizeof(detect_result_group_t));
 
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;

    // stride 8
    int stride0 = 8;
    int grid_h0 = model_in_h / stride0;
    int grid_w0 = model_in_w / stride0;

    int validCount0 = process(input0, (int *)anchor0, grid_h0, grid_w0, model_in_h, model_in_w,
                              stride0, filterBoxes, objProbs, classId, conf_threshold, qnt_zps[0], qnt_scales[0]);

    // stride 16
    int stride1 = 16;
    int grid_h1 = model_in_h / stride1;
    int grid_w1 = model_in_w / stride1;
    int validCount1 = process(input1, (int *)anchor1, grid_h1, grid_w1, model_in_h, model_in_w,
                              stride1, filterBoxes, objProbs, classId, conf_threshold, qnt_zps[1], qnt_scales[1]);

    // stride 32
    int stride2 = 32;
    int grid_h2 = model_in_h / stride2;
    int grid_w2 = model_in_w / stride2;
    int validCount2 = process(input2, (int *)anchor2, grid_h2, grid_w2, model_in_h, model_in_w,
                              stride2, filterBoxes, objProbs, classId, conf_threshold, qnt_zps[2], qnt_scales[2]);

    int validCount = validCount0 + validCount1 + validCount2;
    if (validCount <= 0) return 0;

    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i)
        indexArray.push_back(i);

    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classId), std::end(classId));
    for (auto c : class_set)
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);

    int last_count = 0;
    group->count = 0;
    for (int i = 0; i < validCount; ++i) {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) continue;
        int n = indexArray[i];
        float x1 = filterBoxes[n * 4 + 0] - pads.left;
        float y1 = filterBoxes[n * 4 + 1] - pads.top;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        group->results[last_count].box.left   = (int)(clamp(x1, 0, model_in_w) / scale_w);
        group->results[last_count].box.top    = (int)(clamp(y1, 0, model_in_h) / scale_h);
        group->results[last_count].box.right  = (int)(clamp(x2, 0, model_in_w) / scale_w);
        group->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);
        group->results[last_count].prop = obj_conf;
        const char *label = (id >= 0 && id < OBJ_CLASS_NUM && labels[id] != nullptr) ? labels[id] : "unknown";
        strncpy(group->results[last_count].name, label, OBJ_NAME_MAX_SIZE - 1);
        group->results[last_count].name[OBJ_NAME_MAX_SIZE - 1] = '\0';
        last_count++;
    }
    group->count = last_count;
    return 0;
}
```

三路输出都处理完后，代码会：

- 合并候选框。
- 按概率排序。
- 按类别做 NMS。
- 最终填充 `detect_result_group_t`。

你在读这个函数时要抓住：

- 它不是单纯“检测”，而是“解码 + 排序 + NMS + 结构化输出”。
- `group->count` 才是最后有效结果数量。
- `name` 来自标签文件，不在模型里硬编码。

`post_process()` 不是只做检测，它还负责把三个尺度的候选框合并、按置信度排序、按类别做 NMS，并把坐标从模型输入尺寸映射回原图。`pads` 用来扣掉 letterbox 带来的边缘填充，`scale_w/scale_h` 则负责把坐标按缩放比例还原。最终写入 `detect_result_group_t` 的不是“若干候选”，而是一个可以直接拿去显示的结果包。

`process()` 扫的是三路输出头的 anchor/grid，`nms()` 则在同类候选之间去重，最后 `detect_result_group_t` 才成为可以直接给显示层用的结果包。`setLabelNamePath()`、`readLines()` 和 `loadLabelName()` 则把标签文件路径、逐行读取和缓存分开，避免 `post_process()` 每帧都碰磁盘。

### `deinitPostProcess()`

`deinitPostProcess()` 是后处理模块的收尾清理函数。

```cpp
void deinitPostProcess()
{
    for (int i = 0; i < OBJ_CLASS_NUM; i++) {
        if (labels[i] != nullptr) {
            free(labels[i]);
            labels[i] = nullptr;
        }
    }
}
```

它的作用很明确：

- 释放 `labels` 里每个类别名的堆内存。
- 把指针清空，避免重复释放。
- 它是后处理模块的唯一显式清理入口。

你读这个函数时要把它和 `loadLabelName()` 一起看。

这个收尾函数只释放 `loadLabelName()` 动态分配出来的标签字符串，并把指针清空。因为标签文件是整个后处理模块共享的，所以不能在每次 `post_process()` 后都释放，只能在模块退出时统一收尾。

## 6. 读这份文档应该抓什么

- `rkYolov5s` 负责把模型和推理流程封装起来。
- `preprocess` 负责把输入帧变成模型能吃的格式。
- `postprocess` 负责把输出张量变成框和类别。
- `detect_result_group_t` 是结果最终载体。

如果你把这三块看懂，推理链路就完整了。

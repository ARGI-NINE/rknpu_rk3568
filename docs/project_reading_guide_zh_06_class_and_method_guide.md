# 类与方法总览

前面 01-05 已经把这个项目按调用链拆开了，但真正读实现时，最容易卡住的往往不是“流程”，而是“对象怎么持有资源、方法之间怎么交接所有权”。

这份文档换一个视角：不再按功能链路看，而是按类和方法看。重点放在 RK3568 / Linux 环境下每个对象负责哪一段生命周期、哪一段缓冲、哪一个 release contract。你把这份和前面的 02-04 对照着看，类名、方法名和调用点会更容易对上。

## 1. `rkYolov5s`

这个类不是“一个单纯的模型推理函数”，而是把模型文件、RKNN 上下文、输入输出张量属性、zero-copy 内存和后处理状态绑成一个可复用的执行单元。`rknnPool` 复制和共享的也是这个单元，而不是一张孤立的推理调用。

先看接口。

```cpp
class rkYolov5s {
private:
    int ret;
    std::string model_path;
    unsigned char *model_data;

    rknn_context ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;

    rknn_tensor_mem *input_mems[1];
    rknn_tensor_mem *output_mems[3];

    int channel, width, height;
    float nms_threshold, box_conf_threshold;

    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;

public:
    rkYolov5s(const std::string &model_path);
    ~rkYolov5s();

    int init(rknn_context *ctx_in = nullptr, bool share_weight = false);
    rknn_context *get_pctx();

    int infer(void *frame_data, int img_w, int img_h, int src_format,
              detect_result_group_t *det_group, float *scale_w, float *scale_h);
};
```

### 1.1 构造和析构

构造函数只做初始化，不碰硬件资源。

```cpp
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
```

这段的重点是：

- 这里没有创建 RKNN 上下文，也没有分配输入输出内存。
- `input_mems[0]` 和 `output_mems[0..2]` 只在 `init()` 成功后才有效。
- 阈值直接从宏读进来，说明后处理参数是类的固定运行时契约。

析构函数负责按逆序回收。

```cpp
rkYolov5s::~rkYolov5s()
{
    deinitPostProcess();

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
```

这里要抓住的顺序是：

1. 先销毁 zero-copy 内存。
2. 再销毁 RKNN 上下文。
3. 最后释放模型文件和属性数组。

这个顺序不是习惯问题，而是上下文、DMA 内存和后处理状态之间有依赖关系。先把 `rknn_destroy_mem()` 和 `rknn_destroy()` 做完，后面的 `free()` 才不会踩到悬空句柄。

### 1.2 `init()`

初始化流程先找标签文件，再读模型，再创建或者复用上下文，最后把输入输出内存绑到 RKNN。

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

这段说明两件事：

- `share_weight` 为真时，会复用第一个模型实例的上下文。
- `rknnPool` 正是依赖这个分支，让多个 worker 共享权重，但保留各自独立的执行状态。

后面的 `rknn_query()` 不是调试打印，而是运行时契约的来源。

```cpp
    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    input_attrs = (rknn_tensor_attr *)calloc(io_num.n_input, sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
    }

    output_attrs = (rknn_tensor_attr *)calloc(io_num.n_output, sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
    }
```

这里的重点是：

- 输入输出数量、tensor 格式和尺寸都不是硬编码的，而是从模型里问出来的。
- 这些属性会直接决定 `infer()` 里 RGA 怎么缩放、`post_process()` 怎么反量化。
- 这个工程在 RK3568 上不走 `rknn_set_core_mask()`，因为板端只有单 NPU，调度交给 Linux 驱动队列即可。

最后是 zero-copy 绑定。

```cpp
    input_attrs[0].type = RKNN_TENSOR_UINT8;
    input_attrs[0].fmt  = RKNN_TENSOR_NHWC;
    input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);
    ret = rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]);

    for (uint32_t i = 0; i < io_num.n_output; i++) {
        output_mems[i] = rknn_create_mem(ctx, output_attrs[i].size_with_stride);
        ret = rknn_set_io_mem(ctx, output_mems[i], &output_attrs[i]);
    }
```

这说明 `rkYolov5s` 的输入不是“复制一份图像再送 NPU”，而是先让 RGA 把图像直接写到 RKNN 申请的 DMA 内存里，再让 `rknn_run()` 直接吃这块内存。

### 1.3 `get_pctx()`

这个函数很短，但语义很重要。

```cpp
rknn_context *rkYolov5s::get_pctx()
{
    return &ctx;
}
```

它只是把内部上下文地址暴露出去，给 `rknnPool` 的后续实例做 `rknn_dup_context()` 用。也就是说，这里不是“把控制权交出去”，而是“把共享权重的入口暴露出去”。

`taskQueueSize()` 和 `inflightSize()` 也是同一类监控接口：前者看当前排队等待 worker 处理的任务数，后者看已经 put 进去但还没被 get 走的帧数。它们不改变调度，只是让你在日志里看清池子有没有积压。

### 1.4 `infer()`

`infer()` 把输入帧送进 RGA、NPU 和后处理，最后把检测结果和缩放比例一起返回。

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

这段代码的链路很清楚：

- `rga_resize_convert()` 负责格式转换和缩放。
- `rknn_run()` 负责推理。
- `post_process()` 负责把 int8 输出还原成可读检测框。

还有两个细节值得记住：

- `pads` 在这里清零，说明当前路径把 letterbox 的边距处理收进后处理里了。
- `out_scales` 和 `out_zps` 是在 `init()` 里缓存好的量化参数，`infer()` 每帧都复用，不重新拼临时向量。

### 1.5 关键不变量

- `input_mems[0]` 只承载 zero-copy 输入，格式固定为 `UINT8 + NHWC`。
- `output_mems[0..2]` 对应 YOLOv5 的三个输出头，后处理直接从 `virt_addr` 读。
- RK3568 这条路径不依赖 `rknn_set_core_mask()`。
- `init()` 失败后不要继续调用 `infer()`，因为输入输出属性和内存句柄还不完整。

## 2. `rknnPool`

这个模板类负责把“采集 / 解码 -> 推理 -> 显示”拆成可并发、可顺序消费的任务链。它最重要的事情不是调线程，而是把帧的所有权、顺序和结果缓存都管住。

先看契约结构。

```cpp
struct FrameTask {
    uint8_t seq;
    unsigned char* data;
    int width;
    int height;
    int format;
    size_t dataSize;
    uint64_t frame_id;
    long long src_ts_us;
    void (*release_fn)(unsigned char*, void*){nullptr};
    void* release_ctx{nullptr};
};

struct ResultSlot {
    std::atomic<bool> ready{false};
    detect_result_group_t det{};
    float scale_w{0.0f};
    float scale_h{0.0f};
    unsigned char* frame_data{nullptr};
    int frame_w{0};
    int frame_h{0};
    int frame_fmt{0};
    uint64_t frame_id{0};
    long long src_ts_us{0};
    void (*release_fn)(unsigned char*, void*){nullptr};
    void* release_ctx{nullptr};
};

enum class PutRejectReason {
    kNone = 0,
    kInvalidReleaseContract,
    kInflightLimit,
    kStopping,
};
```

这几种结构已经把整个契约说完了：

- `FrameTask` 是送入 worker 的输入。
- `ResultSlot` 是 worker 返回的结果缓存。
- `PutRejectReason` 说明 `put()` 为什么会拒绝接收帧。

### 2.1 构造、析构和 `init()`

构造函数只设置基本状态。

```cpp
rknnPool<Model>::rknnPool(const std::string& modelPath, int threadNum, int maxQueueSize)
    : threadNum_(threadNum), maxQueueSize_(maxQueueSize), modelPath_(modelPath),
      nextGetSeq_(0), inflight_(0), stop_(false) {
    if (maxQueueSize_ < 1) {
        maxQueueSize_ = 1;
    }
}
```

这说明两个不变量：

- 队列长度至少是 1，不能退化成“没有缓冲”。
- `nextGetSeq_` 从 0 开始，后面靠 `uint8_t` 自然回绕。

`init()` 先创建模型实例，再启动 worker 线程。

```cpp
int rknnPool<Model>::init() {
    for (int i = 0; i < threadNum_; i++)
        models_.push_back(std::make_shared<Model>(modelPath_.c_str()));

    for (int i = 0; i < threadNum_; i++) {
        int ret = models_[i]->init(models_[0]->get_pctx(), i != 0);
        if (ret != 0) {
            printf("rknnPool: model[%d] init failed (%d)\n", i, ret);
            return ret;
        }
    }

    for (int i = 0; i < threadNum_; i++)
        workers_.emplace_back(&rknnPool::workerFunc, this, i);
```

这段的语义很直接：

- 第一个模型完整初始化，加载权重并创建上下文。
- 后续模型通过 `dup_context` 共享权重，但各自保留独立执行状态。
- worker 线程只有在模型都准备好之后才启动。

这就是 `rknnPool` 为什么不是“线程池 + 任务队列”的普通写法，而是“模型实例池 + 结果序列号”的原因。

### 2.2 `put()`

`put()` 不是简单入队，它会先检查 release contract，再检查 in-flight 上限，然后才把帧的所有权转给池子。

```cpp
int rknnPool<Model>::put(unsigned char* frameData, int w, int h, int format, size_t dataSize,
                         uint64_t frameId, long long srcTsUs,
                         void (*release_fn)(unsigned char*, void*),
                         void* release_ctx,
                         PutRejectReason* outRejectReason) {
    if (!hasValidReleaseContract(release_fn, release_ctx)) {
        if (outRejectReason) {
            *outRejectReason = PutRejectReason::kInvalidReleaseContract;
        }
        return -1;
    }

    int inflight_now = inflight_.load(std::memory_order_relaxed);
    while (true) {
        if (inflight_now >= 255) {
            if (outRejectReason) {
                *outRejectReason = PutRejectReason::kInflightLimit;
            }
            releaseFrameData(frameData, release_fn, release_ctx);
            return -1;
        }
        if (inflight_.compare_exchange_weak(inflight_now, inflight_now + 1,
                                            std::memory_order_acq_rel,
                                            std::memory_order_relaxed)) {
            break;
        }
    }

    uint8_t seq = nextPutSeq_.fetch_add(1, std::memory_order_relaxed);
```

这里有三个特别关键的点：

- release contract 为空时，`put()` 在“所有权转移之前”就直接拒绝。
- `inflight_` 上限卡在 255，是为了和 `uint8_t seq` 的 256 槽环形结果表保持一致。
- 成功入队后，帧的归还不再靠调用者手动 `free()`，而是靠 `release_fn(frameData, release_ctx)`。

后面的入队部分会把序列号、尺寸、格式和回调一起塞进 `FrameTask`，再压进有界队列。如果池子已经停止，`put()` 会把帧释放掉并返回失败。

### 2.3 `get()` 和 `notifyGetters()`

`get()` 是单消费者、按序消费。

```cpp
int rknnPool<Model>::get(detect_result_group_t& det, float& scaleW, float& scaleH,
                         unsigned char** frameData, int* frameW, int* frameH, int* frameFmt,
                         uint64_t* frameId, long long* srcTsUs,
                         void (**frameReleaseFn)(unsigned char*, void*),
                         void** frameReleaseCtx) {
    {
        std::unique_lock<std::mutex> lock(resultMtx_);
        resultCv_.wait(lock, [this]{
            if (results_[nextGetSeq_].ready.load(std::memory_order_acquire)) {
                return true;
            }
            return getStopRequested_.load(std::memory_order_acquire);
        });
    }

    if (!results_[nextGetSeq_].ready.load(std::memory_order_acquire)) {
        return -1;
    }

    ResultSlot& slot = results_[nextGetSeq_];
    det    = slot.det;
    scaleW = slot.scale_w;
    scaleH = slot.scale_h;
    *frameData = slot.frame_data;
    *frameW    = slot.frame_w;
    *frameH    = slot.frame_h;
    *frameFmt  = slot.frame_fmt;
```

这段代码说明 `get()` 的行为是顺序严格的：它只等 `nextGetSeq_` 对应的那个槽位。

所以即使多个 worker 同时完成推理，消费端看到的结果顺序也不会乱。

取出结果后，`get()` 会清空槽位并推进序号。

```cpp
    slot.frame_data = nullptr;
    slot.frame_id = 0;
    slot.src_ts_us = 0;
    slot.release_fn = nullptr;
    slot.release_ctx = nullptr;
    slot.ready.store(false, std::memory_order_relaxed);

    nextGetSeq_++;
    inflight_.fetch_sub(1, std::memory_order_relaxed);
    return 0;
}

void rknnPool<Model>::notifyGetters() {
    getStopRequested_.store(true, std::memory_order_release);
    resultCv_.notify_all();
}
```

`notifyGetters()` 的用途很单一：在停止阶段把阻塞中的 `get()` 唤醒，让消费线程能干净退出。

### 2.4 私有帮助函数和不变量

`releaseFrameData()` 是整个类里最硬的一条契约。

```cpp
static void releaseFrameData(unsigned char* data,
                             void (*release_fn)(unsigned char*, void*),
                             void* release_ctx) {
    if (!data) {
        return;
    }
    if (release_fn && release_ctx) {
        release_fn(data, release_ctx);
        return;
    }
    fprintf(stderr,
            "rknnPool: fatal release contract missing for data=%p (fn=%s, ctx=%s)\n",
            static_cast<void*>(data),
            release_fn ? "set" : "null",
            release_ctx ? "set" : "null");
    fflush(stderr);
    abort();
}
```

这意味着：

- 只要数据已经进入 `rknnPool`，就必须能按回调归还。
- 任何把 `release_fn` 或 `release_ctx` 弄丢的地方，都会被当成致命错误。

`workerFunc()` 也有一个很重要的行为：如果帧尺寸或 dataSize 不合法，它不会把坏数据送进 `infer()`，而是直接写一个空结果槽并推进序号。

这保证了“队列里不能有坏帧把后面的序号卡住”。

最后记住这几个不变量：

- `results_` 是 256 槽环形缓冲，`uint8_t` 序号自然回绕。
- `nextGetSeq_` 只由 `get()` 的单消费者线程访问。
- `inflight_` 统计的是“已经 put 进来、但还没有 get 走”的帧数。
- 成功 `put()` 后，调用者就不能再把那块帧内存当成自己的所有物。

## 3. `FrameCopyPool`

这个类只在 `main.cc` 里出现，是一个很局部的缓冲池。它的作用不是“提供通用内存池”，而是把采集线程临时拷贝出来的帧复用掉，减少频繁 `malloc/free`。

### 3.1 接口

```cpp
class FrameCopyPool {
public:
    FrameCopyPool(size_t buffer_size, int prealloc_count, int max_cached_count);
    ~FrameCopyPool();

    size_t bufferSize() const;
    unsigned char* acquire();
    void release(unsigned char* buffer);

private:
    size_t buffer_size_;
    int max_cached_count_;
    std::mutex mtx_;
    std::vector<unsigned char*> free_list_;
};
```

### 3.2 构造和析构

构造函数会先把参数限制在合理范围内，再按 `prealloc_count` 预分配一批缓冲。

```cpp
FrameCopyPool(size_t buffer_size, int prealloc_count, int max_cached_count)
    : buffer_size_(buffer_size), max_cached_count_(max_cached_count) {
    if (max_cached_count_ < 1) {
        max_cached_count_ = 1;
    }
    if (prealloc_count < 0) {
        prealloc_count = 0;
    }
    if (prealloc_count > max_cached_count_) {
        prealloc_count = max_cached_count_;
    }
```

这表示它不是无限扩张的池，而是一个有上限的缓存器。

析构函数只释放当前还留在空闲链表里的缓冲。

```cpp
~FrameCopyPool() {
    for (unsigned char* buf : free_list_) {
        free(buf);
    }
    free_list_.clear();
}
```

如果外部还握着缓冲却没有按回调归还，这个类并不会替你回收。它只回收自己还认得的那部分缓存。

`bufferSize()` 则是给上层做尺寸检查用的：`copy_frame_and_submit_to_pool()` 会先拿它确认一帧数据能不能装进池里的单块缓冲，再决定要不要申请并复制。

### 3.3 `bufferSize()` / `acquire()` / `release()`

`acquire()` 和 `release()` 都是很标准的“拿一个、还一个”接口，但它有一个硬约束：`release()` 不能把缓存无限塞回去。

```cpp
unsigned char* acquire() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!free_list_.empty()) {
        unsigned char* buf = free_list_.back();
        free_list_.pop_back();
        return buf;
    }
    return nullptr;
}

void release(unsigned char* buffer) {
    if (!buffer) {
        return;
    }
    std::lock_guard<std::mutex> lock(mtx_);
    if ((int)free_list_.size() >= max_cached_count_) {
        fprintf(stderr,
                "[FATAL] FrameCopyPool release overflow: free_list=%zu max_cached=%d buffer=%p\n",
                free_list_.size(),
                max_cached_count_,
                static_cast<void*>(buffer));
        fflush(stderr);
        abort();
    }
    free_list_.push_back(buffer);
}
```

这里最重要的不是省了多少 malloc，而是把“谁拥有这块帧内存”说清楚了。

### 3.4 相关辅助函数

这个池子还配了两个回调桥接函数：

```cpp
static void release_pooled_buffer(unsigned char* data, void* ctx) {
    if (!data) {
        return;
    }
    if (!ctx) {
        fprintf(stderr,
                "[FATAL] release_pooled_buffer missing pool context for data=%p\n",
                static_cast<void*>(data));
        fflush(stderr);
        abort();
    }
    FrameCopyPool* pool = static_cast<FrameCopyPool*>(ctx);
    pool->release(data);
}
```

它们的作用是把“回收缓冲”变成一个统一的 release contract。这样 `rknnPool`、采集线程和显示线程看到的都不是裸 `free()`，而是带上下文的归还动作。

`compute_frame_size_bytes()` 则负责按像素格式算缓冲大小，常见分支是 `YUYV_422`、`YCbCr_420_SP`、`BGR_888` 和 `RGB_888`。

### 3.5 关键不变量

- `free_list_` 由 `mtx_` 保护。
- `max_cached_count_` 至少是 1。
- `release()` 超过上限会直接 `abort()`，这是故意的防御性设计。
- 这个类只负责缓存，不负责替调用者决定生命周期。

## 4. `V4L2Capture`

这个类把 Linux V4L2 的设备打开、格式协商、MMAP 缓冲、DQBUF / QBUF 契约封装成了一个对象。它不是简单的“打开摄像头”，而是把驱动缓冲的生命周期单独接管。

`getWidth()` / `getHeight()` / `getPixelFormat()` 返回的是协商后的实际参数，不是构造时的请求值；`isOpen()` 只是快速判断 fd 是否有效，便于外层在收尾时决定要不要再做一次释放。

### 4.1 接口

```cpp
class V4L2Capture {
private:
    std::string device_;
    int fd_;
    int reqWidth_;
    int reqHeight_;
    int reqBufferCount_;
    int width_;
    int height_;
    int pixfmt_;

    struct MmapBuffer {
        void *start;
        size_t length;
    };
    std::vector<MmapBuffer> buffers_;
    int bufferCount_;
    int currentIdx_;

    int initFormat();
    int initMmap();

public:
    V4L2Capture(const std::string &device, int width, int height,
                int bufferCount = V4L2_DEFAULT_BUF_COUNT);
    ~V4L2Capture();

    int open();
    int startStream();
    int captureFrame(void **data, int *size);
    void releaseFrame();
    int stopStream();
    void close();
};
```

### 4.2 构造和析构

构造函数把请求尺寸和 buffer 数量记下来，并保证缓冲数至少是 2。

```cpp
V4L2Capture::V4L2Capture(const std::string &device, int width, int height, int bufferCount)
    : device_(device), fd_(-1), reqWidth_(width), reqHeight_(height), reqBufferCount_(bufferCount),
      width_(0), height_(0), pixfmt_(0), bufferCount_(0), currentIdx_(-1)
{
    if (reqBufferCount_ < 2) {
        reqBufferCount_ = 2;
    }
}
```

这说明采集不是单缓冲模式。至少 2 个 buffer 才能把“驱动填充”和“用户态处理”分开。

析构函数很简单：只要设备已经打开，就先停流，再关闭。

```cpp
V4L2Capture::~V4L2Capture()
{
    if (fd_ >= 0) {
        stopStream();
        close();
    }
}
```

### 4.3 `open()` / `initFormat()` / `initMmap()`

`open()` 先打开设备，再检查能力，然后走格式和缓冲初始化。

```cpp
int V4L2Capture::open()
{
    fd_ = ::open(device_.c_str(), O_RDWR);
    if (fd_ < 0) {
        perror("V4L2: open device");
        return -1;
    }

    struct v4l2_capability cap;
    if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
        perror("V4L2: QUERYCAP");
        ::close(fd_); fd_ = -1;
        return -1;
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf(stderr, "V4L2: device does not support capture\n");
        ::close(fd_); fd_ = -1;
        return -1;
    }
```

`initFormat()` 会先试 YUYV，再退回 NV12。

```cpp
int V4L2Capture::initFormat()
{
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width  = reqWidth_;
    fmt.fmt.pix.height = reqHeight_;
    fmt.fmt.pix.field  = V4L2_FIELD_NONE;

    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12;
        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
            perror("V4L2: S_FMT (YUYV/NV12 both failed)");
            return -1;
        }
    }
```

这和 RK3568 上的 RGA 路径是配套的：YUYV 往往更容易直接交给硬件做颜色转换和缩放，NV12 也能走通。

`initMmap()` 则负责把驱动缓冲映射到用户态。

```cpp
int V4L2Capture::initMmap()
{
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count  = (uint32_t)reqBufferCount_;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
        perror("V4L2: REQBUFS");
        return -1;
    }

    bufferCount_ = (int)req.count;
    if (bufferCount_ < 2) {
        fprintf(stderr, "V4L2: insufficient buffers (%d)\n", bufferCount_);
        return -1;
    }
```

### 4.4 `startStream()` / `captureFrame()` / `releaseFrame()` / `stopStream()` / `close()`

`startStream()` 先把所有缓冲队列回驱动，再 `STREAMON`。

`captureFrame()` 用 `VIDIOC_DQBUF` 取出当前 buffer，并把 `buffers_[currentIdx_].start` 和 `buf.bytesused` 返回给调用者。

```cpp
int V4L2Capture::captureFrame(void **data, int *size)
{
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
        perror("V4L2: DQBUF");
        return -1;
    }

    currentIdx_ = buf.index;
    *data = buffers_[currentIdx_].start;
    *size = (int)buf.bytesused;
    return 0;
}
```

这个接口最容易误读的地方是：返回来的不是“你自己分配的内存”，而是驱动映射出来的 buffer。它只能借用，不能 `free()`。

所以 `releaseFrame()` 要做的事情就是把当前 buffer 再 `QBUF` 回去。

`stopStream()` 对应 `STREAMOFF`，`close()` 则负责 unmap 每个 buffer，清空列表并关闭 fd。

### 4.5 关键不变量

- `captureFrame()` 返回的指针在 `releaseFrame()` 前有效。
- `currentIdx_` 记录的是当前 dequeued 的 buffer，必须按原样 requeue。
- `initMmap()` 至少要求 2 个 buffer。
- `startStream()` 必须在 `open()` 之后调用。

## 5. `MppDecoder`

这个类把“文件解复用 + 码流过滤 + MPP 解码”包在一起，最终输出 NV12 帧。它的对象边界很清楚：输入是文件路径，输出是可直接送 RGA 的图像平面。

`getWidth()` / `getHeight()` 返回当前解码流的实际尺寸。它们会随着 `info_change` 更新，所以不是一开始打开文件时的静态值。

### 5.1 接口

```cpp
class MppDecoder {
public:
    MppDecoder();
    ~MppDecoder();

    int  open(const char *filepath);
    void close();
    int  readFrame(unsigned char **frame_data, int *width, int *height);

    int  getWidth()  const { return video_width_; }
    int  getHeight() const { return video_height_; }

private:
    AVFormatContext *fmt_ctx_;
    AVBSFContext    *bsf_ctx_;
    AVPacket        *pkt_;
    AVPacket        *filtered_pkt_;
    int              video_stream_idx_;

    MppCtx          mpp_ctx_;
    MppApi         *mpi_;
    MppBufferGroup  frm_grp_;

    int    video_width_;
    int    video_height_;
    unsigned char *frame_buf_;
    size_t         frame_buf_size_;

    bool eos_sent_;
    bool eos_got_;

    int feedPacket(void *data, int size, int64_t pts);
    int sendEos();
};
```

### 5.2 构造和析构

构造函数把所有句柄都置空，把 EOS 状态置为 false。

```cpp
MppDecoder::MppDecoder()
    : fmt_ctx_(nullptr), bsf_ctx_(nullptr), pkt_(nullptr), filtered_pkt_(nullptr),
      video_stream_idx_(-1),
      mpp_ctx_(nullptr), mpi_(nullptr), frm_grp_(nullptr),
      video_width_(0), video_height_(0),
      frame_buf_(nullptr), frame_buf_size_(0),
    eos_sent_(false), eos_got_(false) {}
```

这说明它的状态机是显式的：没有 open，就没有 decoder；没有 eos，循环就不会提前退出。

析构函数只是调用 `close()`，保证资源释放路径只有一条。

### 5.3 `open()`

`open()` 的流程是：FFmpeg 打开容器 -> 找视频流 -> 判断 codec -> 配 BSF -> 初始化 MPP -> 申请外部 buffer group。

```cpp
int MppDecoder::open(const char *filepath) {
    int ret;

    ret = avformat_open_input(&fmt_ctx_, filepath, nullptr, nullptr);
    if (ret < 0) {
        printf("MppDecoder error: avformat_open_input failed ret=%d\n", ret);
        return -1;
    }
    ret = avformat_find_stream_info(fmt_ctx_, nullptr);
    if (ret < 0) {
        printf("MppDecoder error: avformat_find_stream_info failed ret=%d\n", ret);
        close();
        return -1;
    }
```

然后它会根据 codec 选择 MPP coding type。H.264 和 H.265 需要 BSF，VP9 不需要。

```cpp
    switch (codecpar->codec_id) {
    case AV_CODEC_ID_H264:
        coding   = MPP_VIDEO_CodingAVC;
        bsf_name = "h264_mp4toannexb";
        break;
    case AV_CODEC_ID_HEVC:
        coding   = MPP_VIDEO_CodingHEVC;
        bsf_name = "hevc_mp4toannexb";
        break;
    case AV_CODEC_ID_VP9:
        coding   = MPP_VIDEO_CodingVP9;
        bsf_name = nullptr;
        break;
    default:
        printf("MppDecoder error: unsupported codec_id=%d\n", codecpar->codec_id);
        close();
        return -1;
    }
```

最后这段特别关键：

```cpp
    ret = mpp_buffer_group_get_internal(&frm_grp_, MPP_BUFFER_TYPE_ION);
    if (ret != MPP_OK) {
        printf("MppDecoder error: mpp_buffer_group_get_internal failed ret=%d\n", ret);
        close();
        return -1;
    }
    mpi_->control(mpp_ctx_, MPP_DEC_SET_EXT_BUF_GROUP, frm_grp_);
```

在 RK3568 上，这一步不是可选项。没有外部 buffer group，`decode_put_packet()` 很容易卡在 buffer 不够用的状态里。

### 5.4 `readFrame()`

`readFrame()` 是一个典型的“先取解码结果，没结果再喂压缩包”的循环。

注：这里在 `info_change` 后必须重建 buffer group 并显式回告 `MPP_DEC_SET_INFO_CHANGE_READY`，后续帧才会继续输出。

```cpp
int MppDecoder::readFrame(unsigned char **frame_data, int *width, int *height) {
    if (eos_got_) return -1;

    while (true) {
        // ---- 1. Try to get a decoded frame from MPP ----
        MppFrame frame = nullptr;
        MPP_RET ret = mpi_->decode_get_frame(mpp_ctx_, &frame);

        if (ret == MPP_OK && frame) {
            // Handle resolution change (info_change)
            if (mpp_frame_get_info_change(frame)) {
                RK_U32 w = mpp_frame_get_width(frame);
                RK_U32 h = mpp_frame_get_height(frame);
                printf("[INFO] MPP info_change %ux%u\n", w, h);

                video_width_  = (int)w;
                video_height_ = (int)h;

                if (frm_grp_) {
                    mpp_buffer_group_put(frm_grp_);
                    frm_grp_ = nullptr;
                }
                mpp_buffer_group_get_internal(&frm_grp_, MPP_BUFFER_TYPE_ION);
                mpi_->control(mpp_ctx_, MPP_DEC_SET_EXT_BUF_GROUP, frm_grp_);
                mpi_->control(mpp_ctx_, MPP_DEC_SET_INFO_CHANGE_READY, nullptr);

                mpp_frame_deinit(&frame);
                continue;
            }

            // End of stream
            if (mpp_frame_get_eos(frame)) {
                mpp_frame_deinit(&frame);
                eos_got_ = true;
                return -1;
            }

            // Skip error / discard frames
            if (mpp_frame_get_discard(frame) || mpp_frame_get_errinfo(frame)) {
                mpp_frame_deinit(&frame);
                continue;
            }
```

这段说明 `MppDecoder` 不是单纯“吐一帧就走”，它还会处理三种状态：

- 分辨率变化（`info_change`）。
- 结束标记（`eos`）。
- 错误帧或丢弃帧。

真正输出给上层的是一块被整理过 stride 的 NV12 缓冲。

```cpp
            MppBuffer buf = mpp_frame_get_buffer(frame);
            RK_U32 w          = mpp_frame_get_width(frame);
            RK_U32 h          = mpp_frame_get_height(frame);
            RK_U32 hor_stride = mpp_frame_get_hor_stride(frame);
            RK_U32 ver_stride = mpp_frame_get_ver_stride(frame);

            size_t needed = (size_t)w * h * 3 / 2;
            if (needed > frame_buf_size_) {
                free(frame_buf_);
                frame_buf_ = (unsigned char *)malloc(needed);
                frame_buf_size_ = needed;
            }

            unsigned char *src = (unsigned char *)mpp_buffer_get_ptr(buf);
            for (RK_U32 row = 0; row < h; row++)
                memcpy(frame_buf_ + row * w, src + row * hor_stride, w);

            unsigned char *uv_src = src + hor_stride * ver_stride;
            unsigned char *uv_dst = frame_buf_ + w * h;
            for (RK_U32 row = 0; row < h / 2; row++)
                memcpy(uv_dst + row * w, uv_src + row * hor_stride, w);
```

这里的关键点是：

- MPP 输出里的 stride 不一定等于图像宽度。
- 这个类把 padding 去掉后再给上层，所以上层拿到的是连续 NV12。
- `frame_buf_` 是复用的，调用者不能自己释放。

### 5.5 `close()` / `feedPacket()` / `sendEos()`

`close()` 按相反顺序销毁所有句柄，并把 EOS 状态清零。

`feedPacket()` 会在 `decode_put_packet()` 失败时尝试先 `decode_get_frame()`，必要时重新分配 `frm_grp_`，然后短暂 sleep 再重试。

`sendEos()` 则只是往 decoder 里塞一个空的 EOS packet，通知解码器进入 drain 阶段。

### 5.6 关键不变量

- `readFrame()` 返回的 NV12 数据有效期到下一次 `readFrame()` 或 `close()`。
- `eos_sent_` 表示已经向 MPP 发送了 EOF；`eos_got_` 表示真正把尾帧都读干净了。
- 只支持 H.264 / H.265 / VP9 这几类输入。
- `info_change` 时必须更新 buffer group 和宽高，否则后续帧会错位。

## 6. `DrmDisplay`

这个类直接走 DRM/KMS，不走 GUI 栈。它的职责不是“画窗口”，而是管理双缓冲、页翻转和显示恢复。

`getWidth()` / `getHeight()` 返回当前 mode 的显示尺寸，`getStride()` 返回 back buffer 的像素行跨度，`getBackBuffer()` 则给出当前应该画图的那块内存。

### 6.1 接口

```cpp
class DrmDisplay {
public:
    DrmDisplay();
    ~DrmDisplay();

    int  init();
    void deinit();

    uint32_t* getBackBuffer();
    int  getWidth();
    int  getHeight();
    int  getStride();
    int  flip();
    void clearBackBuffer(uint32_t color);

private:
    int drm_fd_;
    uint32_t conn_id_;
    uint32_t crtc_id_;
    uint32_t fb_id_[2];
    uint32_t bo_handle_[2];
    uint32_t pitch_[2];
    uint32_t size_[2];
    uint32_t* map_[2];
    int width_;
    int height_;
    int front_;

    drmModeCrtcPtr saved_crtc_;
    drmModeModeInfo mode_;
};
```

### 6.2 构造和析构

构造函数把所有句柄清零，`front_` 从 0 开始。

```cpp
DrmDisplay::DrmDisplay()
    : drm_fd_(-1), conn_id_(0), crtc_id_(0),
      width_(0), height_(0), front_(0),
      saved_crtc_(nullptr)
{
    for (int i = 0; i < 2; i++) {
        fb_id_[i]    = 0;
        bo_handle_[i]= 0;
        pitch_[i]    = 0;
        size_[i]     = 0;
        map_[i]      = nullptr;
    }
}
```

析构函数只做一件事：调用 `deinit()`。

### 6.3 `init()` / `deinit()`

`init()` 从 `/dev/dri/card0` 打开 DRM 设备，找连接的 connector，选 preferred mode 或第一个 mode，再找一个可用的 CRTC，然后创建两个 dumb buffer。

```cpp
int DrmDisplay::init()
{
    const char* dev = "/dev/dri/card0";
    drm_fd_ = open(dev, O_RDWR | O_CLOEXEC);
    if (drm_fd_ < 0) {
        printf("DrmDisplay: cannot open %s: %s\n", dev, strerror(errno));
        return -1;
    }

    drmModeRes* res = drmModeGetResources(drm_fd_);
    if (!res) {
        printf("DrmDisplay: drmModeGetResources failed\n");
        close(drm_fd_);
        drm_fd_ = -1;
        return -1;
    }
```

这一步背后的意义是：

- 显示不靠桌面环境，而是直接占用 KMS 管线。
- 这就是为什么这套工程在板端能少一层图形栈开销。

然后它会把两个 dumb buffer mmap 到用户态，设置初始 CRTC，并把其中一个 buffer 设成前台显示。

`deinit()` 会先恢复原来的 CRTC，再把 mmap、fb 和 dumb buffer 按逆序释放掉。

### 6.4 `getBackBuffer()` / `getStride()` / `flip()` / `clearBackBuffer()`

`getBackBuffer()` 返回当前背板对应的映射地址。

`getStride()` 返回的是“每行多少个像素”，不是字节数，因为这里的 framebuffer 是 32bpp。

```cpp
int DrmDisplay::getStride()
{
    int back = 1 - front_;
    return (int)(pitch_[back] / 4);
}
```

`flip()` 会把 back buffer 送给 page flip，然后等 VBLANK 事件。

```cpp
int DrmDisplay::flip()
{
    int back = 1 - front_;
    int ret = drmModePageFlip(drm_fd_, crtc_id_, fb_id_[back],
                              DRM_MODE_PAGE_FLIP_EVENT, this);
    if (ret != 0) {
        printf("DrmDisplay: drmModePageFlip failed: %s\n", strerror(errno));
        return -1;
    }
```

这说明显示是“翻页显示”，不是把画面直接写到前台正在显示的 buffer 上。

`clearBackBuffer()` 则只是把背板全部填成同一个颜色，适合做初始化或者清屏。

### 6.5 关键不变量

- `front_` 只表示当前前台 buffer 的索引，背板永远是 `1 - front_`。
- `saved_crtc_` 会在 `deinit()` 里恢复，避免程序退出后把屏幕留在错误的 mode 上。
- `pitch_[]` 是字节步长，所以 `getStride()` 要除以 4。
- `flip()` 依赖 page flip 事件，不是简单地改一个标志位。

### 6.6 相关绘图辅助函数

`drm_draw_rect()`、`drm_draw_fill_rect()`、`drm_draw_char()` 和 `drm_draw_string()` 不是类成员，但它们和 `DrmDisplay` 是一组使用习惯：先拿 back buffer，再画框和文字，最后 `flip()`。

如果你在调试显示链路，最容易出问题的通常不是这些画图函数本身，而是 `fb_w`、`fb_h` 和 `getStride()` 没对齐。

## 7. 读这份时最该记住的三条线

- `rkYolov5s` 负责单帧推理，不负责线程调度。
- `rknnPool` 负责帧的所有权、顺序和结果缓存，不直接做图像算法。
- `V4L2Capture`、`MppDecoder` 和 `DrmDisplay` 都是在 Linux 内核接口上管理缓冲，用户态拿到的是“借用权”，不是“随便 free 的内存”。

如果你把这三条线记住，再回头看前面的 02-04，就能把类名、方法名和整条链路上的资源边界对应起来。

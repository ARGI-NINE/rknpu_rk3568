# 主循环与任务池

这份文档只讲一件事：`main()` 如何把整个工程串起来，以及 `rknnPool` 如何把“采集 -> 推理 -> 显示”拆成可并发、可顺序消费的任务链。

你应该先看这份，因为它决定了后面所有模块的调用方向。

## 1. 入口先做什么

`main()` 先处理参数，再重置全局状态，然后创建推理池、显示对象和采集/渲染线程。

```cpp
int main(int argc, char** argv)
{
    if (argc < 3 || argc > 4) {
        printf("Usage: %s <model_path> <camera_device_or_video> [thread_num]\n", argv[0]);
        printf("  thread_num: number of inference worker threads (default: 1)\n");
        return -1;
    }

    const char* model_path = argv[1];
    const char* input_source = argv[2];

    int threadNum = 1;
    if (argc >= 4) {
        threadNum = atoi(argv[3]);
        if (threadNum < 1) {
            threadNum = 1;
        }
        if (threadNum > kMaxWorkerThreads) {
            threadNum = kMaxWorkerThreads;
        }
    }
    printf("Inference threads: %d (cap=%d)\n", threadNum, kMaxWorkerThreads);
```

这段的作用很直接。

- 第 1 步，限制参数数量，保证启动方式稳定。
- 第 2 步，确定模型路径和输入源路径。
- 第 3 步，把推理线程数限制在上限内。

你看这段时要抓住一个点：`threadNum` 不是“随便开多少线程都行”，它被 `kMaxWorkerThreads` 卡住了。

这组参数其实已经把程序的运行模型暴露出来了：第一个参数是模型，第二个参数是输入源，第三个参数才是并行度。它不是一个可以无限放大的开关，因为每多一个 worker，就意味着多一份模型上下文、多一部分同步开销和更多缓存压力。把它上限化是为了让这份 demo 在不同板子上都保持可预测的资源占用，而不是把板子直接打满。

## 2. 全局状态和池对象初始化

`main()` 接着重置全局状态，创建 `rknnPool<rkYolov5s>`，然后调用 `pool.init()`。

```cpp
    g_render_stop.store(false, std::memory_order_relaxed);
    g_capture_stop.store(false, std::memory_order_relaxed);
    g_frame_id_gen.store(0, std::memory_order_relaxed);
    g_pool_hit_count.store(0, std::memory_order_relaxed);
    g_pool_exhaust_drop_count.store(0, std::memory_order_relaxed);
    g_inflight_reject_count.store(0, std::memory_order_relaxed);
    g_render_drop_count.store(0, std::memory_order_relaxed);

    rknnPool<rkYolov5s> pool(model_path, threadNum, kPoolTaskQueueMaxSize);
    if (pool.init() != 0) {
        printf("rknnPool init failed!\n");
        return -1;
    }
```

这里的语义是：

- `g_render_stop` 和 `g_capture_stop` 控制两个后台线程的退出。
- `g_frame_id_gen` 给每帧分配递增编号。
- `pool` 是整个推理链路的核心队列。
- `pool.init()` 成功之后，worker 才会启动。

你可以把这一步理解成“先把比赛场地、裁判和工作线程都搭好，再开始收帧”。

这一段的语义是“先把场地搭好，再让线程开工”。`g_render_stop`、`g_capture_stop`、`g_frame_id_gen` 和各种统计计数器都属于运行时状态，重新运行之前必须归零，否则后面的延迟统计和退出逻辑会被上一轮污染。`pool.init()` 也不是简单返回个成功值就结束，它代表模型实例、上下文和 worker 线程都已经准备完毕，后续采集线程才可以往里塞帧。

### `now_wall_time_us()`

这个函数给每一帧打微秒级墙钟时间戳。

```cpp
static long long now_wall_time_us() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1000000LL + tv.tv_usec;
}
```

它的作用是：

- 统一所有帧的时间来源。
- 方便后面算延迟和 FPS。
- 给 `frame_id` 之外再补一个真实时间戳。

这里还有一个很重要的区分：它拿到的是墙钟时间，不是单调时钟。系统时间如果被校准，这个值会跳；但这份代码不是拿它做严格的时长累加，而是拿它给帧、日志和调试输出统一标记，所以这个选择是合理的。你可以把它理解成“这帧什么时候进入流水线”的人类可读时间点。

### `compute_frame_size_bytes()`

这个函数按像素格式计算帧复制大小。

```cpp
static size_t compute_frame_size_bytes(int width, int height, int format) {
    if (width <= 0 || height <= 0) {
        return 0U;
    }
    size_t wh = (size_t)width * (size_t)height;
    if (format == RK_FORMAT_YUYV_422) {
        return wh * 2U;
    }
    if (format == RK_FORMAT_YCbCr_420_SP) {
        return wh * 3U / 2U;
    }
    if (format == RK_FORMAT_BGR_888 || format == RK_FORMAT_RGB_888) {
        return wh * 3U;
    }
    return wh * 3U;
}
```

它的作用是：

- 决定 `FrameCopyPool` 每块 buffer 该多大。
- 决定 `copy_frame_and_submit_to_pool()` 里要拷贝多少字节。
- 避免格式和大小不匹配导致越界。

这里不是在算驱动的 stride，而是在算逻辑像素数据的真实字节数。`YUYV` 是 4:2:2 packed 格式，每像素 2 字节；`NV12` 是 4:2:0 semi-planar，每像素 1.5 字节；`RGB/BGR` 每像素 3 字节。把这个逻辑集中在一个函数里，后面无论是摄像头路径还是视频解码路径，都只需要先统一格式，再按同一套规则申请和校验缓冲大小。以后如果要加新的像素格式，也应该先补这里，再补调用者。

## 3. 显示线程先起

显示对象初始化完成后，`main()` 启动渲染线程。

```cpp
    DrmDisplay drm_display;
    bool drm_ok = true;

    std::thread render_th;
    std::thread capture_th;
    bool render_started = false;
    bool capture_started = false;
    bool v4l2_opened = false;
    bool v4l2_stream_started = false;

    V4L2Capture* v4l2 = nullptr;
    MppDecoder* mpp_dec = nullptr;

    auto cleanup_and_return = [&](int rc) -> int {
        g_capture_stop.store(true, std::memory_order_relaxed);
        pool.notifyGetters();
        if (capture_started && capture_th.joinable()) {
            capture_th.join();
        }
 
        if (render_started) {
            {
                std::lock_guard<std::mutex> lock(g_render_mtx);
                g_render_stop.store(true, std::memory_order_relaxed);
            }
            g_render_cv.notify_one();
            if (render_th.joinable()) {
                render_th.join();
            }
        }
 
        if (drm_ok) {
            drm_display.deinit();
        }
        if (v4l2) {
            if (v4l2_stream_started) {
                v4l2->stopStream();
            }
            if (v4l2_opened) {
                v4l2->close();
            }
            delete v4l2;
            v4l2 = nullptr;
        }
        if (mpp_dec) {
            mpp_dec->close();
            delete mpp_dec;
            mpp_dec = nullptr;
        }
        return rc;
    };

    if (drm_display.init() != 0) {
        printf("Error: DRM display init failed\n");
        return cleanup_and_return(-1);
    }

    render_th = std::thread(render_thread_func, &drm_display);
    render_started = true;
```

这段最重要的点有两个：

- `render_thread_func()` 是独立消费线程，不在主线程里直接画图。
- `cleanup_and_return()` 是统一收尾入口，任何一步失败都从这里返回。

你读这段时要抓住“资源生命周期”这个概念。这个工程不是单函数 demo，而是多个对象一起活、一起退。

上面的完整 `cleanup_and_return()` 已经包含这段收尾逻辑。

这段代码的意思很直接：先停采集，再唤醒主线程和渲染线程，最后按相反顺序回收 DRM、V4L2 和 MPP 资源。这样做是为了保证后台线程退出时不会还在访问已经释放的 buffer 或 fd。

### `render_thread_func()`

渲染线程负责从队列里取帧、画图、释放帧。

```cpp
static void render_thread_func(DrmDisplay* drm)
{
    while (true) {
        RenderItem item;
        {
            std::unique_lock<std::mutex> lock(g_render_mtx);
            g_render_cv.wait(lock, [] { return !g_render_queue.empty() || g_render_stop; });
            if (g_render_stop && g_render_queue.empty()) {
                break;
            }
            item = g_render_queue.front();
            g_render_queue.pop();
        }

        render_display(*drm, item);
        release_frame_buffer(item.data, item.release_fn, item.release_ctx);
    }
}
```

它的作用是：

- 独立消费 `g_render_queue`。
- 每次只处理一帧。
- 处理完立即归还帧缓冲。

这就是为什么显示侧和采集/推理侧可以并行。

这段代码的关键不是“在显示”，而是“把显示从主链路里拆出去”。它先在条件变量上睡眠，只有队列里真的有东西或者收到停止信号才醒；拿到任务后立刻释放锁，再去画图和翻页，所以生产者不会因为绘图慢而被卡住。最后立刻调用 `release_frame_buffer()`，说明渲染线程是这块帧数据的最后一个使用者，画完就必须按契约归还，不能继续持有。

这里还有一个很重要的区分：它记录的是墙钟时间，不是单调时钟。系统时间如果被校准，这个值会跳；但这份代码不是拿它做严格的时长累加，而是拿它给帧、日志和调试输出统一标记，所以这个选择是合理的。你可以把它理解成“这帧什么时候进入流水线”的人类可读时间点。

## 4. 输入源分流

输入源有两条路：

- 摄像头，走 `V4L2Capture`
- 视频文件，走 `MppDecoder`

```cpp
    bool use_v4l2 = is_camera_input_source(input_source);
    int frame_w = 0;
    int frame_h = 0;
    int v4l2_pixfmt = 0;

    if (use_v4l2) {
        char devPath[64];
        if (strlen(input_source) == 1) {
            snprintf(devPath, sizeof(devPath), "/dev/video%c", input_source[0]);
        } else {
            snprintf(devPath, sizeof(devPath), "%s", input_source);
        }

        v4l2 = new V4L2Capture(devPath, kDefaultCaptureWidth, kDefaultCaptureHeight,
                               kV4l2BufferCount);
        if (v4l2->open() != 0) {
            printf("V4L2: failed to open %s\n", devPath);
            return cleanup_and_return(-1);
        }
        v4l2_opened = true;
        if (v4l2->startStream() != 0) {
            printf("V4L2: failed to start stream\n");
            return cleanup_and_return(-1);
        }
        v4l2_stream_started = true;

        frame_w = v4l2->getWidth();
        frame_h = v4l2->getHeight();
        v4l2_pixfmt = v4l2->getPixelFormat();
        printf("V4L2: %s %dx%d pixfmt=0x%08x\n", devPath, frame_w, frame_h, v4l2_pixfmt);
    } else {
        mpp_dec = new MppDecoder();
        if (mpp_dec->open(input_source) != 0) {
            return cleanup_and_return(-1);
        }
        frame_w = mpp_dec->getWidth();
        frame_h = mpp_dec->getHeight();
    }
```

这里要理解两点。

1. `is_camera_input_source()` 是唯一分流点。
2. 摄像头和视频文件都要先拿到宽高，后面才能算复制大小和模型输入大小。

## 5. 帧复制池

这个工程没有把原始帧直接扔进推理池，而是先放进 `FrameCopyPool`。

```cpp
    size_t frame_copy_size = compute_frame_size_bytes(frame_w, frame_h, capture_format);
    if (frame_copy_size == 0U) {
        printf("Invalid frame size: w=%d h=%d fmt=%d\n", frame_w, frame_h, capture_format);
        return cleanup_and_return(-1);
    }
    int frame_pool_prealloc = kPoolTaskQueueMaxSize + kRenderQueueMaxSize + threadNum + 2;
    int frame_pool_max_cached = frame_pool_prealloc + 4;
    FrameCopyPool frame_pool(frame_copy_size, frame_pool_prealloc, frame_pool_max_cached);
```

`FrameCopyPool` 的作用是：

- 复用 `malloc` 出来的缓冲区。
- 避免每帧临时分配和释放。
- 为后面 `rknnPool::put()` 提供稳定的所有权转移。

这一步很关键，因为后面你会看到帧不是直接 `free()` 的，而是通过回调归还。

这里再补一层所有权约束：`FrameCopyPool` 负责把借来的驱动缓冲变成 owned copy，`rknnPool::put()` 接管这块内存后，释放动作必须沿着 `release_fn` / `release_ctx` 回到 `FrameCopyPool`。这样做的目的是让 V4L2 和 MPP 的借用型缓冲不跨越错误的生命周期。

`is_camera_input_source()` 只是把“这是设备还是文件”这件事简单分流。摄像头模式下，输入源通常是一个数字或者设备路径，所以代码会把单个数字转换成 `/dev/videoN`；视频文件模式下则直接交给 `MppDecoder`。这一层的目的不是做字符串判断，而是提前决定后面应该走“驱动采集”还是“文件解码”两条完全不同的资源路径。

### `release_frame_buffer()` 和 `release_pooled_buffer()`

这两个函数是本工程所有帧释放的统一出口。

```cpp
static void release_frame_buffer(unsigned char* data,
                                 FrameReleaseFn release_fn,
                                 void* release_ctx) {
    if (!data) {
        return;
    }
    if (release_fn && release_ctx) {
        release_fn(data, release_ctx);
        return;
    }
    fprintf(stderr,
            "[FATAL] frame release contract missing for data=%p (fn=%s, ctx=%s)\n",
            static_cast<void*>(data),
            release_fn ? "set" : "null",
            release_ctx ? "set" : "null");
    fflush(stderr);
    abort();
}
```

它的作用是：

- 把“回收帧”统一成一个出口。
- 如果没有回调和上下文，就直接视为契约错误。
- 任何帧都必须沿着这条路径归还，不能随便 `free()`。

`release_pooled_buffer()` 是给 `FrameCopyPool` 用的回调封装，`copy_frame_and_submit_to_pool()` 会把它作为 `release_fn` 传进 `rknnPool::put()`。

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

它的作用是：

- 把 `FrameCopyPool` 和 `release_frame_buffer()` 接起来。
- 让 `rknnPool::put()` 拿到的帧，在后面能安全回收到池里。

这一步是整条链路里最重要的所有权转换。采集线程拿到的原始帧通常只是“借来用一下”，摄像头驱动缓冲和解码输出缓冲的生命周期都不归你管，所以这里必须先申请一块真正属于自己的缓冲，再把数据拷进去。这样做虽然多了一次 memcpy，但换来的是明确的生命周期：采集侧可以马上把源缓冲还回去，推理侧则持有一块稳定内存直到渲染结束。`frame_id`、`src_ts_us`、`width/height/format` 和 release 契约一起传下去，保证后面的每一层都不需要再猜这帧从哪来、什么时候来的、怎么还。

这两个函数的本质不是“释放内存”，而是“把资源沿着原路送回去”。`release_frame_buffer()` 只负责分派，不关心这块数据来自 V4L2、MPP、malloc 还是池；`release_pooled_buffer()` 则把真正的归还动作落回 `FrameCopyPool::release()`。如果释放契约缺失，程序直接 `abort()`，不是因为作者爱崩，而是因为这已经属于所有权被破坏，继续跑只会把问题变成更难查的内存损坏。

### `copy_frame_and_submit_to_pool()`

注：如果 `put()` 因释放契约非法而拒绝接管，这块 `owned_buf` 的所有权还没转移出去，所以这里必须立刻归还给 `FrameCopyPool`。

```cpp
static int copy_frame_and_submit_to_pool(const void* src_data,
                                 size_t data_size,
                                 int width,
                                 int height,
                                 int format,
                                 uint64_t frame_id,
                                 long long src_ts_us,
                                 rknnPool<rkYolov5s>* pool,
                                 FrameCopyPool* frame_pool)
{
    if (!src_data || data_size == 0) {
        return -1;
    }

    if (!frame_pool || data_size > frame_pool->bufferSize()) {
        g_pool_exhaust_drop_count.fetch_add(1, std::memory_order_relaxed);
        return 1;
    }

    unsigned char* owned_buf = frame_pool->acquire();
    if (!owned_buf) {
        g_pool_exhaust_drop_count.fetch_add(1, std::memory_order_relaxed);
        return 1;
    }
    g_pool_hit_count.fetch_add(1, std::memory_order_relaxed);

    FrameReleaseFn release_fn = release_pooled_buffer;
    void* release_ctx = frame_pool;

    memcpy(owned_buf, src_data, data_size);
    PutRejectReason reject_reason = PutRejectReason::kNone;
    if (pool->put(owned_buf, width, height, format, data_size,
                  frame_id, src_ts_us, release_fn, release_ctx, &reject_reason) != 0) {
        if (reject_reason == PutRejectReason::kInflightLimit) {
            g_inflight_reject_count.fetch_add(1, std::memory_order_relaxed);
        } else if (reject_reason == PutRejectReason::kInvalidReleaseContract) {
            // put() rejects this reason before ownership transfer; caller must reclaim.
            release_frame_buffer(owned_buf, release_fn, release_ctx);
        }
        return 1;
    }
 
    return 0;
}
```

这段的作用是：

- 从 `FrameCopyPool` 申请一个可复用缓冲。
- 把采集到的原始数据 `memcpy` 进去。
- 再把这块缓冲和释放契约一起交给 `rknnPool::put()`。

它是“采集线程 -> 推理池”的桥。

这里做的是“输入异构 -> 内部同构”的归一化。V4L2 路径拿到的是驱动缓冲，`captureFrame()` 返回之后要尽快把它还回去，所以代码先复制再 `releaseFrame()`；MPP 路径拿到的是解码后的 NV12，也要先复制成池里的稳定缓冲，再把解码器继续往前推。无论源头是摄像头还是视频文件，到了池里以后它们就长得一样了：都有尺寸、格式、帧号、输入时间戳和回收契约。循环结束时设置 `g_capture_stop` 并 `notifyGetters()`，是为了把阻塞在 `pool.get()` 的消费者叫醒，避免退出时主线程卡死。

这一步是整条链路里最重要的所有权转换。采集线程拿到的原始帧通常只是“借来用一下”，摄像头驱动缓冲和解码输出缓冲的生命周期都不归你管，所以这里必须先申请一块真正属于自己的缓冲，再把数据拷进去。这样做虽然多了一次 memcpy，但换来的是明确的生命周期：采集侧可以马上把源缓冲还回去，推理侧则持有一块稳定内存直到渲染结束。`frame_id`、`src_ts_us`、`width/height/format` 和 release 契约一起传下去，保证后面的每一层都不需要再猜这帧从哪来、什么时候来的、怎么还。

补全这里的拒绝分支后，逻辑其实是在区分三种情况：`FrameCopyPool` 没有可用 buffer、池容量不够容纳当前帧，或者 `rknnPool::put()` 因为释放契约不合法而拒绝接管。前两种会记丢帧计数，后者则必须在所有权还没转移之前把 `owned_buf` 归还给池，否则就会把一块本该复用的缓冲变成悬空内存。

## 6. 采集线程启动与主循环

采集线程启动后，主线程开始从池里拿结果。

```cpp
    capture_th = std::thread(capture_thread_func, v4l2, mpp_dec, use_v4l2,
                             frame_w, frame_h, v4l2_pixfmt, &pool, &frame_pool);
    capture_started = true;

    while (true) {
        LoopControlCode ctrl = process_one_pool_result(GetProcessMode::kMainLoop);
        if (ctrl == LoopControlCode::kBreak) {
            break;
        }
        if (ctrl == LoopControlCode::kContinue) {
            continue;
        }
    }
```

这就是整个主循环的核心。

- `capture_thread_func()` 负责生产帧。
- `process_one_pool_result()` 负责消费帧。
- 两者通过 `rknnPool` 连接。

你可以把它记成“一个线程采，一个线程算，主线程收结果”。

### `capture_thread_func()`

这个线程负责把输入源变成统一的帧任务。

注：V4L2 分支会先按像素格式重算 `data_size` 和 `rga_fmt`，因为送进池里的字节数必须和原始帧布局完全一致。

```cpp
static void capture_thread_func(V4L2Capture* v4l2, MppDecoder* mpp_dec,
                                bool use_v4l2, int frame_w, int frame_h,
                                int v4l2_pixfmt,
                                rknnPool<rkYolov5s>* pool,
                                FrameCopyPool* frame_pool)
{
    while (!g_capture_stop.load(std::memory_order_relaxed)) {
        int cap_w = frame_w;
        int cap_h = frame_h;

        if (use_v4l2) {
            void* raw = nullptr;
            int raw_sz = 0;
            if (v4l2->captureFrame(&raw, &raw_sz) != 0) {
                printf("V4L2: captureFrame failed\n");
                break;
            }
            (void)raw_sz;
 
            int rga_fmt = RK_FORMAT_BGR_888;
            size_t data_size = (size_t)cap_w * cap_h * 3;
 
            if (v4l2_pixfmt == 0x56595559 /* V4L2_PIX_FMT_YUYV */) {
                rga_fmt = RK_FORMAT_YUYV_422;
                data_size = (size_t)cap_w * cap_h * 2;
            } else if (v4l2_pixfmt == 0x3231564E /* V4L2_PIX_FMT_NV12 */) {
                rga_fmt = RK_FORMAT_YCbCr_420_SP;
                data_size = (size_t)cap_w * cap_h * 3 / 2;
            }
            int submit_ret = copy_frame_and_submit_to_pool(raw, data_size, cap_w, cap_h,
                                                   rga_fmt, frame_id, src_ts_us, pool, frame_pool);
            v4l2->releaseFrame();
 
            if (submit_ret < 0) {
                break;
            }
        } else {
            unsigned char* nv12_data = nullptr;
            int dec_w = 0;
            int dec_h = 0;
            if (mpp_dec->readFrame(&nv12_data, &dec_w, &dec_h) != 0) {
                break;
            }
 
            long long src_ts_us = now_wall_time_us();
            uint64_t frame_id = g_frame_id_gen.fetch_add(1, std::memory_order_relaxed);
            cap_w = dec_w;
            cap_h = dec_h;
            size_t nv12_size = (size_t)cap_w * cap_h * 3 / 2;
            if (copy_frame_and_submit_to_pool(nv12_data, nv12_size, cap_w, cap_h,
                                      RK_FORMAT_YCbCr_420_SP, frame_id, src_ts_us,
                                      pool, frame_pool) < 0) {
                break;
            }
        }
    }

    g_capture_stop.store(true, std::memory_order_relaxed);
    if (pool) {
        pool->notifyGetters();
    }
}
```

注：线程退出前主动 `notifyGetters()`，是为了把阻塞在 `pool.get()` 的消费者唤醒，避免收尾阶段卡死。

它的作用是：

- V4L2 分支采摄像头。
- MPP 分支读视频文件。
- 统一把数据送进 `copy_frame_and_submit_to_pool()`。
- 出错或退出时唤醒主线程。

`enqueue_render_or_release()` 体现的是这个工程对实时性的取舍：能显示的帧就入队，不能显示的帧就立刻释放；队列满了就丢最旧帧，而不是最晚来的那帧。这样做是为了避免“越算越慢、越慢越旧”，也就是现场预览里最糟糕的那种延迟堆积。`handle_get_success()` 和 `process_one_pool_result()` 则是两个本地小助手：前者只管“拿到结果以后怎么交给渲染”，后者只管“从池里拿结果时怎么把底层返回码翻译成主循环能理解的状态”。这两层拆开后，主循环就不会被一堆 if/else 淹没。

上面的完整 `capture_thread_func()` 已经包含这段退出逻辑。

这段不是普通的“收尾”，而是告诉所有等待中的消费者：输入源已经结束，不要继续阻塞在取帧上。这样主线程后面的 drain loop 才能把池里的最后几帧消费干净。

## 7. 结果处理和退出

主线程拿到推理结果后，会通过 `handle_get_success()` 和 `enqueue_render_or_release()` 送进渲染队列。

### `enqueue_render_or_release()`

这个函数负责把推理结果送到渲染队列，或者在条件不满足时直接释放。

```cpp
static void enqueue_render_or_release(bool drm_ok,
                                      unsigned char* frame_data,
                                      int fw,
                                      int fh,
                                      int ff,
                                      const detect_result_group_t& det,
                                      FrameReleaseFn release_fn,
                                      void* release_ctx)
{
    if (!(drm_ok && frame_data)) {
        g_render_drop_count.fetch_add(1, std::memory_order_relaxed);
        release_frame_buffer(frame_data, release_fn, release_ctx);
        return;
    }

    RenderItem item;
    item.data = frame_data;
    item.width = fw;
    item.height = fh;
    item.format = ff;
    item.det = det;
    item.release_fn = release_fn;
    item.release_ctx = release_ctx;

    {
        std::lock_guard<std::mutex> lock(g_render_mtx);
        while ((int)g_render_queue.size() >= g_render_queue_max_size) {
            RenderItem old = g_render_queue.front();
            g_render_queue.pop();
            g_render_drop_count.fetch_add(1, std::memory_order_relaxed);
            release_frame_buffer(old.data, old.release_fn, old.release_ctx);
        }
        g_render_queue.push(item);
    }
    g_render_cv.notify_one();
}
```

它的作用是：

- `drm_ok` 或 `frame_data` 不满足时，直接释放。
- 队列满时丢最旧的帧。
- 把可以显示的帧交给渲染线程。

```cpp
    auto handle_get_success = [&](const detect_result_group_t& det,
                                  unsigned char* frame_data,
                                  int fw,
                                  int fh,
                                  int ff,
                                  FrameReleaseFn frame_release_fn,
                                  void* frame_release_ctx,
                                  bool enqueue_first) {
        if (enqueue_first) {
            enqueue_render_or_release(drm_ok, frame_data, fw, fh, ff, det,
                                      frame_release_fn, frame_release_ctx);
            frames++;
            log_fps_every_60_frames();
            return;
        }

        frames++;
        log_fps_every_60_frames();
        enqueue_render_or_release(drm_ok, frame_data, fw, fh, ff, det,
                                  frame_release_fn, frame_release_ctx);
    };
```

这段说明一个细节：

- `drain` 阶段和 `main loop` 阶段，入队顺序不是完全一样的。
- 但本质都是把已经算好的结果交给渲染线程。

然后 `process_one_pool_result()` 把 `pool.get()` 的结果桥接出去。

```cpp
    auto process_one_pool_result = [&](GetProcessMode mode) -> LoopControlCode {
        detect_result_group_t det;
        float scale_w_unused = 0.0f;
        float scale_h_unused = 0.0f;
        unsigned char* frame_data = nullptr;
        int fw = 0;
        int fh = 0;
        int ff = 0;
        uint64_t frame_id = 0;
        long long src_ts_us = 0;
        FrameReleaseFn frame_release_fn = nullptr;
        void* frame_release_ctx = nullptr;

        int ret = pool.get(det, scale_w_unused, scale_h_unused,
                           &frame_data, &fw, &fh, &ff, &frame_id, &src_ts_us,
                           &frame_release_fn, &frame_release_ctx);
        if (ret != 0) {
            if (mode == GetProcessMode::kMainLoop &&
                !g_capture_stop.load(std::memory_order_relaxed)) {
                return LoopControlCode::kContinue;
            }
            return LoopControlCode::kBreak;
        }

        bool enqueue_first = (mode == GetProcessMode::kDrainLoop);
        handle_get_success(det, frame_data, fw, fh, ff,
                           frame_release_fn, frame_release_ctx,
                           enqueue_first);
        return LoopControlCode::kHandled;
    };
```

退出时，`cleanup_and_return()` 会统一停采集、停渲染、关 DRM、关输入设备。

`enqueue_first` 不是为了玩顺序切换，而是为了区分“正常消费”和“退出收尾”两种场景。收尾时先把结果送进渲染队列，可以最大化保证已经算出来的帧不会因为主循环结束而丢掉；正常场景下先做统计再入队，则更容易让 FPS 计数贴近推理节奏。

## 8. 主链路应该怎么读

如果你只想先读懂主循环，按这个顺序看：

1. `main()`
2. `capture_thread_func()`
3. `process_one_pool_result()`
4. `handle_get_success()`
5. `enqueue_render_or_release()`
6. `render_thread_func()`
7. `cleanup_and_return()`

然后再看 `rknnPool`，因为主循环和池是绑在一起的。

## 9. `rknnPool` 的数据契约

`rknnPool` 不是“简单线程池”，它同时管任务队列、顺序、结果槽和帧所有权。

```cpp
struct FrameTask {
    uint8_t seq;
    unsigned char* data;  // raw frame data, caller-allocated, ownership transferred
    int width;
    int height;
    int format;           // RGA source format (RK_FORMAT_BGR_888, RK_FORMAT_YUYV_422, etc.)
    size_t dataSize;      // actual byte count of data
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
```

这两个结构把整个推理链路的“输入”和“输出”讲清楚了。

- `FrameTask` 是给 worker 的。
- `ResultSlot` 是 worker 写回来的。
- `frame_data` 不会被 worker 立刻释放，而是继续交给后面的显示路径。

这一节要先把“谁拥有帧、谁负责释放、谁负责顺序”三件事固定下来。`FrameTask` 保存的是输入端的所有信息，`ResultSlot` 保存的是输出端要回传给主线程的一切。特别是 `release_fn/release_ctx`，它们不是附加字段，而是所有权的一部分：没有这两个字段，任务就不完整。

## 10. `rknnPool::init()`

池初始化时，先创建模型实例，再启动 worker。

```cpp
template <typename Model>
int rknnPool<Model>::init() {
    try {
        for (int i = 0; i < threadNum_; i++)
            models_.push_back(std::make_shared<Model>(modelPath_.c_str()));
    } catch (const std::bad_alloc& e) {
        printf("rknnPool: out of memory: %s\n", e.what());
        return -1;
    }

    for (int i = 0; i < threadNum_; i++) {
        int ret = models_[i]->init(models_[0]->get_pctx(), i != 0);
        if (ret != 0) {
            printf("rknnPool: model[%d] init failed (%d)\n", i, ret);
            return ret;
        }
    }

    for (int i = 0; i < threadNum_; i++)
        workers_.emplace_back(&rknnPool::workerFunc, this, i);

    printf("rknnPool: %d workers ready, queue_limit=%d\n", threadNum_, maxQueueSize_);
    return 0;
}
```

要点：

- 第一个模型完整初始化。
- 后面的模型通过 `get_pctx()` 复用权重上下文。
- worker 数量等于推理线程数。

`rknnPool::init()` 先创建模型实例，再逐个初始化 worker，顺序是刻意设计的。第一个模型完整加载权重，其余 worker 通过 `rknn_dup_context()` 复用上下文，所以初始化成本主要集中在一次。等所有模型都准备好以后再启动 worker，避免线程一启动就抢到半初始化对象。

## 11. `rknnPool::put()`

`put()` 的作用是把一帧变成任务，然后放入有界队列。

```cpp
template <typename Model>
int rknnPool<Model>::put(unsigned char* frameData, int w, int h, int format, size_t dataSize,
                         uint64_t frameId, long long srcTsUs,
                         void (*release_fn)(unsigned char*, void*),
                         void* release_ctx,
                         PutRejectReason* outRejectReason) {
    if (outRejectReason) {
        *outRejectReason = PutRejectReason::kNone;
    }

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
        if (inflight_.compare_exchange_weak(
                inflight_now,
                inflight_now + 1,
                std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
            break;
        }
    }
```

这段先做两层保护：

- 释放契约必须完整。
- `inflight_` 不能超过上限。

然后分配序号、封装任务、阻塞入队。

```cpp
    uint8_t seq = nextPutSeq_.fetch_add(1, std::memory_order_relaxed);
    results_[seq].ready.store(false, std::memory_order_relaxed);

    FrameTask task;
    task.seq      = seq;
    task.data     = frameData;
    task.width    = w;
    task.height   = h;
    task.format   = format;
    task.dataSize = dataSize;
    task.frame_id = frameId;
    task.src_ts_us = srcTsUs;
    task.release_fn = release_fn;
    task.release_ctx = release_ctx;

    {
        std::unique_lock<std::mutex> lock(queueMtx_);
        queueNotFull_.wait(lock, [this]{ return (int)taskQueue_.size() < maxQueueSize_ || stop_; });
        if (stop_) {
            if (outRejectReason) {
                *outRejectReason = PutRejectReason::kStopping;
            }
            releaseFrameData(frameData, release_fn, release_ctx);
            inflight_.fetch_sub(1, std::memory_order_relaxed);
            return -1;
        }
        taskQueue_.push(task);
    }
    queueNotEmpty_.notify_one();
    return 0;
}
```

读 `put()` 时你要理解：

- 这里已经发生所有权转移。
- 后面不能再把 `frameData` 当成自己的。
- 如果入队失败，就必须按契约回收。

`put()` 的重点不是单纯入队，而是先做契约检查，再做背压控制，再才是封装任务。`hasValidReleaseContract()` 确认这帧能被正确回收；`inflight_` 限制的是“还没被消费完的任务数”，它防止上游无限制塞帧导致延迟和内存膨胀。`compare_exchange_weak()` 是原子自增，说明这里连并发下的计数都不想靠锁。

## 12. `rknnPool::workerFunc()`

worker 先检查坏帧，再调用模型推理，然后把结果写回环形槽。

```cpp
            const bool invalidDimensions = (task.width <= 0 || task.height <= 0);
            const size_t minDataSize = invalidDimensions
                ? 0U
                : static_cast<size_t>(task.width) * static_cast<size_t>(task.height);
            if (invalidDimensions || task.dataSize < minDataSize) {
                releaseFrameData(task.data, task.release_fn, task.release_ctx);
                task.data = nullptr;

                ResultSlot& slot = results_[task.seq];
                slot.det = detect_result_group_t{};
                slot.scale_w = 0.0f;
                slot.scale_h = 0.0f;
                slot.frame_data = nullptr;
                slot.frame_w = task.width;
                slot.frame_h = task.height;
                slot.frame_fmt = task.format;
                slot.frame_id = task.frame_id;
                slot.src_ts_us = task.src_ts_us;
                slot.release_fn = nullptr;
                slot.release_ctx = nullptr;
                slot.ready.store(true, std::memory_order_release);
                resultCv_.notify_one();
                continue;
            }

            detect_result_group_t det;
            float sw, sh;
            models_[id]->infer(task.data, task.width, task.height, task.format, &det, &sw, &sh);

            ResultSlot& slot = results_[task.seq];
            slot.det = det;
            slot.scale_w = sw;
            slot.scale_h = sh;
            slot.frame_data = task.data;
            slot.frame_w = task.width;
            slot.frame_h = task.height;
            slot.frame_fmt = task.format;
            slot.frame_id = task.frame_id;
            slot.src_ts_us = task.src_ts_us;
            slot.release_fn = task.release_fn;
            slot.release_ctx = task.release_ctx;
            slot.ready.store(true, std::memory_order_release);
            resultCv_.notify_one();
```

这段的核心语义是：

- 坏帧不进推理。
- 好帧进 `infer()`。
- 结果和原始帧一起回到 `ResultSlot`。

worker 的核心不是“只要有任务就算”，而是“先保证这个任务真的可算，再保证结果按序回到 `ResultSlot`”。它先检查尺寸和数据量，坏帧直接释放并在对应槽里写一个空结果，这样主线程不会因为某一帧异常而永久卡住。真正推理完成后，worker 会把检测结果、缩放比例和原始帧元数据一起写回槽位，再把 `ready` 置位并唤醒等待者。因为 worker 线程可能乱序完成，所以 `seq` 的存在很关键：结果先写进自己的 slot，主线程按顺序取，从而让对外表现仍然是稳定顺序。

## 13. `rknnPool::get()` 和退出

`get()` 按序号阻塞等待结果，不是忙等。

```cpp
template <typename Model>
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
```

成功时，它会把结果、原始帧指针和回调一起交给上层。

```cpp
    ResultSlot& slot = results_[nextGetSeq_];
    det    = slot.det;
    scaleW = slot.scale_w;
    scaleH = slot.scale_h;
    *frameData = slot.frame_data;
    *frameW    = slot.frame_w;
    *frameH    = slot.frame_h;
    *frameFmt  = slot.frame_fmt;
    if (frameId) *frameId = slot.frame_id;
    if (srcTsUs) *srcTsUs = slot.src_ts_us;
    if (frameReleaseFn) *frameReleaseFn = slot.release_fn;
    if (frameReleaseCtx) *frameReleaseCtx = slot.release_ctx;
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
```

退出时，`notifyGetters()` 会唤醒所有等待者。

```cpp
template <typename Model>
void rknnPool<Model>::notifyGetters() {
    getStopRequested_.store(true, std::memory_order_release);
    resultCv_.notify_all();
}
```

`get()` 不是“拿一个最快完成的结果”，而是“按提交顺序拿下一个该交给主线程的结果”。它先在 `resultCv_` 上等待 `nextGetSeq_` 对应的槽变成 ready；如果当前槽还没好，就继续等。这样主线程看到的是严格按输入顺序返回的结果，即使底层 worker 是并行完成的。退出时如果已经 stop 且没有可交付结果，就返回失败并让上层收尾。取到结果后，它会把结果槽清空并归还帧的释放契约，表示这个槽位可以被下一轮复用。

## 14. 你读这份文档时应该抓什么

- `main()` 负责调度，不负责算法。
- `capture_thread_func()` 负责把输入变成 `FrameTask`。
- `rknnPool::put()` 负责入队和所有权转移。
- `rknnPool::workerFunc()` 负责推理和写回结果。
- `rknnPool::get()` 负责顺序消费和资源交回。
- `render_thread_func()` 负责最终显示。

如果你先把这六个点看懂，整个工程的主链路就通了。

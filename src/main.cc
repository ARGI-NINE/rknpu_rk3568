#include <stdio.h>
#include <sys/time.h>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <vector>

#include "rkYolov5s.hpp"
#include "rknnPool.hpp"
#include "v4l2_capture.h"
#include "mpp_decoder.h"
#include "rga.h"  // for RK_FORMAT_*
#include "drm_display.h"
#include "preprocess.h"

// ---------------------------------------------------------------------------
// Color palette for bounding boxes (10 bright XRGB8888 colors)
// ---------------------------------------------------------------------------
static const uint32_t kBoxColors[10] = {
    0xFFFF0000, // Red
    0xFF00FF00, // Green
    0xFF0000FF, // Blue
    0xFFFFFF00, // Yellow
    0xFFFF00FF, // Magenta
    0xFF00FFFF, // Cyan
    0xFFFF8000, // Orange
    0xFF80FF00, // Lime
    0xFF0080FF, // Sky blue
    0xFFFF0080, // Pink
};

// ---------------------------------------------------------------------------
// Input source detection
// ---------------------------------------------------------------------------
static bool is_camera_input_source(const char* input) {
    if (strlen(input) == 1 && input[0] >= '0' && input[0] <= '9') {
        return true;
    }
    if (strncmp(input, "/dev/video", 10) == 0) {
        return true;
    }
    return false;
}

static long long now_wall_time_us() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1000000LL + tv.tv_usec;
}

static const int kRenderQueueMaxSize = 2;
static const int kPoolTaskQueueMaxSize = 4;
static const int kMaxWorkerThreads = 16;
static const int kDefaultCaptureWidth = 640;
static const int kDefaultCaptureHeight = 480;
static const int kV4l2BufferCount = 4;

using FrameReleaseFn = void (*)(unsigned char*, void*);

class FrameCopyPool {
public:
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
        for (int i = 0; i < prealloc_count; ++i) {
            unsigned char* buf = (unsigned char*)malloc(buffer_size_);
            if (!buf) {
                break;
            }
            free_list_.push_back(buf);
        }
    }

    ~FrameCopyPool() {
        for (unsigned char* buf : free_list_) {
            free(buf);
        }
        free_list_.clear();
    }

    size_t bufferSize() const {
        return buffer_size_;
    }

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

private:
    size_t buffer_size_;
    int max_cached_count_;
    std::mutex mtx_;
    std::vector<unsigned char*> free_list_;
};

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

// ---------------------------------------------------------------------------
// Frame data container for render queue
// ---------------------------------------------------------------------------
struct RenderItem {
    unsigned char* data;
    int width;
    int height;
    int format;
    detect_result_group_t det;
    FrameReleaseFn release_fn;
    void* release_ctx;
};

// ---------------------------------------------------------------------------
// Render one frame + detection overlay to DRM (RGA hardware only)
// ---------------------------------------------------------------------------
static void render_display(DrmDisplay& drm, const RenderItem& item)
{
    if (!item.data) {
        return;
    }

    int disp_w = drm.getWidth();
    int disp_h = drm.getHeight();
    int stride = drm.getStride();
    uint32_t* fb = drm.getBackBuffer();
    if (!fb) {
        return;
    }

    // RGA hardware conversion only; no CPU fallback.
    int rga_ret = rga_resize_convert_vaddr(
        item.data, item.width, item.height, item.format,
        (void*)fb, disp_w, disp_h, RK_FORMAT_BGRA_8888);
    if (rga_ret != 0) {
        printf("[ERROR] RGA display convert failed (%d), frame dropped\n", rga_ret);
        return;
    }

    // Draw detection boxes.
    for (int i = 0; i < item.det.count; i++) {
        const detect_result_t& r = item.det.results[i];

        int x0 = (int)(r.box.left   * disp_w / item.width);
        int y0 = (int)(r.box.top    * disp_h / item.height);
        int x1 = (int)(r.box.right  * disp_w / item.width);
        int y1 = (int)(r.box.bottom * disp_h / item.height);

        int bw = x1 - x0;
        int bh = y1 - y0;
        if (bw <= 0 || bh <= 0) {
            continue;
        }

        unsigned int hash = 0;
        for (const char* p = r.name; *p; p++) {
            hash = hash * 31 + (unsigned char)*p;
        }
        uint32_t color = kBoxColors[hash % 10];

        drm_draw_rect(fb, stride, disp_h, x0, y0, bw, bh, color, 2);

        char label[48];
        snprintf(label, sizeof(label), "%s %.0f%%", r.name, r.prop * 100.0f);

        int label_h = 7 * 2 + 2;
        int label_y = (y0 - label_h > 0) ? (y0 - label_h) : y0;
        drm_draw_fill_rect(
            fb, stride, disp_h, x0, label_y,
            (int)strlen(label) * 6 * 2 + 4, label_h, 0xC0000000);
        drm_draw_string(fb, stride, disp_h, x0 + 2, label_y + 1, label, color, 2);
    }

    int flip_ret = drm.flip();
    if (flip_ret != 0) {
        printf("[ERROR] drm.flip failed (%d)\n", flip_ret);
    }
}

// ---------------------------------------------------------------------------
// Async render thread
// ---------------------------------------------------------------------------
static std::mutex g_render_mtx;
static std::condition_variable g_render_cv;
static std::queue<RenderItem> g_render_queue;
static std::atomic<bool> g_render_stop{false};
static std::atomic<bool> g_capture_stop{false};
static std::atomic<uint64_t> g_frame_id_gen{0};
static std::atomic<uint64_t> g_pool_hit_count{0};
static std::atomic<uint64_t> g_pool_exhaust_drop_count{0};
static std::atomic<uint64_t> g_inflight_reject_count{0};
static std::atomic<uint64_t> g_render_drop_count{0};
static constexpr int g_render_queue_max_size = kRenderQueueMaxSize;

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

// ---------------------------------------------------------------------------
// Capture thread: decoupled from main loop for pipeline overlap
// ---------------------------------------------------------------------------
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

            long long src_ts_us = now_wall_time_us();
            uint64_t frame_id = g_frame_id_gen.fetch_add(1, std::memory_order_relaxed);
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
    printf("Runtime config: render_queue=%d task_queue=%d "
           "v4l2_default=%dx%d v4l2_buffers=%d drm=%s\n",
           kRenderQueueMaxSize,
           kPoolTaskQueueMaxSize,
           kDefaultCaptureWidth,
           kDefaultCaptureHeight,
           kV4l2BufferCount,
           "/dev/dri/card0");

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

    int capture_format = RK_FORMAT_YCbCr_420_SP;
    if (use_v4l2) {
        capture_format = RK_FORMAT_BGR_888;
        if (v4l2_pixfmt == 0x56595559 /* V4L2_PIX_FMT_YUYV */) {
            capture_format = RK_FORMAT_YUYV_422;
        } else if (v4l2_pixfmt == 0x3231564E /* V4L2_PIX_FMT_NV12 */) {
            capture_format = RK_FORMAT_YCbCr_420_SP;
        }
    }

    size_t frame_copy_size = compute_frame_size_bytes(frame_w, frame_h, capture_format);
    if (frame_copy_size == 0U) {
        printf("Invalid frame size: w=%d h=%d fmt=%d\n", frame_w, frame_h, capture_format);
        return cleanup_and_return(-1);
    }
    int frame_pool_prealloc = kPoolTaskQueueMaxSize + kRenderQueueMaxSize + threadNum + 2;
    int frame_pool_max_cached = frame_pool_prealloc + 4;
    FrameCopyPool frame_pool(frame_copy_size, frame_pool_prealloc, frame_pool_max_cached);
    printf("Frame copy pool: frame_bytes=%zu prealloc=%d max_cached=%d\n",
           frame_copy_size, frame_pool_prealloc, frame_pool_max_cached);

    struct timeval tv;
    gettimeofday(&tv, nullptr);
    long long startTime = tv.tv_sec * 1000LL + tv.tv_usec / 1000;

    int frames = 0;
    long long last_fps60_ms = startTime;

    auto log_fps_every_60_frames = [&]() {
        if (frames % 60 != 0) {
            return;
        }

        struct timeval now_tv;
        gettimeofday(&now_tv, nullptr);
        long long now_ms = now_tv.tv_sec * 1000LL + now_tv.tv_usec / 1000;
        long long window_ms = now_ms - last_fps60_ms;
        if (window_ms <= 0) {
            return;
        }

        double fps60 = 60.0 * 1000.0 / (double)window_ms;
        printf("[PERF] fps_60=%.2f\n", fps60);
        last_fps60_ms = now_ms;
    };

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

    enum class GetProcessMode {
        kMainLoop,
        kDrainLoop,
    };
    enum class LoopControlCode {
        kContinue,
        kBreak,
        kHandled,
    };

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

        (void)scale_w_unused;
        (void)scale_h_unused;
        (void)frame_id;
        (void)src_ts_us;

        bool enqueue_first = (mode == GetProcessMode::kDrainLoop);
        handle_get_success(det, frame_data, fw, fh, ff,
                           frame_release_fn, frame_release_ctx,
                           enqueue_first);
        return LoopControlCode::kHandled;
    };

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

    // Shutdown: capture -> drain -> render.
    g_capture_stop.store(true, std::memory_order_relaxed);
    pool.notifyGetters();
    if (capture_th.joinable()) {
        capture_th.join();
    }

    while (true) {
        LoopControlCode ctrl = process_one_pool_result(GetProcessMode::kDrainLoop);
        if (ctrl == LoopControlCode::kBreak) {
            break;
        }
    }

    gettimeofday(&tv, nullptr);
    long long endTime = tv.tv_sec * 1000LL + tv.tv_usec / 1000;
    printf("Total frames: %d, Average FPS: %.1f\n",
           frames, (double)frames / (double)(endTime - startTime) * 1000.0);
        printf("Pool stats: pool_hit=%llu pool_exhaust_drop=%llu inflight_reject=%llu render_drop=%llu\n",
           (unsigned long long)g_pool_hit_count.load(std::memory_order_relaxed),
           (unsigned long long)g_pool_exhaust_drop_count.load(std::memory_order_relaxed),
            (unsigned long long)g_inflight_reject_count.load(std::memory_order_relaxed),
           (unsigned long long)g_render_drop_count.load(std::memory_order_relaxed));

    return cleanup_and_return(0);
}

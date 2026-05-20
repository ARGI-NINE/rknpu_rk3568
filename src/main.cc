#include <stdio.h>
#include <sys/time.h>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <utility>
#include <vector>
#include <memory>
#include <new>

#include "rkYolov5s.hpp"
#include "rknnPool.hpp"
#include "frame_pools.h"
#include "v4l2_capture.h"
#include "mpp_decoder.h"
#include "mpp_encoder_rtsp.h"
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
static const int kStreamQueueMaxSize = 4;
static const int kMaxWorkerThreads = 16;
static const int kDefaultCaptureWidth = 640;
static const int kDefaultCaptureHeight = 480;
static const int kStreamOutputWidth = 640;
static const int kStreamOutputHeight = 540;
static const int kV4l2BufferCount = 4;
static const char* kRtspPushUrl = "rtsp://192.168.30.26:8554/rk3568-001/cam0";
static const int kDefaultFpsNum = 30;
static const int kDefaultFpsDen = 1;
static const int kDefaultRtspBitrateBps = 0;

static int align_up(int value, int align) {
    if (align <= 0) {
        return value;
    }
    return (value + align - 1) / align * align;
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

static int v4l2_pixfmt_to_rga_format(int pixfmt) {
    if (pixfmt == V4L2_PIX_FMT_YUYV) {
        return RK_FORMAT_YUYV_422;
    }
    if (pixfmt == V4L2_PIX_FMT_NV12) {
        return RK_FORMAT_YCbCr_420_SP;
    }
    return RK_FORMAT_BGR_888;
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

class GrowingFramePool {
public:
    GrowingFramePool(const char* label, int prealloc_count, int max_cached_count)
        : label_(label ? label : "frame"),
          prealloc_count_(prealloc_count < 0 ? 0 : prealloc_count),
          max_cached_count_(max_cached_count < 1 ? 1 : max_cached_count) {
        if (prealloc_count_ > max_cached_count_) {
            prealloc_count_ = max_cached_count_;
        }
    }

    FrameCopyPool* ensureCapacity(size_t required_size,
                                  int width,
                                  int height,
                                  int stride) {
        if (required_size == 0U) {
            return nullptr;
        }
        if (active_pool_ && active_size_bytes_ >= required_size) {
            return active_pool_;
        }

        std::unique_ptr<FrameCopyPool> next_pool(
            new (std::nothrow) FrameCopyPool(required_size, prealloc_count_, max_cached_count_));
        if (!next_pool) {
            fprintf(stderr,
                    "%s pool: failed to allocate generation for %dx%d stride=%d bytes=%zu\n",
                    label_, width, height, stride, required_size);
            return nullptr;
        }

        active_pool_ = next_pool.get();
        active_size_bytes_ = required_size;
        generations_.push_back(std::move(next_pool));
        printf("[INFO] %s pool generation=%zu bytes=%zu geometry=%dx%d stride=%d\n",
               label_, generations_.size(), active_size_bytes_, width, height, stride);
        return active_pool_;
    }

    size_t activeSizeBytes() const {
        return active_size_bytes_;
    }

    size_t generationCount() const {
        return generations_.size();
    }

private:
    const char* label_;
    int prealloc_count_;
    int max_cached_count_;
    FrameCopyPool* active_pool_{nullptr};
    size_t active_size_bytes_{0U};
    std::vector<std::unique_ptr<FrameCopyPool>> generations_;
};

static int ensure_fanout_buffers_for_frame(int width,
                                           int height,
                                           int format,
                                           GrowingFramePool* ai_frame_pools,
                                           GrowingFramePool* stream_buffer_pools,
                                           FrameCopyPool** ai_frame_pool,
                                           FrameCopyPool** stream_buffer_pool,
                                           int* stream_stride) {
    if (!ai_frame_pools || !stream_buffer_pools || !ai_frame_pool ||
        !stream_buffer_pool || !stream_stride) {
        return -1;
    }

    const size_t ai_frame_bytes = compute_frame_size_bytes(width, height, format);
    if (ai_frame_bytes == 0U) {
        return -1;
    }

    FrameCopyPool* next_ai_pool =
        ai_frame_pools->ensureCapacity(ai_frame_bytes, width, height, width);
    if (!next_ai_pool) {
        return -1;
    }

    const int next_stream_width = kStreamOutputWidth;
    const int next_stream_height = kStreamOutputHeight;
    const int next_stream_stride = align_up(next_stream_width, 16);
    if (next_stream_stride < next_stream_width) {
        return -1;
    }
    const size_t stream_frame_bytes =
        static_cast<size_t>(next_stream_stride) * static_cast<size_t>(next_stream_height) * 3U / 2U;
    FrameCopyPool* next_stream_pool = stream_buffer_pools->ensureCapacity(
        stream_frame_bytes, next_stream_width, next_stream_height, next_stream_stride);
    if (!next_stream_pool) {
        return -1;
    }

    *ai_frame_pool = next_ai_pool;
    *stream_buffer_pool = next_stream_pool;
    *stream_stride = next_stream_stride;
    return 0;
}

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
static std::atomic<uint64_t> g_stream_buffer_drop_count{0};
static std::atomic<uint64_t> g_stream_convert_fail_count{0};
static std::atomic<uint64_t> g_stream_encode_count{0};
static std::atomic<uint64_t> g_stream_encode_fail_count{0};
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
                                         size_t src_data_size,
                                         int width,
                                         int height,
                                         int format,
                                         uint64_t frame_id,
                                         long long src_ts_us,
                                         rknnPool<rkYolov5s>* pool,
                                         FrameCopyPool* ai_frame_pool)
{
    const size_t expected_size = compute_frame_size_bytes(width, height, format);
    if (!src_data || expected_size == 0 || src_data_size < expected_size) {
        return -1;
    }

    if (!ai_frame_pool || expected_size > ai_frame_pool->bufferSize()) {
        fprintf(stderr,
                "AI fan-out: invalid pool capacity bytes=%zu required=%zu geometry=%dx%d fmt=%d\n",
                ai_frame_pool ? ai_frame_pool->bufferSize() : 0U,
                expected_size,
                width,
                height,
                format);
        return -1;
    }

    unsigned char* owned_buf = ai_frame_pool->acquire();
    if (!owned_buf) {
        g_pool_exhaust_drop_count.fetch_add(1, std::memory_order_relaxed);
        return 1;
    }
    g_pool_hit_count.fetch_add(1, std::memory_order_relaxed);

    FrameReleaseFn release_fn = release_pooled_buffer;
    void* release_ctx = ai_frame_pool;

    memcpy(owned_buf, src_data, expected_size);
    PutRejectReason reject_reason = PutRejectReason::kNone;
    if (pool->put(owned_buf, width, height, format, expected_size,
                  frame_id, src_ts_us, release_fn, release_ctx, &reject_reason) != 0) {
        if (reject_reason == PutRejectReason::kInflightLimit) {
            g_inflight_reject_count.fetch_add(1, std::memory_order_relaxed);
            return 1;
        } else if (reject_reason == PutRejectReason::kInvalidReleaseContract) {
            // put() rejects this reason before ownership transfer; caller must reclaim.
            release_frame_buffer(owned_buf, release_fn, release_ctx);
            return -1;
        }
        return -1;
    }

    return 0;
}

static int copy_frame_and_enqueue_stream(const void* src_data,
                                         size_t src_data_size,
                                         int width,
                                         int height,
                                         int src_stride,
                                         int src_ver_stride,
                                         int src_format,
                                         uint64_t frame_id,
                                         long long capture_ts_us,
                                         StreamFramePool* stream_frame_pool,
                                         FrameCopyPool* stream_buffer_pool,
                                         int stream_stride) {
    const int stream_width = kStreamOutputWidth;
    const int stream_height = kStreamOutputHeight;
    if (!src_data || !stream_frame_pool || !stream_buffer_pool ||
        width <= 0 || height <= 0 || stream_stride < stream_width) {
        return -1;
    }

    const size_t required_src_size = compute_frame_size_bytes(width, height, src_format);
    if (required_src_size == 0 || src_data_size < required_src_size) {
        return -1;
    }

    const size_t stream_bytes =
        static_cast<size_t>(stream_stride) * static_cast<size_t>(stream_height) * 3U / 2U;
    if (stream_bytes > stream_buffer_pool->bufferSize()) {
        fprintf(stderr,
                "Stream fan-out: invalid pool capacity bytes=%zu required=%zu geometry=%dx%d stride=%d\n",
                stream_buffer_pool ? stream_buffer_pool->bufferSize() : 0U,
                stream_bytes,
                stream_width,
                stream_height,
                stream_stride);
        return -1;
    }

    unsigned char* stream_buf = stream_buffer_pool->acquire();
    if (!stream_buf) {
        g_stream_buffer_drop_count.fetch_add(1, std::memory_order_relaxed);
        return 1;
    }

    if (rga_resize_to_nv12_vaddr(const_cast<void*>(src_data),
                                 width, height,
                                 src_stride, src_ver_stride, src_format,
                                 stream_buf, stream_width, stream_height,
                                 stream_stride, stream_height) != 0) {
        g_stream_convert_fail_count.fetch_add(1, std::memory_order_relaxed);
        release_frame_buffer(stream_buf, release_pooled_buffer, stream_buffer_pool);
        return -1;
    }

    StreamFrame stream_frame;
    stream_frame.data = stream_buf;
    stream_frame.data_size = stream_bytes;
    stream_frame.frame_id = frame_id;
    stream_frame.capture_ts_us = capture_ts_us;
    stream_frame.width = stream_width;
    stream_frame.height = stream_height;
    stream_frame.stride = stream_stride;
    stream_frame.format = RK_FORMAT_YCbCr_420_SP;
    stream_frame.release_fn = release_pooled_buffer;
    stream_frame.release_ctx = stream_buffer_pool;
    stream_frame_pool->enqueue(std::move(stream_frame));
    return 0;
}

static void stream_thread_func(StreamFramePool* stream_frame_pool,
                               MppRtspEncoder* rtsp_encoder) {
    while (true) {
        StreamFrame frame;
        if (!stream_frame_pool->waitAndPop(&frame)) {
            break;
        }

        if (rtsp_encoder->encodeAndPush(frame) != 0) {
            g_stream_encode_fail_count.fetch_add(1, std::memory_order_relaxed);
            release_frame_buffer(frame.data, frame.release_fn, frame.release_ctx);
            g_capture_stop.store(true, std::memory_order_relaxed);
            stream_frame_pool->stop();
            break;
        }

        g_stream_encode_count.fetch_add(1, std::memory_order_relaxed);
        release_frame_buffer(frame.data, frame.release_fn, frame.release_ctx);
    }
}

// ---------------------------------------------------------------------------
// Source thread: capture/decode once, then fan out into independent AI/RTSP paths.
// ---------------------------------------------------------------------------
static void source_thread_func(V4L2Capture* v4l2, MppDecoder* mpp_dec,
                               bool use_v4l2, int v4l2_pixfmt,
                               rknnPool<rkYolov5s>* ai_pool,
                               GrowingFramePool* ai_frame_pools,
                               StreamFramePool* stream_frame_pool,
                               GrowingFramePool* stream_buffer_pools)
{
    int last_src_w = 0;
    int last_src_h = 0;

    while (!g_capture_stop.load(std::memory_order_relaxed)) {
        const void* src_data = nullptr;
        size_t src_size = 0;
        int src_w = 0;
        int src_h = 0;
        int src_fmt = RK_FORMAT_YCbCr_420_SP;
        int src_stride = 0;
        int src_ver_stride = 0;
        long long src_ts_us = now_wall_time_us();
        V4L2Frame v4l2_frame;

        if (use_v4l2) {
            if (v4l2->dequeueFrame(&v4l2_frame) != 0) {
                printf("V4L2: dequeueFrame failed\n");
                break;
            }
            src_data = v4l2_frame.data;
            src_size = static_cast<size_t>(v4l2_frame.size);
            src_w = v4l2->getWidth();
            src_h = v4l2->getHeight();
            src_fmt = v4l2_pixfmt_to_rga_format(v4l2_pixfmt);
            src_stride = src_w;
            src_ver_stride = src_h;
            if (v4l2_frame.capture_ts_us > 0) {
                src_ts_us = v4l2_frame.capture_ts_us;
            }
        } else {
            unsigned char* nv12_data = nullptr;
            if (mpp_dec->readFrame(&nv12_data, &src_w, &src_h) != 0) {
                break;
            }
            src_data = nv12_data;
            src_size = compute_frame_size_bytes(src_w, src_h, RK_FORMAT_YCbCr_420_SP);
            src_fmt = RK_FORMAT_YCbCr_420_SP;
            src_stride = src_w;
            src_ver_stride = src_h;
        }

        if (!use_v4l2 &&
            last_src_w > 0 &&
            last_src_h > 0 &&
            (src_w != last_src_w || src_h != last_src_h)) {
            printf("[INFO] File input geometry change: %dx%d -> %dx%d\n",
                   last_src_w, last_src_h, src_w, src_h);
        }
        last_src_w = src_w;
        last_src_h = src_h;

        FrameCopyPool* ai_frame_pool = nullptr;
        FrameCopyPool* stream_buffer_pool = nullptr;
        int stream_stride = 0;
        if (ensure_fanout_buffers_for_frame(src_w, src_h, src_fmt,
                                            ai_frame_pools, stream_buffer_pools,
                                            &ai_frame_pool, &stream_buffer_pool,
                                            &stream_stride) != 0) {
            if (!use_v4l2) {
                fprintf(stderr,
                        "Source fan-out: failed to configure buffers for %dx%d fmt=%d\n",
                        src_w, src_h, src_fmt);
            }
            if (use_v4l2 && v4l2->queueFrame(v4l2_frame) != 0) {
                printf("V4L2: queueFrame failed\n");
            }
            break;
        }

        const uint64_t frame_id = g_frame_id_gen.fetch_add(1, std::memory_order_relaxed);
        const int ai_ret = copy_frame_and_submit_to_pool(src_data, src_size, src_w, src_h,
                                                         src_fmt, frame_id, src_ts_us,
                                                         ai_pool, ai_frame_pool);
        const int stream_ret = copy_frame_and_enqueue_stream(src_data, src_size, src_w, src_h,
                                                             src_stride, src_ver_stride, src_fmt,
                                                             frame_id, src_ts_us,
                                                             stream_frame_pool, stream_buffer_pool,
                                                             stream_stride);

        if (use_v4l2 && v4l2->queueFrame(v4l2_frame) != 0) {
            printf("V4L2: queueFrame failed\n");
            break;
        }

        if (ai_ret < 0 || stream_ret < 0) {
            break;
        }
    }

    g_capture_stop.store(true, std::memory_order_relaxed);
    if (ai_pool) {
        ai_pool->notifyGetters();
    }
    if (stream_frame_pool) {
        stream_frame_pool->stop();
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
    g_stream_buffer_drop_count.store(0, std::memory_order_relaxed);
    g_stream_convert_fail_count.store(0, std::memory_order_relaxed);
    g_stream_encode_count.store(0, std::memory_order_relaxed);
    g_stream_encode_fail_count.store(0, std::memory_order_relaxed);

    rknnPool<rkYolov5s> ai_pool(model_path, threadNum, kPoolTaskQueueMaxSize);
    if (ai_pool.init() != 0) {
        printf("ai_pool init failed!\n");
        return -1;
    }

    DrmDisplay drm_display;
    bool drm_ok = true;

    std::thread render_th;
    std::thread source_th;
    std::thread stream_th;
    bool render_started = false;
    bool source_started = false;
    bool stream_started = false;
    bool v4l2_opened = false;
    bool v4l2_stream_started = false;

    V4L2Capture* v4l2 = nullptr;
    MppDecoder* mpp_dec = nullptr;
    StreamFramePool stream_frame_pool(kStreamQueueMaxSize);
    MppRtspEncoder rtsp_encoder;

    auto cleanup_and_return = [&](int rc) -> int {
        g_capture_stop.store(true, std::memory_order_relaxed);
        ai_pool.notifyGetters();
        stream_frame_pool.stop();

        if (v4l2 && v4l2_stream_started) {
            v4l2->stopStream();
            v4l2_stream_started = false;
        }

        if (source_started && source_th.joinable()) {
            source_th.join();
        }
        if (stream_started && stream_th.joinable()) {
            stream_th.join();
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
        rtsp_encoder.close();

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
    int fps_num = kDefaultFpsNum;
    int fps_den = kDefaultFpsDen;

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
        fps_num = v4l2->getFpsNum();
        fps_den = v4l2->getFpsDen();
        printf("V4L2: %s %dx%d pixfmt=0x%08x\n", devPath, frame_w, frame_h, v4l2_pixfmt);
    } else {
        mpp_dec = new MppDecoder();
        if (mpp_dec->open(input_source) != 0) {
            return cleanup_and_return(-1);
        }
        frame_w = mpp_dec->getWidth();
        frame_h = mpp_dec->getHeight();
        fps_num = mpp_dec->getFpsNum();
        fps_den = mpp_dec->getFpsDen();
    }

    int capture_format = use_v4l2 ? v4l2_pixfmt_to_rga_format(v4l2_pixfmt)
                                  : RK_FORMAT_YCbCr_420_SP;

    const size_t ai_frame_bytes = compute_frame_size_bytes(frame_w, frame_h, capture_format);
    if (ai_frame_bytes == 0U) {
        printf("Invalid frame size: w=%d h=%d fmt=%d\n", frame_w, frame_h, capture_format);
        return cleanup_and_return(-1);
    }

 const int stream_stride = align_up(kStreamOutputWidth, 16);
 const size_t stream_frame_bytes =
 static_cast<size_t>(stream_stride) * static_cast<size_t>(kStreamOutputHeight) * 3U / 2U;

    int ai_pool_prealloc = kPoolTaskQueueMaxSize + kRenderQueueMaxSize + threadNum + 2;
    int ai_pool_max_cached = ai_pool_prealloc + 4;
    GrowingFramePool ai_frame_pools("AI frame",
                                    ai_pool_prealloc,
                                    ai_pool_max_cached);

    int stream_buffer_prealloc = kStreamQueueMaxSize + 3;
    int stream_buffer_max_cached = stream_buffer_prealloc + 4;
    GrowingFramePool stream_buffer_pools("Stream NV12",
                                         stream_buffer_prealloc,
                                         stream_buffer_max_cached);

    FrameCopyPool* initial_ai_frame_pool = nullptr;
    FrameCopyPool* initial_stream_buffer_pool = nullptr;
    int initial_stream_stride = 0;
    if (ensure_fanout_buffers_for_frame(frame_w, frame_h, capture_format,
                                        &ai_frame_pools, &stream_buffer_pools,
                                        &initial_ai_frame_pool, &initial_stream_buffer_pool,
                                        &initial_stream_stride) != 0) {
        printf("Failed to allocate initial fan-out buffers for %dx%d fmt=%d\n",
               frame_w, frame_h, capture_format);
        return cleanup_and_return(-1);
    }

 if (rtsp_encoder.open(kRtspPushUrl,
 kStreamOutputWidth, kStreamOutputHeight,
 initial_stream_stride, kStreamOutputHeight,
 fps_num, fps_den, kDefaultRtspBitrateBps) != 0) {
 printf("RTSP encoder init failed for %s\n", kRtspPushUrl);
 return cleanup_and_return(-1);
    }

    (void)initial_ai_frame_pool;
    (void)initial_stream_buffer_pool;
    printf("AI frame pool: frame_bytes=%zu prealloc=%d max_cached=%d generations=%zu\n",
           ai_frame_bytes, ai_pool_prealloc, ai_pool_max_cached,
           ai_frame_pools.generationCount());
 printf("Stream path: url=%s geometry=%dx%d frame_bytes=%zu stride=%d fps=%d/%d prealloc=%d max_cached=%d\n",
 kRtspPushUrl, kStreamOutputWidth, kStreamOutputHeight,
 stream_frame_bytes, initial_stream_stride, fps_num, fps_den,
 stream_buffer_prealloc, stream_buffer_max_cached);

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

        int ret = ai_pool.get(det, scale_w_unused, scale_h_unused,
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

    stream_th = std::thread(stream_thread_func, &stream_frame_pool, &rtsp_encoder);
    stream_started = true;
    source_th = std::thread(source_thread_func,
                            v4l2, mpp_dec, use_v4l2, v4l2_pixfmt,
                            &ai_pool, &ai_frame_pools,
                            &stream_frame_pool, &stream_buffer_pools);
    source_started = true;

    while (true) {
        LoopControlCode ctrl = process_one_pool_result(GetProcessMode::kMainLoop);
        if (ctrl == LoopControlCode::kBreak) {
            break;
        }
        if (ctrl == LoopControlCode::kContinue) {
            continue;
        }
    }

    // Shutdown: source -> drain AI -> wait for stream -> render.
    g_capture_stop.store(true, std::memory_order_relaxed);
    ai_pool.notifyGetters();
    if (use_v4l2 && v4l2 && v4l2_stream_started) {
        v4l2->stopStream();
        v4l2_stream_started = false;
    }
    if (source_th.joinable()) {
        source_th.join();
    }

    while (true) {
        LoopControlCode ctrl = process_one_pool_result(GetProcessMode::kDrainLoop);
        if (ctrl == LoopControlCode::kBreak) {
            break;
        }
    }

    stream_frame_pool.stop();
    if (stream_th.joinable()) {
        stream_th.join();
    }

    gettimeofday(&tv, nullptr);
    long long endTime = tv.tv_sec * 1000LL + tv.tv_usec / 1000;
    const long long elapsed_ms = std::max(1LL, endTime - startTime);
    printf("Total frames: %d, Average FPS: %.1f\n",
           frames, (double)frames / (double)elapsed_ms * 1000.0);
    printf("AI stats: pool_hit=%llu pool_exhaust_drop=%llu inflight_reject=%llu render_drop=%llu\n",
           (unsigned long long)g_pool_hit_count.load(std::memory_order_relaxed),
           (unsigned long long)g_pool_exhaust_drop_count.load(std::memory_order_relaxed),
           (unsigned long long)g_inflight_reject_count.load(std::memory_order_relaxed),
           (unsigned long long)g_render_drop_count.load(std::memory_order_relaxed));
    printf("Stream stats: queue_drop_oldest=%llu buffer_drop=%llu convert_fail=%llu encoded=%llu encode_fail=%llu\n",
           (unsigned long long)stream_frame_pool.droppedCount(),
           (unsigned long long)g_stream_buffer_drop_count.load(std::memory_order_relaxed),
           (unsigned long long)g_stream_convert_fail_count.load(std::memory_order_relaxed),
           (unsigned long long)g_stream_encode_count.load(std::memory_order_relaxed),
           (unsigned long long)g_stream_encode_fail_count.load(std::memory_order_relaxed));

    return cleanup_and_return(0);
}

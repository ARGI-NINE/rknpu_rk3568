#ifndef RKNNPOOL_H
#define RKNNPOOL_H

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <atomic>
#include <memory>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include "postprocess.h"

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
    // Frame data preserved for display (worker does NOT free)
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

template <typename Model>
class rknnPool {
private:
    int threadNum_;
    int maxQueueSize_;
    std::string modelPath_;

    // Ring buffer (256 slots, uint8_t index auto-wraps 0-255)
    ResultSlot results_[256];
    std::atomic<uint8_t> nextPutSeq_{0};  // atomic: capture_thread safety
    uint8_t nextGetSeq_;     // only accessed by get() caller (single consumer)
    std::atomic<int> inflight_;  // number of frames in-flight (put but not yet get)

    // Bounded work queue
    std::mutex queueMtx_;
    std::condition_variable queueNotEmpty_;
    std::condition_variable queueNotFull_;
    std::queue<FrameTask> taskQueue_;
    bool stop_;

    // Result notification (replaces spin-wait)
    std::mutex resultMtx_;
    std::condition_variable resultCv_;
    std::atomic<bool> getStopRequested_{false};

    // Model instances (one per worker) and worker threads
    std::vector<std::shared_ptr<Model>> models_;
    std::vector<std::thread> workers_;

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

    static bool hasValidReleaseContract(void (*release_fn)(unsigned char*, void*),
                                        void* release_ctx) {
        return release_fn != nullptr && release_ctx != nullptr;
    }

    void workerFunc(int id) {
        while (true) {
            FrameTask task;
            {
                std::unique_lock<std::mutex> lock(queueMtx_);
                queueNotEmpty_.wait(lock, [this]{ return !taskQueue_.empty() || stop_; });
                if (stop_ && taskQueue_.empty()) return;
                task = taskQueue_.front();
                taskQueue_.pop();
            }
            queueNotFull_.notify_one();

            const bool invalidDimensions = (task.width <= 0 || task.height <= 0);
            const size_t minDataSize = invalidDimensions
                ? 0U
                : static_cast<size_t>(task.width) * static_cast<size_t>(task.height);
            if (invalidDimensions || task.dataSize < minDataSize) {
                printf("rknnPool: skip infer seq=%u invalid frame payload (size=%zu, min=%zu, w=%d, h=%d, fmt=%d)\n",
                       task.seq, task.dataSize, minDataSize, task.width, task.height, task.format);

                // Preserve sequence progress and metadata, but do not feed invalid buffer to infer().
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

            // Run inference on this worker's dedicated model instance
            detect_result_group_t det;
            float sw, sh;
            models_[id]->infer(task.data, task.width, task.height, task.format, &det, &sw, &sh);

            // Store result + frame data into ring buffer (DO NOT free data)
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

            // Wake up get() via condition variable
            resultCv_.notify_one();
        }
    }

public:
    rknnPool(const std::string& modelPath, int threadNum, int maxQueueSize = 4);
    ~rknnPool();
    int init();
    // On success, takes ownership of frameData immediately.
    // The buffer must always be returned via release_fn(frameData, release_ctx), never free() directly.
    int put(unsigned char* frameData, int w, int h, int format, size_t dataSize,
            uint64_t frameId, long long srcTsUs,
            void (*release_fn)(unsigned char*, void*),
            void* release_ctx,
            PutRejectReason* outRejectReason = nullptr);
    // Returns detection + original frame data.
    // Returned frameData must be returned via frameReleaseFn(frameData, frameReleaseCtx), never free() directly.
    int get(detect_result_group_t& det, float& scaleW, float& scaleH,
            unsigned char** frameData, int* frameW, int* frameH, int* frameFmt,
            uint64_t* frameId, long long* srcTsUs,
            void (**frameReleaseFn)(unsigned char*, void*) = nullptr,
            void** frameReleaseCtx = nullptr);
    void notifyGetters();
    int taskQueueSize();
    int inflightSize() const;
};

// ---------------------------------------------------------------------------
// Implementation (header-only template)
// ---------------------------------------------------------------------------

template <typename Model>
rknnPool<Model>::rknnPool(const std::string& modelPath, int threadNum, int maxQueueSize)
    : threadNum_(threadNum), maxQueueSize_(maxQueueSize), modelPath_(modelPath),
      nextGetSeq_(0), inflight_(0), stop_(false) {
    if (maxQueueSize_ < 1) {
        maxQueueSize_ = 1;
    }
}

template <typename Model>
int rknnPool<Model>::init() {
    // Create model instances
    try {
        for (int i = 0; i < threadNum_; i++)
            models_.push_back(std::make_shared<Model>(modelPath_.c_str()));
    } catch (const std::bad_alloc& e) {
        printf("rknnPool: out of memory: %s\n", e.what());
        return -1;
    }

    // First model: full init (loads weights, creates context)
    // Remaining models: dup_context (share weights, independent execution state)
    for (int i = 0; i < threadNum_; i++) {
        int ret = models_[i]->init(models_[0]->get_pctx(), i != 0);
        if (ret != 0) {
            printf("rknnPool: model[%d] init failed (%d)\n", i, ret);
            return ret;
        }
    }

    // Launch worker threads
    for (int i = 0; i < threadNum_; i++)
        workers_.emplace_back(&rknnPool::workerFunc, this, i);

    printf("rknnPool: %d workers ready, queue_limit=%d\n", threadNum_, maxQueueSize_);
    return 0;
}

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
        // Reject before ownership transfer to enforce non-null release contract.
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

    uint8_t seq = nextPutSeq_.fetch_add(1, std::memory_order_relaxed);  // natural overflow at 255→0

    // Clear the destination result slot before enqueueing
    results_[seq].ready.store(false, std::memory_order_relaxed);

    FrameTask task;
    task.seq      = seq;
    task.data     = frameData;  // ownership transferred, no copy
    task.width    = w;
    task.height   = h;
    task.format   = format;
    task.dataSize = dataSize;
    task.frame_id = frameId;
    task.src_ts_us = srcTsUs;
    task.release_fn = release_fn;
    task.release_ctx = release_ctx;

    // Push to bounded work queue (block if full)
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

template <typename Model>
int rknnPool<Model>::get(detect_result_group_t& det, float& scaleW, float& scaleH,
                         unsigned char** frameData, int* frameW, int* frameH, int* frameFmt,
                         uint64_t* frameId, long long* srcTsUs,
                         void (**frameReleaseFn)(unsigned char*, void*),
                         void** frameReleaseCtx) {
    // Block until the expected sequence result is ready, or stop is requested.
    {
        std::unique_lock<std::mutex> lock(resultMtx_);
        resultCv_.wait(lock, [this]{
            if (results_[nextGetSeq_].ready.load(std::memory_order_acquire)) {
                return true;
            }
            return getStopRequested_.load(std::memory_order_acquire);
        });
    }

    // Woken by stop notification and no result available.
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

    nextGetSeq_++;  // natural overflow at 255→0
    inflight_.fetch_sub(1, std::memory_order_relaxed);
    return 0;
}

template <typename Model>
int rknnPool<Model>::taskQueueSize() {
    std::lock_guard<std::mutex> lock(queueMtx_);
    return (int)taskQueue_.size();
}

template <typename Model>
int rknnPool<Model>::inflightSize() const {
    return inflight_.load(std::memory_order_relaxed);
}

template <typename Model>
void rknnPool<Model>::notifyGetters() {
    getStopRequested_.store(true, std::memory_order_release);
    resultCv_.notify_all();
}

template <typename Model>
rknnPool<Model>::~rknnPool() {
    // Signal all workers to stop
    {
        std::lock_guard<std::mutex> lock(queueMtx_);
        stop_ = true;
    }
    queueNotEmpty_.notify_all();
    queueNotFull_.notify_all();
    getStopRequested_.store(true, std::memory_order_release);
    resultCv_.notify_all();

    // Wait for workers to finish
    for (auto& t : workers_) {
        if (t.joinable())
            t.join();
    }

    // Free any undrained tasks in the queue
    while (!taskQueue_.empty()) {
        FrameTask& task = taskQueue_.front();
        releaseFrameData(task.data, task.release_fn, task.release_ctx);
        taskQueue_.pop();
    }
    // Free any undrained result frame data
    for (int i = 0; i < 256; i++) {
        if (results_[i].frame_data) {
            releaseFrameData(results_[i].frame_data, results_[i].release_fn, results_[i].release_ctx);
            results_[i].frame_data = nullptr;
        }
        results_[i].release_fn = nullptr;
        results_[i].release_ctx = nullptr;
    }
}

#endif

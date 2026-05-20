#ifndef FRAME_POOLS_H
#define FRAME_POOLS_H

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <vector>

using FrameReleaseFn = void (*)(unsigned char*, void*);

static inline void release_frame_buffer(unsigned char* data,
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
            unsigned char* buf = static_cast<unsigned char*>(malloc(buffer_size_));
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
        if (static_cast<int>(free_list_.size()) >= max_cached_count_) {
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

static inline void release_pooled_buffer(unsigned char* data, void* ctx) {
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

struct StreamFrame {
    unsigned char* data{nullptr};
    size_t data_size{0};
    uint64_t frame_id{0};
    long long capture_ts_us{0};
    int width{0};
    int height{0};
    int stride{0};
    int format{0};
    FrameReleaseFn release_fn{nullptr};
    void* release_ctx{nullptr};
};

class StreamFramePool {
public:
    explicit StreamFramePool(size_t max_queue_size)
        : max_queue_size_(max_queue_size == 0 ? 1 : max_queue_size) {}

    ~StreamFramePool() {
        stop();
        clear();
    }

    void enqueue(StreamFrame&& frame) {
        StreamFrame dropped;
        bool has_dropped = false;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (stop_) {
                has_dropped = true;
                dropped = frame;
            } else {
                if (queue_.size() >= max_queue_size_) {
                    dropped = queue_.front();
                    queue_.pop_front();
                    has_dropped = true;
                    dropped_count_.fetch_add(1, std::memory_order_relaxed);
                }
                queue_.push_back(frame);
            }
        }

        if (has_dropped) {
            release_frame_buffer(dropped.data, dropped.release_fn, dropped.release_ctx);
        }
        cv_.notify_one();
    }

    bool waitAndPop(StreamFrame* out_frame) {
        if (!out_frame) {
            return false;
        }

        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
        if (queue_.empty()) {
            return false;
        }

        *out_frame = queue_.front();
        queue_.pop_front();
        return true;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
    }

    void clear() {
        std::deque<StreamFrame> pending;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            pending.swap(queue_);
        }

        for (StreamFrame& frame : pending) {
            release_frame_buffer(frame.data, frame.release_fn, frame.release_ctx);
            frame = StreamFrame{};
        }
    }

    uint64_t droppedCount() const {
        return dropped_count_.load(std::memory_order_relaxed);
    }

private:
    size_t max_queue_size_;
    mutable std::mutex mtx_;
    std::condition_variable cv_;
    std::deque<StreamFrame> queue_;
    bool stop_{false};
    std::atomic<uint64_t> dropped_count_{0};
};

#endif

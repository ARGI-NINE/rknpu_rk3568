#ifndef V4L2_CAPTURE_H
#define V4L2_CAPTURE_H

#include <string>
#include <vector>
#include <linux/videodev2.h>

static const int V4L2_DEFAULT_BUF_COUNT = 4;

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

    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    int getPixelFormat() const { return pixfmt_; }
    bool isOpen() const { return fd_ >= 0; }
};

#endif

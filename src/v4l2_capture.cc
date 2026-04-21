#include "v4l2_capture.h"

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

V4L2Capture::V4L2Capture(const std::string &device, int width, int height, int bufferCount)
    : device_(device), fd_(-1), reqWidth_(width), reqHeight_(height), reqBufferCount_(bufferCount),
      width_(0), height_(0), pixfmt_(0), bufferCount_(0), currentIdx_(-1)
{
    if (reqBufferCount_ < 2) {
        reqBufferCount_ = 2;
    }
}

V4L2Capture::~V4L2Capture()
{
    if (fd_ >= 0) {
        stopStream();
        close();
    }
}

int V4L2Capture::open()
{
    fd_ = ::open(device_.c_str(), O_RDWR);
    if (fd_ < 0) {
        perror("V4L2: open device");
        return -1;
    }

    // Check capabilities
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

    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        fprintf(stderr, "V4L2: device does not support streaming\n");
        ::close(fd_); fd_ = -1;
        return -1;
    }

    if (initFormat() != 0) {
        ::close(fd_); fd_ = -1;
        return -1;
    }

    if (initMmap() != 0) {
        ::close(fd_); fd_ = -1;
        return -1;
    }

    return 0;
}

int V4L2Capture::initFormat()
{
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width  = reqWidth_;
    fmt.fmt.pix.height = reqHeight_;
    fmt.fmt.pix.field  = V4L2_FIELD_NONE;

    // Try YUYV first (best for RGA)
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        // Fallback to NV12
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12;
        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
            perror("V4L2: S_FMT (YUYV/NV12 both failed)");
            return -1;
        }
    }

    // Read back actual format
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_G_FMT, &fmt) < 0) {
        perror("V4L2: G_FMT");
        return -1;
    }

    width_  = fmt.fmt.pix.width;
    height_ = fmt.fmt.pix.height;
    pixfmt_ = fmt.fmt.pix.pixelformat;

    printf("V4L2: actual format %dx%d pixfmt=0x%08x\n", width_, height_, pixfmt_);
    return 0;
}

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
    buffers_.assign((size_t)bufferCount_, MmapBuffer{nullptr, 0});

    for (int i = 0; i < bufferCount_; i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;

        if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
            perror("V4L2: QUERYBUF");
            return -1;
        }

        buffers_[i].length = buf.length;
        buffers_[i].start = mmap(NULL, buf.length,
                                 PROT_READ | PROT_WRITE, MAP_SHARED,
                                 fd_, buf.m.offset);
        if (buffers_[i].start == MAP_FAILED) {
            perror("V4L2: mmap");
            return -1;
        }
    }

    return 0;
}

int V4L2Capture::startStream()
{
    // Queue all buffers
    for (int i = 0; i < bufferCount_; i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;

        if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
            perror("V4L2: QBUF");
            return -1;
        }
    }

    // Stream on
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
        perror("V4L2: STREAMON");
        return -1;
    }

    return 0;
}

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
    if (currentIdx_ < 0 || currentIdx_ >= bufferCount_) {
        fprintf(stderr, "V4L2: invalid buffer index %d\n", currentIdx_);
        return -1;
    }
    *data = buffers_[currentIdx_].start;
    *size = (int)buf.bytesused;
    return 0;
}

void V4L2Capture::releaseFrame()
{
    if (currentIdx_ < 0) return;

    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index  = currentIdx_;

    if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
        perror("V4L2: QBUF (release)");
    }

    currentIdx_ = -1;
}

int V4L2Capture::stopStream()
{
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (fd_ >= 0 && ioctl(fd_, VIDIOC_STREAMOFF, &type) < 0) {
        perror("V4L2: STREAMOFF");
        return -1;
    }
    return 0;
}

void V4L2Capture::close()
{
    for (int i = 0; i < bufferCount_; i++) {
        if (buffers_[i].start && buffers_[i].start != MAP_FAILED) {
            munmap(buffers_[i].start, buffers_[i].length);
            buffers_[i].start = nullptr;
        }
    }
    buffers_.clear();
    bufferCount_ = 0;

    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

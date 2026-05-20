# 输入、解码与显示

这份文档讲 `V4L2Capture`、`MppDecoder`、`DrmDisplay`。

它们分别对应三条链路：

- 摄像头输入
- 视频文件解码输入
- 最终显示输出

## 1. `V4L2Capture`

先看接口声明。

```cpp
class V4L2Capture {
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
```

这个类的语义很明确：

- `open()` 打开设备并做格式协商。
- `startStream()` 开始采集。
- `captureFrame()` 取一帧。
- `releaseFrame()` 还帧。
- `stopStream()` 停流。
- `close()` 释放资源。

`V4L2Capture` 不是简单的“打开摄像头”，而是把 Linux V4L2 的设备能力检查、像素格式协商、MMAP 缓冲和 DQBUF/QBUF 契约封装成一个对象。它存在的原因是摄像头帧在 Linux 里本来就是驱动管理的 buffer，用户态只能借用，不能直接改成自己的生命周期。

### 打开、格式协商、mmap

```cpp
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
```

`initFormat()` 会先试 `YUYV`，失败再试 `NV12`。

```cpp
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
```

`initMmap()` 会申请驱动缓冲区并做内存映射。

注：这里 `mmap()` 映射的是驱动已经持有的采集缓冲，不是用户态重新 `malloc` 出来的一块内存。

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
```

你要理解这里的流程：

`open()` 里先 `VIDIOC_QUERYCAP` 是为了确认这个设备真的是可采集、可 streaming 的摄像头节点；`initFormat()` 先试 YUYV 再回退 NV12，是因为前者更方便后续 RGA 处理，后者则是常见兜底格式。`initMmap()` 不是在“分配内存”，而是在把驱动分配的 buffer 通过 mmap 暴露给用户态，这样 `captureFrame()` 才能只借用指针而不再做二次 copy。

- 先看设备是否能采集。
- 再定像素格式。
- 再把驱动缓冲映射到用户态。

`initFormat()` 先尝试 `YUYV`，失败再退到 `NV12`，这个顺序不是随便写的，而是在尽量兼容常见摄像头输出格式。`initMmap()` 则是把驱动申请到的缓冲区映射到用户态，这样后面的 `captureFrame()` 才能直接拿到地址而不是再做一次 copy。换句话说，前面这一步是在把“摄像头设备”变成“可直接读写的内存窗口”。

### 启流、取帧、还帧

```cpp
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
```

这段的意义是：

- `captureFrame()` 拿到的是驱动缓冲的地址。
- `releaseFrame()` 把这个缓冲再放回队列。
- 所以它是一套“取出 -> 用完 -> 还回去”的契约。

`startStream()` 把前面申请好的所有 buffer 重新 `QBUF` 回驱动，然后 `STREAMON` 真正开始采集；`captureFrame()` 则是 `DQBUF` 出当前可用帧，把当前 buffer 的地址和 `bytesused` 返回给上层。这里最关键的一点是：拿到的是借来的驱动缓冲，所以它必须在 copy 完之后立刻通过 `releaseFrame()` 还回去。这个“取出 -> 使用 -> 归还”的契约就是 V4L2 路径能连续跑下去的前提。

### 停流和关闭

```cpp
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
```

`V4L2Capture` 这一节真正要记住的是“借用驱动缓冲，然后尽快还回去”。它不是把摄像头帧拷成新对象，而是把驱动分配的 buffer 通过 mmap 暴露到用户态，`captureFrame()` 只是把当前帧指针和长度交出来，`releaseFrame()` 再把这个 buffer 重新排回队列。你读它时要把“打开设备、约束格式、建立映射、流转 buffer”这条线串起来看。

这里的标准顺序就是：先把所有 buffer 重新 `QBUF` 回驱动，再 `STREAMON`，`DQBUF` 拿到数据后立即 `QBUF` 归还。这样驱动就能持续复用同一批 buffer，不会把内存压力转移到用户态。

## 2. `MppDecoder`

`MppDecoder` 负责把视频文件解码成连续 NV12 帧。

### 接口

```cpp
class MppDecoder {
public:
    MppDecoder();
    ~MppDecoder();

    int  open(const char *filepath);
    void close();

    /**
     * Decode next video frame.
     * @param frame_data  [out] pointer to NV12 pixel data (width*height*3/2 bytes).
     *                    Valid until next readFrame() or close(). Caller must NOT free.
     * @param width       [out] frame width (may differ from container metadata after info-change)
     * @param height      [out] frame height
     * @return 0 on success, -1 on EOF or error
     */
    int  readFrame(unsigned char **frame_data, int *width, int *height);

    int  getWidth()  const { return video_width_; }
    int  getHeight() const { return video_height_; }

private:
    // ffmpeg demux
    AVFormatContext *fmt_ctx_;
    AVBSFContext    *bsf_ctx_;      // h264_mp4toannexb / hevc_mp4toannexb
    AVPacket        *pkt_;
    AVPacket        *filtered_pkt_;
    int              video_stream_idx_;

    // MPP decode
    MppCtx          mpp_ctx_;
    MppApi         *mpi_;
    MppBufferGroup  frm_grp_;

    // Frame info
    int    video_width_;
    int    video_height_;

    // Reusable frame buffer (NV12, stripped of stride padding)
    unsigned char *frame_buf_;
    size_t         frame_buf_size_;

    bool eos_sent_;   // EOS packet sent to MPP
    bool eos_got_;    // EOS frame received from MPP

    // Feed one compressed packet to MPP; may block briefly if decoder is full.
    int feedPacket(void *data, int size, int64_t pts);
    // Send EOS signal to MPP
    int sendEos();
};
```

`MppDecoder` 是“压缩视频 -> 原始 NV12”的桥。它不是单纯调一个解码 API，而是同时管 FFmpeg 的 demux、可选的 bitstream filter、MPP 的解码上下文、外部 buffer group 和后续连续帧的整理，所以它的职责比表面看起来更重。

`MppDecoder` 不是单纯调一个解码 API，而是把 FFmpeg 的 demux、H.264/H.265 的 AVCC→Annex-B 过滤、MPP 解码上下文、外部 buffer group 和 NV12 连续帧整理成一个对象。它存在是因为 RK3568 的视频文件输入本质上要走硬解，MP4 容器里的压缩流也不能直接喂给 MPP。

### open()

`open()` 做的事很多：

1. 用 FFmpeg 打开容器。
2. 找视频流。
3. 根据 codec 选择 MPP coding 和 bitstream filter。
4. 初始化 MPP 解码器。
5. 预分配外部 buffer group。

注：外部 buffer group 必须在第一包进入解码器前准备好，否则 RK3568 上很容易卡在 `BUFFER_FULL`。

```cpp
int MppDecoder::open(const char *filepath) {
    int ret;

    // ---- 1. ffmpeg: open container ----
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
    video_stream_idx_ = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_idx_ < 0) {
        printf("MppDecoder error: no video stream found\n");
        close();
        return -1;
    }

    AVCodecParameters *codecpar = fmt_ctx_->streams[video_stream_idx_]->codecpar;
    video_width_  = codecpar->width;
    video_height_ = codecpar->height;

    // ---- 2. Determine codec → MPP coding type ----
    MppCodingType coding = MPP_VIDEO_CodingUnused;
    const char *bsf_name = nullptr;
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
        bsf_name = nullptr;  // VP9 doesn't need BSF
        break;
    default:
        printf("MppDecoder error: unsupported codec_id=%d\n", codecpar->codec_id);
        close();
        return -1;
    }

    // ---- 3. Setup bitstream filter (AVCC → Annex-B for H.264/HEVC in MP4) ----
    if (bsf_name) {
        const AVBitStreamFilter *bsf = av_bsf_get_by_name(bsf_name);
        if (!bsf) {
            printf("MppDecoder error: bsf '%s' not found\n", bsf_name);
            close();
            return -1;
        }
        ret = av_bsf_alloc(bsf, &bsf_ctx_);
        if (ret < 0) {
            printf("MppDecoder error: av_bsf_alloc failed ret=%d\n", ret);
            close();
            return -1;
        }
        avcodec_parameters_copy(bsf_ctx_->par_in, codecpar);
        bsf_ctx_->time_base_in = fmt_ctx_->streams[video_stream_idx_]->time_base;
        ret = av_bsf_init(bsf_ctx_);
        if (ret < 0) {
            printf("MppDecoder error: av_bsf_init failed ret=%d\n", ret);
            close();
            return -1;
        }
    }

    // ---- 4. Allocate ffmpeg packets ----
    pkt_ = av_packet_alloc();
    filtered_pkt_ = av_packet_alloc();
    if (!pkt_ || !filtered_pkt_) {
        printf("MppDecoder error: av_packet_alloc failed\n");
        close();
        return -1;
    }

    // ---- 5. Init MPP decoder ----
    ret = mpp_create(&mpp_ctx_, &mpi_);
    if (ret != MPP_OK) {
        printf("MppDecoder error: mpp_create failed ret=%d\n", ret);
        close();
        return -1;
    }
 
    // Split mode: MPP will find frame boundaries in the stream
    RK_U32 need_split = 1;
    mpi_->control(mpp_ctx_, MPP_DEC_SET_PARSER_SPLIT_MODE, &need_split);
    ret = mpp_init(mpp_ctx_, MPP_CTX_DEC, coding);
    if (ret != MPP_OK) {
        printf("MppDecoder error: mpp_init failed ret=%d coding=%d\n", ret, (int)coding);
        close();
        return -1;
    }

    // ---- 6. Pre-allocate external buffer group (MUST be before first packet) ----
    // Without this, decode_put_packet fails with MPP_ERR_BUFFER_FULL on RK3568
    ret = mpp_buffer_group_get_internal(&frm_grp_, MPP_BUFFER_TYPE_ION);
    if (ret != MPP_OK) {
        printf("MppDecoder error: mpp_buffer_group_get_internal failed ret=%d\n", ret);
        close();
        return -1;
    }
    mpi_->control(mpp_ctx_, MPP_DEC_SET_EXT_BUF_GROUP, frm_grp_);
    return 0;
}
```

`open()` 的重点不是“能打开就行”，而是把后面解码会用到的所有组件都准备好：先确定编码格式，再决定要不要走 bitstream filter，再创建 packet 和 frame 缓冲，最后把 MPP 上下文、解析模式和外部 buffer group 都连起来。这样一来，后面的 `feedPacket()` 和 `readFrame()` 才能只关心解码，不用每次都重建环境。

这里最关键的两件事是：H.264/H.265 在 MP4 里通常要先经过 bitstream filter 转成 Annex-B，和 `mpp_buffer_group_get_internal()` 一定要在第一包之前完成。否则 `decode_put_packet()` 可能直接因为 buffer 不足而失败。

### feedPacket() 和 sendEos()

```cpp
int MppDecoder::feedPacket(void *data, int size, int64_t pts) {
    MppPacket mpp_pkt = nullptr;
    mpp_packet_init(&mpp_pkt, data, size);
    mpp_packet_set_pts(mpp_pkt, pts);

    MPP_RET ret;
    int retries = 0;
    do {
        ret = mpi_->decode_put_packet(mpp_ctx_, mpp_pkt);
        if (ret == MPP_OK) break;

        // Packet queue full, drain pending frames (especially info_change)
        MppFrame frame = nullptr;
        MPP_RET fret = mpi_->decode_get_frame(mpp_ctx_, &frame);
        if (fret == MPP_OK && frame) {
            if (mpp_frame_get_info_change(frame)) {
                printf("[WARN-MPP] info_change during feedPacket: %dx%d\n",
                       mpp_frame_get_width(frame), mpp_frame_get_height(frame));
                video_width_  = (int)mpp_frame_get_width(frame);
                video_height_ = (int)mpp_frame_get_height(frame);
                if (frm_grp_) {
                    mpp_buffer_group_put(frm_grp_);
                    frm_grp_ = nullptr;
                }
                mpp_buffer_group_get_internal(&frm_grp_, MPP_BUFFER_TYPE_ION);
                mpi_->control(mpp_ctx_, MPP_DEC_SET_EXT_BUF_GROUP, frm_grp_);
                mpi_->control(mpp_ctx_, MPP_DEC_SET_INFO_CHANGE_READY, nullptr);
            }
            mpp_frame_deinit(&frame);
        }

        usleep(2000);
        retries++;
    } while (retries < 100);  // ~200 ms max (more time for info_change to settle)

    mpp_packet_deinit(&mpp_pkt);
    if (ret != MPP_OK) {
        printf("[WARN-MPP] decode_put_packet FINAL FAIL ret=%d retries=%d\n", ret, retries);
        return -1;
    }
    return 0;
}

int MppDecoder::sendEos() {
    MppPacket mpp_pkt = nullptr;
    mpp_packet_init(&mpp_pkt, nullptr, 0);
    mpp_packet_set_eos(mpp_pkt);

    MPP_RET ret;
    int retries = 0;
    do {
        ret = mpi_->decode_put_packet(mpp_ctx_, mpp_pkt);
        if (ret == MPP_OK) break;
        usleep(2000);
        retries++;
    } while (retries < 50);

    mpp_packet_deinit(&mpp_pkt);
    eos_sent_ = true;
    if (ret != MPP_OK) {
        printf("MppDecoder error: sendEos failed ret=%d retries=%d\n", ret, retries);
        return -1;
    }
    return 0;
}
```

`feedPacket()` / `sendEos()` 这一组的核心是“持续喂包，直到解码器接收成功；结束时显式标记 EOS”。这里之所以有重试和 `info_change` 处理，是因为硬件解码器不是单纯的同步函数，它可能因为上下文状态、缓冲状态或流参数变化而要求你重新协调一次。换句话说，这里不是“写一包就完事”，而是在和 MPP 维持一个持续的流式协议。

### readFrame()

`readFrame()` 先尝试直接拿解码结果，如果没有帧就去读包，再把包喂进解码器。

注：这里一旦遇到 `info_change`，就必须重建 buffer group 并显式 `INFO_CHANGE_READY`，否则后续帧不会继续输出。

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

            // Extract NV12 pixel data (handle stride ≠ width)
            MppBuffer buf = mpp_frame_get_buffer(frame);
            if (!buf) {
                mpp_frame_deinit(&frame);
                continue;
            }
            RK_U32 w          = mpp_frame_get_width(frame);
            RK_U32 h          = mpp_frame_get_height(frame);
            RK_U32 hor_stride = mpp_frame_get_hor_stride(frame);
            RK_U32 ver_stride = mpp_frame_get_ver_stride(frame);

            // NV12 size: Y plane (w*h) + UV plane (w*h/2)
            size_t needed = (size_t)w * h * 3 / 2;
            if (needed > frame_buf_size_) {
                free(frame_buf_);
                frame_buf_ = (unsigned char *)malloc(needed);
                if (!frame_buf_) {
                    printf("MppDecoder error: malloc frame buffer failed\n");
                    mpp_frame_deinit(&frame);
                    return -1;
                }
                frame_buf_size_ = needed;
            }

            unsigned char *src = (unsigned char *)mpp_buffer_get_ptr(buf);

            // Copy Y plane (strip horizontal stride padding)
            for (RK_U32 row = 0; row < h; row++)
                memcpy(frame_buf_ + row * w, src + row * hor_stride, w);

            // Copy UV plane (interleaved U/V, h/2 rows)
            unsigned char *uv_src = src + hor_stride * ver_stride;
            unsigned char *uv_dst = frame_buf_ + w * h;
            for (RK_U32 row = 0; row < h / 2; row++)
                memcpy(uv_dst + row * w, uv_src + row * hor_stride, w);

            *frame_data = frame_buf_;
            *width      = (int)w;
            *height     = (int)h;

            mpp_frame_deinit(&frame);
            return 0;
        }

        // ---- 2. No frame ready — feed more compressed packets ----
        if (eos_sent_) {
            // Already sent EOS but no more frames → truly done
            eos_got_ = true;
            return -1;
        }

        // Read next packet from container
        int av_ret = av_read_frame(fmt_ctx_, pkt_);
        if (av_ret < 0) {
            // EOF of file → send EOS to decoder, then loop to drain remaining frames
            if (sendEos() != 0) {
                return -1;
            }
            continue;
        }

        // Skip non-video packets
        if (pkt_->stream_index != video_stream_idx_) {
            av_packet_unref(pkt_);
            continue;
        }

        // Apply bit-stream filter (AVCC → Annex-B) if configured
        if (bsf_ctx_) {
            av_ret = av_bsf_send_packet(bsf_ctx_, pkt_);
            av_packet_unref(pkt_);
            if (av_ret < 0) {
                printf("MppDecoder error: av_bsf_send_packet failed ret=%d\n", av_ret);
                continue;
            }

            while (av_bsf_receive_packet(bsf_ctx_, filtered_pkt_) == 0) {
                if (feedPacket(filtered_pkt_->data, filtered_pkt_->size, filtered_pkt_->pts) != 0) {
                    av_packet_unref(filtered_pkt_);
                    return -1;
                }
                av_packet_unref(filtered_pkt_);
            }
        } else {
            // No BSF needed (e.g. VP9)
            if (feedPacket(pkt_->data, pkt_->size, pkt_->pts) != 0) {
                av_packet_unref(pkt_);
                return -1;
            }
            av_packet_unref(pkt_);
        }
    }
}
```

更准确地说，连续 NV12 的组包是两段拷贝：先拷贝 Y 平面，再拷贝 UV 平面，同时把解码器输出里的 stride 剪成连续布局。

```cpp
RK_U32 hor_stride = mpp_frame_get_hor_stride(frame);
RK_U32 ver_stride = mpp_frame_get_ver_stride(frame);

for (RK_U32 row = 0; row < h; row++)
    memcpy(frame_buf_ + row * w, src + row * hor_stride, w);

unsigned char *uv_src = src + hor_stride * ver_stride;
unsigned char *uv_dst = frame_buf_ + w * h;
for (RK_U32 row = 0; row < h / 2; row++)
    memcpy(uv_dst + row * w, uv_src + row * hor_stride, w);
```

关键点：

- 解码输出不是直接拿来就能用，还要处理 `info_change` 和 `EOS`。
- 最后拷贝成连续 NV12，方便后面的统一处理。

`readFrame()` 先试着直接拿解码结果，如果拿不到帧才去继续喂包，这样可以避免无意义的忙等。它最后把解码器输出整理成连续的 NV12 内存，是为了让后面的推理模块看到一块稳定、紧凑、无需再解释 stride 的输入。`info_change`、`EOS`、`discard` 和 `errinfo` 这些分支都在告诉你：硬件解码输出不是“拿到就能用”，而是需要先筛掉不能用的帧，再把能用的帧整理成统一格式。

### close()

```cpp
void MppDecoder::close() {
    if (mpp_ctx_) {
        mpi_->reset(mpp_ctx_);
        mpp_destroy(mpp_ctx_);
        mpp_ctx_ = nullptr;
        mpi_     = nullptr;
    }
    if (frm_grp_) {
        mpp_buffer_group_put(frm_grp_);
        frm_grp_ = nullptr;
    }
    if (bsf_ctx_) {
        av_bsf_free(&bsf_ctx_);
        bsf_ctx_ = nullptr;
    }
    if (filtered_pkt_) {
        av_packet_free(&filtered_pkt_);
        filtered_pkt_ = nullptr;
    }
    if (pkt_) {
        av_packet_free(&pkt_);
        pkt_ = nullptr;
    }
    if (fmt_ctx_) {
        avformat_close_input(&fmt_ctx_);
        fmt_ctx_ = nullptr;
    }
    free(frame_buf_);
    frame_buf_      = nullptr;
    frame_buf_size_ = 0;
    eos_sent_ = false;
    eos_got_  = false;
}
```

`close()` 则是把这整条视频解码链收干净：重置 MPP、销毁上下文、回收 buffer group、释放 FFmpeg 的包和格式上下文，再把 `frame_buf_` 还给堆。它的存在说明这个解码器不是“临时对象”，而是有明确生命周期的资源管理器。

## 3. `DrmDisplay`

`DrmDisplay` 管的是显示层：双缓冲、翻页、清屏和文字/框绘制。

### 接口

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
};
```

`DrmDisplay` 管的是显示层：双缓冲、翻页、清屏和文字/框绘制。它不用 GUI 框架，而是直接操作 DRM/KMS 的 framebuffer，所以初始化阶段必须把 connector、mode、CRTC 和 dumb buffer 全部准备好。你可以把它理解成“把屏幕变成一个可直接写的内存窗口”。

`DrmDisplay` 直接走 Linux DRM/KMS，不借助 GUI 框架。它把 connector、mode、CRTC、两个 dumb buffer 和保存下来的原始 CRTC 都收起来，目的就是让显示路径只做“画到 back buffer -> page flip -> 恢复系统状态”。

### init()

初始化会做这些事：

- 打开 DRM 设备。
- 找 connector 和 mode。
- 找可用 CRTC。
- 创建两个 dumb buffer。
- 把第一个 buffer 显示出来。

```cpp
int DrmDisplay::init()
{
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
    drmModeConnector* conn = nullptr;
    for (int i = 0; i < res->count_connectors; i++) {
        drmModeConnector* c = drmModeGetConnector(drm_fd_, res->connectors[i]);
        if (c && c->connection == DRM_MODE_CONNECTED && c->count_modes > 0) {
            conn = c;
            break;
        }
        if (c) {
            drmModeFreeConnector(c);
        }
    }
    if (!conn) {
        printf("DrmDisplay: no connected connector found on %s\n", dev);
        drmModeFreeResources(res);
        close(drm_fd_);
        drm_fd_ = -1;
        return -1;
    }
 
    printf("DrmDisplay: using DRM device %s\n", dev);
 
    conn_id_ = conn->connector_id;
 
    /* Pick preferred mode or first mode */
    drmModeModeInfo mode = {};
    bool found_preferred = false;
    for (int i = 0; i < conn->count_modes; i++) {
        if (conn->modes[i].type & DRM_MODE_TYPE_PREFERRED) {
            mode = conn->modes[i];
            found_preferred = true;
            break;
        }
    }
    if (!found_preferred) {
        mode = conn->modes[0];
    }
    width_  = mode.hdisplay;
    height_ = mode.vdisplay;
    mode_   = mode;
    printf("DrmDisplay: mode %dx%d @ %dHz\n", width_, height_, mode.vrefresh);
 
    /* Find a suitable CRTC */
    crtc_id_ = 0;
 
    /* Try encoder's current CRTC first */
    if (conn->encoder_id) {
        drmModeEncoder* enc = drmModeGetEncoder(drm_fd_, conn->encoder_id);
        if (enc) {
            if (enc->crtc_id) {
                crtc_id_ = enc->crtc_id;
            }
            drmModeFreeEncoder(enc);
        }
    }
 
    /* If not found, iterate connector's encoders and find a usable CRTC */
    if (crtc_id_ == 0) {
        for (int i = 0; i < conn->count_encoders; i++) {
            drmModeEncoder* enc = drmModeGetEncoder(drm_fd_, conn->encoders[i]);
            if (!enc) continue;
 
            for (int j = 0; j < res->count_crtcs; j++) {
                if (enc->possible_crtcs & (1u << j)) {
                    crtc_id_ = res->crtcs[j];
                    drmModeFreeEncoder(enc);
                    goto crtc_found;
                }
            }
            drmModeFreeEncoder(enc);
        }
    }
 
crtc_found:
    if (crtc_id_ == 0) {
        printf("DrmDisplay: no suitable CRTC found\n");
        drmModeFreeConnector(conn);
        drmModeFreeResources(res);
        close(drm_fd_);
        drm_fd_ = -1;
        return -1;
    }
 
    /* Save old CRTC for restore on deinit */
    saved_crtc_ = (drmModeCrtcPtr)drmModeGetCrtc(drm_fd_, crtc_id_);
    for (int i = 0; i < 2; i++) {
        struct drm_mode_create_dumb creq = {};
        creq.width  = (uint32_t)width_;
        creq.height = (uint32_t)height_;
        creq.bpp    = 32;
 
        if (drmIoctl(drm_fd_, DRM_IOCTL_MODE_CREATE_DUMB, &creq) < 0) {
            printf("DrmDisplay: DRM_IOCTL_MODE_CREATE_DUMB failed for buf %d: %s\n", i, strerror(errno));
            drmModeFreeConnector(conn);
            drmModeFreeResources(res);
            deinit();
            return -1;
        }
 
        bo_handle_[i] = creq.handle;
        pitch_[i]     = creq.pitch;
        size_[i]      = creq.size;
 
        /* Add framebuffer */
        if (drmModeAddFB(drm_fd_, (uint32_t)width_, (uint32_t)height_,
                         24, 32, pitch_[i], bo_handle_[i], &fb_id_[i]) != 0) {
            printf("DrmDisplay: drmModeAddFB failed for buf %d: %s\n", i, strerror(errno));
            drmModeFreeConnector(conn);
            drmModeFreeResources(res);
            deinit();
            return -1;
        }
 
        /* mmap */
        struct drm_mode_map_dumb mreq = {};
        mreq.handle = bo_handle_[i];
        if (drmIoctl(drm_fd_, DRM_IOCTL_MODE_MAP_DUMB, &mreq) < 0) {
            printf("DrmDisplay: DRM_IOCTL_MODE_MAP_DUMB failed for buf %d: %s\n", i, strerror(errno));
            drmModeFreeConnector(conn);
            drmModeFreeResources(res);
            deinit();
            return -1;
        }
 
        map_[i] = (uint32_t*)mmap(0, size_[i], PROT_READ | PROT_WRITE, MAP_SHARED,
                                   drm_fd_, mreq.offset);
        if (map_[i] == MAP_FAILED) {
            printf("DrmDisplay: mmap failed for buf %d: %s\n", i, strerror(errno));
            map_[i] = nullptr;
            drmModeFreeConnector(conn);
            drmModeFreeResources(res);
            deinit();
            return -1;
        }
        memset(map_[i], 0, size_[i]);
    }
    /* Set initial CRTC with front buffer (index 0) */
    front_ = 0;
    if (drmModeSetCrtc(drm_fd_, crtc_id_, fb_id_[front_], 0, 0,
                       &conn_id_, 1, &mode) != 0) {
        printf("DrmDisplay: initial drmModeSetCrtc failed: %s\n", strerror(errno));
        drmModeFreeConnector(conn);
        drmModeFreeResources(res);
        deinit();
        return -1;
    }
 
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);
 
    printf("DrmDisplay: init OK, %dx%d, double-buffered\n", width_, height_);
    return 0;
}
```

初始化的关键不是只是“打开显示设备”，而是确定当前输出模式并把两个 buffer 都建立起来。之后程序永远在 back buffer 上画，画完再 page flip 到 front buffer，这样观感才不会撕裂。`deinit()` 里先恢复原来的 CRTC，再销毁 dumb buffer 和映射，是为了让程序退出后把屏幕还给系统原来的状态。

这不是普通的“开个窗口”，而是先找当前连着的 connector，再选一个 mode 和可用 CRTC，最后创建两块 dumb buffer。`drmModeSetCrtc()` 把第一个 buffer 设成当前前台，后面只需要在 back buffer 上画完再翻页。

### getBackBuffer() / flip() / clearBackBuffer()

注：`flip()` 里不是发完 page-flip 就结束，而是要等到下一次 VBLANK 真的完成切换后才确认前后台缓冲已交换。

```cpp
uint32_t* DrmDisplay::getBackBuffer()
{
    int back = 1 - front_;
    return map_[back];
}

int DrmDisplay::getWidth()
{
    return width_;
}

int DrmDisplay::getHeight()
{
    return height_;
}

int DrmDisplay::getStride()
{
    int back = 1 - front_;
    return (int)(pitch_[back] / 4);
}

int DrmDisplay::flip()
{
    int back = 1 - front_;
    int ret = drmModePageFlip(drm_fd_, crtc_id_, fb_id_[back],
                              DRM_MODE_PAGE_FLIP_EVENT, this);
    if (ret != 0) {
        printf("DrmDisplay: drmModePageFlip failed: %s\n", strerror(errno));
        return -1;
    }
 
    /* Wait for the page-flip event (fires at next VBLANK) */
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(drm_fd_, &fds);
    struct timeval timeout = {1, 0}; /* 1 second safety timeout */
 
    drmEventContext ev_ctx = {};
    ev_ctx.version = 2;
    ev_ctx.page_flip_handler = [](int /*fd*/, unsigned int /*seq*/,
                                  unsigned int /*tv_sec*/,
                                  unsigned int /*tv_usec*/,
                                  void* /*data*/) { /* noop */ };
 
    int sel = select(drm_fd_ + 1, &fds, NULL, NULL, &timeout);
 
    if (sel > 0) {
        drmHandleEvent(drm_fd_, &ev_ctx);
    } else if (sel == 0) {
        printf("DrmDisplay: pageflip wait timeout (1s)\n");
    } else {
        printf("DrmDisplay: pageflip wait select error: %s\n", strerror(errno));
    }

    front_ = back;
    return 0;
}

void DrmDisplay::clearBackBuffer(uint32_t color)
{
    uint32_t* buf = getBackBuffer();
    if (!buf) return;
 
    int total = (int)(pitch_[1 - front_] / 4) * height_;
    for (int i = 0; i < total; i++) {
        buf[i] = color;
    }
}
```

`getBackBuffer()` / `flip()` / `clearBackBuffer()` 这组函数就是双缓冲的最小闭环：`getBackBuffer()` 总是返回当前不可见的那块内存，`clearBackBuffer()` 用统一底色把它擦干净，`flip()` 通过 page flip 把新画面切到前台。这样做的好处是绘制和显示分离，渲染线程可以放心往 back buffer 里画，不会直接污染正在显示的那一帧。

### deinit()

```cpp
void DrmDisplay::deinit()
{
    if (drm_fd_ < 0) return;
    if (saved_crtc_) {
        drmModeSetCrtc(drm_fd_, saved_crtc_->crtc_id, saved_crtc_->buffer_id,
                       saved_crtc_->x, saved_crtc_->y, &conn_id_, 1, &saved_crtc_->mode);
        drmModeFreeCrtc(saved_crtc_);
        saved_crtc_ = nullptr;
    }
    for (int i = 0; i < 2; i++) {
        /* Unmap */
        if (map_[i]) {
            munmap(map_[i], size_[i]);
            map_[i] = nullptr;
        }
 
        /* Remove FB */
        if (fb_id_[i]) {
            drmModeRmFB(drm_fd_, fb_id_[i]);
            fb_id_[i] = 0;
        }
 
        /* Destroy dumb buffer */
        if (bo_handle_[i]) {
            struct drm_mode_destroy_dumb dreq = {};
            dreq.handle = bo_handle_[i];
            drmIoctl(drm_fd_, DRM_IOCTL_MODE_DESTROY_DUMB, &dreq);
            bo_handle_[i] = 0;
        }
    }
    close(drm_fd_);
    drm_fd_ = -1;
    width_  = 0;
    height_ = 0;
}
```

`deinit()` 的职责是把显示前的系统状态恢复回去，而不是只把自己的对象释放掉。只要是 DRM/KMS 程序，退出顺序就很重要：先恢复 CRTC，再解绑 FB，最后 destroy dumb buffer 和 close fd。这样显示器不会留在一个“程序死了但屏幕状态没还原”的半残状态。

## 4. 绘图辅助

这些函数负责把框和文字画到 framebuffer 上。

```cpp
static inline void put_pixel(uint32_t* fb, int fb_w, int fb_h, int x, int y, uint32_t color)
{
    if (x >= 0 && x < fb_w && y >= 0 && y < fb_h) {
        fb[y * fb_w + x] = color;
    }
}

void drm_draw_fill_rect(uint32_t* fb, int fb_w, int fb_h,
                        int x, int y, int w, int h,
                        uint32_t color)
{
    int x0 = x < 0 ? 0 : x;
    int y0 = y < 0 ? 0 : y;
    int x1 = (x + w) > fb_w ? fb_w : (x + w);
    int y1 = (y + h) > fb_h ? fb_h : (y + h);
 
    for (int row = y0; row < y1; row++) {
        for (int col = x0; col < x1; col++) {
            fb[row * fb_w + col] = color;
        }
    }
}

void drm_draw_rect(uint32_t* fb, int fb_w, int fb_h,
                   int x, int y, int w, int h,
                   uint32_t color, int thickness)
{
    if (thickness < 1) thickness = 1;
 
    /* Top edge */
    drm_draw_fill_rect(fb, fb_w, fb_h, x, y, w, thickness, color);
    /* Bottom edge */
    drm_draw_fill_rect(fb, fb_w, fb_h, x, y + h - thickness, w, thickness, color);
    /* Left edge */
    drm_draw_fill_rect(fb, fb_w, fb_h, x, y, thickness, h, color);
    /* Right edge */
    drm_draw_fill_rect(fb, fb_w, fb_h, x + w - thickness, y, thickness, h, color);
}

void drm_draw_char(uint32_t* fb, int fb_w, int fb_h,
                   int x, int y, char c,
                   uint32_t color, int scale)
{
    if (c < 32 || c > 126) return;
    if (scale < 1) scale = 1;
 
    int idx = c - 32;
    const uint8_t* glyph = font5x7[idx];
 
    for (int col = 0; col < 5; col++) {
        uint8_t column_bits = glyph[col];
        for (int row = 0; row < 7; row++) {
            if (column_bits & (1 << row)) {
                /* Draw a scale x scale block */
                for (int sy = 0; sy < scale; sy++) {
                    for (int sx = 0; sx < scale; sx++) {
                        put_pixel(fb, fb_w, fb_h,
                                  x + col * scale + sx,
                                  y + row * scale + sy,
                                  color);
                    }
                }
            }
        }
    }
}

void drm_draw_string(uint32_t* fb, int fb_w, int fb_h,
                     int x, int y, const char* str,
                     uint32_t color, int scale)
{
    if (!str) return;
    if (scale < 1) scale = 1;
 
    int cursor_x = x;
    /* Character width = 5*scale, plus 1*scale gap between chars */
    int char_advance = 6 * scale;
 
    while (*str) {
        drm_draw_char(fb, fb_w, fb_h, cursor_x, y, *str, color, scale);
        cursor_x += char_advance;
        str++;
    }
}
```

这些函数在文档里应该这样理解：

- `put_pixel()` 是最底层像素写入。
- `drm_draw_rect()` 和 `drm_draw_fill_rect()` 负责框。
- `drm_draw_char()` 和 `drm_draw_string()` 负责文字。

这些绘图函数本质上都是软件层的补丁：它们直接在 framebuffer 上画框、填色、画字符。`put_pixel()` 负责最小写入和边界检查，`drm_draw_fill_rect()` 负责整块填充，`drm_draw_rect()` 负责只画边框，`drm_draw_char()` 和 `drm_draw_string()` 负责把结果写成文字。它们之所以这么朴素，是因为这个工程真正要优化的是视频主链路，不是文字渲染本身。

## 5. 你读这份文档时应该抓什么

- 摄像头输入，看 `V4L2Capture`。
- 视频文件输入，看 `MppDecoder`。
- 最终显示，看 `DrmDisplay`。
- 框和标签显示，看绘图辅助函数。

如果你把这三块看懂，输入和输出两端就通了

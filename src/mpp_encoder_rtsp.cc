#include "mpp_encoder_rtsp.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/error.h>
#include <libavutil/mem.h>
#include <libavutil/rational.h>
}

#include "rockchip/mpp_task.h"
#include "rockchip/rk_mpi_cmd.h"
#include "rockchip/rk_type.h"
#include "rockchip/rk_venc_rc.h"
#include "rga.h"

namespace {

constexpr size_t kEncoderHeaderBufferSize = 64 * 1024;

static int align_up(int value, int align) {
    if (align <= 0) {
        return value;
    }
    return (value + align - 1) / align * align;
}

static int clamp_bitrate(int width, int height, int fps_num, int fps_den, int requested_bps) {
    if (requested_bps > 0) {
        return requested_bps;
    }

    if (fps_num <= 0 || fps_den <= 0) {
        fps_num = 30;
        fps_den = 1;
    }

    const long long fps = std::max(1LL, static_cast<long long>(fps_num) / fps_den);
    long long guess = static_cast<long long>(width) * height * fps;
    guess = std::max(guess, 1000000LL);
    guess = std::min(guess, 8000000LL);
    return static_cast<int>(guess);
}

}  // namespace

MppRtspEncoder::MppRtspEncoder() = default;

MppRtspEncoder::~MppRtspEncoder() {
    close();
}

int MppRtspEncoder::open(const char* rtsp_url,
 int width,
 int height,
 int hor_stride,
                         int ver_stride,
                         int fps_num,
                         int fps_den,
 int bitrate_bps) {
 close();

 rtsp_url_ = rtsp_url ? rtsp_url : "";
 width_ = width;
 height_ = height;
 hor_stride_ = align_up(std::max(width, hor_stride), 16);
    ver_stride_ = std::max(height, ver_stride);
    fps_num_ = (fps_num > 0) ? fps_num : 30;
    fps_den_ = (fps_den > 0) ? fps_den : 1;
    bitrate_bps_ = clamp_bitrate(width_, height_, fps_num_, fps_den_, bitrate_bps);
    first_capture_ts_us_ = -1;
    last_capture_ts_us_ = -1;
    io_opened_ = false;
    header_written_ = false;

 if (rtsp_url_.empty() || width_ <= 0 || height_ <= 0) {
 fprintf(stderr, "RTSP encoder: invalid open arguments\n");
 return -1;
 }

    if (initMpp() != 0) {
        close();
        return -1;
    }

    if (initRtspOutput() != 0) {
        close();
        return -1;
    }

    return 0;
}

int MppRtspEncoder::initMpp() {
    MPP_RET ret = mpp_create(&mpp_ctx_, &mpi_);
    if (ret != MPP_OK) {
        fprintf(stderr, "RTSP encoder: mpp_create failed ret=%d\n", ret);
        return -1;
    }

    RK_S64 timeout = MPP_POLL_BLOCK;
    mpi_->control(mpp_ctx_, MPP_SET_INPUT_TIMEOUT, reinterpret_cast<MppParam>(&timeout));
    mpi_->control(mpp_ctx_, MPP_SET_OUTPUT_TIMEOUT, reinterpret_cast<MppParam>(&timeout));

    ret = mpp_init(mpp_ctx_, MPP_CTX_ENC, MPP_VIDEO_CodingAVC);
    if (ret != MPP_OK) {
        fprintf(stderr, "RTSP encoder: mpp_init failed ret=%d\n", ret);
        return -1;
    }

    ret = mpp_enc_cfg_init(&enc_cfg_);
    if (ret != MPP_OK || !enc_cfg_) {
        fprintf(stderr, "RTSP encoder: mpp_enc_cfg_init failed ret=%d\n", ret);
        return -1;
    }

    ret = mpi_->control(mpp_ctx_, MPP_ENC_GET_CFG, enc_cfg_);
    if (ret != MPP_OK) {
        fprintf(stderr, "RTSP encoder: MPP_ENC_GET_CFG failed ret=%d\n", ret);
        return -1;
    }

    mpp_enc_cfg_set_s32(enc_cfg_, "base:low_delay", 1);
    mpp_enc_cfg_set_s32(enc_cfg_, "prep:width", width_);
    mpp_enc_cfg_set_s32(enc_cfg_, "prep:height", height_);
    mpp_enc_cfg_set_s32(enc_cfg_, "prep:hor_stride", hor_stride_);
    mpp_enc_cfg_set_s32(enc_cfg_, "prep:ver_stride", ver_stride_);
    mpp_enc_cfg_set_s32(enc_cfg_, "prep:format", MPP_FMT_YUV420SP);

    mpp_enc_cfg_set_s32(enc_cfg_, "rc:mode", MPP_ENC_RC_MODE_CBR);
    mpp_enc_cfg_set_s32(enc_cfg_, "rc:bps_target", bitrate_bps_);
    mpp_enc_cfg_set_s32(enc_cfg_, "rc:bps_max", bitrate_bps_ * 17 / 16);
    mpp_enc_cfg_set_s32(enc_cfg_, "rc:bps_min", bitrate_bps_ * 15 / 16);
    mpp_enc_cfg_set_s32(enc_cfg_, "rc:fps_in_flex", 0);
    mpp_enc_cfg_set_s32(enc_cfg_, "rc:fps_in_num", fps_num_);
    mpp_enc_cfg_set_s32(enc_cfg_, "rc:fps_in_denorm", fps_den_);
    mpp_enc_cfg_set_s32(enc_cfg_, "rc:fps_out_flex", 0);
    mpp_enc_cfg_set_s32(enc_cfg_, "rc:fps_out_num", fps_num_);
    mpp_enc_cfg_set_s32(enc_cfg_, "rc:fps_out_denorm", fps_den_);
    mpp_enc_cfg_set_s32(enc_cfg_, "rc:gop", std::max(1, fps_num_ / fps_den_));
    mpp_enc_cfg_set_s32(enc_cfg_, "rc:max_reenc_times", 0);

    mpp_enc_cfg_set_s32(enc_cfg_, "codec:type", MPP_VIDEO_CodingAVC);
    mpp_enc_cfg_set_s32(enc_cfg_, "h264:stream_type", 0);
    mpp_enc_cfg_set_s32(enc_cfg_, "h264:profile", 100);
    mpp_enc_cfg_set_s32(enc_cfg_, "h264:level", 40);
    mpp_enc_cfg_set_s32(enc_cfg_, "h264:cabac_en", 1);
    mpp_enc_cfg_set_s32(enc_cfg_, "h264:cabac_idc", 0);

    ret = mpi_->control(mpp_ctx_, MPP_ENC_SET_CFG, enc_cfg_);
    if (ret != MPP_OK) {
        fprintf(stderr, "RTSP encoder: MPP_ENC_SET_CFG failed ret=%d\n", ret);
        return -1;
    }

    ret = mpp_buffer_group_get_internal(&input_group_, MPP_BUFFER_TYPE_ION);
    if (ret != MPP_OK) {
        fprintf(stderr, "RTSP encoder: mpp_buffer_group_get_internal failed ret=%d\n", ret);
        return -1;
    }

    const size_t frame_bytes =
        static_cast<size_t>(hor_stride_) * static_cast<size_t>(ver_stride_) * 3U / 2U;
    ret = mpp_buffer_get(input_group_, &input_buffer_, frame_bytes);
    if (ret != MPP_OK || !input_buffer_) {
        fprintf(stderr, "RTSP encoder: mpp_buffer_get failed ret=%d size=%zu\n", ret, frame_bytes);
        return -1;
    }

    return 0;
}

int MppRtspEncoder::loadHeadersIntoStream() {
    if (!video_stream_) {
        fprintf(stderr, "RTSP encoder: video stream not initialized for headers\n");
        return -1;
    }

    std::vector<uint8_t> header_storage(kEncoderHeaderBufferSize);
    MppPacket header_packet = nullptr;
    bool own_header_packet = false;

    MPP_RET ret = mpp_packet_init(&header_packet, header_storage.data(), header_storage.size());
    if (ret == MPP_OK && header_packet) {
        own_header_packet = true;
        mpp_packet_set_length(header_packet, 0);
        ret = mpi_->control(mpp_ctx_, MPP_ENC_GET_HDR_SYNC, header_packet);
    }

    if (ret != MPP_OK || !header_packet || mpp_packet_get_length(header_packet) == 0) {
        if (own_header_packet) {
            mpp_packet_deinit(&header_packet);
            own_header_packet = false;
        }
        header_packet = nullptr;
        ret = mpi_->control(mpp_ctx_, MPP_ENC_GET_EXTRA_INFO, &header_packet);
    }
    if (ret != MPP_OK || !header_packet) {
        fprintf(stderr, "RTSP encoder: failed to fetch H.264 headers ret=%d\n", ret);
        return -1;
    }

    const void* header_ptr = mpp_packet_get_pos(header_packet);
    const size_t header_len = mpp_packet_get_length(header_packet);
    if (!header_ptr || header_len == 0) {
        if (own_header_packet) {
            mpp_packet_deinit(&header_packet);
        }
        fprintf(stderr, "RTSP encoder: invalid encoder header packet\n");
        return -1;
    }

    uint8_t* extradata = static_cast<uint8_t*>(
        av_malloc(header_len + AV_INPUT_BUFFER_PADDING_SIZE));
    if (!extradata) {
        if (own_header_packet) {
            mpp_packet_deinit(&header_packet);
        }
        fprintf(stderr, "RTSP encoder: av_malloc extradata failed\n");
        return -1;
    }

    memcpy(extradata, header_ptr, header_len);
    memset(extradata + header_len, 0, AV_INPUT_BUFFER_PADDING_SIZE);
    video_stream_->codecpar->extradata = extradata;
    video_stream_->codecpar->extradata_size = static_cast<int>(header_len);

    if (own_header_packet) {
        mpp_packet_deinit(&header_packet);
    }
    return 0;
}

int MppRtspEncoder::initRtspOutput() {
 avformat_network_init();

 int ret = avformat_alloc_output_context2(&fmt_ctx_, nullptr, "rtsp", rtsp_url_.c_str());
 if (ret < 0 || !fmt_ctx_) {
 fprintf(stderr, "RTSP encoder: avformat_alloc_output_context2 failed ret=%d\n", ret);
 return -1;
    }

    video_stream_ = avformat_new_stream(fmt_ctx_, nullptr);
    if (!video_stream_) {
        fprintf(stderr, "RTSP encoder: avformat_new_stream failed\n");
        return -1;
    }

    video_stream_->time_base = AVRational{1, 90000};
    video_stream_->avg_frame_rate = AVRational{fps_num_, fps_den_};
    video_stream_->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    video_stream_->codecpar->codec_id = AV_CODEC_ID_H264;
    video_stream_->codecpar->width = width_;
    video_stream_->codecpar->height = height_;
    video_stream_->codecpar->format = AV_PIX_FMT_NV12;
    video_stream_->codecpar->bit_rate = bitrate_bps_;

    if (loadHeadersIntoStream() != 0) {
        return -1;
    }

    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "muxdelay", "0", 0);
    av_dict_set(&opts, "pkt_size", "1200", 0);

    if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open2(&fmt_ctx_->pb, rtsp_url_.c_str(), AVIO_FLAG_WRITE, nullptr, &opts);
        if (ret < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
            av_strerror(ret, errbuf, sizeof(errbuf));
            fprintf(stderr, "RTSP encoder: avio_open2 failed: %s\n", errbuf);
            av_dict_free(&opts);
            return -1;
        }
        io_opened_ = true;
    }

    fmt_ctx_->flags |= AVFMT_FLAG_FLUSH_PACKETS;
    ret = avformat_write_header(fmt_ctx_, &opts);
    av_dict_free(&opts);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
        av_strerror(ret, errbuf, sizeof(errbuf));
        fprintf(stderr, "RTSP encoder: avformat_write_header failed: %s\n", errbuf);
        return -1;
    }
    header_written_ = true;

 return 0;
}

int MppRtspEncoder::reopenForFrame(const StreamFrame& frame) {
 if (rtsp_url_.empty() || frame.width <= 0 || frame.height <= 0 || frame.stride < frame.width) {
 return -1;
 }

 const std::string reopen_url = rtsp_url_;
 const int fps_num = fps_num_;
 const int fps_den = fps_den_;
 const int bitrate_bps = bitrate_bps_;
 printf("[INFO] RTSP encoder reopen for geometry=%dx%d stride=%d\n",
        frame.width, frame.height, frame.stride);
 close();
 return open(reopen_url.c_str(),
             frame.width,
             frame.height,
             frame.stride,
             frame.height,
             fps_num,
             fps_den,
             bitrate_bps);
}

int MppRtspEncoder::writeMppPacket(MppPacket packet, bool mark_keyframe) {
    if (!fmt_ctx_ || !video_stream_ || !packet) {
        return -1;
    }

    const void* src_ptr = mpp_packet_get_pos(packet);
    const size_t src_len = mpp_packet_get_length(packet);
    if (!src_ptr || src_len == 0) {
        return 0;
    }

    AVPacket* avpkt = av_packet_alloc();
    if (!avpkt) {
        fprintf(stderr, "RTSP encoder: av_packet_alloc failed\n");
        return -1;
    }

    int ret = av_new_packet(avpkt, static_cast<int>(src_len));
    if (ret < 0) {
        av_packet_free(&avpkt);
        fprintf(stderr, "RTSP encoder: av_new_packet failed ret=%d\n", ret);
        return -1;
    }

    memcpy(avpkt->data, src_ptr, src_len);
    avpkt->stream_index = video_stream_->index;

    const long long capture_ts_us = mpp_packet_get_pts(packet);
    if (first_capture_ts_us_ < 0) {
        first_capture_ts_us_ = capture_ts_us;
    }
    const long long rel_ts_us = std::max(0LL, capture_ts_us - first_capture_ts_us_);
    avpkt->pts = av_rescale_q(rel_ts_us, AVRational{1, 1000000}, video_stream_->time_base);
    avpkt->dts = avpkt->pts;

    long long duration_us = 1000000LL * fps_den_ / std::max(fps_num_, 1);
    if (last_capture_ts_us_ >= 0 && capture_ts_us > last_capture_ts_us_) {
        duration_us = capture_ts_us - last_capture_ts_us_;
    }
    last_capture_ts_us_ = capture_ts_us;
    avpkt->duration = av_rescale_q(duration_us, AVRational{1, 1000000}, video_stream_->time_base);
    if (avpkt->duration <= 0) {
        avpkt->duration = 1;
    }

    if (mark_keyframe) {
        avpkt->flags |= AV_PKT_FLAG_KEY;
    }

    ret = av_interleaved_write_frame(fmt_ctx_, avpkt);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
        av_strerror(ret, errbuf, sizeof(errbuf));
        fprintf(stderr, "RTSP encoder: av_interleaved_write_frame failed: %s\n", errbuf);
        av_packet_free(&avpkt);
        return -1;
    }

    av_packet_free(&avpkt);
    return 0;
}

int MppRtspEncoder::encodeAndPush(const StreamFrame& frame) {
    if (!frame.data || frame.format != RK_FORMAT_YCbCr_420_SP ||
        frame.width <= 0 || frame.height <= 0 || frame.stride < frame.width) {
        return -1;
    }
    if (!input_buffer_ || !mpi_ || !mpp_ctx_ ||
        frame.width != width_ || frame.height != height_) {
        if (reopenForFrame(frame) != 0) {
            fprintf(stderr,
                    "RTSP encoder: failed to reopen for frame w=%d h=%d stride=%d\n",
                    frame.width, frame.height, frame.stride);
            return -1;
        }
    }

    const int src_stride = frame.stride > 0 ? frame.stride : frame.width;
    const size_t min_bytes = static_cast<size_t>(src_stride) * static_cast<size_t>(frame.height) * 3U / 2U;
    if (frame.data_size < min_bytes) {
        fprintf(stderr,
                "RTSP encoder: short frame payload size=%zu required=%zu\n",
                frame.data_size, min_bytes);
        return -1;
    }

    unsigned char* dst = static_cast<unsigned char*>(mpp_buffer_get_ptr(input_buffer_));
    if (!dst) {
        fprintf(stderr, "RTSP encoder: null MPP input buffer\n");
        return -1;
    }

    const size_t dst_bytes =
        static_cast<size_t>(hor_stride_) * static_cast<size_t>(ver_stride_) * 3U / 2U;
    memset(dst, 0, dst_bytes);

    const unsigned char* src = frame.data;
    for (int row = 0; row < height_; ++row) {
        memcpy(dst + row * hor_stride_, src + row * src_stride, width_);
    }

    const unsigned char* src_uv = src + static_cast<size_t>(src_stride) * static_cast<size_t>(height_);
    unsigned char* dst_uv = dst + static_cast<size_t>(hor_stride_) * static_cast<size_t>(ver_stride_);
    for (int row = 0; row < height_ / 2; ++row) {
        memcpy(dst_uv + row * hor_stride_, src_uv + row * src_stride, width_);
    }

    MppFrame mpp_frame = nullptr;
    MPP_RET ret = mpp_frame_init(&mpp_frame);
    if (ret != MPP_OK || !mpp_frame) {
        fprintf(stderr, "RTSP encoder: mpp_frame_init failed ret=%d\n", ret);
        return -1;
    }

    mpp_frame_set_width(mpp_frame, width_);
    mpp_frame_set_height(mpp_frame, height_);
    mpp_frame_set_hor_stride(mpp_frame, hor_stride_);
    mpp_frame_set_ver_stride(mpp_frame, ver_stride_);
    mpp_frame_set_fmt(mpp_frame, MPP_FMT_YUV420SP);
    mpp_frame_set_buffer(mpp_frame, input_buffer_);
    mpp_frame_set_pts(mpp_frame, frame.capture_ts_us);

    ret = mpi_->encode_put_frame(mpp_ctx_, mpp_frame);
    mpp_frame_deinit(&mpp_frame);
    if (ret != MPP_OK) {
        fprintf(stderr, "RTSP encoder: encode_put_frame failed ret=%d\n", ret);
        return -1;
    }

    int write_ret = 0;
    bool saw_packet = false;
    bool need_more_packets = true;
    while (need_more_packets) {
        MppPacket packet = nullptr;
        ret = mpi_->encode_get_packet(mpp_ctx_, &packet);
        if (ret != MPP_OK) {
            fprintf(stderr, "RTSP encoder: encode_get_packet failed ret=%d\n", ret);
            return -1;
        }

        if (!packet) {
            if (saw_packet) {
                fprintf(stderr,
                        "RTSP encoder: encode_get_packet returned null before frame drain completed\n");
                return -1;
            }
            break;
        }

        saw_packet = true;
        const bool frame_done =
            !mpp_packet_is_partition(packet) || mpp_packet_is_eoi(packet);
        write_ret = writeMppPacket(packet, first_capture_ts_us_ < 0);
        mpp_packet_deinit(&packet);
        if (write_ret != 0) {
            return write_ret;
        }

        need_more_packets = !frame_done;
    }

    return write_ret;
}

void MppRtspEncoder::close() {
    if (fmt_ctx_) {
        if (header_written_) {
            av_write_trailer(fmt_ctx_);
        }
        if (io_opened_ && !(fmt_ctx_->oformat->flags & AVFMT_NOFILE) && fmt_ctx_->pb) {
            avio_closep(&fmt_ctx_->pb);
        }
        avformat_free_context(fmt_ctx_);
        fmt_ctx_ = nullptr;
        video_stream_ = nullptr;
    }

    if (input_buffer_) {
        mpp_buffer_put(input_buffer_);
        input_buffer_ = nullptr;
    }
    if (input_group_) {
        mpp_buffer_group_put(input_group_);
        input_group_ = nullptr;
    }
    if (enc_cfg_) {
        mpp_enc_cfg_deinit(enc_cfg_);
        enc_cfg_ = nullptr;
    }
    if (mpp_ctx_) {
        if (mpi_) {
            mpi_->reset(mpp_ctx_);
        }
        mpp_destroy(mpp_ctx_);
        mpp_ctx_ = nullptr;
        mpi_ = nullptr;
    }

    avformat_network_deinit();

    rtsp_url_.clear();
    width_ = 0;
    height_ = 0;
    hor_stride_ = 0;
    ver_stride_ = 0;
    bitrate_bps_ = 0;
    first_capture_ts_us_ = -1;
    last_capture_ts_us_ = -1;
    io_opened_ = false;
    header_written_ = false;
}

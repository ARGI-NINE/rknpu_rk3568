// mpp_decoder.cc — RK3568 hardware video decoder
// Pipeline: MP4 → ffmpeg demux → AVCC→Annex-B → MPP H.264/H.265 decode → NV12 frames

#include "mpp_decoder.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

MppDecoder::MppDecoder()
    : fmt_ctx_(nullptr), bsf_ctx_(nullptr), pkt_(nullptr), filtered_pkt_(nullptr),
      video_stream_idx_(-1),
      mpp_ctx_(nullptr), mpi_(nullptr), frm_grp_(nullptr),
      video_width_(0), video_height_(0), fps_num_(30), fps_den_(1),
      frame_buf_(nullptr), frame_buf_size_(0),
    eos_sent_(false), eos_got_(false) {}

MppDecoder::~MppDecoder() {
    close();
}

// ---------------------------------------------------------------------------
// open(): avformat demux + MPP decoder init
// ---------------------------------------------------------------------------

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
    AVStream *video_stream = fmt_ctx_->streams[video_stream_idx_];
    video_width_  = codecpar->width;
    video_height_ = codecpar->height;
    if (video_stream->avg_frame_rate.num > 0 && video_stream->avg_frame_rate.den > 0) {
        fps_num_ = video_stream->avg_frame_rate.num;
        fps_den_ = video_stream->avg_frame_rate.den;
    } else if (video_stream->r_frame_rate.num > 0 && video_stream->r_frame_rate.den > 0) {
        fps_num_ = video_stream->r_frame_rate.num;
        fps_den_ = video_stream->r_frame_rate.den;
    } else {
        fps_num_ = 30;
        fps_den_ = 1;
    }

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

// ---------------------------------------------------------------------------
// close(): tear down everything
// ---------------------------------------------------------------------------

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
    fps_num_ = 30;
    fps_den_ = 1;
    eos_sent_ = false;
    eos_got_  = false;
}

// ---------------------------------------------------------------------------
// feedPacket(): push one compressed packet into MPP (may retry on BUFFER_FULL)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// sendEos(): signal end-of-stream to the decoder
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// readFrame(): decode next frame → NV12 output
// ---------------------------------------------------------------------------

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

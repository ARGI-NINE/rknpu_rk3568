#ifndef MPP_ENCODER_RTSP_H
#define MPP_ENCODER_RTSP_H

#include <cstdint>
#include <string>

#include "frame_pools.h"
#include "rockchip/rk_mpi.h"
#include "rockchip/mpp_frame.h"
#include "rockchip/mpp_packet.h"
#include "rockchip/mpp_buffer.h"
#include "rockchip/rk_venc_cfg.h"

struct AVFormatContext;
struct AVStream;

class MppRtspEncoder {
public:
    MppRtspEncoder();
    ~MppRtspEncoder();

    int open(const char* rtsp_url,
             int width,
             int height,
             int hor_stride,
             int ver_stride,
             int fps_num,
             int fps_den,
             int bitrate_bps);
    int encodeAndPush(const StreamFrame& frame);
    void close();

private:
    int initMpp();
    int initRtspOutput();
    int loadHeadersIntoStream();
    int reopenForFrame(const StreamFrame& frame);
    int writeMppPacket(MppPacket packet, bool mark_keyframe);

    std::string rtsp_url_;
    int width_{0};
    int height_{0};
    int hor_stride_{0};
    int ver_stride_{0};
    int fps_num_{30};
    int fps_den_{1};
    int bitrate_bps_{0};
    long long first_capture_ts_us_{-1};
    long long last_capture_ts_us_{-1};
    bool io_opened_{false};
    bool header_written_{false};

    MppCtx mpp_ctx_{nullptr};
    MppApi* mpi_{nullptr};
    MppEncCfg enc_cfg_{nullptr};
    MppBufferGroup input_group_{nullptr};
    MppBuffer input_buffer_{nullptr};

    AVFormatContext* fmt_ctx_{nullptr};
    AVStream* video_stream_{nullptr};
};

#endif

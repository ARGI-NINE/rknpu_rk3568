#ifndef MPP_DECODER_H
#define MPP_DECODER_H

#include "rockchip/rk_mpi.h"
#include "rockchip/mpp_frame.h"
#include "rockchip/mpp_packet.h"
#include "rockchip/mpp_buffer.h"

// Forward declarations (ffmpeg)
struct AVFormatContext;
struct AVPacket;
struct AVBSFContext;

/**
 * MppDecoder: Hardware video decoder for RK3568
 *
 * Pipeline: MP4 file → ffmpeg demux → AVCC→Annex-B BSF → MPP H.264/H.265 decode → NV12 frames
 *
 * Usage:
 *   MppDecoder dec;
 *   dec.open("video.mp4");
 *   unsigned char *data; int w, h;
 *   while (dec.readFrame(&data, &w, &h) == 0) {
 *       // data is NV12, valid until next readFrame() or close()
 *   }
 *   dec.close();
 */
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
    int  getFpsNum() const { return fps_num_; }
    int  getFpsDen() const { return fps_den_; }

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
    int    fps_num_;
    int    fps_den_;

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

#endif

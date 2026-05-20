#include <stdio.h>
#include <string.h>
#include "im2d.h"
#include "im2d_buffer.h"
#include "rga.h"
#include "preprocess.h"

namespace {

constexpr int kRgaAlign = 8;

int run_rga_improcess(const char* op_name,
                      rga_buffer_t src,
                      int src_w,
                      int src_h,
                      rga_buffer_t dst,
                      int dst_w,
                      int dst_h,
                      int dst_fmt) {
    if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        fprintf(stderr, "%s invalid geometry src=%dx%d dst=%dx%d\n",
                op_name, src_w, src_h, dst_w, dst_h);
        return -1;
    }
    if (dst_fmt == RK_FORMAT_YCbCr_420_SP && (((dst_w | dst_h) & 1) != 0)) {
        fprintf(stderr, "%s requires even NV12 geometry (%dx%d)\n",
                op_name, dst_w, dst_h);
        return -1;
    }

    rga_buffer_t pat;
    im_rect src_rect;
    im_rect dst_rect;
    im_rect pat_rect;
    im_opt_t opt;
    memset(&pat, 0, sizeof(pat));
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    memset(&pat_rect, 0, sizeof(pat_rect));
    memset(&opt, 0, sizeof(opt));

    src_rect.width = src_w;
    src_rect.height = src_h;
    dst_rect.width = dst_w;
    dst_rect.height = dst_h;

    int ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret) {
        fprintf(stderr, "%s imcheck error: %s\n", op_name, imStrError((IM_STATUS)ret));
        return -1;
    }

    IM_STATUS status = improcess(src, dst, pat,
                                 src_rect, dst_rect, pat_rect,
                                 -1, nullptr, &opt, IM_SYNC);
    if (status != IM_STATUS_SUCCESS) {
        fprintf(stderr, "%s improcess error: %s\n", op_name, imStrError(status));
        return -1;
    }

    return 0;
}

}  // namespace

int rga_resize_convert(void *src_data, int src_w, int src_h, int src_fmt,
                       int dst_fd, int dst_w, int dst_h, int dst_fmt)
{
    if (!src_data || dst_fd < 0) {
        fprintf(stderr, "RGA resize-convert(fd) invalid args\n");
        return -1;
    }

    rga_buffer_t src;
    rga_buffer_t dst;
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));

    // Source: virtual address from user-space buffer
    src = wrapbuffer_virtualaddr(src_data, src_w, src_h, src_fmt);

    // Destination: DMA fd from rknn_create_mem (zero-copy to NPU)
    int wstride = dst_w + (kRgaAlign - dst_w % kRgaAlign) % kRgaAlign;
    int hstride = dst_h;
    dst = wrapbuffer_fd_t(dst_fd, dst_w, dst_h, wstride, hstride, dst_fmt);

    return run_rga_improcess("RGA resize-convert(fd)",
                             src, src_w, src_h,
                             dst, dst_w, dst_h, dst_fmt);
}

int rga_resize_convert_vaddr(void *src_data, int src_w, int src_h, int src_fmt,
                             void *dst_data, int dst_w, int dst_h, int dst_fmt)
{
    if (!src_data || !dst_data) {
        fprintf(stderr, "RGA resize-convert(vaddr) invalid args\n");
        return -1;
    }

    rga_buffer_t src;
    rga_buffer_t dst;
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));

    src = wrapbuffer_virtualaddr(src_data, src_w, src_h, src_fmt);
    dst = wrapbuffer_virtualaddr(dst_data, dst_w, dst_h, dst_fmt);

    return run_rga_improcess("RGA resize-convert(vaddr)",
                             src, src_w, src_h,
                             dst, dst_w, dst_h, dst_fmt);
}

int rga_resize_to_nv12_vaddr(void *src_data, int src_w, int src_h,
                             int src_wstride, int src_hstride, int src_fmt,
                             void *dst_data, int dst_w, int dst_h,
                             int dst_wstride, int dst_hstride)
{
    if (!src_data || !dst_data || src_w <= 0 || src_h <= 0 ||
        dst_w <= 0 || dst_h <= 0 ||
        src_wstride < src_w || src_hstride < src_h ||
        dst_wstride < dst_w || dst_hstride < dst_h) {
        fprintf(stderr, "RGA resize-to-nv12 invalid args\n");
        return -1;
    }

    rga_buffer_t src;
    rga_buffer_t dst;
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));

    src = wrapbuffer_virtualaddr_t(src_data, src_w, src_h,
                                   src_wstride, src_hstride, src_fmt);
    dst = wrapbuffer_virtualaddr_t(dst_data, dst_w, dst_h,
                                   dst_wstride, dst_hstride, RK_FORMAT_YCbCr_420_SP);

    return run_rga_improcess("RGA resize-to-nv12",
                             src, src_w, src_h,
                             dst, dst_w, dst_h, RK_FORMAT_YCbCr_420_SP);
}

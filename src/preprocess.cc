#include <stdio.h>
#include <string.h>
#include "im2d.h"
#include "rga.h"
#include "preprocess.h"

#define RGA_ALIGN 8

int rga_resize_convert(void *src_data, int src_w, int src_h, int src_fmt,
                       int dst_fd, int dst_w, int dst_h, int dst_fmt)
{
    rga_buffer_t src;
    rga_buffer_t dst;
    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));

    // Source: virtual address from user-space buffer
    src = wrapbuffer_virtualaddr(src_data, src_w, src_h, src_fmt);

    // Destination: DMA fd from rknn_create_mem (zero-copy to NPU)
    int wstride = dst_w + (RGA_ALIGN - dst_w % RGA_ALIGN) % RGA_ALIGN;
    int hstride = dst_h;
    dst = wrapbuffer_fd_t(dst_fd, dst_w, dst_h, wstride, hstride, dst_fmt);

    int ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret) {
        fprintf(stderr, "RGA imcheck error: %s\n", imStrError((IM_STATUS)ret));
        return -1;
    }

    IM_STATUS status = imresize(src, dst);
    if (status != IM_STATUS_SUCCESS) {
        fprintf(stderr, "RGA imresize error: %s\n", imStrError(status));
        return -1;
    }

    return 0;
}

int rga_resize_convert_vaddr(void *src_data, int src_w, int src_h, int src_fmt,
                             void *dst_data, int dst_w, int dst_h, int dst_fmt)
{
    rga_buffer_t src;
    rga_buffer_t dst;
    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));

    src = wrapbuffer_virtualaddr(src_data, src_w, src_h, src_fmt);
    dst = wrapbuffer_virtualaddr(dst_data, dst_w, dst_h, dst_fmt);

    int ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret) {
        fprintf(stderr, "RGA imcheck error: %s\n", imStrError((IM_STATUS)ret));
        return -1;
    }

    IM_STATUS status = imresize(src, dst);
    if (status != IM_STATUS_SUCCESS) {
        fprintf(stderr, "RGA imresize error: %s\n", imStrError(status));
        return -1;
    }

    return 0;
}

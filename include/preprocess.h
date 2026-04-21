#ifndef _RKNN_YOLOV5_RK3568_PREPROCESS_H_
#define _RKNN_YOLOV5_RK3568_PREPROCESS_H_

#include "im2d.h"
#include "rga.h"

// RGA hardware-accelerated resize + color conversion (zero-copy to NPU DMA memory)
// src_data: source frame virtual address
// src_w, src_h: source dimensions
// src_fmt: RGA source format (RK_FORMAT_BGR_888, RK_FORMAT_YUYV_422, etc.)
// dst_fd: destination DMA file descriptor (from rknn_create_mem()->fd)
// dst_w, dst_h: model input dimensions
// dst_fmt: RGA destination format (RK_FORMAT_RGB_888)
// Returns 0 on success
int rga_resize_convert(void *src_data, int src_w, int src_h, int src_fmt,
                       int dst_fd, int dst_w, int dst_h, int dst_fmt);

// Fallback version using virtual address for destination
int rga_resize_convert_vaddr(void *src_data, int src_w, int src_h, int src_fmt,
                             void *dst_data, int dst_w, int dst_h, int dst_fmt);

#endif

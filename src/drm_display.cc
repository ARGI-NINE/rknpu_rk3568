#include "drm_display.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <errno.h>

#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm/drm_fourcc.h>

/* ------------------------------------------------------------------ */
/* 5x7 bitmap font for ASCII 32..126                                  */
/* Each character = 5 bytes (5 columns). Each byte encodes 7 rows,    */
/* bit-0 = top row.                                                    */
/* ------------------------------------------------------------------ */
static const uint8_t font5x7[95][5] = {
    /* 32  ' ' */ {0x00, 0x00, 0x00, 0x00, 0x00},
    /* 33  '!' */ {0x00, 0x00, 0x5F, 0x00, 0x00},
    /* 34  '"' */ {0x00, 0x07, 0x00, 0x07, 0x00},
    /* 35  '#' */ {0x14, 0x7F, 0x14, 0x7F, 0x14},
    /* 36  '$' */ {0x24, 0x2A, 0x7F, 0x2A, 0x12},
    /* 37  '%' */ {0x23, 0x13, 0x08, 0x64, 0x62},
    /* 38  '&' */ {0x36, 0x49, 0x55, 0x22, 0x50},
    /* 39  ''' */ {0x00, 0x05, 0x03, 0x00, 0x00},
    /* 40  '(' */ {0x00, 0x1C, 0x22, 0x41, 0x00},
    /* 41  ')' */ {0x00, 0x41, 0x22, 0x1C, 0x00},
    /* 42  '*' */ {0x14, 0x08, 0x3E, 0x08, 0x14},
    /* 43  '+' */ {0x08, 0x08, 0x3E, 0x08, 0x08},
    /* 44  ',' */ {0x00, 0x50, 0x30, 0x00, 0x00},
    /* 45  '-' */ {0x08, 0x08, 0x08, 0x08, 0x08},
    /* 46  '.' */ {0x00, 0x60, 0x60, 0x00, 0x00},
    /* 47  '/' */ {0x20, 0x10, 0x08, 0x04, 0x02},
    /* 48  '0' */ {0x3E, 0x51, 0x49, 0x45, 0x3E},
    /* 49  '1' */ {0x00, 0x42, 0x7F, 0x40, 0x00},
    /* 50  '2' */ {0x42, 0x61, 0x51, 0x49, 0x46},
    /* 51  '3' */ {0x21, 0x41, 0x45, 0x4B, 0x31},
    /* 52  '4' */ {0x18, 0x14, 0x12, 0x7F, 0x10},
    /* 53  '5' */ {0x27, 0x45, 0x45, 0x45, 0x39},
    /* 54  '6' */ {0x3C, 0x4A, 0x49, 0x49, 0x30},
    /* 55  '7' */ {0x01, 0x71, 0x09, 0x05, 0x03},
    /* 56  '8' */ {0x36, 0x49, 0x49, 0x49, 0x36},
    /* 57  '9' */ {0x06, 0x49, 0x49, 0x29, 0x1E},
    /* 58  ':' */ {0x00, 0x36, 0x36, 0x00, 0x00},
    /* 59  ';' */ {0x00, 0x56, 0x36, 0x00, 0x00},
    /* 60  '<' */ {0x08, 0x14, 0x22, 0x41, 0x00},
    /* 61  '=' */ {0x14, 0x14, 0x14, 0x14, 0x14},
    /* 62  '>' */ {0x00, 0x41, 0x22, 0x14, 0x08},
    /* 63  '?' */ {0x02, 0x01, 0x51, 0x09, 0x06},
    /* 64  '@' */ {0x32, 0x49, 0x79, 0x41, 0x3E},
    /* 65  'A' */ {0x7E, 0x11, 0x11, 0x11, 0x7E},
    /* 66  'B' */ {0x7F, 0x49, 0x49, 0x49, 0x36},
    /* 67  'C' */ {0x3E, 0x41, 0x41, 0x41, 0x22},
    /* 68  'D' */ {0x7F, 0x41, 0x41, 0x22, 0x1C},
    /* 69  'E' */ {0x7F, 0x49, 0x49, 0x49, 0x41},
    /* 70  'F' */ {0x7F, 0x09, 0x09, 0x09, 0x01},
    /* 71  'G' */ {0x3E, 0x41, 0x49, 0x49, 0x7A},
    /* 72  'H' */ {0x7F, 0x08, 0x08, 0x08, 0x7F},
    /* 73  'I' */ {0x00, 0x41, 0x7F, 0x41, 0x00},
    /* 74  'J' */ {0x20, 0x40, 0x41, 0x3F, 0x01},
    /* 75  'K' */ {0x7F, 0x08, 0x14, 0x22, 0x41},
    /* 76  'L' */ {0x7F, 0x40, 0x40, 0x40, 0x40},
    /* 77  'M' */ {0x7F, 0x02, 0x0C, 0x02, 0x7F},
    /* 78  'N' */ {0x7F, 0x04, 0x08, 0x10, 0x7F},
    /* 79  'O' */ {0x3E, 0x41, 0x41, 0x41, 0x3E},
    /* 80  'P' */ {0x7F, 0x09, 0x09, 0x09, 0x06},
    /* 81  'Q' */ {0x3E, 0x41, 0x51, 0x21, 0x5E},
    /* 82  'R' */ {0x7F, 0x09, 0x19, 0x29, 0x46},
    /* 83  'S' */ {0x46, 0x49, 0x49, 0x49, 0x31},
    /* 84  'T' */ {0x01, 0x01, 0x7F, 0x01, 0x01},
    /* 85  'U' */ {0x3F, 0x40, 0x40, 0x40, 0x3F},
    /* 86  'V' */ {0x1F, 0x20, 0x40, 0x20, 0x1F},
    /* 87  'W' */ {0x3F, 0x40, 0x38, 0x40, 0x3F},
    /* 88  'X' */ {0x63, 0x14, 0x08, 0x14, 0x63},
    /* 89  'Y' */ {0x07, 0x08, 0x70, 0x08, 0x07},
    /* 90  'Z' */ {0x61, 0x51, 0x49, 0x45, 0x43},
    /* 91  '[' */ {0x00, 0x7F, 0x41, 0x41, 0x00},
    /* 92  '\' */ {0x02, 0x04, 0x08, 0x10, 0x20},
    /* 93  ']' */ {0x00, 0x41, 0x41, 0x7F, 0x00},
    /* 94  '^' */ {0x04, 0x02, 0x01, 0x02, 0x04},
    /* 95  '_' */ {0x40, 0x40, 0x40, 0x40, 0x40},
    /* 96  '`' */ {0x00, 0x01, 0x02, 0x04, 0x00},
    /* 97  'a' */ {0x20, 0x54, 0x54, 0x54, 0x78},
    /* 98  'b' */ {0x7F, 0x48, 0x44, 0x44, 0x38},
    /* 99  'c' */ {0x38, 0x44, 0x44, 0x44, 0x20},
    /* 100 'd' */ {0x38, 0x44, 0x44, 0x48, 0x7F},
    /* 101 'e' */ {0x38, 0x54, 0x54, 0x54, 0x18},
    /* 102 'f' */ {0x08, 0x7E, 0x09, 0x01, 0x02},
    /* 103 'g' */ {0x0C, 0x52, 0x52, 0x52, 0x3E},
    /* 104 'h' */ {0x7F, 0x08, 0x04, 0x04, 0x78},
    /* 105 'i' */ {0x00, 0x44, 0x7D, 0x40, 0x00},
    /* 106 'j' */ {0x20, 0x40, 0x44, 0x3D, 0x00},
    /* 107 'k' */ {0x7F, 0x10, 0x28, 0x44, 0x00},
    /* 108 'l' */ {0x00, 0x41, 0x7F, 0x40, 0x00},
    /* 109 'm' */ {0x7C, 0x04, 0x18, 0x04, 0x78},
    /* 110 'n' */ {0x7C, 0x08, 0x04, 0x04, 0x78},
    /* 111 'o' */ {0x38, 0x44, 0x44, 0x44, 0x38},
    /* 112 'p' */ {0x7C, 0x14, 0x14, 0x14, 0x08},
    /* 113 'q' */ {0x08, 0x14, 0x14, 0x18, 0x7C},
    /* 114 'r' */ {0x7C, 0x08, 0x04, 0x04, 0x08},
    /* 115 's' */ {0x48, 0x54, 0x54, 0x54, 0x20},
    /* 116 't' */ {0x04, 0x3F, 0x44, 0x40, 0x20},
    /* 117 'u' */ {0x3C, 0x40, 0x40, 0x20, 0x7C},
    /* 118 'v' */ {0x1C, 0x20, 0x40, 0x20, 0x1C},
    /* 119 'w' */ {0x3C, 0x40, 0x30, 0x40, 0x3C},
    /* 120 'x' */ {0x44, 0x28, 0x10, 0x28, 0x44},
    /* 121 'y' */ {0x0C, 0x50, 0x50, 0x50, 0x3C},
    /* 122 'z' */ {0x44, 0x64, 0x54, 0x4C, 0x44},
    /* 123 '{' */ {0x00, 0x08, 0x36, 0x41, 0x00},
    /* 124 '|' */ {0x00, 0x00, 0x7F, 0x00, 0x00},
    /* 125 '}' */ {0x00, 0x41, 0x36, 0x08, 0x00},
    /* 126 '~' */ {0x10, 0x08, 0x08, 0x10, 0x10},
};

/* ================================================================== */
/*  DrmDisplay implementation                                         */
/* ================================================================== */

DrmDisplay::DrmDisplay()
    : drm_fd_(-1), conn_id_(0), crtc_id_(0),
      width_(0), height_(0), front_(0),
      saved_crtc_(nullptr)
{
    for (int i = 0; i < 2; i++) {
        fb_id_[i]    = 0;
        bo_handle_[i]= 0;
        pitch_[i]    = 0;
        size_[i]     = 0;
        map_[i]      = nullptr;
    }
}

DrmDisplay::~DrmDisplay()
{
    deinit();
}

int DrmDisplay::init()
{
    const char* dev = "/dev/dri/card0";
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

    /* Create 2 dumb buffers */
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

void DrmDisplay::deinit()
{
    if (drm_fd_ < 0) return;

    /* Restore old CRTC */
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

/* ================================================================== */
/*  Standalone drawing helpers                                        */
/* ================================================================== */

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

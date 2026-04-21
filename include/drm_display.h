#ifndef _DRM_DISPLAY_H_
#define _DRM_DISPLAY_H_

#include <stdint.h>
#include <xf86drmMode.h>

class DrmDisplay {
public:
    DrmDisplay();
    ~DrmDisplay();

    int  init();
    void deinit();

    uint32_t* getBackBuffer();
    int  getWidth();
    int  getHeight();
    int  getStride();  // row stride in pixels (pitch/4), use as fb_w for drawing
    int  flip();
    void clearBackBuffer(uint32_t color);

private:
    int drm_fd_;
    uint32_t conn_id_;
    uint32_t crtc_id_;
    uint32_t fb_id_[2];
    uint32_t bo_handle_[2];
    uint32_t pitch_[2];
    uint32_t size_[2];
    uint32_t* map_[2];
    int width_;
    int height_;
    int front_;

    drmModeCrtcPtr saved_crtc_;
    drmModeModeInfo mode_;
};

/* Standalone C-style drawing helpers */
void drm_draw_rect(uint32_t* fb, int fb_w, int fb_h,
                   int x, int y, int w, int h,
                   uint32_t color, int thickness);

void drm_draw_fill_rect(uint32_t* fb, int fb_w, int fb_h,
                         int x, int y, int w, int h,
                         uint32_t color);

void drm_draw_char(uint32_t* fb, int fb_w, int fb_h,
                   int x, int y, char c,
                   uint32_t color, int scale);

void drm_draw_string(uint32_t* fb, int fb_w, int fb_h,
                     int x, int y, const char* str,
                     uint32_t color, int scale);

#endif /* _DRM_DISPLAY_H_ */

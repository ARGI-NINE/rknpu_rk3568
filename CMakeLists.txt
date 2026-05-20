cmake_minimum_required(VERSION 3.10)

project(rknn_yolov5_rk3568)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_rk3568_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Bundled dependencies
set(RKNN_PATH ${CMAKE_SOURCE_DIR}/3rdparty/rknn)
set(RKNN_RT_LIB ${RKNN_PATH}/lib/librknnrt.so)

set(RGA_PATH ${CMAKE_SOURCE_DIR}/3rdparty/rga)
set(RGA_LIB ${RGA_PATH}/lib/librga.so)

set(MPP_PATH ${CMAKE_SOURCE_DIR}/3rdparty/mpp)
set(MPP_LIB ${MPP_PATH}/lib/librockchip_mpp.so)

set(FFMPEG_ROOT "/opt/tool_chain/ffmpeg_tools")
set(LIBDRM_ROOT "/opt/tool_chain/libdrm_tools")

include_directories(${RKNN_PATH}/include)
include_directories(${RGA_PATH}/include)
include_directories(${MPP_PATH}/include)
include_directories(${MPP_PATH}/include/rockchip)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${FFMPEG_ROOT}/usr/include)
include_directories(${FFMPEG_ROOT}/include)
include_directories(${LIBDRM_ROOT}/include)
include_directories(${LIBDRM_ROOT}/include/libdrm)

link_directories(${FFMPEG_ROOT}/usr/lib/aarch64-linux-gnu)
link_directories(${FFMPEG_ROOT}/usr/lib)
link_directories(${FFMPEG_ROOT}/lib/aarch64-linux-gnu)
link_directories(${FFMPEG_ROOT}/lib)
link_directories(${LIBDRM_ROOT}/lib)
link_directories(${LIBDRM_ROOT}/lib/aarch64-linux-gnu)

add_executable(rknn_yolov5_rk3568
    src/main.cc
    src/rkYolov5s.cc
    src/preprocess.cc
    src/postprocess.cc
    src/v4l2_capture.cc
    src/mpp_decoder.cc
    src/mpp_encoder_rtsp.cc
    src/drm_display.cc
)

target_link_libraries(rknn_yolov5_rk3568
    ${RKNN_RT_LIB}
    ${RGA_LIB}
    ${MPP_LIB}
    avformat
    avcodec
    avutil
    drm
    pthread
)

set(CMAKE_INSTALL_RPATH "lib")
install(TARGETS rknn_yolov5_rk3568 DESTINATION ./)
install(PROGRAMS
    ${RKNN_RT_LIB}
    ${RGA_LIB}
    ${MPP_LIB}
    DESTINATION lib)
install(FILES
    ${CMAKE_SOURCE_DIR}/model/yolov5s-640-640.rknn
    ${CMAKE_SOURCE_DIR}/model/yolov5s_relu-640-640-rk3568.rknn
    ${CMAKE_SOURCE_DIR}/model/coco_80_labels_list.txt
    DESTINATION model)
install(DIRECTORY video/ DESTINATION ./video)

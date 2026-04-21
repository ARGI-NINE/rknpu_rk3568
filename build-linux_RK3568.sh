#!/bin/bash
set -e

ROOT_PWD=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR=$ROOT_PWD/build

mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../rk3568.toolchain.cmake \
  -DFFMPEG_ROOT=/opt/tool_chain/ffmpeg_tools \
  -DLIBDRM_ROOT=/opt/tool_chain/libdrm_tools

make -j4
make install

# RK3568 YOLOv5s 实时推理工程（rknn_yolov5_rk3568）

## 1. 项目定位与目标

`rknn_yolov5_rk3568` 是一个面向 RK3568 板端部署的端到端目标检测工程，不是单点 API 示例。项目目标是：

1. 在 RK3568 上打通摄像头与视频文件两条实时链路。
2. 通过 RGA + RKNN zero-copy + 并发流水线降低 CPU 参与度与内存搬运开销。
3. 提供可复用的工程化骨架（采集/解码/推理/显示/并发有明确边界）。

仓库主入口与关键实现：
- `src/main.cc`
- `src/rkYolov5s.cc`
- `include/rknnPool.hpp`
- `src/mpp_decoder.cc`
- `src/drm_display.cc`

## 2. 本仓关键实现要点（可核验）

以下条目仅基于本仓当前源码可核验事实，不依赖外部对比材料：

| 实现要点 | 当前状态 | 证据 |
|---|---|---|
| Camera 与 MP4 双链路共用同一推理/显示后段 | Implemented | `src/main.cc`、`src/mpp_decoder.cc` |
| 预处理由 RGA 执行（格式转换+缩放） | Implemented | `src/preprocess.cc`、`src/rkYolov5s.cc` |
| NPU I/O 采用 zero-copy（`rknn_create_mem + rknn_set_io_mem`） | Implemented | `src/rkYolov5s.cc` |
| `pool.get` 为条件变量阻塞等待 | Implemented | `include/rknnPool.hpp`（`resultCv_.wait(...)`） |
| 结果按 `nextGetSeq_` 顺序提交（256 槽环） | Implemented | `include/rknnPool.hpp` |
| `frame_id/src_ts_us` metadata 从采集到取结果全程透传 | Implemented | `src/main.cc`、`include/rknnPool.hpp` |
| 当前未输出 `latency_60_us/latency_total_us` | Observed | `src/main.cc`（日志仍以 `fps_60` 为主） |
| worker 线程数上限固定为 16（`kMaxWorkerThreads`） | Implemented | `src/main.cc` |

结论：当前实现重点是“可运行的端到端工程链路 + 顺序语义控制 + 板端可调并发”，并已具备持续收敛基础。

## 3. 最终完整链路

### 3.1 Camera 链路（实时）

`V4L2 -> capture_thread -> rknnPool(worker) -> main get(seq) -> render_thread -> DRM`

分阶段说明：
1. `V4L2Capture::captureFrame` 采集原始帧（`src/main.cc`）。
2. capture 线程将帧放入 `rknnPool::put`（带 `frame_id` 与时间戳）。
3. worker 内执行 `RGA preprocess + rknn_run + post_process`（`src/rkYolov5s.cc`）。
4. `rknnPool::get` 按 `nextGetSeq_` 顺序取结果（`include/rknnPool.hpp`）。
5. 渲染线程做 `RGA->BGRA + 画框 + drm.flip()`（`src/main.cc` + `src/drm_display.cc`）。

### 3.2 MP4 链路（实时）

`ffmpeg demux -> Annex-B BSF -> MPP decode(NV12) -> capture_thread -> rknnPool -> DRM`

分阶段说明：
1. `avformat_open_input` 打开容器，`av_bsf_*` 处理 H264/H265 码流（`src/mpp_decoder.cc`）。
2. MPP 硬解输出 NV12 帧（`MppDecoder::readFrame`）。
3. 后续路径与 Camera 模式共享同一推理与显示流水线。

## 4. 当前实现状态映射（源码对照）

本节仅依据本仓当前源码状态整理。

| 实现要点 | 当前状态 | 对应实现 |
|---|---|---|
| 去掉 RK3588 core_mask 绑定 | Implemented | `src/rkYolov5s.cc`（注释明确不调用 core_mask） |
| OpenCV 前处理下沉到 RGA | Implemented | `src/preprocess.cc` + `src/rkYolov5s.cc` |
| Zero-copy（create_mem/set_io_mem） | Implemented | `src/rkYolov5s.cc` |
| 乱序执行 + 顺序提交（256 环） | Implemented | `include/rknnPool.hpp` |
| `pool.get` 阻塞等待目标 seq | Implemented | `include/rknnPool.hpp`（条件变量 wait） |
| Worker 数量按板端特性调优 | Partial | 可通过 CLI 第 3 参数调优，受固定上限约束 |
| metadata 统计字段输出 | Partial | metadata 已透传，尚未产出 latency 指标日志 |

结论：核心优化主线已落地，当前处于“工程收敛与稳定性验证”阶段。

## 5. 收敛与当前状态

当前可见状态（低风险、行为保持）：
1. `main.cc` 已将主循环与 drain 阶段共享处理路径收敛为复用逻辑。
2. 渲染入队策略保持不变：队列满时丢弃最旧帧后再入队。

当前状态：
- 已完成：重复逻辑收敛与公共路径提炼。
- 已完成：任务队列阈值与渲染队列阈值保持硬编码（`4/2`）。
- 当前实现：`pool.get` 为条件变量阻塞等待，不使用固定休眠轮询。

## 6. 构建与运行

### 6.1 依赖

工程链接以下库：
- RKNN runtime（`3rdparty/rknn`）
- RGA（`3rdparty/rga`）
- MPP（`3rdparty/mpp`）
- ffmpeg：`avformat/avcodec/avutil`
- libdrm

构建脚本：
- `build-linux_RK3568.sh`

默认 CMake 配置关键项：
- `FFMPEG_ROOT=/opt/tool_chain/ffmpeg_tools`
- `LIBDRM_ROOT=/opt/tool_chain/libdrm_tools`

### 6.2 一键构建（推荐）

```bash
cd rknn_yolov5_rk3568
chmod +x build-linux_RK3568.sh
./build-linux_RK3568.sh
```

脚本执行流程：
1. `cmake .. -DCMAKE_TOOLCHAIN_FILE=../rk3568.toolchain.cmake`
2. `make -j4`
3. `make install`

产物目录：`install/rknn_yolov5_rk3568_Linux`

### 6.3 运行

在 RK3568 板端：

```bash
cd install/rknn_yolov5_rk3568_Linux
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
```

Camera 模式：

```bash
./rknn_yolov5_rk3568 ./model/yolov5s-640-640.rknn /dev/video0 1
```

MP4 模式：

```bash
./rknn_yolov5_rk3568 ./model/yolov5s-640-640.rknn ./video/test.mp4 1
```

参数说明：
- 第 1 个参数：`.rknn` 模型路径
- 第 2 个参数：输入源（`/dev/videoX` 或 mp4 文件）
- 第 3 个参数（可选）：worker 线程数，范围 `[1, 16]`（由 `src/main.cc` 的 `kMaxWorkerThreads` 常量约束，默认值 `1`）

### 6.4 当前固定运行常量（硬编码）

当前实现使用固定常量，不提供环境变量参数入口。

| 项目 | 固定值 | 作用 |
|---|---:|---|
| DRM 设备路径 | `/dev/dri/card0` | 固定使用 `card0`，不做自动探测。 |
| 渲染队列阈值 | `2` | 队列超限时丢弃最旧帧。 |
| 任务队列上限 | `4` | `rknnPool` 任务队列上限。 |
| `pool.get` 行为 | 条件变量阻塞等待 | 等待 `nextGetSeq_` 对应结果或 stop 通知；非固定 `1000us` 轮询。 |
| CLI worker 上限 | `16` | 由 `src/main.cc` 中 `kMaxWorkerThreads` 固定约束。 |
| 摄像头默认分辨率 | `640x480` | 默认请求宽高。 |
| V4L2 `REQBUFS` 数量 | `4` | V4L2 缓冲区请求数量。 |

### 6.5 frame metadata 现状（`frame_id/src_ts_us`）

当前实现状态：
1. `frame_id/src_ts_us` 保持从采集到 `pool.get` 的透传。
2. 主循环与 drain 阶段当前仅保留该字段并显式 `(void)`，未用于延迟统计。
3. 当前性能日志仅输出 `fps_60`，未输出 `latency_60_us` / `latency_total_us`。

该状态不改变推理与显示主流程，metadata 已保留好后续扩展接口。

## 7. 已知问题（观测现象 + 待验证）

1. 观测现象：MP4 路径存在右侧残影/色块。待验证项：逐步核对解码输出参数、RGA 目标参数与显示提交参数的一致性。
2. 观测现象：RK3568 上提高 worker 数后 FPS 提升幅度存在波动。待验证项：按输入源、分辨率、线程数分组采样并复测。
3. 观测现象：交叉编译环境可能出现 pkg-config 依赖解析失败。待验证项：确认 `PKG_CONFIG_LIBDIR` 与 `PKG_CONFIG_SYSROOT_DIR` 配置是否完整。
4. 观测现象：MPP 仅承担解码职责。待验证项：维持“容器解复用 -> 码流转换 -> MPP 解码”路径，不引入离线转码步骤。

## 8. 文档索引

相关文档：
- `docs/three_way_diff_and_architecture.md`
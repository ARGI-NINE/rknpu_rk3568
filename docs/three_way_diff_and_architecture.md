# 架构链路与实现状态快照（本仓）

## 1. 文档范围

本文件仅描述本仓当前可核验的实现状态，不依赖外部仓目录、历史对比材料或跨仓优劣判断。

目标：
1. 给出 Camera/MP4 两条链路的可复核架构说明。
2. 汇总关键实现状态，便于后续收敛与验证。
3. 明确当前已实现项与待补充项，避免超出证据范围的结论。

## 2. 架构链路

### 2.1 Camera 链路

```text
/dev/videoX
  -> V4L2Capture::captureFrame
  -> capture_thread (main.cc)
  -> rknnPool::put
  -> worker: RGA preprocess + rknn_run + post_process
  -> rknnPool::get(nextGetSeq)
  -> enqueue render queue
  -> render_thread: RGA display convert + draw + DRM flip
```

链路特征：
1. 采集线程与渲染线程分离，主循环聚焦结果消费与显示提交。
2. 推理端允许并行执行，输出端按 seq 顺序消费，避免乱序展示。
3. 显示路径直接走 DRM，不依赖 GUI 框架。

### 2.2 MP4 链路

```text
MP4 file
  -> ffmpeg demux (avformat)
  -> H264/H265 AVCC->Annex-B (bsf)
  -> MPP decode to NV12
  -> capture_thread push to rknnPool
  -> (后续与 Camera 链路共享)
```

链路特征：
1. MPP 负责硬解，属于在线解码路径。
2. 一旦进入 `rknnPool`，后段与 Camera 模式统一，便于共用优化与排障手段。

## 3. 关键实现要点（可核验）

| 实现要点 | 状态 | 说明 |
|---|---|---|
| 去 core_mask 绑定策略 | Implemented | `rkYolov5s.cc` 注释明确 RK3568 单核 NPU 场景不调用 core_mask。 |
| RGA 执行前处理 | Implemented | 预处理阶段执行格式转换与缩放，减少 CPU 负担。 |
| RKNN zero-copy I/O | Implemented | 已使用 `rknn_create_mem/rknn_set_io_mem`。 |
| 乱序执行 + 顺序提交 | Implemented | `nextGetSeq_` 与环形槽位机制确保按序输出。 |
| `pool.get` 等待机制 | Implemented | `rknnPool::get` 使用条件变量阻塞等待目标 seq 就绪或 stop 通知，非固定休眠轮询。 |
| frame metadata 透传 | Implemented | `frame_id/src_ts_us` 在 `put -> ResultSlot -> get` 路径透传。 |
| latency 指标输出 | Partial | 当前日志仍以 `fps_60` 为主，尚未输出 `latency_60_us/latency_total_us`。 |
| worker 上限（CLI） | Implemented | 线程数可调，但受 `kMaxWorkerThreads=16` 上限约束。 |

## 4. 实现状态摘要

当前可确认状态：
1. 端到端主链路可闭环：输入采集/解码 -> 推理 -> 渲染。
2. 并发执行与顺序消费语义已经落地。
3. metadata 已保留在主路径中，但延迟字段仍未进入周期性日志输出。

当前收敛方向：
1. 在不改变主链路语义的前提下，继续补齐可观测性指标。
2. 将性能与稳定性结论建立在复测数据上，而非推断。

## 5. 观测与待验证

1. 观测现象：MP4 路径仍可见右侧残影/色块。待验证项：逐段比对解码输出参数、RGA 目标参数、显示提交参数。
2. 观测现象：不同 worker 配置下 FPS 增益存在波动。待验证项：按输入源、分辨率、线程数分层采样后复测。
3. 观测现象：metadata 已透传但延迟统计尚未输出。待验证项：在不改动主流程的前提下补齐 `latency_60_us/latency_total_us` 指标口径与输出时机。
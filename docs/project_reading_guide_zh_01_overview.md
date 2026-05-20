# 项目总览

这个项目是一个面向 RK3568 的 `C++14` 视觉推理工程。

它不是纯算法示例，而是完整工程链路：

`摄像头 / 视频文件 -> 采集 / 解码 -> 前处理 -> RKNN 推理 -> 后处理 -> DRM 显示`

## 1. 它要解决什么

项目把 YOLOv5 的推理过程，封装成一条可以在板端跑起来的流水线。

你在代码里看到的不是“单个模型推理函数”，而是几个模块一起工作：

- `main.cc` 负责调度。
- `rknnPool` 负责线程、队列和结果顺序。
- `rkYolov5s` 负责模型初始化、RGA 前处理、`rknn_run()` 和后处理。
- `V4L2Capture` 负责摄像头采集。
- `MppDecoder` 负责视频文件解码。
- `DrmDisplay` 负责显示。

这里先补一个平台前提：RK3568 只有一颗 NPU，工程里不做 core_mask 绑定，推理并发交给 `rknnPool` 统一调度。`rkYolov5s`、`V4L2Capture`、`MppDecoder` 和 `DrmDisplay` 不是为了把文件拆散，而是为了把 Linux 下驱动缓冲、DMA 内存和 page flip 的生命周期分别交给对应模块管理。

## 2. 构建方式

构建入口是 `build-linux_RK3568.sh`，底层走 CMake 和交叉编译工具链。

```bash
cmake .. -DCMAKE_TOOLCHAIN_FILE=../rk3568.toolchain.cmake
make -j4
make install
```

`rk3568.toolchain.cmake` 说明目标平台是 Linux / aarch64，编译器是 `aarch64-linux-gnu-gcc/g++`。

构建后会生成 `rknn_yolov5_rk3568` 可执行文件，并把模型、库和 `video/` 一起安装到 `install/rknn_yolov5_rk3568_Linux`。

## 3. 运行方式

`README.md` 里给出的运行方式大致是：

```bash
cd install/rknn_yolov5_rk3568_Linux
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
./rknn_yolov5_rk3568 <model_path> <camera_device_or_video> [thread_num]
```

第二个参数可以是：

- `/dev/videoX`，走摄像头采集。
- mp4 之类的视频文件，走 MPP 解码。

第三个参数是 worker 线程数，默认 1。

## 4. 这份项目的核心特点

### 4.1 不只是一条推理链路

它有两条输入分支：

- 摄像头分支
- 视频文件分支

最后都汇到同一条推理和显示链路。

### 4.2 不是“推理完直接显示”

中间还有这些关键步骤：

- 采集或者解码得到原始帧。
- 用 RGA 做缩放和颜色转换。
- 把输入写到 `rknn_create_mem()` 对应的 DMA 内存。
- 执行 `rknn_run()`。
- 做 `post_process()`。
- 把结果送进 DRM 双缓冲显示。

这里再补一个 Linux 侧原因：采集或解码得到的原始帧会先被复制到 `FrameCopyPool`，然后再经过 RGA 做缩放和颜色转换，最终写入 RKNN 申请的 DMA 内存；真正省掉的是 RGA 到 RKNN DMA 内存之间的额外 copy，而不是整条输入链路都没有 CPU copy。显示侧也一样，`DrmDisplay` 直接走 DRM/KMS 双缓冲和 page flip，把 back buffer 切到前台，避免 GUI 栈带来的额外开销和抖动。

### 4.3 资源全部按契约传递

这个工程里最重要的一个概念是“所有权转移”。

帧不是随便 `free()`，而是通过回调传递和归还。

这也是为什么你会看到：

- `release_fn`
- `release_ctx`
- `FrameCopyPool`
- `rknnPool::put/get`

这些都在保证同一块缓冲不会被重复释放。

## 5. 推荐先读什么

如果你是第一次看这个工程，建议按这个顺序：

1. `project_reading_guide_zh_02_main_and_pool.md`
2. `project_reading_guide_zh_03_backend.md`
3. `project_reading_guide_zh_04_input_display.md`
4. `project_reading_guide_zh_05_function_index.md`

## 6. 一句话总结

这个项目的本质是：

**把输入、推理、后处理、显示全部工程化，并通过任务池把帧和结果按序传递起来。**
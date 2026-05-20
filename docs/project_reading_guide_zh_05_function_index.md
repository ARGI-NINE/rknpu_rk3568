# 函数索引

这份文档是回查索引。

它不重复讲实现，只做两件事：

1. 告诉你每个关键函数在哪个文件。
2. 告诉你应该去前面哪份文档读它。

## 1. 主循环与任务池

这些函数已经在 `project_reading_guide_zh_02_main_and_pool.md` 里逐段讲过。

| 函数 | 文件 | 作用 |
| --- | --- | --- |
| `main()` | `src/main.cc` | 启动入口，参数解析、线程创建、主循环 |
| `is_camera_input_source()` | `src/main.cc` | 判断输入源是摄像头还是视频文件 |
| `now_wall_time_us()` | `src/main.cc` | 获取微秒时间戳 |
| `release_frame_buffer()` | `src/main.cc` | 统一帧释放出口 |
| `compute_frame_size_bytes()` | `src/main.cc` | 根据格式计算复制大小 |
| `enqueue_render_or_release()` | `src/main.cc` | 渲染队列入队或直接释放 |
| `render_thread_func()` | `src/main.cc` | 渲染消费者线程 |
| `copy_frame_and_submit_to_pool()` | `src/main.cc` | 采集线程把帧送进 `rknnPool` |
| `capture_thread_func()` | `src/main.cc` | 采集线程主逻辑 |
| `handle_get_success()` | `src/main.cc` | 主线程拿到结果后的分发逻辑 |
| `process_one_pool_result()` | `src/main.cc` | 主线程从池里消费一个结果 |
| `cleanup_and_return()` | `src/main.cc` | 统一退出收尾 |
| `FrameCopyPool` | `src/main.cc` | 复用采集帧缓冲 |
| `FrameTask / ResultSlot` | `include/rknnPool.hpp` | 任务和结果的数据契约 |
| `rknnPool::init()` | `include/rknnPool.hpp` | 创建模型实例并启动 worker |
| `rknnPool::put()` | `include/rknnPool.hpp` | 入队并转移帧所有权 |
| `rknnPool::workerFunc()` | `include/rknnPool.hpp` | 执行推理并写回结果槽 |
| `rknnPool::get()` | `include/rknnPool.hpp` | 顺序取出结果 |
| `rknnPool::notifyGetters()` | `include/rknnPool.hpp` | 唤醒等待中的消费者 |

## 2. 推理共享后段

这些函数已经在 `project_reading_guide_zh_03_backend.md` 里讲过。

| 函数 | 文件 | 作用 |
| --- | --- | --- |
| `rkYolov5s::init()` | `src/rkYolov5s.cc` | 加载模型、创建上下文、绑定 IO 内存 |
| `rkYolov5s::get_pctx()` | `src/rkYolov5s.cc` | 暴露内部 RKNN 上下文 |
| `rkYolov5s::infer()` | `src/rkYolov5s.cc` | 预处理、推理、后处理 |
| `build_label_path_from_model()` | `src/rkYolov5s.cc` | 从模型路径推导标签路径 |
| `load_model()` | `src/rkYolov5s.cc` | 读取模型文件到内存 |
| `setLabelNamePath()` | `src/postprocess.cc` | 设置标签文件路径 |
| `readLines()` | `src/postprocess.cc` | 逐行读取标签文件 |
| `loadLabelName()` | `src/postprocess.cc` | 把标签读入内存 |
| `process()` | `src/postprocess.cc` | 单个输出头解码 |
| `post_process()` | `src/postprocess.cc` | 三头输出合并、NMS、结果打包 |
| `deinitPostProcess()` | `src/postprocess.cc` | 释放标签缓存 |
| `rga_resize_convert()` | `src/preprocess.cc` | RGA zero-copy 预处理 |
| `rga_resize_convert_vaddr()` | `src/preprocess.cc` | RGA vaddr fallback |
| `detect_result_group_t` | `include/postprocess.h` | 检测结果最终结构 |

## 3. 输入与显示

这些函数已经在 `project_reading_guide_zh_04_input_display.md` 里讲过。

| 函数 | 文件 | 作用 |
| --- | --- | --- |
| `V4L2Capture::open()` | `src/v4l2_capture.cc` | 打开摄像头并做格式协商 |
| `V4L2Capture::initFormat()` | `src/v4l2_capture.cc` | 设置像素格式 |
| `V4L2Capture::initMmap()` | `src/v4l2_capture.cc` | 申请并映射驱动缓冲 |
| `V4L2Capture::startStream()` | `src/v4l2_capture.cc` | 开始采集 |
| `V4L2Capture::captureFrame()` | `src/v4l2_capture.cc` | 取帧 |
| `V4L2Capture::releaseFrame()` | `src/v4l2_capture.cc` | 还帧 |
| `V4L2Capture::stopStream()` | `src/v4l2_capture.cc` | 停流 |
| `V4L2Capture::close()` | `src/v4l2_capture.cc` | 释放摄像头资源 |
| `MppDecoder::open()` | `src/mpp_decoder.cc` | 打开视频文件并初始化解码器 |
| `MppDecoder::feedPacket()` | `src/mpp_decoder.cc` | 向 MPP 喂包 |
| `MppDecoder::sendEos()` | `src/mpp_decoder.cc` | 发送 EOS |
| `MppDecoder::readFrame()` | `src/mpp_decoder.cc` | 取出解码后的 NV12 帧 |
| `MppDecoder::close()` | `src/mpp_decoder.cc` | 清理 FFmpeg 和 MPP 资源 |
| `DrmDisplay::init()` | `src/drm_display.cc` | 初始化 DRM 和双缓冲 |
| `DrmDisplay::getBackBuffer()` | `src/drm_display.cc` | 获取背面缓冲 |
| `DrmDisplay::clearBackBuffer()` | `src/drm_display.cc` | 清空背面缓冲 |
| `DrmDisplay::flip()` | `src/drm_display.cc` | 页面翻转 |
| `DrmDisplay::deinit()` | `src/drm_display.cc` | 恢复并释放 DRM 资源 |
| `put_pixel()` | `src/drm_display.cc` | 最底层像素写入 |
| `drm_draw_rect()` | `src/drm_display.cc` | 画边框 |
| `drm_draw_fill_rect()` | `src/drm_display.cc` | 画实心矩形 |
| `drm_draw_char()` | `src/drm_display.cc` | 画单个字符 |
| `drm_draw_string()` | `src/drm_display.cc` | 画字符串 |

## 4. 怎么用这份索引

- 你想知道“这函数在哪”，先查这里。
- 你想知道“这函数怎么讲”，去对应的详细文档。
- 你想知道“这一类函数在链路里承担什么角色”，先回 `02_main_and_pool` 或 `03_backend` 或 `04_input_display`。

## 5. 读代码时的优先级

如果你只想抓最关键的函数，优先顺序是：

1. `main()`
2. `capture_thread_func()`
3. `rknnPool::put()`
4. `rknnPool::workerFunc()`
5. `rknnPool::get()`
6. `rkYolov5s::infer()`
7. `post_process()`
8. `V4L2Capture::captureFrame()`
9. `MppDecoder::readFrame()`
10. `DrmDisplay::flip()`
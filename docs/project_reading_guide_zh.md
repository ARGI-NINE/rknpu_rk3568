# rknn_yolov5_rk3568 阅读总目录

这个目录不是单独的一篇说明，而是一组按调用链拆开的讲解文档。

项目本体是一条完整的板端视觉流水线：

`输入源 -> 采集/解码 -> 前处理 -> RKNN 推理 -> 后处理 -> DRM 显示`

## 阅读顺序

1. [Overview](project_reading_guide_zh_01_overview.md)
2. [Main and Pool](project_reading_guide_zh_02_main_and_pool.md)
3. [Backend](project_reading_guide_zh_03_backend.md)
4. [Input and Display](project_reading_guide_zh_04_input_display.md)
5. [Function Index](project_reading_guide_zh_05_function_index.md)
6. [Class and Method Guide](project_reading_guide_zh_06_class_and_method_guide.md)

## 每份文档负责什么

`project_reading_guide_zh_01_overview.md`
: 项目目标、编译方式、运行入口、整体模块图。

`project_reading_guide_zh_02_main_and_pool.md`
: 从 `main()` 开始，逐段解释主循环、线程、结果队列、`rknnPool` 的数据契约。

`project_reading_guide_zh_03_backend.md`
: 逐函数解释 `rkYolov5s`、`preprocess`、`postprocess`，把推理和结果解码讲清楚。

`project_reading_guide_zh_04_input_display.md`
: 逐函数解释 `V4L2Capture`、`MppDecoder`、`DrmDisplay` 和绘图辅助函数。

`project_reading_guide_zh_05_function_index.md`
: 把项目里所有关键函数按模块列成索引，方便回查。

`project_reading_guide_zh_06_class_and_method_guide.md`
: 逐类解释 `rkYolov5s`、`rknnPool`、`FrameCopyPool`、`V4L2Capture`、`MppDecoder`、`DrmDisplay` 的构造、方法和不变量。

## 读法

- 先读 `01_overview`，知道项目整体是干什么的。
- 再读 `02_main_and_pool`，先把一帧从输入到输出的主循环看懂。
- 然后读 `03_backend`，看推理和后处理怎么实现。
- 再读 `04_input_display`，把输入和显示两端补齐。
- 最后用 `05_function_index` 回查任意函数。
- 如果你想按对象粒度把类、方法和资源边界再过一遍，再读 `06_class_and_method_guide`。

## 说明

这套文档的目标不是“讲概念”，而是“按代码链路讲调用”。

所以每一份都尽量按下面顺序写：

1. 先给代码片段。
2. 再解释这段代码在链路中的作用。
3. 再解释它调用了什么、依赖了什么、结果传到哪里。
4. 最后补一句你读这一段时要抓的重点。
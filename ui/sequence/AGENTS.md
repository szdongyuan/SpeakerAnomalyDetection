# 电机测试工作台

## 控制器的职责

[`MotorWorkbenchController`](motor_workbench_controller.py) 是主工作台的中介者，负责决定跨控件的业务更新。它持有最近一次成功应用的配置快照，把配置转换成视图数据，并通过控件的公开接口提交。`MainWindow.init_sequence_widget()` 负责创建控制器、注入依赖和连接 Qt 信号，刷新与显示更新在 GUI 线程执行。

| 组件 | 职责 |
| --- | --- |
| `MainWindow` | 组装仓储、控制器、视图和视频服务。把配置提交及视频设置事件接到控制器入口。 |
| `MotorWorkbenchController` | 读取配置快照、判断是否需要刷新、生成同序工位数据，并转发打开视频设置的意图。 |
| [`MotorWorkbenchConfigRepository`](../../base/motor_workbench_config.py) | 只读加载产品注册表中的活动产品，以及该产品所引用的队列。负责 JSON、字段和引用校验，返回 `WorkbenchConfig`、`None` 或抛出 `WorkbenchConfigurationError`。 |
| 工位面板、分析网格和活动产品显示 | 接收控制器给出的数据显示。折叠、滚动、缩放等局部状态由所属控件管理。 |
| 配置编辑端及视频服务 | 配置端负责编辑和写入产品、队列及注册表。视频服务负责设置窗口、权限、设备探测和保存。 |

## 配置刷新

1. 工作台启动时，主窗口调用 `refresh_configuration()`。产品保存及配置删除成功后，经 `MainWindow.on_workbench_configuration_committed()` 走同一个刷新入口。队列保存和活动产品切换的发送端接入时，也应在文件和注册表成功写入后通知此入口。通知只表示落盘内容有变化，控制器每次都从仓储重读。关闭编辑窗口、取消和保存失败不触发刷新。
2. 控制器按产品配置中的工位顺序处理数据。用 `station_id` 匹配工况的 `applicable_station_ids`，再按工况顺序和队列的 `analysis_list.display_sequence` 生成指标。一个工位的同名指标分别保留，并补上对应工况名称。
3. 控制器把工位名称和指标交给 `MotorStationTestPanel.show_workbench_stations()`，把同序工位名称交给 `StationPlotGrid.show_workbench_columns()`，再更新只读的活动产品名称。有效新配置以 `--` 初始化正转、反转和指标值，分析网格按工位生成两行空图。
4. 相同快照的重复通知只重新读取配置，保留已经显示的结果。有效注册表中没有活动产品时，控制器应用空态，保留区域标题并显示零工位、零列。配置缺失或无效时静默处理，首次加载显示空态，后续刷新保留最近一次成功应用的配置及显示。意外异常继续向调用方传播。

## 视频设置与后续扩展

- 视频卡片的「设置」按钮发出 `settings_requested`，主窗口将它连接到 `MotorWorkbenchController.open_video_settings()`。控制器通过注入的入口调用 `VideoController.show_settings()`，视频服务负责复用已打开的窗口。
- 新增跨控件业务行为时，先定义控件发出的用户意图和控制器的处理入口，再由控制器调用受影响控件的公开接口。业务判断、业务服务调用和跨控件状态同步都放在控制器侧。控件只渲染自己的数据、发出意图并维护局部展示状态。
- 录制、分析和数据库接入属于后续扩展。新增接入时，把服务依赖和结果回调接到控制器，由控制器在 GUI 线程决定哪些视图需要更新。工作台的旧产品测试入口当前处于禁用状态。
- 独立构造的 `StationPlotGrid` 默认提供两行三列。独立调用者可用 `set_grid_size()` 和网格公开接口供数。主工作台的工位列数据统一来自控制器。

修改刷新规则、跨控件更新或视频意图接线时，更新对应的 [控制器测试](../../unit_test/ui/test_motor_workbench_controller.py) 或 [主窗口集成测试](../../unit_test/ui/test_motor_workbench_integration.py)。

## 模板绘图配置

- `ui/ui_config/motor_workbench_plot_layout.json` 支持模板格式。纯配置解析在 `base/workbench_plot_config.py`，通过已有仓储和配置加载入口调用，不引入 Qt 依赖。
- `workbench_ui_projection.py` 根据当前使用配置筛选适用于各工位的工况，保留产品配置中的顺序、ID 和名称，再绑定分析项与 WAV 通道。绘图 JSON 不重复保存示例工况 ID。
- `StationPlotGrid.show_workbench_layout()` 仍是布局入口。每个模板图框使用一个 `ConditionPlotCell`，继承单图 `StationPlotCell`，只创建一个 PlotWidget 和一套纵轴；工况是在同一画布内的水平区域，不能再拆成多个小图。
- 当前模板中 SPL 和 FFT 都使用 condition_local：各工况分别显示自己的时间或频率，SPL 自动范围从现有队列录音时长取得。保留显式配置 cumulative_time 的支持。
- condition_display.condition_gap_px 默认 16，表示工况之间的逻辑像素留白，不代表时间或频率间隔。窗口变化时重算曲线位置，单工况不留间隔；分隔虚线居中。不同工况的曲线分别绘制，不跨分隔线连接。
- 样式翻译在 `station_plot_style.py`；控件不读取文件、不启动分析、不生成示例曲线。真实结果通过已有控制器的分析事件轮询进入展示层。
- 模板 `display_downsampling` 只控制显示：none 全点；keep_every_nth_point 配 keep_one_point_every_n_points 正整数；preserve_min_max 配 pixel_bucket_width 正整数（默认 1）。曲线完整数据保留，主图和详情按各自视口取点。取点实现位于 `curve_sampling.py` / `sampled_plot_curve.py`，自动纵轴范围使用完整数据；上下限不降采样。x 有限，y 的 null/NaN 保留为断线。不得影响算法、判定或 CSV。
- 模板图标题栏提供放大按钮。`plot_detail_dialog.py` 复用 `ConditionPlotCell` 和内存中的工况曲线，允许缩放和拖动；关闭后重新打开恢复完整视图，不单独设置恢复按钮。同图只打开一个窗口。源图更新和清空同步到详情，切换配置销毁源图时关闭详情。详情不读取 CSV、不调用分析服务。
- 放大入口采用 16px 蓝灰色向外双箭头，点击区域 24px，由 Qt 绘制，无新增图片依赖。工作台工位标题栏只显示名称，高度 36px；不显示列下拉框、播放或手动 OK/NG。顶部模式选择保持原状，切换模式不重新显示这些列控件。独立使用 StationPlotGrid 的通用列控件 API 保留。
- WAV 显示通道从 1 开始，Qt 图框 API 的 row/column 从 0 开始。通道转换统一由展示映射处理。
- 新配置的解析和渲染分别在 `unit_test/base/test_workbench_plot_config.py` 和 `unit_test/ui/test_condition_plot_layout.py` 验证；测试不得依赖用户的本地 JSON。

## 分析结果展示接入

- `MotorWorkbenchController` 复用现有录制回调、分析任务提交和 `poll_analysis_events()`；不增加 UI 轮询器或分析进程。分析结果先完成当前配置版本、任务及录制请求归属检查，再通知展示层和原有事件订阅者。
- `workbench_plot_presenter.py` 只负责 GUI 线程中的工位/工况区域更新与展示状态；`workbench_plot_results.py` 将任务的 `display_payload` 转为曲线、上下限或时频矩阵，不读 CSV、不计算指标、不执行阈值判定。
- 按 `station_id`、`condition_id`、`config_key`、`analysis_type`、`channel_index` 精确匹配；通道索引从 0 转为 WAV 显示通道号（从 1 开始）。绘图配置只选择展示范围，不改变分析任务。
- 控制器按现有工况顺序循环；首工况开始采集时清空上轮所有区域，其他工况开始时只清空自身。旧请求、旧配置和重复终态事件不得覆盖当前结果。状态与完整曲线同步到详情窗口。
- 单项或单通道失败保留其他有效曲线；错误通过区域文字、悬停提示及日志呈现。总体 SPL 阈值不能绘制成瞬时曲线上下限；仅文件保存失败不丢弃已计算的曲线。
- `unit_test/ui/test_workbench_plot_results.py` 覆盖结果映射、状态、迟到事件和真实 WAV 常驻进程集成；硬件采集需另行现场验收。

## 超限、时频图和振动 RMS 展示

- 新工作台独立绘制，不调用或改造旧 `signal_analysis_window.py`。上下限插值和 OK/NG 使用后端完整数据判定；前端只把 `violations` 的时间/频率区间映射成高亮。高亮不降采样，孤立超限点有圆点标记。总体 SPL 限值只显示判定和悬停摘要，不转换为瞬时曲线限值。
- `plot_style.exceedance_highlight` 控制超限颜色、线型、线宽及点大小；不控制后端判定。工况标题显示后端判定，部分失败仍有状态提示，未判定不能显示成 OK。
- `Spec` 使用 `chart_type: heatmap`，每个图框在每个工况只选择一个 WAV 通道。`ConditionHeatmapCell` 复用工况分区、状态、放大窗口，`spectrogram_data.py` 校验 frequency × time 矩阵，`spectrogram_image.py` 按真实频率行边界绘制。线性时间帧和非等距频率支持；对数频率显示排除 DC，原矩阵保留。
- 各工况共享一条色标；`spectrogram.color_range` 可为 auto（当前图框全部工况矩阵范围）或 fixed（min_db/max_db）。色标不是阈值，不推断 OK/NG。当前 Spec 后端单位为变换幅度 dB re 20 µPa。
- `VIBRATION_RMS` 是预留的展示类型，使用 `line`、时间轴及显式 `value_unit: m/s²` 或 `g`，只接受 acceleration 通道。后端执行器尚未支持振动 RMS，本批不添加算法或更改分析任务协议。测试 RMS 数据仅验证绘图，不作为实际测量结果或后端验收。后续后端接入须核对类型名和单位约定。
- 绘图模板可显式配置 `data_source: simulated` 展示振动 RMS 示例；省略该字段或配置 `analysis_result` 使用真实分析结果。模拟模式必须指定展示通道，通道仅为示例编号，不对应硬件。投影不建立分析绑定，由 `workbench_plot_demo.py` 提供确定性示例曲线，展示层填充。初始化为空图，跟随当前录制请求的采集/分析状态；仅在该工况的各工位终态结果统一展示时，为成功完成分析的工位生成模拟曲线。取消、失败及尚未测试的工况不生成曲线，新一轮清空旧曲线。主图、详情及工况标题均标注模拟数据；不加入测试队列、任务、判定或 CSV。控件本身仍不生成模拟数据。
- 已提交的同一次录制请求，其各参与工位的 `task_result`（包括分析失败）在 presenter 中暂存，全部返回后于同一 GUI 更新中绘制。不是固定等四个工位，也不等待下一个工况；等待期间对客户保持「分析中」，不显示内部工位同步状态。新一轮或配置失效时清理暂存结果，其他事件订阅者仍即时收到每项结果。后端进程和事件轮询不变，进程退出/任务失败继续由分析服务形成终态事件。
- pyqtgraph 网格保留原生主次级和密度透明度，不能把所有网格线覆盖为完全不透明。JSON 继续配置颜色、线型、线宽；时频图默认不画网格。
- 空图不显示“暂无分析结果”，保留坐标轴、工况名称和分隔线。实际的待采集、采集中、分析中、失败及配置错误提示继续显示，主图和详情一致。
- 时频图纵轴统一显示 kHz（例如 10000 Hz 显示 10），只缩放刻度文字；后端矩阵、频率坐标和 JSON 频率范围仍使用 Hz，线性与对数轴均适用。
- 扩展测试位于 `unit_test/ui/test_workbench_extended_plots.py`：真实测试 WAV 的 SPL 插值判定及 STFT/CQT 矩阵进入图框，RMS 使用模拟回传数据；校验单点高亮、单位不匹配、矩阵方向/坐标、共同色标、裁剪、状态与详情同步。

- SPL 使用工况范围内的蓝灰色水平虚线标注后端 `metrics.overall_spl`，虚线置于曲线下层。数值只显示一位小数及计权单位，不加“总体”前缀；在各工况坐标轴上方居中对齐，与工况名称之间留出 12px 间距，不得用图内文字或白底标签遮挡曲线。多通道分行并标明 CH 编号。该线表示测量总体值，不是阈值；不从绘图点重新计算，不汇总不同通道。未返回有效总体值时不画线、不编造数字。主图与详情共用此展示，新一轮清空时同步移除。
- SPL 数值区域由展示映射按配置中的显示通道数预留，结果到达、部分工况完成或清空均不得改变绘图区大小。顶部图例与图标题、放大按钮共用标题行，溢出项通过“更多”菜单查看；显式配置底部图例时仍保留底部位置。主图和详情共用相同布局规则。
- 工况画布的实际数据区最小高度为 120px。控件最小高度还需包含标题、SPL 数值预留、坐标轴和底部图例；时频图设置外向刻度后重新计算。工作台沿用现有滚动区域，小窗口通过纵向滚动查看其他行，不压缩数据区来强行显示所有图。
- 同一布局的折线图统一按最多的 SPL 数值行数预留顶部区域，RMS 等无总体数值的图也保留该空间；时频图不增加数值区。预留在配置投影时确定，不随结果改变。折线图不绘制顶部边线，并隐藏恰好位于纵轴范围顶端的网格横线；内部网格和顶部刻度文字保持不变。

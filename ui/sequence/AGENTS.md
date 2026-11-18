# 电机测试工作台

## 唯一工作台装配与资源所有权

- `SequenceWindow`、`SequenceToolsBar` 只有当前工作台构造方式，不接受 `workbench_mode`，不创建旧模式、手动分析、重录、触发测试、旧计数板或隐藏兼容控件。启动预加载不能引用已退役的旧序列运行时。
- `WorkbenchEditorContext` 持有独立队列的内存编辑上下文及设备/校准来源；通道同步不写配置。`WorkbenchInputOpsMixin` 负责型号、S/N、扫码偏好与扫码监听，主窗口注入提交准入；扫码只更新 S/N，不开始录制。忙碌时禁止改写和提交 S/N，保留焦点、光标、选择与复制导航。`workbench_layout.py` 只装配当前 splitter、左面板和画板区域。
- `MainWindow` 拥有当前录制、CSV、归档、分析、硬件与应用退出流程，并向控制器提供真实实例状态。视图的输入清理只释放自己的订阅和扫码资源；服务是否由本窗口创建或由外部注入，继续按各服务的既有所有权与应用关闭约定处理。
- 产品选择与首次打开编辑器检查完整忙碌状态；父编辑器内打开子编辑器只检查错误与关闭状态。已经落盘的配置提交通知在采集、归档及分析期间仍即时刷新，旧录制请求继续使用冻结快照；错误与关闭状态拒绝提交刷新。扫码设置、校准可与非原生资源收尾重叠，不能恢复旧锁轮逻辑。
- 对应覆盖为 [编辑上下文测试](../../unit_test/ui/test_workbench_editor_context.py)、[输入测试](../../unit_test/ui/test_workbench_input_ops.py)、[布局测试](../../unit_test/ui/test_workbench_layout.py)、[运行时清理测试](../../unit_test/ui/test_workbench_runtime_cleanup.py) 及 [启动测试](../../unit_test/ui/test_workbench_recording_startup.py)。当前数据格式、分析算法与设备协议保持原契约。

## 控制器的职责

[`MotorWorkbenchController`](motor_workbench_controller.py) 是主工作台的中介者，负责决定跨控件的业务更新。它持有最近一次成功应用的配置快照，把配置转换成视图数据，并通过控件的公开接口提交。`MainWindow.init_sequence_widget()` 负责创建控制器、注入依赖和连接 Qt 信号，刷新与显示更新在 GUI 线程执行。

| 组件 | 职责 |
| --- | --- |
| `MainWindow` | 组装仓储、控制器、视图和视频服务，注入产品管理端、录制与分析忙碌查询及关闭状态。把产品选择、打开下拉列表、配置提交及视频设置事件接到控制器入口，通过 GUI 状态刷新同步选择可用性。 |
| `MotorWorkbenchController` | 判断是否需要刷新，从同一快照准备工位指标、画板展示和业务查询。负责产品列表、来源身份、切换准入、候选准备、持久化后应用，并转发打开视频设置的意图。 |
| [`MotorWorkbenchConfigRepository`](../../base/motor_workbench_config.py) | 只读加载产品注册表、活动或指定候选产品、引用的队列和 [画板布局](../ui_config/motor_workbench_plot_layout.json)。负责 JSON、字段和引用校验，返回 `WorkbenchConfig`、`None` 或抛出 `WorkbenchConfigurationError`。成功切换提交后接受新来源身份，继续支持显式注入布局路径。 |
| 工位面板、分析网格和产品选择控件 | 接收控制器给出的数据显示。产品选择控件维护完整列表、当前文件、提示与可用状态。折叠、滚动、缩放等局部状态由所属控件管理。 |
| 配置编辑端及视频服务 | 配置端负责编辑和写入产品、队列及注册表。视频服务负责设置窗口、权限、设备探测和保存。 |

## 配置刷新

1. 工作台启动时，主窗口调用 `refresh_configuration()`。产品保存、另存为、重命名及配置删除成功后，经 `MainWindow.on_workbench_configuration_committed()` 走同一个刷新入口，同步产品列表、当前来源及视图。队列保存的发送端接入时，也应在文件和注册表成功写入后通知此入口。通知只表示落盘内容有变化，控制器每次都从仓储重读。关闭编辑窗口、取消和保存失败不触发刷新。
2. 控制器按产品配置中的工位顺序处理数据。用 `station_id` 匹配工况的 `applicable_station_ids`，按共享队列的 `analysis_list.display_sequence` 生成一次指标。每个分析项 `config_key` 对应一行，正转、反转共享该行的两个数值列，不添加工况名称后缀；同类型的不同配置键仍各占一行。
3. 控制器按 `ui/ui_config/motor_workbench_plot_layout.json` 指定的尺寸和逐格设置准备画板。默认路径按应用根目录定位，支持从其他工作目录启动。所选布局文件不存在时，`WorkbenchUiConfig.layout` 为 `None`，画板显示零行零列；产品、工位指标、队列及录制和分析快照照常加载，不生成或写入默认布局。补入有效布局文件后，普通配置刷新即可恢复画板。已有布局文件不可读、JSON 无效（包括 `null`）或字段不合法时仍报告配置错误。列按产品工位顺序排列，剩余列留空。每格为前两个适用工况保留位置，按队列显示顺序中的分析项名称和类型匹配。缺少分析项或类型不符时保留工况名并提示。单位不一致时保留第一工况的绑定、工况名和单位，提示中列出两种单位。
4. 两个视图的数据都准备好后，控制器调用 `MotorStationTestPanel.show_workbench_stations()` 和 `StationPlotGrid.show_workbench_layout()`，再同步产品选择和业务快照。应用新快照时，工位正转、反转和指标值重置为 `--`，清空所有画板的旧曲线。
5. 来源文件和快照都相同的重复通知保留已经显示的结果。来源文件变化时，即使配置内容相同，也清空旧结果、重置工况游标并更新分析版本。有效注册表中没有活动产品时，控制器清空业务快照并显示零工位、零列，保留区域标题。必要业务配置缺失或配置无效时记录诊断，首次加载显示空态，后续刷新保留最近一次成功应用的配置及显示。意外异常继续向调用方传播。

## 产品选择与准入

- 「使用配置」按登记顺序展示产品，使用文件名作为项数据。重名产品的显示文本包含文件名，工具提示包含完整产品名和文件名。打开列表调用 `refresh_product_catalog()` 重读登记信息，程序刷新阻断选择信号，保留未选择与空列表状态。
- `select_product(file_name)` 先检查准入和最新登记，再读取一次候选并准备一次投影。提交前再次检查准入，通过产品管理端保存 `active_file` 后，接受来源并应用同一份候选和投影。选择当前来源时保留结果与注册表。读取、校验或保存失败时恢复当前选项并报告原因，保留已应用状态。
- 同一个准入判断覆盖控制器在途状态、录制与资源收尾、归档、运行及排队中的分析和应用关闭状态。`refresh_recording_state()` 同步刷新选择可用性，实际选择入口也检查准入，拒绝忙碌期间直接发出的选择信号。任务失败、取消和正常结束后，待资源释放完成即可恢复选择。
- 产品列表读取失败时保留最近有效列表与选项并报告诊断，后续打开或配置提交通知可以重试。主工作台不读取或写入模式偏好，产品切换后 OK / NG 手动标记按钮继续隐藏。

## 业务查询

- `get_plot_bindings(row, column)` 按画板位置返回两个可空的绑定。绑定包含工位 ID、工况 ID、分析项名称和类型。
- `get_condition_config(condition_id)` 查询所有已加载工况的只读快照，包括未分配到画板的工况。工况通过适用工位 ID 引用工位。其 `export_process_config()` 每次返回独立的普通配置字典及列表，保留队列字段和分析通道索引。
- `get_station_channels(station_id)` 按工位 ID 返回产品声明顺序中的共享通道映射。同一工位的工况使用同一组通道。逻辑通道由工位 ID 和通道 ID 定位，保留工位内通道名称。设备引用和从零开始的物理通道索引共同标识物理通道。

## 视频设置与后续扩展

- 视频卡片的「设置」按钮发出 `settings_requested`，主窗口将它连接到 `MotorWorkbenchController.open_video_settings()`。控制器通过注入的入口调用 `VideoController.show_settings()`，视频服务负责复用已打开的窗口。
- 新增跨控件业务行为时，先定义控件发出的用户意图和控制器的处理入口，再由控制器调用受影响控件的公开接口。业务判断、业务服务调用和跨控件状态同步都放在控制器侧。控件只渲染自己的数据、发出意图并维护局部展示状态。
- 录制和自动分析已接到控制器；后续服务或数据库接入继续通过控制器在 GUI 线程决定哪些视图需要更新。旧产品测试入口及专属运行时已经移除。
- 独立构造的 `StationPlotGrid` 默认提供两行三列。独立调用者可用 `set_grid_size()` 和网格公开接口供数。主工作台的工位列数据统一来自控制器。控制器初始化时调用网格公开接口隐藏 OK / NG 手动标记按钮，新列继承该状态，配置刷新、清空恢复和重启均保持隐藏；独立网格默认值和显式显隐 API 保持不变。

修改刷新规则、跨控件更新或视频意图接线时，更新对应的 [控制器测试](../../unit_test/ui/test_motor_workbench_controller.py) 或 [主窗口集成测试](../../unit_test/ui/test_motor_workbench_integration.py)。

## 模板绘图配置

- `ui/ui_config/motor_workbench_plot_layout.json` 支持模板格式。纯配置解析在 `base/workbench_plot_config.py`，通过已有仓储和配置加载入口调用，不引入 Qt 依赖。
- `workbench_ui_projection.py` 根据当前使用配置筛选适用于各工位的工况，保留产品配置中的顺序、ID 和名称，再绑定分析项与 WAV 通道。绘图 JSON 不重复保存示例工况 ID。
- `StationPlotGrid.show_workbench_layout()` 仍是布局入口。每个模板图框使用一个 `ConditionPlotCell`，继承单图 `StationPlotCell`，只创建一个 PlotWidget 和一套纵轴；工况是在同一画布内的水平区域，不能再拆成多个小图。
- 当前模板中 SPL 和 FFT 都使用 condition_local：各工况分别显示自己的时间或频率，SPL 自动范围从现有队列录音时长取得。保留显式配置 cumulative_time 的支持。
- condition_display.condition_gap_px 默认 16，表示工况之间的逻辑像素留白，不代表时间或频率间隔。窗口变化时重算曲线位置，单工况不留间隔；分隔虚线居中。不同工况的曲线分别绘制，不跨分隔线连接。
- 样式翻译在 `station_plot_style.py`；控件不读取文件、不启动分析、不生成示例曲线。真实结果通过已有控制器的分析事件轮询进入展示层。
- 模板 `display_downsampling` 只控制显示：none 全点；keep_every_nth_point 配 keep_one_point_every_n_points 正整数；preserve_min_max 配 pixel_bucket_width 正整数（默认 1）。曲线完整数据保留，主图和详情按各自视口取点。取点实现位于 `ui/plotting/sampling.py` / `ui/plotting/sampled_curve.py`，原路径仅作兼容导出；自动纵轴范围使用完整数据；上下限不降采样。x 有限，y 的 null/NaN 保留为断线。不得影响算法、判定或 CSV。
- 模板图标题栏提供放大按钮。`plot_detail_dialog.py` 复用 `ConditionPlotCell` 和内存中的工况曲线。小图和详情均允许缩放、拖动及右键菜单；右键菜单保持 pyqtgraph 原生内容和行为，不改名或替换 `View All`。鼠标悬停时提供左下角还原按钮，恢复配置的完整工况范围并清除手动缩放状态。结果更新和窗口缩放保留手动调整的坐标轴，新一轮清空及关闭后重新打开详情恢复完整视图。同图只打开一个窗口。源图更新和清空同步到详情，切换配置销毁源图时关闭详情。详情不读取 CSV、不调用分析服务。
- 放大入口采用 16px 蓝灰色向外双箭头，点击区域 24px，由 Qt 绘制，无新增图片依赖。工作台工位标题栏固定高度 44px，使用 Microsoft YaHei UI（微软雅黑 UI）18px 常规字重，水平、垂直居中，默认与 OK/NG 状态保持相同高度。该字体仅用于工位标题，不修改全局字体。默认只显示名称；当前轮最终结果出来后显示“工位名称 · OK/NG”，所有状态保持中性浅灰蓝底色；工位名称和分隔点始终保持深蓝灰色常规字重，仅 OK/NG 结果文字分别使用深绿色/深红色并加粗，配对应颜色的 2px 底部细线；长名称省略时为结果文字保留空间，不对整条标题背景着色。恢复默认时移除结果文字与彩色底线，恢复普通分隔线。标题不显示测试中、待测试。不显示列下拉框、播放或手动 OK/NG。主工作台不提供顶部模式选择，配置刷新和产品切换不重新显示这些列控件。独立使用 StationPlotGrid 的通用列控件 API 保留。
- WAV 显示通道从 1 开始，Qt 图框 API 的 row/column 从 0 开始。通道转换统一由展示映射处理。
- 新配置的解析和渲染分别在 `unit_test/base/test_workbench_plot_config.py` 和 `unit_test/ui/test_condition_plot_layout.py` 验证；测试不得依赖用户的本地 JSON。

## 分析结果展示接入

- `MotorWorkbenchController` 复用现有录制回调、分析任务提交和 `poll_analysis_events()`；不增加 UI 轮询器或分析进程。分析结果先完成当前配置版本、任务及录制请求归属检查，再通知展示层和原有事件订阅者。
- `workbench_plot_presenter.py` 只负责 GUI 线程中的工位/工况区域更新与展示状态；`workbench_plot_results.py` 将任务的 `display_payload` 转为曲线、上下限或时频矩阵，不读 CSV、不计算指标、不执行阈值判定。
- 按 `station_id`、`condition_id`、`config_key`、`analysis_type`、`channel_index` 精确匹配；通道索引从 0 转为 WAV 显示通道号（从 1 开始）。绘图配置只选择展示范围，不改变分析任务。
- 控制器按现有工况顺序循环；首工况开始采集时清空上轮所有区域，其他工况开始时只清空自身。旧请求、旧配置和重复终态事件不得覆盖当前结果。状态与完整曲线同步到详情窗口。
- 工位标题复用 `WorkbenchResultState` 的 `complete/result`，由控制器在身份校验及当前轮次检查后更新，不从绘图项重新判定。新轮次录制获接收时经 `_activate_recording_round()` 清除所有标题结果，拒绝启动保留原结果，同轮第二工况不重复重置。重新应用配置恢复默认标题；无绘图布局时不更新标题。
- 单项或单通道失败保留其他有效曲线；错误通过区域文字、悬停提示及日志呈现。总体 SPL 阈值可作为标注“总体上限/下限”的水平参考线，不能用于瞬时曲线逐点判定；仅文件保存失败不丢弃已计算的曲线。
- `unit_test/ui/test_workbench_plot_results.py` 覆盖结果映射、状态、迟到事件和真实 WAV 常驻进程集成；硬件采集需另行现场验收。

## 当日 WAV 记录分析查看

- 成功录制完成时，在归档前从冻结请求及完成描述符保留 `RecordAnalysisReference`：请求 ID、源工位/工况 ID、实际绝对 WAV 路径和 WAV 列通道身份。日记录的显示工位序号、文件名及原记录结果不承担定位职责；旧记录缺少引用时继续参与筛选统计，但查看操作禁用并说明原因。明细操作绑定实际记录，分页和刷新不能捕获可变行号。
- 明细通过 `record_analysis_requested(record, parent)` 向控制器发意图。`get_current_analysis_snapshot()` 只提供当前成功应用且有效的分析配置及版本；`RecordAnalysisCoordinator` 在打开时冻结它们，按源身份和 WAV 通道顺序构建该文件的全部当前队列项。当前设备离线不阻止查看；配置不存在、不匹配或队列为空时显示明确状态。已打开会话不受后续配置提交影响，重新打开采用新快照。
- 查看复用 `MainWindow` 的同一个 `AnalysisService`、常驻 worker 和唯一 50ms 分析轮询器，不增加服务或分析定时器。请求显式 `persist_artifacts=False`、结果目录为 `None`；执行器完成原算法与判定后跳过导出，返回内存结果及空 artifacts。普通生产任务默认仍导出 CSV，原标签、工位、画板、日记录、轮次与统计认证及更新链路保持原职责。
- 控制器先按登记的任务所有权把查看事件交给协调器，其终态需认证任务、源工位和工况。查看结果不能进入生产 presenter、标签提交或统计链路；关闭窗口后仍保留最小任务归属到终态，迟到结果只消费而不渲染。`service_closed` 和同批生产终态继续完整送交原生命周期入口，不能因查看任务分流提前建立标签排空屏障。
- `record_analysis_results.py` 仅映射内存结果，dialog/presenter/renderers 只管理选择和原生 Qt/pyqtgraph 内容。SPL/FFT/LOUD 曲线实际调用 `ui.plotting.renderers.draw_curve`，与主图共用 `resolve_curve_colors` 和 `analysis_curve_pen`；同配置多通道同色，保留现有通道名称，不按通道另取调色板。每个曲线/频段面板使用真实 `StationPlotLegend` 图例行及画布，不创建图标题或空标题行；图例设置 `align_right=True`，独占完整宽度的 24px 行，字体、画笔示例和“更多”菜单直接复用主界面组件，不创建图内图例。以实际 QPen 按 WAV 通道、角色及颜色/宽度/线型去重，名称沿用主界面 CHn、上下限/总体限值、超限和参考规则；隐藏阈值时同步移除对应条目。LOUD 每个曲线面板独立图例，热图仅保留色条。`show_threshold_lines` 控制上下限及标量参考线，不影响返回判定或超限高亮；baseline、FFT 超限画笔也由公共入口解析。查看窗口仍持有完整数据、采样刷新和销毁责任，调用曲线入口时显式传 sampling（禁用取点使用 `{'method': 'none'}`）以保留完整数据所有权，不重复传 connect。
- FBA 使用独立 BandRenderer 布局，通过公共 `draw_bars`、`draw_curve`、`draw_text` 创建图形；按频段索引等距绘柱，同域多通道在总宽 0.7 内按顺序并排，域身份包含标签和中心坐标。正常柱取公共主曲线画笔色，返回 exceedance 超限柱及 NG 取公共超限画笔色；允许的局部 null/NaN 为零高灰柱，原数据不变。上下限按柱中心绘圆点/三角标记，采用公共限值画笔并保留断线，只显示水平网格。保留离散频段标签、单位和总体值；共享图例列出正常柱的公共主曲线笔及实际可见的上下限和超限笔，不漏掉返回的角色说明。Spec/LOUD 热图实际调用公共 `draw_binned_image`，矩阵方向、物理格边界、DC/对数处理、色阶和单位保持返回语义，坐标轴、色条和布局由查看窗口管理。LOUD 按该配置返回的时间曲线、Bark 特征响度曲线/热图和汇总展示，不添加第六种算法或伪造曲线。未支持项保留标签及实际诊断，局部失败不丢弃其他通道。切换不重算、不读 WAV 或 CSV。摘要只显示人类可读测量值、单位及阈值，完整内部指标保留在模型中；删除工位/工况/时间信息行、重复分析项标题、每通道名称/完成状态和所有图标题。单通道摘要不带 CH 前缀，多通道仅用简短 CHn。原记录结果保留右上，本次判定按返回通道逐项显示在其下方；切换、等待和失败不得残留旧判定，不推断汇总；只有结果值按 `MOTOR_STATION_STATUS_COLORS` 着色。坐标复用主界面 `_BottomAxis` / `StyledGridAxis` 及 `(7, 8, 7, 5)` 内边距，使端点文字在界内并沿用防重叠；不改共享类、刻度坐标或数据。字体使用 `MAIN_UI_SMALL_FONT_FAMILY_NAME` 16px；秒轴显示“时间（s）”，Hz/kHz 横轴显示“频率（Hz）”/“频率（kHz）”，不换算数据或改写实际单位；Bark 保持原语义。左侧导航使用共用 `WorkbenchHorizontalScrollBar`，objectName 为 `workbenchPlotHorizontalScrollBar`，沿用共用样式及原生按需滚动、按钮、范围与选择行为。SPL 计权优先 metrics.weighting、明确返回单位、当前项配置及共用 Z 默认，只规范显示文本，FFT delta 的 dB、LOUD 与热图语义及原始元数据不变。内容区域的现有垂直滚动条采用 `workbenchStationVerticalScrollBar` 与 `MOTOR_WORKBENCH_SCROLLBAR_STYLE`，保持按需显示。默认单图按视口伸缩并保留至少 120px 实际绘图区，小窗口和多面板通过区域滚动，失败说明紧凑靠上排列。
- 每个明细最多一个原生模态子窗口，重复查看同记录只聚焦；查看窗口不提供“返回明细”按钮。Escape、系统关闭及父窗口销毁都释放 presenter、图形和矩阵订阅，并清空共享图例条目、溢出菜单引用，清除柱图与 NG 标注引用，保留明细筛选/页码/滚动与焦点。查看窗口借用服务，不能 shutdown、删除工位或终止 worker；应用关闭仍由 `MainWindow` 消费真实关闭通知及整批终态，再等待原标签桥排空。
- 新增回归位于 `unit_test/base/test_record_analysis_reference.py`、`unit_test/base/test_record_analysis_request.py` 及 `unit_test/ui/test_record_analysis_{results,dialog,presenter,renderers,legend,coordinator,entry,integration}.py`。真实集成测试只替换采集/登录/配置边界，使用独立校准双通道 WAV、SQLite、真实分析 worker、主窗口原定时器和标签桥，覆盖只读隔离、完整生产正反转、配置冻结、关闭迟到结果、父窗口销毁和混合关闭批次。
- 跨界面画笔与公共函数真实委托回归位于 `unit_test/ui/test_record_analysis_shared_plotting.py`，覆盖默认/自定义/非法颜色、阈值开关、baseline、FFT 超限、多通道和真实 WAV 数据。可重复视觉证据命令：`python scripts/render_record_analysis_view.py --output-dir .worktrees/record-analysis-view-evidence`。脚本仅在指定目录生成合成 WAV、独立配置、十张截图及 `evidence.json`，在独立计算进程得到真实结果后装配实际窗口组件；不访问用户数据库/配置。证据包含默认与小窗口、多通道、单/多通道 FBA、Spec、LOUD 特征响度及失败项、单通道 A 计权 SPL 与 NG，以及原记录 NG/当前 OK 的对照，使用明显的非默认分析项颜色；JSON 记录原记录/本次判定的可见文本、坐标标签/字体、空标题列表、共享图例名称/字体/画笔/右对齐/溢出菜单，以及配置颜色、实际画笔/柱色、尺寸、控件边界、选择和完整数据范围；渲染产物不提交。

## 超限、时频图和振动 RMS 展示

- 阈值绘图共用 `workbench_plot_results.build_condition_result` → `ConditionPlotCell` → `station_plot_style.draw_curves`：分析后端提供同坐标的 `display_payload.x/y/upper/lower`、单位和 `violations`，显示层统一绘制 `curve/upper/lower/exceedance`，不得为每种分析项重复实现阈值比较或绘图。新增类型只接入分析结果适配、绘图模板类型及坐标单位；通过公共绘图测试不代表该类型的分析执行器已经接入。
- `LOUD` 折线模板接收现有响度分析结果，横轴为秒，纵轴单位跟随分析项 `advanced.curve_y_unit`（sone/phon），不使用声压级的 dB 单位。曲线阈值和超限区间复用公共结果映射；稳态平均、最大瞬态等标量判定不伪造逐点阈值。绘图模板不修改分析项判定配置。FFT 与响度的真实 WAV 阈值、主图/详情及清空回归位于 `unit_test/ui/test_workbench_analysis_thresholds.py`。
- 分析项 `display.show_threshold_lines`（默认 true）统一控制主图与详情的阈值线及对应图例，不影响后端判定和超限高亮。
- 主图与详情调用 `ui/plotting`；远端已删除的分析窗口不恢复，公共模块不得依赖旧窗口。上下限插值和 OK/NG 使用后端完整数据判定；前端只把 `violations` 的时间/频率区间映射成高亮。高亮不降采样，孤立超限点有圆点标记。总体 SPL 限值以工况内水平参考线展示，不转换为瞬时逐点判定。
- `plot_style.exceedance_highlight` 控制超限颜色、线型、线宽及点大小；不控制后端判定。工况标题不显示 OK/NG，后端判定保留在悬停提示中；部分失败等状态提示和模拟数据标注继续显示。主图与详情保持一致。

上述 `plot_style` 阈值样式覆盖仅适用于旧版布局。v2 使用 `schema_version: 2`、`column_source: product_stations` 和 `row_templates`，按产品的实际工位数量及顺序展开；不再固定四列。分析项提供曲线和阈值颜色，公共绘图提供对应算法原有线型与线宽；v2 拒绝重复定义 `limit_lines` / `exceedance_highlight`。JSON 保留图表选择、通道筛选、显示范围、网格、图例和显示采样。

响度分段曲线保留全部原始样本，以首末原始值延展分析区间边缘，不能用第二个点替换首点；后端展示坐标使用 `base/analysis_plot_geometry.py`。后端 `display_payload.plot_curve` 保存展示坐标及对应限值，原始 `x/y/upper/lower`、判定及 CSV 不变；高亮仍依据原始点的后端超限区间。响度阈值单位与显示单位不一致时，主图和详情均不绘制该阈值或其高亮，原判定保留。稳态/最大响度标量限值可在同量纲坐标轴画水平参考线，特征响度超限总量不得画在响度时间曲线上。
- FFT 差值模式不叠加绝对参考谱；overlay 模式保留灰色参考层。固定横轴范围使用 `ui/plotting/geometry.py` 裁剪线段，保留边界交点及 NaN 断线；对数坐标在变换后的坐标中插值，不改变源数据、判定和导出。
- 主图和详情 Spec 使用后端完整、已标定且无 80 dB 截底的矩阵。颜色范围属于显示设置，不是判定阈值。曲线与超限回归见 `unit_test/ui/test_plot_surface_consistency.py`，矩阵一致性见 `unit_test/ui/test_spec_coordinate_parity.py`，裁剪边界见 `unit_test/ui/test_plot_geometry.py`。
- Spec 主图与详情共用 `ui/plotting/geometry.py` 的采样格边界和 `SpectrogramImage` 渲染，不能把采样中心当作图像外边界。对数频率先按真实 Hz 计算格边界，再转换显示坐标；颜色范围只影响显示，原矩阵不变。主图与详情的坐标对照位于 `unit_test/ui/test_spec_coordinate_parity.py`。
- `Spec` 使用 `chart_type: heatmap`，每个图框在每个工况只选择一个 WAV 通道。`ConditionHeatmapCell` 复用工况分区、状态、放大窗口，`spectrogram_data.py` 校验 frequency × time 矩阵，`ui/plotting/spectrogram.py` 按真实频率行边界绘制，原 `spectrogram_image.py` 仅作兼容导出。线性时间帧和非等距频率支持；对数频率显示排除 DC，原矩阵保留。
- 各工况共享一条色标；`spectrogram.color_range` 可为 auto（当前图框全部工况矩阵范围）或 fixed（min_db/max_db）。色标不是阈值，不推断 OK/NG。当前 Spec 后端单位为变换幅度 dB re 20 µPa。
- `VIBRATION_RMS` 是预留的展示类型，使用 `line`、时间轴及显式 `value_unit: m/s²` 或 `g`，只接受 acceleration 通道。后端执行器尚未支持振动 RMS，本批不添加算法或更改分析任务协议。测试 RMS 数据仅验证绘图，不作为实际测量结果或后端验收。后续后端接入须核对类型名和单位约定。
- 绘图模板可显式配置 `data_source: simulated` 展示振动 RMS 示例；省略该字段或配置 `analysis_result` 使用真实分析结果。模拟模式必须指定展示通道，通道仅为示例编号，不对应硬件。投影不建立分析绑定，由 `workbench_plot_demo.py` 提供确定性示例曲线，展示层填充。初始化为空图，跟随当前录制请求的采集/分析状态；仅在该工况的各工位终态结果统一展示时，为成功完成分析的工位生成模拟曲线。取消、失败及尚未测试的工况不生成曲线，新一轮清空旧曲线。主图、详情及工况标题均标注模拟数据；不加入测试队列、任务、判定或 CSV。控件本身仍不生成模拟数据。
- 已提交的同一次录制请求，其各参与工位的 `task_result`（包括分析失败）在 presenter 中暂存，全部返回后于同一 GUI 更新中绘制。不是固定等四个工位，也不等待下一个工况；等待期间对客户保持「分析中」，不显示内部工位同步状态。新一轮或配置失效时清理暂存结果，其他事件订阅者仍即时收到每项结果。后端进程和事件轮询不变，进程退出/任务失败继续由分析服务形成终态事件。
- pyqtgraph 网格保留原生主次级和密度透明度，不能把所有网格线覆盖为完全不透明。JSON 继续配置颜色、线型、线宽；时频图默认不画网格。
- 空图不显示“暂无分析结果”，保留坐标轴、工况名称和分隔线。实际的待采集、采集中、分析中、失败及配置错误提示继续显示，主图和详情一致。
- 时频图纵轴统一显示 kHz（例如 10000 Hz 显示 10），只缩放刻度文字；后端矩阵、频率坐标和 JSON 频率范围仍使用 Hz，线性与对数轴均适用。
- 扩展测试位于 `unit_test/ui/test_workbench_extended_plots.py`：真实测试 WAV 的 SPL 插值判定及 STFT/CQT 矩阵进入图框，RMS 使用模拟回传数据；校验单点高亮、单位不匹配、矩阵方向/坐标、共同色标、裁剪、状态与详情同步。

- SPL 仅以数值标注后端 `metrics.overall_spl`，不绘制总体值水平参考线。数值只显示一位小数及计权单位，不加“总体”前缀；在各工况坐标轴上方居中对齐，与工况名称之间留出 12px 间距，不得用图内文字或白底标签遮挡曲线。多通道分行并标明 CH 编号。该数值表示测量总体值，不是阈值；不从绘图点重新计算，不汇总不同通道。未返回有效总体值时不显示数值。主图与详情共用此展示，新一轮清空时同步移除。
- SPL 数值区域由展示映射按配置中的显示通道数预留，结果到达、部分工况完成或清空均不得改变绘图区大小。顶部图例在标题行下方独占一行，使用图框完整宽度，标题和放大按钮留在标题行；只有整行仍放不下时才通过“更多”菜单查看。图例启用时预留固定行高，不随结果到达或清空改变；显式配置底部图例时仍保留底部位置。主图和详情共用相同布局规则。
- 工况画布的实际数据区最小高度为 120px。控件最小高度还需包含标题、SPL 数值预留、坐标轴和底部图例；时频图设置外向刻度后重新计算。工作台沿用现有滚动区域，小窗口通过纵向滚动查看其他行，不压缩数据区来强行显示所有图。
- 同一布局的折线图统一按最多的 SPL 数值行数预留顶部区域，RMS 等无总体数值的图也保留该空间；时频图不增加数值区。预留在配置投影时确定，不随结果改变。折线图不绘制顶部边线，并隐藏恰好位于纵轴范围顶端的网格横线；内部网格和顶部刻度文字保持不变。

## 分析结果同步

- `MainWindow` 在 GUI 线程轮询分析事件，`MotorWorkbenchController` 及其拥有的实例状态校验任务、配置来源、分析版本、内部录制轮次、工位和工况身份，再调用左侧工位面板及信息汇总的公开接口。业务关联使用配置 ID，显示序号只在视图边界映射。
- 中介者仅对已接纳任务、原始事件身份一致且首次接收的分析终态提交单 WAV 标签意图，使用任务接纳时冻结的绝对 WAV 路径；未知、身份不符和重复终态不得写库。标签按该 WAV 的 `instance_results` 复用共用判定规则：任意 NG 则 NG，有 OK 则 OK，否则 `not_labeled`；CSV 保存失败不取消已有判定。标签不等待另一方向或其他工位，与下述正反转齐全后的工位结果及统计分别更新。
- `MainWindow` 为归档与标签仓储注入同一数据库。控制器只认证和路由，不访问数据库；`AudioLabelRepository` 只按既有路径更新既有记录的标签并返回稳定 ID，不插入、重命名或更改其他元数据。`AudioLabelSyncService` 在实例队列的单一后台线程顺序执行，`AudioLabelSyncBridge` 把已提交成功或失败通知排队交回 GUI 线程；失败保留分析显示并报告诊断。已打开的音频管理列表仅订阅其对话框生命周期内的成功通知，按稳定 ID/路径同步缓存与可见行，保留筛选、排序、选择和仍可见的播放状态；重新打开从数据库读取。
- 产品配置中的第一工况对应正转，第二工况对应反转；结果按工况 ID 更新共享指标行的对应列。分析完成的 SPL/FBA 显示返回的总体值、两位小数及单位；成功但没有有效总体值的项显示「分析完成」，失败或缺项显示 `--`。控件只渲染，不重新计算阈值。
- 工位状态行为「工位、正转、反转、结果」四列，展开的指标表仍为「指标、正转、反转」三列。方向状态及指标在各自结果返回时即时更新；工位结果复用现有轮次终态齐全标志，未齐全时保持 `--`，直到本轮最后一个工况分析终态返回后才刷新。方向判定和齐全后的工位结果均为任意 NG 则 NG、全 `--` 则 `--`、其余 OK。更新一个方向保留另一个方向、其他工位和折叠状态。
- 内部轮次把两个工况的任务关联起来，分别保存方向值和是否收到终态的标记。只有本工位本轮双方终态齐全时才统计一次；第一条结果、未知任务和重复终态不增加计数。双方均已返回 `--` 时计为未标记，OK 与已返回 `--` 的组合计为 OK。
- 工位结果和主界面统计继续等待本轮正反转终态齐全，保留原轮次规则与 `bind_result_summary` / `set_daily_records` 入口。「查看明细」通过独立的 `bind_wav_details` / `set_daily_wav_records` 接收保存成功的 WAV 快照，每个完成描述符中的文件登记一行未标记，再由首次合法单文件分析终态更新结果，不等待另一工况。成功文件不因无分析或归档失败消失；合法旧任务只更新其冻结文件身份。
- [`WorkbenchWavState`](../../base/workbench_wav_state.py) 按请求与源工位身份保存实例内账本，文件名取实际完成路径的 basename，日期与时间取该请求冻结的录制开始时间；[`DailyWavRecordStore`](../../base/motor_daily_wav_records.py) 独立校验、统计、筛选并分页 WAV 快照。明细六列为完成时间、工位、文件名称、工况、结果、操作，文件名保留完整 tooltip。第一、第二工况显示正转、反转，第三个及更多工况保留该请求冻结的配置名称。明细的 WAV 总数与合格率可不同于主界面轮次统计，两种快照不得互相覆盖。
- 新一轮正常接纳录制后重置当前显示；拒绝录制保留原显示。第一工况失败后重试，使用最终产生正转 WAV 的那次录制开始时间。每轮统计日期及记录时间来自传给 WAV 请求的同一份第一工况冻结录制时间，不取分析回调时间。
- 已知旧轮次的迟到结果只补齐所属轮次及当日汇总，不覆盖当前显示；跨日轮次结果归属正转录制日，WAV 明细分别归属每个请求的录制日。切换产品或开始新一轮保留当日累计，相同来源及快照的重复刷新保留显示。
- 首页通过独立的 `bind_summary_counts` / `set_summary_counts` 接收日期与 OK、NG、未标记计数快照。启动时只读一次 `ui/ui_config/workbench_summary_state.json`，有效同日快照作为基数叠加本次运行的配对结果。WAV 明细启动时只读一次 `ui/ui_config/workbench_wav_details_state.json`，首次发布即合并有效同日恢复行与本次录制。联合身份重复时保留恢复行及其判定，分析引用保留原 WAV 路径和通道。两份文件缺失或过期时各自从空状态开始，读取失败记日志。现有分析轮询在无事件时也采样本地日期，跨日刷新首页和已打开的明细窗口，旧日迟到终态仍归属原录制日。
- 正常关闭在录制、归档、原始 CSV、分析事件和音频标签收尾之后，采样一次本地日期，据此发布首页和明细的最终快照，并分别原子保存一次。刷新、保存及失败提示期间忽略重入的关闭事件。两份快照均尝试保存后分别记录失败日志，并提示「本次新增统计未保存」或「本次 WAV 明细未保存」。失败文件保留上次提交版本，退出正常完成。
- 旧轮次、旧配置或旧产品的合法迟到终态仍回写其冻结路径所指的旧 WAV，不能用当前工位汇总或当前产品路径替代。应用开始关闭后继续接收这些已接纳任务的终态；退出须实际消费 `service_closed` 并完成同一次轮询返回的整批事件（包括重入期间的后续终态），不能只检查分析服务的 `closed` 属性。此屏障之后才关闭标签提交入口并非阻塞地排空：后台线程已退出且所有成功/失败通知已交付 GUI 后才允许最终退出。
- `SequenceToolsBar` 不创建模式选择和手动分析控件或其隐藏兼容对象。录制后的自动分析保持现有入口。

修改结果关联和统计规则时，更新 [结果同步测试](../../unit_test/ui/test_workbench_result_sync.py)；修改 WAV 明细时，更新 [日快照测试](../../unit_test/base/test_motor_daily_wav_records.py)、[身份账本测试](../../unit_test/base/test_workbench_wav_state.py)、[明细窗口测试](../../unit_test/ui/test_motor_daily_details_dialog.py) 及 [WAV 链路测试](../../unit_test/ui/test_workbench_wav_details_sync.py)，并用 [合成渲染脚本](../../scripts/render_motor_daily_details.py) 核对六列与独立统计。修改标签路由、真实持久化或退出屏障时，更新 [标签链路测试](../../unit_test/ui/test_workbench_audio_label_sync.py) 及 [分析生命周期测试](../../unit_test/ui/test_workbench_analysis_lifecycle.py)。修改工具栏显隐时，更新 [工具栏测试](../../unit_test/ui/test_sequence_toolbar_responsive.py)。

## 每日数字轮次

- 主窗口在配置与录制入口开放前，通过 `bind_round_counter` 注入 [`WorkbenchRoundCounter`](../../base/workbench_round_counter.py)、[`WorkbenchRoundStore`](../../base/workbench_round_store.py) 和 `SequenceWindow.show_workbench_round()`。工具栏原样本位置显示只读「轮次」，全软件同一自然日共用数字，切换产品不重置。
- 状态单独保存在应用根目录 `ui/ui_config/workbench_round_state.json`，包含版本、日期和正整数。首次缺失从当天 1 初始化，不从 `recorded_number.json` 的旧样本或隐藏轮次字段迁移。工作台不恢复或保存这两个旧字段，型号与 S/N 仍走已有保存入口。不创建隐藏旧轮次兼容对象。
- 内部 UUID 关联录制与分析任务；每日数字用于展示和新 WAV 原样本编号槽位，不能用数字替换 UUID。第一工况接纳时冻结数字和日期，同轮后续工况复用；WAV 自身时间戳继续取每次录制开始时间。
- 全部工况成功保存并完成归档收尾后，全部应等待分析完成与准备下一轮录制两个入口中先发生者推进并持久化一次。等待集合按冻结的工况、适用工位和分析项构造，不看全局队列空闲。旧轮迟到或重复结果继续补齐原有结果与日汇总，不再次推进当前数字；没有分析项时不等待不存在的事件。
- 现有 100ms 硬件状态刷新经 `refresh_recording_state()` 检查本地日期。空闲换日保存并显示 1；旧日仍有工况未录完时保留整轮数字，收尾后新日从 1 开始，旧日分析不能把新日 1 推到 2。重启只恢复已保存数字，工况从第一项开始，不恢复任务或进度。
- 状态读取或保存失败通过现有 GUI 通知报告，损坏文件不能静默覆盖。准备的新数字未保存成功时不得开始相应录制；后续刷新重试同一待提交转换，不能重复加数。保持现有分析诊断，不新增分析重试或失败恢复策略。

修改数字规则时，更新 [纯计数器测试](../../unit_test/base/test_workbench_round_counter.py)、[控制器轮次测试](../../unit_test/ui/test_workbench_round_counter_integration.py) 和 [主窗口集成测试](../../unit_test/ui/test_motor_workbench_integration.py)。

## 工位指标表展示

- 工位名称、状态、指标正文及两类表头统一宋体 12px，区域标题保留 14px；继续使用现有全局字号缩放。三列指标表按视口宽度分配 40% / 30% / 剩余宽度，首列左对齐，正反转表头与数值居中，单元格及表头左右内边距 6px。外层四列状态行仍为 46% / 其余三等分，保留折叠、自动换行与无横向滚动。指标行保留原有最小行高（34px），长文本按当前列宽增长，缩放后重新计算行高和表格总高，避免裁掉正文及单位。
- 左右工作台声压单位复用 `workbench_units.format_workbench_spl_unit()`，按实际计权显示 `dB(计权)`，包含 A/B/C/D/Z，缺省、None 及 Z（None）显示 Z。右侧仍从已加载配置取值；左侧优先返回的 `metrics.weighting`，缺少该字段时保留 `metrics.unit` 或无单位，不能由旧单位猜计权，不查询产品配置。
- 调整这些展示规则时更新工位面板、结果状态和投影测试，并用 `scripts/render_motor_station_panel.py` 检查 300/340/600px 实际控件，确保常见数字、单位和「分析完成」完整显示。

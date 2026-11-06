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

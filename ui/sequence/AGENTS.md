# 电机测试工作台

- 由 `MainWindow` 组装 `SequenceWindow`、`MotorWorkbenchController` 和视频服务。所有新增的跨控件业务更新经控制器提交。控件接收显示数据、发出用户意图并管理折叠、滚动、缩放等局部状态。
- 通过控制器协调产品、测试队列、录制、分析和数据库业务。控件之间通过控制器传递业务意图。工作台仓储只读配置。配置端在成功提交产品或队列文件及注册表后，调用 `MainWindow.on_workbench_configuration_committed()`。工作台用落盘快照更新视图。
- 生产工作台由控制器从已保存的活动产品生成工位。控制器通过 `MotorStationTestPanel.show_workbench_stations` 和 `StationPlotGrid.show_workbench_columns` 提交同序工位。无活动产品时传入空序列，保留区域标题并显示零工位和零列。
- 独立构造的 `StationPlotGrid` 默认提供两行三列。独立调用者可通过 `set_grid_size` 设置尺寸并用网格公开接口设置列名、选择项和图表数据。工作台使用 `show_workbench_columns` 按产品配置重建两行空图。

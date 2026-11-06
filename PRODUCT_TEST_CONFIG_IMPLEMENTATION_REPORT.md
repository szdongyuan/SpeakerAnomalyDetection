# 产品测试配置阶段实施与验收记录

更新日期：2026-11-06。

## 1. 当前结论

产品配置的主体编辑、字段校验、文件读写和引用保护已实现，界面反馈已完成多轮修正。已保存配置的切换编辑缺陷现已修复；复选框现已使用 Qt 原生控件，不再依赖新增对号图片。

最终收尾检查及默认通道行调整完成：本阶段最新相关回归 703 passed、1 skipped、1 warning、18 subtests passed，未发现本轮产品配置改造的新增提交阻塞项，可以按配置阶段范围提交。此前两个待处理工作流用例已修正并通过。额外发现的 3 项 PCM24 测试失败已在未包含本轮改动的 HEAD 快照复现，单独记录于第 12 节。最新界面调整见第 13 节。尚未暂存、提交或推送。

新配置的采集与分析执行流程尚未接通，这是本阶段的明确边界；配置选择曾被该限制阻断，现已修复；执行能力仍未接通。

## 2. 已实现内容

- 前端拆分工位、通道映射、公共工况和局部样式；后端拆分模型、校验和队列上下文，复用原管理器、JSON 读写、引用扫描及删除服务。
- 产品字段为 project_id、project_name、result_root_directory、stations、conditions；内部对象使用稳定 UUID，另存重建身份与内部引用。
- 配置文件按项目名称保存，例如 `电机A.json`，不附加短编号；完整 project_id 仅作为内部稳定身份。名称修改时同步更新文件名、注册表和活动文件引用。
- 通道数组位置对应工位 WAV 位置，界面从第 1 通道开始，分析数据下标从 0 开始。名称可修改，不提供上下移动。
- 原队列编辑界面继续使用；分析项选哪个通道，就对应各适用工位 WAV 中的那个通道。普通队列不要求新增字段或执行转换。
- 已取消“另存为 WAV 通道队列”和转换时重置通道的逻辑。对已有 analysis_channel_basis 字段的读写及相关保护仍保留，不作为新队列必填条件。
- 共用队列检查各适用工位的通道存在性和信号类型。队列保存同时检查磁盘项目与草稿，草稿不再遮蔽磁盘引用。
- 移除电压、原始音频 WAV/CSV 选项及复制工况到端口；产品分段分析仅保留 none、time。公共历史负载解析逻辑未整体删除。
- 旧产品配置模块、旧窗口及对应旧格式专用测试已替换；旧 JSON 不自动迁移或覆盖。

## 3. 界面与提示修正

| 内容 | 当前行为 |
|---|---|
| 字体及颜色 | 系统已有宋体 SimSun，原浅蓝分区、浅灰背景 |
| 通道映射高度 | 默认一行待填写通道，按内容增长至四行，更多通道滚动 |
| 工况表 | 获得剩余纵向空间，分段分析位于最后一列 |
| 适用工位 | 白底复选菜单，空框/框内对号区分选择状态 |
| 分段弹窗 | 不分段、按时间，移除负载输入，缩短弹窗 |
| 工位双层边框 | 旧按钮立即隐藏后再延迟释放，避免模态事件循环中叠加 |
| 保存后的提示 | 已知执行限制不再弹出“请修正配置”，真实错误仍提示 |
| 主界面自动提示 | 不再把执行未接通当成缺少配置自动弹窗；测试按钮禁用并说明原因 |

实际 Qt 验证覆盖工位切换、双击改名、滚动、多选勾选、时间参数往返、保存通知和主界面首次/再次显示。重叠按钮问题通过“先启动外层事件循环，再打开已保存配置模态窗口”复现并验证修正。

本地截图位于忽略的 tmp 目录，包括 station-modal-before.png、station-modal-after.png、station-checkbox-final.png、segments-time-only.png 和 product-config-empty-two-rows.png。这些是不同轮次的局部证据，不作为单张完整最终界面的交付文件。

## 4. 缺陷处理与剩余事项

### 4.1 已保存配置切换编辑（已修复）

复现步骤：保存配置 A、B，保持活动配置 A；在主界面“使用配置”中选择 B；随后打开产品配置编辑器。

修复前结果：选择退回，注册表仍指向 A，编辑器仍加载 A。隔离目录中的实际 Qt 控件复现结果为：选择 B 后活动文件仍为 A，选择框索引变为 -1，编辑器项目名称为 A。

修复前原因：ui/sequence/sequence_widget_config_ops.py 的 on_using_file_combobox_changed 在保存选择前调用 build_product_test_refresh_snapshot。新执行未接通时该函数抛出 ProductTestRuntimeUnavailableError，导致配置选择一并失败。编辑器按注册表 active_file 加载，所以继续打开 A。

修正行为：配置选择路径单独处理 ProductTestRuntimeUnavailableError，文件读取和结构检查成功后允许保存活动配置；执行状态继续阻止测试。列表刷新及切换失败回退不再清空仅因执行未接通而不可运行的配置。注册表读写异常会提示并保留原选择。没有更改产品字段，也没有提前接通采集分析进程。

新增回归已通过：A→选择 B→打开编辑器显示 B→修改并保存 B→重开 B→切回 A；测试仍被禁用，A 文件未被覆盖。另覆盖目标文件损坏、丢失、注册表写入返回失败或抛出异常、当前测试锁定五种失败情况。

### 4.2 复选框图片依赖及字体测试（已处理）

适用工位菜单已改用 QWidgetAction 内的 QCheckBox，方框和对号由 Qt 绘制，原 check.svg 引用及对应自定义指示器样式已移除。该本地图片无需提交。报表筛选布局测试已改为检查共享字体常量，保留原有控件边界、间距及尺寸断言。

## 5. 前期验证记录（最终收尾见第 12 节）

2026-11-06 提交前针对性回归：**517 passed、1 skipped、2 warnings、18 subtests passed，59.05 秒**。

实际命令：

```powershell
$env:QT_QPA_PLATFORM='windows'
python -m pytest unit_test/test_product_test_project_config.py unit_test/test_product_test_project_config_dialog.py unit_test/ui/test_product_wav_queue.py unit_test/ui/test_shared_queue_save.py unit_test/ui/test_queue_display.py unit_test/base/test_sequence_queue_references.py unit_test/base/test_config_deletion.py unit_test/base/test_product_test_config_refresh.py unit_test/ui/test_product_test_program_runtime_refresh.py unit_test/ui/test_product_config_startup.py unit_test/ui/test_dialog_enter_queues.py unit_test/ui/test_config_delete_dialog.py unit_test/base/test_hardware_analysis_channel_sync.py unit_test/test_sequence_toolbar_style.py unit_test/ui/test_dialog_enter_analysis.py unit_test/base/test_analysis_segments.py -q --disable-warnings --tb=short
```

静态验证：44 个现存变更 Python 文件语法解析通过；git diff --check 通过。使用已有系统 Python 3.12 和 PyQt，未安装依赖或字体。上述 517 项为修正配置切换前的回归记录；本轮切换修正后的针对性回归为 55 passed、1 warning（7.11 秒）。

4.1 的切换问题最初由额外操作复现发现，本轮新增测试先复现失败，再验证修正通过。测试通过仍不代表完整采集分析执行链的验收。

历史扩展检查仍有 WAV 子类型 FLOAT/PCM_24 断言差异、仪表盘像素检查失败等记录；此前报表字体断言不一致已修正并重新验证；这些不是本轮 517 项测试的通过结论，也没有完成全部失败的基线归因。全量仓库、真实硬件、打包资源、现场节拍及长时间运行均未完成验收。

## 6. 前期提交准备记录（最终清单见第 12 节）

提交前检查时，工作区有 50 个 Git 可见变更文件，其中 35 个已跟踪文件、15 个未跟踪文件；已跟踪差异为 +1048/-10503 行。该数字包含此前多轮改造，不代表本轮文档更新的行数，且不包含被忽略的新图标和 tmp 文件。实际提交前需重新统计。

建议分组：

1. 产品配置实现、队列衔接、旧模块清理及回归测试，不新增图片资源。
2. 公共字体和共享下拉样式及对应测试，单独审查其他页面影响。
3. 本实施方案及记录随收尾状态核对后纳入；临时诊断、截图、用户配置、录音和日志排除。

本轮修改配置选择逻辑、对应回归测试及两份实施文档。没有暂存、提交或推送。复选框和字体测试收尾后，按更新后的候选文件进行最终提交检查。

配置切换修正后的实际验证命令：

```powershell
$env:QT_QPA_PLATFORM='windows'
python -m pytest unit_test/ui/test_product_test_program_runtime_refresh.py unit_test/ui/test_product_config_startup.py unit_test/ui/test_queue_display.py unit_test/test_product_test_project_config_dialog.py unit_test/base/test_product_test_config_refresh.py -q --disable-warnings --tb=short
```

## 7. 复选框与字体测试收尾验证

2026-11-06：43 passed、2 warnings、18 subtests passed（6.01 秒）。执行以下相关集合，并检查实际 Qt 菜单截图及复选框点击后的配置更新：

```powershell
$env:QT_QPA_PLATFORM='windows'
python -m pytest unit_test/ui/test_analysis_report_wav_dialog.py unit_test/test_sequence_toolbar_style.py unit_test/test_product_test_project_config_dialog.py -q --disable-warnings --tb=short
```

本轮不提交图片、不暂存代码、不执行 Git 提交或推送。

## 8. 项目名称文件名（2026-11-06）

保存服务已改为 `project_name.json`，例如 `电机A.json`。文件内部 project_id 及工位、通道、工况 ID 不因改名而变化。同目录下不允许仅大小写不同的重名文件；注册表未登记的同名文件也受保护。名称校验覆盖非法字符、Windows 保留名称、结尾句点、长度及 program_registry.json 冲突。另存为必须使用不同名称。

当前 stations/conditions 格式的 UUID 文件名仍可读取，再次保存该配置时才转换为项目名称。改名同步修改注册表和 active_file；非活动项目改名保持原活动配置。写入新文件、更新注册表或清理旧文件遇到可处理失败时返回错误并回滚；未提供跨文件断电事务保证，也未批量迁移用户已有配置。

验证结果：

- 第 5 节完整针对性命令重新运行：553 passed、1 skipped、2 warnings、18 subtests passed，58.59 秒。覆盖保存、导入、另存为、删除、队列引用及真实 Qt 控件的配置切换/改名/重新打开路径。
- 随后补充写入失败及同名另存等用例，运行 `python -m pytest unit_test/test_product_test_project_config.py -q --disable-warnings --tb=short`：58 passed、1 warning，1.71 秒。两组结果存在重叠，不累加统计。
- 新增针对性验证包括中文名称、大小写重名、大小写改名及回滚、未登记文件保护、UUID 文件名兼容、活动路径更新、非活动配置改名、旧文件删除失败回滚、写入失败保留原配置。
- git diff --check 通过。测试使用临时目录，未改动实际用户配置，没有暂存、提交或推送。

本轮更改配置保存服务和相关测试，并同步实施方案；测试队列格式、录音路径及采集分析执行边界保持此前约定。

## 9. 部分配置损坏时的共享队列校验（2026-11-06）

提交前检查发现：队列保存入口仅在引用扫描没有任何问题时执行通道校验。因此，即使 A 可正常读取，只要另一份 B 配置损坏，修改 A 队列时也会跳过通道检查。隔离目录对照验证中，相同的 SPL 振动通道选择在 B 正常时被拒绝，在 B 损坏时确认后写入。

修正：移除“扫描无问题”这一通道校验前提。可读取的已保存配置和当前草稿继续校验，损坏的工况结构单独跳过，由现有引用扫描及共享确认展示读取问题。SPL 选择振动通道、选择不存在的通道等确定错误在共享确认前返回失败，队列原文件不变；合法修改仍可按原有确认流程保存。既有带 analysis_channel_basis=wav 标记的额外限制保持不变。

新增 24 项组合回归，覆盖已保存配置和草稿、文件缺失、JSON 损坏、工况列表和条目结构损坏，以及合法/信号类型不匹配/不存在的通道。修复前 16 项错误通道用例失败（被错误保存），8 项合法修改通过；修复后全部通过。

实际验证命令：

```powershell
$env:QT_QPA_PLATFORM='windows'
python -m pytest unit_test/ui/test_product_wav_queue.py unit_test/test_product_test_project_config.py unit_test/ui/test_shared_queue_save.py unit_test/base/test_sequence_queue_references.py -q --disable-warnings --tb=short
```

结果：247 passed、2 warnings，13.74 秒；git diff --check 通过。所有测试使用临时配置，损坏文件没有被自动修复或覆盖。本轮涉及两个业务文件、一个测试文件及两份实施文档，未暂存或提交。

验收边界：此前扩展检查的结果为 639 passed、4 failed、1 skipped；4 项失败位于 test_vkinging_settings_workflow.py（两项弹窗可见性、两项采样率保存场景）。本轮仅修复共享通道校验，不将上述 4 项记为已解决，也不据此宣称整体已具备提交条件。

## 10. 支持仅录音队列（2026-11-06）

原队列目录将“分析项为空”判为不可用，导致合法的仅录音队列无法在工况中正常选择，相关配置保存也被拒绝。现调整为录音参数合法即可作为仅录音队列使用，工况“判定与分析”列显示“仅录音”；没有分析项不代表可以自动输出 OK/NG。保存后的提示同步改为可在产品测试配置中选择使用。损坏文件、无效录音时长/采样率、错误分析项顺序及不存在的已选分析项仍被拒绝。

验证命令与结果：

```powershell
$env:QT_QPA_PLATFORM='windows'
python -m pytest unit_test/test_product_test_project_config.py unit_test/test_product_test_project_config_dialog.py unit_test/ui/test_queue_display.py unit_test/ui/test_product_wav_queue.py unit_test/ui/test_shared_queue_save.py -q --disable-warnings --tb=short
# 254 passed, 2 warnings, 14.86 秒
python -m pytest unit_test/ui/test_vkinging_settings_workflow.py -q -k saved_profile --disable-warnings --tb=short
# 2 passed, 10 deselected, 2 warnings, 4.64 秒
```

新增仅录音队列的目录状态、选择、摘要和产品保存回归，以及无效录音参数/分析项的反向检查。修复前 3 项有效队列用例失败，修复后通过。实际 Qt 界面截图已核对，位于本地忽略目录 tmp/recording-only-condition.png，不纳入提交；git diff --check 通过。

此前 4 项待处理失败中的两个采样率保存场景现已通过；另外两项下拉弹窗可见性失败尚未处理。本轮修改 4 个业务文件、3 个测试文件和两份实施文档，未改变配置字段或提前接通采集分析执行链，未暂存或提交。

## 11. 录音参数确认后的产品字段弹窗（2026-11-06）

用户现场发现：在录制音频窗口确认参数后，出现缺少 project_name、result_root_directory、工位通道的保存失败提示。原因是子参数确认触发队列自动保存，保存检查对父产品草稿执行了完整产品必填校验，并将自动保存失败直接弹窗。

修正了两个边界：草稿仅检查与当前队列有关的参数及已建立的工位通道映射，产品名称、结果目录等必填字段留到产品保存时检查；未配置通道的草稿工位不作为已有映射处理。已保存产品继续完整校验，不能通过清空草稿绕过磁盘上的有效映射检查。自动保存失败保留 dirty 状态及原文件并记录日志，只有主动保存队列时显示错误。

新增回归在修复前复现了完整草稿要求和自动保存弹窗，修复后通过；另用真实 RecordConfigWindow 模态循环及确定按钮验证“父配置未填完整→修改录音时长→确认→队列自动保存”，没有产品必填字段弹窗。

针对性命令同第 9 节，结果为 257 passed、2 warnings，12.07 秒。测试数据均在临时目录，未改动用户配置。

同期复查：此前两个下拉弹窗用例在 showPopup 后立即检查可见性，等待界面处理后列表能够显示，尚无证据证明量程下拉本身无法使用；继续执行后，又暴露该测试构造的注册表缺少项目名称、产品数据仅为引用扫描所需的部分字段，不满足当前保存校验。此处未修改这两个测试，也没有放宽已保存产品的校验规则。

修正后的扩展回归（第 5 节集合加手动播放权限、串口工况、报表字体及 Vkinging 工作流测试）结果：676 passed、2 failed、1 skipped、3 warnings、18 subtests passed，61.46 秒。失败仍为上述两个下拉弹窗工作流用例。git diff --check 通过，未暂存或提交。

## 12. 最终收尾与提交核对（2026-11-06）

### 12.1 工作流测试修正

本次修改 test_vkinging_settings_workflow.py 中既有测试，未修改量程控件、录音格式、校准元数据或录音执行实现：

- 展开 Vking 量程列表后使用 QTest.qWaitForWindowExposed，最多等待 1 秒，再检查可见性及完整的 7 档选项。
- 使用现有产品配置 fixture 和真实保存服务创建两份完整配置，保留 Q、Q_alias 共享同一文件及 R 不受影响的校验。原引用扫描专用的残缺数据不再作为可保存产品使用。
- 断言对齐当前 HEAD 已有的 PCM24 和元数据版本 2（factor_source=calibrated）。PCM24 数据用独立字面量整数码检查，实时预览仍检查采集浮点幅值。
- 录音完成回调使用现有 prepare_main_recording_result 准备结果及校验最终波形包，保留完成接收、资源释放、通道顺序、采样率、量程及校准存储不被改写等检查。

独立执行该测试文件：12 passed、2 warnings，5.76 秒。随后关闭 pytest 缓存插件进行最终集合验证，避免当前环境对 .pytest_cache 的写入权限警告；未关闭业务断言。

### 12.2 最终针对性回归

```powershell
$env:QT_QPA_PLATFORM='windows'
python -m pytest unit_test/test_product_test_project_config.py unit_test/test_product_test_project_config_dialog.py unit_test/ui/test_product_wav_queue.py unit_test/ui/test_shared_queue_save.py unit_test/ui/test_queue_display.py unit_test/base/test_sequence_queue_references.py unit_test/base/test_config_deletion.py unit_test/base/test_product_test_config_refresh.py unit_test/ui/test_product_test_program_runtime_refresh.py unit_test/ui/test_product_config_startup.py unit_test/ui/test_dialog_enter_queues.py unit_test/ui/test_config_delete_dialog.py unit_test/base/test_hardware_analysis_channel_sync.py unit_test/test_sequence_toolbar_style.py unit_test/ui/test_dialog_enter_analysis.py unit_test/base/test_analysis_segments.py unit_test/test_manual_playback_permission.py unit_test/test_serial_product_condition_runtime.py unit_test/ui/test_analysis_report_wav_dialog.py unit_test/ui/test_vkinging_settings_workflow.py unit_test/ui/test_config_dialog_theme.py -q -rs --tb=short -p no:cacheprovider
```

结果：**699 passed、1 skipped、1 warning、18 subtests passed，70.86 秒**。跳过项为删除服务的符号链接测试，原因是 Windows 返回 WinError 1314（没有创建符号链接权限）；警告为 hardware_window.py 使用 cgi 的既有弃用提示。日志保存在本地忽略目录 tmp/product-config-scoped-regression-20261106.txt。

45 个现存变更 Python 文件通过 AST 语法解析，git diff --check 通过。删除的旧配置模块没有残留直接导入；仍存在的旧命名菜单回调不等于引用已删除模块。check.svg 无代码引用，不需要提交图片。真实采集硬件、打包、现场节拍和长时间运行未在本轮验收。

### 12.3 额外检查的既有失败

额外加入 unit_test/base/test_pcm24_pipeline.py 时，结果为 701 passed、3 failed、1 skipped。3 项失败均在该文件第 113 行：旧测试要求主录音 sample_digest 为 SHA256 字符串，当前实现不再为主录音生成该字段。最终波形包校验也明确拒绝主录音携带该旧字段。

通过 git archive HEAD 在 tmp/product-config-head-xen641vz 创建隔离快照，基线为 22883b568f382089bf4de4724c137c3229f81ecb。在该快照执行以下命令，得到相同的 3 项失败、2 项通过（9.69 秒）：

```powershell
python -m pytest unit_test/base/test_pcm24_pipeline.py -q --tb=line -p no:cacheprovider
```

失败参数为 soundcard_mono-main、soundcard_multi-main、ve-main，均未运行到本轮修改的分析任务构建代码。基线日志为 tmp/product-config-head-pcm24-20261106.txt。该文件及录音实现保持不变，作为独立测试维护事项保留；不能把针对性回归通过表述为全仓库测试全部通过。

### 12.4 提交范围和结论

- 当前候选共 51 个文件：36 个已跟踪变更、15 个新增文件。已跟踪差异为 +1450/-10509 行，不含新增文件；这是整批多轮改造的累计差异，不是本次测试收尾的变更量。
- 本次收尾实际修改 1 个测试文件及两份实施文档；临时日志、基线快照和候选清单位于忽略目录。
- 建议将配置实现、旧模块删除、关联回归及实施文档作为配置阶段提交；公共字体和共享下拉样式的 3 个文件单独提交，便于检查其他页面影响。
- 本地候选清单：tmp/product-config-submission-candidates-20261106.md。不纳入截图、check.svg、用户配置、日志、录音及临时诊断文件。
- 配置选择和编辑已与测试执行许可分开；自动化设备适配层和新采集分析链路不属于本次完成内容。可以提交配置阶段代码，不代表完整多工位系统已可运行。

未执行 Git 暂存、提交、推送或分支切换。

## 13. 默认一行通道映射（2026-11-06）

新建项目和新增工位均默认包含 CH1 待填写行；信号类型及硬件引用不预设。添加后续通道沿用此规则，已有配置原样加载。保留原表头，表格实际显示一至四行，超过后滚动。信号类型列调整为 180 像素，以完整显示“请选择信号类型”。最后一行移除按钮禁用，保留说明提示；多行删除仍需确认，并按原规则更新 WAV 顺序。

产品保存继续执行完整校验，提示具体工位及 WAV 通道缺少信号类型或硬件通道。空白行只存在于编辑草稿，不会以合法映射写入配置文件。共享队列编辑允许尚未明确用途的草稿通道，但继续检查其他已明确通道，磁盘上的产品引用仍执行完整校验。未新增配置字段或迁移用户已有文件。

新增测试在修改前复现三个失败：新建项目没有默认行、表格至少占两行、保存提示未区分待选字段。修改后新增场景通过，另覆盖新增工位的独立 ID、最后一行不弹删除确认、删除至一行、第五行滚动、填写后保存和读取、占位行不阻止队列编辑且不遮蔽其他通道错误。既有测试 fixture 明确设置自己的有效通道，不再依赖新建草稿为空。

验证记录：

- 产品配置服务、配置窗口及共享 WAV 队列三组：120 passed（9.02 秒）。
- 按第 12.2 节同一命令运行完整相关集合：703 passed、1 skipped、1 warning、18 subtests passed（67.59 秒）。跳过和警告原因同第 12.2 节；日志为 tmp/product-config-default-row-regression.txt。
- 随后仅加宽信号类型列，复跑配置窗口：19 passed（5.22 秒）。实际 Qt 截图检查了一行、四行布局及完整提示文字，截图位于 tmp/product-config-default-one-row.png 和 tmp/product-config-default-four-rows.png，不纳入提交。
- git diff --check 通过；新增逻辑复查未发现需要增加的抽象或兜底。最后一行删除保护与保存校验分别位于交互入口和持久化边界。

本轮涉及 4 个业务文件、3 个测试文件和两份实施文档。所有文件均属于已有未提交改造，本轮未暂存、提交或推送；没有改动用户配置或实际硬件。

## 14. 与远端最新基线集成及提交核对（2026-11-06）

本节记录本次提交前的最终集成结果；第 12、13 节属于原工作目录上的历史验证，不代表最新基线的完整结果。

从远端 develop_hsdk 的 58815a5 创建 codex/product-test-config-20261106，使用独立工作目录整合候选变更。提交前再次拉取，远端基线未变化。原工作目录及未提交文件保留。

### 14.1 集成范围

- 保留最新主窗口与 MotorWorkbenchController 的刷新职责，只将产品窗口成功保存后的通知接至 projects_changed 信号。取消或失败不触发配置提交刷新。
- 更新主窗口集成测试及配置刷新测试，使其使用最新 StationPlotGrid 和控制器接口，不恢复旧波形面板实现。
- 保留上游新增的字体、菜单及工具栏断言；字体期望使用当前公共字体常量。共享字体沿用本项目约定的宋体，未新增字体资源。
- 新增真实产品保存后由 MotorWorkbenchConfigRepository 读取的回归，检查项目、工位、工况、物理通道及分析通道引用。
- 最终候选 52 个文件，包括为最新主窗口通知接口补充的 test_motor_workbench_integration.py。配置主体及集成测试、公共样式分为两次提交。截图、用户配置、录音和 tmp 目录不纳入提交。

### 14.2 最新验证结果

使用第 12.2 节的相关集合，移出单独验证的 test_sequence_toolbar_style.py，并加入以下三组工作台回归：

```text
unit_test/base/test_motor_workbench_config.py
unit_test/ui/test_motor_workbench_controller.py
unit_test/ui/test_motor_workbench_integration.py
```

QT_QPA_PLATFORM=windows，pytest 参数为 -q -rs --tb=short -p no:cacheprovider。结果：**805 passed、1 failed、1 skipped、1 warning，92.51 秒**。失败项为 test_default_layout_is_loaded_from_the_application_root：仓库没有跟踪 ui/sequence/motor_workbench_plot_layout.json，干净检出缺少该默认布局文件。对应实现和测试均未被本次修改。跳过项仍为符号链接权限不足，警告仍为 cgi 弃用提示。

公共字体测试单独验证：

```powershell
$env:QT_QPA_PLATFORM='offscreen'
python -m pytest unit_test/test_sequence_toolbar_style.py -q -p no:cacheprovider --tb=short
```

结果：**17 passed、1 warning、9 subtests passed，3.48 秒**。Windows 原生平台下同文件为 16 passed、1 failed；失败发生在寻找已显示的 tooltip 标签时，未进入字体断言。

使用 git archive 在独立目录还原未修改的 58815a5，执行上述默认布局及原生 tooltip 两项失败用例，同样得到 **2 failed**（8.57 秒）。因此两项均可在本次变更前复现，没有通过删除断言或添加用户本地配置使其表面通过。日志位于本地忽略目录 tmp/product-config-latest-regression.txt、tmp/product-config-latest-baseline.txt。

### 14.3 交付边界

本次交付产品配置编辑、持久化、校验、队列引用及主界面配置通知。自动化设备消息适配层、新采集与各工位分析执行链仍未接通。现有工作台只读展示的工位数量限制不在本次扩展范围。旧 test_groups 格式不自动迁移；配置格式不支持时明确拒绝读取并保留文件。

本轮没有真实硬件、打包或长时间运行验收；不得将上述相关回归结果表述为全仓库全部测试通过。Git 提交及 PR 信息以实际创建结果为准。

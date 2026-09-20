# Enter Confirmation Semantics Implementation Plan

> **For agentic workers:** REQUIRED: Use the `subagent-driven-development` skill to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为已确认范围内的界面统一回车确认行为，保护参数编辑，令校准及音频管理回车无按钮响应，并将四类确认框的回车改为正向确认。

**Architecture:** 新建由对话框实例拥有、显式接入的 Qt 回车策略，窗口注册现有确认按钮或无目标。策略处理输入上下文、按键传播、焦点与模态窗口归属；所有业务操作继续通过原按钮点击路径执行。公共分析参数组件允许随必要共用改动影响被排除的分析项，不增加兼容分支。

**Tech Stack:** Python 3.12、PyQt5 / Qt 5.15、pytest、PyQt5.QtTest、现有 UI 测试夹具。

---

## 输入、范围与执行约束

- Spec: `docs/double-sdd/specs/2026-09-18-enter-confirm-semantics-design.md`，以用户最终确认版本为准。
- 本计划承接 writing-specs 的精确 runtime metadataPath：`.worktrees/20260918-enter-confirm-semantics.metadata.json`。后续原样传给 `subagent-driven-development`，不使用 `RUN_ID` 或 `NO_UPSTREAM_METADATA`。
- 用户批准执行计划后，从原始主工作区使用项目安装的 `.double-sdd/scripts/double_sdd/setup_worktree.py` 建立 `codex/enter-confirm-semantics` 独立工作树。必须使用 metadata 的不可变 `mainBase` 作为最终清理锚点；不要以临时规格提交作为新的清理基线。
- 所有下列命令在该独立工作树根目录运行。PowerShell 首先设置 `$env:QT_QPA_PLATFORM='offscreen'`；使用 `python -m pytest`。当前本机 Python 已有 PyQt5，测试目录 `unit_test/ui/conftest.py` 提供 `ui_qapp`。
- 任务按 1 → 6 顺序实施，每个任务交给新 implementer；每次实施后依 `code-review` 技能并行启动 spec/quality 审阅，先判定真实规格问题再采纳质量意见。全部任务结束后执行任务 7 的最终验证。
- 实施、文档和预存状态提交均为临时检查点；最终按 finishing 技能保留业务文件内容为未提交改动，不创建用户未批准的永久提交。不得把本地未跟踪文件、数据库、录音、运行资源或其他任务的工作树加入提交。
- 只验收 SPL、Spec、FBA、FFT 参数页。SPLF、FR、HD、RB、PRB、LP、LOUD、参考频谱、AI 分析配置不做专项适配、隔离或回归；公共参数组件改变其回车行为可接受。AI 训练/评估/模型管理/模型信息/音频选择仍不接入新策略。
- 鼠标、空格、Tab、Escape、修饰键组合、扫码输入与通道位置编辑保持原行为。原生文件选择框及未列出的通用消息框不改默认行为。
- 使用实例或类范围状态，不新增应用全局“当前确认按钮”、全局 Enter 快捷键或一次性模块级常量。禁止宽泛捕获异常后静默放行到任意默认按钮；不包装或重写已有业务异常与校验边界。

## 文件责任分配

| 单元 | 负责文件 | 职责 |
| --- | --- | --- |
| 共用策略 | 新建 `ui/dialog_enter_policy.py` | 显式接入、输入分类、事件与生命周期、目标点击 |
| 系统与账户窗口 | `ui/acquisition_config_window.py`、`ui/hardware_window.py`、`ui/serial_discrete_input_config_dialog.py`、`ui/video_settings_dialog.py`、`ui/calibration_window.py`、`ui/login_window.py` | 确定/保存/关闭/无目标注册 |
| 分析参数与子弹窗 | `ui/ui_analysis_config/common_widgets.py`、`ui/ui_analysis_config/threshold_config_widget.py`、`ui/ui_analysis_config/curve_color_config_widget.py`、`ui/output_load_config_dialog.py`、`ui/custom_ui_widget/audio_clip_extraction_dialog.py` | 四类参数页及公共子弹窗接入，保留校验 |
| 报告与归档 | `ui/analysis_report_export_dialog.py`、`ui/analysis_report_wav_dialog.py`、`ui/archive_audio_filter_dialog.py`、`ui/archive_audio_package_dialog.py`、`ui/archive_audio_data_dialog.py`、`ui/archive_audio_analysis_dialog.py`、`ui/segmented_analysis_results_dialog.py` | 报告/筛选/打包目标及无操作窗口 |
| 队列与产品配置 | `ui/operation_sequence.py`、`ui/product_test_project_config_dialog.py`、`ui/product_test_program_config_dialog.py` | 保存、复制工况、动态编辑器 |
| 四类确认框 | `ui/archive_audio_delete_dialog.py`、`ui/shared_queue_save_dialog.py`、`ui/sequence/sequence_widget_round_reset_ops.py`、`ui/sequence/motor_mode_switch_panel.py` | 正向确认与默认按钮显示一致 |
| 测试 | 新建 `unit_test/ui/test_dialog_enter_policy.py`、`unit_test/ui/test_dialog_enter_system.py`、`unit_test/ui/test_dialog_enter_analysis.py`、`unit_test/ui/test_dialog_enter_archive.py`、`unit_test/ui/test_dialog_enter_queues.py`、`unit_test/ui/test_dialog_enter_confirmations.py`；修改已有相关回归测试 | 使用真实 Qt 键盘事件与实际窗口，临时数据隔离 |

`ConfigDialogBase`、`AudioDataManageDialog` 默认不启用策略，不为继承关系添加隐含功能。四个参数页可通过已有共用 footer 完成接入，通常无需修改四个独立配置模块；若其初始化路径有差异，仅在对应 `spl_config_dialog.py`、`spec_config_dialog.py`、`fba_config_dialog.py`、`fft_config_dialog.py` 增加接入，不修改分析逻辑。

没有按钮的结果 `QWidget`（如 `AnalysisMultichannelResultWindow`）原本没有回车按钮响应，保留代码并以验证覆盖即可；不要为了凑接入数量修改无行为窗口。

## Task 1: 实现实例级回车策略与事件契约

**Create:** `ui/dialog_enter_policy.py`、`unit_test/ui/test_dialog_enter_policy.py`。

- [x] **Step 1: 写真实 Qt 失败测试，确定公共接口。** 导出 `install_dialog_enter_policy(dialog, confirm_button=None)`，返回一个由 dialog 持有的 QObject 策略。重复安装返回同一个策略并更新目标；对象提供 `set_confirm_button(button_or_none)`。没有调用安装函数的窗口不改变行为。下列完整测试作为首个红灯用例：

```python
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QDialog, QLineEdit, QPushButton, QVBoxLayout
from ui.dialog_enter_policy import install_dialog_enter_policy


@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Enter])
def test_editor_is_protected_and_auxiliary_focus_confirms(ui_qapp, key):
    dialog = QDialog()
    layout = QVBoxLayout(dialog)
    editor = QLineEdit()
    auxiliary = QPushButton("浏览")
    confirm = QPushButton("确定")
    for widget in (editor, auxiliary, confirm):
        layout.addWidget(widget)
    clicks = []
    auxiliary.clicked.connect(lambda: clicks.append("auxiliary"))
    confirm.clicked.connect(lambda: clicks.append("confirm"))
    policy = install_dialog_enter_policy(dialog, confirm)
    assert policy is install_dialog_enter_policy(dialog, confirm)
    dialog.show()
    dialog.activateWindow()
    ui_qapp.processEvents()
    try:
        editor.setFocus()
        QTest.keyClick(editor, key)
        assert clicks == []
        auxiliary.setFocus()
        QTest.keyClick(auxiliary, key)
        assert clicks == ["confirm"]
        confirm.setEnabled(False)
        QTest.keyClick(auxiliary, key)
        assert clicks == ["confirm"]
    finally:
        dialog.close()
        dialog.deleteLater()
        ui_qapp.processEvents()
```

- [x] **Step 2: 运行红灯。** `python -m pytest unit_test/ui/test_dialog_enter_policy.py -q`。期望缺少策略模块或与旧行为不符的断言失败；排除依赖导入等无关失败后再实现。
- [x] **Step 3: 按以下事件顺序实现最小策略。** 保留对象在 `dialog._enter_policy`，以 QObject parent 管理释放。窗口 show/子控件添加时更新辅助按钮的 `autoDefault/default`，观察动态表格编辑器；禁止无选择地改公共基类所有实例。识别最近所属顶层窗口，父策略不得操作子模态窗口。仅处理无修饰 Return/Enter；其余按键透传。先处理输入法 preedit/活动下拉框/菜单，再识别参数输入及祖先。控件自身允许处理原生编辑，传播到对话框的同一次输入回车必须被截断。不能只在父 `keyPressEvent` 看当时焦点，因为编辑完成可能改变焦点。为每次物理按键保留必要的实例级上下文，防止输入完成后变成父确认。非输入回车先消费事件，再至多一次 `confirm_button.click()`；无目标、隐藏/禁用目标直接消费。自动重复不得启动按钮操作。
- [x] **Step 4: 覆盖输入原生行为而非一律吞键。** 新增参数化用例：只读路径 QLineEdit、QSpinBox/QDoubleSpinBox 内部编辑器、QDateTimeEdit、可编辑/不可编辑 QComboBox、打开的下拉框、QTextEdit/QPlainTextEdit 换行、动态 QTableWidget 编辑器及编辑后焦点改变。记录 `editingFinished`/编辑结果与按钮点击，证明前者保留且后者为零。只读多行说明按非输入上下文处理。
- [x] **Step 5: 覆盖窗口与按键边界。** 加入无目标所有按钮零点击、辅助焦点确认、目标隐藏/禁用不回退、默认按钮视觉语义、重复安装不双击、重复 show、动态按钮、删除/销毁、两个独立窗口、嵌套原生/自定义模态窗口、按住回车后的重复事件。用 `QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier, "\r", True, 1)` 模拟 repeat；用 QInputMethodEvent 建立候选输入状态，再发送 Enter，确认不提交窗口。活跃原生子框若接收同一长按的重复事件，不得由父触发；确需阻断该长按重复时只在父本次按键到 release 的局部生命周期处理，不改变子框正常新按键语义。
- [x] **Step 6: 覆盖既有交互。** mouseClick 和 Space 仍点击辅助按钮，Tab 移焦，Escape 保留原有 reject，带修饰键事件不由策略新增触发。未安装策略的相邻窗口保持 Qt 原行为；单独演示未安装的 `ConfigDialogBase` 未改变。
- [x] **Step 7: 运行全部策略测试。** `python -m pytest unit_test/ui/test_dialog_enter_policy.py -q`，期望全通过，且测试不得通过 mock 掉键盘派发或核心分类函数实现。
- [x] **Step 8: 临时检查点并审阅。** `git add ui/dialog_enter_policy.py unit_test/ui/test_dialog_enter_policy.py`，`git commit -m "feat: add scoped dialog Enter policy"`。检查无应用全局确认状态、无宽泛捕获、无继承泄漏。

## Task 2: 系统配置、账户窗口与校准

**Modify:** 文件责任表“系统与账户窗口”六个文件。
**Create:** `unit_test/ui/test_dialog_enter_system.py`。
**Existing regression:** `unit_test/ui/test_recording_advanced_settings.py`、`unit_test/ui/test_input_calibration.py`、`unit_test/ui/test_video_controller.py`。

- [x] **Step 1: 在实际窗口上写失败测试。** 分别为录音、硬件、串口、视频读写两模式、登录/添加账号/修改密码构造实例；替换硬件发现、账户数据库及提交回调的外部副作用，再创建窗口，使 clicked 接到可观察处理函数。发送 Enter 验证参数零点击、辅助按钮焦点只触发目标。录音夹具沿用 `RecordConfigWindow({}, mic={"name": "Soundcard"}, speaker={"name": "Output"})`。
- [x] **Step 2: 为校准写失败测试。** 构造 CalibrationWindow 的输出与输入页，不启动真实采集；在数值框、播放/测试/保存/校准/重置/退出按钮及标签页焦点发送两种 Enter，所有业务按钮计数应为零。鼠标仍能触发现有回调。
- [x] **Step 3: 运行红灯。** `python -m pytest unit_test/ui/test_dialog_enter_system.py -q`，期望旧的取消/刷新/播放行为造成失败。
- [x] **Step 4: 接入显式按钮。** 各模块导入 `install_dialog_enter_policy`；按钮及原 clicked 连接完成后安装：录音 `ok_btn`（在其布局函数有引用时安装）、硬件 `self.ok_btn`、串口 `self.ok_btn`、登录 `self.login_button`、添加账号 `add_user_button`、改密码 `change_pwd_button`。不要将录音规则无条件安装到 BaseConfigWindow 所有潜在子类。视频使用 `self.buttons.button(QDialogButtonBox.Close if read_only else QDialogButtonBox.Save)`。校准在外层完成 UI 后执行 `install_dialog_enter_policy(self, None)`，现有 Input/OutputCalibration 只保留业务逻辑。
- [x] **Step 5: 运行目标及回归。** `python -m pytest unit_test/ui/test_dialog_enter_policy.py unit_test/ui/test_dialog_enter_system.py unit_test/ui/test_recording_advanced_settings.py unit_test/ui/test_input_calibration.py unit_test/ui/test_video_controller.py -q`。期望全部通过；硬件测试只观察选择确认信号，不访问真实硬件或注册表配置写入。
- [x] **Step 6: 临时检查点并审阅。** `git add ui/acquisition_config_window.py ui/hardware_window.py ui/serial_discrete_input_config_dialog.py ui/video_settings_dialog.py ui/calibration_window.py ui/login_window.py unit_test/ui/test_dialog_enter_system.py`，`git commit -m "fix: normalize Enter in system dialogs"`。

## Task 3: 四类分析参数与公共编辑子弹窗

**Modify:** `ui/ui_analysis_config/common_widgets.py`、`ui/ui_analysis_config/threshold_config_widget.py`、`ui/ui_analysis_config/curve_color_config_widget.py`、`ui/output_load_config_dialog.py`、`ui/custom_ui_widget/audio_clip_extraction_dialog.py`。
**Create:** `unit_test/ui/test_dialog_enter_analysis.py`。
**Existing tests:** `unit_test/ui/test_analysis_config_common_widgets.py`、`unit_test/ui/test_spl_spec_config_dialog.py`、`unit_test/ui/test_fba_config_dialog.py`、`unit_test/ui/test_fft_config_dialog.py`、`unit_test/ui/test_manual_limit_segments.py`、`unit_test/ui/test_plot_view_and_curve_style.py`。

- [x] **Step 1: 写四项实际配置的失败矩阵。** 沿用已有配置夹具，仅参数化 SPL、Spec、FBA、FFT；验证数值/通道/时间/上下限编辑回车无点击，非输入及“设为默认/恢复默认/取消/导航”焦点回车仅确认。确认必须走各项原本校验与保存回调，不直接 accept。至少测试一次 show/hide/show 后规则仍有效。
- [x] **Step 2: 为公共子弹窗写失败测试。** 上下限弹窗确认而不导出 CSV；颜色弹窗确认而不改变颜色；分段设置确定；片段提取确认而不加载音频。无效上下限和未选片段不能跳过现有校验。原生选择框 mock 为若被非预期调用即抛 AssertionError。
- [x] **Step 3: 运行红灯。** `python -m pytest unit_test/ui/test_dialog_enter_analysis.py -q`。
- [x] **Step 4: 接入并协调 showEvent。** `SemanticAnalysisConfigDialogBase._create_semantic_footer_layout` 注册 `self.semantic_ok_btn`；`AnalysisConfigDialogBase.create_standard_button_layout` 注册该方法的 `ok_btn`，消除 footer 代码创建的目标引用盲区。已接入窗口在 showEvent 交给策略刷新，未接入保留原行为。公共 footer 对其他分析项造成影响不做隔离。上下限/颜色分别注册 `self.confirm_button`；分段注册 `self.buttons.button(QDialogButtonBox.Ok)`；片段注册布局中已有 `ok_btn`。不修改 AI 训练等非参数窗口。
- [x] **Step 5: 更新相关旧断言。** 共用组件测试中若把“所有按钮都不是默认”当作目的，改为真实键盘行为：输入框不提交，非输入只确认。不能删掉输入保护覆盖，也不为 SPLF、LP、LOUD 等新增兼容条件。
- [x] **Step 6: 跑目标测试与四项回归。** `python -m pytest unit_test/ui/test_dialog_enter_analysis.py unit_test/ui/test_spl_spec_config_dialog.py unit_test/ui/test_fba_config_dialog.py unit_test/ui/test_fft_config_dialog.py unit_test/ui/test_manual_limit_segments.py unit_test/ui/test_plot_view_and_curve_style.py unit_test/ui/test_analysis_config_common_widgets.py -q`。旧测试文件可能含其他分析项：只记录排除项旧回车预期的失败，不用适配排除项来满足它；四项及其共用行为用例必须通过。其他类型的真实回归需按当前改动边界排查，不能一概忽略。
- [x] **Step 7: 临时检查点并审阅。** 对本任务确实改动的上述文件执行逐项 `git add -- <paths>`；若修改四项独立模块也必须包含其明确路径，禁止 `git add .`。`git commit -m "fix: confirm supported analysis dialogs with Enter"`。

## Task 4: 报告、筛选、归档与无操作结果窗口

**Modify:** 文件责任表“报告与归档”七个文件。
**Create:** `unit_test/ui/test_dialog_enter_archive.py`。
**Existing tests:** `unit_test/ui/test_analysis_report_export_dialog.py`、`unit_test/ui/test_analysis_report_wav_dialog.py`、`unit_test/ui/test_archive_audio_filter_dialog.py`、`unit_test/ui/test_archive_audio_package_dialog.py`、`unit_test/ui/test_archive_audio_analysis_dialog.py`。

- [x] **Step 1: 写实际窗口失败测试。** 报告选择项目/查看 WAV 焦点回车只进入导出入口；WAV 空/有数据明细窗口确认；WAV 筛选应用；归档筛选应用；打包继续。用现有 CandidateTableModel / ProjectReportIndex 测试数据，禁止用实际项目扫描作为测试前提。
- [x] **Step 2: 把报告防护写成键盘验收。** 沿用 `test_export_button_stays_actionable_and_validates_missing_project` 的 warning mock，用真实键盘进入 `_start_export`。未选项目精确提示“请先选择项目目录。”；已载入型号/样本但取消全部 WAV 时提示“请至少勾选一个 WAV。”；空索引先被型号校验挡住；缺分析项、缺可导出数值保持现有提示；这些情况下保存对话框和导出线程均不得启动。扫描/导出忙碌时目标禁用，回车零导出。
- [x] **Step 3: 写无目标及 AI 隔离测试。** 仅在 ArchiveAudioDataDialog 实例启用无目标，所有筛选/全部显示/排序/打包/删除按钮均不得因回车执行。用数据库 stub 和播放服务 stub。同时创建未安装策略的 AudioDataManageDialog/SelectAudioDataView，确认新策略没有继承泄漏；沿用前期审计的字典参数 `SelectAudioDataView(logger, {})`，不使用其错误的默认列表参数。
- [x] **Step 4: 运行红灯。** `python -m pytest unit_test/ui/test_dialog_enter_archive.py -q`。
- [x] **Step 5: 完成接入。** 注册报告 `self.export_button`，WAV 筛选 `self.apply_button`，WAV 明细 `self.confirm_button`，归档筛选 `self.apply_button`，打包 `self.continue_button`。ArchiveAudioDataDialog、ArchiveAudioAnalysisDialog、SegmentedAnalysisResultsDialog 注册 None；不改 AudioDataManageDialog 的基类初始化，不改筛选/删除/播放业务。普通 QWidget 结果窗不因共享策略改变，补零按钮响应验证即可。
- [x] **Step 6: 运行目标与回归。** `python -m pytest unit_test/ui/test_dialog_enter_archive.py unit_test/ui/test_analysis_report_export_dialog.py unit_test/ui/test_analysis_report_wav_dialog.py unit_test/ui/test_archive_audio_filter_dialog.py unit_test/ui/test_archive_audio_package_dialog.py unit_test/ui/test_archive_audio_analysis_dialog.py -q`，期望全通过。
- [x] **Step 7: 临时检查点并审阅。** `git add ui/analysis_report_export_dialog.py ui/analysis_report_wav_dialog.py ui/archive_audio_filter_dialog.py ui/archive_audio_package_dialog.py ui/archive_audio_data_dialog.py ui/archive_audio_analysis_dialog.py ui/segmented_analysis_results_dialog.py unit_test/ui/test_dialog_enter_archive.py`，`git commit -m "fix: route archive and report Enter actions"`。

## Task 5: 测试队列、产品配置与动态编辑器

**Modify:** `ui/operation_sequence.py`、`ui/product_test_project_config_dialog.py`、`ui/product_test_program_config_dialog.py`。
**Create:** `unit_test/ui/test_dialog_enter_queues.py`。
**Existing tests:** `unit_test/test_product_test_project_config_dialog.py`、`unit_test/test_product_test_program_config_dialog.py`。

- [x] **Step 1: 写失败测试。** 测试队列在树/列表未编辑、调整顺序/新建/导入/清空按钮焦点回车只保存；当前产品配置及旧版配置回车保存而不新增工况/配置；复制工况回车确认复制。参数框（包括项目名、关闭测试报文）和动态表格编辑器回车仅完成编辑。使用 tmp_path 管理器和现有假队列数据，不改当前用户配置注册表。
- [x] **Step 2: 运行红灯。** `python -m pytest unit_test/ui/test_dialog_enter_queues.py -q`。
- [x] **Step 3: 接入局部引用。** AnalysisModelSelect 创建底部 `ok_btn` 后安装；产品配置注册 `self.save_btn`；_CopyConditionsDialog 注册 QDialogButtonBox 的 Ok 按钮；旧程序配置注册 `self.save_btn`。当前产品配置的显式 `setAutoDefault(False)/setDefault(False)` 与策略协调，但不删除其端口/工况功能。后续插入的操作/设置按钮及单元格编辑器必须自动受策略管理。
- [x] **Step 4: 运行目标与回归。** `python -m pytest unit_test/ui/test_dialog_enter_queues.py unit_test/test_product_test_project_config_dialog.py unit_test/test_product_test_program_config_dialog.py -q`，期望全通过。共享队列确认流程的旧 Enter 预期留到紧接着的任务 6 更新，不在本任务扩大写入路径。
- [x] **Step 5: 临时检查点并审阅。** `git add ui/operation_sequence.py ui/product_test_project_config_dialog.py ui/product_test_program_config_dialog.py unit_test/ui/test_dialog_enter_queues.py`，`git commit -m "fix: confirm queue and product configuration with Enter"`。

## Task 6: 四类确认框与嵌套确认

**Modify:** `ui/archive_audio_delete_dialog.py`、`ui/shared_queue_save_dialog.py`、`ui/sequence/sequence_widget_round_reset_ops.py`、`ui/sequence/motor_mode_switch_panel.py`。
**Create:** `unit_test/ui/test_dialog_enter_confirmations.py`。
**Modify tests:** `unit_test/ui/test_archive_audio_delete_dialog.py`、`unit_test/ui/test_shared_queue_save.py`；按需补充 `unit_test/ui/test_round_reset.py`、`unit_test/test_recent_session_mode_switch.py`。

- [x] **Step 1: 将明确的旧测试改成正向期待并运行红灯。** `test_confirmation_displays_all_counts_and_enter_defaults_to_cancel` 重命名为回车确认语义；两种删除/移除分支都期待确认按钮为默认与 Accepted。`test_real_shared_message_decision_controls_queue_write` 的写入条件改成 `action in ("confirm", "enter")`，取消/关闭/Escape 继续断言不写入。不要删除关于文件范围、数据库记录和展示文本的断言。
- [x] **Step 2: 补充失败矩阵。** Return/Enter、取消按钮焦点、共享引用只读说明焦点都确认；Space 在取消按钮仍取消；Escape 取消。重置复选框未勾选返回原有“不删除而重置”选择，勾选返回删除选择，键盘本身不勾选。模式切换覆盖 test/mark 两方向。单次 Enter 在父保存打开共享确认后不能顺便接受子框，必须新按一次 Enter。
- [x] **Step 3: 运行红灯。** `python -m pytest unit_test/ui/test_dialog_enter_confirmations.py unit_test/ui/test_archive_audio_delete_dialog.py unit_test/ui/test_shared_queue_save.py -q`。
- [x] **Step 4: 接入与视觉同步。** 删除注册 `self.delete_button` 并撤销取消的默认/初始焦点设置；重置在局部 dialog 上注册 `confirm`，保留 checkbox 控制的文案、删除范围及重试状态；共享队列调用 `setDefaultButton(self.save_button)` 并注册该按钮，保留 `setEscapeButton(self.cancel_button)`；模式提示创建后 `setDefaultButton(QMessageBox.Yes)`，注册 `button(QMessageBox.Yes)`。使用共用策略处理取消焦点与只读说明，不把结果直接写成 Accepted 来绕过真实按钮路径。
- [x] **Step 5: 运行确认框及业务回归。** `python -m pytest unit_test/ui/test_dialog_enter_confirmations.py unit_test/ui/test_archive_audio_delete_dialog.py unit_test/ui/test_shared_queue_save.py unit_test/ui/test_round_reset.py unit_test/test_recent_session_mode_switch.py -q`，期望全部通过。删除测试只操作 tmp_path 与隔离数据库；不能在真实归档上人工确认删除。
- [x] **Step 6: 临时检查点并审阅。** 逐项添加本任务上述四个生产文件与实际修改的四个既有测试文件、新测试文件，`git commit -m "fix: use affirmative Enter in confirmation dialogs"`。

## Task 7: 集成验证与收尾

**Verify only:** 前六个任务的全部修改、规格与计划。若发现真实缺陷，回到拥有该边界的 implementer 修复并再次审阅，不由协调者直接改代码。

- [x] **Step 1: 检查范围及文件状态。** `git diff --check`；`git status --short`；依据规格逐行确认每个目标窗口都在六组测试中有实际实例路径。检查未修改 AI 训练/评估/模型管理/模型信息/音频选择与扫码源文件，未加入运行数据。
- [x] **Step 2: 一次运行所有新增键盘测试。** `python -m pytest unit_test/ui/test_dialog_enter_policy.py unit_test/ui/test_dialog_enter_system.py unit_test/ui/test_dialog_enter_analysis.py unit_test/ui/test_dialog_enter_archive.py unit_test/ui/test_dialog_enter_queues.py unit_test/ui/test_dialog_enter_confirmations.py -q`。期望零失败，完整测试输出作为证据。
- [x] **Step 3: 验证扫码隔离。** `python -m pytest unit_test/test_product_round_barcode_lock.py -q`；再用测试中的 fake host 或安全 UI 实例检查 S/N 原 `returnPressed` 和方向标签 `editingFinished/returnPressed` 未被策略拦截。无需实际扫描器，不触发真实采集。
- [x] **Step 4: 原生 Windows 核验。** 以新的独立诊断进程使用 `QT_QPA_PLATFORM=windows`，仅创建测试用对话框/隔离实际窗口，由 QTest 分别发送 Enter 到 spinbox、下拉列表、动态表格编辑器、辅助按钮与嵌套确认；覆盖一个原生窗口 IME 输入/候选状态。诊断按钮连接信号记录，无文件删除/采集/账户写入。退出后恢复 offscreen 测试环境。若当前会话无法完成真实 IME 候选交互，明确报告该人工覆盖限制，并保留 QInputMethodEvent 自动测试证据，不宣称完成手动输入法验证。
- [x] **Step 5: 复用前六个任务的回归证据。** 不无理由重复整库测试；任务间又修改过共用策略时，重新运行受影响任务的精确回归命令。被排除的分析项只因旧回车预期变化的失败不阻挡交付，明确列出，不为其添加兼容代码。
- [x] **Step 6: 完成 review 与本地集成。** 按 `verification-before-completion` 与 `finishing-a-development-branch` 执行；传递同一 metadataPath。确认无活动 merge/rebase/cherry-pick/bisect，按记录的检查点边界清理临时检查点并保留文件内容；若实施期间原目录引入独立提交，保留其最新基线。保留原先 .gitignore 改动及未跟踪/忽略状态，不删其他任务工作树或文件。用户未批准永久提交时交付未提交业务改动。
- [x] **Step 7: 汇报实际结果。** 简述输入保护、校准/音频管理无响应、四类正向确认、只验收四项分析参数、现有报告防护保留、测试结果与原生/IME 核验限制。不能在仅完成文档时声称回车行为已修改。

## 实施批准与运行状态

用户已批准执行。Task 1–6 均已完成，分别通过规格及质量审查。Task 7 已完成：全部新增键盘测试与扫码回归共 379 项，在合并验证环境及交付后的原目录均通过；完整实现质量审查通过。

| 任务 | 新增键盘用例 | 该组完整回归 |
| --- | ---: | ---: |
| 共用规则 | 63 | 63 |
| 系统、账户、校准 | 47 | 267 |
| 四项分析参数与公共编辑窗 | 74 | 204 |
| 报告、归档、结果 | 54 | 125 |
| 队列与产品配置 | 76 | 173 |
| 四类确认框 | 56 | 188 |

测试使用 Windows 原生 Qt 后端和新的 D:/tmp 临时目录。现有 QMessageBox 在 offscreen 后端有独立可复现的原生崩溃，因此相关用例使用 Windows 后端。报告组一次退出崩溃在随后三次完整重跑中未复现。

共用策略已经覆盖真实小键盘事件的 KeypadModifier、输入回调打开子框、长按重复、动态编辑器与自动化输入法事件；真实输入法候选交互尚无人工验证。

实施期间原工作目录合入独立的视频修复。其重建的三个本地检查点与原检查点补丁完全一致；交付必须保留新的视频提交及原有 .gitignore 改动。

最终交付记录：

- 原工作目录交付后：379 passed，18 项第三方弃用警告，55.59 秒。此前各组现有回归均通过。
- 独立 Windows 原生消息诊断观察到 KeypadModifier，确认参数框不提交、辅助按钮焦点只确认一次。动态编辑器、下拉框、嵌套确认及模拟输入法事件已通过原生 Qt 测试；真实输入法候选交互未做人工验证。
- 额外合并的视频测试中，test_camera_failure_and_recovery_keep_recording_error_visible 在原分支单独运行也失败。其活动定时器会用 stub 的 ready 状态覆盖直接注入的失败状态；该独立既有测试问题未在本次修改。更广范围运行结果为 438 passed、1 项上述失败。
- 37 个交付路径与验证源逐项核对一致（允许 Git 的 CRLF 转换），保留原 .gitignore、原未跟踪文件和新合入的视频提交。业务与测试修改保持未提交；临时工作树及分支已清理，运行 metadata 已 completed。

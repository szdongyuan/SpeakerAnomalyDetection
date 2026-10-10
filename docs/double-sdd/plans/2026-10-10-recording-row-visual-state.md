# Recording Row Visual State Implementation Plan

> **For agentic workers:** REQUIRED: Use the `subagent-driven-development` skill to implement this plan. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 消除录制收尾组结果同步反复重设整表 QSS 的开销，保持工况面板所有业务和视觉行为。

**Architecture:** 在现有样式常量文件定义固定、按对象名限定的 QSS。按钮以 visualState、结果标签以 resultTone 表达视觉状态，仅对属性变化的控件局部 unpolish/polish/update；业务更新路径继续执行。初始化/重建/重置可遍历全部行，日常更新只同步目标行和旧/新查看行。

**Tech Stack:** Python 3.12、PyQt5、pytest、unittest.mock、独立进程 Qt 离屏性能对照。

---

## 输入与文件职责

- 规格：`docs/double-sdd/specs/2026-10-10-recording-row-visual-state-design.md`，全部条款为验收依据。
- 基线提交：`e8d4b3bab29a2778faa18e64476cfbf37c7bf342`。
- 修改 `consts/ui_style_const.py`：固定按钮/标签 QSS，颜色、悬停、字体遵循规格。
- 修改 `ui/sequence/motor_result_panel.py`：安装 QSS、状态计算、局部刷新以及现有生命周期调用点。
- 修改 `unit_test/test_motor_left_panel_layout.py`：将依赖旧动态 QSS 字符串的断言改为等价行为/渲染断言，不削弱业务断言。
- 修改 `unit_test/test_manual_product_condition_cycle.py`：仅补齐测试替身缺少的 default_logger，使既有收尾日志调用可执行；保持全部业务断言。
- 新建 `unit_test/ui/test_motor_row_visual_state.py`：真实 Qt 状态、渲染、调用数量和生命周期回归。
- 新建 `unit_test/ui/benchmark_motor_row_visual_state.py`：独立可运行基准（不由 pytest 自动运行），真实组同步业务入口、独立基线/当前进程、语义一致性检查和 JSON 输出。
- 新建 `docs/double-sdd/reports/2026-10-10-recording-row-visual-state.md`：测试与基准结果、环境、命令、样本、边界及证据位置。

在隔离 worktree 工作。下文 PowerShell 命令均从该根目录执行，解释器使用本机已安装的 `D:/Python/Python312/python.exe`。测试前 `$env:QT_QPA_PLATFORM='offscreen'`。不安装无关依赖，不运行真实采集、音频设备或业务数据库。

## Task 1: 固定 QSS 与定向状态转换

**Files:** 两个生产文件，现有布局/手动工况测试和新建状态测试（见上）。

- [x] **Step 1: 验证修改前回归基线。**

运行 `D:/Python/Python312/python.exe -m pytest unit_test/test_motor_left_panel_layout.py unit_test/test_test_task_status.py unit_test/test_manual_product_condition_cycle.py unit_test/ui/test_recording_gui_finalization.py unit_test/ui/test_sequence_analysis_process_ops.py unit_test/base/test_product_test_config_refresh.py -q`。记录真实结果；已有失败需确认基线归属。执行时已确认基线有23个失败均为共享测试替身缺少 default_logger；仅为该替身添加现有 logging.getLogger(__name__) 后重新验证，不修改生产逻辑或弱化断言。TEMP/TMP 定向到可写测试证据目录，规避系统临时目录权限问题。

- [x] **Step 2: 添加失败的状态及操作数量测试。**

复用布局测试的 QApplication/配置模式，用真实控件验证：初始化 normal/pending；用户查看 old→new；自动准备/采集清除 viewed；recording 优先；未知 tone 显示灰色而 row 保留原 tone；长文本宽度；结果/进度/信号仍更新。计数在初始化完成后开始，仅统计工况按钮/结果标签的 setStyleSheet、setProperty 和局部 polish。对未变化状态重复通知必须为零，对一个普通更新不能调用全行刷新。用 Mock 包装局部刷新函数及实际 QWidget 方法，不替换业务回调。

核心断言例（fixture 构造包含 a、b 两行，采用真实公开配置格式）：

```python
panel.select_condition('a')
assert panel.rows['a']['button'].property('visualState') == 'viewed'
panel.set_condition_result('b', '采集中', 'running')
assert panel.viewed_key == ''
assert panel.rows['a']['button'].property('visualState') == 'normal'
assert panel.rows['b']['button'].property('visualState') == 'recording'
panel.select_condition('b')
assert panel.rows['b']['button'].property('visualState') == 'recording'
panel.set_condition_result('b', 'OK', 'ok')
assert panel.rows['b']['button'].property('visualState') == 'viewed'
assert panel.rows['b']['labels']['result'].property('resultTone') == 'ok'
```

- [x] **Step 3: 运行新测试确认 RED。**

运行 `D:/Python/Python312/python.exe -m pytest unit_test/ui/test_motor_row_visual_state.py -q`。预期因缺少动态属性或重复样式操作失败，而不是导入/fixture 错误。

- [x] **Step 4: 安装固定 QSS，实现最小属性更新辅助函数。**

按钮保留 `testTaskConditionButton`；结果标签使用独立对象名例如 `testTaskConditionResult`。在创建控件时安装共享 QSS 一次。QSS 必须包含 normal/viewed/recording 各自 `:hover` 和 resultTone 四种颜色，数值全部逐项取自规格，结果字体继续使用现有 font 常量。不要让 QLabel 通配规则覆盖其它标签。属性更新逻辑如下（可用等价实例方法）：

```python
def _set_style_property(self, widget, name, value):
    if widget.property(name) == value:
        return
    widget.setProperty(name, value)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
```

每行状态根据 `row['result'] == '采集中'` 优先，否则 key==viewed_key，再否则 normal。tone 外观归一化为 ok/ng/running/pending；不改写 row tone。更新结果文字和 width 的逻辑保留，移除其反复 setStyleSheet。

- [x] **Step 5: 改造现有调用点，保持业务步骤。**

`set_condition_result` 在可能同步触发端口信号前捕获旧 viewed key，完成现有业务修改后同步目标 key 和受影响旧 key（去重）；不能因为属性没变提前 return。`select_condition` 处理旧/新 viewed key，保留重复点击详情折叠、emit 和自动选择语义。`_refresh_port_view` 保留显示/隐藏和汇总，只同步被清除的旧 viewed key。`reset` 在全行默认同步前清除 viewed，或清除后额外同步旧行，防止遗留高亮。初始化/配置重建执行一次全行初始化；不保留旧控件引用。保留现有 `_refresh_row_styles` 作为全行生命周期入口亦可，但普通更新不得调用它。

- [x] **Step 6: 补齐真实渲染与边界测试。**

通过 QWidget.grab 的背景/边框像素及结果 QLabel palette/文本像素验证全部视觉状态；用 Qt 原生 hover 事件路径或 QStyleOption 的 State_MouseOver 绘制验证三种 hover，不以字符串存在代替渲染。保存 normal/viewed/recording/结果图到测试临时目录并在最终报告保留证据。检查 result tone 不改变 name/progress 标签颜色。覆盖同状态不刷新、跨端口自动选择的同步重入、隐藏 recording 行、reset、空配置、完整重建及 preserve_results；分析失败/结果不完整/已完成通道结果不可被占位符覆盖。现有样式字符串测试换成属性加真实渲染/调色板断言。

- [x] **Step 7: GREEN 与检查点。**

运行 Step 1 的完整回归命令加 `unit_test/ui/test_motor_row_visual_state.py`，预期全部 PASS。运行 `git diff --check`。用精确文件路径 git add 五个文件，然后 `git commit -m "perf: use targeted Qt properties for condition row styles"`。这是临时检查点；编排者记录 SHA。交接 RED/GREEN 命令、通过数、风险及截图位置，等候规格/质量双审查。

## Task 2: 真实组同步基准和最终证据

**Files:** 新基准脚本、报告；若性能检查发现 Task 1 缺陷，通过同范围生产文件修正并重跑相关测试。

- [x] **Step 1: 构造可复现的真实调用链。**

参考 `unit_test/ui/benchmark_hidden_history.py` 的独立进程、日志重定向与 Host 模式。基准必须执行 `_update_current_recent_session_result`→`_update_recent_session`→`_refresh_manual_product_condition_results_from_group`→真实 MotorResultPanel，不替换成计时桩。只为外部设备/文件依赖提供内存输入；不实际录音/分析。使用真实配置格式生成 N=100/200，K=20/60/100 已完成 keys，最近历史限制20并断言；completed keys 与历史独立，跨录制保留。至少用一次正常的录制完成标记/历史入口验证构造契约。mark 为主，test 为分支对照。

- [x] **Step 2: 确认基线能暴露问题。**

脚本接受 `--baseline-root`、`--current-root`、`--output-dir` 和可选单场景参数；用 `git archive e8d4b3bab29a2778faa18e64476cfbf37c7bf342` 导出只读基线到独立临时目录。在基线进程记录回调 wall/CPU 时间、行按钮/结果标签 setStyleSheet 次数、setProperty 次数、polish次数及实际行数。200/100 基线应出现约 K*N 按钮样式调用，先保存该结果，再完善当前版本对照；不得改动基线源码来降低工作量。

- [x] **Step 3: 执行独立、顺序对照并验证语义。**

同环境、输入序列及计时边界，基线与当前分别启动独立进程，不能并行。每种场景至少1次预热、3次测量，保留每次原始样本、计数和中位数。计时边界为真实结果更新回调，事件队列清理在计时外并说明。比较基线/当前每行 result、tone、channel结果、选择、端口、viewed、汇总、历史内容（时间等固定化）而不是只比较总数。顺序重复同状态和真正改变一个/多个状态均需测；本次不允许用批量一次刷新的旧实验值冒充结果。

运行示例：`D:/Python/Python312/python.exe unit_test/ui/benchmark_motor_row_visual_state.py --baseline-root <导出的基线目录> --current-root . --output-dir <临时证据目录>`。预期语义一致，当前回调不再对已创建行按钮/结果标签 setStyleSheet，相同属性不再 setProperty/polish；200/100 的 mark 中位数较真实基线下降至少90%。时间不是 CI 硬阈值；未达到目标继续剖析残余开销，结构性断言必须通过。

- [x] **Step 4: 汇总可审查证据。**

报告注明 Python/Qt/平台、基线提交、命令和计时范围，列出六个 mark 场景及 test 对照、原始3样本、中位数、降幅、操作计数、语义比较结果、截图位置和现场未验证边界。结果由脚本输出计算，不手填估计值；不要承诺现场总收尾时间。必要证据保留在任务临时目录（非运行日志/真实录音）。

- [x] **Step 5: 最终回归与临时检查点。**

重新执行 Task 1 Step 7 的完整测试集合和 `git diff --check`。`git add -- unit_test/ui/benchmark_motor_row_visual_state.py`，`git add -f -- docs/double-sdd/reports/2026-10-10-recording-row-visual-state.md`，`git commit -m "test: verify row state rendering and group callback performance"`。交付全部样本及证据，编排者记录检查点、执行本任务双审查及全范围质量审查后按 finishing-a-development-branch 流程将最终内容作为主工作区未提交修改交付。

## 所有任务共同约束与审查重点

- 不新增全局缓存/第二份业务真相/异步任务/定时器；不改组同步业务接口。
- 不新增 broad catch、吞异常或整表重设 QSS 兜底；不为确定性归一化添加防御包装。
- 共享 QSS 遵循 consts 组织；单用途实现状态保存在实例/控件，不添加无必要模块常量。
- 审查明确区分样式操作 O(D) 与仍可能遍历的已有汇总逻辑，不夸大端到端复杂度。
- 保持公开接口、信号、数据库/配置格式和业务保护；不修改无关功能。
- 每个任务由一个 fresh implementer 顺序执行，随后 fresh spec-code-reviewer 与 quality-code-reviewer 并行审查同一 Base..Head；先验证规格反馈再处理质量反馈。全部完成后再进行整范围质量审查。
- 仅临时检查点，无推送/永久提交；整合时保持最终内容并清理本工作流历史，保护其它会话及用户修改。

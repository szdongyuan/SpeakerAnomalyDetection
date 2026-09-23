# Raw Audio CSV Process Implementation Plan

> **For agentic workers:** REQUIRED: Use the `subagent-driven-development` skill to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将原始音频 CSV 导出隔离到单个常驻子进程，在保持文件内容和原子发布的同时，以包含录音预留在内的 16 份总容量保护录音和 GUI。

**Architecture:** 应用注入一个 RawAudioCsvService，监督线程管理 spawn worker、任务账本及路径占用，Qt bridge 只投递小型事件。录音启动前预留名额，完成后原名额转为 FIFO 任务；所有文件修改及退出动作与任务资源释放协调。

**Tech Stack:** Python 3.12、multiprocessing spawn/Pipe、threading、PyQt5 queued signals、soundfile、pytest；Windows 源码及冻结应用验证。

---

## 输入、边界与执行约定

规格：`docs/double-sdd/specs/2026-09-22-raw-audio-csv-process-design.md`（用户已确认，独立规格审查通过）。这是一个功能的一份计划，以下任务顺序执行，每任务由新 implementer 实施、通过 spec/quality review 后再进入下一任务。当前仅规划；用户批准执行之前不创建实现或测试文件。

上游 writing-specs 已显式移交本次精确 metadataPath；执行调度必须继续显式传递收到的原值，不从目录扫描推断、不切换为无上游模式。运行路径和临时 checkpoint 标识只保留在运行元数据，不抄入本计划。不可变设计基点为 `dc9cf104e80179bc176c1dbc7fa57b36e0d69ce1`；实现从包含批准文档的当前状态建立隔离 worktree，最终通过 finishing-a-development-branch 按运行元数据清理临时提交并保留文件内容。

不要编辑未跟踪的 interval launcher、临时录音或任何既有用户文件。不要改 `.agents`/`.codex` 技能。不得把历史表格或间隔调度器优化扩入本功能。源码性能组需要旧线程模式时，只在 benchmark 中构造，不保留产品运行时线程回退开关。

命令均在实施 worktree 根目录运行，PowerShell 使用已安装的 Python。规划时核对到 Python 3.12.7、pytest 9.0.3、soundfile 0.12.1；不要自动更新依赖。规划基线已实际执行：系统默认 pytest 临时目录发生 WinError 5，改用仓库 tmp 下新建且未使用的 basetemp 后 7 项通过（1 项现有 pyqtgraph 弃用警告）。这是测试目录权限问题，无需修改 CSV 源码。

下列所有 pytest 命令必须使用新的 `--basetemp (New-CsvTestTemp)`；每次调用前在当前 PowerShell 会话定义以下函数，不能把已有目录交给 pytest 自动清理。

```powershell
function New-CsvTestTemp {
    $csvTempRoot = [IO.Path]::GetFullPath((Join-Path (Get-Location) 'tmp'))
    if (Test-Path -LiteralPath $csvTempRoot) {
        if ((Get-Item -Force -LiteralPath $csvTempRoot).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Reparse temp root' }
    } else {
        New-Item -ItemType Directory -Path $csvTempRoot | Out-Null
    }
    $csvFreshTemp = [IO.Path]::GetFullPath((Join-Path $csvTempRoot ('csv-pytest-' + [Guid]::NewGuid().ToString('N'))))
    if (!$csvFreshTemp.StartsWith($csvTempRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or (Test-Path -LiteralPath $csvFreshTemp)) { throw 'Unsafe basetemp' }
    return $csvFreshTemp
}
python --version
python -c "import pytest, PyQt5, soundfile; print(pytest.__version__, soundfile.__version__)"
$env:QT_QPA_PLATFORM = 'offscreen'
python -m pytest unit_test/base/test_raw_audio_csv_exporter.py unit_test/ui/test_raw_audio_csv_recording_integration.py -q --basetemp (New-CsvTestTemp)
```

预期：基线成功。若失败，先记录现有问题并按 systematic-debugging 定位；不把环境导入失败当作功能 RED。实施使用 TDD：每个行为批次先写测试、指定运行出现预期失败，再实现、运行变绿；最后运行任务完整范围。每任务依 verification-before-completion 留实际命令/结果，暂存仅该任务列明的文件，检查 `git diff --cached --check` 后创建临时 checkpoint（下列 commit 文案），把 SHA 记录到运行元数据。无需重复跑已通过且未受修改影响的全库测试。

### 跨任务接口约定

以下是要实现的接口合同，不是已存在 API。协议类用 frozen dataclass，所有事件含 task_id/generation，service token 只在父进程使用。

```python
@dataclass(frozen=True)
class CsvReservation:
    token_id: str
    owner_id: str

@dataclass(frozen=True)
class CsvAdmission:
    status: str  # accepted, full, closing, unavailable
    reservation: CsvReservation | None = None

@dataclass(frozen=True)
class CsvExportRequest:
    task_id: str
    recording_id: str
    wav_path: str
    csv_path: str
    raw_channels: tuple[int, ...]
    owner_group: str
    owner_record: str

# Service public operations: no process/file/pipe I/O on the caller thread.
# reserve(owner_id) -> CsvAdmission
# release_reservation(token) -> bool; duplicate release returns False.
# commit(token, request) -> accepted/invalid/path_busy/unavailable
# snapshot() -> immutable counts/state; observation never reserves.
# begin_shutdown() -> None; stops new reserve, honors existing tokens.
# subscribe(callback) -> subscription; events run on supervisor, final closed on completion notifier after supervisor exit.
# try_acquire_mutation(paths) -> permit | None; release_mutation(permit).
# defer_mutation(paths, callback) -> bool; owns exact pending paths until callback ends.
```

状态数：reserved + queued + dispatched/running/releasing <= 16；同一任务恰好归还一次；只有已经返回且资源关闭的任务才可 released。UI 后处理不得延迟服务资源回收。invalid/path_busy 提交消费并终结该 token，产生明确失败，不能遗留预留；成功提交也不得重复登记。

仅协议/文件格式/共享验收常量进入 `consts/raw_audio_csv_consts.py`（容量 16、协议版本、30 秒 ready 期限、5 秒空闲关闭期限）。worker/队列/锁/registry 均为实例字段；禁止单次使用的无边界意义模块常量。容量满/关闭/路径忙用明确状态返回。宽泛 Exception 只允许 worker 单任务边界、监督线程最外层和 Qt 消费者边界（有诊断并不影响服务状态）；其他代码处理已知异常或直接传播，不做静默回退、重复防御包装。

## 文件职责地图

| 文件 | 职责 |
| --- | --- |
| `consts/raw_audio_csv_consts.py` | 共享容量及协议/期限合同 |
| `base/raw_audio_csv_protocol.py` | 小型不可变请求、结果、失败、快照 |
| `base/raw_audio_csv_tasks.py` | 容量账本、预留、FIFO、状态转换、互斥路径许可 |
| `base/raw_audio_csv_exporter.py` | 原格式输出；增加受控临时路径及可观察清理结果，原调用兼容 |
| `base/raw_audio_csv_worker.py` | Qt 无关 worker 顶层入口，顺序执行，结果归一化 |
| `base/raw_audio_csv_service.py` | 监督线程、spawn/IPC、generation、进程退出及排空 |
| `ui/raw_audio_csv_service_bridge.py` | Qt queued 投递、订阅及退出回调 |
| `ui/sequence/sequence_widget_raw_csv_ops.py` | CSV 预留/提交/结果显示/容量原因，避免在 streaming mixin 堆更多生命周期代码 |
| `ui/sequence/recording_process_context.py` | 本次录音的 CSV 预留和开关/归属快照 |
| `main_window_Launcher.py`, `main_window.py`, `ui/sequence/sequence_widget.py` | 一个服务的注入和实际入口 freeze_support、主窗口关闭接续 |
| sequence 的 analysis/recording_process/streaming/serial_trigger/round_reset mixin | 在已有录音、文件操作和关闭边界接入 |
| `ui/archive_audio_data_dialog.py` | 归档确认后再次取得删除许可 |
| `tools/benchmark_raw_audio_csv_process.py` | 小 fixture 自检、完整内容和三组并发实验，JSON 证据 |
| `tools/raw_audio_csv_frozen_smoke.py` | 实际启动入口可调用的冻结进程小样本自检 |

不得为了文件保护改写通用 RecordingManager 的所有方法；本次修改许可放在实际应用调用入口，直接 base 工具调用不引入隐式全局服务。真实存在的独立管理入口也必须明确有/无所属服务，不能为打开一个对话框创建第二个 worker。

## Task 1: 不可变协议、容量账本及路径许可

**Files:** Create `consts/raw_audio_csv_consts.py`, `base/raw_audio_csv_protocol.py`, `base/raw_audio_csv_tasks.py`; Test `unit_test/base/test_raw_audio_csv_tasks.py`。

- [x] **Step 1 — 编写容量与身份测试。** 用纯内存 ledger；覆盖第 17 个 token 拒绝、查询不预留、release 幂等、转提交数量不变、旧 token/重复 task_id/不同 owner 拒绝、不依赖 GUI 来释放。示例测试必须完整可运行：

```python
def test_sixteen_reservations_and_observation_never_allocates():
    ledger = CsvTaskLedger()
    tokens = [ledger.reserve(str(i)).reservation for i in range(16)]
    assert all(tokens)
    for _ in range(20):
        assert ledger.snapshot().outstanding == 16
    assert ledger.reserve("seventeenth").status == "full"
    assert ledger.release_reservation(tokens[0]) is True
    assert ledger.release_reservation(tokens[0]) is False
    assert ledger.reserve("replacement").status == "accepted"
```

- [x] **Step 2 — 运行 RED。** `python -m pytest unit_test/base/test_raw_audio_csv_tasks.py -q --basetemp (New-CsvTestTemp)`；预期新模块/类缺失，然后实施 ledger/protocol，使该批测试通过。`reserve` 在同一 RLock 内检查 phase 和计数并插入唯一 token；snapshot 返回值对象。commit 同锁校验并把 reserved 移到 queued，失败也终结该 token，不能出现增量加一的中间态。
- [x] **Step 3 — 编写并验证路径/关闭批次 RED。** 规范化 `normcase(abspath(path))`；提交与 mutation permit 对同一源/目标互斥，匹配不同路径写法；deferred mutation 在注册时保留未来修改权；FIFO、晚到 commit 在 draining 允许、closed/unavailable 新预留拒绝。禁止拿着 ledger 锁运行文件/用户回调。
- [x] **Step 4 — 实现并验证全部 ledger 行为。** permit 用唯一 owner id，release 按对象身份幂等；有待处理清理权的路径阻止新 commit/修改插队；终态与 released 分开；已经确认无句柄但 unlink 残留的任务可 released 并保留诊断。测试 terminal 后未 released 仍占容量、capacity=16 时 1 running + 14 queued + 1 reserved。`python -m pytest unit_test/base/test_raw_audio_csv_tasks.py -q --basetemp (New-CsvTestTemp)` 预期全部 PASS。
- [x] **Step 5 — 检查并 checkpoint。** 精确暂存本任务四个文件，`git diff --cached --check`；`git commit -m "feat: add bounded CSV task and path ownership ledger"`。review 核对无全局可变 registry、无忙碌异常控制流、无文件 I/O 持锁。

## Task 2: 保持格式的 exporter 扩展和真实 spawn worker

**Files:** Modify `base/raw_audio_csv_exporter.py`; Create `base/raw_audio_csv_worker.py`; Test modify `unit_test/base/test_raw_audio_csv_exporter.py`, create `unit_test/base/test_raw_audio_csv_worker.py`, `unit_test/base/raw_audio_csv_fakes.py`。

- [x] **Step 1 — 写并运行 exporter 批次 RED。** 增加可选 keyword `temporary_path=None` 与可选 `cleanup_failed=None`（仅内部诊断回调，默认原 API 不变）。测试指定任务临时文件、路径不在目标目录时拒绝、排他创建碰撞不覆盖、replace 失败保留旧 CSV、正常失败删除自身临时文件、unlink 失败回调精确路径/异常。`python -m pytest unit_test/base/test_raw_audio_csv_exporter.py -q --basetemp (New-CsvTestTemp)`；确认失败来自新增合同。
- [x] **Step 2 — 最小扩展并 GREEN。** 默认继续 mkstemp；指定路径验证父目录和文件名后 `os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)`。保持 float32/8192、BOM/LF、.9f/.9g、flush/fsync/close/replace 顺序。碰撞文件不是本任务所有，不能在 finally unlink。不要在回调异常上覆盖原始导出异常；worker 传入的回调只记有限诊断。运行同一文件全部 PASS。
- [x] **Step 3 — 写真实 spawn 测试并 RED。** fakes 模块保存顶层 spawn target（阻塞、硬退出、丢终态、错误版本），不把 lambda/monkeypatch 期望跨进程传播。小样本比较原 exporter 与 worker 输出 bytes，覆盖空 WAV、非连续通道、跨块、中文/空格路径；同 worker 连续两任务 PID 一致、异于父 PID，成功后立即可替换/删除 WAV。
- [x] **Step 4 — 实现 worker 并 GREEN。** `raw_audio_csv_worker(control)` 先发送带协议版本 ready，逐条接收 export/shutdown；export 带 service 生成的 generation/临时路径，调用原 exporter。返回后再生成 result（内部起止/行数/字节数）；预期文件异常归一化为 failure。任务最外层 Exception 有完整任务阶段诊断；不能导入 UI/硬件/数据库或把 traceback 无限发送。关闭控制端于 finally。`python -m pytest unit_test/base/test_raw_audio_csv_exporter.py unit_test/base/test_raw_audio_csv_worker.py -q --basetemp (New-CsvTestTemp)` 预期 PASS，包含真实 spawn。
- [x] **Step 5 — Checkpoint。** 精确暂存本任务文件并 diff check；`git commit -m "feat: export compatible raw CSV in a reusable spawn worker"`。不做大文件 benchmark。

## Task 3: 异步监督服务和故障回收

**Files:** Create `base/raw_audio_csv_service.py`; Modify `base/raw_audio_csv_protocol.py`, `base/raw_audio_csv_tasks.py`（仅服务事件需要）, `base/raw_audio_csv_exporter.py`, `base/raw_audio_csv_worker.py`（任务临时文件创建确认）；Test `unit_test/base/test_raw_audio_csv_service.py`, `unit_test/base/raw_audio_csv_fakes.py`, `unit_test/base/test_raw_audio_csv_exporter.py`, `unit_test/base/test_raw_audio_csv_worker.py`。

- [x] **Step 1 — 写并运行调度 RED。** 测试公共方法在调用者线程仅锁/入队，mock process factory/pipe/file 操作如发生在调用线程则失败；reserve 不启动 worker，首个 commit 惰性启动，只有一个 in-flight export，进程健康复用、无子级积压队列。`python -m pytest unit_test/base/test_raw_audio_csv_service.py -q --basetemp (New-CsvTestTemp)`。
- [x] **Step 2 — 实现基础监督循环并 GREEN。** 一个实例监督线程管理 command inbox、FIFO 和 `multiprocessing.connection.wait`/短 poll；进程 sentinel 与管道同时监听，先读取可用终态后判断异常退出。start/send/recv/join/临时清理仅监督线程；每任务终态与 released 标记由 ledger 幂等推进。父进程每任务生成 `.目标文件名.task_id.tmp` 的唯一兄弟临时路径；worker 不得自行给父进程不可知的任务临时名。subscribe 回调在锁外执行且个别消费者失败不阻止释放。
- [x] **Step 3 — 写故障批次 RED，再实现 GREEN。** 强制 exit、publish 后丢消息、旧 generation 消息、Pipe EOF、ready 版本错/超时、启动抛错、终止未确认、临时删除权限错。无可信终态仅报结果未确认，不凭 CSV 存在成功；确认 worker 死亡后才清理精确临时路径/放行源文件。正常任务 failure 继续同 worker；异常死亡为后续任务重建 generation；启动/ready 失败一次后 unavailable，未执行队列明确失败，现存 reserved 保留到录音完成/取消后归还。

  异常退出清理必须有所有权证据：exporter 在排他创建之后、纳入句柄 finally 保护范围内调用可选内部 temporary-created callback；worker 发出带 task_id/generation/path 与 fstat 文件身份（st_dev/st_ino）的小事件。父进程确认死亡后，仅清理已确认且身份仍匹配的非 reparse 最终路径；不扫描、不按名字推定所有权。进程若在创建后但确认发出前死亡，记录未确认残留并保留文件，死亡确认后可释放容量；这是规格允许的残留诊断，不冒险删除。测试未确认硬退出、确认后的部分文件清理、同名替换和邻居文件保留。
- [x] **Step 4 — 写排空 RED，再实现 GREEN。** begin_shutdown 拒绝新 reserve，但等待旧 reserved 的 commit/release；所有任务归零后再发 shutdown；空闲退出超时 5 秒可 terminate 并确认退出；真实导出无固定 timeout。单元可注入 clock/短期限，真实 spawn 至少验证忙碌任务完成和无孤儿进程。`python -m pytest unit_test/base/test_raw_audio_csv_tasks.py unit_test/base/test_raw_audio_csv_service.py unit_test/base/test_raw_audio_csv_worker.py -q --basetemp (New-CsvTestTemp)` 预期 PASS；重复运行仅在修复新故障时需要。
- [x] **Step 5 — Checkpoint。** `git commit -m "feat: supervise bounded CSV process lifecycle asynchronously"`，精确暂存本任务文件并 diff check。审查 ledger/进程死亡/结果先后竞争，不能因 done Qt 消息未消费而死锁。

## Task 4: Qt 桥接、服务注入与入口安全

**Files:** Create `ui/raw_audio_csv_service_bridge.py`, `ui/sequence/sequence_widget_raw_csv_ops.py`; Modify `main_window_Launcher.py`, `main_window.py`, `ui/sequence/sequence_widget.py`; Test `unit_test/ui/test_raw_audio_csv_bridge.py`, `unit_test/ui/test_raw_audio_csv_entrypoints.py`。

- [x] **Step 1 — Qt/入口 RED。** 测试监督事件最终在 QApplication 线程交付、订阅解除/对象销毁后不触碰 QWidget、重复终态不重复弹窗、应用和 sequence 持有同一个 service（归档共享服务的断言留到 Task 6）；直接 main 及 launcher 的 freeze_support 在 GUI 初始化前执行。空闲构造不产生进程。`python -m pytest unit_test/ui/test_raw_audio_csv_bridge.py unit_test/ui/test_raw_audio_csv_entrypoints.py -q --basetemp (New-CsvTestTemp)`。
- [x] **Step 2 — 实现 bridge。** 私有 `pyqtSignal(object)` 连接 `Qt.QueuedConnection`；事件关联原任务归属；service.closed 与 bridge delivery_closed 分开；关闭 GUI 消费者不妨碍服务释放。raw_csv_ops 暂时只初始化/提供查询和标准结果展示，保留旧导出路径到 Task 5 原子切换，不同时运行两套导出。
- [x] **Step 3 — 注入并整理入口。** launcher 创建一次服务/桥接并传入 MainWindow，再传 SequenceWindow；直接 MainWindow 构造缺参时自身拥有服务，传入时不重复拥有；standalone SequenceWindow 明确 own/injected，测试替身支持显式注入。添加 keyword 参数保持原调用兼容。main_window 的 main guard 前移最小 freeze_support 到重 UI 导入前的安全 bootstrap；不要让 child re-import 引发硬件/GUI初始化。归档注入的实际入口改造在 Task 6 完成。
- [x] **Step 4 — GREEN。** `python -m pytest unit_test/ui/test_raw_audio_csv_bridge.py unit_test/ui/test_raw_audio_csv_entrypoints.py unit_test/ui/test_recording_process_integration.py -q --basetemp (New-CsvTestTemp)`。已有线程测试仍保持通过，本任务不改产品导出行为。
- [x] **Step 5 — Checkpoint。** `git commit -m "feat: wire shared CSV process service through Qt entrypoints"`。检查线程亲和与独立窗口所有权，不引入隐式 singleton。

## Task 5: 录音前容量预留及导出线程切换

**Files:** Modify `ui/sequence/sequence_widget_raw_csv_ops.py`, `ui/sequence/sequence_widget.py`, `ui/sequence/recording_process_context.py`, `ui/sequence/sequence_widget_analysis_ops.py`, `ui/sequence/sequence_widget_recording_process_ops.py`, `ui/sequence/sequence_widget_streaming_ops.py`, `ui/sequence/sequence_widget_serial_trigger_ops.py`, `ui/sequence/sequence_widget_barcode_ops.py`（既有方向及延迟触发的 CSV 准入）, `ui/sequence/sequence_widget_ui_ops.py`（常规按钮刷新保留 CSV 拒绝原因）；Test modify `unit_test/ui/test_raw_audio_csv_recording_integration.py`, create `unit_test/ui/test_raw_audio_csv_admission.py`。

- [x] **Step 1 — 准入 RED。** fake service 停在 16 满额，反复查询不占名额，`start_this_play` 不调用 `_reserve_recorded_count_for_run`/不启动硬件；串口未开始时不推进工况，延迟触发开始前再次检查且满额不补录。满额仍允许 WAV-only 与校准。`python -m pytest unit_test/ui/test_raw_audio_csv_admission.py -q --basetemp (New-CsvTestTemp)`。
- [x] **Step 2 — 实现预留边界。** raw_csv_ops 集中实现本次是否需要 CSV、token 获取/释放；纯检查可使用候选工况快照，不改变旧状态。`start_this_play` 在计数/工况副作用前取得 token，直接调用 judge 的入口也经幂等预留方法，不能二次预留。串口 `_start_serial_product_condition` 在 prepare 前先取得名额，并在任何 prepare/preflight 失败路径释放；延迟阶段失败要撤销本次准备，不动已完成工况。校准使用原硬件准入，不绕到 CSV 需求检查；同一录音 token 持有后后续 can_start 不被自己的第 16 个名额拒绝。

  实际入口还包括 `on_clicked_player_btn → _prepare_next_manual_product_condition_recording → start_this_play`；名额需在准备阶段的轮次登记、元数据锁定等副作用之前取得并由后续步骤复用，不能只在 start_this_play 加检查。串口上层在调用 `_start_serial_product_condition` 前设置 `_manual_product_condition_index`，同样要确保最终原子预留失败时不改变该索引；覆盖观察有容量、实际 reserve 已满的竞态。所有准备失败路径归还这次 token，不能清除重入后新录音的预留。

  既有方向/延迟触发位于 barcode mixin。`_start_directional_workflow` 在修改方向轮次、S/N 锁定及检测中状态前预留；`on_directional_triggered` 对 CSV 满额/不可用的拒绝不得写入 `_queued_directional_trigger`，其他既有忙碌原因的暂存行为保留。延迟到期先重新准入，满额不留下自动补录任务，日志区分 CSV 原因；不重构通用方向调度器。
- [x] **Step 3 — 归属/失败批次 RED。** Context 增加 csv_reservation 和 csv_enabled_snapshot；清理旧 context 不得释放新 token。启动失败、元数据失败、取消、WAV 验证失败、late callback、成功 publication 后其他 UI 异常均测试恰好一次转交或归还。关闭期间预留转提交成功；录音后开关变化不改变已经预留的决定。
- [x] **Step 4 — 替换导出并 GREEN。** `_schedule_raw_audio_csv_export` 改为 raw_csv_ops 的小型请求构建/登记/commit，无 Thread。先验证路径/通道、保存原轮次记录，再在任务可派发前完成 round file 登记；失败报告沿用 WAV 安全文案。移除仅用于 CSV 的线程/锁/信号使用，其他 threading 用途按搜索保留；同步等待方法留给 Task 7 完整替换前不得伪造 drain。增加 `test_parent_never_formats_csv`：替换父进程 exporter 为 fail，真实 worker 小样本仍成功。运行 `python -m pytest unit_test/ui/test_raw_audio_csv_admission.py unit_test/ui/test_raw_audio_csv_recording_integration.py unit_test/ui/test_recording_process_integration.py unit_test/test_serial_product_condition_runtime.py -q --basetemp (New-CsvTestTemp)`，预期 PASS。
- [x] **Step 5 — Checkpoint。** `git commit -m "feat: reserve CSV capacity before admitting recordings"`。审查实际所有 start/延迟/失败路径，不能只测一个按钮；本任务 checkpoint 尚有 Task 6/7 的集成工作，不声称功能可交付。

## Task 6: 文件操作互斥、轮次清理与归档接入

**Files:** Modify `base/raw_audio_csv_tasks.py`, `base/raw_audio_csv_service.py`, `main_window.py`, `ui/archive_audio_data_dialog.py`, `ui/sequence/sequence_widget_raw_csv_ops.py`, `ui/sequence/recording_process_context.py`（录音路径许可归属）, `ui/sequence/sequence_widget_streaming_ops.py`, `ui/sequence/sequence_widget_round_reset_ops.py`, `ui/sequence/sequence_widget_serial_trigger_ops.py`, `ui/sequence/sequence_widget_recording_process_ops.py`; Test create `unit_test/ui/test_raw_audio_csv_file_ownership.py`, modify `unit_test/base/test_raw_audio_csv_tasks.py`（原子许可交接）, `unit_test/ui/test_archive_audio_delete_dialog.py`, `unit_test/ui/test_round_reset.py`, `unit_test/test_serial_product_round_cleanup.py`。

- [x] **Step 1 — 文件竞态 RED。** 门闩控制 worker/提交，验证 queued/running/releasing WAV 和目标 CSV 不可修改；新增归档窗口与主窗口、sequence 共享同一 service 的断言；archive 确认弹窗期间新任务进入，确认后能拒绝删除；permit 持有期间 commit 不能抢占。产品数据库标注允许，实际移动旧式标注拒绝；同路径重录保护但按钮仍隐藏。`python -m pytest unit_test/ui/test_raw_audio_csv_file_ownership.py -q --basetemp (New-CsvTestTemp)`。
- [x] **Step 2 — 衔接应用入口。** MainWindow `on_audio_manager_init` 从 static 改为实例入口传 shared service（信号不变）；ArchiveAudioDataDialog 兼容独立无任务入口，并在确认后为每个删除计划的 WAV/CSV 取得 permit，再调用原删除函数，finally release。轮次删除的 permit 覆盖记录全部 owned 路径；round busy 改用服务 snapshot，包括 reserved。不要只检查 `is_busy` 后再独立执行文件 I/O。
- [x] **Step 3 — 串口清理 RED，再 GREEN。** 使用精确旧轮次/路径闭包立即登记 CSV deferred mutation，保留未来清理优先权、禁止同路径新录音或提交插队；CSV supervisor 在自身账本锁外检查所属 RecordingService 的路径租约是否已释放，且 CSV 路径也可用时才原子领取修改许可并删除。现有 RecordingService.defer_path_cleanup 对同一路径只保存一个回调，不能叠加多个生命周期回调；有 CSV 协调服务时采用上述就绪条件，无 CSV 服务的独立旧调用继续原释放回调路径。不要在两个服务锁同时持有时调用对方；保证两个资源先后释放两种顺序都只删一次，释放失败保留文件。只保留现有删除范围，不顺手扩成清空目录。历史 UI 可清除但服务持有独立归属，迟到成功仅记录原任务。
- [x] **Step 4 — 覆盖窗口间与启动竞态并 GREEN。** 在新录音打开可能复用路径前取得互斥的短期路径许可并交接给该录音上下文，避免 check/start 间隙；释放录音后转 CSV 时交接不能出现可删空档。此保护由 CSV 服务账本持有 owner 权限，同一 owner 可以原子升级到 export，失败/取消释放。`python -m pytest unit_test/ui/test_raw_audio_csv_file_ownership.py unit_test/ui/test_archive_audio_delete_dialog.py unit_test/ui/test_round_reset.py unit_test/test_serial_product_round_cleanup.py unit_test/base/test_recording_storage_consistency.py -q --basetemp (New-CsvTestTemp)` 预期 PASS。

  保留现有声卡的可选清理失败语义：`_publish_recording_context` 在 release_warned 时仍可能发布已接纳录音，但 RecordingService 路径租约尚未释放。CSV 结束或 UI context 被清理不能抹掉这层占用；归档、移动、重录和串口清理仍须等待其确认释放。新增回归覆盖 CSV 先结束、录音租约仍在的情况；WAV-only 录音同样不能被新归档入口绕过文件保护。协调时不在持有一个服务锁的情况下调用另一个服务。
- [x] **Step 5 — Checkpoint。** `git commit -m "fix: coordinate CSV ownership with recording file operations"`。审查许可顺序和 deferred cleanup 优先权、标注兼容、无隐式全局注册表。

## Task 7: 非阻塞 drain 与完整窗口生命周期

**Files:** Modify `base/raw_audio_csv_service.py`, `ui/raw_audio_csv_service_bridge.py`, `ui/sequence/sequence_widget_raw_csv_ops.py`, `ui/sequence/sequence_widget_streaming_ops.py`, `ui/sequence/sequence_widget_recording_process_ops.py`, `ui/sequence/sequence_widget_analysis_ops.py`, `main_window.py`, `main_window_Launcher.py`; Test create `unit_test/ui/test_raw_audio_csv_shutdown.py`, modify `unit_test/ui/test_recording_process_integration.py`。

- [x] **Step 1 — 关闭 RED。** 门闩挡住 exporter 后 close，Qt timer 持续跳动，event.ignore 且服务继续任务；close 重入无重复 shutdown。覆盖录音预留尚未转提交、录音取消、late commit、failure、零任务、初始化 hidden close、可见子窗口关闭与主窗口关闭嵌套。`python -m pytest unit_test/ui/test_raw_audio_csv_shutdown.py -q --basetemp (New-CsvTestTemp)`。
- [x] **Step 2 — 实现关闭接续。** 主窗口保留分析未完成阻止策略；真正退出关闭新 CSV/录音准入，录音原策略收尾导致 token 提交/释放，CSV begin_shutdown 排空，closed Qt 通知再 close。状态标签显示保存数量；事件循环继续运行，禁用的是新业务启动不是整个事件分发。监督线程退出后才销毁 bridge；aboutToQuit 仅幂等兜底。
- [x] **Step 3 — 子窗口边界。** visible sequence 只等待服务中的已有保存完成，不关闭借用的应用级 service；独立 own service 在实际退出时 shutdown。隐藏初始化 close 跳过 drain，主窗口递归关闭子窗口使用幂等标识。删除所有 `_wait_for_raw_audio_csv_exports` 同步 join 调用，替换测试里的旧 Mock 断言为异步契约。

  主窗口开始实际退出后，在视频异步关闭之前就禁止新业务启动；是否创建 CSV service 与是否承担应用退出责任是不同概念，launcher 注入的服务也须由主窗口退出流程排空。子窗口任务归属须跟踪到 released，不能因收到 terminal 就视为路径已经释放。录音启动期间已有一次 Qt 事件处理，可能在上下文创建前收到 close；须释放这时的 pending reservation，并阻止该启动调用返回后继续打开硬件，WAV-only 同样覆盖。保留合法晚到提交的 drain 合同，不通过提前销毁 bridge 丢弃生命周期事件。
- [x] **Step 4 — GREEN 与无资源遗留。** `python -m pytest unit_test/ui/test_raw_audio_csv_shutdown.py unit_test/ui/test_recording_process_integration.py unit_test/base/test_raw_audio_csv_service.py -q --basetemp (New-CsvTestTemp)`；断言退出后 child PID 不存活，supervisor join 在后台完成、IPC 已关闭；无静默强杀正在导出的任务，无用 `QApplication.processEvents` 包装 busy wait。
- [x] **Step 5 — Checkpoint。** `git commit -m "fix: drain CSV exports asynchronously during shutdown"`。本任务后搜索线程集合/等待旧方法的引用应仅剩历史文档；程序范围无 CSV 线程实现。

## Task 8: 可归因计时及可复现验证工具

**Files:** Modify `base/raw_audio_csv_service.py`, `base/raw_audio_csv_worker.py`, `base/raw_audio_csv_protocol.py`（不可变诊断字段）, `base/recording_service.py`, `ui/recording_service_bridge.py`（实际产品录音启动、采集事件及 Qt 接收的关联计时）, `ui/raw_audio_csv_service_bridge.py`（Qt 投递延迟）, `ui/sequence/sequence_widget_raw_csv_ops.py`（有界 GUI 分段记录）, `ui/sequence/sequence_widget_streaming_ops.py`, `ui/sequence/sequence_widget_recording_process_ops.py`; Create `tools/benchmark_raw_audio_csv_process.py`, `tools/raw_audio_csv_benchmark_load.py`（真实并发负载与资源采样）, `unit_test/tools/test_raw_audio_csv_benchmark.py`; Modify `unit_test/ui/test_raw_audio_csv_recording_integration.py` 及直接受影响的录音 service/bridge 诊断测试。

- [x] **Step 1 — 诊断 RED。** 测试每 task 一组相关 ID/PID/generation，reserve/submit/dispatch/ready/terminal_received/released 时间；worker 自身 export 耗时；Qt 投递单列。注入 GUI 投递延迟不改变 exporter 内部耗时；禁用日志不破坏资源释放。`python -m pytest unit_test/tools/test_raw_audio_csv_benchmark.py unit_test/ui/test_raw_audio_csv_recording_integration.py -q --basetemp (New-CsvTestTemp)`。
- [x] **Step 2 — 接入分段计时。** 在现有波形/数据库/历史更新/自动分析入队边界放 perf_counter 范围，保留原顺序和行为；有效开始请求及 worker 实际采集事件采用既有录音标识关联，无法观测实际采集事件时标为 unavailable，不冒用文件名或 GUI started。每个进程只计算自身 duration，跨进程只用同一父进程观测边界。 这些启动诊断须覆盖正常产品录音，不能仅由基准工具子类记录；使用小型会话快照按 request_id/PID/generation 保存有效启动、父进程接收并验证 worker started、Qt 投递三个边界。采集观测明确包含 IPC 延迟，不把子进程时钟直接相减；缺失事件保留 unavailable，并测试延迟 Qt 投递不改变采集观测耗时。
- [x] **Step 3 — 实现工具及小样本自检。** argparse 支持 `--mode self-test|content|concurrency`、`--source-wav`、`--reference-json`、`--work-dir`、`--report`；concurrency 支持 `--csv-mode off|thread|process`、`--repetitions`、`--conditions`、`--profile representative|full`。self-test 生成小 float WAV 真 spawn，bytes 对比；content 只输出一个新 CSV 并流式 hash/计行。所有报告和输出排他创建，拒绝覆盖输入/既有基线，先核对目录空间，不把源 WAV 读成大数组传给 worker。
- [x] **Step 4 — 并发 profile 可执行。** concurrency 建立相同 200 工况实际历史/波形 widget 路径、使用相同 WAV/分析配置，经真实 AnalysisProcessService 并发执行；录音端用已有测试 backend 驱动真实 RecordingService 观察 capture start，硬件现场数据另作真实设备确认。thread 仅工具内恢复原 exporter Thread，process 调用新服务，off 不导出；负载/配置/PID/是否模拟采集写 report。50ms QTimer 收集 heartbeat；representative 用同一较短 WAV 每模式三次交错，full 用现有完整 WAV 每模式至少一次，消耗产生的输出必须属于该 run 目录，不删除其他文件。新增测试验证模式隔离、指标计算、缺测不判通过、hash 错误失败、legacy 线程路径不进入产品代码。

  工具的测量前准备可调用现有历史记录构造、分组合并方法批量建立 200 工况状态，然后完整渲染一次；避免准备阶段为每条旧记录重复重建全部单元格，造成无关的超时和控件堆积。须断言并报告真实记录数及渲染状态一致。正式计时仍执行未经优化的产品 `_update_current_recent_session_result → upsert_session → _populate_group_row` 和真实波形发布路径，不能替换为简化循环或 Mock。准备耗时与采样是否覆盖准备阶段要在报告中区分；不改动产品历史控件实现。

  每组 report 还必须包含以下硬门槛，不能只有计时：`resources.samples` 按 1 秒等间隔记录 parent/csv/recording/analysis PID 的累计 CPU 秒、工作集字节及峰值；Windows 可用 ctypes 调用 GetProcessTimes/GetProcessMemoryInfo，避免新增依赖；CPU 采样时记录 monotonic 时间，进程已退出的数据标 missing，不填零。 已退出进程的瞬时采样不能充当有效运行期数据；保留进程句柄取得的可信最终累计 CPU 秒可在独立汇总字段报告，不能伪装成退出后的实时采样。每个参与 PID 必须有可用的运行期内存/CPU 观测及可信累计 CPU 汇总；只有退出后采样或运行期采样失败时仍为 unverified。正常退出后的预期 missing 不否定此前完整的有效观测，避免把所有正常退出实验都判缺测。`recording.expected_frames/actual_frames/sample_rate/channels/trim_frames/drop_count/validation_status` 比对实际关闭后 WAV 及录音诊断，实际帧数必须等于按当前 startup trim 规则计算的目标帧数，模拟和真实硬件标识分开；`analysis.task_id/execution_status/expected_instance_count/completed_instance_count/failure_count` 验证与 baseline 相同配置的分析进入 completed、实例数一致、无新增算法执行失败，OK/NG 判定值按相同输入与基线一致核对，不要求所有样本判 OK。新增测试故意制造长度不足、分析 failure、资源采样缺失，报告必须失败或标 unverified，不判通过。
- [x] **Step 5 — GREEN 与 checkpoint。** `python -m pytest unit_test/tools/test_raw_audio_csv_benchmark.py unit_test/ui/test_raw_audio_csv_recording_integration.py -q --basetemp (New-CsvTestTemp)`；`python tools/benchmark_raw_audio_csv_process.py --mode self-test --work-dir tmp/csv-self-test --report tmp/csv-self-test/result.json` 预期独立 PID、bytes 相等、退出干净。`git commit -m "test: add correlated CSV process and GUI benchmark evidence"`。不得把 self-test 当作 600 秒验收。

## Task 9: Windows 冻结入口烟测及目标回归

**Files:** Create `tools/raw_audio_csv_frozen_smoke.py`, `unit_test/tools/test_raw_audio_csv_frozen_smoke.py`; Modify `main_window_Launcher.py`, `main_window.py`, `unit_test/ui/test_raw_audio_csv_entrypoints.py`（测试入口最小接入）。

- [x] **Step 1 — 入口测试 RED。** 增加显式诊断 CLI `--verify-raw-csv-process <新报告路径>`，仅用户/测试显式传入时运行，不进入产品 GUI；两个实际入口在 freeze_support 之后、GUI 导入前识别参数并调用 helper。helper 用短中文路径 WAV、服务真实 worker、hash/原子发布/关闭完成验证，失败 exit 非零，报告所有 child PID；无参启动流程不变。`python -m pytest unit_test/tools/test_raw_audio_csv_frozen_smoke.py unit_test/ui/test_raw_audio_csv_entrypoints.py -q --basetemp (New-CsvTestTemp)`。
- [x] **Step 2 — 实现并 GREEN。** helper 是测试工具，不能实例化第二个与应用并存的 worker；诊断模式本身不启动 GUI。Windows 测试对子进程命令行/进程树确认没有递归主窗口，无 stdout 控制台依赖；两个入口统一解析函数避免分叉。运行上一步测试成功。
- [ ] **Step 3 — 实际 launcher 冻结验证。** 若未安装 PyInstaller，按现有项目发布环境运行同命令，不静默安装/更新生产依赖，也不能跳过并声称通过。用新隔离输出目录，不编辑现有未跟踪 spec：

```powershell
python -m PyInstaller --noconfirm --clean --onedir --windowed --name raw-csv-process-smoke --distpath tmp/csv-frozen/dist --workpath tmp/csv-frozen/build --specpath tmp/csv-frozen main_window_Launcher.py
$csvExe = (Resolve-Path 'tmp/csv-frozen/dist/raw-csv-process-smoke/raw-csv-process-smoke.exe').Path
$csvReport = [IO.Path]::GetFullPath('tmp/csv-frozen/spawn-report.json')
$csvSmoke = Start-Process -FilePath $csvExe -ArgumentList @('--verify-raw-csv-process', ('"' + $csvReport + '"')) -WindowStyle Hidden -PassThru
```

实施代理用非阻塞轮询观察进程和报告，单次工具等待不超过 60 秒。预期退出码 0、报告 hash 一致、独立 worker PID、无遗留。依照 same command 将入口替换为 main_window.py、使用不同 name/output/report 再做直接入口冻结验证。烟测不替代正常主 GUI 启动：在已有发布所需资源/配置可用的同一隔离包中手工完成正常开窗、小录音导出和正常退出，记录结果。若缺少发布资源、原生界面操作能力或真实设备，把此人工验收项明确转交 Task 10 Step 4 的现场验收清单，保留为未验证；本任务仍完成两个实际入口的自动冻结烟测和目标回归，不因外部人工证据缺失停止后续可独立执行的软件对照。不扩大成发布工程重构，也不能把诊断 CLI 成功视为主界面手工验收通过。

- [x] **Step 4 — 目标回归。** 运行以下集合一次（变更后失败则按 systematic-debugging 修复并重跑受影响项），全部 PASS 才进入最终性能验收：

  现有 `test_recording_gui_finalization.py::test_finalizing_is_once_only_and_cancel_blocks_fallback` 用 `object.__new__(RecordingService)` 绕过构造器，未提供当前生产诊断所需的 `_recording_diagnostics` 等字段；已在本次改动前的实现确认同样失败。本任务可最小修正该测试宿主的初始化以验证原有断言，不为迁就测试改变生产录音行为。`test_recording_storage_consistency.py` 两个外部路径测试要求临时目录位于项目之外，运行它们时使用经过检查的全新外部 basetemp；项目内路径被规范化成相对路径属于预期行为，不修改其生产规则。

```powershell
python -m pytest unit_test/base/test_raw_audio_csv_tasks.py unit_test/base/test_raw_audio_csv_exporter.py unit_test/base/test_raw_audio_csv_worker.py unit_test/base/test_raw_audio_csv_service.py unit_test/ui/test_raw_audio_csv_bridge.py unit_test/ui/test_raw_audio_csv_entrypoints.py unit_test/ui/test_raw_audio_csv_admission.py unit_test/ui/test_raw_audio_csv_recording_integration.py unit_test/ui/test_raw_audio_csv_file_ownership.py unit_test/ui/test_raw_audio_csv_shutdown.py unit_test/ui/test_round_reset.py unit_test/ui/test_archive_audio_delete_dialog.py unit_test/ui/test_recording_process_integration.py unit_test/ui/test_recording_gui_finalization.py unit_test/test_serial_product_round_cleanup.py unit_test/test_serial_product_condition_runtime.py unit_test/base/test_recording_storage_consistency.py unit_test/tools/test_raw_audio_csv_benchmark.py unit_test/tools/test_raw_audio_csv_resources.py unit_test/base/test_recording_start_timing.py unit_test/tools/test_raw_audio_csv_frozen_smoke.py -q --basetemp (New-CsvTestTemp)
```

- [x] **Step 5 — Checkpoint。** `git commit -m "test: verify CSV spawn lifecycle in frozen application entrypoints"`。检查产品无参数行为、release notes 不暴露内部测试操作；不会把 benchmark 临时输出、打包目录加入提交。


执行状态：入口源码实现、目标回归及源码双审已通过。实际冻结构建在现有依赖收集器中反复出现原生访问冲突，未产出可执行包；Step 3 保持未完成并保留证据。后续独立的软件性能对照继续执行，不将源码运行成功替代冻结验收。

## Task 10: 完整 600 秒内容与并发性能验收及交付记录

**Files:** Create `docs/double-sdd/verification/2026-09-22-raw-audio-csv-process.md`。本任务只运行已构建工具与记录证据；发现缺陷时回到所属实现任务修复并复审，不在验证任务混入无关优化。

- [x] **Step 1 — 准备已有 fixture。** 上游实验 JSON 位于原工作区 `tmp/full-csv-benchmark-20260922-105431/result.json`，它含 source 路径、hash、字节数；实施 worktree 不会自动拥有它。通过运行元数据明确原工作区位置，验证 JSON/source 都存在，再用显式绝对路径作为以下参数，不能把示例占位符直接执行。原 WAV、原 CSV 和既有实验输出只读。先检查至少可容纳本轮新增输出的磁盘空间，工具默认拒绝覆盖。
- [x] **Step 2 — 完整内容。** `python tools/benchmark_raw_audio_csv_process.py --mode content --source-wav <existing-600s-wav> --reference-json <existing-result-json> --work-dir tmp/csv-full-content --report tmp/csv-full-content/result.json`。预期 26,460,000 数据行、1,162,232,136 字节、SHA256 `facb5196b26f7ac36fdb470fe8850a89106fd94b7602164b40c24eeba38e5b65`，最后行时间 599.999977324，原 WAV 未修改。长命令保持后台会话，分次读取进度，定期向用户说明结果。不得重复生成旧 baseline 巨型 CSV。
- [x] **Step 3 — 三组对照。** 对 `off/thread/process` 每组执行工具 `--mode concurrency --profile representative --conditions 200 --repetitions 3`，使用相同较短 fixture；实施控制器按 off-thread-process、thread-process-off、process-off-thread 交错各次运行（若工具 repetitions 只连续执行则每次传 1）。再用完整 WAV 每组 `--profile full --repetitions 1`。每次 work-dir/report 唯一，记录分析配置/录音后端/机器负载；content 输出能作为 process full 输出的那次只在负载相同且报告完整时复用，不能借旧离线 80.55 秒替代 thread 并发组。允许清理由本轮明确创建的大 CSV，删除前核实路径均在本轮输出目录且结果 hash/报告已写入，不触碰原实验文件。
- [ ] **Step 4 — 实际现场与阈值判定。** 模拟采集报告明确 labeled simulated，只证实软件链路；在真实目标设备上同负载复核实际采集启动、录音无新增丢帧以及关闭后 WAV 的帧数/采样率/通道数均等于该次请求和 trim 规则确定的目标值。逐项核对自动分析 completed、预期分析实例数全部完成、无新增执行 failure，相同输入的 OK/NG 与基线一致。表中报告每个参与 PID 的 CPU 总秒/平均负载、工作集峰值，解释组间负载差异；资源缺测标 unverified。GUI 收尾三次中位数相对线程组降低至少 30%；启动延迟 <= 线程组中位数 × 1.10 + 100ms；Qt 心跳 p95 不比线程组恶化超过 20ms；完整并发组 GUI 改善同方向。记录每组 CSV 内部耗时、排队、GUI 分段、分析入队和实际采集起点，CSV 总时长不设加速承诺。没有复现竞争、硬件或冻结验收缺失时写“未验证”，不标任务全部完成、不扩大范围；向用户交代具体需要的外部验证条件。
- [ ] **Step 5 — 最终证据与收尾。** verification 文档附环境、命令、退出码、结果表、所有阈值逐项 pass/fail/unverified、数据保真和容量/关闭结果；本地路径/实验报告链接可列在验证文档，临时 checkpoint 和运行文件清单仍只放元数据。先精确 force-add 新 verification 文档（docs 被忽略），严禁 add 整个 docs/tmp；`git diff --cached --check` 后以 `git commit -m "docs: record full CSV process validation evidence"` 建立可清理临时 checkpoint。再 review 最终文件范围，经 finishing-a-development-branch 在同一 metadataPath 下完成整合和内容保留清理；清理后不再创建临时提交。没有用户批准最终 commit 时不要留下临时 checkpoint 作为永久提交；不要删除用户未跟踪/忽略文件。


执行状态：完整数据及十二次软件并发对照已完成，格式、模拟录音、分析和四项性能指标通过；现场硬件、实际冻结包和普通冻结 GUI 验收仍未验证，Step 4 保持未完成。验证文档已建立临时检查点；Step 5 的最终整合清理尚未执行，不将当前结果标为全部验收完成。

## 审查核对表与执行门槛

- [ ] 原始 CSV 数据、路径、开关、原子替换语义无变化；全部任务状态/输出归属独立于当前页面。
- [ ] 16 包括录音预留和未释放任务；读查询不预留；满额不启动/推进，正常旧 token drain 不丢失。
- [ ] GUI 不做 process/file/pipe 等待；worker 只处理小请求与分块 WAV，无隐藏线程格式化回退。
- [ ] 文件删除/移动与提交有真正互斥；两种资源释放顺序安全；DB-only 标注和非 CSV 录音不受多余限制。
- [ ] Windows spawn、冻结入口、正常退出、崩溃边界均有实际证据；500ms 小测试不能替代完整 600 秒验证。
- [ ] 无无理由的 broad catch、静默 fallback 或单次使用的模块常量；清理精确归属文件，未确认释放不谎报成功。
- [ ] 所有任务经过 spec/quality gate，既有改动和工作流临时历史按不可变基点妥善保留/清理。

本计划通过 plan-document-reviewer 后交用户批准执行；收到执行批准时，把上游原样 metadataPath 与本计划/规格路径一起传入 subagent-driven-development。执行待批准期间运行状态保持 active；用户明确终止工作流时由后续所有者按技能处理 abandoned 和保留内容的收尾。

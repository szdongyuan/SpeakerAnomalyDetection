# Raw Audio CSV ZIP Implementation Plan

> **For agentic workers:** REQUIRED: Use the `subagent-driven-development` skill to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在原始音频 CSV 完整导出后生成 `example.csv.zip`，完整校验和发布后删除 `example.csv`，失败保留 CSV，并维持 16 容量和异步关闭。

**Architecture:** 沿用单个常驻 spawn worker、父进程 FIFO 和共享路径许可。新增有界内存 ZIP 组件，worker 顺序执行 CSV/ZIP/校验/清理；协议显式区分两个产物，Qt 只处理小型结果和提示。原始 CSV exporter 保持纯 CSV 接口，压缩不进入 GUI。

**Tech Stack:** Python 3.12、zipfile/zlib/hashlib、multiprocessing spawn、soundfile、PyQt5、pytest、Windows。

---

## 输入、边界及执行准备

Spec: `docs/double-sdd/specs/2026-09-23-raw-audio-csv-zip-design.md`。按 Task 1 → 6 顺序，每项一个新 implementer，之后按 `code-review` 技能进行并行规格/质量审查，规格审查为门禁。任务之间不并行写代码。

当前仓库已包含前一项 CSV 进程改造，其中若干必要模块仍为未跟踪文件。执行必须以本地现有实现为前置状态，不能把 Git 的旧版本当作当前产品，也不能把这些已有文件算作本次 ZIP 新增内容。运行时元数据的精确路径由 writing-plans 交接提供，不在本文件写入临时 SHA 或本机资产清单。

### Orchestrator preflight（执行批准后）

- [x] 验证交接 metadata 的路径、状态、mainBase、preexistingTrackedCheckpoint 及当前原始工作区；确认无 merge/rebase/cherry-pick/bisect，无待处理 staged 意图。记录当前源 HEAD 和内容快照；已确认的 spec/本计划如未 checkpoint，仅提交这两份文档的临时检查点。不可使用 `git add .`。
- [x] 调用安装的 `setup_worktree.py --branch codex/raw-audio-csv-zip --metadata-path <交接的精确路径>`。helper 从 mainBase 建立隔离分支；在新 worktree 中执行 `git merge --ff-only <已核实的原始工作区 HEAD>` 接上前置 tracked 保护检查点和文档，必须证明 fast-forward 且包含原 preexistingTrackedCheckpoint；不要修改 mainBase。
- [x] 将下列前置未跟踪依赖按字节复制到隔离区，复制前核对原元数据记录、源路径无 reparse、目标不存在；文件已被其他任务纳入 Git 且内容相同则无需再复制：
  - `base/raw_audio_csv_protocol.py`, `base/raw_audio_csv_service.py`, `base/raw_audio_csv_tasks.py`, `base/raw_audio_csv_worker.py`, `consts/raw_audio_csv_consts.py`。
  - `ui/raw_audio_csv_service_bridge.py`, `ui/sequence/sequence_widget_raw_csv_ops.py`。
  - `tools/benchmark_raw_audio_csv_process.py`, `tools/raw_audio_csv_benchmark_load.py`, `tools/raw_audio_csv_frozen_smoke.py`。
  - `unit_test/base/raw_audio_csv_fakes.py`, `unit_test/base/test_raw_audio_csv_service.py`, `unit_test/base/test_raw_audio_csv_tasks.py`, `unit_test/base/test_raw_audio_csv_worker.py`, `unit_test/base/test_recording_start_timing.py`。
  - `unit_test/tools/test_raw_audio_csv_benchmark.py`, `unit_test/tools/test_raw_audio_csv_frozen_smoke.py`, `unit_test/tools/test_raw_audio_csv_resources.py`。
  - `unit_test/ui/test_raw_audio_csv_admission.py`, `unit_test/ui/test_raw_audio_csv_bridge.py`, `unit_test/ui/test_raw_audio_csv_entrypoints.py`, `unit_test/ui/test_raw_audio_csv_file_ownership.py`, `unit_test/ui/test_raw_audio_csv_shutdown.py`。
- [x] 把复制的明确依赖在 feature worktree 单独 checkpoint，并在非提交 metadata 中记录 `preexistingDependencyCheckpoint`、逐路径原始 SHA-256 和文件清单；不改变原有 preexistingTrackedCheckpoint。本次实现审查范围从这个依赖检查点起。最终回填按路径将“原有未跟踪内容”与 ZIP 修改区分：原目录相同前置文件应用其修改差异，不能作为新文件覆盖；文档本来就在原目录也必须保留。分离不清楚时停止，不能强制覆盖。
- [x] 不复制 `.recording-timing-*`、interval launcher/spec/test、其他任务文件、缓存或历史实验大文件。必要运行时资产只按前置回归实际需要逐一登记并只读引用/复制，不入 Git。
- [x] 在同一依赖检查点建立只读对照 worktree：`git worktree add --detach <新的 repo-local .worktrees 对照路径> <preexistingDependencyCheckpoint>`；路径、SHA 写入 runtime metadata。该对照保留完整旧 CSV 子进程代码与旧工具，作为 Task 5 公平对照，不在产品增加“跳过 ZIP”开关。
- [x] 运行现有 CSV exporter、worker、service、ledger、Qt/准入/文件/关闭、工具与入口目标测试建立前置基线。失败时先区分已有问题与缺依赖；使用 `systematic-debugging`，不得为了 ZIP 改动无关业务策略。

### 每次测试的新临时目录

所有命令从实际执行的 worktree 根运行。Windows 外部路径语义测试需要仓库外 basetemp；不得复用一个已有目录交给 pytest 删除。

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
function New-CsvZipTestTemp {
    $csvZipParent = Split-Path (git rev-parse --path-format=absolute --git-common-dir) -Parent
    $csvZipTempAnchor = Split-Path $csvZipParent -Parent
    if ((Get-Item -LiteralPath $csvZipTempAnchor).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Unsafe test parent' }
    $csvZipTemp = Join-Path $csvZipTempAnchor ('csv-zip-test-' + [Guid]::NewGuid().ToString('N'))
    if (Test-Path -LiteralPath $csvZipTemp) { throw 'Test directory exists' }
    return $csvZipTemp
}
```

### 文件职责图

| 文件 | 本次职责 |
| --- | --- |
| `base/raw_audio_csv_zip.py`（新增） | 追加 `.zip` 的统一路径函数、有界压缩、完整校验、原子发布、安全清理及阶段结果 |
| `base/raw_audio_csv_exporter.py` | 保留现有纯 CSV API/格式，不把 ZIP 默认塞进此函数 |
| `base/raw_audio_csv_protocol.py`, `consts/raw_audio_csv_consts.py` | 双临时路径、ZIP 结果/警告/分段计时、IPC 版本 |
| `base/raw_audio_csv_worker.py`, `base/raw_audio_csv_service.py`, `base/raw_audio_csv_tasks.py` | 编排、身份确认、三路径许可、警告成功及死亡清理 |
| `ui/sequence/sequence_widget_raw_csv_ops.py`, `ui/sequence/sequence_widget_streaming_ops.py` | 双产物归属、正确提示/成功路径、等待所有阶段 |
| `ui/sequence/sequence_widget_round_reset_ops.py`, `base/test_round_data.py` | ZIP 精确登记、实际存在计数、预期缺失的原始产物不误报 |
| `base/audio_record_package.py`, `ui/archive_audio_data_dialog.py` | 精确配套 `.csv.zip` 收集、归档删除许可与二次检查 |
| `ui/product_test_project_config_dialog.py` | 现有选项显示 ZIP 保存方式，配置键不变 |
| `tools/benchmark_raw_audio_csv_process.py`, `tools/raw_audio_csv_benchmark_load.py`, `tools/raw_audio_csv_frozen_smoke.py` | ZIP 内容证据、分段计时、入口诊断 |

## Task 1: 独立 ZIP 组件与失败保护

**Files:** Create `base/raw_audio_csv_zip.py`, `unit_test/base/test_raw_audio_csv_zip.py`。Read `base/raw_audio_csv_exporter.py`；不要修改其默认行为。

- [x] **Step 1 — 正常路径 RED。** 固定接口：`raw_csv_zip_path(csv_path) -> Path` 返回 `Path(str(csv_path) + '.zip')`；`archive_raw_audio_csv(csv_path, *, temporary_path, temporary_created=None, stage_changed=None) -> CsvZipResult`。frozen dataclass 返回 `archive_path: str`, `csv_bytes: int`, `archive_bytes: int`, `csv_retained: bool`, `cleanup_diagnostics: tuple[str, ...]` 以及压缩/校验/发布/清理各阶段 seconds。编写下面的实际文件测试，并扩展中文空格、多点文件名、空文件/仅表头及跨块数据：

```python
def test_zip_appends_extension_and_preserves_member(tmp_path):
    source = tmp_path / '中文 example.v2.csv'
    original = b'\xef\xbb\xbftime_s,CH1\n0.000000000,0.5\n'
    source.write_bytes(original)
    result = archive_raw_audio_csv(source, temporary_path=tmp_path / '.owned.zip.tmp')
    target = tmp_path / '中文 example.v2.csv.zip'
    assert result.archive_path == str(target)
    assert not source.exists()
    assert not result.csv_retained
    with zipfile.ZipFile(target) as archive:
        assert archive.namelist() == [source.name]
        assert archive.read(source.name) == original
        assert archive.getinfo(source.name).compress_type == zipfile.ZIP_DEFLATED
```

- [x] **Step 2 — 运行 RED。** `python -m pytest unit_test/base/test_raw_audio_csv_zip.py -q --basetemp (New-CsvZipTestTemp)`，确认新模块或契约缺失导致失败。
- [x] **Step 3 — 最小实现并 GREEN。** 验证临时路径同目录且不同于 CSV/ZIP；排他创建后立即确认 `(st_dev, st_ino)`。使用 `os.fdopen(fd, 'w+b')` 承接 fd，`ZipFile(..., ZIP_DEFLATED, compresslevel=1, allowZip64=True)`，成员名 `source.name`，写成员启用 `force_zip64=True`。以局部 1 MiB 缓冲循环读取 CSV 和写成员，同时累计 SHA-256/大小；不能 `.read()` 无参数读取整个文件。关闭 ZipFile 中央目录后 flush/fsync/close 底层文件。重新打开 ZIP 检查唯一成员名/大小，完整读到 EOF 并核对 CRC/SHA；校验关闭后 stat 临时 ZIP，replace 正式目标，再核对 CSV 身份并 unlink。CSV 已不存在属于清理完成；身份变化、权限失败返回保留标记及明确警告。
- [x] **Step 4 — 失败路径 RED。** 新增同名旧 ZIP、ZIP 临时路径碰撞、创建/读/写/fsync/中央目录/校验/replace 失败、CSV 身份替换、CSV 已删除、CSV unlink 失败、临时清理失败的独立用例。损坏成员、额外成员、错误名称、截断和 hash/字节数不一致不得发布/删除 CSV；捕获源读取使用固定块，验证 ZIP64 标志及大尺寸边界逻辑。旧 ZIP 用 hardlink 见证或内容 hash 验证未被原地破坏。
- [x] **Step 5 — 实现失败语义并 GREEN。** `stage_changed` 仅在阶段切换调用：`zip_write`, `zip_verify`, `zip_publish`, `csv_cleanup`；worker 后续用于归一化阶段，不通过新广泛异常包装掩盖原因。每次 finally 只回收本任务排他创建的临时文件；异常路径清理失败作为诊断附带、不能替换原错误。对已成功发布 ZIP 后的 CSV unlink/身份问题只产生带警告结果。所有 fd/ZIP/member 关闭顺序可测试。
- [x] **Step 6 — 回归与 checkpoint。** `python -m pytest unit_test/base/test_raw_audio_csv_zip.py unit_test/base/test_raw_audio_csv_exporter.py -q --basetemp (New-CsvZipTestTemp)` 应全部 PASS。只提交本任务两文件，`git commit -m "feat: add verified atomic raw CSV ZIP archiving"`，审查后再 Task 2。

## Task 2: worker、IPC 与容量/路径生命周期一起接入

**Files:** Modify `base/raw_audio_csv_protocol.py`, `base/raw_audio_csv_worker.py`, `base/raw_audio_csv_service.py`, `base/raw_audio_csv_tasks.py`, `consts/raw_audio_csv_consts.py`, `unit_test/base/raw_audio_csv_fakes.py`, `unit_test/base/test_raw_audio_csv_worker.py`, `unit_test/base/test_raw_audio_csv_service.py`, `unit_test/base/test_raw_audio_csv_tasks.py`。Task 1 组件仅为必要修正才改。

- [x] **Step 1 — 协议/账本 RED。** 命令保留 CSV `temporary_path`，增加明确 ZIP 临时路径。请求仍保存 `csv_path`，统一函数派生 archive_path。结果保留 `csv_path` 的含义及 `bytes_written` 的 CSV 未压缩字节含义，新增 `archive_path`, `archive_bytes`, `csv_retained`, `cleanup_diagnostics` 和分段耗时；总 elapsed 包含所有阶段。升级共享协议版本。测试 WAV/CSV/ZIP 三路径一组占用、同路径别名冲突、与正在录音 permit 原子交接、ZIP mutation 阻止派发和 deferred cleanup 优先权。新增字段用显式 keyword，更新真正调用者/测试，不用宽松 `getattr` 隐藏缺字段。
- [x] **Step 2 — 运行 RED。** `python -m pytest unit_test/base/test_raw_audio_csv_tasks.py unit_test/base/test_raw_audio_csv_worker.py unit_test/base/test_raw_audio_csv_service.py -q --basetemp (New-CsvZipTestTemp)`，确认失败对应新增行为。
- [x] **Step 3 — worker/service 实现。** worker 先调用现有 exporter，取得 CSV 统计后调用 ZIP helper；在统一每任务 Exception 边界记录真实阶段及 root cause。helper 返回 warning 仍组装 CsvResult。Service 在派发前生成两条精确临时路径，使用短角色前缀加唯一 UUID，不能把完整最终文件名再拼 UUID（220 字符 stem 会使临时文件名超过 Windows 单段 255 字符限制）；以固定两项映射保存路径及可选身份；temporary_owned 仅接受当前 task/generation 和预期路径身份，正常与异常退出都清理/reset 全部相关状态。terminal 成功核对 CSV/ZIP 路径和 PID，成功警告也传入 release_task。UI 尚在下一任务适配，本任务以服务/worker 接口测试为门禁，不能称 UI 全绿。
- [x] **Step 4 — 真 spawn RED/GREEN。** fakes 必须是模块顶层、可跨 spawn 导入，通过真实同步原语在 zip_write/zip_verify 门闩等待，不靠 monkeypatch 跨进程。一个 active + 14 queued + 1 reserved 时第 17 次拒绝；CSV 已发布但 ZIP 未完成不能释放；两任务健康 PID 一致，失败后下一任务可用。为 CSV unlink 失败回传带警告成功，资源关闭后计数归零。
- [x] **Step 5 — 异常回收 RED/GREEN。** 覆盖两份临时所有权确认任意顺序、CSV 临时 replace 已消失、ZIP 已发布但终态丢失、未知路径/身份、旧 generation/PID、worker 在各阶段死亡。死亡确认后逐个 lstat 正规文件/身份/reparse 核验再删；没有身份只记录不删；绝不删最终 CSV/ZIP。死亡未确认保持许可不启动第二worker，关闭完成后进程/线程/IPC 归零。
- [x] **Step 6 — 回归与 checkpoint。** `python -m pytest unit_test/base/test_raw_audio_csv_zip.py unit_test/base/test_raw_audio_csv_exporter.py unit_test/base/test_raw_audio_csv_tasks.py unit_test/base/test_raw_audio_csv_worker.py unit_test/base/test_raw_audio_csv_service.py -q --basetemp (New-CsvZipTestTemp)` 应 PASS。显式 add 本任务文件，`git commit -m "feat: keep CSV ZIP export within one worker task lifecycle"`，双审。

## Task 3: Qt 展示、文件归属、归档及轮次清理

**Files:** Modify `ui/sequence/sequence_widget_raw_csv_ops.py`, `ui/sequence/sequence_widget_streaming_ops.py`, `ui/sequence/sequence_widget_round_reset_ops.py`, `base/test_round_data.py`, `base/audio_record_package.py`, `ui/archive_audio_data_dialog.py`, `ui/product_test_project_config_dialog.py`。`main_window.py` 仅更新现有关闭等待提示为“保存/压缩原始 CSV”，入口和启动行为不变。

**Tests:** `unit_test/ui/test_raw_audio_csv_recording_integration.py`, `unit_test/ui/test_raw_audio_csv_file_ownership.py`, `unit_test/ui/test_raw_audio_csv_shutdown.py`, `unit_test/ui/test_raw_audio_csv_bridge.py`, `unit_test/ui/test_raw_audio_csv_admission.py`, `unit_test/ui/test_round_reset.py`, `unit_test/test_serial_product_round_cleanup.py`, `unit_test/base/test_audio_record_package.py`, `unit_test/base/test_audio_record_delete.py`, `unit_test/ui/test_archive_audio_delete_dialog.py`, `unit_test/ui/test_archive_audio_package_dialog.py`, `unit_test/test_product_test_project_config_dialog.py`, `unit_test/ui/test_qt_test_window_cleanup.py`。仅必要时调整 `ui/sequence/sequence_widget_recording_process_ops.py`/`sequence_widget_analysis_ops.py` 中现有移动许可调用，不扩展策略。

- [x] **Step 1 — UI RED。** 测试提交前按同一原录音快照登记 CSV、ZIP；成功日志使用 archive_path；zip_write/zip_verify/zip_publish 失败提示“CSV 已保存，ZIP 失败”，CSV 导出失败提示原 CSV 失败；成功警告显示“ZIP 已保存但 CSV 清理失败”。重复/迟到消息只提示一次，不关联当前新轮次。配置仅更改“WAV + CSV（ZIP 压缩）”文案，键 `export_raw_audio_csv`、默认值和读写不变。
- [x] **Step 2 — 文件清理 RED。** 分别建立仅 ZIP、仅失败回退 CSV、ZIP+CSV、两个都尚未生成的记录，断言摘要只统计实际存在产物。在 `_register_round_file(..., is_raw_csv=True)` 明确允许本录音匹配的 `.csv.zip`，其他 ZIP 不收集。`RoundDataRecord.delete_generated_data` 对精确登记的原始数据预期产物缺失不报丢失错误；其他 WAV/分析文件缺失仍遵循现有告警，不能全局改成 missing_ok。这使压缩正常删除 CSV 或失败没产生 ZIP 都不会制造清理假警报。
- [x] **Step 3 — 归档与竞态 RED。** 外层打包输出不得与任何已选输入指向同一文件；启动线程前在许可内检查规范化路径及已有文件身份别名，拒绝冲突并保留原内容、归还许可。 `collect_audio_package_files` 仅在该录音的精确 raw_csv 路径旁追加其 `.csv.zip`，沿用原始数据类别，不扫描其他ZIP。归档删除和选中文件打包因此都包含现有ZIP（外层打包保留原 ZIP，不解压）；测试不同stem邻居ZIP、中文路径、两份残留计数及路径越界。确认对话框后在许可内重新计划实际配套文件，防止等待确认期间 CSV 变成 ZIP 造成遗漏；许可至少覆盖 WAV 和精确 CSV/ZIP 三者，重新计划仍遵循用户选中的录音/类别，不能新增其他录音。 收集端必须沿用 producer 的 `sanitize_path_component(wav_stem, max_length=220)` 规则（复用共享路径派生，不能另造近似算法）；覆盖可解析长文件名/空格，现有无法解析的历史命名不扩展推断。打包选择/保存对话框返回后，对捕获的录音路径和原选择类别重新收集；通过同一服务取得包含精确 WAV/CSV/ZIP 的许可，持有到打包线程完成或失败，启动异常也归还，避免输入 CSV 在打包过程中被压缩删除。只导出数据库的旧路径不新增文件许可。
- [x] **Step 4 — 运行 RED 并最小实现。** 用本任务 Tests 列表运行 `python -m pytest <这些路径> -q --basetemp (New-CsvZipTestTemp)`，确认新增行为失败后，实施 UI、归属和精确文件收集。不做 GUI 同步压缩或大文件读取，消息格式化使用 worker 已有结果。
- [x] **Step 5 — 生命周期集成。** 使用真实 service 和可暂停压缩的 worker 验证 GUI timer 在压缩/校验和 close 时持续；同路径移动、归档删除、重置不能穿过任务持有的许可；异常轮次 deferred cleanup 等录音/ZIP 都释放且不抢到新轮次。 串口异常回调现有行为只删 WAV 与数据库，保持该删除范围，不新增 CSV/ZIP 删除。由于任务全程持有 WAV，原 `defer_mutation((recorded_path,), ...)` 的交集即可等待完整 ZIP 生命周期；以测试验证后保留该实现，不为形式上的三路径列表扩大删除或移动策略。窗口提示“保存/压缩原始 CSV”；隐藏初始化 close 不 drain，主/可见子窗口沿用既有 own/borrowed 边界。WAV-only/校准/OK-NG 数据库标注不因 ZIP 忙被额外阻止。
- [x] **Step 6 — 回归与 checkpoint。** 验证中已定位测试夹具对 Qt/pyqtgraph 对象的错误销毁：`unit_test/ui/conftest.py` 的清理仅处理 Python 真正持有所有权的普通顶层窗口（以 `sip.ispyowned` 核实），保留 `QMenu` 的框架生命周期；无父窗口不等于没有其他 Qt 持有者，图形场景代理持有的控件不能单独删除。以真实代理控件和普通窗口测试保留/销毁结果，保留全部测试和正常捕获，复跑完整 `-q` 组合，不以禁用捕获代替验证。本任务 Tests 全部 PASS，再运行 `unit_test/ui/test_recording_process_integration.py`, `unit_test/ui/test_recording_gui_finalization.py`, `unit_test/base/test_recording_storage_consistency.py`, `unit_test/test_product_test_project_config.py`。每次使用新 basetemp。显式 add 文件，`git commit -m "feat: manage raw CSV ZIP artifacts across UI and cleanup"`，双审。

## Task 4: 诊断、基准和实际入口适配 ZIP

**Files:** Modify `tools/benchmark_raw_audio_csv_process.py`, `tools/raw_audio_csv_benchmark_load.py`, `tools/raw_audio_csv_frozen_smoke.py`, `unit_test/tools/test_raw_audio_csv_benchmark.py`, `unit_test/tools/test_raw_audio_csv_frozen_smoke.py`, `unit_test/tools/test_raw_audio_csv_resources.py`, `unit_test/ui/test_raw_audio_csv_entrypoints.py`。入口 bootstrap 无新需求时不改 `main_window.py`/`main_window_Launcher.py`。

- [x] **Step 1 — 证据 RED。** 新 `zip_csv_evidence(path)` 流式读取唯一成员，返回成员名、CSV rows/header/last_row/bytes/sha256 和独立 ZIP 字节数；失败不得返回合格内容。测试换行跨块/BOM/尾行/空CSV/非法多成员及CRC。原 `csv_evidence` 仍为旧 thread/off 与对照工具服务。新 process 模式从 result.archive_path 取输出，不能继续读已删除的 raw.csv。
- [x] **Step 2 — 分段计时 RED。** 测试 CSV export_end 保留纯导出含义，ZIP 压缩/校验/发布/清理分段有值且总时间覆盖它们；父进程 submit/dispatch、Qt 回调和 recording capture 时钟独立。保留既有 Windows 64-bit 进程资源采样与最终 CPU 读数，禁止退出后取最后一次活体值冒充退出瞬时值。
- [x] **Step 3 — self-test / content / concurrency 适配并 GREEN。** CLI flags 不变；process 指当前生产 ZIP；legacy thread 仅工具可用，不作为本次纯 CSV 子进程对照。self-test 用纯 exporter 小基线与ZIP成员逐字节比较；content/full 只流式hash，不落地解压。concurrency 保留真实200工况历史、2图、实际生产GUI路径、分析和模拟录音，process结果改为ZIP内容及比率/分段时间。空间预算同时考虑 CSV + ZIP 临时文件 + 旧ZIP，不用预期压缩率假定容量一定够；输出/报告必须新建。
- [x] **Step 4 — 入口诊断 RED/GREEN。** `--verify-raw-csv-process` 继续是显式无GUI模式；在新目录建立旧ZIP及其hardlink见证，确认成功目标身份改变、见证不变，解压内容等于原exporter、最终CSV不存在、ZIP临时文件清空、独立唯一子PID、关闭清洁。覆盖CLI错误/失败收尾以及无 stdout 的 pythonw 路径；正常无参启动保持原样。
- [x] **Step 5 — 验证与 checkpoint。** `python -m pytest unit_test/tools/test_raw_audio_csv_benchmark.py unit_test/tools/test_raw_audio_csv_resources.py unit_test/tools/test_raw_audio_csv_frozen_smoke.py unit_test/ui/test_raw_audio_csv_entrypoints.py unit_test/base/test_recording_start_timing.py -q --basetemp (New-CsvZipTestTemp)` 应 PASS。以新 UUID work-dir/report 运行 `python tools/benchmark_raw_audio_csv_process.py --mode self-test --work-dir <new-dir> --report <new-dir>/result.json`，应内容一致且干净退出；该结果不等于全量验收。`git commit -m "test: adapt raw CSV diagnostics to verified ZIP output"`，双审。

## Task 5: 集成回归、完整内容与并发比较

**Files:** Create `docs/double-sdd/verification/2026-09-23-raw-audio-csv-zip.md`。实验 controller、报告与输出放唯一新 `tmp/` 路径，不提交原始大文件。需修复源码时退回所属任务 implementer，重新双审，不让验证任务自行绕过实现审查。

- [x] **Step 1 — 总目标回归。** 运行 Task 1–4 所有测试路径去重后的全集，并加入 `unit_test/test_serial_product_condition_runtime.py`、`unit_test/ui/test_video_and_round_workflow_integration.py`；必须给出完整命令、退出码、测试数量和警告。已有失败需定位，不修改不相关策略，不以过滤测试伪造通过。
- [x] **Step 2 — 核实输入及只读对照。** 在原目录已有的上一轮验证报告中定位 600 秒源 WAV/参考JSON/代表性30秒fixture，检查文件存在、WAV 44100Hz/2通道/26460000帧及内容基准一致；在新运行目录写输入 manifest。只读对照 worktree 必须仍位于 preexistingDependencyCheckpoint，无源码差异。原始引用路径失效时按现存归档记录解析，不重新生成巨型旧 CSV 基线，不改输入。参考 JSON 可以在本轮新目录生成小型组合副本以补充原有分析基线。
- [x] **Step 3 — 完整内容与公平比较。** 从各自 worktree 根执行同一格式命令：`python tools/benchmark_raw_audio_csv_process.py --mode concurrency --csv-mode process --source-wav <same-fixture> --reference-json <same-reference> --profile representative --conditions 200 --repetitions 1 --work-dir <new-dir> --report <new-dir>/result.json`。对照原CSV与新ZIP各3次，顺序 old-new/new-old/old-new，每次目录唯一，记录首次spawn；另各跑一次 `--profile full` 的600秒fixture。运行期间不并行测试/打包，GUI尺寸/环境/分析配置一致；若未在参考中提供分析判定基线，另跑对应 off 模式作判定基线，不能把已有 `unverified` 改写为 PASS。
- [x] **Step 4 — 核验硬门槛。** 新ZIP成员的完整基准必须匹配 spec 的行数/字节数/SHA；旧CSV文件本身按同一基准核验。读取实际录音输出验证采样率/通道/帧数，无新增模拟丢帧；分析同输入判定一致，无执行错误；每个角色进程采样有效且关闭后PID不存在。把纯CSV导出耗时、ZIP写入/校验耗时、总耗时与GUI处理耗时分别列出，不能把提示到达时间当作压缩耗时。
- [x] **Step 5 — 判定性能。** 分别计算每组三次 GUI完成、实际capture启动中位数及每次Qt心跳p95的中位数。新GUI与capture中位数各 ≤ old×1.1+0.1s，Qt p95 ≤ old+20ms；完整场景报告实际方向和值。单独列压缩率与内存采样间隔/最大观测值，不承诺总导出更快。不通过则按分段定位并回实现任务，不擅自放宽标准。
- [x] **Step 6 — 记录与 checkpoint。** 报告记录环境、命令、输入hash、代码身份、真实报告链接、原始值、计算公式、全部功能/性能状态和外部待验收项；格式异常、不满足条件或只有模拟证据的项目明确未验证。文档审查需独立核对原始报告和实际ZIP内容。`git add -f -- docs/double-sdd/verification/2026-09-23-raw-audio-csv-zip.md` 后 `git commit -m "docs: record CSV ZIP validation evidence"`，双审。

## Task 6: 入口发布验证与本地收尾

**Files:** 必要时补充 `docs/double-sdd/verification/2026-09-23-raw-audio-csv-zip.md`。不编辑现有 interval launcher 的未跟踪 spec，不更改生产依赖版本以绕过本次任务。

- [x] **Step 1 — 两个源码入口。** 各以新绝对报告路径运行 `python main_window.py --verify-raw-csv-process <report>` 和 `python main_window_Launcher.py --verify-raw-csv-process <report>`，随后用同一解释器的 `pythonw.exe` 重复（每次新目录），等待结束、读取JSON和退出码，确认 `.csv.zip`、成员CSV、唯一worker和无残留，不凭无窗口表示成功。
- [x] **Step 2 — 实际冻结入口。** 在有效发布环境、两个唯一新目录分别执行 `python -m PyInstaller --noconfirm --clean --onedir --windowed --name <entry-specific-name> --distpath <new-dir>/dist --workpath <new-dir>/build --specpath <new-dir>/spec main_window_Launcher.py` 及另一目录的 `main_window.py`。用产出EXE执行同一个诊断开关，再验证普通GUI录音/关闭。此前环境有Torch依赖收集访问冲突：若再次明确复现并停止进展，保留日志和未验证状态，核对任务进程归属后收尾，不能无限重复构建或偷偷排除依赖；继续独立可完成验证。
- [x] **Step 3 — 现场验收边界。** 有用户授权和可用设备时完成实际录音/压缩/关闭；缺设备时报告“真实设备未验证”，软件模拟结果不得替代。不得为等待外部验收而擅自把已批准的本地回填与历史清理遗漏；是否接受带外部待验收项的本地交付依照用户指示。
- [x] **Step 4 — 全范围审查与交付。** 最终双审全部本次实现差异；按 `verification-before-completion` 核实证据，使用 `finishing-a-development-branch` 执行用户选定本地交付。文件从执行依赖检查点到最终版本抽取 ZIP 差异，结合原目录文档和元数据的逐路径基线核对前置未跟踪依赖；原有tracked保护检查点内容单独保留。完整内容保留为本地未提交修改，不创建最终提交/PR/push。
- [x] **Step 5 — 临时历史清理。** 归档仍被验证文档引用的实验文件并更新链接，再移除本任务feature/只读对照worktree和临时分支。按检查点分离重建从 recorded mainBase 起的本地内容，验证交付文件及原文件hash、索引意图和其他任务引用；不得直接硬重置、删除原未跟踪文件或重写无关永久提交。清理保护检查点/文档检查点，metadata只在确实交付后标记completed；原始主分支若期间有新用户提交，保留并逐一识别本任务临时记录，不能盲目退回旧base。交付与清理各自报告结果。

## 每项审查必须检查的约束

1. 所有派生输出均为完整 CSV 文件名追加 `.zip`；ZIP唯一成员保留CSV名，不使用 `with_suffix('.zip')` 丢掉 `.csv`。
2. CSV纯导出契约不变。只有ZIP已完整校验并发布才尝试删除CSV；结果统计失败或worker死亡不额外删除最终产物。
3. 一任务一名额，WAV/CSV/ZIP占用持续到实际释放；两份临时文件身份分别跟踪，无父进程目录扫描/未经确认回收。
4. 仅worker任务边界统一广泛捕获；底层异常保留原因，清理失败不掩盖首因，CSV删除失败带警告成功。禁止静默fallback、过度重复防御和无意义throw。
5. 采用实例/局部状态；单次模块级常量仅允许稳定外部协议/格式/序列化边界，共享常量按项目放 `consts`；无需为单处缓冲尺寸增加全局配置。
6. 不使用大数组IPC、整CSV内存读取、新线程/worker队列或GUI阻塞；不顺手重构录音/分析，不批量改历史文件。
7. 代码与测试变更只在授权工作区，遵守每项TDD及新鲜验证；不覆盖其他任务的改动。

Task 6 release evidence: four source diagnostics passed; both frozen builds hit the documented Torch native collector failure. Frozen runtime and physical hardware remain UNVERIFIED. Local software delivery proceeds under the approved scope.

# Recording Final Envelope Implementation Plan

> **For agentic workers:** REQUIRED: Use the `subagent-driven-development` skill to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 主录制使用录制期间累计的整段峰谷完成显示，取消收尾全量WAV回读，同时保留基本检查、质量门、资源生命周期和现有业务流程。

**Architecture:** 先实现不接入生产路由的有界峰谷、流式质量统计和结果协议模块。随后一次性接通采集子进程、service和GUI，避免新旧完成对象混用；最后执行真实进程/Qt回归和本地模拟。其他purpose及旧非进程数组路径保持兼容。

**Tech Stack:** Python 3.12、NumPy、soundfile、multiprocessing、PyQt5、pytest。

---

## 执行上下文与边界

Spec: `docs/double-sdd/specs/2026-10-12-recording-final-envelope-design.md`，已通过文档审查和用户审阅。

所有命令在现有 `.worktrees/codex/recording-final-envelope` 工作树根目录执行，不能重新创建或覆盖该工作树。上游运行记录位于原仓库根目录的 `.worktrees/20261012-142749-recording-final-envelope.metadata.json`；调度时传递原先收到的精确完整metadataPath。基础提交由该运行记录提供，不从当前HEAD重新推断。

工作树已有隐藏全局OK/NG删除，保留其7个未提交文件改动，不能重做或回退。规格已有临时文档检查点。代码执行从用户批准本计划后开始；本计划不授权最终提交、合回主目录或发布。每任务由新implementer按顺序实现，完成后对同一差异启动spec-code-reviewer和quality-code-reviewer，按code-review技能核实意见，规格审查为门禁。保留每任务的精确起始差异，避免将前置删除误记为新增变化。

测试环境：`$env:QT_QPA_PLATFORM='offscreen'`。系统Temp的pytest目录当前不可访问，所有命令指定工作树内新的 `--basetemp=.pytest_cache/<本次运行唯一名>`，不要重用其他进程正使用的目录。新测试和生产模块均不得连接真实音频设备、网络共享或生产录音目录。

已知原有工具栏基线：`test_production_blocked_config_and_barcode_updates_keep_full_hints` 缺少fixture属性；`test_serial_states_paint_without_losing_native_status` 的3个offscreen变体失败。不可顺手修复或隐藏；本方案相关测试出现其他失败必须调查。当前定向标签/模式/串口接线16项通过。

## 文件职责

| 文件 | 职责 |
|---|---|
| 新增 `base/recording_final_envelope.py` | 只处理已经裁剪和量化的块，输出有界位置/幅值；无Qt、文件或IPC |
| 新增 `base/recording_quality_accumulator.py` | 请求内质量指标累计；无历史样本缓存 |
| 修改 `base/recording_settings.py` | 提取批量与流式共用的阈值判定/提示，保持旧函数行为 |
| 修改 `base/recording_process_protocol.py`、`consts/recording_result_consts.py` | 稳定的版本和最终摘要/质量证据值协议 |
| 新增 `base/recording_final_result.py` | 父进程校验、时间轴构造、只读所有权和请求绑定 |
| 修改 `base/recording_capture.py` | writer返回后累计、质量门、元数据检查、main完成payload |
| 修改 `base/recording_service.py` | main无reader交付及原有租约/accepted/released流程 |
| 修改 `ui/sequence/sequence_widget_recording_process_ops.py`、`sequence_widget_streaming_ops.py` | 新结果准入/发布，局部提取共享完成业务，保留旧数组路径 |
| 修改 `base/data_struct/data_deal_struct.py`、`ui/sequence/recording_process_context.py` | 显式保存来源/显示来源所需最小实例状态 |
| 按需修改 `ui/sequence/channel_plot_workspace.py` | 目标条件显示失效/失败占位，不能清空其他条件 |
| 修改既有测试并新增下述测试 | 算法、质量、协议、生命周期、真实Qt路由回归 |
| 新增 `tools/benchmark_recording_final_envelope.py` | 可重复运行的本地模拟与JSON指标；仅开发用途 |

不改动 `signal_analysis_window.py`、`reference_spectrum_analysis_window.py` 的旧原始数组协议，不恢复隐藏入口，不重写CSV或分析工作进程。只有经调用链确认属于本方案的接口变化才扩展范围。

## Task 1: 有界整段峰谷累计器

**Create:** `base/recording_final_envelope.py`, `unit_test/base/test_recording_final_envelope.py`。
**Read:** `base/recording_waveform_preparation.py`, `consts/recording_preview_consts.py`, `base/wav_pcm24.py`。

- [x] 写第一个失败测试：`FinalEnvelopeAccumulator(total_frames=N, channel_count=C, max_points=P)`，提供 `append(float32_frames_first_block)` 和 `finish()`，返回按列排序的 `(int64_indices, float32_amplitudes)` tuple。使用已量化数据，按不整齐块边界追加，比较现有批量函数。示例测试主体如下，测试文件补齐正常imports：

```python
def test_arbitrary_chunks_match_batch_display():
    rng = np.random.default_rng(71)
    data = quantize_pcm24(rng.uniform(-1, 1, (113, 3)).astype(np.float32))
    acc = FinalEnvelopeAccumulator(total_frames=113, channel_count=3, max_points=12)
    for block in np.split(data, [1, 8, 49, 50, 110]):
        acc.append(block)
    for column, (indices, values) in enumerate(acc.finish()):
        expected_time, expected_values = prepare_waveform_display_data(
            data[:, column], 44100, max_points=12)
        np.testing.assert_array_equal(indices.astype(np.float64) / 44100, expected_time)
        np.testing.assert_array_equal(values, expected_values)
        assert len(indices) <= 12
```

- [x] RED：`python -m pytest unit_test/base/test_recording_final_envelope.py -q --basetemp=.pytest_cache/envelope_t1_red`，预期缺少模块/API而失败。
- [x] 实现：N<=P预分配N×C float32显示缓冲；N>P采用B=(P-2)//2、W=(N+B-1)//B，每桶min/max与最早位置。块内完整桶可向量化，跨块边界合并当前桶；相同极值仅在严格改善时更新位置。额外保存首/末点，finish合并位置并去重。禁止逐样本Python循环、历史块列表或N×C长录音数组。
- [x] 验证API边界：N/C/P为有效整数、形状/dtype和连续帧总量匹配；不能超过N；未满N不能输出成功；输入不修改。捕获线程调用前已量化，但模块明确拒绝非有限输入。所有状态为实例字段，无新全局缓存。
- [x] 扩展测试：N=1、N=P、N=P+1、常量、正负脉冲、同值极值跨块、尾桶、5通道、P=4、空块约定（允许形状正确的零行块不改变状态）、超额/不足帧、不同分块结果完全相同。使用大N但少量逐块输入验证内部容量在创建时即有界；完整长样本正确性用中等N，不给单元测试分配600秒音频。
- [x] GREEN：重跑上述文件，全部通过；`git diff --check`；记录数组nbytes和复杂度以供审查。此任务不改生产录制路由。

## Task 2: 流式质量统计及共享判定

**Create:** `base/recording_quality_accumulator.py`, `unit_test/base/test_recording_quality_accumulator.py`。
**Modify:** `base/recording_settings.py`。
**Read/test:** `unit_test/base/test_recording_finalization.py`，现有质量门相关测试。

- [x] RED：用量化后的多通道块建立 `RecordingQualityAccumulator(channel_count=C, range_max=None)`；普通声卡range_max=None，VE使用冻结请求的range_max。`append`只累计；`finish(thresholds)`返回固定大小指标、passed/skipped及原有reason/detail。首个测试比较正常样本流式结果与 `validate_recorded_audio`，运行 `python -m pytest unit_test/base/test_recording_quality_accumulator.py -q --basetemp=.pytest_cache/envelope_t2_red`，预期失败。
- [x] 提取 `validate_recorded_audio` 的指标→阈值判定为共用函数，例如 `validate_recorded_audio_metrics(count, peak, rms, variance, thresholds)`。批量函数仍按原有dtype及算法求指标，再调用该函数；原有enabled=false、空样本提示、比较顺序及格式保持不变。不要复制一套可能漂移的错误文本。
- [x] 实现每块mono：声卡 `quantized.mean(axis=1, dtype=np.float32)`；VE先对多通道除range_max，再求同样均值，另验证原电压mono有限。float64统计维护count、peak、square_sum、mean、M2。合并公式：delta=block_mean-mean，new_mean=mean+delta*m/(n+m)，new_M2=M2+block_M2+delta**2*n*m/(n+m)。RMS=sqrt(square_sum/count)，variance=M2/count。只保留块级临时数组，不保留历史mono。
- [x] 加测试：静音、常量、弱信号、正常信号、多通道抵消、VE先除后均值、trim后样本（由调用方裁剪）、disabled状态、非有限值和电压mono溢出。指标允许float64归约误差（选非退化数据并给出数值容差），判定不加宽阈值。近阈值用np.nextafter及精确可表示样本验证比较方向；若仅舍入造成边界差异则记录具体案例，不能静默放宽阈值。
- [x] GREEN：`python -m pytest unit_test/base/test_recording_quality_accumulator.py unit_test/base/test_recording_finalization.py -q --basetemp=.pytest_cache/envelope_t2_green`。此时生产路径仍未迁移，既有finalization测试应保持旧结果；任何原有失败先与任务开始状态比较。

## Task 3: 摘要协议与父进程准备模块（暂不切换路由）

**Modify:** `base/recording_process_protocol.py`, `consts/recording_result_consts.py`。
**Create:** `base/recording_final_result.py`, `unit_test/base/test_recording_final_result.py`。

- [x] 定义测试用新payload，RED运行 `python -m pytest unit_test/base/test_recording_final_result.py -q --basetemp=.pytest_cache/envelope_t3_red`。
- [x] 增加具名协议类型（名称可用 `RecordingFinalEnvelope` 与 `RecordingQualityEvidence`）：envelope包含版本、frame_count、channels、columns、display_error及quality；quality包含count、status、peak、rms、variance和VE有限值检查状态。status仅passed/skipped，failed不允许出现在completed；disabled时数值证据可为None，但计数及VE检查状态仍需明确。给 `RecordingResult` 末尾增加 `final_envelope=None`，不移动旧位置参数。
- [x] `prepare_main_recording_result(descriptor, request)` 返回专用 `PreparedRecordingResult`，包含descriptor、waveforms和display_error，不能包含multi/mono。检查request_id/path/purpose/rate/channel顺序、raw/final counts、handles_released、metadata_appended的VE要求；main digest字段应为None；新payload版本和N匹配；质量状态匹配冻结enabled，passed指标有限且再次用小规模阈值函数验证，VE标识必须证明有限值检查成功。
- [x] 正常columns为C组一维数组，dtype精确int64/float32、大小相同且1..P、位置严格递增、首0末N-1、幅值有限。display_error分支必须为非空字符串且columns为空；两者混合拒绝。仅此分支可不携带曲线，不能把不合法数据自动改成显示失败。
- [x] 有界分配时间轴，建立无可写别名的只读发布所有权，绑定精确request与descriptor/数组身份。`is_prepared_for(request)`仅做身份及固定大小结构检查，不重新扫描全部点；dataclasses.replace或手工构造不能继承准备证据。pickle传输前后的只读标志不可信，因此在父进程重新包装/必要时复制有界数组。
- [x] GREEN测试：正确稀疏通道顺序、显示错误、质量disabled、恶意shape/dtype/index/通道/版本/数值/缺失证据、质量证据不匹配；伪造/replace、可写别名、setflags(write=True)、pickle往返；验证工厂不调用任何文件API。重跑本文件及 `unit_test/base/test_recording_service.py`，生产service尚未改路由，不能因此破坏旧协议用例。

## Task 4: 一次性切换主录制完成链路

**Modify:** `base/recording_capture.py`, `base/recording_service.py`, `base/data_struct/data_deal_struct.py`, `ui/sequence/sequence_widget_recording_process_ops.py`, `ui/sequence/sequence_widget_streaming_ops.py`, `ui/sequence/recording_process_context.py`, `ui/sequence/channel_plot_workspace.py`（仅所需显示API）。
**Modify tests:** `unit_test/base/test_recording_finalization.py`, `unit_test/base/test_recording_capture.py`, `unit_test/base/test_recording_metadata_append.py`, `unit_test/base/test_recording_service.py`, `unit_test/base/test_recording_service_pipeline.py`, `unit_test/ui/test_recording_process_integration.py`, `unit_test/ui/test_recording_gui_finalization.py`。
**Create:** `unit_test/ui/test_recording_envelope_publication.py`。

此任务是一个完整可评审的纵向切换；不得拆成已启用新子进程却保留旧父进程/GUI的交付状态。前3任务的纯模块均已通过验证。

执行细化：Task4的完整回归包含544项，旧main reader门控fixture迁移需要单独集中处理。按顺序执行两个实现pass：第一pass负责生产链路及base/GUI_finalization/新增测试；完成后第二pass只迁移 `unit_test/ui/test_recording_process_integration.py` 的旧数组/reader门控用例。不得并行修改代码；第二pass以子进程finalization或待接收结果门控替换旧reader门控，保持取消/租约断言。全部完成并验证后，对Task4整体差异做规格和代码双审查；不将中间状态交付为完成。

- [x] 先建立RED测试：在真实spawn main服务注入会抛AssertionError的reader_factory，正常完成仍应result_ready→accepted→released；GUI结果没有multi/mono、真实N和完整整段波形正确。新增用例先运行：`python -m pytest unit_test/ui/test_recording_envelope_publication.py -q --basetemp=.pytest_cache/envelope_t4_red`，确认因为旧reader/数组路由失败。
- [x] 子进程：仅main创建两个实例，N按有效裁剪计算；writer成功返回后先完成质量累计，再在独立显示错误边界累计峰谷。质量错误仍进入已有录制失败边界；显示错误只记录一次、释放累计器并记录display_error，后续仍写WAV。main不构造/更新hashlib摘要；其他purpose不变。最终finish产生有界payload，日志区分累计与物化时间。
- [x] 子进程收尾：沿用counts、WAV头检查、释放和取消时序；把reader对metadata_appended=True的普通声卡校准元数据检查迁入子进程，与VE共用清楚的边界。使用 `normalize_wav_calibration_metadata` 保持原reader比较语义；保留retained_handles/cleanup_paths/close_errors/rollback_errors的所有权。必要质量门失败时返回RecordingFailure，不发送completed；enabled=false发送skipped。元数据结构检查不得读取data样本。
- [x] service：在既有completed描述符验证后，main调用准备工厂并设置session.audio/result_ready；未启动reader则_reader_released=True，但保持_child_released与cleanup门。别的purpose保留原reader_factory和_read流程。所有控制操作仍在service所有者线程，准入、VE双槽位、generation、取消/拒绝、已释放路径及重复终态规则不变；任何准备异常归一化一次为该session协议失败。
- [x] GUI准入只接受main专用prepared结果，确认精确请求及冻结VE配置，继续预检目标窗口。在accepted且released后调用新摘要发布入口，不传None进旧 `_on_streaming_complete` 以触发 `streaming_processor.get_recorded_data()`。旧非进程调用及其他purpose保持原数组行为。
- [x] 从 `_on_streaming_complete` 定向提取必须共用的发布后业务：CSV提交、条件缓存、数据库、记录历史、按钮/条码状态、自动分析入队、串口/手动循环续接。保留执行顺序、reentrancy guard和现有失败处理，避免复制两套长收尾代码。分别提供原数组/新摘要的准备和投图步骤，不在公共业务函数中假设原始数组存在。
- [x] 已保存来源用不可混淆的请求身份及path/rate/N/calibration状态发布；store_wave_data和store_wave_data_multi清空为None，audio_lenth=N。把来源清理纳入DataDealStruct.clear_data等既有重置边界；`_clear_unpublished_recording_source` 区分旧已发布来源与新未发布上下文，不因数组为None就清除旧校准。标记改名后更新当前路径而不改变请求身份；其他条件来源仍独立。
- [x] 显示事务：开始刷新使目标条件显示来源失效；全通道set_data成功才标完成并绑定新请求。第二通道异常时先调用现有rollback，再隐藏该条件旧/部分曲线和交互，显示失败占位；不清空别的条件。摘要display_error走相同占位路径，无WAV回读；仍发布新文件来源并执行文件业务。
- [x] 更新旧测试的已批准变化：main读一次/digest匹配/全长数组断言替换为main不读/摘要等价/无原始数组；仍保留其他purpose的digest损坏检测、原reader直接调用验证。main旧的“磁盘样本修改后必须被摘要比对发现”不再是要求，明确测试分类，不用它要求恢复回读。保留counts、元数据、失败提示、租约及路径保护断言。
- [x] GREEN：先运行上述新增UI文件和 `unit_test/base/test_recording_finalization.py unit_test/base/test_recording_metadata_append.py unit_test/base/test_recording_service.py unit_test/base/test_recording_service_pipeline.py unit_test/ui/test_recording_gui_finalization.py unit_test/ui/test_recording_process_integration.py`，用 `python -m pytest <这些路径> -q --tb=short --basetemp=.pytest_cache/envelope_t4_green`。对受影响fixture作最小适配，不删除生命周期断言；新增全部通过，既有失败须提供修改前证据。审查整个纵向diff后再进入Task5。

## Task 5: 兼容回归、故障覆盖和本地模拟报告

**Modify tests:** `unit_test/base/test_recording_audio_lifetime.py`, `unit_test/base/test_recording_storage_consistency.py`, `unit_test/base/test_recording_worker_pipeline.py`, `unit_test/ui/test_raw_audio_csv_recording_integration.py`, `unit_test/ui/test_original_recording_analysis_route.py`, `unit_test/ui/test_multichannel_recording_workspace.py`, `unit_test/ui/test_ve3668n_recording.py`，及Task4新测试文件。
**Create:** `tools/benchmark_recording_final_envelope.py`, `unit_test/base/test_recording_envelope_benchmark.py`。

- [x] 建立缺少覆盖的失败用例，再补必要的最小实现：preview=false/relative_latest/cumulative及预览故障都能完成整段摘要；trim跨块/大于目标；声卡已追加元数据不匹配和关闭失败；VE质量/元数据失败；cancel/reject/晚到/旧generation/工作进程死亡；显示第二通道失败；新录制取消保留旧已保存来源；标记改名及CSV/分析收到新路径。若Task4已覆盖，不重复写同义测试。
- [x] 验证当前手动/自动分析通过既有文件路径运行，峰谷数组从未写入原始数据字段；测试CSV后台读取可继续，不能用“禁止所有sf.read”的全局猴子补丁误禁独立分析。无回读测试只约束main录制完成链路，子进程允许头和元数据检查。
- [x] 运行聚焦回归命令（每次换新basetemp）：

```powershell
$env:QT_QPA_PLATFORM='offscreen'
python -m pytest unit_test/base/test_recording_final_envelope.py unit_test/base/test_recording_quality_accumulator.py unit_test/base/test_recording_final_result.py unit_test/base/test_recording_capture.py unit_test/base/test_recording_finalization.py unit_test/base/test_recording_metadata_append.py unit_test/base/test_recording_service.py unit_test/base/test_recording_service_pipeline.py unit_test/base/test_recording_worker_pipeline.py unit_test/base/test_recording_storage_consistency.py unit_test/base/test_recording_audio_lifetime.py -q --tb=short --basetemp=.pytest_cache/envelope_base_regression
python -m pytest unit_test/ui/test_recording_envelope_publication.py unit_test/ui/test_recording_gui_finalization.py unit_test/ui/test_recording_process_integration.py unit_test/ui/test_raw_audio_csv_recording_integration.py unit_test/ui/test_original_recording_analysis_route.py unit_test/ui/test_multichannel_recording_workspace.py unit_test/ui/test_ve3668n_recording.py -q --tb=short --basetemp=.pytest_cache/envelope_ui_regression
```

预期无新增失败。根据失败或代码影响再扩展相邻测试，不盲目运行全仓库硬件/长耗时测试。使用verification-before-completion技能获取新鲜结果。

- [x] benchmark先写低时长smoke测试：新脚本命令行接受 `--seconds 2 600 --channels 5 --sample-rate 44100 --block-frames 2048 --output <json>`；拒绝UNC/网络目标，输出目录在本地工作树或用户明确本地目录，输入为按块生成的确定性合成音频，绝不预分配600秒原始多通道数组。RED后实现脚本并GREEN运行 `python -m pytest unit_test/base/test_recording_envelope_benchmark.py -q --basetemp=.pytest_cache/envelope_benchmark_smoke`。
- [x] benchmark对照在相互独立的子进程运行，避免旧路径大数组峰值污染新路径指标。新路径走实际capture/service完成接口；旧路径作为测试专用对照复用保留的ResultReader，提供已知生成器的digest和冻结请求，不能给生产main增加fallback。分开报告capture/write/close、摘要累计/物化、父进程准备、GUI发布及旧read/digest/display；无法直接复用真实GUI时明确该项单独由Qt测试测量，不伪称整界面测量。旧路径读取次数只能在对照模式出现。
- [x] 时间用每进程perf_counter/process_time，IPC阶段只在同一测量者内取区间或报告传输加调度总量，不直接减不同进程未经校准的时间戳。Windows内存可用ctypes的GetProcessMemoryInfo读取工作集/峰值；周期采样父子同时RSS/working-set总和并说明采样误差。进程各自峰值之和不能冒充同时总峰值，tracemalloc不能冒充NumPy原生内存峰值。
- [x] 实跑 `python tools/benchmark_recording_final_envelope.py --seconds 2 600 --channels 5 --sample-rate 44100 --block-frames 2048 --output .pytest_cache/recording-final-envelope-profile.json`，报告墙钟、CPU、阶段、点数、累计数组nbytes、父子各自峰值及同时采样总峰值。用重复追加到不同总N的容量测试证明有界，不能仅凭一次RSS低就声称有界；验证新结果没有完整multi+mono约635MB分配。报告磁盘型号/目录与缓存条件，不把模拟值外推为真实设备保证；CSV/分析默认关闭，说明开启时仍独立读取和竞争磁盘。
- [x] 记录结果到 `docs/double-sdd/reports/2026-10-12-recording-final-envelope-validation.md`（明确实测环境、命令、基线、限制和性能数据，不存生产录音）。运行 `git diff --check`，核对源代码引用和本次改动范围。提交最终整体验证结果给独立quality-code-reviewer；依finishing-a-development-branch保留工作树，未经用户授权不合回或发布。


## 通用实现与审查约束

- 没有新的宽泛catch、silent fallback、重复包装或全局状态。捕获显示异常和资源边界异常必须各有明确恢复语义；质量/计数失败不得落入显示警告。
- 单次使用的普通常量放局部；协议版本等稳定外部边界可放共享consts，不增加全局可变缓存。
- 所有主录制分支必须满足无reader、无全长数组、质量与元数据保障保留。不能借用旧read_result helper绕过新的真实链路测试。
- 其他purpose、非进程录制、自动/手动分析及CSV保持原有文件/数组契约。隐藏全局OK/NG前置删除保持原样。
- 每任务记录RED/GREEN命令、退出码、实际通过/失败数及审查结果；原有失败不写成通过。性能收益以实测报告为准，不承诺固定收尾秒数或5–10MB总内存峰值。

# Raw audio CSV process validation — 2026-09-22

**Overall acceptance: UNVERIFIED.** All measured software gates and the four specified timing comparisons passed in this source/offscreen/simulated-capture experiment. Real-device capture, frozen executable diagnostics, and normal frozen GUI operation remain unverified. This document does not claim project acceptance is complete.

## Evidence and environment

Experiments ran sequentially from 2026-09-22T20:37:49+0800 to 2026-09-22T20:42:11+0800. Windows 10 build 19045, AMD Ryzen 5 5600GT (6 cores / 12 logical CPUs), 16,413,032,448 bytes physical RAM. D: had 285.86 GiB free before trials. No dependencies were installed or changed. The controller requested a quiet machine; no other agent builds or tests were run concurrently. Whole-system background utilization was not independently sampled, so unrelated OS activity cannot be excluded. Participating-process load is fully reported below.

Environment: `{"numpy": "1.26.4", "soundfile": "0.12.1", "PyQt5": "5.15.11", "PyQt5-Qt5": "5.15.2", "pytest": "9.0.3", "PyInstaller": "6.21.0"}`; Python 3.12.7, 64-bit. `QT_QPA_PLATFORM=offscreen` for every mode. Widgets were rendered at 1440×900 offscreen; this is not manual/native GUI evidence.

[Controller](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-task10-56c360747fcd4749993a211b0a89ebd9/controller.py>), [complete commands / exits / report paths](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-task10-56c360747fcd4749993a211b0a89ebd9/manifest.json>), [independent report checks](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-task10-56c360747fcd4749993a211b0a89ebd9/summarize.py>), [computed summary](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-task10-56c360747fcd4749993a211b0a89ebd9/summary.json>). All 12 benchmark commands exited **0**. All raw top-level statuses intentionally remain `unverified` because a CLI run cannot accept hardware or comparative performance. Derived gates below do not rewrite those reports.

## Inputs and full-content fidelity

Original input and baseline: [original 600-second reference](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/full-csv-benchmark-20260922-105431/result.json>). Original WAV: 26,460,000 frames, 44,100 Hz, two channels, 600 seconds, 211,680,392 bytes, SHA256 `64b8934b433b3f137754e5b0944d7772b2ff584d7c53a544c559305f34da0c10`.

Full CSV must have 26,460,000 data rows, 1,162,232,136 bytes, SHA256 `facb5196b26f7ac36fdb470fe8850a89106fd94b7602164b40c24eeba38e5b65`, header `time_s,CH1,CH2`, final row `599.999977324,-0.000335693359,-0.000366210938`. Both new full concurrent thread/process outputs match all five fields.

Previously completed [offline full process content report](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-full-content-592a642e29fb4315ba6b466feb136a87/result.json>) is reused for Task 10 Step 2: PASS, 79.8026734 seconds worker-internal export, 80.0137925 seconds submission-to-release, source unchanged and clean exit. Subsequent changes did not alter the exporter. It is not reused as a concurrent performance observation; new full trials were run under the required load. No original giant CSV baseline or offline full output was regenerated.

Representative input: [prepared 30-second reference](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-representative-fixture-28a9eb6e844b45b6bb16e7e7cfc2ca36/reference.json>). First 1,323,000 frames of the same WAV, FLOAT stereo, 44,100 Hz; WAV SHA256 `d1a9e61d2b12eb20ee447578d33d42ca9e31717d4b4b5fdb2338bfda4aaceff5`. CSV: 1,323,000 rows, 56,601,607 bytes, SHA256 `4828d24a2acd305b642d9c4f10a489a703a53fa75abadd5111535f77063845c0`, final row `29.999977324,0.000549316406,0.000579833984`. All six new representative CSV outputs match. Every run reports source unchanged.

## Controlled workload and measurement boundaries

The existing `tools.raw_audio_csv_benchmark_load` tool loads actual `SequenceWidgetStreamingOpsMixin` publication and analysis mixin paths, `MotorResultPanel`, `AnalysisWaveformPanel`, and `RecentSessionPanel`; source audio is validated by `ResultReader`. The historical state is prepared with product constructors/mergers, then rendered once before timing. Every run confirms 200 history records, 200 grouped conditions, 200 populated condition cells, two physical plots, and one inserted database row. Preparation/first paint is excluded from CPU samples and measured GUI completion; preparation durations are reported separately. During timing the unchanged product history update, waveform projection, SQLite save, and analysis enqueue execute. This is a purpose-built host using real components, not the complete normal MainWindow.

CSV off disables formatting; thread mode invokes the unchanged exporter on a benchmark-only parent thread; process mode uses the real RawAudioCsvService with one independent spawn child. All process trials use `--repetitions 1`, therefore all are **first spawn**. The preceding Task 8 two-repetition smoke established healthy PID reuse functionally, but these results make no steady-reuse performance claim.

Real `AnalysisProcessService` executes two SPL instances (physical channels 0 and 1), Z weighting, upper limit 50.0, same configuration and same input within each profile. Benchmark fallback calibration is 1.0; existing file calibration takes precedence. Both instances and overall judgement are OK for every representative run and NG for every full run; each matches its same-input off baseline. NG is a valid judgement, not an execution failure; execution is completed, 2/2 instances, 0 execution failures, 0 artifact failures. Each profile uses its first CSV-off report as its analysis baseline. Its raw analysis status is necessarily `unverified`; posthoc checks confirm its completed execution and match every later run to its judgement. The new combined references preserve the original CSV reference fields and add exact source SHA/configuration/runs: [representative](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-task10-56c360747fcd4749993a211b0a89ebd9/representative-combined-reference.json>) and [full](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-task10-56c360747fcd4749993a211b0a89ebd9/full-combined-reference.json>). Raw off reports are unchanged.

Next capture uses real RecordingService, process protocol, bridge, result validation and WAV writer, with the writer-paced deterministic sounddevice fake from `unit_test.base.recording_process_fakes`, wrapped by `tools.raw_audio_csv_benchmark_load.recording_dependencies`. It is **SIMULATED**, not target hardware. Every run requests 88,200 raw frames at 44,100 Hz, two channels, trims 441 startup frames, and closes a WAV containing exactly 87,759 frames. Raw/consumed frames equal 88,200, drop count is 0, no overflow callback or capture failure is recorded.

Capture latency is the production accepted request to parent validation of the worker-started event; it includes IPC/supervisor delay. It is not a pure hardware timestamp. Qt-started latency and subsequent delivery delay are separate. All calculated differences below use the same parent clock; child export duration is calculated internally by the child. Absolute child capture observations remain available in raw JSON but are not subtracted from parent timestamps. The next request is issued in the same Qt turn after GUI publication completes.

50 ms precise Qt timer lateness is max(0, inter-tick interval − 50 ms). Each run retains all samples. For the gate, the group statistic is the **median of the three per-run p95 values** (equal run weighting, not pooled samples). Sampling lasts through the entire trial until recording, analysis and CSV finish, so different trial lengths can dilute an initial blocked turn differently; the per-run maximum is also shown.

## Commands and raw results

The controller invokes the following command once per row, substituting the exact source, reference and fresh output directory shown in the manifest. It runs `off, thread, process; thread, process, off; process, off, thread` for the representative fixture, then `off, thread, process` for the full fixture. No comparisons were parallelized.

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python tools/benchmark_raw_audio_csv_process.py --mode concurrency --profile PROFILE --conditions 200 --repetitions 1 --csv-mode MODE --source-wav SOURCE --reference-json REFERENCE --work-dir FRESH_DIR --report FRESH_DIR/result.json
```

GUI/capture/Qt times are seconds; heartbeat values are milliseconds. Each report link provides the raw stages, capture observations, full CSV evidence and live resource samples.

### Representative

| Run / raw report | GUI | Capture observed | Qt started | Qt delivery | Heartbeat p50 / p95 / max | Exit |
| --- | --- | --- | --- | --- | --- | --- |
| [representative-01-off](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-task10-56c360747fcd4749993a211b0a89ebd9/representative-01-off/result.json>) | 0.687176 | 0.334233 | 0.334747 | 0.000514 | 0.052800/17.156080/762.832500 | 0 |
| [representative-02-thread](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-task10-56c360747fcd4749993a211b0a89ebd9/representative-02-thread/result.json>) | 4.587369 | 0.307686 | 0.308325 | 0.000639 | 0.040200/11.137180/4658.019700 | 0 |
| [representative-03-process](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-task10-56c360747fcd4749993a211b0a89ebd9/representative-03-process/result.json>) | 0.668685 | 0.369958 | 0.370682 | 0.000723 | 0.000000/0.947400/747.587800 | 0 |
| [representative-04-thread](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-task10-56c360747fcd4749993a211b0a89ebd9/representative-04-thread/result.json>) | 4.628390 | 0.310044 | 0.310447 | 0.000404 | 0.053900/74.494825/4629.010100 | 0 |
| [representative-05-process](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-task10-56c360747fcd4749993a211b0a89ebd9/representative-05-process/result.json>) | 0.687708 | 0.317544 | 0.317823 | 0.000279 | 0.026250/0.940875/751.778600 | 0 |
| [representative-06-off](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-task10-56c360747fcd4749993a211b0a89ebd9/representative-06-off/result.json>) | 0.639259 | 0.292386 | 0.292776 | 0.000390 | 0.023700/10.688130/701.189900 | 0 |
| [representative-07-process](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-task10-56c360747fcd4749993a211b0a89ebd9/representative-07-process/result.json>) | 0.671164 | 0.318738 | 0.319410 | 0.000672 | 0.016200/0.905105/732.900300 | 0 |
| [representative-08-off](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-task10-56c360747fcd4749993a211b0a89ebd9/representative-08-off/result.json>) | 0.629356 | 0.335737 | 0.336464 | 0.000727 | 0.082200/114.430725/630.589900 | 0 |
| [representative-09-thread](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-task10-56c360747fcd4749993a211b0a89ebd9/representative-09-thread/result.json>) | 4.659697 | 0.298755 | 0.299596 | 0.000841 | 0.000000/69.745835/4660.854800 | 0 |

### Full

| Run / raw report | GUI | Capture observed | Qt started | Qt delivery | Heartbeat p50 / p95 / max | Exit |
| --- | --- | --- | --- | --- | --- | --- |
| [full-01-off](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-task10-56c360747fcd4749993a211b0a89ebd9/full-01-off/result.json>) | 0.643244 | 0.310038 | 0.310356 | 0.000317 | 0.000000/0.884270/712.558700 | 0 |
| [full-02-thread](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-task10-56c360747fcd4749993a211b0a89ebd9/full-02-thread/result.json>) | 5.044781 | 0.340465 | 1.094378 | 0.753913 | 0.000000/1.716560/5243.147500 | 0 |
| [full-03-process](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-task10-56c360747fcd4749993a211b0a89ebd9/full-03-process/result.json>) | 0.667410 | 0.319254 | 0.319998 | 0.000745 | 0.009600/0.782280/734.811800 | 0 |

### Representative medians and thresholds

| Mode | GUI seconds | Capture seconds | Median per-run p95 ms |
| --- | --- | --- | --- |
| off | 0.639259 | 0.334233 | 17.156080 |
| thread | 4.628390 | 0.307686 | 69.745835 |
| process | 0.671164 | 0.318738 | 0.940875 |

| Gate | Observed process | Limit / comparator | Outcome |
| --- | --- | --- | --- |
| GUI reduction ≥30% | 0.671164 s | 3.239873 s; reduction 85.498980% | PASS |
| Capture ≤ 1.10 × thread + 0.100 s | 0.318738 s | 0.438455 s | PASS |
| Heartbeat p95 ≤ thread + 20 ms | 0.940875 ms | 89.745835 ms | PASS |
| Full GUI improves in same direction | 0.667410 s | 5.044781 s thread | PASS |

### GUI stage durations and export timings

All values below are seconds. `GUI complete` is the full parent publication span and includes small unlisted glue work, so it need not equal the sum of the other stages. Off has no CSV duration/queue; thread has no process ready event. Queue is submit→dispatch (including initial readiness wait), and ready is submit→ready. Neither is inferred from child timestamps.

| Run | Setup | CSV submit | Waveform | DB | History | Analysis enqueue | GUI complete |
| --- | --- | --- | --- | --- | --- | --- | --- |
| representative-01-off | 1.222239 | 0.000031 | 0.004940 | 0.111615 | 0.508207 | 0.061732 | 0.687176 |
| representative-02-thread | 1.095226 | 0.000742 | 0.050245 | 0.093462 | 4.390718 | 0.051468 | 4.587369 |
| representative-03-process | 1.123478 | 0.000214 | 0.006677 | 0.096894 | 0.515053 | 0.048986 | 0.668685 |
| representative-04-thread | 1.206004 | 0.000660 | 0.041357 | 0.102075 | 4.434255 | 0.049331 | 4.628390 |
| representative-05-process | 1.107500 | 0.000231 | 0.006774 | 0.107482 | 0.525863 | 0.046690 | 0.687708 |
| representative-06-off | 1.118186 | 0.000033 | 0.004919 | 0.100202 | 0.486958 | 0.046447 | 0.639259 |
| representative-07-process | 1.103017 | 0.000196 | 0.005689 | 0.097381 | 0.519832 | 0.047450 | 0.671164 |
| representative-08-off | 1.124490 | 0.000029 | 0.005140 | 0.079364 | 0.492840 | 0.051277 | 0.629356 |
| representative-09-thread | 1.103515 | 0.000653 | 0.043270 | 0.109634 | 4.458292 | 0.047231 | 4.659697 |
| full-01-off | 1.116127 | 0.000030 | 0.004855 | 0.097983 | 0.491665 | 0.048083 | 0.643244 |
| full-02-thread | 1.109169 | 0.000614 | 0.051057 | 0.089933 | 4.817365 | 0.085170 | 5.044781 |
| full-03-process | 1.114264 | 0.000183 | 0.006397 | 0.090195 | 0.518801 | 0.051183 | 0.667410 |

| Run | CSV internal | CSV submit→release | Queue | Submit→ready | CSV terminal Qt delay |
| --- | --- | --- | --- | --- | --- |
| representative-01-off | — | 0.000000 | — | — | — |
| representative-02-thread | 4.823938 | 4.824512 | — | — | — |
| representative-03-process | 4.364319 | 4.550050 | 0.180571 | 0.180282 | 0.000193 |
| representative-04-thread | 4.864648 | 4.865178 | — | — | — |
| representative-05-process | 4.418105 | 4.598536 | 0.178457 | 0.178155 | 0.000299 |
| representative-06-off | — | 0.000000 | — | — | — |
| representative-07-process | 4.439673 | 4.623405 | 0.174910 | 0.174548 | 0.000161 |
| representative-08-off | — | 0.000000 | — | — | — |
| representative-09-thread | 4.923548 | 4.924053 | — | — | — |
| full-01-off | — | 0.000000 | — | — | — |
| full-02-thread | 85.425487 | 85.425954 | — | — | — |
| full-03-process | 82.718727 | 82.890918 | 0.169475 | 0.168403 | 0.000233 |

CSV total speed is not an acceptance promise. The stage evidence supports attributing the large thread-group GUI delay mainly to contention during history update; process-group history duration approaches off-group duration. This is an inference from stage timing, not a profiler attribution. No history optimization, format precision change, or algorithm change was made.

## CPU and memory evidence

Each row below is one participating PID. CPU interval is measured-trial delta for the parent and cumulative process CPU since birth for newly spawned children; CSV-thread CPU is included in its parent. The cumulative final column is separately retained native GetProcessTimes evidence. `live_trial_end` means a still-live process snapshot, not final lifetime after shutdown; `exited_process_lifetime` is captured through the retained handle after confirmed exit. Mean one-core percentage = 100 × CPU interval / complete trial duration, so a shorter off trial gives child activity a larger average; 100% means one logical core. This is participating-process load, not whole-machine utilization.

Memory is the maximum **observed live working set** (MiB) across registration, 1-second periodic samples and final live observation; setup is excluded. It is not a continuously measured or OS lifetime peak. Every role has valid live CPU/memory observations and a usable CPU summary. Post-exit instantaneous CPU/memory is correctly null and is not counted as valid live data. No sampling failure or incomplete flag was found. CSV service shutdown occurs after this resource interval; clean-exit is separately checked.

| Run | Role | PID | Trial s | CPU interval s | Cumulative final s | Final CPU source | Mean % one core | Observed WS MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| representative-01-off | parent | 25700 | 3.190299 | 0.906250 | 2.656250 | live_trial_end | 28.406427 | 225.917969 |
| representative-01-off | analysis | 26016 | 3.190299 | 1.875000 | 1.875000 | exited_process_lifetime | 58.771917 | 150.085938 |
| representative-01-off | recording | 25548 | 3.190299 | 0.296875 | 0.296875 | exited_process_lifetime | 9.305554 | 37.593750 |
| representative-02-thread | parent | 9988 | 6.766625 | 5.109375 | 6.812500 | live_trial_end | 75.508467 | 226.511719 |
| representative-02-thread | analysis | 24008 | 6.766625 | 1.906250 | 1.906250 | exited_process_lifetime | 28.171354 | 130.652344 |
| representative-02-thread | recording | 20592 | 6.766625 | 0.343750 | 0.343750 | exited_process_lifetime | 5.080080 | 37.562500 |
| representative-03-process | parent | 23512 | 4.680630 | 0.875000 | 2.578125 | live_trial_end | 18.694063 | 225.890625 |
| representative-03-process | csv | 23584 | 4.680630 | 4.171875 | 4.171875 | live_trial_end | 89.130620 | 32.875000 |
| representative-03-process | analysis | 7804 | 4.680630 | 2.031250 | 2.031250 | exited_process_lifetime | 43.396931 | 203.398438 |
| representative-03-process | recording | 21912 | 4.680630 | 0.296875 | 0.296875 | exited_process_lifetime | 6.342628 | 37.886719 |
| representative-04-thread | parent | 19160 | 6.784111 | 5.203125 | 7.046875 | live_trial_end | 76.695755 | 225.242188 |
| representative-04-thread | analysis | 21516 | 6.784111 | 1.921875 | 1.921875 | exited_process_lifetime | 28.329063 | 163.750000 |
| representative-04-thread | recording | 20012 | 6.784111 | 0.265625 | 0.265625 | exited_process_lifetime | 3.915399 | 37.437500 |
| representative-05-process | parent | 24592 | 4.750774 | 0.859375 | 2.609375 | live_trial_end | 18.089158 | 225.738281 |
| representative-05-process | csv | 24588 | 4.750774 | 4.265625 | 4.265625 | live_trial_end | 89.788001 | 32.804688 |
| representative-05-process | analysis | 17436 | 4.750774 | 2.109375 | 2.109375 | exited_process_lifetime | 44.400660 | 200.800781 |
| representative-05-process | recording | 24732 | 4.750774 | 0.343750 | 0.343750 | exited_process_lifetime | 7.235663 | 37.593750 |
| representative-06-off | parent | 24444 | 2.785654 | 0.812500 | 2.609375 | live_trial_end | 29.167294 | 225.558594 |
| representative-06-off | analysis | 22496 | 2.785654 | 1.921875 | 1.921875 | exited_process_lifetime | 68.991868 | 144.609375 |
| representative-06-off | recording | 22676 | 2.785654 | 0.281250 | 0.281250 | exited_process_lifetime | 10.096371 | 37.488281 |
| representative-07-process | parent | 3988 | 4.750642 | 0.906250 | 2.625000 | live_trial_end | 19.076370 | 226.253906 |
| representative-07-process | csv | 26016 | 4.750642 | 4.296875 | 4.296875 | live_trial_end | 90.448305 | 32.761719 |
| representative-07-process | analysis | 21928 | 4.750642 | 1.984375 | 1.984375 | exited_process_lifetime | 41.770672 | 131.628906 |
| representative-07-process | recording | 25064 | 4.750642 | 0.281250 | 0.281250 | exited_process_lifetime | 5.920253 | 37.597656 |
| representative-08-off | parent | 22228 | 2.819464 | 0.843750 | 2.562500 | live_trial_end | 29.925901 | 225.765625 |
| representative-08-off | analysis | 24264 | 2.819464 | 1.937500 | 1.937500 | exited_process_lifetime | 68.718735 | 117.058594 |
| representative-08-off | recording | 24800 | 2.819464 | 0.312500 | 0.312500 | exited_process_lifetime | 11.083667 | 37.664062 |
| representative-09-thread | parent | 17204 | 6.806667 | 5.171875 | 6.890625 | live_trial_end | 75.982492 | 222.460938 |
| representative-09-thread | analysis | 24092 | 6.806667 | 1.968750 | 1.968750 | exited_process_lifetime | 28.923849 | 129.097656 |
| representative-09-thread | recording | 24404 | 6.806667 | 0.265625 | 0.265625 | exited_process_lifetime | 3.902424 | 37.550781 |
| full-01-off | parent | 21560 | 9.247565 | 0.843750 | 3.671875 | live_trial_end | 9.124024 | 513.750000 |
| full-01-off | analysis | 24316 | 9.247565 | 8.296875 | 8.296875 | exited_process_lifetime | 89.719568 | 1925.714844 |
| full-01-off | recording | 25788 | 9.247565 | 0.265625 | 0.265625 | exited_process_lifetime | 2.872378 | 37.593750 |
| full-02-thread | parent | 2228 | 85.562768 | 83.796875 | 86.562500 | live_trial_end | 97.936144 | 520.160156 |
| full-02-thread | analysis | 23644 | 85.562768 | 8.093750 | 8.093750 | exited_process_lifetime | 9.459430 | 1858.398438 |
| full-02-thread | recording | 21860 | 85.562768 | 0.375000 | 0.375000 | exited_process_lifetime | 0.438275 | 37.648438 |
| full-03-process | parent | 20720 | 83.024371 | 0.968750 | 3.687500 | live_trial_end | 1.166826 | 516.328125 |
| full-03-process | csv | 19372 | 83.024371 | 78.656250 | 78.656250 | live_trial_end | 94.738748 | 32.984375 |
| full-03-process | analysis | 692 | 83.024371 | 8.296875 | 8.296875 | exited_process_lifetime | 9.993301 | 1939.656250 |
| full-03-process | recording | 16040 | 83.024371 | 0.453125 | 0.453125 | exited_process_lifetime | 0.545773 | 37.699219 |

The parent CPU interval was 83.796875 seconds in the full thread trial versus 0.968750 seconds in the full process trial; the latter moved 78.656250 seconds of formatting CPU into its CSV worker. Full analysis CPU stayed close (8.093750–8.296875 seconds), while observed analysis memory was about 1.81–1.90 GiB; shorter representative analysis peak observations vary with 1-second sampling. Full process trials therefore preserve the analysis workload while removing most formatting work from the parent.

Independent [PID absence evidence](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-task10-56c360747fcd4749993a211b0a89ebd9/pid-absence.json>) checked all 39 distinct participating PIDs at 20:44:01 and found none remaining. PIDs can be recycled between sequential runs, so resource rows are keyed by run as well as role/PID.

## Functional and external gates

| Gate | Outcome | Evidence / limit |
| --- | --- | --- |
| Exact full and representative CSV | PASS | All 8 new concurrent CSV files match reference hash, row/byte counts, header and last row; prior full content report also PASS. |
| Source unchanged / clean software exit | PASS | All 12 reports; independent post-run check found all 39 distinct participating PIDs absent. |
| Same actual 200-condition GUI load | PASS | 200 records/grouped conditions/cells, two plots, one DB row in every run. |
| Automatic analysis execution / same judgements | PASS | Both instances completed; zero execution/artifact failures; exact config/source and baseline judgement comparisons. |
| Recording length / drop count, simulated | PASS | 87,759 frames after trim; 44,100 Hz stereo; drop count 0 in all runs. |
| Participating PID resources | PASS | Every role has live observations and trusted cumulative CPU summaries. |
| Capacity 16 / path ownership / shutdown regressions | PASS — prior regression evidence | Task 9 target suite 465 passed, plus later focused tests; not a new hardware or load stress assertion. |
| Real-device capture timing / no additional dropped frames | UNVERIFIED | Requires same workload on target physical audio hardware. |
| Frozen executable diagnostic spawn / packaging | UNVERIFIED | Three PyInstaller builds failed in native dependency collection; no EXE produced. |
| Normal frozen GUI / small physical recording / close | UNVERIFIED | Needs supported release environment/resources, interactive GUI and target hardware. |
| Steady-reuse performance | UNVERIFIED / not claimed | Current process comparisons are first spawn; prior reuse smoke is functional evidence only. |
| Whole-project acceptance | UNVERIFIED | Hardware and frozen gates remain open. |

### Existing source regression and frozen evidence

[Task 9 validation record](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-frozen-e477ea8802fe40a7941812011813a6a7/task9-validation.md>) records the complete prescribed target suite: 465 passed / 17 warnings in 127.82 seconds, followed by the 20-test helper/entrypoint suite after helper corrections. The orchestrator also ran 21 focused checks. This document references that prior source evidence; Task 10 made no source/test changes and did not rerun the regression suite. Unit coverage includes capacity reservations/FIFO/failure recovery, path ownership and asynchronous shutdown.

The same record links actual source `pythonw.exe` launcher/direct entrypoint diagnostics: both exited 0, each had exactly one spawn child, no recursive child, no remaining PID, identical small CSV bytes, unchanged WAV, atomic publication witness, and clean worker/service/IPC exit. These are source-only checks.

Actual windowed PyInstaller attempts (launcher, direct, direct refresh) repeatedly hit native Windows access violations in Torch DLL dependency collection. Verified task-owned stuck collectors were terminated, then each parent exited 1. No executable was produced or run; no dependency was changed. Their logs and process identities remain in the Task 9 evidence directory. This is a build-environment validation block, not a passing frozen test.

### Remaining external acceptance

Use the supported release environment to build both actual entrypoints, then run their `--verify-raw-csv-process` diagnostic and verify independent PID, no recursive windows/children, exact bytes and clean exit. In the normal frozen GUI, perform a real small recording/export/close with required application resources. On the target device repeat the same 200-condition, analysis and next-recording load; confirm actual capture timing, frame/rate/channel/trim equality and zero additional dropped frames. Until these are observed, retain UNVERIFIED for hardware/frozen acceptance.

## Verification execution and scope

`python D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/.worktrees/codex/raw-audio-csv-process/tmp/csv-task10-56c360747fcd4749993a211b0a89ebd9/summarize.py` exited 0 after all 12 trials. It asserted input/config identity, exact CSV fields, 200 rendered cells, analysis completion and baseline equality, simulated recording metadata/drop count, resource validity, clean exit and all command exits; it computed each timing threshold without converting external gaps into passes. All exact commands and individual wall times are in the manifest. Documentation-only work has no RED/GREEN cycle: no product behavior or tests were changed.

Plan adjustments: reused the already completed full offline content validation; invoked repetitions=1 to meet the exact interleaved order; combined baseline references were newly created for analysis comparisons. Final integration/cleanup belongs to the workflow owner and is not performed by this verification slice. Original fixtures/reports remain read-only; generated experiment outputs are retained.

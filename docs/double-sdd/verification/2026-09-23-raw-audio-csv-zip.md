# Raw audio CSV ZIP validation — 2026-09-23

**Task 5 software gates and Task 6 source entrypoint diagnostics: PASS. Overall acceptance: UNVERIFIED.** The final 766-test regression, full member fidelity, simulated recording, analysis, participating-process resource checks, three specified representative timing gates and all four actual source entrypoint/interpreter combinations passed. Task 6 source diagnostics used fresh external output directories after identifying Visual Studio Code contention on ZIP files inside the worktree. Both actual PyInstaller builds reproduced the native Torch DLL collection fault; frozen executable diagnostics, normal frozen GUI operation and real-device capture remain unverified. See the Task 6 evidence and limitations below.

## Evidence and reproducibility

Eight sequential benchmark invocations ran from 2026-09-23T13:36:36+0800 to 2026-09-23T13:40:52+0800. Old means the existing CSV child-process implementation; new means CSV export, ZIP write, complete verification, publish and CSV cleanup in the same child process. Both used `--csv-mode process`; no thread comparison or product skip-ZIP option was used. Representative order was old/new, new/old, old/new; full order was old/new. Each invocation used a new output directory and one first-spawn worker, with one repetition. Root confirmed no concurrent agent tests/builds. Whole-system background load was not independently sampled.

Old read-only worktree: `D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-csv-zip-baseline-20260923`, HEAD `2d2406e41b8c0b0d3662dddc6946d714f4254ac2`; clean before and after. New worktree: `D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip`, HEAD `0e58d61facc16828b00d91f31abd5ed6122496fe`. The only tracked local difference at benchmark start was the root-owned plan. No production files or dependency versions were changed by this validation.

Environment: Windows-10-10.0.19045-SP0; Python `3.12.7 (tags/v3.12.7:0b05ead, Oct  1 2024, 03:06:41) [MSC v.1941 64 bit (AMD64)]`; executable `D:\Python\Python312\python.exe`; 64-bit; CPU AMD Ryzen 5 5600GT with Radeon Graphics         (6 cores/12 logical), physical RAM 16,413,032,448 bytes. D: free space at experiment preparation: 279.84 GiB. `QT_QPA_PLATFORM=offscreen`, 1440×900 widgets. Package versions: `{"numpy": "1.26.4", "soundfile": "0.12.1", "PyQt5": "5.15.11", "PyQt5-Qt5": "5.15.2", "pytest": "9.0.3", "PyInstaller": "6.21.0"}`.

[controller](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/controller.py>), [independent Windows process identity observer](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/win_observer.py>), [commands, exits, code identity, input hashes and PID checks](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/manifest.json>), [input manifest](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/input-manifest.json>), [controller log](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/controller-run.log>), [independent content and report checks](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/summarize.py>), [computed summary](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/summary.json>), [summary execution log](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/summary.log>). All eight benchmark exit codes were **0**. Raw top-level statuses remain `unverified`; derived checks below do not rewrite raw reports.

The initial controller preflight exited 1 because optional `psutil` was absent, before any benchmark was launched. The experiment observer was changed to standard-library `ctypes` Win64 read-only process APIs. [Preserved preflight log](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/controller.log>). No installation or benchmark source change was used.

## Integrated regression

The first exact 28-path union aborted with exit 3 during UI fixture teardown. It was returned to the Task 3 owner for repair and dual review. Qt parentlessness did not imply Python ownership; the repaired fixture only schedules Python-owned ordinary windows, preserving framework-owned objects. The final owner-run, uninstrumented normal `-q` **29-path union passed 766 tests with 17 warnings in 142.70 seconds, exit 0**; it includes all original 28 paths plus the new real-Qt ownership regression. This fresh evidence was inspected for Task 5 and reused at the orchestrator’s explicit direction, rather than rerun unchanged. No filters or `-s` workaround were used.

[final full log](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task3-union-lifetime-99246ee3d5844ec892e8a9a01e5d42b9/final-union.log>), [exit code](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task3-union-lifetime-99246ee3d5844ec892e8a9a01e5d42b9/final-union-exit.txt>), [repair RED/GREEN and diagnostic history](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task3-union-lifetime-99246ee3d5844ec892e8a9a01e5d42b9/evidence-summary.txt>), [initial failing union](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/union.log>). An earlier incomplete-scope preliminary run passed 713 tests/17 warnings; it is retained in `regression.log` and is not the acceptance result.

All pytest basetemps were unique new paths under the checked non-reparse `D:/dongyuan_ditting` anchor. The final warnings comprise pyqtgraph datetime deprecation, matplotlib/pyparsing deprecations, `cgi` deprecation and the deliberately exercised numpy overflow path. The complete final command (cwd: new worktree, `QT_QPA_PLATFORM=offscreen`) was:

```powershell
D:/Python/Python312/python.exe -m pytest unit_test/base/test_audio_record_delete.py unit_test/base/test_audio_record_package.py unit_test/base/test_raw_audio_csv_exporter.py unit_test/base/test_raw_audio_csv_service.py unit_test/base/test_raw_audio_csv_tasks.py unit_test/base/test_raw_audio_csv_worker.py unit_test/base/test_raw_audio_csv_zip.py unit_test/base/test_recording_start_timing.py unit_test/base/test_recording_storage_consistency.py unit_test/test_product_test_project_config_dialog.py unit_test/test_product_test_project_config.py unit_test/test_serial_product_condition_runtime.py unit_test/test_serial_product_round_cleanup.py unit_test/tools/test_raw_audio_csv_benchmark.py unit_test/tools/test_raw_audio_csv_frozen_smoke.py unit_test/tools/test_raw_audio_csv_resources.py unit_test/ui/test_archive_audio_delete_dialog.py unit_test/ui/test_archive_audio_package_dialog.py unit_test/ui/test_raw_audio_csv_admission.py unit_test/ui/test_raw_audio_csv_bridge.py unit_test/ui/test_raw_audio_csv_entrypoints.py unit_test/ui/test_raw_audio_csv_file_ownership.py unit_test/ui/test_raw_audio_csv_recording_integration.py unit_test/ui/test_raw_audio_csv_shutdown.py unit_test/ui/test_recording_gui_finalization.py unit_test/ui/test_recording_process_integration.py unit_test/ui/test_round_reset.py unit_test/ui/test_video_and_round_workflow_integration.py unit_test/ui/test_qt_test_window_cleanup.py -q --basetemp D:\dongyuan_ditting\csv-zip-test-6546755ed3ba47ad842d20c8de6e052f
```

## Inputs and complete CSV fidelity

The existing WAVs and prepared reference JSONs were reused. Their identities and soundfile metadata were checked before the first run; source and reference hashes were checked again after all runs. Both rechecks passed. References already contain matching source/configuration and a prior analysis baseline report, so no off-mode run was needed. No extra giant CSV baseline was generated; the full old-process output is the required comparison run. ZIP validation streams the sole member through EOF without extracting it to disk.

- **representative**: 1,323,000 frames, 44,100 Hz, 2 channels; WAV SHA-256 `d1a9e61d2b12eb20ee447578d33d42ca9e31717d4b4b5fdb2338bfda4aaceff5`. [source WAV](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086/source-evidence/tmp/csv-representative-fixture-28a9eb6e844b45b6bb16e7e7cfc2ca36/representative-30s.wav>), [reference JSON](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-inputs-2c34e66eb6af4eaf932863e08e91cc7f/representative-reference.json>). Reference JSON SHA-256 `882db3e76df642a53dc7330489c9b5bbc08b9ddd9416a86475f7b3b668dd2d3a`. CSV: 1,323,000 data rows, 56,601,607 bytes, SHA-256 `4828d24a2acd305b642d9c4f10a489a703a53fa75abadd5111535f77063845c0`; header `time_s,CH1,CH2`; final row `29.999977324,0.000549316406,0.000579833984`.
- **full**: 26,460,000 frames, 44,100 Hz, 2 channels; WAV SHA-256 `64b8934b433b3f137754e5b0944d7772b2ff584d7c53a544c559305f34da0c10`. [source WAV](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/111/gq001/S004-2/222/audio/wav/gq001_S004-2_222_新端口1_R0001_档位1_20260922-100010.wav>), [reference JSON](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-inputs-2c34e66eb6af4eaf932863e08e91cc7f/full-reference.json>). Reference JSON SHA-256 `25e26e40fa33266cba288b8493c16b53b2b833a272d009a6c93ad1d030225156`. CSV: 26,460,000 data rows, 1,162,232,136 bytes, SHA-256 `facb5196b26f7ac36fdb470fe8850a89106fd94b7602164b40c24eeba38e5b65`; header `time_s,CH1,CH2`; final row `599.999977324,-0.000335693359,-0.000366210938`.

Every old CSV and every new ZIP member independently matched all five reference fields: SHA-256, data rows, bytes, header and last row. New outputs are `raw.csv.zip`, each with exactly one DEFLATE member `raw.csv`, no retained final CSV, no cleanup diagnostics and no task `.tmp` files. The source WAVs remained unchanged. Large-input fidelity is proven for the required 1.162 GB member; ZIP64 API/flag and larger boundary behavior are covered by automated tests, not inferred from this member being over 4 GiB (it is not).

| New run | CSV bytes | ZIP bytes | ZIP/CSV ratio | Space reduction |
| --- | --- | --- | --- | --- |
| representative-02-new | 56601607 | 11427687 | 0.201897 | 79.810% |
| representative-03-new | 56601607 | 11427687 | 0.201897 | 79.810% |
| representative-06-new | 56601607 | 11427687 | 0.201897 | 79.810% |
| full-02-new | 1162232136 | 233334718 | 0.200764 | 79.924% |

Ratio = complete archive file bytes / uncompressed CSV member bytes. This is an observed result for these fixtures, not a promised compression rate.

## Workload and clocks

Each run confirms 200 actual historical records, 200 grouped conditions, 200 populated condition cells, two physical plots, one database row and the same 1440×900 offscreen GUI. The benchmark invokes the existing production publication path and real analysis/recording services. Setup/first paint occurs before the measured GUI publication span. Analysis uses SPL, Z weighting, upper limit 50.0 and physical channels [0, 1]; calibration fallback is 1.0, with existing file calibration taking precedence.

The real recording service uses a deterministic paced simulated backend. Independently inspected `capture/next.wav` files have 87,759 frames, 44,100 Hz and two channels: 88,200 raw target frames minus 441 trim frames. All eight reports show zero drops, no capture failure and matching written/expected frames. This does not prove hardware capture behavior. Both analysis instances completed in every run with zero execution/artifact failures, matching the reference judgement and configuration: representative **OK**, full **NG**. NG is the valid reference classification, not an execution failure.

GUI completion is the measured parent publication span. Capture is accepted request → parent observation/validation of worker-started, including IPC/supervisor delay; it is not queued Qt delivery. Qt started and Qt delivery are separately reported. Heartbeat values are lateness beyond the configured 50 ms timer, not the whole timer interval. Per-run p95 is computed from that run’s raw heartbeat samples, then three p95 values are median-aggregated. Setup, stage clocks and full heartbeat arrays remain in the linked raw reports.

| Run / raw report | GUI s | Capture s | Qt started s | Qt delivery s | Heartbeat p50/p95/max ms |
| --- | --- | --- | --- | --- | --- |
| [representative-01-old](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/representative-01-old/result.json>) | 0.754782 | 0.349787 | 0.350006 | 0.000219 | 0.013900/0.776400/755.300800 |
| [representative-02-new](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/representative-02-new/result.json>) | 0.891389 | 0.348275 | 0.349270 | 0.000995 | 0.027100/0.735740/980.844900 |
| [representative-03-new](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/representative-03-new/result.json>) | 0.747097 | 0.378533 | 0.379068 | 0.000535 | 0.040300/1.187960/749.553300 |
| [representative-04-old](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/representative-04-old/result.json>) | 0.820591 | 0.402500 | 0.403208 | 0.000709 | 0.000000/0.706420/934.172300 |
| [representative-05-old](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/representative-05-old/result.json>) | 0.710431 | 0.340736 | 0.341250 | 0.000515 | 0.027200/0.882980/788.201000 |
| [representative-06-new](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/representative-06-new/result.json>) | 0.737459 | 0.357518 | 0.358582 | 0.001064 | 0.000000/0.994585/828.398400 |
| [full-01-old](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/full-01-old/result.json>) | 0.767708 | 0.358068 | 0.359097 | 0.001028 | 0.001800/0.820430/768.401900 |
| [full-02-new](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/full-02-new/result.json>) | 0.953102 | 0.392832 | 0.393780 | 0.000949 | 0.013500/0.816540/953.852000 |

## Representative timing gates

| Mode (three runs) | Median GUI s | Median capture s | Median per-run p95 ms |
| --- | --- | --- | --- |
| old | 0.754782 | 0.349787 | 0.776400 |
| new | 0.747097 | 0.357518 | 0.994585 |

| Gate/formula | New observed | Limit | Result |
| --- | --- | --- | --- |
| GUI ≤ old×1.1+0.1 s | 0.747097 | 0.930260 | PASS |
| Capture ≤ old×1.1+0.1 s | 0.357518 | 0.484766 | PASS |
| Heartbeat p95 ≤ old+20 ms | 0.994585 | 20.776400 | PASS |

Full 600-second GUI: old 0.767708 s, new 0.953102 s; increased by 0.185394 s (+24.149%).
Full 600-second capture: old 0.358068 s, new 0.392832 s; increased by 0.034764 s (+9.709%).
Full 600-second heartbeat p95: old 0.820430 ms, new 0.816540 ms; decreased by 0.003890 ms (-0.474%).

Full observations report actual direction and magnitude; no additional full-data threshold is invented. One full run per version does not estimate variance.

## Export stages and delivery

All values below are seconds. New `csv_export_seconds` measures the pure exporter call. Old worker `elapsed_seconds` also includes CSV inspection/stat overhead, so it is explicitly an old-worker duration and is not represented as equivalent to the new pure-export field. New elapsed spans CSV plus all ZIP/cleanup work and small orchestration/inspection overhead. Submit→release includes initial spawn/queue and parent processing. Qt terminal delivery is separate; it is never substituted for compression time.

| Run | Old worker elapsed | New pure CSV | ZIP write | ZIP verify | ZIP publish | CSV cleanup | New task elapsed | Submit→release |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| representative-01-old | 4.599385 | — | — | — | — | — | — | 4.949974 |
| representative-02-new | — | 4.819899 | 0.565138 | 0.211850 | 0.000328 | 0.003805 | 5.604200 | 5.951586 |
| representative-03-new | — | 4.626020 | 0.564908 | 0.199805 | 0.000386 | 0.003859 | 5.396605 | 5.590543 |
| representative-04-old | 4.578115 | — | — | — | — | — | — | 4.808713 |
| representative-05-old | 4.381860 | — | — | — | — | — | — | 4.580992 |
| representative-06-new | — | 4.547913 | 0.778993 | 0.203454 | 0.000385 | 0.004410 | 5.537374 | 5.735358 |
| full-01-old | 85.646124 | — | — | — | — | — | — | 85.844403 |
| full-02-new | — | 86.972231 | 9.796012 | 4.009376 | 0.000314 | 0.092828 | 100.872641 | 101.082584 |

| Run | Setup s | Queue s | Submit→ready s | CSV Qt delivery s | Resource interval s |
| --- | --- | --- | --- | --- | --- |
| representative-01-old | 1.302887 | 0.343334 | 0.342058 | 0.000221 | 5.092301 |
| representative-02-new | 1.197035 | 0.337934 | 0.337225 | 0.000188 | 6.076188 |
| representative-03-new | 1.131060 | 0.189285 | 0.188902 | 0.000629 | 5.721599 |
| representative-04-old | 1.158995 | 0.223204 | 0.222906 | 0.000195 | 4.937263 |
| representative-05-old | 1.119624 | 0.187862 | 0.187594 | 0.000361 | 4.703873 |
| representative-06-new | 1.112088 | 0.189028 | 0.188719 | 0.000594 | 5.881259 |
| full-01-old | 1.174650 | 0.189636 | 0.189298 | 0.000360 | 85.989618 |
| full-02-new | 1.461098 | 0.200519 | 0.200028 | unavailable | 101.208276 |

| Run | CSV submit | Waveform | Database | History | Analysis enqueue | GUI complete |
| --- | --- | --- | --- | --- | --- | --- |
| representative-01-old | 0.000220 | 0.005945 | 0.121841 | 0.557297 | 0.068830 | 0.754782 |
| representative-02-new | 0.000217 | 0.006100 | 0.107417 | 0.725025 | 0.051950 | 0.891389 |
| representative-03-new | 0.000224 | 0.006559 | 0.103359 | 0.579202 | 0.056915 | 0.747097 |
| representative-04-old | 0.000212 | 0.005920 | 0.095803 | 0.655017 | 0.062950 | 0.820591 |
| representative-05-old | 0.000253 | 0.007025 | 0.089329 | 0.555863 | 0.057313 | 0.710431 |
| representative-06-new | 0.000269 | 0.007539 | 0.132393 | 0.545232 | 0.051368 | 0.737459 |
| full-01-old | 0.000231 | 0.008198 | 0.092892 | 0.550501 | 0.115215 | 0.767708 |
| full-02-new | 0.000536 | 0.006951 | 0.214381 | 0.663987 | 0.066071 | 0.953102 |

`full-02-new` has an empty `csv.qt_terminal_delivery` list. Its CSV Qt terminal-delivery timing is **UNVERIFIED**, not zero; this does not establish a delivery-latency value or a cause. Worker stage timing, content, GUI/capture clocks and the specified median gates are independently populated. All other runs contain one CSV Qt terminal-delivery observation.

ZIP write and full verification add worker time. These results do not promise faster overall export or unchanged backlog at capacity 16.

## Participating-process resources and exit

Resource sampler interval is 1 second, with registration/final samples. The reported memory figure is the maximum valid live working-set observation during the measured interval, excluding GUI setup and exited samples; it is not the absolute lifetime peak. CPU values use valid live baseline and final cumulative native counters. Parent CPU covers the measured interval; newly spawned children cover their process lifetime. The CSV worker is still live at the measured trial end and then shuts down normally; its final CPU source is `live_trial_end`. Recording/analysis exits retain their final cumulative CPU counters through open native handles (`exited_process_lifetime`). Post-exit instantaneous CPU and working-set fields are null, never replaced with stale live samples.

All four roles (parent, CSV, recording, analysis) had live CPU/memory observations, final CPU counters and complete resource records in every run. The Win64 `WindowsProcessMetrics` class AST is unchanged between old and new, including HANDLE/pointer signatures. An independent standard-library Windows observer sampled process identities every 250 ms, retaining native creation FILETIME ticks, then checked all observed/reported PIDs after each invocation exited. Every original process was absent; PID reuse would be distinguished by creation time. Raw identity checks are in the manifest.

| Run | Role / PID | CPU s | Final cumulative CPU s | Final source | Maximum observed live MiB |
| --- | --- | --- | --- | --- | --- |
| representative-01-old | parent / 25692 | 0.921875 | 2.828125 | live_trial_end | 229.253906 |
| representative-01-old | csv / 22108 | 4.406250 | 4.406250 | live_trial_end | 32.757812 |
| representative-01-old | analysis / 20500 | 2.312500 | 2.312500 | exited_process_lifetime | 132.937500 |
| representative-01-old | recording / 19692 | 0.328125 | 0.328125 | exited_process_lifetime | 37.589844 |
| representative-02-new | parent / 22192 | 1.140625 | 2.906250 | live_trial_end | 227.257812 |
| representative-02-new | csv / 14180 | 5.265625 | 5.265625 | live_trial_end | 34.085938 |
| representative-02-new | analysis / 14000 | 2.281250 | 2.281250 | exited_process_lifetime | 152.632812 |
| representative-02-new | recording / 680 | 0.296875 | 0.296875 | exited_process_lifetime | 37.914062 |
| representative-03-new | parent / 13960 | 1.062500 | 2.906250 | live_trial_end | 226.781250 |
| representative-03-new | csv / 7952 | 4.984375 | 4.984375 | live_trial_end | 34.964844 |
| representative-03-new | analysis / 10260 | 2.296875 | 2.296875 | exited_process_lifetime | 140.917969 |
| representative-03-new | recording / 14764 | 0.437500 | 0.437500 | exited_process_lifetime | 38.039062 |
| representative-04-old | parent / 27180 | 1.046875 | 3.000000 | live_trial_end | 228.847656 |
| representative-04-old | csv / 5956 | 4.406250 | 4.406250 | live_trial_end | 32.910156 |
| representative-04-old | analysis / 15480 | 2.187500 | 2.187500 | exited_process_lifetime | 138.703125 |
| representative-04-old | recording / 25948 | 0.359375 | 0.359375 | exited_process_lifetime | 37.578125 |
| representative-05-old | parent / 19688 | 0.953125 | 2.718750 | live_trial_end | 226.234375 |
| representative-05-old | csv / 27144 | 4.140625 | 4.140625 | live_trial_end | 32.757812 |
| representative-05-old | analysis / 26620 | 2.125000 | 2.125000 | exited_process_lifetime | 150.878906 |
| representative-05-old | recording / 21500 | 0.312500 | 0.312500 | exited_process_lifetime | 37.554688 |
| representative-06-new | parent / 9684 | 0.937500 | 2.703125 | live_trial_end | 225.800781 |
| representative-06-new | csv / 11352 | 4.953125 | 4.953125 | live_trial_end | 34.832031 |
| representative-06-new | analysis / 25008 | 2.125000 | 2.125000 | exited_process_lifetime | 150.589844 |
| representative-06-new | recording / 22228 | 0.312500 | 0.312500 | exited_process_lifetime | 37.804688 |
| full-01-old | parent / 17588 | 1.281250 | 4.203125 | live_trial_end | 514.621094 |
| full-01-old | csv / 10940 | 80.578125 | 80.578125 | live_trial_end | 32.949219 |
| full-01-old | analysis / 27416 | 9.218750 | 9.218750 | exited_process_lifetime | 2122.410156 |
| full-01-old | recording / 20876 | 0.531250 | 0.531250 | exited_process_lifetime | 37.652344 |
| full-02-new | parent / 14184 | 1.328125 | 4.562500 | live_trial_end | 514.335938 |
| full-02-new | csv / 12788 | 96.234375 | 96.234375 | live_trial_end | 37.863281 |
| full-02-new | analysis / 22664 | 11.203125 | 11.203125 | exited_process_lifetime | 2099.683594 |
| full-02-new | recording / 26128 | 0.484375 | 0.484375 | exited_process_lifetime | 37.996094 |

## Exact benchmark commands

All commands used `QT_QPA_PLATFORM=offscreen` and `PYTHONUNBUFFERED=1`. Full cwd and argv arrays, per-run exit 0, command wall time, log/report paths and process identities are preserved in the manifest. Command wall time includes CLI startup, content inspection and shutdown; it is not the worker export duration.

**representative-01-old** — cwd `D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-csv-zip-baseline-20260923`; exit 0; wall 8.590640 s. [stdout/stderr log](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/representative-01-old.log>)

```powershell
& 'D:\Python\Python312\python.exe' 'tools/benchmark_raw_audio_csv_process.py' '--mode' 'concurrency' '--csv-mode' 'process' '--source-wav' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\tmp\csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086\source-evidence\tmp\csv-representative-fixture-28a9eb6e844b45b6bb16e7e7cfc2ca36\representative-30s.wav' '--reference-json' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\tmp\csv-zip-inputs-2c34e66eb6af4eaf932863e08e91cc7f\representative-reference.json' '--profile' 'representative' '--conditions' '200' '--repetitions' '1' '--work-dir' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip\tmp\task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5\representative-01-old' '--report' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip\tmp\task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5\representative-01-old\result.json'
```

**representative-02-new** — cwd `D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip`; exit 0; wall 9.375216 s. [stdout/stderr log](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/representative-02-new.log>)

```powershell
& 'D:\Python\Python312\python.exe' 'tools/benchmark_raw_audio_csv_process.py' '--mode' 'concurrency' '--csv-mode' 'process' '--source-wav' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\tmp\csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086\source-evidence\tmp\csv-representative-fixture-28a9eb6e844b45b6bb16e7e7cfc2ca36\representative-30s.wav' '--reference-json' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\tmp\csv-zip-inputs-2c34e66eb6af4eaf932863e08e91cc7f\representative-reference.json' '--profile' 'representative' '--conditions' '200' '--repetitions' '1' '--work-dir' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip\tmp\task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5\representative-02-new' '--report' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip\tmp\task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5\representative-02-new\result.json'
```

**representative-03-new** — cwd `D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip`; exit 0; wall 9.361086 s. [stdout/stderr log](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/representative-03-new.log>)

```powershell
& 'D:\Python\Python312\python.exe' 'tools/benchmark_raw_audio_csv_process.py' '--mode' 'concurrency' '--csv-mode' 'process' '--source-wav' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\tmp\csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086\source-evidence\tmp\csv-representative-fixture-28a9eb6e844b45b6bb16e7e7cfc2ca36\representative-30s.wav' '--reference-json' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\tmp\csv-zip-inputs-2c34e66eb6af4eaf932863e08e91cc7f\representative-reference.json' '--profile' 'representative' '--conditions' '200' '--repetitions' '1' '--work-dir' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip\tmp\task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5\representative-03-new' '--report' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip\tmp\task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5\representative-03-new\result.json'
```

**representative-04-old** — cwd `D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-csv-zip-baseline-20260923`; exit 0; wall 8.068220 s. [stdout/stderr log](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/representative-04-old.log>)

```powershell
& 'D:\Python\Python312\python.exe' 'tools/benchmark_raw_audio_csv_process.py' '--mode' 'concurrency' '--csv-mode' 'process' '--source-wav' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\tmp\csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086\source-evidence\tmp\csv-representative-fixture-28a9eb6e844b45b6bb16e7e7cfc2ca36\representative-30s.wav' '--reference-json' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\tmp\csv-zip-inputs-2c34e66eb6af4eaf932863e08e91cc7f\representative-reference.json' '--profile' 'representative' '--conditions' '200' '--repetitions' '1' '--work-dir' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip\tmp\task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5\representative-04-old' '--report' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip\tmp\task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5\representative-04-old\result.json'
```

**representative-05-old** — cwd `D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-csv-zip-baseline-20260923`; exit 0; wall 7.528928 s. [stdout/stderr log](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/representative-05-old.log>)

```powershell
& 'D:\Python\Python312\python.exe' 'tools/benchmark_raw_audio_csv_process.py' '--mode' 'concurrency' '--csv-mode' 'process' '--source-wav' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\tmp\csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086\source-evidence\tmp\csv-representative-fixture-28a9eb6e844b45b6bb16e7e7cfc2ca36\representative-30s.wav' '--reference-json' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\tmp\csv-zip-inputs-2c34e66eb6af4eaf932863e08e91cc7f\representative-reference.json' '--profile' 'representative' '--conditions' '200' '--repetitions' '1' '--work-dir' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip\tmp\task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5\representative-05-old' '--report' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip\tmp\task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5\representative-05-old\result.json'
```

**representative-06-new** — cwd `D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip`; exit 0; wall 8.824222 s. [stdout/stderr log](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/representative-06-new.log>)

```powershell
& 'D:\Python\Python312\python.exe' 'tools/benchmark_raw_audio_csv_process.py' '--mode' 'concurrency' '--csv-mode' 'process' '--source-wav' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\tmp\csv-history-cleanup-9d84fc25bc7d444caf705f59ed93f086\source-evidence\tmp\csv-representative-fixture-28a9eb6e844b45b6bb16e7e7cfc2ca36\representative-30s.wav' '--reference-json' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\tmp\csv-zip-inputs-2c34e66eb6af4eaf932863e08e91cc7f\representative-reference.json' '--profile' 'representative' '--conditions' '200' '--repetitions' '1' '--work-dir' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip\tmp\task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5\representative-06-new' '--report' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip\tmp\task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5\representative-06-new\result.json'
```

**full-01-old** — cwd `D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-csv-zip-baseline-20260923`; exit 0; wall 92.705936 s. [stdout/stderr log](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/full-01-old.log>)

```powershell
& 'D:\Python\Python312\python.exe' 'tools/benchmark_raw_audio_csv_process.py' '--mode' 'concurrency' '--csv-mode' 'process' '--source-wav' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\111\gq001\S004-2\222\audio\wav\gq001_S004-2_222_新端口1_R0001_档位1_20260922-100010.wav' '--reference-json' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\tmp\csv-zip-inputs-2c34e66eb6af4eaf932863e08e91cc7f\full-reference.json' '--profile' 'full' '--conditions' '200' '--repetitions' '1' '--work-dir' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip\tmp\task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5\full-01-old' '--report' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip\tmp\task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5\full-01-old\result.json'
```

**full-02-new** — cwd `D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip`; exit 0; wall 110.923948 s. [stdout/stderr log](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5/full-02-new.log>)

```powershell
& 'D:\Python\Python312\python.exe' 'tools/benchmark_raw_audio_csv_process.py' '--mode' 'concurrency' '--csv-mode' 'process' '--source-wav' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\111\gq001\S004-2\222\audio\wav\gq001_S004-2_222_新端口1_R0001_档位1_20260922-100010.wav' '--reference-json' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\tmp\csv-zip-inputs-2c34e66eb6af4eaf932863e08e91cc7f\full-reference.json' '--profile' 'full' '--conditions' '200' '--repetitions' '1' '--work-dir' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip\tmp\task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5\full-02-new' '--report' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip\tmp\task5-csv-zip-42cc821b26a24dc3865e25e1181c6fa5\full-02-new\result.json'
```

## Remaining acceptance boundaries

- Hardware capture, real-device loss/overflow behavior and hardware GUI interaction: **UNVERIFIED**; simulation cannot substitute.
- Final-revision source entrypoint checks via both `python.exe` and `pythonw.exe`: **PASS in fresh external output directories**, with internal worktree ZIP contention and failed attempts preserved below.
- Frozen executable diagnostics and normal frozen GUI operation: **UNVERIFIED**. Both Task 6 builds were actually attempted and reproduced the Torch native DLL collection fault; no EXE was produced.
- Local applyback, evidence archival, worktree/history cleanup and final all-diff review: outside Task 5; this document’s current links must be preserved or updated during Task 6.

This task changes only the verification document and ignored experiment material. All output creation was isolated; source WAV/reference inputs, the read-only old-code worktree and root-owned plan/runtime metadata were preserved.


## Task 6 release and final source validation

These checks ran against HEAD `1f94c8519ad867fd4e7620c6937514154879c01e`, with only the root-owned plan modified. No production code, dependency versions, unrelated launcher spec or user application processes were changed. The build attempts overlapped each other and later source diagnostics; these are correctness checks, not additional performance measurements. Processes were launched with `CREATE_NO_WINDOW`, and `QT_QPA_PLATFORM=offscreen` remained set.

### Independent final regression

The orchestrator ran the complete sorted, normal `-q` 29-path suite again: **766 passed, 17 warnings in 203.52 seconds, exit 0**. The actual command, log and exit file were read for this report; this release-validation agent did not rerun that suite. The unique external basetemp was `D:/dongyuan_ditting/csv-zip-final-root-240ee888c09743cc94115ed252480cd1`.

[exact regression command](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/root-final-verification-2bb54ac9c3bd4d1e8faee4e3b8d33578/command.txt>), [final regression log](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/root-final-verification-2bb54ac9c3bd4d1e8faee4e3b8d33578/pytest.log>), [observed exit code](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/root-final-verification-2bb54ac9c3bd4d1e8faee4e3b8d33578/exit.txt>).

### Four actual source entrypoints

The existing non-reparse `D:/dongyuan_ditting` anchor was checked before creating a new UUID output root. Each invocation below used a separate newly created directory and absolute report path, and its process was waited to completion. Every invocation returned **exit 0**, report `status=pass`, 8,193 frames, 303,749 member bytes and a 123,391-byte ZIP. The sole member is `原始 output.csv`, compressed with DEFLATE; its SHA-256 is `80dd677d825ea06607fcaf7cf70a039fa7267adeea4734411b2223c08610dbbf`, and its bytes independently matched the pure-export reference. The final CSV is absent; the old-publication hardlink witness is intact with a different file identity from the new ZIP; no task temporary files remain. Reports also confirm unchanged WAV, no GUI module imports and clean service/IPC shutdown.

A separate Win64 Toolhelp/GetProcessTimes observer sampled recursive descendants every 20 ms and retained process creation FILETIME identities. It independently observed each reported worker, found one application worker per invocation, no recursive application child and no surviving original process identity after exit. Hidden `conhost.exe` children are recorded as console hosts, not recursive application instances. Finite polling does not prove absence of arbitrarily short-lived processes between samples.

| Interpreter | Entry | Parent / worker PID | Result |
| --- | --- | --- | --- |
| python.exe | main_window.py | 21284 / 14804 | [PASS, exit 0](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/external-source/external-python-main_window/report.json>) |
| python.exe | main_window_Launcher.py | 8536 / 25380 | [PASS, exit 0](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/external-source/external-python-main_window_Launcher/report.json>) |
| pythonw.exe | main_window.py | 10260 / 22544 | [PASS, exit 0](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/external-source/external-pythonw-main_window/report.json>) |
| pythonw.exe | main_window_Launcher.py | 14008 / 10520 | [PASS, exit 0](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/external-source/external-pythonw-main_window_Launcher/report.json>) |

[four exact commands and independent checks](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task6-release-d0212d5bf4474221b6b66a5eabe4f8f9/external-source-manifest.json>), [controller](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task6-release-d0212d5bf4474221b6b66a5eabe4f8f9/external-controller.py>), [controller log](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task6-release-d0212d5bf4474221b6b66a5eabe4f8f9/external-controller.log>), [Win64 observer](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task6-release-d0212d5bf4474221b6b66a5eabe4f8f9/win_observer.py>).

Exact source commands (cwd: feature worktree):

```powershell
& 'D:\Python\Python312\python.exe' 'main_window.py' '--verify-raw-csv-process' 'D:\dongyuan_ditting\csv-zip-release-16dc2a3d38df45c296ed395a697fe2b1\external-python-main_window\report.json'
& 'D:\Python\Python312\python.exe' 'main_window_Launcher.py' '--verify-raw-csv-process' 'D:\dongyuan_ditting\csv-zip-release-16dc2a3d38df45c296ed395a697fe2b1\external-python-main_window_Launcher\report.json'
& 'D:\Python\Python312\pythonw.exe' 'main_window.py' '--verify-raw-csv-process' 'D:\dongyuan_ditting\csv-zip-release-16dc2a3d38df45c296ed395a697fe2b1\external-pythonw-main_window\report.json'
& 'D:\Python\Python312\pythonw.exe' 'main_window_Launcher.py' '--verify-raw-csv-process' 'D:\dongyuan_ditting\csv-zip-release-16dc2a3d38df45c296ed395a697fe2b1\external-pythonw-main_window_Launcher\report.json'
```

### Internal-directory contention investigation

Before the external matrix, the direct entry failed five times (initial python, fresh python/pythonw, then shorter-path python/pythonw) with `PermissionError [WinError 5]` at `zip_publish`; Launcher passed with both interpreters in the internal directory. Shorter paths did not fix the issue: direct ZIP target lengths were 206–223 characters and temporary paths 234–251 characters. Every failed run retained the complete 303,749-byte CSV matching the reference and the original ZIP/hardlink identity and contents; cleanup left no temporary files, and reports recorded clean exit. The failed runs are not counted as passes.

The orchestrator/explorer then reproduced the same internal-directory replacement failure using only standard-library ZIP/hardlink operations with no application imports. Internal delayed replacement failed until 8.2–9.0 seconds in the probe, while external delayed replacement and immediate internal replacement passed. A native Windows Restart Manager query during the reproducing probe identified **Visual Studio Code PID 20468** holding the ZIP. This isolates an external filesystem contention mechanism and supports the observed timing difference; it does not establish an application ZIP algorithm defect. Visual Studio Code was not stopped and no settings, automatic retries, product delays or production code were changed. The accepted source matrix above uses new external locations; internal paths remain subject to this observed external contention.

[delay control](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/zip-delay-probe-4617a472f4a64966b891781f467115bf/report.json>), [identified external holder](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/zip-lock-holder-b0b304e06155404a8bea4d3466f3f36b/report.json>), [initial failed report](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task6-release-d0212d5bf4474221b6b66a5eabe4f8f9/python-main_window/report.json>), [fresh internal matrix](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task6-release-d0212d5bf4474221b6b66a5eabe4f8f9/source-manifest.json>), [short internal matrix](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task6-release-d0212d5bf4474221b6b66a5eabe4f8f9/short-source-manifest.json>), [failed-output inspection](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task6-release-d0212d5bf4474221b6b66a5eabe4f8f9/failed-output-inspection.json>), [initial controller failure](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task6-release-d0212d5bf4474221b6b66a5eabe4f8f9/controller.log>). The initial controller expected an archive field after a failed report and exited with `KeyError`; subsequent observers preserved failed reports and process identity evidence explicitly.

### Actual PyInstaller attempts and frozen boundary

Both entrypoints were built using the required unmodified dependency collection and separate new dist/build/spec directories. Each build started at approximately 13:49:34 +0800 and reached dynamic-library collection. Both then repeatedly emitted `Windows fatal exception: access violation` at `torch.__init__._load_dll_libraries` (line 263), the same native loader failure class as the earlier environment evidence. Neither advanced past the last INFO stage, `Looking for dynamic libraries` at approximately 424.39 seconds. These are actual build failures, not an inference from source diagnostic success.

At 14:01:25 +0800, only the two task-owned isolated collector processes were terminated after verifying command lines/parent relationships via CIM and matching native creation FILETIME on the same process handle used for termination. No broad process-name kill was used. The parents then ended with `PyInstaller.isolated._parent.SubprocessDiedError`; this followed intentional bounded collector termination, not spontaneous parent completion.

| Entry | Build parent | Verified collector | Native faults before stop | Parent exit code |
| --- | --- | --- | --- | --- |
| main_window_Launcher.py | 26804 | 20444 | 64 | unavailable |
| main_window.py | 26064 | 13892 | 58 | unavailable |

**Exit-code evidence limitation:** while processes were ending, the independent observer encountered `OpenProcess` WinError 5 and its controller exited 1 before recording the build parent return codes. Both stored parent exit codes remain null; the controller's exit 1 is not a build exit code. A later native query returned error 87 for both absent parents, so their exact exit codes could not be recovered. No new build was run to hide this gap. A separate final identity inspection found all 27 observed Launcher-build identities and all 25 direct-build identities gone (including original creation times), and found no EXE under either dist directory.

[exact build commands and observed identities](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task6-release-d0212d5bf4474221b6b66a5eabe4f8f9/build-manifest.json>), [direct build log](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task6-release-d0212d5bf4474221b6b66a5eabe4f8f9/build-main_window/build.log>), [Launcher build log](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task6-release-d0212d5bf4474221b6b66a5eabe4f8f9/build-main_window_Launcher/build.log>), [CIM identity evidence](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task6-release-d0212d5bf4474221b6b66a5eabe4f8f9/build-cim-1401.json>), [bounded collector termination](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task6-release-d0212d5bf4474221b6b66a5eabe4f8f9/collector-termination.json>), [observer race failure](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task6-release-d0212d5bf4474221b6b66a5eabe4f8f9/controller-fresh.log>), [unrecoverable parent exit query](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task6-release-d0212d5bf4474221b6b66a5eabe4f8f9/parent-exit-query.json>), [final identity and EXE inspection](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task6-release-d0212d5bf4474221b6b66a5eabe4f8f9/build-final-inspection.json>), [final inspection script](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence/tmp/task6-release-d0212d5bf4474221b6b66a5eabe4f8f9/final-check.py>).

Exact attempted build commands (cwd: feature worktree):

```powershell
& 'D:\Python\Python312\python.exe' '-m' 'PyInstaller' '--noconfirm' '--clean' '--onedir' '--windowed' '--name' 'raw-csv-zip-main_window_Launcher' '--distpath' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip\tmp\task6-release-d0212d5bf4474221b6b66a5eabe4f8f9\build-main_window_Launcher\dist' '--workpath' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip\tmp\task6-release-d0212d5bf4474221b6b66a5eabe4f8f9\build-main_window_Launcher\build' '--specpath' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip\tmp\task6-release-d0212d5bf4474221b6b66a5eabe4f8f9\build-main_window_Launcher\spec' 'main_window_Launcher.py'
& 'D:\Python\Python312\python.exe' '-m' 'PyInstaller' '--noconfirm' '--clean' '--onedir' '--windowed' '--name' 'raw-csv-zip-main_window' '--distpath' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip\tmp\task6-release-d0212d5bf4474221b6b66a5eabe4f8f9\build-main_window\dist' '--workpath' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip\tmp\task6-release-d0212d5bf4474221b6b66a5eabe4f8f9\build-main_window\build' '--specpath' 'D:\dongyuan_ditting\SpeakerAnomalyDetection-MW2\.worktrees\codex\raw-audio-csv-zip\tmp\task6-release-d0212d5bf4474221b6b66a5eabe4f8f9\build-main_window\spec' 'main_window.py'
```

No frozen EXE diagnostic could be run because neither build produced an EXE. **Frozen runtime, ordinary frozen GUI recording/close, and real-device capture remain UNVERIFIED.** Native CUA is disabled and real recording hardware was not exercised; software simulation does not replace that acceptance. These external validation boundaries do not constitute local applyback/history-cleanup completion. The orchestrator owns final paired review, evidence archival/link rewriting and local delivery.

## Evidence archival for local delivery

Evidence was copied byte-for-byte into `D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/source-evidence` before temporary worktree removal. Links above use the archived locations; raw report paths and command cwd fields remain historical execution records. Map the former feature-worktree prefix to this archive root, and the external release-report prefix to its `external-source/` directory when following raw paths. Generated build/cache directories are excluded; logs, manifests, outputs and verification data are preserved. [Archive hashes](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/archive-manifest.json>), [external-source hashes](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/external-archive-manifest.json>).

## Final review and local integration

Final integrated specification and engineering-quality reviews both passed. After applying the ZIP patch to the original local checkout, the ZIP helper, entrypoint and concurrent window-restoration regression selection completed with **55 passed, 1 skipped, 16 warnings, exit 0**. The skipped native-window test requires the Windows Qt backend and was not exercised by the offscreen run. The separate full 29-path feature regression remains **766 passed**. Existing window-restoration changes and all unrelated local files were preserved; the index remains empty. [Local test command](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/local-command.txt>), [local test log](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/local-pytest.log>), [exit code](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/local-exit.txt>).

Local apply-back and cleanup completed: all 38 ZIP task paths were delivered, original local changes preserved, and final contents remain uncommitted with an empty index. Main branch was restored to its recorded permanent base after checkpoint-aware content preservation. The ZIP feature branch and its feature/comparison worktrees were removed. The unrelated `codex/main-window-restore-white-flash` branch still references the older shared checkpoints; its history was intentionally preserved. [Delivery verification](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/apply-result.json>), [cleanup verification](<D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2/tmp/csv-zip-delivery-2502141e50fd460fb0402f3c981b0876/cleanup-result.json>).

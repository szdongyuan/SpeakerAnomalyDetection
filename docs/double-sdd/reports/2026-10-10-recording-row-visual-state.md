# Recording row visual state: verification and callback benchmark

Date: 2026-10-10. Baseline: `e8d4b3bab29a2778faa18e64476cfbf37c7bf342`. Current production version is identified by the source SHA-256 values below; final changes are delivered uncommitted in the primary checkout.

The real mark-mode 200-row/100-completed unchanged callback median fell from **4616.951 ms to 19.141 ms (99.59% reduction)**. The 90% reference goal was met. All 12 N/K/mode scenarios and all unchanged/one-changed/five-changed stages passed semantic equivalence and structural gates. No production files changed in Task 2.

## Reproducible method

- Environment: Python 3.12.7, Qt 5.15.2, PyQt 5.15.11, Windows-10-10.0.19045-SP0. Both versions use the same interpreter and environment.
- Each source runs in a separate subprocess, baseline first and current second, without overlap. Each N/K/mode scenario executes one warmup sequence and three measured sequences. Each sequence measures unchanged, one changed label (pending to OK), then five changed labels (one OK to NG and four pending to NG). Test mode uses the identical input sequence as a branch control; group synchronization deliberately does not change row states in that mode.
- The Host inherits the production analysis mixin. Timing covers the actual `_update_current_recent_session_result` → `_update_recent_session` → `_refresh_manual_product_condition_results_from_group` → real `MotorResultPanel` path, including synchronous final-result work. No callback is replaced or batched.
- Configuration uses the real public format, groups of 20 rows, two analysis channels, and SPL metadata. The panel is shown with `WA_DontShowOnScreen` using the offscreen Qt platform. One populated two-channel OK row, selected/viewed state, and visible details are included in the snapshots.
- K−1 completed keys are seeded as accumulated prior recordings; the final key is added by the real completion method. Twenty-one real history appends prove eviction to 20 retained records while K completed keys survive. Group raw results are asserted to contain K keys and actual panel rows are asserted to equal N. Synthetic paths are metadata only; no WAV, device, analysis job, or business database is used.
- Before each measured sequence, the five changing rows are restored through real public panel APIs outside timing. The neutral reset clears completed-analysis placeholder protection. History labels are reset in memory; the measured callback then supplies the current record update. Deferred deletes, event processing and pending paints are drained before and after timing. Widget construction, fixture preparation, snapshots, and event draining are excluded.
- `perf_counter_ns` measures wall time; `process_time_ns` measures process CPU. Counter wrappers run inside both timing boundaries. Windows process CPU samples are quantized, so small callbacks can report 0 ms. Wall time is used for the performance comparison.
- Counters wrap real row-button/result-label `setStyleSheet`, `setProperty`, `update`, and explicit Python `QStyle.polish/unpolish` calls. They do not count implicit internal C++ polishing caused by `setStyleSheet`; baseline polish=0 therefore does not mean no Qt style work. Summary/title/detail styling is outside the row-only count scope.
- Semantic equality compares every row result/tone, channel results/counts/completion/runtime details, displayed row texts and visibility; selection/viewed/port/detail ownership; port/round/final/automatic summaries; complete history order/records/config snapshots; current session; completed keys/manual results and collected group content. Time is fixed at 2026-10-10 12:00:00 and `PYTHONHASHSEED=0`. QSS strings and version-specific dynamic properties are intentionally not compared.
- The baseline is an existing `git archive` export. The script validates all 274 tracked Python files under `ui`, `base`, and `consts` against the baseline commit, allowing only Windows LF/CRLF conversion. The initial integrity check exposed that conversion; source content was not edited.

## Commands

To reproduce after delivery, run from `D:/dongyuan_ditting/SpeakerAnomalyDetection-MW2`:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
$env:QT_QPA_FONTDIR='C:/Windows/Fonts'
$env:TEMP='D:/row-visual-state-evidence-20261010'
$env:TMP=$env:TEMP
D:/Python/Python312/python.exe -B unit_test/ui/benchmark_motor_row_visual_state.py --baseline-root C:/Users/Administrator/AppData/Local/Temp/row-visual-state-20261010-avuqv10q/baseline --current-root . --output-dir D:/row-visual-state-evidence-20261010/benchmark/comparison
```

Use a new output directory for reruns. The parent forces `PYTHONDONTWRITEBYTECODE=1` and `PYTHONHASHSEED=0`; exact subprocess argv is saved in each `command.json`. Optional `--rows 200 --completed 100 --mode mark` selects one scenario. To create a fresh baseline export:

```powershell
git archive --format=zip --output=D:/row-visual-state-evidence-20261010/baseline-repro.zip e8d4b3bab29a2778faa18e64476cfbf37c7bf342
Expand-Archive -LiteralPath D:/row-visual-state-evidence-20261010/baseline-repro.zip -DestinationPath D:/row-visual-state-evidence-20261010/baseline-repro
```

## Raw wall samples and medians

All values are milliseconds; triples are the three chronological measured samples, rounded here to three decimal places. Raw JSON retains full precision and the warmup. `one` and `five` describe changed inputs; test controls have zero changed row states.

| Mode/N/K | Stage | Baseline samples | Current samples | Baseline median | Current median | Reduction |
|---|---|---|---|---:|---:|---:|
| mark-100-20 | unchanged | 447.478 / 458.796 / 450.901 | 3.438 / 3.361 / 3.555 | 450.901 | 3.438 | 99.24% |
| mark-100-20 | one | 457.222 / 459.120 / 447.204 | 3.645 / 4.191 / 3.957 | 457.222 | 3.957 | 99.13% |
| mark-100-20 | five | 452.335 / 455.215 / 447.185 | 4.306 / 4.033 / 5.582 | 452.335 | 4.306 | 99.05% |
| mark-100-60 | unchanged | 1361.844 / 1360.984 / 1382.955 | 10.069 / 9.963 / 10.258 | 1361.844 | 10.069 | 99.26% |
| mark-100-60 | one | 1353.927 / 1383.754 / 1346.967 | 9.796 / 10.637 / 10.831 | 1353.927 | 10.637 | 99.21% |
| mark-100-60 | five | 1359.432 / 1386.622 / 1359.660 | 11.298 / 11.095 / 11.331 | 1359.660 | 11.298 | 99.17% |
| mark-100-100 | unchanged | 2269.408 / 2267.635 / 2311.722 | 16.440 / 16.803 / 17.160 | 2269.408 | 16.803 | 99.26% |
| mark-100-100 | one | 2363.354 / 2290.137 / 2257.793 | 18.404 / 17.294 / 17.294 | 2290.137 | 17.294 | 99.24% |
| mark-100-100 | five | 2311.632 / 2276.196 / 2336.441 | 19.139 / 18.438 / 17.600 | 2311.632 | 18.438 | 99.20% |
| mark-200-20 | unchanged | 900.910 / 909.798 / 892.350 | 4.008 / 4.103 / 4.097 | 900.910 | 4.097 | 99.55% |
| mark-200-20 | one | 914.937 / 897.099 / 923.737 | 4.096 / 4.333 / 4.259 | 914.937 | 4.259 | 99.53% |
| mark-200-20 | five | 898.084 / 902.199 / 902.601 | 4.958 / 4.795 / 4.747 | 902.199 | 4.795 | 99.47% |
| mark-200-60 | unchanged | 2726.408 / 2686.455 / 2677.110 | 12.405 / 11.630 / 11.400 | 2686.455 | 11.630 | 99.57% |
| mark-200-60 | one | 2686.370 / 2692.108 / 2697.777 | 11.898 / 12.158 / 12.806 | 2692.108 | 12.158 | 99.55% |
| mark-200-60 | five | 2703.917 / 2669.756 / 2672.804 | 12.486 / 13.021 / 12.738 | 2672.804 | 12.738 | 99.52% |
| mark-200-100 | unchanged | 4616.951 / 4614.637 / 4721.560 | 19.141 / 19.049 / 19.154 | 4616.951 | 19.141 | 99.59% |
| mark-200-100 | one | 4590.629 / 4588.562 / 4710.921 | 19.745 / 19.503 / 19.871 | 4590.629 | 19.745 | 99.57% |
| mark-200-100 | five | 4638.883 / 4667.549 / 4611.108 | 20.151 / 20.780 / 20.448 | 4638.883 | 20.448 | 99.56% |
| test-100-20 | unchanged | 0.214 / 0.399 / 0.323 | 0.177 / 0.163 / 0.166 | 0.323 | 0.166 | 48.42% |
| test-100-20 | one | 0.131 / 0.173 / 0.170 | 0.139 / 0.138 / 0.127 | 0.170 | 0.138 | 18.76% |
| test-100-20 | five | 0.117 / 0.139 / 0.125 | 0.181 / 0.126 / 0.115 | 0.125 | 0.126 | -0.88% |
| test-100-60 | unchanged | 0.268 / 0.245 / 0.242 | 0.146 / 0.159 / 0.181 | 0.245 | 0.159 | 35.31% |
| test-100-60 | one | 0.171 / 0.145 / 0.171 | 0.181 / 0.169 / 0.147 | 0.171 | 0.169 | 1.46% |
| test-100-60 | five | 0.151 / 0.140 / 0.170 | 0.155 / 0.251 / 0.149 | 0.151 | 0.155 | -2.92% |
| test-100-100 | unchanged | 0.360 / 0.363 / 0.454 | 0.278 / 0.242 / 0.251 | 0.363 | 0.251 | 31.03% |
| test-100-100 | one | 0.292 / 0.258 / 0.308 | 0.267 / 0.250 / 0.253 | 0.292 | 0.253 | 13.29% |
| test-100-100 | five | 0.671 / 0.258 / 0.281 | 0.248 / 0.334 / 0.254 | 0.281 | 0.254 | 9.38% |
| test-200-20 | unchanged | 0.354 / 0.278 / 0.330 | 0.233 / 0.190 / 0.172 | 0.330 | 0.190 | 42.65% |
| test-200-20 | one | 0.330 / 0.200 / 0.216 | 0.164 / 0.181 / 0.165 | 0.216 | 0.165 | 23.65% |
| test-200-20 | five | 0.196 / 0.178 / 0.180 | 0.167 / 0.158 / 0.176 | 0.180 | 0.167 | 7.27% |
| test-200-60 | unchanged | 0.548 / 0.337 / 0.321 | 0.218 / 0.196 / 0.278 | 0.337 | 0.218 | 35.43% |
| test-200-60 | one | 0.295 / 0.228 / 0.238 | 0.200 / 0.251 / 0.254 | 0.238 | 0.251 | -5.25% |
| test-200-60 | five | 0.226 / 0.199 / 0.221 | 0.196 / 0.226 / 0.196 | 0.221 | 0.196 | 11.14% |
| test-200-100 | unchanged | 0.354 / 0.373 / 0.355 | 0.221 / 0.277 / 0.213 | 0.355 | 0.221 | 37.95% |
| test-200-100 | one | 0.298 / 0.345 / 0.265 | 0.205 / 0.245 / 0.201 | 0.298 | 0.205 | 31.07% |
| test-200-100 | five | 0.243 / 0.255 / 0.261 | 0.196 / 0.206 / 0.194 | 0.255 | 0.196 | 23.23% |

## Raw CPU samples and medians

| Mode/N/K | Stage | Baseline CPU samples | Current CPU samples | Baseline median | Current median |
|---|---|---|---|---:|---:|
| mark-100-20 | unchanged | 437.500 / 453.125 / 437.500 | 0.000 / 15.625 / 0.000 | 437.500 | 0.000 |
| mark-100-20 | one | 468.750 / 468.750 / 437.500 | 15.625 / 0.000 / 15.625 | 468.750 | 15.625 |
| mark-100-20 | five | 453.125 / 468.750 / 453.125 | 0.000 / 15.625 / 0.000 | 453.125 | 0.000 |
| mark-100-60 | unchanged | 1359.375 / 1359.375 / 1375.000 | 15.625 / 15.625 / 15.625 | 1359.375 | 15.625 |
| mark-100-60 | one | 1343.750 / 1390.625 / 1343.750 | 15.625 / 15.625 / 0.000 | 1343.750 | 15.625 |
| mark-100-60 | five | 1359.375 / 1390.625 / 1359.375 | 15.625 / 0.000 / 0.000 | 1359.375 | 0.000 |
| mark-100-100 | unchanged | 2265.625 / 2265.625 / 2296.875 | 15.625 / 15.625 / 15.625 | 2265.625 | 15.625 |
| mark-100-100 | one | 2359.375 / 2265.625 / 2265.625 | 15.625 / 15.625 / 15.625 | 2265.625 | 15.625 |
| mark-100-100 | five | 2296.875 / 2265.625 / 2328.125 | 31.250 / 15.625 / 15.625 | 2296.875 | 15.625 |
| mark-200-20 | unchanged | 890.625 / 906.250 / 890.625 | 0.000 / 15.625 / 0.000 | 890.625 | 0.000 |
| mark-200-20 | one | 906.250 / 890.625 / 921.875 | 0.000 / 0.000 / 15.625 | 906.250 | 0.000 |
| mark-200-20 | five | 875.000 / 906.250 / 906.250 | 0.000 / 15.625 / 0.000 | 906.250 | 0.000 |
| mark-200-60 | unchanged | 2718.750 / 2671.875 / 2671.875 | 0.000 / 0.000 / 15.625 | 2671.875 | 0.000 |
| mark-200-60 | one | 2687.500 / 2671.875 / 2703.125 | 15.625 / 0.000 / 15.625 | 2687.500 | 15.625 |
| mark-200-60 | five | 2703.125 / 2671.875 / 2671.875 | 15.625 / 15.625 / 15.625 | 2671.875 | 15.625 |
| mark-200-100 | unchanged | 4593.750 / 4609.375 / 4718.750 | 15.625 / 15.625 / 15.625 | 4609.375 | 15.625 |
| mark-200-100 | one | 4593.750 / 4578.125 / 4703.125 | 15.625 / 31.250 / 15.625 | 4593.750 | 15.625 |
| mark-200-100 | five | 4593.750 / 4640.625 / 4609.375 | 15.625 / 15.625 / 31.250 | 4609.375 | 15.625 |
| test-100-20 | unchanged | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | 0.000 |
| test-100-20 | one | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | 0.000 |
| test-100-20 | five | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | 0.000 |
| test-100-60 | unchanged | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | 0.000 |
| test-100-60 | one | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | 0.000 |
| test-100-60 | five | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | 0.000 |
| test-100-100 | unchanged | 0.000 / 0.000 / 0.000 | 15.625 / 0.000 / 0.000 | 0.000 | 0.000 |
| test-100-100 | one | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | 0.000 |
| test-100-100 | five | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | 0.000 |
| test-200-20 | unchanged | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | 0.000 |
| test-200-20 | one | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | 0.000 |
| test-200-20 | five | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | 0.000 |
| test-200-60 | unchanged | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | 0.000 |
| test-200-60 | one | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | 0.000 |
| test-200-60 | five | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | 0.000 |
| test-200-100 | unchanged | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | 0.000 |
| test-200-100 | one | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | 0.000 |
| test-200-100 | five | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | 0.000 |

## Row operation counts

Counts below are per callback and were identical in all three measured samples. All baseline/current button `setProperty`, explicit polish/unpolish/update counts are zero. Baseline label property/explicit polish/unpolish/update counts are zero. Current button and label `setStyleSheet` counts are zero in every stage. The one/five scenarios change result label tones, so button visual states correctly remain unchanged.

| Mode/N/K | Stage | Baseline button QSS | Baseline label QSS | Current label property | Current label polish | Current label unpolish | Current label update |
|---|---|---:|---:|---:|---:|---:|---:|
| mark-100-20 | unchanged | 2000 | 20 | 0 | 0 | 0 | 0 |
| mark-100-20 | one | 2000 | 20 | 1 | 1 | 1 | 1 |
| mark-100-20 | five | 2000 | 20 | 5 | 5 | 5 | 5 |
| mark-100-60 | unchanged | 6000 | 60 | 0 | 0 | 0 | 0 |
| mark-100-60 | one | 6000 | 60 | 1 | 1 | 1 | 1 |
| mark-100-60 | five | 6000 | 60 | 5 | 5 | 5 | 5 |
| mark-100-100 | unchanged | 10000 | 100 | 0 | 0 | 0 | 0 |
| mark-100-100 | one | 10000 | 100 | 1 | 1 | 1 | 1 |
| mark-100-100 | five | 10000 | 100 | 5 | 5 | 5 | 5 |
| mark-200-20 | unchanged | 4000 | 20 | 0 | 0 | 0 | 0 |
| mark-200-20 | one | 4000 | 20 | 1 | 1 | 1 | 1 |
| mark-200-20 | five | 4000 | 20 | 5 | 5 | 5 | 5 |
| mark-200-60 | unchanged | 12000 | 60 | 0 | 0 | 0 | 0 |
| mark-200-60 | one | 12000 | 60 | 1 | 1 | 1 | 1 |
| mark-200-60 | five | 12000 | 60 | 5 | 5 | 5 | 5 |
| mark-200-100 | unchanged | 20000 | 100 | 0 | 0 | 0 | 0 |
| mark-200-100 | one | 20000 | 100 | 1 | 1 | 1 | 1 |
| mark-200-100 | five | 20000 | 100 | 5 | 5 | 5 | 5 |
| test-100-20 | unchanged | 0 | 0 | 0 | 0 | 0 | 0 |
| test-100-20 | one | 0 | 0 | 0 | 0 | 0 | 0 |
| test-100-20 | five | 0 | 0 | 0 | 0 | 0 | 0 |
| test-100-60 | unchanged | 0 | 0 | 0 | 0 | 0 | 0 |
| test-100-60 | one | 0 | 0 | 0 | 0 | 0 | 0 |
| test-100-60 | five | 0 | 0 | 0 | 0 | 0 | 0 |
| test-100-100 | unchanged | 0 | 0 | 0 | 0 | 0 | 0 |
| test-100-100 | one | 0 | 0 | 0 | 0 | 0 | 0 |
| test-100-100 | five | 0 | 0 | 0 | 0 | 0 | 0 |
| test-200-20 | unchanged | 0 | 0 | 0 | 0 | 0 | 0 |
| test-200-20 | one | 0 | 0 | 0 | 0 | 0 | 0 |
| test-200-20 | five | 0 | 0 | 0 | 0 | 0 | 0 |
| test-200-60 | unchanged | 0 | 0 | 0 | 0 | 0 | 0 |
| test-200-60 | one | 0 | 0 | 0 | 0 | 0 | 0 |
| test-200-60 | five | 0 | 0 | 0 | 0 | 0 | 0 |
| test-200-100 | unchanged | 0 | 0 | 0 | 0 | 0 | 0 |
| test-200-100 | one | 0 | 0 | 0 | 0 | 0 | 0 |
| test-200-100 | five | 0 | 0 | 0 | 0 | 0 | 0 |

The structural gate also requires every counted operation to be zero for unchanged current states and for all test-mode callbacks. Changed label property/polish/unpolish/update counts must be exactly 1 or 5. The baseline gate requires exactly K×N button QSS and K label QSS calls in mark mode, and zero row QSS in test mode. Existing summary calculations can still traverse rows; this establishes O(D) row style operations, not O(D) for the entire callback.

## RED, regression and rendering evidence

The new executable structural regression gate was first run against the untouched baseline:

```powershell
$env:PYTHONHASHSEED='0'
D:/Python/Python312/python.exe -B unit_test/ui/benchmark_motor_row_visual_state.py --worker-root C:/Users/Administrator/AppData/Local/Temp/row-visual-state-20261010-avuqv10q/baseline --output-dir D:/row-visual-state-evidence-20261010/benchmark/red --rows 200 --completed 100 --mode mark --require-target
```

Observed RED: exit 1 at the zero-QSS assertion, with 20,000 button and 100 result-label QSS calls in the unchanged stage. The raw warmup and three measured sequences were saved before the assertion. The full sequential comparison then passed the current structural gate, the baseline K×N gate, within-source repeatability, and cross-source semantic equality for all warmup/measured stage snapshots. No production implementation was changed for this task.

Final seven-module regression:

```powershell
D:/Python/Python312/python.exe -m pytest unit_test/test_motor_left_panel_layout.py unit_test/test_test_task_status.py unit_test/test_manual_product_condition_cycle.py unit_test/ui/test_recording_gui_finalization.py unit_test/ui/test_sequence_analysis_process_ops.py unit_test/base/test_product_test_config_refresh.py unit_test/ui/test_motor_row_visual_state.py -q --basetemp=D:/row-visual-state-evidence-20261010/final-regression
git diff --check
```

Observed result: **187 passed, 39 subtests passed, 16 deprecation warnings, in 7.75 seconds** (exit 0). `git diff --check` passed. No baseline failure remained after Task 1's approved dummy-logger fixture repair; no business assertion was weakened.

After apply-back to the primary checkout, the same seven-module regression passed again: **187 passed, 39 subtests passed, 16 deprecation warnings, in 11.35 seconds** (exit 0). Evidence: [final-main-tests.log](D:/row-visual-state-evidence-20261010/final-main-tests.log); pytest used `--basetemp D:/row-visual-state-evidence-20261010/final-main-tests`. Temporary worktree and branch were removed; the original branch HEAD was restored with all final file contents preserved.

Task 1 rendering evidence remains in [green-tests](D:/row-visual-state-evidence-20261010/green-tests): `test_actual_button_and_hover_r0/normal{,-hover}.png`, `r1/viewed{,-hover}.png`, `r2/recording{,-hover}.png`; `test_result_palette_text_pixel0/result-ok.png` through `pixel4/result-unknown.png` cover ok/ng/running/pending/unknown. The regression tests use rendered pixels/palette and real Qt hover drawing, alongside business and lifecycle assertions.

## Evidence files and source provenance

- [Summary](D:/row-visual-state-evidence-20261010/benchmark/comparison/summary.json): calculated samples, medians, reductions, counts and equivalence result.
- [Baseline raw](D:/row-visual-state-evidence-20261010/benchmark/comparison/baseline/raw.json) and [current raw](D:/row-visual-state-evidence-20261010/benchmark/comparison/current/raw.json): all 12 scenarios, complete warmup and raw samples, counters and semantic snapshots.
- [RED raw](D:/row-visual-state-evidence-20261010/benchmark/red/raw.json): first baseline reproduction.
- Each source folder under `D:/row-visual-state-evidence-20261010/benchmark/comparison` retains `command.json` and `process.log`.

| File | SHA-256 |
|---|---|
| summary.json | `10c781dc3172ef98903f61ac495ab01974c0e504b9d6cc5045b92da4fd66a8a4` |
| baseline/raw.json | `09cc1e97dd7152fa2bb4090a00eb64d28d43d3c5f19c05d9e46cad8929427624` |
| current/raw.json | `0a801bd8611ac73d3cff81b6521052eed4d86624487c7fe4f1b256e3ccfb1b27` |

| Source | Baseline SHA-256 | Current SHA-256 |
|---|---|---|
| ui/sequence/motor_result_panel.py | `5e69f9f55b3cd30b228343db0bc5f7e9923292f03227c366efe91b63d097ecb8` | `5605020c039e7120b7f69bef4966e113f53d30cc8a9c4aa4e80941cfab35cc1d` |
| consts/ui_style_const.py | `50d181262381f97d8d7df1509136b7a562d2a1ba933de6de289a5ebe3aec9e5b` | `43be8c9ebf9e5f376ed79070401e6b0fb8a9649d7a7fbb9252cfcb276d491aa2` |
| ui/sequence/sequence_widget_analysis_ops.py | `32260683fa48176853fc27253f7285c263ee68df162d519bb35e6cf8f1a5a63c` | `32260683fa48176853fc27253f7285c263ee68df162d519bb35e6cf8f1a5a63c` |
| ui/sequence/product_condition_result_ops.py | `a991b9a8ce816803fdb355f438121b3519d1070451d37f3b7260914914de5e36` | `a991b9a8ce816803fdb355f438121b3519d1070451d37f3b7260914914de5e36` |

## Limits

These are instrumented synchronous offscreen callback measurements on this machine. They exclude event-loop paint latency, capture hardware, real recordings, CSV/video/filesystem load, live databases, analysis workers and the full recording-finalization path. Sub-millisecond test-control differences are noise-scale and are not interpreted as product regressions or speed claims. The original batch-once diagnostic values are not used anywhere in this report. No claim is made that field end-to-end finalization time falls by the same percentage.

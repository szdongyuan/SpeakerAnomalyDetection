# Unified adaptive waveform validation

2026-09-23. Worktree: `.worktrees/unified-adaptive-waveform`, branch `codex/unified-adaptive-waveform`. Component/integration state measured: `32488f17275224b38b1c106baf579ccb1568fae0`; reference component: `26c222d014932b401db2f393f6c8784bf38de696`. No merge, push, hardware recording, dependency modification, or user configuration change was performed by this benchmark task.

## Reproduction and boundaries

Run from the isolated worktree in PowerShell:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
$env:PYQTGRAPH_QT_LIB='PyQt5'
python unit_test/ui/benchmark_adaptive_waveform.py --revision 26c222d014932b401db2f393f6c8784bf38de696 --iterations 30 --output .tmp/benchmark-original.json
python unit_test/ui/benchmark_adaptive_waveform.py --iterations 30 --output .tmp/benchmark-current.json
python unit_test/ui/benchmark_adaptive_waveform.py --iterations 30 --output .tmp/benchmark-current-repeat.json
python unit_test/ui/benchmark_adaptive_waveform.py --revision 26c222d014932b401db2f393f6c8784bf38de696 --iterations 30 --output .tmp/benchmark-original-repeat.json
```

Runs execute sequentially, with the regression suite finished before timing. Every scenario warms up five batches, then collects 30 batch durations. One batch includes all visible channels, event processing, and a non-null `PlotWidget.grab()` for each widget. Four-channel cases keep all four widgets visible in one offscreen window. Source arrays and sample-time axes are allocated outside timing; `setData` replacement, internal validation, events, and painting are included. Array concatenation, acquisition, storage, and audio hardware are excluded. Y limits are fixed, so these timings do not benchmark automatic Y-range computation.

Both components use explicit plot-level `setDownsampling(auto=True, mode="peak")` and `setClipToView(True)`. Original PR call sites did not preserve item constructor defaults through `PlotItem.addItem`; this is a matched-settings component comparison, **not** the unmodified original application's default UI performance.

The signals are deterministic 997 Hz sines at 48 kHz, with 0.37 radian phase offsets per channel. Dense update shows the whole file. Sparse pan alternates a 128-sample-wide view by 16 samples. Streaming replaces the full source while zoomed to the sparse span, checking that zoom cannot enable sinc. Cache explicitly refreshes the unchanged sparse view. Requested widths 800/1600 px produce actual ViewBox widths 725/1525 px; each plot is 220 px high.

Environment: Windows 11 `10.0.26200`, Python 3.12.3 64-bit, NumPy 1.26.4, Qt 5.15.2, PyQt5 5.15.11, PyQtGraph 0.13.7. The harness selects PyQt5 and offscreen mode before imports. Both PyQt5 and PyQt6 are installed; selecting PyQt5 resolves the earlier mixed-binding offscreen failure without skipping GUI tests or modifying installed packages.

## Measured results

The attached [benchmark JSON](2026-09-23-unified-adaptive-waveform-benchmarks.json) retains all four complete runs, environment, settings, median/P95, actual widths, and reconstruction counts. Values below are milliseconds per all-channel batch, `median / P95`, rounded to two decimals. They are observations on this machine, not acceptance thresholds or cross-machine promises.

| Seconds | Channels | Width | Scenario | Original median / P95 | Current median / P95 |
| --- | --- | --- | --- | --- | --- |
| 10 | 1 | 800 | dense_update | 14.69 / 21.79 | 15.93 / 17.88 |
| 10 | 1 | 800 | sparse_pan | 12.14 / 13.75 | 9.90 / 12.09 |
| 10 | 1 | 800 | streaming_update | 4.74 / 5.80 | 3.68 / 4.64 |
| 10 | 1 | 800 | sparse_cache | 0.74 / 1.35 | 0.40 / 1.16 |
| 10 | 1 | 1600 | dense_update | 24.44 / 28.44 | 24.86 / 27.18 |
| 10 | 1 | 1600 | sparse_pan | 19.77 / 23.33 | 16.51 / 19.31 |
| 10 | 1 | 1600 | streaming_update | 4.77 / 6.13 | 4.22 / 6.37 |
| 10 | 1 | 1600 | sparse_cache | 1.42 / 2.02 | 0.93 / 1.77 |
| 10 | 4 | 800 | dense_update | 56.57 / 71.63 | 69.56 / 81.95 |
| 10 | 4 | 800 | sparse_pan | 47.59 / 60.59 | 41.39 / 47.93 |
| 10 | 4 | 800 | streaming_update | 18.66 / 38.64 | 15.04 / 16.64 |
| 10 | 4 | 800 | sparse_cache | 3.86 / 5.82 | 2.32 / 3.91 |
| 10 | 4 | 1600 | dense_update | 117.30 / 128.70 | 123.11 / 146.97 |
| 10 | 4 | 1600 | sparse_pan | 87.46 / 96.96 | 73.59 / 79.58 |
| 10 | 4 | 1600 | streaming_update | 21.45 / 26.99 | 15.88 / 19.11 |
| 10 | 4 | 1600 | sparse_cache | 5.36 / 7.44 | 4.64 / 6.46 |
| 60 | 1 | 800 | dense_update | 33.55 / 38.43 | 60.03 / 64.15 |
| 60 | 1 | 800 | sparse_pan | 10.16 / 12.34 | 9.66 / 12.29 |
| 60 | 1 | 800 | streaming_update | 21.86 / 25.81 | 19.09 / 21.91 |
| 60 | 1 | 800 | sparse_cache | 0.67 / 1.14 | 0.57 / 1.43 |
| 60 | 1 | 1600 | dense_update | 42.09 / 44.96 | 71.04 / 78.32 |
| 60 | 1 | 1600 | sparse_pan | 18.69 / 20.85 | 18.80 / 20.80 |
| 60 | 1 | 1600 | streaming_update | 19.82 / 23.37 | 20.54 / 23.27 |
| 60 | 1 | 1600 | sparse_cache | 1.28 / 2.16 | 1.16 / 1.81 |
| 60 | 4 | 800 | dense_update | 130.72 / 157.08 | 239.79 / 251.20 |
| 60 | 4 | 800 | sparse_pan | 40.70 / 46.83 | 42.22 / 46.51 |
| 60 | 4 | 800 | streaming_update | 77.81 / 84.74 | 77.55 / 86.30 |
| 60 | 4 | 800 | sparse_cache | 2.20 / 3.27 | 2.36 / 3.72 |
| 60 | 4 | 1600 | dense_update | 169.39 / 179.50 | 287.20 / 319.81 |
| 60 | 4 | 1600 | sparse_pan | 79.93 / 87.82 | 77.20 / 91.58 |
| 60 | 4 | 1600 | streaming_update | 83.32 / 103.36 | 81.17 / 100.30 |
| 60 | 4 | 1600 | sparse_cache | 4.45 / 6.82 | 4.10 / 5.42 |

## Interpretation and limits

Dense static replacement has a repeatable regression. At 60 seconds/four channels/800 px, the primary original/current medians are 130.72/239.79 ms (P95 157.08/251.20); the reversed-order repeat gives 124.87/246.48 ms (P95 134.95/275.71). At 1600 px, primary medians are 169.39/287.20 ms and repeat medians 173.20/282.09 ms. The result is not explained away as noise.

Code inspection identifies additional correctness work per static replacement: whole-axis finite/uniform validation (`isfinite`, `diff`, `allclose`), full-source finite-value detection, and finite checks before complete peak reduction. Uniform validation is cached for later pans but invalidated on every `setData`. A separate cProfile diagnostic of `benchmark_case(app, load_component(None), 60, 4, 800, 30)` took 13.884 s including setup and all scenarios: `allclose` accumulated 2.593 s, `diff` 0.732 s, the adapter `_getDisplayDataset` 6.516 s (including 4.384 s in the dependency), and `_refresh_smoothing` 4.654 s. These overlapping cumulative values must not be added. The profile confirms substantial validation/reduction work, but is not a precise attribution of every millisecond of the revision difference. It was not included in the timing matrix.

Sparse pan at 60 seconds/four channels/800 px is 40.70/42.22 ms original/current in the primary run and 42.19/40.44 ms in the repeat. Streaming is 77.81/77.55 ms primary and 78.60/75.94 ms repeat. Thus the dense replacement regression is clear, while these pan/streaming differences do not establish a consistent slowdown. Cache timings remain small and variable. No broad superiority claim is justified. The extra validation protects remote nonuniform axes and nonfinite gaps; optimizing static replacement is deferred rather than weakening those contracts.

All full runs enforce zero sinc calls during streaming and unchanged-cache warmup and measured batches. Sparse pan input is at most 195 samples; output is at most 1,581 points at 800 px and 3,181 at 1600 px. These sizes are exactly equal for matched 10/60-second cases in both revisions and well below the 4,160-input/12,288-output guards. Full-array replacement validation remains duration-dependent; these locality claims concern reconstruction and subsequent pan/cache operations.

Painting was exercised through real Qt offscreen events and pixmap grabs, not a numerical-only benchmark. The orchestrator additionally inspected `.tmp/waveform-validation.png`: sparse sine and a tail impulse of amplitude 7 remain visible. That diagnostic image is supplementary and not required to interpret this report. No actual capture/playback device, physical display, real-time scheduling, or production session load was validated. In particular, even the reference's 60-second four-channel streaming batch exceeds 50 ms in these runs; neither implementation has a demonstrated 20 Hz end-to-end recording guarantee.

## Regression evidence

The following prior RED evidence was supplied by the Task 1/2 implementers and recorded by the orchestrator; it was not recreated by mutating production sources during this benchmark task:

- Reference tail regressions: six failures before complete partial-bucket reduction.
- Clear lifecycle left stale children; a remote nonuniform time step incorrectly allowed sinc; dynamic-range cache reuse reduced data twice. Each received a focused fix and passing regression coverage.
- Two Inf-gap failures demonstrated lost discontinuities before preserving nonfinite values through range limiting.
- Integration tests initially had eight failures and four passes before wiring shared items, streaming state, and plot settings.
- Two clip endpoint failures demonstrated loss of the final sample for full-duration/final-sample-only regions; exclusive selection bounds now extend to `N/fs` while plotted samples end at `(N-1)/fs`.

Harness RED: direct invocation with `--iterations 1 --output .tmp/benchmark-smoke.json` initially exited 1 because the planned harness did not exist. GREEN: the same invocation completed all eight configurations, serialized finite JSON, closed plots, and passed sinc/locality assertions. This is a standalone verification tool; it adds no product behavior and no timing assertions to pytest.

Final regression command:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
$env:PYQTGRAPH_QT_LIB='PyQt5'
python -m pytest unit_test/ui/test_adaptive_waveform.py unit_test/ui/test_audio_clip_adaptive_waveform.py unit_test/ui/test_channel_plot_workspace_visibility.py unit_test/ui/test_sequence_wav_calibration_metadata.py unit_test/ui/test_sequence_widget_sn_lock.py unit_test/ui/test_stimulus_window_frequency_stepped.py -q -p no:cacheprovider
git diff --check
```

Task 3 rerun: **286 passed, 5 warnings in 41.57 s**, exit 0. The orchestrator independently obtained 286 passed/5 warnings in 42.77 s before timing began. Warnings are existing dependency deprecations: two protobuf/upb metaclass warnings, librosa's `pkg_resources` import, and deprecated namespace declarations for `pywinusb` and `zope`. No tests were skipped to avoid Qt execution. `python -m py_compile unit_test/ui/benchmark_adaptive_waveform.py` and `git diff --check` also exited 0.

Changes in this task are limited to the benchmark harness, this report, and the attached JSON. Dense replacement slowdown is a documented tradeoff; sparse work remains bounded and no production performance refactor was included.

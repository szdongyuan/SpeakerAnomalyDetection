# Unified Adaptive Waveform Implementation Plan

> **For agentic workers:** REQUIRED: Use the `subagent-driven-development` skill to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify raw-audio waveform previews, preserve dense-view tail peaks, and keep live recording free of sinc computation.

**Architecture:** Build on the display-only AdaptiveWaveformItem from PR #920. Keep all original arrays unchanged, handle peak reduction including partial end buckets inside this component, and wire existing waveform windows into it. Preserve recording/storage/analysis architecture and restore adaptive display after successful recording.

**Tech Stack:** Python, NumPy, PyQt5, PyQtGraph (local version 0.13.7), pytest, Windows PowerShell.

---

## Authoritative context and workflow

- Spec: `docs/double-sdd/specs/2026-09-23-unified-adaptive-waveform-design.md`, user approved 2026-09-23.
- Original code base: `842bb02ae9ca053e3af91d1bb70ec0d200d3a45a`; reference PR commit: `26c222d014932b401db2f393f6c8784bf38de696`.
- Main workspace already contains a user edit to `ui/ui_config/analysis_default_config.json`. Never stage, overwrite, or copy that edit into implementation.
- Record main HEAD after document commits as immutable implementation base. Create `.worktrees/unified-adaptive-waveform` on `codex/unified-adaptive-waveform` from that exact commit. `.worktrees/` is already ignored. Do not merge, push, or publish. Document checkpoints may be committed; implementation checkpoints stay in this worktree.
- Run one fresh implementer per task sequentially. Review each completed task with independent spec and quality reviews for the same recorded base/head. Apply `code-review` to verify findings. If the configured quality-reviewer model is unavailable, use a default agent with the same bounded review brief, never self-review as its replacement.
- Docs are ignored by this repository: explicitly `git add -f -- <document>` when committing approved documents/reports. Never force-add a whole directory of unrelated files.
- Use `systematic-debugging` for failures and `verification-before-completion` before success claims. Existing offscreen Qt exit must be diagnosed, not bypassed by marking GUI tests skipped.
- Environment diagnosis confirmed both PyQt6 and PyQt5 are installed: importing PyQtGraph first can select PyQt6 while application widgets use PyQt5. For every test/benchmark command set `$env:QT_QPA_PLATFORM='offscreen'` and `$env:PYQTGRAPH_QT_LIB='PyQt5'` in that same PowerShell invocation. This fixes the minimal reproduction and existing tests without installation changes. New standalone harness/tests must select PyQt5 before importing PyQtGraph, consistent with application widgets.

## File map

| File | Responsibility |
| --- | --- |
| `ui/adaptive_waveform.py` (new) | Display-only sinc, bounded local reconstruction/cache, safe dense peak display |
| `unit_test/ui/test_adaptive_waveform.py` (new) | Numerical, actual Qt item lifecycle, dense tail and streaming invariants |
| `ui/stimulus_window.py` | Stimulus preview component integration |
| `ui/sequence/channel_plot_workspace.py` | Shared channel item and explicit streaming option |
| `ui/sequence/sequence_widget.py` | Both recording paths and final/static display state, sample times |
| `ui/custom_ui_widget/audio_clip_extraction_dialog.py` | Clip-preview component and exact time axis |
| `unit_test/ui/test_audio_clip_adaptive_waveform.py` (new) | Actual dialog integration and original-array selection |
| Existing tests named below | Targeted integration assertions/compatible doubles |
| `unit_test/ui/benchmark_adaptive_waveform.py` (new) | Reproducible optional offscreen benchmark; not timing assertions in pytest |
| `docs/double-sdd/reports/2026-09-23-unified-adaptive-waveform-validation.md` (new) | Commands, environment, measured comparison and limitations |

## Task 1: Shared waveform component and reliable dense peaks

**Own:** `ui/adaptive_waveform.py`, `unit_test/ui/test_adaptive_waveform.py` only; temporary diagnostics outside tracked sources are allowed.

- [x] **Step 1: Establish the test environment.** Run `python -c "import sys,numpy,pyqtgraph; from PyQt5.QtCore import QT_VERSION_STR; print(sys.version,numpy.__version__,pyqtgraph.__version__,QT_VERSION_STR)"`. Run `python -m pytest unit_test/ui/test_channel_plot_workspace_visibility.py -q -p no:cacheprovider` from the worktree, with `QT_QPA_PLATFORM=offscreen`. Diagnose failures using minimal QApplication → PlotWidget → show → processEvents probes. Do not modify global installs or production modules as a test workaround.
- [x] **Step 2: Add the original PR numerical/UI tests, plus failing tail regressions.** Retrieve reference tests with `git show 26c222d014932b401db2f393f6c8784bf38de696:unit_test/ui/test_adaptive_waveform.py`; write the file using a byte-preserving Python subprocess call instead of PowerShell's legacy output encoding. Before adding implementation, run the file and record expected missing-module failure.
- [x] **Step 3: Bring in the reference shared component only.** Retrieve `ui/adaptive_waveform.py` from that exact commit, not the entire PR. Inspect root causes before changing it. Run numerical and GUI tests. Add an actual PlotWidget regression that forces `setDownsampling(ds=12, auto=False, method="peak")`, uses 48001 and 48005 samples, and puts signed impulses in the first, middle, last and partial-tail interior positions. Inspect actual display data and bounds, not only a stand-alone reducer. Demonstrate the tail test fails against the unchanged PR implementation.
- [x] **Step 4: Implement local peak retention.** Keep xData/yData untouched and preserve visible-range clipping/automatic reduction. Use a component-local reducer/adapter which includes the partial end bucket; no third-party edits or global patches. The core reduction behavior can follow this complete NumPy operation (integration must preserve Qt finite-connect and mapped-mode semantics):

```python
def peak_buckets(x, y, bucket_size):
    starts = np.arange(0, len(y), bucket_size)
    if not len(starts):
        return x[:0], y[:0]
    ends = np.minimum(starts + bucket_size, len(y))
    centers = (starts + ends - 1) // 2
    peaks = np.maximum.reduceat(y, starts)
    troughs = np.minimum.reduceat(y, starts)
    return np.repeat(x[centers], 2), np.column_stack((peaks, troughs)).ravel()
```

This is a reducer sketch, not permission to apply min/max blindly across NaN/Inf breaks. Preserve finite segments or fall back appropriately. Prefer a narrowly scoped adapter over copying PyQtGraph's entire private dataset pipeline. If a dependency hook is necessary, isolate and test it against installed versions. Do not append fake raw samples to solve the tail.

- [x] **Step 5: Complete behavioral tests and fixes.** Cover known sine accuracy/sample reconstruction, constant extension, short/nonfinite data, unsupported maps/nonuniform sampling, repeated cache hits with an interpolation call spy, bounded reconstruction input/output for 10/60-second sources, resize/zoom hysteresis, auto Y including reconstructed peaks, clear/remove/re-add, and streaming→static→streaming. Assert interpolation is never called in streaming, including zoom/resize. Ensure empty data hides all children. Fix only confirmed behavior failures.
- [x] **Step 6: Verify and checkpoint.** Run `python -m pytest unit_test/ui/test_adaptive_waveform.py unit_test/ui/test_channel_plot_workspace_visibility.py -q -p no:cacheprovider`. Run `git diff --check`. Commit only owned files as `tmp/sdd: adaptive waveform with complete peak buckets`. Report RED/GREEN evidence, diagnostics and commit SHA for paired review.

## Task 2: Wire every raw-audio preview and recording transition

**Own:** Four UI call-site files in the map; `unit_test/ui/test_audio_clip_adaptive_waveform.py`; existing `test_sequence_wav_calibration_metadata.py`, `test_sequence_widget_sn_lock.py`, `test_stimulus_window_frequency_stepped.py`, `test_channel_plot_workspace_visibility.py` as needed.

- [x] **Step 1: Inspect existing call-site tests and write failing integration tests.** Add assertions for shared item in stimulus/channel/clip windows. Test both chunk handlers pass streaming=True and both successful completions restore static display with final arrays. Update doubles to accept/record keyword-only streaming without removing their existing assertions. Test no false success on cancelled/failed recordings and that subsequent static load recovers. Run focused cases and record failures on pre-task code.
- [x] **Step 2: Apply reference PR call-site behavior, not a whole-commit cherry-pick.** Inspect `git diff 842bb02 26c222d -- ui/stimulus_window.py ui/sequence/channel_plot_workspace.py ui/sequence/sequence_widget.py`. Use `AdaptiveWaveformItem` for stimulus addItem and channel data; expose `set_data(self, x, y, *, streaming=False)`. Forward this argument to both creation and setData. Both chunk handlers use True; successful final displays use False. Final playback/aligned time axis uses `np.arange(len(aligned_data)) / sample_rate`. Preserve error handling, file writes, recording metadata and existing post-recording followup helper.
- [x] **Step 2a: Preserve downsampling/clipping through PlotItem integration.** Live investigation established `PlotItem.addItem` overwrites item-level `autoDownsample` and `clipToView` defaults with the plot's settings. Configure each raw waveform PlotWidget with `setDownsampling(auto=True, mode="peak")` and `setClipToView(True)` before adding items. Assert all three actual window integrations retain these options and reduce dense data after attachment. Keep public manual downsampling controls usable; do not force constructor options back on every item update.
- [x] **Step 3: Integrate the clip dialog.** Replace `pg.PlotDataItem(pen="k")` with `AdaptiveWaveformItem(pen="k")` and generate `np.arange(len(audio_data)) / sample_rate`. Keep raw data, mono loading, region interactions and sample slicing unchanged. Empty/failed audio loading must still follow existing error handling and clear display.
- [x] **Step 4: Verify original-array selection and time coordinates.** Use deterministic mocked librosa loading with known source values/sample rate, select a region, intercept save/return handling, and assert the selected array equals the same original slice before and after visual zoom. Assert xData equals `np.arange(N)/fs`, last sample `(N-1)/fs`, and yData/source_audio unchanged. Exercise region cancel/fixed-length behavior without a filesystem/audio-device requirement.
- [x] **Step 4a: Preserve the exclusive selection endpoint.** Region selection bounds must extend to `N/fs` even though the last plotted sample is `(N-1)/fs`; otherwise Python's end-exclusive slicing drops the last sample. Verify full-duration selection and final-sample-only selection. Retain prior empty-load failure/clear behavior with an explicit check if the former `time_array[-1]` access is removed.
- [x] **Step 5: Run integration suite.** Run `python -m pytest unit_test/ui/test_adaptive_waveform.py unit_test/ui/test_audio_clip_adaptive_waveform.py unit_test/ui/test_channel_plot_workspace_visibility.py unit_test/ui/test_sequence_wav_calibration_metadata.py unit_test/ui/test_sequence_widget_sn_lock.py unit_test/ui/test_stimulus_window_frequency_stepped.py -q -p no:cacheprovider`. Diagnose pre-existing fixture/dependency issues rather than claiming them passed. Run `git diff --check`.
- [x] **Step 6: Checkpoint.** Commit only owned changes as `tmp/sdd: integrate adaptive waveforms in all raw audio previews`, report targeted/full results and SHA for paired review. Do not modify other analysis plot types or config files.

## Task 3: Reproducible performance evidence and final regression validation

**Own:** `unit_test/ui/benchmark_adaptive_waveform.py`, validation report. If a new production defect appears, report evidence to orchestrator for a bounded fix/re-review rather than silently expanding ownership.

- [x] **Step 1: Add a standalone benchmark harness.** Use argparse options `--revision` (optional git ref; absent imports worktree implementation), `--iterations` (default 30), `--output` (JSON path). A ref loads only `ui/adaptive_waveform.py` via `subprocess.check_output(["git", "show", ref + ":ui/adaptive_waveform.py"])` and a module namespace; no checkout, monkeypatch of installed libraries, or persistent mutation. Run with QApplication in offscreen mode. Validate harness can create/close plots and serialize finite metrics before measuring the full matrix.
- [x] **Step 2: Implement representative cases.** Full cross product: source duration 10/60 seconds at 48 kHz, 1/4 visible channel PlotWidgets, requested widget widths 800/1600 pixels. Record actual ViewBox width. Use same phase-offset deterministic sine data for both revisions. Measure dense setData+event processing+widget grab; sparse pan+events+grab; streaming setData+events+grab; unchanged sparse refresh cache+events+grab. Warm up 5 iterations and report median/P95 for the remaining iterations. Treat a batch across all visible channels as one timing sample. Keep allocation of source signal outside timings; label axis/data replacement included in measured setData consistently. Assert no sinc calls for streaming and no reconstruction calls for unchanged cache scenario. For matched sparse view and width, record interpolation input/output sizes and enforce duration-independent bounds. Record Python/NumPy/Qt/PyQtGraph/OS versions, iterations and timing boundaries in JSON.
- [x] **Step 2a: Compare matching effective plot settings.** Explicitly enable plot-level auto peak downsampling and clipping in both benchmark revisions so the comparison isolates component cost. State in the report that the original PR call sites did not preserve these settings on addItem; the matched-settings benchmark is not a measurement of the unmodified full application's default UI. Keep all four widgets visible for four-channel cases and verify rendering via grab, not only setData calls.
- [x] **Step 3: Execute comparison.** Run `python unit_test/ui/benchmark_adaptive_waveform.py --revision 26c222d014932b401db2f393f6c8784bf38de696 --iterations 30 --output <temporary-original.json>` and `python unit_test/ui/benchmark_adaptive_waveform.py --iterations 30 --output <temporary-current.json>`. The script must add repository root to sys.path if required for direct execution. Compare matched scenarios. Investigate repeatable regressions, report cause/tradeoff; do not invent machine-independent latency acceptance thresholds or assert performance superiority from noise. No hardware claims.
- [x] **Step 4: Run final regression command from Task 2.** Verify git diff --check and inspect complete change scope. Record counts, warnings, failure diagnostics, RED/GREEN tail evidence, GUI verification resolution, benchmark aggregates and remaining hardware limits in `docs/double-sdd/reports/2026-09-23-unified-adaptive-waveform-validation.md`. Include representative/full matrix results or attach tracked compact JSON if necessary; never require an ephemeral output file to interpret the report.
- [x] **Step 5: Checkpoint and reviews.** Commit harness and report as `tmp/sdd: record waveform performance and regression evidence`. Update this plan's checkboxes only for actually verified steps. Run paired task reviews, then final quality review over immutable base..final HEAD. Address verified blockers through implementer, re-run affected checks and reviews. Apply `finishing-a-development-branch` with the already selected outcome: retain isolated branch/worktree for user review, no merge/push.

## Delivery evidence

Report the worktree/branch and final commit, test counts and relevant remaining limitations; link the spec, validation report and main changed files. Main worktree must retain the user's original configuration modification unchanged. No merging original PR or feature worktree is authorized by this plan.

## Completion record

All three tasks passed independent spec and quality reviews. Final whole-range quality review passed for `30c4137..2e38eaa`, independently rerunning 286 tests with five existing dependency warnings. Benchmark matrices and the independent smoke run passed; dense static replacement slowdown and missing hardware acceptance remain disclosed in the validation report. Feature branch/worktree retained without merge or push. Main user config SHA256 remained `39B292184A195D9BC1139F7C16E866D7259054FB5146E5A190919818E953C435`.

## User-requested local integration

After the isolated application preview, the user explicitly requested integration into the main directory and restart. Squash integration onto develop at the recorded base was verified in the main directory: 286 tests passed, five existing dependency warnings, 42.63 seconds. The existing analysis configuration edit remains unstaged and its SHA256 is unchanged. Temporary implementation checkpoints are not included as individual commits in develop. Runtime data in the preview worktree is retained to avoid discarding local preview changes.

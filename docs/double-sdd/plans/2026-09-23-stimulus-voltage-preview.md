# Stimulus Voltage Preview Implementation Plan

> **For agentic workers:** REQUIRED: Use the `subagent-driven-development` skill to implement this plan. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Display stimulus samples in the configured target voltage instead of normalized amplitude.

**Architecture:** A small NumPy conversion helper produces display-only samples and a conversion-status flag. StimulusWindow passes these samples to the existing adaptive plot and refreshes on voltage/type edits, while source data and playback/calibration remain unchanged.

**Tech Stack:** Python, NumPy, PyQt5, PyQtGraph, pytest.

---

Spec: `docs/double-sdd/specs/2026-09-23-stimulus-voltage-preview-design.md` (independently reviewed). Main base `ff39fded979d5a20b9bdbe195d9fb0e99b4b5ade`; worktree `.worktrees/stimulus-voltage-preview`, branch `codex/stimulus-voltage-preview`. Main `ui/ui_config/analysis_default_config.json` is the user's existing edit and must not enter the change. User clarified the bug is the plotted ordinate, not measured playback voltage.

Tests must set `$env:QT_QPA_PLATFORM='offscreen'; $env:PYQTGRAPH_QT_LIB='PyQt5'`. Dependencies are installed; ignored skills are in the main repo `.agents/skills/`. No real audio playback in tests.

## File responsibilities

- Create `ui/stimulus_voltage_preview.py`: pure `build_stimulus_voltage_preview(source, voltage, voltage_type)` returning `(display_samples, is_voltage)`; no Qt, hardware, file or metadata mutations.
- Modify `ui/stimulus_window.py`: plotting conversion/label and control refresh only.
- Create `unit_test/ui/test_stimulus_voltage_preview.py`: numeric/fallback/source-integrity cases.
- Modify `unit_test/ui/test_stimulus_window_frequency_stepped.py`: actual window regressions reusing its fixtures; update two obsolete raw-y display assertions while preserving explicit raw-source assertions.

## Task 1: Correct target-voltage display and refresh

- [x] Add numeric tests before helper implementation. Use `[0, 1, 0, -1]` as a known whole-cycle sine: Peak .1 gives `[0,.1,0,-.1]`; RMS .1 gives `[0,sqrt(2)*.1,0,-sqrt(2)*.1]`. Validate non-unit source scale, noise RMS, signs, zero/empty, zero V, unknown/missing/invalid voltage/type, NaN/Inf gaps, high-amplitude stability and unchanged source. Expectations must be analytical or direct independent statistical assertions, not calls to the helper under test.
- [x] Add actual-window failing tests for .1 V Peak/RMS and type-only change; use fixtures that avoid hardware. Verify graph yData and axis label/units. Add external-WAV mode voltage-only refresh and unchanged source assertions. Run `python -m pytest unit_test/ui/test_stimulus_voltage_preview.py unit_test/ui/test_stimulus_window_frequency_stepped.py -q -p no:cacheprovider`; record missing-helper and display mismatch RED before code.
- [x] Implement pure conversion. Parse finite nonnegative V and recognized case-insensitive type. For finite nonempty nonzero y, compute `scale=max(abs(y))`, `unit=y/scale`; Peak uses `unit*V`, RMS uses `unit/sqrt(mean(unit*unit))*V`. Guard numerical overflow/nonfinite result and fall back to original values with `is_voltage=False`. Empty/all-zero valid inputs stay empty/zero; valid zero V returns zero. Invalid metadata/nonfinite source retains original data and returns False. Never modify input arrays or apply calibration amplitude again.
- [x] Integrate in `graph_stimulus`: retain exact sample-time axis and AdaptiveWaveformItem, pass display_samples instead of stimulus_data, set label text `目标电压` and units `V` when converted; fallback text `原始幅度` with empty units. Keep stimulus_data identity/content and all save/play data unchanged. Preserve source-mode switching and marker behavior.
- [x] Refresh when RMS/Peak changes after updating metadata. Ensure callbacks are safe before source/plot initialization and while restoring config. Preserve numeric voltage edit's existing calibration and generation logic; add graph refresh for the non-generated/external-WAV path. Do not regenerate imported WAV or modify source on type-only edits. Avoid blanket callback rewrites.
- [x] Adapt existing expected-y checks at legacy external-WAV restore and adaptive-plot integration to voltage-unit expectations; retain or strengthen assertions for original stimulus_data. Add a non-unit amplitude frequency_stepped case so multiplication is not applied twice. For deterministic noise/WAV arrays verify complete-waveform normalization stays identical after zooming.
- [x] Add fake playback/save tests or extend existing window tests to capture inputs before/after graph/type-only refresh: source arrays and calibrated amplitude are unchanged. Test initialization with both modes and loaded/restore paths for no Qt callback exceptions. Confirm source zeros/invalids render safe label/fallback without repeated dialogs.
- [x] Run focused tests then this affected suite (includes earlier waveform integrations and calibration/sample-rate consumers):

```powershell
$env:QT_QPA_PLATFORM='offscreen'
$env:PYQTGRAPH_QT_LIB='PyQt5'
python -m pytest unit_test/ui/test_stimulus_voltage_preview.py unit_test/ui/test_adaptive_waveform.py unit_test/ui/test_audio_clip_adaptive_waveform.py unit_test/ui/test_channel_plot_workspace_visibility.py unit_test/ui/test_sequence_wav_calibration_metadata.py unit_test/ui/test_sequence_widget_sn_lock.py unit_test/ui/test_stimulus_window_frequency_stepped.py unit_test/ui/test_stimulus_window_samplerate_authority.py unit_test/ui/test_speaker_calibration_db_consumers.py unit_test/base/test_playback_controller_samplerate.py -q -p no:cacheprovider
python -m pytest unit_test/base/test_play_and_record_channel_forwarding.py -q -p no:cacheprovider
git diff --check
```

Baseline test-isolation note: before this fix, running channel-forwarding tests after UI tests can destroy the global MySignals QObject when a module-scoped QApplication is torn down (113 passed / 2 lifetime failures in the four-file probe). Channel-forwarding alone passes all 62 tests. Run the two commands separately, as above; do not alter audio production code or skip those tests to mask the test-lifetime issue.

- [x] Commit only owned source/test files as a temporary checkpoint; record RED/GREEN and results. Paired spec/quality review covers entire one-task code change and serves as final review. Use default quality agent if configured custom model remains unavailable. Route confirmed fixes back through implementer, not orchestrator edits.
- [x] Squash-integrate into main only after review/testing, preserve original config hash, run focused tests on main and commit one final change. Notify before restarting the app; preserve old runtime data. No push or remote merge, no hardware voltage claim.

## Completion evidence

Actual-window RED: four failures reproduced normalized +/-1 at a .1 V setting and missing edit refresh. Focused GREEN: 148 tests. Independent spec/quality reviews passed. Main-directory squash verification: 384 affected tests passed with five existing dependency warnings in 40.61 seconds; separate channel-forwarding suite 62 passed in 2.02 seconds. User analysis configuration SHA256 remained `39B292184A195D9BC1139F7C16E866D7259054FB5146E5A190919818E953C435`, unstaged. Display values are target volts, not hardware measurements. Playback/save behavior remains unchanged.

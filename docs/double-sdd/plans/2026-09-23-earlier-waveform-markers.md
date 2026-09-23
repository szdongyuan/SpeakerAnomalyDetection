# Earlier Waveform Markers Implementation Plan

> **For agentic workers:** REQUIRED: Use the `subagent-driven-development` skill to implement this plan. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Show original sample markers at smaller zoom in all shared raw-audio previews.

**Architecture:** Change the shared static marker threshold from 12 px sample spacing to 4 px, keeping marker size and reconstruction eligibility. Retain dense peak rendering and streaming bypass.

**Tech Stack:** Python, NumPy, PyQt5, PyQtGraph 0.13.7, pytest.

---

Spec: `docs/double-sdd/specs/2026-09-23-unified-adaptive-waveform-design.md`, user-requested marker amendment. Main integration base `15aadfb8ac0f787d019e423b155cc273f28fdd92`; isolated worktree `.worktrees/earlier-waveform-markers`, branch `codex/earlier-waveform-markers`. Main user configuration edit must remain unchanged. This is a follow-up adjustment to the already approved/implemented waveform design.

Use `$env:QT_QPA_PLATFORM='offscreen'; $env:PYQTGRAPH_QT_LIB='PyQt5'` in each PowerShell verification invocation; both Qt bindings are installed. Skills are at the main repository `.agents/skills/` because ignored directories are not copied into worktrees. No dependency installation needed.

## Task 1: Earlier original-sample markers

**Files:**
- Modify `ui/adaptive_waveform.py` marker visibility condition in `_refresh_smoothing`.
- Modify `unit_test/ui/test_adaptive_waveform.py` focused visibility regression.

- [x] Write a regression using actual PlotWidget/ViewBox: choose fixed Y bounds and X range calculated from `vb.width()` so sample spacing is controlled. Show markers at spacing 4.5 and 6 (below old threshold), hide at 3.5, and continue showing at 12.5. Verify zooming back out hides markers. Check the >=4 boundary using measured actual spacing with floating-point-safe ranges rather than an imprecise expected width. Confirm marker coordinates are original visible samples.
- [x] Run `python -m pytest unit_test/ui/test_adaptive_waveform.py -q -p no:cacheprovider` before implementation; record the earlier-marker failure.
- [x] Change only `self._markers.setVisible(bool(self._smooth and spacing >= 12))` to `self._markers.setVisible(bool(self._smooth and spacing >= 4))`. No new user option, no change to marker diameter/sinc thresholds/dense rendering, no other UI rewrites.
- [x] Verify at the earlier marker zoom that `streaming=True` hides markers and never calls `bandlimited_values` (spy that fails on calls); restoring static display shows them. Existing empty/invalid/clear cases remain passing. Cover a resize that moves effective spacing above/below the threshold; no arbitrary sleeps.
- [x] Run the focused component suite and the existing six-file integration suite: `python -m pytest unit_test/ui/test_adaptive_waveform.py unit_test/ui/test_audio_clip_adaptive_waveform.py unit_test/ui/test_channel_plot_workspace_visibility.py unit_test/ui/test_sequence_wav_calibration_metadata.py unit_test/ui/test_sequence_widget_sn_lock.py unit_test/ui/test_stimulus_window_frequency_stepped.py -q -p no:cacheprovider`. Run `git diff --check`; checkpoint only the two owned files. Report counts and RED/GREEN evidence.
- [x] Paired independent spec/quality review of the task's recorded base..head. Verify findings before changes. Default quality agent may replace unavailable custom gpt-5.4 reviewer role. For this one-task follow-up, the paired whole-code-range quality review also supplies the final implementation quality gate.
- [x] After verification, squash-integrate the small follow-up into the main directory, preserving the unstaged config. Verify component tests on integrated code and commit one final change. No push or remote PR merge. The currently running application needs a restart to load the updated module; do not close the user's active UI solely for this marker preference without notifying them.

## Verification record

RED: 1 failed / 49 passed before threshold change. GREEN: 50 component tests and 286 six-file integration tests passed (five existing dependency warnings). Both independent reviews passed. Main-directory squash result independently passed 50 component tests in 2.42 seconds; user configuration hash unchanged. No change to waveform data, interpolation thresholds or dense rendering.

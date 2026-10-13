# Retire hidden manual label Implementation Plan

> **For agentic workers:** REQUIRED: Use the `subagent-driven-development` skill to implement this plan. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Remove the unreachable global manual OK/NG entry without changing current recording, mode switching, or automatic persistence.

**Architecture:** Delete the retired handler, its exclusive sender-based label helper, and hidden button wiring/construction. Preserve shared recording guards, condition labeling, mode controls, and automatic save paths.

**Tech Stack:** Python, PyQt5, pytest.

---

Spec: `docs/double-sdd/specs/2026-10-12-retire-hidden-manual-label-design.md`.
Execute in the existing `recording-final-envelope` worktree. User has explicitly authorized this conditional deletion; reachability condition is verified. Keep changes uncommitted for inspection. This is not the envelope implementation.

Baseline: the five-file suite produced 40 passed and 4 existing failures before edits: toolbar `test_production_blocked_config_and_barcode_updates_keep_full_hints` (fake lacks `active_product_program_file`) and the three `test_serial_states_paint_without_losing_native_status` variants (offscreen image assertion). Do not fix unrelated code or weaken these assertions. Require no additional failures and run the focused wiring/label/mode tests separately for a clean passing result.

### Task 1: Remove the retired entry and verify surviving flows

**Files:**
- Modify `ui/sequence/sequence_widget_barcode_ops.py`: delete `clicked_ok_or_ng` only; retain shared guard and automatic finalization/persistence.
- Modify `ui/sequence/sequence_widget_streaming_ops.py`: delete exclusive `update_audio_label_info` only.
- Modify `ui/sequence/sequence_widget_ui_ops.py`: remove the two retired button signal connections.
- Modify `ui/sequence/sequencement_count_board.py`: remove `ok_btn`, `ng_btn`, their exclusive styling helper/call and unused imports; retain mode buttons and availability callbacks.
- Test `unit_test/test_recording_label_guard.py`: remove global-handler-only fixtures/tests; preserve both condition guard tests.
- Test `unit_test/test_sequence_count_board_shared_statistics.py`: preserve mode, availability, warning and no-statistics behavior; remove obsolete manual button expectations and unused fixtures.
- Test `unit_test/ui/test_sequence_toolbar_responsive.py`: production wiring regression must work without obsolete button fields or handler stub.

- [x] Update `test_serial_click_uses_production_signal_and_configuration_entry`: remove `clicked_ok_or_ng` from stub method names, and remove `ok_btn`/`ng_btn` from count-board fake. Leave serial-click/dialog assertions intact. Run it before production changes; expect AttributeError from current obsolete wiring.
- [x] Remove exactly the retired source elements above. Check references across production code; do not replace with no-ops or fallbacks. Add no exceptions or global state; do not refactor unrelated controls.
- [x] Remove retired-handler test fixtures/assertions and keep meaningful active behavior coverage. If removing unused `DEFAULT_DIR` import, remove matching obsolete test monkeypatches only.
- [x] Run PowerShell: `$env:QT_QPA_PLATFORM='offscreen'` then `python -m pytest unit_test/test_recording_label_guard.py unit_test/test_recent_session_label_update.py unit_test/test_recent_session_mode_switch.py unit_test/test_sequence_count_board_shared_statistics.py unit_test/ui/test_sequence_toolbar_responsive.py -q --tb=short --basetemp=.pytest_cache/retired_full`. Expect only the four documented baseline failures and no additional failures. Use a fresh worktree-local basetemp per execution to avoid a pre-existing system Temp PermissionError.
- [x] Run `python -m pytest unit_test/test_recording_label_guard.py unit_test/test_recent_session_label_update.py unit_test/test_recent_session_mode_switch.py unit_test/test_sequence_count_board_shared_statistics.py unit_test/ui/test_sequence_toolbar_responsive.py::test_serial_click_uses_production_signal_and_configuration_entry -q --tb=short --basetemp=.pytest_cache/retired_focused`. Expect all selected tests pass.
- [x] Run `rg -n 'clicked_ok_or_ng|update_audio_label_info|count_board\.(ok_btn|ng_btn)' ui unit_test`, expect no remaining references; inspect diff and `git diff --check`.
- [x] Report red/green evidence, changed files and retained behaviors; no commit. Spec and quality reviewers independently review the uncommitted diff from base `e9bf1147f635156a081d893d17dc106cfe404539`, checking scope and preserved live consumers.

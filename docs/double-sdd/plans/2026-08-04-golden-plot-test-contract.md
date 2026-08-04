# Golden Plot Test Contract Synchronization Implementation Plan

> **For agentic workers:** REQUIRED: Use the `subagent-driven-development` skill to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update two stale golden-sample dual-plot tests to verify the existing configure-unlinked/finalize-linked X-axis lifecycle.

**Architecture:** This is a test-only synchronization. Production plotting code, configuration defaults, and SPEC fallback constants remain unchanged. Because the user-selected Git publishing workflow requires all branch and commit operations in the verified original main worktree, execute on the existing `GQ-develop-20260804-enhance-golden-sample-analysis-and-calibration` branch in that worktree and do not create a linked worktree.

**Tech Stack:** Python 3.12, pytest, PyQt5, pyqtgraph

---

### Task 1: Synchronize the Dual-Plot Lifecycle Assertions

**Files:**
- Modify: `unit_test/ui/test_golden_sample_plot_layout.py`
- Modify: `unit_test/ui/test_golden_sample_plot_titles.py`
- Do not modify: `consts/acoustic_analysis/specific_consts/spec_consts.py`
- Do not modify: `ui/ui_config/analysis_default_config.json`
- Do not modify: `unit_test/ui/test_analysis_config_defaults.py`
- Do not modify production code.

- [ ] **Step 1: Reproduce the two stale assertions**

Run from the repository root with `QT_QPA_PLATFORM=offscreen`:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
python -m pytest --no-header -q --tb=short `
  unit_test/ui/test_golden_sample_plot_layout.py::test_dual_mode_orders_equal_height_linked_x_only_plots `
  unit_test/ui/test_golden_sample_plot_titles.py::test_dual_golden_plot_titles_match_existing_layout_contract
```

Expected: both tests fail because `linkedView(ViewBox.XAxis)` is `None`
immediately after `configure_golden_sample_plots()`.

- [ ] **Step 2: Update the layout lifecycle assertions**

In `test_dual_mode_orders_equal_height_linked_x_only_plots`, retain the existing
splitter order, orientation, size, and active-plot assertions. Replace the
immediate-link assertion with explicit before/after lifecycle checks:

```python
envelope_view = envelope.getViewBox()
deviation_view = deviation.getViewBox()
assert deviation_view.linkedView(ViewBox.XAxis) is None
assert envelope_view.linkedView(ViewBox.YAxis) is None
assert deviation_view.linkedView(ViewBox.YAxis) is None

widget._finalize_plot_view_ranges_after_render()

assert deviation_view.linkedView(ViewBox.XAxis) is envelope_view
assert envelope_view.linkedView(ViewBox.YAxis) is None
assert deviation_view.linkedView(ViewBox.YAxis) is None
```

- [ ] **Step 3: Update the title/layout lifecycle assertions**

In `test_dual_golden_plot_titles_match_existing_layout_contract`, retain the
splitter and title assertions. Add the same pre-finalization `XAxis is None`
contract, invoke `_finalize_plot_view_ranges_after_render()`, and then assert the
deviation plot is X-linked to the envelope plot while both Y axes remain
unlinked.

- [ ] **Step 4: Run the two corrected tests**

Run the command from Step 1 again.

Expected: `2 passed`.

- [ ] **Step 5: Run the changed-test-module verification set**

Use a fresh, non-existing repository-local `.tmp` basetemp and run. The
containment check prevents an unsafe computed path, and the existence check
ensures pytest does not remove a pre-existing directory:

```powershell
$repoRoot = [System.IO.Path]::GetFullPath((Get-Location).Path)
$baseTemp = [System.IO.Path]::GetFullPath(
  (Join-Path $repoRoot ('.tmp\pytest-golden-contract-' + [Guid]::NewGuid().ToString('N')))
)
if (-not $baseTemp.StartsWith($repoRoot.TrimEnd('\') + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
  throw "Unsafe basetemp path: $baseTemp"
}
if (Test-Path -LiteralPath $baseTemp) {
  throw "Basetemp unexpectedly exists: $baseTemp"
}
$env:QT_QPA_PLATFORM = 'offscreen'
python -m pytest --no-header -q --tb=short --basetemp $baseTemp `
  unit_test/base/test_channel_calibration_store.py `
  unit_test/base/test_excel_result_exporter.py `
  unit_test/base/test_golden_sample_comparison.py `
  unit_test/base/test_golden_sample_export_payload.py `
  unit_test/ui/test_analysis_config_common_widgets.py `
  unit_test/ui/test_analysis_config_defaults.py `
  unit_test/ui/test_analysis_config_dialog_migration.py `
  unit_test/ui/test_golden_dual_plot_x_auto_range.py `
  unit_test/ui/test_golden_sample_plot_layout.py `
  unit_test/ui/test_golden_sample_plot_titles.py `
  unit_test/ui/test_manual_limit_runtime.py `
  unit_test/ui/test_plot_view_config.py `
  unit_test/ui/test_sequence_channel_calibration.py `
  unit_test/ui/test_sequence_import_runtime_refresh.py
```

Expected: the two golden-plot failures are resolved. The separately accepted
`test_spec_defaults_include_all_spectrum_modes` assertion may remain as the only
failure and must be reported without modifying SPEC constants, JSON defaults,
or that test.

- [ ] **Step 6: Verify scope and whitespace**

Run:

```powershell
git diff --check
git diff -- unit_test/ui/test_golden_sample_plot_layout.py unit_test/ui/test_golden_sample_plot_titles.py
git status --short
```

Expected: no whitespace errors; only the two specified tracked test files
differ from commit `1ccefa9034c571b18b437bcc2fbe5bde9b1ad8b0`, the
pre-implementation content checkpoint. ZIP and EXE files remain unstaged.

- [ ] **Step 7: Commit the implementation checkpoint**

Stage only the two specified test files and commit:

```powershell
git add -- unit_test/ui/test_golden_sample_plot_layout.py unit_test/ui/test_golden_sample_plot_titles.py
git diff --cached --check
git diff --cached -- unit_test/ui/test_golden_sample_plot_layout.py unit_test/ui/test_golden_sample_plot_titles.py
git commit -m "US20260804 synchronize golden plot test contract"
```

Expected: the commit contains no production file, SPEC-default file, ZIP, or
EXE changes.

## Review Requirements

- [ ] Run a paired `spec-code-reviewer` and `quality-code-reviewer` review over the
  implementation checkpoint range.
- [ ] Treat specification compliance as the gate.
- [ ] Reject any suggestion to alter production plotting behavior or the accepted
  SPEC-default mismatch because those changes are outside this plan.
- [ ] After the task review passes, run a final quality review over the complete
  implementation range.

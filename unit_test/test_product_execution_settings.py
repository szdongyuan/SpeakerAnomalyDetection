"""Optional project execution schema and draft-only dialog changes."""
import copy

import pytest
from PyQt5.QtWidgets import QDialog

from base.product_test.config_model import execution_settings, normalize_project
from base.product_test.config_validation import validate_structure
from ui.product_test.execution_settings_dialog import ExecutionSettingsDialog
from unit_test.product_config_fixtures import make_project
from unit_test.test_product_test_project_config_dialog import dialog


def test_old_project_defaults_without_mutating_input(tmp_path):
    project = make_project(tmp_path)
    project.pop("execution")
    assert not validate_structure(project)
    assert normalize_project(project)["execution"] == {"mode": "per_condition", "condition_gap_seconds": .5}
    assert "execution" not in project


@pytest.mark.parametrize("value", [None, {}, {"mode": "full_cycle"},
    {"mode": "unknown", "condition_gap_seconds": .5},
    *[{"mode": "full_cycle", "condition_gap_seconds": gap}
      for gap in (True, "0.5", -1, float("nan"), float("inf"), 86401, 10 ** 400, .0001)],
])
def test_explicit_invalid_settings_are_rejected(tmp_path, value):
    project = make_project(tmp_path)
    project["execution"] = value
    assert validate_structure(project)
    with pytest.raises(ValueError):
        execution_settings(project)


@pytest.mark.parametrize("mode,gap", [("per_condition", .5), ("full_cycle", 0), ("full_cycle", 2.125)])
def test_execution_settings_saved_and_reloaded(dialog, mode, gap):
    dialog.project_data["execution"] = dict(mode=mode, condition_gap_seconds=gap)
    assert dialog._save_project(close_dialog=False)
    assert dialog.manager.load_project(dialog.current_file)[1]["execution"] == dialog.project_data["execution"]


@pytest.mark.parametrize("accept", [False, True])
def test_popup_edits_only_draft_on_accept(dialog, monkeypatch, accept):
    before = copy.deepcopy(dialog.project_data)
    dialog._dirty = False

    def execute(popup):
        assert popup.gap.value() == .5
        assert popup.gap_options.isHidden()
        popup.full_cycle.setChecked(True)
        assert not popup.gap_options.isHidden()
        popup.gap.setValue(1.25)
        popup.per_condition.setChecked(True)
        assert popup.gap_options.isHidden()
        popup.full_cycle.setChecked(True)
        assert popup.gap.value() == 1.25
        return QDialog.Accepted if accept else QDialog.Rejected

    monkeypatch.setattr(ExecutionSettingsDialog, "exec_", execute)
    dialog._edit_execution()
    if accept:
        assert dialog.project_data["execution"] == dict(mode="full_cycle", condition_gap_seconds=1.25)
        assert dialog._dirty
        assert "1.25" in dialog.condition_editor.execution_summary.text()
    else:
        assert dialog.project_data == before and not dialog._dirty
    assert dialog.project_data["conditions"] == before["conditions"]

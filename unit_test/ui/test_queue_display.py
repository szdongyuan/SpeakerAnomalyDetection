"""Default queue labels must not change persisted condition references."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt5.QtWidgets import QComboBox

from base.config_deletion import ConfigDeletionService
from base.load_config import LoadUiConfig
from ui.operation_sequence import AnalysisModelSelect
from ui.product_test_program_config_dialog import ProductTestProgramConfigDialog
from ui.product_test_project_config_dialog import ProductTestProjectConfigDialog
from ui.queue_display import queue_display_name, queue_display_options
from unit_test.test_product_test_program_config_dialog import make_manager as make_program_manager
from unit_test.test_product_test_project_config_dialog import (
    make_manager as make_project_manager,
    make_queue_config,
    project_data,
)


@pytest.fixture
def default_queue(tmp_path, monkeypatch):
    monkeypatch.setattr("ui.queue_display.DEFAULT_DIR", str(tmp_path))
    monkeypatch.setattr("base.config_deletion.DEFAULT_DIR", str(tmp_path))
    path = tmp_path / "ui" / "ui_config" / "sequence_config.json"
    assert LoadUiConfig.save_data_to_json(make_queue_config(), str(path))
    return path


@pytest.mark.parametrize("kind", ["project", "program"])
@pytest.mark.parametrize("aliases,current", [
    (("sequence_config", "默认配置"), "sequence_config"),
    (("默认配置", "sequence_config"), "默认配置"),
    (("sequence_config",), "sequence_config"),
    (("默认配置",), "默认配置"),
])
def test_default_queue_display_preserves_saved_reference(
    qt_app, tmp_path, default_queue, kind, aliases, current
):
    manager = (make_project_manager if kind == "project" else make_program_manager)(tmp_path)
    registry = {name: str(default_queue) for name in aliases}
    registry["using_config_path"] = str(default_queue)
    assert LoadUiConfig.save_data_to_json(registry, manager.queue_registry_path)
    registry_before = Path(manager.queue_registry_path).read_bytes()
    queue_before = default_queue.read_bytes()
    condition = {"condition_name": "档位1", "trigger_state": "", "test_queue": current}
    edited = []
    if kind == "project":
        data = project_data(tmp_path, [condition])
        data["test_groups"] = data["test_groups"][:1]
        success, filename = manager.save_project(None, data)
        assert success, filename
        dialog = ProductTestProjectConfigDialog(manager, queue_editor_callback=edited.append)
        dialog._show_project(data, filename)
        combo, button = dialog._queue_controls_for_row(0)
    else:
        data = {"name": "测试产品", "sub_configs": [condition]}
        success, filename = manager.save_program(None, data)
        assert success, filename
        dialog = ProductTestProgramConfigDialog(manager, queue_editor_callback=edited.append)
        dialog._show_program(data, filename)
        combo = dialog._queue_combobox(0)
        button = dialog.program_table.cellWidget(0, 3).edit_button
    try:
        for _ in range(2):
            assert [combo.itemText(i) for i in range(combo.count())] == ["请选择", "默认配置"]
            assert combo.currentText() == "默认配置"
            assert combo.currentData() == current
            assert button.isEnabled()
            button.click()
            dialog._refresh_queue_options()
        assert edited == [str(default_queue), str(default_queue)]
        if kind == "project":
            collected = dialog.collect_project()
            saved_condition = collected["test_groups"][0]["test_conditions"][0]
            success, message = manager.save_project(filename, collected)
        else:
            collected = dialog.collect_program()
            saved_condition = collected["sub_configs"][0]
            success, message = manager.save_program(filename, collected)
        assert success, message
        assert saved_condition["test_queue"] == current
        persisted = json.loads(Path(manager.program_dir, filename).read_text(encoding="utf-8"))
        persisted_condition = (persisted["test_groups"][0]["test_conditions"][0]
                               if kind == "project" else persisted["sub_configs"][0])
        assert persisted_condition["test_queue"] == current
        assert Path(manager.queue_registry_path).read_bytes() == registry_before
        assert default_queue.read_bytes() == queue_before
        assert ConfigDeletionService(manager).list_targets("queue") == []
    finally:
        dialog._dirty = False
        dialog.close()
        dialog.deleteLater()


def test_new_selection_prefers_default_alias_without_rewriting_catalog(default_queue):
    catalog = {name: {"path": str(default_queue)} for name in ("sequence_config", "默认配置")}
    assert queue_display_options(catalog, list(catalog), "") == [("默认配置", "默认配置")]
    assert set(catalog) == {"sequence_config", "默认配置"}


def test_user_queue_with_same_filename_remains_separate(tmp_path, default_queue):
    user_path = tmp_path / "analysis_sequence_config" / "sequence_config.json"
    catalog = {"sequence_config": {"path": str(user_path)}, "默认配置": {"path": str(default_queue)}}
    assert queue_display_options(catalog, list(catalog), "sequence_config") == [
        ("sequence_config", "sequence_config"), ("默认配置", "默认配置")
    ]
    assert queue_display_name("sequence_config", str(user_path)) == "sequence_config"


def test_equivalent_default_paths_are_deduplicated(default_queue):
    alternate = str(default_queue.parent / ".." / "ui_config" / default_queue.name)
    catalog = {"sequence_config": {"path": alternate}, "默认配置": {"path": str(default_queue)}}
    assert queue_display_options(catalog, list(catalog), "sequence_config") == [
        ("默认配置", "sequence_config")
    ]


def test_unavailable_default_keeps_existing_reference(qt_app, default_queue):
    host = SimpleNamespace(
        queue_catalog={"sequence_config": {"path": str(default_queue), "available": False}},
        _available_queue_names=lambda: [],
    )
    combo = QComboBox()
    ProductTestProjectConfigDialog._populate_queue_combobox(host, combo, "sequence_config")
    assert combo.currentText() == "默认配置（不可用）"
    assert combo.currentData() == "sequence_config"


@pytest.mark.parametrize("registered", [False, True])
def test_editor_default_label_ignores_legacy_alias(default_queue, monkeypatch, registered):
    registry = {"sequence_config": str(default_queue)} if registered else {}
    monkeypatch.setattr(LoadUiConfig, "_load_sequence_config_registry", lambda: registry)
    host = SimpleNamespace(using_config_path=str(default_queue))
    assert AnalysisModelSelect._get_using_config_display_name(host) == "默认配置"

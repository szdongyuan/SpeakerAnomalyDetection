"""Default queue labels must not change persisted condition references."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt5.QtWidgets import QComboBox

from base.config_deletion import ConfigDeletionService
from base.load_config import LoadUiConfig
from ui.operation_sequence import AnalysisModelSelect
from ui.product_test_project_config_dialog import ProductTestProjectConfigDialog
from ui.queue_display import queue_display_name, queue_display_options
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


@pytest.mark.parametrize("kind", ["project"])
@pytest.mark.parametrize("aliases,current", [
    (("sequence_config", "默认配置"), "sequence_config"),
    (("默认配置", "sequence_config"), "默认配置"),
    (("sequence_config",), "sequence_config"),
    (("默认配置",), "默认配置"),
])
def test_default_queue_display_preserves_saved_reference(
    qt_app, tmp_path, default_queue, kind, aliases, current
):
    from unit_test.product_config_fixtures import make_project
    manager = make_project_manager(tmp_path)
    assert LoadUiConfig.save_data_to_json({alias: str(default_queue) for alias in aliases}, manager.queue_registry_path)
    data = make_project(tmp_path)
    data["conditions"][0]["test_queue"] = current
    dialog = ProductTestProjectConfigDialog(manager=manager)
    try:
        dialog._show_project(data, None)
        combo = dialog.condition_editor.table.cellWidget(0, 4).findChild(QComboBox)
        assert combo.currentData() == current
        assert combo.currentText() == "默认配置"
        assert dialog.collect_project()["conditions"][0]["test_queue"] == current
        assert dialog._save_project(close_dialog=False)
        assert manager.load_project(dialog.current_file)[1]["conditions"][0]["test_queue"] == current
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


def test_unavailable_default_keeps_existing_reference(qt_app, default_queue, tmp_path):
    from ui.product_test.condition_editor import ConditionEditor
    from unit_test.product_config_fixtures import make_project
    data = make_project(tmp_path)
    data["conditions"][0]["test_queue"] = "sequence_config"
    editor = ConditionEditor()
    editor.set_project(data, {"sequence_config": {"path": str(default_queue), "available": False}})
    combo = editor.table.cellWidget(0, 4).findChild(QComboBox)
    assert combo.currentText() == "默认配置（不可用）"
    assert combo.currentData() == "sequence_config"
    editor.deleteLater()


@pytest.mark.parametrize("registered", [False, True])
def test_editor_default_label_ignores_legacy_alias(default_queue, monkeypatch, registered):
    registry = {"sequence_config": str(default_queue)} if registered else {}
    monkeypatch.setattr(LoadUiConfig, "_load_sequence_config_registry", lambda: registry)
    host = SimpleNamespace(using_config_path=str(default_queue))
    assert AnalysisModelSelect._get_using_config_display_name(host) == "默认配置"

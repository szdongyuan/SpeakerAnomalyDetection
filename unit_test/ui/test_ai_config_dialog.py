from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

from PyQt5.QtWidgets import QDialog

from base.ai_model_catalog import ModelChoice, ModelCatalogError
from ui.operation_sequence import OptionList
from ui.ui_analysis_config.ai_config_dialog import AIConfigWindow
from base.load_config import LoadUiConfig
from base.analysis_config_validation import validate_sequence_config


def make_dialog(config, catalog):
    manager = SimpleNamespace(config={"AI 1": deepcopy(config)})
    manager.load_config = lambda: deepcopy(manager.config)
    manager.save_default_config = Mock(return_value=True)
    return AIConfigWindow(manager, "AI 1", model_catalog=catalog, available_channels=[0, 2])


def test_no_model_draft_can_be_saved_without_mutating_source(ui_qapp):
    dialog = make_dialog({"analysis_channel": 2}, Mock(list_models=lambda: ()))
    try:
        assert dialog.model_status_label.isHidden()
        assert dialog.analyse_model_combo_box.count() == 1
        dialog.on_click_ok_btn()
        assert dialog.get_accepted_config() == {"analysis_channel": 2, "analyse_model_name": ""}
        assert dialog.result() == QDialog.Accepted
        assert dialog.config_manager.config == {"AI 1": {"analysis_channel": 2}}
    finally:
        dialog.close()


def test_choices_selection_and_cancel_are_draft_only(ui_qapp):
    catalog = Mock(list_models=lambda: (ModelChoice("first", True), ModelChoice("second", True)))
    dialog = make_dialog({"analyse_model_name": "first", "analysis_channel": 0}, catalog)
    try:
        combo = dialog.analyse_model_combo_box
        combo.setCurrentIndex(combo.findData("second"))
        assert dialog.get_default_config()["analyse_model_name"] == "second"
        dialog.reject()
        assert dialog.config_manager.config["AI 1"]["analyse_model_name"] == "first"
        dialog.config_manager.save_default_config.assert_not_called()
    finally:
        dialog.close()


def test_unavailable_saved_model_is_retained_and_not_replaced(ui_qapp):
    dialog = make_dialog({"analyse_model_name": "old", "analysis_channel": 2},
                         Mock(list_models=lambda: (ModelChoice("new", True),)))
    try:
        assert dialog.get_default_config()["analyse_model_name"] == "old"
        assert "不可用" in dialog.analyse_model_combo_box.currentText()
        dialog.on_restore_default_btn_clicked()
        assert dialog.get_default_config()["analyse_model_name"] == "old"
    finally:
        dialog.close()


def test_catalog_failure_is_visible(ui_qapp):
    catalog = Mock()
    catalog.list_models.side_effect = ModelCatalogError("模型目录读取失败")
    dialog = make_dialog({}, catalog)
    try:
        assert "读取失败" in dialog.model_status_label.text()
    finally:
        dialog.close()


def test_queue_factory_opens_ai_dialog_without_model_framework(ui_qapp, monkeypatch):
    monkeypatch.setattr("ui.operation_sequence.AiModelCatalog", lambda *_: Mock(list_models=lambda: ()))
    host = SimpleNamespace(
        config=[SimpleNamespace(analysis_channel_basis="wav", mode="RECORD_ONLY")],
        wav_channels=[0, 1], mic_channels=[],
    )
    dialog = OptionList.create_config_dialog(host, None, SimpleNamespace(config={}), "AI 1", "AI", 100)
    try:
        assert isinstance(dialog, AIConfigWindow)
        assert dialog.get_default_config()["analyse_model_name"] == ""
    finally:
        dialog.close()


def test_new_queue_ai_defaults_work_without_default_config_file(ui_qapp, monkeypatch):
    monkeypatch.setattr(LoadUiConfig, "load_data_from_json", lambda _: (1, "missing"))
    sequence = SimpleNamespace(analysis_list={}, mode="RECORD_ONLY", analysis_channel_basis="wav")
    host = SimpleNamespace(config=[sequence], default_logger=Mock(), wav_channels=[0, 1])
    OptionList.get_item_default_config(host, "AI 分析 ", "AI 分析 1")
    assert sequence.analysis_list["AI 分析 1"] == {
        "type": "AI", "analysis_channel": 0, "analyse_model_name": "",
    }
    validate_sequence_config([{"seq1": {"analysis_list": sequence.analysis_list}}])


def test_save_default_preserves_selected_model_and_channel(ui_qapp, monkeypatch):
    dialog = make_dialog({"analysis_channel": 2}, Mock(list_models=lambda: (ModelChoice("model", True),)))
    try:
        monkeypatch.setattr(dialog, "show_default_save_popup", Mock())
        dialog.analyse_model_combo_box.setCurrentIndex(1)
        dialog.on_default_btn_clicked()
        dialog.config_manager.save_default_config.assert_called_once_with(
            "AI", {"analysis_channel": 2, "analyse_model_name": "model"},
        )
    finally:
        dialog.close()

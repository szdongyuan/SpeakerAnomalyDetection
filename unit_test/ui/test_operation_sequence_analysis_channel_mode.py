import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication, QTreeView, QWidget

from base.load_config import LoadUiConfig
from ui.operation_sequence import AnalysisModelSelect, OptionList
from ui.ui_analysis_config.ai_config_dialog import AIConfigWindow
from ui.ui_analysis_config.common_widgets import ChannelSelectorWidget, MultiChannelSelectorWidget
from ui.ui_analysis_config.fft_config_dialog import FftConfigWindow
from ui.ui_analysis_config.loudness_config_dialog import LoudnessConfigWindow
from ui.ui_analysis_config.spec_config_dialog import SpecConfigWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _ConfigManager:
    def __init__(self, name, analysis_channel):
        self.config = {name: {"analysis_channel": analysis_channel}}

    def load_config(self):
        return self.config


class _Logger:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def warning(self, message):
        self.warnings.append(message)

    def error(self, message):
        self.errors.append(message)


def _option(mode, channels=(0, 2, 7)):
    return SimpleNamespace(
        mic_channels=list(channels),
        config=[SimpleNamespace(mode=mode)],
        default_logger=_Logger(),
    )


def test_analysis_catalog_only_shows_first_release_analysis_types(qapp):
    container = QWidget()
    host = SimpleNamespace(analysis_list=QTreeView(container))
    container.setLayout(AnalysisModelSelect.create_analysis_list_layout(host))
    try:
        analysis_group = host.analysis_model.item(1)
        visible_entries = [
            analysis_group.child(row).text().strip()
            for row in range(analysis_group.rowCount())
        ]
        assert visible_entries == [
            "声压级 (SPL)", "频谱分析 (Spec)", "AI 分析",
            "频段能量 (FBA)", "快速傅里叶变换 (FFT)",
        ]
    finally:
        container.close()


def test_new_queue_defaults_to_automatic_analysis_enabled(qapp):
    container = QWidget()
    select_list = QWidget(container)
    select_list.config = []
    host = SimpleNamespace(select_list=select_list)
    container.setLayout(AnalysisModelSelect.create_select_list_layout(host))
    try:
        assert host.auto_analysis_box.isChecked()
    finally:
        container.close()


def test_saved_automatic_analysis_choice_remains_respected(qapp):
    container = QWidget()
    select_list = QWidget(container)
    select_list.config = [SimpleNamespace(auto_analysis=False)]
    host = SimpleNamespace(select_list=select_list)
    container.setLayout(AnalysisModelSelect.create_select_list_layout(host))
    try:
        assert not host.auto_analysis_box.isChecked()
    finally:
        container.close()


def test_new_spec_item_uses_code_defaults_when_default_file_is_missing(monkeypatch):
    monkeypatch.setattr(
        LoadUiConfig,
        "load_data_from_json",
        lambda _path: (1, "missing"),
    )
    option = SimpleNamespace(
        config=[SimpleNamespace(analysis_list={})],
        default_logger=_Logger(),
    )

    OptionList.get_item_default_config(
        option,
        "频谱分析 (Spec) ",
        "频谱分析 (Spec) 1",
    )

    config = option.config[0].analysis_list["频谱分析 (Spec) 1"]
    assert config == {**SpecConfigWindow.DEFAULT_CONFIG, "type": "Spec"}


def test_new_recorded_analysis_item_defaults_to_all_hardware_channels(monkeypatch):
    monkeypatch.setattr(
        LoadUiConfig,
        "load_data_from_json",
        lambda _path: (0, {"SPL": {}}),
    )
    option = SimpleNamespace(
        config=[SimpleNamespace(mode="RECORD_ONLY", analysis_list={})],
        mic_channels=[0, 2, 7],
        default_logger=_Logger(),
    )

    OptionList.get_item_default_config(
        option,
        "声压级 (SPL) ",
        "声压级 (SPL) 1",
    )

    config = option.config[0].analysis_list["声压级 (SPL) 1"]
    assert config["analysis_channel"] == 0
    assert config["analysis_channels"] == [0, 2, 7]


@pytest.mark.parametrize("analysis_type", ["SPL", "Spec", "FBA", "AI", "LP", "FFT", "LOUD"])
def test_record_only_routes_multiple_selected_channels(qapp, monkeypatch, analysis_type):
    monkeypatch.setattr(AIConfigWindow, "load_model_name_from_db", lambda _self: [])
    monkeypatch.setattr(AIConfigWindow, "cheack_model_list", lambda _self: None)
    option = _option("RECORD_ONLY")
    dialog = OptionList.create_config_dialog(
        option,
        None,
        _ConfigManager(analysis_type, 5),
        analysis_type,
        analysis_type,
        0,
    )

    assert isinstance(dialog.channel_selector, MultiChannelSelectorWidget)
    assert dialog.channel_selector.selected_channels() == [0, 2, 7]
    dialog.channel_selector.set_selected_channels([0, 7])
    saved = dialog.get_default_config()
    assert saved["analysis_channel"] == 0
    assert saved["analysis_channels"] == [0, 7]
    dialog.close()


@pytest.mark.parametrize("analysis_type", ["SPL", "Spec", "FBA"])
def test_import_audio_routes_unrestricted_channel_input(qapp, analysis_type):
    option = _option("IMPORT_AUDIO", channels=(0,))
    dialog = OptionList.create_config_dialog(
        option,
        None,
        _ConfigManager(analysis_type, 127),
        analysis_type,
        analysis_type,
        0,
    )

    assert dialog.channel_selector.spin_box.lineEdit().isReadOnly() is False
    assert dialog.channel_selector.current_channel() == 127
    dialog.close()


@pytest.mark.parametrize("analysis_type", ["SPL", "Spec", "FBA"])
def test_reopening_dialog_uses_current_acquisition_mode(qapp, analysis_type):
    option = _option("RECORD_ONLY", channels=(0, 2, 7))
    manager = _ConfigManager(analysis_type, 5)

    record_dialog = OptionList.create_config_dialog(
        option, None, manager, analysis_type, analysis_type, 0
    )
    assert isinstance(record_dialog.channel_selector, MultiChannelSelectorWidget)
    assert record_dialog.channel_selector.selected_channels() == [0, 2, 7]
    record_dialog.channel_selector.set_selected_channels([0, 2])
    manager.config[analysis_type] = record_dialog.get_default_config()
    record_dialog.close()

    option.config[0].mode = "IMPORT_AUDIO"
    import_dialog = OptionList.create_config_dialog(
        option, None, manager, analysis_type, analysis_type, 0
    )
    assert import_dialog.channel_selector.spin_box.lineEdit().isReadOnly() is False
    assert import_dialog.channel_selector.current_channel() == 0
    import_dialog.channel_selector.spin_box.setValue(6)
    manager.config[analysis_type] = import_dialog.get_default_config()
    assert manager.config[analysis_type]["analysis_channels"] == [0, 2]
    assert manager.config[analysis_type]["analysis_channel"] == 5
    import_dialog.close()

    option.config[0].mode = "RECORD_ONLY"
    second_record_dialog = OptionList.create_config_dialog(
        option, None, manager, analysis_type, analysis_type, 0
    )
    assert isinstance(second_record_dialog.channel_selector, MultiChannelSelectorWidget)
    assert second_record_dialog.channel_selector.selected_channels() == [0, 2]
    second_record_dialog.close()


@pytest.mark.parametrize("analysis_type", ["SPL", "Spec", "FBA"])
@pytest.mark.parametrize("mode", [None, "FUTURE_MODE"])
def test_unknown_mode_rejects_target_dialog_with_warning(
    qapp, analysis_type, mode
):
    option = _option(mode)

    dialog = OptionList.create_config_dialog(
        option,
        None,
        _ConfigManager(analysis_type, 0),
        analysis_type,
        analysis_type,
        0,
    )

    assert dialog is None
    assert len(option.default_logger.warnings) == 1
    assert analysis_type in option.default_logger.warnings[0]
    assert str(mode) in option.default_logger.warnings[0]


@pytest.mark.parametrize("analysis_type", ["SPL", "Spec", "FBA"])
def test_absent_mode_attribute_rejects_target_dialog_with_warning(
    qapp, analysis_type
):
    option = _option("RECORD_ONLY")
    delattr(option.config[0], "mode")

    dialog = OptionList.create_config_dialog(
        option,
        None,
        _ConfigManager(analysis_type, 0),
        analysis_type,
        analysis_type,
        0,
    )

    assert dialog is None
    assert len(option.default_logger.warnings) == 1
    assert analysis_type in option.default_logger.warnings[0]
    assert "None" in option.default_logger.warnings[0]


@pytest.mark.parametrize("analysis_type", ["SPL", "Spec", "FBA"])
def test_empty_acquisition_config_rejects_target_dialog_with_warning(
    qapp, analysis_type
):
    option = _option("RECORD_ONLY")
    option.config = []

    dialog = OptionList.create_config_dialog(
        option,
        None,
        _ConfigManager(analysis_type, 0),
        analysis_type,
        analysis_type,
        0,
    )

    assert dialog is None
    assert len(option.default_logger.warnings) == 1
    assert analysis_type in option.default_logger.warnings[0]
    assert "None" in option.default_logger.warnings[0]


@pytest.mark.parametrize(
    ("analysis_type", "dialog_type"),
    [("FFT", FftConfigWindow), ("LOUD", LoudnessConfigWindow)],
)
def test_non_target_dialogs_are_not_gated_by_acquisition_mode(
    qapp, analysis_type, dialog_type
):
    option = _option("FUTURE_MODE", channels=(0, 2))

    dialog = OptionList.create_config_dialog(
        option,
        None,
        _ConfigManager(analysis_type, 2),
        analysis_type,
        analysis_type,
        0,
    )

    assert isinstance(dialog, dialog_type)
    assert isinstance(dialog.channel_selector, ChannelSelectorWidget)
    assert dialog.channel_selector.current_channel() == 2
    assert option.default_logger.warnings == []
    dialog.close()

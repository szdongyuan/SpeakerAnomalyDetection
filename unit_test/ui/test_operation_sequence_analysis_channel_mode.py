import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication

from ui.operation_sequence import OptionList
from ui.ui_analysis_config.common_widgets import ChannelSelectorWidget
from ui.ui_analysis_config.fft_config_dialog import FftConfigWindow
from ui.ui_analysis_config.loudness_config_dialog import LoudnessConfigWindow


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

    def warning(self, message):
        self.warnings.append(message)


def _option(mode, channels=(0, 2, 7)):
    return SimpleNamespace(
        mic_channels=list(channels),
        config=[SimpleNamespace(mode=mode)],
        default_logger=_Logger(),
    )


@pytest.mark.parametrize("analysis_type", ["SPL", "Spec", "FBA"])
def test_record_only_routes_restricted_selected_channels(qapp, analysis_type):
    option = _option("RECORD_ONLY")
    dialog = OptionList.create_config_dialog(
        option,
        None,
        _ConfigManager(analysis_type, 5),
        analysis_type,
        analysis_type,
        0,
    )

    assert dialog.channel_selector.spin_box.lineEdit().isReadOnly() is True
    assert dialog.channel_selector.current_channel() == 0
    dialog.channel_selector.spin_box.stepBy(-1)
    assert dialog.channel_selector.current_channel() == 7
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
    assert record_dialog.channel_selector.spin_box.lineEdit().isReadOnly() is True
    assert record_dialog.channel_selector.current_channel() == 0
    record_dialog.close()

    option.config[0].mode = "IMPORT_AUDIO"
    import_dialog = OptionList.create_config_dialog(
        option, None, manager, analysis_type, analysis_type, 0
    )
    assert import_dialog.channel_selector.spin_box.lineEdit().isReadOnly() is False
    assert import_dialog.channel_selector.current_channel() == 5
    import_dialog.close()

    option.config[0].mode = "RECORD_ONLY"
    second_record_dialog = OptionList.create_config_dialog(
        option, None, manager, analysis_type, analysis_type, 0
    )
    assert second_record_dialog.channel_selector.spin_box.lineEdit().isReadOnly() is True
    assert second_record_dialog.channel_selector.current_channel() == 0
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

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication

from ui.ui_analysis_config.common_widgets import (
    AnalysisChannelSpinBoxWidget,
    ChannelSelectorWidget,
)
from ui.ui_analysis_config.excel_config_dialog import ExcelConfigWindow
from ui.ui_analysis_config.fba_config_dialog import FbaConfigWindow
from ui.ui_analysis_config.lp_config_dialog import LPConfigWindow
from ui.ui_analysis_config.reference_spectrum_config_dialog import (
    ReferenceSpectrumConfigWindow,
)
from ui.ui_analysis_config.spec_config_dialog import SpecConfigWindow
from ui.ui_analysis_config.spl_config_dialog import SplConfigWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _ConfigManager:
    def __init__(self, config):
        self.config = config

    def load_config(self):
        return self.config

    def save_default_config(self, model_type, config):
        self.config[model_type] = config
        return True


@pytest.mark.parametrize(
    ("dialog_type", "model_type"),
    [
        (SplConfigWindow, "SPL"),
        (SpecConfigWindow, "Spec"),
        (FbaConfigWindow, "FBA"),
    ],
)
def test_only_in_scope_dialogs_use_analysis_channel_spinbox(qapp, dialog_type, model_type):
    dialog = dialog_type(
        _ConfigManager({model_type: {"analysis_channel": 2}}),
        model_type,
        available_channels=[0],
    )

    assert isinstance(dialog.channel_selector, AnalysisChannelSpinBoxWidget)
    assert dialog.channel_selector.spin_box.lineEdit().isReadOnly() is False
    assert dialog.channel_selector.spin_box.value() == 3
    dialog.close()


@pytest.mark.parametrize(
    ("dialog_type", "model_type"),
    [
        (SplConfigWindow, "SPL"),
        (SpecConfigWindow, "Spec"),
        (FbaConfigWindow, "FBA"),
    ],
)
def test_in_scope_dialogs_forward_restricted_channel_mode(
    qapp,
    dialog_type,
    model_type,
):
    dialog = dialog_type(
        _ConfigManager({model_type: {"analysis_channel": 5}}),
        model_type,
        available_channels=[0, 2, 7],
        restrict_analysis_channel=True,
    )

    assert dialog.channel_selector.spin_box.lineEdit().isReadOnly() is True
    assert dialog.channel_selector.current_channel() == 0
    dialog.channel_selector.spin_box.stepBy(-1)
    assert dialog.channel_selector.current_channel() == 7
    dialog.close()


def test_splf_ignores_analysis_channel_restriction(qapp):
    dialog = SplConfigWindow(
        _ConfigManager({"SPLF": {"analysis_channel": 2}}),
        "SPLF",
        available_channels=[0, 2, 7],
        restrict_analysis_channel=True,
    )

    assert isinstance(dialog.channel_selector, ChannelSelectorWidget)
    assert not isinstance(dialog.channel_selector, AnalysisChannelSpinBoxWidget)
    assert dialog.channel_selector.current_channel() == 2
    dialog.close()


def test_lp_retains_combo_channel_selector_and_serialization(qapp):
    dialog = LPConfigWindow(
        _ConfigManager({"LP": {"analysis_channel": 2}}),
        "LP",
        available_channels=[0, 2],
    )

    assert isinstance(dialog.channel_selector, ChannelSelectorWidget)
    assert not isinstance(dialog.channel_selector, AnalysisChannelSpinBoxWidget)
    assert dialog.get_default_config()["analysis_channel"] == 2
    dialog.close()


@pytest.mark.parametrize(
    ("persisted_channel", "available_channels", "expected"),
    [
        ("1.5", [0, 1], 0),
        (128, [0, 128], 128),
    ],
)
def test_lp_retains_legacy_channel_coercion(
    qapp,
    persisted_channel,
    available_channels,
    expected,
):
    dialog = LPConfigWindow(
        _ConfigManager({"LP": {"analysis_channel": persisted_channel}}),
        "LP",
        available_channels=available_channels,
    )

    assert dialog.channel_selector.current_channel() == expected
    assert dialog.get_default_config()["analysis_channel"] == expected
    dialog.close()


@pytest.mark.parametrize(
    ("dialog_type", "model_type", "args"),
    [
        (ReferenceSpectrumConfigWindow, "RSC", ([0, 1],)),
        (ExcelConfigWindow, "Excel", ()),
    ],
)
def test_rsc_and_excel_gain_no_analysis_channel_field(qapp, dialog_type, model_type, args):
    dialog = dialog_type(_ConfigManager({model_type: {}}), model_type, *args)

    assert dialog.findChildren(AnalysisChannelSpinBoxWidget) == []
    assert "analysis_channel" not in dialog.get_default_config()
    dialog.close()

"""Confirmed analysis settings are read without re-entering button handlers."""

from copy import deepcopy
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt5.QtWidgets import QDialog, QMessageBox, QWidget

from ui.custom_ui_widget.widgets import MessageBox
from ui.operation_sequence import OptionList
from ui.ui_analysis_config.common_widgets import SemanticAnalysisConfigDialogBase


KINDS = ("AI", "SPL", "FFT", "FBA", "Spec", "RSC", "LP", "LOUD")


@pytest.fixture
def make_dialog(ui_qapp, monkeypatch, tmp_path):
    dialogs = []
    monkeypatch.setattr("ui.operation_sequence.AiModelCatalog", lambda *_: Mock(list_models=lambda: ()))

    def create(kind):
        name = f"{kind} 1"
        manager = SimpleNamespace(config={name: {"type": kind, "analysis_channel": 0}})
        manager.load_config = lambda: deepcopy(manager.config)
        manager.save_default_config = Mock(return_value=True)
        source = SimpleNamespace(
            config=[SimpleNamespace(mode="RECORD_ONLY", analysis_channel_basis="wav")],
            wav_channels=[0, 1], mic_channels=[],
        )
        dialog = OptionList.create_config_dialog(source, None, manager, name, kind, 48000)
        dialogs.append(dialog)
        if kind == "RSC":
            reference = tmp_path / "reference.wav"
            reference.write_bytes(b"reference fixture")
            dialog.reference_source_path_edit.setText(str(reference))
            dialog._update_action_button_states()
            # Reference generation is a separate I/O boundary; validation still runs.
            monkeypatch.setattr(dialog, "_ensure_reference_data_ready", Mock(return_value=True))
        return name, manager, dialog

    yield create
    for dialog in dialogs:
        dialog.close()
        dialog.deleteLater()


def test_confirmed_snapshot_is_available_in_signal_and_independent_of_mutations(ui_qapp):
    dialog = SemanticAnalysisConfigDialogBase()
    config = {"channels": [0, 1], "display": {"curves": ["loudness_time"]}}
    expected = deepcopy(config)
    observed = []
    dialog.accepted.connect(lambda: observed.append(dialog.get_accepted_config()))
    try:
        assert dialog.get_accepted_config() is None
        dialog.accept_config(config)
        assert observed == [expected]
        config["channels"].append(2)
        observed[0]["display"]["curves"].clear()
        assert dialog.get_accepted_config() == expected
        dialog.reject()
        assert dialog.get_accepted_config() is None
    finally:
        dialog.close()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("confirm", [True, False], ids=["confirm", "cancel"])
def test_queue_reads_confirmed_config_once_and_cancel_preserves_source(
    make_dialog, monkeypatch, kind, confirm,
):
    name, manager, dialog = make_dialog(kind)
    original = deepcopy(manager.config)
    callback = Mock(wraps=dialog._ok_callback)
    monkeypatch.setattr(dialog, "_ok_callback", callback)
    accepted = Mock()
    dialog.accepted.connect(accepted)
    host = QWidget()
    host.config = [SimpleNamespace(
        name="recording", display_sequence=[name], analysis_list=deepcopy(original),
    )]
    host.signal_len = 48000
    host.create_config_dialog = lambda *_: dialog
    host._notify_config_changed = Mock()
    host.add_config = MethodType(OptionList.add_config, host)
    monkeypatch.setattr("ui.operation_sequence.ConfigManager", lambda _: manager)
    expected = []

    def execute_dialog():
        if confirm:
            dialog.semantic_ok_btn.click()
            assert dialog.result() == QDialog.Accepted
            expected.append(dialog.get_accepted_config())
        else:
            dialog.semantic_cancel_btn.click()
            assert dialog.result() == QDialog.Rejected
        assert manager.config == original
        # Reading an accepted result must not collect or validate controls again.
        monkeypatch.setattr(dialog, "get_default_config", Mock(side_effect=AssertionError("collected twice")))
        return dialog.result()

    monkeypatch.setattr(dialog, "exec_", execute_dialog)
    try:
        OptionList.show_dialog(host, name)
        if confirm:
            callback.assert_called_once_with()
            accepted.assert_called_once_with()
            assert host.config[0].analysis_list[name] == {**original[name], **expected[0]}
            host._notify_config_changed.assert_called_once_with()
            if kind == "RSC":
                dialog._ensure_reference_data_ready.assert_called_once_with(show_error_message=True)
        else:
            callback.assert_not_called()
            accepted.assert_not_called()
            assert dialog.get_accepted_config() is None
            assert host.config[0].analysis_list == original
            host._notify_config_changed.assert_not_called()
    finally:
        host.close()
        host.deleteLater()


@pytest.mark.parametrize("kind", ("SPL", "FFT", "FBA", "Spec", "RSC", "LOUD"))
def test_invalid_settings_do_not_close_or_publish_config(make_dialog, monkeypatch, kind):
    _, manager, dialog = make_dialog(kind)
    original = deepcopy(manager.config)
    warnings = Mock()
    monkeypatch.setattr(MessageBox, "warning", warnings)
    monkeypatch.setattr(QMessageBox, "warning", warnings)
    if kind == "SPL":
        dialog.threshold_widget.limit_checkbox.setChecked(True)
        dialog.limit_metric_combo.setCurrentIndex(dialog.limit_metric_combo.findData("overall_spl"))
        dialog.scalar_upper_check.setChecked(True)
        dialog.scalar_lower_check.setChecked(True)
        dialog.scalar_upper_spin.setValue(10)
        dialog.scalar_lower_spin.setValue(20)
    elif kind == "FFT":
        dialog.focus_checkbox.setChecked(True)
        dialog.focus_min_spin.setValue(100)
        dialog.focus_max_spin.setValue(100)
    elif kind == "FBA":
        dialog.f_min_spin.setValue(100)
        dialog.f_max_spin.setValue(100)
    elif kind == "Spec":
        dialog.custom_limit_checkbox.setChecked(True)
        dialog.top_limit_spinbox.setValue(10)
        dialog.bottom_limit_spinbox.setValue(20)
    elif kind == "RSC":
        dialog.custom_band_checkbox.setChecked(True)
        dialog.start_freq_spinbox.setValue(100)
        dialog.end_freq_spinbox.setValue(100)
    else:
        panel = dialog.panel
        panel.limit_checked_box.setChecked(True)
        panel.limit_metric_combo.setCurrentIndex(panel.limit_metric_combo.findData("steady_state_average"))
        panel.upper_enabled_box.setChecked(True)
        panel.lower_enabled_box.setChecked(True)
        panel.upper_spin.setValue(10)
        panel.lower_spin.setValue(20)
    dialog.show()
    dialog.semantic_ok_btn.click()
    assert warnings.call_count == 1
    assert dialog.isVisible()
    assert dialog.result() == QDialog.Rejected
    assert dialog.get_accepted_config() is None
    assert manager.config == original

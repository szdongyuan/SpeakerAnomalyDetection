"""PR2 admission regressions through real Qt buttons and production methods."""
from types import SimpleNamespace, MethodType
import pytest
from PyQt5.QtWidgets import QPushButton
from unit_test.ui.test_ve3668n_hardware import controls, main_window_harness, ControlledDiscovery
from unit_test.base.ve3668n_fakes import device_info
from ui.sequence.sequence_widget_recording_process_ops import SequenceWidgetRecordingProcessOpsMixin as RecordingOps
from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin as UiOps


def attach_real_start_button(window):
    sequence = window.sequence_window
    sequence.player_btn = QPushButton(window)
    sequence.sequence_config = {"configured": True}
    sequence.recording_bridge = window.recording_bridge
    sequence.ve_prewarm_lifetime = window.ve_prewarm_lifetime
    sequence._can_start_recording_workflow = MethodType(RecordingOps._can_start_recording_workflow, sequence)
    sequence.update_player_btn_is_paused = MethodType(UiOps.update_player_btn_is_paused, sequence)
    sequence.update_player_btn_is_paused()
    return sequence


def test_legacy_constructor_shares_application_lifetime(main_window_harness, controls, monkeypatch, ui_qapp):
    monkeypatch.setattr(ui_qapp, "_ve_prewarm_lifetime", None, raising=False)
    window_type = main_window_harness.namespace["MainWindow"]
    windows = []
    try:
        for _ in range(2):
            windows.append(window_type(recording_bridge=main_window_harness.bridge,
                ve_profile_store=controls.profiles, ve_calibration_store=controls.calibrations,
                discovery_factory=ControlledDiscovery,
                hardware_selection_path=controls.path))
        assert windows[0].ve_prewarm_lifetime is windows[1].ve_prewarm_lifetime
        selected = ("vkinging", "legacy", (1,), 51200)
        assert windows[0].ve_prewarm_lifetime.claim("legacy", selected)
        assert not windows[1].ve_prewarm_lifetime.claim("again", selected)
    finally:
        for window in windows:
            window._close_ve_discovery()
            window.close()


@pytest.mark.parametrize("busy", [False, True])
def test_config_close_recalculates_real_start_button(main_window_harness, monkeypatch, busy):
    window = main_window_harness.create()
    sequence = attach_real_start_button(window)
    window.recording_bridge.service.can_start_recording = not busy
    class Dialog:
        def __init__(self, *args, **kwargs):
            self.programs_changed = SimpleNamespace(connect=lambda callback: None)
        def exec(self):
            sequence.update_player_btn_is_paused()
            assert not sequence.player_btn.isEnabled()
    sequence.on_product_test_program_updated = lambda: None
    monkeypatch.setitem(main_window_harness.namespace, "ProductTestProjectConfigDialog", Dialog)
    window.on_product_test_program_config()
    assert not sequence._product_test_program_config_dialog_open
    assert sequence.player_btn.isEnabled() is (not busy)


def test_prewarm_terminal_recovers_real_start_button(main_window_harness):
    from unit_test.ui.test_ve3668n_hardware import _prewarm_completion
    window = main_window_harness.create()
    window.mic = device_info()
    sequence = attach_real_start_button(window)
    sequence.mic = window.mic
    window.recording_bridge.prewarm_status = "accepted"
    assert sequence.player_btn.isEnabled()
    assert window._try_start_ve_prewarm(window.mic, window.mic_channels, "test") == "accepted"
    assert not sequence.player_btn.isEnabled()
    request = window.recording_bridge.prewarm_calls[-1]
    window.recording_bridge.prewarm_callback(_prewarm_completion(request))
    assert sequence.player_btn.isEnabled()

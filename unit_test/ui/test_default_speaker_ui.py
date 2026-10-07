"""Default output UI policy, using real Qt widgets and fake hardware."""
import json
from copy import deepcopy
import pytest
from PyQt5.QtWidgets import QDialog, QGroupBox, QTableView
from ui import hardware_window as hardware_ui
from unit_test.ui.test_ve3668n_hardware import controls, confirm, main_window_harness
from unit_test.base.ve3668n_fakes import device_info
from unit_test.ui.test_ve3668n_recording import host_factory
from base import hardware_selection


def test_hardware_has_only_input_tables(controls, ui_qapp):
    controller = controls.create()
    view = controller.view
    view.show()
    ui_qapp.processEvents()
    assert view.mic_device_table.isVisible() and view.mic_channel_table.isVisible()
    assert min(view.mic_device_table.width(), view.mic_channel_table.width()) > 250
    assert not hasattr(view, "speaker_device_table")
    assert len(view.findChildren(QTableView)) == 2
    assert all("扬声器" not in group.title() for group in view.findChildren(QGroupBox))


@pytest.mark.parametrize("ve", [False, True])
def test_no_output_can_submit_input(controls, ui_qapp, monkeypatch, ve):
    monkeypatch.setattr(controls.audio.sdm, "get_default_device", lambda *a, **k: (0, None))
    mic = device_info(physical_channels=[1]) if ve else controls.old_mic
    if ve:
        controls.profiles.set_sample_rate(mic, 51200, controls.calibrations)
    profile_before = controls.profiles.path.read_bytes() if ve else None
    controller = controls.create(hardware_ui.HardwareSelectionState(api_name="MME", mic_device=mic, mic_channels=[1]))
    if ve:
        confirm(controller, ui_qapp, mic)
    controller.view.ok_btn.click()
    assert controller.view.result() == QDialog.Accepted
    monkeypatch.setattr(controller.view, "on_exec", lambda: QDialog.Accepted)
    accepted, output, output_channels, selected, channels = controller.on_exec()
    assert accepted and output is None and output_channels == []
    assert selected["name"] == mic["name"] and channels == [1]
    saved = json.loads(controls.path.read_text())
    assert "speaker_name" not in saved and "speaker_channels" not in saved
    assert controls.defaults.pair.writes == ([] if ve else [(0, mic["index"])])
    if ve:
        assert controls.profiles.path.read_bytes() == profile_before


@pytest.mark.parametrize("ve", [False, True])
def test_save_failure_keeps_dialog_unaccepted_and_committed_input(controls, ui_qapp, monkeypatch, ve):
    hardware_selection.save_if_changed(controls.old_mic, None, [1], [], path=controls.path)
    before = controls.path.read_bytes()
    initial = hardware_ui.HardwareSelectionState(api_name="MME", mic_device=controls.old_mic, mic_channels=[1], speaker_device=controls.speaker)
    controller = controls.create(initial)
    if ve:
        controller.view.driver_combo.setCurrentText("vkinging")
        confirm(controller, ui_qapp, device_info(physical_channels=[1]))
        controller.view.mic_device_table.model().item(0).setCheckState(2)
        controller.view.mic_channel_table.model().item(0).setCheckState(2)
    else:
        controller.view.mic_channel_table.model().item(0).setCheckState(2)
    def fail(*args, **kwargs):
        raise OSError("configuration write rejected")
    monkeypatch.setattr(hardware_selection, "_atomic_write_json", fail)
    controller.view.ok_btn.click()
    assert controller.view.result() != QDialog.Accepted
    assert controls.path.read_bytes() == before
    assert initial.mic_device == controls.old_mic and initial.mic_channels == [1]
    assert controls.defaults.pair.writes == [] and controls.warnings
    monkeypatch.setattr(controller.view, "on_exec", lambda: QDialog.Rejected)
    assert controller.on_exec()[3:] == (controls.old_mic, [1])


def test_main_status_has_only_input(controls, main_window_harness):
    window = main_window_harness.create()
    window.update_statusbar()
    text = window.device_label.text() + window.device_label.toolTip()
    assert "扬声器" not in text and "输出" not in text and "Old output" not in text



@pytest.mark.parametrize("legacy_output", [None, {"name": "stale output", "hostapi": 1}])
def test_capture_ignores_absent_or_cross_api_legacy_output(host_factory, controls, monkeypatch, legacy_output):
    from ui.sequence import sequence_widget_analysis_ops as analysis
    host = host_factory()
    host.mic, host.mic_channels = controls.old_mic, [1]
    host.refresh_channel_windows()
    original = analysis.LoadUiConfig.get_rec_and_play_dict_base_sequence_dict
    def with_legacy_output(*args):
        playback, record = original(*args)
        record.update(input_device=controls.old_mic, output_device=legacy_output)
        return playback, record
    monkeypatch.setattr(analysis.LoadUiConfig, "get_rec_and_play_dict_base_sequence_dict", with_legacy_output)
    def no_output_lookup(*args, **kwargs):
        raise AssertionError("Pure capture must not query output")
    monkeypatch.setattr(controls.audio.sdm, "get_default_device", no_output_lookup)
    recorded, rate = host.reset_work_pram("not_labeled")
    assert recorded["device"] == controls.old_mic
    assert host._recording_input_channels == (1,)
    assert rate == recorded["sample_rate"]


def test_recording_editor_ignores_legacy_output_without_query(controls, monkeypatch):
    from ui.acquisition_config_window import RecordConfigWindow, SoundDeviceManager
    def unexpected(*args, **kwargs):
        raise AssertionError("Input editor must not resolve output")
    monkeypatch.setattr(SoundDeviceManager, "get_default_device", unexpected)
    view = RecordConfigWindow({}, mic=controls.old_mic)
    assert not hasattr(view, "speaker")
    view.close()


def test_main_driver_comes_from_input_and_does_not_propagate_output(controls, main_window_harness):
    window = main_window_harness.create()
    window.mic, window.mic_channels = controls.old_mic, [1]
    window.speaker = controls.audio.asio_speaker  # Simulate a legacy caller's stale attribute.
    calls = []
    def cancel(**kwargs):
        calls.append(kwargs)
        return False, None, [], window.mic, window.mic_channels
    main_window_harness.namespace["open_hardware_selection_window"] = cancel
    window.on_hardware_window_init()
    assert calls[0]["driver"] == "MME"
    assert "speaker_device" not in calls[0] and "speaker_channels" not in calls[0]


@pytest.mark.parametrize("ve", [False, True])
def test_failed_dialog_save_does_not_publish_draft_to_main(controls, main_window_harness, ui_qapp, monkeypatch, ve):
    window = main_window_harness.create()
    window.mic, window.mic_channels = controls.old_mic, [1]
    window.sequence_window.mic, window.sequence_window.mic_channels = controls.old_mic, [1]
    before = controls.path.read_bytes()
    previous = deepcopy(window.mic)
    def reject_write(*args):
        return False
    monkeypatch.setattr(hardware_selection, "_atomic_write_json", reject_write)
    def interact(controller):
        if ve:
            controller.view.driver_combo.setCurrentText("vkinging")
            confirm(controller, ui_qapp, device_info(physical_channels=[1]))
            controller.view.mic_device_table.model().item(0).setCheckState(2)
            controller.view.mic_channel_table.model().item(0).setCheckState(2)
        else:
            controller.view.mic_channel_table.model().item(0).setCheckState(2)
        controller.view.ok_btn.click()
        assert controller.view.result() != QDialog.Accepted
        return False, None, [], controller._initial_state.mic_device, controller._initial_state.mic_channels
    monkeypatch.setattr(hardware_ui.HardwareSelectionController, "on_exec", interact)
    window.on_hardware_window_init()
    assert window.mic == window.sequence_window.mic == previous
    assert window.mic_channels == window.sequence_window.mic_channels == [1]
    assert controls.path.read_bytes() == before


def test_input_calibration_starts_without_output(controls, monkeypatch):
    from unittest.mock import Mock
    from ui import calibration_window as calibration
    from unit_test.ui.test_recording_process_integration import CapturingBridge
    from base.sound_device_manager import SoundDeviceManager
    monkeypatch.setattr(calibration, "load_mic_channel_v2pa_factors", lambda *a, **k: {})
    queries = []
    monkeypatch.setattr(SoundDeviceManager, "get_default_device", lambda *a, **k: queries.append(a) or (0, None))
    bridge = CapturingBridge()
    bridge.service.cancel = Mock()
    bridge.shutting_down = Mock()
    view = calibration.InputCalibration(controls.old_mic, [1], recording_bridge=bridge)
    try:
        assert view.clicked_calibration()
        assert bridge.request.device["index"] == controls.old_mic["index"]
        assert bridge.request.channels == (1,)
        assert queries == []
    finally:
        view.cancel_calibration()
        view.close()


def test_public_cancel_ignores_malformed_legacy_output_arguments(controls, monkeypatch):
    from unit_test.ui.test_ve3668n_hardware import ControlledDiscovery
    monkeypatch.setattr(hardware_ui.HardwareSelectionView, "on_exec", lambda self: QDialog.Rejected)
    result = hardware_ui.open_hardware_selection_window(
        driver="MME", speaker_device=object(), speaker_channels=object(),
        mic_device=controls.old_mic, mic_channels=[1],
        profile_store=controls.profiles, calibration_store=controls.calibrations,
        discovery_factory=ControlledDiscovery, selection_path=controls.path)
    assert result == (False, controls.speaker, [], controls.old_mic, [1])
    assert not controls.path.exists() and controls.defaults.pair.writes == []

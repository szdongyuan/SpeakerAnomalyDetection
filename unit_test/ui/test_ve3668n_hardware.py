"""Actual Qt widgets, controlled discovery, and isolated stores; never native I/O."""
from copy import deepcopy
import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
import threading
import json

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QComboBox
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QMainWindow, QAction, QApplication, QLabel
from PyQt5.QtTest import QTest

from ui import hardware_window as hardware_ui
from base.ve3668n_discovery import DiscoveryEvent, DiscoveryResult
from base.ve3668n_stores import VEInputProfileStore, VECalibrationStore
from base.ve3668n_input import validate_device_snapshot
from base.ve3668n_prewarm_lifetime import VePrewarmLifetime
from base.recording_process_protocol import VeLifecycleCounts
from base.recording_service import VePrewarmCompletion
from unit_test.base.ve3668n_fakes import device_info
from unit_test.base.test_ve3668n_stores import save_measurement
from unit_test.base.test_ve3668n_hardware_selection import fake_soundcards
from unit_test.ui.test_ve3668n_recording import host_factory, capture_audio, finish_ve_capture
from ui.acquisition_config_window import RecordConfigWindow
from consts.recording_preview_consts import (
    PREVIEW_TIME_MODE_CUMULATIVE,
    RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY,
)


def test_model_uses_only_discovered_ains_and_preserves_ve_on_output_change(ui_qapp):
    device = device_info(physical_channels=[7, 1])
    initial = hardware_ui.HardwareSelectionState(mic_device=device, mic_channels=[7, 1])
    model = hardware_ui.HardwareSelectionModel(initial)
    assert model.channels_for_device(device, "mic") == [7, 1]
    model.set_api("ASIO")
    assert model.state.mic_device == device
    assert model.state.mic_channels == [7, 1]
    assert initial.api_name is None


class ControlledDiscovery:
    def __init__(self, *, on_result):
        self.on_result = on_result
        self.generation = 0
        self.closed = False
        self.cancelled = False

    def start(self):
        return self.refresh()

    def refresh(self):
        self.generation += 1
        return self.generation

    def cancel(self):
        self.generation += 1
        self.cancelled = True

    def close(self):
        self.cancel()
        self.closed = True

    def deliver(self, devices=(), *, generation=None, diagnostics=(), released=True):
        event = DiscoveryEvent(self.generation if generation is None else generation,
            "completed" if devices else "unavailable", DiscoveryResult(tuple(devices), tuple(diagnostics)),
            None, 0, released)
        thread = threading.Thread(target=self.on_result, args=(event,))
        thread.start()
        thread.join()


@pytest.fixture
def controls(ui_qapp, tmp_path, monkeypatch):
    from base import vkinging_sdk
    audio = fake_soundcards(monkeypatch)
    native_calls = []
    def no_sdk(*args, **kwargs):
        native_calls.append((args, kwargs))
        raise AssertionError("Tests must use controlled discovery, never SDK loading")
    monkeypatch.setattr(vkinging_sdk, "VkDaqClient", no_sdk)
    monkeypatch.setattr(hardware_ui, "SoundDeviceManager", audio.sdm)
    warnings = []
    monkeypatch.setattr(hardware_ui.QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    profiles = VEInputProfileStore(tmp_path / "profiles.json")
    calibrations = VECalibrationStore(tmp_path / "calibrations.json")
    busy = SimpleNamespace(value=False)
    windows = []
    def create(initial=None):
        initial = initial or hardware_ui.HardwareSelectionState(mic_device=device_info(), mic_channels=[7, 1])
        model = hardware_ui.HardwareSelectionModel(initial)
        view = hardware_ui.HardwareSelectionView()
        controller = hardware_ui.HardwareSelectionController(model, view,
            profile_store=profiles, calibration_store=calibrations,
            discovery_factory=ControlledDiscovery, busy_check=lambda: busy.value,
            selection_path=tmp_path / "hardware.json")
        windows.append(view)
        return controller
    yield SimpleNamespace(create=create, profiles=profiles, calibrations=calibrations,
        busy=busy, warnings=warnings, path=tmp_path / "hardware.json", old_mic=audio.mic, speaker=audio.speaker,
        defaults=audio.defaults, audio=audio)
    for view in windows:
        view.reject()
    ui_qapp.processEvents()
    assert native_calls == []  # Also detects accidental calls if a boundary caught the error.


def confirm(controller, ui_qapp, *devices):
    controller.view.ve_controls.discovery.service.deliver(devices or (device_info(),))
    ui_qapp.processEvents()



def select_device(controller, machine_id):
    model = controller.view.mic_device_table.model()
    for row in range(model.rowCount()):
        item = model.item(row)
        if item.data(Qt.UserRole).get("machine_id") == machine_id:
            if item.checkState() == Qt.Checked:
                item.setCheckState(Qt.Unchecked)
            model.item(row).setCheckState(Qt.Checked)
            return
    raise AssertionError(f"No discovered device {machine_id}")


def test_driver_selects_vk_inventory_and_gates_channels(controls, ui_qapp):
    controller = controls.create(hardware_ui.HardwareSelectionState(api_name="MME",
        mic_device=controls.old_mic, mic_channels=[1], speaker_device=controls.speaker,
        speaker_channels=[1, 0]))
    view = controller.view
    assert view.driver_combo.findData("vkinging") >= 0
    assert view.findChildren(QComboBox) == [view.driver_combo]
    view.driver_combo.setCurrentIndex(view.driver_combo.findData("vkinging"))
    assert not view.mic_channel_table.isEnabled()
    confirm(controller, ui_qapp, device_info(), device_info(machine_id="two", name="Second"))
    table = view.mic_device_table.model()
    assert table.rowCount() == 2
    assert "test-machine-1" in table.item(0).text()
    assert "Second" in table.item(1).text() and "two" in table.item(1).text()
    assert view.mic_device_table.checked_payload() is None
    assert view.mic_channel_table.model().rowCount() == 0
    table.item(0).setCheckState(Qt.Checked)
    assert view.mic_channel_table.isEnabled()
    view.mic_channel_table.model().item(0).setCheckState(Qt.Checked)
    table.item(1).setCheckState(Qt.Checked)
    assert controller.model.state.mic_device["machine_id"] == "two"
    assert controller.model.state.mic_channels == []
    table.item(1).setCheckState(Qt.Unchecked)
    assert view.mic_channel_table.model().rowCount() == 0
    assert not view.mic_channel_table.isEnabled()
    assert controller.model.state.mic_channels == []
    assert not view.speaker_device_table.isEnabled()
    assert controller.model.state.speaker_device == controls.speaker
    assert controller.model.state.speaker_channels == [1, 0]


def test_refresh_discards_unsaved_device_and_cancel_returns_entry_snapshot(controls, ui_qapp, monkeypatch):
    initial = hardware_ui.HardwareSelectionState(api_name="MME", mic_device=controls.old_mic,
        mic_channels=[1], speaker_device=controls.speaker, speaker_channels=[1, 0])
    controller = controls.create(initial)
    view = controller.view
    view.driver_combo.setCurrentText("vkinging")
    confirm(controller, ui_qapp)
    select_device(controller, "test-machine-1")
    view.mic_channel_table.model().item(0).setCheckState(Qt.Checked)
    view.refresh_btn.click()
    assert controller.model.state.mic_device is None
    assert controller.model.state.mic_channels == []
    assert not view.mic_channel_table.isEnabled()
    confirm(controller, ui_qapp)
    assert view.mic_device_table.checked_payload() is None
    controller._on_ok_clicked()
    assert view.result() != QDialog.Accepted and controls.warnings
    monkeypatch.setattr(view, "on_exec", lambda: QDialog.Rejected)
    assert controller.on_exec() == (False, controls.speaker, [1, 0], controls.old_mic, [1])
    assert not controls.path.exists() and not controls.profiles.path.exists()


def test_busy_deferred_discovery_cannot_override_after_driver_switch(controls, ui_qapp):
    controller = controls.create(hardware_ui.HardwareSelectionState(api_name="MME",
        mic_device=controls.old_mic, mic_channels=[1], speaker_device=controls.speaker))
    view = controller.view
    view.driver_combo.setCurrentText("vkinging")
    controls.busy.value = True
    confirm(controller, ui_qapp)
    controls.busy.value = False
    view.driver_combo.setCurrentText("MME")
    controller._update_busy_state()
    assert_old_soundcard_visible(controller, controls)
    assert view.mic_channel_table.isEnabled()
    assert view.speaker_device_table.isEnabled()


@pytest.mark.parametrize("missing", [False, True])
def test_soundcard_refresh_updates_output_snapshot_or_clears_missing_device(controls, missing):
    controller = controls.create(hardware_ui.HardwareSelectionState(api_name="MME",
        mic_device=controls.old_mic, mic_channels=[1], speaker_device=controls.speaker,
        speaker_channels=[1, 0]))
    refreshed = {**controls.speaker, "index": 22}
    controls.audio.devices["MME"]["output"] = [] if missing else [refreshed]

    controller.view.refresh_btn.click()

    expected = None if missing else refreshed
    assert controller.view.speaker_device_table.checked_payload() == expected
    assert controller.model.state.speaker_device == expected
    assert controller.model.state.speaker_channels == ([] if missing else [1, 0])
    assert not controls.defaults.calls
    controller.view.ok_btn.click()
    if missing:
        assert controller.view.result() != QDialog.Accepted
        assert controls.warnings and not controls.defaults.calls
    else:
        assert controller.view.result() == QDialog.Accepted
        assert controls.defaults.calls == [(1, 22)]
    assert controls.speaker["index"] == 2


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("enter_from_soundcard", [False, True])
@pytest.mark.parametrize("switch_back", [False, True])
def test_vk_refresh_reconciles_output_identity_and_switchback_before_accept(
        controls, ui_qapp, missing, enter_from_soundcard, switch_back):
    from base import hardware_selection
    hardware_selection.save_if_changed(controls.old_mic, controls.speaker, [1], [1, 0],
        path=controls.path)
    initial = hardware_ui.HardwareSelectionState(api_name="MME",
        mic_device=controls.old_mic if enter_from_soundcard else device_info(),
        mic_channels=[1] if enter_from_soundcard else [7, 1],
        speaker_device=controls.speaker, speaker_channels=[1, 0])
    controller = controls.create(initial)
    view = controller.view
    view.driver_combo.setCurrentText("vkinging")
    confirm(controller, ui_qapp)
    refreshed = {**controls.speaker, "index": 22}
    controls.audio.devices["MME"]["output"] = [] if missing else [refreshed]

    view.refresh_btn.click()
    confirm(controller, ui_qapp)

    expected = None if missing else refreshed
    assert view.speaker_device_table.checked_payload() == expected
    assert controller.model.state.speaker_device == expected
    assert controller.model.state.speaker_channels == ([] if missing else [1, 0])
    assert not view.speaker_device_table.isEnabled()
    assert not controls.defaults.pair.writes and not controls.defaults.calls

    if switch_back:
        view.driver_combo.setCurrentText("MME")
        assert view.speaker_device_table.isEnabled()
        assert view.speaker_device_table.checked_payload() == expected
        assert controller.model.state.speaker_device == expected
        assert controller.model.state.mic_device == controls.old_mic
        assert controller.model.state.mic_channels == [1]
        view.driver_combo.setCurrentText("vkinging")
        confirm(controller, ui_qapp)
    select_device(controller, "test-machine-1")
    view.mic_channel_table.model().item(0).setCheckState(Qt.Checked)
    view.ok_btn.click()
    assert view.result() == QDialog.Accepted
    assert controls.defaults.pair.writes == ([] if missing else [(1, 22)])
    assert not controls.defaults.calls
    assert controls.speaker["index"] == 2


def test_saved_device_restores_only_after_discovery_without_profile_write(controls, ui_qapp):
    controller = controls.create()
    panel = controller.view.ve_controls
    assert controller.view.findChildren(QComboBox) == [controller.view.driver_combo]
    assert not controller.model.state.mic_device["available"]
    assert controller.model.state.mic_channels == []
    confirm(controller, ui_qapp)
    table = controller.view.mic_channel_table.model()
    assert [table.item(i).data(Qt.UserRole) for i in range(table.rowCount())] == [7, 1]
    controller._on_ok_clicked()
    assert controller.view.result() == QDialog.Accepted
    assert controller.model.state.mic_channels == [7, 1]
    assert controls.path.exists() and not controls.profiles.path.exists()
    assert panel.discovery.service.closed


@pytest.mark.parametrize("saved_vk", [False, True])
def test_empty_vk_scan_clears_rows_and_status_without_changing_saved_selection(controls, ui_qapp, saved_vk):
    from base import hardware_selection

    mic = device_info() if saved_vk else controls.old_mic
    channels = [7, 1] if saved_vk else [1]
    hardware_selection.save_if_changed(mic, controls.speaker, channels, [1, 0], path=controls.path)
    original = controls.path.read_bytes()
    controller = controls.create(hardware_ui.HardwareSelectionState(
        api_name="MME", mic_device=mic, mic_channels=channels,
        speaker_device=controls.speaker, speaker_channels=[1, 0]))
    view = controller.view
    view.driver_combo.setCurrentText("vkinging")
    assert view.ve_status_label.text()
    view.ve_controls.discovery.service.deliver()
    ui_qapp.processEvents()

    assert view.mic_device_table.model().rowCount() == 0
    assert view.mic_device_table.checked_payload() is None
    assert view.mic_channel_table.model().rowCount() == 0
    assert not view.mic_channel_table.isEnabled()
    assert controller.model.state.mic_channels == []
    assert view.ve_status_label.text() == ""
    assert controls.warnings == []
    assert controls.path.read_bytes() == original
    assert not controls.profiles.path.exists() and not controls.calibrations.path.exists()

    # A later scan restores the retained identity and channel order only if saved.
    view.refresh_btn.click()
    confirm(controller, ui_qapp, device_info(name="reconnected"))
    assert view.mic_device_table.model().rowCount() == 1
    assert controller.model.state.mic_channels == ([7, 1] if saved_vk else [])
    assert view.mic_channel_table.isEnabled() is saved_vk
    if saved_vk:
        assert view.mic_device_table.checked_payload()["name"] == "reconnected"
    assert controls.path.read_bytes() == original
    view.reject()
    assert controls.path.read_bytes() == original


@pytest.mark.parametrize("released", [False, True])
def test_empty_vk_scan_preserves_real_sdk_or_release_diagnostics(controls, ui_qapp, released):
    controller = controls.create()
    view = controller.view
    diagnostic = "fake SDK enumeration failure" if released else "fake SDK close failure"
    view.ve_controls.discovery.service.deliver(diagnostics=(diagnostic,), released=released)
    ui_qapp.processEvents()
    assert view.mic_device_table.model().rowCount() == 0
    assert view.mic_channel_table.model().rowCount() == 0
    assert not view.mic_channel_table.isEnabled()
    assert diagnostic in view.ve_status_label.text()
    assert not controls.path.exists()


def test_async_refresh_close_and_late_generations_are_ignored(controls, ui_qapp):
    controller = controls.create()
    panel = controller.view.ve_controls
    service = panel.discovery.service
    first = service.generation
    controller.refresh_and_render(try_restore=True)
    service.deliver([device_info(name="stale")], generation=first)
    ui_qapp.processEvents()
    assert not controller.model.state.mic_device["available"]
    confirm(controller, ui_qapp, device_info(name="current"))
    assert controller.model.state.mic_device["name"] == "current"
    controller.view.reject()
    service.deliver([device_info(name="after-close")])
    ui_qapp.processEvents()
    assert controller.model.state.mic_device["name"] == "current"
    assert service.closed and not controls.path.exists() and not controls.profiles.path.exists()


def test_channel_click_order_and_open_boundary_do_not_sort_ve(controls, ui_qapp, monkeypatch):
    device = device_info(physical_channels=list(range(8)))
    initial = hardware_ui.HardwareSelectionState(mic_device=device, mic_channels=[])
    controller = controls.create(initial)
    confirm(controller, ui_qapp, device)
    table = controller.view.mic_channel_table.model()
    table.item(7).setCheckState(Qt.Checked)
    table.item(1).setCheckState(Qt.Checked)
    controller._on_ok_clicked()
    assert controller.model.state.mic_channels == [7, 1]
    assert initial.mic_channels == []
    def inspect(self):
        assert self.model.state.mic_channels == []
        self.view.reject()
        return False, None, [], self._initial_state.mic_device, self._initial_state.mic_channels
    monkeypatch.setattr(hardware_ui.HardwareSelectionController, "on_exec", inspect)
    result = hardware_ui.open_hardware_selection_window(mic_device=device, mic_channels=[7, 1],
        profile_store=controls.profiles, calibration_store=controls.calibrations,
        discovery_factory=ControlledDiscovery, selection_path=controls.path)
    assert result[-1] == [7, 1]
    assert hardware_ui._normalize_channel_indices([7, 1]) == [1, 7]


def test_cancel_failure_and_ok_never_write_profile_or_calibration(controls, ui_qapp, monkeypatch):
    device = device_info()
    controls.profiles.set_sample_rate(device, 48000, controls.calibrations)
    save_measurement(controls.calibrations, device, 7)
    before_profile = controls.profiles.path.read_bytes()
    before_calibration = controls.calibrations.path.read_bytes()
    def forbidden(*args):
        raise AssertionError("Hardware selection must never write a profile")
    monkeypatch.setattr(controls.profiles, "set_sample_rate", forbidden)
    initial = hardware_ui.HardwareSelectionState(mic_device=device, mic_channels=[7, 1])
    controller = controls.create(initial)
    confirm(controller, ui_qapp)
    controller.view.reject()
    assert not controls.path.exists()
    controller = controls.create(initial)
    confirm(controller, ui_qapp)
    from base import hardware_selection
    write = hardware_selection._atomic_write_json
    monkeypatch.setattr(hardware_selection, "_atomic_write_json", lambda *args: False)
    controller._on_ok_clicked()
    assert controller.view.result() != QDialog.Accepted
    assert controls.warnings and "未保存" in controller.view.ve_status_label.text()
    assert not controls.path.exists()
    monkeypatch.setattr(hardware_selection, "_atomic_write_json", write)
    controller._on_ok_clicked()
    assert controller.view.result() == QDialog.Accepted
    assert controller.model.state.mic_device["input_config"] == device["input_config"]
    assert controls.profiles.path.read_bytes() == before_profile
    assert controls.calibrations.path.read_bytes() == before_calibration
    assert initial.mic_device == device


def test_switching_devices_keeps_discovery_snapshots_without_reading_profiles(controls, ui_qapp, monkeypatch):
    from unittest.mock import Mock

    one, two = device_info(), device_info(machine_id="two", name="Dev2")
    controls.profiles.set_sample_rate(one, 44100, controls.calibrations)
    controls.profiles.set_sample_rate(two, 48000, controls.calibrations)
    before = controls.profiles.path.read_bytes()
    load = Mock(wraps=controls.profiles.load)
    monkeypatch.setattr(controls.profiles, "load", load)
    controller = controls.create()
    confirm(controller, ui_qapp, one, two)
    assert controller.model.state.mic_device == validate_device_snapshot(one)
    select_device(controller, "two")
    assert controller.model.state.mic_device == validate_device_snapshot(two)
    assert controller.model.state.mic_channels == []
    select_device(controller, one["machine_id"])
    assert controller.model.state.mic_device == validate_device_snapshot(one)
    load.assert_not_called()
    assert controls.profiles.path.read_bytes() == before


def test_busy_after_dialog_open_disables_edits_and_rechecks_ok(controls, ui_qapp):
    controller = controls.create()
    confirm(controller, ui_qapp)
    controls.busy.value = True
    QTest.qWait(150)
    panel = controller.view.ve_controls
    for widget in (controller.view.mic_device_table,
                   controller.view.mic_channel_table, controller.view.driver_combo,
                   controller.view.ok_btn):
        assert not widget.isEnabled()
    controller._on_ok_clicked()
    assert controller.view.result() != QDialog.Accepted and not controls.path.exists()
    controls.busy.value = False
    QTest.qWait(150)
    assert controller.view.mic_channel_table.isEnabled()


def test_unreleased_discovery_is_unavailable_not_busy_forever(controls, ui_qapp):
    controller = controls.create()
    panel = controller.view.ve_controls
    panel.discovery.service.deliver([device_info()], released=False, diagnostics=("launch pending",))
    ui_qapp.processEvents()
    assert not controller.model.state.mic_device["available"]
    assert "launch pending" in controller.view.ve_status_label.text()
    controller._on_ok_clicked()
    assert controller.view.result() != QDialog.Accepted


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("failure", ["unreadable", "malformed", "invalid_rate"])
def test_bad_profile_does_not_disable_discovered_hardware_or_explicit_queue(
        controls, ui_qapp, host_factory, monkeypatch, restored, failure):
    from unittest.mock import Mock
    from ui.sequence.sequence_widget_config_ops import SequenceWidgetConfigOpsMixin

    host = host_factory()
    device = device_info()
    save_measurement(controls.calibrations, device, 7)
    if failure == "unreadable":
        load = Mock(side_effect=OSError("profile denied"))
    else:
        if failure == "malformed":
            controls.profiles.path.write_text("{", encoding="utf-8")
        else:
            controls.profiles.path.write_text(json.dumps({"schema_version": 1, "devices": {
                device["machine_id"]: {**device["input_config"], "sample_rate": 102401}}}), encoding="utf-8")
        load = Mock(wraps=controls.profiles.load)
    monkeypatch.setattr(controls.profiles, "load", load)
    before = controls.profiles.path.read_bytes()
    calibration_before = controls.calibrations.path.read_bytes()
    initial = (hardware_ui.HardwareSelectionState(
        mic_device=device_info(available=False, input_config=None), mic_channels=[7, 1])
        if restored else hardware_ui.HardwareSelectionState(
            api_name="MME", mic_device=controls.old_mic, mic_channels=[1]))
    controller = controls.create(initial)
    controller.view.driver_combo.setCurrentText("vkinging")
    confirm(controller, ui_qapp)
    if not restored:
        assert controller.view.mic_device_table.checked_payload() is None
        select_device(controller, device["machine_id"])
    assert controller.model.state.mic_device["available"]
    assert controller.view.mic_channel_table.isEnabled()
    assert controller.view.mic_channel_table.model().rowCount() == 2
    for row in range(2):
        controller.view.mic_channel_table.model().item(row).setCheckState(Qt.Checked)
    controller._on_ok_clicked()
    assert controller.view.result() == QDialog.Accepted
    assert controller.model.state.mic_channels == [7, 1]
    saved_selection = json.loads(controls.path.read_text(encoding="utf-8"))["input_selection"]
    assert saved_selection["machine_id"] == device["machine_id"]
    assert saved_selection["physical_channels"] == [7, 1]
    load.assert_not_called()
    assert controls.profiles.path.read_bytes() == before
    assert controls.calibrations.path.read_bytes() == calibration_before

    host.mic = deepcopy(controller.model.state.mic_device)
    host.mic_channels = list(controller.model.state.mic_channels)
    host.ve_profile_store, host.ve_calibration_store = controls.profiles, controls.calibrations
    host.refresh_channel_windows()
    detail = host.sequence_config[0]["seq1"]["acq"]["detail"]
    detail.update(sample_rate=96000, ve_range_index=5)
    host.judge_play_and_record()
    session = host._recording_process_session
    request = session.request
    assert request.sample_rate == 96000 and request.device["input_config"]["range_max"] == .1
    assert request.calibration_metadata["recorded_channels"][0]["v2pa_factor"] == 10
    _, audio = capture_audio(request)
    finish_ve_capture(host, session, audio)
    load.assert_not_called()
    assert host.mic["available"]
    assert controls.calibrations.get_factor(request.device, 7) == 10
    assert controls.calibrations.get_factor(request.device, 1) is None
    assert controls.calibrations.path.read_bytes() == calibration_before
    assert controls.profiles.path.read_bytes() == before

    # The bad profile still fails at the boundary that actually needs it.
    del detail["sample_rate"]
    with pytest.raises((OSError, ValueError)):
        SequenceWidgetConfigOpsMixin._current_ve_recording_device(host, host.mic)
    load.assert_called_once()
    assert host.mic["available"]
    assert controls.profiles.path.read_bytes() == before
    assert controls.calibrations.path.read_bytes() == calibration_before


@pytest.mark.parametrize("rate", [44100, 48000, 51200])
@pytest.mark.parametrize("monitor", [False, True])
def test_record_config_effective_rate_preserves_product_rate_and_ignores_monitor(ui_qapp, monkeypatch, rate, monitor):
    from base.sound_device_manager import SoundDeviceManager
    def forbidden(*args, **kwargs):
        raise AssertionError("No automatic output/mic fallback for VE")
    monkeypatch.setattr(SoundDeviceManager, "get_default_device", forbidden)
    device = device_info()
    device["input_config"]["sample_rate"] = rate
    product = {"sample_rate": 96000, "monitor_playback": monitor,
               "monitor_gain_db": 4.5, "use_streaming_recording": True,
               RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY: PREVIEW_TIME_MODE_CUMULATIVE}
    original = deepcopy(product)
    window = RecordConfigWindow(product, mic=device)
    assert window.samplerate_combo.currentText() == "96000"
    assert window.samplerate_combo.isEnabled() and window.samplerate_combo.isEditable()
    assert not hasattr(window, "ve_hint_label")
    assert not hasattr(window, "monitor_checkbox") and not hasattr(window, "monitor_gain_db_input")
    window.streaming_recording_checkbox.setChecked(False)
    window.on_click_ok_btn()
    assert window.final_data["sample_rate"] == 96000
    assert not any(key.startswith("monitor_") for key in window.final_data)
    assert window.final_data[RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY] == PREVIEW_TIME_MODE_CUMULATIVE
    assert product == original
    window.close()


@pytest.fixture
def main_window_harness(controls, monkeypatch):
    """Run complete MainWindow methods on Qt, isolating unrelated DB/login setup."""
    from base import hardware_selection
    source = Path(__file__).resolve().parents[2] / "main_window.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    klass = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
    namespace = {"QMainWindow": QMainWindow, "QAction": QAction, "QApplication": QApplication,
        "QPoint": QPoint, "SoundDeviceManager": hardware_ui.SoundDeviceManager,
        "QMessageBox": hardware_ui.QMessageBox, "restore_or_default": hardware_selection.restore_or_default,
        "save_if_changed": hardware_selection.save_if_changed,
        "open_hardware_selection_window": hardware_ui.open_hardware_selection_window}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[klass], type_ignores=[])), str(source), "exec"), namespace)
    MainWindow = namespace["MainWindow"]
    events = []
    def init_ui(window):
        events.append("sequence initialized")
        window.sequence_window = SimpleNamespace(player_status_flag=False, mic=window.mic,
            speaker=window.speaker, mic_channels=list(window.mic_channels), speaker_channels=[],
            update_v2pa_factor=lambda: events.append("legacy calibration refresh"),
            refresh_channel_windows=lambda: events.append("channels refreshed"))
        window.device_label, window.user_label = QLabel(window), QLabel(window)
    monkeypatch.setattr(MainWindow, "init_ui", init_ui)
    monkeypatch.setattr(MainWindow, "closeEvent", lambda window, event: event.accept())
    class OrderedDiscovery(ControlledDiscovery):
        def __init__(self, **kwargs):
            assert "sequence initialized" in events
            super().__init__(**kwargs)
    controls.path.write_text(json.dumps({"input_selection": {"schema_version": 1,
        "backend": "vkinging", "machine_id": "test-machine-1", "physical_channels": [7, 1]}}), encoding="utf-8")
    class Bridge:
        def __init__(self):
            self.service = SimpleNamespace(busy=False)
            self.hardware_busy = False
            self.release_calls = []
            self.release_callback = None
            self.prewarm_status = "busy"
            self.prewarm_calls = []
            self.prewarm_callback = None

        def release_ve(self, signature, callback):
            self.release_calls.append(signature)
            self.release_callback = callback
            return "pending"

        def prewarm_ve(self, request, callback):
            self.prewarm_calls.append(request)
            self.prewarm_callback = callback
            return self.prewarm_status

        @staticmethod
        def shutdown():
            return None

    bridge = Bridge()
    lifetime = VePrewarmLifetime()
    windows = []
    def create(*, recording_bridge=None, lifetime_instance=None):
        window = MainWindow(ve_prewarm_lifetime=lifetime_instance or lifetime,
            recording_bridge=recording_bridge or bridge, ve_profile_store=controls.profiles,
            ve_calibration_store=controls.calibrations, discovery_factory=OrderedDiscovery,
            hardware_selection_path=controls.path)
        windows.append(window)
        return window
    yield SimpleNamespace(create=create, bridge=bridge, lifetime=lifetime,
                          events=events, namespace=namespace)
    for window in windows:
        window._close_ve_discovery()
        window.close()


class _QtPrewarmService:
    """Service boundary fake used through the real RecordingServiceBridge."""

    def __init__(self, status="accepted"):
        self.status = status
        self.closed = threading.Event()
        self._closing = False
        self._worker = None
        self._capture_session = None
        self._pending_ve_release = None
        self._ownership_uncertain = False
        self.calls = []
        self.callback = None

    def prewarm_ve(self, request, callback):
        self.calls.append(request)
        self.callback = callback
        return self.status

    def release_ve(self, _signature, _callback):
        return "unchanged"

    def shutdown(self, callback=None):
        self._closing = True
        self.closed.set()
        if callback is not None:
            callback()


def _prewarm_completion(request, *, success=True, stage="completed", detail="",
                        ownership_safe=True):
    return VePrewarmCompletion(
        request.warmup_id, request.signature, success, stage, None, detail, (),
        VeLifecycleCounts(1, 1, 1, 1, 1, 1), ownership_safe)


def test_main_startup_is_async_and_shares_instance_stores(main_window_harness, controls, ui_qapp):
    window = main_window_harness.create()
    assert "mic" not in controls.audio.queries
    assert not window.mic["available"] and window.mic_channels == [7, 1]
    assert window.sequence_window.ve_profile_store is controls.profiles
    assert window.sequence_window.ve_calibration_store is controls.calibrations
    window.ve_discovery.service.deliver([device_info(name="reconnected")])
    ui_qapp.processEvents()
    assert window.mic["available"] and window.sequence_window.mic == window.mic
    assert window.mic["name"] == "reconnected"
    assert "test-machine-1" in window.device_label.text() and "可用" in window.device_label.text()
    assert "legacy calibration refresh" not in main_window_harness.events


def test_main_real_qt_bridge_ignores_late_discovery_then_completes_current_generation(
        main_window_harness, ui_qapp):
    from ui.recording_service_bridge import RecordingServiceBridge

    service = _QtPrewarmService("accepted")
    bridge = RecordingServiceBridge(service)
    lifetime = VePrewarmLifetime()
    window = main_window_harness.create(
        recording_bridge=bridge, lifetime_instance=lifetime)
    discovery = window.ve_discovery.service
    stale_generation = discovery.generation
    window.ve_discovery.refresh()

    discovery.deliver([device_info(name="stale")], generation=stale_generation)
    ui_qapp.processEvents()
    assert service.calls == [] and lifetime.snapshot().state == "available"

    discovery.deliver([device_info(name="current")])
    ui_qapp.processEvents()
    assert len(service.calls) == 1 and lifetime.snapshot().state == "pending"
    service.callback(_prewarm_completion(service.calls[0]))
    ui_qapp.processEvents()
    assert lifetime.snapshot().state == "succeeded"


@pytest.mark.parametrize("status,expected", [("busy", "skipped_busy"), ("closing", "skipped_busy")])
def test_main_real_qt_bridge_consumes_immediate_busy_and_closing_without_callback(
        main_window_harness, ui_qapp, status, expected):
    from ui.recording_service_bridge import RecordingServiceBridge

    service = _QtPrewarmService(status)
    bridge = RecordingServiceBridge(service)
    lifetime = VePrewarmLifetime()
    window = main_window_harness.create(
        recording_bridge=bridge, lifetime_instance=lifetime)

    window.ve_discovery.service.deliver([device_info()])
    ui_qapp.processEvents()

    assert len(service.calls) == 1
    assert lifetime.snapshot().state == expected
    assert not bridge.hardware_busy


def test_main_real_qt_bridge_discards_projected_busy_before_service_admission(
        main_window_harness, ui_qapp):
    from ui.recording_service_bridge import RecordingServiceBridge

    service = _QtPrewarmService("accepted")
    service._capture_session = object()
    bridge = RecordingServiceBridge(service)
    lifetime = VePrewarmLifetime()
    window = main_window_harness.create(
        recording_bridge=bridge, lifetime_instance=lifetime)

    window.ve_discovery.service.deliver([device_info()])
    ui_qapp.processEvents()

    assert service.calls == []
    assert lifetime.snapshot().state == "skipped_busy"


@pytest.mark.parametrize("stale_selection,ownership_safe", [(False, True), (True, False)])
def test_main_real_qt_bridge_release_failure_terminalizes_with_original_ownership(
        main_window_harness, ui_qapp, monkeypatch, stale_selection, ownership_safe):
    from ui.recording_service_bridge import RecordingServiceBridge

    critical = []
    monkeypatch.setattr(main_window_harness.namespace["QMessageBox"],
                        "critical", lambda *args: critical.append(args[-1]))
    service = _QtPrewarmService("accepted")
    bridge = RecordingServiceBridge(service)
    lifetime = VePrewarmLifetime()
    window = main_window_harness.create(
        recording_bridge=bridge, lifetime_instance=lifetime)
    window.ve_discovery.service.deliver([device_info()])
    ui_qapp.processEvents()
    request = service.calls[0]
    if stale_selection:
        window.mic = device_info(machine_id="new-selection")
        window.mic_channels = [7, 1]

    service.callback(_prewarm_completion(
        request, success=False, stage="release_ve", detail="release failed",
        ownership_safe=ownership_safe))
    ui_qapp.processEvents()

    snapshot = lifetime.snapshot()
    assert snapshot.state == "failed"
    assert snapshot.failed_signature == request.signature
    assert snapshot.ownership_safe is ownership_safe
    assert getattr(window, "_ve_prewarm_context", None) is None
    assert bool(critical) is not stale_selection
    if stale_selection:
        assert lifetime.admission_for(window._ve_signature(
            window.mic, window.mic_channels)) == "allowed"


def test_main_hardware_accept_and_cancel_do_not_reset_calibration(main_window_harness, controls, ui_qapp, monkeypatch):
    device = device_info()
    controls.profiles.set_sample_rate(device, 51200, controls.calibrations)
    save_measurement(controls.calibrations, device, 7)
    raw = controls.calibrations.path.read_bytes()
    window = main_window_harness.create()
    window.ve_discovery.service.deliver([device])
    ui_qapp.processEvents()
    def accept(controller):
        confirm(controller, ui_qapp)
        panel = controller.view.ve_controls
        controller._on_ok_clicked()
        assert controller.view.result() == QDialog.Accepted
        state = controller.model.state
        return True, state.speaker_device, state.speaker_channels, state.mic_device, state.mic_channels
    monkeypatch.setattr(hardware_ui.HardwareSelectionController, "on_exec", accept)
    window.on_hardware_window_init()
    assert window.mic["input_config"]["sample_rate"] == 51200
    assert window.sequence_window.mic_channels == [7, 1]
    assert controls.calibrations.path.read_bytes() == raw
    assert "legacy calibration refresh" not in main_window_harness.events
    saved, old = controls.path.read_bytes(), window.mic
    def cancel(controller):
        controller.view.reject()
        return False, None, [], device_info(machine_id="should-not-publish"), [0]
    monkeypatch.setattr(hardware_ui.HardwareSelectionController, "on_exec", cancel)
    window.on_hardware_window_init()
    assert window.mic is old and controls.path.read_bytes() == saved


def test_main_shared_recording_busy_guards_hardware_and_calibration(main_window_harness, controls):
    window = main_window_harness.create()
    main_window_harness.bridge.hardware_busy = True
    window.on_hardware_window_init()
    window.on_calibration_window_init()
    assert len(controls.warnings) == 2
    assert not window.mic["available"]


def _configure_main_calibration_admission(
    window, bridge, *, current_type="SPL", old_type="SPL",
    can_start=True, hardware_busy=False, playback=False, closing=False,
):
    from ui.sequence.sequence_widget_recording_process_ops import (
        SequenceWidgetRecordingProcessOpsMixin,
    )

    sequence = window.sequence_window
    sequence.recording_bridge = bridge
    current = {"display_sequence": ["current"],
               "current": {"type": current_type}}
    old = {"display_sequence": ["old"], "old": {"type": old_type}}
    processor = object()
    sequence.analysis_config = current
    sequence._build_recent_session_config_snapshot = lambda: {
        "analysis_config": deepcopy(current),
    }
    sequence._recording_process_contexts = {
        "old": SimpleNamespace(
            enabled_analysis_identifiers=(old_type,),
            recent_session_config_snapshot={"analysis_config": old},
            processor=processor,
        )
    }
    sequence._record_workflow_busy = True
    sequence._recording_closed = False
    sequence._closing = closing
    sequence.player_status_flag = True
    sequence.streaming_processor = processor
    sequence._condition_playback_controller = SimpleNamespace(
        is_audio_playing=lambda: playback)
    sequence.recent_session_panel = None
    for name in (
        "_can_start_recording_workflow",
        "_can_start_calibration_workflow",
    ):
        setattr(sequence, name, MethodType(
            getattr(SequenceWidgetRecordingProcessOpsMixin, name), sequence))
    bridge.service.busy = True
    bridge.service.can_start_recording = can_start
    bridge.hardware_busy = hardware_busy


@pytest.mark.parametrize("current_type", ["SPL", "ED", "future-analysis"])
def test_main_calibration_action_waits_for_recording_publication(
    main_window_harness, monkeypatch, current_type,
):
    window = main_window_harness.create()
    window.access_lvl = "Engineer"
    window.mic = {"backend": "sounddevice", "name": "test-input"}
    _configure_main_calibration_admission(
        window, main_window_harness.bridge, current_type=current_type)
    constructed = []

    class CalibrationDialog:
        input_calibration_flag = False

        def __init__(self, **kwargs):
            constructed.append(kwargs)

        def exec(self):
            return None

    monkeypatch.setitem(
        main_window_harness.namespace, "CalibrationWindow", CalibrationDialog)
    window.hardware_action_calibration.triggered.connect(
        window.on_calibration_window_init)

    window._update_hardware_busy_state()
    assert not window.hardware_action_selection.isEnabled()
    assert not window.hardware_action_calibration.isEnabled()
    window.hardware_action_calibration.trigger()

    assert constructed == []


@pytest.mark.parametrize(
    "case,overrides",
    [
        ("ineligible-old", {"old_type": "future-analysis"}),
        ("ineligible-old-ed", {"old_type": "ED"}),
        ("ineligible-old-and-current", {
            "old_type": "future-analysis", "current_type": "ED"}),
        ("active-capture", {"can_start": False, "hardware_busy": True}),
        ("capacity-full", {"can_start": False}),
        ("playback", {"playback": True}),
        ("release", {"hardware_busy": True}),
        ("closing", {"closing": True}),
    ],
)
def test_main_calibration_action_retains_outer_blockers(
    main_window_harness, monkeypatch, case, overrides,
):
    window = main_window_harness.create()
    window.access_lvl = "Engineer"
    window.mic = {"backend": "sounddevice", "name": "test-input"}
    _configure_main_calibration_admission(
        window, main_window_harness.bridge, **overrides)
    constructed = []

    class CalibrationDialog:
        input_calibration_flag = False

        def __init__(self, **kwargs):
            constructed.append(kwargs)

        def exec(self):
            return None

    monkeypatch.setitem(
        main_window_harness.namespace, "CalibrationWindow", CalibrationDialog)
    window.hardware_action_calibration.triggered.connect(
        window.on_calibration_window_init)

    assert not window._calibration_admission_available(), case
    window._update_hardware_busy_state()
    assert not window.hardware_action_calibration.isEnabled(), case
    window.on_calibration_window_init()

    assert constructed == []


def test_main_background_result_does_not_disable_hardware_dialog(
    main_window_harness, controls, monkeypatch,
):
    window = main_window_harness.create()
    main_window_harness.bridge.service.busy = True
    main_window_harness.bridge.hardware_busy = False
    opened = []
    monkeypatch.setitem(main_window_harness.namespace, "open_hardware_selection_window",
        lambda **kwargs: (opened.append(kwargs) or
            (False, window.speaker, window.speaker_channels, window.mic, window.mic_channels)))

    window.on_hardware_window_init()

    assert len(opened) == 1
    assert main_window_harness.bridge.release_calls == []


def test_main_cancel_and_same_ve_signature_do_not_release(
    main_window_harness, controls, ui_qapp, monkeypatch,
):
    window = main_window_harness.create()
    window.ve_discovery.service.deliver([device_info()])
    ui_qapp.processEvents()
    before = controls.path.read_bytes()
    old = window.mic
    monkeypatch.setitem(main_window_harness.namespace, "open_hardware_selection_window",
        lambda **kwargs: (False, None, [], device_info(machine_id="ignored"), [0]))
    window.on_hardware_window_init()
    assert window.mic is old and controls.path.read_bytes() == before
    assert main_window_harness.bridge.release_calls == []

    same = deepcopy(window.mic)
    monkeypatch.setitem(main_window_harness.namespace, "open_hardware_selection_window",
        lambda **kwargs: (True, window.speaker, window.speaker_channels, same, [7, 1]))
    window.on_hardware_window_init()
    assert main_window_harness.bridge.release_calls == []


@pytest.mark.parametrize("change", ["rate", "channels", "device", "soundcard"])
def test_main_accepted_native_signature_change_publishes_then_releases(
    main_window_harness, controls, ui_qapp, monkeypatch, change,
):
    from base import hardware_selection
    from base.ve3668n_input import ve_acquisition_signature

    window = main_window_harness.create()
    window.ve_discovery.service.deliver([device_info()])
    ui_qapp.processEvents()
    old = deepcopy(window.mic)
    old_channels = list(window.mic_channels)
    new_mic = deepcopy(old)
    new_channels = list(old_channels)
    if change == "rate":
        new_mic["input_config"]["sample_rate"] = 44100
    elif change == "channels":
        new_channels = [1, 7]
    elif change == "device":
        new_mic["machine_id"] = "test-machine-2"
        new_mic["name"] = "other"
    elif change == "soundcard":
        new_mic = controls.old_mic
        new_channels = [0]

    if change != "soundcard":
        hardware_selection.save_ve_selection(new_mic, window.speaker, new_channels, [],
            profile_store=controls.profiles, calibration_store=controls.calibrations,
            path=controls.path)
    observed = []
    main_window_harness.bridge.release_ve = lambda signature, callback: (
        observed.append((signature, window.mic, list(window.mic_channels),
                         controls.path.read_bytes(), callback)) or "pending")
    def accept_selection(**kwargs):
        if change == "rate":
            window.sequence_window.sequence_config = [{"seq1": {"acq": {"detail": {"sample_rate": 44100}}}}]
        return True, window.speaker, window.speaker_channels, new_mic, new_channels
    monkeypatch.setitem(main_window_harness.namespace, "open_hardware_selection_window", accept_selection)

    window.on_hardware_window_init()

    assert len(observed) == 1
    required, published_mic, published_channels, persisted, _ = observed[0]
    assert published_mic == new_mic and published_channels == new_channels
    assert persisted == controls.path.read_bytes()
    expected = None if change == "soundcard" else ve_acquisition_signature(
        new_mic, new_channels, new_mic["input_config"]["sample_rate"])
    assert required == expected


def test_main_release_failure_warns_without_rolling_back_persisted_setting(
    main_window_harness, controls, ui_qapp, monkeypatch,
):
    from base import hardware_selection

    window = main_window_harness.create()
    window.ve_discovery.service.deliver([device_info()])
    ui_qapp.processEvents()
    replacement = deepcopy(window.mic)
    replacement["input_config"]["sample_rate"] = 44100
    hardware_selection.save_ve_selection(replacement, window.speaker, [7, 1], [],
        profile_store=controls.profiles, calibration_store=controls.calibrations,
        path=controls.path)
    persisted = controls.path.read_bytes()
    def accept_selection(**kwargs):
        window.sequence_window.sequence_config = [{"seq1": {"acq": {"detail": {"sample_rate": 44100}}}}]
        return True, window.speaker, window.speaker_channels, replacement, [7, 1]
    monkeypatch.setitem(main_window_harness.namespace, "open_hardware_selection_window", accept_selection)

    window.on_hardware_window_init()
    main_window_harness.bridge.hardware_busy = True  # Retirement is still pending.
    main_window_harness.bridge.release_callback("failed", ("ClearTask failed",))

    assert window.mic == replacement and window.mic["input_config"]["sample_rate"] == 44100
    assert controls.path.read_bytes() == persisted
    assert main_window_harness.bridge.hardware_busy
    assert any("ClearTask failed" in warning and "回收" in warning
               for warning in controls.warnings)


def test_backend_round_trip_retains_unsaved_soundcard_choice_and_then_saves_old_fields(controls, ui_qapp):
    initial = hardware_ui.HardwareSelectionState(api_name="MME", mic_device=controls.old_mic,
        mic_channels=[1], speaker_device=controls.speaker)
    controller = controls.create(initial)
    panel = controller.view.ve_controls
    controller.view.driver_combo.setCurrentText("vkinging")
    confirm(controller, ui_qapp)
    select_device(controller, "test-machine-1")
    for row in range(controller.view.mic_channel_table.model().rowCount()):
        controller.view.mic_channel_table.model().item(row).setCheckState(Qt.Checked)
    controller.view.driver_combo.setCurrentText("MME")
    assert controller.model.state.mic_device == controls.old_mic
    assert controller.view.mic_channel_table.checked_payloads() == [1]
    controller.view.driver_combo.setCurrentText("vkinging")
    confirm(controller, ui_qapp)
    select_device(controller, "test-machine-1")
    controller.view.mic_channel_table.model().item(0).setCheckState(Qt.Checked)
    controller._on_ok_clicked()
    assert controller.view.result() == QDialog.Accepted
    payload = json.loads(controls.path.read_text(encoding="utf-8"))
    assert payload["mic_name"] == "Old mic" and payload["mic_channels"] == [1]


def test_switch_back_write_failure_keeps_explicit_ve_selection(controls, ui_qapp, monkeypatch):
    from base import hardware_selection
    controls.path.write_text(json.dumps({"api_name": "MME", "mic_name": "Old mic", "mic_channels": [1],
        "input_selection": {"schema_version": 1, "backend": "vkinging",
                            "machine_id": "test-machine-1", "physical_channels": [7, 1]}}), encoding="utf-8")
    before = controls.path.read_bytes()
    controller = controls.create()
    controller.view.driver_combo.setCurrentText("MME")
    controller.view.speaker_device_table.set_checked_by_predicate(lambda item: True)
    monkeypatch.setattr(hardware_selection, "_atomic_write_json", lambda *args: False)
    controller._on_ok_clicked()
    assert controller.view.result() != QDialog.Accepted
    assert controls.path.read_bytes() == before


def test_discovery_result_during_busy_does_not_load_profile_until_idle(controls, ui_qapp):
    controller = controls.create()
    controls.busy.value = True  # Before the 100 ms UI timer gets its next tick.
    controller.view.ve_controls.discovery.service.deliver([device_info(name="deferred")])
    ui_qapp.processEvents()
    assert not controller.model.state.mic_device["available"]
    controls.busy.value = False
    QTest.qWait(150)
    assert controller.model.state.mic_device["available"]


def test_disconnect_reconnect_retains_id_order_without_other_device_fallback(controls, ui_qapp):
    controller = controls.create()
    confirm(controller, ui_qapp)
    panel = controller.view.ve_controls
    controller.refresh_and_render(try_restore=True)
    confirm(controller, ui_qapp, device_info(machine_id="other", name="Dev2"))
    assert not controller.model.state.mic_device["available"]
    assert controller.model.state.mic_channels == []
    assert controller.view.mic_channel_table.model().rowCount() == 0
    controller.refresh_and_render(try_restore=True)
    confirm(controller, ui_qapp, device_info(name="new-alias", physical_channels=list(range(8))))
    controller._on_ok_clicked()
    assert controller.view.result() == QDialog.Accepted
    assert controller.model.state.mic_channels == [7, 1]
    assert controller.model.state.mic_device["name"] == "new-alias"


def test_closing_repeated_dialogs_closes_every_discovery_owner(controls, ui_qapp):
    services = []
    for _ in range(5):
        controller = controls.create()
        services.append(controller.view.ve_controls.discovery.service)
        controller.view.show()
        controller.view.close()
    ui_qapp.processEvents()
    assert all(service.closed for service in services)


@pytest.mark.parametrize("corrupt", [{"physical_channels": [None]}, {"physical_channels": ["bad"]},
                                   {"machine_id": {"invalid": "object"}}])
def test_corrupt_saved_record_renders_unavailable_without_coercion(controls, corrupt, monkeypatch):
    import sys
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *details: errors.append(details[1]))
    from base.hardware_selection import unavailable_ve_selection
    raw = {"schema_version": 1, "backend": "vkinging", "machine_id": "test-machine-1",
           "physical_channels": [7, 1], **corrupt}
    unavailable = unavailable_ve_selection(raw)
    controller = controls.create(hardware_ui.HardwareSelectionState(
        mic_device=unavailable, mic_channels=raw["physical_channels"]))
    panel = controller.view.ve_controls
    event = DiscoveryEvent(panel.discovery.service.generation, "completed",
                           DiscoveryResult((device_info(),)), None, 0, True)
    panel._on_discovered(event)
    assert not controller.model.state.mic_device["available"]
    assert controller.model.state.mic_channels == []
    assert "损坏" in controller.view.ve_status_label.text() or "不可用" in controller.view.ve_status_label.text()
    controller._on_ok_clicked()
    assert controller.view.result() != QDialog.Accepted
    assert errors == []


def test_explicit_reselect_same_device_repairs_corrupt_selection(controls, ui_qapp):
    from base.hardware_selection import unavailable_ve_selection
    unavailable = unavailable_ve_selection({"schema_version": 2, "backend": "vkinging",
        "machine_id": "test-machine-1", "physical_channels": [7, 1]})
    controller = controls.create(hardware_ui.HardwareSelectionState(mic_device=unavailable, mic_channels=[7, 1]))
    confirm(controller, ui_qapp)
    panel = controller.view.ve_controls
    assert not controller.model.state.mic_device["available"]
    select_device(controller, "test-machine-1")
    assert controller.model.state.mic_device["available"]
    assert controller.model.state.mic_channels == []
    controller.view.mic_channel_table.model().item(0).setCheckState(Qt.Checked)
    controller._on_ok_clicked()
    assert controller.view.result() == QDialog.Accepted


def test_main_ve_calibration_handoff_uses_shared_stores_without_legacy_reset(main_window_harness, controls, ui_qapp):
    window = main_window_harness.create()
    window.ve_discovery.service.deliver([device_info()])
    ui_qapp.processEvents()
    class CalibrationDialog:
        input_calibration_flag = True
        def __init__(self, **kwargs):
            assert kwargs["recording_bridge"] is main_window_harness.bridge
        def exec(self):
            assert self.ve_profile_store is controls.profiles
            assert self.ve_calibration_store is controls.calibrations
    main_window_harness.namespace["CalibrationWindow"] = CalibrationDialog
    window.on_calibration_window_init()
    assert "legacy calibration refresh" not in main_window_harness.events


def test_main_close_ignores_late_discovery_and_does_not_wait(main_window_harness, ui_qapp):
    window = main_window_harness.create()
    service = window.ve_discovery.service
    original = window.mic
    window._close_ve_discovery()
    service.deliver([device_info(name="late")])
    ui_qapp.processEvents()
    assert window.mic is original and service.closed


@pytest.mark.parametrize("edit", ["device", "backend", "channel_add", "channel_remove", "driver", "speaker"])
def test_busy_pre_timer_rejected_edit_renders_exact_staged_snapshot(controls, ui_qapp, edit):
    one = device_info(physical_channels=list(range(8)))
    two = device_info(machine_id="two", name="Dev2", physical_channels=list(range(8)))
    controls.profiles.set_sample_rate(one, 51200, controls.calibrations)
    controls.profiles.set_sample_rate(two, 51200, controls.calibrations)
    save_measurement(controls.calibrations, two, 6)
    before_profile = controls.profiles.path.read_bytes()
    before_calibration = controls.calibrations.path.read_bytes()
    controller = controls.create(hardware_ui.HardwareSelectionState(
        api_name="MME", mic_device=one, mic_channels=[7, 1], speaker_device=controls.speaker))
    confirm(controller, ui_qapp, one, two)
    view, panel = controller.view, controller.view.ve_controls
    select_device(controller, "two")
    table = view.mic_channel_table.model()
    table.item(6).setCheckState(Qt.Checked)
    table.item(2).setCheckState(Qt.Checked)
    staged = deepcopy(controller.model.state)
    assert staged.mic_channels == [6, 2]
    controller._busy_timer.stop()  # No timer tick or event processing can repair the signal for us.
    controls.busy.value = True
    if edit == "device":
        select_device(controller, one["machine_id"])
    elif edit == "backend":
        controller.view.driver_combo.setCurrentText("MME")
    elif edit == "channel_add":
        table.item(7).setCheckState(Qt.Checked)
    elif edit == "channel_remove":
        table.item(6).setCheckState(Qt.Unchecked)
    elif edit == "driver":
        view.driver_combo.setCurrentText("ASIO")
    else:
        view.speaker_device_table.model().item(0).setCheckState(Qt.Unchecked)
    assert view.mic_device_table.checked_payload()["machine_id"] == "two"
    assert view.driver_combo.currentData() == "vkinging"
    assert view.speaker_device_table.checked_payload() == controls.speaker
    assert view.mic_channel_table.checked_payloads() == [2, 6]
    assert controller.model.state == staged
    assert panel.selected_device == staged.mic_device and panel.selected_channels == [6, 2]
    for widget in (controller.view.mic_device_table,
                   view.driver_combo, view.mic_channel_table, view.speaker_device_table, view.ok_btn):
        assert not widget.isEnabled()
    assert not controls.defaults.pair.writes and not controls.path.exists()
    assert controls.profiles.path.read_bytes() == before_profile
    assert controls.calibrations.path.read_bytes() == before_calibration
    controls.busy.value = False
    controller._update_busy_state()
    controller._on_ok_clicked()
    assert view.result() == QDialog.Accepted
    # The strict save boundary canonicalizes inventory lists to tuples.
    staged.mic_device = validate_device_snapshot(staged.mic_device)
    assert controller.model.state == staged
    assert controls.profiles.load(two, controls.calibrations)["sample_rate"] == 51200
    assert json.loads(controls.path.read_text(encoding="utf-8"))["input_selection"] == {
        "schema_version": 1, "backend": "vkinging", "machine_id": "two", "physical_channels": [6, 2]}
    assert controls.calibrations.path.read_bytes() == before_calibration


@pytest.mark.parametrize("edit", ["mic", "channel", "backend"])
def test_busy_pre_timer_soundcard_edits_keep_view_and_accepted_values(controls, edit):
    controller = controls.create(hardware_ui.HardwareSelectionState(api_name="MME",
        mic_device=controls.old_mic, mic_channels=[1], speaker_device=controls.speaker))
    controller._busy_timer.stop()
    view = controller.view
    controls.busy.value = True
    if edit == "mic":
        view.mic_device_table.model().item(0).setCheckState(Qt.Unchecked)
    elif edit == "channel":
        view.mic_channel_table.model().item(0).setCheckState(Qt.Checked)
    else:
        view.driver_combo.setCurrentText("vkinging")
    assert view.driver_combo.currentData() == "MME"
    assert view.mic_device_table.checked_payload() == controls.old_mic
    assert view.mic_channel_table.checked_payloads() == [1]
    assert not view.mic_channel_table.isEnabled() and not view.mic_device_table.isEnabled()
    assert not controls.defaults.pair.writes
    controls.busy.value = False
    controller._update_busy_state()
    controller._on_ok_clicked()
    assert view.result() == QDialog.Accepted
    assert controller.model.state.mic_device == controls.old_mic
    assert controller.model.state.mic_channels == [1]
    assert controls.defaults.calls == [(1, 2)]


@pytest.mark.parametrize("outcome", ["accept", "absent", "cancel", "failed_save"])
def test_ve_output_is_applied_only_after_successful_accept(controls, ui_qapp, monkeypatch, outcome):
    from base import hardware_selection
    controller = controls.create(hardware_ui.HardwareSelectionState(mic_device=device_info(),
        mic_channels=[7, 1], speaker_device=controls.speaker if outcome != "absent" else None))
    confirm(controller, ui_qapp)
    assert controls.defaults.pair.raw == [None, 9] and not controls.defaults.pair.writes
    if outcome == "cancel":
        controller.view.reject()
    else:
        if outcome == "failed_save":
            monkeypatch.setattr(hardware_selection, "_atomic_write_json", lambda *args: False)
        controller._on_ok_clicked()
    if outcome == "accept":
        assert controller.view.result() == QDialog.Accepted
        assert controls.defaults.pair.raw == [None, 2]
        assert controls.defaults.pair.writes == [(1, 2)]
    else:
        assert controls.defaults.pair.raw == [None, 9] and not controls.defaults.pair.writes
    assert not controls.defaults.pair.reads and not controls.defaults.calls


@pytest.mark.parametrize("outcome", ["cancel", "accept", "failed_save"])
def test_ve_to_soundcard_draft_never_applies_defaults_before_accept(controls, monkeypatch, outcome):
    from base import hardware_selection
    controls.path.write_text(json.dumps({"api_name": "MME", "mic_name": "Old mic", "mic_channels": [1],
        "speaker_name": "Old output", "speaker_channels": [], "input_selection": {
            "schema_version": 1, "backend": "vkinging", "machine_id": "test-machine-1", "physical_channels": [7, 1]}}), encoding="utf-8")
    before = controls.path.read_bytes()
    controller = controls.create(hardware_ui.HardwareSelectionState(api_name="MME",
        mic_device=device_info(), mic_channels=[7, 1], speaker_device=controls.speaker))
    assert controller._soundcard_state is None
    controller.view.driver_combo.setCurrentText("MME")
    assert controller.model.state.mic_device == controls.old_mic
    assert controller.view.mic_channel_table.checked_payloads() == [1]
    assert controls.defaults.pair.raw == [None, 9]
    assert not controls.defaults.calls and not controls.defaults.pair.writes
    assert controls.path.read_bytes() == before
    if outcome == "cancel":
        controller.view.reject()
    else:
        if outcome == "failed_save":
            monkeypatch.setattr(hardware_selection, "_atomic_write_json", lambda *args: False)
        controller._on_ok_clicked()
    if outcome == "accept":
        assert controller.view.result() == QDialog.Accepted
        assert controls.defaults.calls == [(1, 2)] and controls.defaults.pair.raw == [1, 2]
        assert "input_selection" not in json.loads(controls.path.read_text(encoding="utf-8"))
    else:
        assert controller.view.result() != QDialog.Accepted
        assert controls.defaults.pair.raw == [None, 9]
        assert not controls.defaults.calls and not controls.defaults.pair.writes
        assert controls.path.read_bytes() == before


def assert_old_soundcard_visible(controller, controls):
    state, view = controller.model.state, controller.view
    assert state.api_name == view.driver_combo.currentText() == "MME"
    assert state.mic_device == view.mic_device_table.checked_payload() == controls.old_mic
    assert state.speaker_device == view.speaker_device_table.checked_payload() == controls.speaker
    assert state.mic_channels == view.mic_channel_table.checked_payloads() == [1]


@pytest.mark.parametrize("persisted", [False, True])
def test_cross_api_backend_roundtrip_restores_legacy_driver_and_visible_devices(controls, ui_qapp, persisted):
    from base import hardware_selection
    if persisted:
        hardware_selection.save_if_changed(controls.old_mic, controls.speaker, [1], [], path=controls.path)
    before = controls.path.read_bytes() if persisted else None
    controller = controls.create(hardware_ui.HardwareSelectionState(api_name="MME",
        mic_device=controls.old_mic, mic_channels=[1], speaker_device=controls.speaker))
    controller.view.driver_combo.setCurrentText("vkinging")
    confirm(controller, ui_qapp)
    select_device(controller, "test-machine-1")
    controller.view.mic_channel_table.model().item(0).setCheckState(Qt.Checked)
    service = controller.view.ve_controls.discovery.service
    generation = service.generation
    controller.view.driver_combo.setCurrentText("ASIO")
    assert controller.model.state.mic_device is None
    assert controller.model.state.mic_channels == []
    assert controller.view.mic_device_table.model().item(0).text() == controls.audio.asio_mic["name"]
    service.deliver([device_info(name="stale")], generation=generation)
    ui_qapp.processEvents()
    assert controller.model.state.mic_device is None
    controller.view.driver_combo.setCurrentText("MME")
    controller.view.mic_device_table.model().item(0).setCheckState(Qt.Checked)
    controller.view.mic_channel_table.model().item(1).setCheckState(Qt.Checked)
    controller.view.speaker_device_table.model().item(0).setCheckState(Qt.Checked)
    controller.view.driver_combo.setCurrentText("vkinging")
    confirm(controller, ui_qapp)
    select_device(controller, "test-machine-1")
    controller.view.mic_channel_table.model().item(0).setCheckState(Qt.Checked)
    controller.view.driver_combo.setCurrentText("MME")
    assert_old_soundcard_visible(controller, controls)
    assert not controls.defaults.calls and not controls.defaults.pair.writes and not controls.audio.queries
    assert (controls.path.read_bytes() if controls.path.exists() else None) == before


@pytest.mark.parametrize("outcome", ["cancel", "accept", "failed_save"])
def test_cross_api_restart_main_dialog_restores_legacy_context_before_accept(
        main_window_harness, controls, ui_qapp, monkeypatch, outcome):
    from base import hardware_selection
    hardware_selection.save_if_changed(controls.old_mic, controls.speaker, [1], [], path=controls.path)
    for machine_id, output in [("test-machine-1", controls.audio.asio_speaker),
                               ("two", controls.speaker), ("test-machine-1", controls.audio.asio_speaker)]:
        hardware_selection.save_ve_selection(device_info(machine_id=machine_id), output, [7, 1], [],
            path=controls.path, profile_store=controls.profiles, calibration_store=controls.calibrations)
    before = controls.path.read_bytes()
    window = main_window_harness.create()
    assert window.speaker == controls.audio.asio_speaker and window.mic_channels == [7, 1]
    assert controls.defaults.pair.raw == [None, 4] and controls.defaults.pair.writes == [(1, 4)]
    assert not controls.defaults.calls and not controls.audio.queries
    if outcome == "failed_save":
        monkeypatch.setattr(hardware_selection, "_atomic_write_json", lambda *args: False)

    def interact(controller):
        view = controller.view
        assert controller._soundcard_state is None
        assert view.driver_combo.currentText() == "vkinging"
        assert controller.model.state.api_name == "ASIO"
        assert view.speaker_device_table.checked_payload() == controls.audio.asio_speaker
        view.driver_combo.setCurrentText("MME")
        assert_old_soundcard_visible(controller, controls)
        assert controls.defaults.pair.raw == [None, 4] and controls.defaults.pair.writes == [(1, 4)]
        assert not controls.defaults.calls and not controls.defaults.pair.reads and not controls.audio.queries
        assert controls.path.read_bytes() == before
        if outcome != "cancel":
            controller._on_ok_clicked()
        if outcome != "accept":
            assert view.result() != QDialog.Accepted
            view.reject()
        else:
            assert view.result() == QDialog.Accepted
        state = controller.model.state
        return view.result() == QDialog.Accepted, state.speaker_device, state.speaker_channels, state.mic_device, state.mic_channels

    monkeypatch.setattr(hardware_ui.HardwareSelectionController, "on_exec", interact)
    window.on_hardware_window_init()
    if outcome == "accept":
        assert window.mic == window.sequence_window.mic == controls.old_mic
        assert window.mic_channels == window.sequence_window.mic_channels == [1]
        assert window.speaker == controls.speaker
        assert controls.defaults.calls == [(1, 2)] and controls.defaults.pair.raw == [1, 2]
        payload = json.loads(controls.path.read_text(encoding="utf-8"))
        assert payload["api_name"] == "MME" and "input_selection" not in payload
    else:
        assert window.mic["backend"] == "vkinging" and window.mic_channels == [7, 1]
        assert controls.defaults.pair.raw == [None, 4] and controls.defaults.pair.writes == [(1, 4)]
        assert not controls.defaults.calls and controls.path.read_bytes() == before
    assert not controls.defaults.pair.reads and not controls.audio.queries


@pytest.mark.parametrize("inventory", [[1], [1, 7]])
def test_explicit_table_reselect_clears_routes_and_reopen_restores_saved_only(controls, ui_qapp, inventory):
    from base import hardware_selection
    original = device_info()
    controls.profiles.set_sample_rate(original, 44100, controls.calibrations)
    for channel in (7, 1):
        save_measurement(controls.calibrations, original, channel)
    calibration_bytes = controls.calibrations.path.read_bytes()
    profile_bytes = controls.profiles.path.read_bytes()
    controller = controls.create()
    confirm(controller, ui_qapp, device_info(physical_channels=inventory))
    assert controller.model.state.mic_device["available"] is (7 in inventory)
    assert controller.model.state.mic_channels == ([7, 1] if 7 in inventory else [])
    if 7 not in inventory:
        controller._on_ok_clicked()
        assert controller.view.result() != QDialog.Accepted and not controls.path.exists()
    select_device(controller, "test-machine-1")
    assert controller.model.state.mic_channels == []
    table = controller.view.mic_channel_table.model()
    ain2 = next(table.item(row) for row in range(table.rowCount()) if table.item(row).data(Qt.UserRole) == 1)
    ain2.setCheckState(Qt.Checked)
    controller._on_ok_clicked()
    assert controller.view.result() == QDialog.Accepted
    restored, output, channels, output_channels = hardware_selection.restore_or_default(
        path=controls.path, apply_defaults=False)
    reopened = controls.create(hardware_ui.HardwareSelectionState(mic_device=restored, mic_channels=channels,
        speaker_device=output, speaker_channels=output_channels))
    assert reopened.model.state.mic_channels == []
    confirm(reopened, ui_qapp, device_info(name="reconnected", physical_channels=list(range(8))))
    assert reopened.model.state.mic_channels == [1]
    assert reopened.model.state.mic_device["available"]
    assert controls.profiles.path.read_bytes() == profile_bytes
    assert controls.calibrations.path.read_bytes() == calibration_bytes
    assert not controls.defaults.pair.writes and not controls.defaults.calls and not controls.defaults.pair.reads
    assert "mic" not in controls.audio.queries


def test_default_mainwindow_constructs_with_baseline_sequence(controls, ui_qapp, monkeypatch):
    import main_window
    from PyQt5 import sip
    from ui.recording_service_bridge import RecordingServiceBridge
    from base.recording_service import RecordingService
    monkeypatch.setattr(main_window, "restore_or_default", lambda **kw: (None, None, [], []))
    monkeypatch.setattr(main_window.QMessageBox, "warning", lambda *args: None)
    from base.load_config import LoadUiConfig
    monkeypatch.setattr(LoadUiConfig, "get_tcp_config", lambda: ("127.0.0.1", 12345))
    monkeypatch.setattr(main_window.MainWindow, "get_current_version", lambda self: "test")
    for name in ("critical", "information", "question"):
        monkeypatch.setattr(main_window.QMessageBox, name, lambda *a, **kw: main_window.QMessageBox.No)
    monkeypatch.setattr(main_window.LoginWindow, "on_exec", lambda self: ("Engineer", "test"))
    service = RecordingService()
    bridge = RecordingServiceBridge(service)
    window = None
    try:
        window = main_window.MainWindow(recording_bridge=bridge,
            ve_profile_store=controls.profiles, ve_calibration_store=controls.calibrations,
            discovery_factory=ControlledDiscovery, hardware_selection_path=controls.path)
        assert window.sequence_window.recording_bridge is bridge
        assert window._calibration_admission_available()
        window._update_hardware_busy_state()
    finally:
        if window is not None:
            window._close_ve_discovery()
            sip.delete(window)
        bridge.shutdown()
        assert service.closed.wait(5)

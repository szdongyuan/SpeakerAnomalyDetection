import json
import os
import sqlite3
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from PyQt5.QtWidgets import QApplication, QDialog

import ui.hardware_window as module
from ui.hardware_window import (
    HardwareSelectionController,
    HardwareSelectionModel,
    HardwareSelectionState,
    HardwareSelectionView,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _device(index, name, hostapi=0, inputs=0, outputs=0):
    return {
        "index": index,
        "name": name,
        "hostapi": hostapi,
        "default_samplerate": 48000.0,
        "max_input_channels": inputs,
        "max_output_channels": outputs,
    }


def _asset(
    hardware_id,
    display_name,
    device_name,
    hostapi_name="API",
    hardware_type="audio_interface",
    inputs=0,
    outputs=0,
    samplerate=48000,
    bit_depth=32,
    latency_ms=100,
):
    return {
        "hardware_id": hardware_id,
        "hardware_type": hardware_type,
        "display_name": display_name,
        "device_name": device_name,
        "hostapi_name": hostapi_name,
        "samplerate": samplerate,
        "bit_depth": bit_depth,
        "latency_ms": latency_ms,
        "max_input_channels": inputs,
        "max_output_channels": outputs,
        "updated_at": "2026-06-23 10:00:00",
    }


class FakeRepository:
    def __init__(
        self,
        assets=None,
        channels=None,
        tables_exist=True,
        tables_error=None,
        list_error=None,
        channel_error=None,
    ):
        self.assets = list(assets or [])
        self.channels = dict(channels or {})
        self._tables_exist = tables_exist
        self.tables_error = tables_error
        self.list_error = list_error
        self.channel_error = channel_error
        self.tables_checked = 0
        self.selection_reads = 0
        self.channel_reads = []

    def tables_exist(self):
        self.tables_checked += 1
        if self.tables_error:
            raise self.tables_error
        return self._tables_exist

    def list_assets_for_selection(self):
        self.selection_reads += 1
        if self.list_error:
            raise self.list_error
        grouped = {}
        for asset in self.assets:
            api_group = grouped.setdefault(asset["hostapi_name"], {"input": [], "output": []})
            if asset["max_input_channels"] > 0:
                api_group["input"].append(dict(asset))
            if asset["max_output_channels"] > 0:
                api_group["output"].append(dict(asset))
        return grouped

    def list_channels(self, hardware_id, direction=None):
        self.channel_reads.append((hardware_id, direction))
        if self.channel_error:
            raise self.channel_error
        rows = [dict(row) for row in self.channels.get(hardware_id, [])]
        if direction is not None:
            rows = [row for row in rows if row.get("direction") == direction]
        return rows


def _input_channel(hardware_id, label, index):
    return {
        "channel_id": f"{hardware_id}-{label}",
        "hardware_id": hardware_id,
        "direction": "input",
        "channel_index": index,
        "channel_label": label,
    }


def _build_controller(qapp, repository, monkeypatch, runtime_devices=None, warnings=None):
    monkeypatch.setattr(module.SoundDeviceManager, "get_device_info", staticmethod(lambda: runtime_devices or {}))
    warnings = warnings if warnings is not None else []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    model = HardwareSelectionModel(HardwareSelectionState(api_name="API"), repository=repository)
    view = HardwareSelectionView()
    controller = HardwareSelectionController(model, view)
    qapp.processEvents()
    return controller, view, warnings


def test_normalize_channel_indices_rejects_zero_prefixed_labels():
    assert module._normalize_channel_indices(["In0", "Out0"]) == []
    assert module._normalize_channel_indices(["In0", "Out0", "In1"]) == [0]


def test_hardware_window_no_longer_exports_removed_message_and_exception_globals():
    assert not hasattr(module, "INVALID_MIC_CHANNEL_ROWS_MESSAGE")
    assert not hasattr(module, "REPOSITORY_READ_EXCEPTIONS")


def test_model_refresh_reads_registered_assets_without_runtime_enumeration(monkeypatch):
    speaker = _asset("speaker-1", "Registered speaker", "Runtime speaker", outputs=2)
    mic = _asset("mic-1", "Registered mic", "Runtime mic", inputs=2)
    repository = FakeRepository([speaker, mic])

    def fail_runtime_enumeration():
        raise AssertionError("runtime enumeration must not be used for listing")

    monkeypatch.setattr(module.SoundDeviceManager, "get_device_info", staticmethod(fail_runtime_enumeration))

    model = HardwareSelectionModel(repository=repository)
    model.refresh()

    assert repository.tables_checked == 1
    assert repository.selection_reads == 1
    assert model.api_names() == ["API"]
    assert model.speaker_devices() == [speaker]
    assert model.mic_devices() == [mic]


def test_hardware_selection_view_has_no_manual_refresh_button(qapp):
    view = HardwareSelectionView()
    try:
        assert not hasattr(view, "refresh_btn")
        button_texts = [
            button.text()
            for button in view.findChildren(module.PushButton)
        ]
        assert " 刷  新 " not in button_texts
    finally:
        view.close()


def test_selection_controller_initial_render_still_loads_registered_assets(qapp, monkeypatch):
    speaker = _asset("speaker-1", "Registered speaker", "Runtime speaker", outputs=2)
    repository = FakeRepository([speaker])

    controller, view, _warnings = _build_controller(qapp, repository, monkeypatch)
    try:
        assert repository.tables_checked == 1
        assert repository.selection_reads == 1
        assert controller.model.api_names() == ["API"]
    finally:
        view.close()


def test_model_groups_distinct_api_names_and_filters_speaker_microphone_candidates():
    speaker = _asset("speaker-1", "Speaker", "Runtime speaker", hostapi_name="API-A", outputs=2)
    mic = _asset("mic-1", "Mic", "Runtime mic", hostapi_name="API-B", inputs=2)
    interface = _asset("iface-1", "Interface", "Runtime iface", hostapi_name="API-A", inputs=4, outputs=4)
    silent = _asset("other-1", "Other", "Runtime other", hostapi_name="API-C", inputs=0, outputs=0)
    repository = FakeRepository([speaker, mic, interface, silent])

    model = HardwareSelectionModel(repository=repository)
    model.refresh()

    assert model.api_names() == ["API-A", "API-B", "API-C"]
    assert model.state.api_name == "API-A"
    assert model.speaker_devices() == [speaker, interface]
    assert model.mic_devices() == [interface]

    model.set_api("API-B")
    assert model.speaker_devices() == []
    assert model.mic_devices() == [mic]

    model.set_api("API-C")
    assert model.speaker_devices() == []
    assert model.mic_devices() == []


def test_selecting_microphone_loads_registered_input_channels_by_label(qapp, monkeypatch):
    mic = _asset("mic-1", "Registered mic", "Runtime mic", inputs=3)
    repository = FakeRepository(
        [mic],
        channels={
            "mic-1": [
                _input_channel("mic-1", "In10", 9),
                _input_channel("mic-1", "In2", 1),
                {**_input_channel("mic-1", "Out1", 0), "direction": "output"},
            ]
        },
    )

    controller, view, _warnings = _build_controller(qapp, repository, monkeypatch)
    try:
        view.mic_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "mic-1")
        qapp.processEvents()

        channel_model = view.mic_channel_table.model()
        assert repository.channel_reads[-1] == ("mic-1", "input")
        assert [channel_model.item(row, 0).text() for row in range(channel_model.rowCount())] == ["In2", "In10"]
        assert [channel_model.item(row, 0).data(module.Qt.UserRole) for row in range(channel_model.rowCount())] == [1, 9]
    finally:
        view.close()


def test_controller_restores_legacy_runtime_initial_devices_to_registered_assets(qapp, monkeypatch):
    speaker = _asset("speaker-1", "Registered speaker", "Runtime speaker", outputs=2, samplerate=44100)
    mic = _asset("mic-1", "Registered mic", "Runtime mic", inputs=2, samplerate=48000)
    repository = FakeRepository(
        [speaker, mic],
        channels={"mic-1": [_input_channel("mic-1", "In1", 0), _input_channel("mic-1", "In2", 1)]},
    )
    warnings = []
    monkeypatch.setattr(module.SoundDeviceManager, "get_device_info", staticmethod(lambda: {}))
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))

    initial_state = HardwareSelectionState(
        speaker_device={
            "name": "Runtime speaker",
            "hostapi_name": "API",
            "hostapi": 77,
            "default_samplerate": 96000.0,
            "max_output_channels": 12,
        },
        mic_device={
            "name": "Runtime mic",
            "hostapi_name": "API",
            "hostapi": 88,
            "default_samplerate": 44100.0,
            "max_input_channels": 16,
        },
        mic_channels=[1],
    )
    model = HardwareSelectionModel(initial_state, repository=repository)
    view = HardwareSelectionView()
    controller = HardwareSelectionController(model, view)
    try:
        qapp.processEvents()

        selected_speaker = view.speaker_device_table.checked_payload()
        selected_mic = view.mic_device_table.checked_payload()
        channel_model = view.mic_channel_table.model()

        assert warnings == []
        assert selected_speaker["hardware_id"] == "speaker-1"
        assert selected_mic["hardware_id"] == "mic-1"
        assert [channel_model.item(row, 0).checkState() for row in range(channel_model.rowCount())] == [
            module.Qt.Unchecked,
            module.Qt.Checked,
        ]
        assert model.state.speaker_device["hardware_id"] == "speaker-1"
        assert model.state.mic_device["hardware_id"] == "mic-1"
    finally:
        view.close()


@pytest.mark.parametrize("restored_channels", [["In0"], ["Out0"]])
def test_open_hardware_selection_window_rejects_malformed_restored_channel_labels(
    qapp, monkeypatch, restored_channels
):
    mic = _asset("mic-1", "Registered mic", "Runtime mic", inputs=1)
    repository = FakeRepository([mic], channels={"mic-1": [_input_channel("mic-1", "In1", 0)]})
    selected_channels = []

    def fake_exec(view):
        qapp.processEvents()
        selected_channels.extend(view.mic_channel_table.checked_payloads())
        view.reject()
        return QDialog.Rejected

    monkeypatch.setattr(module.HardwareSelectionView, "exec", fake_exec)

    speaker, selected_mic, returned_channels = module.open_hardware_selection_window(
        driver="API",
        mic_device={"hardware_id": "mic-1"},
        mic_channels=restored_channels,
        repository=repository,
    )

    assert speaker is None
    assert selected_mic == {"hardware_id": "mic-1"}
    assert returned_channels == []
    assert selected_channels == []


def test_missing_hardware_tables_warns_and_keeps_dialog_open(qapp, monkeypatch):
    repository = FakeRepository(tables_exist=False)
    warnings = []

    controller, view, warnings = _build_controller(qapp, repository, monkeypatch, warnings=warnings)
    try:
        assert view.result() != QDialog.Accepted
        assert warnings
        assert warnings[-1][2] == "硬件管理表不存在，请使用最新版数据库"
        assert controller.model.api_names() == []
    finally:
        view.close()


@pytest.mark.parametrize(
    "repository",
    [
        FakeRepository(tables_error=sqlite3.Error("probe failed")),
        FakeRepository(list_error=sqlite3.Error("read failed")),
    ],
)
def test_refresh_read_failure_warns_and_renders_empty_state(qapp, monkeypatch, repository):
    controller, view, warnings = _build_controller(qapp, repository, monkeypatch)
    try:
        assert view.result() != QDialog.Accepted
        assert warnings
        assert "failed" in warnings[-1][2]
        assert controller.model.api_names() == []
        assert view.driver_combo.count() == 0
        assert view.speaker_device_table.model().rowCount() == 0
        assert view.mic_device_table.model().rowCount() == 0
        assert view.mic_channel_table.model().rowCount() == 0
    finally:
        view.close()


def test_channel_read_failure_warns_and_renders_empty_channels(qapp, monkeypatch):
    mic = _asset("mic-1", "Registered mic", "Runtime mic", inputs=2)
    repository = FakeRepository([mic], channel_error=sqlite3.Error("channel read failed"))

    controller, view, warnings = _build_controller(qapp, repository, monkeypatch)
    try:
        view.mic_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "mic-1")
        qapp.processEvents()

        assert warnings
        assert "channel read failed" in warnings[-1][2]
        assert view.mic_channel_table.model().rowCount() == 0
    finally:
        view.close()


def test_malformed_registered_mic_channel_index_warns_and_renders_empty_channels(qapp, monkeypatch):
    mic = _asset("mic-1", "Registered mic", "Runtime mic", inputs=2)
    repository = FakeRepository(
        [mic],
        channels={"mic-1": [_input_channel("mic-1", "Bad channel", "not-an-integer")]},
    )

    controller, view, warnings = _build_controller(qapp, repository, monkeypatch)
    try:
        view.mic_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "mic-1")
        qapp.processEvents()

        assert warnings
        assert warnings[-1][2] == "已注册麦克风通道数据无效，请检查硬件管理数据库。"
        assert view.mic_channel_table.model().rowCount() == 0
    finally:
        view.close()


@pytest.mark.parametrize(
    "select_speaker,select_mic,select_channel",
    [
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ],
)
def test_ok_clicked_rejects_missing_selection_before_runtime_matching(
    qapp, monkeypatch, select_speaker, select_mic, select_channel
):
    speaker = _asset("speaker-1", "Registered speaker", "Runtime speaker", outputs=2)
    mic = _asset("mic-1", "Registered mic", "Runtime mic", inputs=2)
    repository = FakeRepository([speaker, mic], channels={"mic-1": [_input_channel("mic-1", "In1", 0)]})
    runtime_calls = []

    def runtime_enumeration():
        runtime_calls.append(True)
        return {}

    warnings = []
    controller, view, warnings = _build_controller(
        qapp,
        repository,
        monkeypatch,
        runtime_devices=None,
        warnings=warnings,
    )
    monkeypatch.setattr(module.SoundDeviceManager, "get_device_info", staticmethod(runtime_enumeration))
    try:
        if select_speaker:
            view.speaker_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "speaker-1")
        if select_mic:
            view.mic_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "mic-1")
        if select_channel:
            controller._restore_channels(view.mic_channel_table, [0])

        controller._on_ok_clicked()

        assert runtime_calls == []
        assert view.result() != QDialog.Accepted
        assert warnings
        assert "请选择扬声器和麦克风设备，以及麦克风通道" in warnings[-1][2]
    finally:
        view.close()


def test_ok_clicked_warns_and_stays_open_when_runtime_match_is_missing(qapp, monkeypatch):
    speaker = _asset("speaker-1", "Registered speaker", "Runtime speaker", outputs=2)
    mic = _asset("mic-1", "Registered mic", "Runtime mic", inputs=2)
    repository = FakeRepository([speaker, mic], channels={"mic-1": [_input_channel("mic-1", "In1", 0)]})
    runtime_devices = {"API": {"input": [_device(1, "Other mic", inputs=2)], "output": [_device(2, "Runtime speaker", outputs=2)]}}

    controller, view, warnings = _build_controller(qapp, repository, monkeypatch, runtime_devices=runtime_devices)
    try:
        view.speaker_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "speaker-1")
        view.mic_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "mic-1")
        controller._restore_channels(view.mic_channel_table, [0])

        controller._on_ok_clicked()

        assert view.result() != QDialog.Accepted
        assert warnings
        assert "不可用" in warnings[-1][2] or "不存在" in warnings[-1][2]
    finally:
        view.close()


def test_ok_clicked_warns_and_stays_open_when_runtime_match_is_ambiguous(qapp, monkeypatch):
    speaker = _asset("speaker-1", "Registered speaker", "Runtime speaker", outputs=2)
    mic = _asset("mic-1", "Registered mic", "Runtime mic", inputs=2)
    repository = FakeRepository([speaker, mic], channels={"mic-1": [_input_channel("mic-1", "In1", 0)]})
    runtime_devices = {
        "API": {
            "input": [_device(1, "Runtime mic", inputs=2), _device(3, "Runtime mic", inputs=4)],
            "output": [_device(2, "Runtime speaker", outputs=2)],
        }
    }

    controller, view, warnings = _build_controller(qapp, repository, monkeypatch, runtime_devices=runtime_devices)
    try:
        view.speaker_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "speaker-1")
        view.mic_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "mic-1")
        controller._restore_channels(view.mic_channel_table, [0])

        controller._on_ok_clicked()

        assert view.result() != QDialog.Accepted
        assert warnings
        assert "多个" in warnings[-1][2] or "重复" in warnings[-1][2]
    finally:
        view.close()


def test_ok_clicked_rejects_matched_mic_with_zero_runtime_inputs(qapp, monkeypatch):
    speaker = _asset("speaker-1", "Registered speaker", "Runtime speaker", outputs=2)
    mic = _asset("mic-1", "Registered mic", "Runtime mic", inputs=2)
    repository = FakeRepository([speaker, mic], channels={"mic-1": [_input_channel("mic-1", "In1", 0)]})
    runtime_devices = {"API": {"input": [_device(1, "Runtime mic", inputs=0)], "output": [_device(2, "Runtime speaker", outputs=2)]}}
    applied = []
    saved = []
    monkeypatch.setattr(module.SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(module.SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    controller, view, warnings = _build_controller(qapp, repository, monkeypatch, runtime_devices=runtime_devices)
    try:
        view.speaker_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "speaker-1")
        view.mic_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "mic-1")
        controller._restore_channels(view.mic_channel_table, [0])

        controller._on_ok_clicked()

        assert view.result() != QDialog.Accepted
        assert saved == []
        assert applied == []
        assert warnings
    finally:
        view.close()


def test_ok_clicked_rejects_selected_mic_channel_beyond_runtime_capacity(qapp, monkeypatch):
    speaker = _asset("speaker-1", "Registered speaker", "Runtime speaker", outputs=2)
    mic = _asset("mic-1", "Registered mic", "Runtime mic", inputs=4)
    repository = FakeRepository([speaker, mic], channels={"mic-1": [_input_channel("mic-1", "In3", 2)]})
    runtime_devices = {"API": {"input": [_device(1, "Runtime mic", inputs=2)], "output": [_device(2, "Runtime speaker", outputs=2)]}}
    applied = []
    saved = []
    monkeypatch.setattr(module.SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(module.SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    controller, view, warnings = _build_controller(qapp, repository, monkeypatch, runtime_devices=runtime_devices)
    try:
        view.speaker_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "speaker-1")
        view.mic_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "mic-1")
        controller._restore_channels(view.mic_channel_table, [2])

        controller._on_ok_clicked()

        assert view.result() != QDialog.Accepted
        assert saved == []
        assert applied == []
        assert warnings
    finally:
        view.close()


def test_ok_clicked_rejects_matched_speaker_with_zero_runtime_outputs(qapp, monkeypatch):
    speaker = _asset("speaker-1", "Registered speaker", "Runtime speaker", outputs=2)
    mic = _asset("mic-1", "Registered mic", "Runtime mic", inputs=2)
    repository = FakeRepository([speaker, mic], channels={"mic-1": [_input_channel("mic-1", "In1", 0)]})
    runtime_devices = {"API": {"input": [_device(1, "Runtime mic", inputs=2)], "output": [_device(2, "Runtime speaker", outputs=0)]}}
    applied = []
    saved = []
    monkeypatch.setattr(module.SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(module.SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    controller, view, warnings = _build_controller(qapp, repository, monkeypatch, runtime_devices=runtime_devices)
    try:
        view.speaker_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "speaker-1")
        view.mic_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "mic-1")
        controller._restore_channels(view.mic_channel_table, [0])

        controller._on_ok_clicked()

        assert view.result() != QDialog.Accepted
        assert saved == []
        assert applied == []
        assert warnings
    finally:
        view.close()


def test_ok_clicked_matches_runtime_devices_applies_order_and_returns_augmented_payload(qapp, monkeypatch):
    speaker = _asset("speaker-1", "Registered speaker", "Runtime speaker", outputs=2, samplerate=44100, latency_ms=80)
    mic = _asset("mic-1", "Registered mic", "Runtime mic", inputs=2, bit_depth=24)
    repository = FakeRepository([speaker, mic], channels={"mic-1": [_input_channel("mic-1", "In1", 0)]})
    runtime_mic = _device(7, "Runtime mic", hostapi=0, inputs=2)
    runtime_mic["custom_runtime_key"] = "preserved"
    runtime_speaker = _device(9, "Runtime speaker", hostapi=0, outputs=2)
    runtime_speaker["samplerate"] = 96000
    runtime_devices = {"API": {"input": [runtime_mic], "output": [runtime_speaker]}}
    applied = []
    saved = []
    monkeypatch.setattr(module.SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(module.SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    controller, view, warnings = _build_controller(qapp, repository, monkeypatch, runtime_devices=runtime_devices)
    try:
        view.speaker_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "speaker-1")
        view.mic_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "mic-1")
        controller._restore_channels(view.mic_channel_table, [0])

        controller._on_ok_clicked()

        selected_speaker, selected_mic, selected_channels = controller.get_selected_devices()
        assert warnings == []
        assert applied == [(7, 9)]
        assert saved == [(selected_mic, selected_speaker, [0])]
        assert view.result() == QDialog.Accepted
        assert selected_channels == [0]
        assert selected_mic is not runtime_mic
        assert selected_mic["index"] == 7
        assert selected_mic["custom_runtime_key"] == "preserved"
        assert selected_mic["hardware_id"] == "mic-1"
        assert selected_mic["device_name"] == "Runtime mic"
        assert selected_mic["bit_depth"] == 24
        assert selected_speaker is not runtime_speaker
        assert selected_speaker["index"] == 9
        assert selected_speaker["samplerate"] == 96000
        assert selected_speaker["hardware_id"] == "speaker-1"
        assert selected_speaker["latency_ms"] == 80
    finally:
        view.close()


def test_same_audio_interface_for_mic_and_speaker_applies_same_runtime_index(qapp, monkeypatch):
    interface = _asset("iface-1", "Registered interface", "Runtime iface", inputs=2, outputs=2)
    repository = FakeRepository([interface], channels={"iface-1": [_input_channel("iface-1", "In1", 0)]})
    runtime_iface = _device(5, "Runtime iface", hostapi=0, inputs=2, outputs=2)
    runtime_devices = {"API": {"input": [runtime_iface], "output": [runtime_iface]}}
    applied = []
    monkeypatch.setattr(module.SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(module.SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: None))

    controller, view, _warnings = _build_controller(qapp, repository, monkeypatch, runtime_devices=runtime_devices)
    try:
        view.speaker_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "iface-1")
        view.mic_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "iface-1")
        controller._restore_channels(view.mic_channel_table, [0])

        controller._on_ok_clicked()

        assert applied == [(5, 5)]
        assert view.result() == QDialog.Accepted
    finally:
        view.close()


def test_ok_clicked_keeps_dialog_open_when_save_selected_devices_fails(qapp, monkeypatch):
    speaker = _asset("speaker-1", "Registered speaker", "Speaker", outputs=2)
    mic = _asset("mic-1", "Registered mic", "Mic", inputs=2)
    repository = FakeRepository([speaker, mic], channels={"mic-1": [_input_channel("mic-1", "In1", 0)]})
    runtime_devices = {"API": {"input": [_device(1, "Mic", inputs=2)], "output": [_device(2, "Speaker", outputs=2)]}}
    applied = []
    monkeypatch.setattr(module.SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(
        module.SoundDeviceManager,
        "save_selected_devices",
        staticmethod(lambda *args: (_ for _ in ()).throw(OSError("disk full"))),
    )
    warnings = []

    controller, view, warnings = _build_controller(
        qapp,
        repository,
        monkeypatch,
        runtime_devices=runtime_devices,
        warnings=warnings,
    )
    try:
        view.speaker_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "speaker-1")
        view.mic_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "mic-1")
        controller._restore_channels(view.mic_channel_table, [0])

        controller._on_ok_clicked()

        assert view.result() != QDialog.Accepted
        assert applied == []
        assert warnings
        assert "保存" in warnings[-1][2] or "硬件" in warnings[-1][2]
    finally:
        view.close()


def test_ok_clicked_rolls_back_partial_selection_save_failure(qapp, monkeypatch, tmp_path):
    speaker = _asset("speaker-1", "Registered speaker", "Speaker", outputs=2)
    mic = _asset("mic-1", "Registered mic", "Mic", inputs=2)
    repository = FakeRepository([speaker, mic], channels={"mic-1": [_input_channel("mic-1", "In1", 0)]})
    runtime_devices = {"API": {"input": [_device(1, "Mic", inputs=2)], "output": [_device(2, "Speaker", outputs=2)]}}
    config_path = tmp_path / "last_audio_devices.json"
    previous_text = '{\n  "mic": {"name": "Previous mic"},\n  "speaker": {"name": "Previous speaker"},\n  "mic_channels": [1]\n}\n'
    config_path.write_text(previous_text, encoding="utf-8")
    applied = []

    def partial_save(*_args):
        config_path.write_text('{"mic": {"name": "Partial"', encoding="utf-8")
        raise OSError("disk full after partial write")

    monkeypatch.setattr(module.sound_device_manager_module, "AUDIO_DEVICE_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(module.SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(module.SoundDeviceManager, "save_selected_devices", staticmethod(partial_save))

    controller, view, warnings = _build_controller(qapp, repository, monkeypatch, runtime_devices=runtime_devices)
    try:
        view.speaker_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "speaker-1")
        view.mic_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "mic-1")
        controller._restore_channels(view.mic_channel_table, [0])

        controller._on_ok_clicked()

        assert view.result() != QDialog.Accepted
        assert warnings
        assert "保存" in warnings[-1][2]
        assert applied == []
        assert config_path.read_text(encoding="utf-8") == previous_text
    finally:
        view.close()


def test_ok_clicked_rolls_back_persisted_selection_when_runtime_apply_fails(qapp, monkeypatch, tmp_path):
    speaker = _asset("speaker-1", "Registered speaker", "Speaker", outputs=2)
    mic = _asset("mic-1", "Registered mic", "Mic", inputs=2)
    repository = FakeRepository([speaker, mic], channels={"mic-1": [_input_channel("mic-1", "In1", 0)]})
    runtime_devices = {"API": {"input": [_device(1, "Mic", inputs=2)], "output": [_device(2, "Speaker", outputs=2)]}}
    config_path = tmp_path / "last_audio_devices.json"
    previous_payload = {
        "mic": {"name": "Previous mic", "hostapi_name": "API", "default_samplerate": 44100},
        "speaker": {"name": "Previous speaker", "hostapi_name": "API", "default_samplerate": 44100},
        "mic_channels": [2],
    }
    config_path.write_text(json.dumps(previous_payload, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(module.sound_device_manager_module, "AUDIO_DEVICE_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(module.SoundDeviceManager, "get_api_info", staticmethod(lambda _hostapi=None: {"name": "API"}))
    monkeypatch.setattr(
        module.SoundDeviceManager,
        "change_default_device",
        staticmethod(lambda *args: (_ for _ in ()).throw(RuntimeError("apply failed"))),
    )

    controller, view, warnings = _build_controller(qapp, repository, monkeypatch, runtime_devices=runtime_devices)
    try:
        view.speaker_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "speaker-1")
        view.mic_device_table.set_checked_by_predicate(lambda payload: payload["hardware_id"] == "mic-1")
        controller._restore_channels(view.mic_channel_table, [0])

        controller._on_ok_clicked()

        assert view.result() != QDialog.Accepted
        assert warnings
        assert "应用" in warnings[-1][2]
        assert json.loads(config_path.read_text(encoding="utf-8")) == previous_payload
    finally:
        view.close()

"""Real UI/store/snapshot/WAV acceptance with only hardware replaced."""
from copy import deepcopy
import json

import numpy as np
import pytest
import soundfile as sf
from PyQt5.QtTest import QTest

from base import soundcard_calibration_manager as mic_store
from base.recording_calibration_snapshot import build_recording_wav_calibration_metadata
from base.recording_process_protocol import FrozenConfig
from base.streaming_file_writer import StreamingWavWriter
from base.ve3668n_stores import VECalibrationStore, VEInputProfileStore
from base.wav_calibration_metadata import (
    WavCalibrationMetadataReadStatus,
    append_wav_calibration_metadata,
    inspect_wav_calibration_metadata,
    read_wav_calibration_metadata,
    resolve_wav_channel_v2pa_factor,
)
from ui.calibration_window import CalibrationWindow
from unit_test.base.ve3668n_fakes import device_info
from unit_test.ui.test_ve3668n_calibration import CalibrationBridgeFake


@pytest.fixture
def windows(ui_qapp):
    opened = []

    def make(device, channels, bridge, **kwargs):
        window = CalibrationWindow(
            deepcopy(device), channels, recording_bridge=bridge, **kwargs,
        )
        opened.append(window)
        window.show()
        window.activateWindow()
        ui_qapp.processEvents()
        return window

    yield make
    for window in opened:
        window.close()
        window.deleteLater()
    ui_qapp.processEvents()


def _edit_and_blur(window, text, qapp):
    panel = window.input_cal_wnd
    line = panel.v2pa_factor_lineedit
    assert line.isEnabled() and not line.isReadOnly()
    line.setFocus()
    qapp.processEvents()
    assert line.hasFocus()
    line.selectAll()
    QTest.keyClicks(line, text)
    panel.standard_spl_ii.setFocus()
    qapp.processEvents()
    assert not line.hasFocus()
    assert float(line.text()) == float(text)
    assert window.input_calibration_flag
    assert panel.streaming_processor is None


def _write_recording(path, snapshot, sample_rate):
    # Binary fractions are exactly representable in PCM24: any coefficient
    # application to stored samples would be observable, without rounding noise.
    samples = np.array([[0.25, -0.5], [-0.75, 0.125], [0.0, 0.5]], dtype=np.float32)
    writer = StreamingWavWriter(str(path), sample_rate=sample_rate, channels=2)
    try:
        writer.write_chunk(samples)
    finally:
        writer.finalize()
    original = path.read_bytes()
    assert append_wav_calibration_metadata(path, snapshot.to_dict())
    # Only RIFF size and appended metadata change, never the original audio.
    assert path.read_bytes()[8:len(original)] == original[8:]
    read_samples, read_rate = sf.read(path, dtype="float32", always_2d=True)
    assert read_rate == sample_rate
    np.testing.assert_array_equal(read_samples, samples)


def _assert_file_factor(path, factor):
    result = inspect_wav_calibration_metadata(path)
    assert result.status is WavCalibrationMetadataReadStatus.VALID
    metadata = read_wav_calibration_metadata(path)
    assert metadata == result.metadata
    resolution = resolve_wav_channel_v2pa_factor(metadata, 0)
    assert resolution.factor == factor
    assert resolution.has_valid_metadata and resolution.used_file_metadata
    assert metadata["recorded_channels"][0]["calibrated"] is True
    return metadata


def _assert_history_and_new_recording(tmp_path, first, second, rate, k1, k2, a_bytes, expected_first):
    path_a = tmp_path / "a.wav"
    _write_recording(tmp_path / "b.wav", second, rate)
    assert _assert_file_factor(path_a, k1) == expected_first
    assert _assert_file_factor(tmp_path / "b.wav", k2) == second.to_dict()
    assert first.to_dict() == expected_first
    assert path_a.read_bytes() == a_bytes


def test_mic_edit_round_trip_keeps_old_wav_snapshot(ui_qapp, tmp_path, monkeypatch, windows):
    path = tmp_path / "mic.json"
    monkeypatch.setattr(mic_store, "MIC_INPUT_CALIBRATION_PATH", str(path))
    monkeypatch.setattr(mic_store.SoundDeviceManager, "get_api_info",
                        lambda _index: {"name": "Test API"})
    device = {"index": 7, "name": "Test Microphone", "hostapi": 3, "max_input_channels": 3}
    bridge = CalibrationBridgeFake(tmp_path)
    mic_store.save_mic_channel_calibration(
        8.0, device, 0, 94, 48000, 10,
        calibrated_at="2026-10-07T15:00:00+08:00",
    )
    untouched = mic_store.load_mic_channel_calibrations(device)[0]
    k1, k2 = 0.12345678901234567, 1.234567890123456e-12
    window = windows(device, [2, 0], bridge)
    assert window.input_cal_wnd.current_channel == 2
    assert window.input_cal_wnd.v2pa_factor_lineedit.text() == ""
    _edit_and_blur(window, repr(k1), ui_qapp)
    window.close()
    reopened = windows(device, [2, 0], bridge)
    assert float(reopened.input_cal_wnd.v2pa_factor_lineedit.text()) == k1
    first = FrozenConfig.snapshot(build_recording_wav_calibration_metadata([2, 0], device))
    expected_first = first.to_dict()
    _write_recording(tmp_path / "a.wav", first, 48000)
    a_bytes = (tmp_path / "a.wav").read_bytes()
    _assert_file_factor(tmp_path / "a.wav", k1)

    _edit_and_blur(reopened, repr(k2), ui_qapp)
    second = FrozenConfig.snapshot(build_recording_wav_calibration_metadata([2, 0], device))
    assert second.to_dict()["recorded_channels"][0]["standard_spl"] is None
    _assert_history_and_new_recording(tmp_path, first, second, 48000, k1, k2, a_bytes, expected_first)
    records = mic_store.load_mic_channel_calibrations(device)
    assert records[0] == untouched
    assert records[2]["v2pa_factor"] == k2
    assert all(records[2][key] is None for key in
               ("standard_spl_db", "sample_rate_hz", "duration_seconds"))
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 3
    assert bridge.sessions == []


def test_ve_edit_round_trip_keeps_old_wav_snapshot(ui_qapp, tmp_path, windows):
    device = device_info(physical_channels=[7, 1, 3])
    profiles = VEInputProfileStore(tmp_path / "profiles.json")
    store = VECalibrationStore(tmp_path / "ve.json")
    profiles.set_sample_rate(device, 51200, store)
    untouched = store.save(
        device, 3, v2pa_factor=8.0, standard_spl=94,
        calibration_sample_rate=51200, calibration_duration_seconds=10,
        calibrated_at="2026-10-07T15:00:00+08:00",
    )
    bridge = CalibrationBridgeFake(tmp_path)
    k1, k2 = 0.12345678901234567, 1.234567890123456e-12
    window = windows(device, [7, 1], bridge, ve_profile_store=profiles, ve_calibration_store=store)
    assert window.input_cal_wnd.current_channel == 7
    assert window.input_cal_wnd.v2pa_factor_lineedit.text() == ""
    _edit_and_blur(window, repr(k1), ui_qapp)
    window.close()
    # Fresh store instances prove reopening reads persisted data, not a UI cache.
    store = VECalibrationStore(store.path)
    profiles = VEInputProfileStore(profiles.path)
    reopened = windows(device, [7, 1], bridge, ve_profile_store=profiles, ve_calibration_store=store)
    # Opening retains the existing preference for an uncalibrated channel.
    combo = reopened.input_cal_wnd.channel_combo_box
    combo.setCurrentIndex(combo.findData(7))
    ui_qapp.processEvents()
    assert reopened.input_cal_wnd.current_channel == 7
    assert float(reopened.input_cal_wnd.v2pa_factor_lineedit.text()) == k1
    assert reopened.input_cal_wnd.channel_status_label.text() == "状态: 校准有效"
    first = FrozenConfig.snapshot(build_recording_wav_calibration_metadata(
        [7, 1], device, ve_calibration_store=store,
    ))
    expected_first = first.to_dict()
    _write_recording(tmp_path / "a.wav", first, 51200)
    a_bytes = (tmp_path / "a.wav").read_bytes()
    _assert_file_factor(tmp_path / "a.wav", k1)

    _edit_and_blur(reopened, repr(k2), ui_qapp)
    second = FrozenConfig.snapshot(build_recording_wav_calibration_metadata(
        [7, 1], device, ve_calibration_store=store,
    ))
    _assert_history_and_new_recording(tmp_path, first, second, 51200, k1, k2, a_bytes, expected_first)
    for snapshot in (first, second):
        metadata = snapshot.to_dict()
        assert metadata["schema_version"] == 2
        calibrated, uncalibrated = metadata["recorded_channels"]
        assert [channel["physical_input_channel"] for channel in (calibrated, uncalibrated)] == [7, 1]
        assert calibrated["factor_source"] == "calibrated"
        assert all(calibrated["calibration"][key] is None for key in
                   ("standard_spl", "sample_rate", "duration_seconds"))
        assert uncalibrated["factor_source"] == "none"
        assert uncalibrated["calibrated"] is False
        assert uncalibrated["v2pa_factor"] is None
        assert uncalibrated["calibration"] is None
        resolution = resolve_wav_channel_v2pa_factor(metadata, 1)
        assert resolution.factor == 1.0
        assert resolution.has_valid_metadata and not resolution.used_file_metadata
    assert store.get_record(device, 3) == untouched
    assert store.get_factor(device, 7) == k2
    assert json.loads(store.path.read_text(encoding="utf-8"))["schema_version"] == 2
    assert json.loads(profiles.path.read_text(encoding="utf-8"))["schema_version"] == 1
    assert bridge.sessions == []

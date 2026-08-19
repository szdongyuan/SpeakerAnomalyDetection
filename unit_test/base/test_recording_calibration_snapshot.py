import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base import recording_calibration_snapshot


class FakeLogger:
    def __init__(self):
        self.messages = []

    def warning(self, message):
        self.messages.append(message)

    def error(self, message):
        self.messages.append(message)


def test_build_recording_wav_metadata_uses_active_channel_order(monkeypatch):
    monkeypatch.setattr(
        recording_calibration_snapshot,
        "load_mic_channel_v2pa_factors",
        lambda hardware_id=None, db_path=None: {2: 2.5, 4: 4.5},
        raising=False,
    )
    monkeypatch.setattr(
        recording_calibration_snapshot,
        "load_mic_channel_standard_spl",
        lambda hardware_id=None, db_path=None: {2: 94.0, 4: 114.0},
        raising=False,
    )

    metadata = recording_calibration_snapshot.build_recording_wav_calibration_metadata(
        [2, 4], hardware_id="mic-1", logger=FakeLogger()
    )

    assert metadata == {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True},
            {"wav_channel_index": 1, "v2pa_factor": 4.5, "standard_spl": 114.0, "calibrated": True},
        ]
    }


def test_build_recording_wav_metadata_treats_manual_factor_as_calibrated(monkeypatch):
    monkeypatch.setattr(
        recording_calibration_snapshot,
        "load_mic_channel_v2pa_factors",
        lambda hardware_id=None, db_path=None: {2: 1.234567},
        raising=False,
    )
    monkeypatch.setattr(
        recording_calibration_snapshot,
        "load_mic_channel_standard_spl",
        lambda hardware_id=None, db_path=None: {},
        raising=False,
    )

    metadata = recording_calibration_snapshot.build_recording_wav_calibration_metadata(
        [2], hardware_id="mic-1", logger=FakeLogger()
    )

    assert metadata == {
        "recorded_channels": [
            {
                "wav_channel_index": 0,
                "v2pa_factor": 1.234567,
                "standard_spl": None,
                "calibrated": True,
            }
        ]
    }


def test_build_recording_wav_metadata_marks_uncalibrated_channels(monkeypatch):
    monkeypatch.setattr(
        recording_calibration_snapshot,
        "load_mic_channel_v2pa_factors",
        lambda hardware_id=None, db_path=None: {},
        raising=False,
    )
    monkeypatch.setattr(
        recording_calibration_snapshot,
        "load_mic_channel_standard_spl",
        lambda hardware_id=None, db_path=None: {},
        raising=False,
    )

    metadata = recording_calibration_snapshot.build_recording_wav_calibration_metadata(
        [2], hardware_id="mic-1", logger=FakeLogger()
    )

    assert metadata == {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": None, "standard_spl": None, "calibrated": False},
        ]
    }


def test_build_recording_wav_metadata_marks_all_channels_uncalibrated_when_lookup_fails(monkeypatch):
    logger = FakeLogger()

    def fail_lookup(hardware_id=None, db_path=None):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(recording_calibration_snapshot, "load_mic_channel_v2pa_factors", fail_lookup, raising=False)

    metadata = recording_calibration_snapshot.build_recording_wav_calibration_metadata(
        [1, 3], hardware_id="mic-1", logger=logger
    )

    assert metadata == {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": None, "standard_spl": None, "calibrated": False},
            {"wav_channel_index": 1, "v2pa_factor": None, "standard_spl": None, "calibrated": False},
        ]
    }
    assert logger.messages

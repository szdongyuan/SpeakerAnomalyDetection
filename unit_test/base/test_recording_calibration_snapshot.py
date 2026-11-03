import sys
import json
from pathlib import Path

import pytest

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

def test_vkinging_snapshot_uses_current_device_and_saved_channel_conditions(monkeypatch):
    conditions = {'input_mode': 'IEPE', 'unit': 'V', 'range_min': -10.0, 'range_max': 10.0}
    class Repository:
        def __init__(self, db_path):
            pass
        def list_channel_calibrations(self, hardware_id, direction, calibration_type):
            assert (hardware_id, direction, calibration_type) == ('machine-A', 'input', 'mic_v2pa')
            return [{'channel_index': 15, 'factor_value': 3.5, 'standard_spl': 94.0,
                     'coefficients_json': json.dumps(conditions)}]
    monkeypatch.setattr(recording_calibration_snapshot, 'HardwareManagementRepository', Repository)
    result = recording_calibration_snapshot.build_vkinging_recording_wav_calibration_metadata(
        (15, 0, 8), device={'hardware_id': 'machine-A', 'hostapi_name': 'vkinging',
                          'model': 'raw-model', 'samplerate': 51200, 'bit_depth': 64,
                          'physical_channels': (0, 8, 15)}, range_index=2)
    assert result['backend'] == 'vkinging'
    assert result['schema_version'] == 1
    assert result['acquisition'] == {'hardware_id': 'machine-A', 'model': 'raw-model',
                                    'sample_rate': 51200, 'bit_depth': 64, 'input_mode': 'IEPE',
                                    'unit': 'V', 'range_min': -2.5, 'range_max': 2.5}
    assert [ch['physical_channel_index'] for ch in result['recorded_channels']] == [15, 0, 8]
    assert result['recorded_channels'][0] == {
        'wav_channel_index': 0, 'physical_channel_index': 15, 'calibrated': True,
        'v2pa_factor': 3.5, 'standard_spl': 94.0, 'calibration_conditions': conditions}
    assert result['recorded_channels'][1]['calibrated'] is False


@pytest.mark.parametrize('bad_json', ['not json', '{}', 'null'])
def test_vkinging_snapshot_does_not_absorb_malformed_database_data(monkeypatch, bad_json):
    class Repository:
        def __init__(self, db_path):
            pass
        def list_channel_calibrations(self, *args):
            return [{'channel_index': 15, 'factor_value': 2.0, 'standard_spl': None,
                     'coefficients_json': bad_json}]
    monkeypatch.setattr(recording_calibration_snapshot, 'HardwareManagementRepository', Repository)
    with pytest.raises(ValueError):
        recording_calibration_snapshot.build_vkinging_recording_wav_calibration_metadata(
            (15,), device={'hardware_id': 'machine-A', 'hostapi_name': 'vkinging',
                           'samplerate': 51200, 'bit_depth': 32, 'physical_channels': (15,)},
            range_index=0)

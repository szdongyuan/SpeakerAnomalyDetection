import json
import inspect

import pytest

from base.hardware_management import HardwareManagementRepository
from base.soundcard_calibration_manager import SoundcardCalibrationManager
from consts import error_code
from unit_test.base.test_hardware_management import create_system_db, runtime_device
from unit_test.compare_methods import assert_equal


@pytest.mark.parametrize(
    "amplitude, voltage, result_set",
    [
        ([], 3, (error_code.INVALID_DATA_LOADING, "Input data cannot be None.")),
        (0.1, [], (error_code.INVALID_DATA_LOADING, "Input data cannot be None.")),
        ([0.1], 3, (error_code.INVALID_TYPE_DATA, "Input data must be numeric.")),
        (0.1, 3, (error_code.OK, "Successfully add data.")),
    ],
)
def test_add_data(amplitude, voltage, result_set):
    result = SoundcardCalibrationManager().add_data(amplitude, voltage)
    assert result == result_set


@pytest.mark.parametrize(
    "coefficients, target_voltage, result_ret",
    [
        ([0.05038, -0.0008392], 3, 0.1503),
        ([0.05038, -0.0008392], [1, 2], [0.0495, 0.0999]),
        ([0.05038, -0.0008392], [], []),
    ],
)
def test_predict_amplitude(coefficients, target_voltage, result_ret):
    result = SoundcardCalibrationManager().predict_amplitude(coefficients, target_voltage)
    assert_equal(result, result_ret)


def test_fit_saves_output_calibration_to_database(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=0, outputs=1), "Windows WASAPI", "Speaker", 48000)

    manager = SoundcardCalibrationManager(db_path=str(db_path), speaker_hardware_id=asset["hardware_id"])
    manager.add_data(0, 0, validation=False)
    manager.add_data(0.5, 1.0)

    code, coefficients = manager.fit()

    assert code == error_code.OK
    row = repo.get_output_amplitude_calibration(asset["hardware_id"])
    assert row["max_voltage"] == 1.0
    assert "max_voltage" not in json.loads(row["coefficients_json"])
    assert_equal(coefficients, [0.5, 0.0])


def test_fit_blocks_without_registered_speaker_hardware(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    manager = SoundcardCalibrationManager(db_path=str(db_path))
    manager.add_data(0, 0, validation=False)
    manager.add_data(0.5, 1.0)

    code, message = manager.fit()

    assert code == error_code.INVALID_DATA_LOADING
    assert "registered speaker hardware" in message


def test_calibrate_amplitude_reads_database_and_blocks_when_missing(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=0, outputs=1), "Windows WASAPI", "Speaker", 48000)

    missing_code, _ = SoundcardCalibrationManager(
        db_path=str(db_path),
        speaker_hardware_id=asset["hardware_id"],
    ).calibrate_amplitude(0.5)
    assert missing_code == error_code.INVALID_DATA_LOADING

    repo.update_output_amplitude_calibration(asset["hardware_id"], [0.5, 0.0], max_voltage=1.0)
    code, result = SoundcardCalibrationManager(
        db_path=str(db_path),
        speaker_hardware_id=asset["hardware_id"],
    ).calibrate_amplitude(0.5)

    assert code == error_code.OK
    assert result == (0.25, 1.0)


def test_output_calibration_api_exposes_only_database_backed_paths():
    assert list(inspect.signature(SoundcardCalibrationManager.fit).parameters) == ["self", "threshold"]
    assert list(inspect.signature(SoundcardCalibrationManager.calibrate_amplitude).parameters) == ["self", "target_voltage"]
    assert not hasattr(SoundcardCalibrationManager, "save_coefficients_to_json")


def test_db_output_calibration_ignores_legacy_json_when_file_path_omitted(tmp_path, monkeypatch):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=0, outputs=1), "Windows WASAPI", "Speaker", 48000)
    repo.update_output_amplitude_calibration(asset["hardware_id"], [0.5, 0.0], max_voltage=1.0)

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "calibration_coefficients.json").write_text(
        json.dumps({"calibration_coefficients": [99.0, 0.0], "max_voltage": 99.0}),
        encoding="utf-8",
    )

    code, result = SoundcardCalibrationManager(
        db_path=str(db_path),
        speaker_hardware_id=asset["hardware_id"],
    ).calibrate_amplitude(0.5)

    assert code == error_code.OK
    assert result == (0.25, 1.0)

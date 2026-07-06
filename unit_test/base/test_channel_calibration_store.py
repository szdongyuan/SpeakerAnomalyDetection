import json
import inspect
import sqlite3

import pytest

from base.hardware_management import HardwareManagementRepository
from base.soundcard_calibration_manager import (
    MicChannelCalibrationResult,
    clear_mic_channel_v2pa_factors,
    format_input_channel_label,
    load_mic_channel_v2pa_factors,
    replace_mic_channel_v2pa_factors,
    resolve_analysis_v2pa_factor_for_channel,
    resolve_mic_channel_v2pa_factor,
    save_mic_channel_v2pa_factor,
)
from consts import model_consts
from unit_test.base.test_hardware_management import create_system_db, runtime_device


def test_db_load_and_replace_channel_v2pa_factors(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=2, outputs=0), "Windows WASAPI", "Mic", 48000)

    replace_mic_channel_v2pa_factors(
        {0: 1.25, 1: 2.5},
        channel_standard_spl={0: 94, 1: 114},
        hardware_id=asset["hardware_id"],
        db_path=str(db_path),
    )

    assert load_mic_channel_v2pa_factors(hardware_id=asset["hardware_id"], db_path=str(db_path)) == {0: 1.25, 1: 2.5}


def test_db_load_rejects_invalid_persisted_mic_calibration_row(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=1, outputs=0), "Windows WASAPI", "Mic", 48000)
    with sqlite3.connect(str(db_path)) as connection:
        connection.execute(
            """
            UPDATE hardware_channel_calibrations
            SET calibration_type = ?, factor_value = ?
            WHERE hardware_id = ? AND direction = 'input' AND channel_index = 0
            """,
            ("mic_v2pa", "bad", asset["hardware_id"]),
        )

    with pytest.raises(ValueError, match="Invalid microphone calibration payload"):
        load_mic_channel_v2pa_factors(hardware_id=asset["hardware_id"], db_path=str(db_path))


def test_db_replace_preserves_existing_standard_spl_for_reused_channel(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=2, outputs=0), "Windows WASAPI", "Mic", 48000)
    repo.update_mic_channel_calibrations(asset["hardware_id"], {0: 1.25}, channel_standard_spl={0: 114})

    replace_mic_channel_v2pa_factors(
        {0: 1.5, 1: 2.5},
        channel_standard_spl={1: 94},
        hardware_id=asset["hardware_id"],
        db_path=str(db_path),
    )

    assert repo.get_channel_calibration(
        asset["hardware_id"], "input", 0, "mic_v2pa"
    )["standard_spl"] == 114
    assert repo.get_channel_calibration(
        asset["hardware_id"], "input", 1, "mic_v2pa"
    )["standard_spl"] == 94


def test_db_clear_channel_v2pa_factors(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=2, outputs=0), "Windows WASAPI", "Mic", 48000)
    repo.update_mic_channel_calibrations(asset["hardware_id"], {0: 1.25, 1: 2.5}, channel_standard_spl={0: 94, 1: 94})

    clear_mic_channel_v2pa_factors(hardware_id=asset["hardware_id"], channel_indices=[1], db_path=str(db_path))

    assert load_mic_channel_v2pa_factors(hardware_id=asset["hardware_id"], db_path=str(db_path)) == {0: 1.25}


def test_db_resolve_analysis_factor_uses_exact_hardware_channel(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=2, outputs=0), "Windows WASAPI", "Mic", 48000)
    repo.update_mic_channel_calibrations(asset["hardware_id"], {1: 2.5}, channel_standard_spl={1: 94})

    factor = resolve_analysis_v2pa_factor_for_channel(
        1,
        hardware_id=asset["hardware_id"],
        db_path=str(db_path),
    )

    assert factor == 2.5


def test_db_resolve_analysis_factor_warns_and_returns_one_when_missing(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=1, outputs=0), "Windows WASAPI", "Mic", 48000)
    warnings = []

    factor = resolve_analysis_v2pa_factor_for_channel(
        0,
        hardware_id=asset["hardware_id"],
        db_path=str(db_path),
        warn_callback=warnings.append,
    )

    assert factor == 1.0
    assert warnings == ["麦克风未进行校准，结果仅供参考。"]
    assert repo.get_channel_calibration(asset["hardware_id"], "input", 0, "mic_v2pa") is None


def test_db_resolve_analysis_factor_warns_every_time_missing(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=1, outputs=0), "Windows WASAPI", "Mic", 48000)
    warnings = []

    first = resolve_analysis_v2pa_factor_for_channel(0, hardware_id=asset["hardware_id"], db_path=str(db_path), warn_callback=warnings.append)
    second = resolve_analysis_v2pa_factor_for_channel(0, hardware_id=asset["hardware_id"], db_path=str(db_path), warn_callback=warnings.append)

    assert first == 1.0
    assert second == 1.0
    assert warnings == ["麦克风未进行校准，结果仅供参考。", "麦克风未进行校准，结果仅供参考。"]

def test_db_resolve_channel_factor_does_not_fallback_to_another_channel(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=2, outputs=0), "Windows WASAPI", "Mic", 48000)
    repo.update_mic_channel_calibrations(asset["hardware_id"], {1: 2.5}, channel_standard_spl={1: 94})

    result = resolve_mic_channel_v2pa_factor(0, hardware_id=asset["hardware_id"], db_path=str(db_path))

    assert isinstance(result, MicChannelCalibrationResult)
    assert result.factor is None
    assert result.requested_channel == 0
    assert result.source_channel is None
    assert result.used_fallback is False
    assert result.has_any_calibration is True


def test_db_resolve_analysis_factor_ignores_legacy_file_when_file_path_omitted(tmp_path, monkeypatch):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=1, outputs=0), "Windows WASAPI", "Mic", 48000)
    repo.update_mic_channel_calibrations(asset["hardware_id"], {0: 2.5}, channel_standard_spl={0: 94})
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "mic_channel_calibration.json").write_text(
        json.dumps({"version": 1, "channels": {"0": {"v2pa_factor": 99.0}}}),
        encoding="utf-8",
    )

    assert resolve_analysis_v2pa_factor_for_channel(0, hardware_id=asset["hardware_id"], db_path=str(db_path)) == 2.5


def test_mic_calibration_helpers_expose_only_database_backed_paths():
    helpers = [
        load_mic_channel_v2pa_factors,
        save_mic_channel_v2pa_factor,
        clear_mic_channel_v2pa_factors,
        replace_mic_channel_v2pa_factors,
        resolve_mic_channel_v2pa_factor,
        resolve_analysis_v2pa_factor_for_channel,
    ]

    for helper in helpers:
        assert "file_path" not in inspect.signature(helper).parameters


def test_format_input_channel_label_uses_one_based_display():
    assert format_input_channel_label(0) == "In1"
    assert format_input_channel_label("2") == "In3"

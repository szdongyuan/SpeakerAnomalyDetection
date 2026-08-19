import json
import math
import sqlite3

import pytest

from base.db_manager import DataSave
from base.hardware_management import (
    HardwareManagementRepository,
    HardwareRuntimeMatchError,
    HardwareValidationError,
    augment_runtime_device,
    build_channel_placeholders,
    build_selected_device_payload,
    match_runtime_device,
)
from consts import error_code
from consts import model_consts


def test_hardware_database_constants_match_expected_schema():
    import sqlite3
    from consts import model_consts

    assert model_consts.HARDWARE_ASSETS_TABLE == "hardware_assets"
    assert model_consts.HARDWARE_CHANNEL_CALIBRATIONS_TABLE == "hardware_channel_calibrations"
    assert model_consts.HARDWARE_ASSET_COLUMNS == (
        "hardware_id",
        "hardware_type",
        "display_name",
        "device_name",
        "hostapi_name",
        "samplerate",
        "bit_depth",
        "latency_ms",
        "max_input_channels",
        "max_output_channels",
        "updated_at",
    )
    assert model_consts.HARDWARE_CHANNEL_CALIBRATION_COLUMNS == (
        "channel_id",
        "hardware_id",
        "direction",
        "channel_index",
        "channel_label",
        "calibration_type",
        "factor_value",
        "standard_spl",
        "max_voltage",
        "coefficients_json",
        "updated_at",
    )
    assert sqlite3.Error in model_consts.SQLITE_REPOSITORY_EXCEPTIONS


def test_hardware_management_no_longer_exports_moved_constants():
    import base.hardware_management as module

    for name in (
        "MISSING_HARDWARE_TABLES_MESSAGE",
        "HARDWARE_ASSET_COLUMNS",
        "CHANNEL_COLUMNS",
        "VALID_BIT_DEPTHS",
    ):
        assert not hasattr(module, name)


def create_system_db(db_path):
    data_save = DataSave(str(db_path))
    try:
        code, message = data_save.create_system_tables()
    finally:
        if data_save.connection is not None:
            data_save.close()
    assert code == error_code.OK, message


def runtime_device(name="USB Mic", inputs=2, outputs=1, hostapi=0, index=7):
    return {
        "index": index,
        "name": name,
        "hostapi": hostapi,
        "default_samplerate": 48000.0,
        "max_input_channels": inputs,
        "max_output_channels": outputs,
        "extra_runtime_key": "preserved",
    }


def registered_asset(**overrides):
    asset = {
        "hardware_id": "hardware-1",
        "hardware_type": "audio_interface",
        "display_name": "Desk Interface",
        "device_name": "USB Mic",
        "hostapi_name": "Windows WASAPI",
        "samplerate": 48000,
        "bit_depth": 32,
        "latency_ms": 100,
        "max_input_channels": 2,
        "max_output_channels": 1,
        "updated_at": "2026-06-23 12:00:00",
    }
    asset.update(overrides)
    return asset


def test_tables_exist_requires_both_hardware_tables(tmp_path):
    missing_db = tmp_path / "missing.db"
    repo = HardwareManagementRepository(str(missing_db))
    assert repo.tables_exist() is False

    with sqlite3.connect(missing_db) as connection:
        connection.execute("CREATE TABLE hardware_assets(hardware_id TEXT PRIMARY KEY)")
    assert repo.tables_exist() is False

    full_db = tmp_path / "system_data.db"
    create_system_db(full_db)
    assert HardwareManagementRepository(str(full_db)).tables_exist() is True


def test_register_asset_inserts_asset_and_channel_placeholders(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))

    asset = repo.register_asset(
        runtime_device(inputs=2, outputs=2),
        hostapi_name="Windows WASAPI",
        display_name="Desk Interface",
        samplerate=48000,
        bit_depth=32,
        latency_ms=100,
    )

    stored_asset = repo.get_asset(asset["hardware_id"])
    assert stored_asset["display_name"] == "Desk Interface"
    assert stored_asset["hardware_type"] == "audio_interface"
    assert stored_asset["device_name"] == "USB Mic"
    assert stored_asset["hostapi_name"] == "Windows WASAPI"
    assert stored_asset["samplerate"] == 48000
    assert stored_asset["bit_depth"] == 32
    assert stored_asset["latency_ms"] == 100

    channels = repo.list_channels(asset["hardware_id"])
    assert [(row["direction"], row["channel_index"], row["channel_label"]) for row in channels] == [
        ("input", 0, "In1"),
        ("input", 1, "In2"),
        ("output", 0, "Out1"),
        ("output", 1, "Out2"),
    ]
    assert all(row["calibration_type"] is None for row in channels)
    assert all(row["factor_value"] is None for row in channels)
    assert all(row["standard_spl"] is None for row in channels)
    assert all(row["max_voltage"] is None for row in channels)
    assert all(row["coefficients_json"] is None for row in channels)


def test_update_and_read_mic_channel_calibration(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=2, outputs=0), "Windows WASAPI", "Mic", 48000)

    repo.update_mic_channel_calibrations(
        asset["hardware_id"],
        {1: 2.5},
        channel_standard_spl={1: 94},
    )

    row = repo.get_channel_calibration(
        asset["hardware_id"],
        "input",
        1,
        "mic_v2pa",
    )
    assert row["factor_value"] == 2.5
    assert row["standard_spl"] == 94
    assert row["calibration_type"] == "mic_v2pa"
    assert row["max_voltage"] is None
    assert row["coefficients_json"] is None


def test_update_output_calibration_stores_max_voltage_outside_coefficients_json(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=0, outputs=2), "Windows WASAPI", "Speaker", 48000)

    repo.update_output_amplitude_calibration(asset["hardware_id"], [0.5, 0.1], max_voltage=1.25)

    row = repo.get_channel_calibration(
        asset["hardware_id"],
        "output",
        0,
        "output_amplitude",
    )
    assert row["calibration_type"] == "output_amplitude"
    assert row["factor_value"] is None
    assert row["standard_spl"] is None
    assert row["max_voltage"] == 1.25
    assert json.loads(row["coefficients_json"]) == {"calibration_coefficients": [0.5, 0.1]}


def test_calibration_update_rejects_missing_channel_placeholder(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=1, outputs=0), "Windows WASAPI", "Mic", 48000)

    with pytest.raises(HardwareValidationError, match="channel"):
        repo.update_mic_channel_calibrations(asset["hardware_id"], {2: 2.5}, channel_standard_spl={2: 94})


def test_mic_nullable_multi_channel_update_rolls_back_before_missing_placeholder(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=1, outputs=0), "Windows WASAPI", "Mic", 48000)
    repo.update_mic_channel_calibrations(
        asset["hardware_id"], {0: 2.0}, channel_standard_spl={0: 114}
    )

    with pytest.raises(HardwareValidationError, match="channel"):
        repo.update_mic_channel_calibrations(
            asset["hardware_id"],
            {0: 1.25, 1: 2.5},
            channel_standard_spl={0: None, 1: 94},
        )

    row = repo.get_channel_calibration(asset["hardware_id"], "input", 0, "mic_v2pa")
    assert row["factor_value"] == 2.0
    assert row["standard_spl"] == 114


def test_output_calibration_rejects_missing_channel_placeholder(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=0, outputs=0), "Windows WASAPI", "Speaker", 48000)

    with pytest.raises(HardwareValidationError, match="channel"):
        repo.update_output_amplitude_calibration(asset["hardware_id"], [0.5, 0.0], max_voltage=1.0)


def test_mic_calibration_validates_factor_and_standard_spl(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=1, outputs=0), "Windows WASAPI", "Mic", 48000)

    for factor in (0, -1, math.inf, math.nan, "bad"):
        with pytest.raises(HardwareValidationError):
            repo.update_mic_channel_calibrations(
                asset["hardware_id"],
                {0: factor},
                channel_standard_spl={0: 94},
            )

    for standard_spl in (math.inf, math.nan, "bad"):
        with pytest.raises(HardwareValidationError):
            repo.update_mic_channel_calibrations(
                asset["hardware_id"],
                {0: 1.25},
                channel_standard_spl={0: standard_spl},
            )


def test_mic_calibration_accepts_explicit_manual_null_standard_spl(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=1, outputs=0), "Windows WASAPI", "Mic", 48000)

    repo.update_mic_channel_calibrations(
        asset["hardware_id"], {0: 1.234567}, channel_standard_spl={0: None}
    )

    row = repo.get_channel_calibration(asset["hardware_id"], "input", 0, "mic_v2pa")
    assert row["factor_value"] == 1.234567
    assert row["standard_spl"] is None


def test_mic_calibration_still_rejects_missing_provenance_key(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=1, outputs=0), "Windows WASAPI", "Mic", 48000)

    with pytest.raises(HardwareValidationError, match="standard_spl"):
        repo.update_mic_channel_calibrations(
            asset["hardware_id"], {0: 1.25}, channel_standard_spl={}
        )

    assert repo.get_channel_calibration(asset["hardware_id"], "input", 0, "mic_v2pa") is None


@pytest.mark.parametrize(
    "coefficients",
    [
        [math.inf, 0.0],
        [math.nan, 0.0],
        ["bad", 0.0],
        [0.5],
        [0.5, 0.1, 0.0],
    ],
)
def test_output_calibration_rejects_invalid_coefficients_before_writing(tmp_path, coefficients):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=0, outputs=1), "Windows WASAPI", "Speaker", 48000)

    with pytest.raises(HardwareValidationError):
        repo.update_output_amplitude_calibration(asset["hardware_id"], coefficients, max_voltage=1.0)

    assert repo.get_output_amplitude_calibration(asset["hardware_id"]) is None


def test_output_calibration_rejects_invalid_max_voltage(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=0, outputs=1), "Windows WASAPI", "Speaker", 48000)

    for max_voltage in (0, -1, math.inf, math.nan, "bad"):
        with pytest.raises(HardwareValidationError):
            repo.update_output_amplitude_calibration(asset["hardware_id"], [0.5, 0.0], max_voltage=max_voltage)

    assert repo.get_output_amplitude_calibration(asset["hardware_id"]) is None


def test_clear_calibration_methods_reset_columns(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    mic = repo.register_asset(runtime_device(inputs=2, outputs=0), "Windows WASAPI", "Mic", 48000)
    speaker = repo.register_asset(runtime_device(inputs=0, outputs=1), "Windows WASAPI", "Speaker", 48000)

    repo.update_mic_channel_calibrations(mic["hardware_id"], {0: 1.25, 1: 2.5}, channel_standard_spl={0: 94, 1: 114})
    repo.update_output_amplitude_calibration(speaker["hardware_id"], [0.5, 0.1], max_voltage=1.25)

    repo.clear_mic_channel_calibrations(mic["hardware_id"], channel_indices=[1])
    assert repo.get_channel_calibration(mic["hardware_id"], "input", 0, "mic_v2pa")
    assert repo.get_channel_calibration(mic["hardware_id"], "input", 1, "mic_v2pa") is None

    repo.clear_output_amplitude_calibration(speaker["hardware_id"])
    assert repo.get_output_amplitude_calibration(speaker["hardware_id"]) is None


def test_register_asset_returns_inserted_asset_when_post_commit_readback_fails(tmp_path, monkeypatch):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))

    def fail_readback(_hardware_id):
        raise sqlite3.Error("post-commit read failed")

    monkeypatch.setattr(repo, "get_asset", fail_readback)

    asset = repo.register_asset(
        runtime_device(inputs=1, outputs=0),
        hostapi_name="Windows WASAPI",
        display_name="Readback Resistant Mic",
        samplerate=48000,
    )

    fresh_repo = HardwareManagementRepository(str(db_path))
    stored_asset = fresh_repo.get_asset(asset["hardware_id"])
    assert asset["display_name"] == "Readback Resistant Mic"
    assert asset["hardware_type"] == "microphone"
    assert asset["updated_at"] is None
    assert stored_asset["hardware_id"] == asset["hardware_id"]
    assert stored_asset["display_name"] == "Readback Resistant Mic"
    assert [row["channel_index"] for row in fresh_repo.list_channels(asset["hardware_id"], "input")] == [0]
    assert len(fresh_repo.list_assets()) == 1


def test_registration_rolls_back_asset_when_channel_insert_fails(tmp_path, monkeypatch):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))

    def duplicate_channel_ids(hardware_id, max_input_channels, max_output_channels):
        return [
            {
                "channel_id": "duplicate-channel",
                "hardware_id": hardware_id,
                "direction": "input",
                "channel_index": 0,
                "channel_label": "In1",
            },
            {
                "channel_id": "duplicate-channel",
                "hardware_id": hardware_id,
                "direction": "input",
                "channel_index": 1,
                "channel_label": "In2",
            },
        ]

    monkeypatch.setattr("base.hardware_management.build_channel_placeholders", duplicate_channel_ids)

    with pytest.raises(sqlite3.IntegrityError):
        repo.register_asset(
            runtime_device(inputs=2, outputs=0),
            hostapi_name="Windows WASAPI",
            display_name="Failing Mic",
            samplerate=48000,
        )

    assert repo.list_assets() == []


def test_delete_asset_cascades_channel_rows(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(
        runtime_device(inputs=1, outputs=1),
        hostapi_name="Windows WASAPI",
        display_name="Desk Interface",
        samplerate=48000,
    )

    assert len(repo.list_channels(asset["hardware_id"])) == 2
    assert repo.delete_asset(asset["hardware_id"]) is True
    assert repo.get_asset(asset["hardware_id"]) is None
    assert repo.list_channels(asset["hardware_id"]) == []


def test_update_asset_fields_allows_only_mutable_fields(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(
        runtime_device(inputs=1, outputs=0),
        hostapi_name="Windows WASAPI",
        display_name="Original",
        samplerate=44100,
    )

    assert repo.update_asset_fields(
        asset["hardware_id"],
        {"display_name": "Updated", "samplerate": 48000, "bit_depth": 64, "latency_ms": 25},
    )
    updated = repo.get_asset(asset["hardware_id"])
    assert updated["display_name"] == "Updated"
    assert updated["samplerate"] == 48000
    assert updated["bit_depth"] == 64
    assert updated["latency_ms"] == 25

    with pytest.raises(HardwareValidationError):
        repo.update_asset_fields(asset["hardware_id"], {"device_name": "Other"})


@pytest.mark.parametrize(
    "field,value",
    [
        ("display_name", ""),
        ("samplerate", 96000),
        ("bit_depth", 20),
        ("latency_ms", -1),
        ("latency_ms", 1001),
        ("latency_ms", 10.5),
    ],
)
def test_invalid_registration_values_are_rejected_before_sql(tmp_path, field, value):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    kwargs = {
        "hostapi_name": "Windows WASAPI",
        "display_name": "Valid Name",
        "samplerate": 48000,
        "bit_depth": 32,
        "latency_ms": 100,
    }
    kwargs[field] = value

    with pytest.raises(HardwareValidationError):
        repo.register_asset(runtime_device(), **kwargs)

    assert repo.list_assets() == []


def test_missing_hostapi_or_device_are_rejected(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))

    with pytest.raises(HardwareValidationError):
        repo.register_asset(runtime_device(), hostapi_name="", display_name="Name", samplerate=48000)

    with pytest.raises(HardwareValidationError):
        repo.register_asset(
            runtime_device(name=""),
            hostapi_name="Windows WASAPI",
            display_name="Name",
            samplerate=48000,
        )


def test_duplicate_hostapi_and_device_registration_is_allowed(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))

    first = repo.register_asset(runtime_device(), "Windows WASAPI", "First", 48000)
    second = repo.register_asset(runtime_device(), "Windows WASAPI", "Second", 48000)

    assert first["hardware_id"] != second["hardware_id"]
    assets = repo.list_assets()
    assert [asset["display_name"] for asset in assets] == ["First", "Second"]


def test_list_assets_for_selection_groups_microphones_and_speakers(tmp_path):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    repo = HardwareManagementRepository(str(db_path))
    repo.register_asset(runtime_device(name="Mic", inputs=2, outputs=0), "API A", "Mic", 48000)
    repo.register_asset(runtime_device(name="Speaker", inputs=0, outputs=2), "API A", "Speaker", 48000)
    repo.register_asset(runtime_device(name="Interface", inputs=2, outputs=2), "API B", "Interface", 48000)

    grouped = repo.list_assets_for_selection()

    assert [asset["display_name"] for asset in grouped["API A"]["input"]] == ["Mic"]
    assert [asset["display_name"] for asset in grouped["API A"]["output"]] == ["Speaker"]
    assert [asset["display_name"] for asset in grouped["API B"]["input"]] == ["Interface"]
    assert [asset["display_name"] for asset in grouped["API B"]["output"]] == ["Interface"]


def test_build_channel_placeholders_uses_zero_based_indexes_and_labels():
    channels = build_channel_placeholders("hardware-1", 2, 1)

    assert [(row["direction"], row["channel_index"], row["channel_label"]) for row in channels] == [
        ("input", 0, "In1"),
        ("input", 1, "In2"),
        ("output", 0, "Out1"),
    ]
    assert all(row["hardware_id"] == "hardware-1" for row in channels)


def test_match_runtime_device_success_no_match_and_ambiguous():
    asset = registered_asset()
    devices = [
        runtime_device(name="Other", hostapi=0),
        runtime_device(name="USB Mic", hostapi=1),
    ]
    hostapis = {0: "MME", 1: "Windows WASAPI"}

    matched = match_runtime_device(asset, devices, lambda device: hostapis[device["hostapi"]])
    assert matched["name"] == "USB Mic"

    with pytest.raises(HardwareRuntimeMatchError, match="not currently available"):
        match_runtime_device(asset, devices[:1], lambda device: hostapis[device["hostapi"]])

    with pytest.raises(HardwareRuntimeMatchError, match="matches multiple"):
        match_runtime_device(asset, devices + [runtime_device(name="USB Mic", hostapi=1)], lambda device: hostapis[device["hostapi"]])


def test_augment_runtime_device_preserves_runtime_keys_and_adds_registered_metadata():
    runtime = runtime_device(name="Runtime Name", inputs=8, outputs=8)
    asset = registered_asset(device_name="Registered Device", max_input_channels=2, max_output_channels=1)

    payload = augment_runtime_device(runtime, asset)

    assert payload["name"] == "Runtime Name"
    assert payload["max_input_channels"] == 8
    assert payload["max_output_channels"] == 8
    assert payload["extra_runtime_key"] == "preserved"
    assert payload["hardware_id"] == "hardware-1"
    assert payload["display_name"] == "Desk Interface"
    assert payload["device_name"] == "Registered Device"
    assert payload["hostapi_name"] == "Windows WASAPI"
    assert payload["samplerate"] == 48000
    assert payload["bit_depth"] == 32
    assert payload["latency_ms"] == 100


def test_build_selected_device_payload_uses_version_three_schema_without_runtime_indexes():
    mic = augment_runtime_device(runtime_device(index=3), registered_asset(hardware_id="mic-1"))
    speaker = augment_runtime_device(
        runtime_device(name="Speaker", outputs=2, inputs=0, hostapi=1),
        registered_asset(hardware_id="speaker-1", device_name="Speaker"),
    )

    payload = build_selected_device_payload(mic, speaker, [0, "1"])

    assert payload["version"] == 3
    assert payload["mic_channels"] == [0, 1]
    assert "index" not in payload["mic"]
    assert "index" not in payload["speaker"]
    assert payload["mic"]["name"] == payload["mic"]["device_name"]
    assert payload["speaker"]["name"] == payload["speaker"]["device_name"]
    assert payload["mic"]["hardware_id"] == "mic-1"
    assert payload["speaker"]["hardware_id"] == "speaker-1"


def test_build_selected_device_payload_version_three_allows_missing_speaker():
    mic = augment_runtime_device(
        runtime_device(index=3, inputs=2),
        registered_asset(hardware_id="mic-1"),
    )

    payload = build_selected_device_payload(mic, None, [0, "1"])

    assert payload["version"] == 3
    assert payload["mic"]["hardware_id"] == "mic-1"
    assert payload["speaker"] is None
    assert payload["mic_channels"] == [0, 1]

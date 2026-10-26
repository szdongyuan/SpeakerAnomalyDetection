"""Independent VK calibration text settings; no real config or device I/O."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from base.ve3668n_calibration_settings import (
    VECalibrationSettingsError,
    build_ve_calibration_device,
    load_ve_calibration_settings,
)
from consts import model_consts
from consts.ve3668n_consts import (
    VE_CALIBRATION_DEFAULT_RANGE_VOLTS,
    VE_CALIBRATION_DEFAULT_SAMPLE_RATE,
    VE_CALIBRATION_SETTINGS_RELATIVE_PATH,
    VE_DEFAULT_SAMPLE_RATE,
)
from unit_test.base.ve3668n_fakes import device_info, input_config


def test_missing_file_uses_independent_defaults_without_creating_file(tmp_path):
    target = tmp_path / "missing.txt"
    settings = load_ve_calibration_settings(target)
    assert (settings.sample_rate, settings.range_volts) == (48000, 2.5)
    assert settings.sample_rate == VE_CALIBRATION_DEFAULT_SAMPLE_RATE
    assert settings.range_volts == VE_CALIBRATION_DEFAULT_RANGE_VOLTS
    assert VE_DEFAULT_SAMPLE_RATE == 51200
    assert not target.exists()
    source = device_info(input_config=input_config(96000))
    before = deepcopy(source)
    result = build_ve_calibration_device(source, settings)
    assert result["input_config"] == input_config(
        48000, range_min=-2.5, range_max=2.5)
    assert source == before
    assert result is not source
    assert result["input_config"] is not source["input_config"]
    for key in source.keys() - {"input_config", "physical_channels"}:
        assert result[key] == source[key]
    assert result["physical_channels"] == tuple(source["physical_channels"])
    result["input_config"]["range_max"] = 5
    assert source == before


@pytest.mark.parametrize("rate", [4000, 44100, 48000, 51200, 96000, 102400])
def test_supported_sample_rates(tmp_path, rate):
    target = tmp_path / "settings.txt"
    target.write_text(f"sample_rate={rate}\nrange_volts=2.5\n", encoding="utf-8")
    assert load_ve_calibration_settings(target).sample_rate == rate


@pytest.mark.parametrize("limit", ["10", "5", "2.5", "1", "0.5", "0.1", "0.02", "2.50"])
def test_supported_voltage_ranges(tmp_path, limit):
    target = tmp_path / "settings.txt"
    target.write_text(f"sample_rate=48000\nrange_volts={limit}\n", encoding="utf-8")
    settings = load_ve_calibration_settings(target)
    assert settings.range_volts == float(limit)
    result = build_ve_calibration_device(device_info(), settings)
    assert result["input_config"]["range_min"] == -float(limit)
    assert result["input_config"]["range_max"] == float(limit)


def test_whitespace_bom_blank_lines_and_field_order(tmp_path):
    target = tmp_path / "settings.txt"
    target.write_text("\ufeff\n  range_volts \t= 2.50 \n\n sample_rate = 48000 \n", encoding="utf-8")
    settings = load_ve_calibration_settings(str(target))
    assert (settings.sample_rate, settings.range_volts) == (48000, 2.5)


def test_reads_both_fields_again_after_file_edit(tmp_path):
    target = tmp_path / "settings.txt"
    target.write_text("sample_rate=48000\nrange_volts=2.5\n", encoding="utf-8")
    first = load_ve_calibration_settings(target)
    target.write_text("sample_rate=51200\nrange_volts=0.1\n", encoding="utf-8")
    second = load_ve_calibration_settings(target)
    assert (first.sample_rate, first.range_volts) == (48000, 2.5)
    assert (second.sample_rate, second.range_volts) == (51200, 0.1)
    assert target.read_text(encoding="utf-8") == "sample_rate=51200\nrange_volts=0.1\n"
    with pytest.raises(FrozenInstanceError):
        first.sample_rate = 51200


@pytest.mark.parametrize("text,reason", [
    ("", "required"),
    (" \n\t\n", "required"),
    ("2.5", "line 1"),
    ("sample_rate=48000", "required"),
    ("range_volts=2.5", "required"),
    ("sample_rate=48000\nrange_volts=2.5\nsample_rate=51200", "sample_rate"),
    ("sample_rate=48000\nrange_volts=2.5\nrange_volts=5", "range_volts"),
    ("sample_rate=48000\nrange_volts=2.5\nunknown=1", "unknown"),
    ("sample_rate 48000\nrange_volts=2.5", "line 1"),
    ("sample_rate==48000\nrange_volts=2.5", "line 1"),
    ("sample_rate=48000\nrange_volts=2.5\n# comment", "line 3"),
    ("[calibration]\nsample_rate=48000\nrange_volts=2.5", "line 1"),
    ("sample_rate=48000 Hz\nrange_volts=2.5", "sample_rate"),
    ("sample_rate=48000\nrange_volts=2.5 V", "range_volts"),
])
def test_invalid_file_structure_is_not_defaulted(tmp_path, text, reason):
    target = tmp_path / "settings.txt"
    target.write_text(text, encoding="utf-8")
    with pytest.raises(VECalibrationSettingsError, match=reason):
        load_ve_calibration_settings(target)
    assert target.read_text(encoding="utf-8") == text


@pytest.mark.parametrize("rate", [
    "48000.0", "4.8e4", "True", "3999", "102401", "0", "-1", "", "+48000", "４８０００",
])
def test_invalid_sample_rate_is_not_coerced_or_defaulted(tmp_path, rate):
    target = tmp_path / "settings.txt"
    target.write_text(f"sample_rate={rate}\nrange_volts=2.5\n", encoding="utf-8")
    with pytest.raises(VECalibrationSettingsError, match="sample_rate"):
        load_ve_calibration_settings(target)


@pytest.mark.parametrize("limit", [
    "nan", "inf", "-inf", "0", "-2.5", "3", "2.5001", "2.5,5", "2.5 5", "True", "", "2.5 # comment",
])
def test_invalid_voltage_range_is_not_coerced_or_defaulted(tmp_path, limit):
    target = tmp_path / "settings.txt"
    target.write_text(f"sample_rate=48000\nrange_volts={limit}\n", encoding="utf-8")
    with pytest.raises(VECalibrationSettingsError, match="range_volts"):
        load_ve_calibration_settings(target)


def test_invalid_utf8_is_a_diagnostic_read_error(tmp_path):
    target = tmp_path / "invalid.txt"
    target.write_bytes(b"sample_rate=48000\nrange_volts=\xff\n")
    with pytest.raises(VECalibrationSettingsError, match="invalid.txt") as caught:
        load_ve_calibration_settings(target)
    assert isinstance(caught.value.__cause__, UnicodeError)


@pytest.mark.parametrize("error_type", [PermissionError, IsADirectoryError, OSError])
def test_nonmissing_read_errors_do_not_use_defaults(tmp_path, monkeypatch, error_type):
    target = tmp_path / "unreadable.txt"
    failure = error_type("test read failure")

    def fail_read(path, *, encoding):
        assert path == target
        raise failure

    monkeypatch.setattr(Path, "read_text", fail_read)
    with pytest.raises(VECalibrationSettingsError, match="unreadable.txt") as caught:
        load_ve_calibration_settings(target)
    assert caught.value.__cause__ is failure


def test_default_path_is_resolved_at_call_time_independently_of_cwd(tmp_path, monkeypatch):
    app_root = tmp_path / "application"
    target = app_root / VE_CALIBRATION_SETTINGS_RELATIVE_PATH
    target.parent.mkdir(parents=True)
    target.write_text("sample_rate=96000\nrange_volts=0.02\n", encoding="utf-8")
    other = tmp_path / "other-cwd"
    other.mkdir()
    monkeypatch.setattr(model_consts, "DEFAULT_DIR", str(app_root))
    monkeypatch.chdir(other)
    settings = load_ve_calibration_settings()
    assert (settings.sample_rate, settings.range_volts) == (96000, 0.02)


@pytest.mark.parametrize("overrides,reason", [
    ({"input_mode": "invalid"}, "input_mode"),
    ({"unit": "Pa"}, "unit"),
])
def test_builder_preserves_hardware_mode_and_unit_validation(tmp_path, overrides, reason):
    settings = load_ve_calibration_settings(tmp_path / "missing.txt")
    config = input_config(96000)
    config.update(overrides)
    source = device_info(input_config=config)
    before = deepcopy(source)
    with pytest.raises(ValueError, match=reason):
        build_ve_calibration_device(source, settings)
    assert source == before


@pytest.mark.parametrize("obsolete", [
    {"sample_rate": "broken", "range_min": None, "range_max": object()},
    {"sample_rate": 0, "range_min": -3, "range_max": 3},
    {},
])
def test_builder_replaces_obsolete_source_acquisition_fields(tmp_path, obsolete):
    settings = load_ve_calibration_settings(tmp_path / "missing.txt")
    source = device_info(input_config={"input_mode": "IEPE", "unit": "V", **obsolete})
    result = build_ve_calibration_device(source, settings)
    assert result["input_config"] == input_config(48000, range_min=-2.5, range_max=2.5)

"""Independent VK calibration settings; no Qt, logging, cache, or writes."""
from dataclasses import dataclass
from pathlib import Path

from base.ve3668n_input import (
    validate_device_snapshot,
    validate_sample_rate,
    validate_voltage_range,
)
from consts import model_consts
from consts.ve3668n_consts import (
    VE_CALIBRATION_DEFAULT_RANGE_VOLTS,
    VE_CALIBRATION_DEFAULT_SAMPLE_RATE,
    VE_CALIBRATION_SETTINGS_RELATIVE_PATH,
)


class VECalibrationSettingsError(ValueError):
    """An explicit calibration settings file cannot authorize acquisition."""


@dataclass(frozen=True)
class VECalibrationSettings:
    sample_rate: int
    range_volts: float


def default_path():
    return Path(model_consts.DEFAULT_DIR) / VE_CALIBRATION_SETTINGS_RELATIVE_PATH


def _parse_settings(text):
    fields = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if line.count("=") != 1:
            raise VECalibrationSettingsError(f"line {line_number}: expected key=value")
        key, value = (part.strip() for part in line.split("=", 1))
        if key not in ("sample_rate", "range_volts") or key in fields:
            raise VECalibrationSettingsError(
                f"line {line_number}: unknown or duplicate key {key!r}")
        fields[key] = value
    if set(fields) != {"sample_rate", "range_volts"}:
        raise VECalibrationSettingsError("sample_rate and range_volts are both required")
    raw_rate = fields["sample_rate"]
    if not raw_rate.isascii() or not raw_rate.isdecimal():
        raise VECalibrationSettingsError("sample_rate must be a decimal integer in Hz")
    try:
        rate = validate_sample_rate(int(raw_rate))
    except ValueError as exc:
        raise VECalibrationSettingsError(f"sample_rate: {exc}") from exc
    try:
        limit = float(fields["range_volts"])
        _, limit = validate_voltage_range(-limit, limit)
    except ValueError as exc:
        raise VECalibrationSettingsError(f"range_volts: {exc}") from exc
    return VECalibrationSettings(rate, limit)


def load_ve_calibration_settings(path=None):
    target = default_path() if path is None else Path(path)
    try:
        text = target.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return VECalibrationSettings(
            VE_CALIBRATION_DEFAULT_SAMPLE_RATE, VE_CALIBRATION_DEFAULT_RANGE_VOLTS)
    except (OSError, UnicodeError) as exc:
        raise VECalibrationSettingsError(f"cannot read {target}: {exc}") from exc
    return _parse_settings(text)


def build_ve_calibration_device(source_device, settings):
    config = dict(source_device["input_config"])
    config.update(sample_rate=settings.sample_rate,
                  range_min=-settings.range_volts, range_max=settings.range_volts)
    return validate_device_snapshot({**source_device, "input_config": config})

"""Prepare calibrated output and retain one successful output context per owner."""

import math
from dataclasses import dataclass

from base.soundcard_calibration_manager import SoundcardCalibrationManager
from consts import error_code


@dataclass(frozen=True)
class OutputCalibrationPreparation:
    amplitude: float | None
    max_voltage: float | None
    error: str | None = None
    cacheable: bool = False


def resolve_output_amplitude(speaker, voltage, voltage_type):
    """Read one hardware-specific snapshot before checking and predicting output.

    Voltage types retain the existing convention: both RMS and Peak targets are
    passed directly to the calibration formula in the configured voltage units.
    """
    hardware_id = speaker.get("hardware_id") if speaker else None
    if not hardware_id:
        return OutputCalibrationPreparation(amplitude=0.0, max_voltage=None)

    manager = SoundcardCalibrationManager(speaker_hardware_id=hardware_id)
    code, calibration = manager.load_output_amplitude_calibration()
    if code != error_code.OK:
        return OutputCalibrationPreparation(amplitude=0.0, max_voltage=None)

    max_voltage = calibration["max_voltage"]
    try:
        voltage = float(voltage)
    except (TypeError, ValueError, OverflowError):
        return OutputCalibrationPreparation(
            amplitude=None, max_voltage=max_voltage,
            error="激励电压必须为有限数值，请调整激励电压。",
        )
    if not math.isfinite(voltage):
        return OutputCalibrationPreparation(
            amplitude=None, max_voltage=max_voltage,
            error="激励电压必须为有限数值，请调整激励电压。",
        )
    if voltage > max_voltage:
        return OutputCalibrationPreparation(
            amplitude=None,
            max_voltage=max_voltage,
            error=(f"配置电压为 {voltage:g} V，当前扬声器校准上限为 {max_voltage:g} V，"
                   "请调整激励电压或重新校准。"),
        )
    amplitude = manager.predict_amplitude(calibration["calibration_coefficients"], voltage)
    if not math.isfinite(amplitude):
        return OutputCalibrationPreparation(
            amplitude=None, max_voltage=max_voltage,
            error="输出校准无法计算有效幅值，请重新校准。",
        )
    return OutputCalibrationPreparation(
        amplitude=float(amplitude), max_voltage=max_voltage, cacheable=True,
    )


class OutputCalibrationCache:
    """A single successful preparation, owned by the calling window instance."""

    def __init__(self):
        self.invalidate()

    def get_or_resolve(self, key, resolve):
        if self._result is not None and self._key == key:
            return self._result
        self.invalidate()
        result = resolve()
        if result.cacheable and result.error is None:
            self._key, self._result = key, result
        return result

    def invalidate(self):
        self._key = None
        self._result = None

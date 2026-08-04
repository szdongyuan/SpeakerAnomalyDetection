import json
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from base.hardware_management import HardwareManagementRepository
from base.log_manager import LogManager
from base.sound_device_manager import SoundDeviceManager
from consts import error_code, model_consts


@dataclass(frozen=True)
class MicChannelCalibrationResult:
    factor: Optional[float]
    requested_channel: int
    source_channel: Optional[int]
    used_fallback: bool
    has_any_calibration: bool


@dataclass(frozen=True)
class AnalysisV2paPreparation:
    factor: Optional[float]
    error: Optional[str] = None


def _normalize_channel_index(channel_index):
    try:
        normalized = int(channel_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("channel_index must be a non-negative integer.") from exc
    if normalized < 0:
        raise ValueError("channel_index must be a non-negative integer.")
    return normalized


def _normalize_positive_factor(v2pa_factor):
    try:
        normalized = float(v2pa_factor)
    except (TypeError, ValueError) as exc:
        raise ValueError("v2pa_factor must be a finite positive number.") from exc
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError("v2pa_factor must be a finite positive number.")
    return normalized


def _normalize_standard_spl(standard_spl):
    if standard_spl is None:
        return None
    try:
        numeric = float(standard_spl)
    except (TypeError, ValueError) as exc:
        raise ValueError("standard_spl must be a finite number.") from exc
    if not math.isfinite(numeric):
        raise ValueError("standard_spl must be a finite number.")
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _selected_hardware_id(device_key):
    saved_devices = SoundDeviceManager.load_selected_devices() or {}
    device_payload = saved_devices.get(device_key)
    if isinstance(device_payload, dict):
        hardware_id = device_payload.get("hardware_id")
        if hardware_id:
            return hardware_id
    return None


def _resolve_mic_hardware_id(hardware_id=None):
    resolved = hardware_id or _selected_hardware_id("mic")
    if not resolved:
        raise ValueError("registered microphone hardware_id is required for calibration")
    return resolved


def _resolve_speaker_hardware_id(hardware_id=None):
    resolved = hardware_id or _selected_hardware_id("speaker")
    if not resolved:
        raise ValueError("registered speaker hardware_id is required for calibration")
    return resolved


def _repository(db_path=None):
    return HardwareManagementRepository(db_path)


def _load_mic_channel_calibration_rows(hardware_id=None, db_path=None):
    resolved_hardware_id = _resolve_mic_hardware_id(hardware_id)
    return _repository(db_path).list_channel_calibrations(
        resolved_hardware_id,
        "input",
        "mic_v2pa",
    )


def load_mic_channel_v2pa_factors(hardware_id=None, db_path=None):
    factors = {}
    for row in _load_mic_channel_calibration_rows(hardware_id=hardware_id, db_path=db_path):
        try:
            factors[_normalize_channel_index(row["channel_index"])] = _normalize_positive_factor(row["factor_value"])
        except ValueError as exc:
            channel_label = format_input_channel_label(row.get("channel_index", 0))
            raise ValueError(f"Invalid microphone calibration payload for {channel_label}: {exc}") from exc
    return factors


def save_mic_channel_v2pa_factor(channel_index, v2pa_factor, standard_spl=None, hardware_id=None, db_path=None):
    normalized_channel = _normalize_channel_index(channel_index)
    resolved_hardware_id = _resolve_mic_hardware_id(hardware_id)
    _repository(db_path).update_mic_channel_calibrations(
        resolved_hardware_id,
        {normalized_channel: _normalize_positive_factor(v2pa_factor)},
        channel_standard_spl={normalized_channel: _normalize_standard_spl(standard_spl)},
    )


def clear_mic_channel_v2pa_factors(hardware_id=None, channel_indices=None, db_path=None):
    _repository(db_path).clear_mic_channel_calibrations(
        _resolve_mic_hardware_id(hardware_id),
        channel_indices=channel_indices,
    )


def replace_mic_channel_v2pa_factors(channel_factors, channel_standard_spl=None, hardware_id=None, db_path=None):
    resolved_hardware_id = _resolve_mic_hardware_id(hardware_id)
    channel_standard_spl = channel_standard_spl or {}
    normalized_factors = {}
    for channel_index, v2pa_factor in channel_factors.items():
        normalized_factors[_normalize_channel_index(channel_index)] = _normalize_positive_factor(v2pa_factor)

    normalized_standard_spl_by_channel = {}
    for channel_index, standard_spl in channel_standard_spl.items():
        normalized_standard_spl_by_channel[_normalize_channel_index(channel_index)] = _normalize_standard_spl(standard_spl)

    existing_standard_spl_by_channel = {}
    for row in _load_mic_channel_calibration_rows(hardware_id=resolved_hardware_id, db_path=db_path):
        if row.get("standard_spl") is not None:
            existing_standard_spl_by_channel[_normalize_channel_index(row["channel_index"])] = row["standard_spl"]

    complete_standard_spl = {}
    for channel_index in normalized_factors:
        if channel_index in normalized_standard_spl_by_channel:
            complete_standard_spl[channel_index] = normalized_standard_spl_by_channel[channel_index]
        elif channel_index in existing_standard_spl_by_channel:
            complete_standard_spl[channel_index] = existing_standard_spl_by_channel[channel_index]
        else:
            raise ValueError("standard_spl is required for mic_v2pa calibration")

    _repository(db_path).update_mic_channel_calibrations(
        resolved_hardware_id,
        normalized_factors,
        channel_standard_spl=complete_standard_spl,
    )


def resolve_mic_channel_v2pa_factor(channel_index, hardware_id=None, db_path=None):
    requested_channel = _normalize_channel_index(channel_index)
    factors = load_mic_channel_v2pa_factors(hardware_id=hardware_id, db_path=db_path)
    if requested_channel in factors:
        return MicChannelCalibrationResult(
            factor=factors[requested_channel],
            requested_channel=requested_channel,
            source_channel=requested_channel,
            used_fallback=False,
            has_any_calibration=True,
        )
    return MicChannelCalibrationResult(
        factor=None,
        requested_channel=requested_channel,
        source_channel=None,
        used_fallback=False,
        has_any_calibration=bool(factors),
    )


def format_input_channel_label(channel_index):
    return f"In{int(channel_index) + 1}"


def resolve_analysis_v2pa_factor_for_channel(raw_channel, warn_callback=None, hardware_id=None, db_path=None):
    channel = max(0, int(raw_channel or 0))
    result = resolve_mic_channel_v2pa_factor(channel, hardware_id=hardware_id, db_path=db_path)
    if result.factor is None:
        if warn_callback:
            warn_callback("麦克风未进行校准，结果仅供参考。")
        return 1.0
    return float(result.factor)


class AnalysisV2paBatch:
    def __init__(self, resolver=None):
        self._resolver = resolver or resolve_analysis_v2pa_factor_for_channel
        self._preparations = {}
        self._messages = []
        self._message_set = set()

    @staticmethod
    def _normalize_channel(raw_channel):
        try:
            return max(0, int(raw_channel or 0))
        except (TypeError, ValueError, OverflowError):
            return 0

    def _capture_message(self, message):
        if message is None:
            return
        text = str(message)
        if text and text not in self._message_set:
            self._message_set.add(text)
            self._messages.append(text)

    def resolve(self, raw_channel) -> AnalysisV2paPreparation:
        channel = self._normalize_channel(raw_channel)
        if channel in self._preparations:
            return self._preparations[channel]

        try:
            factor = self._resolver(channel, warn_callback=self._capture_message)
            preparation = AnalysisV2paPreparation(factor=factor)
        except ValueError as exc:
            error = str(exc)
            self._capture_message(error)
            preparation = AnalysisV2paPreparation(factor=None, error=error)

        self._preparations[channel] = preparation
        return preparation

    @property
    def messages(self):
        return tuple(self._messages)

    def warning_text(self):
        if len(self._messages) == 1:
            return self._messages[0]
        return "\n".join(f"• {message}" for message in self._messages)




class SoundcardCalibrationManager(object):

    def __init__(self, db_path=None, speaker_hardware_id=None):
        self.amplitudes = []
        self.voltages = []
        self.db_path = db_path
        self.speaker_hardware_id = speaker_hardware_id
        self.logger = LogManager.set_log_handler("soundcard_core")

    def _resolved_speaker_hardware_id(self):
        return _resolve_speaker_hardware_id(self.speaker_hardware_id)

    def add_data(self, amplitude, voltage, validation=True):
        """
        Add amplitude and voltage data.
        Args:
            validation: bool
            amplitude: int or float
                The input amplitude value.
            voltage: int or float
                The input voltage value.
        Returns:
             A tuple containing the status code and message.
        """
        if validation:
            if not amplitude or not voltage:
                return error_code.INVALID_DATA_LOADING, "Input data cannot be None."
            if not isinstance(amplitude, (int, float)) or not isinstance(voltage, (int, float)):
                return error_code.INVALID_TYPE_DATA, "Input data must be numeric."
        self.amplitudes.append(amplitude)
        self.voltages.append(voltage)
        return error_code.OK, "Successfully add data."

    def fit(self, threshold=0.001):
        """
        Fit amplitude and voltage data to obtain a linear relationship.
        Returns:
             A tuple containing the status code and the fitting function.
        """
        if not self.amplitudes or not self.voltages:
            self.logger.error("Amplitudes and voltages must not be empty.")
            return error_code.INVALID_DATA_LOADING, "Amplitudes and voltages must not be empty."
        if len(self.amplitudes) != len(self.voltages):
            self.logger.error("Amplitudes and voltages must have the same length.")
            return error_code.INVALID_DATA_LOADING, "Amplitudes and voltages must have the same length."
        coefficients, residuals, *_ = np.polyfit(self.voltages, self.amplitudes, 1, full=True)
        if len(self.voltages) > 2:
            if len(residuals) == 0:
                self.logger.error("Residuals is empty.")
                return error_code.INVALID_CALIBRATION, "Residuals is empty, please readjust."
            mse = np.nanmean(residuals**2)
            if mse > threshold or mse < 0 or not np.isfinite(mse):
                self.logger.error("Calibration is not accurate, please readjust.")
                return error_code.INVALID_CALIBRATION, "Calibration is not accurate, please readjust."
        save_code, msg = self.save_output_amplitude_calibration(coefficients, max(self.voltages))
        if save_code == error_code.OK:
            return error_code.OK, coefficients
        return save_code, msg

    @staticmethod
    def predict_amplitude(coefficients, target_voltage):
        """
        Predict the corresponding amplitude based on the fitting function and target voltage.
        Args:
            coefficients: list
                fit coefficients.
            target_voltage: int or float
                The target voltage value.
        Returns:
            predict_amplitude: float
                The predicted amplitude(four decimal places).
        """
        fit_function = np.poly1d(coefficients)
        predict_amplitude = fit_function(target_voltage)
        return np.round(predict_amplitude, 4)

    def calibrate_amplitude(self, target_voltage):
        """
        Args:
            target_voltage: int or float or list
        Returns:
            predict_amplitude: int or float or list
                The amplitude corresponding to the target voltage.
        """
        load_code, load_data = self.load_output_amplitude_calibration()
        if load_code == error_code.OK:
            coefficients_data = load_data.get("calibration_coefficients")
            max_voltage = load_data.get("max_voltage")
            predict_amplitude = self.predict_amplitude(coefficients_data, target_voltage)
            return error_code.OK, (predict_amplitude, max_voltage)
        self.logger.error("Failed to load coefficients, please calibrate first.")
        return error_code.INVALID_DATA_LOADING, "Failed to load coefficients, please calibrate first."

    def save_output_amplitude_calibration(self, coefficients, max_voltage):
        if isinstance(coefficients, np.ndarray):
            coefficients = coefficients.tolist()
        try:
            hardware_id = self._resolved_speaker_hardware_id()
            HardwareManagementRepository(self.db_path).update_output_amplitude_calibration(
                hardware_id,
                coefficients,
                max_voltage=max_voltage,
            )
            return error_code.OK, "Successfully saved output calibration to database."
        except Exception as e:
            err_msg = str(e)
            self.logger.error(err_msg)
            return error_code.INVALID_DATA_LOADING, err_msg

    def load_output_amplitude_calibration(self, channel_index=0):
        try:
            hardware_id = self._resolved_speaker_hardware_id()
            row = HardwareManagementRepository(self.db_path).get_output_amplitude_calibration(
                hardware_id,
                channel_index=channel_index,
            )
            if row is None:
                return error_code.INVALID_DATA_LOADING, "Failed to load coefficients, please calibrate first."
            payload = json.loads(row["coefficients_json"])
            coefficients = payload.get("calibration_coefficients") if isinstance(payload, dict) else None
            if set(payload.keys()) != {"calibration_coefficients"}:
                return error_code.INVALID_DATA_LOADING, "Invalid output calibration payload."
            if not isinstance(coefficients, list) or len(coefficients) != 2:
                return error_code.INVALID_DATA_LOADING, "Invalid output calibration coefficients."
            normalized_coefficients = []
            for coefficient in coefficients:
                numeric = float(coefficient)
                if not math.isfinite(numeric):
                    return error_code.INVALID_DATA_LOADING, "Invalid output calibration coefficients."
                normalized_coefficients.append(numeric)
            max_voltage = float(row["max_voltage"])
            if not math.isfinite(max_voltage) or max_voltage <= 0:
                return error_code.INVALID_DATA_LOADING, "Invalid output calibration max voltage."
            return error_code.OK, {
                "calibration_coefficients": normalized_coefficients,
                "max_voltage": max_voltage,
            }
        except Exception as e:
            err_msg = str(e)
            self.logger.error(err_msg)
            return error_code.INVALID_DATA_LOADING, err_msg

    def get_max_output_voltage(self, channel_index=0):
        code, data = self.load_output_amplitude_calibration(channel_index=channel_index)
        if code == error_code.OK:
            return data["max_voltage"]
        return None

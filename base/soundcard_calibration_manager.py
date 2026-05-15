import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from base.log_manager import LogManager
from consts import error_code, model_consts, running_consts


@dataclass(frozen=True)
class MicChannelCalibrationResult:
    factor: Optional[float]
    requested_channel: int
    source_channel: Optional[int]
    used_fallback: bool
    has_any_calibration: bool


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


def _channel_calibration_path(file_path=None):
    MIC_CHANNEL_CALIBRATION_FILE = running_consts.DEFAULT_DIR + "ui/ui_config/mic_channel_calibration.json"
    return os.fspath(file_path or MIC_CHANNEL_CALIBRATION_FILE)


def _load_mic_channel_calibration_payload(file_path=None):
    path = _channel_calibration_path(file_path=file_path)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_mic_channel_v2pa_factors(file_path=None):
    payload = _load_mic_channel_calibration_payload(file_path=file_path)
    channels = payload.get("channels")
    if not isinstance(channels, dict):
        return {}

    factors = {}
    for channel_key, channel_payload in channels.items():
        if not isinstance(channel_payload, dict):
            continue
        try:
            channel_index = _normalize_channel_index(channel_key)
            factors[channel_index] = _normalize_positive_factor(channel_payload.get("v2pa_factor"))
        except ValueError:
            continue
    return factors


def save_mic_channel_v2pa_factor(channel_index, v2pa_factor, standard_spl=None, file_path=None):
    normalized_channel = _normalize_channel_index(channel_index)
    normalized_factor = _normalize_positive_factor(v2pa_factor)
    path = _channel_calibration_path(file_path=file_path)

    channels_payload = {}
    existing_payload = _load_mic_channel_calibration_payload(file_path=path)
    existing_channels = existing_payload.get("channels")
    if isinstance(existing_channels, dict):
        for saved_channel, channel_payload in existing_channels.items():
            if not isinstance(channel_payload, dict):
                continue
            try:
                normalized_saved_channel = _normalize_channel_index(saved_channel)
                preserved_payload = {
                    "v2pa_factor": _normalize_positive_factor(channel_payload.get("v2pa_factor")),
                    "calibrated_at": channel_payload.get("calibrated_at") or datetime.now().strftime("%Y-%m-%d"),
                }
            except (TypeError, ValueError):
                continue
            try:
                normalized_saved_spl = _normalize_standard_spl(channel_payload.get("standard_spl"))
            except (TypeError, ValueError):
                normalized_saved_spl = None
            if normalized_saved_spl is not None:
                preserved_payload["standard_spl"] = normalized_saved_spl
            channels_payload[str(normalized_saved_channel)] = preserved_payload

    channel_payload = {
        "v2pa_factor": normalized_factor,
        "calibrated_at": datetime.now().strftime("%Y-%m-%d"),
    }
    normalized_standard_spl = _normalize_standard_spl(standard_spl)
    if normalized_standard_spl is not None:
        channel_payload["standard_spl"] = normalized_standard_spl
    channels_payload[str(normalized_channel)] = channel_payload

    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "channels": channels_payload}, f, ensure_ascii=False, indent=2)


def resolve_mic_channel_v2pa_factor(channel_index, file_path=None):
    requested_channel = _normalize_channel_index(channel_index)
    factors = load_mic_channel_v2pa_factors(file_path=file_path)
    if requested_channel in factors:
        return MicChannelCalibrationResult(
            factor=factors[requested_channel],
            requested_channel=requested_channel,
            source_channel=requested_channel,
            used_fallback=False,
            has_any_calibration=True,
        )
    if factors:
        fallback_channel = min(factors)
        return MicChannelCalibrationResult(
            factor=factors[fallback_channel],
            requested_channel=requested_channel,
            source_channel=fallback_channel,
            used_fallback=True,
            has_any_calibration=True,
        )
    return MicChannelCalibrationResult(
        factor=None,
        requested_channel=requested_channel,
        source_channel=None,
        used_fallback=False,
        has_any_calibration=False,
    )


def format_input_channel_label(channel_index):
    return f"In{int(channel_index) + 1}"


def resolve_analysis_v2pa_factor_for_channel(raw_channel, warn_callback=None, file_path=None):
    channel = max(0, int(raw_channel or 0))
    result = resolve_mic_channel_v2pa_factor(channel, file_path=file_path)
    if result.factor is None:
        raise ValueError(f"{format_input_channel_label(channel)} 未找到输入通道校准数据，请先完成输入校准。")
    if result.used_fallback and warn_callback:
        warn_callback(
            f"{format_input_channel_label(channel)} 未校准，本次使用 "
            f"{format_input_channel_label(result.source_channel)} 的校准系数。"
        )
    return float(result.factor)


def get_mic_v2pa_factor():
    """
        Reads the microphone calibration v2pa_factor value from a specified file.

        This method is static because it does not depend on the instance state of the class and can operate independently.
        The v2pa_factor value is read from a file as it may vary based on environmental conditions and needs to be
    dynamically adjusted.

        Return:
            The microphone calibration v2pa_factor value. Returns 0.0 if reading the file fails.
    """
    file_path = running_consts.DEFAULT_DIR + "ui/ui_config/mic_calibration.txt"
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            v2pa_factor = lines[1].strip()
            return float(v2pa_factor)
    except Exception as e:
        return 0.0


class SoundcardCalibrationManager(object):

    def __init__(self):
        self.amplitudes = []
        self.voltages = []
        self.logger = LogManager.set_log_handler("soundcard_core")

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

    def fit(self, threshold=0.001, json_file_name="calibration_coefficients.json"):
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
        save_code, msg = self.save_coefficients_to_json(coefficients, max(self.voltages), json_file_name)
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

    def calibrate_amplitude(self, target_voltage, json_file_name="calibration_coefficients.json"):
        """
        Args:
            target_voltage: int or float or list
            json_file_name: str
                The json file name of calibration coefficient.
        Returns:
            predict_amplitude: int or float or list
                The amplitude corresponding to the target voltage.
        """
        load_code, load_data = self.load_data_from_json(json_file_name)
        if load_code == error_code.OK:
            coefficients_data = load_data.get("calibration_coefficients")
            max_voltage = load_data.get("max_voltage")
            predict_amplitude = self.predict_amplitude(coefficients_data, target_voltage)
            return error_code.OK, (predict_amplitude, max_voltage)
        self.logger.error("Failed to load coefficients, please calibrate first.")
        return error_code.INVALID_DATA_LOADING, "Failed to load coefficients, please calibrate first."

    def save_coefficients_to_json(self, coefficients, max_voltages, json_file_name):
        """
        Save calibration coefficients and voltages to a JSON file.
        Args:
            coefficients: list or np.ndarray
                Calibration coefficients to save.
            max_voltages: int or float
                Calibration max voltages to save.
            json_file_name: str
                Name of the JSON file to save the data.
        Returns:
            A tuple containing an error code and a message indicating success or failure.
        """
        if not json_file_name.endswith(".json"):
            json_file_name = os.path.splitext(json_file_name)[0]
            json_file_name += ".json"
        json_file_path = model_consts.JSON_DIR_PATH + "/" + json_file_name
        directory = os.path.dirname(json_file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        if not isinstance(coefficients, (list, np.ndarray)):
            return error_code.INVALID_TYPE_DATA, "Coefficients must be a list or numpy array."
        coefficients = coefficients.tolist() if isinstance(coefficients, np.ndarray) else coefficients
        data = {"calibration_coefficients": coefficients, "max_voltage": max_voltages}
        try:
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file, indent=3)
                self.logger.info(f"Coefficients saved to {json_file_path}.")
                return error_code.OK, f"Successfully save the coefficients to {json_file_path}."
        except Exception as e:
            err_msg = "Failed saving coefficients to json. %s" % (str(e)[:50])
            self.logger.error(err_msg)
            return error_code.INVALID_SAVE, err_msg

    def load_data_from_json(self, json_file_name):
        """
        Load calibration coefficients and voltages from a JSON file.
        Args:
            json_file_name: str
                The name of the JSON file to load data from.
        Returns:
                A tuple containing an error code and the loaded data or an error message.
        """
        json_file_path = model_consts.JSON_DIR_PATH + "/" + json_file_name
        if not os.path.exists(json_file_path):
            return error_code.INVALID_DATA_LOADING, "This json file does not exist."
        try:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)
                return error_code.OK, data
        except Exception as e:
            err_msg = "Failed to load coefficients data from json.%s" % (str(e)[:50])
            self.logger.error(err_msg)
            return error_code.INVALID_DATA_LOADING, err_msg

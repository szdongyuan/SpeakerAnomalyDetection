import json
import math
import os
import tempfile
import threading
from datetime import datetime

import numpy as np

from base.log_manager import LogManager
from base.sound_device_manager import SoundDeviceManager
from consts import error_code, model_consts


MIC_INPUT_CALIBRATION_PATH = os.path.join(
    model_consts.JSON_DIR_PATH,
    "mic_input_calibration.json",
)
MIC_INPUT_CALIBRATION_VERSION = 1
_mic_input_calibration_io_lock = threading.Lock()


def _device_value(device, key):
    getter = getattr(device, "get", None)
    return getter(key) if callable(getter) else None


def build_mic_input_identity(input_device, input_channel):
    """Return the stable identity used to bind a single-channel calibration."""
    if (
        input_device is None
        or isinstance(input_channel, bool)
        or not isinstance(input_channel, (int, np.integer))
    ):
        return None

    try:
        channel_index = int(input_channel)
        hostapi_index = int(_device_value(input_device, "hostapi"))
    except (TypeError, ValueError, OverflowError):
        return None

    device_name = str(_device_value(input_device, "name") or "").strip()
    if not device_name or channel_index < 0:
        return None

    try:
        api_info = SoundDeviceManager.get_api_info(hostapi_index)
        api_name = str(_device_value(api_info, "name") or "").strip()
    except Exception:
        return None

    if not api_name:
        return None

    return {
        "api_name": api_name,
        "device_name": device_name,
        "channel_index": channel_index,
    }


def _validated_mic_input_calibration(payload):
    if not isinstance(payload, dict):
        return None
    version = payload.get("version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != MIC_INPUT_CALIBRATION_VERSION
    ):
        return None

    input_config = payload.get("input")
    calibration = payload.get("calibration")
    if not isinstance(input_config, dict) or not isinstance(calibration, dict):
        return None

    api_name = input_config.get("api_name")
    device_name = input_config.get("device_name")
    channel_index = input_config.get("channel_index")
    if not isinstance(api_name, str) or not api_name.strip():
        return None
    if not isinstance(device_name, str) or not device_name.strip():
        return None
    if isinstance(channel_index, bool) or not isinstance(channel_index, int) or channel_index < 0:
        return None

    factor_value = calibration.get("v2pa_factor")
    standard_spl_value = calibration.get("standard_spl_db")
    sample_rate_value = calibration.get("sample_rate_hz")
    duration_value = calibration.get("duration_seconds")
    numeric_types = (int, float, np.integer, np.floating)
    if any(
        isinstance(value, bool) or not isinstance(value, numeric_types)
        for value in (factor_value, standard_spl_value, duration_value)
    ):
        return None
    if isinstance(sample_rate_value, bool) or not isinstance(
        sample_rate_value,
        (int, np.integer),
    ):
        return None

    factor = float(factor_value)
    standard_spl = float(standard_spl_value)
    sample_rate = int(sample_rate_value)
    duration = float(duration_value)

    calibrated_at = calibration.get("calibrated_at")
    if not math.isfinite(factor) or factor <= 0.0:
        return None
    if not math.isfinite(standard_spl) or standard_spl <= 0.0:
        return None
    if sample_rate <= 0 or not math.isfinite(duration) or duration <= 0.0:
        return None
    if not isinstance(calibrated_at, str) or not calibrated_at.strip():
        return None

    return {
        "version": MIC_INPUT_CALIBRATION_VERSION,
        "input": {
            "api_name": api_name.strip(),
            "device_name": device_name.strip(),
            "channel_index": channel_index,
        },
        "calibration": {
            "v2pa_factor": factor,
            "standard_spl_db": standard_spl,
            "sample_rate_hz": sample_rate,
            "duration_seconds": duration,
            "calibrated_at": calibrated_at.strip(),
        },
    }


def load_mic_input_calibration(calibration_path=None):
    """Load and validate the persisted single-channel microphone calibration."""
    path = calibration_path or MIC_INPUT_CALIBRATION_PATH
    try:
        with _mic_input_calibration_io_lock:
            with open(path, "r", encoding="utf-8") as calibration_file:
                payload = json.load(calibration_file)
    except (OSError, ValueError, TypeError):
        return None
    return _validated_mic_input_calibration(payload)


def _atomic_write_json(path, payload):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=".mic_input_calibration_",
        suffix=".json.tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as calibration_file:
            json.dump(payload, calibration_file, ensure_ascii=False, indent=2)
            calibration_file.flush()
            os.fsync(calibration_file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def save_mic_input_calibration(
    v2pa_factor,
    input_device,
    input_channel,
    standard_spl_db,
    sample_rate_hz,
    duration_seconds,
    calibration_path=None,
    calibrated_at=None,
):
    """Atomically save a calibration bound to one input device and channel."""
    identity = build_mic_input_identity(input_device, input_channel)
    if identity is None:
        return False, "无法识别当前输入设备或通道。"

    payload = {
        "version": MIC_INPUT_CALIBRATION_VERSION,
        "input": identity,
        "calibration": {
            "v2pa_factor": v2pa_factor,
            "standard_spl_db": standard_spl_db,
            "sample_rate_hz": sample_rate_hz,
            "duration_seconds": duration_seconds,
            "calibrated_at": calibrated_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    }
    validated_payload = _validated_mic_input_calibration(payload)
    if validated_payload is None:
        return False, "输入校准结果无效，未保存。"

    path = calibration_path or MIC_INPUT_CALIBRATION_PATH
    try:
        with _mic_input_calibration_io_lock:
            _atomic_write_json(path, validated_payload)
    except OSError as exc:
        return False, f"输入校准配置保存失败：{str(exc)[:80]}"
    return True, "输入校准配置保存成功。"


def resolve_mic_input_calibration(input_device, input_channels, calibration_path=None):
    """Resolve a factor only when the current device and one channel match exactly."""
    if not isinstance(input_channels, (list, tuple)) or len(input_channels) != 1:
        return 0.0
    identity = build_mic_input_identity(input_device, input_channels[0])
    if identity is None:
        return 0.0

    payload = load_mic_input_calibration(calibration_path)
    if payload is None or payload["input"] != identity:
        return 0.0
    return float(payload["calibration"]["v2pa_factor"])


def get_mic_v2pa_factor(input_device=None, input_channels=None, calibration_path=None):
    """
        Read the single-channel microphone factor for the active hardware selection.

        Return:
            The microphone calibration v2pa_factor. Returns 0.0 when the
            selection is missing, ambiguous, invalid, or does not match the
            saved calibration.
    """
    return resolve_mic_input_calibration(
        input_device,
        input_channels,
        calibration_path=calibration_path,
    )


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
            mse = np.nanmean(residuals ** 2)
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
        load_code,  load_data = self.load_data_from_json(json_file_name)
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
        if not json_file_name.endswith('.json'):
            json_file_name = os.path.splitext(json_file_name)[0]
            json_file_name += '.json'
        json_file_path = model_consts.JSON_DIR_PATH + "/" + json_file_name
        directory = os.path.dirname(json_file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        if not isinstance(coefficients, (list, np.ndarray)):
            return error_code.INVALID_TYPE_DATA, "Coefficients must be a list or numpy array."
        coefficients = coefficients.tolist() if isinstance(coefficients, np.ndarray) else coefficients
        data = {
            "calibration_coefficients": coefficients,
            "max_voltage": max_voltages
        }
        try:
            with open(json_file_path, 'w') as json_file:
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
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)
                return error_code.OK, data
        except Exception as e:
            err_msg = "Failed to load coefficients data from json.%s" % (str(e)[:50])
            self.logger.error(err_msg)
            return error_code.INVALID_DATA_LOADING, err_msg

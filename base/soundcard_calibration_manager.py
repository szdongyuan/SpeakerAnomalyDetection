import json
import math
import os
import tempfile
import threading
from datetime import datetime

import numpy as np
import sounddevice as sd

from base.log_manager import LogManager
from base.sound_device_manager import SoundDeviceManager
from consts import error_code, model_consts


MIC_INPUT_CALIBRATION_PATH = os.path.join(
    model_consts.JSON_DIR_PATH,
    "mic_input_calibration.json",
)
MIC_INPUT_CALIBRATION_VERSION = 2
_mic_input_calibration_io_lock = threading.Lock()


class MicCalibrationError(Exception):
    """Base error for the microphone calibration file boundary."""


class MicCalibrationFormatError(MicCalibrationError):
    """The calibration JSON exists but is unsupported or invalid."""


class MicCalibrationIOError(MicCalibrationError):
    """The calibration JSON could not be read or atomically updated."""


def _device_value(device, key):
    getter = getattr(device, "get", None)
    return getter(key) if callable(getter) else None


def build_mic_input_identity(input_device):
    """Return the normalized identity shared by all channels of a device."""
    if input_device is None:
        return None

    try:
        hostapi_index = int(_device_value(input_device, "hostapi"))
    except (TypeError, ValueError, OverflowError):
        return None

    device_name = str(_device_value(input_device, "name") or "").strip()
    if not device_name:
        return None

    try:
        api_info = SoundDeviceManager.get_api_info(hostapi_index)
        api_name = str(_device_value(api_info, "name") or "").strip()
    except (TypeError, ValueError, OverflowError, OSError, sd.PortAudioError):
        return None

    if not api_name:
        return None

    return {
        "api_name": api_name,
        "device_name": device_name,
    }


def _validated_channel_index(input_channel):
    if isinstance(input_channel, bool) or not isinstance(input_channel, (int, np.integer)):
        raise ValueError("The physical input channel must be a non-negative integer.")
    channel_index = int(input_channel)
    if channel_index < 0:
        raise ValueError("The physical input channel must be a non-negative integer.")
    return channel_index


def _decimal_channel_key_to_index(channel_key):
    if len(channel_key) <= 9:
        return int(channel_key)

    channel_index = 0
    for offset in range(0, len(channel_key), 9):
        chunk = channel_key[offset:offset + 9]
        channel_index = channel_index * (10 ** len(chunk)) + int(chunk)
    return channel_index


def _channel_index_to_decimal_key(channel_index):
    if channel_index < 1_000_000_000:
        return str(channel_index)

    chunks = []
    while channel_index:
        channel_index, remainder = divmod(channel_index, 1_000_000_000)
        chunks.append(remainder)
    return str(chunks[-1]) + "".join(
        f"{chunk:09d}"
        for chunk in reversed(chunks[:-1])
    )


def _validated_timestamp(value, error_type):
    if not isinstance(value, str) or not value.strip():
        raise error_type("calibrated_at must be an ISO-8601 timestamp with an offset")
    normalized = value.strip()
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise error_type("calibrated_at must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise error_type("calibrated_at must include a UTC offset")
    return normalized


def _float_value(value, field_name, error_type):
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise error_type(f"{field_name} must be a representable number") from exc


def _validated_record(record, error_type):
    expected_fields = {
        "v2pa_factor",
        "standard_spl_db",
        "sample_rate_hz",
        "duration_seconds",
        "calibrated_at",
    }
    if not isinstance(record, dict) or set(record) != expected_fields:
        raise error_type("A channel calibration record has an invalid structure")

    numeric_types = (int, float, np.integer, np.floating)
    factor = record["v2pa_factor"]
    standard_spl = record["standard_spl_db"]
    sample_rate = record["sample_rate_hz"]
    duration = record["duration_seconds"]
    if isinstance(factor, bool) or not isinstance(factor, numeric_types):
        raise error_type("v2pa_factor must be numeric")
    factor_value = _float_value(factor, "v2pa_factor", error_type)
    if not math.isfinite(factor_value) or factor_value <= 0.0:
        raise error_type("v2pa_factor must be finite and positive")
    if isinstance(standard_spl, bool) or not isinstance(standard_spl, numeric_types):
        raise error_type("standard_spl_db must be numeric")
    standard_spl_value = _float_value(standard_spl, "standard_spl_db", error_type)
    if not math.isfinite(standard_spl_value) or standard_spl_value not in (94.0, 114.0):
        raise error_type("standard_spl_db must be exactly 94 or 114")
    if isinstance(standard_spl, np.integer):
        standard_spl = int(standard_spl)
    elif isinstance(standard_spl, np.floating):
        standard_spl = float(standard_spl)
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise error_type("sample_rate_hz must be an integer")
    sample_rate_value = int(sample_rate)
    if sample_rate_value <= 0:
        raise error_type("sample_rate_hz must be positive")
    if isinstance(duration, bool) or not isinstance(duration, numeric_types):
        raise error_type("duration_seconds must be numeric")
    duration_value = _float_value(duration, "duration_seconds", error_type)
    if not math.isfinite(duration_value) or duration_value <= 0.0:
        raise error_type("duration_seconds must be finite and positive")

    return {
        "v2pa_factor": factor_value,
        "standard_spl_db": standard_spl,
        "sample_rate_hz": sample_rate_value,
        "duration_seconds": duration_value,
        "calibrated_at": _validated_timestamp(record["calibrated_at"], error_type),
    }


def _validated_mic_input_registry(payload):
    if not isinstance(payload, dict) or set(payload) != {"version", "devices"}:
        raise MicCalibrationFormatError("The microphone calibration root is invalid")
    version = payload["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 2:
        raise MicCalibrationFormatError("Unsupported microphone calibration version")
    if not isinstance(payload["devices"], list):
        raise MicCalibrationFormatError("devices must be a list")

    canonical_devices = []
    seen_identities = set()
    for device in payload["devices"]:
        if not isinstance(device, dict) or set(device) != {"input", "channels"}:
            raise MicCalibrationFormatError("A device calibration has an invalid structure")
        identity = device["input"]
        if not isinstance(identity, dict) or set(identity) != {"api_name", "device_name"}:
            raise MicCalibrationFormatError("A device identity has an invalid structure")
        api_name = identity["api_name"]
        device_name = identity["device_name"]
        if not isinstance(api_name, str) or not api_name.strip():
            raise MicCalibrationFormatError("api_name must be a non-empty string")
        if not isinstance(device_name, str) or not device_name.strip():
            raise MicCalibrationFormatError("device_name must be a non-empty string")
        canonical_identity = {
            "api_name": api_name.strip(),
            "device_name": device_name.strip(),
        }
        identity_key = (canonical_identity["api_name"], canonical_identity["device_name"])
        if identity_key in seen_identities:
            raise MicCalibrationFormatError("Duplicate microphone device identity")
        seen_identities.add(identity_key)

        channels = device["channels"]
        if not isinstance(channels, dict):
            raise MicCalibrationFormatError("channels must be an object")
        canonical_channels = {}
        for channel_key, record in channels.items():
            if (
                not isinstance(channel_key, str)
                or not channel_key.isascii()
                or not channel_key.isdecimal()
                or (channel_key != "0" and channel_key.startswith("0"))
            ):
                raise MicCalibrationFormatError("Channel keys must be canonical non-negative integers")
            canonical_channels[channel_key] = _validated_record(
                record,
                MicCalibrationFormatError,
            )
        canonical_devices.append({
            "input": canonical_identity,
            "channels": canonical_channels,
        })

    return {"version": MIC_INPUT_CALIBRATION_VERSION, "devices": canonical_devices}


def _load_mic_input_calibration_unlocked(path):
    try:
        with open(path, "r", encoding="utf-8") as calibration_file:
            payload = json.load(calibration_file)
    except FileNotFoundError:
        return {"version": MIC_INPUT_CALIBRATION_VERSION, "devices": []}
    except json.JSONDecodeError as exc:
        raise MicCalibrationFormatError("The microphone calibration JSON is malformed") from exc
    except RecursionError as exc:
        raise MicCalibrationFormatError("The microphone calibration JSON is too deeply nested") from exc
    except ValueError as exc:
        raise MicCalibrationFormatError("The microphone calibration JSON is invalid") from exc
    except UnicodeError as exc:
        raise MicCalibrationFormatError("The microphone calibration JSON encoding is invalid") from exc
    except OSError as exc:
        raise MicCalibrationIOError("The microphone calibration file could not be read") from exc

    if (
        isinstance(payload, dict)
        and not isinstance(payload.get("version"), bool)
        and isinstance(payload.get("version"), int)
        and payload.get("version") == 1
    ):
        return {"version": MIC_INPUT_CALIBRATION_VERSION, "devices": []}
    return _validated_mic_input_registry(payload)


def load_mic_input_calibration(calibration_path=None):
    """Load the canonical version-2 microphone calibration registry."""
    path = calibration_path or MIC_INPUT_CALIBRATION_PATH
    with _mic_input_calibration_io_lock:
        return _load_mic_input_calibration_unlocked(path)


def _atomic_write_json(path, payload):
    directory = os.path.dirname(path) or "."
    fd = None
    temporary_path = None
    try:
        os.makedirs(directory, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=".mic_input_calibration_",
            suffix=".json.tmp",
            dir=directory,
        )
        calibration_file = os.fdopen(fd, "w", encoding="utf-8")
        fd = None
        with calibration_file:
            json.dump(payload, calibration_file, ensure_ascii=False, indent=2)
            calibration_file.flush()
            os.fsync(calibration_file.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError, UnicodeError) as exc:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        raise MicCalibrationIOError("The microphone calibration file could not be updated") from exc


def _device_entry(registry, identity):
    return next(
        (device for device in registry["devices"] if device["input"] == identity),
        None,
    )


def _validated_save_record(
    v2pa_factor,
    standard_spl_db,
    sample_rate_hz,
    duration_seconds,
    calibrated_at,
):
    return _validated_record(
        {
            "v2pa_factor": v2pa_factor,
            "standard_spl_db": standard_spl_db,
            "sample_rate_hz": sample_rate_hz,
            "duration_seconds": duration_seconds,
            "calibrated_at": calibrated_at,
        },
        ValueError,
    )


def save_mic_channel_calibration(
    v2pa_factor,
    input_device,
    input_channel,
    standard_spl_db,
    sample_rate_hz,
    duration_seconds,
    calibration_path=None,
    calibrated_at=None,
):
    """Atomically save one physical-channel calibration record."""
    identity = build_mic_input_identity(input_device)
    if identity is None:
        raise ValueError("The input device identity is invalid.")
    channel_index = _validated_channel_index(input_channel)
    if calibrated_at is None:
        calibrated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    record = _validated_save_record(
        v2pa_factor,
        standard_spl_db,
        sample_rate_hz,
        duration_seconds,
        calibrated_at,
    )

    path = calibration_path or MIC_INPUT_CALIBRATION_PATH
    with _mic_input_calibration_io_lock:
        registry = _load_mic_input_calibration_unlocked(path)
        device = _device_entry(registry, identity)
        if device is None:
            device = {"input": identity, "channels": {}}
            registry["devices"].append(device)
        device["channels"][_channel_index_to_decimal_key(channel_index)] = record
        _atomic_write_json(path, _validated_mic_input_registry(registry))


def clear_mic_channel_calibrations(
    input_device,
    input_channels,
    calibration_path=None,
):
    """Remove selected physical-channel records, returning whether data changed."""
    identity = build_mic_input_identity(input_device)
    if identity is None:
        raise ValueError("The input device identity is invalid.")
    if not isinstance(input_channels, (list, tuple)):
        raise ValueError("Input channels must be a sequence.")
    channels = {_validated_channel_index(channel) for channel in input_channels}

    path = calibration_path or MIC_INPUT_CALIBRATION_PATH
    with _mic_input_calibration_io_lock:
        registry = _load_mic_input_calibration_unlocked(path)
        device = _device_entry(registry, identity)
        if device is None:
            return False
        changed = False
        for channel in channels:
            channel_key = _channel_index_to_decimal_key(channel)
            if device["channels"].pop(channel_key, None) is not None:
                changed = True
        if not changed:
            return False
        if not device["channels"]:
            registry["devices"].remove(device)
        _atomic_write_json(path, _validated_mic_input_registry(registry))
        return True


def load_mic_channel_calibrations(input_device, calibration_path=None):
    """Return calibration records keyed by physical channel for one device."""
    identity = build_mic_input_identity(input_device)
    if identity is None:
        return {}
    registry = load_mic_input_calibration(calibration_path)
    device = _device_entry(registry, identity)
    if device is None:
        return {}
    return {
        _decimal_channel_key_to_index(channel): dict(record)
        for channel, record in device["channels"].items()
    }


def load_mic_channel_v2pa_factors(input_device, calibration_path=None):
    """Return finite positive factors keyed by exact physical channel."""
    return {
        channel: float(record["v2pa_factor"])
        for channel, record in load_mic_channel_calibrations(
            input_device,
            calibration_path,
        ).items()
    }


def resolve_mic_channel_v2pa_factor(
    input_device,
    input_channel,
    calibration_path=None,
):
    """Return one exact physical-channel factor, or None when it is absent."""
    try:
        channel_index = _validated_channel_index(input_channel)
    except ValueError:
        return None
    identity = build_mic_input_identity(input_device)
    if identity is None:
        return None
    registry = load_mic_input_calibration(calibration_path)
    device = _device_entry(registry, identity)
    if device is None:
        return None
    channel_key = _channel_index_to_decimal_key(channel_index)
    record = device["channels"].get(channel_key)
    return None if record is None else float(record["v2pa_factor"])


def resolve_mic_input_calibration(input_device, input_channels, calibration_path=None):
    """Resolve a factor only when the current device and one channel match exactly."""
    if not isinstance(input_channels, (list, tuple)) or len(input_channels) != 1:
        return 0.0
    if build_mic_input_identity(input_device) is None:
        return 0.0
    factor = resolve_mic_channel_v2pa_factor(
        input_device,
        input_channels[0],
        calibration_path,
    )
    if factor is None:
        return 0.0
    return float(factor)


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

import json
import os
import tempfile

from base.log_manager import LogManager

from consts.calibration_consts import (
    INPUT_CALIBRATION_MODE_MANUAL,
    INPUT_CALIBRATION_MODE_STANDARD_SPL,
)
from consts.running_consts import DEFAULT_DIR


_DEFAULT_PATH = os.path.join(
    DEFAULT_DIR,
    "ui",
    "ui_config",
    "input_calibration_preferences.json",
)


def _warn(logger, message):
    target_logger = logger if logger is not None else LogManager.set_log_handler("core")
    target_logger.warning(message)


def load_input_calibration_mode(path=None, logger=None):
    target_path = os.fspath(path or _DEFAULT_PATH)
    try:
        with open(target_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except FileNotFoundError:
        return INPUT_CALIBRATION_MODE_STANDARD_SPL
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _warn(
            logger,
            f"Failed to load input calibration preferences from {target_path}: {exc}",
        )
        return INPUT_CALIBRATION_MODE_STANDARD_SPL

    valid_modes = (
        INPUT_CALIBRATION_MODE_STANDARD_SPL,
        INPUT_CALIBRATION_MODE_MANUAL,
    )
    if (
        not isinstance(payload, dict)
        or type(payload.get("version")) is not int
        or payload["version"] != 1
        or type(payload.get("mode")) is not str
        or payload["mode"] not in valid_modes
    ):
        _warn(logger, f"Invalid input calibration preferences in {target_path}")
        return INPUT_CALIBRATION_MODE_STANDARD_SPL
    return payload["mode"]


def save_input_calibration_mode(mode, path=None, logger=None):
    valid_modes = (
        INPUT_CALIBRATION_MODE_STANDARD_SPL,
        INPUT_CALIBRATION_MODE_MANUAL,
    )
    if mode not in valid_modes:
        raise ValueError(f"Unsupported input calibration mode: {mode!r}")

    target_path = os.path.abspath(os.fspath(path or _DEFAULT_PATH))
    parent_dir = os.path.dirname(target_path) or "."
    temp_file_fd = None
    temp_file_path = None
    try:
        os.makedirs(parent_dir, exist_ok=True)
        temp_file_fd, temp_file_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(target_path)}.",
            suffix=".tmp",
            dir=parent_dir,
        )
        with os.fdopen(temp_file_fd, "w", encoding="utf-8") as file:
            temp_file_fd = None
            json.dump({"version": 1, "mode": mode}, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_file_path, target_path)
        temp_file_path = None
        return True
    except OSError as exc:
        _warn(
            logger,
            f"Failed to save input calibration preferences to {target_path}: {exc}",
        )
        return False
    finally:
        if temp_file_fd is not None:
            try:
                os.close(temp_file_fd)
            except OSError:
                pass
        if temp_file_path is not None:
            try:
                os.remove(temp_file_path)
            except OSError:
                pass

import json
import os
import tempfile

from consts.running_consts import DEFAULT_DIR


_PREFERENCES_VERSION = 1
_SUPPRESS_KEY = "suppress_uncalibrated_microphone_warning"
_DEFAULT_PATH = os.path.join(
    DEFAULT_DIR,
    "ui",
    "ui_config",
    "analysis_warning_preferences.json",
)


def _log_warning(logger, message):
    warning = getattr(logger, "warning", None)
    if callable(warning):
        warning(message)


def is_uncalibrated_microphone_warning_suppressed(path=None, logger=None):
    target_path = os.fspath(path or _DEFAULT_PATH)
    try:
        with open(target_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except FileNotFoundError:
        return False
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _log_warning(
            logger,
            f"Failed to load analysis warning preferences from {target_path}: {exc}",
        )
        return False

    if (
        not isinstance(payload, dict)
        or type(payload.get("version")) is not int
        or payload["version"] != _PREFERENCES_VERSION
        or type(payload.get(_SUPPRESS_KEY)) is not bool
    ):
        _log_warning(
            logger,
            f"Invalid analysis warning preferences in {target_path}",
        )
        return False
    return payload[_SUPPRESS_KEY]


def _cleanup_temporary_file(temp_fd, temp_path, target_path, logger):
    if temp_fd is not None:
        try:
            os.close(temp_fd)
        except OSError as exc:
            _log_warning(
                logger,
                "Failed to close temporary analysis warning preference file "
                f"for {target_path}: {exc}",
            )
    if temp_path is not None:
        try:
            os.unlink(temp_path)
        except OSError as exc:
            _log_warning(
                logger,
                "Failed to remove temporary analysis warning preference file "
                f"{temp_path}: {exc}",
            )


def save_uncalibrated_microphone_warning_suppressed(path=None, logger=None):
    target_path = os.fspath(path or _DEFAULT_PATH)
    parent = os.path.dirname(os.path.abspath(target_path)) or "."
    temp_fd = None
    temp_path = None
    try:
        os.makedirs(parent, exist_ok=True)
        temp_fd, temp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(target_path)}.",
            suffix=".tmp",
            dir=parent,
        )
        with os.fdopen(temp_fd, "w", encoding="utf-8") as file:
            temp_fd = None
            json.dump(
                {
                    "version": _PREFERENCES_VERSION,
                    _SUPPRESS_KEY: True,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, target_path)
        temp_path = None
        return True
    except OSError as exc:
        _log_warning(
            logger,
            f"Failed to save analysis warning preferences to {target_path}: {exc}",
        )
        return False
    finally:
        _cleanup_temporary_file(temp_fd, temp_path, target_path, logger)

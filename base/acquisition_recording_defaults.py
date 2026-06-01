import json
import os
import tempfile
from copy import deepcopy

from consts.running_consts import (
    DEFAULT_DIR,
    DEFAULT_PLAY_AND_RECORD_DETAIL,
    DEFAULT_RECORD_ONLY_DETAIL,
    VALID_ACQUISITION_MODES,
)


def get_acquisition_default_config_path():
    return os.path.join(DEFAULT_DIR, "ui", "ui_config", "acquisition_default_config.json")


def _as_float(value, default):
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default):
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value, default):
    return value if isinstance(value, bool) else default


def normalize_play_record_detail(detail):
    normalized = deepcopy(detail) if isinstance(detail, dict) else {}
    normalized["use_streaming_recording"] = _as_bool(
        normalized.get(
            "use_streaming_recording",
            DEFAULT_PLAY_AND_RECORD_DETAIL["use_streaming_recording"],
        ),
        DEFAULT_PLAY_AND_RECORD_DETAIL["use_streaming_recording"],
    )
    return normalized


def normalize_record_only_detail(detail):
    normalized = deepcopy(detail) if isinstance(detail, dict) else {}
    normalized["total_time"] = _as_float(
        normalized.get("total_time"),
        DEFAULT_RECORD_ONLY_DETAIL["total_time"],
    )
    normalized["sample_rate"] = _as_int(
        normalized.get("sample_rate"),
        DEFAULT_RECORD_ONLY_DETAIL["sample_rate"],
    )
    monitor_playback = normalized.get("monitor_playback", DEFAULT_RECORD_ONLY_DETAIL["monitor_playback"])
    normalized["monitor_playback"] = _as_bool(monitor_playback, DEFAULT_RECORD_ONLY_DETAIL["monitor_playback"])
    monitor_input_channel = _as_int(
        normalized.get("monitor_input_channel"),
        DEFAULT_RECORD_ONLY_DETAIL["monitor_input_channel"],
    )
    if monitor_input_channel < 0:
        monitor_input_channel = DEFAULT_RECORD_ONLY_DETAIL["monitor_input_channel"]
    normalized["monitor_input_channel"] = monitor_input_channel
    normalized["monitor_gain_db"] = _as_float(
        normalized.get("monitor_gain_db"),
        DEFAULT_RECORD_ONLY_DETAIL["monitor_gain_db"],
    )
    normalized["use_streaming_recording"] = _as_bool(
        normalized.get(
            "use_streaming_recording",
            DEFAULT_RECORD_ONLY_DETAIL["use_streaming_recording"],
        ),
        DEFAULT_RECORD_ONLY_DETAIL["use_streaming_recording"],
    )
    return normalized


def _built_in_defaults():
    return {
        "PLAY_AND_RECORD": normalize_play_record_detail(DEFAULT_PLAY_AND_RECORD_DETAIL),
        "RECORD_ONLY": normalize_record_only_detail(DEFAULT_RECORD_ONLY_DETAIL),
    }


def _read_json(path, logger=None, missing_ok=True):
    if not os.path.exists(path):
        if not missing_ok:
            logger.warning(f"Acquisition default config does not exist: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as exc:
        logger.warning(f"Failed to load acquisition default config: {exc}")
        return None
    if not isinstance(data, dict):
        logger.warning("Acquisition default config must be a JSON object.")
        return None
    return data


def load_acquisition_defaults(path=None, logger=None):
    target_path = os.fspath(path or get_acquisition_default_config_path())
    data = _read_json(target_path, logger=logger)
    if data is None:
        return _built_in_defaults()

    cfg = deepcopy(data)
    cfg["PLAY_AND_RECORD"] = normalize_play_record_detail(cfg.get("PLAY_AND_RECORD"))
    cfg["RECORD_ONLY"] = normalize_record_only_detail(cfg.get("RECORD_ONLY"))
    return cfg


def save_acquisition_default(mode, detail, path=None, logger=None):
    if mode not in VALID_ACQUISITION_MODES:
        logger.warning(f"Invalid acquisition default mode: {mode}")
        return False
    if not isinstance(detail, dict):
        logger.warning("Acquisition default detail must be a dict.")
        return False

    target_path = os.fspath(path or get_acquisition_default_config_path())
    try:
        parent = os.path.dirname(target_path) or "."
        os.makedirs(parent, exist_ok=True)
        data = _read_json(target_path, logger=logger) or {}
        existing_detail = data.get(mode)
        if not isinstance(existing_detail, dict):
            existing_detail = {}
        merged_detail = deepcopy(existing_detail)
        merged_detail.update(deepcopy(detail))
        if mode == "PLAY_AND_RECORD":
            data[mode] = normalize_play_record_detail(merged_detail)
        else:
            data[mode] = normalize_record_only_detail(merged_detail)

        fd, temp_path = tempfile.mkstemp(
            prefix=".acquisition_default_",
            suffix=".json",
            dir=parent,
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
            os.replace(temp_path, target_path)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise
        return True
    except Exception as exc:
        logger.warning(f"Failed to save acquisition default config: {exc}")
        return False

import hashlib
import os
import re
import logging

import numpy as np

from base.file_ops import FileOps
from base.pre_processing.swept_sine_chirps import StimulusSignal
from base.stimulus_signal.frequency_stepped import (
    generate_frequency_stepped,
    resolve_frequency_stepped_schedule,
)
from base.stimulus_signal.methods import normalize_stimulus_method
from base.save_data import save_audio_simple
from base.load_audio import load_audio_simple
from base.log_manager import LogManager
from consts.frequency_stepped_consts import (
    FREQUENCY_STEPPED_FILENAME_KEYS,
    FREQUENCY_STEPPED_GENERATOR_KEYS,
    FREQUENCY_STEPPED_METHOD,
    FREQUENCY_STEPPED_MODES,
    SUPPORTED_STIMULUS_METHODS,
)
from consts import model_consts
from consts.running_consts import DEFAULT_DIR


def _get_logger(logger=None):
    if logger is not None:
        return logger
    try:
        return LogManager.set_log_handler("core")
    except Exception:
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger("core")


def _sanitize_filename(name: str) -> str:
    """
    Make a string safe for use as a filename on Windows/macOS/Linux.
    """
    if not name:
        return "stimulus"
    # Replace reserved characters on Windows: \ / : * ? " < > |
    name = re.sub(r'[\\\\/:*?"<>|]+', "_", name)
    name = name.strip().strip(".")
    return name or "stimulus"


def _build_stimulus_name_from_info(stimulus_info: dict) -> str:
    # Keep behavior aligned with ui/stimulus_window.py -> save_stimulus_to_json()
    try:
        info = stimulus_info or {}
        if normalize_stimulus_method(info.get("stimulus_method", "")) == FREQUENCY_STEPPED_METHOD:
            values = []
            for key in FREQUENCY_STEPPED_FILENAME_KEYS:
                if key not in info or info.get(key) is None:
                    continue
                value = info.get(key)
                if key == "stimulus_method":
                    value = FREQUENCY_STEPPED_METHOD
                values.append(str(value))
            raw = "_".join(values)
        else:
            raw = "_".join(str(v) for v in info.values())
    except Exception:
        raw = ""
    raw = _sanitize_filename(raw)
    # Prevent extremely long filenames on Windows
    if len(raw) > 180:
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
        raw = raw[:160] + "_" + digest
    return raw


def _iter_candidate_paths(raw_path: str, base_dirs: list) -> list:
    """
    Build candidate absolute paths from a raw path string.
    raw_path may be absolute, relative to DEFAULT_DIR, or relative to config dir.
    """
    if not raw_path:
        return []
    candidates = []

    # Keep original (in case it's already absolute or correct relative to cwd)
    candidates.append(raw_path)

    # Absolute path
    if os.path.isabs(raw_path):
        candidates.append(raw_path)

    # Relative to each base dir (DEFAULT_DIR, config dir, etc.)
    for base_dir in base_dirs:
        if base_dir:
            candidates.append(os.path.join(base_dir, raw_path))

    # Normalize, de-dup, and unify separators
    normalized = []
    seen = set()
    for p in candidates:
        if not p:
            continue
        try:
            pp = os.path.abspath(p)
        except Exception:
            pp = p
        pp = pp.replace("\\", "/")
        if pp not in seen:
            seen.add(pp)
            normalized.append(pp)
    return normalized


def _try_load_existing_wav(detail: dict, sample_rate: int, base_dirs: list, logger=None):
    """
    Try loading wav from detail["stimulus_signal_path"] then detail["load_stimulus_signal_path"].
    Returns (y, used_abs_path) or (None, None).
    """
    logger = _get_logger(logger)

    for key in ("stimulus_signal_path", "load_stimulus_signal_path"):
        raw = (detail or {}).get(key)
        for p in _iter_candidate_paths(raw, base_dirs):
            if not p:
                continue
            if not os.path.exists(p):
                continue
            try:
                y, _ = load_audio_simple(p, sample_rate)
                if y is None:
                    continue
                return y, p
            except Exception as e:
                logger.warning(f"Failed to load stimulus wav at {p}: {e}")
                continue
    return None, None


def _valid_frequency_stepped_mode(value):
    if value is None:
        return None
    mode = str(value).strip()
    if mode in FREQUENCY_STEPPED_MODES:
        return mode
    return None


def _require_supported_stimulus_method(method):
    if method not in SUPPORTED_STIMULUS_METHODS:
        raise ValueError(f"Unsupported stimulus_method: {method}")


def _required_frequency_stepped_sample_rate(stimulus_info: dict) -> int:
    info = stimulus_info or {}
    if "sample_rate" not in info:
        raise ValueError("frequency_stepped requires sample_rate")
    value = info.get("sample_rate")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("frequency_stepped sample_rate must be a positive integer")
    try:
        sample_rate = int(value)
    except (TypeError, ValueError):
        raise ValueError("frequency_stepped sample_rate must be a positive integer") from None
    try:
        if float(value) != sample_rate:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("frequency_stepped sample_rate must be a positive integer") from None
    if sample_rate <= 0:
        raise ValueError("frequency_stepped sample_rate must be a positive integer")
    return sample_rate


def _repair_frequency_stepped_mode_fields(stimulus_info: dict) -> dict:
    repaired = dict(stimulus_info or {})
    mode = _valid_frequency_stepped_mode(repaired.get("frequency_mode"))
    stimulus_type = _valid_frequency_stepped_mode(repaired.get("stimulus_type"))
    if mode is None:
        if stimulus_type is None:
            raise ValueError("frequency_stepped requires frequency_mode or stimulus_type")
        mode = stimulus_type
    repaired["frequency_mode"] = mode
    repaired["stimulus_type"] = mode
    return repaired


def _valid_retained_frequencies(value):
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        frequencies = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if frequencies.ndim != 1 or frequencies.size == 0:
        return None
    if not np.all(np.isfinite(frequencies)) or not np.all(frequencies > 0):
        return None
    return [float(frequency) for frequency in frequencies]


def _frequency_stepped_generator_kwargs(stimulus_info: dict) -> dict:
    """
    Return only keyword arguments accepted by generate_frequency_stepped().
    """
    info = dict(stimulus_info or {})
    method = normalize_stimulus_method(info.get("stimulus_method", ""))
    if method != FREQUENCY_STEPPED_METHOD:
        raise ValueError(f"Unsupported {FREQUENCY_STEPPED_METHOD} stimulus_method: {info.get('stimulus_method')}")

    repaired = _repair_frequency_stepped_mode_fields(info)
    repaired["sample_rate"] = _required_frequency_stepped_sample_rate(repaired)
    kwargs = {}
    for key in FREQUENCY_STEPPED_GENERATOR_KEYS:
        if key in repaired:
            kwargs[key] = repaired[key]
    kwargs["frequency_mode"] = repaired["frequency_mode"]
    kwargs["stimulus_type"] = repaired["frequency_mode"]
    kwargs["amplitude"] = repaired.get("amplitude", 1.0)
    kwargs["generate_waveform"] = True

    retained_frequencies = _valid_retained_frequencies(repaired.get("frequencies"))
    if retained_frequencies is not None:
        kwargs["frequencies"] = retained_frequencies
    elif "frequencies" in kwargs:
        del kwargs["frequencies"]

    return kwargs


def _merge_frequency_stepped_metadata(stimulus_info: dict, generated_metadata: dict) -> dict:
    merged = dict(generated_metadata)
    for key, value in (stimulus_info or {}).items():
        if key in merged and key not in ("stimulus_label",):
            continue
        merged[key] = value
    merged["stimulus_method"] = FREQUENCY_STEPPED_METHOD
    merged["frequency_mode"] = generated_metadata["frequency_mode"]
    merged["stimulus_type"] = generated_metadata["frequency_mode"]
    return merged


def _generate_frequency_stepped_runtime(stimulus_info: dict):
    repaired_info = _repair_frequency_stepped_mode_fields(
        {**(stimulus_info or {}), "stimulus_method": FREQUENCY_STEPPED_METHOD}
    )
    sample_rate = _required_frequency_stepped_sample_rate(repaired_info)
    schedule = resolve_frequency_stepped_schedule(
        repaired_info,
        sample_rate,
    )
    kwargs = _frequency_stepped_generator_kwargs({**repaired_info, **schedule.metadata})
    result = generate_frequency_stepped(**kwargs)
    metadata = _merge_frequency_stepped_metadata(repaired_info, result.metadata)
    result.metadata.clear()
    result.metadata.update(metadata)
    return result


def generate_and_save_stimulus(detail: dict, logger=None):
    """
    Generate stimulus from detail["stimulus_info"], save to STORED_STIMULUS_PATH,
    and update detail["stimulus_signal_path"] (relative to DEFAULT_DIR).

    Returns (stimulus_data, sample_rate, saved_abs_path) or (None, sr, None) on failure.
    """
    logger = _get_logger(logger)
    stimulus_info = (detail or {}).get("stimulus_info") or {}
    method = normalize_stimulus_method(stimulus_info.get("stimulus_method", "chirp"))
    _require_supported_stimulus_method(method)
    if method == FREQUENCY_STEPPED_METHOD:
        sample_rate = _required_frequency_stepped_sample_rate(stimulus_info)
    else:
        sample_rate = int(stimulus_info.get("sample_rate") or detail.get("sample_rate") or 44100)
    total_time = float(stimulus_info.get("total_time") or detail.get("total_time") or 0.0)

    # Generate data
    stimulus_data = None
    try:
        if method == FREQUENCY_STEPPED_METHOD:
            result = _generate_frequency_stepped_runtime(stimulus_info)
            stimulus_data = result.data
            sample_rate = int(result.sample_rate)
            stimulus_info.clear()
            stimulus_info.update(result.metadata)
            detail["stimulus_info"] = stimulus_info
            detail["alignment_sample_count"] = result.metadata.get("alignment_sample_count")
        else:
            create_function_dict = {
                "chirp": StimulusSignal().generate_chirps,
                "step": StimulusSignal().generate_steps,
                "noise": StimulusSignal().generate_noise,
            }
            create_fn = create_function_dict.get(method)
            if create_fn is None:
                raise ValueError(f"Unsupported stimulus_method: {method}")
            if stimulus_info.get("stimulus_method") != method:
                stimulus_info = dict(stimulus_info)
                stimulus_info["stimulus_method"] = method
            stimulus_data, _sr = create_fn(**stimulus_info)
            if _sr:
                sample_rate = int(_sr)
        if stimulus_data is None:
            raise ValueError("Generated stimulus_data is None")
        stimulus_data = np.asarray(stimulus_data, dtype="float32")
    except Exception as e:
        logger.error(f"Failed to generate stimulus from stimulus_info: {e}")
        if method == FREQUENCY_STEPPED_METHOD:
            raise ValueError(f"Failed to generate frequency_stepped stimulus: {e}") from e
        stimulus_data = None

    # Fallback to silence if generation failed
    if stimulus_data is None or stimulus_data.size == 0:
        num_samples = int(max(total_time, 0.0) * sample_rate)
        stimulus_data = np.zeros(max(num_samples, 1), dtype="float32")

    # Save to disk
    try:
        stimulus_name = _build_stimulus_name_from_info(stimulus_info)
        save_path = os.path.join(model_consts.STORED_STIMULUS_PATH, f"{stimulus_name}.wav")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_audio_simple(save_path, stimulus_data, sample_rate)
        rel = FileOps.get_relative_path(save_path, DEFAULT_DIR)
        detail["stimulus_signal_path"] = rel
        # We re-generated, so the original loaded wav is not relevant anymore
        detail["load_stimulus_signal_path"] = None
        return stimulus_data, sample_rate, save_path
    except Exception as e:
        logger.error(f"Failed to save regenerated stimulus wav: {e}")
        return stimulus_data, sample_rate, None


def set_data_struct_stimulus_signal(data_struct, detail, using_config_path: str = None, logger=None) -> bool:
    """
    Safe variant: if stimulus wav is missing, regenerate from stimulus_info, save it,
    and update detail['stimulus_signal_path'] so caller can write back config.

    Returns:
        bool: whether detail was modified (path fixed/regenerated)
    """
    logger = _get_logger(logger)
    if detail is None:
        return False

    stimulus_info = detail.get("stimulus_info") or {}
    method = normalize_stimulus_method(stimulus_info.get("stimulus_method", "chirp"))
    _require_supported_stimulus_method(method)
    if method == FREQUENCY_STEPPED_METHOD:
        sample_rate = _required_frequency_stepped_sample_rate(stimulus_info)
    else:
        sample_rate = int(stimulus_info.get("sample_rate") or detail.get("sample_rate") or 44100)

    config_dir = None
    if using_config_path:
        try:
            config_dir = os.path.dirname(using_config_path)
        except Exception:
            config_dir = None
    base_dirs = [DEFAULT_DIR, config_dir]

    # 1) Try load existing wav. frequency_stepped metadata is authoritative, not a saved WAV artifact.
    if method == FREQUENCY_STEPPED_METHOD:
        stimulus_signal = None
    else:
        stimulus_signal, _used_path = _try_load_existing_wav(detail, sample_rate, base_dirs, logger=logger)
    modified = False

    # 2) Regenerate when missing / unloadable
    if stimulus_signal is None:
        stimulus_signal, sample_rate, saved_path = generate_and_save_stimulus(detail, logger=logger)
        modified = True
        if saved_path:
            logger.info(f"Stimulus wav missing; regenerated and saved: {saved_path}")

    # Apply to data_struct
    if hasattr(data_struct, "alignment_sample_count"):
        delattr(data_struct, "alignment_sample_count")
    current_stimulus_info = detail.get("stimulus_info") or stimulus_info
    if method == FREQUENCY_STEPPED_METHOD:
        data_struct.stimulus_info = current_stimulus_info
    else:
        data_struct.stimulus_info = dict(current_stimulus_info)
        data_struct.stimulus_info.pop("alignment_sample_count", None)
    data_struct.stimulus_data = stimulus_signal
    data_struct.sample_rate = sample_rate
    if method == FREQUENCY_STEPPED_METHOD:
        try:
            data_struct.alignment_sample_count = int(data_struct.stimulus_info.get("alignment_sample_count"))
        except (TypeError, ValueError, AttributeError):
            pass
    return modified


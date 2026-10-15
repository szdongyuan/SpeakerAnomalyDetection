import hashlib
import copy
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


def _try_load_existing_wav(detail: dict, sample_rate: int, base_dirs: list, logger=None, keys=None):
    """
    Try loading wav from configured detail path keys.
    Returns (y, used_abs_path) or (None, None).
    """
    logger = _get_logger(logger)

    if keys is None:
        keys = ("stimulus_signal_path", "load_stimulus_signal_path")

    for key in keys:
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


def _waveform_sample_count(stimulus_signal) -> int:
    signal = np.asarray(stimulus_signal)
    if signal.ndim == 0:
        return 0
    return int(signal.shape[0])


def _sync_loaded_wav_runtime_metadata(stimulus_info: dict, stimulus_signal, sample_rate: int) -> None:
    if not isinstance(stimulus_info, dict):
        return
    sample_count = _waveform_sample_count(stimulus_signal)
    stimulus_info["sample_rate"] = int(sample_rate)
    stimulus_info["total_time"] = float(sample_count) / float(sample_rate)
    for key in ("sample_count", "num_samples", "playback_sample_count", "alignment_sample_count"):
        if key in stimulus_info:
            stimulus_info[key] = sample_count


def _require_supported_stimulus_method(method):
    if method not in SUPPORTED_STIMULUS_METHODS:
        raise ValueError(f"Unsupported stimulus_method: {method}")


def _runtime_sample_rate_value(runtime_sample_rate) -> int:
    if isinstance(runtime_sample_rate, (bool, np.bool_)):
        raise ValueError("runtime_sample_rate must be a positive integer")
    try:
        sample_rate = int(runtime_sample_rate)
    except (TypeError, ValueError):
        raise ValueError("runtime_sample_rate must be a positive integer") from None
    try:
        if float(runtime_sample_rate) != sample_rate:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("runtime_sample_rate must be a positive integer") from None
    if sample_rate <= 0:
        raise ValueError("runtime_sample_rate must be a positive integer")
    return sample_rate


def _runtime_stimulus_info(detail: dict, runtime_sample_rate: int) -> dict:
    stimulus_info = dict((detail or {}).get("stimulus_info") or {})
    method = normalize_stimulus_method(stimulus_info.get("stimulus_method", "chirp"))
    _require_supported_stimulus_method(method)
    stimulus_info["stimulus_method"] = method
    stimulus_info["sample_rate"] = runtime_sample_rate
    if detail is not None:
        detail["stimulus_info"] = stimulus_info
        detail["sample_rate"] = runtime_sample_rate
    return stimulus_info


def _frequency_stepped_generator_kwargs(schedule_metadata: dict) -> dict:
    """Select generator inputs from an already resolved schedule."""
    return {
        key: schedule_metadata[key]
        for key in FREQUENCY_STEPPED_GENERATOR_KEYS
        if key in schedule_metadata
    }


def _merge_frequency_stepped_metadata(stimulus_info: dict, generated_metadata: dict) -> dict:
    merged = dict(generated_metadata)
    for key, value in (stimulus_info or {}).items():
        if key in merged and key not in ("stimulus_label",):
            continue
        merged[key] = value
    return merged


def _generate_frequency_stepped_runtime(stimulus_info: dict, sample_rate: int):
    schedule = resolve_frequency_stepped_schedule(stimulus_info, sample_rate)
    kwargs = _frequency_stepped_generator_kwargs(schedule.metadata)
    result = generate_frequency_stepped(**kwargs)
    metadata = _merge_frequency_stepped_metadata(stimulus_info, result.metadata)
    result.metadata.clear()
    result.metadata.update(metadata)
    return result


def _generate_stimulus_data(detail: dict, runtime_sample_rate: int, logger=None):
    sample_rate = _runtime_sample_rate_value(runtime_sample_rate)
    stimulus_info = _runtime_stimulus_info(detail, sample_rate)
    return _generate_stimulus_data_at_rate(detail, stimulus_info, sample_rate, logger=logger)


def _generate_stimulus_data_at_rate(detail: dict, stimulus_info: dict, sample_rate: int, logger=None):
    """Generate with the runtime rate and method already resolved at the caller boundary."""
    logger = _get_logger(logger)
    method = stimulus_info["stimulus_method"]
    total_time = float(stimulus_info.get("total_time") or (detail or {}).get("total_time") or 0.0)

    stimulus_data = None
    try:
        if method == FREQUENCY_STEPPED_METHOD:
            result = _generate_frequency_stepped_runtime(stimulus_info, sample_rate)
            stimulus_data = result.data
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
            create_fn = create_function_dict[method]
            stimulus_data, _sr = create_fn(**stimulus_info)
            if _sr:
                sample_rate = int(_sr)
            detail["stimulus_info"] = stimulus_info
        if stimulus_data is None:
            raise ValueError("Generated stimulus_data is None")
        stimulus_data = np.asarray(stimulus_data, dtype="float32")
    except Exception as e:
        logger.error(f"Failed to generate stimulus from stimulus_info: {e}")
        if method == FREQUENCY_STEPPED_METHOD:
            raise ValueError(f"Failed to generate frequency_stepped stimulus: {e}") from e
        stimulus_data = None

    if stimulus_data is None or stimulus_data.size == 0:
        num_samples = int(max(total_time, 0.0) * sample_rate)
        stimulus_data = np.zeros(max(num_samples, 1), dtype="float32")

    return stimulus_data, sample_rate, stimulus_info, method


def generate_and_save_stimulus(detail: dict, *, runtime_sample_rate: int, logger=None):
    """
    Generate stimulus from detail["stimulus_info"], save to STORED_STIMULUS_PATH,
    and update detail["stimulus_signal_path"] (relative to DEFAULT_DIR).

    Returns (stimulus_data, sample_rate, saved_abs_path) or (None, sr, None) on failure.
    """
    sample_rate = _runtime_sample_rate_value(runtime_sample_rate)
    stimulus_info = _runtime_stimulus_info(detail, sample_rate)
    return _generate_and_save_stimulus_at_rate(detail, stimulus_info, sample_rate, logger=logger)


def _generate_and_save_stimulus_at_rate(detail: dict, stimulus_info: dict, sample_rate: int, logger=None):
    logger = _get_logger(logger)
    stimulus_data, sample_rate, stimulus_info, _method = _generate_stimulus_data_at_rate(
        detail, stimulus_info, sample_rate, logger=logger,
    )

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


def set_data_struct_stimulus_signal(
    data_struct,
    detail,
    using_config_path: str = None,
    *,
    runtime_sample_rate: int,
    logger=None,
) -> bool:
    """
    Safe variant: if stimulus wav is missing, regenerate from stimulus_info, save it,
    and update detail['stimulus_signal_path'] so caller can write back config.

    Returns:
        bool: whether detail was modified (path fixed/regenerated)
    """
    logger = _get_logger(logger)
    if detail is None:
        return False

    sample_rate = _runtime_sample_rate_value(runtime_sample_rate)
    stimulus_info = _runtime_stimulus_info(detail, sample_rate)
    method = stimulus_info["stimulus_method"]

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
        stimulus_signal, sample_rate, saved_path = _generate_and_save_stimulus_at_rate(
            detail,
            stimulus_info,
            sample_rate,
            logger=logger,
        )
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


def set_data_struct_analysis_reference_signal(
    data_struct,
    detail,
    using_config_path: str = None,
    *,
    runtime_sample_rate: int,
    logger=None,
) -> bool:
    logger = _get_logger(logger)
    if detail is None:
        return False

    sample_rate = _runtime_sample_rate_value(runtime_sample_rate)
    runtime_detail = copy.deepcopy(detail)
    stimulus_info = _runtime_stimulus_info(runtime_detail, sample_rate)
    method = stimulus_info["stimulus_method"]

    config_dir = None
    if using_config_path:
        try:
            config_dir = os.path.dirname(using_config_path)
        except Exception:
            config_dir = None
    base_dirs = [DEFAULT_DIR, config_dir]
    configured_external_wav = bool((runtime_detail or {}).get("load_stimulus_signal_path"))

    if method == FREQUENCY_STEPPED_METHOD:
        stimulus_signal = None
    elif configured_external_wav:
        stimulus_signal, _used_path = _try_load_existing_wav(
            runtime_detail,
            sample_rate,
            base_dirs,
            logger=logger,
            keys=("load_stimulus_signal_path",),
        )
    else:
        stimulus_signal = None

    if stimulus_signal is None and configured_external_wav and method != FREQUENCY_STEPPED_METHOD:
        return False

    if stimulus_signal is not None and configured_external_wav and method != FREQUENCY_STEPPED_METHOD:
        _sync_loaded_wav_runtime_metadata(stimulus_info, stimulus_signal, sample_rate)

    if stimulus_signal is None:
        stimulus_signal, sample_rate, stimulus_info, method = _generate_stimulus_data_at_rate(
            runtime_detail,
            stimulus_info,
            sample_rate,
            logger=logger,
        )

    if hasattr(data_struct, "alignment_sample_count"):
        delattr(data_struct, "alignment_sample_count")
    current_stimulus_info = runtime_detail.get("stimulus_info") or stimulus_info
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
    return True


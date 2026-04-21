"""Global recording-pipeline settings + recording-validity gate.

Owns ``configs/recording_settings.json`` (defaults shared across all
product configs) and :func:`validate_recorded_audio` (the gate that
rejects silent / stuck / unplugged recordings before they reach AI).

Three-layer precedence, lowest first:
1. ``_HARDCODE_DEFAULTS`` -- safety net when the JSON is missing/corrupt.
2. ``configs/recording_settings.json`` -- global, edited once.
3. ``seqN.acq.detail`` -- per-product override of any single field.
   ``startup_trim_ms: 0`` is honoured as an explicit opt-out.

``audio_validation`` fields:
* ``min_rms_dbfs`` -- "no signal" detector (device off, mic unplugged).
  Set 6~10 dB below the lowest RMS observed on known-good samples.
* ``min_peak`` -- complement to RMS; catches "almost dead" recordings
  where a single edge spike would otherwise pass.
* ``min_variance`` -- "stuck data line" detector (constant / DC buffer).
"""

import json
import math
import os
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np

from consts import model_consts


_HARDCODE_DEFAULTS: Dict[str, Any] = {
    "startup_trim_ms": 0,
    "monitor_fade_in_ms": 1.0,
    "audio_validation": {
        "enabled": False,
        "min_rms_dbfs": -65.0,
        "min_peak": 1e-4,
        "min_variance": 1e-10,
    },
}


# Tests monkeypatch this and call :func:`reset_cache` to drive a temp file.
_GLOBAL_SETTINGS_PATH = (
    model_consts.DEFAULT_DIR + "configs/recording_settings.json"
)


_cache_lock = threading.Lock()
_cached_settings: Optional[Dict[str, Any]] = None
_cached_path: Optional[str] = None


# Re-exported under the old ``base.audio_quality`` name so existing
# imports keep working after the rename.
AUDIO_VALIDATION_FALLBACK = dict(_HARDCODE_DEFAULTS["audio_validation"])


def reset_cache() -> None:
    """Drop the cached settings; next call re-parses the JSON file."""
    global _cached_settings, _cached_path
    with _cache_lock:
        _cached_settings = None
        _cached_path = None


def _coerce_audio_validation(raw: Any) -> Dict[str, Any]:
    """Drop unknown / ``None`` keys; return a complete validation block."""
    base = dict(_HARDCODE_DEFAULTS["audio_validation"])
    if not isinstance(raw, dict):
        return base
    for key in base:
        if key in raw and raw[key] is not None:
            base[key] = raw[key]
    return base


def _read_settings_file(path: str) -> Dict[str, Any]:
    """Read the JSON; any error path falls back to ``_HARDCODE_DEFAULTS``."""
    merged: Dict[str, Any] = {
        "startup_trim_ms": _HARDCODE_DEFAULTS["startup_trim_ms"],
        "monitor_fade_in_ms": _HARDCODE_DEFAULTS["monitor_fade_in_ms"],
        "audio_validation": dict(_HARDCODE_DEFAULTS["audio_validation"]),
    }
    if not path or not os.path.isfile(path):
        return merged
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return merged
    if not isinstance(raw, dict):
        return merged

    if "startup_trim_ms" in raw and raw["startup_trim_ms"] is not None:
        merged["startup_trim_ms"] = raw["startup_trim_ms"]
    if "monitor_fade_in_ms" in raw and raw["monitor_fade_in_ms"] is not None:
        merged["monitor_fade_in_ms"] = raw["monitor_fade_in_ms"]
    merged["audio_validation"] = _coerce_audio_validation(
        raw.get("audio_validation")
    )
    return merged


def get_global_settings() -> Dict[str, Any]:
    """Return a defensive copy of the resolved global settings (cached)."""
    global _cached_settings, _cached_path
    with _cache_lock:
        if _cached_settings is None or _cached_path != _GLOBAL_SETTINGS_PATH:
            _cached_settings = _read_settings_file(_GLOBAL_SETTINGS_PATH)
            _cached_path = _GLOBAL_SETTINGS_PATH
        out = dict(_cached_settings)
        out["audio_validation"] = dict(_cached_settings["audio_validation"])
        return out


def _coerce_positive_ms(raw: Any) -> Optional[float]:
    """Return a finite ``>= 0`` float, or ``None`` for any malformed input.

    Booleans are treated as malformed -- almost always a config-author
    typo (``True`` would otherwise silently mean "1 ms").
    """
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def resolve_startup_trim_ms(acq_detail: Any) -> float:
    """Resolve ``startup_trim_ms`` (per-product > global > hardcode).

    ``0`` is honoured at every layer as an explicit "do not trim".
    A malformed per-product value falls through to the global default
    rather than collapsing to 0, so a typo cannot silently disable
    the trim everyone else relies on.
    """
    if isinstance(acq_detail, dict) and "startup_trim_ms" in acq_detail:
        local = _coerce_positive_ms(acq_detail.get("startup_trim_ms"))
        if local is not None:
            return local
    global_value = _coerce_positive_ms(
        get_global_settings().get("startup_trim_ms")
    )
    if global_value is not None:
        return global_value
    return float(_HARDCODE_DEFAULTS["startup_trim_ms"])


def resolve_monitor_fade_in_ms(acq_detail: Any) -> float:
    """Resolve ``monitor_fade_in_ms`` (same precedence as startup trim).

    ``0`` means "hard cut, no fade" -- accepts whatever click the
    hardware produces; only useful for diagnostics.
    """
    if isinstance(acq_detail, dict) and "monitor_fade_in_ms" in acq_detail:
        local = _coerce_positive_ms(acq_detail.get("monitor_fade_in_ms"))
        if local is not None:
            return local
    global_value = _coerce_positive_ms(
        get_global_settings().get("monitor_fade_in_ms")
    )
    if global_value is not None:
        return global_value
    return float(_HARDCODE_DEFAULTS["monitor_fade_in_ms"])


def resolve_audio_validation_thresholds(acq_detail: Any) -> Dict[str, Any]:
    """Resolve the ``audio_validation`` block (3-layer field-by-field merge).

    Unknown override keys are dropped; ``None`` at any layer is treated
    as "not specified" so the next layer wins. ``enabled`` follows the
    same rules so a product config can both enable and disable the
    gate independently of the global setting.
    """
    merged = dict(_HARDCODE_DEFAULTS["audio_validation"])

    global_block = get_global_settings().get("audio_validation")
    if isinstance(global_block, dict):
        for key in merged:
            if key in global_block and global_block[key] is not None:
                merged[key] = global_block[key]

    if isinstance(acq_detail, dict):
        local_block = acq_detail.get("audio_validation")
        if isinstance(local_block, dict):
            for key in merged:
                if key in local_block and local_block[key] is not None:
                    merged[key] = local_block[key]

    return merged


def merge_audio_validation_thresholds(acq_detail) -> dict:
    """Backwards-compatible alias for :func:`resolve_audio_validation_thresholds`."""
    return resolve_audio_validation_thresholds(acq_detail)


def _to_mono_float(samples) -> np.ndarray:
    arr = np.asarray(samples)
    if arr.size == 0:
        return arr.astype(np.float32, copy=False)
    if arr.ndim > 1:
        # Average across channels: a single dead channel must not fail
        # the whole recording, but if every channel is dead the mean
        # is also dead.
        arr = arr.mean(axis=1)
    return arr.astype(np.float32, copy=False)


def validate_recorded_audio(
    samples, thresholds: dict
) -> Tuple[bool, str, str]:
    """Return ``(ok, reason, detail)`` for the recorded audio buffer.

    * ``reason`` -- short user-facing message (shown in the operator
      dialog; no numbers so the dialog stays compact).
    * ``detail`` -- one-line diagnostic with measured values and
      configured thresholds, intended for the log so the offline
      analyst can still tell a hardware fault from an over-tight
      threshold. Empty string when ``ok`` is ``True``.
    """
    if not isinstance(thresholds, dict):
        thresholds = AUDIO_VALIDATION_FALLBACK
    if not thresholds.get("enabled", True):
        return True, "", ""

    mono = _to_mono_float(samples)
    if mono.size == 0:
        return False, "录音缓冲区为空，未采集到任何样本。", "samples=0"

    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    variance = float(np.var(mono, dtype=np.float64))

    min_peak = float(thresholds.get("min_peak", AUDIO_VALIDATION_FALLBACK["min_peak"]))
    min_rms_dbfs = float(thresholds.get("min_rms_dbfs", AUDIO_VALIDATION_FALLBACK["min_rms_dbfs"]))
    min_variance = float(thresholds.get("min_variance", AUDIO_VALIDATION_FALLBACK["min_variance"]))

    rms_dbfs = -math.inf if rms <= 0 else 20.0 * math.log10(rms)
    rms_text = "-inf" if math.isinf(rms_dbfs) else f"{rms_dbfs:.1f}"

    if peak < min_peak or rms_dbfs < min_rms_dbfs:
        detail = (
            f"weak_signal rms={rms_text}dBFS peak={peak:.4f} "
            f"threshold_rms>={min_rms_dbfs:.1f}dBFS "
            f"threshold_peak>={min_peak:.4f}"
        )
        return False, "录音信号过弱，疑似未通电或采集链路异常。", detail

    if variance < min_variance:
        detail = (
            f"flat_signal variance={variance:.3e} "
            f"threshold_variance>={min_variance:.3e}"
        )
        return (
            False,
            "录音几乎为常数，疑似数据线路断开或采集卡未输出有效数据。",
            detail,
        )

    return True, "", ""

"""Sound-quality service layer.

This module is the single entry point used by the test queue and other
project pipelines. It does not own audio capture or microphone calibration:
callers must pass an already-recorded signal, its sample rate, and the
project's effective ``v2pa_factor``.

Configuration is read from the ``SQ`` node of the project analysis-default
config (``ui/ui_config/analysis_default_config.json``). Missing or partial
configs are tolerated through ``.get(..., default)`` chains so that the
service stays compatible with both the shipped defaults and any user-edited
config.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np

from base.core_algorithm.sound_quality.psychoacoustic_constants import (
    LOUDNESS_DEFAULT_OUTPUT_TIME_RESOLUTION_S,
    LOUDNESS_TV_SUMMARY_SKIP_S,
)

from base.core_algorithm.sound_quality.loudness import (
    LoudnessAnalyzer,
    LoudnessMethod,
    LoudnessResult,
    sones_to_phons,
)


SUPPORTED_SUMMARY_METRICS = (
    "specific_loudness_sum_sone",
    "specific_loudness_summed_exceedance",
    "steady_state_average_sone",
    "steady_state_average_phon",
    "max_transient_sone",
    "max_transient_phon",
    "nmax_sone",
    "lnmax_phon",
    "mean_sone",
    "mean_phon",
    "mean_loudness",
)
SUPPORTED_CURVES = ("loudness_time", "specific_loudness_profile")
SUPPORTED_HEATMAPS = ("specific_loudness",)


@dataclass
class LoudnessRunResult:
    """Per-algorithm result wrapper for the time-varying loudness step."""

    enabled: bool
    skipped_reason: Optional[str]
    raw_result: Optional[LoudnessResult]
    summary: Dict[str, float]
    display_payload: Dict[str, object]
    save_payload: Dict[str, object]


@dataclass
class SoundQualityRunResult:
    """Top-level result of one ``run_sound_quality`` call."""

    enabled: bool
    skipped_reason: Optional[str] = None
    field_type: str = "free"
    loudness: Optional[LoudnessRunResult] = None
    pending_algorithms: List[str] = field(default_factory=list)


def run_sound_quality(
    signal_v: np.ndarray,
    sample_rate: int,
    project_v2pa_factor: float,
    sq_config: Optional[dict],
) -> SoundQualityRunResult:
    """Run the configured sound-quality algorithms on one recorded signal.

    Parameters
    ----------
    signal_v:
        Mono recorded signal in volts (or whatever raw unit the recording
        module produces). It will be multiplied by ``project_v2pa_factor``
        inside the loudness algorithm to obtain Pa.
    sample_rate:
        Recording sample rate in Hz.
    project_v2pa_factor:
        Microphone calibration multiplier from the project calibration
        module (V -> Pa). Must be > 0 for SQ to run.
    sq_config:
        The ``SQ`` node from ``analysis_default_config.json``. ``None`` or
        ``{}`` is treated as "SQ disabled".
    """

    cfg = sq_config or {}
    if not cfg.get("enabled", False):
        return SoundQualityRunResult(enabled=False, skipped_reason="SQ.enabled is false")

    # Keep the application-facing loudness path on a single, predictable
    # reference. Free/diffuse field selection is a psychoacoustic correction,
    # not a microphone calibrator setting, and exposing it caused confusing
    # calibration-source comparisons.
    field_type = "free"

    items = cfg.get("items", {}) or {}
    pending = [
        key for key in ("SHRP", "ROUGH", "FLUC", "TON", "PR", "TNR")
        if items.get(key, {}).get("enabled", False)
    ]

    loud_cfg = items.get("LOUD", {}) or {}
    loud_result = _run_loudness(
        signal_v=signal_v,
        sample_rate=sample_rate,
        project_v2pa_factor=project_v2pa_factor,
        field_type=field_type,
        loud_cfg=loud_cfg,
    )

    return SoundQualityRunResult(
        enabled=True,
        field_type=field_type,
        loudness=loud_result,
        pending_algorithms=pending,
    )


def _run_loudness(
    signal_v: np.ndarray,
    sample_rate: int,
    project_v2pa_factor: float,
    field_type: str,
    loud_cfg: dict,
) -> LoudnessRunResult:
    if not loud_cfg.get("enabled", False):
        return _empty_loudness("LOUD.enabled is false")

    try:
        v2pa = float(project_v2pa_factor)
    except (TypeError, ValueError):
        return _empty_loudness("project_v2pa_factor is not numeric")
    if not np.isfinite(v2pa) or v2pa <= 0.0:
        return _empty_loudness("project_v2pa_factor must be > 0")

    if not isinstance(signal_v, np.ndarray) or signal_v.size == 0:
        return _empty_loudness("recorded signal is empty")
    try:
        sample_rate_int = int(sample_rate)
    except (TypeError, ValueError):
        return _empty_loudness("sample_rate must be an int")
    if sample_rate_int <= 0:
        return _empty_loudness("sample_rate must be > 0")

    method_value = str(loud_cfg.get("method", LoudnessMethod.TIME_VARYING_ISO532_1.value))
    try:
        method = LoudnessMethod(method_value)
    except ValueError:
        method = LoudnessMethod.TIME_VARYING_ISO532_1
    advanced_cfg = loud_cfg.get("advanced", {}) or {}
    output_time_resolution_ms = _positive_float_option(advanced_cfg, "output_time_resolution_ms")
    output_time_resolution_s = (
        output_time_resolution_ms / 1000.0
        if output_time_resolution_ms is not None
        else LOUDNESS_DEFAULT_OUTPUT_TIME_RESOLUTION_S
    )
    stationary_frame_duration_s = _positive_float_option(
        advanced_cfg,
        "stationary_frame_duration_s",
        fallback_key="frame_duration_s",
    )
    stationary_hop_duration_s = _stationary_hop_duration_option(advanced_cfg, stationary_frame_duration_s)
    analysis_signal = signal_v.astype(np.float64, copy=False)
    analysis_signal, source_start_time_s, source_end_time_s = _apply_loudness_time_range(
        analysis_signal,
        sample_rate_int,
        advanced_cfg,
    )
    analysis_signal = _apply_loudness_noise_reduction(analysis_signal, sample_rate_int, advanced_cfg)

    try:
        raw = LoudnessAnalyzer(sample_rate_int).compute(
            analysis_signal,
            method=method,
            field_type=field_type,
            v2pa_factor=v2pa,
            frame_duration_s=stationary_frame_duration_s,
            hop_duration_s=stationary_hop_duration_s,
            output_time_resolution_s=output_time_resolution_s,
        )
    except (RuntimeError, ValueError, TypeError, FloatingPointError) as exc:
        return _empty_loudness(f"loudness computation failed: {exc}")

    display_cfg = loud_cfg.get("display", {}) or {}
    save_cfg = loud_cfg.get("save", {}) or {}
    requested_metrics = _requested_loudness_summary_metrics(display_cfg, save_cfg, advanced_cfg)
    summary = _build_loudness_summary(raw, advanced_cfg, requested_metrics=requested_metrics)

    analysis_raw = _with_analysis_time_range_metadata(raw, source_start_time_s, source_end_time_s)
    display_time_s = _source_time_axis(analysis_raw)
    display_payload = _build_loudness_display_payload(
        analysis_raw,
        summary,
        display_cfg,
        advanced_cfg,
        time_s=display_time_s,
    )
    save_payload = _build_loudness_save_payload(
        analysis_raw,
        summary,
        save_cfg,
        advanced_cfg,
        time_s=display_time_s,
    )

    return LoudnessRunResult(
        enabled=True,
        skipped_reason=None,
        raw_result=analysis_raw,
        summary=summary,
        display_payload=display_payload,
        save_payload=save_payload,
    )


def _empty_loudness(reason: str) -> LoudnessRunResult:
    return LoudnessRunResult(
        enabled=False,
        skipped_reason=reason,
        raw_result=None,
        summary={},
        display_payload={"summary_cards": [], "curves": [], "heatmaps": []},
        save_payload={"summary": None, "curve": None, "specific_loudness": None},
    )


def _positive_float_option(config: dict, key: str, *, fallback_key: Optional[str] = None) -> Optional[float]:
    value = config.get(key)
    if value is None and fallback_key is not None:
        value = config.get(fallback_key)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or numeric <= 0.0:
        return None
    return numeric


def _finite_float_option(config: dict, key: str) -> Optional[float]:
    value = config.get(key)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _stationary_hop_duration_option(config: dict, frame_duration_s: Optional[float]) -> Optional[float]:
    overlap_percent = _finite_float_option(config, "stationary_overlap_percent")
    if overlap_percent is not None and frame_duration_s is not None:
        overlap_ratio = min(max(overlap_percent, 0.0), 90.0) / 100.0
        return max(frame_duration_s * (1.0 - overlap_ratio), 0.001)
    return _positive_float_option(config, "stationary_hop_duration_s", fallback_key="hop_duration_s")


def _apply_loudness_time_range(
    signal: np.ndarray,
    sample_rate: int,
    advanced_cfg: dict,
) -> tuple[np.ndarray, float, Optional[float]]:
    """Slice the signal to the user-configured analysis window."""
    if not advanced_cfg.get("analysis_time_range_enabled", False):
        return signal, 0.0, None
    start_sec = max(0.0, float(advanced_cfg.get("analysis_start_time_sec", 0.0) or 0.0))
    end_sec = max(0.0, float(advanced_cfg.get("analysis_end_time_sec", 0.0) or 0.0))
    start_sample = min(int(np.floor(start_sec * sample_rate)), signal.size)
    end_sample = min(int(np.ceil(end_sec * sample_rate)), signal.size) if end_sec > 0.0 else signal.size
    if end_sample <= start_sample:
        return signal, 0.0, None
    return (
        signal[start_sample:end_sample],
        start_sample / float(sample_rate),
        end_sample / float(sample_rate),
    )


def _with_analysis_time_range_metadata(
    raw: LoudnessResult,
    source_start_time_s: float,
    source_end_time_s: Optional[float],
) -> LoudnessResult:
    """Attach source-window metadata without changing segment-relative time."""
    if source_end_time_s is None:
        return raw

    metadata = dict(raw.metadata or {})
    source_start = float(source_start_time_s)
    source_end = float(source_end_time_s)
    metadata.update(
        {
            "analysis_time_range_enabled": True,
            "analysis_source_start_s": source_start,
            "analysis_source_end_s": source_end,
            "analysis_duration_s": max(0.0, source_end - source_start),
        }
    )
    return replace(raw, metadata=metadata)


def _source_time_axis(raw: LoudnessResult) -> np.ndarray:
    """Return the time axis to expose in UI/export payloads."""
    time_s = np.asarray(raw.time_s, dtype=np.float64)
    metadata = dict(raw.metadata or {})
    if not metadata.get("analysis_time_range_enabled", False):
        return time_s
    try:
        source_start_s = float(metadata.get("analysis_source_start_s", 0.0) or 0.0)
    except (TypeError, ValueError):
        return time_s
    if not np.isfinite(source_start_s) or source_start_s <= 0.0:
        return time_s
    return time_s + source_start_s


def _apply_loudness_noise_reduction(
    signal: np.ndarray,
    sample_rate: int,
    advanced_cfg: dict,
) -> np.ndarray:
    """Apply spectral subtraction if a noise reference file is configured."""
    if not advanced_cfg.get("background_noise_processing_enabled", False):
        return signal
    noise_path = str(advanced_cfg.get("background_noise_file_path", "") or "").strip()
    if not noise_path:
        return signal

    import os
    if not os.path.isfile(noise_path):
        return signal

    try:
        import soundfile as sf
        noise_data, sr_noise = sf.read(noise_path, dtype="float64", always_2d=True)
        if noise_data.ndim == 2:
            noise_data = noise_data[:, 0]
        else:
            noise_data = np.asarray(noise_data, dtype=np.float64)
        if sr_noise != sample_rate:
            from scipy.signal import resample
            target_len = int(round(noise_data.size * sample_rate / float(sr_noise)))
            if target_len <= 0:
                return signal
            noise_data = resample(noise_data, target_len)
    except Exception:
        return signal

    try:
        from .noise_reduction import spectral_subtract_audio
        result = spectral_subtract_audio(
            signal,
            noise_data,
            n_fft=int(advanced_cfg.get("background_noise_n_fft", 4096) or 4096),
            hop_size=int(advanced_cfg.get("background_noise_hop_size", 1024) or 1024),
            alpha=float(advanced_cfg.get("background_noise_oversubtraction_factor", 1.0) or 1.0),
            spectral_floor=float(advanced_cfg.get("background_noise_spectral_floor", 0.02) or 0.02),
            min_gain_db=float(advanced_cfg.get("background_noise_min_gain_db", -20.0) or -20.0),
            frequency_smoothing_bins=int(
                advanced_cfg.get("background_noise_frequency_smoothing_bins", 3) or 3
            ),
            gain_time_smoothing=float(
                advanced_cfg.get("background_noise_gain_time_smoothing", 0.6) or 0.6
            ),
        )
        return result.signal
    except Exception:
        return signal


def _requested_loudness_summary_metrics(display_cfg: dict, save_cfg: dict, advanced_cfg: dict) -> set[str]:
    configured = list(display_cfg.get("summary_metrics", []) or [])
    if save_cfg.get("summary", False):
        configured.extend(save_cfg.get("summary_metrics", []) or [])
    return {
        _resolve_loudness_summary_metric_key(key, advanced_cfg)
        for key in configured
        if isinstance(key, str)
    }


def _build_loudness_summary(
    raw: LoudnessResult,
    advanced_cfg: Optional[dict] = None,
    requested_metrics: Optional[set[str]] = None,
) -> Dict[str, float]:
    loudness = np.asarray(raw.loudness_sone, dtype=np.float64)
    if loudness.size == 0:
        return {}
    cfg = advanced_cfg or {}
    requested = set(requested_metrics) if requested_metrics is not None else {
        "specific_loudness_summed_exceedance",
        "max_transient_sone",
        "max_transient_phon",
        "mean_sone",
        "mean_phon",
        "nmax_sone",
        "lnmax_phon",
    }

    loudness_phon = np.asarray(raw.loudness_level_phon, dtype=np.float64)

    # Skip initial IIR filterbank warm-up (applies to both TIME_VARYING and
    # PER_SEGMENT — PER_SEGMENT internally reuses the time-varying frontend,
    # so a short first-segment is also contaminated by the startup transient).
    time_s_arr = np.asarray(raw.time_s, dtype=np.float64).reshape(-1)
    if time_s_arr.size == loudness.size and time_s_arr.size > 1:
        skip_idx = int(np.searchsorted(time_s_arr, LOUDNESS_TV_SUMMARY_SKIP_S, side="left"))
        skip_idx = min(skip_idx, loudness.size - 1)
        if skip_idx > 0:
            loudness = loudness[skip_idx:]
            if loudness_phon.size == time_s_arr.size:
                loudness_phon = loudness_phon[skip_idx:]

    summary: Dict[str, float] = {}

    need_max = bool({"max_transient_sone", "max_transient_phon", "nmax_sone", "lnmax_phon"} & requested)
    if need_max:
        nmax = float(np.max(loudness))
        nmax_phon = float(sones_to_phons(np.asarray([nmax], dtype=np.float64))[0])
        if "max_transient_sone" in requested:
            summary["max_transient_sone"] = nmax
        if "max_transient_phon" in requested:
            summary["max_transient_phon"] = nmax_phon
        if "nmax_sone" in requested:
            summary["nmax_sone"] = nmax
        if "lnmax_phon" in requested:
            summary["lnmax_phon"] = nmax_phon

    if "mean_sone" in requested or "steady_state_average_sone" in requested:
        mean_sone_val = float(np.mean(loudness))
        if "mean_sone" in requested:
            summary["mean_sone"] = mean_sone_val
        if "steady_state_average_sone" in requested:
            summary["steady_state_average_sone"] = mean_sone_val
    if "mean_phon" in requested or "steady_state_average_phon" in requested:
        mean_phon_val = float(np.mean(loudness_phon)) if loudness_phon.size else float("nan")
        if "mean_phon" in requested:
            summary["mean_phon"] = mean_phon_val
        if "steady_state_average_phon" in requested:
            summary["steady_state_average_phon"] = mean_phon_val

    if "specific_loudness_summed_exceedance" in requested:
        summary["specific_loudness_summed_exceedance"] = _specific_loudness_summed_exceedance(raw, cfg)
    if "specific_loudness_sum_sone" in requested:
        summary["specific_loudness_sum_sone"] = _specific_loudness_sum(raw, cfg)

    return summary


def _specific_loudness_matrix(raw: LoudnessResult) -> tuple[np.ndarray, np.ndarray]:
    specific = np.asarray(raw.specific_loudness, dtype=np.float64)
    bark_axis = np.asarray(raw.bark_axis, dtype=np.float64).reshape(-1)
    if specific.size == 0 or bark_axis.size == 0:
        return np.empty((0, 0), dtype=np.float64), bark_axis
    if specific.ndim == 1:
        specific = specific.reshape(-1, 1)
    if specific.ndim != 2:
        return np.empty((0, 0), dtype=np.float64), bark_axis
    if specific.shape[0] != bark_axis.size and specific.shape[1] == bark_axis.size:
        specific = specific.T
    if specific.shape[0] != bark_axis.size:
        return np.empty((0, 0), dtype=np.float64), bark_axis
    return specific, bark_axis


def _specific_loudness_keep_mask(raw: LoudnessResult, frame_count: int, advanced_cfg: dict) -> Optional[np.ndarray]:
    time_s = np.asarray(raw.time_s, dtype=np.float64).reshape(-1)
    if time_s.size != frame_count:
        return None
    skip_s = _finite_float_option(advanced_cfg, "specific_loudness_time_skip_s")
    if skip_s is None:
        skip_s = LOUDNESS_TV_SUMMARY_SKIP_S
    if skip_s <= 0.0:
        return np.ones(frame_count, dtype=bool)
    keep = time_s >= float(skip_s)
    return keep if np.any(keep) else np.ones(frame_count, dtype=bool)


def _specific_loudness_profile(raw: LoudnessResult, advanced_cfg: dict) -> tuple[np.ndarray, np.ndarray, str]:
    specific, bark_axis = _specific_loudness_matrix(raw)
    if specific.size == 0 or bark_axis.size == 0:
        return np.asarray([], dtype=np.float64), bark_axis, "steady_average"

    mode = str(advanced_cfg.get("specific_loudness_profile_mode", "steady_average") or "steady_average").lower()
    keep = _specific_loudness_keep_mask(raw, specific.shape[1], advanced_cfg)
    specific_for_stats = specific[:, keep] if keep is not None else specific

    if mode in {"max_loudness", "max_transient"}:
        loudness = np.asarray(raw.loudness_sone, dtype=np.float64).reshape(-1)
        if keep is not None and loudness.size == keep.size:
            candidate_indexes = np.flatnonzero(keep)
        else:
            candidate_indexes = np.arange(specific.shape[1], dtype=np.int64)
        if loudness.size == specific.shape[1] and candidate_indexes.size:
            idx = int(candidate_indexes[int(np.argmax(loudness[candidate_indexes]))])
        else:
            frame_power = np.sum(np.maximum(specific_for_stats, 0.0), axis=0)
            idx_local = int(np.argmax(frame_power)) if frame_power.size else 0
            idx = int(np.flatnonzero(keep)[idx_local]) if keep is not None and np.any(keep) else idx_local
        return specific[:, idx].astype(np.float64, copy=False), bark_axis, "max_loudness"

    return np.mean(specific_for_stats, axis=1), bark_axis, "steady_average"


def _bark_step(bark_axis: np.ndarray) -> float:
    if bark_axis.size > 1:
        dz = float(np.median(np.diff(bark_axis)))
        if np.isfinite(dz) and dz > 0.0:
            return dz
    return 0.1


def _specific_loudness_sum(raw: LoudnessResult, advanced_cfg: dict) -> float:
    profile, bark_axis, _mode = _specific_loudness_profile(raw, advanced_cfg)
    if profile.size == 0 or bark_axis.size == 0:
        return float("nan")
    return float(np.sum(np.maximum(profile, 0.0)) * _bark_step(bark_axis))


def interpolate_ref_line(bark_axis: np.ndarray, ref_breakpoints: list) -> np.ndarray:
    """Interpolate a SSTS reference line to the given Bark axis."""
    ref_bark = np.array([p[0] for p in ref_breakpoints], dtype=np.float64)
    ref_value = np.array([p[1] for p in ref_breakpoints], dtype=np.float64)
    return np.interp(bark_axis, ref_bark, ref_value)


def _specific_loudness_summed_exceedance(raw: LoudnessResult, advanced_cfg: dict) -> float:
    """Summed exceedance of the selected specific-loudness profile.

    Supports two modes:
      1. Frequency-dependent reference line (SSTS standard):
         Set "specific_loudness_exceedance_ref_line" to "ref1", "ref2", "ref3", or "ref4".
         Computes sum(max(N'(z) - Ref(z), 0) * dz) in sone.

      2. Fixed threshold (legacy):
         Set "specific_loudness_exceedance_threshold_sone_per_bark" to a scalar value.
         Computes sum(max(N'(z) - T, 0) * dz) in sone.

    If both are specified, the reference line takes priority.
    """
    from .psychoacoustic_constants import SSTS_SPECIFIC_LOUDNESS_REF_LINES

    profile, bark_axis, _mode = _specific_loudness_profile(raw, advanced_cfg)
    if profile.size == 0 or bark_axis.size == 0:
        return float("nan")

    # Mode 1: frequency-dependent reference line
    ref_line_key = (advanced_cfg or {}).get("specific_loudness_exceedance_ref_line")
    if ref_line_key and str(ref_line_key).lower() in SSTS_SPECIFIC_LOUDNESS_REF_LINES:
        ref_breakpoints = SSTS_SPECIFIC_LOUDNESS_REF_LINES[str(ref_line_key).lower()]
        threshold_curve = interpolate_ref_line(bark_axis, ref_breakpoints)
        excess = np.maximum(profile - threshold_curve, 0.0)
        return float(np.sum(excess) * _bark_step(bark_axis))

    # Mode 2: fixed scalar threshold (legacy)
    threshold = _finite_float_option(advanced_cfg, "specific_loudness_exceedance_threshold_sone_per_bark")
    if threshold is None or threshold <= 0.0:
        return float("nan")
    threshold_value = float(threshold)
    excess = np.maximum(profile - threshold_value, 0.0)
    return float(np.sum(excess) * _bark_step(bark_axis))


def _build_loudness_display_payload(
    raw: LoudnessResult,
    summary: Dict[str, float],
    display_cfg: dict,
    advanced_cfg: dict,
    time_s: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    display_time_s = np.asarray(raw.time_s if time_s is None else time_s, dtype=np.float64)
    metric_keys = [
        _resolve_loudness_summary_metric_key(key, advanced_cfg)
        for key in (display_cfg.get("summary_metrics", []) or [])
    ]
    metric_order = {
        "steady_state_average_sone": 0,
        "steady_state_average_phon": 0,
        "max_transient_sone": 1,
        "max_transient_phon": 1,
        "specific_loudness_sum_sone": 2,
        "specific_loudness_summed_exceedance": 3,
    }
    metric_keys = sorted(
        metric_keys,
        key=lambda key: (metric_order.get(key, 100), metric_keys.index(key)),
    )
    summary_cards = [
        {"key": key, "value": summary.get(key), "label": _metric_label(key), "unit": _metric_unit(key)}
        for key in metric_keys
        if key in SUPPORTED_SUMMARY_METRICS
    ]

    curves: List[Dict[str, object]] = []
    for curve_key in display_cfg.get("curves", []) or []:
        if curve_key not in SUPPORTED_CURVES:
            continue
        if curve_key == "loudness_time":
            curves.append(
                {
                    "key": "loudness_time",
                    "x": display_time_s,
                    "y": np.asarray(raw.loudness_sone, dtype=np.float64),
                    "x_label": "time / s",
                    "y_label": "Loudness / sone",
                }
            )
        elif curve_key == "specific_loudness_profile":
            profile, bark_axis, profile_mode = _specific_loudness_profile(raw, advanced_cfg)
            if profile.size and bark_axis.size:
                curves.append(
                    {
                        "key": "specific_loudness_profile",
                        "x": bark_axis,
                        "y": profile,
                        "x_label": "Bark",
                        "y_label": "特征响度 / (sone/Bark)",
                        "profile_mode": profile_mode,
                    }
                )

    heatmaps: List[Dict[str, object]] = []
    show_heatmap = bool(advanced_cfg.get("show_specific_loudness_heatmap", False))
    if show_heatmap and "specific_loudness" in (display_cfg.get("heatmaps", []) or []):
        heatmaps.append(
            {
                "key": "specific_loudness",
                "x": display_time_s,
                "y": np.asarray(raw.bark_axis, dtype=np.float64),
                "z": np.asarray(raw.specific_loudness, dtype=np.float64),
                "x_label": "time / s",
                "y_label": "Bark",
                "z_label": "N' / (sone/Bark)",
            }
        )

    return {"summary_cards": summary_cards, "curves": curves, "heatmaps": heatmaps}


def _resolve_loudness_summary_metric_key(key: str, advanced_cfg: dict) -> str:
    aliases = {
        "specific_loudness_exceedance": "specific_loudness_summed_exceedance",
    }
    if key in aliases:
        return aliases[key]
    curve_y_unit = str((advanced_cfg or {}).get("curve_y_unit", "sone") or "sone").lower()
    if key == "steady_state_average_loudness":
        return "steady_state_average_phon" if curve_y_unit == "phon" else "steady_state_average_sone"
    if key == "max_transient_loudness":
        return "max_transient_phon" if curve_y_unit == "phon" else "max_transient_sone"
    if key not in {"mean_loudness", "mean_sone", "mean_phon"}:
        return key
    return "mean_phon" if curve_y_unit == "phon" else "mean_sone"


def _build_loudness_save_payload(
    raw: LoudnessResult,
    summary: Dict[str, float],
    save_cfg: dict,
    advanced_cfg: dict,
    time_s: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    payload: Dict[str, object] = {"summary": None, "curve": None, "specific_loudness": None}
    save_time_s = np.asarray(raw.time_s if time_s is None else time_s, dtype=np.float64)

    if save_cfg.get("summary", False):
        payload["summary"] = {
            "field_type": raw.metadata.get("field_type", "free"),
            "method": raw.metadata.get("method"),
            "sample_rate_input_hz": raw.metadata.get("sample_rate_input_hz"),
            "sample_rate_internal_hz": raw.metadata.get("sample_rate_internal_hz"),
            "summary_metrics": summary,
        }

    if save_cfg.get("curve", False):
        payload["curve"] = {
            "time_s": save_time_s,
            "loudness_sone": np.asarray(raw.loudness_sone, dtype=np.float64),
            "loudness_level_phon": np.asarray(raw.loudness_level_phon, dtype=np.float64),
        }

    save_specific = bool(save_cfg.get("specific_loudness", False)) or bool(advanced_cfg.get("save_specific_loudness_npz", False))
    if save_specific:
        payload["specific_loudness"] = {
            "time_s": save_time_s,
            "bark_axis": np.asarray(raw.bark_axis, dtype=np.float64),
            "specific_loudness": np.asarray(raw.specific_loudness, dtype=np.float64),
        }

    return payload


def _metric_label(key: str) -> str:
    return {
        "specific_loudness_sum_sone": "特征响度总贡献",
        "specific_loudness_summed_exceedance": "特征响度超限总量",
        "steady_state_average_sone": "稳态平均响度",
        "steady_state_average_phon": "稳态平均响度",
        "max_transient_sone": "最大瞬态响度",
        "max_transient_phon": "最大瞬态响度",
        "nmax_sone": "Nmax",
        "lnmax_phon": "LNmax",
        "mean_sone": "Mean Loudness",
        "mean_phon": "Mean LN",
    }.get(key, key)


def _metric_unit(key: str) -> str:
    return {
        "specific_loudness_sum_sone": "sone",
        "specific_loudness_summed_exceedance": "sone",
        "steady_state_average_sone": "sone",
        "steady_state_average_phon": "phon",
        "max_transient_sone": "sone",
        "max_transient_phon": "phon",
        "nmax_sone": "sone",
        "lnmax_phon": "phon",
        "mean_sone": "sone",
        "mean_phon": "phon",
    }.get(key, "")

from typing import Dict, List, Tuple, Any

import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths

from base.pre_processing.audio_equalizer import AudioEqualizer
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from base.utils.smooth import smooth as smooth_fn


def _parse_ranges(frequency_ranges_str: str) -> List[Tuple[float, float]]:
    if not frequency_ranges_str:
        return []
    parts = [p.strip() for p in frequency_ranges_str.split(";") if p.strip()]
    ranges: List[Tuple[float, float]] = []
    for p in parts:
        try:
            lo, hi = p.split(",")
            lo_f, hi_f = float(lo.strip()), float(hi.strip())
            if hi_f > lo_f:
                ranges.append((lo_f, hi_f))
        except Exception:
            continue
    return ranges

def _design_and_apply_filter(
    audio_signal: np.ndarray,
    sample_rate: int,
    enabled: bool,
    frequency_ranges_str: str,
    filter_type: str,
    filter_order: int,
) -> np.ndarray:
    """
    Parse frequency ranges and apply multi-band filtering using the equalizer.

    This function merges the previous range parsing + multi-band application and
    includes exception handling. On any error, the original signal is returned.
    """
    if not enabled:
        return audio_signal

    try:
        ranges = _parse_ranges(frequency_ranges_str)
        if not ranges:
            return audio_signal

        transition_width = 1.0 / max(1, int(filter_order))

        is_bandpass = (str(filter_type) == "bandpass")
        if is_bandpass:
            # Sum bandpass components across ranges
            return AudioEqualizer.apply_multi_band_equalizer(
                audio_signal=audio_signal,
                sample_rate=float(sample_rate),
                frequency_ranges=ranges,
                gains=1.0,
                gain_mode="linear",
                window_type=None,
                transition_width=transition_width,
                transition_type="cosine",
                mode="sum",
            )
        else:
            # Cascade bandstop across ranges (in-band gain=0)
            return AudioEqualizer.apply_multi_band_equalizer(
                audio_signal=audio_signal,
                sample_rate=float(sample_rate),
                frequency_ranges=ranges,
                gains=0.0,
                gain_mode="linear",
                window_type=None,
                transition_width=transition_width,
                transition_type="cosine",
                mode="cascade",
            )
    except Exception:
        return audio_signal


def _smooth_series(
    data: np.ndarray,
    enabled: bool,
    window_unit: str,
    window_time_sec: float,
    window_points: int,
    smoothing_algorithm_id: int,
    sample_rate: int,
) -> np.ndarray:
    if not enabled:
        return data

    if window_unit == "time":
        window_size_points = max(1, int(window_time_sec * sample_rate))
    else:
        window_size_points = max(1, int(window_points))

    if window_size_points % 2 == 0:
        window_size_points += 1

    if smoothing_algorithm_id == 1:
        kernel = np.ones(window_size_points, dtype=float) / float(window_size_points)
        return np.convolve(data, kernel, mode="same")
    elif smoothing_algorithm_id == 2:
        # Savitzky-Golay smoothing window must be odd
        polyorder = 3
        win = int(window_size_points)
        max_valid = int(len(data)) if len(data) % 2 == 1 else int(len(data)) - 1
        if max_valid < 1:
            return data
        win = min(win, max_valid)
        if win % 2 == 0:
            win = max(1, win - 1)
        if win <= polyorder:
            win = polyorder + 2  # make it at least 5
            if win % 2 == 0:
                win += 1
        # if it still exceeds the data length or is not enough to support SG, it will be degraded to mean smoothing
        if win > len(data) or win <= polyorder:
            k = max(1, min(len(data), 5))
            kernel = np.ones(k, dtype=float) / float(k)
            return np.convolve(data, kernel, mode="same")
        return smooth_fn(data, win, method="savgol")
    else:
        sigma = max(1, int(round(window_size_points / 6)))
        return smooth_fn(data, window_size_points, method="gaussian", sigma=sigma)


def _compute_rms_spl_series(
    audio_signal: np.ndarray,
    sample_rate: int,
    window_unit: str,
    window_time_sec: float,
    window_points: int,
    ref_pressure: float = 20e-6,
    v2pa_factor: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    if window_unit == "time":
        win = max(1, int(round(window_time_sec * sample_rate)))
    else:
        win = max(1, int(window_points))

    if win <= 1:
        rms_envelope = np.abs(audio_signal)
    else:
        # sliding RMS (kept for RMS envelope output consistency)
        pad = win // 2
        padded = np.pad(audio_signal, (pad, pad), mode="constant", constant_values=0)
        sq = padded ** 2
        kernel = np.ones(win, dtype=float) / float(win)
        mean_sq = np.convolve(sq, kernel, mode="same")[pad:-pad]
        rms_envelope = np.sqrt(np.maximum(mean_sq, 1e-30))

    spl_db_series = AudioThdFrequencyResponseAnalysis.spl_calculation(
        recorded_signal=audio_signal,
        reference_pressure=ref_pressure,
        window_size=win,
        method="rms",
        padding_mode="constant",
        padding_cval=0.0,
        v2pa_factor=v2pa_factor
    )
    return rms_envelope, np.asarray(spl_db_series, dtype=float)


def _normalize_rms(rms_envelope: np.ndarray) -> np.ndarray:
    max_val = float(np.max(rms_envelope)) if rms_envelope.size > 0 else 1.0
    if max_val <= 0:
        max_val = 1.0
    return rms_envelope / max_val


def _choose_distance(nms_enabled: bool, nms_unit: str, nms_time_sec: float, nms_points: int, sample_rate: int) -> int:
    # get the max points between peaks, for consistency of the processing
    if not nms_enabled:
        return 1
    if nms_unit == "time":
        return max(1, int(nms_time_sec * sample_rate))
    return max(1, int(nms_points))


def _duration_filter(
    db_series_smoothed: np.ndarray,
    rms_normalized_series_smoothed: np.ndarray,
    peaks: np.ndarray,
    sample_rate: int,
    enabled: bool,
    duration_reference_unit: str,
    duration_reference_value: float,
    min_duration_sec: float,
    max_duration_sec: float,
) -> Tuple[List[int], List[float]]:
    if not enabled or len(peaks) == 0:
        return list(peaks.astype(int)), []

    valid_indices: List[int] = []
    widths_sec: List[float] = []

    if duration_reference_unit == "peak":
        # reference peak ratio p (0~1) -> rel_height = 1 - p
        p = float(np.clip(duration_reference_value, 0.0, 1.0))
        rel_h = 1.0 - p
        w, wl, wr, _ = peak_widths(rms_normalized_series_smoothed, peaks, rel_height=rel_h)
        dur_sec = w / sample_rate
        for idx, width_s in zip(peaks, dur_sec):
            if (min_duration_sec <= width_s <= max_duration_sec) or (min_duration_sec == 0 and max_duration_sec == 0):
                valid_indices.append(int(idx))
                widths_sec.append(float(width_s))
        return valid_indices, widths_sec
    else:
        # dB absolute threshold
        thr = float(duration_reference_value)
        for pk in peaks:
            left = int(pk)
            right = int(pk)
            # find the points where the left and right sides drop to the threshold
            while left > 0 and db_series_smoothed[left] > thr:
                left -= 1
            while right < len(db_series_smoothed) - 1 and db_series_smoothed[right] > thr:
                right += 1
            width_s = max(0, right - left) / sample_rate
            if (min_duration_sec <= width_s <= max_duration_sec) or (min_duration_sec == 0 and max_duration_sec == 0):
                valid_indices.append(int(pk))
                widths_sec.append(float(width_s))
        return valid_indices, widths_sec


def _prominence_filter(series: np.ndarray, peaks: np.ndarray, enabled: bool, min_prom: float, unit: str, wlen_points: int) -> Tuple[List[int], List[float]]:
    if not enabled or len(peaks) == 0:
        return list(peaks.astype(int)), []
    wlen = None
    if wlen_points > 1:
        wlen = int(wlen_points)
    prominences, left_bases, right_bases = peak_prominences(series, peaks, wlen=wlen)
    valid: List[int] = []
    vals: List[float] = []
    for i, pk in enumerate(peaks):
        if prominences[i] >= min_prom:
            valid.append(int(pk))
            vals.append(float(prominences[i]))
    return valid, vals


def peak_detection(
    audio_signal: np.ndarray,
    sample_rate: int,
    config: Dict[str, Any],
    v2pa_factor: float = None,
) -> Dict[str, Any]:
    """
    peak detection (PD) based on configurable parameters.

    the input parameter config must be consistent with the output of UI PDConfigWindow.get_default_config.

    the return value contains the peak location, number, whether it passed the test, and the intermediate sequence for plotting/debugging.
    """

    # 1) pre-processing: filter
    filtered = _design_and_apply_filter(
        audio_signal=audio_signal,
        sample_rate=sample_rate,
        enabled=bool(config.get("filter_enabled", False)),
        frequency_ranges_str=str(config.get("filter_ranges", "")),
        filter_type=str(config.get("filter_type", "bandpass")),
        filter_order=int(config.get("filter_order", 4)),
    )

    # 2) compute SPL/RMS series (used for amplitude/threshold and duration measurement)
    rms_envelope, spl_db_series_raw = _compute_rms_spl_series(
        filtered,
        sample_rate,
        window_unit=str(config.get("spl_window_unit", "time")),
        window_time_sec=float(config.get("spl_window_time_sec", 0.050)),
        window_points=int(config.get("spl_window_points", 0)),
        ref_pressure=20e-6,
        v2pa_factor=v2pa_factor
    )

    # 3) normalization and smoothing (consistent pre-processing for dB and RMS series)
    rms_normalized_series_raw = _normalize_rms(rms_envelope)
    rms_normalized_series_smoothed = _smooth_series(
        data=rms_normalized_series_raw,
        enabled=bool(config.get("smooth_enabled", False)),
        window_unit=str(config.get("smooth_unit", "time")),
        window_time_sec=float(config.get("smooth_time_sec", 0.02)),
        window_points=int(config.get("smooth_points", 0)),
        smoothing_algorithm_id=int(config.get("smooth_algo", 1)),
        sample_rate=sample_rate,
    )
    db_series_smoothed = _smooth_series(
        data=spl_db_series_raw,
        enabled=bool(config.get("smooth_enabled", False)),
        window_unit=str(config.get("smooth_unit", "time")),
        window_time_sec=float(config.get("smooth_time_sec", 0.02)),
        window_points=int(config.get("smooth_points", 0)),
        smoothing_algorithm_id=int(config.get("smooth_algo", 1)),
        sample_rate=sample_rate,
    )

    # detection series selection (based on the peak size unit)
    use_rms_for_height = str(config.get("peak_size_unit", "db")) == "rms"
    detection_series = rms_normalized_series_smoothed if use_rms_for_height else db_series_smoothed
    detection_series_unit = "rms_norm" if use_rms_for_height else "db"

    # 4) peak detection
    height_enabled = bool(config.get("peak_size_enabled", True))
    min_height_threshold = None
    if height_enabled:
        min_height_threshold = float(config.get("peak_min_value", 0.0))
        if detection_series_unit == "rms_norm":
            # protect the threshold range [0,1]
            min_height_threshold = float(np.clip(min_height_threshold, 0.0, 1.0))

    min_distance_points = _choose_distance(
        nms_enabled=bool(config.get("nms_enabled", False)),
        nms_unit=str(config.get("nms_unit", "time")),
        nms_time_sec=float(config.get("nms_time_sec", 0.5)),
        nms_points=int(config.get("nms_points", 0)),
        sample_rate=sample_rate,
    )

    peaks, properties = find_peaks(detection_series, height=min_height_threshold, distance=min_distance_points)

    # 5) prominence (significance) constraint (optional)
    # calculate the window length (from the advanced item "peak prominence calculation window")
    prominence_window_points = 0
    convex_unit = str(config.get("convex_unit", "audio"))
    if convex_unit == "audio":
        ratio = float(config.get("convex_audio_ratio", 1.0))
        prominence_window_points = max(0, int(round(len(detection_series) * max(0.0, ratio))))
    elif convex_unit == "points":
        prominence_window_points = int(config.get("convex_points", 1024))
    else:  # time
        prominence_window_points = max(0, int(round(float(config.get("convex_time_sec", 0.0)) * sample_rate)))

    prom_enabled = bool(config.get("peak_slope_enabled", False))
    min_prominence = float(config.get("peak_min_slope", 0.0))
    # prominence unit selection: calculate on the corresponding sequence
    use_rms_for_prom = str(config.get("peak_slope_unit", "db")) == "rms"
    prom_series = rms_normalized_series_smoothed if use_rms_for_prom else db_series_smoothed
    peaks_after_prominence_filter, prominence_values = _prominence_filter(
        series=prom_series,
        peaks=peaks,
        enabled=prom_enabled,
        min_prom=min_prominence,
        unit=str(config.get("peak_slope_unit", "db")),
        wlen_points=prominence_window_points,
    )

    peaks_np = np.array(peaks_after_prominence_filter, dtype=int)

    # 6) duration constraint (optional)
    is_duration_enabled = bool(config.get("duration_enabled", False))
    min_duration_sec = float(config.get("duration_min", 0.0))
    max_duration_sec = float(config.get("duration_max", 0.5))
    duration_reference_unit = str(config.get("duration_ref_unit", "peak"))
    duration_reference_value = float(config.get("duration_ref_value", 0.5))

    peaks_after_duration_filter, width_secs = _duration_filter(
        db_series_smoothed=db_series_smoothed,
        rms_normalized_series_smoothed=rms_normalized_series_smoothed,
        peaks=peaks_np,
        sample_rate=sample_rate,
        enabled=is_duration_enabled,
        duration_reference_unit=duration_reference_unit,
        duration_reference_value=duration_reference_value,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
    )

    final_peaks = np.array(peaks_after_duration_filter, dtype=int)

    # 7) number limit (peak count)
    if bool(config.get("peak_count_enabled", True)):
        max_count = int(config.get("peak_count", 3))
        if final_peaks.size > max_count:
            # take the top N peaks
            peak_heights = detection_series[final_peaks]
            order = np.argsort(peak_heights)[::-1][:max_count]
            final_peaks = final_peaks[order]

    final_peaks.sort()

    # 8) test determination
    test_operator = str(config.get("test_peak_op", "≥"))
    test_threshold_value = int(config.get("test_peak_value", 3))
    num_detected_peaks = int(final_peaks.size)
    if test_operator == ">":
        passed = num_detected_peaks > test_threshold_value
    elif test_operator == "<":
        passed = num_detected_peaks < test_threshold_value
    elif test_operator == "=":
        passed = num_detected_peaks == test_threshold_value
    elif test_operator in ("≥", ">="):
        passed = num_detected_peaks >= test_threshold_value
    elif test_operator in ("≤", "<="):
        passed = num_detected_peaks <= test_threshold_value
    else:
        passed = num_detected_peaks >= test_threshold_value

    peak_times_sec = (final_peaks / float(sample_rate)).tolist()

    result: Dict[str, Any] = {
        "peaks_index": final_peaks.tolist(),
        "peaks_time_sec": peak_times_sec,
        "num_peaks": int(final_peaks.size),
        "passed": bool(passed),
        # for easy plotting/debugging
        "detect_series": detection_series.tolist(),
        "detect_series_unit": detection_series_unit,
        "rms_series": rms_normalized_series_smoothed.tolist(),
        "spl_db_series": db_series_smoothed.tolist(),
    }

    # additional optional information
    if prominence_values:
        result["prominences"] = prominence_values
    if width_secs:
        result["widths_sec"] = width_secs

    return result

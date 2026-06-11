import warnings
from copy import deepcopy

import numpy as np
from scipy import ndimage, signal


DEFAULT_FRAME_LENGTH_MS = 30.0
DEFAULT_FRAME_SHIFT_MS = 10.0
DEFAULT_WINDOW_TYPE = "hamming"
DEFAULT_STFT_NFFT = 2048
DEFAULT_ENVELOPE_WINDOW_MS = 10.0
DEFAULT_ENVELOPE_SHIFT_MS = 1.0
DEFAULT_INTERNAL_MOD_FREQ_STEP_HZ = 1.0
DEFAULT_SIGNAL_FREQ_DISPLAY_STEP_HZ = 1.0
DEFAULT_MOD_FREQ_BIN_HZ = 1.0
DEFAULT_SMOOTHING_POINTS = 3
LOW_ENERGY_RATIO = 0.18
HOTSPOT_MIN_SIGNAL_FREQ_HZ = 500.0
MAX_HOTSPOTS = 8
MODULATION_CHUNK_ROWS = 32
DEFAULT_FAN_RPM = 4500.0
DEFAULT_BLADE_COUNT = 2
DEFAULT_ROTATION_HARMONICS = 2
DEFAULT_BPF_HARMONICS = 1
DEFAULT_MAIN_TONES_HZ = (1200.0, 3500.0)
DEFAULT_MAIN_TONE_SEARCH_WIDTH_HZ = 160.0
DEFAULT_TONE_BAND_HZ = DEFAULT_MAIN_TONE_SEARCH_WIDTH_HZ / 2.0
DEFAULT_MECHANICAL_MATCH_TOLERANCE_HZ = 20.0
DEFAULT_SHOW_GLOBAL_HOTSPOTS = True
DEFAULT_MIN_MODULATION_DEPTH_PERCENT = 1.0


DEFAULT_MODULATION_CONFIG = {
    "main_tones_hz": list(DEFAULT_MAIN_TONES_HZ),
    "fan_rpm": DEFAULT_FAN_RPM,
    "blade_count": DEFAULT_BLADE_COUNT,
    "threshold_percent": 10.0,
    "signal_freq_range_hz": [0.0, 10000.0],
    "mod_freq_range_hz": [0.0, 200.0],
    "window_type": DEFAULT_WINDOW_TYPE,
    "stft_nfft": DEFAULT_STFT_NFFT,
    "frame_length_ms": DEFAULT_FRAME_LENGTH_MS,
    "frame_shift_ms": DEFAULT_FRAME_SHIFT_MS,
    "envelope_window_ms": DEFAULT_ENVELOPE_WINDOW_MS,
    "envelope_shift_ms": DEFAULT_ENVELOPE_SHIFT_MS,
    "signal_freq_display_step_hz": DEFAULT_SIGNAL_FREQ_DISPLAY_STEP_HZ,
    "mod_freq_bin_hz": DEFAULT_MOD_FREQ_BIN_HZ,
    "smoothing_points": DEFAULT_SMOOTHING_POINTS,
    "rotation_harmonics": DEFAULT_ROTATION_HARMONICS,
    "bpf_harmonics": DEFAULT_BPF_HARMONICS,
    "mechanical_freqs_hz": [],
    "main_tone_search_width_hz": DEFAULT_MAIN_TONE_SEARCH_WIDTH_HZ,
    "mechanical_match_tolerance_hz": DEFAULT_MECHANICAL_MATCH_TOLERANCE_HZ,
    "show_global_hotspots": DEFAULT_SHOW_GLOBAL_HOTSPOTS,
    "min_modulation_depth_percent": DEFAULT_MIN_MODULATION_DEPTH_PERCENT,
    "tone_band_hz": DEFAULT_TONE_BAND_HZ,
    "core_freq_lines_khz": [0.5, 1.0, 2.0, 4.0, 8.0],
    "analysis_channel": 0,
}


def default_modulation_config():
    return deepcopy(DEFAULT_MODULATION_CONFIG)


def _clean_audio(audio_signal):
    audio = np.asarray(audio_signal, dtype=np.float64)
    if audio.ndim != 1:
        raise ValueError("audio_signal must be a one-dimensional array.")
    if audio.size < 128:
        raise ValueError("audio_signal is too short for modulation analysis.")
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = audio - np.mean(audio)
    return audio


def _range_pair(value, default_pair):
    if value is None:
        return float(default_pair[0]), float(default_pair[1])
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").replace("\n", ",").split(",") if p.strip()]
    else:
        parts = list(value) if isinstance(value, (list, tuple)) else []
    if len(parts) < 2:
        return float(default_pair[0]), float(default_pair[1])
    lo = float(parts[0])
    hi = float(parts[1])
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _float_list(value, default_values=()):
    if value is None:
        return [float(v) for v in default_values]
    if isinstance(value, str):
        raw_items = value.replace(";", ",").replace("\n", ",").replace("\t", ",").split(",")
    elif isinstance(value, (list, tuple, np.ndarray)):
        raw_items = list(value)
    else:
        raw_items = [value]

    out = []
    for item in raw_items:
        if isinstance(item, dict):
            item = item.get("freq_hz", item.get("frequency_hz", item.get("hz")))
        elif isinstance(item, (list, tuple)) and item:
            item = item[0]
        text = str(item).strip()
        if not text:
            continue
        out.append(float(text))
    return out


def _choose_envelope_stft_params(
    sample_rate,
    sample_count,
    envelope_window_ms,
    envelope_shift_ms,
    stft_nperseg,
    stft_nfft,
):
    if stft_nperseg is None:
        nperseg = int(round(sample_rate * envelope_window_ms / 1000.0))
    else:
        nperseg = int(stft_nperseg)
    nperseg = min(nperseg, sample_count)

    hop_length = max(1, int(round(sample_rate * envelope_shift_ms / 1000.0)))
    nfft = int(stft_nfft)

    if nperseg < 128:
        raise ValueError("audio_signal is too short for the configured envelope window.")
    if hop_length <= 0:
        raise ValueError("envelope_shift_ms must produce a positive hop length.")
    if nfft < nperseg:
        raise ValueError("stft_nfft must be greater than or equal to stft_nperseg.")

    if hop_length >= nperseg:
        hop_length = max(1, nperseg // 4)

    noverlap = nperseg - hop_length
    return nperseg, nfft, noverlap, hop_length


def _choose_modulation_frame_params(frame_length_ms, frame_shift_ms, envelope_rate_hz):
    frame_points = int(round(frame_length_ms * envelope_rate_hz / 1000.0))
    hop_points = int(round(frame_shift_ms * envelope_rate_hz / 1000.0))
    frame_points = max(4, frame_points)
    hop_points = max(1, hop_points)
    return frame_points, hop_points


def _build_mod_freq_axis(mod_freq_range_hz, envelope_nyquist_hz, mod_freq_step_hz):
    effective_mod_freq_range_hz = (
        float(mod_freq_range_hz[0]),
        min(float(mod_freq_range_hz[1]), envelope_nyquist_hz),
    )
    if effective_mod_freq_range_hz[1] < effective_mod_freq_range_hz[0]:
        raise ValueError("No modulation frequency bins remain after filtering.")

    step_hz = DEFAULT_INTERNAL_MOD_FREQ_STEP_HZ
    if mod_freq_step_hz is not None and mod_freq_step_hz > 0:
        step_hz = float(mod_freq_step_hz)

    mod_freqs = np.arange(
        effective_mod_freq_range_hz[0],
        effective_mod_freq_range_hz[1] + step_hz * 0.5,
        step_hz,
    )
    return mod_freqs, effective_mod_freq_range_hz


def _compute_modulation_depth(
    audio,
    sample_rate,
    signal_freq_range_hz,
    mod_freq_range_hz,
    frame_length_ms,
    frame_shift_ms,
    window_type,
    stft_nperseg,
    stft_nfft,
    envelope_window_ms,
    envelope_shift_ms,
    mod_freq_step_hz,
    main_tones_hz=None,
    tone_band_hz=DEFAULT_TONE_BAND_HZ,
):
    nperseg, nfft, noverlap, envelope_hop_length = _choose_envelope_stft_params(
        sample_rate,
        audio.size,
        envelope_window_ms,
        envelope_shift_ms,
        stft_nperseg,
        stft_nfft,
    )

    freqs, _, stft = signal.stft(
        audio,
        fs=sample_rate,
        window=window_type,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary=None,
        padded=False,
    )
    envelope = np.abs(stft)

    roi_main_tones_hz = _float_list(main_tones_hz, ())
    tone_band_hz = max(float(tone_band_hz), 0.0)
    full_range_mask = (freqs >= signal_freq_range_hz[0]) & (freqs <= signal_freq_range_hz[1])
    if main_tones_hz is None or len(roi_main_tones_hz) == 0:
        freq_mask = full_range_mask
        analysis_scope = "full"
    else:
        freq_mask = np.zeros_like(freqs, dtype=bool)
        for tone_freq in roi_main_tones_hz:
            roi_mask = (
                (freqs >= tone_freq - tone_band_hz)
                & (freqs <= tone_freq + tone_band_hz)
            )
            freq_mask |= roi_mask
        freq_mask &= full_range_mask
        analysis_scope = "main_tone_roi"
    signal_freqs = freqs[freq_mask]
    envelope = envelope[freq_mask]

    if envelope.shape[0] == 0:
        raise ValueError("No signal frequency bins remain after filtering.")

    envelope_rate_hz = sample_rate / envelope_hop_length
    envelope_nyquist_hz = envelope_rate_hz / 2.0
    mod_freqs, effective_mod_freq_range_hz = _build_mod_freq_axis(
        mod_freq_range_hz,
        envelope_nyquist_hz,
        mod_freq_step_hz,
    )
    if effective_mod_freq_range_hz[1] < float(mod_freq_range_hz[1]):
        warnings.warn(
            "Requested modulation upper frequency is above the envelope Nyquist frequency. "
            f"With envelope_shift_ms={envelope_shift_ms:g}, the effective modulation range is "
            f"0-{effective_mod_freq_range_hz[1]:.1f} Hz.",
            RuntimeWarning,
            stacklevel=2,
        )

    modulation_frame_points, modulation_hop_points = _choose_modulation_frame_params(
        frame_length_ms,
        frame_shift_ms,
        envelope_rate_hz,
    )
    if envelope.shape[1] < modulation_frame_points:
        raise ValueError("Not enough high-rate envelope frames for modulation analysis.")

    frame_starts = np.arange(
        0,
        envelope.shape[1] - modulation_frame_points + 1,
        modulation_hop_points,
    )
    if frame_starts.size == 0:
        raise ValueError("No modulation analysis frames were produced.")

    time_window = signal.get_window(window_type, modulation_frame_points, fftbins=True)
    time_indices = np.arange(modulation_frame_points)
    modulation_basis = time_window[:, np.newaxis] * np.exp(
        -2j
        * np.pi
        * time_indices[:, np.newaxis]
        * mod_freqs[np.newaxis, :]
        / envelope_rate_hz
    )
    window_sum = np.sum(time_window)

    mod_depth = np.zeros((envelope.shape[0], mod_freqs.size), dtype=np.float64)
    for row_start in range(0, envelope.shape[0], MODULATION_CHUNK_ROWS):
        row_stop = min(row_start + MODULATION_CHUNK_ROWS, envelope.shape[0])
        envelope_chunk = envelope[row_start:row_stop]
        envelope_windows = np.stack(
            [
                envelope_chunk[:, frame_start : frame_start + modulation_frame_points]
                for frame_start in frame_starts
            ],
            axis=1,
        )
        frame_mean = np.mean(envelope_windows, axis=2, keepdims=True)
        centered_windows = envelope_windows - frame_mean

        mod_coefficients = centered_windows.reshape(-1, modulation_frame_points) @ modulation_basis
        mod_amplitude = 2.0 * np.abs(mod_coefficients) / np.maximum(window_sum, 1e-12)
        frame_depth = 100.0 * mod_amplitude / np.maximum(frame_mean.reshape(-1, 1), 1e-12)
        mod_depth[row_start:row_stop] = frame_depth.reshape(
            row_stop - row_start,
            frame_starts.size,
            mod_freqs.size,
        ).mean(axis=1)

    mod_depth[:, np.isclose(mod_freqs, 0.0)] = 0.0

    mean_flat = np.mean(envelope, axis=1)
    if np.max(mean_flat) > 0:
        valid_energy = mean_flat >= LOW_ENERGY_RATIO * np.max(mean_flat)
        mod_depth[~valid_energy, :] = 0.0

    stft_params = {
        "analysis_method": "high_rate_subband_envelope",
        "frame_length_ms": frame_length_ms,
        "frame_shift_ms": frame_shift_ms,
        "envelope_window_ms": envelope_window_ms,
        "envelope_shift_ms": 1000.0 * envelope_hop_length / sample_rate,
        "window_type": window_type,
        "stft_nperseg": nperseg,
        "stft_nfft": nfft,
        "stft_noverlap": noverlap,
        "stft_hop_length": envelope_hop_length,
        "modulation_frame_points": modulation_frame_points,
        "modulation_hop_points": modulation_hop_points,
        "modulation_frame_count": int(frame_starts.size),
        "internal_signal_freq_resolution_hz": sample_rate / nfft,
        "internal_mod_freq_resolution_hz": envelope_rate_hz / modulation_frame_points,
        "envelope_rate_hz": envelope_rate_hz,
        "envelope_nyquist_hz": envelope_nyquist_hz,
        "requested_mod_freq_range_hz": mod_freq_range_hz,
        "effective_mod_freq_range_hz": effective_mod_freq_range_hz,
        "analysis_scope": analysis_scope,
        "main_tones_hz": roi_main_tones_hz if roi_main_tones_hz else None,
        "tone_band_hz": tone_band_hz,
        "computed_signal_freq_count": int(signal_freqs.size),
    }
    return mod_depth, signal_freqs, mod_freqs, stft_params


def _smooth_modulation_depth(mod_depth, smoothing_points):
    if smoothing_points is None or smoothing_points <= 1:
        return mod_depth
    points = int(round(smoothing_points))
    if points <= 1:
        return mod_depth
    return ndimage.uniform_filter(mod_depth, size=(points, points), mode="nearest")


def _bin_modulation_axis(mod_depth, mod_freqs, mod_freq_bin_hz, mod_freq_range_hz):
    if mod_freq_bin_hz is None or mod_freq_bin_hz <= 0:
        return mod_depth, mod_freqs

    bin_hz = float(mod_freq_bin_hz)
    start_hz, stop_hz = mod_freq_range_hz
    binned_freqs = np.arange(start_hz, stop_hz + bin_hz * 0.5, bin_hz)
    binned_depth = np.zeros((mod_depth.shape[0], binned_freqs.size), dtype=mod_depth.dtype)

    half_bin = bin_hz / 2.0
    for index, center_hz in enumerate(binned_freqs):
        in_bin = (mod_freqs >= center_hz - half_bin) & (mod_freqs < center_hz + half_bin)
        if index == binned_freqs.size - 1:
            in_bin |= np.isclose(mod_freqs, center_hz)
        if np.any(in_bin):
            binned_depth[:, index] = np.max(mod_depth[:, in_bin], axis=1)

    return binned_depth, binned_freqs


def _interpolate_signal_axis(mod_depth, signal_freqs, signal_freq_display_step_hz):
    if signal_freq_display_step_hz is None or signal_freq_display_step_hz <= 0:
        return mod_depth, signal_freqs

    step_hz = float(signal_freq_display_step_hz)
    raw_step_hz = _axis_step(signal_freqs)
    if raw_step_hz is None or raw_step_hz <= 0:
        return mod_depth, signal_freqs

    gap_threshold_hz = max(step_hz * 2.0, raw_step_hz * 2.5)
    split_points = np.where(np.diff(signal_freqs) > gap_threshold_hz)[0] + 1
    index_segments = np.split(np.arange(signal_freqs.size), split_points)
    display_depth_parts = []
    display_freq_parts = []

    for indices in index_segments:
        segment_freqs = signal_freqs[indices]
        segment_depth = mod_depth[indices]
        if segment_freqs.size < 2:
            display_depth_parts.append(segment_depth)
            display_freq_parts.append(segment_freqs)
            continue

        start_hz = np.ceil(segment_freqs[0] / step_hz) * step_hz
        stop_hz = np.floor(segment_freqs[-1] / step_hz) * step_hz
        display_freqs = np.arange(start_hz, stop_hz + step_hz * 0.5, step_hz)
        if display_freqs.size < 2:
            display_depth_parts.append(segment_depth)
            display_freq_parts.append(segment_freqs)
            continue

        display_depth = np.empty((display_freqs.size, mod_depth.shape[1]), dtype=mod_depth.dtype)
        for col in range(mod_depth.shape[1]):
            display_depth[:, col] = np.interp(display_freqs, segment_freqs, segment_depth[:, col])
        display_depth_parts.append(display_depth)
        display_freq_parts.append(display_freqs)

    return np.vstack(display_depth_parts), np.concatenate(display_freq_parts)


def _axis_step(axis):
    if axis.size < 2:
        return None
    return float(np.median(np.diff(axis)))


def _odd_window_bins(width_hz, axis):
    step_hz = _axis_step(axis)
    if step_hz is None or step_hz <= 0:
        return 3
    bins = max(3, int(round(width_hz / step_hz)))
    return bins if bins % 2 == 1 else bins + 1


def _find_hotspots(mod_depth, signal_freqs, mod_freqs, threshold_percent):
    if mod_depth.size == 0:
        return []

    search = np.array(mod_depth, copy=True)
    search[:, mod_freqs < 1.0] = 0.0
    search[signal_freqs < HOTSPOT_MIN_SIGNAL_FREQ_HZ, :] = 0.0

    local_max = search == ndimage.maximum_filter(
        search,
        size=(
            _odd_window_bins(120.0, signal_freqs),
            _odd_window_bins(8.0, mod_freqs),
        ),
        mode="nearest",
    )
    candidate_rows, candidate_cols = np.where(local_max & (search >= threshold_percent))

    candidates = []
    for row, col in zip(candidate_rows, candidate_cols):
        candidates.append(
            {
                "signal_freq_khz": float(signal_freqs[row] / 1000.0),
                "signal_freq_hz": float(signal_freqs[row]),
                "mod_freq_hz": float(mod_freqs[col]),
                "mod_depth_percent": float(search[row, col]),
            }
        )

    candidates.sort(key=lambda item: item["mod_depth_percent"], reverse=True)
    return candidates[:MAX_HOTSPOTS]


def _filter_hotspots_near_unmodulated_main_tones(hotspots, main_tone_results, main_tone_search_width_hz):
    if not hotspots or not main_tone_results:
        return hotspots

    half_width_hz = max(float(main_tone_search_width_hz), 0.0) / 2.0
    if half_width_hz <= 0:
        return hotspots

    unmodulated_ranges = []
    for item in main_tone_results:
        if bool(item.get("has_modulation_peak", True)):
            continue
        center_hz = float(item.get("target_signal_freq_hz", item.get("analysis_signal_freq_hz", 0.0)))
        unmodulated_ranges.append((center_hz - half_width_hz, center_hz + half_width_hz))

    if not unmodulated_ranges:
        return hotspots

    filtered = []
    for hotspot in hotspots:
        signal_hz = hotspot.get("signal_freq_hz")
        if signal_hz is None:
            signal_hz = float(hotspot.get("signal_freq_khz", 0.0)) * 1000.0
        signal_hz = float(signal_hz)
        if any(low <= signal_hz <= high for low, high in unmodulated_ranges):
            continue
        filtered.append(hotspot)
    return filtered


def _normalize_main_tones(main_tones, main_tones_hz):
    if main_tones is None:
        main_tones = main_tones_hz
    if main_tones is None:
        return []

    normalized = []
    for index, item in enumerate(main_tones, start=1):
        if isinstance(item, dict):
            freq_hz = item.get("freq_hz", item.get("frequency_hz", item.get("hz")))
            label = item.get("label", item.get("name"))
        elif isinstance(item, (tuple, list)):
            freq_hz = item[0]
            label = item[1] if len(item) > 1 else None
        else:
            freq_hz = item
            label = None

        freq_hz = float(freq_hz)
        if label is None:
            label = f"main tone {index}"
        normalized.append(
            {
                "id": index,
                "label": str(label),
                "freq_hz": freq_hz,
            }
        )

    return normalized


def _mechanical_modulation_refs(
    fan_rpm,
    blade_count,
    mod_freq_range_hz,
    rotation_harmonics,
    bpf_harmonics,
    extra_mechanical_freqs_hz,
):
    refs = []
    start_hz, stop_hz = mod_freq_range_hz

    def add_ref(freq_hz, label):
        if freq_hz < start_hz or freq_hz > stop_hz:
            return
        for ref in refs:
            if abs(ref["freq_hz"] - freq_hz) < 1e-6:
                ref["label"] = f"{ref['label']} / {label}"
                return
        refs.append({"freq_hz": float(freq_hz), "label": label})

    if fan_rpm is not None and fan_rpm > 0:
        rotation_hz = float(fan_rpm) / 60.0
        for harmonic in range(1, int(rotation_harmonics) + 1):
            add_ref(rotation_hz * harmonic, f"rotation {harmonic}x")

        if blade_count is not None and blade_count > 0:
            bpf_hz = rotation_hz * float(blade_count)
            for harmonic in range(1, int(bpf_harmonics) + 1):
                add_ref(bpf_hz * harmonic, f"BPF {harmonic}x")

    if extra_mechanical_freqs_hz:
        for index, freq_hz in enumerate(extra_mechanical_freqs_hz, start=1):
            add_ref(float(freq_hz), f"extra {index}")

    refs.sort(key=lambda ref: ref["freq_hz"])
    return refs


def _mechanical_ref_freqs(mechanical_refs):
    return np.asarray([ref["freq_hz"] for ref in mechanical_refs], dtype=np.float64)


def _evaluate_main_tones(
    mod_depth,
    signal_freqs,
    mod_freqs,
    main_tones,
    threshold_percent,
    mechanical_refs,
    mechanical_match_tolerance_hz,
    main_tone_search_width_hz,
    min_modulation_depth_percent,
):
    if not main_tones:
        return []

    mod_mask = mod_freqs >= 1.0
    if not np.any(mod_mask):
        return []

    results = []
    mechanical_mod_freqs = _mechanical_ref_freqs(mechanical_refs)
    for tone in main_tones:
        tone_hz = tone["freq_hz"]
        half_width_hz = max(float(main_tone_search_width_hz), 0.0) / 2.0
        if half_width_hz > 0:
            row_indices = np.where(
                (signal_freqs >= tone_hz - half_width_hz)
                & (signal_freqs <= tone_hz + half_width_hz)
            )[0]
        else:
            row_indices = np.array([], dtype=int)
        if row_indices.size == 0:
            row_indices = np.asarray([int(np.argmin(np.abs(signal_freqs - tone_hz)))], dtype=int)

        col_indices = np.where(mod_mask)[0]
        tone_depth = mod_depth[np.ix_(row_indices, col_indices)]
        any_row_offset, any_col_offset = np.unravel_index(
            int(np.argmax(tone_depth)),
            tone_depth.shape,
        )
        row = int(row_indices[any_row_offset])
        any_col = int(col_indices[any_col_offset])
        max_any_depth_percent = float(mod_depth[row, any_col])
        max_any_mod_freq_hz = float(mod_freqs[any_col])
        has_modulation_peak = max_any_depth_percent >= float(min_modulation_depth_percent)

        if mechanical_mod_freqs.size:
            mechanical_cols = []
            for mechanical_hz in mechanical_mod_freqs:
                nearest_col = int(np.argmin(np.abs(mod_freqs - mechanical_hz)))
                if mod_mask[nearest_col]:
                    mechanical_cols.append(nearest_col)
            mechanical_cols = np.asarray(sorted(set(mechanical_cols)), dtype=int)
        else:
            mechanical_cols = col_indices

        mechanical_row = row
        if mechanical_cols.size:
            mechanical_depth_window = mod_depth[np.ix_(row_indices, mechanical_cols)]
            mechanical_row_offset, mechanical_col_offset = np.unravel_index(
                int(np.argmax(mechanical_depth_window)),
                mechanical_depth_window.shape,
            )
            mechanical_row = int(row_indices[mechanical_row_offset])
            local_col = int(mechanical_cols[mechanical_col_offset])
        else:
            local_col = any_col

        mechanical_depth_percent = float(mod_depth[mechanical_row, local_col])
        mechanical_mod_freq_hz = float(mod_freqs[local_col])
        mod_freq_hz = max_any_mod_freq_hz if has_modulation_peak else None

        if has_modulation_peak and mechanical_mod_freqs.size:
            nearest_index = int(np.argmin(np.abs(mechanical_mod_freqs - mod_freq_hz)))
            nearest_mech_hz = float(mechanical_mod_freqs[nearest_index])
            nearest_mech_label = mechanical_refs[nearest_index]["label"]
            mechanical_delta_hz = abs(mod_freq_hz - nearest_mech_hz)
            mechanical_match = mechanical_delta_hz <= mechanical_match_tolerance_hz
        else:
            nearest_mech_hz = None
            nearest_mech_label = None
            mechanical_delta_hz = None
            mechanical_match = True

        depth_pass = max_any_depth_percent <= threshold_percent
        is_valid = depth_pass and mechanical_match
        fail_reasons = []
        if not depth_pass:
            fail_reasons.append("AM depth above threshold")
        if has_modulation_peak and not mechanical_match:
            fail_reasons.append("modulation frequency not mechanically matched")

        results.append(
            {
                "main_tone_id": tone["id"],
                "main_tone_label": tone["label"],
                "target_signal_freq_hz": float(tone_hz),
                "analysis_signal_freq_hz": float(signal_freqs[row]),
                "signal_freq_khz": float(signal_freqs[row] / 1000.0),
                "mod_freq_hz": mod_freq_hz,
                "peak_mod_freq_hz": max_any_mod_freq_hz,
                "mod_depth_percent": max_any_depth_percent,
                "am_depth_percent": max_any_depth_percent,
                "mechanical_mod_freq_hz": mechanical_mod_freq_hz,
                "mechanical_mod_depth_percent": mechanical_depth_percent,
                "max_any_mod_freq_hz": max_any_mod_freq_hz,
                "max_any_mod_depth_percent": max_any_depth_percent,
                "nearest_mechanical_mod_freq_hz": nearest_mech_hz,
                "nearest_mechanical_label": nearest_mech_label,
                "mechanical_delta_hz": mechanical_delta_hz,
                "mechanical_match": bool(mechanical_match),
                "has_modulation_peak": bool(has_modulation_peak),
                "min_modulation_depth_percent": float(min_modulation_depth_percent),
                "depth_pass": bool(depth_pass),
                "is_valid": bool(is_valid),
                "status": "valid" if is_valid else "invalid",
                "reason": (
                    "; ".join(fail_reasons)
                    if fail_reasons
                    else ("pass" if has_modulation_peak else "no AM peak")
                ),
            }
        )

    return results


def compute_modulation_map(audio_signal, sample_rate, config=None):
    cfg = default_modulation_config()
    if isinstance(config, dict):
        for key, value in config.items():
            if value is not None or key == "main_tones_hz":
                cfg[key] = value

    audio = _clean_audio(audio_signal)
    sample_rate = int(sample_rate)

    signal_freq_range_hz = _range_pair(cfg.get("signal_freq_range_hz"), DEFAULT_MODULATION_CONFIG["signal_freq_range_hz"])
    mod_freq_range_hz = _range_pair(cfg.get("mod_freq_range_hz"), DEFAULT_MODULATION_CONFIG["mod_freq_range_hz"])
    main_tones_hz = _float_list(cfg.get("main_tones_hz"), DEFAULT_MAIN_TONES_HZ)
    mechanical_freqs_hz = _float_list(cfg.get("mechanical_freqs_hz"), ())
    tone_band_hz = float(cfg.get("tone_band_hz", DEFAULT_TONE_BAND_HZ))

    mod_freq_bin_hz = float(cfg.get("mod_freq_bin_hz", DEFAULT_MOD_FREQ_BIN_HZ))
    internal_mod_freq_step_hz = DEFAULT_INTERNAL_MOD_FREQ_STEP_HZ
    if mod_freq_bin_hz > 0:
        internal_mod_freq_step_hz = min(internal_mod_freq_step_hz, mod_freq_bin_hz)

    mod_depth, signal_freqs, mod_freqs, stft_params = _compute_modulation_depth(
        audio,
        sample_rate,
        signal_freq_range_hz,
        mod_freq_range_hz,
        float(cfg.get("frame_length_ms", DEFAULT_FRAME_LENGTH_MS)),
        float(cfg.get("frame_shift_ms", DEFAULT_FRAME_SHIFT_MS)),
        str(cfg.get("window_type", DEFAULT_WINDOW_TYPE) or DEFAULT_WINDOW_TYPE),
        cfg.get("stft_nperseg"),
        int(cfg.get("stft_nfft", DEFAULT_STFT_NFFT)),
        float(cfg.get("envelope_window_ms", DEFAULT_ENVELOPE_WINDOW_MS)),
        float(cfg.get("envelope_shift_ms", DEFAULT_ENVELOPE_SHIFT_MS)),
        internal_mod_freq_step_hz,
        main_tones_hz if cfg.get("main_tones_hz") is not None else None,
        tone_band_hz,
    )
    effective_mod_freq_range_hz = stft_params["effective_mod_freq_range_hz"]
    mod_depth, mod_freqs = _bin_modulation_axis(
        mod_depth,
        mod_freqs,
        mod_freq_bin_hz,
        effective_mod_freq_range_hz,
    )
    mod_depth, signal_freqs = _interpolate_signal_axis(
        mod_depth,
        signal_freqs,
        float(cfg.get("signal_freq_display_step_hz", DEFAULT_SIGNAL_FREQ_DISPLAY_STEP_HZ)),
    )
    mod_depth = _smooth_modulation_depth(
        mod_depth,
        int(cfg.get("smoothing_points", DEFAULT_SMOOTHING_POINTS)),
    )
    stft_params.update(
        {
            "signal_freq_display_step_hz": cfg.get("signal_freq_display_step_hz"),
            "mod_freq_bin_hz": mod_freq_bin_hz,
            "smoothing_points": cfg.get("smoothing_points"),
            "signal_freq_axis_step_hz": _axis_step(signal_freqs),
            "mod_freq_axis_step_hz": _axis_step(mod_freqs),
        }
    )

    normalized_main_tones = _normalize_main_tones(None, main_tones_hz)
    mechanical_refs = _mechanical_modulation_refs(
        float(cfg.get("fan_rpm", DEFAULT_FAN_RPM) or 0.0),
        int(cfg.get("blade_count", DEFAULT_BLADE_COUNT) or 0),
        effective_mod_freq_range_hz,
        int(cfg.get("rotation_harmonics", DEFAULT_ROTATION_HARMONICS) or 0),
        int(cfg.get("bpf_harmonics", DEFAULT_BPF_HARMONICS) or 0),
        mechanical_freqs_hz,
    )
    mechanical_mod_freqs = _mechanical_ref_freqs(mechanical_refs)
    threshold_percent = float(cfg.get("threshold_percent", 10.0))
    min_modulation_depth_percent = float(
        cfg.get("min_modulation_depth_percent", DEFAULT_MIN_MODULATION_DEPTH_PERCENT)
    )

    main_tone_results = _evaluate_main_tones(
        mod_depth,
        signal_freqs,
        mod_freqs,
        normalized_main_tones,
        threshold_percent,
        mechanical_refs,
        float(cfg.get("mechanical_match_tolerance_hz", DEFAULT_MECHANICAL_MATCH_TOLERANCE_HZ)),
        float(cfg.get("main_tone_search_width_hz", DEFAULT_MAIN_TONE_SEARCH_WIDTH_HZ)),
        min_modulation_depth_percent,
    )
    global_hotspots = _find_hotspots(
        mod_depth,
        signal_freqs,
        mod_freqs,
        threshold_percent,
    )
    global_hotspots = _filter_hotspots_near_unmodulated_main_tones(
        global_hotspots,
        main_tone_results,
        float(cfg.get("main_tone_search_width_hz", DEFAULT_MAIN_TONE_SEARCH_WIDTH_HZ)),
    )

    return {
        "mod_depth_matrix": mod_depth,
        "signal_freq_axis_hz": signal_freqs,
        "mod_freq_axis_hz": mod_freqs,
        "hotspots": global_hotspots if bool(cfg.get("show_global_hotspots", DEFAULT_SHOW_GLOBAL_HOTSPOTS)) else [],
        "global_hotspots": global_hotspots,
        "main_tone_results": main_tone_results,
        "main_tone_validity_table": main_tone_results,
        "input_main_tones": normalized_main_tones,
        "mechanical_references": mechanical_refs,
        "mechanical_mod_freqs_hz": mechanical_mod_freqs,
        "stft_params": stft_params,
        "threshold_percent": threshold_percent,
        "min_modulation_depth_percent": min_modulation_depth_percent,
        "analysis_scope": stft_params.get("analysis_scope"),
        "main_tones_hz": stft_params.get("main_tones_hz"),
        "tone_band_hz": stft_params.get("tone_band_hz"),
        "computed_signal_freq_count": stft_params.get("computed_signal_freq_count"),
        "core_freq_lines_khz": _float_list(cfg.get("core_freq_lines_khz"), DEFAULT_MODULATION_CONFIG["core_freq_lines_khz"]),
        "main_tone_search_width_hz": float(
            cfg.get("main_tone_search_width_hz", DEFAULT_MAIN_TONE_SEARCH_WIDTH_HZ)
        ),
        "mechanical_match_tolerance_hz": float(
            cfg.get("mechanical_match_tolerance_hz", DEFAULT_MECHANICAL_MATCH_TOLERANCE_HZ)
        ),
    }

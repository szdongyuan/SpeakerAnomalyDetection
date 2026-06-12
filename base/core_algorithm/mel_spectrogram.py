from copy import deepcopy

import numpy as np
from scipy import signal


REFERENCE_PRESSURE_PA = 20e-6
DEFAULT_FMIN_HZ = 100.0
DEFAULT_FMAX_HZ = 20_000.0
DEFAULT_FRAME_LENGTH_MS = 30.0
DEFAULT_FRAME_SHIFT_MS = 10.0
DEFAULT_WINDOW = "hamming"
DEFAULT_STFT_NFFT = 4096
DEFAULT_N_MELS = 128
DEFAULT_CORE_RANGE_HZ = (2000.0, 5000.0)
DEFAULT_MEL_SCALE_RANGE = (0.0, 8000.0)
DEFAULT_MAIN_TONES_HZ = (1200.0, 3500.0)
DEFAULT_MAIN_TONE_SEARCH_WIDTH_HZ = 160.0
POWER_FLOOR_PA2 = 1e-30


DEFAULT_MEL_CONFIG = {
    "main_tones_hz": list(DEFAULT_MAIN_TONES_HZ),
    "main_tone_search_width_hz": DEFAULT_MAIN_TONE_SEARCH_WIDTH_HZ,
    "fmin_hz": DEFAULT_FMIN_HZ,
    "fmax_hz": DEFAULT_FMAX_HZ,
    "frame_length_ms": DEFAULT_FRAME_LENGTH_MS,
    "frame_shift_ms": DEFAULT_FRAME_SHIFT_MS,
    "window": DEFAULT_WINDOW,
    "stft_nfft": DEFAULT_STFT_NFFT,
    "n_mels": DEFAULT_N_MELS,
    "color_map": "magma",
    "dynamic_range_db": 65.0,
    "core_range_hz": list(DEFAULT_CORE_RANGE_HZ),
    "mel_scale_range": list(DEFAULT_MEL_SCALE_RANGE),
    "analysis_channel": 0,
}


def default_mel_config():
    return deepcopy(DEFAULT_MEL_CONFIG)


def hz_to_mel(freq_hz):
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + freq_hz / 700.0)


def mel_to_hz(mel_value):
    mel_value = np.asarray(mel_value, dtype=np.float64)
    return 700.0 * (10.0 ** (mel_value / 2595.0) - 1.0)


def _clean_audio(audio_signal):
    audio = np.asarray(audio_signal, dtype=np.float64)
    if audio.ndim != 1:
        raise ValueError("audio_signal must be a one-dimensional array.")
    if audio.size < 2:
        raise ValueError("audio_signal is too short for Mel spectrogram analysis.")
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio - np.mean(audio)


def _validate_positive(name, value):
    if value is None:
        raise ValueError(f"{name} is required.")
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value


def _range_pair(value, default_pair):
    if value is None:
        return float(default_pair[0]), float(default_pair[1])
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
    else:
        parts = list(value) if isinstance(value, (list, tuple, np.ndarray)) else []
    if len(parts) < 2:
        return float(default_pair[0]), float(default_pair[1])
    lo = float(parts[0])
    hi = float(parts[1])
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _a_weighting_db(freqs_hz):
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    freqs = np.maximum(freqs, 1e-12)
    f2 = freqs * freqs

    numerator = (12_200.0**2) * (f2**2)
    denominator = (
        (f2 + 20.6**2)
        * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2))
        * (f2 + 12_200.0**2)
    )
    return 20.0 * np.log10(numerator / denominator) + 2.0


def _mel_filter_bank(freqs_hz, n_mels, fmin_hz, fmax_hz):
    mel_edges = np.linspace(hz_to_mel(fmin_hz), hz_to_mel(fmax_hz), n_mels + 2)
    hz_edges = mel_to_hz(mel_edges)
    filters = np.zeros((n_mels, freqs_hz.size), dtype=np.float64)

    for index in range(n_mels):
        left_hz = hz_edges[index]
        center_hz = hz_edges[index + 1]
        right_hz = hz_edges[index + 2]

        left_mask = (freqs_hz >= left_hz) & (freqs_hz <= center_hz)
        right_mask = (freqs_hz >= center_hz) & (freqs_hz <= right_hz)

        if center_hz > left_hz:
            filters[index, left_mask] = (freqs_hz[left_mask] - left_hz) / (center_hz - left_hz)
        if right_hz > center_hz:
            filters[index, right_mask] = (right_hz - freqs_hz[right_mask]) / (right_hz - center_hz)

    valid_rows = filters.sum(axis=1) > 0
    if not np.all(valid_rows):
        missing_count = int(np.size(valid_rows) - np.count_nonzero(valid_rows))
        raise ValueError(f"{missing_count} mel filters have no FFT bins; reduce n_mels or widen range.")

    return filters, mel_edges, hz_edges


def _frequency_bin_widths(freqs_hz):
    if freqs_hz.size < 2:
        raise ValueError("At least two frequency bins are required.")
    widths = np.empty_like(freqs_hz, dtype=np.float64)
    widths[1:-1] = 0.5 * (freqs_hz[2:] - freqs_hz[:-2])
    widths[0] = freqs_hz[1] - freqs_hz[0]
    widths[-1] = freqs_hz[-1] - freqs_hz[-2]
    return widths


def _mel_display_range(mel_scale_range):
    mel_min, mel_max = _range_pair(mel_scale_range, DEFAULT_MEL_SCALE_RANGE)
    if mel_min < 0 or mel_max <= mel_min:
        raise ValueError("mel_scale_range must define a non-negative ascending Mel range.")
    return mel_min, mel_max


def _power_to_db(power_pa2):
    return 10.0 * np.log10(np.maximum(power_pa2, POWER_FLOOR_PA2) / REFERENCE_PRESSURE_PA**2)


def _mel_channel_edges(mel_axis, mel_edges):
    channel_edges = np.empty(mel_axis.size + 1, dtype=np.float64)
    channel_edges[0] = float(mel_edges[0])
    channel_edges[-1] = float(mel_edges[-1])
    if mel_axis.size > 1:
        channel_edges[1:-1] = 0.5 * (mel_axis[:-1] + mel_axis[1:])
    return channel_edges


def _first_number(mapping, keys):
    for key in keys:
        if key not in mapping:
            continue
        try:
            value = float(mapping.get(key))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def _normalize_main_tones(raw_tones, search_width_hz=0.0):
    if raw_tones is None:
        return []
    if isinstance(raw_tones, dict):
        if isinstance(raw_tones.get("dominant_tones"), (list, tuple, np.ndarray)):
            raw_tones = raw_tones.get("dominant_tones")
        elif isinstance(raw_tones.get("main_tones"), (list, tuple, np.ndarray)):
            raw_tones = raw_tones.get("main_tones")
        else:
            raw_tones = [raw_tones]
    elif not isinstance(raw_tones, (list, tuple, np.ndarray)):
        raw_tones = [raw_tones]

    normalized = []
    for order, raw in enumerate(raw_tones):
        source = "configured_main_tone"
        label = ""
        level_db = None
        prominence_db = None
        if isinstance(raw, dict):
            frequency_hz = _first_number(
                raw,
                (
                    "frequency_hz",
                    "freq_hz",
                    "main_tone_hz",
                    "target_signal_freq_hz",
                    "tone_hz",
                ),
            )
            f_low = _first_number(raw, ("f_low", "freq_low_hz", "band_low_hz", "interval_low_hz"))
            f_high = _first_number(raw, ("f_high", "freq_high_hz", "band_high_hz", "interval_high_hz"))
            label = str(raw.get("interval_label", raw.get("label", "")) or "")
            source = str(raw.get("source", source) or source)
            level_db = _first_number(raw, ("level_db", "peak_db", "level_dba"))
            prominence_db = _first_number(raw, ("prominence_db",))
        else:
            try:
                frequency_hz = float(raw)
            except (TypeError, ValueError):
                continue
            f_low = None
            f_high = None
            label = f"Tone {order + 1}"

        if frequency_hz is None and f_low is not None and f_high is not None:
            frequency_hz = 0.5 * (f_low + f_high)
        if frequency_hz is None or not np.isfinite(frequency_hz) or frequency_hz <= 0:
            continue

        half_width_hz = max(float(search_width_hz or 0.0), 0.0) / 2.0
        if f_low is None or not np.isfinite(f_low):
            f_low = frequency_hz - half_width_hz
        if f_high is None or not np.isfinite(f_high):
            f_high = frequency_hz + half_width_hz
        if f_high < f_low:
            f_low, f_high = f_high, f_low

        normalized.append(
            {
                "order": int(order),
                "frequency_hz": float(frequency_hz),
                "f_low": float(max(f_low, 0.0)),
                "f_high": float(max(f_high, 0.0)),
                "interval_label": label,
                "source": source,
                "level_db": level_db,
                "prominence_db": prominence_db,
            }
        )
    return normalized


def _hotspot_from_row(
    row,
    mel_power_pa2,
    mel_db_a,
    mel_axis,
    mel_axis_edges,
    mel_center_freqs_hz=None,
    *,
    kind="mel_band",
):
    mel_value = float(mel_axis[row])
    mel_low = float(mel_axis_edges[row])
    mel_high = float(mel_axis_edges[row + 1])
    if mel_center_freqs_hz is not None and len(mel_center_freqs_hz) > row:
        freq_hz = float(mel_center_freqs_hz[row])
    else:
        freq_hz = float(mel_to_hz(mel_value))
    band_power_pa2 = np.nanmean(mel_power_pa2, axis=1)
    return {
        "kind": kind,
        "mel": mel_value,
        "mel_low": mel_low,
        "mel_high": mel_high,
        "freq_hz": freq_hz,
        "freq_low_hz": float(mel_to_hz(mel_low)),
        "freq_high_hz": float(mel_to_hz(mel_high)),
        "level_dba": float(_power_to_db(band_power_pa2[row])),
        "peak_level_dba": float(np.nanmax(mel_db_a[row])),
        "aggregation": "mean_over_time",
        "mel_band_index": int(row),
    }


def _find_hotspot(mel_power_pa2, mel_db_a, mel_axis, mel_axis_edges, mel_center_freqs_hz=None):
    if mel_db_a.size == 0:
        return None

    band_power_pa2 = np.nanmean(mel_power_pa2, axis=1)
    row = int(np.nanargmax(band_power_pa2))
    return _hotspot_from_row(row, mel_power_pa2, mel_db_a, mel_axis, mel_axis_edges, mel_center_freqs_hz)


def _find_main_tone_hotspots(mel_power_pa2, mel_db_a, mel_axis, mel_axis_edges, mel_center_freqs_hz, main_tones):
    if mel_db_a.size == 0 or not main_tones:
        return []

    band_power_pa2 = np.nanmean(mel_power_pa2, axis=1)
    analysis_low_hz = float(mel_to_hz(mel_axis_edges[0]))
    analysis_high_hz = float(mel_to_hz(mel_axis_edges[-1]))
    hotspots = []
    for tone in main_tones:
        tone_freq = float(tone["frequency_hz"])
        search_low_hz = max(float(tone.get("f_low", tone_freq)), analysis_low_hz)
        search_high_hz = min(float(tone.get("f_high", tone_freq)), analysis_high_hz)
        if search_high_hz < search_low_hz:
            search_low_hz = search_high_hz = min(max(tone_freq, analysis_low_hz), analysis_high_hz)

        search_low_mel = float(hz_to_mel(search_low_hz))
        search_high_mel = float(hz_to_mel(search_high_hz))
        row_mask = (mel_axis_edges[:-1] <= search_high_mel) & (mel_axis_edges[1:] >= search_low_mel)
        rows = np.flatnonzero(row_mask)
        if rows.size:
            local_power = band_power_pa2[rows]
            row = int(rows[int(np.nanargmax(local_power))])
        else:
            row = int(np.nanargmin(np.abs(np.asarray(mel_center_freqs_hz, dtype=float) - tone_freq)))

        hotspot = _hotspot_from_row(
            row,
            mel_power_pa2,
            mel_db_a,
            mel_axis,
            mel_axis_edges,
            mel_center_freqs_hz,
            kind="main_tone_mel_band",
        )
        hotspot.update(
            {
                "source": str(tone.get("source", "configured_main_tone") or "configured_main_tone"),
                "main_tone_order": int(tone.get("order", len(hotspots))),
                "main_tone_frequency_hz": tone_freq,
                "main_tone_label": str(tone.get("interval_label", "") or ""),
                "main_tone_band_low_hz": float(tone.get("f_low", tone_freq)),
                "main_tone_band_high_hz": float(tone.get("f_high", tone_freq)),
                "fft_interval_label": str(tone.get("interval_label", "") or ""),
                "fft_band_low_hz": float(tone.get("f_low", tone_freq)),
                "fft_band_high_hz": float(tone.get("f_high", tone_freq)),
                "fft_level_db": tone.get("level_db"),
                "fft_prominence_db": tone.get("prominence_db"),
                "search_freq_low_hz": float(search_low_hz),
                "search_freq_high_hz": float(search_high_hz),
                "search_mel_low": float(search_low_mel),
                "search_mel_high": float(search_high_mel),
            }
        )
        hotspots.append(hotspot)
    return hotspots


def compute_mel_spectrogram(audio_signal, sample_rate, config=None, v2pa_factor=None):
    cfg = default_mel_config()
    if isinstance(config, dict):
        cfg.update({key: value for key, value in config.items() if value is not None})

    pressure_scale_pa_per_sample = _validate_positive("v2pa_factor", v2pa_factor)
    audio = _clean_audio(audio_signal)
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero.")

    fmin_hz = float(cfg.get("fmin_hz", DEFAULT_FMIN_HZ))
    fmax_hz = float(cfg.get("fmax_hz", DEFAULT_FMAX_HZ))
    frame_length_ms = float(cfg.get("frame_length_ms", DEFAULT_FRAME_LENGTH_MS))
    frame_shift_ms = float(cfg.get("frame_shift_ms", DEFAULT_FRAME_SHIFT_MS))
    window = str(cfg.get("window", DEFAULT_WINDOW) or DEFAULT_WINDOW)
    stft_nfft = int(cfg.get("stft_nfft", DEFAULT_STFT_NFFT))
    n_mels = int(cfg.get("n_mels", DEFAULT_N_MELS))
    main_tone_search_width_hz = float(
        cfg.get("main_tone_search_width_hz", DEFAULT_MAIN_TONE_SEARCH_WIDTH_HZ) or 0.0
    )

    if fmin_hz < 0 or fmax_hz <= fmin_hz:
        raise ValueError("fmin_hz and fmax_hz must define a non-negative ascending frequency range.")
    if fmax_hz > sample_rate / 2.0:
        raise ValueError(f"fmax_hz={fmax_hz:g} exceeds Nyquist frequency {sample_rate / 2.0:g}.")
    if frame_length_ms <= 0 or frame_shift_ms <= 0:
        raise ValueError("frame_length_ms and frame_shift_ms must be greater than zero.")
    if n_mels <= 0:
        raise ValueError("n_mels must be greater than zero.")

    nperseg = int(round(sample_rate * frame_length_ms / 1000.0))
    hop_length = int(round(sample_rate * frame_shift_ms / 1000.0))
    noverlap = nperseg - hop_length

    if nperseg <= 0 or nperseg > audio.size:
        raise ValueError("frame_length_ms produces a frame longer than the audio signal.")
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("frame_shift_ms must be no larger than frame_length_ms.")
    if stft_nfft < nperseg:
        raise ValueError("stft_nfft must be no smaller than the frame length in samples.")

    pressure_pa = audio * pressure_scale_pa_per_sample
    freqs_hz, times_s, psd_pa2_per_hz = signal.spectrogram(
        pressure_pa,
        fs=sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=stft_nfft,
        scaling="density",
        mode="psd",
    )

    freq_mask = (freqs_hz >= fmin_hz) & (freqs_hz <= fmax_hz)
    freqs_hz = freqs_hz[freq_mask]
    psd_pa2_per_hz = psd_pa2_per_hz[freq_mask]

    if freqs_hz.size < 2:
        raise ValueError("Not enough frequency bins remain after filtering.")
    if times_s.size == 0:
        raise ValueError("No time frames were produced by the spectrogram.")

    a_weighting_power = 10.0 ** (_a_weighting_db(freqs_hz) / 10.0)
    psd_a_pa2_per_hz = psd_pa2_per_hz * a_weighting_power[:, np.newaxis]

    bin_widths_hz = _frequency_bin_widths(freqs_hz)
    power_a_pa2 = psd_a_pa2_per_hz * bin_widths_hz[:, np.newaxis]

    mel_filters, mel_edges, hz_edges = _mel_filter_bank(freqs_hz, n_mels, fmin_hz, fmax_hz)
    mel_power_pa2 = mel_filters @ power_a_pa2
    mel_db_a = _power_to_db(mel_power_pa2)

    total_power_pa2_by_frame = np.sum(power_a_pa2, axis=0)
    mean_square_a_pa2 = float(np.mean(total_power_pa2_by_frame))
    overall_spl_dba = float(_power_to_db(mean_square_a_pa2))

    true_mel_axis = mel_edges[1:-1]
    mel_center_freqs_hz = hz_edges[1:-1]
    mel_display_min, mel_display_max = _mel_display_range(cfg.get("mel_scale_range", DEFAULT_MEL_SCALE_RANGE))
    core_low_hz, core_high_hz = _range_pair(cfg.get("core_range_hz", DEFAULT_CORE_RANGE_HZ), DEFAULT_CORE_RANGE_HZ)
    core_low_mel = float(hz_to_mel(core_low_hz))
    core_high_mel = float(hz_to_mel(core_high_hz))
    mel_axis_edges = _mel_channel_edges(true_mel_axis, mel_edges)
    mel_axis = true_mel_axis
    global_hotspot = _find_hotspot(mel_power_pa2, mel_db_a, mel_axis, mel_axis_edges, mel_center_freqs_hz)
    configured_main_tones = cfg.get("main_tones_hz")
    if configured_main_tones is None:
        configured_main_tones = cfg.get("main_tones", cfg.get("dominant_tones"))
    main_tones = _normalize_main_tones(configured_main_tones, main_tone_search_width_hz)
    main_tone_hotspots = _find_main_tone_hotspots(
        mel_power_pa2,
        mel_db_a,
        mel_axis,
        mel_axis_edges,
        mel_center_freqs_hz,
        main_tones,
    )
    hotspot = main_tone_hotspots[0] if main_tone_hotspots else global_hotspot

    params = {
        "sample_rate_hz": sample_rate,
        "pressure_scale_pa_per_sample": pressure_scale_pa_per_sample,
        "calibration_source": "v2pa_factor",
        "v2pa_factor": pressure_scale_pa_per_sample,
        "fmin_hz": fmin_hz,
        "fmax_hz": fmax_hz,
        "frame_length_ms": frame_length_ms,
        "frame_shift_ms": frame_shift_ms,
        "frame_length_samples": nperseg,
        "hop_length_samples": hop_length,
        "window": window,
        "stft_nfft": stft_nfft,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "n_mels": n_mels,
        "main_tones_hz": [float(item["frequency_hz"]) for item in main_tones],
        "main_tone_search_width_hz": float(main_tone_search_width_hz),
        "reference_pressure_pa": REFERENCE_PRESSURE_PA,
        "log_compression": "10*log10(power_pa2/reference_pressure_pa^2)",
        "level_unit": "dB(A)",
        "mel_scale_range": [float(mel_display_min), float(mel_display_max)],
        "mel_display_range": [float(mel_display_min), float(mel_display_max)],
        "analysis_mel_range": [float(mel_edges[0]), float(mel_edges[-1])],
        "analysis_freq_range_hz": [float(fmin_hz), float(fmax_hz)],
        "core_range_hz": [float(core_low_hz), float(core_high_hz)],
        "core_mel_range": [float(core_low_mel), float(core_high_mel)],
        "filter_mel_range": [float(mel_edges[0]), float(mel_edges[-1])],
        "main_tone_source": str(cfg.get("main_tone_source", "configured_main_tones") or "configured_main_tones")
        if main_tones
        else "",
    }

    return {
        "mel_db_a": mel_db_a,
        "times_s": times_s,
        "mel_axis": mel_axis,
        "mel_axis_edges": mel_axis_edges,
        "mel_true_axis": true_mel_axis,
        "mel_center_freqs_hz": mel_center_freqs_hz,
        "mel_freq_edges_hz": hz_edges,
        "overall_spl_dba": float(overall_spl_dba),
        "hotspot": hotspot,
        "global_hotspot": global_hotspot,
        "main_tone_hotspots": main_tone_hotspots,
        "main_tone_hotspot_count": int(len(main_tone_hotspots)),
        "params": params,
    }

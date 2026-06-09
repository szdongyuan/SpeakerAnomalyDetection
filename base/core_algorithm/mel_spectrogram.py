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
DEFAULT_SAMPLE_TO_PA = 0.05
DEFAULT_MIC_SENSITIVITY_V_PER_PA = 0.01
DEFAULT_CORE_RANGE_HZ = (2000.0, 5000.0)


DEFAULT_MEL_CONFIG = {
    "sample_to_pa": DEFAULT_SAMPLE_TO_PA,
    "mic_sensitivity_v_per_pa": DEFAULT_MIC_SENSITIVITY_V_PER_PA,
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


def _find_hotspot(mel_db_a, times_s, mel_axis):
    if mel_db_a.size == 0:
        return None

    row, col = np.unravel_index(np.nanargmax(mel_db_a), mel_db_a.shape)
    mel_value = float(mel_axis[row])
    return {
        "time_s": float(times_s[col]),
        "mel": mel_value,
        "freq_hz": float(mel_to_hz(mel_value)),
        "level_dba": float(mel_db_a[row, col]),
    }


def compute_mel_spectrogram(audio_signal, sample_rate, config=None):
    cfg = default_mel_config()
    if isinstance(config, dict):
        cfg.update({key: value for key, value in config.items() if value is not None})

    sample_to_pa = _validate_positive("sample_to_pa", cfg.get("sample_to_pa"))
    mic_sensitivity_v_per_pa = _validate_positive(
        "mic_sensitivity_v_per_pa",
        cfg.get("mic_sensitivity_v_per_pa"),
    )
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

    if fmin_hz <= 0 or fmax_hz <= fmin_hz:
        raise ValueError("fmin_hz and fmax_hz must define a positive frequency range.")
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

    pressure_pa = audio * sample_to_pa
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
    mel_db_a = 10.0 * np.log10(np.maximum(mel_power_pa2, 1e-30) / REFERENCE_PRESSURE_PA**2)

    total_power_pa2_by_frame = np.sum(power_a_pa2, axis=0)
    mean_square_a_pa2 = float(np.mean(total_power_pa2_by_frame))
    overall_spl_dba = 10.0 * np.log10(max(mean_square_a_pa2, 1e-30) / REFERENCE_PRESSURE_PA**2)

    mel_axis = mel_edges[1:-1]
    hotspot = _find_hotspot(mel_db_a, times_s, mel_axis)

    params = {
        "sample_rate_hz": sample_rate,
        "sample_to_pa": sample_to_pa,
        "mic_sensitivity_v_per_pa": mic_sensitivity_v_per_pa,
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
        "reference_pressure_pa": REFERENCE_PRESSURE_PA,
    }

    return {
        "mel_db_a": mel_db_a,
        "times_s": times_s,
        "mel_axis": mel_axis,
        "mel_freq_edges_hz": hz_edges,
        "overall_spl_dba": float(overall_spl_dba),
        "hotspot": hotspot,
        "params": params,
    }

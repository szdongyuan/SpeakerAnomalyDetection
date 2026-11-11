"""Shared acoustic calculations without plotting or application state."""

from typing import Optional

import numpy as np
from scipy.fftpack import hilbert
from scipy.ndimage import maximum_filter, uniform_filter1d


def compute_cqt(
    y, sr=44100, hop_length=128, n_fft=1024, fmin=None, fmax=None,
    bins_per_octave=None, n_bins=None,
):
    """
    Compute the Constant-Q Transform (CQT) of an audio signal.

    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : number > 0
        Sampling rate of y
    hop_length : int > 0
        Number of samples between frames
    n_fft : int > 0
        window size, used to determine frequency resolution, 为了可以像stft那样使用
    fmin : float > 0
        Minimum frequency
    fmax : float > 0
        Maximum frequency. If None, defaults to the note C9.
    bins_per_octave : int > 0 or None
        Number of bins per octave. If None, calculated based on n_fft.
    n_bins : int > 0 or None
        Total number of CQT bins. If None, calculated based on fmin, fmax and bins_per_octave.

    Returns:
        - C : np.ndarray
            Constant-Q transform of y, a complex number
        - freqs : np.ndarray
            Frequencies corresponding to each bin of CQT
        - times : np.ndarray
            Time points corresponding to each frame of CQT
    """

    import librosa

    if fmin is None:
        fmin = librosa.note_to_hz('C1')  # 32.7 Hz

    if fmax is None:
        fmax = librosa.note_to_hz('C9')

    if bins_per_octave is None:
        # 表示每个八度内有多少频率点，对应的频率对数增加, 增加n_fft会增加频率分辨率，和预期相符
        bins_per_octave = int(12 * np.log2(n_fft / 1024) + 24)
        bins_per_octave = max(12, bins_per_octave)


    if n_bins is None:
        n_octaves = np.log2(fmax / fmin)    # 八度，表示频率区间跨越了多少频率翻倍的区间
        n_bins = int(np.ceil(n_octaves * bins_per_octave))   # n_bins ≈ bins_per_octave * log2(fmax/fmin)，最终的频域采样点数

    # fmax parameter is not supported in librosa.cqt
    C = librosa.cqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave
    )

    freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
    times = librosa.times_like(C, sr=sr, hop_length=hop_length)
    return C, freqs, times


def spl_calculation(
    recorded_signal,
    reference_pressure: float = 20e-6,
    window_size: int = 1201,
    method: str = "rms",
    padding_mode: str = "zero",
    padding_cval: Optional[float] = 0.0,
    v2pa_factor: Optional[float] = None,
    trim_edges: bool = False,
):
    """
        Compute SPL (dB) with a sliding window.

        - method: "rms" | "envelope" | "hilbert" (default "rms")
        - padding_mode: "reflect" | "nearest" | "mirror" | "wrap" | "constant"
          Accepts alias "zero" → "constant"
        - padding_cval: used only when padding_mode == "constant"
    """
    method = str(method).strip().lower()
    padding_mode_in = str(padding_mode).strip().lower()

    alias_to_mode = {
        "zero": "constant",
        "zeros": "constant",
        "const": "constant",
        "constant": "constant",
    }
    valid_modes = {"reflect", "nearest", "mirror", "wrap", "constant"}
    mode = alias_to_mode.get(padding_mode_in, padding_mode_in)
    if mode not in valid_modes:
        mode = "constant"

    valid_methods = {"rms", "envelope", "hilbert"}
    if method not in valid_methods:
        method = "rms"

    if mode == "constant":
        cval = 0.0 if padding_cval is None else float(padding_cval)
    else:
        cval = 0.0

    signal_float = np.asarray(recorded_signal, dtype=float)
    if method == "envelope":
        v_amp_list = maximum_filter(np.abs(signal_float), size=window_size, mode=mode, cval=cval)
    elif method == "hilbert":
        amplitude_envelope = np.abs(hilbert(signal_float))
        v_amp_list = uniform_filter1d(
            amplitude_envelope,
            size=window_size,
            axis=0,
            mode=mode,
            cval=cval
        )
    else:
        v_amp_list = np.sqrt(
            uniform_filter1d(signal_float ** 2, size=window_size, axis=0, mode=mode, cval=cval)
        )

    if not v2pa_factor:
        v2pa_factor = 1
    pa_amp_list = np.asarray(v_amp_list, dtype=float) * v2pa_factor
    pa_amp_list = np.maximum(pa_amp_list, 1.0e-10)    # Avoid log of zero or negative due to numerical issues
    spl = 20 * np.log10(pa_amp_list / float(reference_pressure))

    if not trim_edges:
        return spl

    n = int(signal_float.size)
    w = int(window_size)
    if w <= 1 or w > n:
        return spl
    out_len = n - w + 1
    start = w // 2
    end = start + out_len
    return spl[start:end]

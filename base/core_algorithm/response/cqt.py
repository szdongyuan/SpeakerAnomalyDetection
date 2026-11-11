"""Constant-Q transform calculation without plotting or application state."""

import numpy as np


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

import numpy as np
from typing import Literal, Tuple
from scipy.signal import bilinear, lfilter, filtfilt, freqz, firwin2, butter, sosfiltfilt


def _calibrate_1khz(b: np.ndarray, a: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize the filter to 0 dB (magnitude 1) at 1 kHz.

    Parameters:
    - b, a: IIR/FIR filter coefficients
    - fs: Sample rate (Hz)

    Returns:
    - (b_cal, a): Normalized numerator coefficients and original denominator coefficients
    """
    _, h1k = freqz(b, a, worN=[1000], fs=fs)
    gain = abs(h1k[0])
    return b / gain, a


def A_weighting_filter(fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate discrete coefficients (b, a) for A-weighting filter, calibrated to 0 dB at 1 kHz."""
    "Reference: https://en.wikipedia.org/wiki/A-weighting?utm_source=chatgpt.com"
    f1, f2, f3, f4 = 20.6, 107.7, 737.9, 12194.0
    A1000 = 2.0
    NUM = [(2*np.pi*f4)**2 * (10**(A1000/20.0)), 0, 0, 0, 0]
    DEN = np.convolve([1, 4*np.pi*f4, (2*np.pi*f4)**2], [1, 4*np.pi*f1, (2*np.pi*f1)**2])
    DEN = np.convolve(DEN, [1, 2*np.pi*f2])
    DEN = np.convolve(DEN, [1, 2*np.pi*f3])
    b, a = bilinear(NUM, DEN, fs)
    return _calibrate_1khz(b, a, fs)


def B_weighting_filter(fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate discrete coefficients (b, a) for B-weighting filter, calibrated to 0 dB at 1 kHz."""
    "Reference: https://en.wikipedia.org/wiki/A-weighting?utm_source=chatgpt.com"
    f1, f2, f4 = 20.6, 158.5, 12194.0
    B1000 = 0.17
    NUM = [(2*np.pi*f4)**2 * (10**(B1000/20.0)), 0, 0, 0]
    DEN = np.convolve([1, 4*np.pi*f4, (2*np.pi*f4)**2], [1, 4*np.pi*f1, (2*np.pi*f1)**2])
    DEN = np.convolve(DEN, [1, 2*np.pi*f2])
    b, a = bilinear(NUM, DEN, fs)
    return _calibrate_1khz(b, a, fs)


def C_weighting_filter(fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate discrete coefficients (b, a) for C-weighting filter, calibrated to 0 dB at 1 kHz."""
    "Reference: https://en.wikipedia.org/wiki/A-weighting?utm_source=chatgpt.com"
    f1, f4 = 20.6, 12194.0
    C1000 = 0.06
    NUM = [(2*np.pi*f4)**2 * (10**(C1000/20.0)), 0, 0]
    DEN = np.convolve([1, 4*np.pi*f4, (2*np.pi*f4)**2], [1, 4*np.pi*f1, (2*np.pi*f1)**2])
    b, a = bilinear(NUM, DEN, fs)
    return _calibrate_1khz(b, a, fs)


def _d_weighting_wikipedia_mag(freqs_hz: np.ndarray) -> np.ndarray:
    """
    Calculate target magnitude curve (phase not considered) based on Wikipedia D-weighting formula, normalized at 1 kHz.
    Returns minimum value limited to 1e-12 to avoid numerical issues.
    "Reference: https://en.wikipedia.org/wiki/A-weighting?utm_source=chatgpt.com"
    """
    f = np.asarray(freqs_hz, dtype=float)
    f2 = f * f
    h = ((1037918.48 - f2)**2 + 1080768.16 * f2) / ((9837328 - f2)**2 + 11723776 * f2)
    denom = (f2 + 79919.29) * (f2 + 1345600.0)
    RD = (f / (6.8966888496476e-5)) * np.sqrt(h / denom)
    f0 = 1000.0
    f0_2 = f0 * f0
    h0 = ((1037918.48 - f0_2)**2 + 1080768.16 * f0_2) / ((9837328 - f0_2)**2 + 11723776 * f0_2)
    RD0 = (f0 / (6.8966888496476e-5)) * np.sqrt(h0 / ((f0_2 + 79919.29) * (f0_2 + 1345600.0)))
    mag = RD / RD0
    return np.maximum(mag, 1e-12)


def D_weighting_filter(fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate equivalent FIR filter (b, a) for D-weighting.
    Note: Uses `firwin2` to fit the target magnitude curve; zero phase is ensured by `filtfilt`.
    """
    numtaps = 2049
    f = np.linspace(0.0, fs/2, numtaps)
    mag = _d_weighting_wikipedia_mag(f)
    b = firwin2(numtaps, f/(fs/2), mag, window='hamming')
    a = np.array([1.0])
    return _calibrate_1khz(b, a, fs)

def apply_weighting_filter(
    signal: np.ndarray,
    fs: float,
    weighting: Literal['A', 'B', 'C', 'D', 'Z', 'a', 'b', 'c', 'd', 'z'] = 'A',
    zero_phase: bool = False,
) -> np.ndarray:
    """
    Apply A/B/C/D/Z weighting filter to time-domain signal.

    Parameters:
    - signal: Input 1D or 2D array; for 2D arrays, channels are along columns
    - fs: Sample rate (Hz)
    - weighting: Weighting type (case-insensitive), including A/B/C/D/Z
    - zero_phase: True uses `filtfilt` zero-phase (causes magnitude squared, dB doubled, not suitable for SPL/Leq/dBA calculation);
                  False uses `lfilter` (correct for SPL/Leq/dBA calculation)

    Returns:
    - Weighted signal (same shape as input)

    Note:
    - For SPL / Leq / dBA calculation, must use zero_phase=False (default)
    - filtfilt causes magnitude squared, doubling dB values, which is incorrect
    """
    w = str(weighting).upper()
    if w == 'Z':
        return signal
    elif w == 'A':
        b, a = A_weighting_filter(fs)
    elif w == 'B':
        b, a = B_weighting_filter(fs)
    elif w == 'C':
        b, a = C_weighting_filter(fs)
    elif w == 'D':
        b, a = D_weighting_filter(fs)
    else:
        return signal

    if zero_phase:
        return filtfilt(b, a, signal)
    return lfilter(b, a, signal)




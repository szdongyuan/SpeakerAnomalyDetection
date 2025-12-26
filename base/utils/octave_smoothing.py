"""
Octave-band smoothing utilities for frequency-domain analysis.

Provides functions to smooth data to standard octave-band grid points.
Commonly used for THD, RB, and PRB analysis with chirp signals.
"""
import numpy as np
from typing import Optional, Tuple


def hz_to_octave(freq_hz: np.ndarray, ref_freq: float = 1000.0) -> np.ndarray:
    """
    Convert frequency in Hz to octave units relative to reference frequency.

    Args:
        freq_hz: Frequency values in Hz
        ref_freq: Reference frequency in Hz (default 1000 Hz)

    Returns:
        Octave values relative to reference
    """
    return np.log2(np.asarray(freq_hz, dtype=np.float64) / float(ref_freq))


def octave_to_hz(octave: np.ndarray, ref_freq: float = 1000.0) -> np.ndarray:
    """
    Convert octave units to frequency in Hz.

    Args:
        octave: Octave values relative to reference
        ref_freq: Reference frequency in Hz (default 1000 Hz)

    Returns:
        Frequency values in Hz
    """
    return float(ref_freq) * (2.0 ** np.asarray(octave, dtype=np.float64))


def generate_octave_grid(
    f_min: float,
    f_max: float,
    fraction: int = 6,
    ref_freq: float = 1000.0,
    include_endpoints: bool = True
) -> np.ndarray:
    """
    Generate standard octave-band center frequencies.

    Args:
        f_min: Minimum frequency in Hz
        f_max: Maximum frequency in Hz
        fraction: Octave fraction (1, 3, 6, 12, 24, 48)
        ref_freq: Reference frequency in Hz (default 1000 Hz)
        include_endpoints: If True, ensure f_min and f_max are included (default True)

    Returns:
        Array of octave-band center frequencies in Hz

    Example:
        >>> grid = generate_octave_grid(100, 200, fraction=6)
        >>> len(grid)  # Should be 7 points (100-200Hz is 1 octave, 1/6 oct = 6 intervals + 1)
        7
    """
    if fraction not in [1, 3, 6, 12, 24, 48]:
        raise ValueError(f"Octave fraction must be one of [1, 3, 6, 12, 24, 48], got {fraction}")

    # Grid spacing in octave units
    step = 1.0 / fraction

    if include_endpoints:
        # Generate grid that includes both endpoints
        # Convert f_min and f_max to octave units relative to ref_freq
        oct_min = hz_to_octave(f_min, ref_freq)
        oct_max = hz_to_octave(f_max, ref_freq)

        # Calculate number of steps needed
        n_octaves = oct_max - oct_min
        n_steps = int(np.round(n_octaves / step))

        # Generate grid from f_min to f_max with exact number of steps
        octave_grid = np.linspace(oct_min, oct_max, n_steps + 1)
    else:
        # Original behavior: align to standard grid points
        oct_min = hz_to_octave(f_min, ref_freq)
        oct_max = hz_to_octave(f_max, ref_freq)

        # Round to nearest grid point
        grid_start = np.ceil(oct_min / step) * step
        grid_end = np.floor(oct_max / step) * step

        # Generate grid in octave units
        n_points = int(np.round((grid_end - grid_start) / step)) + 1
        octave_grid = np.linspace(grid_start, grid_end, n_points)

    # Convert back to Hz
    freq_grid = octave_to_hz(octave_grid, ref_freq)

    return freq_grid


def smooth_to_octave_grid(
    frequencies: np.ndarray,
    values: np.ndarray,
    fraction: int = 6,
    ref_freq: float = 1000.0,
    method: str = "linear",
    f_min: Optional[float] = None,
    f_max: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth data to octave-band grid points.

    This function resamples the input data to standard octave-band center frequencies
    using the specified interpolation method.

    Args:
        frequencies: Input frequency array in Hz
        values: Input value array (same length as frequencies)
        fraction: Octave fraction (1, 3, 6, 12, 24, 48)
                 - 1: 1/1 octave (coarse)
                 - 3: 1/3 octave (common in acoustics)
                 - 6: 1/6 octave (recommended for THD/RB/PRB)
                 - 12: 1/12 octave (fine)
                 - 24: 1/24 octave (very fine)
                 - 48: 1/48 octave (extremely fine)
        ref_freq: Reference frequency in Hz (default 1000 Hz)
        method: Interpolation method
                - "linear": Linear interpolation (default)
                - "nearest": Nearest neighbor
                - "log": Logarithmic interpolation (recommended for frequency domain)
        f_min: Minimum frequency for output grid (default: min of input frequencies)
        f_max: Maximum frequency for output grid (default: max of input frequencies)

    Returns:
        Tuple of (smoothed_frequencies, smoothed_values)

    Example:
        >>> freqs = np.array([100, 150, 200, 300, 500, 1000, 2000])
        >>> thd = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
        >>> smooth_freqs, smooth_thd = smooth_to_octave_grid(freqs, thd, fraction=6)
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    if frequencies.shape != values.shape:
        raise ValueError(f"Frequencies and values must have the same shape, got {frequencies.shape} and {values.shape}")

    if len(frequencies) == 0:
        return np.array([]), np.array([])

    # Determine frequency range
    if f_min is None:
        f_min = np.min(frequencies)
    if f_max is None:
        f_max = np.max(frequencies)

    # Generate octave grid (include endpoints to ensure f_min and f_max are covered)
    freq_grid = generate_octave_grid(f_min, f_max, fraction, ref_freq, include_endpoints=True)

    # Interpolate values to octave grid
    if method == "linear":
        smoothed_values = np.interp(freq_grid, frequencies, values)
    elif method == "nearest":
        # Find nearest neighbor for each grid point
        smoothed_values = np.zeros_like(freq_grid)
        for i, f_grid in enumerate(freq_grid):
            idx = np.argmin(np.abs(frequencies - f_grid))
            smoothed_values[i] = values[idx]
    elif method == "log":
        # Logarithmic interpolation (interpolate in log-frequency domain)
        log_freq_input = np.log10(frequencies)
        log_freq_grid = np.log10(freq_grid)
        smoothed_values = np.interp(log_freq_grid, log_freq_input, values)
    else:
        raise ValueError(f"Unknown interpolation method: {method}. Use 'linear', 'nearest', or 'log'")

    return freq_grid, smoothed_values


def get_octave_fraction_name(fraction: int) -> str:
    """
    Get the standard name for an octave fraction.

    Args:
        fraction: Octave fraction (1, 3, 6, 12, 24, 48)

    Returns:
        String name (e.g., "1/6 octave")
    """
    names = {
        1: "1/1 octave",
        3: "1/3 octave",
        6: "1/6 octave",
        12: "1/12 octave",
        24: "1/24 octave",
        48: "1/48 octave"
    }
    return names.get(fraction, f"1/{fraction} octave")

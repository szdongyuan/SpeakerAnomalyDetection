import numpy as np
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d


def smooth(data, window_size, method="savgol", padding_mode="same", **kwargs):
    """
    Smooth the data using the specified method.

    Parameters:
        data: np.ndarray
            The data to smooth.
        window_size: int
            The size of the window to use for smoothing, must be grid, not time.
        method: str
            The method to use for smoothing.
        return: np.ndarray
            The smoothed data. If the method is not supported, return the **original data**.
    """
    method = method.strip().lower()
    padding_mode = padding_mode.strip().lower()

    if method == "mean":
        return np.convolve(data, np.ones(window_size) / window_size, mode=padding_mode)
    if method == "rms":
        return np.sqrt(np.convolve(data ** 2, np.ones(window_size) / window_size, mode=padding_mode))
    elif method == "savgol":
        window_length = int(window_size)
        if window_length < 1:
            return data
        if window_length % 2 == 0:
            window_length += 1
        if window_length > len(data):
            window_length = len(data) if len(data) % 2 == 1 else max(1, len(data) - 1)
            if window_length < 3:
                sigma = kwargs.get("sigma", max(1, int(window_size / 6)))
                return gaussian_filter1d(data, sigma)
        polyorder = min(3, window_length - 1)
        return savgol_filter(data, window_length, polyorder)
    elif method == "median":
        k = int(window_size)
        if k < 1:
            return data
        if k % 2 == 0:
            k += 1
        if k > len(data):
            k = len(data) if len(data) % 2 == 1 else max(1, len(data) - 1)
        return medfilt(data, kernel_size=k)
    elif method == "gaussian":
        sigma = kwargs.get("sigma", max(1, int(window_size / 6)))
        return gaussian_filter1d(data, sigma)
    else:
        return data
    

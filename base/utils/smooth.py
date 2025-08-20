import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


def smooth(data, window_size, method="savgol", **kwargs):
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

    if method == "mean":
        return np.convolve(data, np.ones(window_size)/window_size, mode="valid")
    if method == "rms":
        return np.sqrt(np.convolve(data**2, np.ones(window_size)/window_size, mode="valid"))
    elif method == "savgol":
        return savgol_filter(data, window_size, 3)
    elif method == "median":
        return np.convolve(data, np.ones(window_size)/window_size, mode="valid")
    elif method == "gaussian":
        sigma = kwargs.get("sigma", max(1, int(window_size / 6)))
        return gaussian_filter1d(data, sigma)
    else:
        return data
    

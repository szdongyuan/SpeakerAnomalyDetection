"""Pure display geometry shared by analysis windows and result presentation.

These helpers never change the samples used for metrics, judgement or export.
"""

import numpy as np


def loudness_time_curve(time_s, loudness, *, method, start_s=0.0, end_s=None):
    """Extend segment edges without dropping or replacing measured samples."""
    time_s = np.asarray(time_s, dtype=float)
    loudness = np.asarray(loudness, dtype=float)
    if not (method == 'per_segment' and time_s.size and loudness.size
            and np.isfinite(time_s[0]) and np.isfinite(loudness[0]) and time_s[0] > 0):
        return time_s, loudness
    plot_x = np.insert(time_s, 0, start_s)
    plot_y = np.insert(loudness, 0, loudness[0])
    if end_s is not None and np.isfinite(end_s) and end_s > plot_x[-1]:
        plot_x = np.append(plot_x, end_s)
        plot_y = np.append(plot_y, plot_y[-1])
    return plot_x, plot_y

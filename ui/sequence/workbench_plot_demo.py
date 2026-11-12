"""Deterministic display samples, isolated from analysis tasks and saved results."""

import numpy as np


def simulated_rms_series(segment, settings, station_column, condition_index):
    axis = settings['x_axis']
    low, high = ((0, segment.duration_seconds or 3.0) if axis['auto_range']
                 else (axis['range_min'], axis['range_max']))
    time = np.linspace(low, high, 121)
    phase = np.linspace(0, 6 * np.pi, time.size)
    # These are illustrative RMS-shaped curves, not computed sensor measurements.
    scale = 1.0 if settings['value_unit'] == 'm/s²' else 1.0 / 9.80665
    return [dict(
        wav_channel=channel, name=name, role='curve', x=time,
        y=scale * (0.05 + 0.006 * station_column + 0.008 * condition_index
                   + 0.012 * index + 0.007 * np.sin(phase + index + station_column)
                   + 0.003 * np.sin(2.3 * phase + condition_index)),
    ) for index, (channel, name) in enumerate(segment.channels)]

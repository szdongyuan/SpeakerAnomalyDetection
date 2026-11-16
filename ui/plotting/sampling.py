"""Select original curve points for display; never alter measurement arrays."""

import numpy as np


def select_display_indices(x, y, settings, x_range, width):
    """X is already mapped to the canvas (including logarithmic FFT regions).

    Preserve endpoints and NaN breaks. Pixel buckets retain first, last, minimum
    and maximum in source order, rather than manufacturing an envelope curve.
    Offscreen buckets are collapsed too, so zooming restores visible detail.
    """
    size = len(x)
    method = settings['method']
    if method == 'none' or size < 3:
        return np.arange(size)
    valid = np.isfinite(y)
    boundaries = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    gaps = np.flatnonzero(np.diff(np.r_[False, ~valid, False])).reshape(-1, 2)
    # Two missing endpoints preserve the break without rendering a long NaN run.
    selected = [gaps[:, 0], gaps[:, 1] - 1]
    for start, stop in boundaries.reshape(-1, 2):
        if method == 'keep_every_nth_point':
            step = settings['keep_one_point_every_n_points']
            # Keep the sampling grid anchored to the original array indices.
            selected.append(np.r_[start, np.arange(((start + step - 1) // step) * step, stop, step), stop - 1])
            continue
        low, high = x_range
        bucket_count = max(1, int(width / settings['pixel_bucket_width']))
        pixels = np.floor(np.clip((x[start:stop] - low) / (high - low), -1, 2) * bucket_count)
        pixels = np.clip(pixels, -1, bucket_count)
        edges = np.r_[0, np.flatnonzero(np.diff(pixels)) + 1, stop - start]
        for left, right in zip(edges[:-1] + start, edges[1:] + start):
            values = y[left:right]
            selected.append(np.array([left, right - 1, left + values.argmin(), left + values.argmax()]))
    return np.unique(np.concatenate(selected))

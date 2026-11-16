"""Map display geometry without changing source samples or verdicts."""

import numpy as np


def bin_edges(centres):
    """Convert at least two sample centres to physical cell boundaries."""
    middle = (centres[:-1] + centres[1:]) / 2
    return np.r_[centres[0] - (middle[0] - centres[0]), middle,
                 centres[-1] + (centres[-1] - middle[-1])]


def frequency_bin_geometry(frequency, *, logarithmic=False):
    """Return the first visible row and its physical Hz edges; omit DC on log axes."""
    first = 1 if logarithmic and frequency[0] == 0 else 0
    edges = bin_edges(frequency)[first:]
    edges[0] = max(frequency[first] / 2 if logarithmic else 0, edges[0])
    return first, edges


def clip_curve_x(x, y, bounds, *, log_x=False, log_y=False):
    """Keep boundary intersections and gaps, including nonmonotonic paths.

    Intersections are computed in screen-axis coordinates, as pyqtgraph does
    when joining log-transformed points. Outside excursions remain disconnected.
    """
    low, high = bounds
    inside = (x >= low) & (x <= high)
    if inside.all():
        return x, y
    keep = inside.copy()
    keep[:-1] |= inside[1:]
    keep[1:] |= inside[:-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        screen_x = np.log10(x) if log_x else x
        screen_y = np.log10(y) if log_y else y
    valid = (np.isfinite(screen_x[:-1]) & np.isfinite(screen_x[1:])
             & np.isfinite(screen_y[:-1]) & np.isfinite(screen_y[1:]))
    keys, edge_x, edge_y = [], [], []
    for bound in (low, high):
        crossed = valid & (((x[:-1] < bound) & (x[1:] > bound))
                           | ((x[:-1] > bound) & (x[1:] < bound)))
        indices = np.flatnonzero(crossed)
        if not indices.size:
            continue
        screen_bound = np.log10(bound) if log_x else bound
        fraction = ((screen_bound - screen_x[indices])
                    / (screen_x[indices + 1] - screen_x[indices]))
        values = screen_y[indices] + fraction * (screen_y[indices + 1] - screen_y[indices])
        keys.extend(indices + fraction)
        edge_x.extend(np.full(indices.size, bound))
        edge_y.extend(10. ** values if log_y else values)
        keep[indices] = True
        keep[indices + 1] = True
    indices = np.flatnonzero(keep)
    order = np.argsort(np.r_[indices, keys], kind='stable')
    clipped_x = np.r_[np.clip(x[indices], low, high), edge_x][order]
    clipped_y = np.r_[np.where(inside[indices], y[indices], np.nan), edge_y][order]
    # Outside vertices split re-entry paths, but do not add empty endpoints.
    visible = np.r_[inside[indices], np.ones(len(keys), dtype=bool)][order]
    present = np.flatnonzero(visible)
    if not present.size:
        return x[:0], y[:0]
    span = slice(present[0], present[-1] + 1)
    return clipped_x[span], clipped_y[span]

"""Display-only target-voltage normalization; audio samples remain untouched."""

import numpy as np


def build_stimulus_voltage_preview(source, voltage, voltage_type):
    """Return independent display samples and whether their unit is volts.

    Statistics cover the complete source, including silence and fadeout. Invalid
    metadata or unsafe conversion retains raw amplitude (and nonfinite gaps).
    """
    raw = np.array(source, dtype=float, copy=True)
    try:
        target = float(voltage)
    except (TypeError, ValueError, OverflowError):
        return raw, False
    kind = voltage_type.lower() if isinstance(voltage_type, str) else None
    if not np.isfinite(target) or target < 0 or kind not in {'peak', 'rms'}:
        return raw, False
    if not np.all(np.isfinite(raw)):
        return raw, False
    if raw.size == 0:
        return raw, True
    scale = np.max(np.abs(raw))
    if scale == 0 or target == 0:
        return np.zeros_like(raw), True
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        unit = raw / scale
        if kind == 'rms':
            unit = unit / np.sqrt(np.mean(unit * unit))
        display = unit * target
    if not np.all(np.isfinite(display)):
        return raw, False
    return display, True

"""Golden-result loading and deviation policies shared by analysis paths."""
import json
import os

import numpy as np

from consts.running_consts import DEFAULT_DIR


def resolve_golden_baseline_path(path):
    if not path or not isinstance(path, str):
        return None
    normalized = path.replace("\\", "/").strip()
    if not normalized:
        return None
    if os.path.isabs(normalized):
        return normalized
    return os.path.join(DEFAULT_DIR, normalized).replace("\\", "/")


def load_golden_baseline_result(analysis_config, title_name):
    if not isinstance(analysis_config, dict):
        return None
    resolved = resolve_golden_baseline_path(
        analysis_config.get("golden_sample_result_path"))
    if not resolved or not os.path.exists(resolved):
        return None
    try:
        with open(resolved, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    items = payload.get("items")
    item = items.get(title_name) if isinstance(items, dict) else None
    result = item.get("result") if isinstance(item, dict) else None
    return result if isinstance(result, dict) else None


def golden_deviation_curve(x_current, y_current, x_base, y_base):
    """Match the production signed current-minus-interpolated-base curve."""
    x_current = np.asarray(x_current, dtype=float)
    y_current = np.asarray(y_current, dtype=float)
    x_base = np.asarray(x_base, dtype=float)
    y_base = np.asarray(y_base, dtype=float)
    if (x_current.size == 0 or y_current.size == 0
            or x_base.size == 0 or y_base.size == 0):
        return y_current
    valid = np.isfinite(x_base) & np.isfinite(y_base)
    x_base, y_base = x_base[valid], y_base[valid]
    if x_base.size < 2:
        return y_current
    order = np.argsort(x_base)
    x_base, y_base = x_base[order], y_base[order]
    x_base, unique = np.unique(x_base, return_index=True)
    y_base = y_base[unique]
    if x_base.size < 2:
        return y_current
    baseline = np.interp(x_current, x_base, y_base)
    in_range = ((x_current >= float(np.min(x_base)))
                & (x_current <= float(np.max(x_base))))
    return y_current - np.where(in_range, baseline, np.nan)

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.signal import find_peaks, peak_prominences


@dataclass(frozen=True)
class FrequencyInterval:
    f_low: float
    f_high: float
    label: str


def _format_interval_label(f_low: float, f_high: float) -> str:
    return f"{f_low:.4g}-{f_high:.4g}Hz"


def parse_frequency_intervals(text: str | None) -> list[FrequencyInterval]:
    intervals: list[FrequencyInterval] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line:
            parts = [p.strip() for p in line.split(",") if p.strip()]
        else:
            parts = [p.strip() for p in line.replace("\t", " ").split(" ") if p.strip()]
        if len(parts) == 1 and "-" in parts[0]:
            low_text, high_text = [p.strip() for p in parts[0].split("-", 1)]
            f_low, f_high = float(low_text), float(high_text)
            label = _format_interval_label(f_low, f_high)
        elif len(parts) >= 2:
            f_low, f_high = float(parts[0]), float(parts[1])
            label = " ".join(parts[2:]).strip() if len(parts) >= 3 else _format_interval_label(f_low, f_high)
        else:
            raise ValueError(f"无法解析频率区间: {raw!r}")
        if not (f_low >= 0 and f_high > f_low):
            raise ValueError(f"频率区间上限必须大于下限: {raw!r}")
        intervals.append(FrequencyInterval(f_low=f_low, f_high=f_high, label=label))

    intervals.sort(key=lambda item: item.f_low)
    for index in range(1, len(intervals)):
        if intervals[index].f_low < intervals[index - 1].f_high:
            raise ValueError("频率区间不允许重叠，请检查相邻区间边界。")
    return intervals


def default_interval(frequency: Iterable[float], label: str = "Overall") -> list[FrequencyInterval]:
    freq = np.asarray(list(frequency), dtype=float)
    freq = freq[np.isfinite(freq)]
    freq = freq[freq > 0]
    if freq.size == 0:
        return []
    return [FrequencyInterval(float(np.nanmin(freq)), float(np.nanmax(freq)), label)]


def find_dominant_fft_peaks(
    frequency_hz,
    level_db,
    intervals: list[FrequencyInterval] | None = None,
    *,
    min_prominence_db: float = 3.0,
) -> list[dict]:
    freq = np.asarray(frequency_hz, dtype=float)
    levels = np.asarray(level_db, dtype=float)
    intervals = intervals or default_interval(freq)
    tones: list[dict] = []
    for interval in intervals:
        mask = (
            np.isfinite(freq)
            & np.isfinite(levels)
            & (freq > 0)
            & (freq >= float(interval.f_low))
            & (freq <= float(interval.f_high))
        )
        local_freq = freq[mask]
        local_levels = levels[mask]
        if local_freq.size == 0:
            continue

        peaks, properties = find_peaks(local_levels, prominence=max(float(min_prominence_db), 0.0))
        source = "fft_peak"
        if peaks.size:
            prominences = np.asarray(properties.get("prominences", []), dtype=float)
            if prominences.size == peaks.size:
                peak_order = np.lexsort((prominences, local_levels[peaks]))
                selected_peak = int(peaks[int(peak_order[-1])])
                selected_prominence = float(prominences[int(peak_order[-1])])
            else:
                selected_peak = int(peaks[int(np.nanargmax(local_levels[peaks]))])
                selected_prominence = float(peak_prominences(local_levels, [selected_peak])[0][0])
        else:
            selected_peak = int(np.nanargmax(local_levels))
            selected_prominence = 0.0
            source = "fft_max"

        tones.append(
            {
                "interval_label": interval.label,
                "f_low": float(interval.f_low),
                "f_high": float(interval.f_high),
                "frequency_hz": float(local_freq[selected_peak]),
                "level_db": float(local_levels[selected_peak]),
                "prominence_db": selected_prominence,
                "source": source,
            }
        )
    return tones


def find_dominant_fba_bands(
    bands,
    levels,
    intervals: list[FrequencyInterval] | None = None,
) -> list[dict]:
    levels_arr = np.asarray(levels, dtype=float)
    if intervals is None:
        centers = [float(getattr(band, "f_center", np.nan)) for band in bands]
        intervals = default_interval(centers)

    tones: list[dict] = []
    for interval in intervals:
        indices = []
        for index, band in enumerate(bands):
            f_low = float(getattr(band, "f_low", np.nan))
            f_high = float(getattr(band, "f_high", np.nan))
            if np.isfinite(f_low) and np.isfinite(f_high) and f_high >= interval.f_low and f_low <= interval.f_high:
                indices.append(index)
        if not indices:
            continue
        local_levels = levels_arr[indices]
        finite = np.isfinite(local_levels)
        if not np.any(finite):
            continue
        finite_indices = np.asarray(indices, dtype=int)[finite]
        selected_index = int(finite_indices[int(np.nanargmax(levels_arr[finite_indices]))])
        band = bands[selected_index]
        tones.append(
            {
                "interval_label": interval.label,
                "f_low": float(interval.f_low),
                "f_high": float(interval.f_high),
                "frequency_hz": float(getattr(band, "f_center", np.nan)),
                "level_db": float(levels_arr[selected_index]),
                "source": "fba_band",
                "band_label": str(getattr(band, "label", "")),
                "band_low_hz": float(getattr(band, "f_low", np.nan)),
                "band_high_hz": float(getattr(band, "f_high", np.nan)),
            }
        )
    return tones

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base.core_algorithm.response.dominant_tone_analyzer import (
    find_dominant_fba_bands,
    find_dominant_fft_peaks,
    parse_frequency_intervals,
)


def test_parse_frequency_intervals_accepts_labels_and_rejects_overlap():
    intervals = parse_frequency_intervals("100, 500, Low\n500 2000 Mid")

    assert intervals[0].f_low == 100.0
    assert intervals[0].f_high == 500.0
    assert intervals[0].label == "Low"
    assert intervals[1].label == "Mid"


def test_find_dominant_fft_peaks_returns_one_peak_per_interval():
    freq = np.arange(0, 2001, 10, dtype=float)
    levels = np.zeros_like(freq)
    levels[np.argmin(np.abs(freq - 300))] = 20.0
    levels[np.argmin(np.abs(freq - 1200))] = 30.0
    intervals = parse_frequency_intervals("100, 500, Low\n500, 1800, Mid")

    tones = find_dominant_fft_peaks(freq, levels, intervals, min_prominence_db=3.0)

    assert [tone["interval_label"] for tone in tones] == ["Low", "Mid"]
    assert tones[0]["frequency_hz"] == 300.0
    assert tones[0]["level_db"] == 20.0
    assert tones[0]["source"] == "fft_peak"
    assert tones[1]["frequency_hz"] == 1200.0


def test_find_dominant_fft_peaks_falls_back_to_max_when_no_peak_prominent():
    freq = np.array([100.0, 200.0, 300.0])
    levels = np.array([10.0, 12.0, 11.0])
    intervals = parse_frequency_intervals("100, 300, Flat")

    tones = find_dominant_fft_peaks(freq, levels, intervals, min_prominence_db=20.0)

    assert tones[0]["frequency_hz"] == 200.0
    assert tones[0]["source"] == "fft_max"


def test_find_dominant_fft_peaks_prefers_highest_level_among_prominent_candidates():
    freq = np.array([100.0, 110.0, 120.0, 200.0, 210.0, 220.0, 300.0])
    levels = np.array([45.0, 60.0, 44.0, 10.0, 50.0, 10.0, 20.0])
    intervals = parse_frequency_intervals("100, 300, Low")

    tones = find_dominant_fft_peaks(freq, levels, intervals, min_prominence_db=3.0)

    assert tones[0]["frequency_hz"] == 110.0
    assert tones[0]["level_db"] == 60.0
    assert tones[0]["prominence_db"] < 40.0
    assert tones[0]["source"] == "fft_peak"


def test_find_dominant_fba_bands_uses_highest_band_within_each_interval():
    bands = [
        types.SimpleNamespace(label="100", f_low=80.0, f_high=125.0, f_center=100.0),
        types.SimpleNamespace(label="315", f_low=250.0, f_high=400.0, f_center=315.0),
        types.SimpleNamespace(label="1k", f_low=800.0, f_high=1250.0, f_center=1000.0),
    ]
    levels = np.array([20.0, 35.0, 30.0])
    intervals = parse_frequency_intervals("50, 500, Low\n500, 2000, High")

    tones = find_dominant_fba_bands(bands, levels, intervals)

    assert tones[0]["interval_label"] == "Low"
    assert tones[0]["frequency_hz"] == 315.0
    assert tones[0]["band_label"] == "315"
    assert tones[0]["source"] == "fba_band"
    assert tones[1]["frequency_hz"] == 1000.0

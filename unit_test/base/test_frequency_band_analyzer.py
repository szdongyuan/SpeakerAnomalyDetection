import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base.core_algorithm.response.frequency_band_analyzer import (
    FrequencyBandAnalyzer,
    Threshold,
)


def _sine_wave(frequency_hz: float, *, sample_rate: int = 48000, duration_s: float = 0.5):
    time_s = np.arange(int(sample_rate * duration_s), dtype=np.float64) / sample_rate
    return 0.02 * np.sin(2.0 * np.pi * frequency_hz * time_s)


def _custom_analyzer():
    return FrequencyBandAnalyzer(
        strategy="custom",
        weighting="Z",
        f_min=20,
        f_max=20000,
        custom_edges=[
            (900, 1100, "1 kHz"),
            (2900, 3100, "3 kHz"),
        ],
    )


def test_custom_band_analysis_places_tone_energy_in_matching_band():
    result = _custom_analyzer().analyze(_sine_wave(1000.0), 48000)

    assert [band.label for band in result.bands] == ["1 kHz", "3 kHz"]
    assert result.band_levels_db[0] > result.band_levels_db[1] + 80.0
    assert np.isfinite(result.overall_db)


def test_v2pa_factor_changes_band_and_overall_levels_by_expected_amount():
    signal = _sine_wave(1000.0)
    reference = _custom_analyzer().analyze(signal, 48000, v2pa_factor=1.0)
    doubled = _custom_analyzer().analyze(signal, 48000, v2pa_factor=2.0)
    expected_delta_db = 20.0 * np.log10(2.0)

    assert np.isclose(
        doubled.band_levels_db[0] - reference.band_levels_db[0],
        expected_delta_db,
    )
    assert np.isclose(doubled.overall_db - reference.overall_db, expected_delta_db)


def test_threshold_interpolates_and_marks_only_exceeded_band():
    threshold = Threshold.per_band({100.0: 50.0, 200.0: 70.0})
    assert threshold.get_limit(150.0) == pytest.approx(60.0)

    result = _custom_analyzer().analyze(_sine_wave(1000.0), 48000)
    limit = float(result.band_levels_db[0] - 1.0)
    compared = FrequencyBandAnalyzer.compare_threshold(
        result,
        Threshold.uniform(limit),
        use_weighted=False,
    )

    assert compared.exceeded_bands == [0]
    assert compared.threshold_results[0].exceeded is True
    assert compared.threshold_results[1].exceeded is False


def test_band_above_nyquist_is_reported_as_unavailable():
    analyzer = FrequencyBandAnalyzer(
        strategy="custom",
        weighting="Z",
        custom_edges=[(25000, 26000, "above Nyquist")],
    )

    result = analyzer.analyze(_sine_wave(1000.0), 48000)

    assert np.isnan(result.band_levels_db[0])
    assert np.isnan(result.band_levels_weighted_db[0])


def test_invalid_reference_pressure_is_rejected():
    with pytest.raises(ValueError, match="p_ref must be finite positive"):
        _custom_analyzer().analyze(_sine_wave(1000.0), 48000, p_ref=0.0)

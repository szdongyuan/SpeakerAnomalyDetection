import numpy as np
import pytest
from unittest.mock import Mock

from base.core_algorithm.harmonic_distortion.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer


def _pa_for_spl(spl_db: float) -> float:
    return float(20e-6 * np.power(10.0, float(spl_db) / 20.0))


def _make_spectrum_matrix_for_single_frame(*, sr: int, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        spectrum_matrix: (n_rfft_bins+1, 1) with dummy row 0.
        rfft_freqs: (n_rfft_bins,) frequency axis for reference.
    """
    n_rfft_bins = (int(n_fft) // 2) + 1
    spec = np.zeros((n_rfft_bins + 1, 1), dtype=float)
    freqs = np.fft.rfftfreq(int(n_fft), d=1.0 / float(sr))
    assert freqs.size == n_rfft_bins
    return spec, freqs


def test_prb_delta_specific_zero_when_only_fundamental():
    analyzer = Mock(spec=HarmonicDistortionAnalyzer)
    analyzer.sample_rate = 48000
    from base.core_algorithm.harmonic_distortion.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer as HD
    analyzer.compute_perceptual_thd_batch = HD.compute_perceptual_thd_batch.__get__(analyzer)

    n_fft = 9600  # 5 Hz resolution
    spec, freqs = _make_spectrum_matrix_for_single_frame(sr=analyzer.sample_rate, n_fft=n_fft)
    fund_hz = 200.0
    fund_bin = int(round(fund_hz * n_fft / analyzer.sample_rate))
    # Dummy row means rfft bin k is row k+1.
    spec[fund_bin + 1, 0] = _pa_for_spl(80.0)

    mask = np.zeros_like(spec)
    fundamental_bins = np.array([fund_bin + 1])
    fundamental_freqs = np.array([fund_hz])

    phons = analyzer.compute_perceptual_thd_batch(
        spec,
        mask,
        fundamental_bins,
        fundamental_freqs,
        masking_config={"prb_loudness_method": "delta_specific", "fundamental_neighbor_bins": 1},
        n_fft=n_fft,
    )
    assert phons.shape == (1,)
    assert np.isfinite(phons[0])
    assert phons[0] <= 1.0


def test_prb_delta_specific_increases_with_harmonic_level():
    analyzer = Mock(spec=HarmonicDistortionAnalyzer)
    analyzer.sample_rate = 48000
    from base.core_algorithm.harmonic_distortion.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer as HD
    analyzer.compute_perceptual_thd_batch = HD.compute_perceptual_thd_batch.__get__(analyzer)

    n_fft = 9600
    fund_hz = 200.0
    harm_hz = 1000.0  # 5th harmonic
    fund_bin = int(round(fund_hz * n_fft / analyzer.sample_rate))
    harm_bin = int(round(harm_hz * n_fft / analyzer.sample_rate))

    def run(harm_spl_db: float) -> float:
        spec, _ = _make_spectrum_matrix_for_single_frame(sr=analyzer.sample_rate, n_fft=n_fft)
        spec[fund_bin + 1, 0] = _pa_for_spl(80.0)
        spec[harm_bin + 1, 0] = _pa_for_spl(harm_spl_db)
        mask = np.zeros_like(spec)
        mask[harm_bin + 1, 0] = 1.0
        return float(
            analyzer.compute_perceptual_thd_batch(
                spec,
                mask,
                np.array([fund_bin + 1]),
                np.array([fund_hz]),
                masking_config={"prb_loudness_method": "delta_specific", "fundamental_neighbor_bins": 1},
                n_fft=n_fft,
            )[0]
        )

    low = run(30.0)
    mid = run(50.0)
    high = run(70.0)
    assert 0.0 <= low <= mid <= high


def test_prb_delta_specific_edge_window_avoids_hard_zero_at_high_freq():
    analyzer = Mock(spec=HarmonicDistortionAnalyzer)
    analyzer.sample_rate = 48000
    from base.core_algorithm.harmonic_distortion.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer as HD
    analyzer.compute_perceptual_thd_batch = HD.compute_perceptual_thd_batch.__get__(analyzer)

    n_fft = 9600
    spec, _ = _make_spectrum_matrix_for_single_frame(sr=analyzer.sample_rate, n_fft=n_fft)
    fund_hz = 200.0
    fund_bin = int(round(fund_hz * n_fft / analyzer.sample_rate))
    # Put a strong high-frequency component above 15500 Hz so bark(freq) > 24.
    harm_hz = 20000.0
    harm_bin = int(round(harm_hz * n_fft / analyzer.sample_rate))

    spec[fund_bin + 1, 0] = _pa_for_spl(80.0)
    spec[harm_bin + 1, 0] = _pa_for_spl(100.0)

    mask = np.zeros_like(spec)
    mask[harm_bin + 1, 0] = 1.0

    phons = analyzer.compute_perceptual_thd_batch(
        spec,
        mask,
        np.array([fund_bin + 1]),
        np.array([fund_hz]),
        masking_config={"prb_loudness_method": "delta_specific", "fundamental_neighbor_bins": 1},
        n_fft=n_fft,
    )
    assert phons.shape == (1,)
    assert np.isfinite(phons[0])
    assert phons[0] > 0.0


def test_prb_delta_specific_requires_48khz():
    analyzer = Mock(spec=HarmonicDistortionAnalyzer)
    analyzer.sample_rate = 44100
    from base.core_algorithm.harmonic_distortion.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer as HD
    analyzer.compute_perceptual_thd_batch = HD.compute_perceptual_thd_batch.__get__(analyzer)

    spec = np.zeros((33, 1))
    mask = np.zeros_like(spec)
    with pytest.raises(ValueError, match=">= 48000"):
        analyzer.compute_perceptual_thd_batch(
            spec,
            mask,
            np.array([1]),
            np.array([100.0]),
            masking_config={"prb_loudness_method": "delta_specific"},
            n_fft=64,
        )

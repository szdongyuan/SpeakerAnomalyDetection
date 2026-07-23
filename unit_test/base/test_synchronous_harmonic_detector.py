import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base.core_algorithm.harmonic_distortion.synchronous_harmonic_detector import (
    SynchronousHarmonicDetector,
)


def _signal(sample_rate, sample_count, f0, amplitudes, phases=None, dc=0.0):
    phases = phases or {}
    n = np.arange(sample_count, dtype=np.float64)
    y = np.full(sample_count, float(dc), dtype=np.float64)
    for order, amplitude in amplitudes.items():
        phase = float(phases.get(order, 0.0))
        y += float(amplitude) * np.sin(phase + 2.0 * np.pi * order * f0 * n / sample_rate)
    return y


def test_detector_recovers_nonbin_multiharmonic_signal_with_dc_and_phase():
    sample_rate = 48000
    f0 = 331.7
    amplitudes = {1: 0.8, 2: 0.07, 5: 0.025, 11: 0.012, 35: 0.004}
    phases = {1: 0.4, 2: -1.2, 5: 2.1, 11: -0.7, 35: 1.6}
    segment = _signal(sample_rate, 4096, f0, amplitudes, phases, dc=0.31)

    detector = SynchronousHarmonicDetector()
    measured, distortion = detector.analyze(
        segment,
        f0=f0,
        sample_rate=sample_rate,
        harmonic_orders=[2, 5, 11, 35],
        stft_window_type="hann",
    )

    for order, expected in amplitudes.items():
        assert measured[order] == pytest.approx(expected, rel=1e-8, abs=1e-10)
    expected_distortion = 100.0 * np.sqrt(0.07**2 + 0.025**2 + 0.012**2 + 0.004**2) / 0.8
    assert distortion == pytest.approx(expected_distortion, rel=1e-8, abs=1e-10)


def test_detector_is_phase_invariant_for_multiharmonic_signals():
    sample_rate = 48000
    f0 = 455.3
    amplitudes = {1: 1.2, 2: 0.08, 7: 0.03, 10: 0.015}
    phase_sets = [
        {1: 0.0, 2: 0.0, 7: 0.0, 10: 0.0},
        {1: 0.8, 2: -0.2, 7: 1.4, 10: -2.1},
        {1: -1.3, 2: 2.7, 7: -0.9, 10: 0.5},
    ]

    detector = SynchronousHarmonicDetector()
    reference_distortion = None
    for phases in phase_sets:
        segment = _signal(sample_rate, 5000, f0, amplitudes, phases, dc=-0.17)
        measured, distortion = detector.analyze(
            segment,
            f0=f0,
            sample_rate=sample_rate,
            harmonic_orders=[2, 7, 10],
            stft_window_type="hann",
        )

        for order, expected in amplitudes.items():
            assert measured[order] == pytest.approx(expected, rel=1e-8, abs=1e-10)
        if reference_distortion is None:
            reference_distortion = distortion
        else:
            assert distortion == pytest.approx(reference_distortion, rel=1e-8, abs=1e-10)


def test_detector_suppresses_low_order_leakage_for_rb_orders():
    sample_rate = 48000
    f0 = 217.9
    amplitudes = {order: 0.55 / order for order in range(1, 10)}
    segment = _signal(sample_rate, 8192, f0, amplitudes, dc=0.09)

    _, distortion = SynchronousHarmonicDetector().analyze(
        segment,
        f0=f0,
        sample_rate=sample_rate,
        harmonic_orders=list(range(10, 36)),
        stft_window_type="hann",
    )

    assert distortion <= 1e-8


def test_duplicate_selected_orders_are_counted_once():
    sample_rate = 48000
    f0 = 613.2
    amplitudes = {1: 1.0, 2: 0.1, 3: 0.2}
    segment = _signal(sample_rate, 4096, f0, amplitudes, dc=0.03)

    _, distortion = SynchronousHarmonicDetector().analyze(
        segment,
        f0=f0,
        sample_rate=sample_rate,
        harmonic_orders=[2, 2, 3],
        stft_window_type="hann",
    )

    expected = 100.0 * np.sqrt(0.1**2 + 0.2**2) / 1.0
    assert distortion == pytest.approx(expected, rel=1e-8, abs=1e-10)


def test_legal_selected_orders_above_nyquist_are_ignored():
    sample_rate = 48000
    f0 = 15000.0
    segment = _signal(sample_rate, 128, f0, {1: 0.75}, phases={1: 0.4}, dc=0.2)

    measured, distortion = SynchronousHarmonicDetector().analyze(
        segment,
        f0=f0,
        sample_rate=sample_rate,
        harmonic_orders=[2, 3, 35],
        stft_window_type="hann",
    )

    assert set(measured) == {1}
    assert measured[1] == pytest.approx(0.75, rel=1e-8, abs=1e-10)
    assert distortion == 0.0


@pytest.mark.parametrize("f0", [0.0, -1.0, 24000.0, 48000.0, np.inf, np.nan])
def test_invalid_f0_raises_value_error(f0):
    with pytest.raises(ValueError):
        SynchronousHarmonicDetector().analyze(
            np.ones(64),
            f0=f0,
            sample_rate=48000,
            harmonic_orders=[2],
            stft_window_type="hann",
        )


@pytest.mark.parametrize("order", [1, 0, 36, 2.5, True, False])
def test_invalid_selected_orders_raise_value_error(order):
    with pytest.raises(ValueError):
        SynchronousHarmonicDetector().analyze(
            np.ones(64),
            f0=1000.0,
            sample_rate=48000,
            harmonic_orders=[order],
            stft_window_type="hann",
        )


@pytest.mark.parametrize(
    "window",
    [
        np.ones(63),
        np.array([1.0, np.nan] + [1.0] * 62),
        np.zeros(64),
    ],
)
def test_invalid_explicit_windows_raise_value_error(window):
    with pytest.raises(ValueError):
        SynchronousHarmonicDetector().analyze(
            np.ones(64),
            f0=1000.0,
            sample_rate=48000,
            harmonic_orders=[2],
            stft_window_type=window,
        )


@pytest.mark.parametrize("window", ["hann", ("kaiser", 8.0), np.hanning(4096)])
def test_valid_string_tuple_and_explicit_windows_work(window):
    sample_rate = 48000
    f0 = 389.4
    amplitudes = {1: 0.9, 2: 0.04}
    segment = _signal(sample_rate, 4096, f0, amplitudes, phases={1: 0.3, 2: -0.8}, dc=-0.11)

    measured, distortion = SynchronousHarmonicDetector().analyze(
        segment,
        f0=f0,
        sample_rate=sample_rate,
        harmonic_orders=[2],
        stft_window_type=window,
    )

    assert measured[1] == pytest.approx(0.9, rel=1e-8, abs=1e-10)
    assert measured[2] == pytest.approx(0.04, rel=1e-8, abs=1e-10)
    assert distortion == pytest.approx(100.0 * 0.04 / 0.9, rel=1e-8, abs=1e-10)


def test_underdetermined_input_returns_finite_minimum_norm_result():
    measured, distortion = SynchronousHarmonicDetector().analyze(
        np.array([0.5], dtype=np.float64),
        f0=1000.0,
        sample_rate=48000,
        harmonic_orders=[2, 3],
        stft_window_type="hann",
    )

    assert measured
    assert all(np.isfinite(value) for value in measured.values())
    assert np.isfinite(distortion)

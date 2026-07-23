import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base.core_algorithm.harmonic_distortion.step_signal_hd import StepSignalHD
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from base.stimulus_signal.frequency_stepped import generate_frequency_stepped
from consts.harmonic_detection_consts import (
    HARMONIC_DETECTION_METHOD_FOURIER,
    HARMONIC_DETECTION_METHOD_KEY,
    HARMONIC_DETECTION_METHOD_SYNCHRONOUS,
)


def _tone(sample_rate, sample_count, f0, amplitudes, phase=0.0, dc=0.0):
    n = np.arange(sample_count, dtype=np.float64)
    y = np.full(sample_count, dc, dtype=np.float64)
    for order, amplitude in amplitudes.items():
        y += amplitude * np.sin(phase + 2.0 * np.pi * order * f0 * n / sample_rate)
    return y


def _frequency_stepped_recording(sample_rate, frequencies, harmonic_amplitudes_by_step):
    generated = generate_frequency_stepped(
        sample_rate=sample_rate,
        repeat_times=1,
        min_duration=0.012,
        min_cycles=8,
        frequency_mode="custom_linear",
        frequencies=frequencies,
        generate_waveform=False,
    )
    recording = np.zeros(generated.metadata["alignment_sample_count"], dtype=float)
    for segment in generated.segments:
        recording[segment.start_sample:segment.end_sample] = _tone(
            sample_rate,
            segment.sample_count,
            segment.frequency_hz,
            harmonic_amplitudes_by_step[segment.step_index],
            phase=0.2 * (segment.step_index + 1),
            dc=0.05,
        )
    return generated.metadata, recording


def test_ordinary_step_uses_synchronous_fit_and_averages_percentages_per_repetition():
    sample_rate = 48000
    metadata = {
        "stimulus_method": "steps",
        "stimulus_type": "linear",
        "start_freq": 997.3,
        "stop_freq": 997.3,
        "num_steps": 1,
        "repeat_times": 2,
        "total_time": 0.08,
    }
    step_samples = int(sample_rate * metadata["total_time"] / metadata["repeat_times"])
    first = _tone(sample_rate, step_samples, 997.3, {1: 1.0, 2: 0.10}, phase=0.3, dc=0.2)
    second = _tone(sample_rate, step_samples, 997.3, {1: 2.0, 2: 0.40}, phase=-1.1, dc=-0.4)
    recording = np.concatenate([first, second])

    x, h, thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        recording,
        sample_rate,
        {"stimulus_metadata": metadata, "harmonic_orders": [2], "stft_window_type": "hann"},
    )

    assert x.tolist() == pytest.approx([997.3])
    assert h.shape == (6, 1)
    assert h[0, 0] == pytest.approx(997.3)
    assert h[1, 0] == pytest.approx(1.5, rel=1e-8, abs=1e-10)
    assert h[2, 0] == pytest.approx(0.25, rel=1e-8, abs=1e-10)
    assert thd[0] == pytest.approx(15.0, rel=1e-8, abs=1e-10)


def test_ordinary_step_rb_does_not_leak_low_order_harmonics():
    sample_rate = 48000
    metadata = {
        "stimulus_method": "steps",
        "stimulus_type": "linear",
        "start_freq": 217.9,
        "stop_freq": 217.9,
        "num_steps": 1,
        "repeat_times": 1,
        "total_time": 0.2,
    }
    step_samples = int(sample_rate * metadata["total_time"])
    recording = _tone(
        sample_rate,
        step_samples,
        217.9,
        {order: 0.55 / order for order in range(1, 10)},
        phase=0.41,
        dc=0.09,
    )

    _, _, thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        recording,
        sample_rate,
        {"stimulus_metadata": metadata, "harmonic_orders": list(range(10, 36)), "stft_window_type": "hann"},
    )

    assert thd[0] <= 1e-8


def test_ordinary_step_result_shape_is_preserved_for_two_linear_frequencies():
    sample_rate = 48000
    metadata = {
        "stimulus_method": "steps",
        "stimulus_type": "linear",
        "start_freq": 997.3,
        "stop_freq": 1500.5,
        "num_steps": 2,
        "repeat_times": 1,
        "total_time": 0.08,
    }
    step_samples = int(sample_rate * metadata["total_time"] / metadata["num_steps"])
    recording = np.concatenate(
        [
            _tone(sample_rate, step_samples, 997.3, {1: 1.0, 2: 0.04}, phase=0.2, dc=0.01),
            _tone(sample_rate, step_samples, 1500.5, {1: 0.8, 3: 0.08}, phase=-0.7, dc=-0.02),
        ]
    )

    x, h, thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        recording,
        sample_rate,
        {"stimulus_metadata": metadata, "harmonic_orders": [2, 3], "stft_window_type": "hann"},
    )

    assert x.tolist() == pytest.approx([997.3, 1500.5])
    assert h.shape == (6, 2)
    assert h[0].tolist() == pytest.approx([997.3, 1500.5])
    assert thd.tolist() == pytest.approx([4.0, 10.0], rel=1e-8, abs=1e-10)


def test_ordinary_step_h_rows_use_fitted_amplitudes_and_zero_nyquist_exclusions():
    sample_rate = 48000
    metadata = {
        "stimulus_method": "steps",
        "stimulus_type": "linear",
        "start_freq": 13000.0,
        "stop_freq": 13000.0,
        "num_steps": 1,
        "repeat_times": 1,
        "total_time": 0.04,
    }
    step_samples = int(sample_rate * metadata["total_time"])
    recording = _tone(sample_rate, step_samples, 13000.0, {1: 0.7}, phase=0.8, dc=0.05)

    x, h, thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        recording,
        sample_rate,
        {"stimulus_metadata": metadata, "harmonic_orders": [2, 3, 4, 5], "stft_window_type": "hann"},
    )

    assert x.tolist() == pytest.approx([13000.0])
    assert h.shape == (6, 1)
    assert h[1, 0] == pytest.approx(0.7, rel=1e-8, abs=1e-10)
    np.testing.assert_array_equal(h[2:6, 0], np.zeros(4))
    assert thd[0] == 0.0


@pytest.mark.parametrize("harmonic_orders", [[1], [0], [36], [2.5], [True]])
def test_ordinary_step_invalid_selected_harmonic_orders_propagate_value_error(harmonic_orders):
    sample_rate = 48000
    metadata = {
        "stimulus_method": "steps",
        "stimulus_type": "linear",
        "start_freq": 1000.0,
        "stop_freq": 1000.0,
        "num_steps": 1,
        "repeat_times": 1,
        "total_time": 0.04,
    }
    recording = _tone(sample_rate, int(sample_rate * metadata["total_time"]), 1000.0, {1: 1.0})

    with pytest.raises(ValueError):
        AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
            recording,
            sample_rate,
            {"stimulus_metadata": metadata, "harmonic_orders": harmonic_orders},
        )


@pytest.mark.parametrize("harmonic_orders", [[1], [0], [36], [2.5], [True]])
def test_ordinary_step_fourier_invalid_selected_harmonic_orders_raise_value_error(harmonic_orders):
    sample_rate = 48000
    metadata = {
        "stimulus_method": "steps",
        "stimulus_type": "linear",
        "start_freq": 1000.0,
        "stop_freq": 1000.0,
        "num_steps": 1,
        "repeat_times": 1,
        "total_time": 0.04,
    }
    recording = _tone(sample_rate, int(sample_rate * metadata["total_time"]), 1000.0, {1: 1.0})

    with pytest.raises(ValueError):
        AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
            recording,
            sample_rate,
            {
                "stimulus_metadata": metadata,
                "harmonic_orders": harmonic_orders,
                HARMONIC_DETECTION_METHOD_KEY: HARMONIC_DETECTION_METHOD_FOURIER,
            },
        )


def test_ordinary_step_explicit_synchronous_does_not_use_fourier_thd_batch(monkeypatch):
    from base.core_algorithm.harmonic_distortion.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer

    def fail_fourier(self, *args, **kwargs):
        raise AssertionError("synchronous ordinary step must not call Fourier compute_thd_batch")

    monkeypatch.setattr(HarmonicDistortionAnalyzer, "compute_thd_batch", fail_fourier)

    sample_rate = 48000
    metadata = {
        "stimulus_method": "steps",
        "stimulus_type": "linear",
        "start_freq": 997.3,
        "stop_freq": 997.3,
        "num_steps": 1,
        "repeat_times": 1,
        "total_time": 0.04,
    }
    recording = _tone(sample_rate, int(sample_rate * metadata["total_time"]), 997.3, {1: 1.0, 2: 0.1})

    _, h, thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        recording,
        sample_rate,
        {
            "stimulus_metadata": metadata,
            "harmonic_orders": [2],
            HARMONIC_DETECTION_METHOD_KEY: HARMONIC_DETECTION_METHOD_SYNCHRONOUS,
        },
    )

    assert h[1, 0] == pytest.approx(1.0, rel=1e-8, abs=1e-10)
    assert thd[0] == pytest.approx(10.0, rel=1e-8, abs=1e-10)


def test_ordinary_step_fourier_uses_stft_nearest_bin_and_not_synchronous_detector(monkeypatch):
    from base.core_algorithm.harmonic_distortion.synchronous_harmonic_detector import SynchronousHarmonicDetector

    def fail_sync(self, *args, **kwargs):
        raise AssertionError("fourier ordinary step must not call synchronous detector")

    monkeypatch.setattr(SynchronousHarmonicDetector, "analyze", fail_sync)

    sample_rate = 48000
    metadata = {
        "stimulus_method": "steps",
        "stimulus_type": "linear",
        "start_freq": 1000.0,
        "stop_freq": 1000.0,
        "num_steps": 1,
        "repeat_times": 1,
        "total_time": 0.05,
    }
    recording = _tone(sample_rate, int(sample_rate * metadata["total_time"]), 1000.0, {1: 1.0, 2: 0.1})

    x, h, thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        recording,
        sample_rate,
        {
            "stimulus_metadata": metadata,
            "harmonic_orders": [2],
            "stft_window_type": "hann",
            HARMONIC_DETECTION_METHOD_KEY: HARMONIC_DETECTION_METHOD_FOURIER,
        },
    )

    assert x.tolist() == pytest.approx([1000.0])
    assert h.shape == (6, 1)
    assert h[0, 0] == pytest.approx(1000.0)
    assert h[1, 0] > 0.0
    assert h[2, 0] / h[1, 0] == pytest.approx(0.1, rel=1e-6, abs=1e-8)
    assert thd[0] == pytest.approx(10.0, rel=1e-6, abs=1e-8)


def test_step_signal_hd_fourier_requires_harmonic_mask():
    sample_rate = 48000
    metadata = {
        "stimulus_method": "steps",
        "stimulus_type": "linear",
        "start_freq": 1000.0,
        "stop_freq": 1000.0,
        "num_steps": 1,
        "repeat_times": 1,
        "total_time": 0.04,
    }
    recording = _tone(sample_rate, int(sample_rate * metadata["total_time"]), 1000.0, {1: 1.0, 2: 0.1})

    with pytest.raises(ValueError, match="requires a harmonic mask"):
        StepSignalHD(sample_rate=sample_rate).compute_distortion(
            recording,
            metadata,
            [2],
            harmonic_mask=None,
            harmonic_detection_method=HARMONIC_DETECTION_METHOD_FOURIER,
        )


def test_ordinary_step_fourier_and_synchronous_differ_for_nonbin_tone():
    sample_rate = 48000
    metadata = {
        "stimulus_method": "steps",
        "stimulus_type": "linear",
        "start_freq": 997.3,
        "stop_freq": 997.3,
        "num_steps": 1,
        "repeat_times": 1,
        "total_time": 0.04,
    }
    recording = _tone(sample_rate, int(sample_rate * metadata["total_time"]), 997.3, {1: 1.0, 2: 0.1}, dc=0.03)

    _, _, sync_thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        recording,
        sample_rate,
        {"stimulus_metadata": metadata, "harmonic_orders": [2], HARMONIC_DETECTION_METHOD_KEY: "synchronous"},
    )
    _, _, fourier_thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        recording,
        sample_rate,
        {"stimulus_metadata": metadata, "harmonic_orders": [2], HARMONIC_DETECTION_METHOD_KEY: "fourier"},
    )

    assert sync_thd[0] == pytest.approx(10.0, rel=1e-8, abs=1e-10)
    assert fourier_thd[0] != pytest.approx(sync_thd[0], rel=1e-3, abs=1e-3)


def test_standard_step_invalid_direct_detection_method_raises_value_error():
    sample_rate = 48000
    metadata = {
        "stimulus_method": "steps",
        "stimulus_type": "linear",
        "start_freq": 1000.0,
        "stop_freq": 1000.0,
        "num_steps": 1,
        "repeat_times": 1,
        "total_time": 0.04,
    }
    recording = _tone(sample_rate, int(sample_rate * metadata["total_time"]), 1000.0, {1: 1.0})

    with pytest.raises(ValueError, match="Unsupported harmonic detection method"):
        AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
            recording,
            sample_rate,
            {"stimulus_metadata": metadata, "harmonic_orders": [2], HARMONIC_DETECTION_METHOD_KEY: "bad"},
        )


def test_step_signal_hd_stft_helper_remains_available_for_perceptual_consumers():
    sample_rate = 48000
    segment = _tone(sample_rate, 128, 1000.0, {1: 1.0})

    spectrum = StepSignalHD(sample_rate=sample_rate)._compute_stft(segment, 128, 128, "hann")

    assert spectrum.shape == (65, 1)
    assert np.all(np.isfinite(spectrum))


def test_chirp_hd_still_uses_chirp_stft_path(monkeypatch):
    from base.core_algorithm.harmonic_distortion.chirp_signal_hd import ChirpSignalHD
    from base.core_algorithm.harmonic_distortion.synchronous_harmonic_detector import SynchronousHarmonicDetector

    called = {"chirp": False, "sync": False}

    def fake_chirp(self, *args, **kwargs):
        called["chirp"] = True
        return {"frequencies": np.array([1000.0]), "thd": np.array([0.0]), "times": np.array([0.0])}

    def fake_sync(self, *args, **kwargs):
        called["sync"] = True
        raise AssertionError("chirp HD/RB must not call synchronous detector")

    monkeypatch.setattr(ChirpSignalHD, "compute_distortion", fake_chirp)
    monkeypatch.setattr(SynchronousHarmonicDetector, "analyze", fake_sync)

    metadata = {
        "stimulus_method": "chirps",
        "start_freq": 100,
        "stop_freq": 1000,
        "repeat_times": 1,
        "total_time": 0.1,
        "stimulus_type": "linear",
    }
    x, h, thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        np.zeros(4800, dtype=float),
        48000,
        {
            "stimulus_metadata": metadata,
            "harmonic_orders": [2],
            "stft_window_size": 256,
            "stft_hop_size": 128,
            HARMONIC_DETECTION_METHOD_KEY: HARMONIC_DETECTION_METHOD_SYNCHRONOUS,
        },
    )

    assert called == {"chirp": True, "sync": False}
    assert x.tolist() == [1000.0]
    assert h.shape == (6, 1)
    assert thd.tolist() == [0.0]


def test_chirp_hd_ignores_fourier_selector_and_stays_on_chirp_path(monkeypatch):
    from base.core_algorithm.harmonic_distortion.chirp_signal_hd import ChirpSignalHD
    from base.core_algorithm.harmonic_distortion.synchronous_harmonic_detector import SynchronousHarmonicDetector

    called = {"chirp": False, "sync": False}

    def fake_chirp(self, *args, **kwargs):
        called["chirp"] = True
        return {"frequencies": np.array([1000.0]), "thd": np.array([0.0]), "times": np.array([0.0])}

    def fake_sync(self, *args, **kwargs):
        called["sync"] = True
        raise AssertionError("chirp HD/RB must not call synchronous detector")

    monkeypatch.setattr(ChirpSignalHD, "compute_distortion", fake_chirp)
    monkeypatch.setattr(SynchronousHarmonicDetector, "analyze", fake_sync)

    metadata = {
        "stimulus_method": "chirps",
        "start_freq": 100,
        "stop_freq": 1000,
        "repeat_times": 1,
        "total_time": 0.1,
        "stimulus_type": "linear",
    }
    x, h, thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        np.zeros(4800, dtype=float),
        48000,
        {
            "stimulus_metadata": metadata,
            "harmonic_orders": [2],
            "stft_window_size": 256,
            "stft_hop_size": 128,
            HARMONIC_DETECTION_METHOD_KEY: HARMONIC_DETECTION_METHOD_FOURIER,
        },
    )

    assert called == {"chirp": True, "sync": False}
    assert x.tolist() == [1000.0]
    assert h.shape == (6, 1)
    assert thd.tolist() == [0.0]


def test_ordinary_step_prb_stays_on_perceptual_stft_path(monkeypatch):
    from base.core_algorithm.harmonic_distortion.perceptual_step_signal_hd import PerceptualStepSignalHD
    from base.core_algorithm.harmonic_distortion.step_signal_hd import StepSignalHD
    from base.core_algorithm.harmonic_distortion.synchronous_harmonic_detector import SynchronousHarmonicDetector

    called = {"perceptual": False, "fourier": False, "sync": False}

    def fake_perceptual(self, recorded_signal, stimulus_metadata, harmonic_orders, harmonic_mask=None, **kwargs):
        called["perceptual"] = True
        assert harmonic_mask is not None
        return {
            "frequencies": np.array([1000.0]),
            "perceptual_loudness": np.array([12.5]),
            "num_repetitions": 1,
            "spectrum_matrix": np.zeros((2, 1), dtype=float),
        }

    def fake_sync(self, *args, **kwargs):
        called["sync"] = True
        raise AssertionError("ordinary step PRB must not call synchronous detector")

    def fake_fourier(self, *args, **kwargs):
        called["fourier"] = True
        raise AssertionError("ordinary step PRB must not call standard Fourier compute_thd_batch")

    monkeypatch.setattr(PerceptualStepSignalHD, "compute_distortion", fake_perceptual)
    monkeypatch.setattr(StepSignalHD, "compute_thd_batch", fake_fourier)
    monkeypatch.setattr(SynchronousHarmonicDetector, "analyze", fake_sync)

    metadata = {
        "stimulus_method": "steps",
        "stimulus_type": "linear",
        "start_freq": 1000.0,
        "stop_freq": 1000.0,
        "num_steps": 1,
        "repeat_times": 1,
        "total_time": 0.04,
    }
    x, harmonic, loudness = AudioThdFrequencyResponseAnalysis().calculate_perceptual_thd_three_phase(
        np.zeros(1920, dtype=float),
        48000,
        {"stimulus_metadata": metadata, "harmonic_orders": [10]},
        v2pa_factor=1.0,
    )

    assert called == {"perceptual": True, "fourier": False, "sync": False}
    assert x.tolist() == [1000.0]
    assert harmonic.tolist() == [10]
    assert loudness.tolist() == [12.5]


def test_frequency_stepped_prb_stays_on_perceptual_stft_path(monkeypatch):
    from base.core_algorithm.harmonic_distortion.perceptual_step_signal_hd import PerceptualStepSignalHD
    from base.core_algorithm.harmonic_distortion.step_signal_hd import StepSignalHD
    from base.core_algorithm.harmonic_distortion.synchronous_harmonic_detector import SynchronousHarmonicDetector

    called = {"frequency_stepped": False, "perceptual": 0, "fourier": False, "sync": False}
    original_frequency_stepped = AudioThdFrequencyResponseAnalysis._calculate_frequency_stepped_perceptual_thd

    def fake_frequency_stepped(self, *args, **kwargs):
        called["frequency_stepped"] = True
        return original_frequency_stepped(self, *args, **kwargs)

    def fake_perceptual_batch(self, spectrum_matrix, mask_matrix, fundamental_bins, fundamental_freqs, **kwargs):
        called["perceptual"] += 1
        return np.array([float(fundamental_freqs[0]) / 100.0])

    def fake_sync(self, *args, **kwargs):
        called["sync"] = True
        raise AssertionError("frequency-stepped PRB must not call synchronous detector")

    def fake_fourier(self, *args, **kwargs):
        called["fourier"] = True
        raise AssertionError("frequency-stepped PRB must not call standard Fourier compute_thd_batch")

    monkeypatch.setattr(
        AudioThdFrequencyResponseAnalysis,
        "_calculate_frequency_stepped_perceptual_thd",
        fake_frequency_stepped,
    )
    monkeypatch.setattr(PerceptualStepSignalHD, "compute_perceptual_thd_batch", fake_perceptual_batch)
    monkeypatch.setattr(StepSignalHD, "compute_thd_batch", fake_fourier)
    monkeypatch.setattr(SynchronousHarmonicDetector, "analyze", fake_sync)

    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.012,
        min_cycles=8,
        frequency_mode="custom_linear",
        frequencies=[2000.0, 1000.0],
        generate_waveform=False,
    )
    recording = np.zeros(generated.metadata["alignment_sample_count"], dtype=float)
    x, harmonic, loudness = AudioThdFrequencyResponseAnalysis().calculate_perceptual_thd_three_phase(
        recording,
        48000,
        {"stimulus_metadata": generated.metadata, "harmonic_orders": [10]},
        v2pa_factor=1.0,
    )

    assert called == {"frequency_stepped": True, "perceptual": 2, "fourier": False, "sync": False}
    assert x.tolist() == [1000.0, 2000.0]
    assert harmonic.tolist() == [10]
    assert loudness.tolist() == [10.0, 20.0]


def test_frequency_stepped_uses_synchronous_fit_and_preserves_duplicates():
    sample_rate = 48000
    metadata, recording = _frequency_stepped_recording(
        sample_rate,
        [1400.7, 997.3, 997.3],
        {
            0: {1: 1.5, 3: 0.15},
            1: {1: 1.0, 2: 0.05},
            2: {1: 2.0, 2: 0.40},
        },
    )

    x, h, thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        recording,
        sample_rate,
        {"stimulus_metadata": metadata, "harmonic_orders": [2, 3]},
    )

    assert x.tolist() == pytest.approx([997.3, 997.3, 1400.7])
    assert h.shape == (6, 3)
    assert h[0].tolist() == pytest.approx([997.3, 997.3, 1400.7])
    assert h[1].tolist() == pytest.approx([1.0, 2.0, 1.5], rel=1e-8, abs=1e-10)
    assert h[2].tolist() == pytest.approx([0.05, 0.40, 0.0], rel=1e-8, abs=1e-10)
    assert h[3].tolist() == pytest.approx([0.0, 0.0, 0.15], rel=1e-8, abs=1e-10)
    np.testing.assert_allclose(h[4:6], 0.0, rtol=1e-8, atol=1e-10)
    assert thd.tolist() == pytest.approx([5.0, 20.0, 10.0], rel=1e-8, abs=1e-10)


def test_frequency_stepped_explicit_synchronous_does_not_use_fourier_thd_batch(monkeypatch):
    from base.core_algorithm.harmonic_distortion.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer

    def fail_fourier(self, *args, **kwargs):
        raise AssertionError("synchronous frequency_stepped must not call Fourier compute_thd_batch")

    monkeypatch.setattr(HarmonicDistortionAnalyzer, "compute_thd_batch", fail_fourier)

    sample_rate = 48000
    metadata, recording = _frequency_stepped_recording(sample_rate, [997.3], {0: {1: 1.0, 2: 0.1}})

    _, h, thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        recording,
        sample_rate,
        {
            "stimulus_metadata": metadata,
            "harmonic_orders": [2],
            HARMONIC_DETECTION_METHOD_KEY: HARMONIC_DETECTION_METHOD_SYNCHRONOUS,
        },
    )

    assert h[1, 0] == pytest.approx(1.0, rel=1e-8, abs=1e-10)
    assert thd[0] == pytest.approx(10.0, rel=1e-8, abs=1e-10)


def test_frequency_stepped_fourier_uses_stft_nearest_bin_and_not_synchronous_detector(monkeypatch):
    from base.core_algorithm.harmonic_distortion.synchronous_harmonic_detector import SynchronousHarmonicDetector

    def fail_sync(self, *args, **kwargs):
        raise AssertionError("fourier frequency_stepped must not call synchronous detector")

    monkeypatch.setattr(SynchronousHarmonicDetector, "analyze", fail_sync)

    sample_rate = 48000
    metadata, recording = _frequency_stepped_recording(
        sample_rate,
        [1000.0, 1000.0, 2000.0],
        {
            0: {1: 1.0, 2: 0.05},
            1: {1: 3.0, 2: 0.60},
            2: {1: 2.0, 3: 0.10},
        },
    )

    x, h, thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        recording,
        sample_rate,
        {
            "stimulus_metadata": metadata,
            "harmonic_orders": [2, 3],
            HARMONIC_DETECTION_METHOD_KEY: HARMONIC_DETECTION_METHOD_FOURIER,
        },
    )

    assert x.tolist() == pytest.approx([1000.0, 1000.0, 2000.0])
    assert h.shape == (6, 3)
    assert h[0].tolist() == pytest.approx([1000.0, 1000.0, 2000.0])
    assert h[1].tolist() == pytest.approx([h[1, 0], h[1, 1], h[1, 2]])
    assert h[2, 0] / h[1, 0] == pytest.approx(0.05, rel=1e-6, abs=1e-8)
    assert h[2, 1] / h[1, 1] == pytest.approx(0.20, rel=1e-6, abs=1e-8)
    assert h[3, 2] / h[1, 2] == pytest.approx(0.05, rel=1e-6, abs=1e-8)
    assert thd.tolist() == pytest.approx([5.0, 20.0, 5.0], rel=1e-6, abs=1e-8)


def test_frequency_stepped_fourier_and_synchronous_differ_for_nonbin_tone():
    sample_rate = 48000
    metadata, recording = _frequency_stepped_recording(sample_rate, [997.3], {0: {1: 1.0, 2: 0.1}})

    _, _, sync_thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        recording,
        sample_rate,
        {"stimulus_metadata": metadata, "harmonic_orders": [2], HARMONIC_DETECTION_METHOD_KEY: "synchronous"},
    )
    _, _, fourier_thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        recording,
        sample_rate,
        {"stimulus_metadata": metadata, "harmonic_orders": [2], HARMONIC_DETECTION_METHOD_KEY: "fourier"},
    )

    assert sync_thd[0] == pytest.approx(10.0, rel=1e-8, abs=1e-10)
    assert fourier_thd[0] != pytest.approx(sync_thd[0], rel=1e-5, abs=1e-5)


def test_frequency_stepped_empty_segments_return_empty_public_contract():
    sample_rate = 48000
    generated = generate_frequency_stepped(
        sample_rate=sample_rate,
        repeat_times=1,
        min_duration=0.012,
        min_cycles=8,
        frequency_mode="custom_linear",
        frequencies=[1000.0],
        generate_waveform=False,
    )

    x, h, thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        np.array([], dtype=float),
        sample_rate,
        {"stimulus_metadata": generated.metadata, "harmonic_orders": [2]},
    )

    np.testing.assert_array_equal(x, np.array([]))
    assert h.shape == (6, 0)
    np.testing.assert_array_equal(h, np.zeros((6, 0), dtype=float))
    np.testing.assert_array_equal(thd, np.array([]))


def test_frequency_stepped_fourier_empty_segments_return_empty_public_contract():
    sample_rate = 48000
    generated = generate_frequency_stepped(
        sample_rate=sample_rate,
        repeat_times=1,
        min_duration=0.012,
        min_cycles=8,
        frequency_mode="custom_linear",
        frequencies=[1000.0],
        generate_waveform=False,
    )

    x, h, thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        np.array([], dtype=float),
        sample_rate,
        {
            "stimulus_metadata": generated.metadata,
            "harmonic_orders": [2],
            HARMONIC_DETECTION_METHOD_KEY: HARMONIC_DETECTION_METHOD_FOURIER,
        },
    )

    np.testing.assert_array_equal(x, np.array([]))
    assert h.shape == (6, 0)
    np.testing.assert_array_equal(h, np.zeros((6, 0), dtype=float))
    np.testing.assert_array_equal(thd, np.array([]))


def test_frequency_stepped_h_rows_use_zero_for_nyquist_excluded_harmonics():
    sample_rate = 48000
    metadata, recording = _frequency_stepped_recording(
        sample_rate,
        [12000.0],
        {0: {1: 0.7}},
    )

    x, h, thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        recording,
        sample_rate,
        {"stimulus_metadata": metadata, "harmonic_orders": [2, 3, 4, 5]},
    )

    assert x.tolist() == pytest.approx([12000.0])
    assert h[1, 0] == pytest.approx(0.7, rel=1e-8, abs=1e-10)
    np.testing.assert_array_equal(h[2:6, 0], np.zeros(4))
    assert thd[0] == 0.0


@pytest.mark.parametrize("harmonic_orders", [[1], [0], [36], [2.5], [True]])
def test_frequency_stepped_invalid_selected_harmonic_orders_propagate_value_error(harmonic_orders):
    sample_rate = 48000
    metadata, recording = _frequency_stepped_recording(
        sample_rate,
        [1000.0],
        {0: {1: 1.0, 2: 0.1}},
    )

    with pytest.raises(ValueError):
        AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
            recording,
            sample_rate,
            {"stimulus_metadata": metadata, "harmonic_orders": harmonic_orders},
        )


@pytest.mark.parametrize("harmonic_orders", [[1], [0], [36], [2.5], [True]])
def test_frequency_stepped_fourier_invalid_selected_harmonic_orders_raise_before_mask(
    harmonic_orders,
    monkeypatch,
):
    from base.core_algorithm.harmonic_distortion.harmonic_index_builder import HarmonicIndexBuilder
    from base.core_algorithm.harmonic_distortion.synchronous_harmonic_detector import SynchronousHarmonicDetector

    def fail_mask(self, *args, **kwargs):
        raise AssertionError("invalid Fourier orders must be rejected before mask construction")

    def fail_sync(self, *args, **kwargs):
        raise AssertionError("fourier frequency_stepped validation must not call synchronous detector")

    monkeypatch.setattr(HarmonicIndexBuilder, "create_mask_from_indices", fail_mask)
    monkeypatch.setattr(SynchronousHarmonicDetector, "analyze", fail_sync)

    sample_rate = 48000
    metadata, recording = _frequency_stepped_recording(
        sample_rate,
        [1000.0],
        {0: {1: 1.0, 2: 0.1}},
    )

    with pytest.raises(ValueError):
        AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
            recording,
            sample_rate,
            {
                "stimulus_metadata": metadata,
                "harmonic_orders": harmonic_orders,
                HARMONIC_DETECTION_METHOD_KEY: HARMONIC_DETECTION_METHOD_FOURIER,
            },
        )


def test_frequency_stepped_invalid_direct_detection_method_raises_value_error():
    sample_rate = 48000
    metadata, recording = _frequency_stepped_recording(sample_rate, [1000.0], {0: {1: 1.0}})

    with pytest.raises(ValueError, match="Unsupported harmonic detection method"):
        AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
            recording,
            sample_rate,
            {"stimulus_metadata": metadata, "harmonic_orders": [2], HARMONIC_DETECTION_METHOD_KEY: "bad"},
        )


def test_frequency_stepped_legal_selected_orders_above_nyquist_are_ignored():
    sample_rate = 48000
    metadata, recording = _frequency_stepped_recording(
        sample_rate,
        [10000.0],
        {0: {1: 0.9}},
    )

    x, h, thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        recording,
        sample_rate,
        {"stimulus_metadata": metadata, "harmonic_orders": [3]},
    )

    assert x.tolist() == pytest.approx([10000.0])
    assert h[1, 0] == pytest.approx(0.9, rel=1e-8, abs=1e-10)
    assert h[3, 0] == 0.0
    assert thd[0] == 0.0


def test_frequency_stepped_fourier_legal_selected_orders_above_nyquist_are_ignored():
    sample_rate = 48000
    metadata, recording = _frequency_stepped_recording(
        sample_rate,
        [10000.0],
        {0: {1: 0.9}},
    )

    x, h, thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        recording,
        sample_rate,
        {
            "stimulus_metadata": metadata,
            "harmonic_orders": [3],
            HARMONIC_DETECTION_METHOD_KEY: HARMONIC_DETECTION_METHOD_FOURIER,
        },
    )

    assert x.tolist() == pytest.approx([10000.0])
    assert h[1, 0] > 0.0
    assert h[3, 0] == 0.0
    assert thd[0] == 0.0

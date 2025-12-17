# unit_test/base/pre_processing/test_noise_spectrum_scaling_regression.py
import numpy as np

from base.pre_processing.perceptual_step_signal_hd import PerceptualStepSignalHD
from base.pre_processing.psychoacoustic_utils import compute_noise_spectrum


def test_noise_spectrum_scaling_matches_stft_and_does_not_zero_prb():
    """
    Regression: compute_noise_spectrum must use a magnitude scaling compatible with SciPy STFT,
    otherwise quadrature subtraction can over-subtract and drive PRB to 0.
    """
    sample_rate = 48000
    analyzer = PerceptualStepSignalHD(sample_rate=sample_rate)

    stimulus_metadata = {
        "num_steps": 2,
        "repeat_times": 1,
        "total_time": 10.0,  # 5s per step
        "start_freq": 100,
        "stop_freq": 200,
        "stimulus_type": "linear",
    }

    # Strong 1 kHz tone (acts as 10th harmonic of 100 Hz and 5th harmonic of 200 Hz).
    amplitude_peak = np.sqrt(2.0)  # 1 Pa RMS if calibrated in Pa
    t = np.arange(int(sample_rate * stimulus_metadata["total_time"])) / sample_rate
    recorded_signal = amplitude_peak * np.sin(2.0 * np.pi * 1000.0 * t)

    # Background noise segment used for noise spectrum estimation (low level).
    rng = np.random.default_rng(0)
    pre_alignment_data = rng.normal(0.0, 1e-4, size=int(0.5 * sample_rate))
    noise_spectrum = compute_noise_spectrum(pre_alignment_data, sample_rate)

    # Use a negative calibration offset to catch calibration mismatch between signal and noise spectrum.
    result = analyzer.compute_distortion(
        recorded_signal,
        stimulus_metadata,
        harmonic_orders=[5, 10],
        harmonic_mask=None,
        masking_config=None,
        spl_calibration_db=-20.0,
        noise_spectrum=noise_spectrum,
    )

    loudness = np.asarray(result["perceptual_loudness"], dtype=float)
    assert loudness.shape == (stimulus_metadata["num_steps"],)
    assert np.all(loudness > 0.0)


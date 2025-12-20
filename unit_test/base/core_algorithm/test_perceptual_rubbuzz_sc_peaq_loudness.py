import numpy as np
import pytest


@pytest.fixture(scope="module")
def peaq_model():
    from base.core_algorithm.perceptual_rubbuzz_sc.peaq_loudness_model import PEAQLoudnessModel

    sample_rate = 48000
    n_fft = 4096
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return PEAQLoudnessModel(freqs)


def _tone_spectrum_pa(freqs_hz: np.ndarray, *, tone_hz: float, spl_db: float) -> np.ndarray:
    """Single-bin spectrum helper for paper-style FFT spectra in dB SPL vs Hz."""
    p_rms = 20e-6 * (10.0 ** (spl_db / 20.0))
    idx = int(np.argmin(np.abs(freqs_hz - tone_hz)))
    spec = np.zeros(freqs_hz.size, dtype=np.float64)
    spec[idx] = p_rms
    return spec


def test_peaq_loudness_calibration_100db_1khz(peaq_model):
    """
    Paper 4.8: const chosen so that 100 dB SPL @ 1 kHz yields 64 sones (= 100 phons).
    """
    freqs = peaq_model.rfft_freqs_hz
    spec = _tone_spectrum_pa(freqs, tone_hz=1000.0, spl_db=100.0)
    out = peaq_model.compute_loudness_from_single_spectrum(spec)
    assert out.n_total_sones[0] == pytest.approx(64.0, abs=1e-9)
    assert out.n_total_phons[0] == pytest.approx(100.0, abs=1e-9)


def test_peaq_partial_loudness_zero_when_test_equals_ref(peaq_model):
    freqs = peaq_model.rfft_freqs_hz
    test = _tone_spectrum_pa(freqs, tone_hz=1000.0, spl_db=100.0)
    ref = test.copy()
    out = peaq_model.compute_partial_loudness_from_spectra(test, ref)
    assert out.n_total_sones[0] == pytest.approx(0.0, abs=1e-12)


def test_peaq_partial_loudness_zero_with_level_adaptation(peaq_model):
    """
    Paper 4.3: reference (stimulus) spectrum is scaled to match response spectrum level.
    If both are the same tone, TotalNL should be 0 even if the raw reference level differs.
    """
    freqs = peaq_model.rfft_freqs_hz
    test = _tone_spectrum_pa(freqs, tone_hz=1000.0, spl_db=100.0)
    ref = _tone_spectrum_pa(freqs, tone_hz=1000.0, spl_db=80.0)
    out = peaq_model.compute_partial_loudness_from_spectra(test, ref)
    assert out.n_total_sones[0] == pytest.approx(0.0, abs=1e-12)


def test_peaq_partial_loudness_increases_with_added_component(peaq_model):
    freqs = peaq_model.rfft_freqs_hz
    ref = _tone_spectrum_pa(freqs, tone_hz=1000.0, spl_db=100.0)
    test = ref.copy()
    test += _tone_spectrum_pa(freqs, tone_hz=2000.0, spl_db=60.0)

    out = peaq_model.compute_partial_loudness_from_spectra(test, ref)
    assert out.n_total_sones[0] > 0.0


def test_peaq_partial_loudness_supports_multi_frame_inputs(peaq_model):
    freqs = peaq_model.rfft_freqs_hz
    ref0 = _tone_spectrum_pa(freqs, tone_hz=1000.0, spl_db=100.0)
    test0 = ref0.copy()

    # Frame 1 has an additional component; Frame 0 is identical.
    test1 = ref0 + _tone_spectrum_pa(freqs, tone_hz=3000.0, spl_db=60.0)

    test = np.stack([test0, test1], axis=1)  # (n_bins, 2)
    ref = np.stack([ref0, ref0], axis=1)

    out = peaq_model.compute_partial_loudness_from_spectra(test, ref)
    assert out.n_total_sones.shape == (2,)
    assert out.n_total_sones[0] == pytest.approx(0.0, abs=1e-12)
    assert out.n_total_sones[1] > 0.0


def test_peaq_interpolate_specific_matches_np_interp(peaq_model):
    freqs = peaq_model.rfft_freqs_hz
    spec = _tone_spectrum_pa(freqs, tone_hz=1000.0, spl_db=80.0)
    out = peaq_model.compute_loudness_from_single_spectrum(spec)

    target_hz = np.array([1000.0, 1500.0], dtype=np.float64)
    z = out.band_center_bark
    y = out.n_specific_sones_per_band[:, 0]

    # Use the model's hz->bark mapping for interpolation points.
    from base.core_algorithm.perceptual_rubbuzz_sc.peaq_loudness_model import hz_to_bark_peaq

    target_bark = hz_to_bark_peaq(target_hz)
    expected = np.array([np.interp(target_bark[0], z, y), np.interp(target_bark[1], z, y)], dtype=np.float64)

    got = out.interpolate_specific_sones(target_hz)[:, 0]
    assert got == pytest.approx(expected, abs=1e-12)


def test_peaq_interpolate_specific_supports_per_frame_targets(peaq_model):
    freqs = peaq_model.rfft_freqs_hz
    # Two frames with different added components to make the profile differ across time.
    base = _tone_spectrum_pa(freqs, tone_hz=1000.0, spl_db=80.0)
    frame0 = base.copy()
    frame1 = base + _tone_spectrum_pa(freqs, tone_hz=3000.0, spl_db=60.0)
    spec = np.stack([frame0, frame1], axis=1)
    out = peaq_model.compute_loudness_from_single_spectrum(spec)

    # Sample 1000 Hz for frame0, 3000 Hz for frame1.
    targets = np.array([[1000.0, 3000.0]], dtype=np.float64)  # (M=1, T=2)
    got = out.interpolate_specific_sones(targets)
    assert got.shape == (1, 2)
    assert got[0, 0] != got[0, 1]


def test_peaq_partial_loudness_can_disable_level_adaptation(peaq_model):
    """
    If the reference is near-silence (e.g., extracted f0 bin), scaling it to match the test peak is wrong.
    Ensure callers can explicitly disable level adaptation.
    """
    freqs = peaq_model.rfft_freqs_hz
    test = _tone_spectrum_pa(freqs, tone_hz=1000.0, spl_db=100.0)
    # A tiny low-frequency component (acts like noise floor / leakage).
    ref = _tone_spectrum_pa(freqs, tone_hz=100.0, spl_db=20.0)

    out_adapt = peaq_model.compute_partial_loudness_from_spectra(test, ref, apply_level_adaptation=True)
    out_no_adapt = peaq_model.compute_partial_loudness_from_spectra(test, ref, apply_level_adaptation=False)
    assert out_no_adapt.n_total_sones[0] >= out_adapt.n_total_sones[0]


def test_peaq_ear_weighting_term3_exponent_affects_high_freq_loudness(peaq_model):
    """
    The Eq.1 term3 exponent controls high-frequency roll-off.
    If we reduce it (e.g. 3.6 -> 3.5), high-frequency tones should become louder.
    """
    from base.core_algorithm.perceptual_rubbuzz_sc.peaq_loudness_model import PEAQLoudnessConfig, PEAQLoudnessModel

    freqs = peaq_model.rfft_freqs_hz
    # Build a second model on the exact same frequency axis but with a different ear-weighting exponent.
    model_less_rolloff = PEAQLoudnessModel(
        freqs,
        config=PEAQLoudnessConfig(ear_weighting_term3_exponent=3.5),
    )

    test_20k = _tone_spectrum_pa(freqs, tone_hz=20000.0, spl_db=80.0)
    ref_silence = np.zeros_like(test_20k)

    out_default = peaq_model.compute_partial_loudness_from_spectra(test_20k, ref_silence, apply_level_adaptation=False)
    out_less = model_less_rolloff.compute_partial_loudness_from_spectra(
        test_20k, ref_silence, apply_level_adaptation=False
    )
    assert out_less.n_total_sones[0] > out_default.n_total_sones[0]


def test_peaq_ehs_increases_with_harmonic_structure(peaq_model):
    freqs = peaq_model.rfft_freqs_hz
    f0 = 100.0

    # Fundamental-only spectrum.
    spec_f0 = _tone_spectrum_pa(freqs, tone_hz=f0, spl_db=80.0)

    # Add a dense harmonic series (stronger harmonic structure).
    spec_h = spec_f0.copy()
    for h in range(2, 40):
        spec_h += _tone_spectrum_pa(freqs, tone_hz=h * f0, spl_db=65.0)

    ehs_f0 = peaq_model.compute_error_harmonic_structure(spec_f0, np.array([f0], dtype=np.float64))[0]
    ehs_h = peaq_model.compute_error_harmonic_structure(spec_h, np.array([f0], dtype=np.float64))[0]
    assert ehs_h > ehs_f0


def test_peaq_ehs_is_scale_invariant(peaq_model):
    freqs = peaq_model.rfft_freqs_hz
    f0 = 200.0

    spec = _tone_spectrum_pa(freqs, tone_hz=f0, spl_db=80.0)
    for h in range(2, 20):
        spec += _tone_spectrum_pa(freqs, tone_hz=h * f0, spl_db=60.0)

    ehs_1 = peaq_model.compute_error_harmonic_structure(spec, f0)[0]
    ehs_2 = peaq_model.compute_error_harmonic_structure(spec * 10.0, f0)[0]
    assert ehs_1 == pytest.approx(ehs_2, abs=1e-12)

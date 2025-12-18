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


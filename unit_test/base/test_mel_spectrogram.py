import inspect

import librosa
import numpy as np
import pytest

from base.core_algorithm.response.mel_spectrogram_analyzer import (
    InvalidMelBandConfigurationError,
    MelSpectrogramAnalyzer,
)
from consts.acoustic_analysis.specific_consts import spec_consts


def _tone(frequency_hz, amplitude, sample_rate, duration_s=0.5):
    time_s = np.arange(int(sample_rate * duration_s), dtype=np.float64) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * time_s)


def test_mel_spectrogram_tracks_tone_band_and_shape():
    sample_rate = 16000
    analyzer = MelSpectrogramAnalyzer()

    times_s, mel_frequencies_hz, mel_power_db = analyzer.analyze(
        _tone(1000.0, 0.25, sample_rate),
        sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=40,
        fmin_hz=0.0,
        fmax_hz=8000.0,
        window="hann",
    )

    assert mel_power_db.shape[0] == 40
    assert mel_power_db.shape[1] == times_s.size
    assert mel_frequencies_hz.shape == (40,)
    assert np.all(np.isfinite(mel_power_db))

    dominant_band = int(np.argmax(np.mean(mel_power_db, axis=1)))
    dominant_frequency_hz = mel_frequencies_hz[dominant_band]
    assert 700.0 <= dominant_frequency_hz <= 1300.0


def test_mel_spectrogram_preserves_six_db_amplitude_change():
    sample_rate = 16000
    analyzer = MelSpectrogramAnalyzer()
    _, _, low_power_db = analyzer.analyze(
        _tone(1000.0, 0.1, sample_rate),
        sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=40,
        fmin_hz=0.0,
        fmax_hz=8000.0,
        window="hann",
    )
    _, _, high_power_db = analyzer.analyze(
        _tone(1000.0, 0.2, sample_rate),
        sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=40,
        fmin_hz=0.0,
        fmax_hz=8000.0,
        window="hann",
    )

    level_delta_db = float(np.max(high_power_db) - np.max(low_power_db))
    assert level_delta_db == pytest.approx(20.0 * np.log10(2.0), abs=1e-5)


def test_mel_spectrogram_matches_librosa_slaney_power_definition():
    sample_rate = 16000
    signal = _tone(1000.0, 0.2, sample_rate)
    _, _, mel_power_db = MelSpectrogramAnalyzer().analyze(
        signal,
        sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=40,
        fmin_hz=0.0,
        fmax_hz=8000.0,
        window="hann",
    )

    expected_power = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window="hann",
        center=True,
        pad_mode="constant",
        power=2.0,
        n_mels=40,
        fmin=0.0,
        fmax=8000.0,
        htk=False,
        norm="slaney",
    )
    expected_db = librosa.power_to_db(
        expected_power,
        ref=1.0,
        amin=1e-10,
        top_db=None,
    )

    np.testing.assert_allclose(mel_power_db, expected_db, rtol=1e-7, atol=1e-6)


def test_mel_spectrogram_defaults_to_nyquist_frequency():
    sample_rate = 8000

    _, mel_frequencies_hz, _ = MelSpectrogramAnalyzer().analyze(
        _tone(500.0, 0.2, sample_rate),
        sample_rate,
        n_fft=512,
        hop_length=128,
        n_mels=32,
        fmax_hz=None,
    )

    expected_centers_hz = librosa.mel_frequencies(
        n_mels=34,
        fmin=0.0,
        fmax=sample_rate / 2.0,
        htk=False,
    )[1:-1]
    np.testing.assert_allclose(mel_frequencies_hz, expected_centers_hz)
    assert mel_frequencies_hz[-1] < sample_rate / 2.0


def test_mel_spectrogram_rejects_frequency_above_nyquist():
    sample_rate = 8000

    with pytest.raises(ValueError, match="Nyquist"):
        MelSpectrogramAnalyzer().analyze(
            _tone(500.0, 0.2, sample_rate),
            sample_rate,
            fmax_hz=5000.0,
        )


def test_mel_spectrogram_rejects_unsupported_band_configuration():
    sample_rate = 48000

    with pytest.warns(UserWarning, match="Empty filters"), pytest.raises(
        InvalidMelBandConfigurationError,
        match="减少频带数量或增大 FFT 点数",
    ):
        MelSpectrogramAnalyzer().analyze(
            np.zeros(sample_rate, dtype=np.float64),
            sample_rate,
            n_fft=512,
            hop_length=256,
            n_mels=128,
            fmin_hz=0,
            fmax_hz=10000,
        )


def test_mel_analyzer_api_has_no_calibration_parameter():
    parameters = inspect.signature(MelSpectrogramAnalyzer.analyze).parameters

    assert parameters["n_fft"].default == spec_consts.DEFAULT_SPEC_N_FFT
    assert parameters["hop_length"].default == spec_consts.DEFAULT_SPEC_HOP_LENGTH
    assert parameters["n_mels"].default == spec_consts.DEFAULT_MEL_BAND_COUNT
    assert parameters["fmin_hz"].default == spec_consts.DEFAULT_MEL_FMIN_HZ
    assert parameters["window"].default == spec_consts.DEFAULT_SPEC_WINDOW
    assert "v2pa_factor" not in parameters
    assert "reference_pressure" not in parameters

import mock
import numpy as np
import pytest

from base.soundcard_audio_processor import SoundcardAudioProcessor
from consts import error_code


class TestSoundcardAudioProcessor(object):

    test_path = "base.soundcard_audio_processor.SoundcardAudioProcessor"

    @pytest.mark.parametrize("corr_ret, stimulus_signal, recorded_signal, result_set", [
        (1, [1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1], -4),
        (0, [1, 1, 1], [1, 1, 1], -2),
        (0, [], [], 1),
    ])
    @mock.patch("scipy.signal.correlate")
    def test_calculate_alignment(self, mock_corr, corr_ret, stimulus_signal, recorded_signal, result_set):
        mock_corr.return_value = corr_ret
        result = SoundcardAudioProcessor().calculate_alignment(stimulus_signal, recorded_signal)
        assert result == result_set


def test_sd_play_rec_rejects_mismatched_sample_rates(monkeypatch):
    calls = []
    processor = SoundcardAudioProcessor()
    record = {"sr": 48000, "sample_rate": 48000, "num_frames": 1}
    stimulus = {"data": np.array([0.0], dtype=np.float32), "amplitude": 1.0, "sr": 44100}

    monkeypatch.setattr("base.soundcard_audio_processor.sd.playrec", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr("base.soundcard_audio_processor.save_audio_simple", lambda *args, **kwargs: calls.append(args))

    code, msg = processor.sd_play_rec(record, stimulus, "out.wav")

    assert code != error_code.OK
    assert "采样率" in msg or "sample rate" in msg.lower()
    assert calls == []


def test_sd_play_rec_rejects_missing_sample_rates(monkeypatch):
    calls = []
    processor = SoundcardAudioProcessor()
    record = {"num_frames": 1}
    stimulus = {"data": np.array([0.0], dtype=np.float32), "amplitude": 1.0}

    monkeypatch.setattr("base.soundcard_audio_processor.sd.playrec", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr("base.soundcard_audio_processor.save_audio_simple", lambda *args, **kwargs: calls.append(args))

    code, msg = processor.sd_play_rec(record, stimulus, "out.wav")

    assert code != error_code.OK
    assert "采样率" in msg or "sample rate" in msg.lower()
    assert calls == []


@pytest.mark.parametrize("sample_rate", [float("inf"), float("nan")])
def test_sd_play_rec_rejects_non_finite_sample_rates(monkeypatch, sample_rate):
    calls = []
    processor = SoundcardAudioProcessor()
    record = {"sr": sample_rate, "sample_rate": sample_rate, "num_frames": 1}
    stimulus = {"data": np.array([0.0], dtype=np.float32), "amplitude": 1.0, "sr": 48000}

    monkeypatch.setattr("base.soundcard_audio_processor.sd.playrec", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr("base.soundcard_audio_processor.save_audio_simple", lambda *args, **kwargs: calls.append(args))

    code, msg = processor.sd_play_rec(record, stimulus, "out.wav")

    assert code != error_code.OK
    assert "采样率" in msg or "sample rate" in msg.lower()
    assert calls == []


def test_sd_rec_rejects_conflicting_sample_rates_before_recording(monkeypatch):
    calls = []

    monkeypatch.setattr("base.soundcard_audio_processor.sd.rec", lambda *args, **kwargs: calls.append((args, kwargs)))

    code, msg = SoundcardAudioProcessor.sd_rec(
        {
            "num_frames": 1,
            "sample_rate": 44100,
            "sr": 48000,
        }
    )

    assert code != error_code.OK
    assert "采样率" in msg or "sample rate" in msg.lower()
    assert calls == []


def test_sd_rec_rejects_missing_sample_rate_before_recording(monkeypatch):
    calls = []

    monkeypatch.setattr("base.soundcard_audio_processor.sd.rec", lambda *args, **kwargs: calls.append((args, kwargs)))

    code, msg = SoundcardAudioProcessor.sd_rec({"num_frames": 1})

    assert code != error_code.OK
    assert "采样率" in msg or "sample rate" in msg.lower()
    assert calls == []


def test_sd_rec_passes_float64_dtype_from_bit_depth(monkeypatch):
    calls = []

    def fake_rec(**kwargs):
        calls.append(kwargs)
        return np.zeros((2, 1), dtype=np.float64)

    monkeypatch.setattr("base.soundcard_audio_processor.sd.rec", fake_rec)

    code, data = SoundcardAudioProcessor.sd_rec(
        {
            "num_frames": 2,
            "sample_rate": 48000,
            "channels": 1,
            "bit_depth": 64,
        }
    )

    assert code == error_code.OK
    assert calls[0]["dtype"] == "float64"
    assert data.dtype == np.float64


def test_sd_play_rec_passes_float64_dtype_and_saves_bit_depth(monkeypatch):
    calls = {}

    def fake_playrec(data, samplerate, channels, blocking, dtype, device=None):
        calls["dtype"] = dtype
        return np.asarray(data, dtype=np.float64).reshape(-1, 1)

    monkeypatch.setattr("base.soundcard_audio_processor.sd.playrec", fake_playrec)
    monkeypatch.setattr(SoundcardAudioProcessor, "calculate_alignment", lambda self, stimulus, recorded: 0)
    monkeypatch.setattr(
        "base.soundcard_audio_processor.save_audio_simple",
        lambda path, audio, sr, bit_depth=32: calls.update(
            saved_path=path,
            saved_dtype=np.asarray(audio).dtype,
            saved_sr=sr,
            saved_bit_depth=bit_depth,
        ),
    )

    code, data = SoundcardAudioProcessor().sd_play_rec(
        {"prepare_frames": 0, "prolong_frames": 0, "sr": 48000, "bit_depth": 64},
        {"data": np.array([0.1, 0.2], dtype=np.float32), "amplitude": 1.0, "sr": 48000},
        "out.wav",
    )

    assert code == error_code.OK
    assert calls["dtype"] == "float64"
    assert calls["saved_path"] == "out.wav"
    assert calls["saved_sr"] == 48000
    assert calls["saved_bit_depth"] == 64
    assert calls["saved_dtype"] == np.float64
    assert data.dtype == np.float64

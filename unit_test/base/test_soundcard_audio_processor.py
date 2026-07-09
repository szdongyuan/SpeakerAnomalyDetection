import mock
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base import play_and_record
from base.soundcard_audio_processor import SoundcardAudioProcessor
from consts import error_code


class TestSoundcardAudioProcessor(object):

    test_path = "base.soundcard_audio_processor.SoundcardAudioProcessor"

    @pytest.mark.parametrize("corr_ret, stimulus_signal, recorded_signal, result_set", [
        (1, [1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1], -4),
        (0, [1, 1, 1], [1, 1, 1], -2),
        (0, [], [], 1),
    ])
    @mock.patch("base.soundcard_audio_processor.correlate")
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


def test_sd_play_rec_saves_explicit_calibration_metadata(monkeypatch):
    calls = []
    metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True}
        ]
    }
    processor = SoundcardAudioProcessor()
    record = {"sr": 48000, "sample_rate": 48000, "num_frames": 4}
    stimulus = {"data": np.array([0.0, 1.0], dtype=np.float32), "amplitude": 1.0, "sr": 48000}

    monkeypatch.setattr(
        "base.soundcard_audio_processor.sd.playrec",
        lambda *args, **kwargs: np.array([[0.0], [1.0], [0.0], [0.0]], dtype=np.float32),
    )
    monkeypatch.setattr(
        "base.soundcard_audio_processor.save_audio_with_calibration_metadata",
        lambda path, data, sr, calibration_metadata=None, logger=None: calls.append(
            (path, np.asarray(data), sr, calibration_metadata)
        ),
    )

    code, recorded = processor.sd_play_rec(record, stimulus, "record.wav", calibration_metadata=metadata)

    assert code == error_code.OK
    assert np.asarray(recorded).shape == (2,)
    assert calls[0][0] == "record.wav"
    assert calls[0][2] == 48000
    assert calls[0][3] is metadata


def test_sd_play_rec_saves_record_dict_calibration_metadata(monkeypatch):
    calls = []
    metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 3.5, "standard_spl": 114.0, "calibrated": True}
        ]
    }
    processor = SoundcardAudioProcessor()
    record = {"sr": 48000, "sample_rate": 48000, "num_frames": 4, "wav_calibration_metadata": metadata}
    stimulus = {"data": np.array([0.0, 1.0], dtype=np.float32), "amplitude": 1.0, "sr": 48000}

    monkeypatch.setattr(
        "base.soundcard_audio_processor.sd.playrec",
        lambda *args, **kwargs: np.array([[0.0], [1.0], [0.0], [0.0]], dtype=np.float32),
    )
    monkeypatch.setattr(
        "base.soundcard_audio_processor.save_audio_with_calibration_metadata",
        lambda path, data, sr, calibration_metadata=None, logger=None: calls.append(
            (path, np.asarray(data), sr, calibration_metadata)
        ),
    )

    code, recorded = processor.sd_play_rec(record, stimulus, "record.wav")

    assert code == error_code.OK
    assert np.asarray(recorded).shape == (2,)
    assert calls[0][3] is metadata


def test_sd_play_rec_uses_selected_nonzero_input_channel_for_saved_metadata(monkeypatch):
    calls = {}
    metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 4.5, "standard_spl": 94.0, "calibrated": True}
        ]
    }
    processor = SoundcardAudioProcessor()
    record = {
        "sr": 48000,
        "sample_rate": 48000,
        "input_channels": [2],
        "prepare_frames": 0,
        "prolong_frames": 0,
    }
    stimulus = {"data": np.array([0.0, 1.0, 0.0], dtype=np.float32), "amplitude": 1.0, "sr": 48000}
    recorded_channels = np.array(
        [
            [10.0, 20.0, 30.0],
            [11.0, 21.0, 31.0],
            [12.0, 22.0, 32.0],
        ],
        dtype=np.float32,
    )

    def fake_playrec(playback_data, samplerate, channels, blocking, dtype=None):
        calls["playrec"] = {
            "playback_data": np.asarray(playback_data),
            "samplerate": samplerate,
            "channels": channels,
            "blocking": blocking,
            "dtype": dtype,
        }
        return recorded_channels

    monkeypatch.setattr("base.soundcard_audio_processor.sd.playrec", fake_playrec)
    monkeypatch.setattr(SoundcardAudioProcessor, "calculate_alignment", staticmethod(lambda ref, rec: 0))
    monkeypatch.setattr(
        "base.soundcard_audio_processor.save_audio_with_calibration_metadata",
        lambda path, data, sr, calibration_metadata=None, logger=None: calls.update(
            saved_path=path,
            saved_data=np.asarray(data),
            saved_sample_rate=sr,
            saved_metadata=calibration_metadata,
        ),
    )

    code, recorded = processor.sd_play_rec(record, stimulus, "record.wav", calibration_metadata=metadata)

    assert code == error_code.OK
    assert calls["playrec"]["channels"] == 3
    assert calls["saved_path"] == "record.wav"
    assert calls["saved_sample_rate"] == 48000
    assert calls["saved_metadata"] is metadata
    np.testing.assert_array_equal(calls["saved_data"], recorded_channels[:, 2])
    np.testing.assert_array_equal(recorded, recorded_channels[:, 2])


def test_sd_play_rec_projects_multi_channel_metadata_to_saved_mono_channel(monkeypatch):
    calls = {}
    metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 4.5, "standard_spl": 94.0, "calibrated": True},
            {"wav_channel_index": 1, "v2pa_factor": 8.5, "standard_spl": 114.0, "calibrated": True},
        ]
    }
    processor = SoundcardAudioProcessor()
    record = {
        "sr": 48000,
        "sample_rate": 48000,
        "input_channels": [2, 4],
        "prepare_frames": 0,
        "prolong_frames": 0,
        "wav_calibration_metadata": metadata,
    }
    stimulus = {"data": np.array([0.0, 1.0, 0.0], dtype=np.float32), "amplitude": 1.0, "sr": 48000}
    recorded_channels = np.array(
        [
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [11.0, 21.0, 31.0, 41.0, 51.0],
            [12.0, 22.0, 32.0, 42.0, 52.0],
        ],
        dtype=np.float32,
    )

    monkeypatch.setattr("base.soundcard_audio_processor.sd.playrec", lambda *args, **kwargs: recorded_channels)
    monkeypatch.setattr(SoundcardAudioProcessor, "calculate_alignment", staticmethod(lambda ref, rec: 0))
    monkeypatch.setattr(
        "base.soundcard_audio_processor.save_audio_with_calibration_metadata",
        lambda path, data, sr, calibration_metadata=None, logger=None: calls.update(
            saved_data=np.asarray(data),
            saved_metadata=calibration_metadata,
        ),
    )

    code, recorded = processor.sd_play_rec(record, stimulus, "record.wav")

    assert code == error_code.OK
    np.testing.assert_array_equal(calls["saved_data"], recorded_channels[:, 2])
    np.testing.assert_array_equal(recorded, recorded_channels[:, 2])
    assert calls["saved_metadata"] == {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 4.5, "standard_spl": 94.0, "calibrated": True}
        ]
    }


def test_record_without_play_persists_metadata(monkeypatch):
    calls = []
    metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True}
        ]
    }
    recorded_dict = {"sr": 48000, "sample_rate": 48000, "wav_calibration_metadata": metadata}

    monkeypatch.setattr(
        play_and_record.SoundcardAudioProcessor,
        "sd_rec",
        staticmethod(lambda _recorded_dict: (error_code.OK, np.array([0.1, 0.2], dtype=np.float32))),
    )
    monkeypatch.setattr(
        play_and_record,
        "save_audio_with_calibration_metadata",
        lambda path, data, sr, calibration_metadata=None, logger=None: calls.append((path, sr, calibration_metadata)),
    )
    monkeypatch.setattr(play_and_record.RecordingManager, "save_signal_info_to_db", lambda self, info, stimulus: (0, "ok"))

    code, recorded = play_and_record.record_without_play(recorded_dict, "record.wav", {})

    assert code == error_code.OK
    assert np.asarray(recorded).shape == (2,)
    assert calls == [("record.wav", 48000, metadata)]


def test_play_last_stimulus_wave_forwards_metadata(monkeypatch):
    calls = []
    metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True}
        ]
    }
    recorded_dict = {"sr": 48000, "sample_rate": 48000, "wav_calibration_metadata": metadata}
    stimulus_dict = {"sr": 48000, "data": np.zeros(2, dtype=np.float32), "amplitude": 1.0}
    play_and_record.data_struct.sample_rate = 48000
    play_and_record.data_struct.stimulus_info = {"repeat_times": 1}

    def fake_sd_play_rec(self, record, stimulus, path, calibration_metadata=None):
        calls.append((path, calibration_metadata))
        return error_code.OK, np.array([0.1, 0.2], dtype=np.float32)

    monkeypatch.setattr(play_and_record.SoundcardAudioProcessor, "sd_play_rec", fake_sd_play_rec)
    monkeypatch.setattr(play_and_record.RecordingManager, "save_signal_info_to_db", lambda self, info, stimulus: (0, "ok"))

    code, recorded = play_and_record.play_last_stimulus_wave(stimulus_dict, recorded_dict, "record.wav", {})

    assert code == error_code.OK
    assert np.asarray(recorded).shape == (2,)
    assert calls == [("record.wav", metadata)]


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

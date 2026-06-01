import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base import play_and_record
from base.soundcard_audio_processor import SoundcardAudioProcessor
from consts import error_code


def test_stream_record_without_play_forwards_selected_input_channels(monkeypatch):
    calls = []

    class FakeProcessor:
        def start_streaming_rec(self, **kwargs):
            calls.append(kwargs)
            return play_and_record.error_code.OK, "ok"

    monkeypatch.setattr(play_and_record, "StreamingAudioProcessor", FakeProcessor)

    recorded_dict = {
        "sample_rate": 48000,
        "num_frames": 1024,
        "device": {"index": 1, "max_input_channels": 4},
        "output_device": {"index": 2, "max_output_channels": 2},
        "input_channels": [0, 1],
        "monitor_playback": True,
        "monitor_input_channel": 1,
        "monitor_gain_db": -6.0,
    }

    processor, sample_rate = play_and_record.stream_record_without_play(recorded_dict, "unused.wav", {})

    assert isinstance(processor, FakeProcessor)
    assert sample_rate == 48000
    assert calls == [
        {
            "sample_rate": 48000,
            "target_samples": 1024,
            "device": recorded_dict["device"],
            "output_device": recorded_dict["output_device"],
            "input_channels": [0, 1],
            "monitor_playback": True,
            "monitor_input_channel": 1,
            "monitor_gain_db": -6.0,
        }
    ]


def test_sd_rec_slices_selected_input_channels(monkeypatch):
    calls = []
    captured = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
        ],
        dtype=np.float32,
    )

    def fake_rec(frames, samplerate, channels, device=None, blocking=True):
        calls.append(
            {
                "frames": frames,
                "samplerate": samplerate,
                "channels": channels,
                "device": device,
                "blocking": blocking,
            }
        )
        return captured

    monkeypatch.setattr("base.soundcard_audio_processor.sd.rec", fake_rec)

    code, data = SoundcardAudioProcessor.sd_rec(
        {
            "num_frames": 2,
            "sample_rate": 48000,
            "channels": 2,
            "device": {"index": 5},
            "input_channels": [0, 2],
        }
    )

    assert code == error_code.OK
    assert calls == [
        {
            "frames": 2,
            "samplerate": 48000,
            "channels": 3,
            "device": 5,
            "blocking": True,
        }
    ]
    np.testing.assert_array_equal(data, captured[:, [0, 2]])


def test_sd_play_rec_forwards_selected_input_output_devices(monkeypatch):
    calls = []
    captured = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ],
        dtype=np.float32,
    )

    def fake_playrec(data, samplerate, channels, blocking=True, device=None):
        calls.append(
            {
                "samplerate": samplerate,
                "channels": channels,
                "blocking": blocking,
                "device": device,
            }
        )
        return captured

    monkeypatch.setattr("base.soundcard_audio_processor.sd.playrec", fake_playrec)
    monkeypatch.setattr("base.soundcard_audio_processor.save_audio_simple", lambda *args, **kwargs: None)
    monkeypatch.setattr(SoundcardAudioProcessor, "calculate_alignment", lambda self, stimulus, recorded: 0)

    code, data = SoundcardAudioProcessor().sd_play_rec(
        {
            "prepare_frames": 0,
            "prolong_frames": 0,
            "input_device": {"index": 5, "max_input_channels": 2},
            "output_device": {"index": 7, "max_output_channels": 2},
        },
        {
            "data": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "amplitude": 1.0,
            "sr": 48000,
        },
        "unused.wav",
    )

    assert code == error_code.OK
    assert calls == [
        {
            "samplerate": 48000,
            "channels": 1,
            "blocking": True,
            "device": (5, 7),
        }
    ]
    np.testing.assert_array_equal(data, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_sd_rec_uses_sr_when_sample_rate_missing(monkeypatch):
    calls = []
    captured = np.array([1.0, 2.0], dtype=np.float32)

    def fake_rec(frames, samplerate, channels, device=None, blocking=True):
        calls.append(
            {
                "frames": frames,
                "samplerate": samplerate,
                "channels": channels,
                "device": device,
                "blocking": blocking,
            }
        )
        return captured

    monkeypatch.setattr("base.soundcard_audio_processor.sd.rec", fake_rec)

    code, data = SoundcardAudioProcessor.sd_rec(
        {
            "num_frames": 2,
            "sr": 48000,
            "channels": 1,
            "device": {"index": 5},
        }
    )

    assert code == error_code.OK
    assert calls == [
        {
            "frames": 2,
            "samplerate": 48000,
            "channels": 1,
            "device": 5,
            "blocking": True,
        }
    ]
    np.testing.assert_array_equal(data, captured)


def test_sd_rec_without_input_channels_preserves_mono_vector(monkeypatch):
    captured = np.array(
        [
            [1.0],
            [2.0],
        ],
        dtype=np.float32,
    )

    def fake_rec(frames, samplerate, channels, device=None, blocking=True):
        return captured

    monkeypatch.setattr("base.soundcard_audio_processor.sd.rec", fake_rec)

    code, data = SoundcardAudioProcessor.sd_rec(
        {
            "num_frames": 2,
            "sample_rate": 48000,
        }
    )

    assert code == error_code.OK
    np.testing.assert_array_equal(data, np.array([1.0, 2.0], dtype=np.float32))

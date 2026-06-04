from types import SimpleNamespace

from base import play_and_record
from base import stimulus_resolver
import numpy as np
import pytest

from base.load_config import LoadUiConfig
from base.soundcard_audio_processor import SoundcardAudioProcessor, alignment_reference_from_stimulus
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


def test_alignment_reference_helper_is_public():
    np.testing.assert_array_equal(
        alignment_reference_from_stimulus(
            {"data": np.arange(4, dtype=float), "alignment_sample_count": 2}
        ),
        np.array([0.0, 1.0]),
    )


def test_sd_play_expands_mono_data_to_requested_output_channels(monkeypatch):
    calls = []

    def fake_play(data, samplerate, device, blocking):
        calls.append(
            {
                "data": np.asarray(data),
                "samplerate": samplerate,
                "device": device,
                "blocking": blocking,
            }
        )

    monkeypatch.setattr("base.soundcard_audio_processor.sd.play", fake_play)

    code, msg = SoundcardAudioProcessor.sd_play(
        {
            "data": np.array([0.1, 0.2], dtype=np.float32),
            "amplitude": 2.0,
            "sr": 48000,
            "device": 21,
            "blocking": False,
            "output_channels": 2,
        }
    )

    assert code == error_code.OK
    assert msg == "play successfully"
    assert len(calls) == 1
    assert calls[0]["samplerate"] == 48000
    assert calls[0]["device"] == 21
    assert calls[0]["blocking"] is False
    assert calls[0]["data"].shape == (2, 2)
    np.testing.assert_array_equal(
        calls[0]["data"],
        np.array([[0.2, 0.2], [0.4, 0.4]], dtype=np.float32),
    )


def test_sd_play_expands_single_column_mono_data_to_requested_output_channels(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "base.soundcard_audio_processor.sd.play",
        lambda data, samplerate, device, blocking: calls.append(np.asarray(data)),
    )

    code, _ = SoundcardAudioProcessor.sd_play(
        {
            "data": np.array([[0.1], [0.2]], dtype=np.float32),
            "amplitude": 2.0,
            "sr": 48000,
            "output_channels": 3,
        }
    )

    assert code == error_code.OK
    assert len(calls) == 1
    assert calls[0].shape == (2, 3)
    np.testing.assert_array_equal(
        calls[0],
        np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4]], dtype=np.float32),
    )


def test_sd_play_preserves_existing_multichannel_data_with_output_channels(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "base.soundcard_audio_processor.sd.play",
        lambda data, samplerate, device, blocking: calls.append(np.asarray(data)),
    )

    code, _ = SoundcardAudioProcessor.sd_play(
        {
            "data": np.array([[0.1, 0.3], [0.2, 0.4]], dtype=np.float32),
            "amplitude": 2.0,
            "sr": 48000,
            "output_channels": 4,
        }
    )

    assert code == error_code.OK
    assert len(calls) == 1
    assert calls[0].shape == (2, 2)
    np.testing.assert_array_equal(
        calls[0],
        np.array([[0.2, 0.6], [0.4, 0.8]], dtype=np.float32),
    )


def test_sd_play_keeps_mono_shape_when_output_channels_absent(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "base.soundcard_audio_processor.sd.play",
        lambda data, samplerate, device, blocking: calls.append(np.asarray(data)),
    )

    code, _ = SoundcardAudioProcessor.sd_play(
        {"data": np.array([0.1, 0.2], dtype=np.float32), "amplitude": 2.0, "sr": 48000}
    )

    assert code == error_code.OK
    assert len(calls) == 1
    assert calls[0].shape == (2,)
    np.testing.assert_array_equal(calls[0], np.array([0.2, 0.4], dtype=np.float32))


@pytest.mark.parametrize("invalid_output_channels", [True, False, 0, -1, 1.2, "2"])
def test_sd_play_ignores_invalid_output_channels(monkeypatch, invalid_output_channels):
    calls = []
    monkeypatch.setattr(
        "base.soundcard_audio_processor.sd.play",
        lambda data, samplerate, device, blocking: calls.append(np.asarray(data)),
    )

    code, _ = SoundcardAudioProcessor.sd_play(
        {
            "data": np.array([0.1, 0.2], dtype=np.float32),
            "amplitude": 2.0,
            "sr": 48000,
            "output_channels": invalid_output_channels,
        }
    )

    assert code == error_code.OK
    assert len(calls) == 1
    assert calls[0].shape == (2,)


def test_blocking_play_record_aligns_and_saves_tail_free_frequency_stepped_recording(monkeypatch):
    calls = {}

    def fake_playrec(playback_data, samplerate, channels, blocking):
        calls["playback_len"] = len(playback_data)
        recorded = np.concatenate([np.zeros(3), np.asarray(playback_data, dtype=float)])
        return recorded.reshape(-1, 1)

    def fake_calculate_alignment(ref, rec):
        calls["alignment_reference"] = np.asarray(ref)
        return 5

    monkeypatch.setattr("base.soundcard_audio_processor.sd.playrec", fake_playrec)
    monkeypatch.setattr(SoundcardAudioProcessor, "calculate_alignment", staticmethod(fake_calculate_alignment))
    monkeypatch.setattr(
        "base.soundcard_audio_processor.save_audio_simple",
        lambda path, audio, sr: calls.update(saved_len=len(audio), saved=np.asarray(audio), sr=sr),
    )

    stimulus_dict = {
        "data": np.arange(8, dtype=float),
        "amplitude": 2.0,
        "sr": 48000,
        "alignment_sample_count": 5,
    }
    record_dict = {"prepare_frames": 2, "prolong_frames": 4}

    code, aligned = SoundcardAudioProcessor().sd_play_rec(record_dict, stimulus_dict, "unused.wav")

    assert code == error_code.OK
    assert calls["playback_len"] == 14
    assert calls["saved_len"] == 5
    assert len(aligned) == 5
    np.testing.assert_array_equal(calls["alignment_reference"], np.arange(5, dtype=float) * 2.0)
    np.testing.assert_array_equal(aligned, np.arange(5, dtype=float) * 2.0)


def test_runtime_dict_builder_includes_alignment_sample_count_from_data_struct():
    data_struct = SimpleNamespace(
        stimulus_data=np.arange(8, dtype=float),
        stimulus_info={"amplitude": 0.5, "stimulus_method": "frequency_stepped"},
        sample_rate=48000,
        alignment_sample_count=5,
    )

    stimulus_dict, recorded_dict = LoadUiConfig.get_rec_and_play_dict_base_sequence_dict(data_struct)

    assert stimulus_dict["alignment_sample_count"] == 5
    np.testing.assert_array_equal(stimulus_dict["data"], np.arange(8, dtype=float))
    assert recorded_dict["num_frames"] == 8 + int(0.5 * 48000)


def test_runtime_dict_builder_prefers_data_struct_alignment_sample_count():
    data_struct = SimpleNamespace(
        stimulus_data=np.arange(8, dtype=float),
        stimulus_info={
            "amplitude": 0.5,
            "stimulus_method": "frequency_stepped",
            "alignment_sample_count": 7,
        },
        sample_rate=48000,
        alignment_sample_count=5,
    )

    stimulus_dict, _ = LoadUiConfig.get_rec_and_play_dict_base_sequence_dict(data_struct)

    assert stimulus_dict["alignment_sample_count"] == 5


def test_runtime_dict_builder_includes_alignment_sample_count_from_stimulus_info():
    data_struct = SimpleNamespace(
        stimulus_data=np.arange(8, dtype=float),
        stimulus_info={
            "amplitude": 0.5,
            "stimulus_method": "frequency_stepped",
            "alignment_sample_count": 5,
        },
        sample_rate=48000,
    )

    stimulus_dict, _ = LoadUiConfig.get_rec_and_play_dict_base_sequence_dict(data_struct)

    assert stimulus_dict["alignment_sample_count"] == 5


def test_runtime_dict_builder_does_not_leak_stale_alignment_count_to_legacy_stimulus(monkeypatch):
    data_struct = SimpleNamespace()

    def fake_generate(detail, logger=None):
        info = detail.get("stimulus_info") or {}
        if info.get("stimulus_method") == "frequency_stepped":
            info["stimulus_method"] = "frequency_stepped"
            info["alignment_sample_count"] = 5
            detail["stimulus_info"] = info
            detail["alignment_sample_count"] = 5
            return np.arange(8, dtype=np.float32), 48000, "generated.wav"
        return np.arange(6, dtype=np.float32), 48000, "legacy.wav"

    monkeypatch.setattr(stimulus_resolver, "generate_and_save_stimulus", fake_generate)

    stepped_detail = {
        "stimulus_info": {
            "stimulus_method": "frequency_stepped",
            "amplitude": 0.5,
            "sample_rate": 48000,
        }
    }
    stimulus_resolver.set_data_struct_stimulus_signal(data_struct, stepped_detail)
    stepped_stimulus_dict, _ = LoadUiConfig.get_rec_and_play_dict_base_sequence_dict(data_struct)
    assert stepped_stimulus_dict["alignment_sample_count"] == 5

    legacy_detail = {
        "stimulus_info": {
            "stimulus_method": "chirp",
            "amplitude": 0.5,
            "sample_rate": 48000,
            "alignment_sample_count": 5,
        }
    }
    stimulus_resolver.set_data_struct_stimulus_signal(data_struct, legacy_detail)
    legacy_stimulus_dict, _ = LoadUiConfig.get_rec_and_play_dict_base_sequence_dict(data_struct)

    assert "alignment_sample_count" not in legacy_stimulus_dict


def test_blocking_play_record_without_alignment_sample_count_keeps_existing_full_length_behavior(
    monkeypatch,
):
    calls = {}

    def fake_playrec(playback_data, samplerate, channels, blocking):
        calls["playback_len"] = len(playback_data)
        recorded = np.concatenate([np.zeros(3), np.asarray(playback_data, dtype=float)])
        return recorded.reshape(-1, 1)

    monkeypatch.setattr("base.soundcard_audio_processor.sd.playrec", fake_playrec)
    monkeypatch.setattr(SoundcardAudioProcessor, "calculate_alignment", staticmethod(lambda ref, rec: 3))
    monkeypatch.setattr(
        "base.soundcard_audio_processor.save_audio_simple",
        lambda path, audio, sr: calls.update(saved_len=len(audio), saved=np.asarray(audio), sr=sr),
    )

    stimulus_dict = {"data": np.arange(8, dtype=float), "amplitude": 2.0, "sr": 48000}
    record_dict = {"prepare_frames": 2, "prolong_frames": 4}

    code, aligned = SoundcardAudioProcessor().sd_play_rec(record_dict, stimulus_dict, "unused.wav")

    assert code == error_code.OK
    assert calls["playback_len"] == 14
    assert calls["saved_len"] == 8
    assert len(aligned) == 8


def test_blocking_play_record_clamps_negative_alignment_offset(monkeypatch):
    calls = {}

    def fake_playrec(playback_data, samplerate, channels, blocking):
        recorded = np.asarray(playback_data, dtype=float)
        return recorded.reshape(-1, 1)

    monkeypatch.setattr("base.soundcard_audio_processor.sd.playrec", fake_playrec)
    monkeypatch.setattr(SoundcardAudioProcessor, "calculate_alignment", staticmethod(lambda ref, rec: -3))
    monkeypatch.setattr(
        "base.soundcard_audio_processor.save_audio_simple",
        lambda path, audio, sr: calls.update(saved_len=len(audio), saved=np.asarray(audio), sr=sr),
    )

    stimulus_dict = {
        "data": np.arange(8, dtype=float),
        "amplitude": 1.0,
        "sr": 48000,
        "alignment_sample_count": 5,
    }
    record_dict = {"prepare_frames": 0, "prolong_frames": 0}

    code, aligned = SoundcardAudioProcessor().sd_play_rec(record_dict, stimulus_dict, "unused.wav")

    assert code == error_code.OK
    assert calls["saved_len"] == 5
    assert len(aligned) == 5
    np.testing.assert_array_equal(aligned, np.arange(5, dtype=float))
    np.testing.assert_array_equal(calls["saved"], np.arange(5, dtype=float))


def test_blocking_play_record_pads_alignment_offset_near_recording_end(monkeypatch):
    calls = {}

    def fake_playrec(playback_data, samplerate, channels, blocking):
        recorded = np.asarray(playback_data, dtype=float)
        return recorded.reshape(-1, 1)

    monkeypatch.setattr("base.soundcard_audio_processor.sd.playrec", fake_playrec)
    monkeypatch.setattr(SoundcardAudioProcessor, "calculate_alignment", staticmethod(lambda ref, rec: 6))
    monkeypatch.setattr(
        "base.soundcard_audio_processor.save_audio_simple",
        lambda path, audio, sr: calls.update(saved_len=len(audio), saved=np.asarray(audio), sr=sr),
    )

    stimulus_dict = {
        "data": np.arange(8, dtype=float),
        "amplitude": 1.0,
        "sr": 48000,
        "alignment_sample_count": 5,
    }
    record_dict = {"prepare_frames": 0, "prolong_frames": 0}

    code, aligned = SoundcardAudioProcessor().sd_play_rec(record_dict, stimulus_dict, "unused.wav")

    assert code == error_code.OK
    assert calls["saved_len"] == 5
    assert len(aligned) == 5
    np.testing.assert_array_equal(aligned, np.array([6.0, 7.0, 0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(calls["saved"], np.array([6.0, 7.0, 0.0, 0.0, 0.0]))


def test_stream_play_and_record_plays_full_tail_but_returns_tail_free_alignment_reference(monkeypatch):
    calls = []

    class FakeProcessor:
        def start_streaming_playrec(self, **kwargs):
            calls.append(kwargs)
            return play_and_record.error_code.OK, "ok"

    monkeypatch.setattr(play_and_record, "StreamingAudioProcessor", FakeProcessor)
    stimulus_dict = {
        "data": np.arange(8, dtype=float),
        "amplitude": 1.0,
        "sr": 48000,
        "alignment_sample_count": 5,
    }
    recorded_dict = {
        "prepare_frames": 2,
        "prolong_frames": 4,
        "input_device": {"index": 1, "max_input_channels": 2},
        "output_device": {"index": 2, "max_output_channels": 2},
    }

    processor, alignment_reference, sample_rate = play_and_record.stream_play_and_record(
        stimulus_dict, recorded_dict, "unused.wav", {}
    )

    assert isinstance(processor, FakeProcessor)
    assert sample_rate == 48000
    assert calls[0]["target_samples"] == 14
    assert calls[0]["stimulus_dict"] is stimulus_dict
    np.testing.assert_array_equal(alignment_reference, np.arange(5, dtype=float))


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

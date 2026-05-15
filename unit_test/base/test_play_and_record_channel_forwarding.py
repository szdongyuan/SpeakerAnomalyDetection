from base import play_and_record


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

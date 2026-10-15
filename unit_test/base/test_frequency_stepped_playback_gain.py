"""Capture real adapter output without opening any audio devices."""

import numpy as np
import pytest

import base.soundcard_audio_processor as soundcard
import base.streaming_audio_processor as streaming
from base.stimulus_signal.frequency_stepped import generate_frequency_stepped
from consts import error_code


@pytest.fixture
def unit_wave():
    return generate_frequency_stepped(
        sample_rate=48000, frequency_mode="custom_linear",
        frequencies=[1000, 2000], min_duration=0.01, min_cycles=4,
    ).data


@pytest.mark.parametrize("gain", [0.25, 0.5])
def test_sd_play_applies_gain_once(monkeypatch, unit_wave, gain):
    original = unit_wave.copy()
    captured = []
    monkeypatch.setattr(soundcard.sd, "play", lambda data, **kwargs: captured.append(data.copy()))

    code, message = soundcard.SoundcardAudioProcessor.sd_play(
        dict(data=unit_wave, amplitude=gain, sr=48000, output_channels=1)
    )

    assert code == error_code.OK, message
    np.testing.assert_allclose(captured[0], original * gain)
    np.testing.assert_array_equal(unit_wave, original)


@pytest.mark.parametrize("gain", [0.25, 0.5])
def test_sd_play_rec_applies_gain_once(monkeypatch, tmp_path, unit_wave, gain):
    original = unit_wave.copy()
    captured = []

    def playrec(data, **kwargs):
        captured.append(data.copy())
        return np.zeros((len(data), 1))

    monkeypatch.setattr(soundcard.sd, "playrec", playrec)
    monkeypatch.setattr(soundcard, "save_audio_simple", lambda *args, **kwargs: None)
    processor = soundcard.SoundcardAudioProcessor()
    monkeypatch.setattr(processor, "calculate_alignment", lambda *args: 0)

    code, recording = processor.sd_play_rec(
        dict(sr=48000, input_channels=[0], prepare_frames=0, prolong_frames=0),
        dict(data=unit_wave, amplitude=gain, sr=48000),
        str(tmp_path / "recording.wav"),
    )

    assert code == error_code.OK
    assert len(recording) == len(unit_wave)
    np.testing.assert_allclose(captured[0], original * gain)
    np.testing.assert_array_equal(unit_wave, original)


@pytest.mark.parametrize("gain", [0.25, 0.5])
def test_streaming_callback_applies_gain_once(monkeypatch, unit_wave, gain):
    original = unit_wave.copy()
    stream_args = {}

    class FakeStream:
        def __init__(self, **kwargs):
            stream_args.update(kwargs)

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(streaming.sd, "Stream", FakeStream)
    monkeypatch.setattr(streaming.sd, "query_devices", lambda **kwargs: {"max_input_channels": 1})
    processor = streaming.StreamingAudioProcessor()
    code, message = processor.start_streaming_playrec(
        dict(data=unit_wave, amplitude=gain), sample_rate=48000,
        # Keep the callback below the recording stop threshold to avoid a worker thread.
        target_samples=len(unit_wave) + 1,
        prepare_frames=0, prolong_frames=0, discard_initial_samples=0, input_channels=[0],
    )

    assert code == error_code.OK, message
    outdata = np.full((len(unit_wave), 1), np.nan, dtype=np.float32)
    try:
        stream_args["callback"](np.zeros_like(outdata), outdata, len(unit_wave), None, None)
        np.testing.assert_allclose(outdata[:, 0], original * gain, rtol=1e-6, atol=1e-8)
        np.testing.assert_array_equal(unit_wave, original)
    finally:
        processor.stop_streaming()

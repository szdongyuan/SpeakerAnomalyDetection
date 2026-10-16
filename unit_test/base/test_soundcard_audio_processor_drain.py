"""Exercise the real playback callback without opening audio hardware."""

from threading import Event, Thread

import numpy as np
import pytest

import base.soundcard_audio_processor as soundcard
from base.stimulus_signal.frequency_stepped import generate_frequency_stepped
from consts import error_code


class FakeOutput:
    def __init__(self):
        self.frames = (2048,)
        self.auto_finish = True
        self.failure = None
        self.streams = []
        self.play_calls = []
        self.events = []
        self.submitted = Event()

    def stop(self):
        self.events.append("stop")

    def play(self, data, **kwargs):
        self.play_calls.append((data.copy(), kwargs))

    def open(self, **kwargs):
        self.events.append("open")
        if self.failure == "open":
            raise RuntimeError("open failed")
        stream = FakeStream(self, kwargs)
        self.streams.append(stream)
        return stream


class CopyFailureBuffer:
    def fill(self, value):
        pass

    def __setitem__(self, key, value):
        raise RuntimeError("output copy failed")


class FakeStream:
    def __init__(self, output, kwargs):
        self.output = output
        self.kwargs = kwargs
        self.blocks = []
        self.closed = False
        self.termination = None

    def start(self):
        self.output.events.append("start")
        if self.output.failure == "start":
            raise RuntimeError("start failed")
        for index in range(10000):
            frames = self.output.frames[index % len(self.output.frames)]
            block = np.full((frames, self.kwargs["channels"]), 99, dtype=self.kwargs["dtype"])
            buffer = CopyFailureBuffer() if self.output.failure == "copy" else block
            try:
                self.kwargs["callback"](buffer, frames, None, None)
            except (soundcard.sd.CallbackStop, soundcard.sd.CallbackAbort) as error:
                self.termination = type(error)
            # CallbackStop submits the final block as well.
            self.blocks.append(block.copy())
            if self.termination is not None:
                self.output.events.append("submitted")
                self.output.submitted.set()
                if self.output.auto_finish:
                    self.finish()
                return
        raise AssertionError("callback never terminated")

    def finish(self):
        self.output.events.append("finished")
        self.kwargs["finished_callback"]()

    def close(self):
        self.output.events.append("close")
        self.closed = True


@pytest.fixture
def output(monkeypatch):
    output = FakeOutput()
    monkeypatch.setattr(soundcard.sd, "OutputStream", output.open)
    monkeypatch.setattr(soundcard.sd, "stop", output.stop)
    monkeypatch.setattr(soundcard.sd, "play", output.play)
    return output


def play(data, **kwargs):
    params = dict(data=data, amplitude=0.25, sr=48000, device=5, drain_output=True)
    params.update(kwargs)
    return soundcard.SoundcardAudioProcessor.sd_play(params)


def assert_output(output, expected):
    assert output.play_calls == []
    assert len(output.streams) == 1
    stream = output.streams[0]
    assert stream.termination is soundcard.sd.CallbackStop
    assert stream.closed
    assert output.events == ["stop", "open", "start", "submitted", "finished", "close"]
    captured = np.concatenate(stream.blocks)
    np.testing.assert_allclose(captured[:len(expected)], expected, rtol=1e-6, atol=1e-8)
    silence = 128 if captured.dtype == np.uint8 else 0
    np.testing.assert_array_equal(captured[len(expected):], silence)
    assert stream.kwargs["samplerate"] == 48000
    assert stream.kwargs["device"] == 5
    assert stream.kwargs["channels"] == expected.shape[1]
    return captured


def test_complete_frequency_stepped_wave_including_last_step_and_fadeout(output):
    signal = generate_frequency_stepped(
        sample_rate=48000, frequency_mode="octave", start_freq=80,
        stop_freq=20000, resolution="R10", min_duration=0.02,
        min_cycles=20, repeat_times=1,
    )
    original = signal.data.copy()
    code, message = play(signal.data)
    assert code == error_code.OK, message
    captured = assert_output(output, (original * 0.25).reshape(-1, 1))[:, 0]
    last = signal.segments[-1]
    np.testing.assert_allclose(
        captured[last.start_sample:last.end_sample],
        original[last.start_sample:last.end_sample] * 0.25, rtol=1e-6, atol=1e-8,
    )
    np.testing.assert_allclose(
        captured[last.boundary_zero_sample + 1:len(original)],
        original[last.boundary_zero_sample + 1:] * 0.25, rtol=1e-6, atol=1e-8,
    )
    np.testing.assert_array_equal(signal.data, original)


@pytest.mark.parametrize("length,frames", [(4096, (2048,)), (17, (2048,)), (71, (3, 17, 8))])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("channels,replicate", [(1, None), (2, None), (1, 2), (1, 3)])
def test_frame_sizes_channels_dtype_and_gain(output, length, frames, dtype, channels, replicate):
    output.frames = frames
    data = np.linspace(-0.9, 0.9, length * channels, dtype=dtype)
    if channels > 1:
        data = data.reshape(length, channels)
    original = data.copy()
    expected = (data * 0.25).reshape(length, channels)
    if replicate:
        expected = np.tile(expected, (1, replicate))
    code, message = play(data, output_channels=replicate)
    assert code == error_code.OK, message
    assert_output(output, expected)
    assert output.streams[0].kwargs["dtype"] == np.dtype("float32")
    np.testing.assert_array_equal(data, original)


def test_two_dimensional_mono_is_replicated(output):
    data = np.array([[0.1], [-0.2]], dtype=np.float32)
    code, message = play(data, output_channels=3)
    assert code == error_code.OK, message
    assert_output(output, np.tile(data * 0.25, (1, 3)))


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int8, np.uint8])
def test_integer_audio_dtype_is_preserved(output, dtype):
    data = np.arange(12, dtype=dtype)
    code, message = play(data, amplitude=1)
    assert code == error_code.OK, message
    assert_output(output, data.reshape(-1, 1))
    assert output.streams[0].kwargs["dtype"] == np.dtype(dtype)


def test_uint8_silence_remains_silent_through_partial_block_padding(output):
    data = np.full(12, 128, dtype=np.uint8)
    code, message = play(data, amplitude=1)
    assert code == error_code.OK, message
    captured = assert_output(output, data.reshape(-1, 1))
    assert captured.shape == (2048, 1)
    np.testing.assert_array_equal(captured, 128)


@pytest.mark.parametrize("shape", [(0,), (0, 2)])
def test_empty_audio_does_not_open_stream(output, shape):
    code, message = play(np.empty(shape))
    assert code == error_code.OK, message
    assert output.events == []
    assert output.play_calls == []


@pytest.mark.parametrize("data", [np.array(1.0), np.zeros((2, 2, 2))])
def test_invalid_shape_is_rejected_before_device_actions(output, data):
    code, message = play(data)
    assert code == error_code.INVALID_PLAY
    assert "dimensional" in message
    assert output.events == []
    assert output.play_calls == []


def test_nonblocking_drain_is_rejected_before_device_actions(output):
    code, message = play(np.ones(3), blocking=False)
    assert code == error_code.INVALID_PLAY
    assert "blocking" in message
    assert output.events == []
    assert output.play_calls == []


@pytest.mark.parametrize("drain", [None, False])
@pytest.mark.parametrize("blocking", [None, True, False])
def test_default_path_preserves_convenience_play(output, drain, blocking):
    data = np.array([0.2, -0.6])
    params = dict(data=data, amplitude=0.25, sr=44100, device=7)
    if drain is not None:
        params["drain_output"] = drain
    if blocking is not None:
        params["blocking"] = blocking
    code, message = soundcard.SoundcardAudioProcessor.sd_play(params)
    assert code == error_code.OK, message
    assert output.events == []
    assert len(output.play_calls) == 1
    actual, kwargs = output.play_calls[0]
    np.testing.assert_array_equal(actual, data * 0.25)
    assert kwargs == dict(samplerate=44100, device=7, blocking=True if blocking is None else blocking)


def test_waits_for_finished_notification_before_returning_or_closing(output):
    output.auto_finish = False
    returned = Event()
    result = []

    def run():
        result.append(play(np.ones(9)))
        returned.set()

    worker = Thread(target=run)
    worker.start()
    try:
        assert output.submitted.wait(2), "draining callback was not used"
        assert not returned.wait(0.05)
        assert not output.streams[0].closed
        assert output.events == ["stop", "open", "start", "submitted"]
        output.streams[0].finish()
        assert returned.wait(2)
        assert result[0][0] == error_code.OK
        assert output.streams[0].closed
        assert output.events[-2:] == ["finished", "close"]
    finally:
        for stream in output.streams:
            stream.finish()
        worker.join(2)
        assert not worker.is_alive()


@pytest.mark.parametrize("failure", ["open", "start", "wait"])
def test_stream_failure_returns_error_and_closes_created_stream(output, monkeypatch, failure):
    output.failure = failure
    if failure == "wait":
        class FailingWaitEvent:
            def set(self):
                pass

            def wait(self):
                raise RuntimeError("wait failed")

        monkeypatch.setattr(soundcard, "Event", FailingWaitEvent, raising=False)
    code, message = play(np.ones(9))
    assert code == error_code.INVALID_PLAY
    assert f"{failure} failed" in message
    assert output.play_calls == []
    assert len(output.streams) == (0 if failure == "open" else 1)
    assert all(stream.closed for stream in output.streams)


def test_callback_copy_failure_wakes_waiter_without_finished_notification(output):
    output.failure = "copy"
    output.auto_finish = False
    result = []
    worker = Thread(target=lambda: result.append(play(np.ones(9))))
    worker.start()
    try:
        worker.join(2)
        assert not worker.is_alive(), "callback error left caller waiting"
        assert result[0][0] == error_code.INVALID_PLAY
        assert "output copy failed" in result[0][1]
        assert output.streams[0].termination is soundcard.sd.CallbackAbort
        assert output.streams[0].closed
        assert "finished" not in output.events
        assert output.play_calls == []
    finally:
        for stream in output.streams:
            stream.finish()
        worker.join(2)
        assert not worker.is_alive()

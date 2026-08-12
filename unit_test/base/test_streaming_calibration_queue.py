from unittest import mock
from types import SimpleNamespace

import numpy as np
import pytest

from base.streaming_audio_processor import StreamingAudioProcessor
from consts import error_code


@pytest.fixture(autouse=True)
def isolated_logger():
    logger = SimpleNamespace(
        info=mock.Mock(),
        warning=mock.Mock(),
        error=mock.Mock(),
    )
    with mock.patch(
        "base.streaming_audio_processor.LogManager.set_log_handler",
        return_value=logger,
    ):
        yield


class _FakeInputStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def test_single_physical_input_channel_is_selected_and_accumulated():
    created_streams = []

    def create_stream(**kwargs):
        stream = _FakeInputStream(**kwargs)
        created_streams.append(stream)
        return stream

    processor = StreamingAudioProcessor()
    input_device = {
        "index": 7,
        "name": "Two Channel Input",
        "hostapi": 1,
        "max_input_channels": 2,
    }

    with mock.patch(
        "base.streaming_audio_processor.sd.InputStream",
        side_effect=create_stream,
    ):
        code, _ = processor.start_streaming_rec(
            sample_rate=44100,
            target_samples=10,
            device=input_device,
            input_channels=[1],
        )

    assert code == error_code.OK
    assert processor._rec_in_sel == [1]
    assert created_streams[0].kwargs["channels"] == 2
    assert created_streams[0].kwargs["device"] == 7

    callback = created_streams[0].kwargs["callback"]
    callback(
        np.array([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32),
        2,
        None,
        None,
    )
    emit = mock.Mock()
    fake_sign = SimpleNamespace(
        stream_audio_chunk_signal=SimpleNamespace(emit=emit)
    )
    with mock.patch("base.streaming_audio_processor.sign", fake_sign):
        processor.process_queue(emit_signal=False)

    np.testing.assert_array_equal(
        processor.get_recorded_data(),
        np.array([10.0, 20.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        processor.get_recorded_data_multi(),
        np.array([[10.0], [20.0]], dtype=np.float32),
    )
    emit.assert_not_called()
    processor.stop_streaming()


def test_process_queue_keeps_legacy_array_payload_compatible():
    processor = StreamingAudioProcessor()
    processor.audio_queue.put_nowait(np.array([1.0, 2.0], dtype=np.float32))

    processor.process_queue(emit_signal=False)

    np.testing.assert_array_equal(
        processor.get_recorded_data(),
        np.array([1.0, 2.0], dtype=np.float32),
    )
    assert processor.get_recorded_data_multi().shape == (2, 1)


def test_process_queue_still_emits_waveform_payload_by_default():
    processor = StreamingAudioProcessor()
    processor.audio_queue.put_nowait(np.array([1.0, 2.0], dtype=np.float32))
    emit = mock.Mock()
    fake_sign = SimpleNamespace(
        stream_audio_chunk_signal=SimpleNamespace(emit=emit)
    )

    with mock.patch("base.streaming_audio_processor.sign", fake_sign):
        processor.process_queue()

    emitted_payload = emit.call_args.args[0]
    np.testing.assert_array_equal(
        emitted_payload["mono"],
        np.array([1.0, 2.0], dtype=np.float32),
    )
    assert emitted_payload["multi"].shape == (2, 1)


def test_invalid_selected_channel_is_rejected_before_opening_stream():
    processor = StreamingAudioProcessor()
    input_device = {
        "index": 7,
        "name": "Two Channel Input",
        "hostapi": 1,
        "max_input_channels": 2,
    }

    with mock.patch("base.streaming_audio_processor.sd.InputStream") as input_stream:
        code, message = processor.start_streaming_rec(
            sample_rate=44100,
            target_samples=10,
            device=input_device,
            input_channels=[2],
        )

    assert code == error_code.INVALID_RECORD
    assert "Invalid input_channels" in message
    assert processor.is_recording is False
    input_stream.assert_not_called()


def test_stream_open_failure_resets_recording_state():
    processor = StreamingAudioProcessor()

    with mock.patch(
        "base.streaming_audio_processor.sd.InputStream",
        side_effect=OSError("device unavailable"),
    ):
        code, message = processor.start_streaming_rec(
            sample_rate=44100,
            target_samples=10,
            device={"index": 7, "max_input_channels": 1},
            input_channels=[0],
        )

    assert code == error_code.INVALID_RECORD
    assert "device unavailable" in message
    assert processor.is_recording is False
    assert processor.error_occurred is True

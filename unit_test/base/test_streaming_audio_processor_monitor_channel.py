import numpy as np
import pytest

from base import streaming_audio_processor
from base.streaming_audio_processor import StreamingAudioProcessor
from consts import error_code


def _prepare_processor(input_channels, monitor_input_channel):
    processor = StreamingAudioProcessor()
    processor._rec_in_sel = list(input_channels)
    processor.monitor_gain_linear = 1.0
    processor.target_samples = 100
    processor.samples_captured = 0
    processor._monitor_input_column = processor._resolve_monitor_input_column(
        input_channels,
        monitor_input_channel,
    )
    return processor


def test_monitor_duplex_callback_outputs_selected_input_channel():
    processor = _prepare_processor([1, 3], 3)
    indata = np.array(
        [
            [0.00, 0.10, 0.20, 0.30],
            [0.01, 0.11, 0.21, 0.31],
        ],
        dtype=np.float32,
    )
    outdata = np.zeros((2, 2), dtype=np.float32)

    processor.monitor_duplex_callback(indata, outdata, 2, None, None)

    np.testing.assert_allclose(outdata[:, 0], [0.30, 0.31])
    np.testing.assert_allclose(outdata[:, 1], [0.30, 0.31])
    payload = processor.audio_queue.get_nowait()
    np.testing.assert_allclose(payload["multi"], [[0.10, 0.30], [0.11, 0.31]])


def test_monitor_duplex_callback_falls_back_to_first_selected_channel():
    processor = _prepare_processor([1, 3], 9)
    indata = np.array(
        [
            [0.00, 0.10, 0.20, 0.30],
            [0.01, 0.11, 0.21, 0.31],
        ],
        dtype=np.float32,
    )
    outdata = np.zeros((2, 1), dtype=np.float32)

    processor.monitor_duplex_callback(indata, outdata, 2, None, None)

    np.testing.assert_allclose(outdata[:, 0], [0.10, 0.11])


def test_monitor_duplex_callback_applies_gain_and_clips():
    processor = _prepare_processor([0], 0)
    processor.monitor_gain_linear = 2.0
    indata = np.array([[0.75], [-0.75]], dtype=np.float32)
    outdata = np.zeros((2, 1), dtype=np.float32)

    processor.monitor_duplex_callback(indata, outdata, 2, None, None)

    np.testing.assert_allclose(outdata[:, 0], [1.0, -1.0])


def test_monitor_duplex_callback_uses_trimmed_payload_for_final_chunk(monkeypatch):
    started_threads = []

    class FakeThread:
        def __init__(self, target, daemon=False):
            self.target = target
            self.daemon = daemon
            started_threads.append(self)

        def start(self):
            return None

    monkeypatch.setattr(streaming_audio_processor.threading, "Thread", FakeThread)

    processor = _prepare_processor([1, 3], 3)
    processor.target_samples = 1
    indata = np.array(
        [
            [0.00, 0.10, 0.20, 0.30],
            [0.01, 0.11, 0.21, 0.31],
        ],
        dtype=np.float32,
    )
    outdata = np.full((2, 1), -9.0, dtype=np.float32)

    processor.monitor_duplex_callback(indata, outdata, 2, None, None)

    np.testing.assert_allclose(outdata[:, 0], [0.30, 0.0])
    payload = processor.audio_queue.get_nowait()
    np.testing.assert_allclose(payload["multi"], [[0.10, 0.30]])
    assert started_threads


def test_monitor_duplex_outputs_silence_during_discarded_warmup():
    processor = _prepare_processor([0], 0)
    processor.discard_initial_samples = 5
    processor.samples_discarded = 0
    indata = np.arange(4, dtype=np.float32).reshape(-1, 1)
    outdata = np.full((4, 1), -9.0, dtype=np.float32)

    processor.monitor_duplex_callback(indata, outdata, 4, None, None)

    np.testing.assert_allclose(outdata[:, 0], np.zeros(4, dtype=np.float32))
    assert processor.audio_queue.empty()
    assert processor.samples_captured == 0
    assert processor.samples_discarded == 4


def test_monitor_duplex_pads_silence_for_partial_discarded_warmup():
    processor = _prepare_processor([0], 0)
    processor.discard_initial_samples = 2
    processor.samples_discarded = 0
    indata = (np.arange(5, dtype=np.float32) / 10.0).reshape(-1, 1)
    outdata = np.full((5, 1), -9.0, dtype=np.float32)

    processor.monitor_duplex_callback(indata, outdata, 5, None, None)

    np.testing.assert_allclose(outdata[:, 0], [0.0, 0.0, 0.2, 0.3, 0.4])
    payload = processor.audio_queue.get_nowait()
    np.testing.assert_allclose(payload["mono"], [0.2, 0.3, 0.4])
    np.testing.assert_allclose(payload["multi"], [[0.2], [0.3], [0.4]])
    assert processor.samples_captured == 3
    assert processor.samples_discarded == 2


def test_start_streaming_rec_initializes_monitor_input_column(monkeypatch):
    created = []

    class FakeStream:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.started = False

        def start(self):
            self.started = True

    monkeypatch.setattr(streaming_audio_processor.sd, "Stream", FakeStream)

    processor = streaming_audio_processor.StreamingAudioProcessor()
    code, message = processor.start_streaming_rec(
        sample_rate=44100,
        target_samples=8,
        device={"name": "Mic", "index": 1, "max_input_channels": 8},
        input_channels=[1, 3],
        output_device={"name": "Speaker", "index": 2, "max_output_channels": 2},
        monitor_playback=True,
        monitor_gain_db=0.0,
        monitor_input_channel=3,
    )

    assert code == error_code.OK
    assert "monitor" in message
    assert processor._monitor_input_column == 1
    assert processor._rec_in_sel == [1, 3]
    assert created[0]["channels"] == (8, 2)
    assert created[0]["device"] == (1, 2)


def test_start_streaming_rec_preserves_final_positional_input_channels(monkeypatch):
    created = []

    class FakeStream:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def start(self):
            return None

    monkeypatch.setattr(streaming_audio_processor.sd, "Stream", FakeStream)

    processor = streaming_audio_processor.StreamingAudioProcessor()
    code, _ = processor.start_streaming_rec(
        44100,
        8,
        None,
        {"name": "Mic", "index": 1, "max_input_channels": 8},
        {"name": "Speaker", "index": 2, "max_output_channels": 2},
        True,
        0.0,
        3,
        [1, 3],
    )

    assert code == error_code.OK
    assert processor._rec_in_sel == [1, 3]
    assert processor._monitor_input_column == 1
    assert created[0]["channels"] == (8, 2)


def test_start_streaming_rec_uses_device_max_input_channels(monkeypatch):
    created = []

    class FakeInputStream:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.started = False

        def start(self):
            self.started = True

    monkeypatch.setattr(streaming_audio_processor.sd, "InputStream", FakeInputStream)

    processor = streaming_audio_processor.StreamingAudioProcessor()
    code, message = processor.start_streaming_rec(
        sample_rate=44100,
        target_samples=8,
        device={"name": "Mic", "index": 1, "max_input_channels": 8},
        input_channels=[1, 3],
    )

    assert code == error_code.OK
    assert message
    assert processor._rec_in_sel == [1, 3]
    assert processor.input_channels == 8
    assert created[0]["channels"] == 8
    assert created[0]["device"] == 1


@pytest.mark.parametrize("value", [None, -1, True, False, "bad"])
def test_start_streaming_rec_invalid_discard_initial_samples_defaults_to_zero(monkeypatch, value):
    class FakeInputStream:
        def __init__(self, **kwargs):
            pass

        def start(self):
            return None

    monkeypatch.setattr(streaming_audio_processor.sd, "InputStream", FakeInputStream)

    processor = streaming_audio_processor.StreamingAudioProcessor()
    code, _ = processor.start_streaming_rec(
        sample_rate=44100,
        target_samples=8,
        device={"name": "Mic", "index": 1, "max_input_channels": 2},
        discard_initial_samples=value,
    )

    assert code == error_code.OK
    assert processor.discard_initial_samples == 0
    assert processor.samples_discarded == 0


def test_start_streaming_playrec_uses_device_max_input_channels(monkeypatch):
    created = []

    class FakeStream:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.started = False

        def start(self):
            self.started = True

    monkeypatch.setattr(streaming_audio_processor.sd, "Stream", FakeStream)

    processor = streaming_audio_processor.StreamingAudioProcessor()
    code, message = processor.start_streaming_playrec(
        stimulus_dict={"data": np.array([0.1, 0.2], dtype=np.float32), "amplitude": 1.0},
        sample_rate=44100,
        target_samples=4,
        input_device={"name": "Mic", "index": 1, "max_input_channels": 8},
        output_device={"name": "Speaker", "index": 2, "max_output_channels": 2},
        prepare_frames=1,
        prolong_frames=1,
        input_channels=[1, 3],
    )

    assert code == error_code.OK
    assert "play+record" in message
    assert processor._rec_in_sel == [1, 3]
    assert processor.input_channels == 8
    assert created[0]["channels"] == (8, 1)
    assert created[0]["device"] == (1, 2)


@pytest.mark.parametrize("value", [None, -1, True, False, "bad"])
def test_start_streaming_playrec_invalid_discard_initial_samples_defaults_to_zero(monkeypatch, value):
    class FakeStream:
        def __init__(self, **kwargs):
            pass

        def start(self):
            return None

    monkeypatch.setattr(streaming_audio_processor.sd, "Stream", FakeStream)

    processor = streaming_audio_processor.StreamingAudioProcessor()
    code, _ = processor.start_streaming_playrec(
        stimulus_dict={"data": np.array([0.1, 0.2], dtype=np.float32), "amplitude": 1.0},
        sample_rate=44100,
        target_samples=2,
        input_device={"name": "Mic", "index": 1, "max_input_channels": 2},
        output_device={"name": "Speaker", "index": 2, "max_output_channels": 1},
        prepare_frames=0,
        prolong_frames=0,
        discard_initial_samples=value,
    )

    assert code == error_code.OK
    assert processor.discard_initial_samples == 0
    assert processor.samples_discarded == 0


def test_playrec_duplex_callback_queues_selected_channels(monkeypatch):
    created = []

    class FakeStream:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def start(self):
            return None

    monkeypatch.setattr(streaming_audio_processor.sd, "Stream", FakeStream)

    processor = streaming_audio_processor.StreamingAudioProcessor()
    code, _ = processor.start_streaming_playrec(
        stimulus_dict={"data": np.array([0.5, 0.6], dtype=np.float32), "amplitude": 1.0},
        sample_rate=44100,
        target_samples=2,
        input_device={"name": "Mic", "index": 1, "max_input_channels": 4},
        output_device={"name": "Speaker", "index": 2, "max_output_channels": 1},
        prepare_frames=0,
        prolong_frames=0,
        input_channels=[1, 3],
    )

    assert code == error_code.OK
    callback = created[0]["callback"]
    indata = np.array([[0.0, 0.1, 0.2, 0.3], [0.01, 0.11, 0.21, 0.31]], dtype=np.float32)
    outdata = np.zeros((2, 1), dtype=np.float32)

    callback(indata, outdata, 2, None, None)

    payload = processor.audio_queue.get_nowait()
    np.testing.assert_allclose(payload["multi"], [[0.1, 0.3], [0.11, 0.31]])
    np.testing.assert_allclose(payload["mono"], [0.2, 0.21])


def test_playrec_duplex_prepends_delay_silence_and_discards_warmup(monkeypatch):
    created = []

    class FakeStream:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def start(self):
            return None

    monkeypatch.setattr(streaming_audio_processor.sd, "Stream", FakeStream)

    processor = streaming_audio_processor.StreamingAudioProcessor()
    code, _ = processor.start_streaming_playrec(
        stimulus_dict={"data": np.array([10.0, 11.0], dtype=np.float32), "amplitude": 1.0},
        sample_rate=44100,
        target_samples=4,
        input_device={"name": "Mic", "index": 1, "max_input_channels": 1},
        output_device={"name": "Speaker", "index": 2, "max_output_channels": 1},
        prepare_frames=1,
        prolong_frames=1,
        discard_initial_samples=2,
    )

    assert code == error_code.OK
    callback = created[0]["callback"]
    indata = np.arange(4, dtype=np.float32).reshape(-1, 1)
    outdata = np.full((4, 1), -9.0, dtype=np.float32)

    callback(indata, outdata, 4, None, None)

    np.testing.assert_allclose(outdata[:, 0], [0.0, 0.0, 0.0, 10.0])
    payload = processor.audio_queue.get_nowait()
    np.testing.assert_allclose(payload["mono"], [2.0, 3.0])
    np.testing.assert_allclose(payload["multi"], [[2.0], [3.0]])
    assert processor.samples_captured == 2
    assert processor.samples_discarded == 2


def test_playrec_duplex_callback_trims_final_multichannel_chunk(monkeypatch):
    created = []
    started_threads = []

    class FakeThread:
        def __init__(self, target, daemon=False):
            self.target = target
            self.daemon = daemon
            started_threads.append(self)

        def start(self):
            return None

    class FakeStream:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def start(self):
            return None

    monkeypatch.setattr(streaming_audio_processor.threading, "Thread", FakeThread)
    monkeypatch.setattr(streaming_audio_processor.sd, "Stream", FakeStream)

    processor = streaming_audio_processor.StreamingAudioProcessor()
    code, _ = processor.start_streaming_playrec(
        stimulus_dict={"data": np.array([0.5, 0.6], dtype=np.float32), "amplitude": 1.0},
        sample_rate=44100,
        target_samples=1,
        input_device={"name": "Mic", "index": 1, "max_input_channels": 4},
        output_device={"name": "Speaker", "index": 2, "max_output_channels": 1},
        prepare_frames=0,
        prolong_frames=0,
        input_channels=[1, 3],
    )

    assert code == error_code.OK
    callback = created[0]["callback"]
    indata = np.array([[0.0, 0.1, 0.2, 0.3], [0.01, 0.11, 0.21, 0.31]], dtype=np.float32)
    outdata = np.zeros((2, 1), dtype=np.float32)

    callback(indata, outdata, 2, None, None)

    payload = processor.audio_queue.get_nowait()
    np.testing.assert_allclose(payload["multi"], [[0.1, 0.3]])
    np.testing.assert_allclose(payload["mono"], [0.2])
    assert processor.samples_captured == 1
    assert started_threads


def test_playrec_processed_multichannel_chunks_keep_mono_recorded_data(monkeypatch):
    created = []
    stream_signal = _FakeStreamSignal()
    monkeypatch.setattr(
        streaming_audio_processor,
        "sign",
        type("FakeSignals", (), {"stream_audio_chunk_signal": stream_signal})(),
    )

    class FakeStream:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def start(self):
            return None

    monkeypatch.setattr(streaming_audio_processor.sd, "Stream", FakeStream)

    processor = streaming_audio_processor.StreamingAudioProcessor()
    code, _ = processor.start_streaming_playrec(
        stimulus_dict={"data": np.array([0.5, 0.6], dtype=np.float32), "amplitude": 1.0},
        sample_rate=44100,
        target_samples=4,
        input_device={"name": "Mic", "index": 1, "max_input_channels": 4},
        output_device={"name": "Speaker", "index": 2, "max_output_channels": 1},
        prepare_frames=0,
        prolong_frames=0,
        input_channels=[1, 3],
    )

    assert code == error_code.OK
    callback = created[0]["callback"]
    indata = np.array([[0.0, 0.1, 0.2, 0.3], [0.01, 0.11, 0.21, 0.31]], dtype=np.float32)
    outdata = np.zeros((2, 1), dtype=np.float32)

    callback(indata, outdata, 2, None, None)
    processor.process_queue()

    np.testing.assert_allclose(processor.get_recorded_data(), [0.2, 0.21])
    assert processor.get_recorded_data().ndim == 1
    np.testing.assert_allclose(processor.get_recorded_data_multi(), [[0.1, 0.3], [0.11, 0.31]])


def test_start_streaming_rec_uses_default_input_device_max_channels(monkeypatch):
    created = []

    class FakeInputStream:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def start(self):
            return None

    def fake_query_devices(*args, **kwargs):
        assert kwargs == {"kind": "input"}
        return {"name": "Default Mic", "index": 5, "max_input_channels": 6}

    monkeypatch.setattr(streaming_audio_processor.sd, "InputStream", FakeInputStream)
    monkeypatch.setattr(streaming_audio_processor.sd, "query_devices", fake_query_devices)

    processor = streaming_audio_processor.StreamingAudioProcessor()
    code, _ = processor.start_streaming_rec(sample_rate=44100, target_samples=8, input_channels=[1])

    assert code == error_code.OK
    assert created[0]["channels"] == 6


@pytest.mark.parametrize("input_channels", [True, -1, [True], [False], [-1], [1.0], [1.5], ["1"], [8]])
def test_start_streaming_rec_rejects_invalid_input_channels(input_channels):
    processor = streaming_audio_processor.StreamingAudioProcessor()
    code, message = processor.start_streaming_rec(
        sample_rate=44100,
        target_samples=8,
        device={"name": "Mic", "index": 1, "max_input_channels": 8},
        input_channels=input_channels,
    )

    assert code == error_code.INVALID_RECORD
    assert "input_channels" in message
    assert processor.is_recording is False


@pytest.mark.parametrize("input_channels", [True, -1, [True], [False], [-1], [1.0], [1.5], ["1"], [8]])
def test_start_streaming_playrec_rejects_invalid_input_channels_without_recording(input_channels):
    processor = streaming_audio_processor.StreamingAudioProcessor()
    code, message = processor.start_streaming_playrec(
        stimulus_dict={"data": np.array([0.1, 0.2], dtype=np.float32), "amplitude": 1.0},
        sample_rate=44100,
        target_samples=2,
        input_device={"name": "Mic", "index": 1, "max_input_channels": 8},
        output_device={"name": "Speaker", "index": 2, "max_output_channels": 1},
        prepare_frames=0,
        prolong_frames=0,
        input_channels=input_channels,
    )

    assert code == error_code.INVALID_RECORD
    assert "input_channels" in message
    assert processor.is_recording is False


@pytest.mark.parametrize("input_channels, expected", [(None, [0]), (False, [0]), (0, [0]), ([], [0]), (2, [0, 1])])
def test_start_streaming_rec_input_channel_compatibility_values(monkeypatch, input_channels, expected):
    class FakeInputStream:
        def __init__(self, **kwargs):
            pass

        def start(self):
            return None

    monkeypatch.setattr(streaming_audio_processor.sd, "InputStream", FakeInputStream)

    processor = streaming_audio_processor.StreamingAudioProcessor()
    code, _ = processor.start_streaming_rec(
        sample_rate=44100,
        target_samples=8,
        device={"name": "Mic", "index": 1, "max_input_channels": 8},
        input_channels=input_channels,
    )

    assert code == error_code.OK
    assert processor._rec_in_sel == expected


@pytest.mark.parametrize("device_info", [{"name": "Default Mic", "max_input_channels": 0}, None])
def test_start_streaming_rec_rejects_default_input_without_positive_max_channels(monkeypatch, device_info):
    def fake_query_devices(*args, **kwargs):
        assert kwargs == {"kind": "input"}
        return device_info

    monkeypatch.setattr(streaming_audio_processor.sd, "query_devices", fake_query_devices)

    processor = streaming_audio_processor.StreamingAudioProcessor()
    code, message = processor.start_streaming_rec(sample_rate=44100, target_samples=8, input_channels=[0])

    assert code == error_code.INVALID_RECORD
    assert "max_input_channels" in message


def test_start_streaming_rec_numpy_input_channels_validate_inside_failed_start_path():
    processor = streaming_audio_processor.StreamingAudioProcessor()

    code, message = processor.start_streaming_rec(
        sample_rate=44100,
        target_samples=8,
        device={"name": "Mic", "index": 1, "max_input_channels": 4},
        input_channels=np.array([0, 4]),
    )

    assert code == error_code.INVALID_RECORD
    assert "input_channels" in message
    assert processor.is_recording is False


def test_record_only_audio_callback_preserves_selected_channels_and_counts_frames(monkeypatch):
    started_threads = []

    class FakeThread:
        def __init__(self, target, daemon=False):
            self.target = target
            self.daemon = daemon
            started_threads.append(self)

        def start(self):
            return None

    monkeypatch.setattr(streaming_audio_processor.threading, "Thread", FakeThread)

    processor = StreamingAudioProcessor()
    processor._rec_in_sel = [1, 3]
    processor.target_samples = 3
    processor.samples_captured = 0
    indata = np.array(
        [
            [0.00, 0.10, 0.20, 0.30],
            [0.01, 0.11, 0.21, 0.31],
            [0.02, 0.12, 0.22, 0.32],
            [0.03, 0.13, 0.23, 0.33],
            [0.04, 0.14, 0.24, 0.34],
        ],
        dtype=np.float32,
    )

    processor._audio_callback(indata, 5, None, None)

    assert processor.samples_captured == 3
    payload = processor.audio_queue.get_nowait()
    np.testing.assert_allclose(
        payload["multi"],
        [[0.10, 0.30], [0.11, 0.31], [0.12, 0.32]],
    )
    assert payload["multi"].shape == (3, 2)
    np.testing.assert_allclose(payload["mono"], [0.20, 0.21, 0.22])
    assert started_threads


def test_record_only_audio_callback_discards_initial_samples_before_queue(monkeypatch):
    started_threads = []

    class FakeThread:
        def __init__(self, target, daemon=False):
            self.target = target
            self.daemon = daemon
            started_threads.append(self)

        def start(self):
            return None

    monkeypatch.setattr(streaming_audio_processor.threading, "Thread", FakeThread)

    processor = StreamingAudioProcessor()
    processor._rec_in_sel = [0]
    processor.target_samples = 3
    processor.samples_captured = 0
    processor.discard_initial_samples = 2
    processor.samples_discarded = 0

    processor._audio_callback(np.arange(5, dtype=np.float32).reshape(-1, 1), 5, None, None)

    payload = processor.audio_queue.get_nowait()
    np.testing.assert_array_equal(payload["mono"], np.array([2, 3, 4], dtype=np.float32))
    np.testing.assert_array_equal(payload["multi"], np.array([[2], [3], [4]], dtype=np.float32))
    assert processor.samples_captured == 3
    assert processor.samples_discarded == 2
    assert started_threads


def test_record_only_warmup_spanning_full_callback_does_not_queue_or_count_retained_samples(monkeypatch):
    started_threads = []

    class FakeThread:
        def __init__(self, target, daemon=False):
            self.target = target
            self.daemon = daemon
            started_threads.append(self)

        def start(self):
            return None

    monkeypatch.setattr(streaming_audio_processor.threading, "Thread", FakeThread)

    processor = StreamingAudioProcessor()
    processor._rec_in_sel = [0]
    processor.target_samples = 3
    processor.samples_captured = 0
    processor.discard_initial_samples = 5
    processor.samples_discarded = 0

    processor._audio_callback(np.arange(4, dtype=np.float32).reshape(-1, 1), 4, None, None)

    assert processor.audio_queue.empty()
    assert processor.samples_captured == 0
    assert processor.samples_discarded == 4
    assert started_threads == []

    processor._audio_callback(np.arange(4, 8, dtype=np.float32).reshape(-1, 1), 4, None, None)

    payload = processor.audio_queue.get_nowait()
    np.testing.assert_array_equal(payload["mono"], np.array([5, 6, 7], dtype=np.float32))
    assert processor.samples_captured == 3
    assert processor.samples_discarded == 5
    assert started_threads


def test_playrec_warmup_spanning_full_callback_does_not_queue_or_count_retained_samples(monkeypatch):
    created = []
    started_threads = []

    class FakeThread:
        def __init__(self, target, daemon=False):
            self.target = target
            self.daemon = daemon
            started_threads.append(self)

        def start(self):
            return None

    class FakeStream:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def start(self):
            return None

    monkeypatch.setattr(streaming_audio_processor.threading, "Thread", FakeThread)
    monkeypatch.setattr(streaming_audio_processor.sd, "Stream", FakeStream)

    processor = streaming_audio_processor.StreamingAudioProcessor()
    code, _ = processor.start_streaming_playrec(
        stimulus_dict={"data": np.array([0.5, 0.6], dtype=np.float32), "amplitude": 1.0},
        sample_rate=44100,
        target_samples=3,
        input_device={"name": "Mic", "index": 1, "max_input_channels": 1},
        output_device={"name": "Speaker", "index": 2, "max_output_channels": 1},
        prepare_frames=0,
        prolong_frames=0,
        discard_initial_samples=5,
    )

    assert code == error_code.OK
    callback = created[0]["callback"]
    callback(np.arange(4, dtype=np.float32).reshape(-1, 1), np.zeros((4, 1), dtype=np.float32), 4, None, None)

    assert processor.audio_queue.empty()
    assert processor.samples_captured == 0
    assert processor.samples_discarded == 4
    assert started_threads == []

    callback(np.arange(4, 8, dtype=np.float32).reshape(-1, 1), np.zeros((4, 1), dtype=np.float32), 4, None, None)

    payload = processor.audio_queue.get_nowait()
    np.testing.assert_array_equal(payload["mono"], np.array([5, 6, 7], dtype=np.float32))
    assert processor.samples_captured == 3
    assert processor.samples_discarded == 5
    assert started_threads


class _FakeStreamSignal:
    def __init__(self):
        self.emitted = []

    def emit(self, chunk):
        self.emitted.append(chunk)


def test_process_queue_emits_payload_and_get_recorded_data_preserves_multichannel(monkeypatch):
    stream_signal = _FakeStreamSignal()
    monkeypatch.setattr(
        streaming_audio_processor,
        "sign",
        type("FakeSignals", (), {"stream_audio_chunk_signal": stream_signal})(),
    )

    processor = StreamingAudioProcessor()
    payload = {
        "mono": np.array([0.20, 0.21], dtype=np.float32),
        "multi": np.array([[0.10, 0.30], [0.11, 0.31]], dtype=np.float32),
    }
    processor.audio_queue.put_nowait(payload)

    processor.process_queue()

    emitted = stream_signal.emitted
    assert len(emitted) == 1
    assert emitted[0] is payload
    np.testing.assert_allclose(processor.get_recorded_data(), payload["multi"])


def test_process_queue_remains_compatible_with_legacy_mono_chunks(monkeypatch):
    stream_signal = _FakeStreamSignal()
    monkeypatch.setattr(
        streaming_audio_processor,
        "sign",
        type("FakeSignals", (), {"stream_audio_chunk_signal": stream_signal})(),
    )

    processor = StreamingAudioProcessor()
    chunk = np.array([0.10, 0.20], dtype=np.float32)
    processor.audio_queue.put_nowait(chunk)

    processor.process_queue()

    emitted = stream_signal.emitted
    assert emitted == [chunk]
    np.testing.assert_allclose(processor.get_recorded_data(), chunk)


def test_get_recorded_data_unwraps_directly_accumulated_payloads_as_mono():
    processor = StreamingAudioProcessor()
    processor.accumulated_chunks.append(
        {
            "mono": np.array([0.20, 0.21], dtype=np.float32),
            "multi": np.array([[0.10, 0.30], [0.11, 0.31]], dtype=np.float32),
        }
    )
    processor.accumulated_chunks.append(
        {
            "mono": np.array([0.22], dtype=np.float32),
            "multi": np.array([[0.12, 0.32]], dtype=np.float32),
        }
    )

    np.testing.assert_allclose(
        processor.get_recorded_data(),
        np.array([0.20, 0.21, 0.22], dtype=np.float32),
    )

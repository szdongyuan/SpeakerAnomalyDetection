import ast
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import sip
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from base import streaming_audio_processor
from base.streaming_audio_processor import StreamingAudioProcessor
from base.utils import custom_signals
from base.utils.custom_signals import MySignals
from ui.sequence import sequence_widget_analysis_ops
from ui.sequence.multichannel_waveform_session import MultichannelWaveformSession
from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin


ROOT = Path(__file__).resolve().parents[2]


def _class_method_source(path, class_name, method_name):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return ast.get_source_segment(source, method_node)


def test_main_ui_wires_streaming_events_with_queued_delivery_before_capture():
    ui_ops = _class_method_source(
        ROOT / "ui/sequence/sequence_widget_ui_ops.py",
        "SequenceWidgetUiOpsMixin",
        "init_ui",
    )
    analysis_ops = (ROOT / "ui/sequence/sequence_widget_analysis_ops.py").read_text(
        encoding="utf-8"
    )

    assert "stream_audio_queue_ready_signal.connect" in ui_ops
    assert "self._on_streaming_queue_ready" in ui_ops
    assert "stream_audio_recording_finished_signal.connect" in ui_ops
    assert "self._on_streaming_recording_finished" in ui_ops
    assert ui_ops.count("Qt.QueuedConnection") >= 2
    assert ui_ops.index("stream_audio_queue_ready_signal.connect") < analysis_ops.index(
        "stream_record_without_play("
    )


def test_main_streaming_lifecycle_has_no_poll_timer():
    widget_init = _class_method_source(
        ROOT / "ui/sequence/sequence_widget.py", "SequenceWindow", "__init__"
    )
    start = _class_method_source(
        ROOT / "ui/sequence/sequence_widget_analysis_ops.py",
        "SequenceWidgetAnalysisOpsMixin",
        "judge_play_and_record",
    )
    streaming_source = (
        ROOT / "ui/sequence/sequence_widget_streaming_ops.py"
    ).read_text(encoding="utf-8")

    for source in (widget_init, start, streaming_source):
        assert "streaming_poll_timer" not in source
        assert "_poll_streaming_queue" not in source
        assert ".start(50)" not in source
    assert "QTimer(" in widget_init  # unrelated timers remain supported


class _FakeProcessor:
    def __init__(self, events=None):
        self.process_queue = mock.Mock(
            side_effect=(lambda: events.append("drain")) if events is not None else None
        )


class _StreamingHost(SequenceWidgetStreamingOpsMixin):
    def __init__(self, processor, events=None):
        self.streaming_processor = processor
        self._streaming_completion_processor = None
        self.events = events if events is not None else []

    def _on_streaming_complete(self):
        self.events.append("complete")


def test_queue_ready_drains_only_the_active_processor():
    active = _FakeProcessor()
    stale = _FakeProcessor()
    host = _StreamingHost(active)

    host._on_streaming_queue_ready(stale)
    host._on_streaming_queue_ready(active)

    stale.process_queue.assert_not_called()
    active.process_queue.assert_called_once_with()


def test_queue_ready_ignores_processor_after_completion_dispatch_begins():
    active = _FakeProcessor()
    host = _StreamingHost(active)
    host._streaming_completion_processor = active

    host._on_streaming_queue_ready(active)

    active.process_queue.assert_not_called()


@pytest.mark.parametrize("clear_processor", [False, True])
def test_recording_finished_drains_before_completion_and_is_idempotent(clear_processor):
    events = []
    active = _FakeProcessor(events)
    stale = _FakeProcessor(events)
    host = _StreamingHost(active, events)

    def complete():
        events.append("complete")
        if clear_processor:
            host.streaming_processor = None

    host._on_streaming_complete = complete
    host._on_streaming_recording_finished(stale)
    host._on_streaming_recording_finished(active)
    host._on_streaming_recording_finished(active)

    assert events == ["drain", "complete"]
    stale.process_queue.assert_not_called()
    active.process_queue.assert_called_once_with()


class _Writer:
    def __init__(self, *args, **kwargs):
        self.chunks = []

    def write_chunk(self, chunk):
        self.chunks.append(chunk)


class _PlotWindow:
    def __init__(self, channel, calls):
        self.channel_index = channel
        self._calls = calls

    def set_data(self, time_axis, amplitude):
        self.calls.append(
            SimpleNamespace(
                channel=self.channel_index,
                time=np.array(time_axis, copy=True),
                amplitude=np.array(amplitude, copy=True),
            )
        )

    @property
    def calls(self):
        return self._calls


class _Workspace:
    def __init__(self, channels):
        self.calls = []
        self._windows = [_PlotWindow(channel, self.calls) for channel in channels]

    def all_subwindows(self):
        return list(self._windows)


class _ChunkHost(SequenceWidgetStreamingOpsMixin):
    def __init__(self, processor):
        self.streaming_processor = processor
        self._streaming_completion_processor = None
        self.streaming_wav_writer = _Writer()
        self.data_struct = SimpleNamespace(sample_rate=48_000)
        self._direction_waveform_cache = {}
        self._streaming_first_chunk_logged = True
        self.default_logger = logging.getLogger(__name__)
        self._recording_input_channels = (0, 2)
        self.channel_workspace = _Workspace(self._recording_input_channels)
        self._streaming_waveform_session = MultichannelWaveformSession(
            max_points=self._WAVEFORM_DISPLAY_MAX_POINTS
        )
        self._streaming_waveform_generation = 0
        self._streaming_waveform_refresh_scheduled = False
        self._streaming_waveform_pending = False
        self._streaming_waveform_live_enabled = False
        self._streaming_waveform_failure_logged = False
        self._streaming_chunk_contract_failed = False
        self.scheduled = []

    def _resolve_active_recording_waveform_direction(self):
        return "01"

    def _resolve_recording_acq_detail(self):
        return {"startup_trim_ms": 0}

    def _schedule_streaming_waveform_callback(self, callback):
        self.scheduled.append(callback)


def test_real_queue_event_chain_preserves_waveform_and_wav_chunk(monkeypatch):
    # Earlier Qt tests may tear down the QApplication that owned the imported
    # module singleton. Keep that failure mode explicit while giving this test
    # sole ownership of the bus used by StreamingAudioProcessor.
    dead_global_bus = MySignals()
    sip.delete(dead_global_bus)
    monkeypatch.setattr(custom_signals, "sign", dead_global_bus)
    with pytest.raises(RuntimeError, match="has been deleted"):
        dead_global_bus.stream_audio_chunk_signal.emit({})

    signal_bus = MySignals()
    monkeypatch.setattr(streaming_audio_processor, "sign", signal_bus)
    with mock.patch(
        "base.streaming_audio_processor.LogManager.set_log_handler",
        return_value=logging.getLogger(__name__),
    ):
        processor = StreamingAudioProcessor()
    host = _ChunkHost(processor)
    host._begin_streaming_waveform_session(48_000, 0, "01")
    payload = {
        "mono": np.array([2.0, 3.0, 4.0], dtype=np.float32),
        "multi": np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]], dtype=np.float32),
    }
    processor.audio_queue.put_nowait(payload)
    receiver = host.on_audio_chunk_received
    signal_bus.stream_audio_chunk_signal.connect(receiver)
    try:
        host._on_streaming_queue_ready(processor)
    finally:
        signal_bus.stream_audio_chunk_signal.disconnect(receiver)

    assert processor.audio_queue.empty()
    np.testing.assert_array_equal(host.streaming_wav_writer.chunks[0], payload["multi"])
    assert host.channel_workspace.calls == []
    assert host._direction_waveform_cache == {}
    assert len(host.scheduled) == 1

    host.scheduled.pop()()

    assert [call.channel for call in host.channel_workspace.calls] == [0, 2]
    np.testing.assert_array_equal(
        host.channel_workspace.calls[0].amplitude,
        payload["multi"][:, 0],
    )
    np.testing.assert_array_equal(
        host.channel_workspace.calls[1].amplitude,
        payload["multi"][:, 1],
    )
    for live_call in host.channel_workspace.calls:
        np.testing.assert_allclose(live_call.time, np.arange(3) / 48_000)
    assert host._direction_waveform_cache == {}


class _Button:
    def setDisabled(self, disabled):
        self.disabled = disabled


class _WorkflowHost(SequenceWidgetAnalysisOpsMixin, SequenceWidgetStreamingOpsMixin):
    def __init__(self, recorded_dict, events):
        self._recorded_dict = recorded_dict
        self.events = events
        self._record_workflow_busy = False
        self.last_play_count = None
        self.analysis_window = []
        self._analysis_result_summary_window = None
        self.player_status_flag = False
        self.replayer_btn = _Button()
        self.data_btn = _Button()
        self.streaming_processor = None
        self._streaming_completion_processor = object()
        self.streaming_wav_writer = None
        self.streaming_stimulus_data = None
        self.streaming_mode = None
        self._streaming_waveform_session = MultichannelWaveformSession(
            max_points=self._WAVEFORM_DISPLAY_MAX_POINTS
        )
        self._streaming_waveform_generation = 0
        self._streaming_waveform_refresh_scheduled = False
        self._streaming_waveform_pending = False
        self._streaming_waveform_live_enabled = False
        self._streaming_waveform_failure_logged = False
        self._streaming_chunk_contract_failed = False
        self._recording_input_channels = (0, 1)
        self.channel_workspace = _Workspace(self._recording_input_channels)
        self.scheduled = []
        self._active_input_channels = [0, 1]
        self.recorded_path = "record.wav"
        self.recorded_signal_info = {"name": "capture"}
        self.default_logger = logging.getLogger(__name__)

    def checked_work_status_message(self):
        return False

    def _close_analysis_windows(self):
        return None

    def _clear_plot_area(self):
        return None

    def _cleanup_streaming_resources(self):
        self.streaming_processor = None
        self.streaming_wav_writer = None

    def update_player_btn_is_playing(self):
        return None

    def reset_work_pram(self, label, count=None):
        return self._recorded_dict, 48_000

    def _should_use_streaming_recording(self):
        return True

    def _begin_recent_session_for_current_run(self):
        return None

    def _capture_recording_wav_calibration_metadata(self):
        return None

    def _schedule_streaming_waveform_callback(self, callback):
        self.scheduled.append(callback)

    def _on_streaming_complete(self):
        self.events.append("complete")


@pytest.mark.parametrize("monitor_playback", [False, True])
def test_main_workflow_routes_start_through_event_completion(monkeypatch, monitor_playback):
    events = []
    recorded_dict = {
        "sample_rate": 48_000,
        "monitor_playback": monitor_playback,
        "use_streaming_recording": True,
    }
    processor = _FakeProcessor(events)
    calls = []

    def start_streaming(actual_dict, path, signal_info):
        calls.append((actual_dict, path, signal_info))
        return processor, "ok"

    monkeypatch.setattr(sequence_widget_analysis_ops, "StreamingWavWriter", _Writer)
    monkeypatch.setattr(sequence_widget_analysis_ops, "stream_record_without_play", start_streaming)
    host = _WorkflowHost(recorded_dict, events)

    host.judge_play_and_record()

    assert calls == [(recorded_dict, "record.wav", {"name": "capture"})]
    assert host.streaming_processor is processor
    assert not hasattr(host, "streaming_poll_timer")
    assert host._streaming_completion_processor is None

    host._on_streaming_recording_finished(processor)

    assert events == ["drain", "complete"]
    processor.process_queue.assert_called_once_with()

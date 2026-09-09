import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np


def _request(request_id, path):
    return SimpleNamespace(request_id=request_id, path=path, channels=(0,), sample_rate=48000)


def test_context_registry_keeps_old_result_owned_after_new_capture_becomes_active():
    from ui.sequence.recording_process_context import RecordingProcessContext
    from ui.sequence.sequence_widget_recording_process_ops import (
        SequenceWidgetRecordingProcessOpsMixin,
    )

    host = SimpleNamespace()
    host._recording_process_contexts = {}
    host._active_recording_process_id = "B"
    request_a = _request("A", "A.wav")
    request_b = _request("B", "B.wav")
    session_a = SimpleNamespace(request=request_a)
    session_b = SimpleNamespace(request=request_b)
    context_a = RecordingProcessContext(
        request_a, "forward", ("A", None), False, session=session_a)
    context_b = RecordingProcessContext(
        request_b, "reverse", ("B", None), True, session=session_b)
    host._recording_process_contexts.update(A=context_a, B=context_b)

    lookup = SequenceWidgetRecordingProcessOpsMixin._recording_context_for_session.__get__(host)
    owns = SequenceWidgetRecordingProcessOpsMixin._is_current_recording_process.__get__(host)

    assert lookup(session_a) is context_a
    assert lookup(session_b) is context_b
    assert owns(session_a)
    assert owns(session_b)
    assert host._active_recording_process_id == "B"


def test_contexts_freeze_request_path_direction_and_tcp_destination_independently():
    from ui.sequence.recording_process_context import RecordingProcessContext

    request_a = _request("A", "A.wav")
    request_b = _request("B", "B.wav")
    context_a = RecordingProcessContext(request_a, "forward", ("A", ("127.0.0.1", 1)), False)
    context_b = RecordingProcessContext(request_b, "reverse", ("B", ("127.0.0.1", 2)), True)

    assert context_a.request.path == "A.wav"
    assert context_a.direction == "forward"
    assert context_a.tcp_completion == ("A", ("127.0.0.1", 1))
    assert context_b.request.path == "B.wav"
    assert context_b.direction == "reverse"
    assert context_b.tcp_completion == ("B", ("127.0.0.1", 2))


def test_preview_ownership_is_only_the_active_capture_not_any_known_context():
    from ui.sequence.recording_process_context import RecordingProcessContext
    from ui.sequence.sequence_widget_recording_process_ops import (
        SequenceWidgetRecordingProcessOpsMixin,
    )

    request_a = _request("A", "A.wav")
    request_b = _request("B", "B.wav")
    session_a = SimpleNamespace(request=request_a)
    session_b = SimpleNamespace(request=request_b)
    host = SimpleNamespace(
        _recording_process_contexts={
            "A": RecordingProcessContext(
                request_a, "forward", None, True, session=session_a),
            "B": RecordingProcessContext(
                request_b, "reverse", None, True, session=session_b),
        },
        _active_recording_process_id="B",
    )
    active = SequenceWidgetRecordingProcessOpsMixin._is_active_recording_process.__get__(host)

    assert not active(session_a)
    assert active(session_b)


class _ResultHost:
    def __init__(self):
        from ui.sequence.sequence_widget_recording_process_ops import (
            SequenceWidgetRecordingProcessOpsMixin,
        )

        self._recording_process_contexts = {}
        self._active_recording_process_id = "B"
        self._record_workflow_busy = True
        self._normalize_final_recording_array = lambda data, _channels: data
        self._validate_final_waveform_workspace = lambda _channels: ("windows",)
        self._on_streaming_complete = mock.Mock(return_value=True)
        self._finalize_recording_channel_selection = mock.Mock()
        self._send_recording_tcp_finish = mock.Mock()
        self._handle_invalid_recording = mock.Mock()
        self._clear_unpublished_recording_source = mock.Mock()
        self.default_logger = SimpleNamespace(info=lambda *_: None, warning=lambda *_: None, error=lambda *_: None)
        class InlineExecutor:
            @staticmethod
            def submit(request_id, work, deliver):
                from ui.sequence.request_scoped_recording_executor import (
                    RequestScopedExecutionOutcome,
                )
                deliver(RequestScopedExecutionOutcome(request_id, value=work()))
                return True
        self._get_request_scoped_recording_executor = lambda: InlineExecutor()
        for name in (
            "_recording_context_for_session", "_is_active_recording_process",
            "_on_process_recording_result", "_on_process_recording_accepted",
            "_on_process_recording_released", "_on_process_recording_release_failed",
            "_publish_recording_context",
            "_notify_process_recording_finished", "_drop_recording_context",
            "_recording_contexts", "_sync_recording_workflow_busy",
            "_on_process_recording_failed", "_on_process_recording_cancelled",
            "_cancel_process_recording",
            "_recording_context_owns_active_workflow",
            "_submit_request_scoped_recording_publication",
            "_deliver_request_scoped_recording_publication",
            "_retry_request_scoped_recording_publication",
            "_request_scoped_recording_generation",
            "_request_scoped_generation_is_current",
        ):
            setattr(self, name, getattr(SequenceWidgetRecordingProcessOpsMixin, name).__get__(self))


def _result_session(request_id, path):
    request = SimpleNamespace(
        request_id=request_id, path=path, channels=(0,), sample_rate=48000,
        device={"backend": "soundcard"},
    )
    session = SimpleNamespace(
        request=request, state="completed", released=threading.Event(),
        release_error=None, accept_result=mock.Mock(), reject_result=mock.Mock(),
        cancel=mock.Mock(),
    )
    descriptor = SimpleNamespace(
        channels=(0,), path=path, sample_rate=48000, final_frames=2, warnings=(),
    )
    multi = np.asarray([[1.0], [2.0]], dtype=np.float32)
    audio = SimpleNamespace(descriptor=descriptor, multi=multi, mono=multi[:, 0].copy())
    return session, audio


def test_a_result_publishes_by_frozen_context_while_b_remains_active():
    from ui.sequence.recording_process_context import RecordingProcessContext

    host = _ResultHost()
    session_a, audio_a = _result_session("A", "A.wav")
    session_b, _audio_b = _result_session("B", "B.wav")
    context_a = RecordingProcessContext(session_a.request, "forward", ("A", None), False,
                                        session=session_a, final_windows=("A-windows",))
    context_b = RecordingProcessContext(session_b.request, "reverse", ("B", None), True,
                                        session=session_b, final_windows=("B-windows",))
    host._recording_process_contexts.update(A=context_a, B=context_b)

    host._on_process_recording_result(session_a, audio_a)
    host._on_process_recording_accepted(session_a, audio_a)
    session_a.released.set()
    host._on_process_recording_released(session_a)

    assert session_a.accept_result.call_count == 1
    assert host._recording_process_contexts == {"B": context_b}
    assert host._active_recording_process_id == "B"
    call = host._on_streaming_complete.call_args
    assert call.kwargs["recording_context"] is context_a
    assert call.kwargs["final_waveform_windows"] == ("A-windows",)
    host._finalize_recording_channel_selection.assert_not_called()


def test_delayed_a_failure_does_not_clear_or_fail_active_b():
    from ui.sequence.recording_process_context import RecordingProcessContext

    host = _ResultHost()
    session_a, _ = _result_session("A", "A.wav")
    session_b, _ = _result_session("B", "B.wav")
    context_a = RecordingProcessContext(session_a.request, "forward", None, False, session=session_a)
    context_b = RecordingProcessContext(session_b.request, "reverse", None, True, session=session_b)
    host._recording_process_contexts.update(A=context_a, B=context_b)

    host._on_process_recording_failed(
        session_a, SimpleNamespace(stage="reader", message="late A failure"))

    assert host._recording_process_contexts == {"B": context_b}
    assert host._active_recording_process_id == "B"
    host._handle_invalid_recording.assert_not_called()
    host._clear_unpublished_recording_source.assert_not_called()


def test_b_result_can_publish_before_a_and_each_context_finishes_once():
    from ui.sequence.recording_process_context import RecordingProcessContext

    host = _ResultHost()
    session_a, audio_a = _result_session("A", "A.wav")
    session_b, audio_b = _result_session("B", "B.wav")
    context_a = RecordingProcessContext(
        session_a.request, "forward", ("A", None), False, session=session_a)
    context_b = RecordingProcessContext(
        session_b.request, "reverse", ("B", None), True, session=session_b)
    host._recording_process_contexts.update(A=context_a, B=context_b)

    for session, audio in ((session_b, audio_b), (session_a, audio_a)):
        host._on_process_recording_result(session, audio)
        host._on_process_recording_accepted(session, audio)
        session.released.set()
        host._on_process_recording_released(session)

    assert host._recording_process_contexts == {}
    assert [call.kwargs["recording_context"] for call in
            host._on_streaming_complete.call_args_list] == [context_b, context_a]


def test_close_cancels_only_active_b_while_a_finalizer_remains_request_owned():
    from ui.sequence.recording_process_context import RecordingProcessContext

    host = _ResultHost()
    session_a, _ = _result_session("A", "A.wav")
    session_b, _ = _result_session("B", "B.wav")
    context_a = RecordingProcessContext(session_a.request, "forward", None, False, session=session_a, final=True)
    context_b = RecordingProcessContext(session_b.request, "reverse", None, True, session=session_b)
    host._recording_process_contexts.update(A=context_a, B=context_b)
    host._recording_process_session = session_b
    host._recording_process_audio = None
    host._recording_process_validated_audio = None
    host._recording_ve_device = None
    host._recording_wav_calibration_metadata = None

    host._cancel_process_recording()

    session_b.cancel.assert_called_once()
    session_a.cancel.assert_not_called()
    assert not context_a.cancelled
    assert context_b.cancelled and context_b.cleanup_owned

    host._drop_recording_context = mock.Mock(wraps=host._drop_recording_context)
    host._on_process_recording_cancelled(session_b, SimpleNamespace())
    assert host._recording_process_contexts == {"A": context_a, "B": context_b}
    session_b.released.set()
    host._on_process_recording_released(session_b)
    host._on_process_recording_released(session_b)

    assert host._recording_process_contexts == {"A": context_a}
    assert host._active_recording_process_id is None
    host._drop_recording_context.assert_called_once_with(context_b)


def test_cleanup_owned_release_failed_retires_context_exactly_once():
    from ui.sequence.recording_process_context import RecordingProcessContext

    host = _ResultHost()
    session, _audio = _result_session("B", "B.wav")
    context = RecordingProcessContext(
        session.request, "reverse", None, False, session=session,
        cancelled=True, cleanup_owned=True)
    host._recording_process_contexts = {"B": context}
    host._active_recording_process_id = "B"
    host._recording_process_session = session
    host._drop_recording_context = mock.Mock(wraps=host._drop_recording_context)

    host._on_process_recording_release_failed(session, "cleanup denied")
    assert host._recording_process_contexts == {}
    assert host._active_recording_process_id is None
    assert host._record_workflow_busy is False
    host._on_process_recording_release_failed(session, "duplicate")
    host._on_process_recording_released(session)
    host._on_process_recording_cancelled(session, SimpleNamespace())
    host._on_process_recording_failed(
        session, SimpleNamespace(stage="late", message="duplicate"))

    assert host._recording_process_contexts == {}
    assert host._active_recording_process_id is None
    assert host._record_workflow_busy is False
    host._drop_recording_context.assert_called_once_with(context)
    host._handle_invalid_recording.assert_not_called()

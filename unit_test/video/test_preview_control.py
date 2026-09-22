"""Preview control must not affect admission or the recording state machine."""

from dataclasses import replace
import multiprocessing
import pickle
from unittest.mock import Mock

import numpy as np
import pytest

from base.video.models import Command, CommandKind, Event, EventKind, VideoState
from base.video.preview import PreviewMailbox
from base.video.runtime import VideoRuntime
from base.video.service import VideoService
from base.video.worker import SimulationOptions, simulated_video_worker
from unit_test.video.test_recording import config_for


def preview_command(enabled, revision=1, generation=1):
    return Command(CommandKind.SET_PREVIEW, generation, f"preview-{revision}",
                   preview_enabled=enabled, preview_revision=revision)


def test_disabled_preview_skips_conversion_but_admits_recording_frame(tmp_path):
    mailbox = Mock(width=160, height=90)
    runtime = VideoRuntime(Mock(), mailbox, 1, replace(config_for(tmp_path), enabled=False),
                           capture_factory=Mock())
    runtime.online = runtime.accept_frames = True
    runtime.session_id = "recording"
    frame = Mock(planes=[])
    frame.to_ndarray.side_effect = AssertionError("preview conversion must be skipped")
    runtime.on_frame(frame, 1.0)
    item = runtime.work.get_nowait()
    assert item[:2] == ("frame", "recording") and item[2][0] is frame
    assert runtime.raw_frames == 1
    mailbox.publish.assert_not_called()


def test_conversion_finishing_after_toggle_cannot_publish_old_frame(tmp_path):
    channel, mailbox = Mock(), Mock(width=160, height=90)
    runtime = VideoRuntime(channel, mailbox, 1, config_for(tmp_path), capture_factory=Mock())
    runtime.online = True

    def convert(**_kwargs):
        runtime._command(preview_command(False, 1))
        runtime._command(preview_command(True, 2))
        return np.zeros((90, 160, 3), dtype=np.uint8)

    frame = Mock(to_ndarray=Mock(side_effect=convert))
    runtime.on_frame(frame, 1.0)
    mailbox.publish.assert_not_called()
    frame.to_ndarray.side_effect = None
    frame.to_ndarray.return_value = np.zeros((90, 160, 3), dtype=np.uint8)
    runtime.on_frame(frame, 2.0)
    assert mailbox.publish.call_args.args[1] == 2
    runtime._command(preview_command(False, 1))
    runtime._command(preview_command(False, 3, generation=2))
    assert runtime.preview_enabled and runtime.preview_revision == 2
    assert channel.send.call_count == 2


def test_full_preview_command_queue_preserves_active_recording():
    service = VideoService(worker_target=simulated_video_worker, worker_options=SimulationOptions())
    service._started = True  # Isolate bounded-queue rejection without a draining supervisor.
    service._state.apply(Event(EventKind.READY, 1, 1, 1.0))
    assert service._state.request_start("session")
    service._state.apply(Event(EventKind.STARTED, 1, 2, 2.0, session_id="session"))
    for revision in range(service._commands.maxsize):
        assert service.set_preview(bool(revision % 2))
    before, revision = service.status, service.preview_revision
    assert not service.set_preview(False)
    assert service.status == before and service.preview_revision == revision
    assert before.recording == "recording" and before.record_intent


def test_preview_ack_preserves_session_and_old_events_cannot_override():
    state = VideoState(1)
    state.apply(Event(EventKind.READY, 1, 1, 1.0))
    state.request_start("session")
    state.apply(Event(EventKind.STARTED, 1, 2, 2.0, session_id="session"))
    before = state.status
    ack = Event(EventKind.PREVIEW_CHANGED, 1, 3, 3.0, command_id="preview-2",
                preview_enabled=False, preview_revision=2)
    assert pickle.loads(pickle.dumps(ack)) == ack
    assert state.apply(ack)
    assert state.status == replace(before, preview_enabled=False, preview_revision=2)
    assert not state.apply(replace(ack, generation=2, sequence=4, preview_revision=3))
    assert not state.apply(replace(ack, sequence=4, preview_revision=1, preview_enabled=True))
    assert not state.status.preview_enabled
    state.apply(Event(EventKind.COMPLETED, 1, 5, 5.0, session_id="session"))
    assert state.request_start("next")
    assert not state.status.preview_enabled and state.status.preview_revision == 2


@pytest.mark.parametrize("fields", [
    {"session_id": "recording"}, {"preview_enabled": None},
    {"preview_revision": 0}, {"preview_revision": True},
])
def test_invalid_preview_command_rejected(fields):
    with pytest.raises(ValueError, match="preview"):
        replace(preview_command(False), **fields)


def test_mailbox_carries_preview_revision_without_changing_frame_sequence():
    mailbox = PreviewMailbox(multiprocessing.get_context("spawn"), width=2, height=2)
    assert mailbox.publish(bytes(12), revision=7)
    frame = mailbox.latest()
    assert frame.revision == 7 and frame.sequence == 1
    assert mailbox.latest(after_sequence=frame.sequence) is None

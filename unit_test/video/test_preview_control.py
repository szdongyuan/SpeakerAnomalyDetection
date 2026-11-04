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
from base.video.config import VideoConfig


def preview_command(enabled, revision=1, generation=1):
    return Command(CommandKind.SET_PREVIEW, generation, f"preview-{revision}",
                   preview_enabled=enabled, preview_revision=revision)


def test_disabled_preview_skips_conversion(tmp_path):
    mailbox = Mock(width=160, height=90)
    runtime = VideoRuntime(Mock(), mailbox, 1, VideoConfig(),
                           capture_factory=Mock())
    runtime.online = True
    frame = Mock(planes=[])
    frame.to_ndarray.side_effect = AssertionError("preview conversion must be skipped")
    runtime.on_frame(frame, 1.0)
    mailbox.publish.assert_not_called()


def test_conversion_finishing_after_toggle_cannot_publish_old_frame(tmp_path):
    channel, mailbox = Mock(), Mock(width=160, height=90)
    runtime = VideoRuntime(channel, mailbox, 1, VideoConfig(enabled=True, device_id="USB"), capture_factory=Mock())
    runtime.online = True

    def convert(**_kwargs):
        runtime.set_preview(preview_command(False, 1))
        runtime.set_preview(preview_command(True, 2))
        return np.zeros((90, 160, 3), dtype=np.uint8)

    frame = Mock(to_ndarray=Mock(side_effect=convert))
    runtime.on_frame(frame, 1.0)
    mailbox.publish.assert_not_called()
    frame.to_ndarray.side_effect = None
    frame.to_ndarray.return_value = np.zeros((90, 160, 3), dtype=np.uint8)
    runtime.on_frame(frame, 2.0)
    assert mailbox.publish.call_args.args[1] == 2
    runtime.set_preview(preview_command(False, 1))
    runtime.set_preview(preview_command(False, 3, generation=2))
    assert runtime.preview_enabled and runtime.preview_revision == 2
    assert channel.send.call_count == 2


def test_full_preview_command_queue_preserves_preview_state():
    service = VideoService(worker_target=simulated_video_worker, worker_options=SimulationOptions())
    service._started = True  # Isolate bounded-queue rejection without a draining supervisor.
    service._state.apply(Event(EventKind.READY, 1, 1, 1.0))
    for revision in range(service._commands.maxsize):
        assert service.set_preview(bool(revision % 2))
    before, revision = service.status, service.preview_revision
    assert not service.set_preview(False)
    assert service.status == before and service.preview_revision == revision
    assert before.connection == "ready"


def test_preview_ack_preserves_connection_and_old_events_cannot_override():
    state = VideoState(1)
    state.apply(Event(EventKind.READY, 1, 1, 1.0))
    before = state.status
    ack = Event(EventKind.PREVIEW_CHANGED, 1, 3, 3.0, command_id="preview-2",
                preview_enabled=False, preview_revision=2)
    assert pickle.loads(pickle.dumps(ack)) == ack
    assert state.apply(ack)
    assert state.status == replace(before, preview_enabled=False, preview_revision=2)
    assert not state.apply(replace(ack, generation=2, sequence=4, preview_revision=3))
    assert not state.apply(replace(ack, sequence=4, preview_revision=1, preview_enabled=True))
    assert not state.status.preview_enabled


@pytest.mark.parametrize("fields", [
    {"preview_enabled": None},
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

"""Exercise gap recovery with production codecs, muxing and worker threads."""

import threading
import time
from types import SimpleNamespace

import pytest

from base.video.models import Command, CommandKind, EventKind
from base.video.recording import EncodedFrame, FrameEncoder, RecordingSession
from base.video.runtime import VideoRuntime
from unit_test.video.test_recording import config_for, decoded, frame
from unit_test.video.test_runtime import SyntheticCapture, wait_for


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_recovery_waits_until_post_gap_media_is_readable(tmp_path, codec):
    session = RecordingSession(config_for(tmp_path, codec=codec), "same-file")
    encoder = FrameEncoder(session.config)
    session.prepare(encoder.parameters)
    writer = session.segment_writer
    for i in range(15):
        session.write(encoder.write(frame(i), 100 + i / 10), 100 + i / 10)
    assert writer.has_written_media
    session.gap("eight second interruption")
    assert session.segment_writer is writer
    assert list(session.directory.glob("*.mp4")) == [writer.path]
    assert not session.write(EncodedFrame(()), 109.4)
    # This packet may flush only the PRE-gap fragment. That is not recovery.
    assert not session.write(encoder.write(frame(15), 109.5), 109.5)
    assert session.summary["state"] == "recovering"
    recovered = False
    for i in range(16, 40):
        stamp = 108 + i / 10
        if session.write(encoder.write(frame(i), stamp), stamp):
            recovered = True
            assert any(t >= 9.5 for t, _ in decoded(writer.path))
            break
    assert recovered
    assert session.summary["state"] == "recording"
    session.write_tail(encoder.finish())
    session.finish()
    files = list(session.directory.glob("*.mp4"))
    assert len(files) == 1
    times = [t for t, _ in decoded(files[0])]
    assert times[:15] == pytest.approx([i / 10 for i in range(15)])
    assert times[15] == pytest.approx(9.5)


def drain(runtime):
    wait_for(lambda: runtime.work.empty() and not runtime.encoder_busy
             and runtime.encoded.snapshot()["items"] == 0 and not runtime.writer_busy)


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_repeated_eight_second_gaps_keep_one_file(tmp_path, codec):
    runtime = VideoRuntime(None, None, 1, config_for(tmp_path, codec=codec),
                           capture_factory=SyntheticCapture)
    runtime.online, runtime.preview_due = True, float("inf")
    runtime.writer.start()
    runtime.encoder_thread.start()
    stamps = []
    try:
        runtime._command(Command(CommandKind.START, 1, "start", "same-file"))
        wait_for(lambda: runtime.accept_frames)
        for i in range(75):
            if i in (25, 50):
                runtime.on_connection(False, "test gap")
                runtime.on_connection(True, "")
            stamp = 100 + i / 10 + 8 * ((i >= 25) + (i >= 50))
            stamps.append(stamp - 100)
            runtime.on_frame(frame(i), stamp)
            drain(runtime)
        assert sum(e[0] == EventKind.STARTED for e in runtime.events.queue) == 3
        runtime._command(Command(CommandKind.STOP, 1, "stop", "same-file"))
        wait_for(lambda: not runtime.session_id)
        files = list(tmp_path.glob("video/*/*.mp4"))
        assert len(files) == 1
        assert [t for t, _ in decoded(files[0])] == pytest.approx(stamps, abs=1 / 90000)
    finally:
        runtime._command(Command(CommandKind.SHUTDOWN, 1, "close"))
        runtime.encoder_thread.join(5)
        runtime.writer.join(5)
    assert not runtime.encoder_thread.is_alive() and not runtime.writer.is_alive()


@pytest.mark.parametrize("cancel", [CommandKind.STOP, CommandKind.SHUTDOWN])
def test_stop_during_unconfirmed_recovery_cannot_restart_recording(tmp_path, cancel):
    runtime = VideoRuntime(None, None, 1, config_for(tmp_path), capture_factory=SyntheticCapture)
    runtime.online, runtime.preview_due = True, float("inf")
    runtime.writer.start()
    runtime.encoder_thread.start()
    try:
        runtime._command(Command(CommandKind.START, 1, "start", "cancel-recovery"))
        wait_for(lambda: runtime.accept_frames)
        for i in range(15):
            runtime.on_frame(frame(i), 100 + i / 10)
            drain(runtime)
        runtime.on_connection(False, "gap")
        runtime.on_connection(True, "")
        runtime.on_frame(frame(15), 110)
        drain(runtime)
        assert sum(e[0] == EventKind.STARTED for e in runtime.events.queue) == 1
        assert any(e[0] == EventKind.RECOVERING for e in runtime.events.queue)
        runtime._command(Command(cancel, 1, "cancel", "cancel-recovery"))
        runtime.on_connection(False, "late disconnect")
        runtime.on_connection(True, "")
        runtime.on_frame(frame(16), 110.1)
        wait_for(lambda: not runtime.session_id)
        assert not runtime.accept_frames
        assert sum(e[0] == EventKind.STARTED for e in runtime.events.queue) == 1
        files = list(tmp_path.glob("video/*/*.mp4"))
        assert len(files) == 1 and len(decoded(files[0])) == 16
    finally:
        runtime._command(Command(CommandKind.SHUTDOWN, 1, "close"))
        runtime.encoder_thread.join(5)
        runtime.writer.join(5)
    assert not runtime.encoder_thread.is_alive() and not runtime.writer.is_alive()


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_gap_crosses_boundary_without_empty_files_or_resetting_schedule(tmp_path, codec):
    config = SimpleNamespace(**(vars(config_for(tmp_path, codec=codec))
                               | {"segment_duration_seconds": 1}))
    runtime = VideoRuntime(None, None, 1, config, capture_factory=SyntheticCapture)
    runtime.online, runtime.preview_due = True, float("inf")
    runtime.writer.start()
    runtime.encoder_thread.start()
    try:
        runtime._command(Command(CommandKind.START, 1, "start", "boundary"))
        wait_for(lambda: runtime.accept_frames)
        runtime.on_frame(frame(0), 100)
        drain(runtime)
        runtime.on_connection(False, "offline across boundary")
        drain(runtime)
        assert not [p for p in tmp_path.glob("video/*/*.mp4") if ".recording." not in p.name]
        # Timer closes the old segment even when no new camera frame arrives.
        wait_for(lambda: len([p for p in tmp_path.glob("video/*/*.mp4")
                              if ".recording." not in p.name]) == 1, timeout=3)
        assert not list(tmp_path.glob("video/*/*.recording.mp4"))
        time.sleep(1.1)  # A wholly empty time slot must not create a file.
        assert len(list(tmp_path.glob("video/*/*.mp4"))) == 1
        runtime.on_connection(True, "")
        for i, stamp in enumerate((102.5, 102.9, 103.0, 103.1), 1):
            runtime.on_frame(frame(i), stamp)
            drain(runtime)
        runtime._command(Command(CommandKind.STOP, 1, "stop", "boundary"))
        wait_for(lambda: not runtime.session_id)
        files = sorted(tmp_path.glob("video/*/*.mp4"))
        assert [len(decoded(p)) for p in files] == [1, 2, 2]
        assert not list(tmp_path.glob("video/*/*.recording.mp4"))
    finally:
        runtime._command(Command(CommandKind.SHUTDOWN, 1, "close"))
        runtime.encoder_thread.join(5)
        runtime.writer.join(5)
    assert not runtime.encoder_thread.is_alive() and not runtime.writer.is_alive()


@pytest.mark.parametrize("codec", ["h264", "h265"])
@pytest.mark.parametrize("cancel", [None, CommandKind.STOP, CommandKind.SHUTDOWN])
def test_inflight_pre_boundary_frame_is_ordered_before_segment_end(tmp_path, codec, cancel):
    config = SimpleNamespace(**(vars(config_for(tmp_path, codec=codec))
                               | {"segment_duration_seconds": 1}))
    mailbox = SimpleNamespace(width=160, height=90, publish=lambda *args: None)
    runtime = VideoRuntime(None, mailbox, 1, config, capture_factory=SyntheticCapture)
    runtime.online, runtime.preview_due = True, float("inf")
    entered, release = threading.Event(), threading.Event()
    errors = []

    class DelayedPreviewFrame:
        def __init__(self):
            self.actual = frame(1)
            self.planes = self.actual.planes

        def to_ndarray(self, **kwargs):
            entered.set()
            assert release.wait(5), "preview test was not released"
            return self.actual.to_ndarray(**kwargs)

        def reformat(self, **kwargs):
            return self.actual.reformat(**kwargs)

    def capture():
        try:
            runtime.on_frame(DelayedPreviewFrame(), 100.9)
        except Exception as exc:
            errors.append(exc)

    capture_thread = threading.Thread(target=capture)
    runtime.writer.start()
    runtime.encoder_thread.start()
    try:
        runtime._command(Command(CommandKind.START, 1, "start", "inflight"))
        wait_for(lambda: runtime.accept_frames)
        runtime.on_frame(frame(0), 100)
        drain(runtime)
        time.sleep(.8)
        runtime.preview_enabled, runtime.preview_due = True, 0
        capture_thread.start()
        assert entered.wait(2)
        if cancel is None:
            time.sleep(.5)  # Cross the boundary with an old-period frame in preview conversion.
            assert not [p for p in tmp_path.glob("video/*/*.mp4") if ".recording." not in p.name]
        else:
            # User cancellation must not wait for a stuck preview conversion.
            runtime._command(Command(cancel, 1, "cancel", "inflight"))
            wait_for(lambda: not runtime.session_id)
        release.set()
        capture_thread.join(2)
        assert not capture_thread.is_alive() and not errors
        drain(runtime)
        if cancel is None:
            runtime.preview_enabled = False
            runtime.on_frame(frame(2), 101.4)
            drain(runtime)
            runtime._command(Command(CommandKind.STOP, 1, "stop", "inflight"))
            wait_for(lambda: not runtime.session_id)
        files = sorted(tmp_path.glob("video/*/*.mp4"))
        counts = [len(decoded(p)) for p in files]
        assert counts == ([2, 1] if cancel is None else [1])
    finally:
        release.set()
        if capture_thread.ident is not None:
            capture_thread.join(2)
        runtime._command(Command(CommandKind.SHUTDOWN, 1, "close"))
        runtime.encoder_thread.join(5)
        runtime.writer.join(5)
    assert not runtime.encoder_thread.is_alive() and not runtime.writer.is_alive()


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_frame_arriving_after_seal_does_not_reopen_closed_time_slot(tmp_path, codec):
    config = SimpleNamespace(**(vars(config_for(tmp_path, codec=codec))
                               | {"segment_duration_seconds": 1}))
    runtime = VideoRuntime(None, None, 1, config, capture_factory=SyntheticCapture)
    runtime.online, runtime.preview_due = True, float("inf")
    runtime.writer.start()
    runtime.encoder_thread.start()
    try:
        runtime._command(Command(CommandKind.START, 1, "start", "late-frame"))
        wait_for(lambda: runtime.accept_frames)
        runtime.on_frame(frame(0), 100)
        drain(runtime)
        wait_for(lambda: len([p for p in tmp_path.glob("video/*/*.mp4")
                              if ".recording." not in p.name]) == 1, timeout=3)
        # Captured before the boundary, but not delivered until AFTER sealing.
        # Preserve the frame in the next file without reopening the closed slot.
        for i, stamp in enumerate((100.9, 101.1, 102.1), 1):
            runtime.on_frame(frame(i), stamp)
            drain(runtime)
        runtime._command(Command(CommandKind.STOP, 1, "stop", "late-frame"))
        wait_for(lambda: not runtime.session_id)
        files = sorted(tmp_path.glob("video/*/*.mp4"))
        assert [len(decoded(p)) for p in files] == [1, 2, 1]
        assert [t for t, _ in decoded(files[1])] == pytest.approx([0, .2])
    finally:
        runtime._command(Command(CommandKind.SHUTDOWN, 1, "close"))
        runtime.encoder_thread.join(5)
        runtime.writer.join(5)
    assert not runtime.encoder_thread.is_alive() and not runtime.writer.is_alive()

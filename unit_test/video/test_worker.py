import multiprocessing
import time

from base.video.preview import PreviewMailbox
from base.video.models import Command, CommandKind, EventKind
from base.video.worker import SimulationOptions, simulated_video_worker


def receive_until(channel, kind, timeout=4):
    events = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if channel.poll(0.05):
            event = channel.recv()
            events.append(event)
            if event.kind == kind:
                return events
    raise AssertionError(f"worker did not send {kind}")


def test_worker_delivers_frames_and_confirms_preview_commands():
    context = multiprocessing.get_context("spawn")
    channel, child = context.Pipe()
    mailbox = PreviewMailbox(context)
    process = context.Process(
        target=simulated_video_worker,
        args=(child, mailbox, 7, SimulationOptions()),
    )
    process.start()
    child.close()
    try:
        receive_until(channel, EventKind.READY)
        # Preview reads are nonblocking and can coincide with the worker's publish lock.
        preview_deadline = time.monotonic() + 2
        while mailbox.latest() is None:
            assert time.monotonic() < preview_deadline, "worker did not publish a preview"
            time.sleep(.01)
        channel.send(Command(CommandKind.SET_PREVIEW, 7, "hide", preview_enabled=False, preview_revision=1))
        ack = receive_until(channel, EventKind.PREVIEW_CHANGED)[-1]
        assert ack.command_id == "hide" and not ack.preview_enabled
        channel.send(Command(CommandKind.SHUTDOWN, 7, "exit-1"))
        receive_until(channel, EventKind.CLOSED)
        process.join(3)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(3)
        process.close()
        channel.close()

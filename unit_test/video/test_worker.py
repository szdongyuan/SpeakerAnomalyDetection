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


def test_worker_acceptance_precedes_start_and_commands_have_identity():
    context = multiprocessing.get_context("spawn")
    channel, child = context.Pipe()
    mailbox = PreviewMailbox(context)
    process = context.Process(
        target=simulated_video_worker,
        args=(child, mailbox, 7, SimulationOptions(start_delay=0.15)),
    )
    process.start()
    child.close()
    try:
        receive_until(channel, EventKind.READY)
        channel.send(Command(CommandKind.START, 7, "start-1", "session-1"))
        events = receive_until(channel, EventKind.STARTED)
        accepted = next(event for event in events if event.kind == EventKind.ACCEPTED)
        started = events[-1]
        assert accepted.command_id == started.command_id == "start-1"
        assert accepted.sequence < started.sequence
        assert started.monotonic_time - accepted.monotonic_time >= 0.1
        assert started.session_id == "session-1"
        assert started.generation == 7
        assert mailbox.latest() is not None
        channel.send(Command(CommandKind.SHUTDOWN, 7, "exit-1"))
        events = receive_until(channel, EventKind.CLOSED)
        assert any(event.kind == EventKind.COMPLETED for event in events)
        process.join(3)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(3)
        process.close()
        channel.close()

from dataclasses import FrozenInstanceError
import pickle
import subprocess
import sys

import pytest

from base.video.models import Command, CommandKind, Event, EventKind


def test_messages_are_immutable_and_spawn_serializable():
    command = Command(CommandKind.SHUTDOWN, 1, "c1")
    event = Event(EventKind.READY, 1, 1, 10)
    assert pickle.loads(pickle.dumps(command)) == command
    assert pickle.loads(pickle.dumps(event)) == event
    with pytest.raises(FrozenInstanceError):
        command.command_id = "other"


def test_invalid_message_identity():
    with pytest.raises(ValueError):
        Command(CommandKind.SHUTDOWN, 0, "c1")
    with pytest.raises(ValueError):
        Event(EventKind.READY, 0, 1, 0)
    with pytest.raises(ValueError):
        Event(EventKind.READY, 1, 0, 0)
    with pytest.raises(ValueError):
        Event(EventKind.READY, 1, 1, float("nan"))


def test_worker_and_demo_import_do_not_load_qt_or_camera_drivers():
    code = (
        "import sys; import base.video.service, base.video.worker, ui.video_demo; "
        "assert not any(n.startswith(('PyQt5', 'cv2', 'av')) for n in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr

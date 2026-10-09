"""Spawn-safe entrypoints that exercise business diagnostics with real file routes."""
from pathlib import Path

import pytest

from unit_test.logging_test_support import isolated_project_logger


def csv_logging_worker(control, directory):
    from base.log_manager import LogManager
    from base.raw_audio_csv_worker import raw_audio_csv_worker

    assert LogManager._runtime is None
    with pytest.MonkeyPatch.context() as monkeypatch:
        with isolated_project_logger(Path(directory), monkeypatch):
            raw_audio_csv_worker(control)


def video_logging_worker(control, directory):
    from base.log_manager import LogManager
    from base.video.config import VideoConfig
    from base.video.runtime import usb_video_worker

    assert LogManager._runtime is None
    with pytest.MonkeyPatch.context() as monkeypatch:
        with isolated_project_logger(Path(directory), monkeypatch):
            usb_video_worker(control, None, 7, VideoConfig())

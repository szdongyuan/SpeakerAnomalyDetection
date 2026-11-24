"""Shared Qt application lifetime for tests across directories."""

import os

import pytest


@pytest.fixture(autouse=True)
def isolate_file_dialog_history(tmp_path, monkeypatch):
    from ui import file_dialog_history

    target = tmp_path / "file_dialog_history.json"
    monkeypatch.setattr(file_dialog_history, "_history_path", lambda: target)
    return target


@pytest.fixture(scope="session")
def qt_app():
    # Keep the application alive across module fixtures so shared signals survive.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app

"""Shared Qt application lifetime for tests across directories."""

import os

import pytest


@pytest.fixture(scope="session")
def qt_app():
    # Keep the application alive across module fixtures so shared signals survive.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app

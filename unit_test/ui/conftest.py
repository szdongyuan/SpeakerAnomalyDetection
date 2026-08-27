import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope="session", autouse=True)
def ui_qapp():
    return QApplication.instance() or QApplication([])

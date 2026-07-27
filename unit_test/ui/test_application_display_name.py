import os
import sys
from pathlib import Path

import pytest
from PyQt5.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from consts.running_consts import APP_DISPLAY_NAME
from ui.splash_screen_window import Splash


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app


def test_splash_screen_uses_application_display_name(qapp):
    splash = Splash()

    assert APP_DISPLAY_NAME == "希听声学检测"
    assert splash.product_name_label.text() == "欢迎使用希听声学检测系统"
    assert splash.product_name_label.font().family() == "SimSun"
    assert splash.product_name_label.font().pixelSize() == 22

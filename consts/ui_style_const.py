import re

from PyQt5.QtWidgets import QApplication


_BASE_SCREEN_HEIGHT = 1080
_MIN_FONT_SCALE = 0.70
_MAX_FONT_SCALE = 1.50
CUSTOM_WIDGET_CONTROL_VERTICAL_PADDING_PX = 8
CUSTOM_WIDGET_BUTTON_VERTICAL_PADDING_PX = 10


def _get_font_scale():
    app = QApplication.instance()
    if app is None:
        return 1.0
    screen = app.primaryScreen()
    if screen is None:
        return 1.0
    size = screen.size()
    if size.width() <= 0 or size.height() <= 0:
        return 1.0
    scale = size.height() / float(_BASE_SCREEN_HEIGHT)
    return max(_MIN_FONT_SCALE, min(scale, _MAX_FONT_SCALE))


_FONT_SCALE = _get_font_scale()
_FONT_SIZE_PATTERN = re.compile(r"(font-size\s*:\s*)(\d+(?:\.\d+)?)(\s*)(pt|px)(\s*;)", re.IGNORECASE)


def _format_size(value):
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")

def scale_size_px(size_px):
    return max(1, int(round(size_px * _FONT_SCALE)))

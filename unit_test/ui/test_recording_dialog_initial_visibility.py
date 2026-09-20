import logging

import pytest
from PyQt5.QtCore import QCoreApplication, QEvent, QObject
from PyQt5.QtWidgets import QWidget

from consts.recording_preview_consts import RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY
from ui.acquisition_config_window import RecordConfigWindow
from unit_test.base.ve3668n_fakes import device_info


@pytest.mark.parametrize("vk", [False, True], ids=["soundcard", "ve"])
@pytest.mark.parametrize("streaming", [False, True], ids=["preview-off", "preview-on"])
@pytest.mark.parametrize("invalid_mode", [False, True], ids=["default-mode", "invalid-mode"])
def test_constructor_does_not_show_top_level_widgets(
    ui_qapp, monkeypatch, vk, streaming, invalid_mode
):
    class ShowRecorder(QObject):
        def __init__(self):
            super().__init__()
            self.shown = []

        def eventFilter(self, obj, event):
            if (
                event.type() == QEvent.Show
                and isinstance(obj, QWidget)
                and obj.isWindow()
            ):
                self.shown.append((type(obj).__name__, obj.windowTitle()))
            return False

    monkeypatch.setattr(
        "ui.acquisition_config_window.LogManager.set_log_handler",
        lambda *_args, **_kwargs: logging.getLogger("record-dialog-visibility-test"),
    )
    detail = {"total_time": 4.0, "sample_rate": 48000,
              "use_streaming_recording": streaming}
    if invalid_mode:
        detail[RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY] = "broken"
    mic = device_info() if vk else {"name": "Input", "backend": "sounddevice"}
    recorder = ShowRecorder()
    window = None
    ui_qapp.installEventFilter(recorder)
    try:
        window = RecordConfigWindow(detail, mic=mic, speaker={"name": "Output"})
        assert not window.isVisible()
        assert recorder.shown == []
    finally:
        ui_qapp.removeEventFilter(recorder)
        if window is not None:
            window.close()
            window.deleteLater()
            QCoreApplication.sendPostedEvents(window, QEvent.DeferredDelete)

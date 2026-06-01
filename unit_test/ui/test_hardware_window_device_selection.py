import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication, QDialog

import ui.hardware_window as module
from ui.hardware_window import (
    HardwareSelectionController,
    HardwareSelectionModel,
    HardwareSelectionState,
    HardwareSelectionView,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _device(index, name, hostapi=0, inputs=0, outputs=0):
    return {
        "index": index,
        "name": name,
        "hostapi": hostapi,
        "default_samplerate": 48000.0,
        "max_input_channels": inputs,
        "max_output_channels": outputs,
    }


def test_ok_clicked_keeps_dialog_open_when_save_selected_devices_fails(qapp, monkeypatch):
    speaker = _device(2, "Speaker", outputs=2)
    mic = _device(1, "Mic", inputs=2)
    monkeypatch.setattr(
        module.HardwareSelectionModel,
        "refresh",
        lambda self: (
            setattr(self, "devices_by_api", {"API": {"input": [mic], "output": [speaker]}})
            or setattr(self.state, "api_name", "API")
        ),
    )
    monkeypatch.setattr(module.SoundDeviceManager, "change_default_device", staticmethod(lambda *args: None))
    monkeypatch.setattr(
        module.SoundDeviceManager,
        "save_selected_devices",
        staticmethod(lambda *args: (_ for _ in ()).throw(OSError("disk full"))),
    )
    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))

    model = HardwareSelectionModel(HardwareSelectionState(api_name="API"))
    view = HardwareSelectionView()
    controller = HardwareSelectionController(model, view)
    try:
        view.speaker_device_table.set_checked_by_predicate(lambda payload: payload == speaker)
        view.mic_device_table.set_checked_by_predicate(lambda payload: payload == mic)
        controller._restore_channels(view.mic_channel_table, [0])

        controller._on_ok_clicked()

        assert view.result() != QDialog.Accepted
        assert warnings
        assert "保存" in warnings[-1][2] or "硬件" in warnings[-1][2]
    finally:
        view.close()

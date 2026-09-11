from copy import deepcopy

import pytest
from PyQt5.QtWidgets import QLabel

from consts import model_consts
from ui.acquisition_config_window import RecordConfigWindow
from unit_test.base.ve3668n_fakes import device_info


@pytest.fixture(params=[False, True], ids=["soundcard", "vk"])
def window_factory(ui_qapp, monkeypatch, request):
    opened, warnings = [], []
    monkeypatch.setattr("ui.acquisition_config_window.QMessageBox.warning",
                        lambda *args: warnings.append(args[-1]))

    def create(detail):
        window = RecordConfigWindow(
            detail, mic=device_info() if request.param else {"name": "input"},
            speaker={"name": "output"},
        )
        opened.append(window)
        return window

    create.warnings = warnings
    yield create
    for window in opened:
        window.close()


@pytest.mark.parametrize("recording_root", ["", "  relative/absent/audio  ", None])
def test_record_config_preserves_stored_root_exactly(window_factory, recording_root):
    raw = {model_consts.RECORDING_ROOT_CONFIG_KEY: recording_root}
    original = deepcopy(raw)
    window = window_factory(raw)

    window.on_click_ok_btn()

    assert window.final_data is not None
    assert window.final_data[model_consts.RECORDING_ROOT_CONFIG_KEY] == recording_root
    assert raw == original
    assert window_factory.warnings == []


def test_record_config_does_not_insert_missing_root(window_factory):
    window = window_factory({"sample_rate": 48000})

    window.on_click_ok_btn()

    assert model_consts.RECORDING_ROOT_CONFIG_KEY not in window.final_data


def test_record_config_has_no_recording_root_editor(window_factory):
    window = window_factory({})

    assert not hasattr(window, "recording_root_input")
    assert not hasattr(window, "select_recording_root_action")
    assert not hasattr(window, "default_recording_root_btn")
    assert all("音频保存根目录" not in label.text() for label in window.findChildren(QLabel))

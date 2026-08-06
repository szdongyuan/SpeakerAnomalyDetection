import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication

from consts import model_consts
from ui import acquisition_config_window
from ui.acquisition_config_window import RecordConfigWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _create_window(recording_root=""):
    return RecordConfigWindow(
        {
            "total_time": 2.0,
            "sample_rate": 44100,
            model_consts.RECORDING_ROOT_CONFIG_KEY: recording_root,
        },
        mic={"name": "input"},
        speaker={"name": "output", "max_output_channels": 2},
    )


def test_record_config_loads_and_persists_recording_root(qapp, tmp_path):
    window = _create_window(str(tmp_path))

    assert window.recording_root_input.text() == str(tmp_path)
    assert window.recording_root_input.toolTip() == os.path.abspath(tmp_path)

    window.on_click_ok_btn()

    assert window.final_data[model_consts.RECORDING_ROOT_CONFIG_KEY] == os.path.abspath(
        tmp_path
    )


def test_record_config_defaults_to_application_storage(qapp):
    window = _create_window()

    assert window.recording_root_input.toolTip() == os.path.abspath(
        os.path.normpath(model_consts.STORED_RECORDED_PATH)
    )

    window.on_click_ok_btn()

    assert window.final_data[model_consts.RECORDING_ROOT_CONFIG_KEY] == ""


def test_record_config_uses_concise_default_path_placeholder(qapp):
    window = _create_window()
    recording_root_layout = (
        window.recording_root_input.parentWidget()
        .layout()
        .itemAtPosition(6, 1)
        .layout()
    )

    assert window.recording_root_input.placeholderText() == "audio_data/stored_data"
    assert window.default_recording_root_btn.text() == "默认路径"
    assert recording_root_layout.spacing() == 8
    assert not hasattr(window, "select_recording_root_btn")
    assert not hasattr(window, "recording_root_hint")


def test_record_config_selects_existing_directory(qapp, tmp_path, monkeypatch):
    window = _create_window()
    monkeypatch.setattr(
        acquisition_config_window.QFileDialog,
        "getExistingDirectory",
        lambda *args: str(tmp_path),
    )

    assert window.select_recording_root_action in window.recording_root_input.actions()
    assert not window.select_recording_root_action.icon().isNull()
    assert window.select_recording_root_action.toolTip() == "选择音频保存根目录"

    window.select_recording_root_action.trigger()

    assert window.recording_root_input.text() == os.path.normpath(tmp_path)


def test_record_config_default_path_button_clears_custom_root(qapp, tmp_path):
    window = _create_window(str(tmp_path))

    window.default_recording_root_btn.click()

    assert window.recording_root_input.text() == ""
    assert window.recording_root_input.toolTip() == os.path.abspath(
        os.path.normpath(model_consts.STORED_RECORDED_PATH)
    )


def test_record_config_tooltip_tracks_manually_entered_root(qapp, tmp_path):
    window = _create_window()
    custom_root = tmp_path / "a-very-long-product-audio-storage-directory"

    window.recording_root_input.setText(str(custom_root))

    assert window.recording_root_input.toolTip() == os.path.abspath(custom_root)


def test_record_config_rejects_missing_directory(qapp, tmp_path, monkeypatch):
    window = _create_window()
    window.recording_root_input.setText(str(tmp_path / "missing"))
    warnings = []
    monkeypatch.setattr(
        acquisition_config_window.QMessageBox,
        "warning",
        lambda *args: warnings.append(args[2]),
    )

    window.on_click_ok_btn()

    assert window.final_data is None
    assert warnings == ["音频保存根目录不存在，请重新选择。"]

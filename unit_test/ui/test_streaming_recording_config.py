import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ui.acquisition_config_window import RecordConfigWindow
from consts.recording_preview_consts import (
    PREVIEW_TIME_MODE_RELATIVE_LATEST,
    RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY,
)


@pytest.fixture(scope="module")
def qapp(ui_qapp):
    return ui_qapp


def test_record_config_defaults_to_blocking_and_persists_streaming_choice(qapp):
    window = RecordConfigWindow(
        {
            "total_time": 2.0,
            "sample_rate": 48000,
            "monitor_playback": False,
        },
        mic={"name": "input"},
        speaker={"name": "output", "max_output_channels": 2},
    )

    assert window.streaming_recording_checkbox.isChecked() is False
    assert not hasattr(window, "monitor_checkbox")
    assert not hasattr(window, "monitor_gain_db_input")
    assert window.preview_time_mode_combo.currentData() == PREVIEW_TIME_MODE_RELATIVE_LATEST

    window.streaming_recording_checkbox.setChecked(True)
    window.on_click_ok_btn()

    assert window.final_data["use_streaming_recording"] is True
    assert (
        window.final_data[RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY]
        == PREVIEW_TIME_MODE_RELATIVE_LATEST
    )


def test_legacy_corrupt_monitor_fields_are_ignored_on_open_and_save(qapp):
    window = RecordConfigWindow(
        {"total_time": 2.0, "sample_rate": 48000, "monitor_playback": True,
         "monitor_gain_db": "corrupt", "monitor_fade_in_ms": {},
         "use_streaming_recording": False},
        mic={"name": "input"}, speaker=None,
    )
    assert not hasattr(window, "monitor_checkbox")
    assert not hasattr(window, "monitor_gain_db_input")
    assert not window.streaming_recording_checkbox.isChecked()
    window.on_click_ok_btn()
    assert not any(key.startswith("monitor_") for key in window.final_data)
    assert window.final_data["use_streaming_recording"] is False

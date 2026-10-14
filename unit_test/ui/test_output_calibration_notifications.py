"""Application events refresh calibrated output through real UI/sequence methods."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from base.load_config import LoadUiConfig
from base.soundcard_calibration_manager import SoundcardCalibrationManager
from consts import error_code
from ui import calibration_window
from ui.sequence import sequence_widget
from unit_test.base.test_hardware_management import runtime_device
from unit_test.ui.test_main_window_startup_device_recovery import main_window_module, _window as main_window
from unit_test.ui.test_sequence_output_calibration import calibrated, qapp, window, _start_harness


@pytest.fixture
def dialog(calibrated, monkeypatch):
    monkeypatch.setattr(calibration_window.InputCalibration, "_load_selected_input_hardware", lambda self: None)
    monkeypatch.setattr(calibration_window.SoundDeviceManager, "load_selected_devices", staticmethod(lambda: {}))
    monkeypatch.setattr(calibration_window, "load_input_calibration_mode", lambda: "standard_spl")
    messages = []
    monkeypatch.setattr(calibration_window.MessageBox, "exec_", lambda self: messages.append(self.text()))
    dlg = calibration_window.CalibrationWindow()
    dlg.speaker = calibrated.speaker
    dlg.output_cal_wnd.default_logger = SimpleNamespace(info=lambda *a: None, error=lambda *a: None)
    dlg.messages = messages
    yield dlg
    dlg.close()


def persist(dlg, *, voltage=2.0, amplitude=0.2):
    dlg.output_cal_wnd.calibration_param = {"calibration_nums": 1, "amplitude_list": [amplitude]}
    dlg.output_cal_wnd.output_voltage_value = [voltage]
    dlg.clicked_calibration_button()


def attach_main(module, win):
    main = main_window(module)
    main.sequence_window = win
    main.mic, main.speaker, main.mic_channels = win.mic, win.speaker, win.mic_channels
    main.device_workflow_available = True
    win.player_status_flag = win._record_workflow_busy = False
    # Rebuild the actual runtime fields cleared by the main-window attach callback.
    def rebuild():
        info = win.sequence_config[0]["seq1"]["acq"]["detail"]["stimulus_info"]
        win.data_struct.stimulus_info = deepcopy(info)
        win.data_struct.stimulus_data = np.array([0.2, -0.4], dtype=np.float32)
        win.data_struct.sample_rate = 48000
    win.init_data_struct_stimulus_config = rebuild
    return main


def close_modal(main_window_module, monkeypatch, main, dlg, action=lambda: None):
    monkeypatch.setattr(main_window_module, "CalibrationWindow", lambda: dlg)
    monkeypatch.setattr(dlg, "exec", action)
    main.on_calibration_window_init()


def test_success_is_sticky_and_reminder_appears_in_single_success_popup(dialog, calibrated):
    assert dialog.output_calibration_changed is False
    persist(dialog)
    assert dialog.output_calibration_changed is True
    assert len(dialog.messages) == 1
    assert dialog.messages[0].count("输出校准已更新，请重新设置激励信号。") == 1
    assert calibrated.repo.get_output_amplitude_calibration(calibrated.speaker["hardware_id"])["max_voltage"] == 2.0
    dialog.output_cal_wnd.output_voltage_value = []
    dialog.clicked_calibration_button()
    dialog.clicked_reset_button()
    assert dialog.output_calibration_changed is True
    assert len(dialog.messages) == 2
    assert "输出校准已更新" not in dialog.messages[-1]


@pytest.mark.parametrize("event", ["cancel", "reset", "incomplete", "save_failure", "input"])
def test_unsuccessful_or_input_events_preserve_cached_output(dialog, calibrated, main_window_module, monkeypatch, event):
    win = window(calibrated)
    main = attach_main(main_window_module, win)
    win.reset_work_pram("first")
    input_updates = []
    win.update_v2pa_factor = lambda: input_updates.append(True)
    def action():
        if event == "reset":
            dialog.clicked_reset_button()
        elif event == "incomplete":
            dialog.clicked_calibration_button()
        elif event == "save_failure":
            monkeypatch.setattr(SoundcardCalibrationManager, "save_output_amplitude_calibration", lambda *a: (error_code.INVALID_DATA_LOADING, "save failed"))
            persist(dialog)
        elif event == "input":
            dialog.tabwidget.setCurrentIndex(1)
            monkeypatch.setattr(dialog.input_cal_wnd, "clicked_calibration", lambda: True)
            dialog.clicked_calibration_button()
            dialog.input_cal_wnd.calibration_popup(True)
    close_modal(main_window_module, monkeypatch, main, dialog, action)
    assert dialog.output_calibration_changed is False
    assert all("输出校准已更新" not in msg for msg in dialog.messages)
    assert win.reset_work_pram("next")[0]["amplitude"] == 0.25
    assert len(calibrated.reads) == 1
    assert input_updates == ([True] if event == "input" else [])


@pytest.mark.parametrize("streaming,replay", [(False, False), (True, False), (False, True), (True, True)])
def test_persisted_recalibration_modal_return_rejects_lower_max_then_recovers(dialog, calibrated, main_window_module, monkeypatch, streaming, replay):
    win = window(calibrated, amplitude=0.9)
    main = attach_main(main_window_module, win)
    starts, attempts = _start_harness(win, streaming=streaming)
    win.judge_play_and_record(is_replay=replay)
    close_modal(main_window_module, monkeypatch, main, dialog, lambda: persist(dialog, voltage=0.5))
    win.judge_play_and_record(is_replay=replay)
    assert len(starts) == len(attempts) == 1
    assert win._output_calibration_cache._result is None
    assert len(calibrated.reads) == 2
    assert "0.5 V" in calibrated.warnings[-1]
    close_modal(main_window_module, monkeypatch, main, dialog, lambda: persist(dialog, voltage=2.0))
    win.judge_play_and_record(is_replay=replay)
    win.judge_play_and_record(is_replay=replay)
    assert [start[1]["amplitude"] for start in starts] == pytest.approx([0.25, 0.1, 0.1])
    assert len(calibrated.reads) == 3
    assert len(calibrated.predictions) == 2


@pytest.mark.parametrize("event", ["same", "rename_index", "mic", "cancel", "first", "startup", "switch"])
def test_selection_reminder_and_cache_follow_stable_speaker_identity(calibrated, main_window_module, monkeypatch, event):
    win = window(calibrated)
    main = attach_main(main_window_module, win)
    win.reset_work_pram("first")
    speaker, mic = dict(win.speaker), dict(win.mic)
    notices = []
    monkeypatch.setattr(main_window_module.MessageBox, "warning", lambda *args: notices.append(args[-1]))
    monkeypatch.setattr(main_window_module.SoundDeviceManager, "get_api_info", lambda *a: {"name": "Test"})
    if event in {"startup", "switch"}:
        other = calibrated.repo.register_asset(runtime_device(name="B", inputs=0, outputs=1, index=8), "Windows WASAPI", "B", 48000)
        calibrated.repo.update_output_amplitude_calibration(other["hardware_id"], [0.3, 0.0], max_voltage=2.0)
        speaker["hardware_id"] = other["hardware_id"]
    elif event == "rename_index":
        speaker.update(index=99, display_name="Renamed")
    elif event == "mic":
        mic.update(hardware_id="other-mic")
    elif event == "first":
        main.speaker = None
        win.speaker = None
        win.invalidate_output_calibration()
    monkeypatch.setattr(main_window_module, "open_hardware_selection_window", lambda **kw: (None, None, []) if event == "cancel" else (speaker, mic, [0]))
    if event == "startup":
        main._apply_audio_devices(mic, speaker, [0])
    else:
        main.on_hardware_window_init()
    assert notices == (["扬声器已变更，请重新设置激励信号。"] if event == "switch" else [])
    if event in {"switch", "startup"}:
        assert win._output_calibration_cache._result is None
    stimulus, _, _ = win.reset_work_pram("next")
    assert stimulus["amplitude"] == (0.3 if event in {"switch", "startup"} else 0.25)
    assert len(calibrated.reads) == (2 if event in {"switch", "startup", "first"} else 1)


@pytest.mark.parametrize("event", ["unavailable", "deleted", "speaker_removed"])
def test_output_loss_clears_cache_before_references_change(calibrated, main_window_module, event):
    win = window(calibrated)
    main = attach_main(main_window_module, win)
    win.reset_work_pram("first")
    if event == "deleted":
        main.on_selected_audio_hardware_deleted(win.speaker["hardware_id"])
    elif event == "speaker_removed":
        main._apply_audio_devices(win.mic, None, [0])
    else:
        main._apply_audio_devices(win.mic, win.speaker, [0], available=False)
    assert win._output_calibration_cache._result is None


@pytest.mark.parametrize("fields", [{"display_name": "New name"}, {"samplerate": 44100}, {"bit_depth": 64}, {"latency_ms": 10}])
def test_supported_metadata_edits_preserve_calibration_and_cache(calibrated, main_window_module, monkeypatch, fields):
    win = window(calibrated)
    main = attach_main(main_window_module, win)
    win.reset_work_pram("first")
    hardware_id = win.speaker["hardware_id"]
    before = calibrated.repo.get_output_amplitude_calibration(hardware_id)
    calibrated.repo.update_asset_fields(hardware_id, fields)
    assert calibrated.repo.get_output_amplitude_calibration(hardware_id) == before
    monkeypatch.setattr(main_window_module.SoundDeviceManager, "save_selected_devices", lambda *a: None)
    main.on_registered_audio_hardware_updated(hardware_id, fields)
    assert win._output_calibration_cache._result is not None
    # Sample-rate compatibility is independently enforced at recording start.
    assert win._prepare_output_calibration(win.sequence_config[0]["seq1"]["acq"]["detail"]).amplitude == 0.25
    assert len(calibrated.reads) == 1



@pytest.mark.parametrize("streaming,replay", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("event", ["reload", "edit", "switch"])
def test_real_config_callbacks_rebuild_once_across_output_paths(calibrated, monkeypatch, streaming, replay, event):
    win = window(calibrated)
    starts, _ = _start_harness(win, streaming=streaming)
    win.count_board = None
    win._set_sequence_config_available_state = lambda value: None
    win.default_logger.warning = lambda *args: pytest.fail(str(args))
    win.refresh_channel_windows = lambda: None
    win.init_data_struct_stimulus_config = lambda: None
    win.init_fft_and_stft_flag = lambda: None
    win.update_using_file_combobox = lambda: None
    win.judge_play_and_record(is_replay=replay)
    config = deepcopy(win.sequence_config)
    if event == "edit":
        config[0]["seq1"]["acq"]["detail"]["stimulus_info"]["voltage"] = 2.0
    monkeypatch.setattr(LoadUiConfig, "load_sequence_config_from_json", lambda *args: (0, config))
    if event == "reload":
        win.get_sequence_config_from_json()
    elif event == "edit":
        sequence_widget.SequenceWindow.on_sequence_config_updated(win)
    else:
        from PyQt5.QtWidgets import QComboBox, QLineEdit
        win.using_file_combobox = QComboBox()
        win.using_file_combobox.addItem("Other", "other.json")
        win.lineedit_s_or_n = QLineEdit("SN001")
        monkeypatch.setattr(LoadUiConfig, "update_using_config_path", lambda *args: None)
        sequence_widget.SequenceWindow.on_using_file_combobox_changed(win, "Other")
    assert win._output_calibration_cache._result is None
    win.judge_play_and_record(is_replay=replay)
    win.judge_play_and_record(is_replay=replay)
    assert [entry[1]["amplitude"] for entry in starts] == [0.25] + [0.5 if event == "edit" else 0.25] * 2
    assert len(calibrated.reads) == len(calibrated.predictions) == 2


def test_same_coefficients_success_then_failure_still_invalidates_on_close(dialog, calibrated, main_window_module, monkeypatch):
    win = window(calibrated)
    main = attach_main(main_window_module, win)
    win.reset_work_pram("first")
    def action():
        persist(dialog, voltage=2.0, amplitude=0.5)
        dialog.output_cal_wnd.output_voltage_value = []
        dialog.clicked_calibration_button()
        dialog.clicked_reset_button()
    close_modal(main_window_module, monkeypatch, main, dialog, action)
    assert win._output_calibration_cache._result is None
    assert win.reset_work_pram("next")[0]["amplitude"] == pytest.approx(0.25)
    win.reset_work_pram("repeat")
    assert len(calibrated.reads) == len(calibrated.predictions) == 2


def test_user_switch_from_retained_unavailable_speaker_still_reminds(calibrated, main_window_module, monkeypatch):
    win = window(calibrated)
    main = attach_main(main_window_module, win)
    main._apply_audio_devices(win.mic, win.speaker, [0], available=False)
    other = dict(win.speaker, hardware_id="speaker-b")
    monkeypatch.setattr(main_window_module.SoundDeviceManager, "get_api_info", lambda *a: {"name": "Test"})
    monkeypatch.setattr(main_window_module, "open_hardware_selection_window", lambda **kw: (other, win.mic, [0]))
    notices = []
    monkeypatch.setattr(main_window_module.MessageBox, "warning", lambda *args: notices.append(args[-1]))
    main.on_hardware_window_init()
    assert notices == ["扬声器已变更，请重新设置激励信号。"]
    assert main.device_workflow_available is True

"""Connected Qt queue editing and condition recording; all device I/O is fake."""
from copy import deepcopy
import json
from types import MethodType

import numpy as np
import pytest
import soundfile as sf
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QDialog

from base.recording_capture import RecordingCapture
from base.recording_process_protocol import RecordingResult
from base.recording_result_reader import RecordingAudio
from base.sequence_queue_references import SequenceQueueReferenceScanner
from base.ve3668n_capture import Ve3668nInputStream
from base.wav_calibration_metadata import inspect_wav_calibration_metadata
from consts.recording_preview_consts import (
    PREVIEW_TIME_MODE_CUMULATIVE, RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY,
)
from consts.ve3668n_consts import VE_RANGE_LABELS
from ui.acquisition_config_window import RecordConfigWindow
from ui.hardware_window import HardwareSelectionState
from ui.operation_sequence import AnalysisModelSelect
from ui.sequence.sequence_widget_config_ops import SequenceWidgetConfigOpsMixin
from ui.shared_queue_save_dialog import SharedQueueSaveDialog
from unit_test.base.test_sequence_queue_references import project, write_json
from unit_test.base.ve3668n_fakes import CaptureSDK, device_info
from unit_test.ui.test_ve3668n_hardware import controls, confirm, select_device
from unit_test.ui.test_ve3668n_queue_parameters import condition_host, enter_condition
from unit_test.ui.test_ve3668n_recording import (
    host_factory, finish_ve_capture, save_calibration,
)


@pytest.mark.parametrize("reopen_collapsed", [False, True])
def test_discovery_queue_edit_shared_save_condition_capture(
        controls, condition_host, ui_qapp, tmp_path, monkeypatch, reopen_collapsed):
    host = condition_host
    controller = controls.create(HardwareSelectionState(
        api_name="MME", mic_device=controls.old_mic, mic_channels=[1]))
    view = controller.view
    view.driver_combo.setCurrentIndex(view.driver_combo.findData("vkinging"))
    assert not view.mic_channel_table.isEnabled()
    confirm(controller, ui_qapp, device_info())
    assert view.mic_device_table.checked_payload() is None
    assert view.mic_channel_table.model().rowCount() == 0
    select_device(controller, "test-machine-1")
    for row in range(view.mic_channel_table.model().rowCount()):
        view.mic_channel_table.model().item(row).setCheckState(Qt.Checked)
    controller._on_ok_clicked()
    assert view.result() == QDialog.Accepted
    host.mic = deepcopy(controller.model.state.mic_device)
    host.mic_channels = list(controller.model.state.mic_channels)
    host.speaker = controller.model.state.speaker_device
    assert host.mic_channels == [7, 1] and host.speaker is None
    host.refresh_channel_windows()
    save_calibration(host, physical=7, factor=12)
    profile_before = host.ve_profile_store.path.read_bytes()
    calibration_before = host.ve_calibration_store.path.read_bytes()

    products = tmp_path / "products"
    product_registry = products / "registry.json"
    queue_registry = tmp_path / "queues.json"
    q, r = tmp_path / "Q.json", tmp_path / "R.json"
    payload = [{"seq1": {
        "acq": {"name": "录制音频", "mode": "RECORD_ONLY", "detail": {
            "sample_rate": 48000, "ve_range_index": 1, "total_time": .1,
            "startup_trim_ms": 10, "use_streaming_recording": False,
            "audio_validation": {"enabled": False}, "monitor_playback": True,
            "monitor_gain_db": "obsolete", "monitor_fade_in_ms": "obsolete",
        }},
        "analysis_list": {"display_sequence": [], "default_ai": None, "auto_analysis": True},
    }}]
    write_json(q, payload)
    write_json(r, payload)
    write_json(queue_registry, {"Q": str(q), "R": str(r), "using_config_path": str(q)})
    write_json(product_registry, {"configs": [{"file": "one.json"}, {"file": "two.json"}]})
    write_json(products / "one.json", project("Q", "R", name="First"))
    write_json(products / "two.json", project(str(q), name="Second"))
    monkeypatch.setattr("base.load_config.SEQUENCE_CONFIG_REGISTRY_PATH", str(queue_registry))
    scanner = SequenceQueueReferenceScanner(products, product_registry, queue_registry)
    editor = AnalysisModelSelect(str(q), mic=host.mic, mic_channels=host.mic_channels,
                                 reference_scanner=scanner)
    before = q.read_bytes(), r.read_bytes(), queue_registry.read_bytes()
    dialogs = []

    def edit_recording(dialog):
        dialogs.append(dialog)
        dialog.show()
        assert dialog.recording_advanced_panel.isHidden()
        dialog.samplerate_combo.setEditText("96000")
        dialog.recording_advanced_toggle.click()
        dialog.streaming_recording_checkbox.setChecked(True)
        dialog.preview_time_mode_combo.setCurrentIndex(
            dialog.preview_time_mode_combo.findData(PREVIEW_TIME_MODE_CUMULATIVE))
        dialog.ve_range_combo.setCurrentIndex(5)
        ui_qapp.processEvents()
        dialog.ve_range_combo.showPopup()
        ui_qapp.processEvents()
        popup = dialog.ve_range_combo.view()
        assert popup.isVisible()
        assert [dialog.ve_range_combo.itemText(i) for i in range(7)] == list(VE_RANGE_LABELS)
        assert dialog.ve_range_combo.count() == 7
        dialog.ve_range_combo.hidePopup()
        dialog.on_click_ok_btn()
        assert dialog.final_data is not None
        return dialog.final_data

    monkeypatch.setattr(RecordConfigWindow, "exec", edit_recording)
    try:
        editor.select_list.show_dialog(editor.select_list.config[0].name)
        edited = editor.select_list.config[0].detail
        assert edited["sample_rate"] == 96000 and edited["ve_range_index"] == 5
        assert edited["use_streaming_recording"] is True
        assert edited[RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY] == PREVIEW_TIME_MODE_CUMULATIVE
        assert not any(key.startswith("monitor_") for key in edited)
        assert editor.dirty
        assert (q.read_bytes(), r.read_bytes(), queue_registry.read_bytes()) == before

        if reopen_collapsed:
            def accept_collapsed(dialog):
                dialogs.append(dialog)
                dialog.show()
                ui_qapp.processEvents()
                assert dialog.recording_advanced_panel.isHidden()
                assert dialog.streaming_recording_checkbox.isChecked()
                dialog.on_click_ok_btn()
                return dialog.final_data
            monkeypatch.setattr(RecordConfigWindow, "exec", accept_collapsed)
            editor.select_list.show_dialog(editor.select_list.config[0].name)

        real_exec = SharedQueueSaveDialog.exec_
        decisions = iter((False, True))
        shown = []

        def decide_shared(dialog):
            shown.append(dialog.details.toPlainText())
            assert "First／Port／A" in shown[-1] and "Second／Port／A" in shown[-1]
            assert "First／Port／B" not in shown[-1]
            button = dialog.save_button if next(decisions) else dialog.cancel_button
            QTimer.singleShot(0, button.click)
            return real_exec(dialog)

        monkeypatch.setattr(SharedQueueSaveDialog, "exec_", decide_shared)
        editor.ok_btn_clicked()
        assert editor.dirty
        assert (q.read_bytes(), r.read_bytes(), queue_registry.read_bytes()) == before
        editor.ok_btn_clicked()
        assert len(shown) == 2 and not editor.dirty
        assert q.read_bytes() != before[0] and r.read_bytes() == before[1]
        saved = json.loads(q.read_text(encoding="utf-8"))[0]["seq1"]["acq"]["detail"]
        assert saved["sample_rate"] == 96000 and saved["ve_range_index"] == 5

        # The real manual condition entry loads the saved queue and builds the
        # real frozen request. Only the process transport is injected by fixture.
        for name in ("_load_sequence_config_for_product_condition", "_resolve_sequence_queue_path",
                     "_apply_sequence_config_from_path"):
            setattr(host, name, MethodType(getattr(SequenceWidgetConfigOpsMixin, name), host))
        host.product_test_condition_configs = [
            {"key": "r", "condition_name": "R", "test_queue": str(r)},
            {"key": "q", "condition_name": "Q", "test_queue": str(q)},
        ]
        for index, rate, limit, live in ((1, 96000, .1, True), (0, 48000, 5, False),
                                         (1, 96000, .1, True)):
            host._manual_product_condition_index = index
            enter_condition(host, "manual")
            session = host._recording_process_session
            request = session.request
            assert request.sample_rate == rate and request.channels == (7, 1)
            assert request.target_samples == int(.1 * rate) + round(.01 * rate)
            assert request.trim_samples == round(.01 * rate)
            assert request.effective_streaming is live and dict(request.monitor) == {}
            sdk = CaptureSDK()
            sdk.values_by_physical_channel = {7: .0125, 1: -.025}
            capture = RecordingCapture(request, ve_stream_factory=lambda **kwargs:
                Ve3668nInputStream(**kwargs, sdk_factory=lambda: sdk))
            capture.start()
            result = capture.wait(5)
            assert isinstance(result, RecordingResult), result
            channel_calls = [call for call in sdk.trace
                             if call["operation"] == "create_iepe_voltage_channel"]
            clock_calls = [call for call in sdk.trace
                           if call["operation"] == "configure_sample_clock"]
            assert len(channel_calls) == len(clock_calls) == 1
            assert channel_calls[0]["kwargs"] == {
                "range_min": -limit, "range_max": limit}
            assert clock_calls[0]["args"][1] == rate
            assert sdk.calls("verify_actual_sample_rate") == 1
            data, actual_rate = sf.read(request.path, dtype="float32", always_2d=True)
            assert actual_rate == rate and len(data) == int(.1 * rate)
            assert sf.info(request.path).subtype == "FLOAT"
            np.testing.assert_array_equal(data, np.tile(np.float32([.0125, -.025]), (len(data), 1)))
            metadata = inspect_wav_calibration_metadata(request.path).metadata
            assert metadata["acquisition"]["sample_rate"] == rate
            assert metadata["acquisition"]["range_min"] == -limit
            assert metadata["acquisition"]["range_max"] == limit
            channel = metadata["recorded_channels"][0]
            assert channel["physical_input_channel"] == 7
            assert channel["v2pa_factor"] == 12 and channel["factor_source"] == "measured"
            assert metadata["recorded_channels"][1]["factor_source"] == "none"
            preview = capture.snapshot(generation=session.generation, sequence=1)
            if live:
                assert preview.time_mode == PREVIEW_TIME_MODE_CUMULATIVE
                assert preview.waveforms[0].time[0] == 0
                assert preview.waveforms[0].time[-1] == pytest.approx((len(data) - 1) / rate)
                np.testing.assert_allclose(preview.waveforms[0].amplitude, .0125)
            else:
                assert preview is None
            finish_ve_capture(host, session, RecordingAudio(result, data, data.mean(axis=1)))
            assert host.ve_calibration_store.get_factor(request.device, 7) == 12
            assert host.ve_profile_store.path.read_bytes() == profile_before
            assert host.ve_calibration_store.path.read_bytes() == calibration_before
    finally:
        for dialog in dialogs:
            dialog.close()
        editor._allow_close = True
        editor.close()


@pytest.mark.parametrize("entry", ["direct", "project", "program"])
def test_queue_editor_after_startup_uses_saved_profile_for_missing_rate(
        host_factory, ui_qapp, tmp_path, monkeypatch, entry):
    from pathlib import Path
    from unittest.mock import Mock
    import main_window
    from unit_test.ui.test_ve3668n_prewarm_trigger import _WindowHarness, _discovery
    from unit_test.test_product_test_project_config_dialog import (
        make_manager, prepare_project,
    )
    from ui.product_test_project_config_dialog import ProductTestProjectConfigDialog
    from ui.product_test_program_config_dialog import ProductTestProgramConfigDialog
    from unit_test.test_product_test_program_config_dialog import (
        make_manager as make_program_manager, prepare_program,
    )

    host = host_factory(48000)
    window = _WindowHarness()
    window.sequence_window = host
    window.ve_profile_store = host.ve_profile_store
    window.ve_calibration_store = host.ve_calibration_store
    window.mic = device_info(available=False, input_config=None)
    window.mic_channels = host.mic_channels
    window._on_ve_discovery_result(_discovery(device_info()))
    assert window.mic["available"]
    assert window.mic["input_config"]["sample_rate"] == 51200
    effective = SequenceWidgetConfigOpsMixin._current_ve_recording_device(host, window.mic)
    assert effective["input_config"]["sample_rate"] == 48000
    before_device = deepcopy(window.mic)
    profile_before = host.ve_profile_store.path.read_bytes()
    host.on_sequence_config_updated = Mock()
    callback = MethodType(main_window.MainWindow._open_analysis_model_select, window)
    parent = None
    if entry == "program":
        manager = make_program_manager(tmp_path)
        prepare_program(manager)
        parent = ProductTestProgramConfigDialog(manager, contextual_queue_editor_callback=callback)
        queue_path = Path(manager.load_queue_catalog()["queue_6000"]["path"])
    else:
        manager = make_manager(tmp_path)
        prepare_project(manager, tmp_path)
        queue_path = Path(next(iter(manager.load_queue_catalog().values()))["path"])
        if entry == "project":
            parent = ProductTestProjectConfigDialog(manager, contextual_queue_editor_callback=callback)
    payload = [{"seq1": {
        "acq": {"name": "录制音频", "mode": "RECORD_ONLY", "detail": {
            "total_time": .1, "ve_range_index": 5}},
        "analysis_list": {"display_sequence": [], "default_ai": None, "auto_analysis": True},
    }}]
    write_json(queue_path, payload)
    original_bytes = queue_path.read_bytes()
    product_before = {path: path.read_bytes() for path in Path(manager.program_dir).glob("*.json")}
    monkeypatch.setattr("base.load_config.SEQUENCE_CONFIG_REGISTRY_PATH", manager.queue_registry_path)
    scanner = SequenceQueueReferenceScanner(manager.program_dir, manager.registry_path, manager.queue_registry_path)
    real_editor = AnalysisModelSelect
    monkeypatch.setattr(main_window, "AnalysisModelSelect", lambda path, **kwargs:
                        real_editor(path, reference_scanner=scanner, **kwargs))
    decisions = iter((False, True))
    observed_rates = []

    def recording_exec(dialog):
        try:
            observed_rates.append(dialog.samplerate_combo.currentText())
            assert observed_rates[-1] == "48000"
            if next(decisions):
                dialog.on_click_ok_btn()
                assert dialog.final_data["sample_rate"] == 48000
            else:
                dialog.on_click_cancel_btn()
            return dialog.final_data
        finally:
            dialog.close()

    def editor_exec(editor):
        try:
            editor.select_list.show_dialog(editor.select_list.config[0].name)
            assert "sample_rate" not in editor.select_list.config[0].detail
            assert queue_path.read_bytes() == original_bytes
            editor.select_list.show_dialog(editor.select_list.config[0].name)
            editor.confirm_shared_save = lambda *args: True
            editor.ok_btn_clicked()
        finally:
            editor._allow_close = True
            editor.close()

    monkeypatch.setattr(RecordConfigWindow, "exec", recording_exec)
    monkeypatch.setattr(AnalysisModelSelect, "exec", editor_exec)
    try:
        if entry == "direct":
            callback(str(queue_path))
        elif entry == "project":
            _, button = parent._queue_controls_for_row(0)
            parent._edit_queue_for_button(button)
        else:
            parent._edit_queue_cell(parent.program_table.cellWidget(0, 3))
        assert observed_rates == ["48000", "48000"]
        saved = json.loads(queue_path.read_text(encoding="utf-8"))[0]["seq1"]["acq"]["detail"]
        assert saved["sample_rate"] == 48000 and saved["ve_range_index"] == 5
        assert window.mic == before_device
        assert host.ve_profile_store.path.read_bytes() == profile_before
        assert all(path.read_bytes() == content for path, content in product_before.items())
        host.on_sequence_config_updated.assert_called_once()
    finally:
        if parent is not None:
            parent._dirty = False
            parent.close()


@pytest.mark.parametrize("failure", ["unreadable", "malformed"])
def test_queue_profile_error_is_repairable_without_mutating_device_or_profile(
        host_factory, monkeypatch, failure):
    from unittest.mock import Mock

    host = host_factory(48000)
    mic = device_info()
    original = deepcopy(mic)
    detail = {"ve_range_index": 5}
    if failure == "unreadable":
        monkeypatch.setattr(host.ve_profile_store, "load", Mock(side_effect=OSError("profile denied")))
    else:
        host.ve_profile_store.path.write_text("{", encoding="utf-8")
    before = host.ve_profile_store.path.read_bytes()
    warnings = Mock()
    monkeypatch.setattr("ui.acquisition_config_window.QMessageBox.warning", warnings)
    dialog = RecordConfigWindow(detail, mic=mic, ve_profile_provider=lambda device:
                                host.ve_profile_store.load(device, host.ve_calibration_store))
    try:
        assert dialog._sample_rate_load_error
        assert dialog.samplerate_combo.toolTip() == dialog._sample_rate_load_error
        dialog.on_click_ok_btn()
        assert dialog.final_data is None
        assert warnings.call_args.args[-1] == dialog._sample_rate_load_error
        dialog.samplerate_combo.setEditText("96000")
        dialog.on_click_ok_btn()
        assert dialog.final_data["sample_rate"] == 96000
        assert dialog.final_data["ve_range_index"] == 5
        assert mic == original and mic["available"]
        assert detail == {"ve_range_index": 5}
        assert host.ve_profile_store.path.read_bytes() == before
    finally:
        dialog.close()


@pytest.mark.parametrize("rate", [48000, None, True, 48000.5])
def test_explicit_queue_rate_and_range_remain_repairable_without_profile_reads(
        ui_qapp, monkeypatch, rate):
    from unittest.mock import Mock

    provider = Mock(side_effect=AssertionError("Explicit rate must not read profile"))
    monkeypatch.setattr("ui.acquisition_config_window.QMessageBox.warning", Mock())
    detail = {"sample_rate": rate, "ve_range_index": -1}
    dialog = RecordConfigWindow(detail, mic=device_info(), ve_profile_provider=provider)
    try:
        assert dialog.samplerate_combo.currentText() == str(rate)
        dialog.on_click_ok_btn()
        assert dialog.final_data is None
        dialog.samplerate_combo.setEditText("96000")
        dialog.on_click_ok_btn()
        assert dialog.final_data is None and dialog.recording_advanced_toggle.isChecked()
        dialog.ve_range_combo.setCurrentIndex(5)
        dialog.on_click_ok_btn()
        assert dialog.final_data["sample_rate"] == 96000 and dialog.final_data["ve_range_index"] == 5
        provider.assert_not_called()
        assert detail == {"sample_rate": rate, "ve_range_index": -1}
    finally:
        dialog.close()


def test_missing_profile_uses_runtime_default_without_writing_a_profile(host_factory):
    host = host_factory(48000)
    host.ve_profile_store.path.unlink()
    mic = device_info()
    runtime = SequenceWidgetConfigOpsMixin._current_ve_recording_device(host, mic)
    dialog = RecordConfigWindow({}, mic=mic, ve_profile_provider=lambda device:
                                host.ve_profile_store.load(device, host.ve_calibration_store))
    try:
        assert int(dialog.samplerate_combo.currentText()) == runtime["input_config"]["sample_rate"] == 51200
        dialog.on_click_ok_btn()
        assert dialog.final_data["sample_rate"] == 51200
        assert not host.ve_profile_store.path.exists()
    finally:
        dialog.close()


def test_soundcard_editor_does_not_read_vk_profile(ui_qapp):
    from unittest.mock import Mock

    provider = Mock(side_effect=AssertionError("Soundcard must not read VK profile"))
    dialog = RecordConfigWindow({}, mic={"name": "Soundcard"}, speaker={"name": "Output"},
                                ve_profile_provider=provider)
    try:
        dialog.on_click_ok_btn()
        assert dialog.final_data["sample_rate"] == 44100
        provider.assert_not_called()
    finally:
        dialog.close()

"""Current queue editing resolves VE parameters against shared instance stores."""
from copy import deepcopy
import json
from types import MethodType

import pytest
from PyQt5.QtWidgets import QMessageBox

from base.sequence_queue_references import SequenceQueueReferenceScanner
from ui.acquisition_config_window import RecordConfigWindow
from ui.operation_sequence import AnalysisModelSelect
from ui.sequence.workbench_editor_context import resolve_editor_ve_device
from unit_test.base.test_sequence_queue_references import write_json
from unit_test.base.vkinging_fakes import device_info
from unit_test.ui.vkinging_editor_fixtures import editor_factory as host_factory


@pytest.mark.parametrize("entry", ["direct", "project"])
def test_queue_editor_after_startup_uses_saved_profile_for_missing_rate(
        host_factory, ui_qapp, tmp_path, monkeypatch, entry):
    from pathlib import Path
    from unittest.mock import Mock
    import main_window
    from unit_test.ui.test_vkinging_prewarm_trigger import _WindowHarness, _discovery
    from unit_test.test_product_test_project_config_dialog import (
        make_manager, prepare_project,
    )
    from ui.product_test_project_config_dialog import ProductTestProjectConfigDialog

    monkeypatch.setattr(QMessageBox, "information", lambda *_: QMessageBox.Ok)
    host = host_factory(48000)
    window = _WindowHarness()
    window.sequence_window = host
    from ui.sequence.workbench_editor_context import WorkbenchEditorContext
    window.editor_context = WorkbenchEditorContext(None, host.queue_data, host.default_logger)
    window.ve_profile_store = host.ve_profile_store
    window.ve_calibration_store = host.ve_calibration_store
    window.mic = device_info(available=False, input_config=None)
    window.mic_channels = host.mic_channels
    window._on_ve_discovery_result(_discovery(device_info()))
    assert window.mic["available"]
    assert window.mic["input_config"]["sample_rate"] == 51200
    effective = resolve_editor_ve_device(host.queue_data, window.mic,
                                        host.ve_profile_store, host.ve_calibration_store)
    assert effective["input_config"]["sample_rate"] == 48000
    before_device = deepcopy(window.mic)
    profile_before = host.ve_profile_store.path.read_bytes()
    window._on_configuration_deleted = Mock()
    notice = Mock()
    monkeypatch.setattr("ui.operation_sequence.QMessageBox.information", notice)
    callback = MethodType(main_window.MainWindow._open_analysis_model_select, window)
    parent = None
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    queue_name, queue_info = next(iter(manager.load_queue_catalog().items()))
    queue_path = Path(queue_info["path"])
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
        else:
            parent._edit_queue(queue_name)
        assert observed_rates == ["48000", "48000"]
        saved = json.loads(queue_path.read_text(encoding="utf-8"))[0]["seq1"]["acq"]["detail"]
        assert saved["sample_rate"] == 48000 and saved["ve_range_index"] == 5
        assert window.mic == before_device
        assert host.ve_profile_store.path.read_bytes() == profile_before
        assert all(path.read_bytes() == content for path, content in product_before.items())
        assert "测试队列已保存" in notice.call_args.args[2]
    finally:
        if parent is not None:
            parent._dirty = False
            parent.close()


@pytest.mark.parametrize("failure", ["unreadable", "malformed"])
def test_queue_profile_error_is_repairable_without_mutating_device_or_profile(
        host_factory, monkeypatch, failure):
    from unittest.mock import Mock
    from ui.vkinging_presentation import ve_failure_text

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
        assert dialog.samplerate_combo.toolTip() == ve_failure_text("configuration")
        dialog.on_click_ok_btn()
        assert dialog.final_data is None
        assert warnings.call_args.args[-1] == ve_failure_text("configuration")
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
    runtime = resolve_editor_ve_device(host.queue_data, mic,
                                     host.ve_profile_store, host.ve_calibration_store)
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

"""Exercise MainWindow's workbench bindings with synthetic committed files."""

import json
import logging
import os
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from threading import Event, Thread
from time import monotonic
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import soundfile as sf
from PyQt5.QtCore import QCoreApplication, QEvent, QObject, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QLabel, QMainWindow, QMenuBar, QMessageBox, QWidget

import main_window as main_module
from base import hardware_analysis_channel_sync, load_config, recording_calibration_snapshot, station_stage_reservations
from base.config_deletion import ConfigDeletionService
from base.motor_workbench_config import MotorWorkbenchConfigRepository
from base.product_test_project_config import ProductTestProjectConfigManager
from base.recording_management import RecordingManager
from base.recording_process_protocol import RecordingCancelled, RecordingFailure
from base.recording_service import RecordingService
from base.streaming_file_writer import StreamingWavWriter
from base.wav_calibration_metadata import read_wav_calibration_metadata
from base.workbench_recording_archive import WorkbenchRecordingArchive, WorkbenchRecordingArchiveError
from base.workbench_recording_protocol import StationWav, WorkbenchRecordingCompletion
from consts import error_code, running_consts
from main_window import MainWindow
from ui.config_delete_dialog import ConfigDeleteDialog
from ui.recording_service_bridge import RecordingServiceBridge
from ui.sequence import (
    sequence_tools_bar, sequence_widget, sequence_widget_serial_trigger_ops, workbench_toolbar_preferences,
)
from ui.sequence.sequence_widget import SequenceWindow
from ui.video_controller import VideoController
from unit_test.base.recording_process_fakes import generated_audio
from unit_test.base.test_motor_workbench_config import acquisition, config_files, write_json
from unit_test.base.vkinging_fakes import device_info
from unit_test.ui.test_workbench_recording_startup import startup


class _CommitDialog(QObject):
    projects_changed = pyqtSignal()

    def __init__(self, execute):
        super().__init__()
        self.exec = execute
        self.initial_load_succeeded = True


class _ConfigurationTeam(QObject):
    committed = pyqtSignal()


class _Hardware(QObject):
    sig_barcode = pyqtSignal(str)
    sig_trigger = pyqtSignal()
    sig_directional_trigger = pyqtSignal(str)
    sig_serial_full_frame = pyqtSignal(object)
    sig_serial_trigger_status = pyqtSignal(object)

    get_serial_discrete_input_status = staticmethod(lambda: {})


class _WorkbenchWindow(MainWindow):
    def __init__(self, video_path):
        QMainWindow.__init__(self)
        self.default_logger = logging.getLogger(__name__)
        self.mic, self.mic_channels = None, []
        self.access_lvl = None
        self._hardware_analysis_sync_pending = False
        self.recording_bridge = Mock()
        self.recording_bridge.busy = False
        self.recording_bridge.hardware_busy = False
        self.recording_bridge.service.busy = False
        self.raw_audio_csv_bridge = Mock()
        self.ve_prewarm_lifetime = None
        self.video_controller = VideoController(self, config_path=video_path, start_automatically=False)
        self.video_controller.probe_devices = Mock()
        self.video_controller.show_settings = Mock(wraps=self.video_controller.show_settings)
        self.init_sequence_widget()

    def init_menu(self):
        return QMenuBar()

    def set_title(self):
        return QWidget()

    def _hardware_selection_admission_available(self):
        return True

    update_statusbar = Mock()
    on_access_lvl_changed = Mock()


def _write_plot_program(files, *, custom_layout):
    def channel(channel_id, name, device, index):
        return {
            "channel_id": channel_id, "channel_name": name, "signal_type": "audio",
            "hardware_channel": {"device_ref": device, "physical_channel_index": index},
        }

    stations = [
        {"station_id": "s1", "station_name": "工位1", "channels": [
            channel("ch1", "CH1", "device-a", 0), channel("ch2", "CH2", "device-a", 1),
        ]},
        {"station_id": "s2", "station_name": "工位2", "channels": [
            channel("input1", "CH1", "device-b", 0),
        ]},
        {"station_id": "s3", "station_name": "工位3", "channels": []},
        {"station_id": "s4", "station_name": "工位4", "channels": []},
    ]
    if custom_layout:
        stations = stations[:2]

    def condition(number, queue):
        return {
            "condition_id": f"c{number}", "condition_name": f"工况{number}", "test_queue": queue,
            "applicable_station_ids": [station["station_id"] for station in stations],
            "trigger_state": "", "segmented_analysis": {"mode": "none"},
        }

    queues = json.loads(files["queues"].read_text(encoding="utf-8"))
    queues.update({"Plot A": queues["Shared"], "Plot B": queues["Empty"]})
    write_json(files["queues"], queues)
    write_json(files["product"], {
        "project_id": "motor", "project_name": "电机",
        "result_root_directory": str(files["product"].parent / "unused-results"),
        "stations": stations,
        "conditions": [condition(1, "Plot A"), condition(2, "Plot B")],
    })
    spl = "声压级"
    fft = "快速傅里叶变换 (FFT) 1"
    other_fft = "另一频谱"
    for file, weighting in (("shared", "A"), ("empty", "a")):
        write_json(files[file], [{"seq1": {
            "acq": acquisition(),
            "analysis_list": {
                "display_sequence": [spl, fft, other_fft],
                spl: {"type": "SPL", "weighting": weighting, "analysis_channels": [1, 0]},
                fft: {"type": "FFT", "analysis_channels": [1, 0]},
                other_fft: {"type": "FFT", "baseline_display_mode": "delta"},
            },
        }}])
    if custom_layout:
        spl_cell = {"title": "自定义声压", "analysis_item": spl, "analysis_type": "SPL"}
        fft_cell = {"title": "自定义频谱", "analysis_item": fft, "analysis_type": "FFT"}
        write_json(files["layout"], {"rows": 3, "columns": 2, "cells": [
            [spl_cell, spl_cell], [fft_cell, fft_cell],
            [fft_cell, {"title": "基线差值", "analysis_item": other_fft, "analysis_type": "FFT"}],
        ]})
    else:
        spl_cell = {"title": "声压级（SPL）1", "analysis_item": spl, "analysis_type": "SPL"}
        fft_cell = {"title": fft, "analysis_item": fft, "analysis_type": "FFT"}
        write_json(files["layout"], {"rows": 2, "columns": 4, "cells": [[spl_cell] * 4, [fft_cell] * 4]})


@pytest.fixture
def workbench(qt_app, tmp_path, monkeypatch, config_files, request):
    startup = getattr(request, "param", "with fallback")
    selection_test = request.node.originalname in (
        "test_toolbar_product_selection_switch_and_restart",
        "test_toolbar_product_selection_failures_restore_current_product",
        "test_toolbar_product_selection_busy_and_recovery",
        "test_toolbar_product_selection_preserves_mode_and_independent_queue",
    )
    if startup in ("unselected", "empty", "missing registry", "unreadable registry"):
        registry = json.loads(config_files["programs"].read_text(encoding="utf-8"))
        registry["active_file"] = None
        if startup == "empty":
            registry["configs"] = []
        write_json(config_files["programs"], registry)
        if startup == "missing registry":
            config_files["programs"].unlink()
    if startup in ("default plots", "custom plots"):
        _write_plot_program(config_files, custom_layout=startup == "custom plots")
    independent_path = tmp_path / "independent.json"
    independent = [{"seq1": {
        "acq": {"mode": "RECORD_ONLY", "detail": {"sample_rate": 48000}},
        "analysis_list": {"display_sequence": ["fft"], "fft": {"type": "FFT", "analysis_channel": 0}},
    }}]
    if startup == "invalid later acquisition":
        independent.append({"seq2": {"acq": []}})
    elif startup == "invalid unscheduled type":
        independent[0]["seq1"]["analysis_list"]["unscheduled"] = {"type": ["FFT"]}
    write_json(independent_path, independent)
    queue_registry = json.loads(config_files["queues"].read_text(encoding="utf-8"))
    independent_entry = os.path.relpath(independent_path, config_files["queues"].parent)
    if startup != "no independent queue":
        queue_registry["A Independent"] = independent_entry
    if startup in ("no independent queue", "invalid later acquisition", "invalid unscheduled type"):
        # An empty analysis order also qualifies for independent startup when acq.detail is present.
        del queue_registry["Empty"]
        product = json.loads(config_files["product"].read_text(encoding="utf-8"))
        product["conditions"] = [
            item for item in product["conditions"] if item["test_queue"] != "Empty"
        ]
        write_json(config_files["product"], product)
    other_path = tmp_path / "other.json"
    other_queue = [{"seq1": {"analysis_list": {"display_sequence": []}}}]
    if startup == "selected removed AI queue":
        other_queue[0]["seq1"] = {
            "acq": {"mode": "RECORD_ONLY", "detail": {"sample_rate": 48000}},
            "analysis_list": {"display_sequence": ["AI"], "AI": {"type": "AI"}},
        }
        queue_registry["using_config_path"] = os.path.relpath(other_path, config_files["queues"].parent)
    write_json(other_path, other_queue)
    queue_registry["Other queue"] = str(other_path)
    if startup == "selected relative queue":
        queue_registry["using_config_path"] = independent_entry
    write_json(config_files["queues"], queue_registry)
    if startup == "display-only queue":
        shared = json.loads(config_files["shared"].read_text(encoding="utf-8"))
        del shared[0]["seq1"]["acq"]
        write_json(config_files["shared"], shared)
    elif startup == "invalid product":
        config_files["product"].write_text("{", encoding="utf-8")
    elif startup == "missing product":
        write_json(config_files["programs"], {
            "active_file": "missing.json",
            "configs": [{"file": "missing.json", "project_name": "缺失产品"}],
        })
    elif startup == "missing layout":
        config_files["layout"].unlink()
    original = {path: path.read_bytes() for path in (*config_files.values(), independent_path, other_path) if path.exists()}
    if startup == "unreadable registry":
        open_path = Path.open

        def open_readable_path(path, mode="r", *args, **kwargs):
            if path == config_files["programs"] and mode == "r":
                raise PermissionError("product registry permission denied")
            return open_path(path, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", open_readable_path)

    repository = MotorWorkbenchConfigRepository(
        config_files["programs"], config_files["queues"], plot_layout_path=config_files["layout"],
    )
    load_active = Mock(wraps=repository.load_active)
    monkeypatch.setattr(repository, "load_active", load_active)
    monkeypatch.setattr(main_module, "MotorWorkbenchConfigRepository", lambda: repository, raising=False)
    monkeypatch.setattr(load_config, "SEQUENCE_CONFIG_REGISTRY_PATH", str(config_files["queues"]))
    forbidden = Mock(side_effect=AssertionError("workbench entered the old product runtime or wrote a configuration"))
    for owner, names in (
        (SequenceWindow, ("_get_product_program_manager", "_prepare_initial_product_configuration",
                          "_finish_initial_product_configuration", "_init_product_progress_runtime",
                          "_offer_product_test_resume", "_start_serial_product_listener",
                          "_sync_product_progress_after_trigger_switch",
                          "_prepare_next_manual_product_condition_recording", "start_this_play",
                          "_start_directional_workflow", "update_recorded_signal_info_to_db",
                          "_start_process_recording", "_start_next_queued_analysis")),
        (StreamingWavWriter, ("__init__",)),
        (load_config.LoadUiConfig, ("update_using_config_path", "save_sequence_config_to_json")),
        (ProductTestProjectConfigManager, ("save_project",) if selection_test else ("save_registry", "save_project")),
        (hardware_analysis_channel_sync, ("synchronize_product_analysis_channels",)),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)

    manager = ProductTestProjectConfigManager(
        program_dir=str(config_files["programs"].parent), registry_path=str(config_files["programs"]),
        queue_registry_path=str(config_files["queues"]),
    )
    save_registry = Mock(wraps=manager.save_registry)
    monkeypatch.setattr(manager, "save_registry", save_registry)
    monkeypatch.setattr(main_module, "ProductTestProjectConfigManager", lambda: manager, raising=False)

    hardware = _Hardware()
    monkeypatch.setattr(sequence_widget, "UnifiedHardwareManager", lambda: hardware)
    monkeypatch.setattr(sequence_widget.LogManager, "set_log_handler", logging.getLogger)
    legacy_recorded_info = {"sequence_mode": "test"} if startup == "legacy test mode" else {}
    monkeypatch.setattr(sequence_widget.LoadUiConfig, "load_last_recorded_info", lambda _: legacy_recorded_info)
    for name in ("restore_scanner_checkbox_state", "init_lineedit_text"):
        monkeypatch.setattr(SequenceWindow, name, lambda self: None)
    monkeypatch.setattr(SequenceWindow, "_load_analysis_window_geometry", lambda self: {})
    monkeypatch.setattr(SequenceWindow, "closeEvent", QWidget.closeEvent)
    warning = Mock()
    information = Mock()
    monkeypatch.setattr(QMessageBox, "warning", warning)
    monkeypatch.setattr(QMessageBox, "information", information)

    monkeypatch.setattr(workbench_toolbar_preferences, "DEFAULT_DIR", tmp_path)
    preference_path = tmp_path / "ui" / "ui_config" / "workbench_toolbar_preferences.ini"
    windows = []

    def open_window():
        window = _WorkbenchWindow(tmp_path / "video.json")
        windows.append(window)
        return window

    try:
        yield SimpleNamespace(window=open_window(), open_window=open_window, preference_path=preference_path,
                              legacy_recorded_info=legacy_recorded_info, files=config_files, original=original,
                              load_active=load_active, independent_path=independent_path, other_path=other_path,
                              hardware=hardware, warning=warning, information=information,
                              repository=repository, manager=manager, save_registry=save_registry, startup=startup,
                              forbidden=forbidden)
    finally:
        for window in reversed(windows):
            qt_app.removeEventFilter(window.sequence_window)
            if window.video_controller.dialog is not None:
                window.video_controller.dialog.close()
            window.video_controller.shutdown()
            window.video_controller._timer.stop()
            window.hide()
            window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def _assert_mark_buttons(grid, *, visible):
    assert grid._headers
    for header in grid._headers.values():
        assert header.ok.isHidden() is (not visible)
        assert header.ng.isHidden() is (not visible)
        assert not header.play.isHidden()


@pytest.mark.parametrize("workbench", ["missing layout"], indirect=True)
def test_startup_without_layout_loads_business_state_and_refresh_recovers_plots(workbench):
    window = workbench.window
    sequence = window.sequence_window
    controller = window.workbench_controller
    assert sequence.using_file_combobox.currentData() == workbench.files["product"].name
    assert sequence.left_panel.get_station_count() == 3
    assert sequence.left_panel.get_station_data(1)["rows"] == [["B", "--", "--"], ["A", "--", "--"]]
    assert (sequence.plot_grid._rows, sequence.plot_grid._columns) == (0, 0)
    assert sequence.plot_grid._cells == sequence.plot_grid._headers == {}
    recording = controller.get_recording_config()
    analysis = controller.get_analysis_config()
    assert recording.context.project_id == "motor"
    assert recording.parameters_for("c1")["total_time"] == 2.5
    assert tuple(analysis.test_queues) == ("Shared", "Empty")
    assert controller.get_condition_config("c3").applicable_station_ids == ()
    assert not workbench.files["layout"].exists()
    for path, data in workbench.original.items():
        assert path.read_bytes() == data

    cell = {"title": "恢复的声压", "analysis_item": "B", "analysis_type": "SPL"}
    write_json(workbench.files["layout"], {"rows": 1, "columns": 3, "cells": [[cell] * 3]})
    window.on_workbench_configuration_committed()

    assert (sequence.plot_grid._rows, sequence.plot_grid._columns) == (1, 3)
    assert sequence.plot_grid._cells[0, 0].header.left.toolTip() == "恢复的声压"
    assert controller.get_plot_bindings(0, 0)[0].condition_id == "c2"
    assert controller.get_recording_config() == recording
    assert controller.get_analysis_config() == analysis
    workbench.forbidden.assert_not_called()


@pytest.mark.parametrize("workbench", ["legacy test mode"], indirect=True)
def test_workbench_toolbar_mode_switch_and_restart(workbench, monkeypatch):
    sequence = workbench.window.sequence_window
    combo, grid = sequence.condition_mode_combobox, sequence.plot_grid
    path = workbench.preference_path
    assert not combo.isHidden()
    assert [combo.itemText(i) for i in range(combo.count())] == ["测试", "标记"]
    assert combo.currentData() == "mark"
    assert sequence.count_board.mode == "test"
    _assert_mark_buttons(grid, visible=True)
    assert not path.exists()

    grid.set_curve_data(0, 0, [{"x": [1, 2], "y": [3, 4]}])
    sequence.left_panel.set_station_results(1, "OK", "NG")
    legacy_mode_changed = Mock()
    sequence.count_board.register_mode_change_callback(legacy_mode_changed)
    persist_legacy = Mock()
    monkeypatch.setattr(sequence, "_persist_sequence_page_state", persist_legacy)
    save = Mock(wraps=workbench_toolbar_preferences.save_mode)
    monkeypatch.setattr(workbench_toolbar_preferences, "save_mode", save)

    combo.setCurrentIndex(combo.findData("test"))
    _assert_mark_buttons(grid, visible=False)
    curve, = grid._cells[0, 0].plot_widget.getPlotItem().listDataItems()
    x, y = curve.getData()
    np.testing.assert_array_equal(x, [1, 2])
    np.testing.assert_array_equal(y, [3, 4])
    station = sequence.left_panel.get_station_data(1)
    assert (station["forward"], station["reverse"]) == ("OK", "NG")
    assert sequence.count_board.mode == "test"
    legacy_mode_changed.assert_not_called()
    persist_legacy.assert_not_called()
    workbench.forbidden.assert_not_called()
    save.assert_called_once_with("test", workbench.window.default_logger)
    saved = path.read_bytes(), path.stat().st_mtime_ns

    sequence.count_board.on_mark_btn_clicked()
    legacy_mode_changed.assert_called_once()
    persist_legacy.assert_called_once_with("mark")
    assert combo.currentData() == "test"
    _assert_mark_buttons(grid, visible=False)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == saved
    assert save.call_count == 1

    workbench.legacy_recorded_info["sequence_mode"] = "mark"
    reopened = workbench.open_window().sequence_window
    assert reopened.count_board.mode == "mark"
    assert reopened.condition_mode_combobox.currentData() == "test"
    _assert_mark_buttons(reopened.plot_grid, visible=False)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == saved
    assert save.call_count == 1

    reopened_combo = reopened.condition_mode_combobox
    reopened_combo.setCurrentIndex(reopened_combo.findData("mark"))
    _assert_mark_buttons(reopened.plot_grid, visible=True)
    assert save.call_count == 2
    saved = path.read_bytes(), path.stat().st_mtime_ns
    workbench.legacy_recorded_info["sequence_mode"] = "test"
    reopened = workbench.open_window().sequence_window
    assert reopened.count_board.mode == "test"
    assert reopened.condition_mode_combobox.currentData() == "mark"
    _assert_mark_buttons(reopened.plot_grid, visible=True)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == saved
    assert save.call_count == 2


@pytest.mark.parametrize("workbench", ["custom plots"], indirect=True)
def test_workbench_toolbar_mode_survives_zero_columns_and_refresh(workbench, monkeypatch):
    window, files = workbench.window, workbench.files
    sequence = window.sequence_window
    combo, grid = sequence.condition_mode_combobox, sequence.plot_grid
    path = workbench.preference_path
    save = Mock(wraps=workbench_toolbar_preferences.save_mode)
    monkeypatch.setattr(workbench_toolbar_preferences, "save_mode", save)
    combo.setCurrentIndex(combo.findData("test"))
    assert len(grid._headers) == 2
    _assert_mark_buttons(grid, visible=False)

    registry = json.loads(files["programs"].read_text(encoding="utf-8"))
    write_json(files["programs"], {"active_file": None, "configs": []})
    window.on_workbench_configuration_committed()
    assert not grid._headers
    assert combo.currentData() == "test"
    save.assert_called_once_with("test", window.default_logger)

    assert combo.isEnabled()
    combo.setCurrentIndex(combo.findData("mark"))
    assert workbench_toolbar_preferences.load_mode(window.default_logger) == "mark"
    combo.setCurrentIndex(combo.findData("test"))
    assert workbench_toolbar_preferences.load_mode(window.default_logger) == "test"
    saved = path.read_bytes(), path.stat().st_mtime_ns
    save.reset_mock()

    layout = json.loads(files["layout"].read_text(encoding="utf-8"))
    layout["columns"] = 3
    for row in layout["cells"]:
        row.append(row[-1])
    write_json(files["layout"], layout)
    write_json(files["programs"], registry)
    window.on_workbench_configuration_committed()
    assert len(grid._headers) == 3
    assert combo.currentData() == "test"
    _assert_mark_buttons(grid, visible=False)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == saved
    save.assert_not_called()
    combo.setCurrentIndex(combo.findData("mark"))
    _assert_mark_buttons(grid, visible=True)


def test_workbench_toolbar_mode_save_failure_keeps_current_display(workbench, monkeypatch, caplog):
    sequence = workbench.window.sequence_window
    combo, grid = sequence.condition_mode_combobox, sequence.plot_grid
    path = workbench.preference_path
    failed_output = Mock()
    failed_output.open.return_value = False
    failed_output.errorString.return_value = "simulated Qt save error"

    with monkeypatch.context() as patch:
        patch.setattr(workbench_toolbar_preferences, "QSaveFile", Mock(return_value=failed_output))
        combo.setCurrentIndex(combo.findData("test"))

    assert combo.currentData() == "test"
    _assert_mark_buttons(grid, visible=False)
    assert "simulated Qt save error" in caplog.text and str(path) in caplog.text
    assert not path.exists()
    combo.setCurrentIndex(combo.findData("mark"))
    _assert_mark_buttons(grid, visible=True)
    combo.setCurrentIndex(combo.findData("test"))
    reopened = workbench.open_window().sequence_window
    assert reopened.condition_mode_combobox.currentData() == "test"
    _assert_mark_buttons(reopened.plot_grid, visible=False)


def _assert_display(window, product, names, *, selection_enabled=True):
    sequence = window.sequence_window
    assert sequence.using_file_combobox.currentText() == product
    assert sequence.using_file_combobox.isEnabled() is selection_enabled
    assert sequence.using_file_combobox.currentData() == (
        Path(window.workbench_controller._applied_source_identity).name if product else None
    )
    assert sequence.left_panel.get_station_count() == len(names)
    station_rows = sequence.left_panel.station_panel._stations
    assert tuple(station.header.accessibleName() for station in station_rows) == names
    expected_columns = names + ("",) * (4 - len(names)) if names else ()
    assert tuple(header.name.toolTip() for header in sequence.plot_grid._headers.values()) == expected_columns
    assert len(sequence.plot_grid._cells) == len(expected_columns)
    if 0 < len(names) < 4:
        empty = sequence.plot_grid._cells[0, 3]
        assert empty.header.left.toolTip() == empty.subheader.left.toolTip() == ""


def _plot_labels(cell):
    return (
        cell.header.left.toolTip(), cell.subheader.left.toolTip(),
        cell.subheader.first_half.toolTip(), cell.subheader.second_half.toolTip(),
    )


def _register_toolbar_product(workbench, *, equal=False):
    files = workbench.files
    product = json.loads(files["product"].read_text(encoding="utf-8"))
    if not equal:
        product.update(project_id="second", project_name="新产品")
        product["stations"][0]["station_name"] = "新工位"
    second = files["product"].with_name("second.json")
    write_json(second, product)
    registry = json.loads(files["programs"].read_text(encoding="utf-8"))
    registry["configs"].append({"file": second.name, "project_name": product["project_name"]})
    write_json(files["programs"], registry)
    workbench.window.sequence_window.using_file_combobox.before_show_popup()
    return second, product


def _seed_toolbar_results(sequence):
    sequence.left_panel.set_station_results(1, "OK", "NG")
    for row, column in sequence.plot_grid._cells:
        sequence.plot_grid.set_curve_data(row, column, [{"x": [1, 2], "y": [3, 4]}])


@pytest.mark.parametrize("equal", [False, True])
def test_toolbar_product_selection_switch_and_restart(workbench, equal):
    window = workbench.window
    sequence = window.sequence_window
    controller = window.workbench_controller
    second, product = _register_toolbar_product(workbench, equal=equal)
    combo = sequence.using_file_combobox
    assert [combo.itemData(i) for i in range(combo.count())] == [workbench.files["product"].name, second.name]
    if equal:
        assert combo.itemText(0) != combo.itemText(1)
        assert all(combo.itemData(i) in combo.itemText(i) for i in range(2))
    _seed_toolbar_results(sequence)
    old_version = controller._analysis_version
    expected_registry = json.loads(workbench.files["programs"].read_text(encoding="utf-8"))
    expected_registry["active_file"] = second.name
    combo.setCurrentIndex(combo.findData(second.name))
    assert json.loads(workbench.files["programs"].read_text(encoding="utf-8")) == expected_registry
    assert controller._repository.source_identity == str(second)
    assert controller.get_recording_config().context.project_id == product["project_id"]
    assert controller._analysis_version != old_version
    assert sequence.left_panel.get_station_data(1)["forward"] == "--"
    assert not any(cell.plot_widget.getPlotItem().listDataItems() for cell in sequence.plot_grid._cells.values())
    assert sequence.left_panel.station_panel._stations[0].header.accessibleName() == product["stations"][0]["station_name"]
    assert next(iter(sequence.plot_grid._headers.values())).name.toolTip() == product["stations"][0]["station_name"]
    _seed_toolbar_results(sequence)
    before = workbench.files["programs"].read_bytes()
    combo.currentIndexChanged.emit(combo.currentIndex())
    assert sequence.left_panel.get_station_data(1)["forward"] == "OK"
    assert workbench.files["programs"].read_bytes() == before
    assert workbench.save_registry.call_count == 1
    reopened = workbench.open_window()
    assert reopened.sequence_window.using_file_combobox.currentData() == second.name
    assert reopened.workbench_controller.get_recording_config().context.project_id == product["project_id"]
    if equal:
        combo.setCurrentIndex(combo.findData(workbench.files["product"].name))
        assert controller._repository.source_identity == str(workbench.files["product"])
    workbench.forbidden.assert_not_called()


@pytest.mark.parametrize("workbench", ["unselected", "empty", "missing registry", "unreadable registry"], indirect=True)
def test_toolbar_product_selection_startup_empty_and_unreadable(workbench, caplog):
    combo = workbench.window.sequence_window.using_file_combobox
    assert combo.currentIndex() == -1
    assert combo.count() == (1 if workbench.startup == "unselected" else 0)
    assert combo.placeholderText() == ("未选择产品" if combo.count() else "暂无配置")
    assert workbench.window.workbench_controller.get_recording_config() is None
    workbench.save_registry.assert_not_called()
    for path, data in workbench.original.items():
        assert path.read_bytes() == data
    if not workbench.files["programs"] in workbench.original:
        assert not workbench.files["programs"].exists()
    if workbench.startup in ("missing registry", "unreadable registry"):
        workbench.warning.assert_called_once()
        assert str(workbench.files["programs"]) in workbench.warning.call_args.args[2]
        assert any("Workbench product selection" in record.message for record in caplog.get_records("setup"))
    else:
        workbench.warning.assert_not_called()
    workbench.forbidden.assert_not_called()


@pytest.mark.parametrize("failure", ["missing", "product", "queue", "layout", "removed", "write"])
def test_toolbar_product_selection_failures_restore_current_product(workbench, monkeypatch, failure):
    second, _ = _register_toolbar_product(workbench)
    sequence = workbench.window.sequence_window
    controller = workbench.window.workbench_controller
    combo = sequence.using_file_combobox
    _seed_toolbar_results(sequence)
    config = controller.get_recording_config()
    if failure == "missing":
        second.unlink()
    elif failure in ("product", "queue", "layout"):
        path = {"product": second, "queue": workbench.files["shared"], "layout": workbench.files["layout"]}[failure]
        path.write_text("{", encoding="utf-8")
    elif failure == "removed":
        load = workbench.manager.load_registry
        monkeypatch.setattr(workbench.manager, "load_registry", lambda: dict(load(), configs=load()["configs"][:1]))
    else:
        workbench.save_registry.return_value = False
        workbench.save_registry.side_effect = None
    before = workbench.files["programs"].read_bytes()
    combo.setCurrentIndex(combo.findData(second.name))
    assert workbench.files["programs"].read_bytes() == before
    assert controller.get_recording_config() is config
    assert controller._repository.source_identity == str(workbench.files["product"])
    assert combo.currentData() == workbench.files["product"].name
    assert sequence.left_panel.get_station_data(1)["forward"] == "OK"
    assert all(len(cell.plot_widget.getPlotItem().listDataItems()) == 1 for cell in sequence.plot_grid._cells.values())
    workbench.warning.assert_called_once()
    assert workbench.warning.call_args.args[2]
    workbench.forbidden.assert_not_called()


@pytest.mark.parametrize("phase", ["cleanup", "finalizing", "analysis", "queued analysis",
                                   "resident analysis", "resident queued analysis", "recording", "closing"])
def test_toolbar_product_selection_busy_and_recovery(workbench, monkeypatch, phase):
    second, _ = _register_toolbar_product(workbench)
    window = workbench.window
    sequence = window.sequence_window
    combo = sequence.using_file_combobox
    window.hardware_action_selection = Mock()
    window.hardware_action_calibration = Mock()
    monkeypatch.setattr(window, "_calibration_admission_available", lambda: True)
    window.recording_bridge.service.busy = phase == "finalizing"
    window.recording_bridge.hardware_busy = phase == "cleanup"
    sequence._analysis_round_config_locked = phase == "analysis"
    monkeypatch.setattr(sequence, "_analysis_has_pending_tasks", lambda: phase == "queued analysis")
    sequence.player_status_flag = phase == "recording"
    window._closing = phase == "closing"
    window.analysis_service = SimpleNamespace(
        pending_task_count={"resident analysis": 1, "resident queued analysis": 2}.get(phase, 0),
    )
    window._update_hardware_busy_state()
    assert not combo.isEnabled()
    combo.setCurrentIndex(combo.findData(second.name))
    assert combo.currentData() == workbench.files["product"].name
    workbench.save_registry.assert_not_called()
    window.recording_bridge.service.busy = False
    window.recording_bridge.hardware_busy = False
    sequence._analysis_round_config_locked = False
    monkeypatch.setattr(sequence, "_analysis_has_pending_tasks", lambda: False)
    sequence.player_status_flag = False
    window._closing = False
    window.analysis_service.pending_task_count = 0
    window._update_hardware_busy_state()
    assert combo.isEnabled()
    combo.setCurrentIndex(combo.findData(second.name))
    assert combo.currentData() == second.name
    assert workbench.save_registry.call_count == 1


def test_toolbar_product_selection_popup_and_editor_refresh(workbench):
    window = workbench.window
    combo = window.sequence_window.using_file_combobox
    second, product = _register_toolbar_product(workbench)
    assert combo.currentData() == workbench.files["product"].name
    assert second.name in combo.itemData(1, Qt.ToolTipRole)
    assert product["project_name"] in combo.itemData(1, Qt.ToolTipRole)
    # A real editor commit updates both catalog and selected file without a toolbar save.
    product["project_name"] = "重命名<产品>"
    write_json(second, product)
    registry = json.loads(workbench.files["programs"].read_text(encoding="utf-8"))
    registry["configs"][1]["project_name"] = product["project_name"]
    registry["active_file"] = second.name
    write_json(workbench.files["programs"], registry)
    window.on_workbench_configuration_committed()
    assert combo.currentData() == second.name
    assert combo.currentText() == product["project_name"]
    assert "&lt;产品&gt;" in combo.toolTip()
    before = workbench.files["programs"].read_bytes()
    workbench.files["programs"].write_text("{", encoding="utf-8")
    combo.showPopup()
    combo.hidePopup()
    assert combo.currentData() == second.name and combo.count() == 2
    workbench.files["programs"].write_bytes(before)
    combo.before_show_popup()
    assert combo.currentData() == second.name
    write_json(workbench.files["programs"], {"active_file": None, "configs": registry["configs"][:1]})
    window.on_workbench_configuration_committed()
    assert combo.count() == 1 and combo.currentIndex() == -1
    assert combo.placeholderText() == "未选择产品"
    workbench.save_registry.assert_not_called()
    workbench.warning.assert_called_once()
    workbench.forbidden.assert_not_called()


@pytest.mark.parametrize("mode", ["test", "mark"])
def test_toolbar_product_selection_preserves_mode_and_independent_queue(workbench, mode):
    second, _ = _register_toolbar_product(workbench)
    sequence = workbench.window.sequence_window
    sequence.condition_mode_combobox.setCurrentIndex(sequence.condition_mode_combobox.findData(mode))
    queue_path = sequence.using_config_path
    sequence.using_file_combobox.setCurrentIndex(sequence.using_file_combobox.findData(second.name))
    assert sequence.condition_mode_combobox.currentData() == mode
    _assert_mark_buttons(sequence.plot_grid, visible=mode == "mark")
    assert sequence.using_config_path == queue_path
    reopened = workbench.open_window().sequence_window
    assert reopened.condition_mode_combobox.currentData() == mode
    assert reopened.using_file_combobox.currentData() == second.name
    _assert_mark_buttons(reopened.plot_grid, visible=mode == "mark")
    workbench.forbidden.assert_not_called()


@pytest.mark.parametrize("workbench", ["default plots"], indirect=True)
def test_default_plots_and_station_channels_from_real_startup(workbench):
    window = workbench.window
    grid = window.sequence_window.plot_grid
    controller = window.workbench_controller
    assert tuple(header.name.toolTip() for header in grid._headers.values()) == tuple(
        f"工位{i}" for i in range(1, 5)
    )
    assert set(grid._cells) == {(row, column) for row in range(2) for column in range(4)}
    for column in range(4):
        assert _plot_labels(grid._cells[0, column]) == ("声压级（SPL）1", "dB(A)", "工况1", "工况2")
        assert _plot_labels(grid._cells[1, column]) == (
            "快速傅里叶变换 (FFT) 1", "dB(Z)", "工况1", "工况2",
        )

    first = controller.get_condition_config("c1")
    second = controller.get_condition_config("c2")
    s1, s2 = first.applicable_station_ids[:2]
    assert second.applicable_station_ids[:2] == (s1, s2) == ("s1", "s2")
    channels = controller.get_station_channels(s1)
    assert controller.get_station_channels(second.applicable_station_ids[0]) is channels
    assert [
        (ch.channel_id, ch.channel_name, ch.signal_type, ch.hardware_channel.device_ref,
         ch.hardware_channel.physical_channel_index)
        for ch in channels + controller.get_station_channels(s2)
    ] == [
        ("ch1", "CH1", "audio", "device-a", 0),
        ("ch2", "CH2", "audio", "device-a", 1),
        ("input1", "CH1", "audio", "device-b", 0),
    ]
    for condition in (first, second):
        assert controller.get_analysis_config().test_queues[condition.test_queue].analysis_config["声压级"]["analysis_channels"] == (1, 0)
        exported = controller.get_analysis_config().describe()["test_queues"][condition.test_queue]
        assert exported["analysis_config"]["声压级"]["analysis_channels"] == [1, 0]
    assert {path: path.read_bytes() for path in workbench.original} == workbench.original
    workbench.forbidden.assert_not_called()


@pytest.mark.parametrize("workbench", ["custom plots"], indirect=True)
def test_custom_layout_rebinds_one_cell_through_real_workbench(workbench):
    window = workbench.window
    grid = window.sequence_window.plot_grid
    assert set(grid._cells) == {(row, column) for row in range(3) for column in range(2)}
    assert _plot_labels(grid._cells[2, 0]) == ("自定义频谱", "dB(Z)", "工况1", "工况2")
    assert _plot_labels(grid._cells[2, 1]) == ("基线差值", "dB(Z)", "工况1", "工况2")
    bindings = window.workbench_controller.get_plot_bindings(2, 1)
    assert tuple((b.station_id, b.condition_id, b.analysis_item, b.analysis_type) for b in bindings) == (
        ("s2", "c1", "另一频谱", "FFT"), ("s2", "c2", "另一频谱", "FFT"),
    )
    workbench.forbidden.assert_not_called()


@pytest.mark.parametrize("workbench", ["custom plots"], indirect=True)
def test_spl_lowercase_and_uppercase_weighting_share_a_plot_after_refresh(workbench):
    window, files = workbench.window, workbench.files
    for file, weighting in (("shared", "c"), ("empty", "C")):
        queue = json.loads(files[file].read_text(encoding="utf-8"))
        queue[0]["seq1"]["analysis_list"]["声压级"]["weighting"] = weighting
        write_json(files[file], queue)
    window.on_workbench_configuration_committed()

    spl = window.sequence_window.plot_grid._cells[0, 0]
    assert _plot_labels(spl) == ("自定义声压", "dB(C)", "工况1", "工况2")
    assert spl.header.right.toolTip() == ""
    controller = window.workbench_controller
    assert tuple(binding.condition_id for binding in controller.get_plot_bindings(0, 0)) == ("c1", "c2")
    exported = controller.get_analysis_config().describe()["test_queues"][controller.get_condition_config("c1").test_queue]
    assert exported["analysis_config"]["声压级"]["weighting"] == "c"


@pytest.mark.parametrize("workbench", ["custom plots"], indirect=True)
def test_each_snapshot_change_resets_all_results_and_invalid_refresh_preserves_them(workbench, caplog):
    window, files = workbench.window, workbench.files
    sequence = window.sequence_window
    controller = window.workbench_controller
    grid = sequence.plot_grid
    panel = sequence.left_panel
    queue = json.loads(files["shared"].read_text(encoding="utf-8"))
    detail = queue[0]["seq1"]["acq"]["detail"]
    analysis = queue[0]["seq1"]["analysis_list"]["声压级"]
    product = json.loads(files["product"].read_text(encoding="utf-8"))
    channel = product["stations"][0]["channels"][0]["hardware_channel"]
    layout = json.loads(files["layout"].read_text(encoding="utf-8"))

    def seed_results():
        for station in (1, 2):
            panel.set_station_results(station, "OK", "NG")
        for row, column in grid._cells:
            grid.set_curve_data(row, column, [{"x": [1, 2], "y": [3, 4]}])

    def assert_results(status, curve_count):
        assert all(panel.get_station_data(i)["forward"] == status for i in (1, 2))
        assert all(len(cell.plot_widget.getPlotItem().listDataItems()) == curve_count
                   for cell in grid._cells.values())

    changes = (
        ("shared", queue, detail, "total_time", 3),
        ("shared", queue, detail, "sample_rate", 48000),
        ("shared", queue, analysis, "weighting", "Z"),
        ("shared", queue, analysis, "analysis_channels", [0, 1]),
        ("product", product, channel, "physical_channel_index", 2),
        ("layout", layout, layout["cells"][0][0], "title", "新标题"),
    )
    for file, document, target, key, value in changes:
        seed_results()
        target[key] = value
        write_json(files[file], document)
        window.on_workbench_configuration_committed()
        assert_results("OK", 1) if key == "sample_rate" else assert_results("--", 0)
        if target is detail:
            if key == "sample_rate":
                assert key not in controller.get_recording_config().parameters_for("c1")
            else:
                assert controller.get_recording_config().parameters_for("c1")[key] == value
        elif target is analysis:
            assert controller.get_analysis_config().describe()["test_queues"][controller.get_condition_config("c1").test_queue]["analysis_config"]["声压级"][key] == value

    assert controller.get_station_channels("s1")[0].hardware_channel.physical_channel_index == 2
    spl = grid._cells[0, 0]
    assert _plot_labels(spl) == ("新标题", "dB(Z)", "工况1", "")
    assert spl.header.right.toolTip() == "工况1 dB(Z)\n工况2 dB(A)"
    assert spl.header.right._full_text == "单位不一致"
    prior = controller.get_condition_config("c1")
    prior_bindings = controller.get_plot_bindings(0, 0)
    seed_results()
    window.on_workbench_configuration_committed()
    assert_results("OK", 1)
    assert controller.get_condition_config("c1") is prior
    product["conditions"][0]["test_queue"] = "Missing queue"
    write_json(files["product"], product)
    window.on_workbench_configuration_committed()
    assert_results("OK", 1)
    assert controller.get_condition_config("c1") is prior
    assert controller.get_plot_bindings(0, 0) is prior_bindings
    assert "Missing queue" in caplog.text
    product["conditions"][0]["test_queue"] = prior.test_queue
    write_json(files["product"], product)
    files["layout"].write_text("{", encoding="utf-8")
    window.on_workbench_configuration_committed()
    assert_results("OK", 1)
    assert controller.get_condition_config("c1") is prior
    assert controller.get_plot_bindings(0, 0) is prior_bindings
    assert str(files["layout"]) in caplog.text

    write_json(files["programs"], {"active_file": None, "configs": []})
    window.on_workbench_configuration_committed()
    _assert_display(window, "", ())
    assert controller.get_condition_config("c1") is None
    assert controller.get_station_channels("s1") is None
    assert controller.get_plot_bindings(0, 0) == (None, None)
    workbench.forbidden.assert_not_called()


def test_startup_reads_both_views_and_uses_independent_fallback_in_memory(workbench):
    window = workbench.window
    _assert_display(window, "电机", ("工位三", "工位一", "工位二"))
    sequence = window.sequence_window
    assert sequence.using_config_path == str(workbench.independent_path)
    assert sequence.count_board.analysis_config is sequence.analysis_config
    assert sequence.left_panel.get_station_data(1)["rows"] == [["B", "--", "--"], ["A", "--", "--"]]
    assert sequence.left_panel.get_station_data(1)["forward"] == "--"
    assert sequence.left_panel.get_station_data(1)["reverse"] == "--"
    controller = window.workbench_controller
    assert controller.get_recording_config().parameters_for("c1") is controller.get_recording_config().parameters_for("c2")
    sequence.using_file_combobox.addItem("草稿", "draft.json")
    sequence.using_file_combobox.setCurrentIndex(1)
    workbench.load_active.assert_called_once_with()
    assert {path: path.read_bytes() for path in workbench.original} == workbench.original


@pytest.mark.parametrize("workbench", ["selected relative queue"], indirect=True)
def test_selected_relative_independent_queue_is_available_to_editor(workbench, monkeypatch):
    editor = Mock(return_value=_CommitDialog(lambda: None))
    monkeypatch.setattr(main_module, "AnalysisModelSelect", editor)
    workbench.window.analysis_model_select()
    assert editor.call_args.args == (str(workbench.independent_path),)
    workbench.load_active.assert_called_once_with()


@pytest.mark.parametrize("workbench", ["selected removed AI queue"], indirect=True)
def test_unsupported_selected_queue_falls_back_to_valid_independent_queue(workbench):
    window = workbench.window
    sequence = window.sequence_window
    assert sequence.using_config_path == str(workbench.independent_path)
    assert sequence.analysis_config["fft"]["type"] == "FFT"
    _assert_display(window, "电机", ("工位三", "工位一", "工位二"))
    assert {path: path.read_bytes() for path in workbench.original} == workbench.original


@pytest.mark.parametrize("workbench", ["no independent queue"], indirect=True)
def test_workbench_starts_with_recording_parameters_and_no_independent_queue(workbench):
    window = workbench.window
    assert window.sequence_window.using_config_path is None
    _assert_display(window, "电机", ("工位三", "工位一", "工位二"))


@pytest.mark.parametrize("workbench", ["display-only queue"], indirect=True)
def test_display_only_queue_is_rejected_without_disrupting_independent_analysis(workbench):
    _assert_display(workbench.window, "", ())
    sequence = workbench.window.sequence_window
    assert sequence.using_config_path == str(workbench.independent_path)
    assert sequence.analysis_config["fft"]["type"] == "FFT"


@pytest.mark.parametrize("workbench", ["invalid product", "missing product"], indirect=True)
def test_first_failed_load_silently_leaves_both_views_empty(workbench):
    _assert_display(workbench.window, "", ())
    assert workbench.window.findChild(QLabel, "workbenchConfigurationError") is None
    workbench.warning.assert_not_called()
    workbench.information.assert_not_called()


def test_refresh_reads_committed_files_and_keeps_last_display_on_error(workbench, monkeypatch):
    window, files = workbench.window, workbench.files
    product = json.loads(files["product"].read_text(encoding="utf-8"))
    product["project_name"] = "修改后"
    product["stations"].reverse()

    def save_product():
        write_json(files["product"], product)
        registry = json.loads(files["programs"].read_text(encoding="utf-8"))
        registry["configs"][0]["project_name"] = product["project_name"]
        write_json(files["programs"], registry)
        dialog.projects_changed.emit()

    dialog = _CommitDialog(save_product)
    monkeypatch.setattr(main_module, "ProductTestProjectConfigDialog", lambda *args, **kwargs: dialog)
    window.on_product_test_program_config()
    _assert_display(window, "修改后", ("工位二", "工位一", "工位三"))

    editor = _ConfigurationTeam()
    editor.committed.connect(window.on_workbench_configuration_committed)
    write_json(files["shared"], [{
        "seq1": {"acq": acquisition(), "analysis_list": {"display_sequence": ["C"], "C": {}}},
    }])
    editor.committed.emit()
    assert window.sequence_window.left_panel.get_station_data(3)["rows"] == [["C", "--", "--"]]

    window.sequence_window.left_panel.set_station_results(1, "OK", "NG")
    editor.committed.emit()
    assert window.sequence_window.left_panel.get_station_data(1)["forward"] == "OK"

    files["shared"].write_text("{", encoding="utf-8")
    editor.committed.emit()
    _assert_display(window, "修改后", ("工位二", "工位一", "工位三"))
    assert window.findChild(QLabel, "workbenchConfigurationError") is None
    assert window.sequence_window.left_panel.get_station_data(1)["forward"] == "OK"
    files["shared"].unlink()
    editor.committed.emit()
    _assert_display(window, "修改后", ("工位二", "工位一", "工位三"))
    assert window.sequence_window.left_panel.get_station_data(1)["forward"] == "OK"
    workbench.warning.assert_not_called()
    workbench.information.assert_not_called()

    registry = json.loads(files["programs"].read_text(encoding="utf-8"))
    next_product = files["product"].with_name("second.json")
    product["project_id"] = "second"
    product["project_name"] = "另一产品"
    product["conditions"] = []
    product["stations"].append({"station_id": "s4", "station_name": "工位四", "channels": []})
    write_json(next_product, product)
    registry["active_file"] = next_product.name
    registry["configs"].append({"file": next_product.name, "project_name": product["project_name"]})
    write_json(files["programs"], registry)
    window.sequence_window._record_workflow_busy = True
    editor.committed.emit()
    _assert_display(window, "另一产品", ("工位二", "工位一", "工位三", "工位四"), selection_enabled=False)
    assert window.sequence_window.left_panel.get_station_data(1)["forward"] == "--"

    registry["active_file"] = None
    write_json(files["programs"], registry)
    editor.committed.emit()
    _assert_display(window, "", (), selection_enabled=False)
    assert files["programs"].read_bytes() == json.dumps(registry, ensure_ascii=False).encode()


@pytest.mark.parametrize("kind", ["product", "other queue", "selected queue"])
def test_successful_deletion_notifies_after_matching_independent_cleanup(workbench, monkeypatch, kind):
    window, files = workbench.window, workbench.files
    sequence = window.sequence_window
    assert sequence.using_config_path == str(workbench.independent_path)

    def open_editor(*args, deletion_completed, **kwargs):
        def execute():
            if kind == "product":
                write_json(files["programs"], {"active_file": None, "configs": []})
                files["product"].unlink()
                target = SimpleNamespace(kind="product", path=str(files["product"]), action="删除")
            else:
                path = workbench.independent_path if kind == "selected queue" else workbench.other_path
                queue_key = "A Independent" if kind == "selected queue" else "Other queue"
                registry = json.loads(files["queues"].read_text(encoding="utf-8"))
                del registry[queue_key]
                write_json(files["queues"], registry)
                path.unlink()
                target = SimpleNamespace(kind="queue", path=str(path), action="删除")
            deletion_completed(target)

        return _CommitDialog(execute)

    if kind == "product":
        monkeypatch.setattr(main_module, "ProductTestProjectConfigDialog", open_editor)
        window.on_product_test_program_config()
        _assert_display(window, "", ())
    else:
        monkeypatch.setattr(main_module, "AnalysisModelSelect", open_editor)
        window.analysis_model_select()
        _assert_display(window, "电机", ("工位三", "工位一", "工位二"))

    assert workbench.load_active.call_count == 2
    if kind == "selected queue":
        assert sequence.using_config_path is None
        assert sequence.sequence_config == []
        assert sequence.analysis_config == sequence.count_board.analysis_config == {}
    else:
        assert sequence.using_config_path == str(workbench.independent_path)
        assert sequence.analysis_config["fft"]["type"] == "FFT"


@pytest.mark.parametrize("action", ["draft", "cancel", "failed save"])
def test_uncommitted_dialog_exit_leaves_workbench_untouched(workbench, monkeypatch, action):
    draft = {}

    def execute():
        if action != "cancel":
            draft["project_name"] = "未提交的产品名称"
        return action != "cancel"

    def open_editor(*args, **kwargs):
        return _CommitDialog(execute)

    monkeypatch.setattr(main_module, "ProductTestProjectConfigDialog", open_editor)
    monkeypatch.setattr(main_module, "AnalysisModelSelect", open_editor)
    workbench.window.on_product_test_program_config()
    workbench.window.analysis_model_select()
    workbench.load_active.assert_called_once_with()
    assert {path: path.read_bytes() for path in workbench.original} == workbench.original


def _confirm_hardware(window, monkeypatch):
    selection = (True, None, None, {"name": "Synthetic", "index": 12}, [2, 4])
    monkeypatch.setattr(main_module, "open_hardware_selection_window", lambda **kwargs: selection)
    window.on_hardware_window_init()


@pytest.mark.parametrize("deletion_failed", [False, True])
def test_hardware_confirmation_updates_independent_channels_without_writing(workbench, monkeypatch, deletion_failed):
    window = workbench.window
    sequence = window.sequence_window
    if deletion_failed:
        sequence._configuration_deletion_failed("配置列表未恢复")
    _confirm_hardware(window, monkeypatch)
    assert sequence.analysis_config["fft"]["analysis_channel"] == 2
    assert sequence.analysis_config["fft"]["analysis_channels"] == [2, 4]
    assert sequence.count_board.analysis_config is sequence.analysis_config
    assert sequence._configuration_deletion_busy() is deletion_failed
    assert not window._hardware_analysis_sync_pending
    assert {path: path.read_bytes() for path in workbench.original} == workbench.original


@pytest.mark.parametrize("workbench", ["invalid later acquisition", "invalid unscheduled type"], indirect=True)
def test_malformed_independent_queue_is_not_loaded_or_updated(workbench, monkeypatch):
    window = workbench.window
    _confirm_hardware(window, monkeypatch)
    assert window.sequence_window.using_config_path is None
    assert window.sequence_window.sequence_config == []
    assert not window._hardware_analysis_sync_pending
    _assert_display(window, "电机", ("工位三", "工位一", "工位二"))
    assert {path: path.read_bytes() for path in workbench.original} == workbench.original


def test_hardware_confirmation_rejects_malformed_runtime_queue_before_mutation(workbench, monkeypatch, caplog):
    window = workbench.window
    sequence = window.sequence_window
    sequence.sequence_config.append({"seq2": {"analysis_list": {"invalid": {"type": ["FFT"]}}}})
    _confirm_hardware(window, monkeypatch)
    assert sequence.analysis_config["fft"]["analysis_channel"] == 0
    assert sequence.count_board.analysis_config is sequence.analysis_config
    assert window._hardware_analysis_sync_pending
    assert "Independent analysis queue" in caplog.text
    assert {path: path.read_bytes() for path in workbench.original} == workbench.original


def test_product_manual_login_and_serial_paths_do_not_enter_runtime(workbench, monkeypatch):
    window = workbench.window
    sequence = window.sequence_window
    assert not sequence.player_btn.isEnabled()
    sequence.player_btn.clicked.emit()
    workbench.information.assert_not_called()
    workbench.warning.assert_not_called()

    login = SimpleNamespace(on_exec=lambda: ("Engineer", "test"))
    monkeypatch.setattr(main_module, "LoginWindow", lambda: login)
    start_preview = Mock()
    monkeypatch.setattr(window.video_controller, "start_preview", start_preview)
    window.on_login_window_init()
    start_preview.assert_called_once_with()
    assert window.access_lvl == "Engineer"
    assert not sequence._serial_trigger_runtime_initialized

    serial = sequence_widget_serial_trigger_ops
    monkeypatch.setattr(serial.LoadUiConfig, "load_serial_discrete_input_config", lambda: (error_code.OK, {}))
    save_serial = Mock(return_value=True)
    monkeypatch.setattr(serial.LoadUiConfig, "save_serial_discrete_input_config", save_serial)
    monkeypatch.setattr(serial, "SerialDiscreteInputConfigDialog", lambda *args, **kwargs:
                        SimpleNamespace(exec=lambda: ("save", {"enabled": True})))
    sequence.serial_trigger_btn.click()
    save_serial.assert_called_once()
    workbench.hardware.sig_trigger.emit()
    workbench.hardware.sig_directional_trigger.emit("forward")
    workbench.hardware.sig_serial_full_frame.emit({"raw_hex": "AB"})
    workbench.forbidden.assert_not_called()
    assert not sequence.player_btn.isEnabled()


def test_settings_click_reaches_only_the_workbench_controller_and_reuses_dialog(workbench, qt_app):
    window = workbench.window
    card = window.sequence_window.left_panel.video_monitor_panel.live_panel
    window.show()
    qt_app.processEvents()
    controller = window.workbench_controller
    with patch.object(controller, "_show_video_settings", wraps=controller._show_video_settings) as forward:
        QTest.mouseClick(card.settings_button, Qt.LeftButton)
        dialog = window.video_controller.dialog
        assert dialog is not None and dialog.isVisible()
        forward.assert_called_once_with()
        window.video_controller.show_settings.assert_called_once_with()
        window.video_controller.probe_devices.assert_called_once_with()
        QTest.mouseClick(card.settings_button, Qt.LeftButton)
        assert window.video_controller.dialog is dialog
        assert forward.call_count == window.video_controller.show_settings.call_count == 2
        window.video_controller.probe_devices.assert_called_once_with()


@pytest.fixture
def recording_icon_colors(qt_app, monkeypatch, tmp_path):
    icon_dir = tmp_path / "ui/ui_pic/sequence_pic"
    icon_dir.mkdir(parents=True)
    play_color, pause_color = QColor("green"), QColor("blue")
    for name, color in (("play", play_color), ("pause", pause_color)):
        pixmap = QPixmap(35, 35)
        pixmap.fill(color)
        assert pixmap.save(str(icon_dir / f"{name}.png"))
    monkeypatch.setattr(sequence_tools_bar, "DEFAULT_DIR", tmp_path.as_posix() + "/")
    return play_color, pause_color


def _icon_color(button, mode=QIcon.Normal):
    return button.icon().pixmap(button.iconSize(), mode).toImage().pixelColor(0, 0)


def _set_recording_identity(window, model="M-02", sample="007"):
    sequence = window.sequence_window
    sequence.lineedit_type.setText(model)
    sequence.toolsbar.sample_number_lineedit.setText(sample)
    return sequence


def test_real_toolbar_waits_for_login_and_resource_admission_then_delivers_group_on_gui_thread(
    startup, config_files, qt_app, monkeypatch, tmp_path, recording_icon_colors,
):
    play_color, pause_color = recording_icon_colors
    from ui.sequence import motor_workbench_controller as controller_module
    monkeypatch.setattr(controller_module, "datetime", SimpleNamespace(
        now=lambda: datetime(2026, 11, 10, 14, 3, 16, 123999)))
    archive_entered, archive_release = Event(), Event()
    save_archive = WorkbenchRecordingArchive.save

    def hold_archive(archive, *args):
        archive_entered.set()
        if not archive_release.wait(5):
            raise TimeoutError("test did not release the archive writer")
        save_archive(archive, *args)

    monkeypatch.setattr(WorkbenchRecordingArchive, "save", hold_archive)
    _write_plot_program(config_files, custom_layout=True)
    product = json.loads(config_files["product"].read_text(encoding="utf-8"))
    first, second = product["stations"]
    first["channels"][0]["hardware_channel"]["physical_channel_index"] = 7
    second["channels"][0]["hardware_channel"] = {"device_ref": "device-a", "physical_channel_index": 7}
    for condition in product["conditions"]:
        condition["applicable_station_ids"].reverse()
    write_json(config_files["product"], product)
    queue = json.loads(config_files["shared"].read_text(encoding="utf-8"))
    queue[0]["seq1"]["acq"]["detail"].update({"recording_root": str(tmp_path / "audio"), "startup_trim_ms": 0})
    write_json(config_files["shared"], queue)
    repository = MotorWorkbenchConfigRepository(
        config_files["programs"], config_files["queues"], plot_layout_path=config_files["layout"],
    )
    monkeypatch.setattr(main_module, "MotorWorkbenchConfigRepository", lambda: repository)
    monkeypatch.setattr(load_config, "SEQUENCE_CONFIG_REGISTRY_PATH", str(config_files["queues"]))
    discovered = device_info()
    write_json(startup.path, {"input_selection": {
        "schema_version": 1, "backend": "vkinging", "machine_id": discovered["machine_id"],
        "physical_channels": [1],
    }})
    monkeypatch.setattr(QMessageBox, "information", Mock())
    monkeypatch.setattr(QMessageBox, "warning", Mock())
    bridge = startup.bridge
    submissions = []

    def service_start(request, callbacks):
        session = SimpleNamespace(request=request)
        submissions.append((session, callbacks))
        return session

    bridge.service.start = service_start
    qt_delivery = RecordingServiceBridge(bridge.service)
    bridge.start = qt_delivery.start

    def login(window):
        bridge.change_worker(1)
        window.ve_discovery.service.answer(discovered)
        qt_app.processEvents()
        assert len(bridge.prewarms) == 1
        bridge.finish_prewarm()
        assert window.workbench_recording_device.snapshot is not None
        button = window.sequence_window.player_btn
        assert not button.isEnabled()
        assert _icon_color(button) == play_color
        button.clicked.emit()
        assert submissions == []
        return "Engineer", "test"

    window = startup.launch(
        login, ve_profile_store=SimpleNamespace(load=lambda *_: discovered["input_config"]),
        ve_calibration_store=SimpleNamespace(observe=lambda _: {}),
    )
    sequence = _set_recording_identity(window)
    sequence.barcode_scanner_box.setChecked(False)
    sequence.lineedit_s_or_n.setText("IGNORED")
    button = window.sequence_window.player_btn
    _wait_for_qt(button.isEnabled)
    assert window.sequence_window.using_file_combobox.isEnabled()
    snapshot = window.workbench_recording_device.snapshot

    # Both resource transitions leave the selected device capability intact.
    for bridge_busy, service_available in ((True, True), (False, False)):
        bridge.hardware_busy = bridge_busy
        bridge.service.can_start_recording = service_available
        window._update_hardware_busy_state()
        assert window.workbench_recording_device.snapshot is snapshot
        assert not button.isEnabled()
        button.clicked.emit()
        assert submissions == []
    bridge.hardware_busy = False
    bridge.service.can_start_recording = True
    window._update_hardware_busy_state()
    assert button.isEnabled()

    completions = []
    analysis_inputs = []
    window.workbench_analysis_input_ready.connect(analysis_inputs.append)
    window.workbench_recording_completed.connect(
        lambda completed: completions.append((QThread.currentThread(), completed)))
    QTest.mouseClick(button, Qt.LeftButton)
    button.clicked.emit()
    assert len(submissions) == 1
    session, callbacks = submissions[0]
    request = session.request
    assert request.channels == (7, 1)
    assert request.sample_rate == 51200
    assert request.target_samples == 128000
    assert not button.isEnabled()
    assert _icon_color(button, QIcon.Disabled) == pause_color
    group = request.station_group
    assert (group.project_id, group.condition_id) == ("motor", "c1")
    payload = WorkbenchRecordingCompletion(request.request_id, group.project_id, group.condition_id,
                                           request.sample_rate, request.target_samples, tuple(
        StationWav(station.station_id, str(Path(group.final_path) / station.filename), station.channels)
        for station in group.stations
    ))

    def finish():
        callbacks.workbench_completed(session, payload)
        callbacks.workbench_completed(session, payload)
        callbacks.released(session)

    worker = Thread(target=finish)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert completions == []
    try:
        _wait_for_qt(archive_entered.is_set)
        assert not button.isEnabled()
        assert _icon_color(button, QIcon.Disabled) == pause_color
        button.clicked.emit()
        assert len(submissions) == 1
        assert completions == analysis_inputs == []
    finally:
        archive_release.set()
        _wait_for_qt(lambda: completions)
    assert completions == [(qt_app.thread(), payload)]
    assert len(analysis_inputs) == 1
    assert analysis_inputs[0].completion is payload
    from base.workbench_analysis_input import build_analysis_payload
    assert len(build_analysis_payload(analysis_inputs)["tasks"]) == len(payload.stations)
    first_wav, second_wav = completions[0][1].stations
    assert (first_wav.station_id, second_wav.station_id) == ("s1", "s2")
    wav_dir = Path(product["result_root_directory"]) / "电机" / "audio"
    assert Path(group.final_path) == Path(request.path).parent == wav_dir
    assert Path(first_wav.path) == wav_dir / "2026-11-10_14-03-16-123_工位1_工况1_M-02_007.wav"
    assert Path(second_wav.path) == wav_dir / "2026-11-10_14-03-16-123_工位2_工况1_M-02_007.wav"
    assert Path(first_wav.path).is_absolute()
    assert [(c.channel_id, c.physical_channel_index, c.wav_channel_index) for c in first_wav.channels] == [
        ("ch1", 7, 0), ("ch2", 1, 1),
    ]
    assert [(c.channel_id, c.physical_channel_index, c.wav_channel_index) for c in second_wav.channels] == [
        ("input1", 7, 0),
    ]
    assert button.isEnabled()
    assert _icon_color(button) == play_color
    QTest.mouseClick(button, Qt.LeftButton)
    second_session, second_callbacks = submissions[1]
    second_request = second_session.request
    assert second_request.station_group.condition_id == "c2"
    second_callbacks.failed(second_session, RecordingFailure(
        second_request.request_id, "publish", second_request.path, "publication failed",
    ))
    second_callbacks.release_failed(second_session, "stage lease retained")
    second_callbacks.release_failed(second_session, "duplicate diagnostic")
    qt_app.processEvents()
    assert "stage lease retained" in window.statusBar().currentMessage()
    assert second_request.request_id not in qt_delivery._callbacks
    assert second_request.request_id not in qt_delivery._finished
    assert not any(key == second_request.request_id for key, _ in qt_delivery._delivered)
    assert completions == [(qt_app.thread(), payload)]
    assert len(analysis_inputs) == 1
    assert analysis_inputs[0].completion is payload
    from base.workbench_analysis_input import build_analysis_payload
    assert len(build_analysis_payload(analysis_inputs)["tasks"]) == len(payload.stations)
    assert button.isEnabled()
    QTest.mouseClick(button, Qt.LeftButton)
    assert submissions[2][0].request.station_group.condition_id == "c2"


def test_workbench_completion_signal_rejects_untyped_payload(workbench):
    with pytest.raises(TypeError, match="unexpected type"):
        workbench.window.workbench_recording_completed.emit({})


def _recording_program(files, audio_root, monkeypatch):
    _write_plot_program(files, custom_layout=True)
    product = json.loads(files["product"].read_text(encoding="utf-8"))
    product["result_root_directory"] = str(audio_root)
    first, second = product["stations"]
    first["channels"][0]["hardware_channel"]["physical_channel_index"] = 2
    first["channels"][1]["hardware_channel"]["physical_channel_index"] = 0
    second["channels"][0]["hardware_channel"] = {"device_ref": "device-a", "physical_channel_index": 0}
    product["conditions"][0]["applicable_station_ids"].reverse()
    product["conditions"][1]["applicable_station_ids"] = ["s2"]
    write_json(files["product"], product)
    for key, frames in (("shared", 8), ("empty", 10)):
        queue = json.loads(files[key].read_text(encoding="utf-8"))
        acq = queue[0]["seq1"]["acq"]
        acq.update(name="录制音频", mode="RECORD_ONLY")
        acq["detail"].update(
            recording_root=str(audio_root.parent / "queue-output"), startup_trim_ms=0.05,
            total_time=frames / 48000, audio_validation={"enabled": False},
        )
        write_json(files[key], queue)
    repository = MotorWorkbenchConfigRepository(
        files["programs"], files["queues"], plot_layout_path=files["layout"],
    )
    monkeypatch.setattr(main_module, "MotorWorkbenchConfigRepository", lambda: repository)
    manager = ProductTestProjectConfigManager(
        program_dir=str(files["programs"].parent), registry_path=str(files["programs"]),
        queue_registry_path=str(files["queues"]),
    )
    monkeypatch.setattr(main_module, "ProductTestProjectConfigManager", lambda: manager, raising=False)
    monkeypatch.setattr(load_config, "SEQUENCE_CONFIG_REGISTRY_PATH", str(files["queues"]))
    monkeypatch.setattr(QMessageBox, "information", Mock())
    monkeypatch.setattr(QMessageBox, "warning", Mock())
    return product


@pytest.mark.parametrize("output_location", ["inside", "outside"])
def test_toolbar_product_selection_records_first_condition_for_new_product(
    startup, config_files, tmp_path, monkeypatch, output_location,
):
    application_root = tmp_path if output_location == "inside" else tmp_path / "application"
    monkeypatch.setattr(running_consts, "DEFAULT_DIR", str(application_root))
    old_root, new_root = tmp_path / "old-output", tmp_path / "new-output"
    product = _recording_program(config_files, old_root, monkeypatch)
    second_file = config_files["product"].with_name("second.json")
    write_json(second_file, dict(product, project_id="second", project_name="第二产品",
                                 result_root_directory=str(new_root), conditions=product["conditions"][::-1]))
    registry = json.loads(config_files["programs"].read_text(encoding="utf-8"))
    registry["configs"].append({"file": second_file.name, "project_name": "第二产品"})
    write_json(config_files["programs"], registry)
    service = RecordingService(
        backend_factory="unit_test.base.recording_process_fakes:workbench_soundcard_dependencies",
        backend_options={"trace_dir": str(tmp_path), "frames": 12, "chunk_frames": 4},
    )
    bridge = RecordingServiceBridge(service)
    requests = []
    start = bridge.start

    def capture(request, callbacks):
        requests.append(request)
        return start(request, callbacks)

    monkeypatch.setattr(bridge, "start", capture)

    def login(window):
        _wait_for_qt(lambda: window.workbench_recording_device.snapshot is not None)
        return "Engineer", "test"

    startup.login(login)
    window = None
    try:
        window = main_module.MainWindow(recording_bridge=bridge)
        sequence = _set_recording_identity(window)
        completions = []
        window.workbench_recording_completed.connect(completions.append)
        _wait_for_qt(sequence.player_btn.isEnabled)
        QTest.mouseClick(sequence.player_btn, Qt.LeftButton)
        _wait_for_qt(lambda: len(completions) == 1 and sequence.using_file_combobox.isEnabled())
        assert completions[0].condition_id == "c1"
        assert window.workbench_controller._condition_index == 1
        combo = sequence.using_file_combobox
        combo.setCurrentIndex(combo.findData(second_file.name))
        assert combo.currentData() == second_file.name
        assert window.workbench_controller._condition_index == 0
        _wait_for_qt(sequence.player_btn.isEnabled)
        QTest.mouseClick(sequence.player_btn, Qt.LeftButton)
        _wait_for_qt(lambda: len(completions) == 2 and sequence.player_btn.isEnabled())
        result = completions[1]
        assert (result.project_id, result.condition_id) == ("second", "c2")
        assert requests[1].station_group.project_id == "second" and requests[1].station_group.condition_id == "c2"
        paths = [Path(station.path) for station in result.stations]
        assert paths and all(path.parent == new_root / "第二产品" / "audio" for path in paths)
        for path in paths:
            data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            assert sample_rate == 48000
            np.testing.assert_array_equal(data, generated_audio(2, 10)[:, [0]])
        rows = _archive_rows(startup.database_path)
        normalize_path = RecordingManager.normalize_audio_path_for_db
        assert all(rows[normalize_path(path)][0] == "第二产品" for path in paths)
        assert all(rows[normalize_path(station.path)][0] == "电机" for station in completions[0].stations)
        assert set(rows) == {
            normalize_path(station.path) for completion in completions for station in completion.stations
        }
        assert all(Path(path).is_absolute() is (output_location == "outside") for path in rows)
    finally:
        if window is None:
            bridge.shutdown()
            assert service.closed.wait(10)
        else:
            window.close()
            _wait_for_qt(lambda: getattr(window, "_exit_cleanup_complete", False))


@pytest.mark.parametrize("terminal", ["success", "failure", "cancelled", "archive failure"])
def test_toolbar_product_selection_recording_terminals_and_cleanup_recover(
    startup, config_files, tmp_path, monkeypatch, terminal,
):
    product = _recording_program(config_files, tmp_path / "audio", monkeypatch)
    second = config_files["product"].with_name("second.json")
    write_json(second, dict(product, project_id="second", project_name="第二产品"))
    registry = json.loads(config_files["programs"].read_text(encoding="utf-8"))
    registry["configs"].append({"file": second.name, "project_name": "第二产品"})
    write_json(config_files["programs"], registry)
    bridge = startup.bridge
    bridge.service.busy = False
    submissions = []

    def start(request, callbacks):
        session = SimpleNamespace(request=request)
        submissions.append((session, callbacks))
        return session

    bridge.start = start

    def login(_):
        bridge.change_worker(1)
        bridge.answer_probe()
        return "Engineer", "test"

    manager = main_module.ProductTestProjectConfigManager()
    save = Mock(wraps=manager.save_registry)
    monkeypatch.setattr(manager, "save_registry", save)
    window = startup.launch(login)
    sequence = _set_recording_identity(window)
    combo = sequence.using_file_combobox
    controller = window.workbench_controller
    archives = []
    controller.bind_recording(replace(controller._recording, submit_archive=lambda *args: archives.append(args)))
    _wait_for_qt(sequence.player_btn.isEnabled)
    QTest.mouseClick(sequence.player_btn, Qt.LeftButton)
    assert not combo.isEnabled()
    combo.setCurrentIndex(combo.findData(second.name))
    assert combo.currentData() == config_files["product"].name
    save.assert_not_called()
    session, callbacks = submissions[-1]
    request = session.request
    bridge.hardware_busy = bridge.service.busy = True
    if terminal == "failure":
        callbacks.failed(session, RecordingFailure(request.request_id, "capture", request.path, "capture failed"))
    elif terminal == "cancelled":
        callbacks.cancelled(session, RecordingCancelled(request.request_id, request.path, 0, 0))
    else:
        group = request.station_group
        result = WorkbenchRecordingCompletion(
            request.request_id, group.project_id, group.condition_id, request.sample_rate,
            request.target_samples - request.trim_samples, tuple(
                StationWav(station.station_id, str(Path(group.final_path) / station.filename), station.channels)
                for station in group.stations
            ),
        )
        callbacks.workbench_completed(session, result)
        assert controller._in_flight.archiving and not combo.isEnabled()
        combo.currentIndexChanged.emit(combo.findData(second.name))
        save.assert_not_called()
        archived = archives[0][-1]
        archived(result.request_id, RuntimeError("archive failed") if terminal == "archive failure" else None)
    assert controller._in_flight is None
    assert not combo.isEnabled()
    window._update_hardware_busy_state()
    assert not combo.isEnabled()
    combo.setCurrentIndex(combo.findData(second.name))
    assert combo.currentData() == config_files["product"].name
    save.assert_not_called()
    bridge.hardware_busy = bridge.service.busy = False
    callbacks.released(session)
    assert combo.isEnabled()
    assert window._hardware_busy_timer.isActive()
    sequence._analysis_round_config_locked = True
    _wait_for_qt(lambda: not combo.isEnabled())
    combo.currentIndexChanged.emit(combo.findData(second.name))
    save.assert_not_called()
    sequence._analysis_round_config_locked = False
    _wait_for_qt(lambda: combo.isEnabled())
    pending = [True]
    monkeypatch.setattr(sequence, "_analysis_has_pending_tasks", lambda: pending[0])
    _wait_for_qt(lambda: not combo.isEnabled())
    combo.currentIndexChanged.emit(combo.findData(second.name))
    save.assert_not_called()
    pending[0] = False
    _wait_for_qt(lambda: combo.isEnabled())
    combo.setCurrentIndex(combo.findData(second.name))
    assert combo.currentData() == second.name and save.call_count == 1


def test_active_switch_refreshes_recording_with_in_flight_snapshot(
    startup, config_files, tmp_path, monkeypatch,
):
    files = config_files
    product = _recording_program(files, tmp_path / "audio", monkeypatch)
    bridge = startup.bridge
    submissions = []

    def start(request, callbacks):
        session = SimpleNamespace(request=request)
        submissions.append((session, callbacks))
        return session

    def finish():
        expected_count = len(completed) + 1
        session, callbacks = submissions[-1]
        request = session.request
        group = request.station_group
        result = WorkbenchRecordingCompletion(
            request.request_id, group.project_id, group.condition_id,
            request.sample_rate, request.target_samples - request.trim_samples, tuple(
                StationWav(station.station_id, str(Path(group.final_path) / station.filename), station.channels)
                for station in group.stations
            ),
        )
        callbacks.workbench_completed(session, result)
        _wait_for_qt(lambda: len(completed) == expected_count)

    def login(_):
        bridge.change_worker(1)
        bridge.answer_probe()
        return "Engineer", "test"

    bridge.start = start
    window = startup.launch(login)
    _set_recording_identity(window)
    button = window.sequence_window.player_btn
    _wait_for_qt(button.isEnabled)
    editor = _ConfigurationTeam()
    editor.committed.connect(window.on_workbench_configuration_committed)
    completed = []
    window.workbench_recording_completed.connect(completed.append)
    QTest.mouseClick(button, Qt.LeftButton)
    assert submissions[-1][0].request.station_group.condition_id == "c1"
    finish()

    before = files["shared"].read_bytes()
    window.sequence_window.left_panel.set_station_results(1, "OK", "NG")
    files["shared"].write_text("{", encoding="utf-8")
    editor.committed.emit()
    assert window.sequence_window.left_panel.get_station_data(1)["forward"] == "OK"
    assert not button.isEnabled() and "配置不可用" in button.toolTip()
    button.clicked.emit()
    assert len(submissions) == 1
    files["shared"].write_bytes(before)
    editor.committed.emit()
    assert button.isEnabled()
    assert window.sequence_window.left_panel.get_station_data(1)["forward"] == "OK"
    QTest.mouseClick(button, Qt.LeftButton)
    assert submissions[-1][0].request.station_group.condition_id == "c2"

    assert not button.isEnabled()
    manager = ProductTestProjectConfigManager(
        program_dir=str(files["programs"].parent), registry_path=str(files["programs"]),
        queue_registry_path=str(files["queues"]),
    )
    second_file = files["product"].with_name("second.json")
    product.update(project_id="second", project_name="第二产品", conditions=product["conditions"][::-1])
    write_json(second_file, product)
    registry = manager.load_registry()
    registry["configs"].append({"file": second_file.name, "project_name": "第二产品"})
    registry["active_file"] = second_file.name
    assert manager.save_registry(registry)
    editor.committed.emit()
    assert window.sequence_window.using_file_combobox.currentText() == "第二产品"
    finish()
    assert [result.condition_id for result in completed] == ["c1", "c2"]
    QTest.mouseClick(button, Qt.LeftButton)
    switched = submissions[-1][0].request.station_group
    assert (switched.project_id, switched.condition_id) == ("second", "c2")
    finish()

    deletion = ConfigDeleteDialog(
        ConfigDeletionService(manager), "product", window, completed=window._on_configuration_deleted,
    )
    for index in range(deletion.config_list.count()):
        item = deletion.config_list.item(index)
        if item.data(Qt.UserRole).key == second_file.name:
            item.setCheckState(Qt.Checked)
    deletion.delete_button.click()
    assert [target.key for target in deletion.completed_targets] == [second_file.name]
    assert not second_file.exists() and manager.load_registry()["active_file"] is None
    _assert_display(window, "", ())
    assert not button.isEnabled()
    deletion.deleteLater()


def _wait_for_qt(predicate, timeout=12):
    deadline = monotonic() + timeout
    while not predicate():
        assert monotonic() < deadline, "Timed out waiting for the workbench recording"
        QTest.qWait(10)


def _archive_row_count(database_path):
    with closing(sqlite3.connect(database_path)) as database:
        return database.execute("SELECT COUNT(*) FROM audio_data_table").fetchone()[0]


def _archive_rows(database_path):
    manager = RecordingManager()
    manager.db_path = database_path
    code, rows = manager.get_record_audio_data()
    assert code == error_code.OK, rows
    return {path: tuple(metadata) for _, path, *metadata in rows}


@pytest.mark.parametrize("delivery", ["queued", "writing", "writing-fails"])
def test_close_drains_workbench_archive(startup, config_files, tmp_path, monkeypatch, caplog, delivery):
    _recording_program(config_files, tmp_path / "audio", monkeypatch)
    bridge = startup.bridge
    qt_delivery = RecordingServiceBridge(bridge.service)
    submissions = []

    def service_start(request, callbacks):
        session = SimpleNamespace(request=request)
        submissions.append((session, callbacks))
        return session

    bridge.service.start = service_start
    bridge.service.shutdown = bridge.shutdown
    bridge.start = qt_delivery.start
    bridge.shutdown = qt_delivery.shutdown

    archive_entered, release_archive = Event(), Event()
    save_archive = WorkbenchRecordingArchive.save

    def save(archive, completion, product_name, recorded_on):
        archive_entered.set()
        if not release_archive.wait(12):
            raise TimeoutError("test did not release the archive writer")
        if delivery == "writing-fails":
            raise WorkbenchRecordingArchiveError("shutdown archive test failure")
        save_archive(archive, completion, product_name, recorded_on)

    monkeypatch.setattr(WorkbenchRecordingArchive, "save", save)

    def login(_):
        bridge.change_worker(1)
        bridge.answer_probe()
        return "Engineer", "test"

    window = startup.launch(login)
    _set_recording_identity(window)
    window.show()
    completions, analysis_inputs = [], []
    window.workbench_recording_completed.connect(completions.append)
    window.workbench_analysis_input_ready.connect(analysis_inputs.append)

    # Drain other shutdown dependencies so they cannot consume the recording
    # bridge's Qt events before the close path reaches the archive check.
    window.video_controller.shutdown()
    window.raw_audio_csv_bridge.begin_shutdown()
    _wait_for_qt(lambda: window.video_controller.is_shutdown_complete and window.raw_audio_csv_bridge.service_closed)
    QTest.mouseClick(window.sequence_window.player_btn, Qt.LeftButton)
    (session, callbacks), = submissions
    request = session.request
    group = request.station_group
    final_dir = Path(group.final_path)
    final_dir.mkdir(parents=True)
    stations = tuple(StationWav(station.station_id, str(final_dir / station.filename), station.channels)
                     for station in group.stations)
    for station in stations:
        sf.write(station.path, np.zeros((8, len(station.channels))), request.sample_rate)
    result = WorkbenchRecordingCompletion(request.request_id, group.project_id, group.condition_id,
                                          request.sample_rate, 8, stations)

    def publish():
        callbacks.workbench_completed(session, result)
        callbacks.released(session)
        if delivery == "queued":
            bridge.service.closed.set()

    publisher = Thread(target=publish)
    publisher.start()
    publisher.join(timeout=5)
    assert not publisher.is_alive()
    try:
        if delivery == "queued":
            assert bridge.service.closed.is_set() and not archive_entered.is_set()
        else:
            _wait_for_qt(archive_entered.is_set)
            assert not bridge.service.closed.is_set()
        assert not window.close()
        _wait_for_qt(lambda: archive_entered.is_set() and window._workbench_archive_close_state == "drained")
        assert bridge.service.closed.is_set()
        assert window.isVisible()
        assert _archive_row_count(startup.database_path) == 0
        assert completions == analysis_inputs == []
        assert not window.sequence_window.player_btn.isEnabled()
    finally:
        release_archive.set()
        _wait_for_qt(lambda: archive_entered.is_set() and not window._workbench_archive_bridge.pending)

    _wait_for_qt(lambda: window.analysis_service.closed and getattr(window, "_exit_cleanup_complete", False))
    assert not window.isVisible()
    assert completions == [result]
    assert analysis_inputs == []
    if delivery == "writing-fails":
        assert _archive_row_count(startup.database_path) == 0
        failure, = (record for record in caplog.records if "shutdown archive test failure" in record.getMessage())
        request_id, paths, _ = failure.args
        assert request_id == request.request_id
        assert paths == [station.path for station in stations]
        assert "入库失败" not in window.statusBar().currentMessage()
    else:
        assert _archive_row_count(startup.database_path) == len(stations)


def test_real_toolbar_reports_uncreatable_product_root_without_fallback(startup, config_files, tmp_path, monkeypatch):
    root = tmp_path / "blocked-root"
    root.write_bytes(b"existing file")
    _recording_program(config_files, root, monkeypatch)
    from consts import model_consts
    default = tmp_path / "default-output"
    monkeypatch.setattr(model_consts, "STORED_RECORDED_PATH", str(default))
    service = RecordingService(
        backend_factory="unit_test.base.recording_process_fakes:workbench_soundcard_dependencies",
        backend_options={"trace_dir": str(tmp_path), "frames": 12, "chunk_frames": 4},
    )
    bridge = RecordingServiceBridge(service)

    def login(window):
        _wait_for_qt(lambda: window.workbench_recording_device.snapshot is not None)
        return "Engineer", "test"

    startup.login(login)
    try:
        window = main_module.MainWindow(recording_bridge=bridge)
        _set_recording_identity(window)
        completions = []
        window.workbench_recording_completed.connect(completions.append)
        button = window.sequence_window.player_btn
        _wait_for_qt(button.isEnabled)
        QTest.mouseClick(button, Qt.LeftButton)
        _wait_for_qt(lambda: QMessageBox.warning.called and button.isEnabled())
        message = QMessageBox.warning.call_args.args[-1]
        assert "stage" in message and "blocked-root" in message
        assert not completions
        assert root.read_bytes() == b"existing file"
        assert not default.exists()
        assert not (tmp_path / "queue-output").exists()
    finally:
        bridge.shutdown()
        assert service.closed.wait(10)


def test_real_toolbar_publishes_spawned_station_wavs_retries_and_wraps(
    startup, config_files, tmp_path, monkeypatch, qt_app,
):
    application_root = tmp_path / "application"
    monkeypatch.setattr(running_consts, "DEFAULT_DIR", str(application_root))
    audio_root = application_root / "audio"
    _recording_program(config_files, audio_root, monkeypatch)
    product_dir = audio_root / "电机"
    wav_dir = product_dir / "audio"
    from ui.sequence import motor_workbench_controller as controller_module
    monkeypatch.setattr(controller_module, "datetime", SimpleNamespace(
        now=lambda: datetime(2026, 11, 9, 13, 14, 15, 123999)))
    monkeypatch.setattr(recording_calibration_snapshot, "load_mic_channel_calibrations",
                        lambda *_: {2: {"v2pa_factor": 2.5, "standard_spl_db": 94.0}})
    service = RecordingService(
        backend_factory="unit_test.base.recording_process_fakes:workbench_soundcard_dependencies",
        backend_options={"trace_dir": str(tmp_path), "frames": 12, "chunk_frames": 4, "manual": True},
    )
    bridge = RecordingServiceBridge(service)
    submissions = []
    real_start = bridge.start
    publish = station_stage_reservations.publish_verified_wav
    fail_publish = Event()

    def publish_wav(fd, parent, target):
        if fail_publish.is_set():
            fail_publish.clear()
            raise PermissionError("injected workbench publish failure")
        publish(fd, parent, target)

    monkeypatch.setattr(station_stage_reservations, "publish_verified_wav", publish_wav)

    def start(request, callbacks):
        submissions.append(request)
        return real_start(request, callbacks)

    def login(window):
        device = window.workbench_recording_device
        _wait_for_qt(lambda: device.snapshot is not None)
        assert device.snapshot.channels == (0, 1, 2)
        assert window.mic_channels == [0]
        assert not window.sequence_window.player_btn.isEnabled()
        return "Engineer", "test"

    monkeypatch.setattr(bridge, "start", start)
    startup.login(login)
    try:
        window = main_module.MainWindow(recording_bridge=bridge)
        sequence = _set_recording_identity(window)
        monkeypatch.setattr(sequence.hw_manager, "start_scanner_and_sensor_listeners", lambda: True)
        sequence._apply_scanner_enabled_state(True, persist=False)
        sequence.lineedit_s_or_n.setText("MANUAL")
        sequence.hw_manager.sig_barcode.emit("SN-008")
        assert sequence.lineedit_s_or_n.text() == "SN-008"
        button = window.sequence_window.player_btn
        completions, analysis_inputs = [], []
        window.workbench_recording_completed.connect(
            lambda result: completions.append((QThread.currentThread(), result)),
        )
        window.workbench_analysis_input_ready.connect(analysis_inputs.append)
        _wait_for_qt(button.isEnabled)
        QTest.mouseClick(button, Qt.LeftButton)
        button.clicked.emit()
        assert len(submissions) == 1
        for index in range(3):
            (tmp_path / f"feed-{index}").touch()
        _wait_for_qt(lambda: len(completions) == 1 and button.isEnabled())
        first_paths = set(audio_root.rglob("*.wav"))
        assert first_paths == {
            wav_dir / "2026-11-09_13-14-15-123_工位1_工况1_M-02_007_SN-008.wav",
            wav_dir / "2026-11-09_13-14-15-123_工位2_工况1_M-02_007_SN-008.wav",
        }
        from base.workbench_analysis_input import build_analysis_payload
        tasks = build_analysis_payload(analysis_inputs)["tasks"]
        assert {Path(task["wav_path"]) for task in tasks} == first_paths
        first_rows = _archive_rows(startup.database_path)
        assert set(first_rows) == {path.relative_to(application_root).as_posix() for path in first_paths}

        (tmp_path / "fail-next-write").touch()
        QTest.mouseClick(button, Qt.LeftButton)
        _wait_for_qt(lambda: QMessageBox.warning.called and button.isEnabled())
        assert "injected workbench write failure" in QMessageBox.warning.call_args.args[-1]
        assert len(completions) == len(analysis_inputs) == 1
        assert set(audio_root.rglob("*.wav")) == first_paths
        assert _archive_rows(startup.database_path) == first_rows

        fail_publish.set()
        QTest.mouseClick(button, Qt.LeftButton)
        _wait_for_qt(lambda: QMessageBox.warning.call_count == 2 and button.isEnabled())
        assert "injected workbench publish failure" in QMessageBox.warning.call_args.args[-1]
        assert len(completions) == len(analysis_inputs) == 1
        assert set(wav_dir.iterdir()) == first_paths
        assert _archive_rows(startup.database_path) == first_rows

        QTest.mouseClick(button, Qt.LeftButton)
        _wait_for_qt(lambda: len(completions) == 2 and button.isEnabled())

        archived_before_failure = _archive_rows(startup.database_path)
        with closing(sqlite3.connect(startup.database_path)) as database, database:
            database.execute("""
                CREATE TRIGGER fail_workbench_archive BEFORE INSERT ON audio_data_table
                WHEN NEW.file_path = 'audio/电机/audio/2026-11-09_13-14-15-123_工位2_工况1_M-02_007_2_SN-008.wav'
                BEGIN SELECT RAISE(ABORT, 'injected workbench archive failure'); END
            """)
        QTest.mouseClick(button, Qt.LeftButton)
        _wait_for_qt(lambda: len(completions) == 3 and button.isEnabled())
        message = window.statusBar().currentMessage()
        assert "工况1" in message and "音频已保存，入库失败" in message
        assert "injected workbench archive failure" in message
        assert QMessageBox.warning.call_count == 2
        assert _archive_rows(startup.database_path) == archived_before_failure
        assert analysis_inputs[-1].completion is completions[-1][1]
        QTest.mouseClick(button, Qt.LeftButton)
        _wait_for_qt(lambda: len(completions) == 4 and button.isEnabled())

        assert [request.station_group.condition_id for request in submissions] == ["c1", "c2", "c2", "c2", "c1", "c2"]
        assert all(request.channels == (0, 1, 2) and request.sample_rate == 48000 for request in submissions)
        assert set(audio_root.iterdir()) == {product_dir}
        assert set(product_dir.iterdir()) == {wav_dir}
        assert all(path.is_file() and path.suffix == ".wav" for path in wav_dir.iterdir())
        assert all(Path(request.path).parent == wav_dir for request in submissions)
        assert not (application_root / "queue-output").exists()
        trace = json.loads((tmp_path / "trace.json").read_text(encoding="utf-8"))
        assert trace["capture_pid"] == trace["writer_pid"] != os.getpid()
        assert [analysis.completion for analysis in analysis_inputs] == [result for _, result in completions]
        expected_rows = {}
        expected_paths = set()
        for index, ((thread, completed), request) in enumerate(zip(completions, (submissions[0], *submissions[3:]))):
            group = request.station_group
            first = completed.condition_id == "c1"
            frames = 8 if first else 10
            assert thread == qt_app.thread()
            assert (completed.request_id, completed.project_id, completed.condition_id) == (
                request.request_id, "motor", group.condition_id,
            )
            assert (completed.sample_rate, completed.final_frames) == (48000, frames)
            expected = [("s1", "工位1", (2, 0)), ("s2", "工位2", (0,))] if first else [
                ("s2", "工位2", (0,)),
            ]
            condition = "工况1" if first else "工况2"
            collision = "_2" if index >= 2 else ""
            assert len(completed.stations) == len(expected)
            assert not Path(request.path).exists()
            for station, (station_id, station_name, columns) in zip(completed.stations, expected):
                filename = f"2026-11-09_13-14-15-123_{station_name}_{condition}_M-02_007{collision}_SN-008.wav"
                assert station.station_id == station_id and Path(station.path) == wav_dir / filename
                expected_paths.add(Path(station.path))
                assert [(channel.physical_channel_index, channel.wav_channel_index) for channel in station.channels] == [
                    (physical, wav) for wav, physical in enumerate(columns)
                ]
                info = sf.info(station.path)
                assert (info.frames, info.channels, info.samplerate, info.subtype) == (
                    frames, len(columns), 48000, "PCM_24",
                )
                if index != 2:
                    relative_path = Path(station.path).relative_to(application_root).as_posix()
                    expected_rows[relative_path] = ("电机", info.samplerate, "2026-11-09", "not_labeled", None, None)
                audio, _ = sf.read(station.path, dtype="float32", always_2d=True)
                np.testing.assert_array_equal(audio, generated_audio(2, frames)[:, columns])
                metadata = read_wav_calibration_metadata(station.path)
                assert [(channel["wav_channel_index"], channel["physical_input_channel"], channel["v2pa_factor"])
                        for channel in metadata["recorded_channels"]] == [
                    (wav, physical, 2.5 if physical == 2 else None) for wav, physical in enumerate(columns)
                ]
        assert set(wav_dir.iterdir()) == expected_paths
        assert _archive_rows(startup.database_path) == expected_rows
    finally:
        bridge.shutdown()
        assert service.closed.wait(10)


def test_real_toolbar_archives_click_snapshot_across_midnight_and_product_switch(
    startup, config_files, tmp_path, monkeypatch, recording_icon_colors,
):
    play_color, pause_color = recording_icon_colors
    monkeypatch.setattr(running_consts, "DEFAULT_DIR", str(tmp_path / "application"))
    old_root, new_root = tmp_path / "external" / "old", tmp_path / "external" / "new"
    product = _recording_program(config_files, old_root, monkeypatch)
    from ui.sequence import motor_workbench_controller as controller_module
    now = datetime(2026, 11, 9, 23, 59, 59)
    monkeypatch.setattr(controller_module, "datetime", SimpleNamespace(now=lambda: now))

    archive_entered, release_archive = Event(), Event()
    save_archive = WorkbenchRecordingArchive.save

    def save(archive, *args):
        archive_entered.set()
        if not release_archive.wait(12):
            raise TimeoutError("test did not release the archive writer")
        save_archive(archive, *args)

    monkeypatch.setattr(WorkbenchRecordingArchive, "save", save)
    service = RecordingService(
        backend_factory="unit_test.base.recording_process_fakes:workbench_soundcard_dependencies",
        backend_options={"trace_dir": str(tmp_path), "frames": 12, "chunk_frames": 4, "pause_finalize": True},
    )
    bridge = RecordingServiceBridge(service)
    trace = tmp_path / "trace.json"
    release_finalize = tmp_path / "release-finalize"

    def finalizer_is_waiting():
        return trace.is_file() and json.loads(trace.read_text(encoding="utf-8")).get("finalize_waiting_for_release")

    def login(window):
        _wait_for_qt(lambda: window.workbench_recording_device.snapshot is not None)
        button = window.sequence_window.player_btn
        assert not button.isEnabled() and _icon_color(button) == play_color
        return "Engineer", "test"

    startup.login(login)
    window = None
    try:
        window = main_module.MainWindow(recording_bridge=bridge)
        sequence = _set_recording_identity(window)
        monkeypatch.setattr(sequence.hw_manager, "start_scanner_and_sensor_listeners", lambda: True)
        sequence._apply_scanner_enabled_state(True, persist=False)
        sequence.lineedit_s_or_n.setText("SN-OLD")
        completions = []
        window.workbench_recording_completed.connect(completions.append)
        button = window.sequence_window.player_btn
        _wait_for_qt(button.isEnabled)
        assert button.isEnabled() and _icon_color(button) == play_color
        QTest.mouseClick(button, Qt.LeftButton)
        assert not button.isEnabled() and _icon_color(button, QIcon.Disabled) == pause_color

        _wait_for_qt(finalizer_is_waiting)
        assert not button.isEnabled() and _icon_color(button, QIcon.Disabled) == pause_color
        assert not completions and _archive_row_count(startup.database_path) == 0

        now = datetime(2026, 11, 10, 0, 0, 1)
        _set_recording_identity(window, "M-03", "009")
        sequence.lineedit_s_or_n.clear()
        manager = ProductTestProjectConfigManager(
            program_dir=str(config_files["programs"].parent), registry_path=str(config_files["programs"]),
            queue_registry_path=str(config_files["queues"]),
        )
        second_file = config_files["product"].with_name("second.json")
        write_json(second_file, dict(product, project_id="second", project_name="第二产品",
                                     result_root_directory=str(new_root), conditions=product["conditions"][::-1]))
        registry = manager.load_registry()
        registry["configs"].append({"file": second_file.name, "project_name": "第二产品"})
        registry["active_file"] = second_file.name
        assert manager.save_registry(registry)
        window.on_workbench_configuration_committed()
        assert window.sequence_window.using_file_combobox.currentText() == "第二产品"
        assert not button.isEnabled() and _icon_color(button, QIcon.Disabled) == pause_color
        release_finalize.touch()

        _wait_for_qt(archive_entered.is_set)
        old_dir = old_root / "电机" / "audio"
        old_paths = {
            old_dir / f"2026-11-09_23-59-59-000_{station}_工况1_M-02_007_SN-OLD.wav"
            for station in ("工位1", "工位2")
        }
        assert set(old_dir.iterdir()) == old_paths
        assert not button.isEnabled() and _icon_color(button, QIcon.Disabled) == pause_color
        assert not completions and _archive_row_count(startup.database_path) == 0
        release_archive.set()

        _wait_for_qt(lambda: len(completions) == 1 and button.isEnabled())
        assert _icon_color(button) == play_color
        first = completions[0]
        assert (first.project_id, first.condition_id) == ("motor", "c1")
        assert {Path(station.path) for station in first.stations} == old_paths
        expected_rows = {path.as_posix(): ("电机", sf.info(path).samplerate, "2026-11-09", "not_labeled", None, None)
                         for path in old_paths}
        assert _archive_rows(startup.database_path) == expected_rows

        QTest.mouseClick(button, Qt.LeftButton)
        _wait_for_qt(lambda: len(completions) == 2 and button.isEnabled())
        second = completions[1]
        assert (second.project_id, second.condition_id) == ("second", "c2")
        assert sequence.barcode_scanner_box.isChecked()
        new_path = new_root / "第二产品" / "audio" / "2026-11-10_00-00-01-000_工位2_工况2_M-03_009.wav"
        assert tuple(Path(station.path) for station in second.stations) == (new_path,)
        expected_rows[new_path.as_posix()] = ("第二产品", sf.info(new_path).samplerate, "2026-11-10",
                                             "not_labeled", None, None)
        assert _archive_rows(startup.database_path) == expected_rows
        assert _icon_color(button) == play_color
    finally:
        release_finalize.touch()
        release_archive.set()
        if window is None:
            bridge.shutdown()
            assert service.closed.wait(10)
        else:
            window.close()
            _wait_for_qt(lambda: getattr(window, "_exit_cleanup_complete", False))

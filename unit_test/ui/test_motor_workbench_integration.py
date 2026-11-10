"""Exercise MainWindow's workbench bindings with synthetic committed files."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from threading import Thread
from time import monotonic
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import soundfile as sf
from PyQt5.QtCore import QCoreApplication, QEvent, QObject, QThread, Qt, pyqtSignal
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QLabel, QMainWindow, QMenuBar, QMessageBox, QWidget

import main_window as main_module
from base import hardware_analysis_channel_sync, load_config, recording_calibration_snapshot
from base.config_deletion import ConfigDeletionService
from base.motor_workbench_config import MotorWorkbenchConfigRepository
from base.product_test_project_config import ProductTestProjectConfigManager
from base.recording_process_protocol import RecordingFailure
from base.recording_service import RecordingService
from base.streaming_file_writer import StreamingWavWriter
from base.wav_calibration_metadata import read_wav_calibration_metadata
from base.workbench_recording_protocol import StationWav, WorkbenchRecordingCompletion
from consts import error_code
from main_window import MainWindow
from ui.config_delete_dialog import ConfigDeleteDialog
from ui.recording_service_bridge import RecordingServiceBridge
from ui.sequence import sequence_widget, sequence_widget_serial_trigger_ops
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
    original = {path: path.read_bytes() for path in (*config_files.values(), independent_path, other_path)}

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
        (ProductTestProjectConfigManager, ("save_registry", "save_project")),
        (hardware_analysis_channel_sync, ("synchronize_product_analysis_channels",)),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)

    hardware = _Hardware()
    monkeypatch.setattr(sequence_widget, "UnifiedHardwareManager", lambda: hardware)
    monkeypatch.setattr(sequence_widget.LogManager, "set_log_handler", logging.getLogger)
    monkeypatch.setattr(sequence_widget.LoadUiConfig, "load_last_recorded_info", lambda _: {})
    for name in ("restore_scanner_checkbox_state", "_restore_last_sequence_mode", "init_lineedit_text"):
        monkeypatch.setattr(SequenceWindow, name, lambda self: None)
    monkeypatch.setattr(SequenceWindow, "_load_analysis_window_geometry", lambda self: {})
    monkeypatch.setattr(SequenceWindow, "closeEvent", QWidget.closeEvent)
    warning = Mock()
    information = Mock()
    monkeypatch.setattr(QMessageBox, "warning", warning)
    monkeypatch.setattr(QMessageBox, "information", information)

    window = _WorkbenchWindow(tmp_path / "video.json")
    try:
        yield SimpleNamespace(window=window, files=config_files, original=original,
                              load_active=load_active, independent_path=independent_path, other_path=other_path,
                              hardware=hardware, warning=warning, information=information,
                              forbidden=forbidden)
    finally:
        sequence = window.sequence_window
        qt_app.removeEventFilter(sequence)
        if window.video_controller.dialog is not None:
            window.video_controller.dialog.close()
        window.video_controller.shutdown()
        window.video_controller._timer.stop()
        window.hide()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def _assert_display(window, product, names):
    sequence = window.sequence_window
    assert sequence.using_file_combobox.currentText() == product
    assert not sequence.using_file_combobox.isEnabled()
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
    _assert_display(window, "另一产品", ("工位二", "工位一", "工位三", "工位四"))
    assert window.sequence_window.left_panel.get_station_data(1)["forward"] == "--"

    registry["active_file"] = None
    write_json(files["programs"], registry)
    editor.committed.emit()
    _assert_display(window, "", ())
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


def test_real_toolbar_waits_for_login_and_resource_admission_then_delivers_group_on_gui_thread(
    startup, config_files, qt_app, monkeypatch, tmp_path,
):
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
        button.clicked.emit()
        assert submissions == []
        return "Engineer", "test"

    window = startup.launch(
        login, ve_profile_store=SimpleNamespace(load=lambda *_: discovered["input_config"]),
        ve_calibration_store=SimpleNamespace(observe=lambda _: {}),
    )
    button = window.sequence_window.player_btn
    assert button.isEnabled()
    assert not window.sequence_window.using_file_combobox.isEnabled()
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
    qt_app.processEvents()
    assert completions == [(qt_app.thread(), payload)]
    assert len(analysis_inputs) == 1
    assert analysis_inputs[0].completion is payload
    from base.workbench_analysis_input import build_analysis_payload
    assert len(build_analysis_payload(analysis_inputs)["tasks"]) == len(payload.stations)
    first_wav, second_wav = completions[0][1].stations
    assert (first_wav.station_id, second_wav.station_id) == ("s1", "s2")
    assert Path(first_wav.path) == Path(group.final_path) / "工位1.wav"
    assert Path(second_wav.path) == Path(group.final_path) / "工位2.wav"
    assert Path(group.final_path).parent == Path(product["result_root_directory"]) / "电机" / "工况1"
    assert Path(first_wav.path).is_absolute()
    assert [(c.channel_id, c.physical_channel_index, c.wav_channel_index) for c in first_wav.channels] == [
        ("ch1", 7, 0), ("ch2", 1, 1),
    ]
    assert [(c.channel_id, c.physical_channel_index, c.wav_channel_index) for c in second_wav.channels] == [
        ("input1", 7, 0),
    ]
    assert button.isEnabled()
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
    monkeypatch.setattr(load_config, "SEQUENCE_CONFIG_REGISTRY_PATH", str(files["queues"]))
    monkeypatch.setattr(QMessageBox, "information", Mock())
    monkeypatch.setattr(QMessageBox, "warning", Mock())
    return product


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

    def login(_):
        bridge.change_worker(1)
        bridge.answer_probe()
        return "Engineer", "test"

    bridge.start = start
    window = startup.launch(login)
    button = window.sequence_window.player_btn
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
        completions = []
        window.workbench_recording_completed.connect(completions.append)
        button = window.sequence_window.player_btn
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
    audio_root = tmp_path / "audio"
    _recording_program(config_files, audio_root, monkeypatch)
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
        button = window.sequence_window.player_btn
        completions = []
        window.workbench_recording_completed.connect(
            lambda result: completions.append((QThread.currentThread(), result)),
        )
        assert button.isEnabled()
        QTest.mouseClick(button, Qt.LeftButton)
        button.clicked.emit()
        assert len(submissions) == 1
        for index in range(3):
            (tmp_path / f"feed-{index}").touch()
        _wait_for_qt(lambda: len(completions) == 1 and button.isEnabled())
        first_paths = set(audio_root.rglob("*.wav"))

        (tmp_path / "fail-next-write").touch()
        QTest.mouseClick(button, Qt.LeftButton)
        _wait_for_qt(lambda: QMessageBox.warning.called and button.isEnabled())
        assert "injected workbench write failure" in QMessageBox.warning.call_args.args[-1]
        assert len(completions) == 1 and set(audio_root.rglob("*.wav")) == first_paths
        QTest.mouseClick(button, Qt.LeftButton)
        _wait_for_qt(lambda: len(completions) == 2 and button.isEnabled())
        QTest.mouseClick(button, Qt.LeftButton)
        _wait_for_qt(lambda: len(completions) == 3 and button.isEnabled())

        assert [request.station_group.condition_id for request in submissions] == ["c1", "c2", "c2", "c1"]
        assert all(request.channels == (0, 1, 2) and request.sample_rate == 48000 for request in submissions)
        assert len({path.parent for path in audio_root.rglob("*.wav")}) == 3
        assert not (tmp_path / "queue-output").exists()
        trace = json.loads((tmp_path / "trace.json").read_text(encoding="utf-8"))
        assert trace["capture_pid"] == trace["writer_pid"] != os.getpid()
        for index, ((thread, completed), request) in enumerate(zip(completions, (submissions[0], *submissions[2:]))):
            group = request.station_group
            first = completed.condition_id == "c1"
            frames = 8 if first else 10
            assert thread == qt_app.thread()
            assert (completed.request_id, completed.project_id, completed.condition_id) == (
                request.request_id, "motor", group.condition_id,
            )
            assert (completed.sample_rate, completed.final_frames) == (48000, frames)
            expected = [("s1", "工位1.wav", (2, 0)), ("s2", "工位2.wav", (0,))] if first else [
                ("s2", "工位2.wav", (0,)),
            ]
            final = Path(completed.stations[0].path).parent
            assert final.parent == audio_root / "电机" / ("工况1" if first else "工况2")
            assert final.name == "2026-11-09_13-14-15-123" + ("_2" if index == 2 else "")
            assert len(completed.stations) == len(expected)
            assert not Path(request.path).exists()
            assert sorted(path.name for path in final.iterdir()) == [entry[1] for entry in expected]
            for station, (station_id, filename, columns) in zip(completed.stations, expected):
                assert station.station_id == station_id and Path(station.path) == final / filename
                assert [(channel.physical_channel_index, channel.wav_channel_index) for channel in station.channels] == [
                    (physical, wav) for wav, physical in enumerate(columns)
                ]
                info = sf.info(station.path)
                assert (info.frames, info.channels, info.samplerate, info.subtype) == (
                    frames, len(columns), 48000, "PCM_24",
                )
                audio, _ = sf.read(station.path, dtype="float32", always_2d=True)
                np.testing.assert_array_equal(audio, generated_audio(2, frames)[:, columns])
                metadata = read_wav_calibration_metadata(station.path)
                assert [(channel["wav_channel_index"], channel["physical_input_channel"], channel["v2pa_factor"])
                        for channel in metadata["recorded_channels"]] == [
                    (wav, physical, 2.5 if physical == 2 else None) for wav, physical in enumerate(columns)
                ]
    finally:
        bridge.shutdown()
        assert service.closed.wait(10)

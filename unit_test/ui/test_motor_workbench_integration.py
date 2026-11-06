"""Exercise MainWindow's workbench bindings with synthetic committed files."""

import json
import logging
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from PyQt5.QtCore import QCoreApplication, QEvent, QObject, Qt, pyqtSignal
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QLabel, QMainWindow, QMenuBar, QMessageBox, QWidget

import main_window as main_module
from base import hardware_analysis_channel_sync, load_config
from base.motor_workbench_config import MotorWorkbenchConfigRepository
from base.product_test_project_config import ProductTestProjectConfigManager
from base.streaming_file_writer import StreamingWavWriter
from consts import error_code
from main_window import MainWindow
from ui.sequence import sequence_widget, sequence_widget_serial_trigger_ops
from ui.sequence.sequence_widget import SequenceWindow
from ui.video_controller import VideoController
from unit_test.base.test_motor_workbench_config import acquisition, config_files, write_json


class _CommitDialog(QObject):
    programs_changed = pyqtSignal()

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
        assert condition.queue.analysis["声压级"]["analysis_channels"] == (1, 0)
        exported = condition.export_process_config()
        assert exported["analysis_config"]["声压级"]["analysis_channels"] == [1, 0]
        assert exported["sequence_config"][0]["seq1"]["analysis_list"]["声压级"]["analysis_channels"] == [1, 0]
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
    exported = controller.get_condition_config("c1").export_process_config()
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
        assert_results("--", 0)
        if target is detail:
            assert controller.get_condition_config("c1").queue.acquisition["detail"][key] == value
        elif target is analysis:
            assert controller.get_condition_config("c1").export_process_config()["analysis_config"]["声压级"][key] == value

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
    assert controller.get_condition_config("c1").queue is controller.get_condition_config("c2").queue
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
        dialog.programs_changed.emit()

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
    assert "工作台暂未接入产品测试" in workbench.information.call_args.args

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

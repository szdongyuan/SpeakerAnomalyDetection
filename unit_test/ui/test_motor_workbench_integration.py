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
from consts import error_code
from main_window import MainWindow
from ui.sequence import sequence_widget, sequence_widget_serial_trigger_ops
from ui.sequence.sequence_widget import SequenceWindow
from ui.video_controller import VideoController
from unit_test.base.test_motor_workbench_config import config_files, write_json


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


@pytest.fixture
def workbench(qt_app, tmp_path, monkeypatch, config_files, request):
    startup = getattr(request, "param", "with fallback")
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
    if startup == "invalid product":
        config_files["product"].write_text("{", encoding="utf-8")
    elif startup == "missing product":
        write_json(config_files["programs"], {
            "active_file": "missing.json",
            "configs": [{"file": "missing.json", "project_name": "缺失产品"}],
        })
    original = {path: path.read_bytes() for path in (*config_files.values(), independent_path, other_path)}

    repository = MotorWorkbenchConfigRepository(config_files["programs"], config_files["queues"])
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
                          "_start_directional_workflow", "update_recorded_signal_info_to_db")),
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
    assert tuple(header.name.toolTip() for header in sequence.plot_grid._headers.values()) == names
    assert len(sequence.plot_grid._cells) == 2 * len(names)


def test_startup_reads_both_views_and_uses_independent_fallback_in_memory(workbench):
    window = workbench.window
    _assert_display(window, "电机", ("工位三", "工位一", "工位二"))
    sequence = window.sequence_window
    assert sequence.using_config_path == str(workbench.independent_path)
    assert sequence.count_board.analysis_config is sequence.analysis_config
    assert sequence.left_panel.get_station_data(1)["rows"] == [["B", "--", "--"], ["A", "--", "--"]]
    assert sequence.left_panel.get_station_data(1)["forward"] == "--"
    assert sequence.left_panel.get_station_data(1)["reverse"] == "--"
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
def test_workbench_starts_when_its_queues_have_only_display_fields(workbench):
    window = workbench.window
    assert window.sequence_window.using_config_path is None
    _assert_display(window, "电机", ("工位三", "工位一", "工位二"))


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
    write_json(files["shared"], [{"seq1": {"analysis_list": {"display_sequence": ["C"], "C": {}}}}])
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

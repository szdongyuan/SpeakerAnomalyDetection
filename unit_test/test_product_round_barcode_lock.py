import ast
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QApplication, QCheckBox, QLineEdit, QWidget

from ui.sequence.sequence_widget_barcode_ops import SequenceWidgetBarcodeOpsMixin
from ui.sequence.sequence_widget_config_ops import SequenceWidgetConfigOpsMixin


ANALYSIS_OPS_PATH = (
    Path(__file__).resolve().parents[1]
    / "ui"
    / "sequence"
    / "sequence_widget_analysis_ops.py"
)


def _load_analysis_method(method_name):
    module_tree = ast.parse(ANALYSIS_OPS_PATH.read_text(encoding="utf-8"))
    mixin = next(
        node
        for node in module_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "SequenceWidgetAnalysisOpsMixin"
    )
    method = next(
        node
        for node in mixin.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    test_class = ast.ClassDef(
        name="TestAnalysisMixin",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[test_class], type_ignores=[])
            ),
            str(ANALYSIS_OPS_PATH),
            "exec",
        ),
        namespace,
    )
    return getattr(namespace["TestAnalysisMixin"], method_name)


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))


class _LineEdit:
    def __init__(self, text=""):
        self._text = text
        self._readonly = False
        self._enabled = True
        self._tooltip = ""

    def text(self):
        return self._text

    def setText(self, text):
        self._text = str(text)

    def clear(self):
        self._text = ""

    def isReadOnly(self):
        return self._readonly

    def setReadOnly(self, readonly):
        self._readonly = bool(readonly)

    def setEnabled(self, enabled):
        self._enabled = bool(enabled)

    def setDisabled(self, disabled):
        self._enabled = not bool(disabled)

    def isEnabled(self):
        return self._enabled

    def toolTip(self):
        return self._tooltip

    def setToolTip(self, tooltip):
        self._tooltip = str(tooltip)

    def setFocus(self):
        return None

    def selectAll(self):
        return None


class _CheckBox:
    def __init__(self, checked=True):
        self._checked = checked

    def isChecked(self):
        return self._checked

    def setChecked(self, checked):
        self._checked = bool(checked)


class _HardwareManager:
    barcode_source = "hid"

    def ensure_config_loaded(self, force_reload=False):
        return bool(force_reload)

    def start_scanner_and_sensor_listeners(self):
        return True

    def stop_scanner_and_sensor_listeners(self):
        return None


class _ProductRoundBarcodeHost(SequenceWidgetBarcodeOpsMixin):
    _reset_manual_product_condition_cycle = _load_analysis_method(
        "_reset_manual_product_condition_cycle"
    )
    _advance_manual_product_condition_cycle_after_recording = (
        _load_analysis_method(
            "_advance_manual_product_condition_cycle_after_recording"
        )
    )
    _reserve_recorded_count_for_run = _load_analysis_method(
        "_reserve_recorded_count_for_run"
    )

    def __init__(self, barcode="SN001"):
        self.product_test_condition_configs = [
            {"key": "q6000", "test_queue": "queue_6000"},
            {"key": "q7000", "test_queue": "queue_7000"},
        ]
        self._manual_product_condition_index = 0
        self._manual_product_condition_group_id = "round-1"
        self._displayed_manual_product_condition_group_id = "round-1"
        self._manual_product_condition_results = {}
        self._manual_product_condition_completed_keys = set()
        self._manual_product_condition_counted_group_labels = {}
        self._active_product_condition_key = "q6000"
        self._active_product_condition_config = dict(
            self.product_test_condition_configs[0]
        )
        self._waveform_display_override_direction = "q6000"
        self._current_trigger_direction = "q6000"
        self._current_cycle_recorded_count = "round-1"
        self._serial_product_condition_executing = False
        self._serial_product_waiting_for_close = False
        self.product_test_close_trigger_state = ""
        self._sn_locked_for_cycle = False
        self._sn_locked_for_product_round = False
        self._record_workflow_busy = False
        self._barcode_commit_suppressed_until = 0.0
        self._last_committed_barcode = None
        self._last_committed_barcode_time = 0.0
        self._barcode_commit_dedup_window_sec = 0.8
        self._barcode_capture_buffer = ""
        self._barcode_capture_first_ts = None
        self._barcode_capture_last_ts = None
        self._barcode_capture_target_lineedit = None
        self._barcode_capture_target_text = None
        self._barcode_capture_target_cursor_pos = None
        self._barcode_first_char_ts = None
        self._barcode_last_char_ts = None
        self._current_run_recording_token = ""
        self.last_play_count = None
        self.lineedit_s_or_n = _LineEdit(barcode)
        self.barcode_scanner_box = _CheckBox(True)
        self.hw_manager = _HardwareManager()
        self.default_logger = _Logger()
        self.left_panel = None

    def _generate_recording_token(self):
        return "round-generated"

    def _get_active_product_condition_key(self):
        return str(self._active_product_condition_key or "")

    def _is_manual_product_condition_cycle_active(self):
        return bool(self._get_active_product_condition_key())

    def _product_condition_sequence(self):
        return [dict(item) for item in self.product_test_condition_configs]

    @staticmethod
    def _product_condition_runtime_key(condition, index=0):
        return str(condition.get("key") or index)

    def _get_recording_direction(self):
        return ""

    def _reset_product_condition_display_state(self):
        return None


class _BarcodeEventFilterHost(
    SequenceWidgetConfigOpsMixin,
    SequenceWidgetBarcodeOpsMixin,
    QWidget,
):
    def __init__(self):
        super().__init__()
        self._analysis_window_key_by_obj = {}
        self.lineedit_type = QLineEdit(self)
        self.lineedit_s_or_n = QLineEdit(self)
        self.lineedit_s_or_n.setText("SN001")
        self.barcode_scanner_box = QCheckBox(self)
        self.barcode_scanner_box.setChecked(True)
        self._hid_mode_active_until = 0.0
        self._current_trigger_direction = "q6000"
        self._sn_locked_for_cycle = False
        self._sn_locked_for_product_round = True
        self._sync_sn_lock_ui()
        self._sn_clear_on_next_scan = True
        self.player_status_flag = False
        self._barcode_debounce_timer = type(
            "TimerStub",
            (),
            {"stop": lambda self: None},
        )()
        self.default_logger = _Logger()


def test_product_round_locks_existing_barcode_and_rejects_all_new_scans():
    host = _ProductRoundBarcodeHost("SN001")

    assert host._reserve_recorded_count_for_run() == "round-1"
    assert host.lineedit_s_or_n.isReadOnly()

    for barcode in ("SN001", "SN002"):
        host._commit_barcode(barcode, source="hid")
        assert host.lineedit_s_or_n.text() == "SN001"

    assert any("产品测试轮次" in message for _level, message in host.default_logger.messages)


def test_product_round_can_lock_an_empty_barcode_and_ignore_mid_round_scan():
    host = _ProductRoundBarcodeHost("")

    host._reserve_recorded_count_for_run()
    host._commit_barcode("SN001", source="serial")

    assert host.lineedit_s_or_n.isReadOnly()
    assert host.lineedit_s_or_n.text() == ""


def test_keyboard_wedge_input_cannot_clear_locked_product_round_barcode():
    app = QApplication.instance() or QApplication([])
    host = _BarcodeEventFilterHost()
    host.show()
    host.lineedit_s_or_n.setFocus()
    app.processEvents()

    try:
        key_event = QKeyEvent(
            QEvent.KeyPress,
            Qt.Key_X,
            Qt.NoModifier,
            "X",
        )
        host.eventFilter(host.lineedit_s_or_n, key_event)

        assert host.lineedit_s_or_n.isReadOnly()
        assert host.lineedit_s_or_n.text() == "SN001"
        assert host._sn_clear_on_next_scan is True
    finally:
        host.close()


def test_product_round_unlocks_and_clears_only_after_last_condition():
    host = _ProductRoundBarcodeHost("SN001")
    host._reserve_recorded_count_for_run()

    host._advance_manual_product_condition_cycle_after_recording()

    assert host._manual_product_condition_index == 1
    assert host.lineedit_s_or_n.isReadOnly()
    assert host.lineedit_s_or_n.text() == "SN001"

    host._active_product_condition_key = "q7000"
    host._active_product_condition_config = dict(
        host.product_test_condition_configs[1]
    )
    host._reserve_recorded_count_for_run()
    host._advance_manual_product_condition_cycle_after_recording()

    assert host._manual_product_condition_index == 0
    assert host._manual_product_condition_group_id == ""
    assert not host.lineedit_s_or_n.isReadOnly()
    assert host.lineedit_s_or_n.text() == ""


def test_manual_product_round_auto_closes_after_last_condition_even_with_close_trigger():
    host = _ProductRoundBarcodeHost("SN001")
    host.product_test_close_trigger_state = "01 04 02 00 00 B9 30"
    host._reserve_recorded_count_for_run()

    host._advance_manual_product_condition_cycle_after_recording()
    host._active_product_condition_key = "q7000"
    host._active_product_condition_config = dict(
        host.product_test_condition_configs[1]
    )
    host._reserve_recorded_count_for_run()
    host._advance_manual_product_condition_cycle_after_recording()

    assert host._manual_product_condition_index == 0
    assert host._manual_product_condition_group_id == ""
    assert host._serial_product_waiting_for_close is False
    assert not host.lineedit_s_or_n.isReadOnly()
    assert host.lineedit_s_or_n.text() == ""


def test_product_round_reset_unlocks_and_clears_barcode():
    host = _ProductRoundBarcodeHost("SN001")
    host._reserve_recorded_count_for_run()

    host._reset_manual_product_condition_cycle(clear_waveforms=False)

    assert not host.lineedit_s_or_n.isReadOnly()
    assert host.lineedit_s_or_n.text() == ""


def test_direction_unlock_does_not_release_product_round_lock():
    host = _ProductRoundBarcodeHost("SN001")
    host._lock_sn_for_cycle()
    host._lock_sn_for_product_round()

    host._unlock_sn_for_cycle()

    assert host.lineedit_s_or_n.isReadOnly()
    assert "产品测试轮次" in host.lineedit_s_or_n.toolTip()


def test_recording_unlock_does_not_release_product_round_lock():
    host = _ProductRoundBarcodeHost("SN001")
    host._sn_locked_for_recording = True
    host._sn_readonly_before_recording = False
    host._sn_tooltip_before_recording = ""
    host._lock_sn_for_product_round()

    host._unlock_sn_after_recording_if_needed()

    assert host._sn_locked_for_product_round is True
    assert host.lineedit_s_or_n.isReadOnly()
    assert "产品测试轮次" in host.lineedit_s_or_n.toolTip()


def test_scanner_toggle_keeps_locked_round_barcode_and_readonly_state():
    host = _ProductRoundBarcodeHost("SN001")
    host._lock_sn_for_product_round()

    host._apply_scanner_enabled_state(False, persist=False)
    assert host.lineedit_s_or_n.text() == "SN001"
    assert not host.lineedit_s_or_n.isEnabled()

    host._apply_scanner_enabled_state(True, persist=False)
    assert host.lineedit_s_or_n.text() == "SN001"
    assert host.lineedit_s_or_n.isEnabled()
    assert host.lineedit_s_or_n.isReadOnly()

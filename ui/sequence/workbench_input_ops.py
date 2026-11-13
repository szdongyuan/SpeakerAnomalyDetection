"""Workbench operator metadata and barcode input, independent of recording."""
import time

from PyQt5.QtCore import QEvent, QSignalBlocker, QTimer, Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QApplication, QMessageBox

from base.load_config import LoadUiConfig
from base.product_model_validation import validate_product_model
from base.save_data import save_recorded_data_to_json
from ui.sequence.barcode_router import BarcodeRouter


class WorkbenchInputOpsMixin:
    """Host supplies the visible edits, checkbox, hardware manager and logger."""

    def init_workbench_input(self, can_commit):
        self._can_commit_barcode = can_commit
        self._workbench_input_closed = False
        self._product_model_validation_open = False
        self._barcode_signals_bound = False
        self._barcode_fast_input_max_seconds = 0.4
        self._barcode_min_length_for_auto_commit = 4
        self._barcode_commit_dedup_window_sec = 0.8
        self._last_committed_barcode = None
        self._last_committed_barcode_time = 0.0
        self._hid_mode_active_until = 0.0
        self._clear_barcode_capture()
        self._barcode_router = BarcodeRouter(self)
        self._barcode_debounce_timer = QTimer(self)
        self._barcode_debounce_timer.setSingleShot(True)
        self._barcode_debounce_timer.setInterval(120)
        self._barcode_debounce_timer.timeout.connect(self._barcode_router.on_barcode_debounce_timeout)
        self.lineedit_s_or_n.returnPressed.connect(self._barcode_router.on_barcode_return_pressed)
        self.lineedit_s_or_n.textChanged.connect(self._barcode_router.on_barcode_text_changed)
        self.lineedit_type.editingFinished.connect(self._finish_product_model_edit)
        self.barcode_scanner_box.clicked.connect(self.clicked_scanner)
        saved = LoadUiConfig.load_last_recorded_info(self.default_logger)
        self.lineedit_type.setText(saved.get("product_model", "S004-1") if isinstance(saved, dict) else "S004-1")
        self.lineedit_type.setReadOnly(True)
        QApplication.instance().installEventFilter(self)
        self.bind_hw_signals()
        self.restore_scanner_checkbox_state()

    def _clear_barcode_capture(self):
        self._barcode_first_char_ts = None
        self._barcode_last_char_ts = None
        self._barcode_capture_buffer = ""
        self._barcode_capture_first_ts = None
        self._barcode_capture_last_ts = None
        self._barcode_capture_target_lineedit = None
        self._barcode_capture_target_text = None
        self._barcode_capture_target_cursor_pos = None

    @staticmethod
    def _normalize_saved_scanner_checkbox_state(value):
        if isinstance(value, (bool, int, float)):
            return bool(value)
        return isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on", "enabled"}

    def bind_hw_signals(self):
        if not self._workbench_input_closed and not self._barcode_signals_bound:
            self.hw_manager.sig_barcode.connect(self.on_barcode_received)
            self._barcode_signals_bound = True

    def restore_scanner_checkbox_state(self):
        saved = LoadUiConfig.load_last_recorded_info(self.default_logger)
        value = saved.get("scanner_barcode_check") if isinstance(saved, dict) else None
        self._apply_scanner_enabled_state(self._normalize_saved_scanner_checkbox_state(value), persist=False)

    def clicked_scanner(self):
        self._apply_scanner_enabled_state(self.barcode_scanner_box.isChecked())

    def _apply_scanner_enabled_state(self, enabled, persist=True):
        if self._workbench_input_closed:
            return
        enabled = bool(enabled)
        self.barcode_scanner_box.setChecked(enabled)
        self._barcode_debounce_timer.stop()
        self._clear_barcode_capture()
        self.lineedit_s_or_n.setEnabled(enabled)
        if enabled:
            if not self.hw_manager.ensure_config_loaded(force_reload=True):
                self.default_logger.warning("扫码枪配置加载失败，将按已有配置或默认 HID 自动识别模式启动。")
            self.lineedit_s_or_n.setFocus()
            self.lineedit_s_or_n.selectAll()
            if not self.hw_manager.start_scanner_listeners():
                self.default_logger.warning("硬件初始化异常，已降级为普通键盘输入模式")
        else:
            self.lineedit_s_or_n.clear()
            self.hw_manager.stop_scanner_listeners()
        if persist:
            save_recorded_data_to_json(scanner_barcode_check=enabled)

    def on_barcode_received(self, barcode):
        if self._workbench_input_closed:
            return
        self._hid_mode_active_until = time.monotonic() + 1.0
        self._barcode_debounce_timer.stop()
        self._clear_barcode_capture()
        self._commit_barcode(barcode, source=self.hw_manager.barcode_source)

    @staticmethod
    def _normalize_barcode(text):
        return "" if text is None else str(text).strip()

    @staticmethod
    def _barcode_has_invalid_chars(barcode):
        found = [ch for ch in barcode if ch in '\\:*?"<>|' or ch.isspace() or ord(ch) < 32 or ord(ch) == 127]
        return bool(found), found

    def _warn_invalid_barcode(self, barcode, invalid_chars):
        unique_chars = sorted(set(invalid_chars), key=lambda ch: (ord(ch), ch))
        self.default_logger.info(f"[barcode][ui] 条码格式异常 chars={unique_chars}, barcode={barcode!r}")
        chars_display = "  ".join(repr(ch) for ch in unique_chars)
        QMessageBox.warning(
            self, "条码格式异常",
            '条码允许包含 /，但不能包含换行、回车、制表符、空格、控制字符或以下字符：\\ : * ? " < > |。\n\n'
            f"异常字符: {chars_display}\n\n请重新扫码后再开始录音。",
        )

    def _validate_current_barcode_before_recording(self):
        barcode = self._normalize_barcode(self.lineedit_s_or_n.text())
        invalid, chars = self._barcode_has_invalid_chars(barcode)
        if invalid:
            self._warn_invalid_barcode(barcode, chars)
            self.lineedit_s_or_n.setFocus()
            self.lineedit_s_or_n.selectAll()
            return False
        return True

    def _commit_barcode(self, barcode, source="wedge"):
        barcode = self._normalize_barcode(barcode)
        if (self._workbench_input_closed or not barcode or not self.barcode_scanner_box.isChecked()
                or not self._can_commit_barcode()):
            return
        now = time.monotonic()
        if (self._last_committed_barcode == barcode
                and now - self._last_committed_barcode_time < self._barcode_commit_dedup_window_sec):
            self._clear_barcode_capture()
            return
        self._last_committed_barcode = barcode
        self._last_committed_barcode_time = now
        focus = QApplication.focusWidget()
        invalid, chars = self._barcode_has_invalid_chars(barcode)
        if invalid:
            self._warn_invalid_barcode(barcode, chars)
            with QSignalBlocker(self.lineedit_s_or_n):
                self.lineedit_s_or_n.clear()
            self._clear_barcode_capture()
            return
        with QSignalBlocker(self.lineedit_s_or_n):
            self.lineedit_s_or_n.setText(barcode)
        self._barcode_debounce_timer.stop()
        self._clear_barcode_capture()
        save_recorded_data_to_json(scanner_barcode=barcode)
        self.default_logger.info(f"[barcode][ui] S/N 已更新: {barcode!r} (source={source})")
        if focus is not self.lineedit_type:
            self.lineedit_s_or_n.setFocus()
            self.lineedit_s_or_n.selectAll()

    def _validate_product_model(self):
        if self._product_model_validation_open:
            return False
        try:
            model = validate_product_model(self.lineedit_type.text())
        except ValueError as error:
            self._product_model_validation_open = True
            try:
                QMessageBox.warning(self, "提示", str(error))
                self.lineedit_type.setFocus()
                self.lineedit_type.selectAll()
            finally:
                self._product_model_validation_open = False
            return False
        self.lineedit_type.setText(model)
        return True

    def _finish_product_model_edit(self):
        if self._workbench_input_closed or not self._validate_product_model():
            return
        save_recorded_data_to_json(product_model=self.lineedit_type.text(), scanner_barcode=self.lineedit_s_or_n.text())
        self.lineedit_type.setReadOnly(True)

    def eventFilter(self, obj, event):
        if self._workbench_input_closed:
            return super().eventFilter(obj, event)
        if obj is self.lineedit_type:
            if event.type() == QEvent.MouseButtonPress and obj.isReadOnly():
                obj.setReadOnly(False)
                obj.setFocus()
                obj.selectAll()
                return True
            return super().eventFilter(obj, event)
        if (event.type() == QEvent.KeyPress and self.barcode_scanner_box.isChecked()
                and QApplication.focusWidget() is self.lineedit_s_or_n
                and not self._can_commit_barcode()):
            # Keep Qt's focus, cursor, selection and copy behavior while the
            # current S/N is frozen. Editing shortcuts can have empty text too.
            if (event.key() in (Qt.Key_Tab, Qt.Key_Backtab, Qt.Key_Left, Qt.Key_Right,
                                Qt.Key_Up, Qt.Key_Down, Qt.Key_Home, Qt.Key_End)
                    or event.matches(QKeySequence.Copy)
                    or event.matches(QKeySequence.SelectAll)):
                return super().eventFilter(obj, event)
            return True
        if self._barcode_router.handle_keypress(obj, event) is True:
            return True
        return super().eventFilter(obj, event)

    def close_workbench_input(self):
        if self._workbench_input_closed:
            return
        self._workbench_input_closed = True
        self._barcode_debounce_timer.stop()
        self._clear_barcode_capture()
        QApplication.instance().removeEventFilter(self)
        if self._barcode_signals_bound:
            self.hw_manager.sig_barcode.disconnect(self.on_barcode_received)
            self._barcode_signals_bound = False
        self.lineedit_s_or_n.returnPressed.disconnect(self._barcode_router.on_barcode_return_pressed)
        self.lineedit_s_or_n.textChanged.disconnect(self._barcode_router.on_barcode_text_changed)
        self.lineedit_type.editingFinished.disconnect(self._finish_product_model_edit)
        self.barcode_scanner_box.clicked.disconnect(self.clicked_scanner)
        self.hw_manager.stop_scanner_listeners()

"""Qt presentation adapter for sequence trigger interactions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any
from weakref import ref

from PyQt5.QtCore import QSignalBlocker, Qt
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLineEdit,
    QPlainTextEdit,
    QTextEdit,
)

from ui.custom_ui_widget.widgets import MessageBox
from ui.sequence.sn_regex_manage_dialog import SnRegexManageDialog
from ui.tcp_config_dialog import TcpConfigDialog


_MODE_DISPLAY_NAMES = {
    "RECORD_ONLY": "仅录制",
    "PLAY_AND_RECORD": "播放录制",
    "IMPORT_AUDIO": "导入音频",
    "IMPORT_STIMULUS_AUDIO": "导入激励与音频",
}


class SequenceTriggerView:
    """Wrap widgets and dialogs; it contains no trigger admission decisions."""

    def __init__(
        self,
        *,
        parent: Any = None,
        serial_input: Any = None,
        scanner_checkbox: Any = None,
        product_input: Any = None,
        count_input: Any = None,
        prepare_for_continuous_scan: Callable[[], None] | None = None,
        regex_dialog_factory: Callable[..., Any] = SnRegexManageDialog,
        tcp_dialog_factory: Callable[..., Any] = TcpConfigDialog,
        message_box: Any = MessageBox,
    ) -> None:
        self.parent = parent
        self.serial_input = serial_input
        self.scanner_checkbox = scanner_checkbox
        self.product_input = product_input
        self.count_input = count_input
        self._prepare_for_continuous_scan = prepare_for_continuous_scan
        self._regex_dialog_factory = regex_dialog_factory
        self._tcp_dialog_factory = tcp_dialog_factory
        self._message_box = message_box
        self._regex_dialog = None
        self._tcp_dialog = None
        self._external_mode_warning_box = None
        self._external_mode_warning_token = None

    def is_scanner_checked(self) -> bool:
        checkbox = self.scanner_checkbox
        return bool(checkbox is not None and checkbox.isChecked())

    def is_serial_enabled(self) -> bool:
        widget = self.serial_input
        return bool(widget is not None and widget.isEnabled())

    def serial_text(self) -> str:
        widget = self.serial_input
        return "" if widget is None else widget.text()

    def set_serial_text(self, text: str) -> None:
        widget = self.serial_input
        if widget is None:
            return
        try:
            with QSignalBlocker(widget):
                widget.setText(text)
        except (RuntimeError, TypeError):
            widget.setText(text)

    def clear_serial_text(self) -> None:
        widget = self.serial_input
        if widget is None:
            return
        try:
            with QSignalBlocker(widget):
                widget.clear()
        except (RuntimeError, TypeError):
            widget.clear()

    def set_serial_enabled(self, enabled: bool) -> None:
        if self.serial_input is not None:
            self.serial_input.setEnabled(bool(enabled))

    def set_serial_read_only(self, read_only: bool) -> None:
        if self.serial_input is not None:
            self.serial_input.setReadOnly(bool(read_only))

    def set_scanner_enabled(self, enabled: bool) -> None:
        if self.scanner_checkbox is not None:
            self.scanner_checkbox.setEnabled(bool(enabled))

    def focus_serial_input(self, *, select_all: bool = False) -> None:
        widget = self.serial_input
        if widget is None:
            return
        widget.setFocus()
        if select_all:
            widget.selectAll()

    @staticmethod
    def focus_widget() -> Any:
        return QApplication.focusWidget()

    @staticmethod
    def is_protected_input_widget(widget: Any) -> bool:
        if widget is None:
            return False
        if isinstance(
            widget,
            (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QComboBox),
        ):
            return True
        return False

    def prepare_for_continuous_scan(self) -> None:
        if self._prepare_for_continuous_scan is not None:
            self._prepare_for_continuous_scan()

    def show_invalid_barcode(
        self, barcode: str, invalid_characters: tuple[str, ...]
    ) -> None:
        chars_display = "  ".join(
            repr(character) for character in sorted(set(invalid_characters))
        )
        self._message_box.warning(
            self.parent,
            "条形码包含特殊字符",
            "扫描到的内容包含无法用于文件名的特殊字符：\n\n"
            f"    {chars_display}\n\n"
            f"条形码内容：{barcode}\n\n"
            "请检查条形码内容，或使用不含特殊字符的条形码。\n"
            "关闭此窗口后可重新扫码。",
        )

    def show_regex_rejection(
        self,
        rule: Mapping[str, Any],
        sn_text: str,
        value_label: str,
        retry_hint: str,
    ) -> None:
        sn_display = sn_text if sn_text else "（空）"
        self._message_box.warning(
            self.parent,
            "SN 正则校验失败",
            "当前 SN 内容不符合已启用规则：\n\n"
            f"规则名称：{rule['name']}\n"
            f"规则表达式：{rule['pattern']}\n"
            f"{value_label}：{sn_display}\n\n"
            f"{retry_hint}",
        )

    def show_mode_rejection(self, trigger_source: str, mode: str | None) -> None:
        current_mode = self.mode_display_name(mode)
        text = (
            f"当前工作模式为 {current_mode}，不支持{trigger_source}启动工作流。\n"
            "仅【仅录制】和【播放录制】模式支持该功能。"
        )
        current = self._external_mode_warning_box
        if current is not None and current.isVisible():
            current.raise_()
            current.activateWindow()
            return
        if current is not None:
            self._external_mode_warning_box = None
            self._external_mode_warning_token = None
            current.reject()
        box = self._message_box(self.parent)
        box.setIcon(self._message_box.Warning)
        box.setWindowTitle("提示")
        box.setText(text)
        box.setStandardButtons(self._message_box.Ok)
        box.setWindowModality(Qt.WindowModal)
        box.setAttribute(Qt.WA_DeleteOnClose, True)
        box_ref = ref(box)
        token = object()
        lifecycle = {"resolved": False}

        def clear_reference(*_args: Any) -> None:
            if lifecycle["resolved"]:
                return
            lifecycle["resolved"] = True
            current_box = box_ref()
            if (
                self._external_mode_warning_token is token
                and self._external_mode_warning_box is current_box
            ):
                self._external_mode_warning_box = None
                self._external_mode_warning_token = None

        box.finished.connect(clear_reference)
        box.destroyed.connect(clear_reference)
        self._external_mode_warning_box = box
        self._external_mode_warning_token = token
        box.open()

    @staticmethod
    def mode_display_name(mode: str | None) -> str:
        return _MODE_DISPLAY_NAMES.get(mode, mode or "未配置")

    def show_busy_rejection(self, _trigger_source: str) -> None:
        # Existing behavior logs busy external triggers without adding a modal dialog.
        return None

    def show_workflow_rejection(self, _reason: str) -> None:
        # Workflow rejection is diagnostic unless a domain-specific presenter handles it.
        return None

    def present_tcp_state(self, enabled: bool) -> None:
        self.set_scanner_enabled(not enabled)
        self.set_serial_read_only(enabled)

    def open_regex_dialog(self) -> bool:
        current = self._regex_dialog
        if current is not None:
            raise_window = getattr(current, "raise_", None)
            if callable(raise_window):
                raise_window()
            activate = getattr(current, "activateWindow", None)
            if callable(activate):
                activate()
            return False
        try:
            dialog = self._regex_dialog_factory()
        except TypeError:
            dialog = self._regex_dialog_factory(self.parent)
        if self.parent is not None:
            set_parent = getattr(dialog, "setParent", None)
            if callable(set_parent):
                flags_getter = getattr(dialog, "windowFlags", None)
                flags = flags_getter() if callable(flags_getter) else Qt.Dialog
                try:
                    set_parent(self.parent, flags | Qt.Dialog)
                except TypeError:
                    set_parent(self.parent)
                    set_flags = getattr(dialog, "setWindowFlags", None)
                    if callable(set_flags):
                        set_flags(flags | Qt.Dialog)
        set_modality = getattr(dialog, "setWindowModality", None)
        if callable(set_modality):
            set_modality(Qt.WindowModal)
        set_attribute = getattr(dialog, "setAttribute", None)
        if callable(set_attribute):
            set_attribute(Qt.WA_DeleteOnClose, True)
        self._regex_dialog = dialog
        dialog_ref = ref(dialog)

        def clear_dialog(*_args: Any) -> None:
            current_dialog = dialog_ref()
            if current_dialog is not None and self._regex_dialog is current_dialog:
                self._regex_dialog = None

        for signal_name in ("finished", "destroyed"):
            signal = getattr(dialog, signal_name, None)
            connect = getattr(signal, "connect", None)
            if callable(connect):
                connect(clear_dialog)
        opener = getattr(dialog, "open", None)
        if callable(opener):
            opener()
        else:
            dialog.show()
        return True

    def close_dialogs(self) -> None:
        """Invalidate and close outstanding non-blocking trigger dialogs."""
        dialogs = (
            self._regex_dialog,
            self._tcp_dialog,
            self._external_mode_warning_box,
        )
        self._regex_dialog = None
        self._tcp_dialog = None
        self._external_mode_warning_box = None
        self._external_mode_warning_token = None
        for dialog in dialogs:
            if dialog is None:
                continue
            reject = getattr(dialog, "reject", None)
            close = getattr(dialog, "close", None)
            try:
                if callable(reject):
                    reject()
                elif callable(close):
                    close()
            except (RuntimeError, TypeError):
                continue

    def open_tcp_dialog(
        self,
        enabled: bool,
        host: str,
        port: Any,
        on_accepted: Callable[[tuple[bool, Any, Any]], None],
        on_rejected: Callable[[], None],
    ) -> bool:
        current = getattr(self, "_tcp_dialog", None)
        if current is not None:
            raise_window = getattr(current, "raise_", None)
            if callable(raise_window):
                raise_window()
            activate = getattr(current, "activateWindow", None)
            if callable(activate):
                activate()
            return False
        try:
            dialog = self._tcp_dialog_factory(enabled, host, port, self.parent)
        except TypeError:
            dialog = self._tcp_dialog_factory(enabled, host, port)
        window_flags = getattr(dialog, "windowFlags", None)
        set_window_flags = getattr(dialog, "setWindowFlags", None)
        if callable(window_flags) and callable(set_window_flags):
            set_window_flags(window_flags() | Qt.Dialog)
        set_window_modality = getattr(dialog, "setWindowModality", None)
        if callable(set_window_modality):
            set_window_modality(Qt.WindowModal)
        set_attribute = getattr(dialog, "setAttribute", None)
        if callable(set_attribute):
            set_attribute(Qt.WA_DeleteOnClose, True)
        self._tcp_dialog = dialog
        dialog_ref = ref(dialog)
        lifecycle = {"resolved": False}

        def clear_reference() -> None:
            current_dialog = dialog_ref()
            if (
                current_dialog is not None
                and getattr(self, "_tcp_dialog", None) is current_dialog
            ):
                self._tcp_dialog = None

        def resolve(*_args: Any) -> None:
            if lifecycle["resolved"]:
                return
            lifecycle["resolved"] = True
            clear_reference()
            current_dialog = dialog_ref()
            if current_dialog is None:
                on_rejected()
                return
            if bool(getattr(current_dialog, "clicked_ok_flag", False)):
                on_accepted(
                    (
                        bool(getattr(current_dialog, "is_tcp_flag", enabled)),
                        getattr(current_dialog, "ip", host),
                        getattr(current_dialog, "port", port),
                    )
                )
            else:
                on_rejected()

        def destroyed(*_args: Any) -> None:
            if lifecycle["resolved"]:
                return
            lifecycle["resolved"] = True
            clear_reference()
            on_rejected()

        for signal_name in ("accepted", "rejected", "finished"):
            signal = getattr(dialog, signal_name, None)
            connect = getattr(signal, "connect", None)
            if callable(connect):
                connect(resolve)
        destroyed_signal = getattr(dialog, "destroyed", None)
        destroyed_connect = getattr(destroyed_signal, "connect", None)
        if callable(destroyed_connect):
            destroyed_connect(destroyed)
        opener = getattr(dialog, "open", None)
        if callable(opener):
            opener()
        else:
            dialog.show()
        return True

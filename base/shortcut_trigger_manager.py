import time

import keyboard
from PyQt5.QtCore import QObject, pyqtSignal

from base.log_manager import LogManager


class ShortcutTriggerManager(QObject):
    """
    独立的全局快捷键管理器，用于通过键盘快捷键触发录音。

    与 UnifiedHardwareManager 中的光电热键独立：
    - 拥有独立的生命周期（start / stop）
    - 使用 keyboard.remove_hotkey() 精确移除，不影响其他热键
    - 内置按键重复过滤：防止键盘长按产生的高频重复事件
    """

    sig_triggered = pyqtSignal()

    HOTKEY = "F2"
    KEY_REPEAT_GUARD_SEC = 0.3  # 按键重复过滤（秒），仅防止键盘长按产生的高频重复事件

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = LogManager.set_log_handler("core")
        self._hotkey_handle = None
        self._last_trigger_time = 0.0

    def start(self):
        """注册全局快捷键"""
        if self._hotkey_handle is not None:
            return True

        try:
            self._hotkey_handle = keyboard.add_hotkey(
                self.HOTKEY,
                self._on_triggered,
                suppress=True
            )
        except BaseException as error:
            self._hotkey_handle = None
            try:
                self.logger.error(
                    "录音快捷键注册失败: "
                    f"{type(error).__name__}（快捷键: {self.HOTKEY}）"
                )
            except BaseException:
                pass
            return False
        try:
            self.logger.info(f"录音快捷键已注册: {self.HOTKEY}")
        except BaseException:
            pass
        return True

    def stop(self):
        """移除全局快捷键（精确移除，不影响其他热键）"""
        if self._hotkey_handle is None:
            return True

        handle = self._hotkey_handle
        try:
            keyboard.remove_hotkey(handle)
        except BaseException as error:
            self._hotkey_handle = handle
            try:
                self.logger.warning(
                    f"录音快捷键移除异常: {type(error).__name__}"
                )
            except BaseException:
                pass
            return False
        self._hotkey_handle = None
        try:
            self.logger.info(f"录音快捷键已移除: {self.HOTKEY}")
        except BaseException:
            pass
        return True

    def _on_triggered(self):
        """快捷键触发回调（在 keyboard 子线程中执行）

        仅过滤键盘长按产生的高频重复事件（~30ms 间隔），
        实际的录音互斥保护由主线程的 _record_workflow_busy 状态控制。
        """
        now = time.monotonic()
        if now - self._last_trigger_time < self.KEY_REPEAT_GUARD_SEC:
            return
        self._last_trigger_time = now
        self.logger.info(f"快捷键触发录音 ({self.HOTKEY})")
        self.sig_triggered.emit()

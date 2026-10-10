"""Legacy host for mode and mark actions; cumulative statistics are retired."""

from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox, QPushButton, QSizePolicy, QWidget

from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR


class SequenceCountBoard(QWidget):
    """Keep existing mode/mark callers without creating a statistics panel."""

    def __init__(self, analysis_config, parent=None):
        super().__init__(parent)
        self.analysis_config = analysis_config
        self.mode = ""
        self._test_available = True
        self._test_unavailable_reason = ""
        self._test_available_notice = ""
        self._mode_change_callbacks = []
        self.test_btn = QPushButton("测试", self)
        self.mark_btn = QPushButton("标记", self)
        self.ok_btn = QPushButton(" OK ", self)
        self.ng_btn = QPushButton(" NG ", self)
        self.set_btn()
        self.test_btn.clicked.connect(self.on_test_btn_clicked)
        self.mark_btn.clicked.connect(self.on_mark_btn_clicked)
        self.on_mark_btn_clicked()

    def set_btn(self):
        self.ok_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/green_circle.png"))
        self.ok_btn.setStyleSheet(ui_style_const.count_board_ok_button_style)
        self.ok_btn.setMinimumSize(148, 56)
        self.ok_btn.setMaximumSize(260, 56)
        self.ok_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.ok_btn.setIconSize(QSize(20, 20))
        self.ng_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/red_circle.png"))
        self.ng_btn.setStyleSheet(ui_style_const.count_board_ng_button_style)
        self.ng_btn.setMinimumSize(148, 56)
        self.ng_btn.setMaximumSize(260, 56)
        self.ng_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.ng_btn.setIconSize(QSize(20, 20))

    def on_test_btn_clicked(self):
        if not self._test_available:
            QMessageBox.information(self, "提示", self._test_unavailable_reason or "当前配置无法进入测试模式")
            self.on_mark_btn_clicked()
            return
        if self._test_available_notice:
            QMessageBox.information(
                self,
                "测试结果说明",
                self._test_available_notice,
            )
        self.test_btn.setStyleSheet(ui_style_const.count_board_mode_active_style)
        self.mark_btn.setStyleSheet(ui_style_const.count_board_mode_inactive_style)
        self.test_btn.setEnabled(False)
        self.mark_btn.setEnabled(True)
        self.mode = "test"
        self._notify_mode_state_changed()

    def on_mark_btn_clicked(self):
        self.mode = "mark"
        self.test_btn.setStyleSheet(ui_style_const.count_board_mode_inactive_style)
        self.mark_btn.setStyleSheet(ui_style_const.count_board_mode_active_style)
        self.mark_btn.setEnabled(False)
        self.test_btn.setEnabled(bool(self._test_available))
        self._notify_mode_state_changed()

    def set_test_available(self, available: bool, reason: str = "", *, preserve_mode=False):
        """
        Control whether test mode can be entered.
        """
        self._test_available = bool(available)
        self._test_unavailable_reason = (
            "" if self._test_available else str(reason or "")
        )
        self._test_available_notice = (
            str(reason or "") if self._test_available else ""
        )
        try:
            self.test_btn.setEnabled(bool(self._test_available) and self.mode != "test")
            self.test_btn.setToolTip(
                self._test_available_notice
                if self._test_available
                else self._test_unavailable_reason
            )
        except Exception:
            pass
        if (not self._test_available) and self.mode == "test" and not preserve_mode:
            self.on_mark_btn_clicked()
            return
        self._notify_mode_state_changed()

    def register_mode_change_callback(self, callback):
        if callable(callback):
            self._mode_change_callbacks.append(callback)

    def get_mode_state(self) -> dict:
        return {
            "mode": str(self.mode or ""),
            "test_available": bool(self._test_available),
            "test_unavailable_reason": str(self._test_unavailable_reason or ""),
        }

    def _notify_mode_state_changed(self):
        state = self.get_mode_state()
        for callback in list(self._mode_change_callbacks):
            try:
                callback(dict(state))
            except Exception:
                pass

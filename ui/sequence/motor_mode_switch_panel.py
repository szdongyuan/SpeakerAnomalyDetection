from PyQt5.QtWidgets import QHBoxLayout, QMessageBox, QPushButton, QWidget

from consts import ui_style_const


class MotorModeSwitchPanel(QWidget):
    def __init__(self, count_board, parent=None):
        super().__init__(parent)
        self.count_board = count_board
        self.test_btn = QPushButton("测试")
        self.mark_btn = QPushButton("标记")
        self._init_ui()
        self._connect_signals()
        self.sync_from_count_board()

    def _init_ui(self):
        self.test_btn.setStyleSheet(
            ui_style_const.motor_mode_switch_button_base_style + ui_style_const.motor_mode_switch_button_inactive_style
        )
        self.mark_btn.setStyleSheet(
            ui_style_const.motor_mode_switch_button_base_style + ui_style_const.motor_mode_switch_button_inactive_style
        )

        root_layout = QHBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(12)
        root_layout.addStretch()
        root_layout.addWidget(self.test_btn)
        root_layout.addWidget(self.mark_btn)
        root_layout.addStretch()
        self.setLayout(root_layout)

    def _connect_signals(self):
        self.test_btn.clicked.connect(self._on_test_clicked)
        self.mark_btn.clicked.connect(self._on_mark_clicked)
        if hasattr(self.count_board, "register_mode_change_callback"):
            self.count_board.register_mode_change_callback(self.sync_from_count_board)

    _MODE_SWITCH_DIALOG_STYLE = """
        QMessageBox {
            background-color: #f0f0f0;
        }
        QMessageBox QLabel {
            color: #1a1a1a;
            font-family: 'SimSun';
            font-size: 16px;
            background: transparent;
        }
        QMessageBox QPushButton {
            background-color: #e8eef6;
            color: #2c3e5a;
            font-family: 'SimSun';
            font-size: 15px;
            font-weight: bold;
            border: 1px solid #b0c4de;
            border-radius: 4px;
            min-width: 72px;
            min-height: 30px;
            padding: 4px 16px;
        }
        QMessageBox QPushButton:hover {
            background-color: #d6e2f0;
        }
        QMessageBox QPushButton:pressed {
            background-color: #c0d0e4;
        }
    """

    def _confirm_mode_switch(self, target_mode: str) -> bool:
        current_mode = str(getattr(self.count_board, "mode", "") or "")
        if current_mode == target_mode:
            return True
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("切换模式")
        msg_box.setText("切换模式将重新计算汇总信息，是否继续？")
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)
        msg_box.button(QMessageBox.Yes).setText("确定")
        msg_box.button(QMessageBox.No).setText("取消")
        msg_box.setStyleSheet(self._MODE_SWITCH_DIALOG_STYLE)
        return msg_box.exec_() == QMessageBox.Yes

    def _on_test_clicked(self):
        if not self._confirm_mode_switch("test"):
            return
        self.count_board.on_test_btn_clicked()
        self.sync_from_count_board()

    def _on_mark_clicked(self):
        if not self._confirm_mode_switch("mark"):
            return
        self.count_board.on_mark_btn_clicked()
        self.sync_from_count_board()

    def sync_from_count_board(self, _state=None):
        state = self.count_board.get_mode_state() if hasattr(self.count_board, "get_mode_state") else {}
        mode = str(state.get("mode", "") or "")
        test_available = bool(state.get("test_available", True))
        reason = str(state.get("test_unavailable_reason", "") or "")

        is_test_mode = mode == "test"
        is_mark_mode = mode == "mark"

        self.test_btn.setEnabled(test_available and not is_test_mode)
        self.mark_btn.setEnabled(not is_mark_mode)
        self.test_btn.setToolTip("" if test_available else reason)
        self.mark_btn.setToolTip("")

        test_style = ui_style_const.motor_mode_switch_button_active_style if is_test_mode else (
            ui_style_const.motor_mode_switch_button_inactive_style
        )
        mark_style = ui_style_const.motor_mode_switch_button_active_style if is_mark_mode else (
            ui_style_const.motor_mode_switch_button_inactive_style
        )
        self.test_btn.setStyleSheet(ui_style_const.motor_mode_switch_button_base_style + test_style)
        self.mark_btn.setStyleSheet(ui_style_const.motor_mode_switch_button_base_style + mark_style)

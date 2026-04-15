from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QWidget

from consts import ui_style_const


class MotorModeSwitchPanel(QWidget):
    def __init__(self, count_board, parent=None):
        super().__init__(parent)
        self.count_board = count_board
        self.mode_label = QLabel("模式")
        self.test_btn = QPushButton("测试")
        self.mark_btn = QPushButton("标记")
        self._init_ui()
        self._connect_signals()
        self.sync_from_count_board()

    def _init_ui(self):
        frame = QFrame()
        frame.setStyleSheet(ui_style_const.motor_mode_switch_panel_style)

        self.mode_label.setStyleSheet(ui_style_const.motor_mode_switch_label_style)
        self.test_btn.setStyleSheet(
            ui_style_const.motor_mode_switch_button_base_style + ui_style_const.motor_mode_switch_button_inactive_style
        )
        self.mark_btn.setStyleSheet(
            ui_style_const.motor_mode_switch_button_base_style + ui_style_const.motor_mode_switch_button_inactive_style
        )

        frame_layout = QHBoxLayout()
        frame_layout.setContentsMargins(12, 6, 12, 6)
        frame_layout.setSpacing(6)
        frame_layout.addWidget(self.mode_label)
        frame_layout.addStretch()
        frame_layout.addWidget(self.test_btn)
        frame_layout.addWidget(self.mark_btn)
        frame.setLayout(frame_layout)

        root_layout = QHBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(frame)
        self.setLayout(root_layout)

    def _connect_signals(self):
        self.test_btn.clicked.connect(self._on_test_clicked)
        self.mark_btn.clicked.connect(self._on_mark_clicked)
        if hasattr(self.count_board, "register_mode_change_callback"):
            self.count_board.register_mode_change_callback(self.sync_from_count_board)

    def _on_test_clicked(self):
        self.count_board.on_test_btn_clicked()
        self.sync_from_count_board()

    def _on_mark_clicked(self):
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

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.sequence.toolbar_eliding_widgets import (
    ElidingCheckBox, ElidingComboBox, ElidingLabel, ElidingLineEdit, ElidingSpinBox,
    ToolbarFieldLabel,
)
from ui.sequence.toolbar_row_layout import ToolbarRowLayout
from ui.sequence.toolbar_serial_button import ToolbarSerialButton


class RefreshBeforePopupComboBox(ElidingComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.before_show_popup = None

    def showPopup(self):
        if callable(self.before_show_popup):
            self.before_show_popup()
        super().showPopup()


class AnalysisStatusButton(QPushButton):
    """Toolbar analysis action with a compact, non-animated status badge."""

    STATE_IDLE = "idle"
    STATE_ANALYZING = "analyzing"
    STATE_COMPLETED = "completed"
    STATE_FAILED = "failed"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._analysis_state = self.STATE_IDLE
        self._status_badge_width = 18
        self._status_badge = ElidingLabel(self)
        self._status_badge.setObjectName("analysisStatusBadge")
        self._status_badge.setAlignment(Qt.AlignCenter)
        self._status_badge.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._status_badge.hide()

    @property
    def analysis_state(self):
        return self._analysis_state

    @property
    def status_badge(self):
        return self._status_badge

    def set_idle(self):
        self._analysis_state = self.STATE_IDLE
        self._status_badge.hide()
        self._set_status_tooltip("分析")

    def set_analyzing(self, completed, total, source_label=""):
        completed = max(0, int(completed))
        total = max(1, int(total))
        completed = min(completed, total)
        source_text = str(source_label or "所选档位").strip()
        self._analysis_state = self.STATE_ANALYZING
        self._show_badge(
            f"{completed}/{total}",
            ui_style_const.COLOR_PRIMARY,
            minimum_width=34,
        )
        self._set_status_tooltip(
            f"正在分析 {source_text}：{completed}/{total}"
        )

    def set_completed(self, source_label=""):
        source_text = str(source_label or "所选档位").strip()
        self._analysis_state = self.STATE_COMPLETED
        self._show_badge("✓", ui_style_const.COLOR_OK, minimum_width=18)
        self._set_status_tooltip(
            f"{source_text} 手动分析完成，点击查看"
        )

    def set_failed(self, source_label=""):
        source_text = str(source_label or "所选档位").strip()
        self._analysis_state = self.STATE_FAILED
        self._show_badge("!", ui_style_const.COLOR_NG, minimum_width=18)
        self._set_status_tooltip(
            f"{source_text} 手动分析失败，点击查看原因"
        )

    def _show_badge(self, text, background_color, *, minimum_width):
        self._status_badge.setText(text)
        self._status_badge.setStyleSheet(
            f"""
            QLabel#analysisStatusBadge {{
                color: #FFFFFF;
                background-color: {background_color};
                border: none;
                border-radius: 8px;
                padding: 0 3px;
                font-family: {ui_style_const.MAIN_UI_SMALL_FONT_FAMILY};
                font-size: 11px;
                font-weight: 600;
            }}
            """
        )
        self._status_badge_width = max(minimum_width, self._status_badge.sizeHint().width())
        self._position_status_badge()
        self._status_badge.show()
        self._status_badge.raise_()

    def _set_status_tooltip(self, text):
        self.setToolTip(text)
        self.setAccessibleDescription(text)

    def _position_status_badge(self):
        self._status_badge.setFixedSize(
            min(self._status_badge_width, self.width() - 6), 18,
        )
        self._status_badge.move(
            max(2, self.width() - self._status_badge.width() - 3),
            2,
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_status_badge()


class SequenceToolsBar(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sequenceToolsBar")
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.player_btn = QPushButton()
        self.replayer_btn = QPushButton()
        self.serial_trigger_btn = ToolbarSerialButton()
        self.data_btn = AnalysisStatusButton()
        self.using_file_combobox = RefreshBeforePopupComboBox()
        self.sample_number_lineedit = ElidingLineEdit()
        self.current_round_spinbox = ElidingSpinBox()
        # 历史模式同步仍依赖该对象；仅从操作界面隐藏。
        self.condition_mode_combobox = QComboBox(self)
        self.condition_mode_combobox.addItems(["测试", "标记"])
        self.condition_mode_combobox.hide()
        self.lineedit_type = ElidingLineEdit()
        self.lineedit_count = QLineEdit()
        self.lineedit_s_or_n = ElidingLineEdit()
        self.barcode_scanner_box = ElidingCheckBox("S/N：")
        self.serial_trigger_code_label = QLabel("最近接收: -")

        self.init_ui()

    def init_ui(self):
        self.set_play_btn()
        self.set_replay_btn()
        self.set_data_btn()
        self.set_serial_trigger_btn()
        tools_layout = self.create_tools_layout()

        self.setLayout(tools_layout)
        self.setStyleSheet(ui_style_const.toolbar_container_style)

    def create_tools_layout(self):
        line_top = self._create_separator(QFrame.HLine)
        line_bottom = self._create_separator(QFrame.HLine)

        layout = self.create_mainly_layout()

        tools_layout = QVBoxLayout()
        tools_layout.addWidget(line_top)
        tools_layout.addLayout(layout)
        tools_layout.addWidget(line_bottom)

        tools_layout.setSpacing(0)
        tools_layout.setContentsMargins(0, 0, 0, 0)

        return tools_layout

    def create_mainly_layout(self):
        layout = ToolbarRowLayout()
        for button in (self.player_btn, self.data_btn):
            layout.add_control(button, "action", 48, 120)
        layout.add_control(self.serial_trigger_btn, "serial", 48, 124)
        layout.add_control(self._create_separator(QFrame.VLine), "fixed", 1, 1)
        self._configure_fields()
        for text, short_text, widget, preferred in (
            ("型号：", "型号：", self.lineedit_type, 160),
            ("使用配置：", "配置：", self.using_file_combobox, 200),
            ("样本编号：", "样本：", self.sample_number_lineedit, 100),
            ("当前测试轮次：", "轮次：", self.current_round_spinbox, 60),
        ):
            label = ToolbarFieldLabel(text, short_text)
            label.setFixedHeight(40)
            label.setStyleSheet(ui_style_const.toolbar_field_label_style)
            label.setBuddy(widget)
            layout.add_control(label, "label")
            layout.add_control(widget, "input", 55, preferred)
            if widget is self.current_round_spinbox:
                layout.add_control(self.reset_round_button, "fixed", 52, 52)
            layout.add_control(self._create_separator(QFrame.VLine), "fixed", 1, 1)
        layout.add_control(self.barcode_scanner_box, "switch")
        layout.add_control(self.lineedit_s_or_n, "sn", 125, 320)
        layout.add_control(self._create_separator(QFrame.VLine), "fixed", 1, 1)
        return layout

    def set_play_btn(self):
        self._configure_icon_button(
            self.player_btn,
            "开始录制",
            "ui/ui_pic/sequence_pic/play.png",
            QSize(35, 35),
        )

    def set_replay_btn(self):
        self._configure_icon_button(
            self.replayer_btn,
            "重新录制",
            "ui/ui_pic/sequence_pic/replay.png",
            QSize(30, 30),
        )
        self.replayer_btn.setDisabled(True)
        # 当前版本不支持重录；保留按钮对象供既有状态同步代码兼容使用。
        self.replayer_btn.hide()

    def set_data_btn(self):
        self._configure_icon_button(
            self.data_btn,
            "分析",
            "ui/ui_pic/sequence_pic/data.png",
            QSize(35, 35),
        )
        self.data_btn.setEnabled(False)

    def set_serial_trigger_btn(self):
        self._configure_icon_button(
            self.serial_trigger_btn,
            "串口离散输入触发配置",
            "ui/ui_pic/sequence_pic/new_com.png",
            QSize(26, 26),
        )
        self._add_icon_trailing_space(self.serial_trigger_btn, QSize(26, 26), 6)
        self.serial_trigger_btn.setFixedSize(124, 40)
        self.serial_trigger_btn.setText("未连接")
        self.serial_trigger_btn.set_connection_state(False, False)
        hint = "串口离散输入触发配置\n未连接"
        self.serial_trigger_btn.setToolTip(hint)
        self.serial_trigger_btn.setAccessibleName("串口离散输入触发配置，未连接")
        self.serial_trigger_btn.setAccessibleDescription(hint)
        self.serial_trigger_btn.setStyleSheet(
            ui_style_const.serial_trigger_button_base_style
            + ui_style_const.serial_trigger_button_inactive_style
        )

    def _configure_fields(self):
        for editor in (self.lineedit_type, self.sample_number_lineedit,
                       self.lineedit_s_or_n):
            editor.setFixedHeight(35)
            editor.setMinimumWidth(55)
            editor.setAlignment(Qt.AlignCenter)
            editor.setStyleSheet(ui_style_const.toolbar_input_style)
        self.lineedit_type.setAccessibleName("型号")
        self.lineedit_s_or_n.setAccessibleName("S/N")
        self.using_file_combobox.setMinimumWidth(55)
        self.using_file_combobox.setFixedHeight(35)
        self.using_file_combobox.setAccessibleName("使用配置")
        self.using_file_combobox.setStyleSheet(ui_style_const.toolbar_combobox_style)
        self.sample_number_lineedit.setObjectName("sampleNumberLineEdit")
        self.sample_number_lineedit.setToolTip("请输入样本编号")
        self.sample_number_lineedit.setAccessibleName("样本编号")
        self.sample_number_lineedit.setAccessibleDescription("请输入样本编号")
        self.current_round_spinbox.setObjectName("currentRoundSpinBox")
        self.current_round_spinbox.setRange(1, 9999)
        self.current_round_spinbox.setValue(1)
        self.current_round_spinbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.current_round_spinbox.setMinimumWidth(55)
        self.current_round_spinbox.setFixedHeight(35)
        self.current_round_spinbox.setAlignment(Qt.AlignCenter)
        self.current_round_spinbox.setToolTip("请输入当前测试轮次")
        self.current_round_spinbox.setAccessibleName("当前测试轮次")
        self.current_round_spinbox.setAccessibleDescription("请输入当前测试轮次")
        self.current_round_spinbox.setStyleSheet(ui_style_const.toolbar_spinbox_style)
        self.reset_round_button = QPushButton("重置")
        self.reset_round_button.setObjectName("resetCurrentRoundButton")
        self.reset_round_button.setFixedSize(52, 35)
        self.reset_round_button.setToolTip("重置当前轮次")
        self.reset_round_button.setAccessibleName("重置当前轮次")
        self.reset_round_button.setAutoDefault(False)
        self.reset_round_button.setStyleSheet(
            ui_style_const.qpushbutton_style
            + "QPushButton#resetCurrentRoundButton { font-size: 14px; padding: 4px 6px; }"
        )
        self.barcode_scanner_box.setChecked(False)
        self.barcode_scanner_box.setFixedHeight(40)
        self.barcode_scanner_box.setStyleSheet(ui_style_const.toolbar_checkbox_style)
        self.lineedit_s_or_n.setDisabled(True)
        self.lineedit_s_or_n.setMinimumWidth(125)
        self.lineedit_s_or_n.setMaximumWidth(320)
        self.lineedit_s_or_n.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    @staticmethod
    def _create_separator(shape):
        line = QFrame()
        line.setFrameShape(shape)
        line.setStyleSheet(ui_style_const.toolbar_separator_style)
        if shape == QFrame.HLine:
            line.setFixedHeight(1)
        else:
            line.setFixedWidth(1)
        return line

    @staticmethod
    def _configure_icon_button(button, tooltip, icon_path, icon_size):
        button.setFixedSize(48, 40)
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setAccessibleDescription(tooltip)
        button.setStyleSheet(ui_style_const.toolbar_button_style)
        button.setIcon(QIcon(DEFAULT_DIR + icon_path))
        button.setIconSize(icon_size)

    @staticmethod
    def _add_icon_trailing_space(button, icon_size, spacing):
        source_pixmap = button.icon().pixmap(icon_size)
        padded_pixmap = QPixmap(icon_size.width() + spacing, icon_size.height())
        padded_pixmap.fill(Qt.transparent)

        painter = QPainter(padded_pixmap)
        painter.drawPixmap(0, 0, source_pixmap)
        painter.end()

        button.setIcon(QIcon(padded_pixmap))
        button.setIconSize(padded_pixmap.size())

    def mouseMoveEvent(self, a0):
        self.setCursor(Qt.ArrowCursor)


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication([])
    window = SequenceToolsBar()
    window.show()
    app.exec_()

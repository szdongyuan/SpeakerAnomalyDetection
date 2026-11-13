from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.sequence.toolbar_eliding_widgets import (
    ElidingCheckBox, ElidingComboBox, ElidingLineEdit,
    ModelLineEdit, ToolbarFieldLabel,
)
from ui.sequence.toolbar_row_layout import ToolbarRowLayout


class RefreshBeforePopupComboBox(ElidingComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.before_show_popup = None

    def showPopup(self):
        if callable(self.before_show_popup):
            self.before_show_popup()
        super().showPopup()


class SequenceToolsBar(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sequenceToolsBar")
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.player_btn = QPushButton()
        self.using_file_combobox = RefreshBeforePopupComboBox()
        self.sample_number_lineedit = ElidingLineEdit()
        self.lineedit_type = ModelLineEdit()
        self.lineedit_s_or_n = ElidingLineEdit()
        self.barcode_scanner_box = ElidingCheckBox("S/N：")

        self.init_ui()

    def init_ui(self):
        self.set_play_btn()
        tools_layout = self.create_tools_layout()

        self.setLayout(tools_layout)
        self.setStyleSheet(
            ui_style_const.toolbar_container_style + ui_style_const.main_window_tooltip_style
        )

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
        buttons = (self.player_btn,)
        for button in buttons:
            layout.add_control(button, "action", ui_style_const.TOOLBAR_ICON_BUTTON_WIDTH, 120)
        layout.add_control(self._create_separator(QFrame.VLine), "fixed", 1, 1)
        self._configure_fields()
        for text, short_text, widget, preferred in (
            ("型号：", "型号：", self.lineedit_type, 160),
            ("使用配置：", "配置：", self.using_file_combobox, 200),
            ("轮次：", "轮次：", self.sample_number_lineedit, 100),
        ):
            label = ToolbarFieldLabel(text, short_text)
            label.setFixedHeight(40)
            label.setStyleSheet(ui_style_const.toolbar_field_label_style)
            label.setBuddy(widget)
            layout.add_control(label, "label")
            layout.add_control(widget, "input", 55, preferred)
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
        self._recording_start_icon = self.player_btn.icon()
        pause_pixmap = QPixmap(DEFAULT_DIR + "ui/ui_pic/sequence_pic/pause.png")
        self._recording_busy_icon = QIcon(pause_pixmap)
        self._recording_busy_icon.addPixmap(pause_pixmap, QIcon.Disabled)

    def set_recording_busy(self, busy):
        self.player_btn.setIcon(self._recording_busy_icon if busy else self._recording_start_icon)

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
        self.sample_number_lineedit.setReadOnly(True)
        self.sample_number_lineedit.setToolTip("每日共用轮次，自动更新")
        self.sample_number_lineedit.setAccessibleName("轮次")
        self.sample_number_lineedit.setAccessibleDescription("每日共用轮次，自动更新")
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
        button.setFixedSize(ui_style_const.TOOLBAR_ICON_BUTTON_WIDTH,
                            ui_style_const.TOOLBAR_ICON_BUTTON_HEIGHT)
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setAccessibleDescription(tooltip)
        button.setStyleSheet(ui_style_const.main_window_toolbar_button_style)
        button.setIcon(QIcon(DEFAULT_DIR + icon_path))
        button.setIconSize(icon_size)

    def mouseMoveEvent(self, a0):
        self.setCursor(Qt.ArrowCursor)


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication([])
    window = SequenceToolsBar()
    window.show()
    app.exec_()

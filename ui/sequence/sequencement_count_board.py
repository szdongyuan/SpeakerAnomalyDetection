from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QFrame, QWidget, QStackedWidget, QSizePolicy

from consts import ui_style_const
from ui.custom_ui_widget.widgets import PushButton, LineEdit, Label, MarkPushButton, MessageBox
from ui.ui_src import ui_resources


class SequenceCountBoard(QWidget):
    collapsed_changed = pyqtSignal(bool)

    def __init__(self, analysis_config, parent=None):
        super(SequenceCountBoard, self).__init__(parent)

        self.btn_height = ui_style_const.scale_size_px(80)
        self.btn_width = ui_style_const.scale_size_px(180)
        self._minimum_content_width = self.btn_width + ui_style_const.scale_size_px(20)
        self.label_width = ui_style_const.scale_size_px(100)
        self.lineedit_width = max(ui_style_const.scale_size_px(80), self._minimum_content_width - self.label_width)
        self.lineedit_height = ui_style_const.scale_size_px(35)
        self.mode_label_width = ui_style_const.scale_size_px(60)
        self.mode_button_width = max(
            ui_style_const.scale_size_px(70),
            int((self._minimum_content_width - self.mode_label_width) / 2),
        )
        self.icon_size = ui_style_const.scale_size_px(24)

        self.analysis_config = analysis_config
        self.mode = str()
        self._test_available = True
        self._test_unavailable_reason = ""
        self._collapsed = False
        self._compact_resize_enabled = False
        self._collapse_bar_width = ui_style_const.scale_size_px(36)
        self._splitter_gap_width = ui_style_const.scale_size_px(5)

        self.test_btn = PushButton("测试")
        self.mark_btn = PushButton("标记")
        self.test_btn.setObjectName("TestBtn")
        self.mark_btn.setObjectName("MarkBtn")
        self.total_line_edit = LineEdit()
        self.ok_line_edit = LineEdit()
        self.ng_line_edit = LineEdit()
        self.yield_line_edit = LineEdit()
        self.datatime_line_edit = LineEdit()
        self.mark_total_edit = LineEdit("0")
        self.mark_ok_edit = LineEdit("0")
        self.mark_ng_edit = LineEdit("0")
        self.ok_btn = MarkPushButton(" OK ")
        self.ng_btn = MarkPushButton(" NG ")
        self.ok_btn.setObjectName("OkBtn")
        self.ng_btn.setObjectName("NgBtn")
        self.reset_btn = PushButton("重置统计")
        self.reset_btn.setObjectName("ResrtBtn")
        self.collapse_toggle_btn = PushButton("<<")
        self.collapse_toggle_btn.setObjectName("CollapseToggleBtn")
        self.collapse_toggle_btn.setFixedWidth(self._collapse_bar_width)
        self.collapse_toggle_btn.setToolTip("折叠计数板")
        self.collapse_toggle_btn.setStyleSheet(
            """
            #CollapseToggleBtn {
                background-color: transparent;
                border: none;
            }
            #CollapseToggleBtn:hover {
                background-color: transparent;
            }
            #CollapseToggleBtn:pressed {
                background-color: transparent;
            }
            """
        )
        self.collapse_toggle_btn.clicked.connect(self.toggle_collapsed)

        self.set_lineedit()
        self.set_btn()

        self.init_ui()

        self.on_mark_btn_clicked()
        # Do NOT force switch to test mode during init.

    def init_ui(self):
        self.setObjectName("SequenceCountBoard")
        mode_btn_layout = self.create_mode_btn_layout()

        separator_line = QFrame()
        separator_line.setFrameShape(QFrame.HLine)
        separator_line.setFrameShadow(QFrame.Sunken)
        separator_line.setFixedHeight(2)

        test_widget = self.set_test_widget()
        mark_widget = self.set_mark_widget()

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(test_widget)
        self.stacked_widget.addWidget(mark_widget)

        layout = QVBoxLayout()
        layout.addLayout(mode_btn_layout)
        layout.addWidget(separator_line)
        layout.addWidget(self.stacked_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.content_widget = QWidget()
        self.content_widget.setLayout(layout)
        self.content_widget.setObjectName("SequenceCountBoardContent")
        self.content_widget.setMinimumWidth(0)
        self.content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        collapse_bar_layout = QVBoxLayout()
        collapse_bar_layout.addWidget(self.collapse_toggle_btn)
        collapse_bar_layout.addStretch()
        collapse_bar_layout.setContentsMargins(0, 0, 0, 0)
        collapse_bar_layout.setSpacing(0)

        collapse_bar_widget = QWidget()
        collapse_bar_widget.setObjectName("SequenceCountBoardCollapseBar")
        collapse_bar_widget.setFixedWidth(self._collapse_bar_width)
        collapse_bar_widget.setLayout(collapse_bar_layout)

        layout_main = QHBoxLayout()
        layout_main.addWidget(self.content_widget, stretch=1)
        layout_main.addWidget(collapse_bar_widget)
        layout_main.setSpacing(0)
        layout_main.setContentsMargins(0, 0, self._splitter_gap_width, 0)

        self.setLayout(layout_main)
        self._refresh_minimum_width()

    def is_collapsed(self):
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        collapsed = bool(collapsed)
        if self._collapsed == collapsed:
            return

        self._collapsed = collapsed
        self.content_widget.setVisible(not collapsed)
        self.collapse_toggle_btn.setText(">>" if collapsed else "<<")
        self.collapse_toggle_btn.setToolTip("展开计数板" if collapsed else "折叠计数板")
        self._refresh_minimum_width()
        self.updateGeometry()
        self.collapsed_changed.emit(collapsed)

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self.is_collapsed())

    def set_compact_resize_enabled(self, enabled: bool) -> None:
        self._compact_resize_enabled = bool(enabled)
        self._refresh_minimum_width()
        self.updateGeometry()

    def expanded_width_hint(self) -> int:
        layout = self.layout()
        margins = layout.contentsMargins()
        spacing = max(0, layout.spacing())
        expanded_layout_width = (
            max(self._minimum_content_width, self.content_widget.minimumSizeHint().width())
            + self._collapse_bar_width
            + spacing
            + margins.left()
            + margins.right()
        )
        return expanded_layout_width

    def collapsed_width_hint(self) -> int:
        return self._collapse_bar_width + self._splitter_gap_width

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        if self._collapsed or self._compact_resize_enabled:
            hint.setWidth(self.collapsed_width_hint())
        else:
            hint.setWidth(self.expanded_width_hint())
        return hint

    def _refresh_minimum_width(self) -> None:
        if self._collapsed or self._compact_resize_enabled:
            self.setMinimumWidth(self.collapsed_width_hint())
        else:
            self.setMinimumWidth(self.expanded_width_hint())

    def create_horizontal_layout(self, label_str, item):
        label = Label(label_str)
        label.setFixedWidth(self.label_width)

        layout = QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(item, stretch=1)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        return layout

    def set_lineedit(self):
        self.total_line_edit.setAlignment(Qt.AlignCenter)
        self.ok_line_edit.setAlignment(Qt.AlignCenter)
        self.ng_line_edit.setAlignment(Qt.AlignCenter)
        self.yield_line_edit.setAlignment(Qt.AlignCenter)
        self.datatime_line_edit.setAlignment(Qt.AlignCenter)
        self.mark_total_edit.setAlignment(Qt.AlignCenter)
        self.mark_ok_edit.setAlignment(Qt.AlignCenter)
        self.mark_ng_edit.setAlignment(Qt.AlignCenter)

        self.total_line_edit.setDisabled(True)
        self.ok_line_edit.setDisabled(True)
        self.ng_line_edit.setDisabled(True)
        self.yield_line_edit.setDisabled(True)
        self.datatime_line_edit.setDisabled(True)
        self.mark_total_edit.setDisabled(True)
        self.mark_ok_edit.setDisabled(True)
        self.mark_ng_edit.setDisabled(True)

        line_edits = (
            self.total_line_edit,
            self.ok_line_edit,
            self.ng_line_edit,
            self.yield_line_edit,
            self.datatime_line_edit,
            self.mark_total_edit,
            self.mark_ok_edit,
            self.mark_ng_edit,
        )
        for line_edit in line_edits:
            line_edit.setFixedHeight(self.lineedit_height)
            line_edit.setMinimumWidth(self.lineedit_width)
            line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_btn(self):
        self.ok_btn.setIcon(QIcon(":/ui/icon/green_circle.png"))
        self.ok_btn.setFixedSize(self.btn_width, self.btn_height)
        self.ok_btn.setIconSize(QSize(self.icon_size, self.icon_size))
        self.ng_btn.setIcon(QIcon(":/ui/icon/red_circle.png"))
        self.ng_btn.setFixedSize(self.btn_width, self.btn_height)
        self.ng_btn.setIconSize(QSize(self.icon_size, self.icon_size))

    def create_mode_btn_layout(self):
        mode_label = Label("模式：")
        mode_label.setFixedWidth(self.mode_label_width)
        self.test_btn.setFixedSize(self.mode_button_width, self.lineedit_height)
        self.mark_btn.setFixedSize(self.mode_button_width, self.lineedit_height)
        self.test_btn.clicked.connect(self.on_test_btn_clicked)
        self.mark_btn.clicked.connect(self.on_mark_btn_clicked)

        model_button_layout = QHBoxLayout()
        model_button_layout.addWidget(mode_label)
        model_button_layout.addStretch()
        model_button_layout.addWidget(self.test_btn)
        model_button_layout.addWidget(self.mark_btn)
        model_button_layout.setSpacing(0)
        model_button_layout.setContentsMargins(0, 0, 0, 0)

        return model_button_layout

    def set_test_widget(self):
        total_layout = self.create_horizontal_layout("总    数：", self.total_line_edit)
        ok_layout = self.create_horizontal_layout("OK    数：", self.ok_line_edit)
        ng_layout = self.create_horizontal_layout("NG    数：", self.ng_line_edit)
        yield_layout = self.create_horizontal_layout("合 格 率：", self.yield_line_edit)
        datatime_layout = self.create_horizontal_layout("录制日期：", self.datatime_line_edit)
        reset_btn_layout = QHBoxLayout()
        reset_btn_layout.addStretch()
        reset_btn_layout.addWidget(self.reset_btn)
        reset_btn_layout.addStretch()

        test_layout = QVBoxLayout()
        test_layout.setSpacing(ui_style_const.scale_size_px(7))
        test_layout.setContentsMargins(0, 0, 0, 0)
        test_layout.addLayout(total_layout)
        test_layout.addLayout(ok_layout)
        test_layout.addLayout(ng_layout)
        test_layout.addLayout(yield_layout)
        test_layout.addLayout(datatime_layout)
        test_layout.addLayout(reset_btn_layout)
        test_layout.addStretch()

        test_widget = QWidget()
        test_widget.setLayout(test_layout)

        return test_widget

    def set_mark_widget(self):
        total_layout = self.create_horizontal_layout("总    数：", self.mark_total_edit)
        ok_layout = self.create_horizontal_layout("OK    数：", self.mark_ok_edit)
        ng_layout = self.create_horizontal_layout("NG    数：", self.mark_ng_edit)
        ok_btn_layout = QHBoxLayout()
        ok_btn_layout.addStretch()
        ok_btn_layout.addWidget(self.ok_btn)
        ok_btn_layout.addStretch()
        ng_btn_layout = QHBoxLayout()
        ng_btn_layout.addStretch()
        ng_btn_layout.addWidget(self.ng_btn)
        ng_btn_layout.addStretch()

        mark_layout = QVBoxLayout()
        mark_layout.setSpacing(ui_style_const.scale_size_px(7))
        mark_layout.setContentsMargins(0, 0, 0, 0)
        mark_layout.addLayout(total_layout)
        mark_layout.addLayout(ok_layout)
        mark_layout.addLayout(ng_layout)
        mark_layout.addSpacing(10)
        mark_layout.addLayout(ok_btn_layout)
        mark_layout.addLayout(ng_btn_layout)
        mark_layout.addStretch()

        mark_widget = QWidget()
        mark_widget.setLayout(mark_layout)

        return mark_widget

    def on_test_btn_clicked(self):
        if not self._test_available:
            MessageBox.information(self, "提示", self._test_unavailable_reason or "当前配置无法进入测试模式")
            self.on_mark_btn_clicked()
            return
        self.test_btn.setEnabled(False)
        self.mark_btn.setEnabled(True)
        self.stacked_widget.setCurrentIndex(0)
        self.mode = "test"

    def on_mark_btn_clicked(self):
        self.stacked_widget.setCurrentIndex(1)
        self.mode = "mark"
        self.mark_btn.setEnabled(False)
        self.test_btn.setEnabled(bool(self._test_available))

    def bind_mark_action(self, callback):
        """Route the raw mark click through its owning Recording controller."""
        if not callable(callback):
            raise TypeError("mark action callback must be callable")
        previous = getattr(self, "_mark_action_callback", None)
        try:
            self.mark_btn.clicked.disconnect(
                previous if callable(previous) else self.on_mark_btn_clicked
            )
        except (RuntimeError, TypeError):
            pass
        self._mark_action_callback = callback
        self.mark_btn.clicked.connect(callback)

    def set_test_available(self, available: bool, reason: str = ""):
        """
        Control whether test mode can be entered.
        """
        self._test_available = bool(available)
        self._test_unavailable_reason = str(reason or "")
        try:
            self.test_btn.setProperty("available", "true" if self._test_available else "false")
            self.test_btn.style().unpolish(self.test_btn)
            self.test_btn.style().polish(self.test_btn)
            self.test_btn.update()
            self.test_btn.setEnabled(bool(self._test_available) and self.mode != "test")
            self.test_btn.setToolTip(self._test_unavailable_reason if not self._test_available else "")
        except Exception:
            pass
        if (not self._test_available) and self.mode == "test":
            self.on_mark_btn_clicked()

import copy
import ntpath
import os

from PyQt5.QtCore import QEvent, QSize, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from base.product_test_project_config import (
    ProductTestProjectConfigManager,
    normalize_trigger_state,
)
from consts import error_code, ui_style_const
from consts.product_test_project_consts import (
    CONDITION_NAME_KEY,
    GROUP_NAME_KEY,
    PROJECT_NAME_KEY,
    REGISTRY_ACTIVE_FILE_KEY,
    REGISTRY_CONFIGS_KEY,
    REGISTRY_FILE_KEY,
    RESULT_ROOT_DIRECTORY_KEY,
    TEST_CONDITIONS_KEY,
    TEST_GROUPS_KEY,
    TEST_QUEUE_KEY,
    TRIGGER_STATE_KEY,
)
from consts.running_consts import DEFAULT_DIR
from ui.config_dialog_base import ConfigDialogBase


NO_QUEUE_TEXT = "暂无可用测试队列"
UNAVAILABLE_QUEUE_SUFFIX = "（不可用）"


class _RefreshingQueueComboBox(QComboBox):
    def __init__(self, refresh_callback, parent=None):
        super().__init__(parent)
        self._refresh_callback = refresh_callback

    def showPopup(self):
        self._refresh_callback()
        super().showPopup()


class _PortSelectorBar(QWidget):
    BUTTON_WIDTH = 112
    BUTTON_HEIGHT = 32
    BUTTON_TEXT_MARGIN = 18

    currentChanged = pyqtSignal(int)
    scroll_requested = pyqtSignal(int)
    tabTextEdited = pyqtSignal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buttons = []
        self._current_index = -1
        self._editing_index = -1
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self._layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._name_editor = QLineEdit(self)
        self._name_editor.setObjectName("productProjectPortNameEditor")
        self._name_editor.setFixedSize(self.BUTTON_WIDTH, self.BUTTON_HEIGHT)
        self._name_editor.editingFinished.connect(self._finish_inline_rename)
        self._name_editor.installEventFilter(self)
        self._name_editor.hide()

    def count(self):
        return len(self._buttons)

    def addTab(self, text):
        button = QPushButton(self)
        button.setObjectName("productProjectPortButton")
        button.setCheckable(True)
        button.setAutoDefault(False)
        button.setFixedSize(self.BUTTON_WIDTH, self.BUTTON_HEIGHT)
        self._set_button_text(button, text)
        button.installEventFilter(self)
        button.clicked.connect(
            lambda _checked=False, target=button: self._select_button(target)
        )
        self._buttons.append(button)
        self._layout.addWidget(button)
        if self._current_index < 0:
            self.setCurrentIndex(0)
        return len(self._buttons) - 1

    def removeTab(self, index):
        if index < 0 or index >= len(self._buttons):
            return
        button = self._buttons.pop(index)
        self._layout.removeWidget(button)
        button.deleteLater()
        if not self._buttons:
            self.setCurrentIndex(-1)
        elif self._current_index >= len(self._buttons):
            self.setCurrentIndex(len(self._buttons) - 1)
        else:
            self._sync_checked_state()

    def currentIndex(self):
        return self._current_index

    def setCurrentIndex(self, index):
        target_index = index if 0 <= index < len(self._buttons) else -1
        changed = target_index != self._current_index
        self._current_index = target_index
        self._sync_checked_state()
        if changed:
            self.currentChanged.emit(target_index)

    def tabText(self, index):
        if 0 <= index < len(self._buttons):
            return str(self._buttons[index].property("fullText") or "")
        return ""

    def setTabText(self, index, text):
        if 0 <= index < len(self._buttons):
            self._set_button_text(self._buttons[index], text)

    def tabRect(self, index):
        if 0 <= index < len(self._buttons):
            return self._buttons[index].geometry()
        return self.rect()

    def _select_button(self, button):
        try:
            index = self._buttons.index(button)
        except ValueError:
            return
        self.setCurrentIndex(index)

    def _set_button_text(self, button, text):
        full_text = str(text or "")
        available_width = self.BUTTON_WIDTH - self.BUTTON_TEXT_MARGIN
        button.setProperty("fullText", full_text)
        button.setText(
            button.fontMetrics().elidedText(
                full_text,
                Qt.ElideRight,
                available_width,
            )
        )
        button.setToolTip(full_text)

    def eventFilter(self, watched, event):
        if (
            watched in self._buttons
            and event.type() == QEvent.MouseButtonDblClick
            and event.button() == Qt.LeftButton
        ):
            index = self._buttons.index(watched)
            self.setCurrentIndex(index)
            self._start_inline_rename(index)
            return True
        if (
            watched is self._name_editor
            and event.type() == QEvent.KeyPress
            and event.key() == Qt.Key_Escape
        ):
            self._cancel_inline_rename()
            return True
        return super().eventFilter(watched, event)

    def _start_inline_rename(self, index):
        if index < 0 or index >= len(self._buttons):
            return
        button = self._buttons[index]
        self._editing_index = index
        self._name_editor.setGeometry(button.geometry())
        self._name_editor.setText(self.tabText(index))
        self._name_editor.show()
        self._name_editor.raise_()
        self._name_editor.setFocus()
        self._name_editor.selectAll()

    def _finish_inline_rename(self):
        index = self._editing_index
        if index < 0:
            return
        name = self._name_editor.text().strip() or f"新端口{index + 1}"
        self._editing_index = -1
        self._name_editor.hide()
        self.tabTextEdited.emit(index, name)

    def _cancel_inline_rename(self):
        self._editing_index = -1
        self._name_editor.hide()

    def _sync_checked_state(self):
        for index, button in enumerate(self._buttons):
            signals_blocked = button.blockSignals(True)
            button.setChecked(index == self._current_index)
            button.blockSignals(signals_blocked)

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.angleDelta().x()
        if not delta:
            event.ignore()
            return
        self.scroll_requested.emit(-delta)
        event.accept()

    def content_width(self):
        if not self._buttons:
            return 1
        button_widths = self.BUTTON_WIDTH * len(self._buttons)
        return button_widths + self._layout.spacing() * (len(self._buttons) - 1)


class _PortTabScrollArea(QScrollArea):
    TAB_HEIGHT = 40
    SCROLL_BAR_HEIGHT = 8

    def __init__(self, tab_bar, parent=None):
        super().__init__(parent)
        self._tab_bar = tab_bar
        self._sync_timer = QTimer(self)
        self._sync_timer.setSingleShot(True)
        self._sync_timer.timeout.connect(self.sync_content_width)
        self.setWidgetResizable(False)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidget(tab_bar)
        scroll_bar = self.horizontalScrollBar()
        scroll_bar.setFixedHeight(self.SCROLL_BAR_HEIGHT)
        scroll_bar.setStyleSheet(
            """
            QScrollBar:horizontal {
                height: 8px;
                margin: 0;
                border: none;
                border-radius: 4px;
                background: #EDF2F7;
            }
            QScrollBar::handle:horizontal {
                min-width: 36px;
                margin: 1px 0;
                border-radius: 3px;
                background: #B7C5D6;
            }
            QScrollBar::handle:horizontal:hover {
                background: #95A9C0;
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                width: 0px;
                height: 0px;
                border: none;
                background: none;
            }
            QScrollBar::left-arrow:horizontal,
            QScrollBar::right-arrow:horizontal {
                width: 0px;
                height: 0px;
                border: none;
                image: none;
                background: none;
            }
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                border-radius: 4px;
                background: #EDF2F7;
            }
            """
        )
        self.setFixedHeight(self.TAB_HEIGHT)

    def sync_content_width(self):
        content_width = self._tab_bar.content_width()
        viewport_width = max(1, self.viewport().width())
        self._tab_bar.resize(max(1, content_width), self.TAB_HEIGHT)
        scroll_bar_height = (
            self.SCROLL_BAR_HEIGHT if content_width > viewport_width else 0
        )
        target_height = self.TAB_HEIGHT + scroll_bar_height
        if self.height() != target_height:
            self.setFixedHeight(target_height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_timer.start(0)


class _CopyConditionsDialog(ConfigDialogBase):
    def __init__(self, source_name, target_groups, parent=None):
        super().__init__(parent)
        self._checkboxes = []
        self.setWindowTitle("复制工况到其他端口")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setMinimumWidth(420)

        title = QLabel(f"将“{source_name}”的工况复制到：")
        all_checkbox = QCheckBox("全选")
        all_checkbox.toggled.connect(self._set_all_checked)

        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(8)
        for group_index, group_name in target_groups:
            checkbox = QCheckBox(group_name)
            checkbox.setProperty("group_index", group_index)
            list_layout.addWidget(checkbox)
            self._checkboxes.append(checkbox)
        list_layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        scroll_area.setWidget(list_widget)
        scroll_area.setMinimumHeight(160)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        buttons.button(QDialogButtonBox.Ok).setText("确认复制")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        layout.addWidget(title)
        layout.addWidget(all_checkbox)
        layout.addWidget(scroll_area)
        layout.addWidget(buttons)

    def _set_all_checked(self, checked):
        for checkbox in self._checkboxes:
            checkbox.setChecked(checked)

    def selected_group_indices(self):
        return [
            int(checkbox.property("group_index"))
            for checkbox in self._checkboxes
            if checkbox.isChecked()
        ]


class ProductTestProjectConfigDialog(ConfigDialogBase):
    CONDITION_CONTROL_HEIGHT = 30
    CONDITION_CONTROL_FONT_FAMILY = "Microsoft YaHei UI"
    CONDITION_CONTROL_FONT_SIZE = 16
    CONDITION_CONTROL_FONT_STYLE = (
        'font-family: "Microsoft YaHei UI"; font-size: 16px;'
    )

    projects_changed = pyqtSignal()
    programs_changed = pyqtSignal()

    def __init__(self, manager=None, queue_editor_callback=None, parent=None):
        super().__init__(parent)
        self.manager = manager or ProductTestProjectConfigManager()
        self.queue_editor_callback = queue_editor_callback
        self.current_file = None
        self.project_data = self.manager.default_project()
        self.queue_catalog = {}
        self._visible_group_index = -1
        self._loading = False
        self._dirty = False

        self.project_name_input = QLineEdit()
        self.port_count_spinbox = QSpinBox()
        self.port_tabs = _PortSelectorBar()
        self.port_tabs_scroll_area = _PortTabScrollArea(self.port_tabs)
        self.condition_header = QWidget()
        self.condition_section_title = QLabel()
        self.add_condition_btn = QPushButton("+ 添加工况")
        self.delete_condition_btn = QPushButton("删除工况")
        self.copy_conditions_btn = QPushButton("复制到其他端口")
        self.condition_table = QTableWidget(0, 6)
        self.result_root_input = QLineEdit()
        self.select_result_root_btn = QPushButton("选择")
        self.new_project_btn = QPushButton("新建配置")
        self.import_project_btn = QPushButton("导入配置")
        self.save_as_btn = QPushButton("另存为配置")
        self.delete_project_btn = QPushButton("删除配置")
        self.cancel_btn = QPushButton("取消")
        self.save_btn = QPushButton("保存")

        self._init_ui()
        self._connect_signals()
        self._load_initial_project()

    def _init_ui(self):
        self.setObjectName("productTestProjectDialog")
        self.setWindowTitle("产品测试配置")
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setMinimumSize(1020, 650)
        self.resize(1210, 720)

        self.project_name_input.setPlaceholderText("请输入项目名称")
        self.project_name_input.setMinimumWidth(300)
        self.port_count_spinbox.setRange(1, 9999)
        self.port_count_spinbox.setMinimumWidth(120)
        self.port_tabs.setObjectName("productProjectPortTabs")
        self.port_tabs.setMinimumHeight(36)
        self.port_tabs_scroll_area.setObjectName("productProjectPortScrollArea")

        basic_content = QWidget()
        basic_layout = QFormLayout(basic_content)
        basic_layout.setContentsMargins(12, 10, 12, 10)
        basic_layout.setHorizontalSpacing(14)
        basic_layout.setVerticalSpacing(9)
        basic_layout.addRow("项目名称：", self.project_name_input)

        port_row = QWidget()
        port_layout = QHBoxLayout(port_row)
        port_layout.setContentsMargins(0, 0, 0, 0)
        port_layout.setSpacing(10)
        port_layout.addWidget(self.port_count_spinbox)
        port_layout.addWidget(self.port_tabs_scroll_area, 1)
        basic_layout.addRow("端口数量：", port_row)

        self.condition_header.setObjectName("productProjectConditionHeader")
        self.condition_section_title.setObjectName("productProjectConditionTitle")
        for button in (
            self.add_condition_btn,
            self.delete_condition_btn,
            self.copy_conditions_btn,
        ):
            button.setMinimumHeight(30)
            button.setMaximumHeight(30)

        condition_header_layout = QHBoxLayout(self.condition_header)
        condition_header_layout.setContentsMargins(12, 4, 12, 4)
        condition_header_layout.setSpacing(8)
        condition_header_layout.addWidget(self.condition_section_title)
        condition_header_layout.addStretch()
        condition_header_layout.addWidget(self.add_condition_btn)
        condition_header_layout.addWidget(self.delete_condition_btn)
        condition_header_layout.addWidget(self.copy_conditions_btn)

        self._configure_condition_table()

        self.result_root_input.setReadOnly(True)
        self.result_root_input.setPlaceholderText("请选择测试结果根目录")
        self.select_result_root_btn.setIcon(
            QIcon(DEFAULT_DIR + "ui/assets/folder-outline.svg")
        )
        self.select_result_root_btn.setIconSize(QSize(18, 18))
        self.select_result_root_btn.setToolTip("选择测试结果根目录")

        result_content = QWidget()
        result_layout = QFormLayout(result_content)
        result_layout.setContentsMargins(12, 9, 12, 9)
        result_layout.setHorizontalSpacing(14)
        result_layout.setVerticalSpacing(7)

        result_root_row = QWidget()
        result_root_layout = QHBoxLayout(result_root_row)
        result_root_layout.setContentsMargins(0, 0, 0, 0)
        result_root_layout.setSpacing(8)
        result_root_layout.addWidget(self.result_root_input, 1)
        result_root_layout.addWidget(self.select_result_root_btn)
        result_layout.addRow("保存目录：", result_root_row)

        for button in (
            self.add_condition_btn,
            self.delete_condition_btn,
            self.copy_conditions_btn,
            self.select_result_root_btn,
            self.new_project_btn,
            self.import_project_btn,
            self.save_as_btn,
            self.delete_project_btn,
            self.cancel_btn,
            self.save_btn,
        ):
            button.setAutoDefault(False)
            button.setDefault(False)

        for button in (
            self.select_result_root_btn,
            self.new_project_btn,
            self.import_project_btn,
            self.save_as_btn,
            self.delete_project_btn,
            self.cancel_btn,
            self.save_btn,
        ):
            button.setMinimumHeight(36)
        self.save_btn.setObjectName("productProgramPrimaryButton")

        footer = QHBoxLayout()
        footer.setContentsMargins(10, 40, 10, 0)
        footer.setSpacing(8)
        footer.addWidget(self.new_project_btn)
        footer.addWidget(self.import_project_btn)
        footer.addWidget(self.save_as_btn)
        footer.addWidget(self.delete_project_btn)
        footer.addStretch()
        footer.addWidget(self.cancel_btn)
        footer.addWidget(self.save_btn)

        condition_table_container = QWidget()
        condition_table_layout = QVBoxLayout(condition_table_container)
        condition_table_layout.setContentsMargins(10, 0, 10, 0)
        condition_table_layout.addWidget(self.condition_table)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(8)
        layout.addWidget(self._section_header("项目基础信息"))
        layout.addWidget(basic_content)
        layout.addWidget(self.condition_header)
        layout.addWidget(condition_table_container, 1)
        layout.addWidget(self._section_header("测试结果保存"))
        layout.addWidget(result_content)
        layout.addLayout(footer)

        self.apply_config_dialog_theme(
            ui_style_const.product_test_program_dialog_style
            + """
            QDialog#productTestProjectDialog {
                background-color: #F3F6F9;
            }
            QDialog#productTestProjectDialog QLabel,
            QDialog#productTestProjectDialog QPushButton,
            QDialog#productTestProjectDialog QLineEdit,
            QDialog#productTestProjectDialog QSpinBox,
            QDialog#productTestProjectDialog QComboBox,
            QDialog#productTestProjectDialog QTableWidget,
            QDialog#productTestProjectDialog QHeaderView::section {
                font-family: "Microsoft YaHei UI";
                font-size: 16px;
            }
            QDialog#productTestProjectDialog QTableWidget#productProgramTable QHeaderView::section {
                font-weight: 500;
            }
            QLabel#productProjectSectionHeader {
                min-height: 30px;
                max-height: 30px;
                padding: 0 12px;
                color: #1F2937;
                background-color: #D4E1F2;
                border: none;
                font-family: "Microsoft YaHei UI";
                font-size: 16px;
                font-weight: 400;
            }
            QWidget#productProjectConditionHeader {
                min-height: 38px;
                max-height: 38px;
                background-color: #D4E1F2;
                border: none;
            }
            QLabel#productProjectConditionTitle {
                color: #1F2937;
                background-color: transparent;
                font-family: "Microsoft YaHei UI";
                font-size: 16px;
                font-weight: 400;
            }
            QPushButton#productProjectPortButton {
                min-height: 30px;
                padding: 0 8px;
                color: #334155;
                border: 1px solid #C9D6E8;
                border-radius: 3px;
                background: #FFFFFF;
                font-weight: normal;
                outline: none;
            }
            QPushButton#productProjectPortButton:checked {
                color: #245C96;
                border-color: #3B6FB6;
                background: #EAF2FB;
                font-weight: normal;
            }
            QPushButton#productProjectPortButton:hover:!checked {
                background: #F3F7FC;
                border-color: #9EB5D1;
            }
            QLineEdit#productProjectPortNameEditor {
                padding: 0 8px;
                color: #245C96;
                border: 1px solid #3B6FB6;
                border-radius: 3px;
                background: #FFFFFF;
                selection-color: #FFFFFF;
                selection-background-color: #3B6FB6;
            }
            QScrollArea#productProjectPortScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollArea#productProjectPortScrollArea QWidget#qt_scrollarea_viewport {
                background-color: transparent;
            }
            """
        )

    @staticmethod
    def _section_header(text):
        label = QLabel(text)
        label.setObjectName("productProjectSectionHeader")
        return label

    def _configure_condition_table(self):
        self.condition_table.setObjectName("productProgramTable")
        self.condition_table.setHorizontalHeaderLabels(
            [
                "序号",
                "工况名称",
                "状态码",
                "测试队列配置",
                "录音时长",
                "判定与分析",
            ]
        )
        self.condition_table.verticalHeader().setVisible(False)
        self.condition_table.verticalHeader().setDefaultSectionSize(44)
        self.condition_table.horizontalHeader().setMinimumHeight(38)
        self.condition_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.condition_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.condition_table.setAlternatingRowColors(True)
        self.condition_table.setWordWrap(False)
        self.condition_table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.condition_table.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.SelectedClicked
            | QAbstractItemView.EditKeyPressed
        )
        self.condition_table.setStyleSheet(
            ui_style_const.product_test_program_table_style
        )
        header = self.condition_table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignCenter)
        for column in range(6):
            header.setSectionResizeMode(column, QHeaderView.Fixed)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        self.condition_table.setColumnWidth(0, 50)
        self.condition_table.setColumnWidth(1, 140)
        self.condition_table.setColumnWidth(2, 220)
        self.condition_table.setColumnWidth(3, 320)
        self.condition_table.setColumnWidth(4, 90)

    def _connect_signals(self):
        self.project_name_input.textEdited.connect(self._on_project_field_changed)
        self.port_count_spinbox.valueChanged.connect(self._on_port_count_changed)
        self.port_tabs.currentChanged.connect(self._on_port_changed)
        self.port_tabs.tabTextEdited.connect(self._on_port_tab_text_edited)
        self.port_tabs.scroll_requested.connect(self._scroll_port_tabs)
        self.condition_table.itemChanged.connect(self._on_table_item_changed)
        self.add_condition_btn.clicked.connect(self._add_condition)
        self.delete_condition_btn.clicked.connect(self._delete_selected_condition)
        self.copy_conditions_btn.clicked.connect(self._show_copy_conditions_dialog)
        self.select_result_root_btn.clicked.connect(self._select_result_root)
        self.new_project_btn.clicked.connect(self._new_project)
        self.import_project_btn.clicked.connect(self._import_project)
        self.save_as_btn.clicked.connect(self._save_project_as)
        self.delete_project_btn.clicked.connect(self._delete_project)
        self.cancel_btn.clicked.connect(self.reject)
        self.save_btn.clicked.connect(
            lambda _checked=False: self._save_project(close_dialog=True)
        )

    def _load_initial_project(self):
        self.queue_catalog = self.manager.load_queue_catalog()
        registry = self.manager.load_registry()
        active_file = registry.get(REGISTRY_ACTIVE_FILE_KEY)
        if not active_file:
            self._show_project(self.manager.default_project(), None)
            return
        load_code, project_data = self.manager.load_project(active_file)
        if load_code == error_code.OK:
            self._show_project(project_data, active_file)
            return

        QMessageBox.warning(self, "加载失败", str(project_data))
        missing_name = next(
            (
                item.get(PROJECT_NAME_KEY, "")
                for item in registry.get(REGISTRY_CONFIGS_KEY, [])
                if item.get(REGISTRY_FILE_KEY) == active_file
            ),
            "",
        )
        missing_project = self.manager.default_project()
        missing_project[PROJECT_NAME_KEY] = str(missing_name or "")
        self._show_project(missing_project, active_file)

    def _load_project(self, file_name):
        load_code, project_data = self.manager.load_project(file_name)
        if load_code != error_code.OK:
            QMessageBox.warning(self, "加载失败", str(project_data))
            return False
        self.queue_catalog = self.manager.load_queue_catalog()
        self._show_project(project_data, file_name)
        return True

    def _show_project(self, project_data, file_name):
        self._loading = True
        self.current_file = file_name
        self.project_data = copy.deepcopy(project_data)
        self.project_name_input.setText(
            str(self.project_data.get(PROJECT_NAME_KEY, "") or "")
        )
        self.result_root_input.setText(
            str(self.project_data.get(RESULT_ROOT_DIRECTORY_KEY, "") or "")
        )
        groups = self.project_data.get(TEST_GROUPS_KEY, [])
        self.port_count_spinbox.setValue(max(1, len(groups)))
        self._rebuild_port_tabs()
        self._visible_group_index = -1
        if groups:
            self.port_tabs.setCurrentIndex(0)
            self._load_group(0)
        self._loading = False
        self._set_dirty(False)

    def _rebuild_port_tabs(self):
        current_index = max(0, self.port_tabs.currentIndex())
        self.port_tabs.blockSignals(True)
        while self.port_tabs.count():
            self.port_tabs.removeTab(self.port_tabs.count() - 1)
        for group_index, group in enumerate(
            self.project_data.get(TEST_GROUPS_KEY, []), 1
        ):
            group_name = str(group.get(GROUP_NAME_KEY, "") or "").strip()
            self.port_tabs.addTab(group_name or f"新端口{group_index}")
        if self.port_tabs.count():
            self.port_tabs.setCurrentIndex(
                min(current_index, self.port_tabs.count() - 1)
            )
        self.port_tabs.blockSignals(False)
        self.port_tabs_scroll_area.sync_content_width()

    def _on_port_changed(self, index):
        if self._loading or index < 0:
            return
        self._collect_visible_group()
        self._load_group(index)
        self._ensure_port_tab_visible(index)

    def _on_port_tab_text_edited(self, index, text):
        if self._loading or index != self._visible_group_index:
            return
        title = str(text or "").strip() or f"新端口{index + 1}"
        self.port_tabs.setTabText(index, title)
        self.port_tabs_scroll_area.sync_content_width()
        self.condition_section_title.setText(f"工况配置   ·   {title}")
        self._set_dirty(True)

    def _scroll_port_tabs(self, delta):
        scroll_bar = self.port_tabs_scroll_area.horizontalScrollBar()
        scroll_bar.setValue(scroll_bar.value() + delta)

    def _ensure_port_tab_visible(self, index):
        if index < 0 or index >= self.port_tabs.count():
            return
        scroll_bar = self.port_tabs_scroll_area.horizontalScrollBar()
        tab_rect = self.port_tabs.tabRect(index)
        viewport_width = self.port_tabs_scroll_area.viewport().width()
        if tab_rect.left() < scroll_bar.value():
            scroll_bar.setValue(tab_rect.left())
        elif tab_rect.right() > scroll_bar.value() + viewport_width:
            scroll_bar.setValue(tab_rect.right() - viewport_width)

    def _load_group(self, group_index):
        groups = self.project_data.get(TEST_GROUPS_KEY, [])
        if group_index < 0 or group_index >= len(groups):
            return
        self._loading = True
        self._visible_group_index = group_index
        group = groups[group_index]
        group_name = str(group.get(GROUP_NAME_KEY, "") or "")
        self.condition_section_title.setText(
            f"工况配置   ·   {group_name or f'新端口{group_index + 1}'}"
        )
        self.condition_table.setRowCount(0)
        for condition in group.get(TEST_CONDITIONS_KEY, []):
            self._append_condition_row(condition)
        self._loading = False

    def _collect_visible_group(self):
        groups = self.project_data.get(TEST_GROUPS_KEY, [])
        index = self._visible_group_index
        if index < 0 or index >= len(groups):
            return
        conditions = []
        for row in range(self.condition_table.rowCount()):
            condition_item = self.condition_table.item(row, 1)
            trigger_input = self.condition_table.cellWidget(row, 2)
            queue_combobox, _operation_button = self._queue_controls_for_row(row)
            conditions.append(
                {
                    CONDITION_NAME_KEY: (
                        condition_item.text().strip()
                        if condition_item is not None
                        else ""
                    ),
                    TRIGGER_STATE_KEY: normalize_trigger_state(
                        trigger_input.text() if trigger_input is not None else ""
                    ),
                    TEST_QUEUE_KEY: self._combobox_value(queue_combobox),
                }
            )
        groups[index] = {
            GROUP_NAME_KEY: (
                self.port_tabs.tabText(index).strip() or f"新端口{index + 1}"
            ),
            TEST_CONDITIONS_KEY: conditions,
        }

    def collect_project(self):
        self._collect_visible_group()
        project_data = copy.deepcopy(self.project_data)
        project_data[PROJECT_NAME_KEY] = self.project_name_input.text().strip()
        project_data[RESULT_ROOT_DIRECTORY_KEY] = self.result_root_input.text().strip()
        return project_data

    def _append_condition_row(self, condition):
        row = self.condition_table.rowCount()
        self.condition_table.insertRow(row)

        number_item = QTableWidgetItem(str(row + 1))
        number_item.setTextAlignment(Qt.AlignCenter)
        number_item.setFlags(number_item.flags() & ~Qt.ItemIsEditable)
        self.condition_table.setItem(row, 0, number_item)

        condition_item = QTableWidgetItem(
            str(condition.get(CONDITION_NAME_KEY, "") or "")
        )
        self.condition_table.setItem(row, 1, condition_item)

        trigger_input = QLineEdit(
            str(condition.get(TRIGGER_STATE_KEY, "") or "")
        )
        trigger_input.setStyleSheet(self.CONDITION_CONTROL_FONT_STYLE)
        trigger_input.setPlaceholderText("选填")
        trigger_input.setToolTip("完整十六进制报文；全部工况需同时填写或同时留空")
        trigger_input.textEdited.connect(self._on_row_widget_changed)
        self.condition_table.setCellWidget(row, 2, trigger_input)

        queue_combobox = _RefreshingQueueComboBox(self._refresh_queue_options)
        queue_combobox.setStyleSheet(self.CONDITION_CONTROL_FONT_STYLE)
        queue_combobox.setFixedHeight(self.CONDITION_CONTROL_HEIGHT)
        self._populate_queue_combobox(
            queue_combobox, str(condition.get(TEST_QUEUE_KEY, "") or "")
        )
        queue_combobox.currentIndexChanged.connect(self._on_row_widget_changed)

        operation_button = QPushButton()
        operation_button.setStyleSheet(self.CONDITION_CONTROL_FONT_STYLE)
        operation_button.setAutoDefault(False)
        operation_button.setFixedHeight(self.CONDITION_CONTROL_HEIGHT)
        operation_button.setMinimumWidth(
            max(
                64,
                max(
                    self.condition_table.fontMetrics().horizontalAdvance(text)
                    for text in ("新建", "编辑")
                )
                + 24,
            )
        )
        operation_button.clicked.connect(
            lambda _checked=False, button=operation_button: self._edit_queue_for_button(
                button
            )
        )

        queue_cell = QWidget()
        queue_cell.queue_combobox = queue_combobox
        queue_cell.operation_button = operation_button
        queue_layout = QHBoxLayout(queue_cell)
        queue_layout.setContentsMargins(4, 2, 4, 2)
        queue_layout.setSpacing(6)
        queue_layout.addWidget(queue_combobox, 1)
        queue_layout.addWidget(operation_button)
        self.condition_table.setCellWidget(row, 3, queue_cell)

        duration_item = self._read_only_item()
        self.condition_table.setItem(row, 4, duration_item)

        summary_item = self._read_only_item()
        self.condition_table.setItem(row, 5, summary_item)
        self._update_row_summary(row)

    @staticmethod
    def _read_only_item():
        item = QTableWidgetItem()
        item.setTextAlignment(Qt.AlignCenter)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    def _queue_controls_for_row(self, row):
        if row < 0 or row >= self.condition_table.rowCount():
            return None, None
        queue_cell = self.condition_table.cellWidget(row, 3)
        return (
            getattr(queue_cell, "queue_combobox", None),
            getattr(queue_cell, "operation_button", None),
        )

    def _populate_queue_combobox(self, combobox, current_queue):
        signals_blocked = combobox.blockSignals(True)
        combobox.clear()
        queue_names = self._available_queue_names()
        if queue_names:
            combobox.addItem("请选择", "")
            for queue_name in queue_names:
                combobox.addItem(queue_name, queue_name)
        else:
            combobox.addItem(NO_QUEUE_TEXT, "")
        if current_queue and current_queue not in queue_names:
            combobox.addItem(
                f"{current_queue}{UNAVAILABLE_QUEUE_SUFFIX}", current_queue
            )
        selected_index = combobox.findData(current_queue)
        combobox.setCurrentIndex(selected_index if selected_index >= 0 else 0)
        combobox.blockSignals(signals_blocked)

    def _available_queue_names(self):
        return sorted(
            queue_name
            for queue_name, queue_info in self.queue_catalog.items()
            if isinstance(queue_info, dict) and queue_info.get("available", False)
        )

    def _update_row_summary(self, row):
        if row < 0 or row >= self.condition_table.rowCount():
            return
        queue_combobox, operation_button = self._queue_controls_for_row(row)
        queue_name = self._combobox_value(queue_combobox)
        queue_info = self.queue_catalog.get(queue_name)
        duration_item = self.condition_table.item(row, 4)
        summary_item = self.condition_table.item(row, 5)

        duration_text = "--"
        summary_text = "--"
        tooltip = ""
        if queue_name and not queue_info:
            summary_text = "测试队列不存在"
            tooltip = summary_text
        elif queue_info and not queue_info.get("available", False):
            summary_text = str(queue_info.get("reason", "") or "测试队列不可用")
            tooltip = summary_text
        elif queue_info:
            duration = queue_info.get("duration")
            if duration is not None:
                duration_text = f"{duration:g}秒"
            analysis_items = queue_info.get("analysis_items", [])
            judgment_text = (
                "自动判定" if queue_info.get("can_auto_judge", False) else "仅分析"
            )
            summary_text = judgment_text
            if analysis_items:
                summary_text += "；" + "、".join(analysis_items)
            tooltip = summary_text

        duration_item.setText(duration_text)
        summary_item.setText(summary_text)
        summary_item.setToolTip(tooltip)
        queue_available = bool(queue_info and queue_info.get("available", False))
        operation_text = "编辑" if queue_available else "新建"
        operation_button.setText(operation_text)
        operation_button.setToolTip(
            "编辑当前测试队列" if queue_available else "新建测试队列"
        )
        operation_button.setEnabled(callable(self.queue_editor_callback))

    def _on_table_item_changed(self, item):
        if not self._loading:
            self._set_dirty(True)

    def _on_row_widget_changed(self):
        if self._loading:
            return
        row = self._find_widget_row(self.sender())
        if row >= 0:
            self._update_row_summary(row)
        self._set_dirty(True)

    def _find_widget_row(self, widget):
        for row in range(self.condition_table.rowCount()):
            queue_combobox, operation_button = self._queue_controls_for_row(row)
            if widget in (
                self.condition_table.cellWidget(row, 2),
                queue_combobox,
                operation_button,
            ):
                return row
        return -1

    def _on_project_field_changed(self):
        if self._loading:
            return
        self._set_dirty(True)

    def _on_port_count_changed(self, new_count):
        if self._loading:
            return
        self._collect_visible_group()
        groups = self.project_data.setdefault(TEST_GROUPS_KEY, [])
        old_count = len(groups)
        if new_count < old_count:
            result = QMessageBox.question(
                self,
                "减少端口",
                f"将删除最后 {old_count - new_count} 个端口及其工况，确定继续吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if result != QMessageBox.Yes:
                self._loading = True
                self.port_count_spinbox.setValue(old_count)
                self._loading = False
                return
            del groups[new_count:]
        elif new_count > old_count:
            for group_index in range(old_count, new_count):
                groups.append(
                    {
                        GROUP_NAME_KEY: f"新端口{group_index + 1}",
                        TEST_CONDITIONS_KEY: [
                            {
                                CONDITION_NAME_KEY: "档位1",
                                TRIGGER_STATE_KEY: "",
                                TEST_QUEUE_KEY: "",
                            }
                        ],
                    }
                )
        self._rebuild_port_tabs()
        target_index = min(max(0, self._visible_group_index), new_count - 1)
        self.port_tabs.setCurrentIndex(target_index)
        self._load_group(target_index)
        self._set_dirty(True)

    def _add_condition(self):
        used_names = {
            self.condition_table.item(row, 1).text().strip()
            for row in range(self.condition_table.rowCount())
            if self.condition_table.item(row, 1) is not None
        }
        condition_number = 1
        while f"档位{condition_number}" in used_names:
            condition_number += 1
        self._loading = True
        self._append_condition_row(
            {
                CONDITION_NAME_KEY: f"档位{condition_number}",
                TRIGGER_STATE_KEY: "",
                TEST_QUEUE_KEY: "",
            }
        )
        self._loading = False
        row = self.condition_table.rowCount() - 1
        self.condition_table.selectRow(row)
        self.condition_table.scrollToItem(self.condition_table.item(row, 0))
        self._set_dirty(True)

    def _delete_selected_condition(self):
        row = self.condition_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "删除工况", "请先选择要删除的工况")
            return
        self.condition_table.removeRow(row)
        self._renumber_conditions()
        if self.condition_table.rowCount():
            self.condition_table.selectRow(
                min(row, self.condition_table.rowCount() - 1)
            )
        self._set_dirty(True)

    def _renumber_conditions(self):
        for row in range(self.condition_table.rowCount()):
            item = self.condition_table.item(row, 0)
            if item is not None:
                item.setText(str(row + 1))

    def _show_copy_conditions_dialog(self):
        self._collect_visible_group()
        groups = self.project_data.get(TEST_GROUPS_KEY, [])
        source_index = self._visible_group_index
        targets = [
            (index, str(group.get(GROUP_NAME_KEY, "") or f"新端口{index + 1}"))
            for index, group in enumerate(groups)
            if index != source_index
        ]
        if not targets:
            QMessageBox.information(self, "复制工况", "当前没有其他端口可复制")
            return
        source_name = str(groups[source_index].get(GROUP_NAME_KEY, "") or "当前端口")
        dialog = _CopyConditionsDialog(source_name, targets, self)
        if dialog.exec() != dialog.Accepted:
            return
        selected_indices = dialog.selected_group_indices()
        if not selected_indices:
            QMessageBox.information(self, "复制工况", "请至少选择一个目标端口")
            return
        self._copy_conditions_to_groups(selected_indices, confirm_replace=True)

    def _copy_conditions_to_groups(self, target_indices, confirm_replace=True):
        self._collect_visible_group()
        groups = self.project_data.get(TEST_GROUPS_KEY, [])
        source_index = self._visible_group_index
        if source_index < 0 or source_index >= len(groups):
            return False
        valid_targets = sorted(
            {
                index
                for index in target_indices
                if 0 <= index < len(groups) and index != source_index
            }
        )
        if not valid_targets:
            return False
        if confirm_replace and any(
            groups[index].get(TEST_CONDITIONS_KEY) for index in valid_targets
        ):
            result = QMessageBox.question(
                self,
                "替换目标端口工况",
                "所选端口的现有工况将被整体替换，确定继续吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if result != QMessageBox.Yes:
                return False
        copied_conditions = [
            {
                CONDITION_NAME_KEY: str(
                    condition.get(CONDITION_NAME_KEY, "") or ""
                ),
                TRIGGER_STATE_KEY: "",
                TEST_QUEUE_KEY: str(condition.get(TEST_QUEUE_KEY, "") or ""),
            }
            for condition in groups[source_index].get(TEST_CONDITIONS_KEY, [])
        ]
        for target_index in valid_targets:
            groups[target_index][TEST_CONDITIONS_KEY] = copy.deepcopy(
                copied_conditions
            )
        self._set_dirty(True)
        return True

    def _refresh_queue_options(self):
        selected_queues = [
            self._combobox_value(self._queue_controls_for_row(row)[0])
            for row in range(self.condition_table.rowCount())
        ]
        self.queue_catalog = self.manager.load_queue_catalog()
        self._loading = True
        try:
            for row, queue_name in enumerate(selected_queues):
                combobox, _operation_button = self._queue_controls_for_row(row)
                self._populate_queue_combobox(combobox, queue_name)
                self._update_row_summary(row)
        finally:
            self._loading = False

    def _edit_queue_for_button(self, button):
        row = -1
        for candidate_row in range(self.condition_table.rowCount()):
            _queue_combobox, operation_button = self._queue_controls_for_row(
                candidate_row
            )
            if operation_button is button:
                row = candidate_row
                break
        if row < 0 or not callable(self.queue_editor_callback):
            return
        queue_combobox, _operation_button = self._queue_controls_for_row(row)
        queue_name = self._combobox_value(queue_combobox)
        queue_info = self.queue_catalog.get(queue_name, {})
        queue_path = (
            queue_info.get("path")
            if queue_info and queue_info.get("available", False)
            else None
        )
        queue_names_before = set(self.queue_catalog)
        self.queue_editor_callback(queue_path)
        self._refresh_queue_options()
        if not queue_name:
            added_queues = set(self.queue_catalog) - queue_names_before
            if len(added_queues) == 1:
                new_queue = added_queues.pop()
                selected_index = queue_combobox.findData(new_queue)
                if selected_index >= 0:
                    queue_combobox.setCurrentIndex(selected_index)
                    self._update_row_summary(row)
                    self._set_dirty(True)

    def _select_result_root(self):
        current_root = self.result_root_input.text().strip()
        initial_root = current_root if os.path.isdir(current_root) else ""
        selected_root = QFileDialog.getExistingDirectory(
            self, "选择测试结果根目录", initial_root
        )
        if selected_root:
            self.result_root_input.setText(ntpath.normpath(selected_root))
            self._set_dirty(True)

    def _new_project(self):
        if not self._confirm_leave_changes():
            return
        self._show_project(self.manager.default_project(), None)
        self.project_name_input.setFocus()

    def _import_project(self):
        if not self._confirm_leave_changes():
            return
        source_path, _ = QFileDialog.getOpenFileName(
            self, "导入产品测试配置", "", "JSON 配置 (*.json)"
        )
        if not source_path:
            return
        success, message = self.manager.import_project(source_path)
        if not success:
            QMessageBox.warning(self, "导入失败", message)
            return
        self._load_project(message)
        self._emit_projects_changed()

    def _save_project_as(self):
        project_data = self.collect_project()
        new_name, accepted = QInputDialog.getText(
            self,
            "另存为配置",
            "新项目名称：",
            text=project_data.get(PROJECT_NAME_KEY, ""),
        )
        if not accepted:
            return
        success, message = self.manager.save_as(project_data, new_name)
        if not success:
            QMessageBox.warning(self, "另存为失败", message)
            return
        self._load_project(message)
        self._emit_projects_changed()
        QMessageBox.information(self, "另存为成功", "产品测试配置已另存")

    def _delete_project(self):
        if not self.current_file:
            QMessageBox.information(self, "删除配置", "当前配置尚未保存")
            return
        project_name = self.project_name_input.text().strip() or self.current_file
        result = QMessageBox.question(
            self,
            "删除配置",
            f"确定删除产品测试配置“{project_name}”吗？\n"
            "测试队列和已有测试结果不会被删除。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if result != QMessageBox.Yes:
            return
        success, message = self.manager.delete_project(self.current_file)
        if not success:
            QMessageBox.warning(self, "删除失败", message)
            return
        self._emit_projects_changed()
        self._show_project(self.manager.default_project(), None)

    def _save_project(self, close_dialog=True):
        project_data = self.collect_project()
        validation = self.manager.validate_project(
            project_data, self.current_file, self.queue_catalog
        )
        if not validation["can_save"]:
            QMessageBox.warning(
                self, "无法保存", "\n".join(validation["save_errors"])
            )
            return False
        success, message = self.manager.save_project(self.current_file, project_data)
        if not success:
            QMessageBox.warning(self, "保存失败", message)
            return False
        self.current_file = message
        self.project_data = project_data
        self._set_dirty(False)
        self._emit_projects_changed()
        if validation["use_warnings"]:
            result_text = "配置已保存，部分测试队列不能自动输出 OK/NG。"
        else:
            result_text = "产品测试配置已保存。"
        QMessageBox.information(self, "保存成功", result_text)
        if close_dialog:
            self.accept()
        return True

    def _emit_projects_changed(self):
        self.projects_changed.emit()
        self.programs_changed.emit()

    def _set_dirty(self, dirty):
        self._dirty = bool(dirty)

    def _confirm_leave_changes(self):
        if not self._dirty:
            return True
        message_box = QMessageBox(
            QMessageBox.Question,
            "未保存的修改",
            "当前配置尚未保存。是否先保存？",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            self,
        )
        message_box.setDefaultButton(QMessageBox.Save)
        message_box.button(QMessageBox.Save).setText("保存")
        message_box.button(QMessageBox.Discard).setText("不保存")
        message_box.button(QMessageBox.Cancel).setText("取消")
        result = message_box.exec_()
        if result == QMessageBox.Cancel:
            return False
        if result == QMessageBox.Save:
            return self._save_project(close_dialog=False)
        return True

    def reject(self):
        if self._confirm_leave_changes():
            super().reject()

    def closeEvent(self, event):
        if self._confirm_leave_changes():
            event.accept()
        else:
            event.ignore()

    @staticmethod
    def _combobox_value(combobox):
        if combobox is None:
            return ""
        value = combobox.currentData()
        if value is None:
            value = combobox.currentText()
        return str(value or "").strip()


# Keep the historical class name available while main-window call sites migrate.
ProductTestProgramConfigDialog = ProductTestProjectConfigDialog

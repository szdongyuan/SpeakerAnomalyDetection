"""Shared conditions and station applicability; queue dialogs remain unchanged."""
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QComboBox, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QMenu, QLabel, QCheckBox, QWidgetAction,
)
from base.product_test.config_model import new_condition
from ui.output_load_config_dialog import OutputLoadConfigDialog
from ui.queue_display import queue_display_name, queue_display_options
from ui.product_test.style import CONTROL_STYLE, SELECTOR_STYLE, STATION_MENU_STYLE, TABLE_STYLE


class StationSelection(QComboBox):
    changed = pyqtSignal()

    def __init__(self, stations, condition, parent=None):
        super().__init__(parent)
        self.condition = condition
        self.setStyleSheet(SELECTOR_STYLE)
        self.setFixedHeight(30)
        self.addItem("")
        self.menu = QMenu(self)
        self.menu.setStyleSheet(STATION_MENU_STYLE)
        for station in stations:
            action = QWidgetAction(self.menu)
            action.setText(station["station_name"])
            action.setData(station["station_id"])
            action.setCheckable(True)
            action.setChecked(station["station_id"] in condition["applicable_station_ids"])
            checkbox = QCheckBox(station["station_name"], self.menu)
            checkbox.setChecked(action.isChecked())
            action.setDefaultWidget(checkbox)
            self.menu.addAction(action)
            checkbox.toggled.connect(action.setChecked)
            action.toggled.connect(checkbox.setChecked)
            action.toggled.connect(self._changed)
        self.menu.aboutToHide.connect(self.hidePopup)
        self._label()

    def showPopup(self):
        self.menu.setMinimumWidth(self.width())
        self.menu.popup(self.mapToGlobal(QPoint(0, self.height())))

    def _label(self):
        names = [a.text() for a in self.menu.actions() if a.isChecked()]
        text = "、".join(names) if len(names) < 3 else f"已选 {len(names)} 个工位"
        self.setItemText(0, text or "请选择适用工位")
        self.setToolTip("、".join(names))

    def _changed(self):
        self.condition["applicable_station_ids"] = [a.data() for a in self.menu.actions() if a.isChecked()]
        self._label()
        self.changed.emit()


class ConditionEditor(QWidget):
    changed = pyqtSignal()
    edit_queue = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.catalog = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        header = QWidget()
        header.setObjectName("productProjectConditionHeader")
        bar = QHBoxLayout(header)
        bar.setContentsMargins(12, 4, 12, 4)
        bar.setSpacing(8)
        title = QLabel("工况配置")
        title.setObjectName("productProjectConditionTitle")
        bar.addWidget(title)
        self.add_button = QPushButton("+ 添加工况")
        self.remove_button = QPushButton("删除选中工况")
        bar.addStretch()
        bar.addWidget(self.add_button)
        bar.addWidget(self.remove_button)
        for button in (self.add_button, self.remove_button):
            button.setFixedHeight(30)
        layout.addWidget(header)
        self.table = QTableWidget(0, 8)
        self.table.setObjectName("productProgramTable")
        self.table.setStyleSheet(TABLE_STYLE)
        self.table.setAlternatingRowColors(True)
        self.table.setHorizontalHeaderLabels(["选择", "工况名称", "适用工位", "状态码", "测试队列配置", "时长/秒", "判定与分析", "分段分析"])
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(44)
        self.table.horizontalHeader().setMinimumHeight(38)
        for column, width in enumerate([48, 120, 180, 110, 240, 75, 150, 170]):
            self.table.setColumnWidth(column, width)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        table_layout = QHBoxLayout()
        table_layout.setContentsMargins(10, 0, 10, 0)
        table_layout.addWidget(self.table)
        layout.addLayout(table_layout)
        self.add_button.clicked.connect(self._add)
        self.remove_button.clicked.connect(self._remove)

    def set_project(self, project, catalog):
        self.project, self.catalog = project, catalog
        self.table.setRowCount(0)
        for row, condition in enumerate(project["conditions"]):
            self.table.insertRow(row)
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            check.setCheckState(Qt.Unchecked)
            self.table.setItem(row, 0, check)
            for column, field in ((1, "condition_name"), (3, "trigger_state")):
                edit = QLineEdit(condition[field])
                edit.setStyleSheet(CONTROL_STYLE)
                if field == "trigger_state":
                    edit.setPlaceholderText("选填")
                edit.textEdited.connect(lambda value, c=condition, key=field: self._set(c, key, value))
                self._set_cell_control(row, column, edit)
            stations = StationSelection(project["stations"], condition)
            stations.changed.connect(self.changed)
            self._set_cell_control(row, 2, stations)
            cell = QWidget()
            bar = QHBoxLayout(cell)
            bar.setContentsMargins(2, 2, 2, 2)
            queue = QComboBox()
            queue.setStyleSheet(SELECTOR_STYLE)
            queue.setFixedHeight(30)
            queue.addItem("请选择测试队列", "")
            available_names = [name for name, info in catalog.items() if info.get("available")]
            for label, name in queue_display_options(catalog, available_names, condition["test_queue"]):
                queue.addItem(label, name)
            if condition["test_queue"] and queue.findData(condition["test_queue"]) < 0:
                name = condition["test_queue"]
                label = queue_display_name(name, catalog.get(name, {}).get("path"))
                queue.addItem(label + "（不可用）", name)
            queue.setCurrentIndex(max(0, queue.findData(condition["test_queue"])))
            queue.currentIndexChanged.connect(lambda _, c=condition, combo=queue, r=row: self._queue_changed(c, combo.currentData(), r))
            button = QPushButton("编辑" if condition["test_queue"] else "新建")
            button.setStyleSheet(CONTROL_STYLE)
            button.setFixedHeight(30)
            button.setMinimumWidth(64)
            button.setAutoDefault(False)
            button.clicked.connect(lambda _, c=condition: self.edit_queue.emit(c["test_queue"]))
            bar.addWidget(queue, 1)
            bar.addWidget(button)
            self.table.setCellWidget(row, 4, cell)
            segment_cell = QWidget()
            segment_layout = QHBoxLayout(segment_cell)
            segment_layout.setContentsMargins(4, 2, 4, 2)
            segment_layout.setSpacing(6)
            segment_cell.status_label = QLabel()
            segment_cell.status_label.setStyleSheet(CONTROL_STYLE)
            segments = QPushButton("设置")
            segments.setStyleSheet(CONTROL_STYLE)
            segments.setFixedSize(64, 30)
            segments.setAutoDefault(False)
            segments.clicked.connect(lambda _, c=condition, r=row: self._segments(c, r))
            segment_layout.addWidget(segment_cell.status_label, 1)
            segment_layout.addWidget(segments)
            self.table.setCellWidget(row, 7, segment_cell)
            self._summary(condition, row)

    def _set_cell_control(self, row, column, control):
        cell = QWidget()
        layout = QHBoxLayout(cell)
        layout.setContentsMargins(4, 0, 4, 0)
        control.setFixedHeight(30)
        layout.addWidget(control)
        layout.setAlignment(control, Qt.AlignVCenter)
        self.table.setCellWidget(row, column, cell)

    def _set(self, condition, key, value):
        condition[key] = value
        self.changed.emit()

    def _queue_changed(self, condition, name, row):
        condition["test_queue"] = name
        self.table.cellWidget(row, 4).findChild(QPushButton).setText("编辑" if name else "新建")
        self._summary(condition, row)
        self.changed.emit()

    def _summary(self, condition, row):
        info = self.catalog.get(condition["test_queue"], {})
        analysis_text = "、".join(info.get("analysis_items", []))
        if info.get("available") and not analysis_text:
            analysis_text = "仅录音"
        for col, text in ((5, str(info.get("duration") or "--")), (6, analysis_text)):
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, col, item)
        mode = condition["segmented_analysis"]["mode"]
        self.table.cellWidget(row, 7).status_label.setText(
            {"none": "未启用", "time": "按时间"}[mode])

    def _segments(self, condition, row):
        dialog = OutputLoadConfigDialog(condition["condition_name"], condition["segmented_analysis"],
                                        self.catalog.get(condition["test_queue"], {}).get("duration"), self)
        if dialog.exec_() == dialog.Accepted:
            condition["segmented_analysis"] = dialog.settings()
            self._summary(condition, row)
            self.changed.emit()
        dialog.deleteLater()

    def _add(self):
        self.project["conditions"].append(new_condition(len(self.project["conditions"]) + 1))
        self.set_project(self.project, self.catalog)
        self.changed.emit()

    def _remove(self):
        self.project["conditions"][:] = [c for row, c in enumerate(self.project["conditions"])
                                         if self.table.item(row, 0).checkState() != Qt.Checked]
        self.set_project(self.project, self.catalog)
        self.changed.emit()

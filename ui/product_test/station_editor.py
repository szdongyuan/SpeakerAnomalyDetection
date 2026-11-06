"""Station mappings editor. Mutates only the dialog's unsaved draft."""
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton,
    QTableWidget, QTableWidgetItem, QLineEdit, QComboBox, QHeaderView, QMessageBox, QToolButton,
)
from base.product_test.config_model import new_channel, resize_stations, SIGNAL_TYPES
from consts.running_consts import DEFAULT_DIR
from ui.product_test.station_selector import _PortSelectorBar, _PortTabScrollArea
from ui.product_test.style import CONTROL_STYLE, SELECTOR_STYLE, REMOVE_BUTTON_STYLE, TABLE_STYLE
from ui.confirmation_message_box import ConfirmationMessageBox


class StationEditor(QWidget):
    changed = pyqtSignal()

    def __init__(self, hardware_provider, parent=None):
        super().__init__(parent)
        self.hardware_provider = hardware_provider
        self.project = None
        self.loading = False
        self.count = QSpinBox()
        self.count.setRange(1, 9999)
        self.count.setMinimumWidth(120)
        self.tabs = _PortSelectorBar()
        self.tabs.setObjectName("productProjectPortTabs")
        self.tabs.setMinimumHeight(36)
        self.scroll = _PortTabScrollArea(self.tabs)
        self.scroll.setObjectName("productProjectPortScrollArea")
        self.title = QLabel("通道映射")
        self.add_button = QPushButton("+ 添加通道")
        self.table = QTableWidget(0, 5)
        self.table.setObjectName("productProgramTable")
        self.table.setStyleSheet(TABLE_STYLE)
        self.table.setAlternatingRowColors(True)
        self.table.setHorizontalHeaderLabels(["WAV通道", "通道名称", "信号类型", "硬件通道", ""])
        self.table.horizontalHeaderItem(4).setToolTip("移除通道")
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Fixed)
        self.table.setColumnWidth(4, 46)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setColumnWidth(2, 180)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(44)
        self.table.horizontalHeader().setMinimumHeight(38)
        self.table.setColumnWidth(1, 150)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(9)
        bar = QHBoxLayout()
        bar.setSpacing(10)
        bar.addWidget(QLabel("工位数量："))
        bar.addWidget(self.count)
        bar.addWidget(self.scroll, 1)
        layout.addLayout(bar)
        tools = QHBoxLayout()
        tools.addWidget(self.title)
        tools.addStretch()
        tools.addWidget(self.add_button)
        layout.addLayout(tools)
        layout.addWidget(self.table)
        self.count.valueChanged.connect(self._resize)
        self.tabs.currentChanged.connect(self._show_station)
        self.tabs.tabTextEdited.connect(self._rename)
        self.add_button.clicked.connect(self._add)

    def set_project(self, project):
        self.loading = True
        self.project = project
        while self.tabs.count():
            self.tabs.removeTab(self.tabs.count() - 1)
        for station in project["stations"]:
            self.tabs.addTab(station["station_name"])
        self.count.setValue(len(project["stations"]))
        self.loading = False
        self.scroll.sync_content_width()
        self.tabs.setCurrentIndex(0)
        self._show_station(0)

    def _resize(self, count):
        if self.loading or self.project is None:
            return
        old = len(self.project["stations"])
        if count < old and QMessageBox.question(
                self, "减少工位", f"移除最后 {old-count} 个工位及其通道映射，并清理工况中的工位引用。工况本身保留。",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            self.count.blockSignals(True)
            self.count.setValue(old)
            self.count.blockSignals(False)
            return
        resize_stations(self.project, count)
        self.set_project(self.project)
        self.changed.emit()

    def _rename(self, index, name):
        if not self.loading:
            self.project["stations"][index]["station_name"] = name
            self._show_station(index)
            self.changed.emit()

    def _show_station(self, index):
        if self.loading or self.project is None or index < 0:
            return
        station = self.project["stations"][index]
        self.title.setText(f"{station['station_name']} · 通道映射")
        self.table.setRowCount(0)
        available = self.hardware_provider()
        for row, channel in enumerate(station["channels"]):
            self.table.insertRow(row)
            item = QTableWidgetItem(f"第 {row+1} 通道")
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, item)
            name = QLineEdit(channel["channel_name"])
            name.setStyleSheet(CONTROL_STYLE)
            name.textEdited.connect(lambda value, c=channel: self._set(c, "channel_name", value))
            self._set_cell_control(row, 1, name)
            signal = QComboBox()
            signal.setStyleSheet(SELECTOR_STYLE)
            signal.addItem("请选择信号类型", "")
            for key, label in SIGNAL_TYPES.items():
                signal.addItem(label, key)
            signal.setCurrentIndex(signal.findData(channel["signal_type"]))
            signal.currentIndexChanged.connect(lambda _, c=channel, combo=signal: self._set(c, "signal_type", combo.currentData()))
            self._set_cell_control(row, 2, signal)
            hardware = QComboBox()
            hardware.setStyleSheet(SELECTOR_STYLE)
            hardware.addItem("请选择已有硬件通道", {"device_ref": "", "physical_channel_index": 0})
            for entry in available:
                hardware.addItem(entry["label"], {k: entry[k] for k in ("device_ref", "physical_channel_index")})
            selected = hardware.findData(channel["hardware_channel"])
            if selected < 0:
                hw = channel["hardware_channel"]
                hardware.addItem(f"{hw['device_ref']} / 物理CH{hw['physical_channel_index']+1}（未核实）", hw.copy())
                selected = hardware.count() - 1
            hardware.setCurrentIndex(selected)
            hardware.currentIndexChanged.connect(lambda _, c=channel, combo=hardware: self._set(c, "hardware_channel", combo.currentData().copy()))
            self._set_cell_control(row, 3, hardware)
            remove = QToolButton()
            remove.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/main_window_pic/close.svg"))
            remove.setIconSize(QSize(12, 12))
            remove.setFixedWidth(28)
            remove.setToolTip(f"移除此通道：WAV 第 {row + 1} 通道")
            if len(station["channels"]) == 1:
                remove.setEnabled(False)
                remove.setToolTip("每个工位至少保留一个通道")
            remove.setAccessibleName(f"移除 WAV 第 {row + 1} 通道")
            remove.setStyleSheet(REMOVE_BUTTON_STYLE)
            remove.clicked.connect(lambda _, r=row: self._remove(r))
            self._set_cell_control(row, 4, remove)
        visible_rows = min(4, max(1, self.table.rowCount()))
        self.table.setFixedHeight(
            self.table.horizontalHeader().minimumHeight()
            + visible_rows * self.table.verticalHeader().defaultSectionSize()
            + 2 * self.table.frameWidth()
        )

    def _set_cell_control(self, row, column, control):
        cell = QWidget()
        layout = QHBoxLayout(cell)
        layout.setContentsMargins(4, 0, 4, 0)
        control.setFixedHeight(30)
        layout.addWidget(control)
        layout.setAlignment(control, Qt.AlignVCenter)
        self.table.setCellWidget(row, column, cell)

    def _set(self, channel, key, value):
        channel[key] = value
        self.changed.emit()

    def _add(self):
        index = self.tabs.currentIndex()
        channels = self.project["stations"][index]["channels"]
        channels.append(new_channel(len(channels) + 1))
        self._show_station(index)
        self.changed.emit()

    def _remove(self, row):
        index = self.tabs.currentIndex()
        station = self.project["stations"][index]
        if len(station["channels"]) <= 1:
            return
        has_queue = any(station["station_id"] in c["applicable_station_ids"] and c["test_queue"]
                        for c in self.project["conditions"])
        channel = station["channels"][row]
        message = f"移除“{station['station_name']}”的 {channel['channel_name']}（WAV 第 {row + 1} 通道）？"
        if row == len(station["channels"]) - 1:
            impact = ""
        else:
            impact = "后续 WAV 通道序号将前移一位。"
        if has_queue:
            impact = (impact + "\n" if impact else "") + "移除后，请检查测试队列的通道选择。"
        prompt = ConfirmationMessageBox(
            self, "移除通道", message, QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        prompt.setTextFormat(Qt.PlainText)
        prompt.setInformativeText(impact)
        prompt.button(QMessageBox.Yes).setText("移除通道")
        result = prompt.exec_()
        prompt.deleteLater()
        if result != QMessageBox.Yes:
            return
        del station["channels"][row]
        self._show_station(index)
        self.changed.emit()

"""Read-only model registry UI with injected data and no model execution."""

from pathlib import PureWindowsPath

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from base.ai_model_catalog import ModelCatalogError, ModelRecord, ModelRegistryReader
from ui.ai.model_dialog_base import AiModelDialogBase, ModelNameDelegate, model_button, model_table
from ui.ai.model_evaluation_dialog import AiModelEvaluationDialog


class AiModelDetailsDialog(AiModelDialogBase):
    def __init__(self, record: ModelRecord, parent=None):
        super().__init__("模型详情", parent)
        self.record = record
        self.resize(660, 570)
        self.setMinimumSize(500, 400)
        self.add_heading("模型详情")
        scroll = QScrollArea(self)
        scroll.setObjectName("aiModelDetailsScroll")
        scroll.setWidgetResizable(True)
        panel = QWidget(scroll)
        panel.setObjectName("aiModelDetailsBody")
        panel.setAttribute(Qt.WA_StyledBackground, True)
        fields = QGridLayout(panel)
        fields.setContentsMargins(24, 22, 24, 22)
        fields.setHorizontalSpacing(24)
        fields.setVerticalSpacing(18)
        fields.setColumnStretch(1, 1)
        fields.setAlignment(Qt.AlignTop)
        accuracy = "—" if record.accuracy is None else f"{record.accuracy:.2%}"
        pairs = (
            ("模型名称", record.model_name),
            ("输入维度", record.input_dim),
            ("输出维度", str(record.output_dim)),
            ("模型文件", record.model_path),
            ("模型文件情况", "文件存在" if record.model_file_exists else "文件缺失"),
            ("配套配置", record.config_path),
            ("配置文件情况", "文件存在" if record.config_file_exists else "文件缺失"),
            ("登记时间", record.update_date),
            ("历史评估准确率", accuracy),
            ("备注", record.model_description),
        )
        self.field_labels = {}
        for row, (name, value) in enumerate(pairs):
            key = QLabel(name, panel)
            key.setObjectName("aiModelMuted")
            label = QLabel(value or "—", panel)
            label.setTextFormat(Qt.PlainText)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
            label.setWordWrap(True)
            label.setMinimumWidth(0)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            fields.addWidget(key, row, 0, Qt.AlignTop)
            fields.addWidget(label, row, 1)
            self.field_labels[name] = label
        self.field_labels["历史评估准确率"].setToolTip("数据库中的历史记录，不是本次评估结果。")
        scroll.setWidget(panel)
        self.root_layout.addWidget(scroll, 1)
        footer = self.add_footer()
        footer.addStretch()
        close_button = model_button("关闭", self)
        close_button.clicked.connect(self.reject)
        footer.addWidget(close_button)


class AiModelManagerDialog(AiModelDialogBase):
    def __init__(self, model_registry: ModelRegistryReader, parent=None):
        super().__init__("AI模型管理", parent)
        self.model_registry = model_registry
        self.records: tuple[ModelRecord, ...] = ()
        self.resize(940, 600)
        self.setMinimumSize(680, 440)
        heading = self.add_heading("AI模型管理")
        self.count_label = QLabel("0 个模型", self)
        self.count_label.setObjectName("aiModelMuted")
        heading.addWidget(self.count_label)
        self._build_toolbar()
        content = QWidget(self)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        self.error_label = QLabel(content)
        self.error_label.setObjectName("aiModelError")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)
        self.table = model_table(["模型名称", "输入维度", "输出维度", "文件情况", "操作"], content)
        self.table.setItemDelegateForColumn(0, ModelNameDelegate(self.table))
        header = self.table.horizontalHeader()
        for column, width in ((1, 170), (2, 100), (3, 130), (4, 90)):
            header.setSectionResizeMode(column, QHeaderView.Interactive)
            header.resizeSection(column, width)
            self.table.horizontalHeaderItem(column).setTextAlignment(Qt.AlignCenter)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.cellDoubleClicked.connect(lambda row, _column: self.show_details(row))
        layout.addWidget(self.table, 1)
        self.root_layout.addWidget(content, 1)
        footer = self.add_footer()
        self.selection_label = QLabel("—", self)
        self.selection_label.setTextFormat(Qt.PlainText)
        self.selection_label.setObjectName("aiModelMuted")
        self.selection_label.setMinimumWidth(0)
        self.selection_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        footer.addWidget(self.selection_label, 1)
        close_button = model_button("关闭", self)
        close_button.clicked.connect(self.reject)
        footer.addWidget(close_button)
        self.ensurePolished()
        self.load_records()

    def _build_toolbar(self):
        toolbar = QWidget(self)
        toolbar.setObjectName("aiModelToolbar")
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(24, 10, 24, 10)
        layout.setSpacing(8)
        self.import_button = model_button("导入模型", toolbar)
        self.import_button.clicked.connect(
            lambda: QMessageBox.information(self, "提示", "模型导入功能尚未接入。")
        )
        self.import_button.setToolTip("模型导入功能尚未接入。")
        self.evaluate_button = model_button("模型评估", toolbar)
        self.evaluate_button.clicked.connect(self.show_evaluation)
        self.delete_button = model_button("删除模型", toolbar)
        self.delete_button.clicked.connect(
            lambda: QMessageBox.information(self, "提示", "模型删除功能尚未接入。")
        )
        self.delete_button.setToolTip("模型删除功能尚未接入。")
        for button in (self.import_button, self.evaluate_button, self.delete_button):
            layout.addWidget(button)
        layout.addStretch()
        self.root_layout.addWidget(toolbar)

    def load_records(self):
        try:
            records = self.model_registry.list_records()
        except ModelCatalogError as error:
            self.records = ()
            self.error_label.setText(str(error))
            self.error_label.show()
            self.count_label.setText("读取失败")
        else:
            self.records = tuple(records)
            self.error_label.clear()
            self.error_label.hide()
            self.count_label.setText(f"{len(self.records)} 个模型")
        self.table.clearContents()
        self.table.setRowCount(len(self.records))
        for row, record in enumerate(self.records):
            values = (
                record.model_name,
                record.input_dim,
                str(record.output_dim),
                "文件存在" if record.model_file_exists else "文件缺失",
                "",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                if column == 0:
                    item.setData(Qt.UserRole, PureWindowsPath(record.model_path).name)
                    item.setToolTip(f"{record.model_name}\n{record.model_path}")
                else:
                    item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, column, item)
            details = model_button("详情", self.table)
            details.setObjectName("aiModelDetails")
            details.setAccessibleName(f"查看{record.model_name}详情")
            details.clicked.connect(lambda _checked=False, index=row: self.show_details(index))
            self.table.setCellWidget(row, 4, details)
            self.table.setRowHeight(row, 68)
        if self.records:
            self.table.selectRow(0)
        self._selection_changed()

    def _selection_changed(self):
        row = self.table.currentRow()
        selected = self.table.selectionModel().hasSelection() and 0 <= row < len(self.records)
        self.evaluate_button.setEnabled(selected)
        name = self.records[row].model_name if selected else "—"
        self.selection_label.setText(f"已选：{name}" if selected else "—")
        self.selection_label.setToolTip(name)

    def show_details(self, row):
        if not 0 <= row < len(self.records):
            return
        self.table.selectRow(row)
        dialog = AiModelDetailsDialog(self.records[row], self)
        dialog.exec()
        dialog.deleteLater()

    def show_evaluation(self):
        row = self.table.currentRow()
        if not self.table.selectionModel().hasSelection() or not 0 <= row < len(self.records):
            return
        dialog = AiModelEvaluationDialog(self.records[row], self)
        dialog.exec()
        dialog.deleteLater()

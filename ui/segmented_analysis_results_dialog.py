"""Compact segment result viewer alongside the existing whole-recording plots."""

from PyQt5.QtWidgets import QComboBox, QDialog, QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout

from ui.sequence.analysis_report_snapshot import build_segment_report_results
from ui.dialog_enter_policy import install_dialog_enter_policy


class SegmentedAnalysisResultsDialog(QDialog):
    def __init__(self, result, analysis_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("分段分析结果")
        self.resize(850, 430)
        self.rows = build_segment_report_results(result, analysis_config)
        layout = QVBoxLayout(self)
        voltage = result.condition_snapshot.get("input_voltage") or "/"
        layout.addWidget(QLabel(f"输入电压：{voltage}　分析图片为整段录音概览；下表为所选分段的测量结果。"))
        self.segment_selector = QComboBox()
        for row in self.rows:
            self.segment_selector.addItem(f"{row['segment_label']}　分析 {row['window_start_seconds']:g}～{row['window_end_seconds']:g} 秒")
        layout.addWidget(self.segment_selector)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["分析项", "通道", "测量值", "下限", "上限", "状态", "说明"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        self.segment_selector.currentIndexChanged.connect(self._show_segment)
        self._show_segment(0)
        install_dialog_enter_policy(self, None)

    def _show_segment(self, index):
        items = self.rows[index]["analysis_items"]
        self.table.setRowCount(len(items))
        for row, item in enumerate(items):
            values = (item["item_key"], item.get("channel_label") or item["channel_key"], f"{item['measurement']} {item['unit']}",
                      item["lower_limit"], item["upper_limit"], item["status"], item["error"])
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))

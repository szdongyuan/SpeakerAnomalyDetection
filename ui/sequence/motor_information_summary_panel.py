"""Static information-summary template, independent of testing statistics."""

from datetime import date

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from consts.ui_style_const import MOTOR_INFORMATION_SUMMARY_STYLE


class MotorInformationSummaryPanel(QWidget):
    """Show the local date and zero placeholders without history or data binding."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("motorInformationSummaryPanel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(MOTOR_INFORMATION_SUMMARY_STYLE)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)

        heading = QHBoxLayout()
        heading.setContentsMargins(12, 10, 12, 8)
        heading.setSpacing(6)
        heading.addWidget(self._label("信息汇总", "summaryTitle"))
        heading.addStretch()
        heading.addWidget(self._label(date.today().isoformat(), "summaryDate"))
        details = QPushButton("查看明细", self)
        details.setObjectName("summaryDetails")
        details.setEnabled(False)
        heading.addWidget(details)
        layout.addLayout(heading)

        totals = QHBoxLayout()
        totals.setContentsMargins(12, 0, 12, 10)
        totals.setSpacing(4)
        totals.addWidget(self._label("总数", "summaryTotalCaption"), 0, Qt.AlignBaseline)
        totals.addWidget(self._label("0", "summaryTotal"), 0, Qt.AlignBaseline)
        totals.addStretch()
        totals.addWidget(self._label("合格率", "summaryRateCaption"), 0, Qt.AlignBaseline)
        totals.addWidget(self._label("--", "summaryRate"), 0, Qt.AlignBaseline)
        layout.addLayout(totals)

        counts = QWidget(self)
        counts.setObjectName("summaryCounts")
        count_layout = QHBoxLayout(counts)
        count_layout.setContentsMargins(0, 9, 0, 9)
        count_layout.setSpacing(0)
        for index, (caption, name) in enumerate((("OK", "Ok"), ("NG", "Ng"), ("未标记", "Unmarked"))):
            column = QWidget(counts)
            column.setProperty("summarySeparated", index > 0)
            column_layout = QVBoxLayout(column)
            column_layout.setContentsMargins(0, 0, 0, 0)
            column_layout.setSpacing(3)
            column_layout.addWidget(self._label(caption, f"summary{name}Caption"), 0, Qt.AlignHCenter)
            column_layout.addWidget(self._label("0", f"summary{name}"), 0, Qt.AlignHCenter)
            count_layout.addWidget(column, 1)
        layout.addWidget(counts)

    def _label(self, text, name):
        label = QLabel(text, self)
        label.setObjectName(name)
        return label

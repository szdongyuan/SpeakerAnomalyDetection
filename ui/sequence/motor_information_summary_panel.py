"""Summary and modal details for an explicitly supplied daily record snapshot."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from consts.ui_style_const import MOTOR_INFORMATION_SUMMARY_STYLE
from base.motor_daily_records import DailyRecordStore
from ui.sequence.motor_daily_details_dialog import MotorDailyDetailsDialog


class MotorInformationSummaryPanel(QWidget):
    """Own daily records independently of station and test-round business state."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._daily_records = DailyRecordStore()
        self._details_dialog = None
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
        self.date_label = self._label(self._daily_records.record_date.isoformat(), "summaryDate")
        heading.addWidget(self.date_label)
        details = QPushButton("查看明细", self)
        details.setObjectName("summaryDetails")
        details.setCursor(Qt.PointingHandCursor)
        details.clicked.connect(self._open_details)
        heading.addWidget(details)
        layout.addLayout(heading)

        totals = QHBoxLayout()
        totals.setContentsMargins(12, 0, 12, 10)
        totals.setSpacing(4)
        totals.addWidget(self._label("总数", "summaryTotalCaption"), 0, Qt.AlignBaseline)
        self.total_label = self._label("0", "summaryTotal")
        totals.addWidget(self.total_label, 0, Qt.AlignBaseline)
        totals.addStretch()
        totals.addWidget(self._label("合格率", "summaryRateCaption"), 0, Qt.AlignBaseline)
        self.rate_label = self._label("--", "summaryRate")
        totals.addWidget(self.rate_label, 0, Qt.AlignBaseline)
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
            label = self._label("0", f"summary{name}")
            setattr(self, f"{name.lower()}_label", label)
            column_layout.addWidget(label, 0, Qt.AlignHCenter)
            count_layout.addWidget(column, 1)
        layout.addWidget(counts)

    def set_daily_records(self, records, record_date=None):
        """Replace one day's complete snapshot; validation errors leave UI intact."""
        self._daily_records.replace(records, record_date)
        summary = self._daily_records.summary
        self.date_label.setText(self._daily_records.record_date.isoformat())
        for label, value in ((self.total_label, summary.total), (self.ok_label, summary.ok),
                             (self.ng_label, summary.ng), (self.unmarked_label, summary.unmarked),
                             (self.rate_label, summary.pass_rate)):
            label.setText(str(value))
        if self._details_dialog is not None:
            self._details_dialog.refresh_records()

    def _open_details(self):
        if self._details_dialog is None:
            dialog = MotorDailyDetailsDialog(self._daily_records, self.window())
            self._details_dialog = dialog
            # Split sections can put the dialog under a different Qt owner.
            self.destroyed.connect(dialog.deleteLater)
            dialog.destroyed.connect(self._details_destroyed)
        dialog = self._details_dialog
        if not dialog.isVisible():
            dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _details_destroyed(self):
        self._details_dialog = None

    def _label(self, text, name):
        label = QLabel(text, self)
        label.setObjectName(name)
        return label

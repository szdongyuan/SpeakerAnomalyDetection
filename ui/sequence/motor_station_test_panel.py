"""Native, caller-controlled station display with an atomic public data API."""

import math

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QAbstractItemView, QHeaderView, QLabel, QPushButton, QScrollArea,
    QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from consts.ui_style_const import MOTOR_STATION_PANEL_STYLE, MOTOR_STATION_STATUS_COLORS


class _ThreeColumns:
    """Shared geometry for the general header and each station toggle."""

    def _place_columns(self):
        first = int(self.width() * .46)
        second = int(self.width() * .27)
        for label, left, width in zip(
            self._labels, (0, first, first + second),
            (first, second, self.width() - first - second),
        ):
            label.setGeometry(left, 0, width, self.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_columns()


class _GeneralHeader(_ThreeColumns, QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("stationGeneralHeader")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._labels = [QLabel(text, self) for text in ("工位", "正转", "反转")]
        for label, name in zip(self._labels, ("Station", "Forward", "Reverse")):
            label.setObjectName(f"stationGeneral{name}")
            label.setAlignment(Qt.AlignCenter)
        self._labels[0].setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self._labels[0].setContentsMargins(12, 0, 0, 0)
        self.setFixedHeight(max(30, self.fontMetrics().height() + 12))


class _StationHeader(_ThreeColumns, QPushButton):
    def __init__(self, station_id, parent=None):
        super().__init__(parent)
        self.setObjectName(f"stationHeader{station_id}")
        self.setProperty("stationHeader", True)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self._station_id = station_id
        self._labels = [QLabel(self) for _ in range(3)]
        for label in self._labels:
            label.setAttribute(Qt.WA_TransparentForMouseEvents)
            label.setAlignment(Qt.AlignCenter)
        self._labels[0].setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self._labels[0].setContentsMargins(12, 0, 0, 0)
        self._labels[0].setStyleSheet("color: #293b4d;")
        self._labels[1].setObjectName(f"stationForward{station_id}")
        self._labels[2].setObjectName(f"stationReverse{station_id}")
        self.setFixedHeight(max(34, self.fontMetrics().height() + 14))
        self.toggled.connect(self._set_expanded)
        self._set_expanded(False)
        self.set_results("--", "--")

    def _set_expanded(self, expanded):
        arrow = "▼" if expanded else "▶"
        self._labels[0].setText(f"{arrow}  工位 {self._station_id}")
        self.setAccessibleName(f"工位 {self._station_id}")
        self.setAccessibleDescription("展开" if expanded else "收起")

    def set_results(self, forward, reverse):
        for label, status in zip(self._labels[1:], (forward, reverse)):
            label.setText(status)
            # Local explicit colors override generic QWidget/QLabel ancestor styles.
            label.setStyleSheet(f"color: {MOTOR_STATION_STATUS_COLORS[status]};")


class _DetailTable(QTableWidget):
    def __init__(self, station_id, parent=None):
        super().__init__(0, 3, parent)
        self.setObjectName(f"stationTable{station_id}")
        self.setHorizontalHeaderLabels(["指标", "正转", "反转"])
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setFocusPolicy(Qt.NoFocus)
        self.verticalHeader().hide()
        self.horizontalHeader().setSectionsMovable(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.horizontalHeader().setMinimumSectionSize(0)
        self.horizontalHeader().setStretchLastSection(False)
        self.horizontalHeader().setFixedHeight(max(30, self.fontMetrics().height() + 12))
        for column in range(3):
            self.horizontalHeaderItem(column).setTextAlignment(
                Qt.AlignVCenter | (Qt.AlignLeft if column == 0 else Qt.AlignRight)
            )
        self.verticalHeader().setDefaultSectionSize(max(34, self.fontMetrics().height() + 14))
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._update_height()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        width = self.viewport().width()
        first, second = int(width * .46), int(width * .27)
        for column, size in enumerate((first, second, width - first - second)):
            self.setColumnWidth(column, size)

    def _update_height(self):
        self.setFixedHeight(self.horizontalHeader().height() + sum(self.rowHeight(row) for row in range(self.rowCount())))

    def show_rows(self, rows):
        self.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(Qt.ItemIsEnabled)
                item.setTextAlignment(Qt.AlignVCenter | (Qt.AlignLeft if column == 0 else Qt.AlignRight))
                item.setForeground(QBrush(QColor("#293b4d")))
                self.setItem(row, column, item)
        self._update_height()


class _Station(QWidget):
    def __init__(self, station_id, parent=None):
        super().__init__(parent)
        self.rows = []
        self.forward = "--"
        self.reverse = "--"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.header = _StationHeader(station_id, self)
        self.table = _DetailTable(station_id, self)
        self.table.hide()
        layout.addWidget(self.header)
        layout.addWidget(self.table)
        self.header.toggled.connect(self.table.setVisible)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def apply_rows(self, rows):
        self.rows = rows
        self.table.show_rows(rows)


class MotorStationTestPanel(QWidget):
    """Zero-default station list; call public setters on the Qt GUI thread.

    Station IDs are 1-based; row and column indices are 0-based. Setters
    validate complete batches before changing either data or presentation.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("motorStationTestPanel")
        self.setStyleSheet(MOTOR_STATION_PANEL_STYLE)
        self._stations = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        title = QLabel("工位测试", self)
        title.setObjectName("stationTitle")
        layout.addWidget(title)
        self._general_header = _GeneralHeader(self)
        layout.addWidget(self._general_header)
        self._scroll = QScrollArea(self)
        self._scroll.setObjectName("stationScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._content = QWidget()
        self._content.setObjectName("stationContent")
        self._station_layout = QVBoxLayout(self._content)
        self._station_layout.setContentsMargins(0, 0, 0, 0)
        self._station_layout.setSpacing(0)
        self._station_layout.addStretch()
        self._scroll.setWidget(self._content)
        layout.addWidget(self._scroll, 1)
        self._scroll.viewport().installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self._scroll.viewport() and event.type() == QEvent.Resize:
            # Keep the persistent header aligned when the content scrollbar appears.
            # A maximum allows the layout to shrink without retaining the old width
            # as a minimum size for this panel and its containing splitter.
            self._general_header.setMaximumWidth(self._scroll.viewport().width())
        return super().eventFilter(watched, event)

    @staticmethod
    def _integer(value, name):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer (not bool)")
        return value

    @classmethod
    def _count(cls, value, name):
        cls._integer(value, name)
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
        return value

    def _station(self, station_id):
        self._integer(station_id, "station_id")
        if not 1 <= station_id <= len(self._stations):
            raise IndexError(f"station_id {station_id} does not exist")
        return self._stations[station_id - 1]

    @classmethod
    def _index(cls, value, count, name):
        cls._integer(value, name)
        if not 0 <= value < count:
            raise IndexError(f"{name} {value} is out of range")
        return value

    @staticmethod
    def _cell(value):
        if value is None:
            return "--"
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise TypeError("cell value must be str, int, finite float or None")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("cell value must be finite")
        return str(value)

    @classmethod
    def _values(cls, values, length):
        if not isinstance(values, (list, tuple)):
            raise TypeError("values must be a list or tuple")
        if len(values) != length:
            raise ValueError(f"values must contain exactly {length} cells")
        return [cls._cell(value) for value in values]

    @staticmethod
    def _status(value, name):
        if not isinstance(value, str):
            raise TypeError(f"{name} status must be a string")
        normalized = value.strip().upper()
        if normalized not in MOTOR_STATION_STATUS_COLORS:
            raise ValueError(f"{name} status must be --, OK or NG")
        return normalized

    def set_station_count(self, count):
        """Resize the station list, preserving surviving data and fold state."""
        self._count(count, "count")
        while len(self._stations) > count:
            station = self._stations.pop()
            self._station_layout.removeWidget(station)
            station.hide()
            station.setParent(None)
            station.deleteLater()
        while len(self._stations) < count:
            station = _Station(len(self._stations) + 1, self._content)
            self._stations.append(station)
            self._station_layout.insertWidget(len(self._stations) - 1, station)

    def set_station_row_count(self, station_id, count):
        """Resize just this station's rows, initializing new cells to --."""
        station = self._station(station_id)
        self._count(count, "count")
        if count != len(station.rows):
            rows = [row[:] for row in station.rows[:count]]
            rows.extend([["--", "--", "--"] for _ in range(count - len(rows))])
            station.apply_rows(rows)

    def set_station_row(self, station_id, row, values):
        """Replace exactly three cells after validating the complete row."""
        station = self._station(station_id)
        self._index(row, len(station.rows), "row")
        normalized = self._values(values, 3)
        rows = [value[:] for value in station.rows]
        rows[row] = normalized
        station.apply_rows(rows)

    def set_station_column(self, station_id, column, values):
        """Replace a full column; a zero-row station accepts an empty sequence."""
        station = self._station(station_id)
        self._index(column, 3, "column")
        normalized = self._values(values, len(station.rows))
        rows = [row[:] for row in station.rows]
        for row, value in zip(rows, normalized):
            row[column] = value
        station.apply_rows(rows)

    def set_station_cell(self, station_id, row, column, value):
        """Replace one scalar cell without changing table shape or results."""
        station = self._station(station_id)
        self._index(row, len(station.rows), "row")
        self._index(column, 3, "column")
        normalized = self._cell(value)
        rows = [values[:] for values in station.rows]
        rows[row][column] = normalized
        station.apply_rows(rows)

    def set_station_results(self, station_id, forward, reverse):
        """Normalize both statuses before updating either direction."""
        station = self._station(station_id)
        forward = self._status(forward, "forward")
        reverse = self._status(reverse, "reverse")
        station.forward, station.reverse = forward, reverse
        station.header.set_results(forward, reverse)

    def get_station_count(self):
        """Return the number of stations explicitly created by this caller."""
        return len(self._stations)

    def get_station_data(self, station_id):
        """Return an independent snapshot of normalized display strings."""
        station = self._station(station_id)
        return {"station_id": station_id, "forward": station.forward,
                "reverse": station.reverse, "rows": [row[:] for row in station.rows]}

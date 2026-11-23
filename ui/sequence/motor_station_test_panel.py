"""Native, caller-controlled station display with an atomic public data API."""

import math

from PyQt5.QtCore import QEvent, QRect, Qt
from PyQt5.QtGui import QBrush, QColor, QPainter
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QHeaderView, QLabel, QPushButton, QScrollArea,
    QSizePolicy, QStyle, QStyleOption, QStyleOptionViewItem, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from consts.ui_style_const import (
    MOTOR_STATION_PANEL_STYLE, MOTOR_STATION_STATUS_COLORS, MOTOR_WORKBENCH_SCROLLBAR_STYLE,
)


class _ResultColumns:
    """Shared geometry for the general header and each station toggle."""

    def _place_columns(self):
        width = self.contentsRect().width()
        first = int(width * .46)
        count = len(self._labels) - 1
        first = first if count <= 3 else 160
        boundaries = [0, *(first + (width - first) * i // count for i in range(count + 1))]
        for label, left, right in zip(self._labels, boundaries, boundaries[1:]):
            label_width = right - left
            label.setGeometry(self.contentsRect().x() + left, 0, label_width, self.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_columns()


class _GeneralHeader(_ResultColumns, QWidget):
    def __init__(self, condition_names, parent=None):
        super().__init__(parent)
        self.setObjectName("stationGeneralHeader")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._labels = [QLabel(text, self) for text in ("工位", *condition_names, "结果")]
        names = ("Station", *(("Forward", "Reverse")[i] if i < 2 else f"Condition{i}"
                              for i in range(len(condition_names))), "Result")
        for label, name in zip(self._labels, names):
            label.setObjectName(f"stationGeneral{name}")
            label.setAlignment(Qt.AlignCenter)
            label.setTextFormat(Qt.PlainText)
            label.setToolTip(label.text())
            label.setWordWrap(True)
        self._labels[0].setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self._labels[0].setContentsMargins(12, 0, 0, 0)
        self.setFixedHeight(max(30, self.fontMetrics().height() + 12))


class _StationHeader(_ResultColumns, QPushButton):
    def __init__(self, station_id, condition_names, parent=None):
        super().__init__(parent)
        self.setObjectName(f"stationHeader{station_id}")
        self.setProperty("stationHeader", True)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self._labels = [QLabel(self) for _ in range(len(condition_names) + 2)]
        for label in self._labels:
            label.setAttribute(Qt.WA_TransparentForMouseEvents)
            label.setAlignment(Qt.AlignCenter)
        self._branch_left = 12
        self._branch_width = QApplication.style().pixelMetric(QStyle.PM_TreeViewIndentation)
        self._labels[0].setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self._labels[0].setContentsMargins(self._branch_left + self._branch_width, 0, 0, 0)
        self._labels[0].setTextFormat(Qt.PlainText)
        self._labels[0].setStyleSheet("color: #293b4d;")
        for index, label in enumerate(self._labels[1:-1]):
            name = ("Forward", "Reverse")[index] if index < 2 else f"Condition{index}"
            label.setObjectName(f"station{name}{station_id}")
        self._labels[-1].setObjectName(f"stationResult{station_id}")
        self.setFixedHeight(max(34, self.fontMetrics().height() + 14))
        self.toggled.connect(self._set_expanded)
        self._set_expanded(self.isChecked())
        self.set_station_name(f"工位 {station_id}")
        self.set_results(("--",) * len(condition_names), "--")

    def set_station_name(self, name):
        self._labels[0].setText(name)
        self.setAccessibleName(name)

    def _set_expanded(self, expanded):
        self.setAccessibleDescription("展开" if expanded else "收起")
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        option = QStyleOption()
        option.initFrom(self)
        option.rect = QRect(
            self._labels[0].x() + self._branch_left, 0, self._branch_width, self.height(),
        )
        option.state |= QStyle.State_Children
        if self.isChecked():
            option.state |= QStyle.State_Open
        # The button stylesheet paints the background. The application style paints tree branches.
        with QPainter(self) as painter:
            QApplication.style().drawPrimitive(QStyle.PE_IndicatorBranch, option, painter, self)

    def set_results(self, statuses, result):
        incomplete = any("结果不完整" in status for status in (*statuses, result))
        self.setFixedHeight(max(34, self.fontMetrics().height() * (3 if incomplete else 1) + 14))
        for label, status in zip(self._labels[1:], (*statuses, result)):
            label.setText(status)
            label.setWordWrap(True)
            label.setToolTip(status)
            # Local styles override ancestor rules and reset weight on every update.
            label.setStyleSheet(f"color: {MOTOR_STATION_STATUS_COLORS[status]}; font-weight: 400;")


class _DetailTable(QTableWidget):
    def __init__(self, station_id, condition_names, parent=None):
        super().__init__(0, len(condition_names) + 1, parent)
        self.setObjectName(f"stationTable{station_id}")
        self.setHorizontalHeaderLabels(["指标", *condition_names])
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setFocusPolicy(Qt.NoFocus)
        self.verticalHeader().hide()
        self.horizontalHeader().setSectionsMovable(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.horizontalHeader().setMinimumSectionSize(0)
        self.horizontalHeader().setStretchLastSection(False)
        self.horizontalHeader().setFixedHeight(max(30, self.fontMetrics().height() + 12))
        for column in range(self.columnCount()):
            self.horizontalHeaderItem(column).setToolTip(self.horizontalHeaderItem(column).text())
            self.horizontalHeaderItem(column).setTextAlignment(
                Qt.AlignVCenter | (Qt.AlignLeft if column == 0 else Qt.AlignHCenter)
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
        count = self.columnCount() - 1
        first = int(width * .40) if count else width
        if count > 2:
            first = 160
        boundaries = [0, first]
        if count:
            boundaries += [first + (width - first) * i // count for i in range(1, count + 1)]
        for column, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
            self.setColumnWidth(column, right - left)
        self._update_height()

    def _update_height(self):
        option = self.viewOptions()
        option.features |= QStyleOptionViewItem.HasDisplay
        text_margin = self.style().pixelMetric(QStyle.PM_FocusFrameHMargin, option, self)
        for row in range(self.rowCount()):
            height = self.verticalHeader().defaultSectionSize()
            for column in range(self.columnCount()):
                item = self.item(row, column)
                option.rect = QRect(0, 0, self.columnWidth(column), height)
                text_rect = self.style().subElementRect(QStyle.SE_ItemViewItemText, option, self)
                text_rect.adjust(text_margin, 0, -text_margin, 0)
                # The delegate adds text margins inside stylesheet padding; include the row border.
                bounds = option.fontMetrics.boundingRect(text_rect, int(Qt.TextWordWrap), item.text())
                height = max(height, bounds.height() + 1)
            self.setRowHeight(row, height)
        self.setFixedHeight(self.horizontalHeader().height() + sum(self.rowHeight(row) for row in range(self.rowCount())))

    def show_rows(self, rows):
        self.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(Qt.ItemIsEnabled)
                item.setTextAlignment(Qt.AlignVCenter | (Qt.AlignLeft if column == 0 else Qt.AlignHCenter))
                item.setForeground(QBrush(QColor("#293b4d")))
                self.setItem(row, column, item)
        self._update_height()


class _Station(QWidget):
    def __init__(self, station_id, condition_names, parent=None):
        super().__init__(parent)
        self.rows = []
        self.forward = "--"
        self.reverse = "--"
        self.statuses = ("--",) * len(condition_names)
        self.result = "--"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.header = _StationHeader(station_id, condition_names, self)
        self.table = _DetailTable(station_id, condition_names, self)
        self.table.hide()
        layout.addWidget(self.header)
        detail_layout = QVBoxLayout()
        detail_layout.setContentsMargins(9, 0, 9, 0)
        detail_layout.addWidget(self.table)
        layout.addLayout(detail_layout)
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
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(MOTOR_STATION_PANEL_STYLE)
        self._stations = []
        self._condition_names = ("正转", "反转")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        title = QLabel("工位测试", self)
        title.setObjectName("stationTitle")
        layout.addWidget(title)
        self._header_viewport = QWidget(self)
        self._general_header = _GeneralHeader(self._condition_names, self._header_viewport)
        self._header_viewport.setFixedHeight(self._general_header.height())
        layout.addWidget(self._header_viewport)
        self._scroll = QScrollArea(self)
        self._scroll.setObjectName("stationScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.horizontalScrollBar().valueChanged.connect(self._align_header)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scrollbar = self._scroll.verticalScrollBar()
        scrollbar.setObjectName("workbenchStationVerticalScrollBar")
        scrollbar.setStyleSheet(MOTOR_WORKBENCH_SCROLLBAR_STYLE)
        self._content = QWidget()
        self._content.setObjectName("stationContent")
        self._station_layout = QVBoxLayout(self._content)
        self._station_layout.setContentsMargins(0, 0, 0, 0)
        self._station_layout.setSpacing(0)
        self._station_layout.addStretch()
        self._scroll.setWidget(self._content)
        layout.addWidget(self._scroll, 1)
        self._scroll.viewport().installEventFilter(self)
        self._header_viewport.installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched in (self._scroll.viewport(), self._header_viewport) and event.type() == QEvent.Resize:
            # Paint the full header; align only its labels with the viewport.
            self._align_header()
        return super().eventFilter(watched, event)

    def _align_header(self):
        viewport = self._scroll.viewport()
        width = max(self._header_viewport.width(), self._content.minimumWidth() + 2)
        self._general_header.setGeometry(
            -self._scroll.horizontalScrollBar().value(), 0, width, self._header_viewport.height(),
        )
        self._general_header.setContentsMargins(
            viewport.x(), 0, max(1, width - max(viewport.width(), self._content.minimumWidth()) - viewport.x()), 0,
        )
        self._general_header._place_columns()

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
    def _cell(value, name="cell value"):
        if value is None:
            return "--"
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise TypeError(f"{name} must be str, int, finite float or None")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return str(value)

    @classmethod
    def _values(cls, values, length, name="values"):
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"{name} must be a list or tuple")
        if len(values) != length:
            raise ValueError(f"{name} must contain exactly {length} cells")
        return [cls._cell(value, f"{name}[{index}]") for index, value in enumerate(values)]

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
            station = _Station(len(self._stations) + 1, self._condition_names, self._content)
            self._stations.append(station)
            self._station_layout.insertWidget(len(self._stations) - 1, station)

    def show_workbench_stations(self, stations, *, condition_names=("正转", "反转")):
        """Apply (station name, metric names) pairs with fresh result placeholders."""
        if not isinstance(stations, (list, tuple)):
            raise TypeError("stations must be a list or tuple")
        if not isinstance(condition_names, (list, tuple)) or not all(
                isinstance(name, str) and name.strip() for name in condition_names):
            raise ValueError("condition_names must contain nonblank strings")
        display = []
        for index, entry in enumerate(stations):
            field = f"stations[{index}]"
            if not isinstance(entry, (list, tuple)):
                raise TypeError(f"{field} must be a name and metrics pair")
            if len(entry) != 2:
                raise ValueError(f"{field} must contain a name and metrics")
            name, metrics = entry
            if not isinstance(name, str):
                raise TypeError(f"{field}.name must be a string")
            if not name.strip():
                raise ValueError(f"{field}.name must not be blank")
            if not isinstance(metrics, (list, tuple)):
                raise TypeError(f"{field}.metrics must be a list or tuple")
            rows = []
            for metric_index, metric in enumerate(metrics):
                if not isinstance(metric, str):
                    raise TypeError(f"{field}.metrics[{metric_index}] must be a string")
                rows.append([metric, *(["--"] * len(condition_names))])
            display.append((name, rows))

        if tuple(condition_names) != self._condition_names:
            self.set_station_count(0)
            self._condition_names = tuple(condition_names)
            self._general_header.hide()
            self._general_header.setParent(None)
            self._general_header.deleteLater()
            self._general_header = _GeneralHeader(self._condition_names, self._header_viewport)
            self._general_header.show()
            count = len(self._condition_names)
            self._content.setMinimumWidth(160 + 90 * (count + 1) if count > 2 else 0)
            self._align_header()
        self.set_station_count(len(display))
        for station_id, (name, rows) in enumerate(display, start=1):
            station = self._station(station_id)
            station.header.set_station_name(name)
            station.apply_rows(rows)
            self.show_condition_result(station_id, ("--",) * len(self._condition_names), "--", rows)

    def set_station_row_count(self, station_id, count):
        """Resize just this station's rows, initializing new cells to --."""
        station = self._station(station_id)
        self._count(count, "count")
        if count != len(station.rows):
            rows = [row[:] for row in station.rows[:count]]
            rows.extend([["--"] * (len(self._condition_names) + 1) for _ in range(count - len(rows))])
            station.apply_rows(rows)

    def set_station_row(self, station_id, row, values):
        """Replace one row in the configured condition-column shape."""
        station = self._station(station_id)
        self._index(row, len(station.rows), "row")
        normalized = self._values(values, len(self._condition_names) + 1)
        rows = [value[:] for value in station.rows]
        rows[row] = normalized
        station.apply_rows(rows)

    def set_station_column(self, station_id, column, values):
        """Replace a full column; a zero-row station accepts an empty sequence."""
        station = self._station(station_id)
        self._index(column, len(self._condition_names) + 1, "column")
        normalized = self._values(values, len(station.rows))
        rows = [row[:] for row in station.rows]
        for row, value in zip(rows, normalized):
            row[column] = value
        station.apply_rows(rows)

    def set_station_cell(self, station_id, row, column, value):
        """Replace one scalar cell without changing table shape or results."""
        station = self._station(station_id)
        self._index(row, len(station.rows), "row")
        self._index(column, len(self._condition_names) + 1, "column")
        normalized = self._cell(value)
        rows = [values[:] for values in station.rows]
        rows[row][column] = normalized
        station.apply_rows(rows)

    def set_station_results(self, station_id, *statuses, result="--"):
        """Normalize all statuses before updating the station header."""
        station = self._station(station_id)
        self.show_condition_result(station_id, statuses, result, station.rows)

    def show_station_result(self, station_id, forward, reverse, result, rows):
        """Two-condition entry point retained for existing panel callers."""
        self.show_condition_result(station_id, (forward, reverse), result, rows)

    def show_condition_result(self, station_id, statuses, result, rows):
        """Apply an atomic snapshot in the configured condition-column order."""
        station = self._station(station_id)
        field = f"station_id {station_id}"
        if not isinstance(statuses, (list, tuple)):
            raise TypeError(f"{field} statuses must be a list or tuple")
        if len(statuses) != len(self._condition_names):
            raise ValueError(f"{field} statuses must contain exactly {len(self._condition_names)} values")
        statuses = tuple(self._status(value, f"{field} condition") for value in statuses)
        result = self._status(result, f"{field} result")
        if not isinstance(rows, (list, tuple)):
            raise TypeError(f"{field} rows must be a list or tuple")
        if len(rows) != len(station.rows):
            raise ValueError(f"{field} rows must contain exactly {len(station.rows)} rows")
        normalized = [self._values(row, len(self._condition_names) + 1, f"{field} rows[{index}]")
                      for index, row in enumerate(rows)]

        station.apply_rows(normalized)
        station.statuses, station.result = statuses, result
        station.forward = statuses[0] if statuses else "--"
        station.reverse = statuses[1] if len(statuses) > 1 else "--"
        station.header.set_results(statuses, result)

    def get_station_count(self):
        """Return the number of stations explicitly created by this caller."""
        return len(self._stations)

    def get_station_data(self, station_id):
        """Return an independent snapshot of normalized display strings."""
        station = self._station(station_id)
        return {"station_id": station_id, "forward": station.forward,
                "reverse": station.reverse, "result": station.result, "rows": [row[:] for row in station.rows]}

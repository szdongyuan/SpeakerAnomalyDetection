"""Caller-fed plot grid with fixed, horizontally synchronized column controls."""

from numbers import Integral

from PyQt5.QtCore import QEvent, QSignalBlocker, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QStyle, QVBoxLayout, QWidget,
)

from consts.ui_style_const import toolbar_combobox_style
from ui.sequence.station_plot_cell import StationPlotCell


class _ElidedLabel(QLabel):
    def set_full_text(self, text):
        self.setToolTip(text)
        self._elide()

    def _elide(self):
        self.setText(self.fontMetrics().elidedText(self.toolTip(), Qt.ElideRight, self.width()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()


class _ColumnComboBox(QComboBox):
    """Keep user data in Python: QVariant would copy dict/list payloads."""

    def __init__(self, parent):
        super().__init__(parent)
        self.options = []

    def itemData(self, index, role=Qt.UserRole):
        if role == Qt.UserRole:
            return self.options[index][1] if 0 <= index < len(self.options) else None
        return super().itemData(index, role)

    def currentData(self, role=Qt.UserRole):
        return self.itemData(self.currentIndex(), role)


class _ColumnControls(QWidget):
    def __init__(self, column, marks_visible, parent):
        super().__init__(parent)
        self.setObjectName(f'stationColumn{column}')
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.name = _ElidedLabel(self)
        self.name.setObjectName(f'stationName{column}')
        self.name.setMinimumWidth(0)
        self.name.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.name.set_full_text(f'工位 {column + 1}')
        self.combo = _ColumnComboBox(self)
        self.combo.setObjectName(f'stationOptions{column}')
        self.combo.setMinimumWidth(40)
        self.combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.combo.setStyleSheet(toolbar_combobox_style + '''
            QComboBox {
                min-width: 0px; min-height: 0px; max-width: 16777215px;
                font-size: 12px; padding: 3px 18px 3px 3px;
            }
            QComboBox QAbstractItemView { font-size: 12px; }
        ''')
        self.combo.setEnabled(False)
        self.play = QPushButton(self)
        self.play.setObjectName('stationPlay')
        self.play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play.setToolTip('播放')
        self.ok = QPushButton('OK', self)
        self.ok.setObjectName('stationOK')
        self.ng = QPushButton('NG', self)
        self.ng.setObjectName('stationNG')
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)
        layout.addWidget(self.name, 2)
        layout.addWidget(self.combo, 3)
        for button in (self.play, self.ok, self.ng):
            button.setFixedSize(30, 28)
            layout.addWidget(button)
        self.ok.setVisible(marks_visible)
        self.ng.setVisible(marks_visible)


class StationPlotGrid(QWidget):
    """All setters run on the GUI thread; no recording or business connections."""

    column_selection_changed = pyqtSignal(int, int, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('stationPlotGrid')
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._rows = 0
        self._columns = 0
        self._cells = {}
        self._headers = {}
        self._marks_visible = True
        self._resizing_contents = False
        self.setStyleSheet('''
            QWidget#stationPlotGrid {
                background: white; color: #183957; border: 1px solid #cddaea;
            }
            QWidget#stationPlotGrid QWidget { background: white; color: #183957; border: 0; }
            QWidget#stationPlotGrid QLabel { background: transparent; color: #183957; }
            QWidget#stationPlotGrid QLabel#stationPlotTitle {
                background: #3e75ac; color: white; padding-left: 10px;
            }
            QWidget#stationPlotGrid QWidget#stationHeaderRow { background: #f4f7fc; }
            QWidget#stationPlotGrid QScrollArea { border: 0; background: white; }
            QWidget#stationPlotGrid QPushButton {
                min-width: 28px; max-width: 28px;
                min-height: 26px; max-height: 26px; padding: 0px;
                border: 1px solid #CAD7E7; border-radius: 4px;
                background: #FFFFFF; color: #3B6FB6;
            }
            QWidget#stationPlotGrid QPushButton:hover { background: #EDF4FD; }
            QWidget#stationPlotGrid QPushButton:pressed { background: #DCEAFB; }
            QWidget#stationPlotGrid QPushButton#stationOK { color: #25825D; }
            QWidget#stationPlotGrid QPushButton#stationNG { color: #C24C55; }
        ''')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        title = QLabel('分析结果', self)
        title.setObjectName('stationPlotTitle')
        title.setMinimumHeight(36)
        title_font = title.font()
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        self._header_row = QWidget(self)
        self._header_row.setObjectName('stationHeaderRow')
        self._header_row.setAttribute(Qt.WA_StyledBackground, True)
        self._header_row_layout = QHBoxLayout(self._header_row)
        self._header_row_layout.setContentsMargins(0, 0, 0, 0)
        self._header_row_layout.setSpacing(0)
        self._header_scroll = QScrollArea(self._header_row)
        self._header_scroll.setObjectName('stationHeaderScroll')
        self._header_scroll.setFrameShape(QFrame.NoFrame)
        self._header_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._header_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._header_scroll.setFixedHeight(40)
        self._header_content = QWidget()
        self._header_layout = QGridLayout(self._header_content)
        self._header_layout.setContentsMargins(0, 0, 0, 0)
        self._header_layout.setSpacing(0)
        self._header_scroll.setWidget(self._header_content)
        self._header_row_layout.addWidget(self._header_scroll)
        layout.addWidget(self._header_row)

        self._scroll = QScrollArea(self)
        self._scroll.setObjectName('stationPlotScroll')
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._content = QWidget()
        self._plot_layout = QGridLayout(self._content)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        self._plot_layout.setSpacing(0)
        self._scroll.setWidget(self._content)
        layout.addWidget(self._scroll, 1)
        self._scroll.viewport().installEventFilter(self)
        self._scroll.horizontalScrollBar().valueChanged.connect(
            self._header_scroll.horizontalScrollBar().setValue)
        self._header_scroll.horizontalScrollBar().valueChanged.connect(
            self._scroll.horizontalScrollBar().setValue)
        self.set_grid_size(2, 3)

    @staticmethod
    def _integer(value, context, field):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f'{context}: {field} must be an integer, excluding bool')
        return int(value)

    def _column(self, column, method):
        column = self._integer(column, method, 'column')
        if not 0 <= column < self._columns:
            raise IndexError(f'{method}: column {column} does not exist')
        return column

    def _cell(self, row, column, method):
        row = self._integer(row, method, 'row')
        column = self._column(column, method)
        if not 0 <= row < self._rows:
            raise IndexError(f'{method}: cell ({row}, {column}) row does not exist')
        return self._cells[row, column]

    @classmethod
    def _option_index(cls, index, count, context, field):
        index = cls._integer(index, context, field)
        if index < -1 or index >= count:
            raise IndexError(f'{context}: {field} {index} is outside the options')
        return index

    def set_grid_size(self, rows, columns):
        rows = self._integer(rows, 'set_grid_size', 'rows')
        columns = self._integer(columns, 'set_grid_size', 'columns')
        if not 1 <= rows <= 3:
            raise ValueError('set_grid_size: rows must be between 1 and 3')
        if not 1 <= columns <= 4:
            raise ValueError('set_grid_size: columns must be between 1 and 4')
        if (rows, columns) == (self._rows, self._columns):
            return
        for key in list(self._cells):
            if key[0] >= rows or key[1] >= columns:
                cell = self._cells.pop(key)
                self._plot_layout.removeWidget(cell)
                cell.clear_plot()
                # pyqtgraph.close detaches the widget from its Qt parent.
                cell.plot_widget.close()
                cell.plot_widget.deleteLater()
                cell.hide()
                cell.setParent(None)
                cell.deleteLater()
        for column in list(self._headers):
            if column >= columns:
                header = self._headers.pop(column)
                header.combo.currentIndexChanged.disconnect()
                self._header_layout.removeWidget(header)
                header.hide()
                header.setParent(None)
                header.deleteLater()
        for column in range(max(columns, self._columns)):
            self._header_layout.setColumnStretch(column, 1 if column < columns else 0)
            self._plot_layout.setColumnStretch(column, 1 if column < columns else 0)
        for row in range(max(rows, self._rows)):
            self._plot_layout.setRowStretch(row, 1 if row < rows else 0)
        for column in range(columns):
            if column not in self._headers:
                header = _ColumnControls(column, self._marks_visible, self._header_content)
                self._headers[column] = header
                header.combo.currentIndexChanged.connect(
                    lambda index, c=column: self._selection_changed(c, index))
                self._header_layout.addWidget(header, 0, column)
                header.show()
            for row in range(rows):
                if (row, column) not in self._cells:
                    cell = StationPlotCell(self._content, row=row, column=column)
                    cell.setObjectName(f'stationPlotCell{row}_{column}')
                    cell.setAttribute(Qt.WA_StyledBackground, True)
                    self._cells[row, column] = cell
                    self._plot_layout.addWidget(cell, row, column)
                    cell.show()
        self._rows, self._columns = rows, columns
        # Each shared edge has exactly one owner: the cell above/to its left.
        # The enclosing widget owns the outer edges, including scrollbar gutters.
        for column, header in self._headers.items():
            right = 1 if column < columns - 1 else 0
            header.setStyleSheet(f'''
                QWidget#{header.objectName()} {{
                    background: #f4f7fc; border: 0;
                    border-right: {right}px solid #cddaea;
                    border-bottom: 1px solid #cddaea;
                }}
            ''')
        for (row, column), cell in self._cells.items():
            right = 1 if column < columns - 1 else 0
            bottom = 1 if row < rows - 1 else 0
            # QWidget layouts do not reserve QSS borders automatically. Keep the
            # opaque PlotWidget inside them so its white surface cannot cover a line.
            cell.layout().setContentsMargins(0, 0, right, bottom)
            cell.setStyleSheet(f'''
                QWidget#{cell.objectName()} {{
                    background: white; border: 0;
                    border-right: {right}px solid #cddaea;
                    border-bottom: {bottom}px solid #cddaea;
                }}
            ''')
        self._resize_contents()

    def set_station_name(self, column, name):
        column = self._column(column, 'set_station_name')
        if not isinstance(name, str):
            raise TypeError(f'set_station_name column {column}: name must be a string')
        self._headers[column].name.set_full_text(name)

    def set_column_options(self, column, options, current_index=None):
        column = self._column(column, 'set_column_options')
        context = f'set_column_options column {column}'
        if not isinstance(options, (list, tuple)):
            raise TypeError(f'{context}: options must be a list or tuple')
        snapshot = []
        for index, option in enumerate(options):
            field = f'options[{index}]'
            if not isinstance(option, (list, tuple)):
                raise TypeError(f'{context}: {field} must be a text/user_data pair')
            if len(option) != 2:
                raise ValueError(f'{context}: {field} must contain text and user_data')
            if not isinstance(option[0], str):
                raise TypeError(f'{context}: {field}.text must be a string')
            snapshot.append((option[0], option[1]))
        if current_index is None:
            current_index = 0 if snapshot else -1
        current_index = self._option_index(current_index, len(snapshot), context, 'current_index')
        box = self._headers[column].combo
        blocker = QSignalBlocker(box)
        box.clear()
        box.options = snapshot
        box.addItems([text for text, data in snapshot])
        box.setCurrentIndex(current_index)
        box.setEnabled(bool(snapshot))
        del blocker

    def set_column_current_index(self, column, index):
        column = self._column(column, 'set_column_current_index')
        box = self._headers[column].combo
        index = self._option_index(index, box.count(), f'set_column_current_index column {column}', 'index')
        blocker = QSignalBlocker(box)
        box.setCurrentIndex(index)
        del blocker

    def _selection_changed(self, column, index):
        self.column_selection_changed.emit(column, index, self._headers[column].combo.currentData())

    def set_mark_buttons_visible(self, visible):
        if not isinstance(visible, bool):
            raise TypeError('set_mark_buttons_visible: visible must be a bool')
        if visible == self._marks_visible:
            return
        self._marks_visible = visible
        for header in self._headers.values():
            header.ok.setVisible(visible)
            header.ng.setVisible(visible)

    def configure_plot(self, row, column, *, kind, title='', right_text='', x_label='', y_label='',
                       x_unit='', y_unit='', x_range=None, y_range=None, subheader_left_text='',
                       subheader_first_half_text='', subheader_second_half_text=''):
        self._cell(row, column, 'configure_plot').configure_plot(
            kind=kind, title=title, right_text=right_text, x_label=x_label, y_label=y_label, x_unit=x_unit,
            y_unit=y_unit, x_range=x_range, y_range=y_range,
            subheader_left_text=subheader_left_text, subheader_first_half_text=subheader_first_half_text,
            subheader_second_half_text=subheader_second_half_text)

    def set_plot_header(self, row, column, left_text='', right_text=''):
        self._cell(row, column, 'set_plot_header').set_plot_header(left_text, right_text)

    def set_plot_subheader(self, row, column, left_text='', first_half_text='', second_half_text=''):
        self._cell(row, column, 'set_plot_subheader').set_plot_subheader(
            left_text, first_half_text, second_half_text)

    def set_curve_data(self, row, column, series):
        self._cell(row, column, 'set_curve_data').set_curve_data(series)

    def set_heatmap_data(self, row, column, values, *, extent=None, levels=None, colormap='viridis'):
        self._cell(row, column, 'set_heatmap_data').set_heatmap_data(
            values, extent=extent, levels=levels, colormap=colormap)

    def clear_plot(self, row, column):
        self._cell(row, column, 'clear_plot').clear_plot()

    def clear_plots(self):
        for cell in self._cells.values():
            cell.clear_plot()

    def eventFilter(self, watched, event):
        if watched is self._scroll.viewport() and event.type() == QEvent.Resize:
            self._resize_contents()
        return super().eventFilter(watched, event)

    def _resize_contents(self):
        if self._resizing_contents or not self._rows:
            return
        self._resizing_contents = True
        viewport = self._scroll.viewport()
        width = max(self._columns * 280, viewport.width())
        height = max(self._rows * 180, viewport.height())
        self._content.setFixedSize(width, height)
        # Leave the same gutter as the body's vertical scrollbar. Header and
        # body then have identical viewport widths and horizontal scroll ranges.
        gutter = self._scroll.width() - viewport.width()
        self._header_row_layout.setContentsMargins(0, 0, gutter, 0)
        self._header_content.setFixedSize(width, 40)
        self._header_scroll.horizontalScrollBar().setValue(self._scroll.horizontalScrollBar().value())
        self._resizing_contents = False

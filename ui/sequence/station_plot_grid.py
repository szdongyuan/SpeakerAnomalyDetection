"""Caller-fed plot grid with fixed, horizontally synchronized column controls."""

from dataclasses import dataclass, fields
from html import escape
from numbers import Integral

from PyQt5.QtCore import QEvent, QSignalBlocker, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QStyle, QToolTip, QVBoxLayout, QWidget,
)

from consts.running_consts import DEFAULT_DIR
from consts.ui_style_const import (
    MAIN_UI_SMALL_FONT_FAMILY, TOOLBAR_ICON_BUTTON_HEIGHT, TOOLBAR_ICON_BUTTON_WIDTH,
    main_window_tooltip_style, toolbar_combobox_style,
)
from ui.sequence.station_plot_cell import StationPlotCell


_CONTROL_VERTICAL_PADDING = 8
_CONTROL_ROW_HEIGHT = TOOLBAR_ICON_BUTTON_HEIGHT + 2 * _CONTROL_VERTICAL_PADDING


@dataclass(frozen=True)
class WorkbenchPlotCellDisplay:
    title: str = ''
    unit: str = ''
    first_condition: str = ''
    second_condition: str = ''
    warning: str = ''
    warning_detail: str = ''


class _ElidedLabel(QLabel):
    def set_full_text(self, text, *, plain_text=False):
        self.setTextFormat(Qt.PlainText if plain_text else Qt.AutoText)
        self.setToolTip(text)
        self._elide()

    def _elide(self):
        self.setText(self.fontMetrics().elidedText(self.toolTip(), Qt.ElideRight, self.width()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def event(self, event):
        if event.type() == QEvent.ToolTip and self.textFormat() == Qt.PlainText and self.toolTip():
            # Qt's popup interprets markup independently of the label's text format.
            text = escape(self.toolTip())
            QToolTip.showText(event.globalPos(),
                              f'<qt><p style="white-space: pre-wrap;">{text}</p></qt>', self)
            return True
        return super().event(event)


class _ColumnComboBox(QComboBox):
    """Keep user data in Python: QVariant would copy dict/list payloads."""

    def __init__(self, parent):
        super().__init__(parent)
        self.options = []
        self._control_width = 40

    def set_control_width(self, width):
        if width != self._control_width:
            self._control_width = width
            self.updateGeometry()

    def sizeHint(self):
        return QSize(self._control_width, 30)

    def minimumSizeHint(self):
        return self.sizeHint()

    def showPopup(self):
        # Qt may defer showing its popup window until the animation finishes.
        self.view().window().installEventFilter(self)
        super().showPopup()

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Show and watched is self.view().window():
            self._resize_popup(watched)
        return super().eventFilter(watched, event)

    def _resize_popup(self, popup):
        # The native popup can follow the narrow closed control. Give its
        # items room to show their text, within the screen that owns the combo.
        view = self.view()
        bounds = popup.geometry()
        screen = self.screen().availableGeometry()
        chrome_width = bounds.width() - view.viewport().width()
        width = min(screen.width(), max(bounds.width(), view.sizeHintForColumn(0) + chrome_width))
        left = bounds.right() - width + 1 if self.isRightToLeft() else bounds.left()
        bounds.setWidth(width)
        bounds.moveLeft(max(screen.left(), min(left, screen.right() - width + 1)))
        popup.setGeometry(bounds)

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
        self.combo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.combo.setStyleSheet(toolbar_combobox_style + '''
            QComboBox {
                min-width: 0px; max-width: 16777215px;
                min-height: 0px; max-height: 16777215px;
                font-size: 12px; padding: 3px 18px 3px 3px;
            }
            QComboBox QAbstractItemView { font-size: 12px; }
        ''')
        self.combo.setEnabled(False)
        self.play = QPushButton(self)
        self.play.setObjectName('stationPlay')
        play_pixmap = QPixmap(DEFAULT_DIR + 'ui/ui_pic/sequence_pic/b_play.png')
        if play_pixmap.isNull():
            self.play.setIcon(self.play.style().standardIcon(QStyle.SP_MediaPlay))
        else:
            self.play.setIcon(QIcon(play_pixmap))
        self.play.setIconSize(QSize(28, 28))
        self.play.setToolTip('播放')
        self.ok = QPushButton('OK', self)
        self.ok.setObjectName('stationOK')
        self.ng = QPushButton('NG', self)
        self.ng.setObjectName('stationNG')
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, _CONTROL_VERTICAL_PADDING, 6, _CONTROL_VERTICAL_PADDING)
        layout.setSpacing(4)
        layout.addWidget(self.name, 1)
        layout.addWidget(self.combo)
        for button in (self.play, self.ok, self.ng):
            button.setFixedSize(TOOLBAR_ICON_BUTTON_WIDTH, TOOLBAR_ICON_BUTTON_HEIGHT)
            layout.addWidget(button)
        self.set_mark_buttons_visible(marks_visible)

    def set_mark_buttons_visible(self, visible):
        self.ok.setVisible(visible)
        self.ng.setVisible(visible)
        self._resize_combo()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_combo()

    def _resize_combo(self):
        # The old combo took 3/5 of the width left after buttons, margins and
        # spacing. At the 280 px column minimum, half of that still leaves
        # room for all three larger buttons. The name gets the rest.
        old_controls_width = 50 if self.ok.isHidden() else 118
        self.combo.set_control_width(max(40, (self.width() - old_controls_width) * 3 // 10))


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
        # QSS sizes cover the content box, inside the one-pixel button border.
        button_content_width = TOOLBAR_ICON_BUTTON_WIDTH - 2
        button_content_height = TOOLBAR_ICON_BUTTON_HEIGHT - 2
        self.setStyleSheet(main_window_tooltip_style + f'''
            QWidget#stationPlotGrid QLabel#stationPlotTitle,
            QWidget#stationPlotGrid QLabel[objectName^="stationName"],
            QWidget#stationPlotGrid QPushButton#stationPlay,
            QWidget#stationPlotGrid QPushButton#stationOK,
            QWidget#stationPlotGrid QPushButton#stationNG {{
                font-family: {MAIN_UI_SMALL_FONT_FAMILY};
                font-size: 20px;
            }}
            QWidget#stationPlotGrid {{
                background: white; color: #183957; border: 1px solid #cddaea;
            }}
            QWidget#stationPlotGrid QWidget {{ background: white; color: #183957; border: 0; }}
            QWidget#stationPlotGrid QLabel {{ background: transparent; color: #183957; }}
            QWidget#stationPlotGrid QLabel#stationPlotTitle {{
                background: #3e75ac; color: white; padding-left: 10px;
            }}
            QWidget#stationPlotGrid QWidget#stationHeaderRow {{ background: #f4f7fc; }}
            QWidget#stationPlotGrid QScrollArea {{ border: 0; background: white; }}
            QWidget#stationPlotGrid QPushButton {{
                min-width: {button_content_width}px; max-width: {button_content_width}px;
                min-height: {button_content_height}px; max-height: {button_content_height}px;
                padding: 0px;
                border: 1px solid #CAD7E7; border-radius: 4px;
                background: #FFFFFF; color: #3B6FB6;
            }}
            QWidget#stationPlotGrid QPushButton:hover {{ background: #EDF4FD; }}
            QWidget#stationPlotGrid QPushButton:pressed {{ background: #DCEAFB; }}
            QWidget#stationPlotGrid QPushButton#stationOK {{ color: #25825D; font-size: 18px; }}
            QWidget#stationPlotGrid QPushButton#stationNG {{ color: #C24C55; font-size: 18px; }}
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
        self._header_scroll.setFixedHeight(_CONTROL_ROW_HEIGHT)
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
        self._resize_grid(rows, columns)

    def show_workbench_columns(self, names):
        """Apply two fresh plot rows for each station, or clear the station area."""
        if not isinstance(names, (list, tuple)):
            raise TypeError('show_workbench_columns: names must be a list or tuple')
        names = tuple(names)
        if len(names) > 4:
            raise ValueError('show_workbench_columns: at most four station names are supported')
        for column, name in enumerate(names):
            if not isinstance(name, str):
                raise TypeError(f'show_workbench_columns: names[{column}] must be a string')
            if not name.strip():
                raise ValueError(f'show_workbench_columns: names[{column}] must not be blank')

        self._resize_grid(2, len(names))
        self._set_workbench_column_controls(names)
        for cell in self._cells.values():
            cell.configure_plot(kind='curve')
            cell.clear_plot()

    def show_workbench_layout(self, names, cells):
        """Replace all workbench plot contents using a fully validated display batch."""
        context = 'show_workbench_layout'
        if not isinstance(names, (list, tuple)):
            raise TypeError(f'{context}: names must be a list or tuple')
        if not isinstance(cells, (list, tuple)):
            raise TypeError(f'{context}: cells must be a list or tuple')
        names = tuple(names)
        row_count, column_count = len(cells), len(names)
        if row_count == column_count == 0:
            self._resize_grid(0, 0)
            return
        if not 1 <= row_count <= 3:
            raise ValueError(f'{context}: rows must be between 1 and 3')
        if not 1 <= column_count <= 4:
            raise ValueError(f'{context}: columns must be between 1 and 4')
        for column, name in enumerate(names):
            if not isinstance(name, str):
                raise TypeError(f'{context}: names[{column}] must be a string')
        display_fields = fields(WorkbenchPlotCellDisplay)
        rows = []
        for row_index, row in enumerate(cells):
            if not isinstance(row, (list, tuple)):
                raise TypeError(f'{context}: cells[{row_index}] must be a list or tuple')
            row = tuple(row)
            if len(row) != column_count:
                raise ValueError(f'{context}: cells[{row_index}] must have {column_count} columns')
            for column, display in enumerate(row):
                location = f'cells[{row_index}][{column}]'
                if not isinstance(display, WorkbenchPlotCellDisplay):
                    raise TypeError(f'{context}: {location} must be a WorkbenchPlotCellDisplay')
                for field in display_fields:
                    if not isinstance(getattr(display, field.name), str):
                        raise TypeError(f'{context}: {location}.{field.name} must be a string')
            rows.append(row)

        self._resize_grid(row_count, column_count)
        self._set_workbench_column_controls(names)
        empty_display = WorkbenchPlotCellDisplay()
        for (row, column), cell in self._cells.items():
            display = rows[row][column] if names[column] else empty_display
            cell.configure_plot(
                kind='curve', title=display.title, right_text=display.warning,
                right_text_detail=display.warning_detail, subheader_left_text=display.unit,
                subheader_first_half_text=display.first_condition,
                subheader_second_half_text=display.second_condition,
            )
            cell.clear_plot()

    def _set_workbench_column_controls(self, names):
        for column, name in enumerate(names):
            self._headers[column].name.set_full_text(name, plain_text=True)
            self.set_column_options(column, ())

    def _resize_grid(self, rows, columns):
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
        self._header_row.setVisible(bool(columns))
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
            header.set_mark_buttons_visible(visible)

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
        if self._resizing_contents:
            return
        self._resizing_contents = True
        viewport = self._scroll.viewport()
        width = max(self._columns * 280, viewport.width())
        height = max(self._rows * 180 if self._columns else 0, viewport.height())
        self._content.setFixedSize(width, height)
        # Leave the same gutter as the body's vertical scrollbar. Header and
        # body then have identical viewport widths and horizontal scroll ranges.
        gutter = self._scroll.width() - viewport.width()
        self._header_row_layout.setContentsMargins(0, 0, gutter, 0)
        self._header_content.setFixedSize(width, _CONTROL_ROW_HEIGHT)
        self._header_scroll.horizontalScrollBar().setValue(self._scroll.horizontalScrollBar().value())
        self._resizing_contents = False

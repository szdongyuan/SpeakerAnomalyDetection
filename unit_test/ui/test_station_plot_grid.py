from pathlib import Path

import pyqtgraph as pg
import pytest
from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QEvent, QPoint, QPointF, QSize, Qt
from PyQt5.QtGui import QIcon, QImage, QPalette, QWheelEvent
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtTest import QSignalSpy, QTest
from PyQt5.QtWidgets import (
    QComboBox, QLabel, QPushButton, QScrollArea, QStyle, QStyleOptionButton,
    QStyleOptionComboBox, QVBoxLayout, QWidget,
)

from consts.ui_style_const import COMBO_DOWN_ARROW_ICON, toolbar_combobox_style

from ui.sequence.station_plot_cell import StationPlotCell
from ui.sequence.station_plot_grid import StationPlotGrid


@pytest.fixture
def grid(qt_app):
    widget = StationPlotGrid()
    widget.resize(1100, 650)
    widget.show()
    qt_app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def cells(grid):
    return {(cell.row, cell.column): cell for cell in grid.findChildren(StationPlotCell)}


def combo(grid, column):
    return grid.findChild(QComboBox, f'stationOptions{column}')


def controls(grid, column):
    return grid.findChild(QWidget, f'stationColumn{column}')


def curves(cell):
    return cell.plot_widget.getPlotItem().listDataItems()


def test_title_and_station_colors_are_actually_painted(grid):
    title = grid.findChild(QLabel, 'stationPlotTitle')
    assert title.height() >= 36
    title_image = title.grab().toImage()
    assert title_image.pixelColor(title.width() - 3, 3).name() == '#3e75ac'
    assert title.palette().color(QPalette.WindowText).name() == '#ffffff'
    assert any(title_image.pixelColor(x, y).name() == '#ffffff'
               for x in range(min(150, title.width())) for y in range(title.height()))
    icon_size = QSize(28, 28)
    play_icon_path = Path(__file__).resolve().parents[2] / 'ui/sequence/assets/b_play.png'
    expected_icon = QIcon(str(play_icon_path)).pixmap(icon_size)
    assert not expected_icon.isNull()
    for column in range(3):
        header = controls(grid, column)
        assert header.height() == 56
        for button in (header.play, header.ok, header.ng):
            assert (button.width(), button.height()) == (48, 40)
            assert button.y() == 8
        assert header.play.icon().pixmap(icon_size).toImage() == expected_icon.toImage()
        assert header.play.iconSize() == icon_size
        assert (header.combo.height(), header.combo.y()) == (30, 13)
        for button in (header.ok, header.ng):
            option = QStyleOptionButton()
            option.initFrom(button)
            contents = button.style().subElementRect(QStyle.SE_PushButtonContents, option, button)
            text_size = button.fontMetrics().size(Qt.TextShowMnemonic, button.text())
            assert button.font().pixelSize() == 18
            assert text_size.width() <= contents.width()
            assert text_size.height() <= contents.height()
        assert header.grab().toImage().pixelColor(2, 2).name() == '#f4f7fc'
        assert header.name.palette().color(QPalette.WindowText).name() == '#183957'


def test_title_header_and_cells_form_a_continuous_grid(grid, qt_app):
    title = grid.findChild(QLabel, 'stationPlotTitle')
    header = controls(grid, 0)
    body = grid.findChild(QScrollArea, 'stationPlotScroll')
    assert title.mapTo(grid, QPoint()).y() == 1
    assert title.mapTo(grid, QPoint()).y() + title.height() == header.mapTo(grid, QPoint()).y()
    assert header.mapTo(grid, QPoint()).y() + header.height() == body.mapTo(grid, QPoint()).y()
    assert header.mapTo(grid, QPoint()).x() == 1
    for size in [(1100, 650), (530, 330)]:
        grid.resize(*size)
        qt_app.processEvents()
        all_cells = cells(grid)
        for (row, column), cell in all_cells.items():
            if column < 2:
                assert cell.geometry().right() + 1 == all_cells[row, column + 1].x()
            if row < 1:
                assert cell.geometry().bottom() + 1 == all_cells[row + 1, column].y()
        assert body.widget().width() == max(3 * 280, body.viewport().width())
        assert body.widget().height() == max(2 * 180, body.viewport().height())


def test_grid_separators_are_single_painted_pixels_and_outer_frame_is_closed(grid):
    image = grid.grab().toImage()
    assert all(image.pixelColor(x, 0).name() == '#cddaea' for x in range(grid.width()))
    assert all(image.pixelColor(x, grid.height() - 1).name() == '#cddaea'
               for x in range(grid.width()))
    assert all(image.pixelColor(x, y).name() == '#cddaea'
               for x in (0, grid.width() - 1) for y in range(grid.height()))
    all_cells = cells(grid)
    for (row, column), cell in all_cells.items():
        point = cell.mapTo(grid, QPoint())
        x, y = point.x(), point.y()
        if column < 2:
            edge = x + cell.width() - 1
            assert [image.pixelColor(edge + dx, y + cell.height() // 2).name()
                    for dx in (-1, 0, 1)] == ['#ffffff', '#cddaea', '#ffffff']
            header = controls(grid, column)
            hy = header.mapTo(grid, QPoint()).y() + 2
            assert [image.pixelColor(edge + dx, hy).name()
                    for dx in (-1, 0, 1)] == ['#f4f7fc', '#cddaea', '#f4f7fc']
        if row < 1:
            edge = y + cell.height() - 1
            assert [image.pixelColor(x + cell.width() // 2, edge + dy).name()
                    for dy in (-1, 0, 1)] == ['#ffffff', '#cddaea', '#ffffff']
    header = controls(grid, 0)
    edge = header.mapTo(grid, QPoint()).y() + header.height() - 1
    assert [image.pixelColor(3, edge + dy).name()
            for dy in (-1, 0, 1)] == ['#f4f7fc', '#cddaea', '#ffffff']


def test_grid_resize_refreshes_border_ownership_without_replacing_cells(grid, qt_app):
    kept = cells(grid)[0, 0]
    grid.set_curve_data(0, 0, [{'x': [0, 1], 'y': [2, 3]}])
    kept.plot_widget.setRange(xRange=(8, 10), yRange=(20, 30), padding=0)
    curve = curves(kept)[0]
    view = kept.plot_widget.viewRange()
    for rows, columns in [(1, 1), (3, 4), (1, 1)]:
        grid.set_grid_size(rows, columns)
        qt_app.processEvents()
        assert cells(grid)[0, 0] is kept
        assert curves(kept) == [curve]
        assert kept.plot_widget.viewRange() == view
        image = kept.grab().toImage()
        expected_right = '#cddaea' if columns > 1 else '#ffffff'
        expected_bottom = '#cddaea' if rows > 1 else '#ffffff'
        assert image.pixelColor(kept.width() - 1, 2).name() == expected_right
        assert image.pixelColor(kept.width() // 2, kept.height() - 1).name() == expected_bottom


def test_combo_reuses_project_arrow_and_renders_its_resource(grid, qt_app):
    grid.set_column_options(0, [('检测 A', None)])
    box = combo(grid, 0)
    assert toolbar_combobox_style in box.styleSheet()
    assert COMBO_DOWN_ARROW_ICON in box.styleSheet()
    assert QSvgRenderer(COMBO_DOWN_ARROW_ICON).isValid()
    assert not QIcon(COMBO_DOWN_ARROW_ICON).pixmap(10, 10).isNull()
    # Compare the actual arrow pixels with a real combo using the shared style.
    reference = QComboBox()
    try:
        reference.setStyleSheet(toolbar_combobox_style)
        reference.addItem('检测 A')
        reference.resize(max(100, box.width()), box.height())
        reference.show()
        qt_app.processEvents()
        patches = []
        for widget in (box, reference):
            option = QStyleOptionComboBox()
            widget.initStyleOption(option)
            arrow = widget.style().subControlRect(
                QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxArrow, widget)
            if widget is box:
                edit = widget.style().subControlRect(
                    QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, widget)
                assert abs(edit.center().y() - widget.rect().center().y()) <= 1
                assert abs(arrow.center().y() - widget.rect().center().y()) <= 1
            center = arrow.center()
            patches.append(widget.grab().toImage().copy(
                center.x() - 4, center.y() - 4, 10, 10).convertToFormat(QImage.Format_ARGB32))
        assert patches[0] == patches[1]
        assert any(patches[0].pixelColor(x, y).lightness() < 128
                   for x in range(10) for y in range(10))
    finally:
        reference.close()
        reference.deleteLater()


def test_default_empty_grid_and_column_controls(grid):
    assert set(cells(grid)) == {(r, c) for r in range(2) for c in range(3)}
    assert all(cell.kind == 'curve' and not curves(cell) for cell in cells(grid).values())
    assert [label.text() for label in grid.findChildren(QLabel) if label.text() == '分析结果'] == ['分析结果']
    for c in range(3):
        assert grid.findChild(QLabel, f'stationName{c}').text() == f'工位 {c + 1}'
        assert not combo(grid, c).isEnabled()
        buttons = controls(grid, c).findChildren(QPushButton)
        assert len(buttons) == 3
        assert [button.text() for button in buttons] == ['', 'OK', 'NG']
        assert not buttons[0].icon().isNull()
        assert all(button.isVisible() and not button.isCheckable() for button in buttons)


@pytest.mark.parametrize('rows,columns', [(1, 1), (1, 4), (3, 1), (3, 4)])
def test_grid_boundaries(grid, rows, columns):
    assert grid.set_grid_size(rows, columns) is None
    assert set(cells(grid)) == {(r, c) for r in range(rows) for c in range(columns)}


@pytest.mark.parametrize('rows,columns,error', [
    (True, 2, TypeError), (2, False, TypeError), (1.0, 2, TypeError),
    (2, '3', TypeError), (0, 2, ValueError), (4, 2, ValueError),
    (2, 0, ValueError), (2, 5, ValueError),
])
def test_invalid_dimensions_are_atomic(grid, rows, columns, error):
    before = cells(grid)
    with pytest.raises(error, match='set_grid_size.*(rows|columns)'):
        grid.set_grid_size(rows, columns)
    assert cells(grid) == before


def test_mixed_cells_clear_and_view_are_independent(grid):
    grid.set_curve_data(0, 0, [{'x': [0, 1], 'y': [2, 3]}])
    curve = curves(cells(grid)[0, 0])[0]
    view = cells(grid)[0, 0].plot_widget.viewRange()
    grid.configure_plot(1, 2, kind='heatmap', title='Heat', x_label='X')
    grid.set_heatmap_data(1, 2, [[1, 2, 3], [4, 5, 6]], extent=(2, 5, 10, 12))
    assert curves(cells(grid)[0, 0]) == [curve]
    assert cells(grid)[0, 0].plot_widget.viewRange() == view
    grid.clear_plot(1, 2)
    assert cells(grid)[1, 2].kind == 'heatmap'
    assert curves(cells(grid)[0, 0]) == [curve]
    grid.set_heatmap_data(1, 2, [[2]])
    grid.set_column_options(0, [('Keep', object())])
    grid.clear_plots()
    assert all(not curves(cell) for cell in cells(grid).values())
    assert not any(isinstance(item, pg.ImageItem) for item in cells(grid)[1, 2].plot_widget.getPlotItem().items)
    assert combo(grid, 0).currentText() == 'Keep'
    assert cells(grid)[1, 2].kind == 'heatmap'


def test_resize_keeps_intersection_and_releases_removed_objects(grid, qt_app):
    grid.set_curve_data(0, 0, [{'x': [0, 1], 'y': [2, 3]}])
    kept = cells(grid)[0, 0]
    kept.plot_widget.setRange(xRange=(8, 10), yRange=(20, 30), padding=0)
    view = kept.plot_widget.viewRange()
    curve = curves(kept)[0]
    removed = cells(grid)[1, 2]
    grid.configure_plot(1, 2, kind='heatmap')
    grid.set_heatmap_data(1, 2, [[1, 2]])
    removed_plot = removed.plot_widget
    removed_header = controls(grid, 2)
    grid.set_station_name(0, 'Retained')
    token = object()
    grid.set_column_options(0, [('Option', token)])
    original_combo = combo(grid, 0)
    grid.set_grid_size(2, 3)
    assert cells(grid)[0, 0] is kept
    grid.set_grid_size(1, 1)
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert sip.isdeleted(removed) and sip.isdeleted(removed_plot) and sip.isdeleted(removed_header)
    assert cells(grid)[0, 0] is kept
    assert curves(kept) == [curve]
    assert kept.plot_widget.viewRange() == view
    assert combo(grid, 0) is original_combo
    assert combo(grid, 0).currentData() is token
    grid.set_grid_size(3, 4)
    assert cells(grid)[0, 0] is kept
    assert cells(grid)[1, 2].kind == 'curve' and curves(cells(grid)[1, 2]) == []
    assert combo(grid, 2).count() == 0


def test_options_snapshot_opaque_data_and_programmatic_silence(grid):
    spy = QSignalSpy(grid.column_selection_changed)
    token = {'mutable': []}
    source = [('First', object()), ('Second', token)]
    assert grid.set_column_options(1, source, current_index=1) is None
    source[:] = [('Changed', None)]
    assert combo(grid, 1).count() == 2
    assert combo(grid, 1).currentText() == 'Second'
    assert combo(grid, 1).currentData() is token
    grid.set_column_current_index(1, -1)
    assert combo(grid, 1).currentIndex() == -1
    grid.set_column_current_index(1, 0)
    grid.set_column_options(1, [], current_index=-1)
    assert not combo(grid, 1).isEnabled()
    assert len(spy) == 0


@pytest.mark.parametrize('options,index,error', [
    (None, None, TypeError), ([('Valid', None), (8, None)], None, TypeError),
    ([('Valid', None), ('Broken',)], None, ValueError),
    ([('Valid', None), 'Broken'], None, TypeError),
    ([('Valid', None)], True, TypeError), ([('Valid', None)], 1, IndexError),
    ([], 0, IndexError),
])
def test_options_late_error_preserves_all_state(grid, options, index, error):
    first, second = object(), object()
    grid.set_column_options(0, [('Old A', first), ('Old B', second)], 1)
    spy = QSignalSpy(grid.column_selection_changed)
    with pytest.raises(error, match='set_column_options.*column 0.*(options|current_index)'):
        grid.set_column_options(0, options, index)
    box = combo(grid, 0)
    assert [box.itemText(i) for i in range(box.count())] == ['Old A', 'Old B']
    assert box.itemData(0) is first and box.currentData() is second
    assert box.currentIndex() == 1 and box.isEnabled()
    assert len(spy) == 0


@pytest.mark.parametrize('index,error', [(True, TypeError), ('0', TypeError), (-2, IndexError), (2, IndexError)])
def test_invalid_current_index_keeps_selection(grid, index, error):
    grid.set_column_options(0, [('A', None), ('B', None)], 1)
    with pytest.raises(error, match='set_column_current_index.*column 0.*index'):
        grid.set_column_current_index(0, index)
    assert combo(grid, 0).currentIndex() == 1


def test_actual_keyboard_and_popup_selection_emit_once(grid, qt_app):
    token = object()
    grid.resize(842, 650)
    qt_app.processEvents()
    options = [('Mode A', None), ('Inspection mode A left', token), ('Inspection mode A right', None)]
    grid.set_column_options(2, options)
    spy = QSignalSpy(grid.column_selection_changed)
    box = combo(grid, 2)
    closed_width = box.width()
    assert closed_width == 48
    box.setFocus()
    QTest.keyClick(box, Qt.Key_Down)
    assert len(spy) == 1 and spy[0] == [2, 1, token]
    view = box.view()
    item = view.model().index(2, 0)
    for _ in range(2):
        box.showPopup()
        qt_app.processEvents()
        viewport = view.viewport()
        text_width = max(view.fontMetrics().horizontalAdvance(text) for text, _ in options)
        assert viewport.width() >= text_width + 2
        assert box.screen().availableGeometry().contains(view.window().geometry())
        assert box.width() == closed_width
        # Qt can report an item rectangle wider than the popup viewport.
        visible_item = view.visualRect(item).intersected(viewport.rect())
        assert not visible_item.isEmpty()
        QTest.mouseClick(viewport, Qt.LeftButton, pos=visible_item.center())
        assert len(spy) == 2 and spy[1] == [2, 2, None]
        assert not view.isVisible()
    assert combo(grid, 0).currentIndex() == -1


def test_buttons_have_no_action_and_marks_hide_without_space(grid, qt_app):
    spy = QSignalSpy(grid.column_selection_changed)
    grid.set_column_options(0, [('Keep', None)])
    grid.set_curve_data(0, 0, [{'x': [0], 'y': [2]}])
    curve = curves(cells(grid)[0, 0])[0]
    box = combo(grid, 0)
    buttons = controls(grid, 0).findChildren(QPushButton)
    for button in buttons:
        QTest.mouseClick(button, Qt.LeftButton)
        assert not button.isChecked()
    header = controls(grid, 0)
    for window_width, column_width, shown, hidden in [
        (640, 280, 97, 138), (1100, 366, 149, 190), (1550, 516, 239, 280),
    ]:
        grid.resize(window_width, 650)
        qt_app.processEvents()
        assert header.width() == column_width
        assert abs(box.width() - shown / 2) <= 1
        assert (header.play.height(), header.play.y()) == (40, 8)
        assert (box.height(), box.y()) == (30, 13)
        assert header.name.geometry().right() < box.x()
        assert box.geometry().right() < header.play.x()
        name_width = header.name.width()
        assert header.width() - header.ng.geometry().right() == 7
        grid.set_mark_buttons_visible(False)
        qt_app.processEvents()
        assert abs(box.width() - hidden / 2) <= 1
        assert (header.play.height(), header.play.y()) == (40, 8)
        assert (box.height(), box.y()) == (30, 13)
        assert header.name.width() > name_width
        assert header.name.geometry().right() < box.x()
        assert box.geometry().right() < header.play.x()
        assert header.width() - header.play.geometry().right() == 7
        grid.set_mark_buttons_visible(True)
        qt_app.processEvents()
        assert abs(box.width() - shown / 2) <= 1
    before = box.width()
    grid.set_mark_buttons_visible(False)
    grid.set_mark_buttons_visible(False)
    qt_app.processEvents()
    assert buttons[0].isVisible()
    assert not buttons[1].isVisible() and not buttons[2].isVisible()
    assert box.width() > before
    assert curves(cells(grid)[0, 0]) == [curve]
    assert box.currentText() == 'Keep' and len(spy) == 0
    grid.set_grid_size(3, 4)
    assert all(not b.isVisible() for b in controls(grid, 3).findChildren(QPushButton)[1:])
    grid.set_mark_buttons_visible(True)
    qt_app.processEvents()
    assert all(b.isVisible() for b in grid.findChildren(QPushButton))
    assert len(grid.findChildren(QPushButton)) == 12


@pytest.mark.parametrize('value', [None, 0, 1, 'False', []])
def test_mark_visibility_strict_bool_and_instance_isolation(grid, qt_app, value):
    with pytest.raises(TypeError, match='set_mark_buttons_visible.*visible'):
        grid.set_mark_buttons_visible(value)
    assert all(b.isVisible() for b in grid.findChildren(QPushButton))
    other = StationPlotGrid()
    try:
        other.set_mark_buttons_visible(False)
        assert all(b.isVisible() for b in grid.findChildren(QPushButton))
    finally:
        other.deleteLater()


@pytest.mark.parametrize('method,args,error,field', [
    ('set_station_name', (-1, 'x'), IndexError, 'column'),
    ('set_station_name', (0, None), TypeError, 'name'),
    ('set_station_name', (True, 'x'), TypeError, 'column'),
    ('configure_plot', (2, 0), IndexError, 'row'),
    ('set_curve_data', (0, 3, []), IndexError, 'column'),
    ('set_heatmap_data', ('0', 0, [[1]]), TypeError, 'row'),
    ('clear_plot', (0, -1), IndexError, 'column'),
])
def test_public_index_diagnostics(grid, method, args, error, field):
    with pytest.raises(error, match=f'{method}.*{field}'):
        getattr(grid, method)(*args, **({'kind': 'curve'} if method == 'configure_plot' else {}))


@pytest.mark.parametrize('size', [(1300, 800), (530, 330), (1220, 330)])
def test_equal_cells_fixed_header_and_synchronized_scroll(grid, qt_app, size):
    grid.set_grid_size(3, 4)
    grid.configure_plot(0, 1, kind='heatmap')
    grid.set_heatmap_data(0, 1, [[1, 2], [3, 4]])
    grid.resize(*size)
    qt_app.processEvents()
    plots = grid.findChild(QScrollArea, 'stationPlotScroll')
    assert plots is not None
    widths = [cell.width() for cell in cells(grid).values()]
    heights = [cell.height() for cell in cells(grid).values()]
    assert min(widths) >= 280 and max(widths) - min(widths) <= 1
    assert min(heights) >= 180 and max(heights) - min(heights) <= 1
    header_y = controls(grid, 0).mapTo(grid, QPoint()).y()
    plots.verticalScrollBar().setValue(plots.verticalScrollBar().maximum())
    plots.horizontalScrollBar().setValue(plots.horizontalScrollBar().maximum())
    qt_app.processEvents()
    assert controls(grid, 0).mapTo(grid, QPoint()).y() == header_y
    for c in range(4):
        head = controls(grid, c)
        cell = cells(grid)[0, c]
        assert head.mapTo(grid, QPoint()).x() == cell.mapTo(grid, QPoint()).x()
        assert head.width() == cell.width()
        assert head.height() == 56
    last = cells(grid)[2, 3]
    bottom_right = last.mapTo(plots.viewport(), last.rect().bottomRight())
    assert plots.viewport().rect().contains(bottom_right)


def test_long_text_elides_without_expanding_columns_or_buttons(grid, qt_app):
    grid.set_grid_size(1, 1)
    grid.resize(320, 300)
    title = 'A very long plot title ' * 30
    station = '很长的工位名称' * 20
    grid.configure_plot(0, 0, kind='curve', title=title)
    grid.set_station_name(0, station)
    qt_app.processEvents()
    name = grid.findChild(QLabel, 'stationName0')
    assert name.toolTip() == station and name.text() != station
    header = cells(grid)[0, 0].header
    assert header.left.toolTip() == title and header.left.text() != title
    assert cells(grid)[0, 0].width() <= 320
    assert all(button.width() == 48 for button in controls(grid, 0).findChildren(QPushButton))


def test_horizontal_wheel_over_header_keeps_body_synchronized(grid, qt_app):
    grid.resize(530, 330)
    qt_app.processEvents()
    header = grid.findChild(QScrollArea, 'stationHeaderScroll')
    plots = grid.findChild(QScrollArea, 'stationPlotScroll')
    point = header.viewport().rect().center()
    event = QWheelEvent(QPointF(point), QPointF(header.viewport().mapToGlobal(point)),
                        QPoint(), QPoint(-120, 0), Qt.NoButton, Qt.NoModifier,
                        Qt.NoScrollPhase, False)
    QCoreApplication.sendEvent(header.viewport(), event)
    qt_app.processEvents()
    assert header.horizontalScrollBar().value() > 0
    assert header.horizontalScrollBar().value() == plots.horizontalScrollBar().value()
    for c in range(3):
        assert controls(grid, c).mapTo(grid, QPoint()).x() == cells(grid)[0, c].mapTo(grid, QPoint()).x()


def test_local_style_resists_host_and_global_pg_options_unchanged(qt_app):
    host = QWidget()
    host.setStyleSheet('''
        QWidget { background: black; color: red; }
        QPushButton { min-width: 150px; max-width: 150px; min-height: 80px; max-height: 80px; }
    ''')
    before = dict(pg.CONFIG_OPTIONS)
    layout = QVBoxLayout(host)
    widget = StationPlotGrid()
    layout.addWidget(widget)
    host.resize(640, 650)
    try:
        host.show()
        qt_app.processEvents()
        header = controls(widget, 0)
        assert header.width() == 280
        assert (header.combo.width(), header.combo.height(), header.combo.y()) == (48, 30, 13)
        host.setStyleSheet(host.styleSheet() + '''
            QComboBox { min-width: 150px; max-width: 150px; min-height: 20px; max-height: 20px; }
        ''')
        qt_app.processEvents()
        assert (header.combo.width(), header.combo.height(), header.combo.y()) == (48, 30, 13)
        assert header.combo.geometry().right() < header.play.x()
        assert all((b.width(), b.height()) == (48, 40) and b.minimumSize() == b.maximumSize() == b.size()
                   for b in widget.findChildren(QPushButton))
        assert controls(widget, 0).grab().toImage().pixelColor(1, 1).name() != '#000000'
        assert pg.CONFIG_OPTIONS == before
    finally:
        host.deleteLater()


def test_header_grid_retention_independence_and_column_state(grid, qt_app):
    grid.set_column_options(0, [('Keep', object())])
    box = combo(grid, 0)
    selected = box.currentData()
    spy = QSignalSpy(grid.column_selection_changed)
    grid.configure_plot(0, 0, kind='curve', title='SPL 趋势', right_text='MIC 1')
    grid.configure_plot(0, 1, kind='heatmap', title='频谱', right_text='MIC 2')
    grid.set_curve_data(0, 0, [{'x': [0, 1], 'y': [2, 3]}])
    kept = cells(grid)[0, 0]
    curve = curves(kept)[0]
    view = kept.plot_widget.viewRange()
    assert grid.set_plot_header(0, 0, 'Updated', 'MIC 3') is None
    assert kept._configuration['title'] == 'Updated'
    assert kept._configuration['right_text'] == 'MIC 3'
    assert curves(kept) == [curve] and kept.plot_widget.viewRange() == view
    assert cells(grid)[0, 1].header.right.toolTip() == 'MIC 2'
    grid.clear_plots()
    assert kept.header.left.toolTip() == 'Updated'
    assert kept.header.right.toolTip() == 'MIC 3'
    grid.set_grid_size(1, 1)
    grid.set_grid_size(3, 4)
    assert cells(grid)[0, 0] is kept
    assert kept.header.right.toolTip() == 'MIC 3'
    assert cells(grid)[0, 1].header.left.toolTip() == cells(grid)[0, 1].header.right.toolTip() == ''
    assert combo(grid, 0) is box and box.currentData() is selected and len(spy) == 0
    other = StationPlotGrid()
    try:
        other.set_plot_header(0, 0, 'Other', 'MIC 4')
        assert kept.header.left.toolTip() == 'Updated'
    finally:
        other.deleteLater()


@pytest.mark.parametrize('method,field', [
    ('configure_plot', 'title'), ('configure_plot', 'right_text'),
    ('set_plot_header', 'left_text'), ('set_plot_header', 'right_text'),
])
def test_grid_header_errors_are_atomic(grid, method, field):
    grid.configure_plot(1, 2, kind='curve', title='Old', right_text='MIC 1')
    grid.set_curve_data(1, 2, [{'x': [0, 1], 'y': [2, 3]}])
    cell = cells(grid)[1, 2]
    curve, = curves(cell)
    config, view = cell._configuration, cell.plot_widget.viewRange()
    kwargs = ({'kind': 'heatmap', 'title': 'Changed', 'right_text': 'Changed'}
              if method == 'configure_plot' else {'left_text': 'Changed', 'right_text': 'Changed'})
    kwargs[field] = None
    with pytest.raises(TypeError, match=rf'{method}.*\(1, 2\).*{field}'):
        getattr(grid, method)(1, 2, **kwargs)
    assert cell._configuration is config and cell.kind == 'curve'
    assert cell.header.left.toolTip() == 'Old' and cell.header.right.toolTip() == 'MIC 1'
    assert curves(cell) == [curve] and cell.plot_widget.viewRange() == view


@pytest.mark.parametrize('row,column,error', [(True, 0, TypeError), (0, '0', TypeError), (-1, 0, IndexError), (0, 3, IndexError)])
def test_header_setter_index_diagnostics(grid, row, column, error):
    with pytest.raises(error, match='set_plot_header.*(row|column)'):
        grid.set_plot_header(row, column, 'Left', 'Right')



def test_subheader_grid_lifecycle_isolation_and_column_state(grid, qt_app):
    grid.set_column_options(0, [('Keep', object())])
    box = combo(grid, 0)
    selected = box.currentData()
    spy = QSignalSpy(grid.column_selection_changed)
    grid.configure_plot(0, 0, kind='curve', title='Title', right_text='MIC 1', subheader_left_text='dB(A)', subheader_first_half_text='正转', subheader_second_half_text='反转')
    grid.configure_plot(0, 1, kind='heatmap', subheader_left_text='kHz')
    grid.set_curve_data(0, 0, [{'x': [0, 1], 'y': [2, 3]}])
    kept = cells(grid)[0, 0]
    curve = curves(kept)[0]
    view = kept.plot_widget.viewRange()
    assert grid.set_plot_subheader(0, 0, 'unit', 'one', 'two') is None
    assert curves(kept) == [curve] and kept.plot_widget.viewRange() == view
    assert kept.header.left.toolTip() == 'Title'
    assert cells(grid)[0, 1].subheader.left.toolTip() == 'kHz'
    grid.clear_plots()
    grid.set_grid_size(1, 1)
    grid.set_grid_size(3, 4)
    assert cells(grid)[0, 0] is kept and kept.subheader.second_half.toolTip() == 'two'
    assert all((not label.toolTip() for label in (cells(grid)[0, 1].subheader.left, cells(grid)[0, 1].subheader.first_half, cells(grid)[0, 1].subheader.second_half)))
    assert combo(grid, 0) is box and box.currentData() is selected and (len(spy) == 0)
    other = StationPlotGrid()
    try:
        other.set_plot_subheader(0, 0, 'other')
        assert kept.subheader.left.toolTip() == 'unit'
    finally:
        other.deleteLater()


@pytest.mark.parametrize('field', ['left_text', 'first_half_text', 'second_half_text'])
def test_grid_subheader_atomic_diagnostics(grid, field):
    grid.configure_plot(1, 2, kind='curve', title='Old', subheader_second_half_text='old')
    grid.set_curve_data(1, 2, [{'x': [0, 1], 'y': [2, 3]}])
    cell = cells(grid)[1, 2]
    config = cell._configuration
    curve = curves(cell)[0]
    kwargs = dict(left_text='new', first_half_text='new', second_half_text='new')
    kwargs[field] = []
    with pytest.raises(TypeError, match=f'set_plot_subheader.*\\(1, 2\\).*{field}'):
        grid.set_plot_subheader(1, 2, **kwargs)
    assert cell._configuration is config and curves(cell) == [curve]
    assert cell.subheader.second_half.toolTip() == 'old'


@pytest.mark.parametrize('row,column,error', [(True, 0, TypeError), (0, '0', TypeError), (-1, 0, IndexError), (0, 3, IndexError)])
def test_subheader_setter_index_diagnostics(grid, row, column, error):
    with pytest.raises(error, match='set_plot_subheader.*(row|column)'):
        grid.set_plot_subheader(row, column, 'unit', 'one', 'two')

from itertools import pairwise
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pytest
from PyQt5.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt
from PyQt5.QtGui import QFontDatabase, QHelpEvent, QPainter, QPicture, QPalette, QTextDocument
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QToolTip

from ui.sequence.station_plot_cell import StationPlotCell


@pytest.fixture
def cell(qt_app):
    widget = StationPlotCell(row=1, column=2)
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


@pytest.fixture
def yahei_font(qt_app):
    if 'Microsoft YaHei UI' in QFontDatabase().families():
        yield
        return
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    if not font_path.is_file():
        pytest.skip('Microsoft YaHei UI is required for the tick layout regression')
    font_id = QFontDatabase.addApplicationFont(str(font_path))
    assert font_id >= 0
    try:
        yield
    finally:
        QFontDatabase.removeApplicationFont(font_id)


def test_axes_use_main_font_after_reconfiguring(cell):
    plot = cell.plot_widget.getPlotItem()
    for kind in ("curve", "heatmap"):
        cell.configure_plot(kind=kind, x_label="Time", x_unit="s", y_label="Level", y_unit="dB")
        for name, text, unit in (("bottom", "Time", "s"), ("left", "Level", "dB")):
            axis = plot.getAxis(name)
            assert (axis.labelText, axis.labelUnits) == (text, unit)
            assert axis.textPen().color().name() == "#64748b"
            for font in (axis.label.font(), axis.style["tickFont"]):
                assert font is not None
                assert (font.family(), font.pixelSize()) == ("Microsoft YaHei UI", 16)


def data_items(cell):
    return cell.plot_widget.getPlotItem().listDataItems()


def scene_items(cell, item_type):
    return [item for item in cell.plot_widget.scene().items() if isinstance(item, item_type)]


def axis_draw_specs(axis, *, base=False):
    picture = QPicture()
    painter = QPainter(picture)
    try:
        if base:
            return pg.AxisItem.generateDrawSpecs(axis, painter)
        return axis.generateDrawSpecs(painter)
    finally:
        painter.end()


@pytest.mark.parametrize(('populated', 'x_max'), [
    (False, None), (True, 2), (False, 10), (True, 12), (True, 20), (True, 2000),
])
@pytest.mark.parametrize('with_header', [False, True])
def test_endpoint_tick_text_stays_inside_plot_viewport(yahei_font, cell, qt_app, populated, x_max, with_header):
    cell.resize(280, 180)
    cell.show()
    if populated:
        cell.set_curve_data([{'x': [0, 1, 2], 'y': [0, 1, 0]}])
    if x_max is not None:
        cell.configure_plot(kind='curve', x_range=(0, x_max), y_range=(-1, 1))
    if with_header:
        cell.set_plot_header('SPL 趋势', 'MIC 1')
    for _ in range(3):
        qt_app.processEvents()
    view = cell.plot_widget
    axis = view.getPlotItem().getAxis('bottom')
    specs = axis_draw_specs(axis)
    assert specs is not None and specs[2]
    if x_max is not None:
        assert str(x_max) in {text for _, _, text in specs[2]}
    for text_rect, flags, text in specs[2]:
        bounds = axis.mapRectToScene(text_rect)
        viewport = view.mapToScene(view.viewport().rect()).boundingRect()
        assert viewport.contains(bounds), (text, bounds, viewport)


def test_crowded_endpoint_tick_text_stays_visible_without_overlap(yahei_font, cell, qt_app):
    cell.resize(280, 180)
    cell.show()
    cell.configure_plot(kind='curve', x_label='Frequency', y_label='Level',
                        x_range=(999.995, 1000), y_range=(-100, 100))
    for _ in range(3):
        qt_app.processEvents()

    view = cell.plot_widget
    axis = view.getAxis('bottom')
    _, ticks, labels = axis_draw_specs(axis)
    _, original_ticks, original_labels = axis_draw_specs(axis, base=True)
    assert ticks == original_ticks
    assert [text for _, _, text in labels] == [text for _, _, text in original_labels]
    bounds_by_label = {text: axis.mapRectToScene(rect) for rect, _, text in labels}
    viewport = view.mapToScene(view.viewport().rect()).boundingRect()
    assert viewport.contains(bounds_by_label['1000.000'])
    assert '999.998' in bounds_by_label
    for left, right in pairwise(sorted(bounds_by_label.values(), key=lambda rect: rect.left())):
        assert left.right() <= right.left(), (left, right)


def test_overlapping_interior_tick_text_keeps_priority_and_tick_marks(yahei_font, cell, qt_app):
    cell.resize(300, 180)
    cell.show()
    cell.configure_plot(kind='curve', x_label='Frequency', x_unit='Hz', y_label='Level',
                        x_range=(999999.995, 1000000), y_range=(-100, 100))
    for _ in range(3):
        qt_app.processEvents()

    axis = cell.plot_widget.getAxis('bottom')
    _, ticks, labels = axis_draw_specs(axis)
    _, original_ticks, original_labels = axis_draw_specs(axis, base=True)
    assert ticks == original_ticks
    assert [text for _, _, text in original_labels] == ['0.999999996', '0.999999998']
    assert labels == original_labels[:1]


@pytest.mark.parametrize('kind', ['curve', 'heatmap'])
def test_long_title_keeps_axes_and_colorbar_inside_cell(cell, qt_app, kind):
    cell.resize(300, 260)
    cell.show()
    title = 'A very long title with units and measurements ' * 8
    right = 'MIC 1 / a very long microphone description ' * 8
    cell.configure_plot(kind=kind, title=title, right_text=right, x_label='Time', x_unit='s',
                        y_label='Frequency', y_unit='Hz')
    if kind == 'heatmap':
        cell.set_heatmap_data(np.arange(15).reshape(3, 5), extent=(0, 5, 100, 400))
    else:
        cell.set_curve_data([{'x': [0, 1, 2], 'y': [0, 1, 0]}])
    for width in (300, 500, 280):
        cell.resize(width, 260)
        for _ in range(3):
            qt_app.processEvents()
        view = cell.plot_widget
        plot = view.getPlotItem()
        header = cell.header
        assert header.left.toolTip() == title and header.left.text() != title
        assert header.right.toolTip() == right and header.right.text() != right
        assert not plot.titleLabel.isVisible()
        assert header.geometry().bottom() < view.y()
        assert header.rect().contains(header.left.geometry())
        assert header.rect().contains(header.right.geometry())
        assert not header.left.geometry().intersects(header.right.geometry())
        assert header.left.width() == pytest.approx(2 * header.right.width(), abs=2)
        for label in (header.left, header.right):
            assert label.fontMetrics().horizontalAdvance(label.text()) <= label.contentsRect().width()
            assert label.text() == label.fontMetrics().elidedText(
                label.toolTip(), Qt.ElideRight, label.contentsRect().width())
        items = [plot.getAxis('bottom'), plot.getAxis('left')]
        items.extend(scene_items(cell, pg.ColorBarItem))
        for item in items:
            bounds = view.mapFromScene(item.mapRectToScene(item.rect())).boundingRect()
            assert bounds.left() >= 0 and bounds.right() <= view.viewport().width(), (
                kind, width, type(item).__name__, bounds, view.viewport().rect())


def test_default_empty_curve_and_multicurve_replacement(cell):
    assert cell.kind == 'curve'
    assert data_items(cell) == []
    cell.set_curve_data([
        {'x': [2, 0], 'y': [3, 4], 'name': 'first', 'color': '#2478C8'},
        {'x': [5, 7, 8], 'y': [1, 2, 4]},
        {'x': [], 'y': []},
    ])
    curves = data_items(cell)
    assert len(curves) == 2
    np.testing.assert_array_equal(curves[0].getData()[0], [2, 0])
    np.testing.assert_array_equal(curves[1].getData()[0], [5, 7, 8])
    legends = scene_items(cell, pg.LegendItem)
    assert len(legends) == 1
    assert [label.text for sample, label in legends[0].items] == ['first', '曲线 2']
    assert curves[0].opts['pen'].color().name() == '#2478c8'
    cell.set_curve_data([{'x': [0], 'y': [8]}])
    assert len(data_items(cell)) == 1
    assert scene_items(cell, pg.LegendItem) == []


def test_curve_copies_input_and_rejects_late_error_atomically(cell):
    x, y = np.array([0., 1.]), np.array([2., 3.])
    cell.set_curve_data([{'x': x, 'y': y}])
    before = data_items(cell)[0]
    view = cell.plot_widget.viewRange()
    x[:] = 8
    y[:] = 9
    np.testing.assert_array_equal(before.getData()[0], [0, 1])
    np.testing.assert_array_equal(before.getData()[1], [2, 3])
    with pytest.raises(ValueError, match=r'set_curve_data.*\(1, 2\).*series\[1\].y'):
        cell.set_curve_data([{'x': [2], 'y': [3]}, {'x': [0], 'y': [np.nan]}])
    assert data_items(cell) == [before]
    assert cell.plot_widget.viewRange() == view


def test_config_same_kind_keeps_data_and_manual_ranges(cell):
    cell.set_curve_data([{'x': [0, 1], 'y': [2, 3]}])
    curve = data_items(cell)[0]
    cell.configure_plot(kind='curve', title='Title', x_label='Time', x_unit='s',
                        y_label='Level', y_unit='dB', x_range=(10, 20), y_range=(-2, 2))
    assert data_items(cell) == [curve]
    assert cell.plot_widget.getAxis('bottom').labelText == 'Time'
    assert cell.plot_widget.getAxis('bottom').labelUnits == 's'
    assert cell.header.left.toolTip() == 'Title'
    np.testing.assert_allclose(cell.plot_widget.viewRange(), [[10, 20], [-2, 2]])
    with pytest.raises(ValueError, match='y_range'):
        cell.configure_plot(kind='heatmap', title='invalid', y_range=(1, 1))
    assert cell.kind == 'curve'
    assert data_items(cell) == [curve]
    cell.clear_plot()
    assert data_items(cell) == []
    np.testing.assert_allclose(cell.plot_widget.viewRange(), [[10, 20], [-2, 2]])
    assert cell.header.left.toolTip() == 'Title'


def test_clear_auto_range_restores_empty_default(cell):
    cell.set_curve_data([{'x': [100, 200], 'y': [400, 500]}])
    cell.clear_plot()
    np.testing.assert_allclose(cell.plot_widget.viewRange(), [[0, 1], [0, 1]])
    assert cell.plot_widget.getViewBox().autoRangeEnabled() == [True, True]


def test_heatmap_row_major_real_extent_and_noninteractive_colorbar(cell):
    cell.configure_plot(kind='heatmap')
    values = np.array([[1., 2., 3.], [4., 5., 6.]])
    cell.set_heatmap_data(values, extent=(-3, 9, 10, 14), levels=(0, 8), colormap='plasma')
    image, = [item for item in cell.plot_widget.getPlotItem().items if isinstance(item, pg.ImageItem)]
    assert image.axisOrder == 'row-major'
    np.testing.assert_array_equal(image.image, values)
    assert image.mapToParent(QPointF(0, 0)) == QPointF(-3, 10)
    assert image.mapToParent(QPointF(3, 2)) == QPointF(9, 14)
    assert image.mapToParent(QPointF(.5, .5)).y() < image.mapToParent(QPointF(.5, 1.5)).y()
    assert not cell.plot_widget.getViewBox().state['yInverted']
    np.testing.assert_allclose(cell.plot_widget.viewRange(), [[-3, 9], [10, 14]])
    colorbar, = scene_items(cell, pg.ColorBarItem)
    assert colorbar.region is None
    assert colorbar.levels() == (0, 8)
    assert colorbar.colorMap().name == 'plasma'
    values[:] = 99
    np.testing.assert_array_equal(image.image, [[1, 2, 3], [4, 5, 6]])


@pytest.mark.parametrize('kwargs', [
    {'values': [[1], [np.nan]]}, {'values': [[1]], 'extent': (0, 0, 0, 1)},
    {'values': [[1]], 'levels': (2, 1)}, {'values': [[1]], 'colormap': 'missing-map'},
])
def test_invalid_heatmap_preserves_image_colorbar_and_view(cell, kwargs):
    cell.configure_plot(kind='heatmap', x_range=(0, 10))
    cell.set_heatmap_data([[1, 2, 3], [4, 5, 6]])
    images = scene_items(cell, pg.ImageItem)
    bars = scene_items(cell, pg.ColorBarItem)
    view = cell.plot_widget.viewRange()
    with pytest.raises(ValueError, match=r'set_heatmap_data.*\(1, 2\)'):
        cell.set_heatmap_data(**kwargs)
    assert scene_items(cell, pg.ImageItem) == images
    assert scene_items(cell, pg.ColorBarItem) == bars
    assert cell.plot_widget.viewRange() == view


@pytest.mark.parametrize('values', [None, np.empty((0, 2)), [[], []]])
def test_heatmap_clear_removes_scene_and_layout_references(cell, values):
    cell.configure_plot(kind='heatmap', title='Kept')
    cell.set_heatmap_data([[4, 4]])
    colorbar, = scene_items(cell, pg.ColorBarItem)
    assert colorbar.levels() == pytest.approx((3.96, 4.04))
    image, = [item for item in cell.plot_widget.getPlotItem().items if isinstance(item, pg.ImageItem)]
    cell.set_heatmap_data(values)
    assert colorbar.scene() is None
    assert image.scene() is None
    layout = cell.plot_widget.getPlotItem().layout
    assert all(layout.itemAt(i) is not colorbar for i in range(layout.count()))
    assert scene_items(cell, pg.ColorBarItem) == []
    assert cell.kind == 'heatmap'
    assert cell.header.left.toolTip() == 'Kept'


def test_repeated_switch_replace_and_clear_do_not_accumulate(cell):
    layout = cell.plot_widget.getPlotItem().layout
    initial_count = layout.count()
    for _ in range(5):
        cell.configure_plot(kind='heatmap')
        cell.set_heatmap_data([[1, 2], [3, 4]])
        old_bar, = scene_items(cell, pg.ColorBarItem)
        cell.set_heatmap_data([[5, 6]])
        assert old_bar.scene() is None
        assert len(scene_items(cell, pg.ColorBarItem)) == 1
        cell.configure_plot(kind='curve')
        assert scene_items(cell, pg.ColorBarItem) == []
        assert scene_items(cell, pg.ImageItem) == []
        cell.set_curve_data([{'x': [0], 'y': [1]}, {'x': [2], 'y': [3]}])
        assert len(scene_items(cell, pg.LegendItem)) == 1
        cell.clear_plot()
        cell.clear_plot()
        assert scene_items(cell, pg.LegendItem) == []
        assert data_items(cell) == []
        assert layout.count() == initial_count


def test_kind_mismatch_is_explicit_and_instances_are_independent(cell, qt_app):
    other = StationPlotCell(row=0, column=0)
    try:
        other.set_curve_data([{'x': [0, 1], 'y': [4, 5]}])
        curve = data_items(other)[0]
        view = other.plot_widget.viewRange()
        with pytest.raises(ValueError, match=r'set_heatmap_data.*\(1, 2\).*kind'):
            cell.set_heatmap_data([[1]])
        cell.configure_plot(kind='heatmap')
        with pytest.raises(ValueError, match='set_curve_data.*kind'):
            cell.set_curve_data([])
        cell.set_heatmap_data([[1]])
        cell.clear_plot()
        assert data_items(other) == [curve]
        assert other.plot_widget.viewRange() == view
    finally:
        other.close()
        other.deleteLater()


def test_cell_does_not_modify_pyqtgraph_global_options(cell):
    before = dict(pg.CONFIG_OPTIONS)
    cell.configure_plot(kind='heatmap')
    cell.set_heatmap_data([[1]])
    cell.configure_plot(kind='curve')
    cell.set_curve_data([{'x': [1], 'y': [2]}])
    assert pg.CONFIG_OPTIONS == before


def header_text(cell):
    return cell.header.left.toolTip(), cell.header.right.toolTip()


def test_header_tooltip_renders_markup_as_literal_text(cell, qt_app):
    cell.resize(300, 260)
    cell.show()
    text = '<b>left</b> & right'
    cell.set_plot_header(text, text)
    qt_app.processEvents()
    try:
        for label in (cell.header.left, cell.header.right):
            event = QHelpEvent(QEvent.ToolTip, QPoint(1, 1), label.mapToGlobal(QPoint(1, 1)))
            QCoreApplication.sendEvent(label, event)
            qt_app.processEvents()
            QTest.qWait(30)
            tip = next(w for w in qt_app.allWidgets() if w.objectName() == "qtooltip_label" and w.isVisible())
            assert (tip.font().family(), tip.font().pixelSize()) == ("Microsoft YaHei UI", 16)
            document = QTextDocument()
            document.setHtml(QToolTip.text())
            assert document.toPlainText() == text
            QToolTip.hideText()
    finally:
        QToolTip.hideText()


def test_header_line_breaks_stay_single_line_with_full_tooltip(cell, qt_app):
    cell.resize(500, 260)
    cell.show()
    cell.set_plot_header('One line', 'MIC 1')
    qt_app.processEvents()
    height = cell.header.height()
    text = '<b>A</b>\nB\tC\r\nD\u2028E'
    cell.set_plot_header(text, text)
    for _ in range(3):
        qt_app.processEvents()
    assert cell.header.height() == height
    for label in (cell.header.left, cell.header.right):
        assert label.toolTip() == text
        assert not any(char in label.text() for char in '\n\r\t\u2028')
        expected = label.fontMetrics().elidedText('<b>A</b> B C D E', Qt.ElideRight, label.width())
        assert label.text() == expected


@pytest.mark.parametrize('kind', ['curve', 'heatmap'])
def test_header_setter_preserves_config_items_data_and_manual_view(cell, qt_app, kind):
    cell.resize(500, 260)
    cell.show()
    cell.configure_plot(kind=kind, title='Old left', right_text='Old right',
                        x_label='Time', x_unit='s', y_label='Level', y_unit='dB',
                        x_range=(0, 10), y_range=(0, 20))
    if kind == 'curve':
        cell.set_curve_data([{'x': [0, 1], 'y': [2, 3]}])
    else:
        cell.set_heatmap_data([[1, 2], [3, 4]])
    qt_app.processEvents()
    cell.plot_widget.setRange(xRange=(2, 4), yRange=(6, 9), padding=0)
    plot = cell.plot_widget.getPlotItem()
    items, view = list(plot.items), cell.plot_widget.viewRange()
    series, matrix, bar = cell._series, cell._heatmap, cell._colorbar
    old_config = cell._configuration.copy()
    for left, right in [('SPL 趋势', 'MIC 1'), ('Left only', ''), ('', 'Right only'), ('', ''),
                        ('<b>A & B</b>', '<i>MIC 2</i>')]:
        assert cell.set_plot_header(left, right) is None
        for _ in range(3):
            qt_app.processEvents()
        assert header_text(cell) == (left, right)
        assert cell.header.left.text() == left and cell.header.right.text() == right
        assert cell.header.left.textFormat() == cell.header.right.textFormat() == Qt.PlainText
        assert cell.header.left.alignment() & Qt.AlignLeft
        assert cell.header.right.alignment() & Qt.AlignRight
        assert cell.header.left.palette().color(QPalette.WindowText).name() == '#183957'
        assert cell.header.right.palette().color(QPalette.WindowText).name() == '#5d7691'
        for label in (cell.header.left, cell.header.right):
            assert (label.font().family(), label.font().pixelSize()) == ("Microsoft YaHei UI", 13)
        assert cell._configuration == dict(old_config, title=left, right_text=right)
        assert list(plot.items) == items
        assert cell._series is series and cell._heatmap is matrix and cell._colorbar is bar
        assert cell.plot_widget.viewRange() == view
        assert plot.getAxis('bottom').labelText == 'Time'
        assert plot.getAxis('bottom').labelUnits == 's'
    header_geometry = cell.header.geometry()
    cell.plot_widget.setRange(xRange=(20, 30), yRange=(30, 50), padding=0)
    qt_app.processEvents()
    assert cell.header.geometry() == header_geometry
    cell.clear_plot()
    assert header_text(cell) == ('<b>A & B</b>', '<i>MIC 2</i>')
    cell.configure_plot(kind='heatmap' if kind == 'curve' else 'curve', title='New type')
    assert header_text(cell) == ('New type', '')


@pytest.mark.parametrize('left,right', [('Left', 'Right'), ('Left', ''), ('', 'Right'), ('', '')])
def test_header_empty_sides_reclaim_width_and_height(cell, qt_app, left, right):
    cell.resize(400, 260)
    cell.show()
    qt_app.processEvents()
    empty_height = cell.plot_widget.height()
    cell.set_plot_header(left, right)
    for _ in range(3):
        qt_app.processEvents()
    assert cell.header.isVisible() == bool(left or right)
    assert cell.header.left.isVisible() == bool(left)
    assert cell.header.right.isVisible() == bool(right)
    if left and right:
        assert cell.header.left.width() == pytest.approx(2 * cell.header.right.width(), abs=2)
    elif left or right:
        label = cell.header.left if left else cell.header.right
        assert label.width() == cell.header.contentsRect().width() - 16
    if left or right:
        assert cell.plot_widget.height() + cell.header.height() == empty_height
    else:
        assert cell.plot_widget.y() == 0
    cell.set_plot_header()
    for _ in range(3):
        qt_app.processEvents()
    assert not cell.header.isVisible() and cell.plot_widget.height() == empty_height


@pytest.mark.parametrize('kind', ['curve', 'heatmap'])
@pytest.mark.parametrize('method,field,value', [
    ('configure_plot', 'title', None), ('configure_plot', 'right_text', 12),
    ('set_plot_header', 'left_text', None), ('set_plot_header', 'right_text', []),
])
def test_invalid_header_is_atomic_and_has_cell_diagnostic(cell, qt_app, kind, method, field, value):
    cell.configure_plot(kind=kind, title='Old', right_text='MIC 1', x_label='Time', x_range=(0, 5))
    if kind == 'curve':
        cell.set_curve_data([{'x': [0, 1], 'y': [2, 3]}])
    else:
        cell.set_heatmap_data([[1, 2]])
    cell.plot_widget.setRange(xRange=(1, 2), yRange=(3, 4), padding=0)
    plot = cell.plot_widget.getPlotItem()
    config, items, view = cell._configuration, list(plot.items), cell.plot_widget.viewRange()
    series, heatmap, bar = cell._series, cell._heatmap, cell._colorbar
    kwargs = ({'kind': 'heatmap' if kind == 'curve' else 'curve', 'title': 'Changed', 'right_text': 'Changed'}
              if method == 'configure_plot' else {'left_text': 'Changed', 'right_text': 'Changed'})
    kwargs[field] = value
    with pytest.raises(TypeError, match=rf'{method}.*\(1, 2\).*{field}'):
        getattr(cell, method)(**kwargs)
    assert header_text(cell) == ('Old', 'MIC 1')
    assert cell.kind == kind and cell._configuration is config
    assert list(plot.items) == items and cell.plot_widget.viewRange() == view
    assert cell._series is series and cell._heatmap is heatmap and cell._colorbar is bar



def subheader_text(cell):
    return tuple((label.toolTip() for label in (cell.subheader.left, cell.subheader.first_half, cell.subheader.second_half)))

def assert_subheader_alignment(cell):
    view = cell.plot_widget.getViewBox()
    bounds = view.mapRectToScene(view.rect())
    viewport = cell.plot_widget.viewport()
    origin = viewport.mapTo(cell, QPoint(0, 0)).x() - cell.subheader.x()
    left = cell.plot_widget.mapFromScene(QPointF(bounds.left(), bounds.top())).x() + origin
    right = cell.plot_widget.mapFromScene(QPointF(bounds.right(), bounds.top())).x() + origin
    for label, fraction in [(cell.subheader.first_half, 0.25), (cell.subheader.second_half, 0.75)]:
        assert label.geometry().center().x() == pytest.approx(left + (right - left) * fraction, abs=2)
    assert cell.subheader.left.x() == 8
    labels = [cell.subheader.left, cell.subheader.first_half, cell.subheader.second_half]
    assert all((cell.subheader.rect().contains(label.geometry()) for label in labels))
    for before, after in pairwise(labels):
        assert after.x() - (before.x() + before.width()) >= 8
    return left


@pytest.mark.parametrize('unit', ['dB SPL', 'dB(A) SPL', 'dB(A)'])
def test_subheader_minimum_size_unit_text_updates_immediately(yahei_font, cell, qt_app, unit):
    cell.resize(280, 180)
    cell.plot_widget.getAxis('left').setWidth(14)
    cell.show()
    cell.set_plot_subheader('dB', '工况1', '工况2')
    for _ in range(4):
        qt_app.processEvents()

    assert (cell.width(), cell.height()) == (280, 180)
    view_box = cell.plot_widget.getViewBox()
    old_plot_rect = view_box.sceneBoundingRect()
    left, first, second = cell.subheader.left, cell.subheader.first_half, cell.subheader.second_half
    old_unit_width = left.width()
    old_first_rect = first.geometry()
    old_second_rect = second.geometry()

    cell.set_plot_subheader(unit, '工况1', '工况2')
    assert left.text() == unit
    assert left.width() >= left.fontMetrics().horizontalAdvance(unit) > old_unit_width
    assert left.toolTip() == unit
    assert first.text() == '工况1' and second.text() == '工况2'
    assert first.width() < old_first_rect.width()
    assert first.geometry().center() == old_first_rect.center()
    assert second.geometry() == old_second_rect
    assert view_box.sceneBoundingRect() == old_plot_rect
    assert_subheader_alignment(cell)

    cell.set_plot_subheader('dB', '工况1', '工况2')
    assert left.text() == 'dB'
    assert left.width() == old_unit_width
    assert first.geometry() == old_first_rect and second.geometry() == old_second_rect
    assert_subheader_alignment(cell)


@pytest.mark.parametrize('kind', ['curve', 'heatmap'])
def test_subheader_setter_preserves_first_row_items_snapshots_axes_and_manual_range(cell, qt_app, kind):
    cell.resize(600, 300)
    cell.show()
    cell.configure_plot(kind=kind, title='Title', right_text='MIC 1', x_label='Time', y_label='Level', subheader_left_text='unit', subheader_first_half_text='one', subheader_second_half_text='two')
    if kind == 'curve':
        cell.set_curve_data([{'x': [0, 1], 'y': [2, 3]}])
    else:
        cell.set_heatmap_data([[1, 2], [3, 4]])
    for _ in range(3):
        qt_app.processEvents()
    cell.plot_widget.setRange(xRange=(2, 4), yRange=(6, 9), padding=0)
    items = list(cell.plot_widget.getPlotItem().items)
    series, matrix, image, bar = (cell._series, cell._heatmap, cell._image, cell._colorbar)
    view = cell.plot_widget.viewRange()
    old = cell._configuration.copy()
    for texts in [('kHz', '正转', '反转'), ('', '正转', ''), ('kHz', '', '反转'), ('', '', '')]:
        assert cell.set_plot_subheader(*texts) is None
        for _ in range(3):
            qt_app.processEvents()
        assert subheader_text(cell) == texts
        assert header_text(cell) == ('Title', 'MIC 1')
        assert cell._configuration == dict(old, **dict(zip(('subheader_left_text', 'subheader_first_half_text', 'subheader_second_half_text'), texts)))
        assert list(cell.plot_widget.getPlotItem().items) == items
        assert cell._series is series and cell._heatmap is matrix and cell._image is image and cell._colorbar is bar
        assert cell.plot_widget.viewRange() == view
    cell.set_plot_subheader('kHz', '正转', '反转')
    cell.set_plot_header('Updated', 'MIC 2')
    assert subheader_text(cell) == ('kHz', '正转', '反转')
    cell.clear_plot()
    assert subheader_text(cell) == ('kHz', '正转', '反转')
    cell.configure_plot(kind=kind)
    assert subheader_text(cell) == ('', '', '')
    cell.configure_plot(kind='heatmap' if kind == 'curve' else 'curve', subheader_first_half_text='new')
    assert subheader_text(cell) == ('', 'new', '')


@pytest.mark.parametrize('kind', ['curve', 'heatmap'])
@pytest.mark.parametrize('method', ['set_plot_subheader', 'configure_plot'])
@pytest.mark.parametrize('field', ['left_text', 'first_half_text', 'second_half_text'])
def test_invalid_subheader_is_atomic_including_late_cross_kind_error(cell, qt_app, kind, method, field):
    cell.configure_plot(kind=kind, title='Old', right_text='MIC 1', subheader_left_text='unit', subheader_first_half_text='one', subheader_second_half_text='two')
    if kind == 'curve':
        cell.set_curve_data([{'x': [0, 1], 'y': [2, 3]}])
    else:
        cell.set_heatmap_data([[1, 2]])
    cell.plot_widget.setRange(xRange=(1, 2), yRange=(3, 4), padding=0)
    config, items, view = (cell._configuration, list(cell.plot_widget.getPlotItem().items), cell.plot_widget.viewRange())
    series, matrix, image, bar = (cell._series, cell._heatmap, cell._image, cell._colorbar)
    kwargs = dict(left_text='changed', first_half_text='changed', second_half_text='changed')
    if method == 'configure_plot':
        kwargs = {'subheader_' + k: v for k, v in kwargs.items()}
        kwargs.update(kind='heatmap' if kind == 'curve' else 'curve', title='Changed')
        field = 'subheader_' + field
    kwargs[field] = None
    with pytest.raises(TypeError, match=f'{method}.*\\(1, 2\\).*{field}'):
        getattr(cell, method)(**kwargs)
    assert cell._configuration is config and cell.kind == kind
    assert header_text(cell) == ('Old', 'MIC 1') and subheader_text(cell) == ('unit', 'one', 'two')
    assert list(cell.plot_widget.getPlotItem().items) == items and cell.plot_widget.viewRange() == view
    assert cell._series is series and cell._heatmap is matrix and cell._image is image and cell._colorbar is bar


@pytest.mark.parametrize('kind', ['curve', 'heatmap'])
def test_subheader_geometry_follows_real_viewbox_axis_width_colorbar_switch_and_resize(cell, qt_app, kind):
    cell.show()
    for width in (600, 280, 480):
        cell.resize(width, 280)
        cell.configure_plot(kind=kind, title='Title', y_label='Frequency', y_unit='Hz', subheader_left_text='kHz', subheader_first_half_text='正转', subheader_second_half_text='反转')
        if kind == 'heatmap':
            cell.set_heatmap_data([[1, 2], [3, 4]])
        for axis_width in (45, 100):
            cell.plot_widget.getAxis('left').setWidth(axis_width)
            for _ in range(4):
                qt_app.processEvents()
            assert_subheader_alignment(cell)
        if kind == 'heatmap':
            cell.set_heatmap_data(None)
            for _ in range(4):
                qt_app.processEvents()
            assert_subheader_alignment(cell)
    cell.configure_plot(kind='heatmap' if kind == 'curve' else 'curve', subheader_left_text='unit', subheader_first_half_text='one', subheader_second_half_text='two', y_label='Different axis')
    if cell.kind == 'heatmap':
        cell.set_heatmap_data([[1, 2]])
    cell.plot_widget.getAxis('left').setWidth(None)
    cell.plot_widget.setRange(yRange=(123456, 987654), padding=0)
    for _ in range(4):
        qt_app.processEvents()
    assert_subheader_alignment(cell)


def test_subheader_unit_width_follows_axis_resize_and_kind_changes(yahei_font, cell, qt_app):
    unit = 'dB(A) SPL'
    left, first, second = cell.subheader.left, cell.subheader.first_half, cell.subheader.second_half
    cell.show()
    for kind in ('curve', 'heatmap', 'curve'):
        cell.configure_plot(kind=kind, subheader_left_text=unit,
                            subheader_first_half_text='工况1', subheader_second_half_text='工况2')
        if kind == 'heatmap':
            cell.set_heatmap_data([[1, 2], [3, 4]])
        for width in (280, 600, 280):
            cell.resize(width, 180)
            for axis_width in (14, 100, 14):
                cell.plot_widget.getAxis('left').setWidth(axis_width)
                for _ in range(4):
                    qt_app.processEvents()
                plot_left = assert_subheader_alignment(cell)
                assert first.text() == '工况1' and second.text() == '工况2'
                assert left.toolTip() == unit
                if axis_width == 14 and (kind == 'curve' or width == 600):
                    assert left.text() == unit
                    assert first.x() > plot_left + 4
                    assert first.width() >= 2 * first.fontMetrics().height()
                else:
                    assert left.width() == plot_left - 12
                    assert first.x() == plot_left + 4
                    if axis_width == 100:
                        assert left.text() == unit
                    else:
                        assert left.text() != unit


@pytest.mark.parametrize('texts', [('unit', 'one', 'two'), ('', 'one', ''), ('unit', '', 'two'), ('', '', '')])
def test_subheader_fixed_empty_positions_independent_rows_and_height(cell, qt_app, texts):
    cell.resize(500, 280)
    cell.show()
    qt_app.processEvents()
    empty = cell.plot_widget.height()
    initial_texts = ('unit', 'one', 'two')
    labels = (cell.subheader.left, cell.subheader.first_half, cell.subheader.second_half)
    cell.set_plot_subheader(*initial_texts)
    for _ in range(3):
        qt_app.processEvents()
    geometry = [label.geometry() for label in labels]
    cell.set_plot_subheader(*texts)
    for _ in range(3):
        qt_app.processEvents()
    assert not cell.header.isVisible()
    assert cell.subheader.isVisible() == any(texts)
    for label, text in zip(labels, texts):
        assert label.isVisible() == bool(text)
    if any(texts):
        if texts[0] == initial_texts[0]:
            assert [label.geometry() for label in labels] == geometry
        else:
            left, first, second = labels
            assert left.geometry().topLeft() == geometry[0].topLeft()
            assert left.height() == geometry[0].height()
            assert first.geometry().center() == geometry[1].center()
            assert first.height() == geometry[1].height()
            assert second.geometry() == geometry[2]
        assert_subheader_alignment(cell)
        assert empty == cell.plot_widget.height() + cell.subheader.height()
    else:
        assert cell.plot_widget.height() == empty
    cell.set_plot_header('Title', 'MIC 1')
    for _ in range(3):
        qt_app.processEvents()
    assert cell.header.isVisible() and cell.subheader.isVisible() == any(texts)
    if any(texts):
        assert cell.header.geometry().bottom() < cell.subheader.y() < cell.plot_widget.y()
    cell.set_plot_subheader()
    for _ in range(3):
        qt_app.processEvents()
    assert cell.header.isVisible() and not cell.subheader.isVisible()


@pytest.mark.parametrize('kind', ['curve', 'heatmap'])
def test_subheader_long_plain_text_minimum_size_tooltip_and_bounds(cell, qt_app, kind):
    cell.configure_plot(kind=kind, title='Title', right_text='MIC 1', y_label='Frequency', y_unit='Hz')
    if kind == 'heatmap':
        cell.set_heatmap_data([[1, 2], [3, 4]])
    text = '<b>A & B</b> 很长的中文 and English\nnext\tline ' * 12
    cell.set_plot_subheader(text, text, text)
    cell.show()
    for size in [(600, 300), (280, 180)]:
        cell.resize(*size)
        for _ in range(4):
            qt_app.processEvents()
        assert (cell.width(), cell.height()) == size
        assert_subheader_alignment(cell)
        assert cell.minimumWidth() == 280 and cell.minimumHeight() == 180
        for label in (cell.subheader.left, cell.subheader.first_half, cell.subheader.second_half):
            assert label.toolTip() == text and label.textFormat() == Qt.PlainText
            assert not any((char in label.text() for char in '\n\t'))
            assert (label.font().family(), label.font().pixelSize()) == ("Microsoft YaHei UI", 11)
            assert label.palette().color(QPalette.WindowText).name() == '#5d7691'
            assert label.text() == label.fontMetrics().elidedText(' '.join(text.splitlines()).replace('\t', ' '), Qt.ElideRight, label.width())
            event = QHelpEvent(QEvent.ToolTip, QPoint(1, 1), label.mapToGlobal(QPoint(1, 1)))
            QCoreApplication.sendEvent(label, event)
            document = QTextDocument()
            document.setHtml(QToolTip.text())
            assert document.toPlainText() == text
            QToolTip.hideText()
        view = cell.plot_widget
        for item in [view.getAxis('left'), view.getAxis('bottom')] + scene_items(cell, pg.ColorBarItem):
            bounds = view.mapFromScene(item.mapRectToScene(item.rect())).boundingRect()
            assert bounds.left() >= 0 and bounds.right() <= view.viewport().width()

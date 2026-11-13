"""Real current spectrogram axes reserve space for their actual tick fonts."""
from PyQt5.QtGui import QFont, QFontMetrics

from unit_test.ui.test_condition_plot_layout import configuration, grid, show
from unit_test.ui.test_workbench_extended_plots import (
    extended_configuration, instance, matrix_payload, publish,
)
from ui.sequence.workbench_plot_presenter import WorkbenchPlotPresenter


def test_spectrogram_reserves_space_for_large_axis_fonts(grid, configuration, qt_app):
    projection = show(grid, extended_configuration(configuration), qt_app)
    presenter = WorkbenchPlotPresenter(grid)
    presenter.configure(projection, invalidate_requests=True)
    binding = projection.bindings[0][0][0]
    publish(projection, presenter, [instance(binding, matrix_payload())])
    cell = grid._cells[0, 0]
    plot = cell.plot_widget.getPlotItem()
    bottom, left = plot.getAxis('bottom'), plot.getAxis('left')
    color = cell._colorbar.axis
    font = QFont(bottom.style['tickFont'])
    font.setPixelSize(24)
    for axis in (bottom, left, color):
        axis.setTickFont(font)
    for _ in range(8):
        qt_app.processEvents()
        cell.grab()
    metrics = QFontMetrics(font)
    assert bottom.boundingRect().height() >= metrics.height()
    assert left.boundingRect().width() >= metrics.horizontalAdvance('20')
    assert color.boundingRect().width() >= metrics.horizontalAdvance('100')
    assert plot.getViewBox().height() >= 120

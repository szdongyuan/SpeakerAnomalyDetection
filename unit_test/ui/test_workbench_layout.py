from types import SimpleNamespace

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton, QScrollArea, QSplitter, QTableWidget, QVBoxLayout, QWidget

from ui.sequence import workbench_layout
from ui.sequence.motor_dashboard_panel import MotorDashboardPanel
from ui.sequence.motor_video_monitor_panel import MotorVideoMonitorPanel
from ui.sequence.station_plot_grid import StationPlotGrid


@pytest.fixture
def host(qt_app, monkeypatch):
    widget = QWidget()
    dashboard = MotorDashboardPanel(widget)
    video = MotorVideoMonitorPanel(widget)
    widget.left_panel = SimpleNamespace(take_split_sections=lambda: (dashboard, video))
    monkeypatch.setattr(workbench_layout, "resolve_workspace_screen_size", lambda host: (1920, 1080))
    widget.setLayout(workbench_layout.build_workbench_layout(widget))
    yield widget
    widget.close()


def test_layout_uses_real_splitters_and_current_sections_only(host):
    layout = host.layout()
    assert isinstance(layout, QVBoxLayout)
    assert layout.count() == 1
    margins = layout.contentsMargins()
    assert (margins.left(), margins.top(), margins.right(), margins.bottom()) == (8, 12, 8, 12)
    assert layout.spacing() == 0
    main = layout.itemAt(0).widget()
    assert isinstance(main, QSplitter)
    assert main.orientation() == Qt.Horizontal
    sidebar = main.widget(0)
    assert sidebar.orientation() == Qt.Vertical
    assert isinstance(sidebar.widget(0), MotorDashboardPanel)
    assert isinstance(sidebar.widget(1), MotorVideoMonitorPanel)
    assert sidebar.widget(0).minimumWidth() == 340
    assert sidebar.widget(1).minimumWidth() == 340
    assert main.widget(1) is host.plot_grid
    assert isinstance(host.plot_grid, StationPlotGrid)
    assert not main.childrenCollapsible()
    assert not sidebar.childrenCollapsible()


@pytest.mark.parametrize("width,height", [(2400, 1100), (1920, 700), (1600, 480), (1100, 700)])
def test_initial_space_transfers_actual_pixels_once_after_first_visible_layout(host, qt_app, width, height):
    qt_app.processEvents()  # hidden startup must not consume allocation
    host.resize(width, height)
    host.show()
    main = host.layout().itemAt(0).widget()
    sidebar = main.widget(0)
    initial_widths = main.sizes()
    initial_heights = sidebar.sizes()
    minimum_width = max(sidebar.minimumWidth(), sidebar.minimumSizeHint().width())
    video = sidebar.widget(1)
    minimum_height = max(video.minimumHeight(), video.minimumSizeHint().height())
    width_transfer = min(200, max(0, initial_widths[0] - minimum_width))
    height_transfer = min(100, max(0, initial_heights[1] - minimum_height))
    qt_app.processEvents()
    assert main.sizes() == [initial_widths[0] - width_transfer, initial_widths[1] + width_transfer]
    assert sidebar.sizes() == [initial_heights[0] + height_transfer, initial_heights[1] - height_transfer]
    assert video.height() >= minimum_height
    main.moveSplitter(minimum_width + 60, 1)
    sidebar.moveSplitter(initial_heights[0], 1)
    manual = (main.sizes(), sidebar.sizes())
    host.hide()
    host.show()
    qt_app.processEvents()
    assert (main.sizes(), sidebar.sizes()) == manual
    host.resize(width + 100, height + 80)
    resized = (main.sizes(), sidebar.sizes())
    qt_app.processEvents()
    assert (main.sizes(), sidebar.sizes()) == resized


def test_common_and_small_window_keep_plots_and_scrollable_stations(host, qt_app):
    host.resize(1400, 900)
    host.show()
    qt_app.processEvents()
    assert host.plot_grid.width() >= 900
    assert host.plot_grid.height() >= 700
    host.resize(1400, 700)
    sidebar = host.layout().itemAt(0).widget().widget(0)
    dashboard = sidebar.widget(0)
    dashboard.station_panel.show_workbench_stations([
        (f"Station {index}", ["SPL", "FFT", "LOUD"]) for index in range(8)
    ])
    # Expanded metric tables guarantee overflow on both native and offscreen
    # fonts; eight collapsed headers can legitimately fit the native viewport.
    for index in range(1, 9):
        header = dashboard.station_panel.findChild(QPushButton, f"stationHeader{index}")
        header.click()
        assert header.isChecked()
    scroll = dashboard.station_panel.findChild(QScrollArea, "stationScroll")
    for _ in range(8):
        qt_app.processEvents()
        if scroll.verticalScrollBar().maximum() > 0:
            break
    assert dashboard.station_panel.findChild(QTableWidget, "stationTable1").isVisible()
    assert host.minimumSizeHint().height() <= 700
    assert scroll.widget().height() > scroll.viewport().height()
    assert scroll.verticalScrollBar().maximum() > 0
    assert dashboard.summary_panel.isVisible()
    assert dashboard.station_panel.geometry().bottom() < dashboard.summary_panel.y()

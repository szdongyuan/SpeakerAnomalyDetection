"""Current workbench workspace geometry, without legacy runtime dependencies."""
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QSplitter, QVBoxLayout

from ui.sequence.station_plot_grid import StationPlotGrid


class _StationVideoSplitter(QSplitter):
    """Transfer actual video space to stations once the first layout is visible."""

    def __init__(self):
        super().__init__(Qt.Vertical)
        self._initial_height_pending = True

    def showEvent(self, event):
        super().showEvent(event)
        if self._initial_height_pending:
            QTimer.singleShot(0, self._allocate_initial_station_height)

    def _allocate_initial_station_height(self):
        if not self._initial_height_pending or not self.isVisible():
            return
        sizes = self.sizes()
        if len(sizes) != 2 or min(sizes) <= 0:
            return
        video = self.widget(1)
        minimum = max(video.minimumHeight(), video.minimumSizeHint().height())
        transfer = min(100, max(0, sizes[1] - minimum))
        self._initial_height_pending = False
        self.setSizes([sizes[0] + transfer, sizes[1] - transfer])


class _SidebarWorkspaceSplitter(QSplitter):
    """Give initial sidebar space to plots after real layout, once only."""

    def __init__(self):
        super().__init__(Qt.Horizontal)
        self._initial_width_pending = True

    def showEvent(self, event):
        super().showEvent(event)
        if self._initial_width_pending:
            QTimer.singleShot(0, self._allocate_initial_sidebar_width)

    def _allocate_initial_sidebar_width(self):
        if not self._initial_width_pending or not self.isVisible():
            return
        sizes = self.sizes()
        if len(sizes) != 2 or min(sizes) <= 0:
            return
        sidebar = self.widget(0)
        minimum = max(sidebar.minimumWidth(), sidebar.minimumSizeHint().width())
        transfer = min(200, max(0, sizes[0] - minimum))
        self._initial_width_pending = False
        self.setSizes([sizes[0] - transfer, sizes[1] + transfer])


def build_workbench_layout(host):
    """
        Create the main work area layout.

        Arrange task/video panels on the left and independent station plots on the right.

        Returns:
            QVBoxLayout: The configured layout object.
    """
    layout = QVBoxLayout()
    host.plot_grid = StationPlotGrid(host)
    dashboard_panel, video_monitor_panel = host.left_panel.take_split_sections()
    if dashboard_panel is not None:
        dashboard_panel.setMinimumWidth(340)
    if video_monitor_panel is not None:
        video_monitor_panel.setMinimumWidth(340)

    # Compute initial splitter sizes from the current screen so the layout
    # does not stay glued to a small left column on high-DPI / large screens.
    screen_width, screen_height = resolve_workspace_screen_size(host)
    h_left = max(380, int(screen_width * 0.30))
    h_right = max(900, screen_width - h_left)
    v_task = max(420, int(screen_height * 0.60))
    v_video = max(260, screen_height - v_task)

    left_sidebar_splitter = _StationVideoSplitter()
    left_sidebar_splitter.addWidget(dashboard_panel)
    left_sidebar_splitter.addWidget(video_monitor_panel)
    left_sidebar_splitter.setChildrenCollapsible(False)
    left_sidebar_splitter.setStretchFactor(0, 3)
    left_sidebar_splitter.setStretchFactor(1, 2)
    left_sidebar_splitter.setSizes([v_task, v_video])

    main_splitter = _SidebarWorkspaceSplitter()
    main_splitter.addWidget(left_sidebar_splitter)
    main_splitter.addWidget(host.plot_grid)
    main_splitter.setChildrenCollapsible(False)
    main_splitter.setStretchFactor(0, 4)
    main_splitter.setStretchFactor(1, 9)
    main_splitter.setSizes([h_left, h_right])

    layout.addWidget(main_splitter)
    layout.setContentsMargins(8, 12, 8, 12)
    layout.setSpacing(0)
    return layout


def resolve_workspace_screen_size(host):
    """Use the current screen's available size, with the existing startup baseline."""
    screen = host.window().screen()
    if screen is None:
        app = QApplication.instance()
        screen = app.primaryScreen() if app is not None else None
    if screen is None:
        return 1600, 900
    geometry = screen.availableGeometry()
    return (geometry.width() if geometry.width() > 0 else 1600,
            geometry.height() if geometry.height() > 0 else 900)

"""Coordinate product configuration and user intents across workbench views.

Send cross-widget business updates through the controller. Views render data
and emit intents. The repository reads committed configuration, and the
configuration editors own writes and send commit notifications to the caller.
"""

from collections import Counter
from collections.abc import Callable

from base.motor_workbench_config import WorkbenchConfig, WorkbenchConfigurationError


def _station_display(config: WorkbenchConfig) -> tuple[tuple[str, tuple[str, ...]], ...]:
    metrics_by_station = {station.station_id: [] for station in config.stations}
    for condition in config.conditions:
        for station_id in condition.applicable_station_ids:
            metrics_by_station[station_id].extend(
                (metric, condition.condition_name) for metric in condition.metric_names
            )

    display = []
    for station in config.stations:
        metrics = metrics_by_station[station.station_id]
        counts = Counter(metric for metric, _ in metrics)
        labels = tuple(
            f"{metric}（{condition}）" if counts[metric] > 1 else metric
            for metric, condition in metrics
        )
        display.append((station.station_name, labels))
    return tuple(display)


class MotorWorkbenchController:
    """Apply validated snapshots and route intents on the caller's GUI thread.

    The caller owns Qt signal bindings. The product callback receives an empty
    string when no product is applied.
    """

    def __init__(
        self,
        repository,
        station_panel,
        plot_grid,
        set_active_product: Callable[[str], None],
        show_video_settings: Callable[[], None],
    ):
        self._repository = repository
        self._station_panel = station_panel
        self._plot_grid = plot_grid
        self._set_active_product = set_active_product
        self._show_video_settings = show_video_settings
        self._initialized = False
        self._applied_config: WorkbenchConfig | None = None

    def refresh_configuration(self) -> None:
        """Read the active configuration and replace both displays if it changed."""
        if not self._initialized:
            self._apply_configuration(None)
            self._initialized = True

        try:
            config = self._repository.load_active()
        except WorkbenchConfigurationError:
            # A missing or invalid configuration leaves the current display in place.
            return

        if config != self._applied_config:
            self._apply_configuration(config)

    def _apply_configuration(self, config: WorkbenchConfig | None) -> None:
        stations = _station_display(config) if config is not None else ()
        names = tuple(name for name, _ in stations)
        product_name = config.project_name if config is not None else ""

        self._station_panel.show_workbench_stations(stations)
        self._plot_grid.show_workbench_columns(names)
        self._set_active_product(product_name)
        self._applied_config = config

    def open_video_settings(self) -> None:
        """Forward the video settings intent to the existing settings entry point."""
        self._show_video_settings()

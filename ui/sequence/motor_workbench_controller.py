"""Coordinate product configuration and user intents across workbench views.

Send cross-widget business updates through the controller. Views render data
and emit intents. The repository reads committed configuration, and the
configuration editors own writes and send commit notifications to the caller.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, datetime
from uuid import uuid4

from base.workbench_config_values import WorkbenchChannel, WorkbenchCondition
from base.workbench_configuration_loader import WorkbenchConfiguration, WorkbenchConfigurationError
from base.workbench_analysis_input import AnalysisRecordingContext, WorkbenchAnalysisInput, make_analysis_recording_context
from ui.sequence.workbench_ui_projection import build_ui_projection, _PlotBindings
from base.recording_service import RecordingCallbacks
from base.workbench_recording_protocol import WorkbenchRecordingCompletion
from base.workbench_recording_request import (
    WorkbenchRecordingRequestError,
    build_recording_request,
    recording_config_key,
)
from ui.recording_service_bridge import RecordingServiceBridge
from ui.workbench_recording_device import WorkbenchRecordingDevice, WorkbenchRecordingDeviceSnapshot


@dataclass(frozen=True)
class WorkbenchRecordingPorts:
    bridge: RecordingServiceBridge
    device: WorkbenchRecordingDevice
    hardware_available: Callable[[WorkbenchRecordingDeviceSnapshot], bool]
    read_wav_identity: Callable[[], tuple[str, str, str]]
    set_button_state: Callable[[bool, str, bool], None]
    notify_error: Callable[[str], None]
    completed: Callable[[WorkbenchRecordingCompletion], None]
    notify_cleanup_error: Callable[[str], None]
    submit_archive: Callable[[WorkbenchRecordingCompletion, str, date, Callable[[str, Exception | None], None]], None]
    notify_archive_error: Callable[[str], None]
    is_closing: Callable[[], bool] = lambda: False
    ve_calibration_store: object = None


@dataclass(frozen=True)
class _InFlightRecording:
    request_id: str
    configuration_revision: int
    condition_name: str
    product_name: str
    recorded_on: date
    analysis_context: AnalysisRecordingContext | None = None
    archiving: bool = False


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
        self._applied_config: WorkbenchConfiguration | None = None
        self._plot_bindings: tuple[tuple[_PlotBindings, ...], ...] = ()
        self._recording: WorkbenchRecordingPorts | None = None
        self._configuration_valid = False
        self._analysis_version = None
        self._analysis_input_ready = None
        self._recording_key = None
        self._configuration_revision = 0
        self._condition_index = 0
        self._in_flight: _InFlightRecording | None = None

    def refresh_configuration(self) -> None:
        """Read the active configuration and replace both displays if it changed."""
        if not self._initialized:
            self._apply_configuration(None)
            self._initialized = True

        try:
            config = self._repository.load_active()
        except WorkbenchConfigurationError as error:
            logging.getLogger("core").warning("Motor workbench configuration: %s", error)
            self._configuration_valid = False
            self.refresh_recording_state()
            return

        if config != self._applied_config:
            self._apply_configuration(config)
        self._configuration_valid = True
        self._refresh_recording_configuration()

    def bind_recording(self, ports: WorkbenchRecordingPorts) -> None:
        self._recording = ports
        self._refresh_recording_configuration()

    def _refresh_recording_configuration(self):
        if self._recording is None:
            return
        config = self._applied_config
        key = recording_config_key(config.recording, source_identity=self._repository.source_identity) if config is not None else None
        if key != self._recording_key:
            self._recording_key = key
            self._configuration_revision += 1
            self._condition_index = 0
        self.refresh_recording_state()

    def _apply_configuration(self, config: WorkbenchConfiguration | None) -> None:
        if config is None:
            stations, names, cells, bindings, product_name = (), (), (), (), ""
        else:
            projection = build_ui_projection(config.ui, config.analysis)
            stations = projection.stations
            names, cells, bindings = projection.column_names, projection.cells, projection.bindings
            product_name = config.recording.context.project_name

        previous = self.get_analysis_config()
        analysis = config.analysis if config is not None else None
        analysis_version = self._analysis_version
        if analysis != previous:
            analysis_version = uuid4().hex if analysis is not None else None

        self._station_panel.show_workbench_stations(stations)
        self._plot_grid.show_workbench_layout(names, cells)
        self._set_active_product(product_name)
        self._applied_config = config
        self._analysis_version = analysis_version
        self._plot_bindings = bindings

    def _recording_admission(self):
        ports = self._recording
        if ports.is_closing():
            return None, "应用正在退出。"
        attempt = self._in_flight
        if attempt is not None:
            if attempt.archiving:
                return None, f"正在将工况「{attempt.condition_name}」的音频入库。"
            return None, f"正在录制或保存工况「{attempt.condition_name}」的音频。"
        if not self._configuration_valid:
            return None, "工作台配置不可用，请检查产品和测试队列配置。"
        config = self._applied_config
        if config is None:
            return None, "没有活动产品，请先在产品测试配置中选择产品。"
        if not config.recording.context.conditions:
            return None, "当前产品没有工况，请先配置录制工况。"
        device = ports.device
        if not device.login_ready:
            return None, "登录后可以开始录制。"
        snapshot = device.snapshot
        if snapshot is None:
            return None, device.detail or "录制设备不可用，请检查硬件设置。"
        if not ports.hardware_available(snapshot) or not ports.bridge.service.can_start_recording:
            return None, "录制资源正在使用或回收，请等待后重试。"
        return snapshot, f"开始录制工况「{config.recording.context.conditions[self._condition_index].condition_name}」"

    def refresh_recording_state(self) -> None:
        if self._recording is not None:
            snapshot, detail = self._recording_admission()
            self._recording.set_button_state(snapshot is not None, detail, self._in_flight is not None)

    def start_recording(self) -> None:
        """Submit one frozen condition and hold the cursor through archival."""
        recorded_at = datetime.now()
        ports = self._recording
        if ports is None:
            return
        snapshot, detail = self._recording_admission()
        if snapshot is None:
            ports.set_button_state(False, detail, self._in_flight is not None)
            return
        config = self._applied_config
        analysis_version = self._analysis_version
        condition = config.recording.context.conditions[self._condition_index]
        product_model, sample_number, serial_number = ports.read_wav_identity()
        attempt = _InFlightRecording(
            uuid4().hex, self._configuration_revision, condition.condition_name,
            config.recording.context.project_name, recorded_at.date(),
        )
        self._in_flight = attempt
        self.refresh_recording_state()
        try:
            request = build_recording_request(
                config.recording, condition.condition_id, snapshot, attempt.request_id,
                recorded_at=recorded_at, product_model=product_model, sample_number=sample_number,
                serial_number=serial_number,
                preview_time_mode=config.ui.preview_time_modes.get(condition.test_queue),
                ve_calibration_store=ports.ve_calibration_store,
            )
        except WorkbenchRecordingRequestError as exc:
            self._reject_recording(attempt, str(exc))
            return

        attempt = replace(attempt, analysis_context=make_analysis_recording_context(
            config.analysis, analysis_version, request,
        ))
        self._in_flight = attempt

        # The worker can retire while calibration is read. Require the same
        # prepared device immediately before entering service admission.
        if ports.is_closing() or not ports.device.login_ready or ports.device.snapshot is not snapshot:
            self._reject_recording(
                attempt, f"工况「{condition.condition_name}」的录制设备或登录状态已变化，请就绪后重试。",
            )
            return
        try:
            ports.bridge.start(request, RecordingCallbacks(
                workbench_completed=self._recording_completed,
                failed=self._recording_failed,
                cancelled=self._recording_cancelled,
                released=self._recording_released,
                release_failed=lambda session, detail: self._recording_release_failed(attempt, session, detail),
            ))
        except (RuntimeError, ValueError, OSError) as exc:
            if self._in_flight is not attempt:
                # A synchronous terminal callback already completed this request.
                # Consumer failures must not become recording admission errors.
                raise
            self._reject_recording(attempt, f"工况「{condition.condition_name}」未能开始录制：{exc}")

    def _reject_recording(self, attempt, detail):
        if self._in_flight is attempt:
            self._in_flight = None
            self.refresh_recording_state()
            self._report_recording_error(attempt, detail)

    def _report_recording_error(self, attempt, detail):
        logging.getLogger("core").warning("Workbench recording %s: %s", attempt.request_id, detail)
        if not self._recording.is_closing():
            self._recording.notify_error(detail)

    def _matching_recording_terminal(self, session, result):
        attempt = self._in_flight
        if attempt is None or attempt.archiving or attempt.request_id != session.request.request_id:
            return None
        if attempt.request_id != result.request_id:
            return None
        return attempt

    def _take_recording_terminal(self, session, result):
        attempt = self._matching_recording_terminal(session, result)
        if attempt is not None:
            self._in_flight = None
        return attempt

    def _recording_completed(self, session, result):
        attempt = self._matching_recording_terminal(session, result)
        if attempt is None:
            return
        archiving = replace(attempt, archiving=True)
        self._in_flight = archiving
        self.refresh_recording_state()
        try:
            self._recording.submit_archive(
                result, attempt.product_name, attempt.recorded_on,
                lambda request_id, error: self._recording_archived(request_id, result, error),
            )
        except Exception as error:
            if self._in_flight is not archiving:
                # A synchronous callback already finished archival. Preserve consumer errors.
                raise
            logging.getLogger("core").exception("Could not submit workbench archive %s", result.request_id)
            self._recording_archived(result.request_id, result, error)

    def _recording_archived(self, request_id, result, error):
        attempt = self._in_flight
        if attempt is None or not attempt.archiving or attempt.request_id != request_id:
            return
        self._in_flight = None
        if attempt.configuration_revision == self._configuration_revision:
            self._condition_index = (self._condition_index + 1) % len(self._applied_config.recording.context.conditions)
        self.refresh_recording_state()
        if error is not None:
            logging.getLogger("core").warning(
                "Workbench archive %s failed for %s: %s", request_id,
                [station.path for station in result.stations], error,
            )
            if not self._recording.is_closing():
                reason = " ".join(str(error).split())[:160] or type(error).__name__
                self._recording.notify_archive_error(
                    f"工况「{attempt.condition_name}」音频已保存，入库失败：{reason}"
                )
        self._recording.completed(result)
        if self._analysis_input_ready is not None and not self._recording.is_closing():
            self._analysis_input_ready(WorkbenchAnalysisInput(attempt.analysis_context, result))

    def _recording_failed(self, session, failure):
        attempt = self._take_recording_terminal(session, failure)
        if attempt is not None:
            self.refresh_recording_state()
            self._report_recording_error(
                attempt, f"工况「{attempt.condition_name}」录制失败（{failure.stage}）：{failure.message}",
            )

    def _recording_cancelled(self, session, cancellation):
        if self._take_recording_terminal(session, cancellation) is not None:
            self.refresh_recording_state()

    def _recording_release_failed(self, attempt, session, detail):
        if attempt.request_id != session.request.request_id:
            return
        self.refresh_recording_state()
        message = f"工况「{attempt.condition_name}」录制暂存文件清理失败：{detail}"
        logging.getLogger("core").warning("Workbench recording %s: %s", attempt.request_id, message)
        if not self._recording.is_closing():
            self._recording.notify_cleanup_error(message)

    def _recording_released(self, _session):
        self.refresh_recording_state()

    def get_plot_bindings(self, row: int, column: int) -> _PlotBindings:
        """Return the two condition bindings at a zero-based plot position."""
        if 0 <= row < len(self._plot_bindings):
            bindings = self._plot_bindings[row]
            if 0 <= column < len(bindings):
                return bindings[column]
        return None, None

    def get_condition_config(self, condition_id: str) -> WorkbenchCondition | None:
        """Return the saved runtime configuration for a condition."""
        if self._applied_config is None:
            return None
        return self._applied_config.recording.context.get_condition(condition_id)

    def get_station_channels(self, station_id: str) -> tuple[WorkbenchChannel, ...] | None:
        """Return the ordered, shared channel mapping for a station."""
        if self._applied_config is None:
            return None
        station = self._applied_config.recording.context.get_station(station_id)
        return station.channels if station is not None else None

    def get_recording_config(self):
        return self._applied_config.recording if self._applied_config is not None else None

    def get_analysis_config(self):
        return self._applied_config.analysis if self._applied_config is not None else None

    def get_ui_config(self):
        return self._applied_config.ui if self._applied_config is not None else None

    def bind_analysis_input(self, callback: Callable[[WorkbenchAnalysisInput], None] | None) -> None:
        """Deliver successful bound inputs without retaining a history or executing analysis."""
        self._analysis_input_ready = callback

    def open_video_settings(self) -> None:
        """Forward the video settings intent to the existing settings entry point."""
        self._show_video_settings()

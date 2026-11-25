"""Coordinate product configuration and user intents across workbench views.

Send cross-widget business updates through the controller. Views render data
and emit intents. The repository reads committed configuration, and the
configuration editors own writes and send commit notifications to the caller.
"""

import logging
import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4

from base.audio_label_policy import (
    audio_label_for_results, judgement_summary, reconcile_terminal_judgement, required_judgements,
)
from base.analysis.input_adapter import build_analysis_batch
from base.analysis.contracts import AnalysisInputError
from base.audio_label_sync import AudioLabelJob
from base.motor_daily_records import DailyRecordSummary
from base.motor_daily_wav_records import DailyWavRecord
from base.record_analysis_reference import RecordAnalysisReference
from base.workbench_config_values import WorkbenchChannel, WorkbenchCondition
from base.workbench_round_counter import WorkbenchRoundCounter
from base.workbench_wav_details_store import WavDetailsRow, WorkbenchWavDetailsSnapshot
from base.workbench_wav_state import WorkbenchWavState
from base.workbench_round_store import RoundSnapshot, WorkbenchRoundStateError
from base.workbench_summary_store import WorkbenchSummarySnapshot
from base.workbench_configuration_loader import WorkbenchConfiguration, WorkbenchConfigurationError
from base.workbench_analysis_input import (
    AnalysisRecordingContext, WorkbenchAnalysisInput, WorkbenchAnalysisInputError,
    build_analysis_payload, check_recorded_analysis_input, make_analysis_recording_context,
)
from ui.sequence.workbench_ui_projection import build_ui_projection, WorkbenchUiProjection, _PlotBindings
from ui.sequence.workbench_plot_presenter import WorkbenchPlotPresenter
from ui.sequence.workbench_result_state import WorkbenchResultState
from base.recording_service import RecordingCallbacks
from base.workbench_recording_protocol import WorkbenchRecordingCompletion
from base.workbench_recording_request import (
    WorkbenchRecordingRequestError,
    build_recording_request,
    recording_config_key,
)
from ui.recording_service_bridge import RecordingServiceBridge
from ui.workbench_recording_device import WorkbenchRecordingDevice, WorkbenchRecordingDeviceSnapshot


def _tcp_spl_value(payload):
    items = [item for item in payload.get('instance_results', ())
             if item.get('analysis_type') == 'SPL']
    if len(items) != 1 or items[0].get('execution_status') != '分析完成':
        return None
    raw = items[0].get('metrics', {}).get('overall_spl')
    return raw if type(raw) in (int, float) and math.isfinite(raw) else None


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
class WorkbenchProductSelectionPorts:
    product_manager: object
    show_catalog: Callable[[tuple[tuple[str, str], ...], str], None]
    set_enabled: Callable[[bool], None]
    is_busy: Callable[[], bool]
    is_closing: Callable[[], bool]
    notify_error: Callable[[str], None]


@dataclass(frozen=True)
class _RecordingRound:
    round_id: str
    analysis_version: str
    source_identity: str
    display_revision: int
    recorded_at: datetime
    condition_ids: tuple[str, ...]
    stations: tuple[tuple[str, int, tuple[str, ...]], ...]
    product_code: str
    round_snapshot: RoundSnapshot | None = None
    expected_analysis_keys: frozenset[tuple[str, str]] = frozenset()


@dataclass(frozen=True)
class _InFlightRecording:
    request_id: str
    configuration_revision: int
    condition_name: str
    product_name: str
    recorded_on: date
    recorded_at: datetime
    round_context: _RecordingRound
    analysis_context: AnalysisRecordingContext | None = None
    archiving: bool = False
    starts_round: bool = False


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
        self._plot_presenter = WorkbenchPlotPresenter(plot_grid)
        self._plot_grid.set_mark_buttons_visible(False)
        self._set_active_product = set_active_product
        self._show_video_settings = show_video_settings
        self._initialized = False
        self._applied_config: WorkbenchConfiguration | None = None
        self._applied_source_identity = ""
        self._product_selection: WorkbenchProductSelectionPorts | None = None
        self._product_catalog: tuple[tuple[str, str], ...] = ()
        self._plot_bindings: tuple[tuple[_PlotBindings, ...], ...] = ()
        self._recording: WorkbenchRecordingPorts | None = None
        self._configuration_valid = False
        self._analysis_version = None
        self._analysis_input_ready = None
        self._analysis_service = None
        self._record_analysis = None
        self._analysis_event_received = None
        self._submit_audio_label: Callable[[AudioLabelJob], None] | None = None
        self._analysis_active = False
        self._analysis_closing = False
        self._analysis_tasks = {}
        self._latest_analysis_tasks = {}
        self._recording_key = None
        self._configuration_revision = 0
        self._condition_index = 0
        self._in_flight: _InFlightRecording | None = None
        self._metric_keys = {}
        self._display_revision = 0
        self._recording_round: _RecordingRound | None = None
        self._current_round_id: str | None = None
        self._activated_round_ids: set[str] = set()
        self._result_state = WorkbenchResultState()
        self.tcp_result_callback = None
        self._tcp_spl_round = None
        self._tcp_spl_values = {}
        self._plot_columns = {}
        self._set_daily_records = None
        self._set_summary_counts: Callable[[WorkbenchSummarySnapshot], None] | None = None
        self._summary_baseline: WorkbenchSummarySnapshot | None = None
        self._last_summary_day: date | None = None
        self._wav_state = WorkbenchWavState()
        self._set_daily_wav_records = None
        self._wav_details_baseline: WorkbenchWavDetailsSnapshot | None = None
        self._last_wav_details_day: date | None = None
        self._round_counter: WorkbenchRoundCounter | None = None
        self._show_round = None
        self._notify_round_error = None
        self._round_error = None

    @property
    def recording_busy(self) -> bool:
        """Capture and archive ownership remain in flight until completion."""
        return self._in_flight is not None

    def bind_round_counter(self, counter, show_round, notify_error) -> None:
        """Attach the persisted daily number; standalone legacy callers may omit it."""
        self._round_counter = counter
        self._show_round = show_round
        self._notify_round_error = notify_error
        self.refresh_recording_state()

    def _update_round_counter(self, operation, *args) -> bool:
        """Handle the store's semantic errors at the GUI boundary, retaining retries."""
        try:
            operation(*args)
        except WorkbenchRoundStateError as error:
            message = str(error)
            changed = message != self._round_error
            # Modal notifications may pump the periodic GUI refresh themselves.
            self._round_error = message
            if changed:
                logging.getLogger("core").warning("Workbench round state: %s", message)
                self._notify_round_error(message)
            return False
        self._round_error = None
        self._show_round(self._round_counter.current_snapshot.number)
        return True

    def _refresh_round_counter(self, day=None) -> bool:
        if self._round_counter is not None and self._in_flight is not None:
            context = self._in_flight.round_context
            if context.round_id not in self._activated_round_ids:
                # Admission may pump GUI events before the bridge accepts. Keep
                # the already persisted WAV identity until acceptance/rejection.
                day = context.round_snapshot.day
        return (self._round_counter is None or self._update_round_counter(
            self._round_counter.refresh, day if day is not None else datetime.now().date(),
        ))

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

        source_identity = self._repository.source_identity if config is not None else ""
        source_changed = ((self._product_selection is not None or self._round_counter is not None)
                          and source_identity != self._applied_source_identity)
        if config != self._applied_config or source_changed:
            self._apply_configuration(config)
        else:
            self._applied_source_identity = source_identity
        self._configuration_valid = True
        self.refresh_product_catalog()
        self._refresh_recording_configuration()
        self._sync_analysis_stations()

    def bind_product_selection(self, ports: WorkbenchProductSelectionPorts) -> None:
        self._product_selection = ports
        self.refresh_product_catalog()

    def _selected_product_file(self) -> str:
        return Path(self._applied_source_identity).name if self._applied_source_identity else ""

    def _product_selection_allowed(self) -> bool:
        ports = self._product_selection
        return (
            ports is not None and self._in_flight is None
            and not ports.is_closing() and not ports.is_busy()
        )

    def refresh_product_selection_state(self) -> None:
        if self._product_selection is not None:
            self._product_selection.set_enabled(self._product_selection_allowed())

    def _publish_product_selection(self) -> None:
        if self._product_selection is not None:
            self._product_selection.show_catalog(self._product_catalog, self._selected_product_file())
            self.refresh_product_selection_state()

    def _report_product_selection_error(self, message: str) -> None:
        logging.getLogger("core").warning("Workbench product selection: %s", message)
        self._publish_product_selection()
        self._product_selection.notify_error(message)

    def refresh_product_catalog(self) -> None:
        if self._product_selection is None:
            return
        try:
            catalog = self._repository.load_product_catalog()
        except WorkbenchConfigurationError as error:
            self._report_product_selection_error(str(error))
            return
        self._product_catalog = catalog
        self._publish_product_selection()

    def select_product(self, file_name: str) -> None:
        """Prepare one candidate, persist its selection, then publish that snapshot."""
        if not self._product_selection_allowed():
            self._publish_product_selection()
            return
        try:
            catalog = self._repository.load_product_catalog()
            if not any(file == file_name for file, _ in catalog):
                raise WorkbenchConfigurationError(f"产品配置 {file_name!r} 已不在配置列表中")
            self._product_catalog = catalog
            if file_name == self._selected_product_file():
                self._publish_product_selection()
                return
            candidate = self._repository.load_candidate(file_name)
            projection = build_ui_projection(candidate.ui, candidate.analysis)
        except WorkbenchConfigurationError as error:
            self._report_product_selection_error(str(error))
            return
        if not self._product_selection_allowed():
            self._publish_product_selection()
            return
        try:
            saved, message = self._product_selection.product_manager.set_active_product(file_name)
        except (OSError, ValueError) as error:
            self._report_product_selection_error(str(error))
            return
        if not saved:
            self._report_product_selection_error(message)
            return
        self._repository.accept_product_source(file_name)
        self._apply_configuration(candidate, projection)
        self._initialized = True
        self._configuration_valid = True
        self._refresh_recording_configuration()
        self._sync_analysis_stations()

    def bind_analysis_service(self, service, event_received: Callable[[dict], None]) -> None:
        """Attach the backend; activation waits until the main window is ready."""
        if self._analysis_service is not None:
            raise RuntimeError("Analysis service is already bound")
        self._analysis_service = service
        self._analysis_event_received = event_received
        self.refresh_recording_state()

    def bind_record_analysis(self, coordinator) -> None:
        """Borrow the optional record-view coordinator from the host runtime."""
        self._record_analysis = coordinator

    def open_record_analysis(self, record, parent):
        """Forward a saved-record intent without changing production state."""
        if self._record_analysis is not None:
            return self._record_analysis.open_record(record, parent)
        return None

    def bind_result_summary(self, set_daily_records: Callable | None) -> None:
        """Attach the optional complete daily snapshot consumer on the GUI thread."""
        self._set_daily_records = set_daily_records

    def bind_summary_counts(self, render: Callable[[WorkbenchSummarySnapshot], None],
                            baseline: WorkbenchSummarySnapshot | None, day: date) -> None:
        """Attach the home summary renderer and publish the current day's counts."""
        self._set_summary_counts = render
        self._summary_baseline = baseline
        self.refresh_summary_counts(day, force=True)

    def refresh_summary_counts(self, day: date, *, force: bool = False) -> WorkbenchSummarySnapshot | None:
        """Publish on a day change or explicit refresh and return the rendered snapshot."""
        if self._set_summary_counts is None or (day == self._last_summary_day and not force):
            return None
        summary = DailyRecordSummary.from_results(row["result"] for row in self._result_state.records_for(day))
        ok, ng, unmarked = summary.ok, summary.ng, summary.unmarked
        baseline = self._summary_baseline
        if baseline is not None and baseline.day == day:
            ok, ng, unmarked = ok + baseline.ok, ng + baseline.ng, unmarked + baseline.unmarked
        snapshot = WorkbenchSummarySnapshot(day, ok, ng, unmarked)
        self._set_summary_counts(snapshot)
        self._last_summary_day = day
        return snapshot

    def bind_wav_details(
        self, set_daily_wav_records: Callable | None, *,
        baseline: WorkbenchWavDetailsSnapshot | None = None, day: date | None = None,
    ) -> None:
        """Bind the consumer and optionally supply the restored baseline."""
        self._set_daily_wav_records = set_daily_wav_records
        if baseline is not None:
            self._wav_details_baseline = baseline
        self._refresh_live_wav_details(datetime.now().date() if day is None else day)

    def refresh_wav_details(self, day: date, *, force: bool = False) -> WorkbenchWavDetailsSnapshot | None:
        """Merge and publish the day's saved WAVs on rollover or explicit refresh."""
        if day == self._last_wav_details_day and not force:
            return None
        snapshot = WorkbenchWavDetailsSnapshot(day, tuple(
            WavDetailsRow(*key, DailyWavRecord(**fields))
            for key, fields in self._merged_wav_details(day).items()
        ))
        self._publish_wav_details(day, (row.to_display_dict() for row in snapshot.rows))
        return snapshot

    def _merged_wav_details(self, day: date):
        baseline = self._wav_details_baseline
        restored = baseline.rows if baseline is not None and baseline.day == day else ()
        merged = {row.key: row.to_display_dict() for row in restored}
        for key, fields in self._wav_state.identified_records_for(day):
            merged.setdefault(key, fields)
        return merged

    def _refresh_live_wav_details(self, day: date) -> None:
        # The display consumer already validates rows. Only explicit refresh needs a typed snapshot.
        rows = self._merged_wav_details(day).values() if self._set_daily_wav_records is not None else ()
        self._publish_wav_details(day, rows)

    def _publish_wav_details(self, day: date, rows) -> None:
        if self._set_daily_wav_records is not None:
            self._set_daily_wav_records(list(rows), day)
        self._last_wav_details_day = day

    def bind_audio_label_sync(self, submit: Callable[[AudioLabelJob], None] | None) -> None:
        """Attach the optional label job sink; persistence belongs to its owner."""
        self._submit_audio_label = submit

    def activate_analysis(self) -> None:
        if self._analysis_active or self._analysis_closing:
            return
        self._analysis_active = True
        self._sync_analysis_stations()

    def _sync_analysis_stations(self) -> None:
        if not self._analysis_active or self._analysis_closing or not self._configuration_valid:
            return
        analysis = self.get_analysis_config()
        station_ids = [station.station_id for station in analysis.context.stations] if analysis else []
        receipt = self._analysis_service.sync_stations(station_ids)
        if not receipt["accepted"]:
            self._analysis_event_received({
                "event_type": "configuration_rejected", "payload": receipt,
            })
        self.refresh_recording_state()

    def poll_analysis_events(self) -> None:
        """The host's sole timer drives the backend and delivers events on its thread."""
        events = self._analysis_service.poll_events()
        for event in events:
            if self._record_analysis is not None and self._record_analysis.handle_analysis_event(event):
                continue
            task_id = event.get("task_id")
            context = self._analysis_tasks.get(task_id)
            if context is not None:
                key = context["station_id"], context["condition_id"]
                current = (context["analysis_version"] == self._analysis_version
                           and self._latest_analysis_tasks.get(key) == task_id
                           and context["round_id"] == self._current_round_id
                           and self._plot_presenter.matches_request(context))
                if event["event_type"] in {"judgement_ready", "task_result"}:
                    accepted = self._consume_analysis_event(event, context, current)
                    if event["event_type"] == "judgement_ready" and not accepted:
                        continue
                event = {**event, **context, "is_current": current}
                if event["event_type"] == "task_result":
                    del self._analysis_tasks[task_id]
                    if self._latest_analysis_tasks.get(key) == task_id:
                        del self._latest_analysis_tasks[key]
            self._plot_presenter.analysis_event(event)
            self._analysis_event_received(event)
        if events:
            self.refresh_recording_state()

    def _authenticated_analysis_payload(self, event, context):
        """Authenticate raw identities before forwarding merges the task context."""
        task_id = event["task_id"]
        payload = event.get("payload", {})
        expected = {"task_id": task_id, "station_id": context["station_id"],
                    "condition_id": context["condition_id"]}
        if not isinstance(payload, dict):
            logging.getLogger("core").warning(
                "Workbench result task %s station %s condition %s: invalid payload %r",
                task_id, context["station_id"], context["condition_id"], payload,
            )
            return
        for source_name, source in (("event", event), ("payload", payload)):
            for field, identity in expected.items():
                required = source_name == "payload" and field in ("station_id", "condition_id")
                if (field in source or required) and source.get(field) != identity:
                    logging.getLogger("core").warning(
                        "Workbench result task %s station %s condition %s: %s.%s %r does not match %r",
                        task_id, context["station_id"], context["condition_id"],
                        source_name, field, source.get(field), identity,
                    )
                    return
        return payload

    def _consume_analysis_event(self, event, context, is_current):
        payload = self._authenticated_analysis_payload(event, context)
        if payload is None:
            return False
        if event["event_type"] == "judgement_ready":
            return self._consume_early_judgement(event, context, is_current, payload)
        self._consume_analysis_result(event, context, is_current, payload)
        return True

    def _consume_early_judgement(self, event, context, is_current, payload):
        expected_items = context.get("expected_judgements", [])
        if "early_judgements" in context or not expected_items:
            return False
        items = payload.get("instance_results", ())
        if not isinstance(items, (list, tuple)) or not all(isinstance(item, dict) for item in items):
            logging.getLogger("core").warning("Malformed early judgement for %s", event["task_id"])
            return False
        summary = judgement_summary(items, expected_items)
        if summary["missing_required"] or len(items) != len(expected_items):
            logging.getLogger("core").warning("Incomplete early judgement message for %s", event["task_id"])
            return False
        payload = {**payload, "judgement_summary": summary}
        snapshot = self._result_state.receive_judgement(
            context["round_id"], context["station_id"], context["condition_id"], payload,
        )
        if snapshot is None:
            return False
        context["early_judgements"] = deepcopy(items)
        event["payload"] = payload
        if is_current:
            self._show_analysis_snapshot(context, snapshot)
        return True

    def _consume_analysis_result(self, event, context, is_current, payload):
        task_id = event["task_id"]
        if "expected_judgements" in context:
            summary, conflict = reconcile_terminal_judgement(
                payload, context["expected_judgements"], context.get("early_judgements"),
            )
            payload = {**payload, "judgement_summary": summary}
            if conflict:
                evidence_fields = ("config_key", "analysis_type", "channel_index",
                                   "execution_status", "judgement_status", "judgement", "error")
                early_evidence, terminal_evidence = (
                    [{key: item.get(key) for key in evidence_fields} for item in items]
                    for items in (context["early_judgements"], payload.get("instance_results", ()))
                )
                logging.getLogger("core").error(
                    "Judgement conflict for task %s: early=%r terminal=%r original_error=%r",
                    task_id, early_evidence, terminal_evidence, payload.get("error"),
                )
                payload["error"] = {"stage": "judgement", "type": "JudgementConflict",
                                    "message": "提前判定与完整结果不一致，请检查分析诊断"}
            event["payload"] = payload
        snapshot = self._result_state.receive(
            context["round_id"], context["station_id"], context["condition_id"], payload,
        )
        if snapshot is None:
            return
        if self._submit_audio_label is not None:
            self._submit_audio_label(AudioLabelJob(
                task_id, context["wav_path"],
                audio_label_for_results(payload),
                context.get("result_directory", ""), payload.get("judgement_summary"),
            ))
        if self._round_counter is not None:
            self._update_round_counter(
                self._round_counter.analysis_completed, context["round_id"],
                (context["condition_id"], context["station_id"]), datetime.now().date(),
            )
        if is_current:
            if self.tcp_result_callback is not None:
                if self._tcp_spl_round != context['round_id']:
                    self._tcp_spl_round = context['round_id']
                    self._tcp_spl_values = {}
                self._tcp_spl_values[context['station_id'], context['condition_id']] = _tcp_spl_value(payload)
                directions = context['condition_ids']
                self.tcp_result_callback({
                    'type': 'station_result', 'round_id': context['round_id'],
                    'station': context['station_ordinal'],
                    'forward': snapshot.forward, 'reverse': snapshot.reverse,
                    'result': snapshot.result,
                    'forward_spl': self._tcp_spl_values.get((context['station_id'], directions[0])) if directions else None,
                    'reverse_spl': self._tcp_spl_values.get((context['station_id'], directions[1])) if len(directions) > 1 else None,
                })
            self._show_analysis_snapshot(context, snapshot)
        if snapshot.complete and (self._set_daily_records is not None or self._set_summary_counts is not None):
            today = datetime.now().date()
            if self._set_daily_records is not None:
                self._set_daily_records(self._result_state.records_for(today), today)
            self.refresh_summary_counts(today, force=True)
        if self._wav_state.update_result(
            context["request_id"], context["station_id"], payload,
        ):
            self._refresh_live_wav_details(datetime.now().date())

    def _show_analysis_snapshot(self, context, snapshot):
        self._station_panel.show_condition_result(
            context["station_ordinal"], snapshot.condition_statuses,
            snapshot.result, snapshot.rows,
        )
        column = self._plot_columns.get(context["station_id"])
        if snapshot.judgement_complete and column is not None:
            self._plot_grid.set_station_result(column, snapshot.result)

    def _submit_saved_recording(self, item: WorkbenchAnalysisInput, round_context: _RecordingRound) -> None:
        """Submit published file paths, never sample buffers or the current configuration."""
        context = item.context
        condition = context.analysis.context.get_condition(context.condition_id)
        if not context.analysis.test_queues[condition.test_queue].analysis_config["display_sequence"]:
            self._plot_presenter.recording_state(context, '未配置分析项')
            self._analysis_event_received({
                "event_type": "analysis_skipped", "request_id": context.request_id,
                "analysis_version": context.analysis_version, "payload": {"reason": "no_analysis_items"},
            })
            return
        try:
            # Published WAVs are under <project>/audio. Pass the project root;
            # analysis owns the sibling csv/<WAV stem> output directory.
            directories = {
                (context.request_id, station.station_id): str(
                    Path(station.path).parent.parent)
                for station in item.completion.stations
            }
            payload = build_analysis_payload((item,), result_directories=directories)
            batch = build_analysis_batch(payload)
            issues = check_recorded_analysis_input(item, batch)
            expected_by_task = {t.task_id: required_judgements(t.instances) for t in batch.tasks}
        except (WorkbenchAnalysisInputError, AnalysisInputError) as error:
            receipt = {"accepted": False, "precheck_failed": True, "accepted_task_ids": [], "issues": [{
                "task_id": None, "field_path": "$input", "error_type": type(error).__name__,
                "message": str(error),
            }]}
        else:
            receipt = ({"accepted": False, "precheck_failed": True, "accepted_task_ids": [],
                        "issues": [asdict(issue) for issue in issues]}
                       if issues else self._analysis_service.submit_batch(payload))
            if receipt["accepted"]:
                for task in payload["tasks"]:
                    task_id = task["task_id"]
                    _, ordinal, keys = next(station for station in round_context.stations
                                            if station[0] == task["station_id"])
                    self._analysis_tasks[task_id] = {
                        "request_id": context.request_id, "analysis_version": context.analysis_version,
                        "wav_path": task["wav_path"],
                        "expected_judgements": expected_by_task[task_id],
                        "result_directory": task.get("analysis_result_directory", ""),
                        "station_id": task["station_id"], "condition_id": task["condition_id"],
                        "round_id": round_context.round_id,
                        "recorded_at": round_context.recorded_at,
                        "source_identity": round_context.source_identity,
                        "condition_ids": round_context.condition_ids,
                        "station_ordinal": ordinal, "metric_keys": keys,
                        "product_code": round_context.product_code,
                    }
                    self._latest_analysis_tasks[task["station_id"], task["condition_id"]] = task_id
                if context.analysis_version == self._analysis_version and payload['tasks']:
                    self._plot_presenter.begin_batch(context.request_id, payload['tasks'])
        if not receipt['accepted']:
            self._cancel_rejected_round(round_context)
        if context.analysis_version == self._analysis_version:
            self._plot_presenter.recording_state(
                context, ('待分析' if receipt['accepted'] else
                          '检查未通过' if receipt.get('precheck_failed') else '分析未提交'),
                '\n'.join(issue['message'] for issue in receipt.get('issues', [])),
            )
        self._analysis_event_received({
            "event_type": "submission_result", "request_id": context.request_id,
            "analysis_version": context.analysis_version, "payload": receipt,
        })

    def begin_analysis_shutdown(self) -> None:
        if self._analysis_closing:
            return
        self._analysis_closing = True
        if self._record_analysis is not None:
            self._record_analysis.begin_shutdown()
        self._analysis_service.shutdown()

    def _cancel_rejected_round(self, context):
        """A rejected attempt has no verdict and must not wait for a worker event."""
        cancelled = self._result_state.cancel_round(context.round_id)
        if self._round_counter is not None:
            self._update_round_counter(self._round_counter.cancel_round,
                                       context.round_id, datetime.now().date())
        if context.round_id != self._current_round_id:
            return
        self._condition_index = 0
        self._recording_round = None
        self._current_round_id = None
        self._plot_presenter.cancel_round(cancelled)
        for station_id, ordinal, _ in context.stations:
            if station_id in cancelled:
                snapshot = cancelled[station_id]
                self._station_panel.show_condition_result(
                    ordinal, snapshot.condition_statuses, snapshot.result, snapshot.rows,
                )
                column = self._plot_columns.get(station_id)
                if column is not None:
                    self._plot_grid.set_station_result(column, "--")

    def bind_recording(self, ports: WorkbenchRecordingPorts) -> None:
        self._recording = ports
        self._refresh_recording_configuration()

    def _refresh_recording_configuration(self):
        if self._recording is None:
            self.refresh_recording_state()
            return
        config = self._applied_config
        key = recording_config_key(config.recording, source_identity=self._applied_source_identity) if config is not None else None
        if key != self._recording_key:
            self._recording_key = key
            self._configuration_revision += 1
            self._condition_index = 0
        self.refresh_recording_state()

    def _apply_configuration(
        self, config: WorkbenchConfiguration | None, projection: WorkbenchUiProjection | None = None,
    ) -> None:
        source_identity = self._repository.source_identity if config is not None else ""
        if config is None:
            stations, names, cells, bindings, product_name = (), (), (), (), ""
        else:
            if projection is None:
                projection = build_ui_projection(config.ui, config.analysis)
            stations = projection.stations
            names, cells, bindings = projection.column_names, projection.cells, projection.bindings
            product_name = config.recording.context.project_name

        previous = self.get_analysis_config()
        analysis = config.analysis if config is not None else None
        analysis_version = self._analysis_version
        source_changed = ((self._product_selection is not None or self._round_counter is not None)
                          and source_identity != self._applied_source_identity)
        if analysis != previous or source_changed:
            analysis_version = uuid4().hex if analysis is not None else None
            self._latest_analysis_tasks.clear()

        logging.getLogger("core").info("Applying workbench configuration from %s", source_identity)
        self._station_panel.show_workbench_stations(
            stations, condition_names=tuple(c.condition_name for c in analysis.context.conditions)
            if analysis else (),
        )
        self._plot_grid.show_workbench_layout(names, cells)
        self._plot_columns = {
            station.station_id: column
            for column, station in enumerate(analysis.context.stations if analysis else ())
            if column < len(names)
        }
        self._plot_presenter.configure(projection, invalidate_requests=analysis_version != self._analysis_version)
        if self._product_selection is None:
            self._set_active_product(product_name)
        self._applied_config = config
        self._applied_source_identity = source_identity
        self._analysis_version = analysis_version
        self._plot_bindings = bindings
        self._metric_keys = projection.metric_keys if projection is not None else {}
        self._display_revision += 1
        if self._round_counter is not None:
            if self._recording_round is not None:
                pending = self._in_flight.round_context if self._in_flight is not None else None
                day = (pending.round_snapshot.day
                       if pending is not None and pending.round_id not in self._activated_round_ids
                       else datetime.now().date())
                self._update_round_counter(
                    self._round_counter.abandon_round,
                    self._recording_round.round_id, day,
                )
            self._condition_index = 0
            self._configuration_revision += 1
        self._recording_round = None
        self._current_round_id = None
        self._publish_product_selection()

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
        condition = config.recording.context.conditions[self._condition_index]
        if (self._analysis_service is not None
                and config.analysis.test_queues[condition.test_queue].analysis_config["display_sequence"]):
            states = self._analysis_service.station_states
            waiting = [station.station_name for station in config.analysis.stations_for(condition.condition_id)
                       if states.get(station.station_id, {}).get("state") not in {"ready", "busy"}]
            if waiting:
                return None, "分析进程尚未就绪：" + "、".join(waiting)
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
        self.refresh_product_selection_state()
        round_ready = self._refresh_round_counter()
        if self._recording is not None:
            snapshot, detail = self._recording_admission()
            if not round_ready:
                snapshot, detail = None, self._round_error
            self._recording.set_button_state(snapshot is not None, detail, self._in_flight is not None)

    def start_recording(self) -> None:
        """Submit one frozen condition and hold the cursor through archival."""
        recorded_at = datetime.now()
        ports = self._recording
        if ports is None:
            return
        if not self._refresh_round_counter(recorded_at.date()):
            ports.set_button_state(False, self._round_error, self._in_flight is not None)
            return
        snapshot, detail = self._recording_admission()
        if snapshot is None:
            ports.set_button_state(False, detail, self._in_flight is not None)
            return
        config = self._applied_config
        analysis_version = self._analysis_version
        condition = config.recording.context.conditions[self._condition_index]
        product_model, sample_number, serial_number = ports.read_wav_identity()
        round_context = self._recording_round
        if self._condition_index == 0 or round_context is None:
            round_snapshot = None
            expected_keys = frozenset(
                (c.condition_id, station.station_id)
                for c in config.analysis.context.conditions
                if config.analysis.test_queues[c.test_queue].analysis_config["display_sequence"]
                for station in config.analysis.stations_for(c.condition_id)
            )
            if self._round_counter is not None:
                if not self._update_round_counter(self._round_counter.prepare_new_round, recorded_at.date()):
                    ports.set_button_state(False, self._round_error, False)
                    return
                round_snapshot = self._round_counter.current_snapshot
                sample_number = str(round_snapshot.number)
            round_context = _RecordingRound(
                uuid4().hex, analysis_version, self._applied_source_identity, self._display_revision,
                recorded_at, tuple(c.condition_id for c in config.recording.context.conditions),
                tuple((station.station_id, ordinal, self._metric_keys[station.station_id])
                      for ordinal, station in enumerate(config.analysis.context.stations, 1)),
                sample_number,
                round_snapshot, expected_keys,
            )
        elif self._round_counter is not None:
            sample_number = str(round_context.round_snapshot.number)
        attempt = _InFlightRecording(
            uuid4().hex, self._configuration_revision, condition.condition_name,
            config.recording.context.project_name, recorded_at.date(), recorded_at, round_context,
            starts_round=self._condition_index == 0,
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
                started=self._recording_started,
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
        else:
            self._activate_recording_round(attempt)

    def _activate_recording_round(self, attempt):
        """Activate only admitted requests, once, before any terminal consumer runs."""
        context = attempt.round_context
        if context.round_id in self._activated_round_ids:
            return
        self._activated_round_ids.add(context.round_id)
        if self._round_counter is not None:
            self._round_counter.open_round(
                context.round_id, context.round_snapshot, context.expected_analysis_keys,
            )
        self._result_state.open_round(
            context.round_id, analysis_version=context.analysis_version,
            source_identity=context.source_identity, recorded_at=context.recorded_at,
            condition_ids=context.condition_ids, stations=context.stations,
            product_code=context.product_code,
            expected_analysis_keys=context.expected_analysis_keys,
        )
        if context.display_revision != self._display_revision:
            if self._round_counter is not None:
                self._update_round_counter(
                    self._round_counter.abandon_round, context.round_id, datetime.now().date(),
                )
            return
        self._recording_round = context
        self._current_round_id = context.round_id
        self._plot_grid.clear_station_results()
        for _, ordinal, keys in context.stations:
            self._station_panel.show_condition_result(
                ordinal, ("--",) * len(context.condition_ids), "--",
                [[key, *(["--"] * len(context.condition_ids))] for key in keys],
            )

    def _recording_started(self, session):
        attempt = self._in_flight
        if (attempt is None or attempt.request_id != session.request.request_id
                or attempt.analysis_context is None
                or attempt.analysis_context.analysis_version != self._analysis_version):
            return
        self._plot_presenter.recording_started(attempt.analysis_context, starts_round=attempt.starts_round)

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
            self._activate_recording_round(attempt)
            if self._round_counter is not None and self._condition_index == 0:
                # A failed first condition has no saved WAV identity to continue.
                # A later condition retries the existing, partially saved round.
                self._update_round_counter(
                    self._round_counter.abandon_round,
                    attempt.round_context.round_id, datetime.now().date(),
                )
        return attempt

    def _recording_completed(self, session, result):
        attempt = self._matching_recording_terminal(session, result)
        if attempt is None:
            return
        # Completion also establishes ownership if no started notification was delivered.
        self._recording_started(session)
        self._plot_presenter.recording_state(attempt.analysis_context, '保存中')
        archiving = replace(attempt, archiving=True)
        self._in_flight = archiving
        self._activate_recording_round(attempt)
        self.refresh_recording_state()
        station_numbers = {station_id: ordinal
                           for station_id, ordinal, _ in attempt.round_context.stations}
        condition_index = attempt.round_context.condition_ids.index(attempt.analysis_context.condition_id)
        condition = ("正转", "反转")[condition_index] if condition_index < 2 else attempt.condition_name
        if self._wav_state.register_recording(
            attempt.request_id, attempt.recorded_at, condition,
            tuple((station.station_id, station_numbers[station.station_id], station.path)
                  for station in result.stations),
            analysis_references={station.station_id: RecordAnalysisReference(
                result.request_id, station.station_id, result.condition_id, station.path, station.channels,
            ) for station in result.stations},
        ):
            self._refresh_live_wav_details(datetime.now().date())
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
            if self._condition_index == 0 and self._round_counter is not None:
                self._update_round_counter(
                    self._round_counter.recordings_completed,
                    attempt.round_context.round_id, datetime.now().date(),
                )
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
        item = WorkbenchAnalysisInput(attempt.analysis_context, result)
        if self._analysis_service is not None and not self._recording.is_closing():
            # Submit before outward completion callbacks can start another round.
            self._submit_saved_recording(item, attempt.round_context)
        self._recording.completed(result)
        if self._analysis_input_ready is not None and not self._recording.is_closing():
            self._analysis_input_ready(item)

    def _recording_failed(self, session, failure):
        attempt = self._take_recording_terminal(session, failure)
        if attempt is not None:
            self._plot_presenter.recording_state(attempt.analysis_context, '采集失败', failure.message)
            self.refresh_recording_state()
            self._report_recording_error(
                attempt, f"工况「{attempt.condition_name}」录制失败（{failure.stage}）：{failure.message}",
            )

    def _recording_cancelled(self, session, cancellation):
        attempt = self._take_recording_terminal(session, cancellation)
        if attempt is not None:
            self._plot_presenter.recording_state(attempt.analysis_context, '已取消')
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
        """Return the condition bindings at a zero-based plot position."""
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

    def get_current_analysis_snapshot(self):
        """Return the valid applied analysis and version, without reloading files."""
        if not self._configuration_valid:
            return None, None
        return self.get_analysis_config(), self._analysis_version

    def get_ui_config(self):
        return self._applied_config.ui if self._applied_config is not None else None

    def bind_analysis_input(self, callback: Callable[[WorkbenchAnalysisInput], None] | None) -> None:
        """Observe completed inputs; the bound service already owns task submission."""
        self._analysis_input_ready = callback

    def open_video_settings(self) -> None:
        """Forward the video settings intent to the existing settings entry point."""
        self._show_video_settings()

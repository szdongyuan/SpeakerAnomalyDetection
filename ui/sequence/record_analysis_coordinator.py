"""Own record-view sessions while borrowing the host's resident analysis service."""

from collections import OrderedDict
from dataclasses import dataclass, field
import logging
from uuid import uuid4

from base.record_analysis_request import RecordAnalysisRequestError, build_record_analysis_request
from ui.sequence.record_analysis_results import build_record_analysis_results


@dataclass
class _Session:
    session_id: str
    parent_key: int
    record_key: object
    reference: object
    analysis: object
    version: object
    dialog: object
    presenter: object
    items: tuple = ()
    connections: list = field(default_factory=list)


@dataclass
class _Task:
    task_id: str
    station_id: str
    condition_id: str
    session_id: str | None
    provisional: bool = True
    buffered: list = field(default_factory=list)


class RecordAnalysisCoordinator:
    """Host-thread coordinator with injected service, snapshot and view factories.

    ``read_current_analysis()`` returns an immutable config (or None) and its
    version. ``create_view(record, parent)`` returns a dialog and presenter;
    neither this object nor its views owns the injected service's lifecycle.
    """

    def __init__(self, service, read_current_analysis, create_view, *, recent_task_capacity=128):
        if type(recent_task_capacity) is not int or recent_task_capacity < 1:
            raise ValueError('recent_task_capacity must be a positive integer')
        self._service = service
        self._read_current_analysis = read_current_analysis
        self._create_view = create_view
        self._recent_capacity = recent_task_capacity
        self._sessions = {}
        self._parents = {}
        self._tasks = {}
        self._recent = OrderedDict()
        self._logger = logging.getLogger(__name__)
        self._closing = False

    def open_record(self, record, parent):
        """Open/focus the parent's single view and submit at most one WAV task."""
        if self._closing:
            return None
        parent_key = id(parent)
        reference = record.analysis_reference
        record_key = reference.record_key if reference is not None else record
        previous = self._sessions.get(self._parents.get(parent_key))
        if previous is not None:
            if previous.record_key == record_key:
                previous.dialog.raise_()
                previous.dialog.activateWindow()
                return previous.dialog
            self._close_session(previous.session_id)
        dialog, presenter = self._create_view(record, parent)
        analysis, version = self._read_current_analysis()
        session_id, task_id = str(uuid4()), str(uuid4())
        session = _Session(session_id, parent_key, record_key, reference,
                           analysis, version, dialog, presenter)
        self._sessions[session_id] = session
        self._parents[parent_key] = session_id
        for signal, callback in (
            (dialog.view_closed, lambda: self._detach(session_id)),
            (dialog.destroyed, lambda *_: self._detach(session_id, destroyed=True)),
            (parent.destroyed, lambda *_: self._detach(session_id)),
        ):
            signal.connect(callback)
            session.connections.append((signal, callback))
        dialog.show_record(record)
        dialog.show()
        try:
            request = build_record_analysis_request(reference, analysis, task_id)
        except RecordAnalysisRequestError as error:
            presenter.show_status('无法分析该记录', str(error))
            return dialog
        session.items = request.items
        presenter.set_items(request.items)
        if not request.items:
            presenter.show_status('当前工况未配置分析项')
            return dialog
        presenter.show_status('分析中')
        task = _Task(task_id, reference.station_id, reference.condition_id, session_id)
        self._tasks[task_id] = task
        receipt = self._service.submit_batch(request.payload)
        if not receipt['accepted']:
            del self._tasks[task_id]
            task.buffered.clear()
            detail = '\n'.join(
                f"{issue.get('field_path', '')}: {issue['error_type']}: {issue['message']}"
                for issue in receipt['issues'])
            current = self._sessions.get(session_id)
            if current is not None:
                current.presenter.show_status('分析未提交', detail)
            return dialog
        task.provisional = False
        buffered, task.buffered = task.buffered, []
        for event in buffered:
            self.handle_analysis_event(event)
        return dialog

    def handle_analysis_event(self, event):
        """Consume only owned task events; preserve the host's service events."""
        kind = event.get('event_type')
        if kind == 'service_closed':
            self.begin_shutdown()
            return False
        if kind not in ('task_started', 'task_result'):
            return False
        task_id = event.get('task_id')
        if task_id in self._recent:
            return True
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if not self._authenticate(event, task, terminal=kind == 'task_result'):
            return True
        if task.provisional:
            task.buffered.append(event if task.session_id is not None else self._identity_event(event))
            return True
        if kind == 'task_result':
            # Complete ownership before calling the presentation port: Qt/user
            # callbacks can reenter the host event path while content is updated.
            del self._tasks[task_id]
            self._recent[task_id] = None
            while len(self._recent) > self._recent_capacity:
                self._recent.popitem(last=False)
            session = self._sessions.get(task.session_id)
            if session is not None:
                session.presenter.show_result(build_record_analysis_results(
                    session.items, event['payload'], channel_names={
                        ch.wav_channel_index: ch.channel_name for ch in session.reference.channels},
                ))
        return True

    def begin_shutdown(self):
        """Detach views without shutting down or reconfiguring the borrowed service."""
        self._closing = True
        for session_id in tuple(self._sessions):
            self._close_session(session_id)
        # Keep accepted identities until their terminal is consumed. service_closed
        # can precede a terminal in the host's current batch or in submit reentry;
        # moving all of them to bounded recent storage here would lose ownership.

    def _close_session(self, session_id):
        session = self._sessions.get(session_id)
        if session is not None:
            dialog = session.dialog
            self._detach(session_id)
            dialog.close()

    def _detach(self, session_id, *, destroyed=False):
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        self._parents.pop(session.parent_key, None)
        for signal, callback in session.connections:
            try:
                signal.disconnect(callback)
            except RuntimeError:
                # Qt can already have deleted the emitting QObject. Its native
                # connections are gone; our session is removed before cleanup.
                self._logger.debug('Record analysis session %s: QObject already destroyed', session_id)
        if not destroyed:
            session.presenter.dispose()
        for task in self._tasks.values():
            if task.session_id == session_id:
                task.session_id = None
                task.buffered = [self._identity_event(event) for event in task.buffered]

    @staticmethod
    def _identity_event(event):
        """Retain authenticated receipt-order information, never detached data."""
        keys = ('task_id', 'station_id', 'condition_id')
        return dict({key: event[key] for key in keys if key in event},
                    event_type=event['event_type'],
                    payload={key: event['payload'][key] for key in keys if key in event['payload']})

    def _authenticate(self, event, task, *, terminal):
        expected = dict(task_id=task.task_id, station_id=task.station_id,
                        condition_id=task.condition_id)
        payload = event.get('payload')
        if not isinstance(payload, dict):
            self._logger.warning('Record analysis task %s station %s condition %s: invalid payload %r',
                                 task.task_id, task.station_id, task.condition_id, payload)
            return False
        for source_name, source in (('event', event), ('payload', payload)):
            for key, value in expected.items():
                required = (terminal if source_name == 'payload' else key != 'condition_id')
                if (required or key in source) and source.get(key) != value:
                    self._logger.warning(
                        'Record analysis task %s station %s condition %s: %s.%s %r does not match %r',
                        task.task_id, task.station_id, task.condition_id,
                        source_name, key, source.get(key), value)
                    return False
        return True

"""Instance-owned daily workbench round state, independent of Qt and services."""

from dataclasses import dataclass, field
from datetime import date
from typing import Hashable

from base.workbench_round_store import RoundSnapshot, WorkbenchRoundStore


@dataclass
class _RoundState:
    snapshot: RoundSnapshot
    expected_keys: frozenset[tuple[str, str]]
    completed_keys: set[tuple[str, str]] = field(default_factory=set)
    recordings_finished: bool = False
    advanced: bool = False
    abandoned: bool = False


class WorkbenchRoundCounter:
    def __init__(self, store: WorkbenchRoundStore):
        self._store = store
        self._loaded = False
        self._current_snapshot = None
        self._pending_snapshot = None
        self._pending_round_id = None
        self._rounds = {}
        self._active_round_id = None

    @property
    def current_snapshot(self) -> RoundSnapshot | None:
        """Last successfully committed state; never exposes a pending number."""
        return self._current_snapshot

    @property
    def can_record(self) -> bool:
        return self._current_snapshot is not None and self._pending_snapshot is None

    def _commit_pending(self):
        if self._pending_snapshot is not None:
            self._store.save(self._pending_snapshot)
            self._current_snapshot = self._pending_snapshot
            if self._pending_round_id is not None:
                self._rounds[self._pending_round_id].advanced = True
            self._pending_snapshot = None
            self._pending_round_id = None

    def refresh(self, day: date) -> RoundSnapshot:
        """Load or synchronize the local day, retrying any exact pending save."""
        if not self._loaded:
            self._current_snapshot = self._store.load()
            self._loaded = True
        self._commit_pending()
        state = self._active_state()
        if state is not None and not state.recordings_finished and not state.abandoned:
            return self._current_snapshot
        if self._current_snapshot is None or self._current_snapshot.day != day:
            self._pending_snapshot = RoundSnapshot(day, 1)
            if state is not None and not state.advanced and not state.abandoned:
                # A daily reset retires the old day's unconsumed advancement.
                self._pending_round_id = self._active_round_id
            self._commit_pending()
        return self._current_snapshot

    def prepare_new_round(self, day: date) -> RoundSnapshot:
        """Return a persisted identity before an actual recording is started."""
        self.refresh(day)
        state = self._active_state()
        if state is not None and not state.advanced and not state.abandoned:
            if not state.recordings_finished:
                raise ValueError("Cannot prepare a new round while one is unfinished")
            self._advance(self._active_round_id)
        return self._current_snapshot

    def _active_state(self):
        return self._rounds.get(self._active_round_id)

    def open_round(
        self, round_id: Hashable, snapshot: RoundSnapshot, expected_keys
    ):
        """Freeze an accepted UUID's identity and applicable analysis pairs."""
        expected_keys = frozenset(expected_keys)
        existing = self._rounds.get(round_id)
        if existing is not None:
            if existing.snapshot != snapshot or existing.expected_keys != expected_keys:
                raise ValueError("Cannot change a round's frozen identity or tasks")
            return
        state = self._active_state()
        if state is not None and not state.advanced and not state.abandoned:
            raise ValueError("Cannot open a new round while one is unfinished")
        if not self.can_record or snapshot != self._current_snapshot:
            raise ValueError("Cannot open a round with an uncommitted identity")
        self._rounds[round_id] = _RoundState(snapshot, expected_keys)
        self._active_round_id = round_id

    def _advance(self, round_id):
        state = self._rounds[round_id]
        self._pending_snapshot = RoundSnapshot(state.snapshot.day, state.snapshot.number + 1)
        self._pending_round_id = round_id
        self._commit_pending()

    def _complete_if_ready(self, round_id, day):
        self.refresh(day)
        state = self._rounds[round_id]
        if (
            not state.advanced and not state.abandoned
            and state.recordings_finished
            and state.completed_keys == state.expected_keys
        ):
            self._advance(round_id)

    def recordings_completed(self, round_id: Hashable, day: date):
        """Notify only after every condition has saved and finished archiving."""
        state = self._rounds.get(round_id)
        if state is None or state.advanced or state.abandoned:
            return
        state.recordings_finished = True
        self._complete_if_ready(round_id, day)

    def analysis_completed(self, round_id: Hashable, key: tuple[str, str], day: date):
        """Accept an authenticated completion; unrelated and late pairs do nothing."""
        state = self._rounds.get(round_id)
        if state is None or state.advanced or state.abandoned or key not in state.expected_keys:
            return
        state.completed_keys.add(key)
        self._complete_if_ready(round_id, day)

    def abandon_round(self, round_id: Hashable, day: date):
        """Drop a partial round's association without consuming its number."""
        state = self._rounds.get(round_id)
        if state is not None and not state.recordings_finished:
            state.abandoned = True
        self.refresh(day)

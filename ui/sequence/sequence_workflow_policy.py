"""Workflow-owned automatic-analysis policy for admitted sequence sources."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from threading import Lock, local
from typing import Any, Protocol, runtime_checkable

from ui.sequence.sequence_messages import ConfigurationSnapshot


class AutomaticAnalysisSource(Enum):
    RECORDED = "recorded"
    IMPORTED = "imported"


@dataclass(frozen=True, slots=True)
class AutomaticAnalysisDecision:
    workflow_generation: int
    source: AutomaticAnalysisSource
    mode: str | None
    enabled: bool
    reason: str

    def __post_init__(self) -> None:
        if type(self.workflow_generation) is not int or self.workflow_generation < 0:
            raise ValueError(
                "workflow_generation must be a non-negative integer"
            )
        if type(self.source) is not AutomaticAnalysisSource:
            raise TypeError("source must be AutomaticAnalysisSource")
        if self.mode is not None and type(self.mode) is not str:
            raise TypeError("mode must be a plain string or None")
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        if type(self.reason) is not str or not self.reason:
            raise ValueError("reason must be a non-empty plain string")


@runtime_checkable
class AutomaticAnalysisPolicyPort(Protocol):
    def decide_recorded(
        self,
        *,
        workflow_generation: int,
        recording_snapshot: Any,
        configuration_snapshot: ConfigurationSnapshot,
    ) -> AutomaticAnalysisDecision: ...

    def decide_imported(
        self,
        *,
        workflow_generation: int,
        recording_snapshot: Any,
        configuration_snapshot: ConfigurationSnapshot,
    ) -> AutomaticAnalysisDecision: ...


class SequenceAutomaticAnalysisPolicyService:
    """Make one immutable automatic-analysis decision per workflow generation."""

    _RECORDED_MODES = frozenset(("RECORD_ONLY", "PLAY_AND_RECORD"))
    DECISION_HISTORY_LIMIT = 128

    def __init__(self) -> None:
        self._lock = Lock()
        self._decisions: OrderedDict[int, AutomaticAnalysisDecision] = OrderedDict()
        self._local = local()

    @staticmethod
    def _generation(value: Any) -> int:
        if type(value) is not int or value < 0:
            raise ValueError("workflow_generation must be a non-negative integer")
        return value

    def _once(
        self,
        generation: int,
        candidate: AutomaticAnalysisDecision,
    ) -> AutomaticAnalysisDecision:
        with self._lock:
            existing = self._decisions.get(generation)
            if existing is not None:
                self._decisions.move_to_end(generation)
                return existing
            self._decisions[generation] = candidate
            while len(self._decisions) > self.DECISION_HISTORY_LIMIT:
                self._decisions.popitem(last=False)
            return candidate

    def _begin_decision(self, generation: int) -> bool:
        active = getattr(self._local, "active", None)
        if active is None:
            active = set()
            self._local.active = active
        if generation in active:
            return False
        active.add(generation)
        return True

    def _end_decision(self, generation: int) -> None:
        active = getattr(self._local, "active", None)
        if active is not None:
            active.discard(generation)

    @staticmethod
    def _disabled(
        generation: int,
        source: AutomaticAnalysisSource,
        reason: str,
        *,
        mode: str | None = None,
    ) -> AutomaticAnalysisDecision:
        return AutomaticAnalysisDecision(
            workflow_generation=generation,
            source=source,
            mode=mode,
            enabled=False,
            reason=reason,
        )

    def decide_recorded(
        self,
        *,
        workflow_generation: int,
        recording_snapshot: Any,
        configuration_snapshot: ConfigurationSnapshot,
    ) -> AutomaticAnalysisDecision:
        generation = self._generation(workflow_generation)
        source = AutomaticAnalysisSource.RECORDED
        if not self._begin_decision(generation):
            return self._disabled(
                generation, source, "automatic analysis policy reentry was rejected"
            )
        mode = None
        try:
            try:
                if type(configuration_snapshot) is not ConfigurationSnapshot:
                    candidate = self._disabled(
                        generation, source, "configuration snapshot is unavailable"
                    )
                elif not isinstance(recording_snapshot, Mapping):
                    candidate = self._disabled(
                        generation, source, "recording snapshot is unavailable"
                    )
                else:
                    session = recording_snapshot.get("session")
                    if not isinstance(session, Mapping):
                        candidate = self._disabled(
                            generation, source, "recording session snapshot is unavailable"
                        )
                    else:
                        session_generation = session.get("workflow_generation")
                        mode_value = session.get("mode")
                        mode = mode_value if type(mode_value) is str else None
                        analysis_config = session.get("analysis_config")
                        if (
                            type(session_generation) is not int
                            or session_generation != generation
                        ):
                            candidate = self._disabled(
                                generation,
                                source,
                                "recording snapshot has a stale workflow generation",
                                mode=mode,
                            )
                        elif mode not in self._RECORDED_MODES:
                            candidate = self._disabled(
                                generation,
                                source,
                                "recording mode does not support automatic analysis",
                                mode=mode,
                            )
                        elif not isinstance(analysis_config, Mapping):
                            candidate = self._disabled(
                                generation,
                                source,
                                "recording analysis configuration is unavailable",
                                mode=mode,
                            )
                        elif analysis_config.get("auto_analysis") is not True:
                            candidate = self._disabled(
                                generation,
                                source,
                                "automatic analysis is disabled",
                                mode=mode,
                            )
                        else:
                            candidate = AutomaticAnalysisDecision(
                                workflow_generation=generation,
                                source=source,
                                mode=mode,
                                enabled=True,
                                reason="automatic analysis is enabled",
                            )
            except BaseException:
                # Immutable message snapshots are the external policy boundary. A
                # hostile Mapping must not unwind or partially advance Workflow.
                candidate = self._disabled(
                    generation, source, "recording snapshot could not be inspected"
                )
            return self._once(generation, candidate)
        finally:
            self._end_decision(generation)

    def decide_imported(
        self,
        *,
        workflow_generation: int,
        recording_snapshot: Any,
        configuration_snapshot: ConfigurationSnapshot,
    ) -> AutomaticAnalysisDecision:
        generation = self._generation(workflow_generation)
        source = AutomaticAnalysisSource.IMPORTED
        if not self._begin_decision(generation):
            return self._disabled(
                generation, source, "automatic analysis policy reentry was rejected"
            )
        try:
            if type(configuration_snapshot) is not ConfigurationSnapshot:
                candidate = self._disabled(
                    generation, source, "configuration snapshot is unavailable"
                )
            elif recording_snapshot is None:
                candidate = self._disabled(
                    generation, source, "imported recording snapshot is unavailable"
                )
            else:
                candidate = AutomaticAnalysisDecision(
                    workflow_generation=generation,
                    source=source,
                    mode=None,
                    enabled=True,
                    reason="imported audio is analyzed automatically",
                )
            return self._once(generation, candidate)
        finally:
            self._end_decision(generation)

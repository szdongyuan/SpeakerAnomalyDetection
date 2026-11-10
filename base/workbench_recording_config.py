"""Recording parameters, shared by queue and resolved through the context."""

from collections.abc import Mapping
from dataclasses import dataclass

from base.workbench_config_values import WorkbenchContext, _SnapshotValue, _freeze_config


@dataclass(frozen=True, eq=False)
class WorkbenchRecordingConfig(_SnapshotValue):
    context: WorkbenchContext
    parameters_by_queue: Mapping[str, Mapping]

    def __post_init__(self):
        object.__setattr__(self, "parameters_by_queue", _freeze_config(
            self.parameters_by_queue, "parameters_by_queue",
        ))

    def parameters_for(self, condition_id: str) -> Mapping:
        condition = self.context.get_condition(condition_id)
        if condition is None:
            raise ValueError(f"Unknown workbench condition {condition_id!r}")
        return self.parameters_by_queue[condition.test_queue]

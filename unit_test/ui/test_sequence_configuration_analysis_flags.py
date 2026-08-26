from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from ui.sequence.sequence_configuration_analysis_flags import (
    AnalysisFlagProjection,
    AnalysisFlagProjectionRollbackError,
    AnalysisFlagRuntimeCheckpoint,
    DataStructAnalysisFlagProjectionPort,
    SequenceAnalysisFlagProjectionService,
)
from ui.sequence.sequence_configuration_controller import (
    SequenceConfigurationController,
)
from ui.sequence.sequence_configuration_model import SequenceConfigurationModel
from ui.sequence.sequence_configuration_view import SequenceConfigurationView
from ui.sequence.sequence_messages import ConfigurationChanged, ConfigurationSnapshot


ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_WIDGET = ROOT / "ui" / "sequence" / "sequence_widget.py"


def _snapshot(analysis_config, *, mode="RECORD_ONLY"):
    return ConfigurationSnapshot(
        sequence_config=[
            {
                "seq1": {
                    "acq": {"mode": mode, "detail": {}},
                    "analysis_list": analysis_config,
                }
            }
        ],
        analysis_config=analysis_config,
        using_config_path="sequence.json",
    )


def test_projection_derives_exact_legacy_alias_counts_from_frozen_snapshot():
    source = {
        "display_sequence": [
            "fr-cn",
            "hd-cn",
            "fft-cn",
            "fr",
            "hd",
            "fft",
            "spec-cn",
            "spec",
            "ignored",
        ],
        "fr-cn": {"type": "频响 (FR) "},
        "hd-cn": {"type": "谐波失真 (HD) "},
        "fft-cn": {"type": "快速傅里叶变换 (FFT) "},
        "fr": {"type": "FR"},
        "hd": {"type": "HD"},
        "fft": {"type": "FFT"},
        "spec-cn": {"type": "频谱分析 (Spec) "},
        "spec": {"type": "Spec"},
        "ignored": {"type": "SPL"},
    }
    snapshot = _snapshot(source)
    source["fft"]["type"] = "Spec"
    source["display_sequence"].clear()

    service = SequenceAnalysisFlagProjectionService(
        DataStructAnalysisFlagProjectionPort(
            SimpleNamespace(fft_flag=99, stft_flag=88)
        )
    )

    assert service.derive(snapshot.analysis_config) == AnalysisFlagProjection(
        fft_flag=6,
        stft_flag=2,
    )


@pytest.mark.parametrize(
    "analysis_config",
    (
        {"display_sequence": "fft", "fft": {"type": "FFT"}},
        {"display_sequence": ["missing"]},
        {"display_sequence": ["fft"], "fft": []},
        {"display_sequence": ["fft"], "fft": {}},
        {"display_sequence": [object()]},
    ),
)
def test_malformed_projection_fails_closed_without_runtime_mutation(analysis_config):
    runtime = SimpleNamespace(fft_flag=7, stft_flag=8)
    service = SequenceAnalysisFlagProjectionService(
        DataStructAnalysisFlagProjectionPort(runtime)
    )

    with pytest.raises((TypeError, ValueError)):
        service.project(analysis_config)

    assert (runtime.fft_flag, runtime.stft_flag) == (7, 8)


class _FailOncePort:
    def __init__(self, failure):
        self.flags = [7, 8]
        self.failure = failure
        self.apply_calls = 0
        self.restore_calls = 0

    def capture(self):
        return AnalysisFlagRuntimeCheckpoint(
            True,
            self.flags[0],
            True,
            self.flags[1],
        )

    def apply(self, projection):
        self.apply_calls += 1
        self.flags[0] = projection.fft_flag
        if self.apply_calls == 1:
            raise self.failure
        self.flags[1] = projection.stft_flag

    def restore(self, checkpoint):
        self.restore_calls += 1
        self.flags[:] = [checkpoint.fft_value, checkpoint.stft_value]


@pytest.mark.parametrize(
    "failure",
    (
        RuntimeError("projection failed"),
        KeyboardInterrupt("projection interrupted"),
        SystemExit("projection exited"),
    ),
)
def test_partial_projection_restores_exactly_and_next_retry_succeeds(failure):
    port = _FailOncePort(failure)
    service = SequenceAnalysisFlagProjectionService(port)
    config = {
        "display_sequence": ["fft", "spec"],
        "fft": {"type": "FFT"},
        "spec": {"type": "Spec"},
    }

    with pytest.raises(type(failure)) as raised:
        service.project(config)

    assert raised.value is failure
    assert port.flags == [7, 8]
    assert port.restore_calls == 1
    assert service.project(config) == AnalysisFlagProjection(1, 1)
    assert port.flags == [1, 1]


@pytest.mark.parametrize("rejection", (False, 1, "accepted"))
def test_partial_apply_rejection_rolls_back_and_next_retry_succeeds(rejection):
    class RejectOncePort:
        def __init__(self):
            self.flags = [7, 8]
            self.apply_calls = 0

        def capture(self):
            return AnalysisFlagRuntimeCheckpoint(
                True,
                self.flags[0],
                True,
                self.flags[1],
            )

        def apply(self, projection):
            self.apply_calls += 1
            self.flags[0] = projection.fft_flag
            if self.apply_calls == 1:
                return rejection
            self.flags[1] = projection.stft_flag
            return True

        def restore(self, checkpoint):
            self.flags[:] = [checkpoint.fft_value, checkpoint.stft_value]
            return None

    port = RejectOncePort()
    service = SequenceAnalysisFlagProjectionService(port)
    config = {
        "display_sequence": ["fft", "spec"],
        "fft": {"type": "FFT"},
        "spec": {"type": "Spec"},
    }

    with pytest.raises(RuntimeError, match="apply.*rejected"):
        service.project(config)

    assert port.flags == [7, 8]
    assert service.project(config) == AnalysisFlagProjection(1, 1)
    assert port.flags == [1, 1]


@pytest.mark.parametrize("rejection", (False, 1, "restored"))
def test_restore_rejection_propagates_rollback_failure_and_allows_exact_retry(
    rejection,
):
    class RejectRestoreOncePort(_FailOncePort):
        def restore(self, checkpoint):
            self.restore_calls += 1
            if self.restore_calls == 1:
                return rejection
            self.flags[:] = [checkpoint.fft_value, checkpoint.stft_value]
            return True

    port = RejectRestoreOncePort(RuntimeError("apply failed"))
    service = SequenceAnalysisFlagProjectionService(port)
    config = {
        "display_sequence": ["fft", "spec"],
        "fft": {"type": "FFT"},
        "spec": {"type": "Spec"},
    }

    with pytest.raises(RuntimeError, match="rollback.*rejected"):
        service.project(config)

    assert port.flags == [1, 8]
    assert service.project(config) == AnalysisFlagProjection(1, 1)
    assert port.flags == [1, 1]


@pytest.mark.parametrize("primary_kind", ("ordinary", "keyboard", "system-exit"))
@pytest.mark.parametrize(
    "rollback_kind",
    ("success", "false", "ordinary", "keyboard"),
)
def test_projection_failure_priority_matrix_preserves_interruptions_and_retry(
    primary_kind,
    rollback_kind,
):
    primary = {
        "ordinary": RuntimeError("projection failed"),
        "keyboard": KeyboardInterrupt("projection interrupted"),
        "system-exit": SystemExit("projection exited"),
    }[primary_kind]
    rollback_interruption = KeyboardInterrupt("rollback interrupted")

    class MatrixPort:
        def __init__(self):
            self.flags = [7, 8]
            self.apply_calls = 0

        def capture(self):
            return AnalysisFlagRuntimeCheckpoint(True, 7, True, 8)

        def apply(self, projection):
            self.apply_calls += 1
            self.flags[0] = projection.fft_flag
            if self.apply_calls == 1:
                raise primary
            self.flags[1] = projection.stft_flag
            return True

        def restore(self, checkpoint):
            if rollback_kind == "false":
                return False
            if rollback_kind == "ordinary":
                raise RuntimeError("rollback failed")
            if rollback_kind == "keyboard":
                raise rollback_interruption
            self.flags[:] = [checkpoint.fft_value, checkpoint.stft_value]
            return True

    port = MatrixPort()
    service = SequenceAnalysisFlagProjectionService(port)
    config = {
        "display_sequence": ["fft", "spec"],
        "fft": {"type": "FFT"},
        "spec": {"type": "Spec"},
    }

    if primary_kind == "ordinary" and rollback_kind in {"false", "ordinary"}:
        expected_type = AnalysisFlagProjectionRollbackError
        expected_error = None
    elif primary_kind == "ordinary" and rollback_kind == "keyboard":
        expected_type = KeyboardInterrupt
        expected_error = rollback_interruption
    else:
        expected_type = type(primary)
        expected_error = primary

    with pytest.raises(expected_type) as raised:
        service.project(config)

    if expected_error is not None:
        assert raised.value is expected_error
    if primary_kind != "ordinary" and rollback_kind != "success":
        assert any(
            "rollback" in note
            for note in getattr(primary, "__notes__", ())
        )

    assert service.project(config) == AnalysisFlagProjection(1, 1)
    assert port.flags == [1, 1]


def test_projection_rejects_reentry_without_corrupting_outer_result():
    runtime = SimpleNamespace(fft_flag=0, stft_flag=0)
    holder = {}

    class ReentrantPort(DataStructAnalysisFlagProjectionPort):
        def apply(self, projection):
            with pytest.raises(RuntimeError, match="already active"):
                holder["service"].project({"display_sequence": []})
            super().apply(projection)

    service = SequenceAnalysisFlagProjectionService(ReentrantPort(runtime))
    holder["service"] = service

    assert service.project(
        {"display_sequence": ["fft"], "fft": {"type": "FFT"}}
    ) == AnalysisFlagProjection(1, 0)
    assert (runtime.fft_flag, runtime.stft_flag) == (1, 0)


def test_concurrent_projection_calls_are_serialized_and_never_leave_mixed_flags():
    class ConcurrentProbePort:
        def __init__(self):
            self.flags = [0, 0]
            self.active = 0
            self.maximum_active = 0
            self.guard = threading.Lock()

        def capture(self):
            return AnalysisFlagRuntimeCheckpoint(
                True,
                self.flags[0],
                True,
                self.flags[1],
            )

        def apply(self, projection):
            with self.guard:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            self.flags[0] = projection.fft_flag
            time.sleep(0.01)
            self.flags[1] = projection.stft_flag
            with self.guard:
                self.active -= 1

        def restore(self, checkpoint):
            self.flags[:] = [checkpoint.fft_value, checkpoint.stft_value]

    port = ConcurrentProbePort()
    service = SequenceAnalysisFlagProjectionService(port)
    configs = (
        {"display_sequence": ["fft"], "fft": {"type": "FFT"}},
        {"display_sequence": ["spec"], "spec": {"type": "Spec"}},
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(service.project, configs))

    assert set(results) == {
        AnalysisFlagProjection(1, 0),
        AnalysisFlagProjection(0, 1),
    }
    assert port.maximum_active == 1
    assert tuple(port.flags) in {(1, 0), (0, 1)}


def test_configuration_changed_projects_once_and_rejects_duplicate_generation():
    runtime = SimpleNamespace(fft_flag=0, stft_flag=0)
    model = SequenceConfigurationModel(data_struct=runtime)
    controller = SequenceConfigurationController(model, SequenceConfigurationView())
    event = ConfigurationChanged(
        1,
        _snapshot({"display_sequence": ["fft"], "fft": {"type": "FFT"}}),
    )

    assert controller.handle_configuration_changed(event) is True
    assert (runtime.fft_flag, runtime.stft_flag) == (1, 0)
    runtime.fft_flag = 12
    assert controller.handle_configuration_changed(event) is False
    assert runtime.fft_flag == 12


def test_configuration_changed_rolls_back_model_and_flags_when_projection_fails():
    runtime = SimpleNamespace(fft_flag=3, stft_flag=4)
    model = SequenceConfigurationModel(data_struct=runtime)
    old = _snapshot({"display_sequence": []})
    assert model.apply_configuration(old, generation=2) is True
    failure_port = _FailOncePort(RuntimeError("apply failed"))
    failure_port.flags = [runtime.fft_flag, runtime.stft_flag]
    controller = SequenceConfigurationController(
        model,
        SequenceConfigurationView(),
        analysis_flag_projection_service=SequenceAnalysisFlagProjectionService(
            failure_port
        ),
    )
    event = ConfigurationChanged(
        3,
        _snapshot({"display_sequence": ["fft"], "fft": {"type": "FFT"}}),
    )

    assert controller.handle_configuration_changed(event) is False
    assert model.configuration_generation == 2
    assert model.analysis_config == {"display_sequence": []}
    assert failure_port.flags == [3, 4]


def test_configuration_changed_restore_interruption_recovers_then_allows_retry():
    runtime = SimpleNamespace(fft_flag=3, stft_flag=4)
    model = SequenceConfigurationModel(data_struct=runtime)
    old = _snapshot({"display_sequence": []})
    assert model.apply_configuration(old, generation=2) is True
    interruption = KeyboardInterrupt("restore interrupted")

    class RestoreInterruptPort(_FailOncePort):
        def restore(self, checkpoint):
            self.restore_calls += 1
            if self.restore_calls == 1:
                raise interruption
            self.flags[:] = [checkpoint.fft_value, checkpoint.stft_value]

    port = RestoreInterruptPort(RuntimeError("apply failed"))
    port.flags = [3, 4]
    controller = SequenceConfigurationController(
        model,
        SequenceConfigurationView(),
        analysis_flag_projection_service=SequenceAnalysisFlagProjectionService(
            port
        ),
    )
    event = ConfigurationChanged(
        3,
        _snapshot(
            {
                "display_sequence": ["fft"],
                "fft": {"type": "FFT"},
            }
        ),
    )

    with pytest.raises(KeyboardInterrupt) as raised:
        controller.handle_configuration_changed(event)

    assert raised.value is interruption
    assert model.configuration_generation == 2
    assert port.flags == [3, 4]
    assert controller.handle_configuration_changed(event) is True
    assert model.configuration_generation == 3
    assert port.flags == [1, 0]


@pytest.mark.parametrize("primary_kind", ("ordinary", "keyboard", "system-exit"))
@pytest.mark.parametrize(
    "rollback_kind",
    ("success", "false", "ordinary", "keyboard"),
)
def test_configuration_changed_priority_matrix_retries_restore_without_swallowing(
    primary_kind,
    rollback_kind,
):
    primary = {
        "ordinary": RuntimeError("projection failed"),
        "keyboard": KeyboardInterrupt("projection interrupted"),
        "system-exit": SystemExit("projection exited"),
    }[primary_kind]
    rollback_interruption = KeyboardInterrupt("rollback interrupted")

    class MatrixPort:
        def __init__(self):
            self.flags = [3, 4]
            self.apply_calls = 0
            self.restore_calls = 0

        def capture(self):
            return AnalysisFlagRuntimeCheckpoint(
                True,
                self.flags[0],
                True,
                self.flags[1],
            )

        def apply(self, projection):
            self.apply_calls += 1
            self.flags[0] = projection.fft_flag
            if self.apply_calls == 1:
                raise primary
            self.flags[1] = projection.stft_flag
            return True

        def restore(self, checkpoint):
            self.restore_calls += 1
            if self.restore_calls == 1:
                if rollback_kind == "false":
                    return False
                if rollback_kind == "ordinary":
                    raise RuntimeError("rollback failed")
                if rollback_kind == "keyboard":
                    raise rollback_interruption
            self.flags[:] = [checkpoint.fft_value, checkpoint.stft_value]
            return True

    port = MatrixPort()
    model = SequenceConfigurationModel(
        data_struct=SimpleNamespace(fft_flag=3, stft_flag=4)
    )
    assert model.apply_configuration(
        _snapshot({"display_sequence": []}),
        generation=2,
    ) is True
    controller = SequenceConfigurationController(
        model,
        SequenceConfigurationView(),
        analysis_flag_projection_service=SequenceAnalysisFlagProjectionService(
            port
        ),
    )
    event = ConfigurationChanged(
        3,
        _snapshot(
            {
                "display_sequence": ["fft"],
                "fft": {"type": "FFT"},
            }
        ),
    )

    expected_interruption = None
    if primary_kind != "ordinary":
        expected_interruption = primary
    elif rollback_kind == "keyboard":
        expected_interruption = rollback_interruption

    if expected_interruption is None:
        assert controller.handle_configuration_changed(event) is False
    else:
        with pytest.raises(type(expected_interruption)) as raised:
            controller.handle_configuration_changed(event)
        assert raised.value is expected_interruption

    assert model.configuration_generation == 2
    assert port.flags == [3, 4]
    assert controller.projection_consistent is True
    assert controller.handle_configuration_changed(event) is True
    assert model.configuration_generation == 3
    assert port.flags == [1, 0]


def test_configuration_changed_restore_rejection_marks_inconsistent_and_disables():
    class Button:
        def __init__(self):
            self.enabled = True

        def setDisabled(self, disabled):
            self.enabled = not bool(disabled)

        def setEnabled(self, enabled):
            self.enabled = bool(enabled)

        def isEnabled(self):
            return self.enabled

    class RejectingPort:
        def __init__(self):
            self.flags = [3, 4]

        def capture(self):
            return AnalysisFlagRuntimeCheckpoint(True, 3, True, 4)

        def apply(self, projection):
            self.flags[0] = projection.fft_flag
            return False

        def restore(self, _checkpoint):
            return False

    buttons = (Button(), Button(), Button())
    view = SequenceConfigurationView(
        player_button=buttons[0],
        replay_button=buttons[1],
        data_button=buttons[2],
    )
    model = SequenceConfigurationModel(
        data_struct=SimpleNamespace(fft_flag=3, stft_flag=4)
    )
    assert model.apply_configuration(
        _snapshot({"display_sequence": []}),
        generation=2,
    ) is True
    port = RejectingPort()
    controller = SequenceConfigurationController(
        model,
        view,
        analysis_flag_projection_service=SequenceAnalysisFlagProjectionService(
            port
        ),
    )
    event = ConfigurationChanged(
        3,
        _snapshot(
            {
                "display_sequence": ["fft"],
                "fft": {"type": "FFT"},
            }
        ),
    )

    assert controller.handle_configuration_changed(event) is False
    assert model.configuration_generation == 2
    assert controller.projection_consistent is False
    assert "rollback was incomplete" in controller.projection_failure
    assert tuple(button.enabled for button in buttons) == (False, False, False)


def test_configuration_changed_incomplete_restore_disables_but_preserves_interrupt():
    class Button:
        def __init__(self):
            self.enabled = True

        def setDisabled(self, disabled):
            self.enabled = not bool(disabled)

        def setEnabled(self, enabled):
            self.enabled = bool(enabled)

        def isEnabled(self):
            return self.enabled

    interruption = KeyboardInterrupt("projection interrupted")

    class RejectingPort:
        def __init__(self):
            self.flags = [3, 4]
            self.apply_calls = 0

        def capture(self):
            return AnalysisFlagRuntimeCheckpoint(True, 3, True, 4)

        def apply(self, projection):
            self.apply_calls += 1
            self.flags[0] = projection.fft_flag
            if self.apply_calls == 1:
                raise interruption
            self.flags[1] = projection.stft_flag
            return True

        def restore(self, _checkpoint):
            return False

    buttons = (Button(), Button(), Button())
    model = SequenceConfigurationModel(
        data_struct=SimpleNamespace(fft_flag=3, stft_flag=4)
    )
    assert model.apply_configuration(
        _snapshot({"display_sequence": []}),
        generation=2,
    ) is True
    port = RejectingPort()
    controller = SequenceConfigurationController(
        model,
        SequenceConfigurationView(
            player_button=buttons[0],
            replay_button=buttons[1],
            data_button=buttons[2],
        ),
        analysis_flag_projection_service=SequenceAnalysisFlagProjectionService(
            port
        ),
    )
    event = ConfigurationChanged(
        3,
        _snapshot(
            {
                "display_sequence": ["fft"],
                "fft": {"type": "FFT"},
            }
        ),
    )

    with pytest.raises(KeyboardInterrupt) as raised:
        controller.handle_configuration_changed(event)

    assert raised.value is interruption
    assert model.configuration_generation == 2
    assert controller.projection_consistent is False
    assert tuple(button.enabled for button in buttons) == (False, False, False)
    assert controller.handle_configuration_changed(event) is True
    assert model.configuration_generation == 3
    assert port.flags == [1, 0]
    assert controller.projection_consistent is False


def test_sequence_window_only_composes_analysis_flag_owner():
    source = SEQUENCE_WIDGET.read_text(encoding="utf-8")
    tree = ast.parse(source)
    window = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    methods = {
        node.name
        for node in window.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "init_fft_and_stft_flag" not in methods
    assert "init_analysis_flags=" not in source
    assert "analysis_flag_projection_service=" in source

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import json
import os
import sys
import threading
import types
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from base.load_config import LoadUiConfig, PathTransactionCoordinator
from ui.sequence.sequence_configuration_controller import (
    PersistenceAdapter,
    SequenceConfigurationController,
    _ConfigurationTransaction,
    _canonical_json_semantic_token,
    _durable_ownership_token,
)
from ui.sequence.sequence_event_bus import SequenceEventBus
from ui.sequence.sequence_configuration_model import SequenceConfigurationModel
from ui.sequence.sequence_recording_model import (
    RecordingModel,
    RecordingStreamingStimulusPort,
)
from ui.sequence.sequence_configuration_view import SequenceConfigurationView
from ui.sequence.sequence_messages import ConfigurationChanged, ConfigurationSnapshot

try:
    import base.analysis_warning_preferences  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "base.analysis_warning_preferences":
        raise
    warning_preferences = types.ModuleType("base.analysis_warning_preferences")
    warning_preferences.is_uncalibrated_microphone_warning_suppressed = (
        lambda **_kwargs: False
    )
    warning_preferences.save_uncalibrated_microphone_warning_suppressed = (
        lambda **_kwargs: True
    )
    sys.modules["base.analysis_warning_preferences"] = warning_preferences

from ui.sequence.sequence_widget import SequenceWindow
from ui.sequence.sequence_workflow_model import SequenceWorkflowModel


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _Button:
    def __init__(self, enabled=True):
        self.enabled = enabled

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)

    def setDisabled(self, disabled):
        self.enabled = not bool(disabled)

    def isEnabled(self):
        return self.enabled


class _ComboBox:
    def __init__(self):
        self.items = []
        self.index = -1
        self.blocked = False
        self.focus_cleared = False
        self.focused = True

    def blockSignals(self, blocked):
        previous = self.blocked
        self.blocked = bool(blocked)
        return previous

    def clear(self):
        self.items.clear()
        self.index = -1

    def addItem(self, text, data):
        self.items.append((text, data))
        if self.index < 0:
            self.index = 0

    def findText(self, text):
        return next((i for i, item in enumerate(self.items) if item[0] == text), -1)

    def findData(self, data):
        return next((i for i, item in enumerate(self.items) if item[1] == data), -1)

    def setCurrentIndex(self, index):
        self.index = index

    def currentData(self):
        if self.index < 0:
            return None
        return self.items[self.index][1]

    def clearFocus(self):
        self.focus_cleared = True
        self.focused = False

    def hasFocus(self):
        return self.focused

    def setFocus(self):
        self.focused = True


def _sequence(mode="RECORD_ONLY", detail=None, analysis=None):
    return [
        {
            "seq1": {
                "acq": {"mode": mode, "detail": detail or {}},
                "analysis_list": analysis or {},
            }
        }
    ]


def _snapshot(mode="RECORD_ONLY", analysis=None):
    sequence = _sequence(mode=mode, analysis=analysis)
    return ConfigurationSnapshot(
        sequence_config=sequence,
        analysis_config=analysis or {},
        mic={"name": "input"},
        speaker={"name": "output"},
        mic_channels=(0, 2),
        using_config_path="sequence.json",
    )


def _controller(*, model=None, view=None, **kwargs):
    workflow = SequenceWorkflowModel()
    model = model or SequenceConfigurationModel(
        data_struct=SimpleNamespace(), workflow_model=workflow
    )
    return SequenceConfigurationController(
        model,
        view or SequenceConfigurationView(),
        **kwargs,
    )


def test_model_copies_device_channels_and_keeps_stable_data_struct():
    data_struct = SimpleNamespace(sample_rate=None)
    model = SequenceConfigurationModel(data_struct=data_struct)
    mic = {"name": "input", "rates": [44100]}
    speaker = {"name": "output"}
    channels = [0, 2]

    model.mic = mic
    model.speaker = speaker
    model.mic_channels = channels
    mic["rates"].append(48000)
    speaker["name"] = "changed"
    channels.append(3)

    assert model.mic == {"name": "input", "rates": [44100]}
    assert model.speaker == {"name": "output"}
    assert model.mic_channels == [0, 2]
    assert model.data_struct is data_struct


def test_canonical_json_semantic_token_preserves_exact_json_types_and_float_policy():
    assert _canonical_json_semantic_token({"a": 1, "b": [True, None]}) == (
        _canonical_json_semantic_token({"b": [True, None], "a": 1})
    )
    assert _canonical_json_semantic_token(True) != _canonical_json_semantic_token(1)
    assert _canonical_json_semantic_token(1) != _canonical_json_semantic_token(1.0)
    assert _canonical_json_semantic_token(float("nan")) == (
        _canonical_json_semantic_token(float("nan"))
    )
    assert _canonical_json_semantic_token(float("inf")) == (
        _canonical_json_semantic_token(float("inf"))
    )
    assert _canonical_json_semantic_token(float("inf")) != (
        _canonical_json_semantic_token(float("-inf"))
    )
    assert _canonical_json_semantic_token(-0.0) != (
        _canonical_json_semantic_token(0.0)
    )


def test_canonical_json_semantic_token_rejects_unsupported_values_without_equality():
    class EqualityPoison:
        def __init__(self):
            self.calls = []

        def __eq__(self, _other):
            self.calls.append("eq")
            raise AssertionError("live equality must not run")

    poison = EqualityPoison()

    with pytest.raises(TypeError):
        _canonical_json_semantic_token({"poison": poison})

    assert poison.calls == []


def test_configuration_generation_is_shared_with_workflow_and_rejects_stale_event():
    workflow = SequenceWorkflowModel()
    model = SequenceConfigurationModel(
        data_struct=SimpleNamespace(), workflow_model=workflow
    )
    controller = _controller(model=model)
    new_snapshot = _snapshot("PLAY_AND_RECORD", {"display_sequence": []})
    old_snapshot = _snapshot("RECORD_ONLY", {"auto_analysis": False})

    assert model.apply_configuration(new_snapshot, generation=2) is True
    assert controller.handle_configuration_changed(
        ConfigurationChanged(1, old_snapshot)
    ) is False
    assert controller.handle_configuration_changed(
        ConfigurationChanged(2, old_snapshot)
    ) is False

    assert model.configuration_generation == 2
    assert workflow.configuration_generation == 2
    assert workflow.configuration_snapshot is new_snapshot
    assert model.sequence_config == _sequence(
        mode="PLAY_AND_RECORD", analysis={"display_sequence": []}
    )

    model.mic = {"name": "replacement-input"}
    assert workflow.configuration_generation == 2
    assert workflow.configuration_snapshot.mic == {"name": "replacement-input"}


def test_successful_load_increments_generation_and_publishes_supported_snapshot(tmp_path):
    config_path = tmp_path / "sequence.json"
    config_path.write_text("{}", encoding="utf-8")
    workflow = SequenceWorkflowModel()
    model = SequenceConfigurationModel(
        data_struct=SimpleNamespace(), workflow_model=workflow
    )
    events = []
    controller = _controller(
        model=model,
        config_loader=lambda _path: (0, _sequence("RECORD_ONLY")),
        ok_code=0,
        configuration_publisher=events.append,
    )
    model.using_config_path = str(config_path)

    controller.get_sequence_config_from_json()
    controller.get_sequence_config_from_json()

    assert model.configuration_generation == 2
    assert workflow.configuration_generation == 2
    assert [event.configuration_generation for event in events] == [1, 2]
    assert all(
        type(event.configuration_snapshot) is ConfigurationSnapshot for event in events
    )


def test_initial_load_commits_and_publishes_only_final_regenerated_configuration(qapp):
    workflow = SequenceWorkflowModel()
    data_struct = SimpleNamespace(
        stimulus_data=None,
        stimulus_info=None,
        sample_rate=None,
        store_wave_data=None,
        store_wave_data_multi=None,
    )
    model = SequenceConfigurationModel(
        data_struct=data_struct,
        workflow_model=workflow,
    )
    model.using_config_path = "sequence.json"
    saved = []
    bus = SequenceEventBus()
    observed = []
    bus.events.configuration_changed.connect(observed.append, Qt.QueuedConnection)

    def prepare(data, detail, **_kwargs):
        data.stimulus_data = np.ones(8)
        data.stimulus_info = {"sample_rate": 48_000}
        if detail["stimulus_path"] == "original.wav":
            detail["stimulus_path"] = "generated.wav"
            return True
        return False

    def restore_saved(_path, checkpoint):
        saved[:] = deepcopy(checkpoint)
        return True

    controller = _controller(
        model=model,
        config_loader=lambda _path: (
            0,
            _sequence("PLAY_AND_RECORD", {"stimulus_path": "original.wav"}),
        ),
        ok_code=0,
        duplex_sample_rate_resolver=lambda _mic, _speaker: SimpleNamespace(
            ok=True,
            sample_rate=48_000,
            message="",
        ),
        stimulus_setter=prepare,
        config_saver=lambda config, path: saved.append([deepcopy(config), path]) or True,
        config_persistence_snapshotter=lambda _path: deepcopy(saved),
        config_persistence_restorer=restore_saved,
        configuration_publisher=bus.events.configuration_changed.emit,
    )

    assert controller.get_sequence_config_from_json() is None
    assert controller.init_data_struct_stimulus_config() is None
    qapp.processEvents()

    expected_detail = {"stimulus_path": "generated.wav"}
    assert model.sequence_config[0]["seq1"]["acq"]["detail"] == expected_detail
    assert workflow.configuration_snapshot.sequence_config[0]["seq1"]["acq"][
        "detail"
    ] == expected_detail
    assert len(observed) == 1
    assert observed[0].configuration_snapshot.sequence_config[0]["seq1"]["acq"][
        "detail"
    ] == expected_detail
    assert saved == [[model.sequence_config, "sequence.json"]]
    assert model.stimulus_reference_ready is True
    assert np.array_equal(data_struct.stimulus_data, np.ones(8))


def test_apply_and_publish_keeps_commit_when_final_notification_raises():
    workflow = SequenceWorkflowModel()
    model = SequenceConfigurationModel(
        data_struct=SimpleNamespace(), workflow_model=workflow
    )
    old_snapshot = ConfigurationSnapshot(
        sequence_config=_sequence("RECORD_ONLY"),
        analysis_config={"display_sequence": []},
        using_config_path="old.json",
    )
    model.apply_configuration(old_snapshot, generation=3)
    warnings = []
    controller = _controller(
        model=model,
        configuration_publisher=lambda _event: (_ for _ in ()).throw(
            RuntimeError("publish failed")
        ),
        logger=SimpleNamespace(warning=warnings.append),
    )

    result = controller._apply_and_publish(_sequence("IMPORT_AUDIO"), {})

    assert result is True
    assert model.sequence_config == _sequence("IMPORT_AUDIO")
    assert model.analysis_config == {}
    assert model.using_config_path == "old.json"
    assert model.acquisition_mode == "IMPORT_AUDIO"
    assert model.configuration_generation == 4
    assert workflow.configuration_generation == 4
    assert workflow.configuration_snapshot.sequence_config == ConfigurationSnapshot(
        _sequence("IMPORT_AUDIO"), {}
    ).sequence_config
    assert warnings == ["Failed to publish sequence configuration: publish failed"]


def test_apply_and_publish_keeps_commit_when_final_notification_returns_false():
    workflow = SequenceWorkflowModel()
    model = SequenceConfigurationModel(
        data_struct=SimpleNamespace(), workflow_model=workflow
    )
    old_snapshot = ConfigurationSnapshot(
        sequence_config=_sequence("RECORD_ONLY"),
        analysis_config={},
        using_config_path="old.json",
    )
    model.apply_configuration(old_snapshot, generation=2)
    controller = _controller(
        model=model,
        configuration_publisher=lambda _event: False,
    )

    assert controller._apply_and_publish(_sequence("IMPORT_AUDIO"), {}) is True
    assert model.acquisition_mode == "IMPORT_AUDIO"
    assert model.using_config_path == "old.json"
    assert model.configuration_generation == 3
    assert workflow.configuration_generation == 3


def test_registry_fallback_and_combobox_preserve_default_then_sorted_order(tmp_path):
    default = tmp_path / "default.json"
    alpha = tmp_path / "alpha.json"
    zulu = tmp_path / "zulu.json"
    for path in (default, alpha, zulu):
        path.write_text("{}", encoding="utf-8")
    registry = {
        "zulu": str(zulu),
        "默认配置": str(default),
        "alpha": str(alpha),
        "using_config_path": str(tmp_path / "missing.json"),
        "broken": str(tmp_path / "broken.json"),
    }
    persisted = []

    def persist_using_path(path):
        persisted.append(path)
        registry["using_config_path"] = path
        return True

    combo = _ComboBox()
    view = SequenceConfigurationView(combobox=combo)
    controller = _controller(
        view=view,
        registry_loader=lambda: deepcopy(registry),
        using_path_updater=persist_using_path,
    )

    using_path, loaded = controller.get_sequence_config_from_registry()
    controller.add_file_to_using_file_combobox()

    assert using_path == str(alpha)
    assert persisted == [str(alpha)]
    assert loaded == registry
    assert controller.model.registry_entries == (
        ("默认配置", str(default)),
        ("alpha", str(alpha)),
        ("zulu", str(zulu)),
    )
    assert combo.items == list(controller.model.registry_entries)
    assert combo.currentData() == str(alpha)


def test_registry_fallback_persistence_false_keeps_prior_model_and_selection(tmp_path):
    previous = tmp_path / "previous.json"
    fallback = tmp_path / "fallback.json"
    previous.write_text("{}", encoding="utf-8")
    fallback.write_text("{}", encoding="utf-8")
    combo = _ComboBox()
    view = SequenceConfigurationView(combobox=combo)
    model = SequenceConfigurationModel(data_struct=SimpleNamespace())
    model.sequence_config = _sequence("RECORD_ONLY")
    model.analysis_config = {"display_sequence": []}
    model.replace_registry(
        {"previous": str(previous), "using_config_path": str(previous)},
        using_config_path=str(previous),
        entries=(("previous", str(previous)),),
    )
    view.populate_configuration_entries(
        model.registry_entries,
        using_config_path=model.using_config_path,
        clear_first=True,
    )
    warnings = []
    controller = _controller(
        model=model,
        view=view,
        registry_loader=lambda: {
            "fallback": str(fallback),
            "using_config_path": str(tmp_path / "missing.json"),
        },
        using_path_updater=lambda _path: False,
        logger=SimpleNamespace(warning=warnings.append),
    )
    previous_config = deepcopy(model.sequence_config)
    previous_generation = model.configuration_generation

    using_path, _loaded = controller.get_sequence_config_from_registry()

    assert using_path == str(previous)
    assert model.using_config_path == str(previous)
    assert model.sequence_config == previous_config
    assert model.configuration_generation == previous_generation
    assert combo.currentData() == str(previous)
    assert warnings == [
        f"Failed to persist active sequence config path: {fallback}"
    ]


def _registry_fallback_cas_controller(
    *, before_lock=None, writer=None, restorer=None, truth_reader=None
):
    durable = {
        "registry": {
            "fallback": "fallback.json",
            "using_config_path": "missing.json",
        }
    }
    calls = []
    coordinator = PathTransactionCoordinator()
    transaction_key = "startup-fallback-registry-cas"

    @contextmanager
    def persistence_transaction(_target):
        if before_lock is not None:
            before_lock(durable["registry"])
            calls.append("concurrent-write")
        with coordinator.transaction(transaction_key):
            calls.append("lock-enter")
            try:
                yield
            finally:
                calls.append("lock-exit")

    def read_semantic_current(_target):
        normalized = coordinator.normalize_path(transaction_key)
        assert getattr(coordinator._lock_state, "active", {}).get(normalized) == 1
        calls.append("semantic-read")
        return deepcopy(durable["registry"])

    def default_writer(path, _target):
        calls.append("controller-write")
        durable["registry"]["using_config_path"] = path
        return True

    def default_restorer(_target, checkpoint, owned_token):
        calls.append("controller-restore")
        if _durable_ownership_token(durable["registry"]) != owned_token:
            return False
        durable["registry"] = deepcopy(checkpoint)
        return True

    def read_durable_truth(_target):
        if truth_reader is not None:
            return truth_reader(
                durable["registry"], coordinator, transaction_key, calls
            )
        return deepcopy(durable["registry"])

    adapter = SimpleNamespace(
        coordinator=coordinator,
        transaction=persistence_transaction,
        capture=lambda _target: deepcopy(durable["registry"]),
        write=writer or default_writer,
        conditional_restore=restorer or default_restorer,
        ownership_token=_durable_ownership_token,
        read_durable_truth=read_durable_truth,
        read_semantic_current=read_semantic_current,
    )
    combo = _ComboBox()
    view = SequenceConfigurationView(combobox=combo)
    model = SequenceConfigurationModel(data_struct=SimpleNamespace())
    model.replace_registry(
        {
            "previous": "previous.json",
            "using_config_path": "previous.json",
        },
        using_config_path="previous.json",
        entries=(("previous", "previous.json"),),
    )
    view.populate_configuration_entries(
        model.registry_entries,
        using_config_path=model.using_config_path,
        clear_first=True,
    )
    published = []
    controller = _controller(
        model=model,
        view=view,
        registry_loader=lambda: deepcopy(durable["registry"]),
        using_path_updater=(
            lambda _path: calls.append("uncoordinated-direct-write") or True
        ),
        using_path_persistence_adapter=adapter,
        configuration_publisher=published.append,
        path_exists=lambda path: path
        in {
            "previous.json",
            "fallback.json",
            "replacement.json",
            "unrelated.json",
        },
    )
    return controller, model, view, combo, durable, calls, published


@pytest.mark.parametrize(
    ("mutation", "expected_entries"),
    (
        (
            lambda registry: registry.__setitem__(
                "fallback", "replacement.json"
            ),
            (("fallback", "replacement.json"),),
        ),
        (
            lambda registry: registry.pop("fallback"),
            (),
        ),
        (
            lambda registry: registry.__setitem__("fallback", 1),
            (),
        ),
    ),
    ids=("path-rebound", "binding-deleted", "binding-type-changed"),
)
def test_registry_fallback_semantic_cas_rejects_changed_selected_binding_and_reconciles(
    mutation, expected_entries
):
    controller, model, _view, combo, durable, calls, published = (
        _registry_fallback_cas_controller(before_lock=mutation)
    )

    using_path, loaded = controller.get_sequence_config_from_registry()

    assert calls == [
        "concurrent-write",
        "lock-enter",
        "semantic-read",
        "lock-exit",
    ]
    assert "uncoordinated-direct-write" not in calls
    assert "controller-write" not in calls
    assert using_path == "missing.json"
    assert loaded == durable["registry"]
    assert model.registry == durable["registry"]
    assert model.using_config_path == "missing.json"
    assert model.registry_entries == expected_entries
    assert combo.items == (
        list(expected_entries) if expected_entries else [("无配置", None)]
    )
    assert published == []


def test_registry_fallback_semantic_cas_preserves_unrelated_addition_and_field_change():
    def add_unrelated(registry):
        registry["unrelated"] = "unrelated.json"
        registry["metadata"]["revision"] = 2

    controller, model, _view, _combo, durable, calls, published = (
        _registry_fallback_cas_controller(before_lock=add_unrelated)
    )
    durable["registry"]["metadata"] = {"revision": 1}

    using_path, loaded = controller.get_sequence_config_from_registry()

    assert calls == [
        "concurrent-write",
        "lock-enter",
        "semantic-read",
        "controller-write",
        "lock-exit",
    ]
    assert using_path == "fallback.json"
    assert loaded == durable["registry"]
    assert durable["registry"] == {
        "fallback": "fallback.json",
        "metadata": {"revision": 2},
        "unrelated": "unrelated.json",
        "using_config_path": "fallback.json",
    }
    assert model.registry == durable["registry"]
    assert model.using_config_path == "fallback.json"
    assert model.registry_entries == (
        ("fallback", "fallback.json"),
        ("unrelated", "unrelated.json"),
    )
    assert published == []


@pytest.mark.parametrize(
    "post_write_truth",
    ("active-path-mismatch", "selected-binding-mismatch", "read-failure"),
)
def test_registry_fallback_post_write_truth_failure_is_detected_inside_lock_and_compensated(
    post_write_truth,
):
    def read_truth(registry, coordinator, transaction_key, calls):
        normalized = coordinator.normalize_path(transaction_key)
        assert getattr(coordinator._lock_state, "active", {}).get(normalized) == 1
        calls.append("post-write-read")
        if post_write_truth == "read-failure":
            raise OSError("durable registry read failed")
        observed = deepcopy(registry)
        if post_write_truth == "active-path-mismatch":
            observed["using_config_path"] = "replacement.json"
        else:
            observed["fallback"] = "replacement.json"
        return observed

    controller, model, _view, combo, durable, calls, published = (
        _registry_fallback_cas_controller(truth_reader=read_truth)
    )
    previous_registry = model.registry

    using_path, _loaded = controller.get_sequence_config_from_registry()

    assert calls[:5] == [
        "lock-enter",
        "semantic-read",
        "controller-write",
        "post-write-read",
        "lock-exit",
    ]
    assert using_path == "previous.json"
    assert durable["registry"] == {
        "fallback": "fallback.json",
        "using_config_path": "missing.json",
    }
    assert model.registry == previous_registry
    assert model.using_config_path == "previous.json"
    assert combo.currentData() == "previous.json"
    assert controller.persistence_consistent is True
    assert published == []


def test_registry_fallback_captures_successful_post_write_truth_before_lock_exit():
    def read_truth(registry, coordinator, transaction_key, calls):
        normalized = coordinator.normalize_path(transaction_key)
        assert getattr(coordinator._lock_state, "active", {}).get(normalized) == 1
        calls.append("post-write-read")
        registry["unrelated"] = "unrelated.json"
        return deepcopy(registry)

    controller, model, _view, _combo, durable, calls, published = (
        _registry_fallback_cas_controller(truth_reader=read_truth)
    )

    using_path, loaded = controller.get_sequence_config_from_registry()

    assert calls == [
        "lock-enter",
        "semantic-read",
        "controller-write",
        "post-write-read",
        "lock-exit",
    ]
    assert using_path == "fallback.json"
    assert loaded == durable["registry"]
    assert model.registry == durable["registry"]
    assert model.registry_entries == (
        ("fallback", "fallback.json"),
        ("unrelated", "unrelated.json"),
    )
    assert published == []


def test_registry_fallback_projects_locked_truth_before_cooperating_writer_runs():
    start_competing = threading.Event()
    competing_acquired = threading.Event()
    durable_holder = {}
    captured_holder = {}
    competing = PathTransactionCoordinator()
    transaction_key = "startup-fallback-registry-cas"

    def read_truth(registry, coordinator, active_key, calls):
        assert active_key == transaction_key
        normalized = coordinator.normalize_path(active_key)
        assert getattr(coordinator._lock_state, "active", {}).get(normalized) == 1
        calls.append("post-write-read")
        start_competing.set()
        assert not competing_acquired.wait(timeout=0.2)
        captured_holder["registry"] = deepcopy(registry)
        return deepcopy(registry)

    controller, model, _view, _combo, durable, calls, published = (
        _registry_fallback_cas_controller(truth_reader=read_truth)
    )
    durable_holder["durable"] = durable

    def cooperating_writer():
        assert start_competing.wait(timeout=5)
        with competing.transaction(transaction_key):
            competing_acquired.set()
            durable_holder["durable"]["registry"]["late"] = "unrelated.json"

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(cooperating_writer)
        using_path, loaded = controller.get_sequence_config_from_registry()
        assert future.result(timeout=5) is None

    assert calls == [
        "lock-enter",
        "semantic-read",
        "controller-write",
        "post-write-read",
        "lock-exit",
    ]
    assert using_path == "fallback.json"
    assert loaded == captured_holder["registry"]
    assert model.registry == captured_holder["registry"]
    assert model.registry_entries == (("fallback", "fallback.json"),)
    assert durable["registry"]["late"] == "unrelated.json"
    assert published == []


@pytest.mark.parametrize("failure", ("false", "raise"))
def test_registry_fallback_partial_writer_failure_is_compensated(failure):
    durable_holder = {}

    def writer(path, _target):
        durable_holder["durable"]["registry"]["using_config_path"] = path
        if failure == "raise":
            raise OSError("partial registry replacement")
        return False

    controller, model, _view, combo, durable, calls, published = (
        _registry_fallback_cas_controller(writer=writer)
    )
    durable_holder["durable"] = durable
    previous_registry = model.registry

    using_path, _loaded = controller.get_sequence_config_from_registry()

    assert using_path == "previous.json"
    assert durable["registry"] == {
        "fallback": "fallback.json",
        "using_config_path": "missing.json",
    }
    assert model.registry == previous_registry
    assert model.using_config_path == "previous.json"
    assert combo.currentData() == "previous.json"
    assert controller.persistence_consistent is True
    assert controller.persistence_failure is None
    assert "controller-restore" in calls
    assert published == []


@pytest.mark.parametrize("error_type", (KeyboardInterrupt, SystemExit))
def test_registry_fallback_base_exception_during_compensation_still_aborts_reconciles_and_reraises(
    error_type,
):
    primary = error_type("compensation interrupted")
    secondary = error_type("projection rollback interrupted")
    durable_holder = {}

    def writer(path, _target):
        durable_holder["durable"]["registry"]["using_config_path"] = path
        return False

    def restorer(_target, _checkpoint, _owned_state):
        raise primary

    controller, model, _view, combo, durable, _calls, published = (
        _registry_fallback_cas_controller(writer=writer, restorer=restorer)
    )
    durable_holder["durable"] = durable
    controller.view.restore_configuration_state = lambda _state: (_ for _ in ()).throw(
        secondary
    )

    with pytest.raises(error_type) as caught:
        controller.get_sequence_config_from_registry()

    assert caught.value is primary
    assert caught.value.__traceback__ is not None
    traceback_names = []
    current_traceback = caught.value.__traceback__
    while current_traceback is not None:
        traceback_names.append(current_traceback.tb_frame.f_code.co_name)
        current_traceback = current_traceback.tb_next
    assert "restorer" in traceback_names
    assert any(
        "configuration rollback also failed" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    assert durable["registry"]["using_config_path"] == "fallback.json"
    assert model.registry == durable["registry"]
    assert model.using_config_path == "fallback.json"
    assert combo.items == [("fallback", "fallback.json")]
    assert controller.persistence_consistent is False
    assert controller.persistence_failure == (
        "startup active sequence config path persistence failed"
    )
    assert published == []


def test_registry_fallback_failed_compensation_enters_explicit_inconsistent_state():
    durable_holder = {}

    def writer(path, _target):
        durable_holder["durable"]["registry"]["using_config_path"] = path
        return False

    controller, model, _view, combo, durable, _calls, published = (
        _registry_fallback_cas_controller(
            writer=writer,
            restorer=lambda _target, _checkpoint, _owned_state: False,
        )
    )
    durable_holder["durable"] = durable

    using_path, loaded = controller.get_sequence_config_from_registry()

    assert using_path == "fallback.json"
    assert loaded == durable["registry"]
    assert model.registry == durable["registry"]
    assert model.using_config_path == "fallback.json"
    assert combo.items == [("fallback", "fallback.json")]
    assert controller.persistence_consistent is False
    assert controller.persistence_failure == (
        "startup active sequence config path persistence failed"
    )
    assert published == []


def test_registry_fallback_semantic_read_failure_retains_checkpoint_when_truth_is_unverifiable():
    controller, model, view, combo, _durable, calls, published = (
        _registry_fallback_cas_controller()
    )
    view.player_button = _Button(True)
    view.replay_button = _Button(True)
    view.data_button = _Button(True)
    adapter = controller._using_path_persistence_adapter_override

    def fail_semantic_read(_target):
        calls.append("semantic-read-failed")
        raise OSError("semantic registry read failed")

    def fail_recovery_read(_target):
        calls.append("recovery-read-failed")
        raise OSError("durable registry truth unavailable")

    adapter.read_semantic_current = fail_semantic_read
    adapter.read_durable_truth = fail_recovery_read
    previous_registry = deepcopy(model.registry)
    published.clear()

    using_path, loaded = controller.get_sequence_config_from_registry()

    assert calls == [
        "lock-enter",
        "semantic-read-failed",
        "lock-exit",
        "recovery-read-failed",
    ]
    assert using_path == "previous.json"
    assert loaded == previous_registry
    assert model.registry == previous_registry
    assert model.using_config_path == "previous.json"
    assert combo.items == [("previous", "previous.json")]
    assert combo.currentData() == "previous.json"
    assert controller.persistence_consistent is False
    assert controller.persistence_failure
    assert controller.projection_consistent is False
    assert controller.projection_failure
    assert view.capture_action_availability() == (False, False, False)
    assert published == []


def test_empty_registry_population_uses_no_configuration_placeholder():
    combo = _ComboBox()
    controller = _controller(view=SequenceConfigurationView(combobox=combo))
    controller.model.replace_registry({}, using_config_path=None)

    controller.add_file_to_using_file_combobox()

    assert combo.items == [("无配置", None)]


def test_actual_view_restores_runtime_actions_and_refreshes_player_after_invalid_to_valid():
    player = _Button(True)
    replay = _Button(True)
    data = _Button(True)
    refreshes = []
    view = SequenceConfigurationView(
        player_button=player,
        replay_button=replay,
        data_button=data,
    )
    load_results = iter(
        (
            (1, None),
            (0, _sequence("PLAY_AND_RECORD")),
        )
    )
    data_struct = SimpleNamespace(
        store_wave_data=np.ones(2),
        store_wave_data_multi=np.ones((2, 1)),
    )
    controller = _controller(
        model=SequenceConfigurationModel(data_struct=data_struct),
        view=view,
        config_loader=lambda _path: next(load_results),
        ok_code=0,
        availability_changed=lambda: (
            refreshes.append("refresh"),
            player.setEnabled(True),
        ),
    )

    controller.get_sequence_config_from_json()
    assert (player.enabled, replay.enabled, data.enabled) == (False, False, False)

    controller.get_sequence_config_from_json()

    assert refreshes == ["refresh"]
    assert (player.enabled, replay.enabled, data.enabled) == (True, True, True)


def test_actual_view_does_not_restore_actions_after_runtime_data_was_cleared():
    player = _Button(True)
    replay = _Button(True)
    data = _Button(True)
    data_struct = SimpleNamespace(
        store_wave_data=np.ones(2),
        store_wave_data_multi=np.ones((2, 1)),
    )
    view = SequenceConfigurationView(
        player_button=player,
        replay_button=replay,
        data_button=data,
    )
    load_results = iter(
        (
            (0, _sequence("RECORD_ONLY")),
            (1, None),
            (0, _sequence("RECORD_ONLY")),
        )
    )
    controller = _controller(
        model=SequenceConfigurationModel(data_struct=data_struct),
        view=view,
        config_loader=lambda _path: next(load_results),
        ok_code=0,
        availability_changed=lambda: player.setEnabled(True),
    )

    controller.get_sequence_config_from_json()
    assert (replay.enabled, data.enabled) == (True, True)
    controller.get_sequence_config_from_json()
    data_struct.store_wave_data = None
    data_struct.store_wave_data_multi = None

    controller.get_sequence_config_from_json()

    assert player.enabled is True
    assert (replay.enabled, data.enabled) == (False, False)


def test_actual_view_disables_replay_on_enabled_to_import_audio_transition():
    player = _Button(True)
    replay = _Button(True)
    data = _Button(True)
    refreshes = []
    view = SequenceConfigurationView(
        player_button=player,
        replay_button=replay,
        data_button=data,
    )
    load_results = iter(
        (
            (0, _sequence("RECORD_ONLY")),
            (0, _sequence("IMPORT_AUDIO")),
        )
    )
    data_struct = SimpleNamespace(
        store_wave_data=np.ones(2),
        store_wave_data_multi=np.ones((2, 1)),
    )
    controller = _controller(
        model=SequenceConfigurationModel(data_struct=data_struct),
        view=view,
        config_loader=lambda _path: next(load_results),
        ok_code=0,
        availability_changed=lambda: refreshes.append("refresh"),
    )

    controller.get_sequence_config_from_json()
    controller.get_sequence_config_from_json()

    assert refreshes == ["refresh", "refresh"]
    assert player.enabled is True
    assert replay.enabled is False
    assert data.enabled is True


def test_json_load_late_failure_is_contained_and_invisible_to_queued_observer(qapp):
    controller, workflow, view, data_struct, mono, _warnings = (
        _config_update_failure_controller(failure="none")
    )
    controller._config_loader = lambda _path: (0, _sequence("IMPORT_AUDIO"))
    old_model_config = controller.model.sequence_config
    old_workflow_snapshot = workflow.configuration_snapshot
    old_actions = view.capture_action_availability()
    old_presentation = view.capture_configuration_presentation_state()
    view.mark_configuration_loaded = lambda: (_ for _ in ()).throw(
        RuntimeError("loaded marker failed")
    )
    bus = SequenceEventBus()
    publication_calls = []
    observed = []

    def publish(event):
        publication_calls.append(event)
        bus.events.configuration_changed.emit(event)

    controller._publish_configuration = publish
    bus.events.configuration_changed.connect(observed.append, Qt.QueuedConnection)

    assert controller.get_sequence_config_from_json() is None
    qapp.processEvents()

    assert publication_calls == []
    assert observed == []
    assert controller.model.sequence_config is old_model_config
    assert controller.model.acquisition_mode == "RECORD_ONLY"
    assert controller.model.using_config_path == "old.json"
    assert controller.model.configuration_generation == 4
    assert workflow.configuration_generation == 4
    assert workflow.configuration_snapshot is old_workflow_snapshot
    assert data_struct.store_wave_data is mono
    assert view.capture_action_availability() == old_actions
    assert view.capture_configuration_presentation_state() == old_presentation


def test_combobox_persistence_false_restores_path_selection_and_runtime():
    old_path = "old.json"
    new_path = "new.json"
    data = np.ones(2)
    data_struct = SimpleNamespace(
        store_wave_data=data,
        store_wave_data_multi=data.reshape(-1, 1),
    )
    model = SequenceConfigurationModel(data_struct=data_struct)
    model.sequence_config = _sequence("RECORD_ONLY")
    model.analysis_config = {"display_sequence": []}
    model.replace_registry(
        {"old": old_path, "new": new_path, "using_config_path": old_path},
        using_config_path=old_path,
        entries=(("old", old_path), ("new", new_path)),
    )
    combo = _ComboBox()
    view = SequenceConfigurationView(combobox=combo)
    view.populate_configuration_entries(
        model.registry_entries,
        using_config_path=old_path,
        clear_first=True,
    )
    combo.setCurrentIndex(combo.findData(new_path))
    warnings = []
    controller = _controller(
        model=model,
        view=view,
        registry_loader=lambda: deepcopy(model.registry),
        using_path_updater=lambda _path: False,
        using_path_persistence_snapshotter=lambda: old_path,
        using_path_persistence_restorer=lambda _state: True,
        config_loader=lambda path: (
            (0, _sequence("IMPORT_AUDIO"))
            if path == new_path
            else pytest.fail(f"unexpected path: {path}")
        ),
        ok_code=0,
        logger=SimpleNamespace(warning=warnings.append),
    )
    previous_config = deepcopy(model.sequence_config)
    previous_generation = model.configuration_generation

    assert controller.on_using_file_combobox_changed("new") is None

    assert model.using_config_path == old_path
    assert model.sequence_config == previous_config
    assert model.configuration_generation == previous_generation
    assert model.data_struct.store_wave_data is data
    assert combo.currentData() == old_path
    assert warnings == [
        f"Failed to persist active sequence config path: {new_path}"
    ]


@pytest.mark.parametrize("publisher_failure", ["false", "raise"])
def test_combobox_publication_failure_keeps_final_committed_selection(
    publisher_failure,
):
    controller, workflow, view, data_struct, mono, warnings = (
        _config_update_failure_controller(failure="none")
    )
    view.combobox.addItem("new", "new.json")
    view.combobox.setCurrentIndex(view.combobox.findData("new.json"))
    persistence_calls = []
    controller._using_path_updater = (
        lambda path: persistence_calls.append(path) or True
    )
    if publisher_failure == "raise":
        controller._publish_configuration = lambda _event: (
            _ for _ in ()
        ).throw(RuntimeError("publish failed"))
    else:
        controller._publish_configuration = lambda _event: False

    result = controller.on_using_file_combobox_changed("new")

    assert result is None
    assert persistence_calls == ["new.json"]
    assert controller.model.acquisition_mode == "IMPORT_AUDIO"
    assert controller.model.using_config_path == "new.json"
    assert controller.model.configuration_generation == 5
    assert workflow.configuration_generation == 5
    assert controller.model.data_struct is data_struct
    assert data_struct.store_wave_data is None
    assert data_struct.stimulus_data is None
    assert view.combobox.currentData() == "new.json"
    assert (
        view.player_button.enabled,
        view.replay_button.enabled,
        view.data_button.enabled,
    ) == (True, False, False)
    assert any("publish sequence configuration" in message for message in warnings)


def _assert_queued_combobox_rollback(
    controller,
    workflow,
    view,
    data_struct,
    *,
    old_model_config,
    old_workflow_snapshot,
    old_stimulus,
    old_actions,
    mono,
    durable,
    expected_actions=None,
):
    assert controller.model.sequence_config is old_model_config
    assert controller.model.acquisition_mode == "RECORD_ONLY"
    assert controller.model.using_config_path == "old.json"
    assert controller.model.configuration_generation == 4
    assert workflow.configuration_generation == 4
    assert workflow.configuration_snapshot is old_workflow_snapshot
    assert durable["path"] == "old.json"
    assert controller.model.data_struct is data_struct
    assert data_struct.store_wave_data is mono
    assert data_struct.stimulus_data is old_stimulus
    assert view.combobox.currentData() == "old.json"
    assert view.capture_action_availability() == (
        old_actions if expected_actions is None else expected_actions
    )


@pytest.mark.parametrize("failure", ["persistence", "presentation"])
def test_real_queued_bus_cannot_reapply_rejected_combobox_candidate(
    qapp,
    failure,
):
    controller, workflow, view, data_struct, mono, _warnings = (
        _config_update_failure_controller(failure="none")
    )
    old_model_config = controller.model.sequence_config
    old_workflow_snapshot = workflow.configuration_snapshot
    old_stimulus = data_struct.stimulus_data
    old_actions = view.capture_action_availability()
    view.combobox.addItem("new", "new.json")
    view.combobox.setCurrentIndex(view.combobox.findData("new.json"))
    durable = {"path": "old.json"}

    def persist(path):
        if failure == "persistence" and path == "new.json":
            return False
        durable["path"] = path
        return True

    controller._using_path_updater = persist
    if failure == "presentation":
        controller._analysis_config_changed = lambda _config: (_ for _ in ()).throw(
            RuntimeError("projection failed")
        )
    bus = SequenceEventBus()
    controller._publish_configuration = bus.events.configuration_changed.emit
    observed = []
    handled = []
    original_handle = controller.handle_configuration_changed

    def handle(event):
        handled.append(event)
        return original_handle(event)

    controller.handle_configuration_changed = handle
    bus.events.configuration_changed.connect(
        controller.handle_configuration_changed,
        Qt.QueuedConnection,
    )
    bus.events.configuration_changed.connect(observed.append, Qt.QueuedConnection)

    assert controller.on_using_file_combobox_changed("new") is None
    qapp.processEvents()
    qapp.processEvents()

    assert observed == []
    assert handled == []
    _assert_queued_combobox_rollback(
        controller,
        workflow,
        view,
        data_struct,
        old_model_config=old_model_config,
        old_workflow_snapshot=old_workflow_snapshot,
        old_stimulus=old_stimulus,
        old_actions=old_actions,
        mono=mono,
        durable=durable,
        expected_actions=(
            (False, False, False) if failure == "presentation" else None
        ),
    )
    assert controller.projection_consistent is (failure != "presentation")
    assert bool(controller.projection_failure) is (failure == "presentation")


@pytest.mark.parametrize("publisher_failure", ["false", "raise"])
def test_real_queued_bus_keeps_commit_when_publisher_fails_after_emit(
    qapp,
    publisher_failure,
):
    controller, workflow, view, data_struct, mono, _warnings = (
        _config_update_failure_controller(failure="none")
    )
    view.combobox.addItem("new", "new.json")
    view.combobox.setCurrentIndex(view.combobox.findData("new.json"))
    durable = {"path": "old.json"}
    controller._using_path_updater = (
        lambda path: durable.__setitem__("path", path) or True
    )
    bus = SequenceEventBus()
    observed = []
    bus.events.configuration_changed.connect(
        controller.handle_configuration_changed,
        Qt.QueuedConnection,
    )
    bus.events.configuration_changed.connect(observed.append, Qt.QueuedConnection)

    def publish_then_fail(event):
        bus.events.configuration_changed.emit(event)
        bus.events.configuration_changed.emit(event)
        if publisher_failure == "raise":
            raise RuntimeError("publisher failed after emit")
        return False

    controller._publish_configuration = publish_then_fail

    assert controller.on_using_file_combobox_changed("new") is None
    qapp.processEvents()
    qapp.processEvents()

    assert controller.model.acquisition_mode == "IMPORT_AUDIO"
    assert controller.model.using_config_path == "new.json"
    assert controller.model.configuration_generation == 5
    assert workflow.configuration_generation == 5
    assert durable["path"] == "new.json"
    assert data_struct.store_wave_data is None
    assert view.combobox.currentData() == "new.json"
    assert view.capture_action_availability() == (True, False, False)
    assert len(observed) == 2


def test_real_queued_bus_rolls_back_failed_play_record_stimulus_initialization(qapp):
    controller, workflow, view, data_struct, mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    old_model_config = controller.model.sequence_config
    old_workflow_snapshot = workflow.configuration_snapshot
    old_stimulus = data_struct.stimulus_data
    old_actions = view.capture_action_availability()
    view.combobox.addItem("new", "new.json")
    view.combobox.setCurrentIndex(view.combobox.findData("new.json"))
    durable = {"path": "old.json"}
    controller._using_path_updater = (
        lambda path: durable.__setitem__("path", path) or True
    )
    bus = SequenceEventBus()
    controller._publish_configuration = bus.events.configuration_changed.emit
    observed = []
    bus.events.configuration_changed.connect(
        controller.handle_configuration_changed,
        Qt.QueuedConnection,
    )
    bus.events.configuration_changed.connect(observed.append, Qt.QueuedConnection)

    assert controller.on_using_file_combobox_changed("new") is None
    qapp.processEvents()
    qapp.processEvents()

    assert observed == []
    _assert_queued_combobox_rollback(
        controller,
        workflow,
        view,
        data_struct,
        old_model_config=old_model_config,
        old_workflow_snapshot=old_workflow_snapshot,
        old_stimulus=old_stimulus,
        old_actions=old_actions,
        mono=mono,
        durable=durable,
    )


def test_success_publishes_final_regenerated_configuration_after_durable_commit(qapp):
    controller, workflow, view, data_struct, _mono, warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    view.combobox.addItem("new", "new.json")
    view.combobox.setCurrentIndex(view.combobox.findData("new.json"))
    controller._config_loader = lambda _path: (
        0,
        _sequence("PLAY_AND_RECORD", {"stimulus_path": "original.wav"}),
    )
    controller._resolve_duplex_sample_rate = lambda _mic, _speaker: SimpleNamespace(
        ok=True,
        sample_rate=48_000,
        message="",
    )
    order = []

    def regenerate(data, detail, **_kwargs):
        order.append("runtime")
        detail["stimulus_path"] = "generated.wav"
        data.stimulus_data = np.ones(16)
        return True

    saved = []
    controller._stimulus_setter = regenerate
    controller._config_saver = (
        lambda config, path: order.append("save")
        or saved.append((deepcopy(config), path))
        or True
    )
    controller._using_path_updater = lambda path: order.append("persist") or True
    bus = SequenceEventBus()
    observed = []
    bus.events.configuration_changed.connect(observed.append, Qt.QueuedConnection)

    def publish(event):
        order.append("publish")
        bus.events.configuration_changed.emit(event)

    controller._publish_configuration = publish

    assert controller.on_using_file_combobox_changed("new") is None
    qapp.processEvents()

    assert order[-2:] == ["persist", "publish"]
    assert len(observed) == 1
    assert controller.model.configuration_generation == 5
    assert workflow.configuration_generation == 5
    assert controller.model.sequence_config[0]["seq1"]["acq"]["detail"] == {
        "stimulus_path": "generated.wav"
    }
    assert observed[0].configuration_snapshot.sequence_config[0]["seq1"]["acq"][
        "detail"
    ] == {"stimulus_path": "generated.wav"}
    assert saved == [
        (
            controller.model.sequence_config,
            "new.json",
        )
    ]
    assert warnings == []


def test_custom_config_saver_runs_inside_reentrant_selected_path_transaction():
    controller, _workflow, _view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    controller._resolve_duplex_sample_rate = lambda *_args: SimpleNamespace(
        ok=True, sample_rate=48000, message=""
    )
    controller._stimulus_setter = lambda staged, detail, **_kwargs: (
        setattr(staged, "stimulus_data", np.ones(8)),
        detail.__setitem__("stimulus_path", "generated.wav"),
        True,
    )[-1]
    competitor_started = threading.Event()
    competitor_acquired = threading.Event()
    pool = ThreadPoolExecutor(max_workers=1)
    competitor_future = []

    def competing_writer():
        competitor_started.set()
        with LoadUiConfig.sequence_config_file_transaction("new.json"):
            competitor_acquired.set()
            return True

    def custom_saver(_config, path):
        assert path == "new.json"
        competitor_future.append(pool.submit(competing_writer))
        assert competitor_started.wait(timeout=5)
        assert not competitor_acquired.wait(timeout=0.2)
        return True

    controller._config_saver = custom_saver
    try:
        assert controller.on_sequence_config_updated() is True
        assert competitor_future[0].result(timeout=5) is True
    finally:
        pool.shutdown(wait=True)

    assert competitor_acquired.is_set()


def test_controller_default_writer_reuses_its_active_coordinator(tmp_path):
    controller, _workflow, _view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    config_path = tmp_path / "selected.json"
    config_path.write_text(json.dumps(_sequence("PLAY_AND_RECORD")), encoding="utf-8")
    controller._registry_loader = lambda: {
        "selected": str(config_path),
        "using_config_path": str(config_path),
    }
    controller._config_loader = lambda _path: (0, _sequence("PLAY_AND_RECORD"))
    controller._path_exists = os.path.exists
    controller._resolve_duplex_sample_rate = lambda *_args: SimpleNamespace(
        ok=True, sample_rate=48000, message=""
    )
    controller._stimulus_setter = lambda staged, detail, **_kwargs: (
        setattr(staged, "stimulus_data", np.ones(8)),
        detail.__setitem__("stimulus_path", "generated.wav"),
        True,
    )[-1]
    coordinator = controller._path_transaction_coordinator
    original_transaction = coordinator.transaction
    selected_path = coordinator.normalize_path(str(config_path))
    active_depth = 0
    selected_depths = []

    @contextmanager
    def observed_transaction(path):
        nonlocal active_depth
        normalized = coordinator.normalize_path(path)
        with original_transaction(path):
            if normalized == selected_path:
                active_depth += 1
                selected_depths.append(active_depth)
            try:
                yield
            finally:
                if normalized == selected_path:
                    active_depth -= 1

    coordinator.transaction = observed_transaction

    assert controller.on_sequence_config_updated() is True

    assert selected_depths == [1, 2, 2, 2]
    load_code, durable = LoadUiConfig().load_sequence_config_from_json(
        str(config_path)
    )
    assert load_code == 0
    assert durable[0]["seq1"]["acq"]["detail"]["stimulus_path"] == "generated.wav"


def test_custom_config_saver_without_complete_protocol_is_rejected_prewrite():
    controller, workflow, _view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    controller._resolve_duplex_sample_rate = lambda *_args: SimpleNamespace(
        ok=True, sample_rate=48000, message=""
    )
    controller._stimulus_setter = lambda staged, detail, **_kwargs: (
        setattr(staged, "stimulus_data", np.ones(8)),
        detail.__setitem__("stimulus_path", "generated.wav"),
        True,
    )[-1]
    write_calls = []
    controller._config_saver = lambda config, path: write_calls.append(
        (config, path)
    ) or True
    controller._config_persistence_snapshotter = None
    controller._config_persistence_restorer = None
    published = []
    controller._publish_configuration = published.append

    assert controller.on_sequence_config_updated() is False

    assert write_calls == []
    assert workflow.configuration_generation == 4
    assert published == []


def test_custom_path_updater_without_complete_protocol_is_rejected_prewrite():
    controller, workflow, _view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="RECORD_ONLY")
    )
    controller._registry_loader = lambda: {
        "new": "new.json",
        "using_config_path": "missing.json",
    }
    controller._path_exists = lambda path: path == "new.json"
    write_calls = []
    controller._using_path_updater = lambda path: write_calls.append(path) or True
    controller._using_path_persistence_snapshotter = None
    controller._using_path_persistence_restorer = None
    published = []
    controller._publish_configuration = published.append

    assert controller.on_sequence_config_updated() is False

    assert write_calls == []
    assert workflow.configuration_generation == 4
    assert published == []


def test_incomplete_adapter_override_is_rejected_before_its_writer_runs():
    controller, workflow, _view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    controller._resolve_duplex_sample_rate = lambda *_args: SimpleNamespace(
        ok=True, sample_rate=48000, message=""
    )
    controller._stimulus_setter = lambda staged, detail, **_kwargs: (
        setattr(staged, "stimulus_data", np.ones(8)),
        detail.__setitem__("stimulus_path", "generated.wav"),
        True,
    )[-1]
    write_calls = []
    controller._config_persistence_adapter_override = SimpleNamespace(
        transaction=lambda _target: nullcontext(),
        capture=lambda _target: (True, b"old"),
        write=lambda payload, target: write_calls.append((payload, target)) or True,
    )
    published = []
    controller._publish_configuration = published.append

    assert controller.on_sequence_config_updated() is False

    assert write_calls == []
    assert workflow.configuration_generation == 4
    assert published == []


def test_custom_adapter_with_unshared_coordinator_is_rejected_prewrite():
    controller, workflow, _view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    controller._resolve_duplex_sample_rate = lambda *_args: SimpleNamespace(
        ok=True, sample_rate=48000, message=""
    )
    controller._stimulus_setter = lambda staged, detail, **_kwargs: (
        setattr(staged, "stimulus_data", np.ones(8)),
        detail.__setitem__("stimulus_path", "generated.wav"),
        True,
    )[-1]
    durable = {"value": "old"}
    write_calls = []
    controller._config_persistence_adapter_override = PersistenceAdapter(
        coordinator=PathTransactionCoordinator(),
        transaction_key=lambda target: target,
        checkpoint_reader=lambda _target: durable["value"],
        writer=lambda payload, _target: write_calls.append(payload) or True,
        checkpoint_restorer=lambda _target, checkpoint: durable.__setitem__(
            "value", checkpoint
        )
        or True,
        durable_truth_reader=lambda _target: durable["value"],
    )
    published = []
    controller._publish_configuration = published.append

    assert controller.on_sequence_config_updated() is False

    assert write_calls == []
    assert workflow.configuration_generation == 4
    assert published == []


def test_custom_adapter_without_semantic_reader_is_rejected_prewrite():
    controller, workflow, _view, _data_struct, _warnings = (
        _controller_for_persistence_context_failure()
    )
    write_calls = []
    controller._config_persistence_adapter_override = SimpleNamespace(
        coordinator=controller._path_transaction_coordinator,
        transaction=lambda _target: nullcontext(),
        capture=lambda _target: "checkpoint",
        write=lambda payload, target: write_calls.append((payload, target)) or True,
        conditional_restore=lambda *_args: True,
        read_durable_truth=lambda _target: (0, _sequence("PLAY_AND_RECORD")),
    )
    published = []
    controller._publish_configuration = published.append

    assert controller.on_sequence_config_updated() is False

    assert write_calls == []
    assert workflow.configuration_generation == 4
    assert published == []


def test_controller_adopts_coordinator_from_complete_adapter():
    coordinator = PathTransactionCoordinator()
    adapter = PersistenceAdapter(
        coordinator=coordinator,
        transaction_key=lambda target: target,
        checkpoint_reader=lambda _target: "old",
        writer=lambda _payload, _target: True,
        checkpoint_restorer=lambda _target, _checkpoint: True,
        durable_truth_reader=lambda _target: "old",
    )

    controller = _controller(config_persistence_adapter=adapter)

    assert controller._path_transaction_coordinator is coordinator
    assert controller._config_adapter() is adapter


def _controller_for_persistence_context_failure():
    controller, workflow, view, data_struct, _mono, warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    controller._resolve_duplex_sample_rate = lambda *_args: SimpleNamespace(
        ok=True, sample_rate=48000, message=""
    )
    controller._stimulus_setter = lambda staged, detail, **_kwargs: (
        setattr(staged, "stimulus_data", np.ones(8)),
        detail.__setitem__("stimulus_path", "generated.wav"),
        True,
    )[-1]
    return controller, workflow, view, data_struct, warnings


def test_adapter_with_noncallable_protocol_method_is_rejected_prewrite():
    controller, workflow, _view, _data_struct, _warnings = (
        _controller_for_persistence_context_failure()
    )
    write_calls = []
    controller._config_persistence_adapter_override = SimpleNamespace(
        coordinator=controller._path_transaction_coordinator,
        transaction=7,
        capture=lambda _target: "old",
        write=lambda payload, target: write_calls.append((payload, target)) or True,
        conditional_restore=lambda *_args: True,
        read_durable_truth=lambda _target: (0, _sequence("RECORD_ONLY")),
        read_semantic_current=lambda _target: _sequence("PLAY_AND_RECORD"),
    )
    published = []
    controller._publish_configuration = published.append

    assert controller.on_sequence_config_updated() is False

    assert write_calls == []
    assert workflow.configuration_generation == 4
    assert published == []


@pytest.mark.parametrize(
    "failure_kind",
    [
        "transaction_raise",
        "enter_raise",
        "body_raise",
        "body_suppress",
        "exit_raise",
        "exit_false",
    ],
)
def test_adapter_transaction_lifecycle_failure_is_contained_and_coherent(
    failure_kind,
):
    controller, workflow, view, data_struct, warnings = (
        _controller_for_persistence_context_failure()
    )
    old_wave = data_struct.store_wave_data
    old_durable = _sequence("PLAY_AND_RECORD", {"stimulus_path": "original.wav"})
    uncertain_durable = _sequence("IMPORT_AUDIO")
    durable = {"value": deepcopy(old_durable)}
    events = []

    class FaultContext:
        def __enter__(self):
            events.append("enter")
            if failure_kind == "enter_raise":
                durable["value"] = deepcopy(uncertain_durable)
                raise RuntimeError("enter failed after durable mutation")
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            events.append("exit")
            if failure_kind == "exit_raise":
                durable["value"] = deepcopy(uncertain_durable)
                raise RuntimeError("exit failed after durable mutation")
            if failure_kind == "body_suppress":
                return True
            if failure_kind in {"body_raise", "exit_false"}:
                return False
            return True

    def transaction(_target):
        events.append("transaction")
        if failure_kind == "transaction_raise":
            durable["value"] = deepcopy(uncertain_durable)
            raise RuntimeError("transaction failed after durable mutation")
        return FaultContext()

    def write(payload, _target):
        events.append("write")
        durable["value"] = deepcopy(payload)
        return True

    def conditional_restore(_target, checkpoint, owned_token):
        events.append("restore")
        if _durable_ownership_token(durable["value"]) != owned_token:
            return False
        durable["value"] = deepcopy(checkpoint)
        return True

    adapter = SimpleNamespace(
        coordinator=controller._path_transaction_coordinator,
        transaction=transaction,
        capture=lambda _target: deepcopy(durable["value"]),
        write=write,
        conditional_restore=conditional_restore,
        ownership_token=_durable_ownership_token,
        read_durable_truth=lambda _target: (0, deepcopy(durable["value"])),
        read_semantic_current=lambda _target: _sequence("PLAY_AND_RECORD"),
    )
    controller._config_persistence_adapter_override = adapter
    if failure_kind in {"body_raise", "body_suppress"}:
        def fail_capture(*_args, **_kwargs):
            durable["value"] = deepcopy(uncertain_durable)
            raise RuntimeError("body failed after durable mutation")

        controller._capture_persistence_checkpoint = fail_capture
    published = []
    controller._publish_configuration = published.append

    result = controller.on_sequence_config_updated()

    if failure_kind == "exit_false":
        assert result is True
        assert durable["value"][0]["seq1"]["acq"]["detail"][
            "stimulus_path"
        ] == "generated.wav"
        assert controller.persistence_consistent is True
        assert controller.model.acquisition_mode == "PLAY_AND_RECORD"
        assert workflow.configuration_generation == 5
        assert len(published) == 1
        assert events == ["transaction", "enter", "write", "exit"]
        return

    assert result is False

    assert published == []
    assert data_struct.store_wave_data is old_wave
    assert warnings
    if failure_kind in {
        "transaction_raise",
        "enter_raise",
        "body_raise",
        "body_suppress",
        "exit_raise",
    }:
        assert durable["value"] == uncertain_durable
        assert controller.persistence_consistent is False
        assert controller.model.acquisition_mode == "IMPORT_AUDIO"
        assert view.player_button.enabled is False
        if failure_kind == "transaction_raise":
            assert events == ["transaction"]
        elif failure_kind == "enter_raise":
            assert events == ["transaction", "enter"]
        elif failure_kind in {"body_raise", "body_suppress"}:
            assert events == ["transaction", "enter", "exit"]
        else:
            assert events == [
                "transaction",
                "enter",
                "write",
                "exit",
                "restore",
            ]


@pytest.mark.parametrize("base_error_type", [KeyboardInterrupt, SystemExit])
def test_persistence_context_exits_once_and_propagates_base_exception(
    base_error_type,
):
    controller, _workflow, _view, data_struct, _warnings = (
        _controller_for_persistence_context_failure()
    )
    old_wave = data_struct.store_wave_data
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None
    data_struct.store_wave_data = None
    abort_calls = []
    original_abort = transaction.abort
    transaction.abort = lambda: abort_calls.append("abort") or original_abort()
    events = []

    class Context:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc, _traceback):
            events.append(("exit", exc_type, exc))
            return True

    adapter = SimpleNamespace(transaction=lambda _target: Context())
    error = base_error_type("stop")

    with pytest.raises(base_error_type) as captured:
        controller._run_persistence_context(
            transaction,
            adapter,
            "config.json",
            operation="test persistence",
            body=lambda: (_ for _ in ()).throw(error),
        )

    assert captured.value is error
    assert events == ["enter", ("exit", base_error_type, error)]
    assert abort_calls == ["abort"]
    assert data_struct.store_wave_data is old_wave


@pytest.mark.parametrize("exit_result", [False, None])
def test_persistence_context_clean_false_or_none_exit_is_success(exit_result):
    controller, _workflow, _view, _data_struct, _warnings = (
        _controller_for_persistence_context_failure()
    )
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None
    events = []

    class Context:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc, traceback):
            events.append(("exit", exc_type, exc, traceback))
            return exit_result

    adapter = SimpleNamespace(transaction=lambda _target: Context())

    result = controller._run_persistence_context(
        transaction,
        adapter,
        "config.json",
        operation="test persistence",
        body=lambda: events.append("body") or True,
    )

    assert result is True
    assert events == ["enter", "body", ("exit", None, None, None)]


def test_persistence_contextmanager_clean_exit_is_success():
    controller, _workflow, _view, _data_struct, _warnings = (
        _controller_for_persistence_context_failure()
    )
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None
    events = []

    @contextmanager
    def persistence_context():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    adapter = SimpleNamespace(transaction=lambda _target: persistence_context())

    assert controller._run_persistence_context(
        transaction,
        adapter,
        "config.json",
        operation="test persistence",
        body=lambda: events.append("body") or True,
    ) is True
    assert events == ["enter", "body", "exit"]


def test_persistence_context_suppression_cannot_turn_incomplete_body_into_success():
    controller, _workflow, _view, _data_struct, _warnings = (
        _controller_for_persistence_context_failure()
    )
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None
    events = []
    body_error = RuntimeError("suppressed body sentinel")

    class Context:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc, traceback):
            events.append(("exit", exc_type, exc, traceback))
            return True

    def fail_body():
        raise body_error

    adapter = SimpleNamespace(transaction=lambda _target: Context())

    assert controller._run_persistence_context(
        transaction,
        adapter,
        "config.json",
        operation="test persistence",
        body=fail_body,
    ) is False
    assert events[0] == "enter"
    assert events[1][0:3] == ("exit", RuntimeError, body_error)
    assert events[1][3] is body_error.__traceback__


@pytest.mark.parametrize("failure_stage", ["factory", "enter"])
def test_persistence_setup_base_exception_aborts_provisional_state_before_reraise(
    failure_stage,
):
    controller, _workflow, _view, data_struct, _warnings = (
        _controller_for_persistence_context_failure()
    )
    old_wave = data_struct.store_wave_data
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None
    data_struct.store_wave_data = None
    abort_calls = []
    original_abort = transaction.abort
    transaction.abort = lambda: abort_calls.append("abort") or original_abort()
    error = KeyboardInterrupt(f"{failure_stage} sentinel")

    class Context:
        def __enter__(self):
            raise error

        def __exit__(self, *_args):
            pytest.fail("exit must not run when enter did not succeed")

    def context_factory(_target):
        if failure_stage == "factory":
            raise error
        return Context()

    with pytest.raises(KeyboardInterrupt) as captured:
        controller._run_persistence_context(
            transaction,
            SimpleNamespace(transaction=context_factory),
            "config.json",
            operation="test persistence",
            body=lambda: pytest.fail("body must not start"),
        )

    assert captured.value is error
    assert abort_calls == ["abort"]
    assert data_struct.store_wave_data is old_wave


@pytest.mark.parametrize("failure_stage", ("factory", "enter"))
@pytest.mark.parametrize("error_type", (KeyboardInterrupt, SystemExit))
def test_persistence_setup_base_exception_reconciles_partially_mutated_durable_truth(
    failure_stage,
    error_type,
):
    controller, _workflow, view, _data_struct, _warnings = (
        _controller_for_persistence_context_failure()
    )
    durable = {
        "config": _sequence("PLAY_AND_RECORD"),
        "registry": {
            "new": "new.json",
            "using_config_path": "new.json",
        },
    }
    controller._registry_loader = lambda: deepcopy(durable["registry"])
    published = []
    controller._publish_configuration = published.append
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None
    error = error_type(f"{failure_stage} setup interrupted")

    class Context:
        def __enter__(self):
            durable["config"] = _sequence("IMPORT_AUDIO")
            raise error

        def __exit__(self, *_args):
            pytest.fail("exit must not run after failed enter")

    def context_factory(_target):
        if failure_stage == "factory":
            durable["config"] = _sequence("IMPORT_AUDIO")
            raise error
        return Context()

    adapter = SimpleNamespace(
        transaction=context_factory,
        read_durable_truth=lambda _target: (0, deepcopy(durable["config"])),
    )
    controller._last_config_persistence_adapter = adapter

    with pytest.raises(error_type) as captured:
        controller._run_persistence_context(
            transaction,
            adapter,
            "new.json",
            operation="setup mutation",
            body=lambda: pytest.fail("body must not start"),
        )

    assert captured.value is error
    traceback_names = []
    current_traceback = captured.value.__traceback__
    while current_traceback is not None:
        traceback_names.append(current_traceback.tb_frame.f_code.co_name)
        current_traceback = current_traceback.tb_next
    assert ("context_factory" if failure_stage == "factory" else "__enter__") in (
        traceback_names
    )
    assert controller.persistence_consistent is False
    assert controller.persistence_failure == (
        "setup mutation setup interrupted before durable ownership capture"
    )
    assert controller.model.sequence_config == _sequence("IMPORT_AUDIO")
    assert controller.model.registry == durable["registry"]
    assert controller.model.using_config_path == "new.json"
    assert view.combobox.items == [("new", "new.json")]
    assert view.combobox.currentData() == "new.json"
    assert view.capture_action_availability() == (False, False, False)
    assert published == []


@pytest.mark.parametrize("failure_stage", ["writer", "exit"])
@pytest.mark.parametrize("base_error_type", [KeyboardInterrupt, SystemExit])
def test_durable_base_exception_compensates_and_aborts_before_exact_reraise(
    failure_stage,
    base_error_type,
):
    controller, workflow, view, data_struct, _warnings = (
        _controller_for_persistence_context_failure()
    )
    candidate = _sequence("PLAY_AND_RECORD")
    durable = {"value": deepcopy(candidate)}
    controller._config_loader = lambda _path: (0, deepcopy(durable["value"]))
    old_snapshot = controller.model.checkpoint_configuration_state()
    old_actions = view.capture_action_availability()
    old_wave = data_struct.store_wave_data
    events = []
    error = base_error_type(f"{failure_stage} sentinel")

    class Context:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc, traceback):
            events.append(("exit", exc_type, exc, traceback))
            if failure_stage == "exit":
                raise error
            return True

    def writer(payload, _target):
        events.append("write")
        durable["value"] = deepcopy(payload)
        if failure_stage == "writer":
            raise error
        return True

    def conditional_restore(_target, checkpoint, owned_token):
        events.append("restore")
        if _durable_ownership_token(durable["value"]) != owned_token:
            return False
        durable["value"] = deepcopy(checkpoint)
        return True

    controller._config_persistence_adapter_override = SimpleNamespace(
        coordinator=controller._path_transaction_coordinator,
        transaction=lambda _target: Context(),
        capture=lambda _target: deepcopy(durable["value"]),
        write=writer,
        conditional_restore=conditional_restore,
        ownership_token=_durable_ownership_token,
        read_durable_truth=lambda _target: (0, deepcopy(durable["value"])),
        read_semantic_current=lambda _target: deepcopy(durable["value"]),
    )
    published = []
    controller._publish_configuration = published.append

    with pytest.raises(base_error_type) as captured:
        controller.on_sequence_config_updated()

    assert captured.value is error
    traceback_names = []
    current = captured.value.__traceback__
    while current is not None:
        traceback_names.append(current.tb_frame.f_code.co_name)
        current = current.tb_next
    expected_traceback_frame = "writer" if failure_stage == "writer" else "__exit__"
    assert expected_traceback_frame in traceback_names
    assert events[-1] == "restore"
    if failure_stage == "writer":
        assert events[2][0:3] == ("exit", base_error_type, error)
        assert events[2][3] is not None
    else:
        assert events[2] == ("exit", None, None, None)
    assert durable["value"] == candidate
    assert controller.persistence_consistent is True
    assert controller.model.checkpoint_configuration_state() == old_snapshot
    assert workflow.configuration_generation == 4
    assert view.capture_action_availability() == old_actions
    assert data_struct.store_wave_data is old_wave
    assert published == []


@pytest.mark.parametrize(
    ("failure_stage", "primary_type", "rollback_type"),
    (
        ("writer", KeyboardInterrupt, SystemExit),
        ("exit", SystemExit, KeyboardInterrupt),
    ),
)
def test_durable_primary_and_projection_abort_base_exceptions_reconcile_and_mark_projection_untrusted(
    failure_stage,
    primary_type,
    rollback_type,
):
    controller, _workflow, view, _data_struct, _warnings = (
        _controller_for_persistence_context_failure()
    )
    durable = {"value": _sequence("PLAY_AND_RECORD")}
    controller._registry_loader = lambda: {
        "new": "new.json",
        "using_config_path": "new.json",
    }
    primary = primary_type(f"{failure_stage} primary")
    rollback_error = rollback_type("view rollback secondary")
    events = []

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            if failure_stage == "exit":
                raise primary
            return False

    def writer(payload, _target):
        durable["value"] = deepcopy(payload)
        if failure_stage == "writer":
            raise primary
        return True

    def conditional_restore(_target, checkpoint, owned_token):
        events.append("restore")
        if _durable_ownership_token(durable["value"]) != owned_token:
            return False
        durable["value"] = deepcopy(checkpoint)
        return True

    adapter = SimpleNamespace(
        coordinator=controller._path_transaction_coordinator,
        transaction=lambda _target: Context(),
        capture=lambda _target: deepcopy(durable["value"]),
        write=writer,
        conditional_restore=conditional_restore,
        ownership_token=_durable_ownership_token,
        read_durable_truth=lambda _target: (0, deepcopy(durable["value"])),
        read_semantic_current=lambda _target: deepcopy(durable["value"]),
    )
    controller._config_persistence_adapter_override = adapter
    controller.view.restore_configuration_state = lambda _state: (
        (_ for _ in ()).throw(rollback_error)
    )
    published = []
    controller._publish_configuration = published.append

    with pytest.raises(primary_type) as captured:
        controller.on_sequence_config_updated()

    assert captured.value is primary
    assert events == ["restore"]
    assert any(
        "Configuration rollback also failed" in note
        and rollback_type.__name__ in note
        for note in getattr(captured.value, "__notes__", ())
    )
    assert controller.persistence_consistent is True
    assert controller.persistence_failure is None
    assert controller.projection_consistent is False
    assert controller.projection_failure
    assert controller.model.sequence_config == durable["value"]
    assert controller.model.using_config_path == "new.json"
    assert view.combobox.items == [("new", "new.json")]
    assert view.combobox.currentData() == "new.json"
    assert view.capture_action_availability() == (False, False, False)
    assert published == []


def test_durable_base_exception_reports_failed_compensation_and_marks_uncertain():
    controller, _workflow, _view, data_struct, _warnings = (
        _controller_for_persistence_context_failure()
    )
    old_wave = data_struct.store_wave_data
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None
    data_struct.store_wave_data = None
    durable = {"value": "old"}
    events = []
    error = KeyboardInterrupt("writer primary")

    @contextmanager
    def persistence_context():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def writer():
        events.append("write")
        durable["value"] = "owned"
        raise error

    def failed_restore(_owned_state):
        events.append("restore")
        raise RuntimeError("compensation sentinel")

    def body():
        return transaction.attempt_durable_write(
            "interrupting writer",
            writer,
            rollback=failed_restore,
            state_reader=lambda: durable["value"],
        )

    with pytest.raises(KeyboardInterrupt) as captured:
        controller._run_persistence_context(
            transaction,
            SimpleNamespace(transaction=lambda _target: persistence_context()),
            "config.json",
            operation="test persistence",
            body=body,
        )

    assert captured.value is error
    assert events == ["enter", "write", "exit", "restore"]
    assert durable["value"] == "owned"
    assert controller.persistence_consistent is False
    assert data_struct.store_wave_data is old_wave
    assert any(
        "compensation sentinel" in note
        for note in getattr(error, "__notes__", ())
    )


@pytest.mark.parametrize("capture_error_type", (KeyboardInterrupt, SystemExit))
def test_post_writer_ownership_capture_base_exception_outranks_ordinary_writer_failure(
    capture_error_type,
):
    controller, _workflow, view, _data_struct, _warnings = (
        _controller_for_persistence_context_failure()
    )
    durable = {"config": _sequence("PLAY_AND_RECORD")}
    controller._registry_loader = lambda: {
        "new": "new.json",
        "using_config_path": "new.json",
    }
    capture_error = capture_error_type("ownership capture interrupted")
    capture_calls = []
    restore_calls = []

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            return False

    def capture(_target):
        capture_calls.append("capture")
        if len(capture_calls) == 2:
            raise capture_error
        return deepcopy(durable["config"])

    def writer(payload, _target):
        durable["config"] = deepcopy(payload)
        raise OSError("writer failed after replacement")

    adapter = SimpleNamespace(
        coordinator=controller._path_transaction_coordinator,
        transaction=lambda _target: Context(),
        capture=capture,
        write=writer,
        conditional_restore=lambda *_args: restore_calls.append("restore") or True,
        ownership_token=_durable_ownership_token,
        read_durable_truth=lambda _target: (0, deepcopy(durable["config"])),
        read_semantic_current=lambda _target: _sequence("PLAY_AND_RECORD"),
    )
    controller._config_persistence_adapter_override = adapter
    published = []
    controller._publish_configuration = published.append

    with pytest.raises(capture_error_type) as captured:
        controller.on_sequence_config_updated()

    assert captured.value is capture_error
    traceback_names = []
    current = captured.value.__traceback__
    while current is not None:
        traceback_names.append(current.tb_frame.f_code.co_name)
        current = current.tb_next
    assert "capture" in traceback_names
    assert any(
        "writer failed after replacement" in note
        for note in getattr(capture_error, "__notes__", ())
    )
    assert capture_calls == ["capture", "capture"]
    assert restore_calls == []
    assert controller.persistence_consistent is False
    assert controller.persistence_failure == (
        "sequence config persistence interrupted before body completion"
    )
    assert controller.projection_consistent is False
    assert "configuration-dependent projections" in controller.projection_failure
    assert controller.model.sequence_config == durable["config"]
    assert controller.model.using_config_path == "new.json"
    assert view.capture_action_availability() == (False, False, False)
    assert published == []


def test_reconcile_to_durable_truth_returns_explicit_success():
    controller, _workflow, view, _data_struct, _warnings = (
        _controller_for_persistence_context_failure()
    )
    controller.model.apply_configuration(
        ConfigurationSnapshot(
            sequence_config=_sequence("RECORD_ONLY"),
            analysis_config={},
            using_config_path="old.json",
        ),
        generation=4,
    )
    durable_registry = {
        "durable": "old.json",
        "using_config_path": "old.json",
    }
    durable_config = _sequence("RECORD_ONLY")
    controller._last_using_path_persistence_adapter = SimpleNamespace(
        read_durable_truth=lambda _target: deepcopy(durable_registry)
    )
    controller._last_config_persistence_adapter = SimpleNamespace(
        read_durable_truth=lambda _target: (0, deepcopy(durable_config))
    )
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None

    result = controller._reconcile_to_durable_truth(
        transaction,
        abort_transaction=False,
    )

    assert result is True, (
        controller.projection_failure,
        controller._projection_failures,
    )
    assert controller.model.sequence_config == durable_config
    assert controller.model.registry == durable_registry
    assert view.combobox.items == [("durable", "old.json")]
    assert view.capture_action_availability() == (False, False, False)


@pytest.mark.parametrize(
    "failure_stage",
    (
        "registry-read-ordinary",
        "config-read-base",
        "view-population",
        "disable",
        "multiple-view-errors",
    ),
)
def test_interruption_reconciliation_reports_each_failure_and_always_attempts_disable(
    failure_stage,
):
    controller, _workflow, view, _data_struct, _warnings = (
        _controller_for_persistence_context_failure()
    )
    durable_registry = {
        "new": "new.json",
        "using_config_path": "new.json",
    }
    durable_config = _sequence("IMPORT_AUDIO")
    old_registry = deepcopy(controller.model.registry)
    old_config = deepcopy(controller.model.sequence_config)
    expected_failure_messages = []
    calls = []

    def read_registry(_target):
        calls.append("registry-read")
        if failure_stage == "registry-read-ordinary":
            expected_failure_messages.append("registry read ordinary")
            raise OSError("registry read ordinary")
        return deepcopy(durable_registry)

    def read_config(_target):
        calls.append("config-read")
        if failure_stage == "config-read-base":
            expected_failure_messages.append("config read interrupted")
            raise SystemExit("config read interrupted")
        return 0, deepcopy(durable_config)

    controller._last_using_path_persistence_adapter = SimpleNamespace(
        read_durable_truth=read_registry
    )
    controller._last_config_persistence_adapter = SimpleNamespace(
        read_durable_truth=read_config
    )
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None
    original_population = view.populate_configuration_entries
    original_disable = view.set_sequence_config_available

    def populate(*args, **kwargs):
        calls.append("populate")
        if failure_stage in {"view-population", "multiple-view-errors"}:
            message = "population interrupted"
            expected_failure_messages.append(message)
            error_type = (
                SystemExit
                if failure_stage == "multiple-view-errors"
                else RuntimeError
            )
            raise error_type(message)
        return original_population(*args, **kwargs)

    def disable(available, *args, **kwargs):
        calls.append("disable")
        original_disable(available, *args, **kwargs)
        if failure_stage in {"disable", "multiple-view-errors"}:
            message = "disable interrupted"
            expected_failure_messages.append(message)
            raise KeyboardInterrupt(message)

    view.populate_configuration_entries = populate
    view.set_sequence_config_available = disable
    primary = KeyboardInterrupt("persistence primary")
    published = []
    controller._publish_configuration = published.append

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            return False

    with pytest.raises(KeyboardInterrupt) as captured:
        controller._run_persistence_context(
            transaction,
            SimpleNamespace(transaction=lambda _target: Context()),
            "new.json",
            operation="reconciliation failure",
            body=lambda: (_ for _ in ()).throw(primary),
        )

    assert captured.value is primary
    notes = getattr(primary, "__notes__", ())
    assert expected_failure_messages
    for message in expected_failure_messages:
        assert any(message in note for note in notes)
    assert calls[-1] == "disable"
    assert "disable" in calls[calls.index("registry-read") :]
    assert controller.persistence_consistent is False
    assert controller.projection_consistent is False
    assert controller.projection_failure
    assert view.capture_action_availability() == (False, False, False)
    if failure_stage in {
        "registry-read-ordinary",
        "config-read-base",
        "view-population",
        "multiple-view-errors",
    }:
        assert controller.model.registry == old_registry
        assert controller.model.sequence_config == old_config
    else:
        assert controller.model.registry == durable_registry
        assert controller.model.sequence_config == durable_config
    assert published == []


@pytest.mark.parametrize("interrupt_type", (KeyboardInterrupt, SystemExit))
@pytest.mark.parametrize(
    "failure_order",
    ("ordinary-then-interrupt", "interrupt-then-ordinary"),
)
def test_abort_after_durable_failure_prioritizes_interruptions_across_recovery_stages(
    interrupt_type,
    failure_order,
):
    controller, _workflow, view, _data_struct, _warnings = (
        _controller_for_persistence_context_failure()
    )
    durable_registry = {
        "new": "new.json",
        "using_config_path": "new.json",
    }
    durable_config = _sequence("IMPORT_AUDIO")
    controller._last_using_path_persistence_adapter = SimpleNamespace(
        read_durable_truth=lambda _target: deepcopy(durable_registry)
    )
    controller._last_config_persistence_adapter = SimpleNamespace(
        read_durable_truth=lambda _target: (0, deepcopy(durable_config))
    )
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None
    ordinary = RuntimeError("ordinary recovery failure")
    interruption = interrupt_type("recovery interrupted")
    calls = []

    def compensate():
        calls.append("compensate")
        if failure_order == "ordinary-then-interrupt":
            raise ordinary
        raise interruption

    def abort():
        calls.append("abort")
        if failure_order == "ordinary-then-interrupt":
            raise interruption
        raise ordinary

    transaction.compensate_durable_writes = compensate
    transaction.abort = abort
    published = []
    controller._publish_configuration = published.append

    with pytest.raises(interrupt_type) as captured:
        controller._abort_after_durable_failure(
            transaction,
            "durable recovery failed",
        )

    assert captured.value is interruption
    traceback_names = []
    current = captured.value.__traceback__
    while current is not None:
        traceback_names.append(current.tb_frame.f_code.co_name)
        current = current.tb_next
    expected_frame = (
        "abort"
        if failure_order == "ordinary-then-interrupt"
        else "compensate"
    )
    assert expected_frame in traceback_names
    assert calls == ["compensate", "abort"]
    assert any(
        "ordinary recovery failure" in note
        for note in getattr(interruption, "__notes__", ())
    )
    assert controller.persistence_consistent is False
    assert controller.projection_consistent is False
    assert controller.model.registry == durable_registry
    assert controller.model.sequence_config == durable_config
    assert view.capture_action_availability() == (False, False, False)
    assert published == []


@pytest.mark.parametrize("interrupt_type", (KeyboardInterrupt, SystemExit))
def test_compensation_logger_interruption_is_promoted_and_remaining_restorers_run(
    interrupt_type,
):
    controller, _workflow, _view, _data_struct, _warnings = (
        _controller_for_persistence_context_failure()
    )
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None
    calls = []
    interruption = interrupt_type("compensation logging interrupted")

    def remaining_restore(_owned_state):
        calls.append("remaining-restore")
        return True

    def failing_restore(_owned_state):
        calls.append("failing-restore")
        raise RuntimeError("ordinary restorer failure")

    assert transaction.attempt_durable_write(
        "remaining checkpoint",
        lambda: True,
        rollback=remaining_restore,
        state_reader=lambda: "remaining-owned",
    ) is True
    assert transaction.attempt_durable_write(
        "failing checkpoint",
        lambda: True,
        rollback=failing_restore,
        state_reader=lambda: "failing-owned",
    ) is True

    def warning(message):
        calls.append("warning")
        if "ordinary restorer failure" in message:
            raise interruption

    controller._logger = SimpleNamespace(warning=warning)

    with pytest.raises(interrupt_type) as captured:
        transaction.compensate_durable_writes()

    assert captured.value is interruption
    traceback_names = []
    current = captured.value.__traceback__
    while current is not None:
        traceback_names.append(current.tb_frame.f_code.co_name)
        current = current.tb_next
    assert "warning" in traceback_names
    assert calls == ["failing-restore", "warning", "remaining-restore"]
    assert any(
        "ordinary restorer failure" in note
        for note in getattr(interruption, "__notes__", ())
    )
    assert transaction.durable_compensation_failures == (
        "failing checkpoint: RuntimeError: ordinary restorer failure",
    )


@pytest.mark.parametrize(
    ("read_interrupt_type", "disable_interrupt_type"),
    (
        (KeyboardInterrupt, SystemExit),
        (SystemExit, KeyboardInterrupt),
    ),
)
def test_abort_recovery_promotes_earliest_late_interruption_after_multiple_ordinary_failures(
    read_interrupt_type,
    disable_interrupt_type,
):
    controller, _workflow, view, _data_struct, _warnings = (
        _controller_for_persistence_context_failure()
    )
    read_interruption = read_interrupt_type("durable read interrupted")
    disable_interruption = disable_interrupt_type("disable interrupted later")
    calls = []
    controller._last_using_path_persistence_adapter = SimpleNamespace(
        read_durable_truth=lambda _target: (
            calls.append("read"),
            (_ for _ in ()).throw(read_interruption),
        )[-1]
    )
    original_disable = view.set_sequence_config_available

    def disable(*args, **kwargs):
        calls.append("disable")
        original_disable(*args, **kwargs)
        raise disable_interruption

    view.set_sequence_config_available = disable
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None
    transaction.compensate_durable_writes = lambda: (
        calls.append("compensate"),
        (_ for _ in ()).throw(RuntimeError("compensation ordinary")),
    )[-1]
    transaction.abort = lambda: (
        calls.append("abort"),
        (_ for _ in ()).throw(OSError("abort ordinary")),
    )[-1]
    published = []
    controller._publish_configuration = published.append

    with pytest.raises(read_interrupt_type) as captured:
        controller._abort_after_durable_failure(
            transaction,
            "multi-stage recovery failed",
        )

    assert captured.value is read_interruption
    assert calls == ["compensate", "abort", "read", "disable"]
    notes = getattr(read_interruption, "__notes__", ())
    assert any("compensation ordinary" in note for note in notes)
    assert any("abort ordinary" in note for note in notes)
    assert any("disable interrupted later" in note for note in notes)
    assert controller.persistence_consistent is False
    assert controller.projection_consistent is False
    assert controller.projection_failure
    assert view.capture_action_availability() == (False, False, False)
    assert published == []


@pytest.mark.parametrize(
    ("rollback_interrupt_type", "view_interrupt_type"),
    (
        (KeyboardInterrupt, SystemExit),
        (SystemExit, KeyboardInterrupt),
    ),
)
def test_configuration_abort_aggregates_all_projection_failures_before_reraising(
    rollback_interrupt_type,
    view_interrupt_type,
):
    controller, _workflow, view, _data_struct, _warnings = (
        _controller_for_persistence_context_failure()
    )
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None
    calls = []
    rollback_interruption = rollback_interrupt_type("rollback interrupted")
    view_interruption = view_interrupt_type("view restore interrupted later")

    def ordinary_rollback():
        calls.append("ordinary-rollback")
        raise RuntimeError("ordinary rollback failure")

    def interrupting_rollback():
        calls.append("interrupting-rollback")
        raise rollback_interruption

    transaction.attempt(
        "later projection",
        lambda: True,
        rollback=interrupting_rollback,
        required_projection=True,
    )
    transaction.attempt(
        "first projection",
        lambda: True,
        rollback=ordinary_rollback,
        required_projection=True,
    )

    def restore_view(_state):
        calls.append("view-restore")
        raise view_interruption

    view.restore_configuration_state = restore_view

    with pytest.raises(rollback_interrupt_type) as captured:
        transaction.abort()

    assert captured.value is rollback_interruption
    assert calls == [
        "ordinary-rollback",
        "interrupting-rollback",
        "view-restore",
    ]
    notes = getattr(rollback_interruption, "__notes__", ())
    assert any("ordinary rollback failure" in note for note in notes)
    assert any("view restore interrupted later" in note for note in notes)
    assert controller.projection_consistent is False
    assert controller.projection_failure
    assert view.capture_action_availability() == (False, False, False)


@pytest.mark.parametrize("exit_result", [False, None])
def test_persistence_context_unsuppressed_ordinary_exception_is_contained(
    exit_result,
):
    controller, _workflow, _view, _data_struct, warnings = (
        _controller_for_persistence_context_failure()
    )
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None
    body_error = RuntimeError("ordinary body sentinel")
    observed = []

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            observed.append((exc_type, exc, traceback))
            return exit_result

    adapter = SimpleNamespace(transaction=lambda _target: Context())

    assert controller._run_persistence_context(
        transaction,
        adapter,
        "config.json",
        operation="test persistence",
        body=lambda: (_ for _ in ()).throw(body_error),
    ) is False
    assert observed[0][0:2] == (RuntimeError, body_error)
    assert observed[0][2] is body_error.__traceback__
    assert any("ordinary body sentinel" in warning for warning in warnings)


def test_persistence_exit_failure_cannot_mask_keyboard_interrupt():
    controller, _workflow, _view, _data_struct, warnings = (
        _controller_for_persistence_context_failure()
    )
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None
    body_error = KeyboardInterrupt("keyboard sentinel")
    exit_error = RuntimeError("exit cleanup sentinel")
    observed = []

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            observed.append((exc_type, exc, traceback))
            raise exit_error

    def fail_body():
        raise body_error

    adapter = SimpleNamespace(transaction=lambda _target: Context())

    with pytest.raises(KeyboardInterrupt) as captured:
        controller._run_persistence_context(
            transaction,
            adapter,
            "config.json",
            operation="test persistence",
            body=fail_body,
        )

    assert captured.value is body_error
    traceback_names = []
    current = captured.value.__traceback__
    while current is not None:
        traceback_names.append(current.tb_frame.f_code.co_name)
        current = current.tb_next
    assert "fail_body" in traceback_names
    assert observed[0][0:2] == (KeyboardInterrupt, body_error)
    assert any(
        "exit cleanup sentinel" in note
        for note in getattr(body_error, "__notes__", ())
    )
    assert any("exit cleanup sentinel" in warning for warning in warnings)


@pytest.mark.parametrize("exit_error_type", (KeyboardInterrupt, SystemExit))
def test_persistence_exit_base_exception_outranks_ordinary_body_failure_and_recovers(
    exit_error_type,
):
    controller, _workflow, view, _data_struct, _warnings = (
        _controller_for_persistence_context_failure()
    )
    durable_registry = {
        "new": "new.json",
        "using_config_path": "new.json",
    }
    durable_config = _sequence("IMPORT_AUDIO")
    controller._registry_loader = lambda: deepcopy(durable_registry)
    controller._last_config_persistence_adapter = SimpleNamespace(
        read_durable_truth=lambda _target: (0, deepcopy(durable_config))
    )
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None
    body_error = RuntimeError("ordinary body failed")
    exit_error = exit_error_type("exit interrupted")

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            raise exit_error

    published = []
    controller._publish_configuration = published.append

    with pytest.raises(exit_error_type) as captured:
        controller._run_persistence_context(
            transaction,
            SimpleNamespace(transaction=lambda _target: Context()),
            "new.json",
            operation="mixed persistence failure",
            body=lambda: (_ for _ in ()).throw(body_error),
        )

    assert captured.value is exit_error
    traceback_names = []
    current = captured.value.__traceback__
    while current is not None:
        traceback_names.append(current.tb_frame.f_code.co_name)
        current = current.tb_next
    assert "__exit__" in traceback_names
    assert any(
        "ordinary body failed" in note
        for note in getattr(exit_error, "__notes__", ())
    )
    assert controller.persistence_consistent is False
    assert controller.persistence_failure == (
        "mixed persistence failure cleanup interrupted while handling body failure"
    )
    assert controller.projection_consistent is False
    assert "configuration-dependent projections" in controller.projection_failure
    assert controller.model.sequence_config == durable_config
    assert controller.model.registry == durable_registry
    assert controller.model.using_config_path == "new.json"
    assert view.combobox.items == [("new", "new.json")]
    assert view.capture_action_availability() == (False, False, False)
    assert published == []


def test_persistence_exit_failure_cannot_mask_ordinary_body_exception():
    controller, _workflow, _view, _data_struct, warnings = (
        _controller_for_persistence_context_failure()
    )
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None
    body_error = RuntimeError("ordinary primary sentinel")
    exit_error = ValueError("ordinary exit sentinel")
    observed = []

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            observed.append((exc_type, exc, traceback))
            raise exit_error

    def fail_body():
        raise body_error

    adapter = SimpleNamespace(transaction=lambda _target: Context())

    assert controller._run_persistence_context(
        transaction,
        adapter,
        "config.json",
        operation="test persistence",
        body=fail_body,
    ) is False

    assert observed[0][0:2] == (RuntimeError, body_error)
    traceback_names = []
    current = body_error.__traceback__
    while current is not None:
        traceback_names.append(current.tb_frame.f_code.co_name)
        current = current.tb_next
    assert "fail_body" in traceback_names
    assert any(
        "ordinary exit sentinel" in note
        for note in getattr(body_error, "__notes__", ())
    )
    assert any("ordinary exit sentinel" in warning for warning in warnings)


def test_persistence_base_exception_releases_real_path_transaction(tmp_path):
    controller, _workflow, _view, _data_struct, _warnings = (
        _controller_for_persistence_context_failure()
    )
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None
    target = str(tmp_path / "base-exception-release.json")
    adapter = SimpleNamespace(
        transaction=controller._path_transaction_coordinator.transaction
    )

    with pytest.raises(KeyboardInterrupt):
        controller._run_persistence_context(
            transaction,
            adapter,
            target,
            operation="test path release",
            body=lambda: (_ for _ in ()).throw(KeyboardInterrupt("stop")),
        )

    with PathTransactionCoordinator().transaction(target):
        pass


class _CopyPoison:
    def __init__(self):
        self.calls = []

    def __copy__(self):
        self.calls.append("copy")
        raise AssertionError("copy hook must not run")

    def __deepcopy__(self, _memo):
        self.calls.append("deepcopy")
        raise AssertionError("deepcopy hook must not run")

    def __reduce_ex__(self, _protocol):
        self.calls.append("reduce")
        raise AssertionError("reduce hook must not run")


class _CopyTrackedWave(np.ndarray):
    copy_calls = 0

    def copy(self, *args, **kwargs):
        type(self).copy_calls += 1
        return super().copy(*args, **kwargs)

    def __deepcopy__(self, _memo):
        type(self).copy_calls += 1
        raise AssertionError("waveform must not be deep-copied")


def test_configuration_admission_journals_known_fields_without_copying_unrelated_runtime():
    controller, _workflow, _view, data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="IMPORT_AUDIO")
    )
    poison = _CopyPoison()
    data_struct.unrelated_runtime = poison
    controller.model._registry["unrelated"] = poison
    clear_calls = []
    data_struct.clear_data = lambda: clear_calls.append(True) or (_ for _ in ()).throw(
        AssertionError("custom clear_data must not run")
    )
    known = {
        "store_wave_data": data_struct.store_wave_data,
        "store_wave_data_multi": data_struct.store_wave_data_multi,
        "wav_calibration_metadata": data_struct.wav_calibration_metadata,
    }

    assert controller.on_sequence_config_updated() is True

    assert clear_calls == []
    assert poison.calls == []
    assert data_struct.unrelated_runtime is poison
    assert all(getattr(data_struct, name) is None for name in known)


def test_failed_configuration_restores_exact_known_field_references_without_copy_hooks():
    controller, workflow, _view, data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="presentation", mode="IMPORT_AUDIO")
    )
    poison = _CopyPoison()
    data_struct.unrelated_runtime = poison
    split = {"retained": object()}
    fft = object()
    stft = object()
    data_struct.split_repeat_data = split
    data_struct.fft_result = fft
    data_struct.stft_result = stft
    old_fields = {
        name: getattr(data_struct, name)
        for name in (
            "store_wave_data",
            "store_wave_data_multi",
            "wav_calibration_metadata",
            "wav_calibration_metadata_authoritative",
            "wav_calibration_warning_shown",
            "split_repeat_data",
            "fft_result",
            "stft_result",
        )
    }

    assert controller.on_sequence_config_updated() is False

    assert workflow.configuration_generation == 4
    assert poison.calls == []
    assert data_struct.unrelated_runtime is poison
    for name, expected in old_fields.items():
        assert getattr(data_struct, name) is expected


def test_stimulus_setter_receives_only_owned_isolated_fields():
    controller, _workflow, _view, data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    controller._resolve_duplex_sample_rate = lambda *_args: SimpleNamespace(
        ok=True, sample_rate=48000, message=""
    )
    poison = _CopyPoison()
    data_struct.unrelated_runtime = poison
    observed = []

    def prepare(staged, _detail, **_kwargs):
        observed.append(set(vars(staged)))
        assert not hasattr(staged, "unrelated_runtime")
        staged.stimulus_data = np.arange(4.0)
        staged.stimulus_info = {"sample_rate": 48000}
        staged.alignment_sample_count = 4
        return False

    controller._stimulus_setter = prepare

    assert controller.on_sequence_config_updated() is True

    assert observed == [
        {
            "stimulus_data",
            "stimulus_info",
            "sample_rate",
            "alignment_sample_count",
        }
    ]
    assert poison.calls == []
    assert data_struct.unrelated_runtime is poison
    assert np.array_equal(data_struct.stimulus_data, np.arange(4.0))
    assert data_struct.alignment_sample_count == 4


def test_large_waveform_checkpoint_is_reference_only_and_plot_capture_is_lazy():
    controller, _workflow, _view, data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="RECORD_ONLY")
    )
    wave = np.zeros(8 * 1024 * 1024, dtype=np.float64).view(_CopyTrackedWave)
    _CopyTrackedWave.copy_calls = 0
    data_struct.store_wave_data = wave
    data_struct.store_wave_data_multi = wave.reshape(-1, 1)
    plot_capture_calls = []
    controller._plot_state_capturer = lambda: plot_capture_calls.append(True) or (
        wave,
        data_struct.sample_rate,
    )
    controller._plot_state_restorer = lambda _state: True

    assert controller.on_sequence_config_updated() is True

    assert plot_capture_calls == []
    assert _CopyTrackedWave.copy_calls == 0
    assert data_struct.store_wave_data is wave


def test_large_waveform_mode_change_restores_plot_from_original_reference():
    controller, workflow, _view, data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="IMPORT_AUDIO")
    )
    wave = np.zeros(8 * 1024 * 1024, dtype=np.float64).view(_CopyTrackedWave)
    _CopyTrackedWave.copy_calls = 0
    multi = wave.reshape(-1, 1)
    data_struct.store_wave_data = wave
    data_struct.store_wave_data_multi = multi
    plot_events = []
    controller._plot_state_capturer = lambda: plot_events.append("capture") or (
        wave,
        data_struct.sample_rate,
    )
    controller._clear_plot = lambda: plot_events.append("clear")

    def restore_plot(state):
        plot_events.append(("restore", state[0], state[1]))
        return True

    controller._plot_state_restorer = restore_plot
    controller._clear_import_identity = lambda: (_ for _ in ()).throw(
        RuntimeError("later cleanup failed")
    )
    controller._import_identity_state_capturer = lambda: ("old.wav", {"rate": 1})
    controller._import_identity_state_restorer = lambda _state: True

    assert controller.on_sequence_config_updated() is False

    assert workflow.configuration_generation == 4
    assert data_struct.store_wave_data is wave
    assert data_struct.store_wave_data_multi is multi
    assert _CopyTrackedWave.copy_calls == 0
    assert plot_events[0:2] == ["capture", "clear"]
    assert plot_events[2][0] == "restore"
    assert plot_events[2][1] is wave


def test_sequence_window_plot_projection_adapter_captures_waveform_by_reference():
    wave = np.zeros(8, dtype=np.float64)
    plotted = []
    window = SimpleNamespace(
        data_struct=SimpleNamespace(
            store_wave_data=wave,
            store_wave_data_multi=None,
            sample_rate=32000,
        ),
        _clear_plot_area=lambda: plotted.append("clear"),
        plot_waveform_to_workspace=lambda data, rate: plotted.append((data, rate)),
    )

    state = SequenceWindow._capture_plot_projection_state(window)
    SequenceWindow._restore_plot_projection_state(window, state)

    assert state[0] is wave
    assert state[1] == 32000
    assert plotted[0] == "clear"
    assert plotted[1][0] is wave
    assert plotted[1][1] == 32000


def test_custom_adapter_serializes_conditional_restore_with_cooperating_writer(
    tmp_path,
):
    transaction_key = str(tmp_path / "semantic-config-state")
    durable = {"value": "owned"}
    restore_entered = threading.Event()
    release_restore = threading.Event()
    writer_started = threading.Event()
    writer_acquired = threading.Event()
    coordinator = PathTransactionCoordinator()
    competing_coordinator = PathTransactionCoordinator()

    def restore(_target, checkpoint):
        restore_entered.set()
        assert release_restore.wait(timeout=5)
        durable["value"] = checkpoint
        return True

    adapter = PersistenceAdapter(
        coordinator=coordinator,
        transaction_key=lambda _target: transaction_key,
        checkpoint_reader=lambda _target: durable["value"],
        writer=lambda payload, _target: durable.__setitem__("value", payload) or True,
        checkpoint_restorer=restore,
        durable_truth_reader=lambda _target: durable["value"],
    )

    def compensate():
        return adapter.conditional_restore(
            "selected", "old", adapter.ownership_token("owned")
        )

    def write_newer():
        writer_started.set()
        with competing_coordinator.transaction(transaction_key):
            writer_acquired.set()
            durable["value"] = "newer"

    with ThreadPoolExecutor(max_workers=2) as pool:
        compensation = pool.submit(compensate)
        assert restore_entered.wait(timeout=5)
        writer = pool.submit(write_newer)
        assert writer_started.wait(timeout=5)
        assert not writer_acquired.wait(timeout=0.2)
        release_restore.set()
        assert compensation.result(timeout=5) is True
        assert writer.result(timeout=5) is None

    assert durable["value"] == "newer"


def test_custom_adapter_does_not_restore_over_a_newer_owned_state():
    durable = {"value": "newer"}
    restore_calls = []
    adapter = PersistenceAdapter(
        coordinator=PathTransactionCoordinator(),
        transaction_key=lambda target: target,
        checkpoint_reader=lambda _target: durable["value"],
        writer=lambda payload, _target: durable.__setitem__("value", payload) or True,
        checkpoint_restorer=lambda _target, checkpoint: restore_calls.append(
            checkpoint
        )
        or durable.__setitem__("value", checkpoint)
        or True,
        durable_truth_reader=lambda _target: durable["value"],
    )

    assert adapter.conditional_restore(
        "semantic-config", "old", adapter.ownership_token("owned")
    ) is False

    assert durable["value"] == "newer"
    assert restore_calls == []


@pytest.mark.parametrize(
    ("owned_state", "current_state", "restored"),
    (
        (1, True, False),
        (1, 1.0, False),
        (-0.0, 0.0, False),
        (float("nan"), float("nan"), True),
        ({"a": 1, "b": [True]}, {"b": [True], "a": 1}, True),
        ((True, b"old"), (True, b"old"), True),
        ((True, b"old"), (True, b"new"), False),
    ),
    ids=(
        "int-vs-bool",
        "int-vs-float",
        "negative-vs-positive-zero",
        "nan-policy",
        "reordered-object",
        "exact-bytes",
        "different-bytes",
    ),
)
def test_persistence_adapter_uses_exact_fail_closed_ownership_tokens(
    owned_state, current_state, restored
):
    durable = {"value": current_state}
    restore_calls = []
    adapter = PersistenceAdapter(
        coordinator=PathTransactionCoordinator(),
        transaction_key=lambda target: target,
        checkpoint_reader=lambda _target: durable["value"],
        writer=lambda payload, _target: durable.__setitem__("value", payload)
        or True,
        checkpoint_restorer=lambda _target, checkpoint: restore_calls.append(
            checkpoint
        )
        or durable.__setitem__("value", checkpoint)
        or True,
        durable_truth_reader=lambda _target: durable["value"],
    )

    owned_token = adapter.ownership_token(owned_state)
    result = adapter.conditional_restore("exact-ownership", "old", owned_token)

    assert result is restored
    assert restore_calls == (["old"] if restored else [])


def test_persistence_adapter_ownership_token_rejects_unsupported_values_without_equality():
    class EqualityPoison:
        def __init__(self):
            self.calls = []

        def __eq__(self, _other):
            self.calls.append("eq")
            raise AssertionError("live equality must not run")

    poison = EqualityPoison()
    adapter = PersistenceAdapter(
        coordinator=PathTransactionCoordinator(),
        transaction_key=lambda target: target,
        checkpoint_reader=lambda _target: poison,
        writer=lambda _payload, _target: True,
        checkpoint_restorer=lambda _target, _checkpoint: pytest.fail(
            "unsupported ownership must not restore"
        ),
        durable_truth_reader=lambda _target: poison,
    )

    with pytest.raises(TypeError):
        adapter.ownership_token(poison)

    assert poison.calls == []


def test_unsupported_post_write_ownership_refuses_compensation_and_reconciles_inconsistent():
    class Unsupported:
        pass

    controller = _controller(
        registry_loader=lambda: {},
        config_loader=lambda _path: (1, None),
    )
    transaction = controller._begin_configuration_transaction(
        selection_path=None
    )
    assert transaction is not None
    restore_calls = []
    adapter = PersistenceAdapter(
        coordinator=controller._path_transaction_coordinator,
        transaction_key=lambda target: target,
        checkpoint_reader=lambda _target: Unsupported(),
        writer=lambda _payload, _target: False,
        checkpoint_restorer=lambda _target, checkpoint: restore_calls.append(
            checkpoint
        )
        or True,
        durable_truth_reader=lambda _target: {},
    )

    assert transaction.attempt_durable_write(
        "unsupported ownership write",
        adapter.write,
        "candidate",
        "target",
        rollback=lambda owned_token: adapter.conditional_restore(
            "target", "checkpoint", owned_token
        ),
        state_reader=lambda: adapter.capture("target"),
        ownership_tokenizer=adapter.ownership_token,
    ) is False

    assert controller._abort_after_durable_failure(
        transaction, "unsupported ownership"
    ) is False
    assert restore_calls == []
    assert controller.persistence_consistent is False
    assert controller.persistence_failure == "unsupported ownership"


@pytest.mark.parametrize("save_failure", ["false", "raise"])
def test_regenerated_configuration_save_failure_aborts_before_path_commit_or_publish(
    qapp,
    save_failure,
):
    controller, workflow, view, data_struct, mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    old_model_config = controller.model.sequence_config
    old_workflow_snapshot = workflow.configuration_snapshot
    old_stimulus = data_struct.stimulus_data
    old_actions = view.capture_action_availability()
    view.combobox.addItem("new", "new.json")
    view.combobox.setCurrentIndex(view.combobox.findData("new.json"))
    controller._resolve_duplex_sample_rate = lambda _mic, _speaker: SimpleNamespace(
        ok=True,
        sample_rate=48_000,
        message="",
    )

    def regenerate(data, detail, **_kwargs):
        detail["stimulus_path"] = "generated.wav"
        data.stimulus_data = np.ones(16)
        return True

    controller._stimulus_setter = regenerate
    if save_failure == "raise":
        controller._config_saver = lambda *_args: (_ for _ in ()).throw(
            RuntimeError("save failed")
        )
    else:
        controller._config_saver = lambda *_args: False
    persistence_calls = []
    controller._using_path_updater = lambda path: persistence_calls.append(path) or True
    bus = SequenceEventBus()
    observed = []
    controller._publish_configuration = bus.events.configuration_changed.emit
    bus.events.configuration_changed.connect(observed.append, Qt.QueuedConnection)

    assert controller.on_using_file_combobox_changed("new") is None
    qapp.processEvents()

    assert persistence_calls == []
    assert observed == []
    _assert_queued_combobox_rollback(
        controller,
        workflow,
        view,
        data_struct,
        old_model_config=old_model_config,
        old_workflow_snapshot=old_workflow_snapshot,
        old_stimulus=old_stimulus,
        old_actions=old_actions,
        mono=mono,
        durable={"path": "old.json"},
    )


@pytest.mark.parametrize(
    "presentation_failure",
    [
        "reset",
        "missing_prompt",
        "loaded_marker",
        "runtime_resolver",
        "mode_refresh",
        "channel_refresh",
        "availability",
        "focus",
    ],
)
def test_real_queued_bus_rolls_back_late_combobox_presentation_failure(
    qapp,
    presentation_failure,
):
    mode = (
        "PLAY_AND_RECORD"
        if presentation_failure == "runtime_resolver"
        else "IMPORT_AUDIO"
    )
    controller, workflow, view, data_struct, mono, _warnings = (
        _config_update_failure_controller(failure="none", mode=mode)
    )
    old_model_config = controller.model.sequence_config
    old_workflow_snapshot = workflow.configuration_snapshot
    old_stimulus = data_struct.stimulus_data
    old_actions = view.capture_action_availability()
    view.combobox.addItem("new", "new.json")
    view.combobox.setCurrentIndex(view.combobox.findData("new.json"))
    durable = {"path": "old.json"}
    controller._using_path_updater = (
        lambda path: durable.__setitem__("path", path) or True
    )
    if presentation_failure == "reset":
        view.reset_runtime_action_buttons = lambda: (_ for _ in ()).throw(
            RuntimeError("reset failed")
        )
    elif presentation_failure == "missing_prompt":
        controller._config_loader = lambda _path: (1, None)
        view.present_missing_configuration_prompt = (
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("missing prompt failed")
            )
        )
    elif presentation_failure == "loaded_marker":
        view.mark_configuration_loaded = lambda: (_ for _ in ()).throw(
            RuntimeError("loaded marker failed")
        )
    elif presentation_failure == "runtime_resolver":
        controller._resolve_duplex_sample_rate = lambda *_args: (
            _ for _ in ()
        ).throw(RuntimeError("resolver failed"))
    elif presentation_failure == "mode_refresh":
        controller._refresh_test_mode_availability = lambda: (
            _ for _ in ()
        ).throw(RuntimeError("mode refresh failed"))
    elif presentation_failure == "channel_refresh":
        controller._refresh_channels = lambda: (_ for _ in ()).throw(
            RuntimeError("channel refresh failed")
        )
    elif presentation_failure == "availability":
        controller.present_configuration_availability = lambda _available: (
            _ for _ in ()
        ).throw(RuntimeError("availability failed"))
    elif presentation_failure == "focus":
        view.focus_after_selection = lambda: (_ for _ in ()).throw(
            RuntimeError("focus failed")
        )
    bus = SequenceEventBus()
    observed = []
    publication_calls = []

    def publish_twice(event):
        publication_calls.append(event)
        bus.events.configuration_changed.emit(event)
        bus.events.configuration_changed.emit(event)

    controller._publish_configuration = publish_twice
    bus.events.configuration_changed.connect(
        controller.handle_configuration_changed,
        Qt.QueuedConnection,
    )
    bus.events.configuration_changed.connect(observed.append, Qt.QueuedConnection)

    assert controller.on_using_file_combobox_changed("new") is None
    qapp.processEvents()
    qapp.processEvents()

    assert publication_calls == []
    assert observed == []
    _assert_queued_combobox_rollback(
        controller,
        workflow,
        view,
        data_struct,
        old_model_config=old_model_config,
        old_workflow_snapshot=old_workflow_snapshot,
        old_stimulus=old_stimulus,
        old_actions=old_actions,
        mono=mono,
        durable=durable,
        expected_actions=(
            (False, False, False)
            if presentation_failure
            in {"mode_refresh", "channel_refresh", "availability"}
            else None
        ),
    )
    rollback_failed = presentation_failure in {
        "mode_refresh",
        "channel_refresh",
        "availability",
    }
    assert controller.projection_consistent is not rollback_failed
    assert bool(controller.projection_failure) is rollback_failed


def test_late_failure_restores_successful_projections_without_durable_compensation(qapp):
    controller, workflow, view, data_struct, mono, _warnings = (
        _config_update_failure_controller(failure="none")
    )
    old_model_config = controller.model.sequence_config
    old_workflow_snapshot = workflow.configuration_snapshot
    old_stimulus = data_struct.stimulus_data
    old_actions = view.capture_action_availability()
    view._missing_config_prompt_enabled = True
    view._missing_config_prompted = True
    old_prompt_state = (True, True)
    view.combobox.addItem("new", "new.json")
    view.combobox.setCurrentIndex(view.combobox.findData("new.json"))
    analysis_modes = []
    mode_projections = []
    channel_projections = []
    controller._analysis_config_changed = lambda _config: analysis_modes.append(
        controller.model.acquisition_mode
    )
    controller._refresh_test_mode_availability = lambda: mode_projections.append(
        controller.model.acquisition_mode
    )
    controller._refresh_channels = lambda: channel_projections.append(
        controller.model.acquisition_mode
    )
    view.focus_after_selection = lambda: (_ for _ in ()).throw(
        RuntimeError("focus failed")
    )
    persistence_calls = []
    controller._using_path_updater = lambda path: persistence_calls.append(path) or True
    bus = SequenceEventBus()
    observed = []
    controller._publish_configuration = bus.events.configuration_changed.emit
    bus.events.configuration_changed.connect(observed.append, Qt.QueuedConnection)

    assert controller.on_using_file_combobox_changed("new") is None
    qapp.processEvents()

    assert persistence_calls == []
    assert observed == []
    assert analysis_modes == ["IMPORT_AUDIO", "RECORD_ONLY"]
    assert mode_projections == ["IMPORT_AUDIO", "RECORD_ONLY"]
    assert channel_projections == ["IMPORT_AUDIO", "RECORD_ONLY"]
    assert (
        view._missing_config_prompt_enabled,
        view._missing_config_prompted,
    ) == old_prompt_state
    _assert_queued_combobox_rollback(
        controller,
        workflow,
        view,
        data_struct,
        old_model_config=old_model_config,
        old_workflow_snapshot=old_workflow_snapshot,
        old_stimulus=old_stimulus,
        old_actions=old_actions,
        mono=mono,
        durable={"path": "old.json"},
    )


@pytest.mark.parametrize(
    "failure_boundary",
    ["analysis", "mode", "channels", "availability", "focus"],
)
def test_projection_journal_restores_partial_combobox_projection_in_old_model_context(
    failure_boundary,
):
    controller, workflow, view, data_struct, mono, _warnings = (
        _config_update_failure_controller(failure="none")
    )
    old_model_config = controller.model.sequence_config
    old_workflow_snapshot = workflow.configuration_snapshot
    old_stimulus = data_struct.stimulus_data
    old_actions = view.capture_action_availability()
    view.combobox.addItem("new", "new.json")
    view.combobox.setCurrentIndex(view.combobox.findData("new.json"))
    view.combobox.focused = True
    projected = {
        "analysis": "RECORD_ONLY",
        "mode": "RECORD_ONLY",
        "channels": "RECORD_ONLY",
        "availability": "RECORD_ONLY",
    }

    def project(name):
        projected[name] = controller.model.acquisition_mode
        if failure_boundary == name and projected[name] == "IMPORT_AUDIO":
            raise RuntimeError(f"{name} projection failed after mutation")

    controller._analysis_config_changed = lambda _config: project("analysis")
    controller._refresh_test_mode_availability = lambda: project("mode")
    controller._refresh_channels = lambda: project("channels")
    original_availability = controller.present_configuration_availability

    def present_availability(available):
        project("availability")
        original_availability(available)

    controller.present_configuration_availability = present_availability

    if failure_boundary == "focus":
        def fail_focus():
            view.combobox.clearFocus()
            raise RuntimeError("focus failed after mutation")

        view.focus_after_selection = fail_focus

    assert controller.on_using_file_combobox_changed("new") is None

    assert projected == {
        "analysis": "RECORD_ONLY",
        "mode": "RECORD_ONLY",
        "channels": "RECORD_ONLY",
        "availability": "RECORD_ONLY",
    }
    assert view.combobox.hasFocus() is True
    _assert_queued_combobox_rollback(
        controller,
        workflow,
        view,
        data_struct,
        old_model_config=old_model_config,
        old_workflow_snapshot=old_workflow_snapshot,
        old_stimulus=old_stimulus,
        old_actions=old_actions,
        mono=mono,
        durable={"path": "old.json"},
    )


def test_projection_journal_removes_focus_introduced_by_failed_callback():
    controller, _workflow, view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none")
    )
    view.combobox.addItem("new", "new.json")
    view.combobox.setCurrentIndex(view.combobox.findData("new.json"))
    view.combobox.focused = False

    def fail_after_focusing():
        view.combobox.setFocus()
        raise RuntimeError("focus failed after mutation")

    view.focus_after_selection = fail_after_focusing

    assert controller.on_using_file_combobox_changed("new") is None

    assert view.combobox.hasFocus() is False


def test_real_analysis_flag_projection_is_idempotent_and_rolls_back_exactly():
    controller, workflow, _view, data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="RECORD_ONLY")
    )
    analysis = {
        "display_sequence": ["fft"],
        "fft": {"type": "FFT"},
    }
    old_snapshot = ConfigurationSnapshot(
        sequence_config=_sequence("RECORD_ONLY", analysis=analysis),
        analysis_config=analysis,
        using_config_path="old.json",
    )
    assert controller.model.apply_configuration(old_snapshot, generation=4) is True
    data_struct.fft_flag = 9
    data_struct.stft_flag = 8
    controller._config_loader = lambda path: (
        (0, _sequence("RECORD_ONLY", analysis=analysis))
        if path == "new.json"
        else pytest.fail(f"unexpected path: {path}")
    )

    late_failure = RuntimeError("late analysis presentation sentinel")

    def fail_late(_analysis_config):
        raise late_failure

    controller._analysis_config_changed = fail_late
    published = []
    controller._publish_configuration = published.append

    assert controller.on_sequence_config_updated() is False

    assert data_struct.fft_flag == 9
    assert data_struct.stft_flag == 8
    assert workflow.configuration_generation == 4
    assert published == []

    controller._analysis_config_changed = lambda _analysis_config: None

    assert controller.on_sequence_config_updated() is True
    assert data_struct.fft_flag == 1
    assert data_struct.stft_flag == 0

    assert controller.on_sequence_config_updated() is True
    assert data_struct.fft_flag == 1
    assert data_struct.stft_flag == 0


def _config_update_failure_controller(*, failure, mode="IMPORT_AUDIO"):
    old_path = "old.json"
    new_path = "new.json"
    mono = np.ones(4)
    data_struct = SimpleNamespace(
        store_wave_data=mono,
        store_wave_data_multi=mono.reshape(-1, 1),
        sample_rate=32000,
        audio_lenth=4,
        stimulus_data=np.ones(4),
        stimulus_info={"sample_rate": 32000, "total_time": 0.000125},
        wav_calibration_metadata={"source": "old.wav"},
        wav_calibration_metadata_authoritative=True,
        wav_calibration_warning_shown=True,
    )
    if failure == "clear_data":
        data_struct.clear_data = lambda: (_ for _ in ()).throw(
            RuntimeError("clear failed")
        )
    else:
        data_struct.clear_data = lambda: (
            setattr(data_struct, "store_wave_data", None),
            setattr(data_struct, "store_wave_data_multi", None),
        )
    workflow = SequenceWorkflowModel()
    model = SequenceConfigurationModel(
        data_struct=data_struct,
        workflow_model=workflow,
    )
    old_config = _sequence("RECORD_ONLY")
    model.apply_configuration(
        ConfigurationSnapshot(
            sequence_config=old_config,
            analysis_config={"display_sequence": []},
            using_config_path=old_path,
        ),
        generation=4,
    )
    model.replace_registry(
        {"old": old_path, "using_config_path": old_path},
        using_config_path=old_path,
        entries=(("old", old_path),),
    )
    combo = _ComboBox()
    player = _Button(True)
    replay = _Button(True)
    data = _Button(True)
    view = SequenceConfigurationView(
        combobox=combo,
        player_button=player,
        replay_button=replay,
        data_button=data,
    )
    view.populate_configuration_entries(
        model.registry_entries,
        using_config_path=old_path,
        clear_first=True,
    )
    warnings = []
    durable_path = {"value": old_path}

    def persist_path(path):
        durable_path["value"] = path
        return True

    def restore_path(path):
        durable_path["value"] = path
        return True

    def availability_changed():
        if failure == "presentation":
            raise RuntimeError("presentation failed")
        player.setEnabled(True)

    controller = _controller(
        model=model,
        view=view,
        registry_loader=lambda: {
            "new": new_path,
            "using_config_path": new_path,
        },
        config_loader=lambda path: (
            (0, _sequence(mode))
            if path == new_path
            else pytest.fail(f"unexpected path: {path}")
        ),
        using_path_updater=persist_path,
        using_path_persistence_snapshotter=lambda: durable_path["value"],
        using_path_persistence_restorer=restore_path,
        ok_code=0,
        path_exists=lambda _path: True,
        input_sample_rate_resolver=lambda _mic: None,
        duplex_sample_rate_resolver=lambda _mic, _speaker: None,
        availability_changed=availability_changed,
        logger=SimpleNamespace(warning=warnings.append),
    )
    return controller, workflow, view, data_struct, mono, warnings


def test_config_update_does_not_invoke_injected_clear_data_callback():
    controller, workflow, view, data_struct, mono, warnings = (
        _config_update_failure_controller(failure="clear_data")
    )
    original_identity = data_struct

    result = controller.on_sequence_config_updated()

    assert result is True
    assert controller.model.data_struct is original_identity
    assert controller.model.acquisition_mode == "IMPORT_AUDIO"
    assert controller.model.using_config_path == "new.json"
    assert controller.model.configuration_generation == 5
    assert workflow.configuration_generation == 5
    assert data_struct.store_wave_data is None
    assert view.combobox.currentData() == "new.json"
    assert (
        view.player_button.enabled,
        view.replay_button.enabled,
        view.data_button.enabled,
    ) == (True, False, False)
    assert not any("clear failed" in message for message in warnings)


def test_config_update_fallback_uses_explicit_controller_owned_cleanup():
    controller, workflow, view, data_struct, mono, _warnings = (
        _config_update_failure_controller(failure="clear_data")
    )
    persistence_calls = []
    registry = {
        "new": "new.json",
        "using_config_path": "missing.json",
    }
    controller._registry_loader = lambda: deepcopy(registry)
    controller._path_exists = lambda path: path == "new.json"
    controller._using_path_updater = lambda path: (
        persistence_calls.append(path),
        registry.__setitem__("using_config_path", path),
        True,
    )[-1]

    result = controller.on_sequence_config_updated()

    assert result is True
    assert persistence_calls == ["new.json"]
    assert controller.model.acquisition_mode == "IMPORT_AUDIO"
    assert controller.model.using_config_path == "new.json"
    assert controller.model.configuration_generation == 5
    assert workflow.configuration_generation == 5
    assert data_struct.store_wave_data is None
    assert view.combobox.currentData() == "new.json"


def test_config_update_candidate_dependent_projections_see_candidate_mode():
    controller, _workflow, view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none")
    )
    projected_modes = []
    controller._refresh_channels = lambda: projected_modes.append(
        ("channels", controller.model.acquisition_mode)
    )
    controller._refresh_test_mode_availability = lambda: projected_modes.append(
        ("mode", controller.model.acquisition_mode)
    )
    original_availability = controller.present_configuration_availability

    def present_availability(available):
        projected_modes.append(("availability", controller.model.acquisition_mode))
        original_availability(available)

    controller.present_configuration_availability = present_availability

    assert controller.on_sequence_config_updated() is True

    assert projected_modes == [
        ("channels", "IMPORT_AUDIO"),
        ("mode", "IMPORT_AUDIO"),
        ("availability", "IMPORT_AUDIO"),
    ]


def test_config_update_path_probe_exception_is_contained_before_transaction_mutation():
    controller, workflow, view, data_struct, mono, warnings = (
        _config_update_failure_controller(failure="none")
    )
    old_model_config = controller.model.sequence_config
    old_workflow_snapshot = workflow.configuration_snapshot
    old_actions = view.capture_action_availability()
    controller._path_exists = lambda _path: (_ for _ in ()).throw(
        OSError("path probe failed")
    )
    controller._config_loader = lambda _path: (1, None)
    published = []
    controller._publish_configuration = published.append

    assert controller.on_sequence_config_updated() is False

    assert published == []
    assert controller.model.sequence_config is old_model_config
    assert workflow.configuration_snapshot is old_workflow_snapshot
    assert data_struct.store_wave_data is mono
    assert view.combobox.currentData() == "old.json"
    assert view.capture_action_availability() == old_actions
    assert any("probe sequence config path" in message for message in warnings)


@pytest.mark.parametrize("late_failure", ["saver", "persistence", "loaded_marker"])
def test_import_mode_identity_and_plot_survive_late_configuration_update_failure(
    late_failure,
):
    controller, workflow, view, data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="RECORD_ONLY")
    )
    old_config = _sequence("IMPORT_AUDIO")
    controller.model.apply_configuration(
        ConfigurationSnapshot(
            sequence_config=old_config,
            analysis_config={"display_sequence": []},
            using_config_path="old.json",
        ),
        generation=4,
    )
    old_workflow_snapshot = workflow.configuration_snapshot
    recorded_identity = {
        "path": "old.wav",
        "signal_info": {"sample_rate": 32_000},
    }
    plot = ["old-curve"]

    def clear_identity():
        recorded_identity["path"] = None
        recorded_identity["signal_info"] = {}

    controller._clear_import_identity = clear_identity
    controller._clear_plot = plot.clear
    if late_failure == "saver":
        controller._prepare_stimulus_config = lambda *_args: (True, True)
        controller._config_saver = lambda *_args: False
    elif late_failure == "persistence":
        controller._registry_loader = lambda: {
            "new": "new.json",
            "using_config_path": "missing.json",
        }
        controller._path_exists = lambda path: path == "new.json"
        controller._using_path_updater = lambda _path: False
        controller._using_path_persistence_snapshotter = lambda: "missing.json"
        controller._using_path_persistence_restorer = lambda _state: True
    else:
        view.mark_configuration_loaded = lambda: (_ for _ in ()).throw(
            RuntimeError("loaded marker failed")
        )

    assert controller.on_sequence_config_updated() is False

    assert recorded_identity == {
        "path": "old.wav",
        "signal_info": {"sample_rate": 32_000},
    }
    assert plot == ["old-curve"]
    assert controller.model.acquisition_mode == "IMPORT_AUDIO"
    assert controller.model.configuration_generation == 4
    assert workflow.configuration_snapshot is old_workflow_snapshot
    assert data_struct.store_wave_data is not None


def test_config_update_late_failure_does_not_write_regenerated_candidate():
    controller, workflow, view, data_struct, mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    controller._resolve_duplex_sample_rate = lambda _mic, _speaker: SimpleNamespace(
        ok=True,
        sample_rate=48_000,
        message="",
    )

    def regenerate(data, detail, **_kwargs):
        detail["stimulus_path"] = "generated.wav"
        data.stimulus_data = np.ones(16)
        return True

    controller._stimulus_setter = regenerate
    saved = []
    controller._config_saver = lambda config, path: saved.append((config, path)) or True
    published = []
    controller._publish_configuration = published.append
    old_actions = view.capture_action_availability()
    old_presentation = view.capture_configuration_presentation_state()
    view.mark_configuration_loaded = lambda: (_ for _ in ()).throw(
        RuntimeError("loaded marker failed")
    )

    assert controller.on_sequence_config_updated() is False

    assert saved == []
    assert published == []
    assert controller.model.acquisition_mode == "RECORD_ONLY"
    assert controller.model.using_config_path == "old.json"
    assert controller.model.configuration_generation == 4
    assert workflow.configuration_generation == 4
    assert data_struct.store_wave_data is mono
    assert view.combobox.currentData() == "old.json"
    assert view.capture_action_availability() == old_actions
    assert view.capture_configuration_presentation_state() == old_presentation


def test_config_update_publish_failure_keeps_final_commit_and_persisted_fallback():
    controller, workflow, view, data_struct, mono, warnings = (
        _config_update_failure_controller(failure="none")
    )
    persistence_calls = []
    registry = {
        "new": "new.json",
        "using_config_path": "missing.json",
    }
    controller._registry_loader = lambda: deepcopy(registry)
    controller._path_exists = lambda path: path == "new.json"
    controller._using_path_updater = lambda path: (
        persistence_calls.append(path),
        registry.__setitem__("using_config_path", path),
        True,
    )[-1]
    controller._publish_configuration = lambda _event: (_ for _ in ()).throw(
        RuntimeError("publish failed")
    )

    result = controller.on_sequence_config_updated()

    assert result is True
    assert persistence_calls == ["new.json"]
    assert controller.model.acquisition_mode == "IMPORT_AUDIO"
    assert controller.model.using_config_path == "new.json"
    assert controller.model.configuration_generation == 5
    assert workflow.configuration_generation == 5
    assert data_struct.store_wave_data is None
    assert view.combobox.currentData() == "new.json"
    assert (
        view.player_button.enabled,
        view.replay_button.enabled,
        view.data_button.enabled,
    ) == (True, False, False)
    assert any("publish sequence configuration" in message for message in warnings)


def test_config_update_publisher_rejection_does_not_compensate_durable_path():
    controller, workflow, view, _data_struct, _mono, warnings = (
        _config_update_failure_controller(failure="none")
    )
    persistence_calls = []
    registry = {
        "new": "new.json",
        "using_config_path": "missing.json",
    }
    controller._registry_loader = lambda: deepcopy(registry)
    controller._path_exists = lambda path: path == "new.json"

    def update_path(path):
        persistence_calls.append(path)
        registry["using_config_path"] = path
        return True

    controller._using_path_updater = update_path
    controller._publish_configuration = lambda _event: False

    result = controller.on_sequence_config_updated()

    assert result is True
    assert persistence_calls == ["new.json"]
    assert controller.model.using_config_path == "new.json"
    assert controller.model.configuration_generation == 5
    assert workflow.configuration_generation == 5
    assert view.combobox.currentData() == "new.json"
    assert not any("restore prior persisted" in message for message in warnings)


def test_config_update_presentation_failure_restores_projection_and_returns_false():
    controller, workflow, view, data_struct, mono, warnings = (
        _config_update_failure_controller(failure="presentation")
    )

    result = controller.on_sequence_config_updated()

    assert result is False
    assert controller.model.acquisition_mode == "RECORD_ONLY"
    assert controller.model.using_config_path == "old.json"
    assert controller.model.configuration_generation == 4
    assert workflow.configuration_generation == 4
    assert (
        workflow.configuration_snapshot.sequence_config[0]["seq1"]["acq"]["mode"]
        == "RECORD_ONLY"
    )
    assert data_struct.store_wave_data is mono
    assert view.combobox.currentData() == "old.json"
    assert (
        view.player_button.enabled,
        view.replay_button.enabled,
        view.data_button.enabled,
    ) == (False, False, False)
    assert controller.projection_consistent is False
    assert controller.projection_failure
    assert any("present configuration availability" in message for message in warnings)


def test_config_update_success_commits_once_and_returns_true():
    controller, workflow, view, data_struct, _mono, warnings = (
        _config_update_failure_controller(failure="none")
    )

    result = controller.on_sequence_config_updated()

    assert result is True
    assert controller.model.acquisition_mode == "IMPORT_AUDIO"
    assert controller.model.using_config_path == "new.json"
    assert controller.model.configuration_generation == 5
    assert workflow.configuration_generation == 5
    assert data_struct.store_wave_data is None
    assert data_struct.store_wave_data_multi is None
    assert (
        view.player_button.enabled,
        view.replay_button.enabled,
        view.data_button.enabled,
    ) == (True, False, False)
    assert controller.projection_consistent is True
    assert controller.projection_failure is None
    assert warnings == []


def test_config_update_reference_failure_restores_runtime_and_generation():
    path = "import.json"
    old_detail = {"stimulus_info": {"start_freq": 100}}
    new_detail = {"stimulus_info": {"start_freq": 200}}
    mono = np.ones(4)
    reference = np.full(4, 2.0)
    data_struct = SimpleNamespace(
        store_wave_data=mono,
        store_wave_data_multi=mono.reshape(-1, 1),
        sample_rate=32000,
        audio_lenth=4,
        stimulus_data=reference,
        stimulus_info={"sample_rate": 32000, "total_time": 0.000125},
    )
    workflow = SequenceWorkflowModel()
    model = SequenceConfigurationModel(
        data_struct=data_struct,
        workflow_model=workflow,
    )
    model.apply_configuration(
        ConfigurationSnapshot(
            sequence_config=_sequence("IMPORT_STIMULUS_AUDIO", old_detail),
            analysis_config={},
            using_config_path=path,
        ),
        generation=7,
    )
    model.replace_registry(
        {"import": path, "using_config_path": path},
        using_config_path=path,
        entries=(("import", path),),
    )
    player = _Button(True)
    replay = _Button(True)
    data = _Button(True)
    view = SequenceConfigurationView(
        player_button=player,
        replay_button=replay,
        data_button=data,
    )
    warnings = []
    controller = _controller(
        model=model,
        view=view,
        registry_loader=lambda: {"import": path, "using_config_path": path},
        config_loader=lambda _path: (
            0,
            _sequence("IMPORT_STIMULUS_AUDIO", new_detail),
        ),
        analysis_reference_setter=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("reference failed")
        ),
        using_path_updater=lambda _path: True,
        ok_code=0,
        path_exists=lambda _path: True,
        warning=lambda _title, _message: None,
        logger=SimpleNamespace(warning=warnings.append),
    )

    result = controller.on_sequence_config_updated()

    assert result is False
    assert controller.model.sequence_config == _sequence(
        "IMPORT_STIMULUS_AUDIO", old_detail
    )
    assert controller.model.configuration_generation == 7
    assert workflow.configuration_generation == 7
    assert data_struct.store_wave_data is mono
    assert data_struct.stimulus_data is reference
    assert data_struct.stimulus_info == {
        "sample_rate": 32000,
        "total_time": 0.000125,
    }
    assert (player.enabled, replay.enabled, data.enabled) == (True, True, True)
    assert any("refresh import stimulus reference" in message for message in warnings)


def test_missing_config_prompt_is_deferred_and_only_shown_on_first_eligible_show():
    warnings = []
    view = SequenceConfigurationView(
        warning=lambda parent, title, text: warnings.append((parent, title, text)),
        parent="window",
    )

    assert view.present_missing_configuration_prompt([], eligible=False) is False
    assert view.present_missing_configuration_prompt([], eligible=True) is True
    assert view.present_missing_configuration_prompt([], eligible=True) is False
    assert len(warnings) == 1
    assert warnings[0][0:2] == ("window", "提示")
    assert "当前未找到可用配置文件" in warnings[0][2]


def test_mode_availability_and_audio_device_state_are_model_owned():
    updates = []
    model = SequenceConfigurationModel(data_struct=SimpleNamespace())
    controller = _controller(model=model, availability_changed=lambda: updates.append(True))
    model.acquisition_mode = "IMPORT_AUDIO"

    assert controller.is_mode_available_for_external_trigger() is False
    model.acquisition_mode = "PLAY_AND_RECORD"
    assert controller.is_mode_available_for_external_trigger() is True

    assert controller.set_audio_devices_available(False, "device missing") is None
    assert model.audio_devices_available is False
    assert model.audio_devices_unavailable_message == "device missing"
    assert updates == [True]


def test_runtime_sample_rate_resolution_uses_mode_authority():
    input_calls = []
    duplex_calls = []
    controller = _controller(
        input_sample_rate_resolver=lambda mic: input_calls.append(mic) or "input-rate",
        duplex_sample_rate_resolver=lambda mic, speaker: duplex_calls.append(
            (mic, speaker)
        )
        or "duplex-rate",
    )
    controller.model.mic = {"name": "mic"}
    controller.model.speaker = {"name": "speaker"}

    assert controller.resolve_runtime_sample_rate("RECORD_ONLY", {}) == "input-rate"
    assert (
        controller.resolve_runtime_sample_rate(
            "RECORD_ONLY", {"monitor_playback": True}
        )
        == "duplex-rate"
    )
    assert controller.resolve_runtime_sample_rate("PLAY_AND_RECORD", {}) == "duplex-rate"
    assert controller.resolve_runtime_sample_rate("IMPORT_AUDIO", {}) is None
    assert len(input_calls) == 1
    assert len(duplex_calls) == 2


def test_import_reference_refresh_commits_atomically_at_recording_rate():
    data_struct = SimpleNamespace(
        sample_rate=32000,
        audio_lenth=320,
        store_wave_data=np.ones(320),
        store_wave_data_multi=np.ones((320, 1)),
        stimulus_data=np.ones(4),
        stimulus_info={"sample_rate": 32000},
        alignment_sample_count=4,
    )
    data_button = _Button(False)

    def build_reference(staged, _detail, **kwargs):
        assert kwargs["runtime_sample_rate"] == 32000
        staged.stimulus_data = np.full(320, 2.0)
        staged.stimulus_info = {"sample_rate": 32000, "total_time": 0.01}
        staged.alignment_sample_count = 320
        return True

    model = SequenceConfigurationModel(data_struct=data_struct)
    controller = _controller(
        model=model,
        analysis_reference_setter=build_reference,
        data_enabled_setter=data_button.setEnabled,
    )

    assert controller.refresh_import_stimulus_analysis_reference({}) is True
    assert data_struct.sample_rate == 32000
    assert np.array_equal(data_struct.stimulus_data, np.full(320, 2.0))
    assert data_struct.stimulus_info["sample_rate"] == 32000
    assert data_struct.alignment_sample_count == 320
    assert model.stimulus_reference_ready is True
    assert data_button.enabled is True


def test_failed_import_reference_refresh_clears_only_reference_state():
    mono = np.ones(320)
    data_struct = SimpleNamespace(
        sample_rate=32000,
        audio_lenth=320,
        store_wave_data=mono,
        store_wave_data_multi=mono.reshape(-1, 1),
        stimulus_data=np.ones(4),
        stimulus_info={"sample_rate": 32000},
        alignment_sample_count=4,
    )
    warnings = []
    model = SequenceConfigurationModel(data_struct=data_struct)
    controller = _controller(
        model=model,
        analysis_reference_setter=lambda *_args, **_kwargs: False,
        warning=lambda title, message: warnings.append((title, message)),
    )

    assert controller.refresh_import_stimulus_analysis_reference({}) is False
    assert data_struct.sample_rate == 32000
    assert data_struct.store_wave_data is mono
    assert data_struct.stimulus_data is None
    assert data_struct.stimulus_info is None
    assert not hasattr(data_struct, "alignment_sample_count")
    assert model.stimulus_reference_ready is False
    assert warnings == [("提示", "加载分析参考激励失败，请检查激励配置。")]


def _assert_nested_runtime_state_restored(data_struct, expected):
    assert data_struct.nested == expected["nested"]
    assert np.array_equal(data_struct.runtime_array, expected["runtime_array"])
    assert np.array_equal(
        data_struct.stimulus_info["calibration"]["values"],
        expected["stimulus_values"],
    )
    assert not hasattr(data_struct, "partial_field")


@pytest.mark.parametrize("failure_kind", ["false", "raise"])
def test_injected_clear_data_cannot_mutate_unrelated_runtime(
    failure_kind,
):
    controller, _workflow, _view, data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none")
    )
    data_struct.nested = {"channels": [{"gain": 1.25}]}
    data_struct.runtime_array = np.array([1.0, 2.0, 3.0])
    data_struct.stimulus_info = {
        "sample_rate": 32000,
        "calibration": {"values": np.array([4.0, 5.0])},
    }
    expected = {
        "nested": deepcopy(data_struct.nested),
        "runtime_array": data_struct.runtime_array.copy(),
        "stimulus_values": data_struct.stimulus_info["calibration"]["values"].copy(),
    }
    stable_identity = data_struct

    def mutate_then_fail():
        data_struct.nested["channels"][0]["gain"] = 99.0
        data_struct.runtime_array[:] = -1
        data_struct.stimulus_info["calibration"]["values"][:] = -2
        data_struct.partial_field = ["leaked"]
        if failure_kind == "false":
            return False
        raise RuntimeError("clear failed after nested mutation")

    data_struct.clear_data = mutate_then_fail

    assert controller.on_sequence_config_updated() is True

    assert controller.model.data_struct is stable_identity
    assert data_struct.nested == expected["nested"]
    assert np.array_equal(data_struct.runtime_array, expected["runtime_array"])
    assert not hasattr(data_struct, "partial_field")


def test_stimulus_setter_is_staged_and_partial_raise_cannot_touch_live_runtime():
    controller, workflow, _view, data_struct, _mono, warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    controller._resolve_duplex_sample_rate = lambda *_args: SimpleNamespace(
        ok=True, sample_rate=48000, message=""
    )
    data_struct.nested = {"values": [1, {"array": np.array([2.0, 3.0])}]}
    nested = data_struct.nested
    expected_nested = deepcopy(nested)
    old_stimulus_data = data_struct.stimulus_data
    old_stimulus_info = data_struct.stimulus_info
    old_stimulus_data_contents = old_stimulus_data.copy()
    old_stimulus_info_contents = deepcopy(old_stimulus_info)
    data_struct.alignment_sample_count = 17
    stable_identity = data_struct
    sentinel_reached = []
    published = []
    controller._publish_configuration = published.append

    class StagedMutationSentinel(RuntimeError):
        pass

    sentinel = StagedMutationSentinel("owned stimulus staging sentinel")

    def mutate_staged_then_raise(staged, detail, **_kwargs):
        assert staged is not data_struct
        assert set(vars(staged)) == {
            "stimulus_data",
            "stimulus_info",
            "sample_rate",
            "alignment_sample_count",
        }
        staged.stimulus_data = np.array([100.0, 200.0])
        staged.stimulus_info = {"candidate": {"values": [1]}}
        staged.stimulus_info["candidate"]["values"].append(2)
        staged.sample_rate = 96000
        staged.alignment_sample_count = 2
        detail["candidate_mutation"] = True
        sentinel_reached.append(sentinel)
        raise sentinel

    controller._stimulus_setter = mutate_staged_then_raise

    assert controller.on_sequence_config_updated() is False

    assert sentinel_reached == [sentinel]
    assert controller.model.data_struct is stable_identity
    assert data_struct.nested is nested
    assert data_struct.nested["values"][0] == expected_nested["values"][0]
    assert np.array_equal(
        data_struct.nested["values"][1]["array"],
        expected_nested["values"][1]["array"],
    )
    assert data_struct.stimulus_data is old_stimulus_data
    assert np.array_equal(data_struct.stimulus_data, old_stimulus_data_contents)
    assert data_struct.stimulus_info is old_stimulus_info
    assert data_struct.stimulus_info == old_stimulus_info_contents
    assert data_struct.sample_rate == 32000
    assert data_struct.alignment_sample_count == 17
    assert workflow.configuration_generation == 4
    assert published == []
    assert any(str(sentinel) in message for message in warnings)


def test_successful_staged_stimulus_commit_preserves_data_struct_identity():
    controller, _workflow, _view, data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    controller._resolve_duplex_sample_rate = lambda *_args: SimpleNamespace(
        ok=True, sample_rate=48000, message=""
    )
    stable_identity = data_struct
    callback_targets = []

    def prepare(staged, _detail, **_kwargs):
        callback_targets.append(staged)
        staged.stimulus_data = np.arange(4.0)
        staged.stimulus_info = {"nested": {"rates": [48000]}}
        return False

    controller._stimulus_setter = prepare

    assert controller.on_sequence_config_updated() is True

    assert controller.model.data_struct is stable_identity
    assert len(callback_targets) == 1
    assert callback_targets[0] is not stable_identity
    assert np.array_equal(data_struct.stimulus_data, np.arange(4.0))
    assert data_struct.stimulus_info == {"nested": {"rates": [48000]}}


class _MutableBox:
    def __init__(self, values):
        self.values = values

    def __call__(self):
        return self.values


class _DictSubclass(dict):
    pass


class _ListSubclass(list):
    pass


def test_stimulus_staging_isolates_custom_mutable_object_before_raise():
    controller, _workflow, _view, data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    controller._resolve_duplex_sample_rate = lambda *_args: SimpleNamespace(
        ok=True, sample_rate=48000, message=""
    )
    retained_values = [1, 2]
    retained_box = _MutableBox(retained_values)
    data_struct.custom_runtime = retained_box
    callback_targets = []

    def prepare_owned_only(staged, _detail, **_kwargs):
        callback_targets.append(hasattr(staged, "custom_runtime"))
        staged.stimulus_data = np.ones(2)
        staged.stimulus_info = {"sample_rate": 48000}
        return False

    controller._stimulus_setter = prepare_owned_only

    assert controller.on_sequence_config_updated() is True

    assert callback_targets == [False]
    assert data_struct.custom_runtime is retained_box
    assert retained_box.values is retained_values
    assert retained_values == [1, 2]


def test_controller_owned_cleanup_does_not_inspect_unrelated_runtime_graphs():
    controller, _workflow, _view, data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none")
    )
    shared = _ListSubclass([{"value": 1}])
    mapping = _DictSubclass(first=shared, second=shared)
    shared.marker = {"state": "old"}
    mapping.marker = shared.marker
    mapping["cycle"] = mapping
    base = np.arange(8.0)
    view = base[2:6]
    data_struct.graph = mapping
    data_struct.shared = shared
    data_struct.array_view = view
    retained_nested = shared[0]
    callback_calls = []

    def mutate_then_raise():
        callback_calls.append(True)
        shared[0]["value"] = 99
        shared.append({"new": True})
        shared.marker["state"] = "mutated"
        mapping["added"] = "leak"
        mapping.marker = {"state": "replaced"}
        base[:] = -1
        view[:] = -2
        raise RuntimeError("graph mutation failed")

    data_struct.clear_data = mutate_then_raise

    assert controller.on_sequence_config_updated() is True

    assert callback_calls == []
    assert data_struct.graph is mapping
    assert type(mapping) is _DictSubclass
    assert mapping["cycle"] is mapping
    assert mapping["first"] is shared
    assert mapping["second"] is shared
    assert data_struct.shared is shared
    assert type(shared) is _ListSubclass
    assert shared == [{"value": 1}]
    assert mapping.marker is shared.marker
    assert shared.marker == {"state": "old"}
    assert shared[0] is retained_nested
    assert retained_nested == {"value": 1}
    assert data_struct.array_view is view
    assert np.shares_memory(view, base)
    assert np.array_equal(base, np.arange(8.0))
    assert np.array_equal(view, np.arange(8.0)[2:6])


def test_uncopyable_opaque_runtime_field_is_ignored_by_bounded_journal():
    controller, workflow, view, data_struct, mono, warnings = (
        _config_update_failure_controller(failure="none")
    )
    callback_calls = []
    data_struct.opaque_lock = threading.Lock()
    data_struct.clear_data = lambda: callback_calls.append(True)
    old_snapshot = workflow.configuration_snapshot
    old_actions = view.capture_action_availability()

    assert controller.on_sequence_config_updated() is True

    assert callback_calls == []
    assert data_struct.store_wave_data is None
    assert workflow.configuration_snapshot is not old_snapshot
    assert not any("capture configuration transaction state" in message for message in warnings)


def test_object_ndarray_unrelated_graph_is_never_mutated_or_copied():
    controller, _workflow, _view, data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none")
    )
    nested = {"values": [1, {"gain": 2.5}]}
    array_only = {"hidden": [3]}
    custom_box = _MutableBox(["old"])
    base = np.empty(5, dtype=object)
    cycle = np.empty(1, dtype=object)
    cycle[0] = cycle
    base[:] = [nested, nested, cycle, array_only, custom_box]
    view = base[1:3]
    retained_container = [nested, view]
    data_struct.object_view = view
    data_struct.object_base = base
    data_struct.object_cycle = cycle
    data_struct.object_container = retained_container

    def mutate_then_fail():
        nested["values"][1]["gain"] = 99.0
        base[0] = {"replacement": True}
        view[0] = {"view": "replacement"}
        cycle[0] = None
        array_only["hidden"].append(4)
        custom_box.values.append("mutated")
        return False

    data_struct.clear_data = mutate_then_fail

    assert controller.on_sequence_config_updated() is True

    assert data_struct.object_base is base
    assert data_struct.object_view is view
    assert data_struct.object_cycle is cycle
    assert data_struct.object_container is retained_container
    assert np.shares_memory(base, view)
    assert base[0] is nested
    assert base[1] is nested
    assert view[0] is nested
    assert retained_container[0] is nested
    assert retained_container[1] is view
    assert nested == {"values": [1, {"gain": 2.5}]}
    assert cycle[0] is cycle
    assert base[3] is array_only
    assert array_only == {"hidden": [3]}
    assert base[4] is custom_box
    assert custom_box.values == ["old"]


def test_owned_object_ndarray_staging_failure_cannot_leak_into_live_runtime():
    controller, workflow, _view, data_struct, _mono, warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    controller._resolve_duplex_sample_rate = lambda *_args: SimpleNamespace(
        ok=True, sample_rate=48000, message=""
    )
    shared = {"values": [1]}
    base = np.empty(3, dtype=object)
    base[0] = shared
    base[1] = base
    base[2] = {"tail": True}
    view = base[:2]
    container = [shared, view]
    data_struct.object_view = view
    data_struct.object_base = base
    data_struct.object_container = container
    sentinel_reached = []
    published = []
    controller._publish_configuration = published.append

    class ObjectArrayStagingSentinel(RuntimeError):
        pass

    sentinel = ObjectArrayStagingSentinel("owned object-array staging sentinel")

    def mutate_staged_then_raise(staged, _detail, **_kwargs):
        owned = np.empty(2, dtype=object)
        owned[0] = owned
        owned[1] = {"values": [1]}
        staged.stimulus_data = owned
        staged.stimulus_info = {"same": owned}
        staged.stimulus_info["same"][1]["values"].append(2)
        staged.sample_rate = 96000
        staged.alignment_sample_count = 2
        sentinel_reached.append(sentinel)
        raise sentinel

    controller._stimulus_setter = mutate_staged_then_raise

    assert controller.on_sequence_config_updated() is False

    assert sentinel_reached == [sentinel]
    assert base[0] is shared
    assert base[1] is base
    assert shared == {"values": [1]}
    assert container[0] is shared
    assert container[1] is view
    assert np.shares_memory(base, view)
    assert workflow.configuration_generation == 4
    assert published == []
    assert any(str(sentinel) in message for message in warnings)


def test_lock_nested_in_unrelated_object_ndarray_is_not_inspected():
    controller, _workflow, _view, data_struct, _mono, warnings = (
        _config_update_failure_controller(failure="none")
    )
    opaque = np.empty(1, dtype=object)
    opaque[0] = threading.Lock()
    data_struct.object_array = opaque
    callback_calls = []
    data_struct.clear_data = lambda: callback_calls.append(True)

    assert controller.on_sequence_config_updated() is True

    assert callback_calls == []
    assert data_struct.object_array is opaque
    assert not any("capture configuration transaction state" in message for message in warnings)


def test_successful_object_ndarray_commit_keeps_owned_aliases_only():
    controller, _workflow, _view, data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    controller._resolve_duplex_sample_rate = lambda *_args: SimpleNamespace(
        ok=True, sample_rate=48000, message=""
    )
    unrelated = _MutableBox(["keep"])
    data_struct.unrelated_runtime = unrelated

    def prepare(staged, _detail, **_kwargs):
        owned = np.empty(1, dtype=object)
        owned[0] = owned
        staged.stimulus_data = owned
        staged.stimulus_info = {"same": owned}
        return False

    controller._stimulus_setter = prepare

    assert controller.on_sequence_config_updated() is True

    committed = data_struct.stimulus_data
    assert committed[0] is committed
    assert data_struct.stimulus_info["same"] is committed
    assert data_struct.unrelated_runtime is unrelated
    assert unrelated.values == ["keep"]


def test_successful_stimulus_commit_preserves_unrelated_runtime_root_identity_and_type():
    controller, _workflow, _view, data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    controller._resolve_duplex_sample_rate = lambda *_args: SimpleNamespace(
        ok=True, sample_rate=48000, message=""
    )
    controller.model.apply_configuration(
        ConfigurationSnapshot(
            sequence_config=_sequence("PLAY_AND_RECORD"),
            analysis_config={"display_sequence": []},
            using_config_path="old.json",
        ),
        generation=4,
    )
    wave = data_struct.store_wave_data
    analysis_nested = _ListSubclass([{"ok": True}])
    analysis_results = _DictSubclass(result=analysis_nested)
    custom = _MutableBox(["keep"])
    data_struct.analysis_result_dict = analysis_results
    data_struct.custom_runtime = custom

    def prepare(staged, _detail, **_kwargs):
        staged.stimulus_data = np.ones(6)
        staged.stimulus_info = {"sample_rate": 48000}
        return False

    controller._stimulus_setter = prepare

    assert controller.on_sequence_config_updated() is True

    assert data_struct.store_wave_data is wave
    assert data_struct.analysis_result_dict is analysis_results
    assert type(data_struct.analysis_result_dict) is _DictSubclass
    assert data_struct.analysis_result_dict == {"result": analysis_nested}
    assert data_struct.analysis_result_dict["result"] is analysis_nested
    assert data_struct.custom_runtime is custom
    assert custom.values == ["keep"]


def test_projection_hook_half_pair_fails_closed_before_projection_mutation():
    controller, workflow, view, _data_struct, _mono, warnings = (
        _config_update_failure_controller(failure="none", mode="RECORD_ONLY")
    )
    controller.model.apply_configuration(
        ConfigurationSnapshot(
            sequence_config=_sequence("IMPORT_AUDIO"),
            analysis_config={},
            using_config_path="old.json",
        ),
        generation=4,
    )
    identity = {"path": "old.wav"}
    controller._clear_import_identity = lambda: identity.__setitem__("path", None)
    controller._import_identity_state_capturer = lambda: deepcopy(identity)
    controller._import_identity_state_restorer = None
    published = []
    controller._publish_configuration = published.append

    assert controller.on_sequence_config_updated() is False

    assert identity == {"path": "old.wav"}
    assert controller.projection_consistent is False
    assert workflow.configuration_generation == 4
    assert view.player_button.enabled is False
    assert view.replay_button.enabled is False
    assert view.data_button.enabled is False
    assert published == []
    assert any("projection hooks" in message for message in warnings)


def test_mutator_without_hooks_is_rejected_before_any_transaction_mutation():
    controller, workflow, view, _data_struct, _mono, warnings = (
        _config_update_failure_controller(failure="none", mode="RECORD_ONLY")
    )
    controller.model.apply_configuration(
        ConfigurationSnapshot(
            sequence_config=_sequence("IMPORT_AUDIO"),
            analysis_config={},
            using_config_path="old.json",
        ),
        generation=4,
    )
    old_wave = controller.model.data_struct.store_wave_data
    old_snapshot = workflow.configuration_snapshot
    plot = {"wave": [1, 2, 3]}
    identity = {"path": "old.wav"}
    identity_checkpoint = deepcopy(identity)
    mutator_calls = []
    durable_calls = []

    def clear_plot_without_hooks():
        mutator_calls.append("plot")
        plot["wave"].clear()

    def clear_identity_then_raise():
        mutator_calls.append("identity")
        identity["path"] = None
        raise RuntimeError("later required projection failed")

    controller._clear_plot = clear_plot_without_hooks
    controller._plot_state_capturer = None
    controller._plot_state_restorer = None
    controller._clear_import_identity = clear_identity_then_raise
    controller._import_identity_state_capturer = lambda: deepcopy(identity)
    controller._import_identity_state_restorer = lambda state: (
        identity.clear(), identity.update(deepcopy(state)), True
    )[-1]
    published = []
    controller._publish_configuration = published.append
    controller._using_path_updater = durable_calls.append

    assert controller.on_sequence_config_updated() is False

    assert mutator_calls == []
    assert durable_calls == []
    assert plot == {"wave": [1, 2, 3]}
    assert identity == identity_checkpoint
    assert controller.projection_consistent is False
    assert "projection hooks" in controller.projection_failure
    assert workflow.configuration_generation == 4
    assert workflow.configuration_snapshot is old_snapshot
    assert controller.model.data_struct.store_wave_data is old_wave
    assert view.player_button.enabled is False
    assert view.replay_button.enabled is False
    assert view.data_button.enabled is False
    assert published == []
    assert any("projection hooks" in message for message in warnings)


def test_plot_projection_hooks_are_not_required_without_scheduled_cleanup():
    controller, workflow, view, data_struct, _mono, warnings = (
        _config_update_failure_controller(failure="none", mode="RECORD_ONLY")
    )
    data_struct.opaque_lock = threading.Lock()
    mutator_calls = []
    controller._clear_plot = lambda: mutator_calls.append(True)
    controller._plot_state_capturer = None
    controller._plot_state_restorer = None
    old_snapshot = workflow.configuration_snapshot

    assert controller.on_sequence_config_updated() is True

    assert mutator_calls == []
    assert controller.projection_consistent is True
    assert workflow.configuration_snapshot is not old_snapshot
    assert not any("projection hooks" in message for message in warnings)


@pytest.mark.parametrize("restore_failure", ["false", "raise"])
def test_projection_restore_failure_enters_explicit_disabled_inconsistent_state(
    restore_failure,
):
    controller, _workflow, view, _data_struct, _mono, warnings = (
        _config_update_failure_controller(failure="none", mode="RECORD_ONLY")
    )
    controller.model.apply_configuration(
        ConfigurationSnapshot(
            sequence_config=_sequence("IMPORT_AUDIO"),
            analysis_config={},
            using_config_path="old.json",
        ),
        generation=4,
    )
    identity = {"path": "old.wav", "info": {"value": 1}}

    def clear_then_raise():
        identity["path"] = None
        raise RuntimeError("projection failed after mutation")

    def restore_then_fail(state):
        identity.clear()
        identity.update(deepcopy(state))
        identity["restorer_partial"] = True
        if restore_failure == "raise":
            raise RuntimeError("restore failed")
        return False

    controller._clear_import_identity = clear_then_raise
    controller._import_identity_state_capturer = lambda: deepcopy(identity)
    controller._import_identity_state_restorer = restore_then_fail
    controller._clear_plot = None
    published = []
    controller._publish_configuration = published.append

    assert controller.on_sequence_config_updated() is False

    assert controller.projection_consistent is False
    assert controller.projection_failure
    assert view.player_button.enabled is False
    assert view.replay_button.enabled is False
    assert view.data_button.enabled is False
    assert published == []
    assert any("restore after clear retained import identity" in message for message in warnings)


@pytest.mark.parametrize("failure_kind", ["false", "raise"])
def test_analysis_reference_failure_mutates_only_isolated_staging_object(
    failure_kind,
):
    path = "import.json"
    old_detail = {"stimulus_path": "old.wav"}
    new_detail = {"stimulus_path": "new.wav"}
    runtime_array = np.array([1.0, 2.0])
    data_struct = SimpleNamespace(
        sample_rate=32000,
        audio_lenth=2,
        store_wave_data=np.ones(2),
        store_wave_data_multi=np.ones((2, 1)),
        stimulus_data=np.ones(2),
        stimulus_info={"nested": {"values": runtime_array}},
    )
    workflow = SequenceWorkflowModel()
    model = SequenceConfigurationModel(data_struct=data_struct, workflow_model=workflow)
    model.apply_configuration(
        ConfigurationSnapshot(
            sequence_config=_sequence("IMPORT_STIMULUS_AUDIO", old_detail),
            analysis_config={},
            using_config_path=path,
        ),
        generation=2,
    )

    def mutate_staged_then_reject(staged, _detail, **_kwargs):
        staged.stimulus_info = {"nested": {"values": np.array([99.0, 99.0])}}
        staged.partial_field = ["candidate-only"]
        if failure_kind == "raise":
            raise RuntimeError("reference failed after nested mutation")
        return False

    controller = _controller(
        model=model,
        registry_loader=lambda: {"import": path, "using_config_path": path},
        config_loader=lambda _path: (
            0,
            _sequence("IMPORT_STIMULUS_AUDIO", new_detail),
        ),
        analysis_reference_setter=mutate_staged_then_reject,
        using_path_updater=lambda _path: True,
        ok_code=0,
        path_exists=lambda _path: True,
        warning=lambda *_args: None,
    )

    assert controller.on_sequence_config_updated() is False

    assert np.array_equal(
        data_struct.stimulus_info["nested"]["values"], np.array([1.0, 2.0])
    )
    assert not hasattr(data_struct, "partial_field")


@pytest.mark.parametrize("failure_kind", ["false", "raise"])
def test_regenerated_save_partial_failure_is_compensated_before_memory_rollback(
    failure_kind,
):
    controller, workflow, view, data_struct, mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    controller._resolve_duplex_sample_rate = lambda *_args: SimpleNamespace(
        ok=True, sample_rate=48000, message=""
    )
    controller._stimulus_setter = lambda staged, detail, **_kwargs: (
        setattr(staged, "stimulus_data", np.ones(8)),
        detail.__setitem__("stimulus_path", "generated.wav"),
        True,
    )[-1]
    durable = {"config": _sequence("PLAY_AND_RECORD", {"stimulus_path": "old.wav"})}
    old_durable = deepcopy(durable)

    def saver(config, _path):
        durable["config"] = deepcopy(config)
        if config[0]["seq1"]["acq"]["detail"].get("stimulus_path") == "generated.wav":
            if failure_kind == "raise":
                raise OSError("save failed after replacement")
            return False
        return True

    controller._config_saver = saver
    controller._config_persistence_snapshotter = lambda _path: deepcopy(durable)
    controller._config_persistence_restorer = lambda _path, state: (
        durable.clear(), durable.update(deepcopy(state)), True
    )[-1]
    published = []
    controller._publish_configuration = published.append

    assert controller.on_sequence_config_updated() is False

    assert durable == old_durable
    assert controller.persistence_consistent is True
    assert controller.model.acquisition_mode == "RECORD_ONLY"
    assert workflow.configuration_generation == 4
    assert view.combobox.currentData() == "old.json"
    assert data_struct.store_wave_data is mono
    assert published == []


@pytest.mark.parametrize("failure_kind", ["false", "raise"])
def test_later_registry_write_failure_compensates_it_and_prior_regenerated_save(
    failure_kind,
):
    controller, workflow, _view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    controller._registry_loader = lambda: {
        "new": "new.json",
        "using_config_path": "missing.json",
    }
    controller._path_exists = lambda path: path == "new.json"
    controller._resolve_duplex_sample_rate = lambda *_args: SimpleNamespace(
        ok=True, sample_rate=48000, message=""
    )
    controller._stimulus_setter = lambda staged, detail, **_kwargs: (
        setattr(staged, "stimulus_data", np.ones(8)),
        detail.__setitem__("stimulus_path", "generated.wav"),
        True,
    )[-1]
    durable = {
        "config": _sequence("PLAY_AND_RECORD", {"stimulus_path": "old.wav"}),
        "path": "missing.json",
    }
    old_durable = deepcopy(durable)
    controller._config_saver = lambda config, _path: durable.__setitem__(
        "config", deepcopy(config)
    ) or True

    def update_path(path):
        durable["path"] = path
        if path == "new.json":
            if failure_kind == "raise":
                raise OSError("registry failed after replacement")
            return False
        return True

    controller._using_path_updater = update_path
    controller._config_persistence_snapshotter = lambda _path: deepcopy(
        durable["config"]
    )
    controller._config_persistence_restorer = lambda _path, state: durable.__setitem__(
        "config", deepcopy(state)
    ) or True
    controller._using_path_persistence_snapshotter = lambda: durable["path"]
    controller._using_path_persistence_restorer = lambda state: durable.__setitem__(
        "path", state
    ) or True

    assert controller.on_sequence_config_updated() is False

    assert durable == old_durable
    assert controller.persistence_consistent is True
    assert controller.model.acquisition_mode == "RECORD_ONLY"
    assert workflow.configuration_generation == 4


def test_default_json_writers_restore_exact_bytes_after_second_write_failure(tmp_path):
    controller, workflow, _view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    config_path = tmp_path / "candidate.json"
    registry_path = tmp_path / "registry.json"
    original_config = _sequence(
        "PLAY_AND_RECORD", {"stimulus_path": "original.wav"}
    )
    config_bytes = (
        json.dumps(original_config, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    registry_payload = {
        "new": str(config_path),
        "using_config_path": str(tmp_path / "missing.json"),
    }
    registry_bytes = (
        json.dumps(registry_payload, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    config_path.write_bytes(config_bytes)
    registry_path.write_bytes(registry_bytes)
    controller._registry_loader = lambda: LoadUiConfig._load_sequence_config_registry(
        str(registry_path)
    )
    controller._config_loader = LoadUiConfig().load_sequence_config_from_json
    coordinator = controller._path_transaction_coordinator
    call_order = []

    def save_config(config, path):
        call_order.append("config-write")
        return LoadUiConfig.save_sequence_config_to_json(
            config, path, coordinator=coordinator
        )

    controller._config_saver = save_config
    controller._path_exists = os.path.exists
    controller._resolve_duplex_sample_rate = lambda *_args: SimpleNamespace(
        ok=True, sample_rate=48000, message=""
    )

    def prepare(staged, detail, **_kwargs):
        staged.stimulus_data = np.ones(8)
        detail["stimulus_path"] = "generated.wav"
        return True

    controller._stimulus_setter = prepare

    def update_then_reject(path):
        call_order.append("registry-write")
        assert LoadUiConfig.update_using_config_path(
            path,
            str(registry_path),
            coordinator=coordinator,
        ) is True
        return False

    controller._using_path_updater = update_then_reject
    controller._config_persistence_snapshotter = lambda path: (
        LoadUiConfig._capture_file_bytes(path, coordinator=coordinator)
    )

    def restore_config(path, state):
        call_order.append("config-restore")
        return LoadUiConfig._restore_file_bytes_atomically(
            path, state, coordinator=coordinator
        )

    controller._config_persistence_restorer = restore_config
    controller._using_path_persistence_transaction_key = str(registry_path)
    controller._using_path_persistence_snapshotter = lambda: (
        LoadUiConfig._capture_file_bytes(
            str(registry_path), coordinator=coordinator
        )
    )

    def restore_registry(state):
        call_order.append("registry-restore")
        return LoadUiConfig._restore_file_bytes_atomically(
            str(registry_path), state, coordinator=coordinator
        )

    controller._using_path_persistence_restorer = restore_registry

    assert controller.on_sequence_config_updated() is False

    assert config_path.read_bytes() == config_bytes
    assert registry_path.read_bytes() == registry_bytes
    assert list(tmp_path.glob(".*.tmp")) == []
    assert controller.persistence_consistent is True
    assert controller.model.acquisition_mode == "RECORD_ONLY"
    assert workflow.configuration_generation == 4
    assert call_order == [
        "config-write",
        "registry-write",
        "registry-restore",
        "config-restore",
    ]


def test_registry_compensation_does_not_overwrite_concurrent_newer_entry(tmp_path):
    controller, _workflow, view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="RECORD_ONLY")
    )
    controller.model.apply_configuration(
        ConfigurationSnapshot(
            sequence_config=_sequence("IMPORT_AUDIO"),
            analysis_config={},
            using_config_path="old.json",
        ),
        generation=4,
    )
    candidate_path = tmp_path / "new.json"
    candidate_path.write_text(json.dumps(_sequence("RECORD_ONLY")), encoding="utf-8")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "new": str(candidate_path),
                "using_config_path": str(tmp_path / "missing.json"),
            }
        ),
        encoding="utf-8",
    )
    controller._registry_loader = lambda: LoadUiConfig._load_sequence_config_registry(
        str(registry_path)
    )
    controller._config_loader = LoadUiConfig().load_sequence_config_from_json
    controller._path_exists = os.path.exists
    coordinator = controller._path_transaction_coordinator
    controller._using_path_persistence_transaction_key = str(registry_path)
    controller._using_path_updater = lambda path: LoadUiConfig.update_using_config_path(
        path, str(registry_path), coordinator=coordinator
    )
    controller._using_path_persistence_snapshotter = lambda: (
        LoadUiConfig._capture_file_bytes(
            str(registry_path), coordinator=coordinator
        )
    )
    controller._using_path_persistence_restorer = (
        lambda checkpoint, expected_current=None: (
            LoadUiConfig._restore_sequence_registry_checkpoint(
                str(registry_path),
                checkpoint,
                expected_current=expected_current,
                coordinator=coordinator,
            )
        )
    )
    identity = {"path": "old.wav"}

    def concurrent_write_then_fail():
        assert LoadUiConfig.append_sequence_config_registry_entry(
            str(tmp_path / "concurrent.json"), str(registry_path)
        )
        identity["path"] = None
        raise RuntimeError("cleanup failed after concurrent writer")

    controller._clear_import_identity = concurrent_write_then_fail
    controller._import_identity_state_capturer = lambda: deepcopy(identity)
    controller._import_identity_state_restorer = lambda state: (
        identity.clear(), identity.update(deepcopy(state)), True
    )[-1]
    controller._clear_plot = None
    published = []
    controller._publish_configuration = published.append

    assert controller.on_sequence_config_updated() is False

    registry = LoadUiConfig._load_sequence_config_registry(str(registry_path))
    assert registry["concurrent"] == str(tmp_path / "concurrent.json")
    assert registry["using_config_path"] == str(candidate_path)
    assert controller.persistence_consistent is False
    assert controller.projection_consistent is False
    assert "configuration-dependent projections" in controller.projection_failure
    assert view.player_button.enabled is False
    assert published == []


def test_config_compensation_does_not_overwrite_cooperating_newer_writer(tmp_path):
    controller, _workflow, view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    config_path = tmp_path / "selected.json"
    candidate = _sequence(
        "PLAY_AND_RECORD", {"stimulus_path": "original.wav"}
    )
    newer = _sequence("RECORD_ONLY", {"writer": "newer"})
    assert LoadUiConfig.save_sequence_config_to_json(candidate, str(config_path))
    controller.model.apply_configuration(
        ConfigurationSnapshot(
            sequence_config=_sequence("RECORD_ONLY"),
            analysis_config={},
            using_config_path=str(config_path),
        ),
        generation=4,
    )
    controller.model.replace_registry(
        {"selected": str(config_path), "using_config_path": str(config_path)},
        using_config_path=str(config_path),
        entries=(("selected", str(config_path)),),
    )
    controller._registry_loader = lambda: {
        "selected": str(config_path),
        "using_config_path": str(config_path),
    }
    controller._config_loader = LoadUiConfig().load_sequence_config_from_json
    controller._path_exists = os.path.exists
    controller._resolve_duplex_sample_rate = lambda *_args: SimpleNamespace(
        ok=True, sample_rate=48000, message=""
    )
    controller._stimulus_setter = lambda staged, detail, **_kwargs: (
        setattr(staged, "stimulus_data", np.ones(8)),
        detail.__setitem__("stimulus_path", "generated.wav"),
        True,
    )[-1]
    plot = {"wave": [1, 2]}

    def cooperating_write_then_fail():
        assert LoadUiConfig.save_sequence_config_to_json(newer, str(config_path))
        plot["wave"].clear()
        raise RuntimeError("cleanup failed after newer config writer")

    controller._clear_plot = cooperating_write_then_fail
    controller._plot_state_capturer = lambda: deepcopy(plot)
    controller._plot_state_restorer = lambda state: (
        plot.clear(), plot.update(deepcopy(state)), True
    )[-1]
    published = []
    controller._publish_configuration = published.append

    assert controller.on_sequence_config_updated() is False

    load_code, durable = LoadUiConfig().load_sequence_config_from_json(
        str(config_path)
    )
    assert load_code == 0
    assert durable == newer
    assert controller.persistence_consistent is False
    assert controller.projection_consistent is False
    assert "configuration-dependent projections" in controller.projection_failure
    assert controller.model.acquisition_mode == "RECORD_ONLY"
    assert view.player_button.enabled is False
    assert plot == {"wave": [1, 2]}
    assert published == []


def test_regenerated_config_semantic_cas_rejects_newer_cooperating_writer():
    controller, workflow, view, data_struct, mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    candidate = _sequence(
        "PLAY_AND_RECORD", {"stimulus_path": "original.wav"}
    )
    newer = _sequence(
        "PLAY_AND_RECORD",
        {"stimulus_path": "newer.wav", "writer_marker": "newer"},
    )
    durable = {"config": deepcopy(candidate)}
    controller._config_loader = lambda _path: (0, deepcopy(durable["config"]))
    controller._resolve_duplex_sample_rate = lambda *_args: SimpleNamespace(
        ok=True, sample_rate=48000, message=""
    )
    controller._stimulus_setter = lambda staged, detail, **_kwargs: (
        setattr(staged, "stimulus_data", np.ones(8)),
        detail.__setitem__("stimulus_path", "generated.wav"),
        True,
    )[-1]
    coordinator = controller._path_transaction_coordinator
    competing = PathTransactionCoordinator()
    transaction_key = "semantic-config-cas"
    barrier_complete = False
    calls = []

    @contextmanager
    def persistence_transaction(_target):
        nonlocal barrier_complete
        if not barrier_complete:
            with competing.transaction(transaction_key):
                durable["config"] = deepcopy(newer)
                calls.append("newer-write")
            barrier_complete = True
        with coordinator.transaction(transaction_key):
            calls.append("lock-enter")
            yield

    def read_semantic_current(_target):
        normalized = coordinator.normalize_path(transaction_key)
        assert getattr(coordinator._lock_state, "active", {}).get(normalized) == 1
        calls.append("semantic-read")
        return deepcopy(durable["config"])

    def write(payload, _target):
        calls.append("controller-write")
        durable["config"] = deepcopy(payload)
        return True

    adapter = SimpleNamespace(
        coordinator=coordinator,
        transaction=persistence_transaction,
        capture=lambda _target: deepcopy(durable["config"]),
        write=write,
        conditional_restore=lambda _target, checkpoint, owned_token: (
            durable.__setitem__("config", deepcopy(checkpoint))
            if _durable_ownership_token(durable["config"]) == owned_token
            else False
        ),
        ownership_token=_durable_ownership_token,
        read_durable_truth=lambda _target: (0, deepcopy(durable["config"])),
        read_semantic_current=read_semantic_current,
    )
    controller._config_persistence_adapter_override = adapter
    old_snapshot = controller.model.checkpoint_configuration_state()
    old_actions = view.capture_action_availability()
    published = []
    controller._publish_configuration = published.append

    assert controller.on_sequence_config_updated() is False

    assert calls == ["newer-write", "lock-enter", "semantic-read"]
    assert durable["config"] == newer
    assert controller.persistence_consistent is True
    assert controller.model.checkpoint_configuration_state() == old_snapshot
    assert workflow.configuration_generation == 4
    assert view.capture_action_availability() == old_actions
    assert data_struct.store_wave_data is mono
    assert published == []


def test_active_path_semantic_cas_rejects_changed_selected_entry_binding():
    controller, workflow, view, data_struct, mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="RECORD_ONLY")
    )
    durable = {
        "registry": {
            "new": "new.json",
            "using_config_path": "old.json",
        }
    }
    controller._registry_loader = lambda: deepcopy(durable["registry"])
    controller._config_loader = lambda path: (
        (0, _sequence("RECORD_ONLY"))
        if path == "new.json"
        else pytest.fail(f"unexpected path: {path}")
    )
    controller._path_exists = lambda path: path == "new.json"
    coordinator = controller._path_transaction_coordinator
    competing = PathTransactionCoordinator()
    transaction_key = "semantic-registry-cas"
    barrier_complete = False
    calls = []

    @contextmanager
    def persistence_transaction(_target):
        nonlocal barrier_complete
        if not barrier_complete:
            with competing.transaction(transaction_key):
                durable["registry"]["new"] = "replacement.json"
                calls.append("newer-write")
            barrier_complete = True
        with coordinator.transaction(transaction_key):
            calls.append("lock-enter")
            yield

    def read_semantic_current(_target):
        normalized = coordinator.normalize_path(transaction_key)
        assert getattr(coordinator._lock_state, "active", {}).get(normalized) == 1
        calls.append("semantic-read")
        return deepcopy(durable["registry"])

    def write(path, _target):
        calls.append("controller-write")
        durable["registry"]["using_config_path"] = path
        return True

    adapter = SimpleNamespace(
        coordinator=coordinator,
        transaction=persistence_transaction,
        capture=lambda _target: deepcopy(durable["registry"]),
        write=write,
        conditional_restore=lambda _target, checkpoint, owned_token: (
            durable.__setitem__("registry", deepcopy(checkpoint))
            if _durable_ownership_token(durable["registry"]) == owned_token
            else False
        ),
        ownership_token=_durable_ownership_token,
        read_durable_truth=lambda _target: deepcopy(durable["registry"]),
        read_semantic_current=read_semantic_current,
    )
    controller._using_path_persistence_adapter_override = adapter
    old_snapshot = controller.model.checkpoint_configuration_state()
    old_actions = view.capture_action_availability()
    published = []
    controller._publish_configuration = published.append

    assert controller.on_sequence_config_updated() is False

    assert calls == ["newer-write", "lock-enter", "semantic-read"]
    assert durable["registry"] == {
        "new": "replacement.json",
        "using_config_path": "old.json",
    }
    assert controller.persistence_consistent is True
    assert controller.model.checkpoint_configuration_state() == old_snapshot
    assert workflow.configuration_generation == 4
    assert view.capture_action_availability() == old_actions
    assert data_struct.store_wave_data is mono
    assert published == []


def test_active_path_semantic_cas_preserves_unrelated_concurrent_registry_addition():
    controller, workflow, _view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="RECORD_ONLY")
    )
    durable = {
        "registry": {
            "new": "new.json",
            "using_config_path": "old.json",
        }
    }
    controller._registry_loader = lambda: deepcopy(durable["registry"])
    controller._config_loader = lambda path: (
        (0, _sequence("RECORD_ONLY"))
        if path == "new.json"
        else pytest.fail(f"unexpected path: {path}")
    )
    controller._path_exists = lambda path: path == "new.json"
    coordinator = controller._path_transaction_coordinator
    competing = PathTransactionCoordinator()
    transaction_key = "semantic-registry-cas-unrelated"
    barrier_complete = False
    calls = []

    @contextmanager
    def persistence_transaction(_target):
        nonlocal barrier_complete
        if not barrier_complete:
            with competing.transaction(transaction_key):
                durable["registry"]["concurrent"] = "concurrent.json"
                calls.append("newer-unrelated-write")
            barrier_complete = True
        with coordinator.transaction(transaction_key):
            calls.append("lock-enter")
            yield

    def read_semantic_current(_target):
        normalized = coordinator.normalize_path(transaction_key)
        assert getattr(coordinator._lock_state, "active", {}).get(normalized) == 1
        calls.append("semantic-read")
        return deepcopy(durable["registry"])

    def write(path, _target):
        calls.append("controller-write")
        durable["registry"]["using_config_path"] = path
        return True

    controller._using_path_persistence_adapter_override = SimpleNamespace(
        coordinator=coordinator,
        transaction=persistence_transaction,
        capture=lambda _target: deepcopy(durable["registry"]),
        write=write,
        conditional_restore=lambda _target, checkpoint, owned_token: (
            durable.__setitem__("registry", deepcopy(checkpoint))
            if _durable_ownership_token(durable["registry"]) == owned_token
            else False
        ),
        ownership_token=_durable_ownership_token,
        read_durable_truth=lambda _target: deepcopy(durable["registry"]),
        read_semantic_current=read_semantic_current,
    )
    published = []
    controller._publish_configuration = published.append

    assert controller.on_sequence_config_updated() is True

    assert calls == [
        "newer-unrelated-write",
        "lock-enter",
        "semantic-read",
        "controller-write",
    ]
    assert durable["registry"] == {
        "concurrent": "concurrent.json",
        "new": "new.json",
        "using_config_path": "new.json",
    }
    assert controller.persistence_consistent is True
    assert workflow.configuration_generation == 5
    assert len(published) == 1


def test_regenerated_config_semantic_cas_writes_when_checkpoint_is_unchanged():
    controller, workflow, _view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    candidate = _sequence(
        "PLAY_AND_RECORD", {"stimulus_path": "original.wav"}
    )
    durable = {"config": deepcopy(candidate)}
    controller._config_loader = lambda _path: (0, deepcopy(durable["config"]))
    controller._resolve_duplex_sample_rate = lambda *_args: SimpleNamespace(
        ok=True, sample_rate=48000, message=""
    )
    controller._stimulus_setter = lambda staged, detail, **_kwargs: (
        setattr(staged, "stimulus_data", np.ones(8)),
        detail.__setitem__("stimulus_path", "generated.wav"),
        True,
    )[-1]
    coordinator = controller._path_transaction_coordinator
    transaction_key = "semantic-config-cas-success"
    calls = []

    @contextmanager
    def persistence_transaction(_target):
        with coordinator.transaction(transaction_key):
            calls.append("lock-enter")
            yield

    def read_semantic_current(_target):
        normalized = coordinator.normalize_path(transaction_key)
        assert getattr(coordinator._lock_state, "active", {}).get(normalized) == 1
        calls.append("semantic-read")
        return deepcopy(durable["config"])

    def write(payload, _target):
        calls.append("controller-write")
        durable["config"] = deepcopy(payload)
        return True

    controller._config_persistence_adapter_override = SimpleNamespace(
        coordinator=coordinator,
        transaction=persistence_transaction,
        capture=lambda _target: deepcopy(durable["config"]),
        write=write,
        conditional_restore=lambda _target, checkpoint, owned_token: (
            durable.__setitem__("config", deepcopy(checkpoint))
            if _durable_ownership_token(durable["config"]) == owned_token
            else False
        ),
        ownership_token=_durable_ownership_token,
        read_durable_truth=lambda _target: (0, deepcopy(durable["config"])),
        read_semantic_current=read_semantic_current,
    )
    published = []
    controller._publish_configuration = published.append

    assert controller.on_sequence_config_updated() is True

    assert calls == ["lock-enter", "semantic-read", "controller-write"]
    assert durable["config"][0]["seq1"]["acq"]["detail"]["stimulus_path"] == (
        "generated.wav"
    )
    assert workflow.configuration_generation == 5
    assert len(published) == 1


def test_compensation_failure_exposes_inconsistent_state_and_reconciles_to_durable():
    controller, workflow, view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    durable = {"path": "missing.json"}
    candidate = _sequence("PLAY_AND_RECORD")
    controller._registry_loader = lambda: {
        "new": "new.json",
        "using_config_path": durable["path"],
    }
    controller._path_exists = lambda path: path == "new.json"
    controller._config_loader = lambda path: (
        0,
        candidate if path == "new.json" else _sequence("RECORD_ONLY"),
    )
    controller._resolve_duplex_sample_rate = lambda *_args: SimpleNamespace(
        ok=True, sample_rate=48000, message=""
    )
    controller._stimulus_setter = lambda staged, _detail, **_kwargs: (
        setattr(staged, "stimulus_data", np.ones(8)),
        False,
    )[-1]

    def update_path(path):
        durable["path"] = path
        return False

    controller._using_path_updater = update_path
    controller._using_path_persistence_snapshotter = lambda: "missing.json"
    controller._using_path_persistence_restorer = lambda _state: False
    published = []
    controller._publish_configuration = published.append

    assert controller.on_sequence_config_updated() is False

    assert controller.persistence_consistent is False
    assert "persist active sequence config path" in controller.persistence_failure
    assert durable["path"] == "new.json"
    assert controller.model.using_config_path == "new.json"
    assert controller.model.acquisition_mode == "PLAY_AND_RECORD"
    assert (
        workflow.configuration_snapshot.sequence_config[0]["seq1"]["acq"]["mode"]
        == "PLAY_AND_RECORD"
    )
    assert view.player_button.enabled is False
    assert view.replay_button.enabled is False
    assert view.data_button.enabled is False
    assert published == []


@pytest.mark.parametrize("failed_projection", ["identity", "plot"])
def test_required_cleanup_partial_raise_restores_projection_and_durable_state(
    failed_projection,
):
    controller, workflow, _view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="RECORD_ONLY")
    )
    controller.model.apply_configuration(
        ConfigurationSnapshot(
            sequence_config=_sequence("IMPORT_AUDIO"),
            analysis_config={},
            using_config_path="old.json",
        ),
        generation=4,
    )
    durable = {"path": "old.json"}
    identity = {"path": "old.wav", "info": {"nested": [1, 2]}}
    plot = {"curves": [np.array([1.0, 2.0])], "rate": 32000}
    expected_identity = deepcopy(identity)
    expected_plot = deepcopy(plot)

    def clear_identity():
        identity["path"] = None
        if failed_projection == "identity":
            raise RuntimeError("identity failed after partial clear")
        identity["info"] = None

    def clear_plot():
        plot["curves"].clear()
        if failed_projection == "plot":
            raise RuntimeError("plot failed after partial clear")
        plot["rate"] = None

    controller._clear_import_identity = clear_identity
    controller._clear_plot = clear_plot
    controller._import_identity_state_capturer = lambda: deepcopy(identity)
    controller._import_identity_state_restorer = lambda state: (
        identity.clear(), identity.update(deepcopy(state))
    )
    controller._plot_state_capturer = lambda: deepcopy(plot)
    controller._plot_state_restorer = lambda state: (
        plot.clear(), plot.update(deepcopy(state))
    )
    controller._registry_loader = lambda: {
        "new": "new.json",
        "using_config_path": "missing.json",
    }
    controller._path_exists = lambda path: path == "new.json"
    controller._using_path_updater = lambda path: durable.__setitem__("path", path) or True
    controller._using_path_persistence_snapshotter = lambda: durable["path"]
    controller._using_path_persistence_restorer = lambda state: durable.__setitem__(
        "path", state
    ) or True
    published = []
    controller._publish_configuration = published.append

    assert controller.on_sequence_config_updated() is False

    assert identity == expected_identity
    assert plot["rate"] == expected_plot["rate"]
    assert len(plot["curves"]) == 1
    assert np.array_equal(plot["curves"][0], expected_plot["curves"][0])
    assert durable == {"path": "old.json"}
    assert controller.persistence_consistent is True
    assert controller.model.acquisition_mode == "IMPORT_AUDIO"
    assert workflow.configuration_generation == 4
    assert published == []


def _raise_exact(error):
    raise error


def _assert_exact_traceback(caught, error, frame_name):
    assert caught.value is error
    assert caught.value.__traceback__ is not None
    names = []
    traceback = caught.value.__traceback__
    while traceback is not None:
        names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert frame_name in names


@pytest.mark.parametrize("boundary", ("path", "entries", "model", "view"))
@pytest.mark.parametrize("error_type", (RuntimeError, KeyboardInterrupt, SystemExit))
def test_startup_fallback_post_write_registry_projection_failure_recovers_transactionally(
    boundary, error_type
):
    controller, model, view, combo, durable, _calls, published = (
        _registry_fallback_cas_controller()
    )
    view.player_button = _Button(True)
    view.replay_button = _Button(True)
    view.data_button = _Button(True)
    adapter = controller._using_path_persistence_adapter_override
    original_write = adapter.write
    original_probe = controller._path_exists
    original_derivation = controller._derive_ordered_registry_entries
    original_replace = model.replace_registry
    original_populate = view.populate_configuration_entries
    state = {"written": False, "failed": False}
    error = error_type(f"startup {boundary} projection sentinel")

    def write(path, target):
        result = original_write(path, target)
        state["written"] = True
        return result

    def fail_once(callback, *args, **kwargs):
        if state["written"] and not state["failed"]:
            state["failed"] = True
            callback(*args, **kwargs)
            _raise_exact(error)
        return callback(*args, **kwargs)

    adapter.write = write
    if boundary == "path":
        def probe(path):
            if state["written"] and not state["failed"]:
                state["failed"] = True
                _raise_exact(error)
            return original_probe(path)

        controller._path_exists = probe
    elif boundary == "entries":
        controller._derive_ordered_registry_entries = lambda *args, **kwargs: (
            fail_once(original_derivation, *args, **kwargs)
        )
    elif boundary == "model":
        model.replace_registry = lambda *args, **kwargs: fail_once(
            original_replace, *args, **kwargs
        )
    else:
        view.populate_configuration_entries = lambda *args, **kwargs: fail_once(
            original_populate, *args, **kwargs
        )

    if issubclass(error_type, Exception):
        using_path, loaded = controller.get_sequence_config_from_registry()
        assert using_path == durable["registry"]["using_config_path"]
        assert loaded == durable["registry"]
    else:
        with pytest.raises(error_type) as caught:
            controller.get_sequence_config_from_registry()
        _assert_exact_traceback(caught, error, "_raise_exact")

    assert state == {"written": True, "failed": True}
    assert durable["registry"] == {
        "fallback": "fallback.json",
        "using_config_path": "missing.json",
    }
    assert model.registry == durable["registry"]
    assert model.using_config_path == "missing.json"
    assert combo.items == list(model.registry_entries)
    assert view.capture_action_availability() == (False, False, False)
    assert controller.persistence_consistent is True
    assert controller.projection_consistent is False
    assert "configuration-dependent projections" in controller.projection_failure
    assert published == []


@pytest.mark.parametrize(
    "logging_error_type", (RuntimeError, KeyboardInterrupt, SystemExit)
)
def test_startup_fallback_path_probe_logging_failure_does_not_skip_recovery(
    logging_error_type,
):
    controller, model, view, _combo, durable, _calls, published = (
        _registry_fallback_cas_controller()
    )
    view.player_button = _Button(True)
    view.replay_button = _Button(True)
    view.data_button = _Button(True)
    adapter = controller._using_path_persistence_adapter_override
    original_write = adapter.write
    original_probe = controller._path_exists
    state = {"written": False, "probe_failed": False, "logged": False}
    probe_error = RuntimeError("post-write probe failed")
    logging_error = logging_error_type("probe diagnostic logging interrupted")

    def write(path, target):
        result = original_write(path, target)
        state["written"] = True
        return result

    def probe(path):
        if state["written"] and not state["probe_failed"]:
            state["probe_failed"] = True
            raise probe_error
        return original_probe(path)

    def warning(_message):
        if not state["logged"]:
            state["logged"] = True
            _raise_exact(logging_error)

    adapter.write = write
    controller._path_exists = probe
    controller._logger = SimpleNamespace(warning=warning)

    if issubclass(logging_error_type, Exception):
        controller.get_sequence_config_from_registry()
    else:
        with pytest.raises(logging_error_type) as caught:
            controller.get_sequence_config_from_registry()
        _assert_exact_traceback(caught, logging_error, "_raise_exact")
    assert durable["registry"]["using_config_path"] == "missing.json"
    assert model.registry == durable["registry"]
    assert view.capture_action_availability() == (False, False, False)
    assert controller.persistence_consistent is True
    assert controller.projection_consistent is False
    assert "configuration-dependent projections" in controller.projection_failure
    assert published == []


@pytest.mark.parametrize("boundary", ("path", "entries", "model", "view"))
@pytest.mark.parametrize("error_type", (RuntimeError, KeyboardInterrupt, SystemExit))
def test_config_update_post_write_registry_projection_failure_compensates_and_reconciles(
    boundary, error_type
):
    controller, _workflow, view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="RECORD_ONLY")
    )
    model = controller.model
    durable = {
        "new": "new.json",
        "using_config_path": "missing.json",
    }
    controller._registry_loader = lambda: deepcopy(durable)
    controller._path_exists = lambda path: path == "new.json"
    controller._config_loader = lambda path: (
        (0, _sequence("RECORD_ONLY"))
        if path in {"new.json", "missing.json"}
        else pytest.fail(f"unexpected path: {path}")
    )
    state = {"written": False, "failed": False}

    def persist(path):
        durable["using_config_path"] = path
        state["written"] = True
        return True

    def restore(checkpoint):
        durable.clear()
        durable.update(deepcopy(checkpoint))
        return True

    controller._using_path_updater = persist
    controller._using_path_persistence_snapshotter = lambda: deepcopy(durable)
    controller._using_path_persistence_restorer = restore
    original_replace = model.replace_registry
    original_populate = view.populate_configuration_entries
    original_probe = controller._path_exists
    original_derivation = controller._derive_ordered_registry_entries
    error = error_type(f"update {boundary} projection sentinel")

    def fail_once(callback, *args, **kwargs):
        if state["written"] and not state["failed"]:
            state["failed"] = True
            callback(*args, **kwargs)
            _raise_exact(error)
        return callback(*args, **kwargs)

    if boundary == "path":
        def probe(path):
            if state["written"] and not state["failed"]:
                state["failed"] = True
                _raise_exact(error)
            return original_probe(path)

        controller._path_exists = probe
    elif boundary == "entries":
        controller._derive_ordered_registry_entries = lambda *args, **kwargs: (
            fail_once(original_derivation, *args, **kwargs)
        )
    elif boundary == "model":
        model.replace_registry = lambda *args, **kwargs: fail_once(
            original_replace, *args, **kwargs
        )
    else:
        view.populate_configuration_entries = lambda *args, **kwargs: fail_once(
            original_populate, *args, **kwargs
        )
    published = []
    controller._publish_configuration = published.append
    if issubclass(error_type, Exception):
        assert controller.on_sequence_config_updated() is False
    else:
        with pytest.raises(error_type) as caught:
            controller.on_sequence_config_updated()
        _assert_exact_traceback(caught, error, "_raise_exact")

    assert state == {"written": True, "failed": True}
    assert durable == {
        "new": "new.json",
        "using_config_path": "missing.json",
    }
    assert model.registry == durable
    assert model.using_config_path == "missing.json"
    assert view.capture_action_availability() == (False, False, False)
    assert controller.persistence_consistent is True
    assert controller.projection_consistent is False
    assert "configuration-dependent projections" in controller.projection_failure
    assert published == []


@pytest.mark.parametrize("boundary", ("path", "entries", "model", "view"))
@pytest.mark.parametrize("error_type", (RuntimeError, KeyboardInterrupt, SystemExit))
def test_combobox_selection_post_write_registry_projection_failure_recovers(
    boundary, error_type
):
    controller, _workflow, view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="RECORD_ONLY")
    )
    model = controller.model
    model.apply_configuration(
        ConfigurationSnapshot(
            sequence_config=_sequence("RECORD_ONLY"),
            analysis_config={},
            using_config_path="old.json",
        ),
        generation=4,
    )
    durable = {
        "old": "old.json",
        "new": "new.json",
        "using_config_path": "old.json",
    }
    model.replace_registry(
        durable,
        using_config_path="old.json",
        entries=(("new", "new.json"), ("old", "old.json")),
    )
    view.populate_configuration_entries(
        model.registry_entries,
        using_config_path="old.json",
        clear_first=True,
    )
    controller._registry_loader = lambda: deepcopy(durable)
    controller._config_loader = lambda path: (
        (0, _sequence("RECORD_ONLY"))
        if path in {"new.json", "old.json"}
        else pytest.fail(f"unexpected path: {path}")
    )
    controller._path_exists = lambda path: path in {"new.json", "old.json"}
    state = {"written": False, "failed": False}

    def persist(path):
        durable["using_config_path"] = path
        state["written"] = True
        return True

    def restore(checkpoint):
        durable.clear()
        durable.update(deepcopy(checkpoint))
        return True

    controller._using_path_updater = persist
    controller._using_path_persistence_snapshotter = lambda: deepcopy(durable)
    controller._using_path_persistence_restorer = restore
    original_probe = controller._path_exists
    original_derivation = controller._derive_ordered_registry_entries
    original_replace = model.replace_registry
    original_populate = view.populate_configuration_entries
    error = error_type(f"selection {boundary} projection sentinel")

    def fail_once(callback, *args, **kwargs):
        if state["written"] and not state["failed"]:
            state["failed"] = True
            callback(*args, **kwargs)
            _raise_exact(error)
        return callback(*args, **kwargs)

    if boundary == "path":
        def probe(path):
            if state["written"] and not state["failed"]:
                state["failed"] = True
                _raise_exact(error)
            return original_probe(path)

        controller._path_exists = probe
    elif boundary == "entries":
        controller._derive_ordered_registry_entries = lambda *args, **kwargs: (
            fail_once(original_derivation, *args, **kwargs)
        )
    elif boundary == "model":
        model.replace_registry = lambda *args, **kwargs: fail_once(
            original_replace, *args, **kwargs
        )
    else:
        view.populate_configuration_entries = lambda *args, **kwargs: fail_once(
            original_populate, *args, **kwargs
        )
    published = []
    controller._publish_configuration = published.append
    view.combobox.setCurrentIndex(view.combobox.findText("new"))

    if issubclass(error_type, Exception):
        assert controller.on_using_file_combobox_changed("new") is None
    else:
        with pytest.raises(error_type) as caught:
            controller.on_using_file_combobox_changed("new")
        _assert_exact_traceback(caught, error, "_raise_exact")

    assert state == {"written": True, "failed": True}
    assert durable["using_config_path"] == "old.json"
    assert model.registry == durable
    assert model.using_config_path == "old.json"
    assert view.capture_action_availability() == (False, False, False)
    assert controller.persistence_consistent is True
    assert controller.projection_consistent is True, (
        controller.projection_failure,
        controller._projection_failures,
    )
    assert published == []


@pytest.mark.parametrize("boundary", ("path", "entries", "model", "view"))
@pytest.mark.parametrize("error_type", (RuntimeError, KeyboardInterrupt, SystemExit))
def test_central_durable_reconciliation_registry_projection_is_transactional(
    boundary, error_type
):
    controller, _workflow, view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="RECORD_ONLY")
    )
    model = controller.model
    previous_registry = deepcopy(model.registry)
    previous_path = model.using_config_path
    previous_items = list(view.combobox.items)
    durable = {"new": "new.json", "using_config_path": "new.json"}
    controller._registry_loader = lambda: deepcopy(durable)
    controller._config_loader = lambda _path: (0, _sequence("RECORD_ONLY"))
    controller._path_exists = lambda _path: True
    transaction = controller._begin_configuration_transaction(
        selection_path=previous_path
    )
    assert transaction is not None
    original_probe = controller._path_exists
    original_derivation = controller._derive_ordered_registry_entries
    original_replace = model.replace_registry
    original_populate = view.populate_configuration_entries
    calls = []
    error = error_type(f"central {boundary} projection sentinel")

    def fail_once(name, callback, *args, **kwargs):
        calls.append(name)
        callback(*args, **kwargs)
        _raise_exact(error)

    if boundary == "path":
        controller._path_exists = lambda path: fail_once(
            "path", original_probe, path
        )
    elif boundary == "entries":
        controller._derive_ordered_registry_entries = lambda *args, **kwargs: (
            fail_once("entries", original_derivation, *args, **kwargs)
        )
    elif boundary == "model":
        model.replace_registry = lambda *args, **kwargs: fail_once(
            "model", original_replace, *args, **kwargs
        )
    else:
        view.populate_configuration_entries = lambda *args, **kwargs: fail_once(
            "view", original_populate, *args, **kwargs
        )

    if issubclass(error_type, Exception):
        assert controller._reconcile_to_durable_truth(transaction) is False
    else:
        with pytest.raises(error_type) as caught:
            controller._reconcile_to_durable_truth(transaction)
        _assert_exact_traceback(caught, error, "_raise_exact")

    assert calls
    assert model.registry == previous_registry
    assert model.using_config_path == previous_path
    assert view.combobox.items == previous_items
    assert view.capture_action_availability() == (False, False, False)
    assert controller.projection_consistent is False
    assert controller.projection_failure


def test_registry_projection_recovery_marks_partial_channel_rollback_inconsistent():
    controller, _workflow, view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="IMPORT_AUDIO")
    )
    durable = {"new": "new.json", "using_config_path": "missing.json"}
    controller._registry_loader = lambda: deepcopy(durable)
    controller._config_loader = lambda path: (
        (0, _sequence("IMPORT_AUDIO"))
        if path == "new.json"
        else (0, _sequence("RECORD_ONLY"))
    )
    controller._path_exists = lambda path: path == "new.json"
    controller._using_path_persistence_snapshotter = lambda: deepcopy(durable)

    def persist(path):
        durable["using_config_path"] = path
        return True

    def restore(checkpoint):
        durable.clear()
        durable.update(deepcopy(checkpoint))
        return True

    controller._using_path_updater = persist
    controller._using_path_persistence_restorer = restore
    channel = {"mode": "old"}
    channel_calls = []
    rollback_failure = {"enabled": True}

    def refresh_channels():
        channel_calls.append(controller.model.acquisition_mode)
        if len(channel_calls) == 1:
            channel["mode"] = "candidate"
            return True
        if rollback_failure["enabled"]:
            channel["mode"] = "partially-restored"
            raise RuntimeError("channel rollback failed after partial mutation")
        channel["mode"] = controller.model.acquisition_mode
        return True

    controller._refresh_channels = refresh_channels
    original_populate = view.populate_configuration_entries
    projection_failed = {"value": False}

    def fail_first_committed_population(*args, **kwargs):
        original_populate(*args, **kwargs)
        if (
            durable["using_config_path"] == "new.json"
            and not projection_failed["value"]
        ):
            projection_failed["value"] = True
            raise RuntimeError("committed registry view projection failed")

    view.populate_configuration_entries = fail_first_committed_population
    published = []
    controller._publish_configuration = published.append

    assert controller.on_sequence_config_updated() is False

    assert projection_failed["value"] is True
    assert channel_calls == ["IMPORT_AUDIO", "RECORD_ONLY"]
    assert channel == {"mode": "partially-restored"}
    assert controller.model.acquisition_mode == "RECORD_ONLY"
    assert controller.model.registry == durable
    assert controller.model.using_config_path == "missing.json"
    assert view.combobox.items == list(controller.model.registry_entries)
    assert controller.projection_consistent is False
    assert "channels" in controller.projection_failure
    assert "rollback" in controller.projection_failure
    assert view.capture_action_availability() == (False, False, False)
    assert published == []

    controller.get_sequence_config_from_registry()

    assert controller.model.registry == durable
    assert controller.model.using_config_path == "new.json"
    assert controller.projection_consistent is False
    assert "channels" in controller.projection_failure
    assert channel == {"mode": "partially-restored"}
    assert view.capture_action_availability() == (False, False, False)

    controller.get_sequence_config_from_json()

    assert controller.model.acquisition_mode == "IMPORT_AUDIO"
    assert controller.projection_consistent is False
    assert "channels" in controller.projection_failure
    assert channel == {"mode": "partially-restored"}
    assert view.capture_action_availability() == (False, False, False)

    rollback_failure["enabled"] = False
    controller.on_using_file_combobox_changed("new")

    assert channel == {"mode": "IMPORT_AUDIO"}
    assert controller.projection_consistent is True
    assert controller.projection_failure is None
    assert view.capture_action_availability() == (True, False, False)


def test_startup_fallback_successful_retry_clears_only_registry_diagnostics():
    controller, model, view, combo, durable, _calls, published = (
        _registry_fallback_cas_controller()
    )
    view.player_button = _Button(True)
    view.replay_button = _Button(True)
    view.data_button = _Button(True)
    original_populate = view.populate_configuration_entries
    fail_projection = {"value": True}

    def populate(*args, **kwargs):
        original_populate(*args, **kwargs)
        if fail_projection["value"]:
            raise RuntimeError("startup projection temporarily unavailable")

    view.populate_configuration_entries = populate

    controller.get_sequence_config_from_registry()

    assert controller.projection_consistent is False
    assert controller.projection_failure
    fail_projection["value"] = False

    using_path, registry = controller.get_sequence_config_from_registry()

    assert using_path == "fallback.json"
    assert registry == durable["registry"]
    assert model.registry == durable["registry"]
    assert model.using_config_path == durable["registry"]["using_config_path"]
    assert combo.items == list(model.registry_entries)
    assert combo.currentData() == model.using_config_path
    assert controller.projection_consistent is False
    assert controller.projection_failure
    assert "registry" not in controller._projection_failures
    assert set(controller._projection_failures).issubset(
        {"configuration", "view"}
    )
    assert controller.persistence_consistent is True
    assert controller.persistence_failure is None
    assert view.capture_action_availability() == (False, False, False)
    assert published == []


def test_central_reconciliation_successful_retry_resets_diagnostics_only_after_complete_projection():
    controller, _workflow, view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="RECORD_ONLY")
    )
    controller.model.apply_configuration(
        ConfigurationSnapshot(
            sequence_config=_sequence("RECORD_ONLY"),
            analysis_config={},
            using_config_path="old.json",
        ),
        generation=4,
    )
    durable = {
        "old": "old.json",
        "unrelated": "unrelated.json",
        "using_config_path": "old.json",
    }
    controller._registry_loader = lambda: deepcopy(durable)
    controller._config_loader = lambda _path: (0, _sequence("RECORD_ONLY"))
    controller._path_exists = lambda _path: True
    original_populate = view.populate_configuration_entries
    fail_projection = {"value": True}

    def populate(*args, **kwargs):
        original_populate(*args, **kwargs)
        if fail_projection["value"]:
            fail_projection["value"] = False
            raise RuntimeError("central projection temporarily unavailable")

    view.populate_configuration_entries = populate
    first = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert first is not None

    assert controller._reconcile_to_durable_truth(
        first, abort_transaction=False
    ) is False
    assert controller.projection_consistent is False
    assert controller.projection_failure

    controller._mark_persistence_inconsistent(
        "registry", "stale persistence diagnostic"
    )
    second = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert second is not None

    assert controller._reconcile_to_durable_truth(second) is True
    assert controller.model.registry == durable
    assert controller.model.using_config_path == "old.json"
    assert view.combobox.items == list(controller.model.registry_entries)
    assert view.combobox.currentData() == "old.json"
    assert controller.projection_consistent is True
    assert controller.projection_failure is None
    assert controller.persistence_consistent is True
    assert controller.persistence_failure is None


def test_central_reconciliation_partial_retry_keeps_failure_diagnostics():
    controller, _workflow, view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="RECORD_ONLY")
    )
    durable = {"new": "new.json", "using_config_path": "new.json"}
    controller._registry_loader = lambda: deepcopy(durable)
    controller._config_loader = lambda _path: (0, _sequence("RECORD_ONLY"))
    controller._path_exists = lambda _path: True
    original_populate = view.populate_configuration_entries

    def populate_then_fail(*args, **kwargs):
        original_populate(*args, **kwargs)
        raise RuntimeError("view remained partially unavailable")

    view.populate_configuration_entries = populate_then_fail
    first = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert first is not None
    assert controller._reconcile_to_durable_truth(first) is False
    controller._mark_persistence_inconsistent(
        "registry", "durable state remains unverified"
    )

    second = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert second is not None
    assert controller._reconcile_to_durable_truth(second) is False

    assert controller.projection_consistent is False
    assert controller.projection_failure
    assert controller.persistence_consistent is False
    assert controller.persistence_failure == "durable state remains unverified"
    assert view.capture_action_availability() == (False, False, False)


def test_registry_success_does_not_clear_configuration_file_persistence_failure():
    controller, model, view, combo, durable, _calls, _published = (
        _registry_fallback_cas_controller()
    )
    view.player_button = _Button(True)
    view.replay_button = _Button(True)
    view.data_button = _Button(True)
    controller._mark_persistence_inconsistent(
        "configuration_file",
        "configuration file durability remains uncertain",
    )

    using_path, registry = controller.get_sequence_config_from_registry()

    assert using_path == "fallback.json"
    assert registry == durable["registry"]
    assert model.registry == durable["registry"]
    assert combo.items == list(model.registry_entries)
    assert controller.persistence_consistent is False
    assert controller.persistence_failure == (
        "configuration file durability remains uncertain"
    )
    assert view.capture_action_availability() == (False, False, False)

    controller._config_loader = lambda _path: (0, _sequence("RECORD_ONLY"))
    controller._resolve_input_sample_rate = lambda _mic: None

    controller.get_sequence_config_from_json()

    assert controller.persistence_consistent is False
    assert controller.persistence_failure == (
        "configuration file durability remains uncertain"
    )
    assert view.capture_action_availability() == (False, False, False)


def test_successful_config_write_repairs_only_config_persistence_surface():
    controller, _workflow, view, _data_struct, _warnings = (
        _controller_for_persistence_context_failure()
    )
    durable = {"config": _sequence("RECORD_ONLY")}

    def writer(payload, _target):
        durable["config"] = deepcopy(payload)
        return True

    controller._config_persistence_adapter_override = SimpleNamespace(
        coordinator=controller._path_transaction_coordinator,
        transaction=lambda _target: nullcontext(),
        capture=lambda _target: deepcopy(durable["config"]),
        write=writer,
        conditional_restore=lambda *_args: True,
        ownership_token=_durable_ownership_token,
        read_durable_truth=lambda _target: (0, deepcopy(durable["config"])),
        read_semantic_current=lambda _target: deepcopy(durable["config"]),
    )
    controller._mark_persistence_inconsistent(
        "configuration_file", "configuration file durability remains uncertain"
    )
    controller._mark_persistence_inconsistent(
        "registry", "registry durability remains uncertain"
    )
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None

    assert controller._attempt_config_persistence(
        transaction,
        deepcopy(durable["config"]),
        controller.model.using_config_path,
        semantic_checkpoint=deepcopy(durable["config"]),
    ) is True

    assert controller.persistence_consistent is False
    assert controller._persistence_failures == {
        "registry": "registry durability remains uncertain"
    }
    assert controller.persistence_failure == "registry durability remains uncertain"
    assert view.capture_action_availability() == (False, False, False)


def test_public_consistency_diagnostics_aggregate_surfaces_deterministically():
    view = SequenceConfigurationView(
        player_button=_Button(True),
        replay_button=_Button(True),
        data_button=_Button(True),
    )
    controller = _controller(view=view)
    controller._mark_projection_inconsistent("view", "view split")
    controller._mark_projection_inconsistent("analysis", "analysis split")
    controller._mark_projection_inconsistent("registry", "registry split")
    controller._mark_persistence_inconsistent(
        "configuration_file", "configuration file uncertain"
    )
    controller._mark_persistence_inconsistent(
        "registry", "registry persistence uncertain"
    )

    assert controller.projection_failure == (
        "registry: registry split; analysis: analysis split; view: view split"
    )
    assert controller.persistence_failure == (
        "registry: registry persistence uncertain; "
        "configuration_file: configuration file uncertain"
    )

    controller._finalize_projection_success(
        {"registry"}, registry_persistence_verified=True
    )

    assert controller.projection_failure == (
        "analysis: analysis split; view: view split"
    )
    assert controller.persistence_failure == "configuration file uncertain"
    assert view.capture_action_availability() == (False, False, False)


def test_mixed_durable_compensation_marks_only_failed_owner_surface():
    controller, _workflow, view, _data_struct, _warnings = (
        _controller_for_persistence_context_failure()
    )
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None
    restores = []

    assert transaction.attempt_durable_write(
        "write configuration file",
        lambda: True,
        rollback=lambda _owned: restores.append("configuration_file") or False,
        state_reader=lambda: {"owner": "configuration"},
        surface="configuration_file",
    ) is True
    assert transaction.attempt_durable_write(
        "write registry",
        lambda: True,
        rollback=lambda _owned: restores.append("registry") or True,
        state_reader=lambda: {"owner": "registry"},
        surface="registry",
    ) is True

    assert controller._abort_after_durable_failure(
        transaction, "mixed durable persistence failure"
    ) is False

    assert restores == ["registry", "configuration_file"]
    assert controller._persistence_failures == {
        "configuration_file": "mixed durable persistence failure"
    }
    assert controller.persistence_failure == "mixed durable persistence failure"
    assert view.capture_action_availability() == (False, False, False)


def test_central_reconcile_keeps_failure_when_durable_configuration_changed():
    controller, _workflow, view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="RECORD_ONLY")
    )
    durable = {"new": "new.json", "using_config_path": "new.json"}
    controller._registry_loader = lambda: deepcopy(durable)
    controller._config_loader = lambda _path: (
        0,
        _sequence("PLAY_AND_RECORD"),
    )
    channel_projection = {"mode": "RECORD_ONLY"}
    analysis_projection = {"mode": "RECORD_ONLY"}
    controller._refresh_channels = lambda: channel_projection.__setitem__(
        "mode", controller.model.acquisition_mode
    )
    controller._analysis_config_changed = (
        lambda _config: analysis_projection.__setitem__(
            "mode", controller.model.acquisition_mode
        )
    )
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None

    assert controller._reconcile_to_durable_truth(transaction) is False

    assert controller.model.acquisition_mode == "PLAY_AND_RECORD"
    assert channel_projection == {"mode": "RECORD_ONLY"}
    assert analysis_projection == {"mode": "RECORD_ONLY"}
    assert controller.projection_consistent is False
    assert "configuration" in controller.projection_failure
    assert view.capture_action_availability() == (False, False, False)


def test_json_load_preserves_registry_projection_failure_until_verified_reconcile():
    durable = {
        "durable": "durable.json",
        "using_config_path": "durable.json",
    }
    model = SequenceConfigurationModel(data_struct=SimpleNamespace())
    model.replace_registry(
        durable,
        using_config_path="durable.json",
        entries=(("durable", "durable.json"),),
    )
    combo = _ComboBox()
    view = SequenceConfigurationView(
        combobox=combo,
        player_button=_Button(True),
        replay_button=_Button(True),
        data_button=_Button(True),
    )
    view.populate_configuration_entries(
        (("stale-view", "stale.json"),),
        using_config_path="stale.json",
        clear_first=True,
    )
    controller = _controller(
        model=model,
        view=view,
        registry_loader=lambda: deepcopy(durable),
        config_loader=lambda _path: (0, _sequence("RECORD_ONLY")),
        ok_code=0,
        path_exists=lambda _path: True,
        input_sample_rate_resolver=lambda _mic: None,
    )
    controller._mark_projection_inconsistent(
        "registry", "pre-existing registry projection split"
    )
    controller._mark_persistence_inconsistent(
        "registry", "pre-existing registry persistence failure"
    )
    view.set_sequence_config_available(False)

    controller.get_sequence_config_from_json()

    assert model.registry == durable
    assert model.using_config_path == "durable.json"
    assert combo.items == [("stale-view", "stale.json")]
    assert combo.currentData() == "stale.json"
    assert controller.projection_consistent is False
    assert controller.projection_failure == (
        "pre-existing registry projection split"
    )
    assert controller.persistence_consistent is False
    assert controller.persistence_failure == (
        "pre-existing registry persistence failure"
    )
    assert view.capture_action_availability() == (False, False, False)

    transaction = controller._begin_configuration_transaction(
        selection_path=model.using_config_path
    )
    assert transaction is not None
    assert controller._reconcile_to_durable_truth(transaction) is True

    assert model.registry == durable
    assert model.using_config_path == "durable.json"
    assert combo.items == [("durable", "durable.json")]
    assert combo.currentData() == "durable.json"
    assert controller.projection_consistent is True
    assert controller.projection_failure is None
    assert controller.persistence_consistent is True
    assert controller.persistence_failure is None


def test_config_update_success_does_not_clear_unverified_projection_failure():
    controller, _workflow, view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="RECORD_ONLY")
    )
    channel_projection = {"mode": "stale"}
    channel_refreshes = []

    def refresh_channels():
        channel_refreshes.append(controller.model.acquisition_mode)
        channel_projection["mode"] = controller.model.acquisition_mode

    controller._refresh_channels = refresh_channels
    controller._mark_projection_inconsistent(
        "channels", "pre-existing channel projection failure"
    )
    view.set_sequence_config_available(False)

    assert controller.on_sequence_config_updated() is True

    assert controller.model.acquisition_mode == "RECORD_ONLY"
    assert channel_refreshes == []
    assert channel_projection == {"mode": "stale"}
    assert controller.projection_consistent is False
    assert controller.projection_failure == (
        "pre-existing channel projection failure"
    )
    assert view.capture_action_availability() == (False, False, False)


def _config_update_with_verified_fallback_persistence():
    controller, workflow, view, data_struct, mono, warnings = (
        _config_update_failure_controller(failure="none", mode="IMPORT_AUDIO")
    )
    durable_registry = {
        "new": "new.json",
        "using_config_path": "missing.json",
    }
    controller._registry_loader = lambda: deepcopy(durable_registry)
    controller._path_exists = lambda path: path == "new.json"
    controller._config_loader = lambda path: (
        (0, _sequence("IMPORT_AUDIO"))
        if path == "new.json"
        else (0, _sequence("RECORD_ONLY"))
    )
    controller._using_path_updater = lambda path: (
        durable_registry.__setitem__("using_config_path", path),
        True,
    )[-1]
    controller._using_path_persistence_snapshotter = lambda: deepcopy(
        durable_registry
    )

    def restore_registry(checkpoint):
        durable_registry.clear()
        durable_registry.update(deepcopy(checkpoint))
        return True

    controller._using_path_persistence_restorer = restore_registry
    return (
        controller,
        workflow,
        view,
        data_struct,
        mono,
        warnings,
        durable_registry,
    )


def test_config_update_verified_fallback_repairs_only_registry_persistence():
    controller, _workflow, view, *_rest = (
        _config_update_with_verified_fallback_persistence()
    )
    controller._mark_persistence_inconsistent(
        "registry", "registry durability remains uncertain"
    )
    controller._mark_persistence_inconsistent(
        "configuration_file", "configuration durability remains uncertain"
    )
    controller._mark_projection_inconsistent(
        "analysis", "analysis projection remains uncertain"
    )

    assert controller.on_sequence_config_updated() is True

    assert controller._persistence_failures == {
        "configuration_file": "configuration durability remains uncertain"
    }
    assert controller._projection_failures == {
        "analysis": "analysis projection remains uncertain"
    }
    assert view.capture_action_availability() == (False, False, False)


def test_config_update_verified_fallback_repairs_registry_and_enables_readiness():
    controller, _workflow, view, *_rest = (
        _config_update_with_verified_fallback_persistence()
    )
    controller._mark_persistence_inconsistent(
        "registry", "registry durability remains uncertain"
    )

    assert controller.on_sequence_config_updated() is True

    assert controller.persistence_consistent is True
    assert controller.persistence_failure is None
    assert view.capture_action_availability() == (True, False, False)


def test_config_update_without_path_persistence_does_not_repair_registry_failure():
    controller, _workflow, view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="IMPORT_AUDIO")
    )
    controller._mark_persistence_inconsistent(
        "registry", "registry durability remains uncertain"
    )

    assert controller.on_sequence_config_updated() is True

    assert controller.persistence_consistent is False
    assert controller.persistence_failure == "registry durability remains uncertain"
    assert view.capture_action_availability() == (False, False, False)


@pytest.mark.parametrize("failure_boundary", ("postwrite", "projection"))
def test_config_update_partial_fallback_does_not_repair_registry_failure(
    failure_boundary,
):
    controller, _workflow, view, *_rest = (
        _config_update_with_verified_fallback_persistence()
    )
    controller._mark_persistence_inconsistent(
        "registry", "registry durability remains uncertain"
    )
    if failure_boundary == "postwrite":
        controller._validate_committed_active_path_registry = (
            lambda *_args, **_kwargs: None
        )
    else:
        original_populate = view.populate_configuration_entries
        populate_calls = {"count": 0}

        def fail_committed_projection(*args, **kwargs):
            populate_calls["count"] += 1
            original_populate(*args, **kwargs)
            if populate_calls["count"] == 2:
                raise RuntimeError("committed registry projection failed")

        view.populate_configuration_entries = fail_committed_projection

    assert controller.on_sequence_config_updated() is False

    assert controller.persistence_consistent is False
    assert controller.persistence_failure == "registry durability remains uncertain"
    assert view.capture_action_availability() == (False, False, False)


def _install_deferred_cleanup_failure(controller, boundary, failure):
    def callback(name):
        if name == boundary:
            if isinstance(failure, BaseException):
                raise failure
            return False
        return True

    controller._clear_plot = lambda: callback("plot")
    controller._plot_state_capturer = lambda: {"plot": "old"}
    controller._plot_state_restorer = lambda _state: True
    controller._clear_import_identity = lambda: callback("import")
    controller._import_identity_state_capturer = lambda: {"path": "old.wav"}
    controller._import_identity_state_restorer = lambda _state: True


@pytest.mark.parametrize("boundary", ("plot", "import"))
@pytest.mark.parametrize(
    "failure_factory",
    (
        lambda: None,
        lambda: KeyboardInterrupt("deferred cleanup interrupted"),
        lambda: SystemExit("deferred cleanup exited"),
    ),
    ids=("ordinary", "keyboard-interrupt", "system-exit"),
)
def test_config_update_deferred_cleanup_failure_preserves_entry_provenance(
    boundary,
    failure_factory,
):
    (
        controller,
        _workflow,
        view,
        _data_struct,
        _mono,
        _warnings,
        durable_registry,
    ) = _config_update_with_verified_fallback_persistence()
    published = []
    controller._publish_configuration = published.append
    controller._mark_persistence_inconsistent(
        "registry", "pre-existing registry durability failure"
    )
    controller._mark_persistence_inconsistent(
        "configuration_file", "pre-existing config durability failure"
    )
    controller._mark_projection_inconsistent(
        "analysis", "pre-existing analysis projection failure"
    )
    failure = failure_factory()
    _install_deferred_cleanup_failure(controller, boundary, failure)

    if isinstance(failure, BaseException):
        with pytest.raises(type(failure)) as raised:
            controller.on_sequence_config_updated()
        assert raised.value is failure
    else:
        assert controller.on_sequence_config_updated() is False

    assert durable_registry["using_config_path"] == "missing.json"
    assert controller._persistence_failures == {
        "registry": "pre-existing registry durability failure",
        "configuration_file": "pre-existing config durability failure",
    }
    assert controller._projection_failures == {
        "analysis": "pre-existing analysis projection failure"
    }
    assert view.capture_action_availability() == (False, False, False)
    assert published == []


@pytest.mark.parametrize(
    "failure",
    (
        RuntimeError("final view commit failed"),
        KeyboardInterrupt("final view commit interrupted"),
        SystemExit("final view commit exited"),
    ),
    ids=("ordinary", "keyboard-interrupt", "system-exit"),
)
def test_config_update_final_view_failure_preserves_entry_provenance(failure):
    (
        controller,
        _workflow,
        view,
        _data_struct,
        _mono,
        _warnings,
        durable_registry,
    ) = _config_update_with_verified_fallback_persistence()
    published = []
    controller._publish_configuration = published.append
    controller._mark_persistence_inconsistent(
        "registry", "pre-existing registry durability failure"
    )
    original_present = controller.present_configuration_availability
    calls = {"count": 0}

    def fail_final_present(available):
        calls["count"] += 1
        original_present(available)
        if calls["count"] == 2:
            raise failure

    controller.present_configuration_availability = fail_final_present

    if isinstance(failure, Exception):
        assert controller.on_sequence_config_updated() is False
    else:
        with pytest.raises(type(failure)) as raised:
            controller.on_sequence_config_updated()
        assert raised.value is failure

    assert durable_registry["using_config_path"] == "missing.json"
    assert controller.persistence_failure == (
        "pre-existing registry durability failure"
    )
    assert view.capture_action_availability() == (False, False, False)
    assert published == []


def test_config_update_registry_writer_interrupt_preserves_entry_provenance():
    (
        controller,
        _workflow,
        view,
        _data_struct,
        _mono,
        _warnings,
        durable_registry,
    ) = _config_update_with_verified_fallback_persistence()
    controller._mark_persistence_inconsistent(
        "registry", "pre-existing registry durability failure"
    )
    published = []
    controller._publish_configuration = published.append
    interruption = KeyboardInterrupt("registry writer interrupted")

    def interrupted_writer(path):
        durable_registry["using_config_path"] = path
        raise interruption

    controller._using_path_updater = interrupted_writer

    with pytest.raises(KeyboardInterrupt) as raised:
        controller.on_sequence_config_updated()

    assert raised.value is interruption
    assert durable_registry["using_config_path"] == "missing.json"
    assert controller.persistence_failure == (
        "pre-existing registry durability failure"
    )
    assert view.capture_action_availability() == (False, False, False)
    assert published == []


def test_transaction_failure_provenance_merge_keeps_new_failures_idempotently():
    controller, _workflow, _view, *_rest = (
        _config_update_with_verified_fallback_persistence()
    )
    controller._mark_persistence_inconsistent(
        "registry", "entry registry failure"
    )
    controller._mark_projection_inconsistent(
        "analysis", "entry analysis failure"
    )
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None
    controller._persistence_failure_reasons["registry"] = (
        "new registry failure",
    )
    controller._projection_failure_reasons["analysis"] = (
        "new analysis failure",
    )
    controller._sync_consistency_diagnostics()

    transaction.restore_failure_provenance()
    first_persistence = dict(controller._persistence_failures)
    first_projection = dict(controller._projection_failures)
    transaction.restore_failure_provenance()

    assert controller._persistence_failures == first_persistence == {
        "registry": "entry registry failure; new registry failure"
    }
    assert controller._projection_failures == first_projection == {
        "analysis": "entry analysis failure; new analysis failure"
    }


def _restore_nested_failure_provenance_in_order(order):
    controller, _workflow, _view, *_rest = (
        _config_update_with_verified_fallback_persistence()
    )
    first = "first reason; delimiter is data"
    second = "second reason"
    controller._mark_persistence_inconsistent("registry", first)
    outer = _ConfigurationTransaction(
        controller,
        selection_path=controller.model.using_config_path
    )
    controller._clear_persistence_failure("registry")
    controller._mark_persistence_inconsistent("registry", second)
    inner = _ConfigurationTransaction(
        controller,
        selection_path=controller.model.using_config_path
    )
    controller._clear_persistence_failure("registry")
    controller._mark_persistence_inconsistent("registry", first)
    assert controller._failure_reason_checkpoint_refs == {
        ("persistence", "registry", first): 1,
        ("persistence", "registry", second): 1,
    }

    transactions = {"outer": outer, "inner": inner}
    for name in order:
        transactions[name].restore_failure_provenance()
    for name in reversed(order):
        transactions[name].restore_failure_provenance()

    outer.commit_failure_provenance()
    inner.commit_failure_provenance()
    assert controller._failure_reason_checkpoint_refs == {}

    return controller, first, second


@pytest.mark.parametrize(
    "order",
    (("outer", "inner"), ("inner", "outer")),
)
def test_nested_transaction_provenance_uses_stable_first_seen_order(order):
    controller, first, second = _restore_nested_failure_provenance_in_order(
        order
    )

    assert controller._persistence_failure_reasons == {
        "registry": (first, second)
    }
    assert controller._persistence_failures == {
        "registry": f"{first}; {second}"
    }
    assert controller.persistence_failure == f"{first}; {second}"
    controller._clear_persistence_failure("registry")
    assert controller._failure_reason_ordinals == {}


def test_failure_reason_ordering_is_scoped_by_category_surface_and_controller():
    first_controller, _workflow, _view, *_rest = (
        _config_update_with_verified_fallback_persistence()
    )
    second_controller, _workflow, _view, *_rest = (
        _config_update_with_verified_fallback_persistence()
    )
    shared_reason = "same; exact reason"
    later_reason = "later reason"

    first_controller._mark_persistence_inconsistent("registry", shared_reason)
    first_controller._record_projection_failure("registry", shared_reason)
    first_controller._mark_persistence_inconsistent(
        "configuration_file", shared_reason
    )
    first_controller._clear_persistence_failure("registry")
    first_controller._mark_persistence_inconsistent("registry", later_reason)
    first_controller._mark_persistence_inconsistent("registry", shared_reason)

    second_controller._mark_persistence_inconsistent("registry", later_reason)
    second_controller._mark_persistence_inconsistent("registry", shared_reason)

    assert first_controller._persistence_failure_reasons == {
        "registry": (later_reason, shared_reason),
        "configuration_file": (shared_reason,),
    }
    assert first_controller._projection_failure_reasons == {
        "registry": (shared_reason,)
    }
    assert second_controller._persistence_failure_reasons == {
        "registry": (later_reason, shared_reason)
    }
    assert first_controller._failure_reason_ordinals == {
        ("projection", "registry", shared_reason): 1,
        ("persistence", "configuration_file", shared_reason): 2,
        ("persistence", "registry", later_reason): 3,
        ("persistence", "registry", shared_reason): 4,
    }
    assert second_controller._failure_reason_ordinals == {
        ("persistence", "registry", later_reason): 0,
        ("persistence", "registry", shared_reason): 1,
    }


def test_failure_reason_order_metadata_is_pruned_after_unique_reasons_clear():
    controller, _workflow, _view, *_rest = (
        _config_update_with_verified_fallback_persistence()
    )

    for index in range(10_000):
        controller._record_projection_failure("analysis", f"reason-{index}")
        controller._clear_projection_failure("analysis")

    assert controller._projection_failure_reasons == {}
    assert controller._failure_reason_ordinals == {}
    assert controller._failure_reason_checkpoint_refs == {}


def test_failure_reason_checkpoint_refs_release_on_commit_abort_and_interrupt():
    controller, _workflow, view, *_rest = (
        _config_update_with_verified_fallback_persistence()
    )
    reason = "checkpoint reason"
    key = ("persistence", "registry", reason)
    controller._mark_persistence_inconsistent("registry", reason)
    transaction = controller._begin_configuration_transaction(
        selection_path=controller.model.using_config_path
    )
    assert transaction is not None
    assert controller._failure_reason_checkpoint_refs == {key: 1}

    transaction.abort()

    assert controller._failure_reason_checkpoint_refs == {}
    controller._clear_persistence_failure("registry")
    assert key not in controller._failure_reason_ordinals

    controller._mark_persistence_inconsistent("registry", reason)
    assert controller.on_sequence_config_updated() is True
    assert controller._failure_reason_checkpoint_refs == {}
    assert key not in controller._failure_reason_ordinals

    controller._mark_persistence_inconsistent("registry", reason)
    interruption = KeyboardInterrupt("checkpoint cleanup interrupted")
    view.mark_configuration_loaded = lambda: (_ for _ in ()).throw(
        interruption
    )
    with pytest.raises(KeyboardInterrupt) as raised:
        controller.on_sequence_config_updated()

    assert raised.value is interruption
    assert controller._failure_reason_checkpoint_refs == {}
    controller._clear_persistence_failure("registry")
    assert key not in controller._failure_reason_ordinals


@pytest.mark.parametrize(
    "failure",
    (
        RuntimeError("outer projection failed"),
        KeyboardInterrupt("outer projection interrupted"),
        SystemExit("outer projection exited"),
    ),
)
def test_reentrant_config_update_is_rejected_before_mutation_and_write(failure):
    controller, workflow, view, data_struct, mono, _warnings, durable = (
        _config_update_with_verified_fallback_persistence()
    )
    writes = []
    original_updater = controller._using_path_updater
    controller._using_path_updater = lambda path: (
        writes.append(path),
        original_updater(path),
    )[-1]
    published = []
    controller._publish_configuration = published.append
    nested_results = []
    original_populate = view.populate_configuration_entries
    reentered = {"value": False}

    def reenter_then_fail(*args, **kwargs):
        original_populate(*args, **kwargs)
        if not reentered["value"]:
            reentered["value"] = True
            nested_results.append(controller.on_sequence_config_updated())
            raise failure

    view.populate_configuration_entries = reenter_then_fail

    if isinstance(failure, Exception):
        assert controller.on_sequence_config_updated() is False
    else:
        with pytest.raises(type(failure)) as raised:
            controller.on_sequence_config_updated()
        assert raised.value is failure

    assert nested_results == [False]
    assert writes == []
    assert durable["using_config_path"] == "missing.json"
    assert controller.model.using_config_path == "old.json"
    assert controller.model.acquisition_mode == "RECORD_ONLY"
    assert controller.model.configuration_generation == 4
    assert workflow.configuration_generation == 4
    assert view.combobox.items == [("old", "old.json")]
    assert view.combobox.currentData() == "old.json"
    assert data_struct.store_wave_data is mono
    assert published == []
    assert view.capture_action_availability() == (True, True, True)
    assert controller.persistence_consistent is True
    assert controller.projection_consistent is True
    assert controller._failure_reason_checkpoint_refs == {}
    assert controller._active_configuration_transaction_token is None


@pytest.mark.parametrize(
    ("entry_name", "args", "recording", "expected"),
    (
        (
            "get_sequence_config_from_registry",
            (),
            False,
            None,
        ),
        ("update_using_file_combobox", (), False, None),
        ("get_sequence_config_from_json", (), False, None),
        (
            "on_using_file_combobox_changed",
            ("recording-selection",),
            True,
            None,
        ),
        (
            "on_using_file_combobox_changed",
            ("invalid-selection-index",),
            False,
            None,
        ),
        ("init_data_struct_stimulus_config", (), False, None),
        ("on_sequence_config_updated", (), False, False),
    ),
    ids=(
        "startup-registry",
        "startup-selection-refresh",
        "json-load",
        "combobox-recording",
        "combobox-invalid-index",
        "runtime-initialization",
        "config-update",
    ),
)
def test_reentrant_public_configuration_entries_are_absolute_noops(
    entry_name,
    args,
    recording,
    expected,
):
    controller, _workflow, _view, *_rest = (
        _config_update_with_verified_fallback_persistence()
    )
    touches = []

    class WorkflowProbe:
        @property
        def player_status_flag(self):
            touches.append("workflow.player_status_flag")
            return recording

    class ModelProbe:
        def __getattr__(self, name):
            touches.append(f"model.{name}")
            if name == "_workflow_model":
                return WorkflowProbe()
            if name == "using_config_path":
                return "old.json"
            if name == "registry":
                return {}
            if name == "registry_entries":
                return ()
            if name == "sequence_config":
                return []
            return None

    class BoundaryProbe:
        def __init__(self, boundary):
            self.boundary = boundary

        def __getattr__(self, name):
            def touched(*_args, **_kwargs):
                touches.append(f"{self.boundary}.{name}")
                return False

            return touched

    def callback(name, result=None):
        def touched(*_args, **_kwargs):
            touches.append(name)
            return result

        return touched

    controller.model = ModelProbe()
    controller.view = BoundaryProbe("view")
    controller._logger = BoundaryProbe("logger")
    controller._registry_loader = callback("registry-read", {})
    controller._config_loader = callback("configuration-read", (None, None))
    controller._path_exists = callback("path-read", False)
    controller._using_path_updater = callback("active-path-write", False)
    controller._config_saver = callback("configuration-write", False)
    controller._publish_configuration = callback("event-publish", False)
    controller._warning = callback("warning")
    controller._refresh_channels = callback("channel-refresh")
    controller._analysis_config_changed = callback("analysis-config")
    controller._refresh_test_mode_availability = callback("mode-availability")
    controller._active_configuration_transaction_token = object()

    result = getattr(controller, entry_name)(*args)

    assert result == expected
    assert touches == []


def test_non_reentrant_recording_selection_still_restores_and_warns_once():
    controller, _workflow, view, *_rest = (
        _config_update_with_verified_fallback_persistence()
    )
    controller.model._workflow_model = SimpleNamespace(player_status_flag=True)
    view.combobox.addItem("new", "new.json")
    view.combobox.setCurrentIndex(view.combobox.findData("new.json"))
    warnings = []
    logs = []
    registry_reads = []
    controller._warning = lambda title, message: warnings.append((title, message))
    controller._logger = SimpleNamespace(warning=logs.append)
    controller._registry_loader = lambda: registry_reads.append(True) or {}

    assert controller.on_using_file_combobox_changed("new") is None

    assert view.combobox.currentData() == "old.json"
    assert warnings == [("警告", "正在录音，请稍后...")]
    assert logs == ["已恢复到之前的配置选项"]
    assert registry_reads == []
    assert controller._active_configuration_transaction_token is None


def test_reentrant_guard_is_instance_scoped_and_releases_after_success():
    first, _workflow, first_view, *_first_rest = (
        _config_update_with_verified_fallback_persistence()
    )
    second, _workflow, _second_view, *_second_rest = (
        _config_update_with_verified_fallback_persistence()
    )
    first_published = []
    second_published = []
    first._publish_configuration = first_published.append
    second._publish_configuration = second_published.append
    nested_same = []
    nested_other = []
    original_populate = first_view.populate_configuration_entries
    reentered = {"value": False}

    def reenter_once(*args, **kwargs):
        original_populate(*args, **kwargs)
        if not reentered["value"]:
            reentered["value"] = True
            nested_same.append(first.on_sequence_config_updated())
            nested_other.append(second.on_sequence_config_updated())

    first_view.populate_configuration_entries = reenter_once

    assert first.on_sequence_config_updated() is True
    assert nested_same == [False]
    assert nested_other == [True]
    assert len(first_published) == 1
    assert len(second_published) == 1
    assert first._active_configuration_transaction_token is None
    assert second._active_configuration_transaction_token is None
    assert first._failure_reason_checkpoint_refs == {}
    assert second._failure_reason_checkpoint_refs == {}
    assert first.persistence_consistent is True
    assert first.projection_consistent is True

    assert first.on_sequence_config_updated() is True
    assert len(first_published) == 2


@pytest.mark.parametrize(
    "failure",
    (
        RuntimeError("transaction checkpoint unavailable"),
        KeyboardInterrupt("transaction checkpoint interrupted"),
        SystemExit("transaction checkpoint exited"),
    ),
)
def test_configuration_transaction_factory_failure_releases_reentrant_guard(
    failure,
):
    controller, _workflow, view, *_rest = (
        _config_update_with_verified_fallback_persistence()
    )
    original_capture = view.capture_configuration_state
    calls = {"count": 0}

    def fail_once(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise failure
        return original_capture(*args, **kwargs)

    view.capture_configuration_state = fail_once

    if isinstance(failure, Exception):
        assert controller.on_sequence_config_updated() is False
    else:
        with pytest.raises(type(failure)) as raised:
            controller.on_sequence_config_updated()
        assert raised.value is failure
    assert controller._active_configuration_transaction_token is None
    assert controller._failure_reason_checkpoint_refs == {}
    assert controller.on_sequence_config_updated() is True


def test_config_update_early_view_interrupt_aborts_entire_owner_transaction():
    controller, workflow, view, data_struct, mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="IMPORT_AUDIO")
    )
    published = []
    controller._publish_configuration = published.append
    original_populate = view.populate_configuration_entries
    interruption = KeyboardInterrupt("early configuration view interrupted")
    calls = {"count": 0}

    def mutate_then_interrupt(*args, **kwargs):
        calls["count"] += 1
        original_populate(*args, **kwargs)
        if calls["count"] == 1:
            raise interruption

    view.populate_configuration_entries = mutate_then_interrupt

    with pytest.raises(KeyboardInterrupt) as raised:
        controller.on_sequence_config_updated()

    assert raised.value is interruption
    assert controller.model.using_config_path == "old.json"
    assert controller.model.acquisition_mode == "RECORD_ONLY"
    assert controller.model.configuration_generation == 4
    assert workflow.configuration_generation == 4
    assert view.combobox.items == [("old", "old.json")]
    assert view.combobox.currentData() == "old.json"
    assert data_struct.store_wave_data is mono
    assert published == []
    assert view.capture_action_availability() == (True, True, True)
    assert controller._persistence_failure_reasons == {}
    assert controller._projection_failure_reasons == {}


def test_config_update_mid_channel_exit_aborts_entire_owner_transaction():
    controller, workflow, view, data_struct, mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="IMPORT_AUDIO")
    )
    published = []
    controller._publish_configuration = published.append
    interruption = SystemExit("channel projection exited")
    channel_state = {"mode": "RECORD_ONLY"}
    calls = {"count": 0}

    def project_or_restore_channels():
        calls["count"] += 1
        if calls["count"] == 1:
            channel_state["mode"] = controller.model.acquisition_mode
            raise interruption
        channel_state["mode"] = controller.model.acquisition_mode

    controller._refresh_channels = project_or_restore_channels

    with pytest.raises(SystemExit) as raised:
        controller.on_sequence_config_updated()

    assert raised.value is interruption
    assert controller.model.using_config_path == "old.json"
    assert controller.model.acquisition_mode == "RECORD_ONLY"
    assert controller.model.configuration_generation == 4
    assert workflow.configuration_generation == 4
    assert channel_state == {"mode": "RECORD_ONLY"}
    assert view.combobox.currentData() == "old.json"
    assert data_struct.store_wave_data is mono
    assert published == []
    assert view.capture_action_availability() == (True, True, True)
    assert controller._persistence_failure_reasons == {}
    assert controller._projection_failure_reasons == {}


def test_combobox_mid_analysis_interrupt_aborts_entire_owner_transaction():
    controller, workflow, view, data_struct, mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="IMPORT_AUDIO")
    )
    view.combobox.addItem("new", "new.json")
    view.combobox.setCurrentIndex(view.combobox.findData("new.json"))
    published = []
    controller._publish_configuration = published.append
    interruption = KeyboardInterrupt("analysis projection interrupted")
    calls = {"count": 0}

    def project_or_restore_analysis(_config):
        calls["count"] += 1
        if calls["count"] == 1:
            raise interruption

    controller._analysis_config_changed = project_or_restore_analysis

    with pytest.raises(KeyboardInterrupt) as raised:
        controller.on_using_file_combobox_changed("new")

    assert raised.value is interruption
    assert controller.model.using_config_path == "old.json"
    assert controller.model.acquisition_mode == "RECORD_ONLY"
    assert controller.model.configuration_generation == 4
    assert workflow.configuration_generation == 4
    assert view.combobox.currentData() == "old.json"
    assert data_struct.store_wave_data is mono
    assert published == []
    assert view.capture_action_availability() == (True, True, True)
    assert controller._persistence_failure_reasons == {}
    assert controller._projection_failure_reasons == {}


def test_json_load_mid_view_exit_aborts_entire_owner_transaction():
    controller, workflow, view, data_struct, mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="IMPORT_AUDIO")
    )
    controller._config_loader = lambda _path: (0, _sequence("IMPORT_AUDIO"))
    published = []
    controller._publish_configuration = published.append
    interruption = SystemExit("loaded view projection exited")
    view.mark_configuration_loaded = lambda: (_ for _ in ()).throw(
        interruption
    )

    with pytest.raises(SystemExit) as raised:
        controller.get_sequence_config_from_json()

    assert raised.value is interruption
    assert controller.model.using_config_path == "old.json"
    assert controller.model.acquisition_mode == "RECORD_ONLY"
    assert controller.model.configuration_generation == 4
    assert workflow.configuration_generation == 4
    assert view.combobox.currentData() == "old.json"
    assert data_struct.store_wave_data is mono
    assert published == []
    assert view.capture_action_availability() == (True, True, True)
    assert controller._persistence_failure_reasons == {}
    assert controller._projection_failure_reasons == {}


def test_runtime_initialization_interrupt_aborts_entire_owner_transaction():
    controller, workflow, view, data_struct, mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="PLAY_AND_RECORD")
    )
    controller.model.apply_configuration(
        ConfigurationSnapshot(
            sequence_config=_sequence("PLAY_AND_RECORD"),
            analysis_config={},
            using_config_path="old.json",
        ),
        generation=5,
    )
    controller._resolve_duplex_sample_rate = lambda *_args: SimpleNamespace(
        ok=True, sample_rate=48000, message=""
    )
    interruption = KeyboardInterrupt("runtime preparation interrupted")

    def mutate_staging_then_interrupt(staged, *_args, **_kwargs):
        staged.stimulus_data = np.ones(8)
        raise interruption

    controller._stimulus_setter = mutate_staging_then_interrupt
    published = []
    controller._publish_configuration = published.append
    old_runtime_sample_rate = controller.model.runtime_sample_rate

    with pytest.raises(KeyboardInterrupt) as raised:
        controller.init_data_struct_stimulus_config()

    assert raised.value is interruption
    assert controller.model.configuration_generation == 5
    assert workflow.configuration_generation == 5
    assert controller.model.runtime_sample_rate == old_runtime_sample_rate
    assert data_struct.store_wave_data is mono
    assert data_struct.stimulus_data.shape == (4,)
    assert view.combobox.currentData() == "old.json"
    assert published == []
    assert view.capture_action_availability() == (True, True, True)
    assert controller._persistence_failure_reasons == {}
    assert controller._projection_failure_reasons == {}


def test_verified_active_path_projection_clears_preexisting_projection_failure():
    controller, _workflow, view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="IMPORT_AUDIO")
    )
    view.combobox.addItem("new", "new.json")
    view.combobox.setCurrentIndex(view.combobox.findData("new.json"))
    controller._mark_projection_inconsistent(
        "registry", "pre-existing projection failure"
    )
    view.set_sequence_config_available(False)

    assert controller.on_using_file_combobox_changed("new") is None

    assert controller.model.registry == {
        "new": "new.json",
        "using_config_path": "new.json",
    }
    assert controller.model.using_config_path == "new.json"
    assert view.combobox.items == [("new", "new.json")]
    assert view.combobox.currentData() == "new.json"
    assert controller.projection_consistent is True
    assert controller.projection_failure is None
    assert view.capture_action_availability() == (True, False, False)


@pytest.mark.parametrize("surface", ("analysis", "channels"))
def test_active_path_selection_projects_owned_analysis_flags_and_preserves_incomplete_surface(
    surface,
):
    controller, _workflow, view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="RECORD_ONLY")
    )
    view.combobox.addItem("new", "new.json")
    view.combobox.setCurrentIndex(view.combobox.findData("new.json"))
    analysis_calls = []
    controller._analysis_config_changed = analysis_calls.append
    controller._refresh_channels = None
    controller._mark_projection_inconsistent(
        surface, f"pre-existing {surface} projection failure"
    )
    controller._mark_projection_inconsistent(
        "registry", "pre-existing registry projection failure"
    )

    controller.on_using_file_combobox_changed("new")

    assert analysis_calls == [{}]
    assert controller.model.data_struct.fft_flag == 0
    assert controller.model.data_struct.stft_flag == 0
    assert controller.projection_consistent is False
    assert controller._projection_failures == {
        surface: f"pre-existing {surface} projection failure"
    }
    assert view.capture_action_availability() == (False, False, False)


@pytest.mark.parametrize("surface", ("channels", "runtime", "view"))
def test_registry_only_fallback_preserves_unprojected_surface_failure(surface):
    controller, _model, view, _combo, _durable, _calls, _published = (
        _registry_fallback_cas_controller()
    )
    view.player_button = _Button(True)
    view.replay_button = _Button(True)
    view.data_button = _Button(True)
    controller._mark_projection_inconsistent(
        surface, f"pre-existing {surface} projection failure"
    )

    controller.get_sequence_config_from_registry()

    assert controller._projection_failures == {
        surface: f"pre-existing {surface} projection failure"
    }
    assert view.capture_action_availability() == (False, False, False)


def test_config_update_repairs_only_surfaces_with_completed_owner_projections():
    controller, _workflow, view, _data_struct, _mono, _warnings = (
        _config_update_failure_controller(failure="none", mode="IMPORT_AUDIO")
    )
    calls = []
    controller._refresh_channels = lambda: calls.append("channels")
    controller._analysis_flag_projection_service.project = (
        lambda _config: calls.append("analysis-flags")
    )
    controller._analysis_config_changed = (
        lambda _config: calls.append("analysis-config")
    )
    controller._refresh_test_mode_availability = (
        lambda: calls.append("mode-availability")
    )
    controller._clear_plot = lambda: calls.append("plot") or True
    controller._plot_state_capturer = lambda: {"plot": "old"}
    controller._plot_state_restorer = lambda _state: True
    controller._clear_import_identity = lambda: calls.append("import") or True
    controller._import_identity_state_capturer = lambda: {"path": "old.wav"}
    controller._import_identity_state_restorer = lambda _state: True
    for surface in (
        "registry",
        "configuration",
        "channels",
        "analysis",
        "mode_availability",
        "runtime",
        "plot_import",
        "view",
    ):
        controller._mark_projection_inconsistent(
            surface, f"pre-existing {surface} projection failure"
        )

    assert controller.on_sequence_config_updated() is True

    assert calls == [
        "channels",
        "analysis-flags",
        "analysis-config",
        "mode-availability",
        "plot",
        "import",
    ]
    assert controller.projection_consistent is True
    assert controller.projection_failure is None
    assert view.capture_action_availability() == (True, False, False)


def test_runtime_reference_readiness_requires_matching_positive_sample_rates():
    data_struct = SimpleNamespace(
        sample_rate=32000,
        audio_lenth=320,
        store_wave_data=np.ones(320),
        store_wave_data_multi=np.ones((320, 1)),
        stimulus_data=np.ones(320),
        stimulus_info={"sample_rate": 32000, "total_time": 0.01},
    )
    controller = _controller(model=SequenceConfigurationModel(data_struct=data_struct))

    assert controller.has_imported_recording_runtime_state() is True
    assert controller.has_import_stimulus_runtime_reference() is True
    data_struct.stimulus_info["sample_rate"] = 44100
    assert controller.has_import_stimulus_runtime_reference() is False


def test_facade_projects_configuration_properties_and_explicit_delegates():
    recording_model = RecordingModel()
    model = SequenceConfigurationModel(
        data_struct=SimpleNamespace(sample_rate=None),
        streaming_stimulus_port=RecordingStreamingStimulusPort(recording_model),
    )
    facade = SimpleNamespace(
        configuration_model=model,
        recording_model=recording_model,
    )
    mic = {"name": "mic", "nested": [1]}
    channels = [0, 1]

    SequenceWindow.mic.fset(facade, mic)
    SequenceWindow.speaker.fset(facade, {"name": "speaker"})
    SequenceWindow.mic_channels.fset(facade, channels)
    SequenceWindow.sequence_config.fset(facade, _sequence("RECORD_ONLY"))
    SequenceWindow.analysis_config.fset(facade, {"display_sequence": []})
    SequenceWindow.using_config_path.fset(facade, "sequence.json")
    SequenceWindow.streaming_stimulus_data.fset(facade, np.ones(2))
    mic["nested"].append(2)
    channels.append(2)

    assert SequenceWindow.mic.fget(facade) == {"name": "mic", "nested": [1]}
    assert SequenceWindow.speaker.fget(facade) == {"name": "speaker"}
    assert SequenceWindow.mic_channels.fget(facade) == [0, 1]
    assert SequenceWindow.sequence_config.fget(facade) == _sequence("RECORD_ONLY")
    assert SequenceWindow.analysis_config.fget(facade) == {"display_sequence": []}
    assert SequenceWindow.using_config_path.fget(facade) == "sequence.json"
    assert SequenceWindow.data_struct.fget(facade) is model.data_struct
    assert np.array_equal(SequenceWindow.streaming_stimulus_data.fget(facade), np.ones(2))


def test_facade_configuration_methods_are_small_named_controller_delegates():
    calls = []

    class Controller:
        def __getattr__(self, name):
            return lambda *args, **kwargs: calls.append((name, args, kwargs)) or name

    facade = SimpleNamespace(configuration_controller=Controller())
    delegates = {
        "get_sequence_config_from_registry": (),
        "get_sequence_config_from_json": (),
        "update_using_file_combobox": (),
        "add_file_to_using_file_combobox": (),
        "on_using_file_combobox_changed": ("name",),
        "restore_previous_configuration": (),
        "on_sequence_config_updated": ("ignored",),
        "init_data_struct_stimulus_config": (),
        "_refresh_import_stimulus_analysis_reference": ({},),
        "_has_imported_recording_runtime_state": (),
        "_has_import_stimulus_runtime_reference": (),
        "_validate_import_stimulus_analysis_readiness": (),
        "set_audio_devices_available": (False, "missing"),
    }

    for method_name, args in delegates.items():
        result = getattr(SequenceWindow, method_name)(facade, *args)
        if method_name == "on_sequence_config_updated":
            assert result is None
        else:
            assert result

    assert [call[0] for call in calls] == [
        "get_sequence_config_from_registry",
        "get_sequence_config_from_json",
        "update_using_file_combobox",
        "add_file_to_using_file_combobox",
        "on_using_file_combobox_changed",
        "restore_previous_configuration",
        "on_sequence_config_updated",
        "init_data_struct_stimulus_config",
        "refresh_import_stimulus_analysis_reference",
        "has_imported_recording_runtime_state",
        "has_import_stimulus_runtime_reference",
        "validate_import_stimulus_analysis_readiness",
        "set_audio_devices_available",
    ]

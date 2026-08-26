from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import gc
import os
import pickle
import subprocess
import sys
from threading import Barrier, Event, Thread, current_thread
import weakref

import numpy as np
import pytest
from PyQt5.QtCore import (
    QCoreApplication,
    QEvent,
    QEventLoop,
    QObject,
    QThread,
    QTimer,
    Qt,
    pyqtSignal,
)
from PyQt5.QtWidgets import QApplication
from scipy.io import wavfile

from ui.sequence.sequence_event_bus import SequenceEventBus as _SequenceEventBus
from ui.sequence import sequence_messages as sequence_messages_module
from ui.sequence.sequence_messages import (
    BeginRecordingRequested,
    CancelRecordingRequested,
    ConfigurationSnapshot,
)
from ui.sequence import sequence_recording_controller as recording_controller_module
from ui.sequence import sequence_recording_model as recording_model_module
from ui.sequence.sequence_recording_controller import (
    BlockingRecordingAdapter,
    PreparedRecordingSession,
    RecordingAcquisitionContext,
    SequenceRecordingController,
)
from ui.sequence.sequence_recording_model import (
    RecordingModel,
    RecordingSessionSnapshot,
    RecordingState,
    StagedRecording,
    freeze_recording_session_value,
    thaw_recording_session_value,
)
from ui.sequence.sequence_recording_transaction import RecordingTransaction
from ui.sequence.sequence_recording_view import SequenceRecordingView


_QT_APP = QApplication.instance() or QApplication([])


def SequenceEventBus(parent=None):
    """Build an explicitly authorized standalone Recording test harness."""
    return _SequenceEventBus(
        parent,
        standalone_recording_admission=True,
    )


class _QtProbe(QObject):
    reached = pyqtSignal()


def _wait_for_qt_signal(signal, trigger=lambda: None):
    loop = QEventLoop()
    observed = []

    def finish(*_args):
        observed.append(True)
        loop.quit()

    signal.connect(finish)
    QTimer.singleShot(5_000, loop.quit)
    trigger()
    if not observed:
        loop.exec()
    try:
        signal.disconnect(finish)
    except (RuntimeError, TypeError):
        # The QObject owning ``destroyed`` is intentionally gone.
        pass
    assert observed, "timed out waiting for queued Qt signal"


def test_player_button_routes_recording_through_trigger_and_workflow():
    source_path = Path(__file__).parents[2] / "ui" / "sequence" / "sequence_widget.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    facade = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    method = next(
        node
        for node in facade.body
        if isinstance(node, ast.FunctionDef) and node.name == "on_clicked_player_btn"
    )
    called_attributes = {
        node.func.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "request_start" in called_attributes
    assert "start_this_play" not in called_attributes


def test_facade_routes_workflow_admitted_analysis_to_analysis_controller():
    source_path = Path(__file__).parents[2] / "ui" / "sequence" / "sequence_widget.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    facade = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    init = next(
        node
        for node in facade.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    workflow_call = next(
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SequenceWorkflowController"
    )
    keywords = {item.arg for item in workflow_call.keywords}
    analysis_controller_calls = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SequenceAnalysisController"
    ]
    analysis_wiring = next(
        node
        for node in facade.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_wire_analysis_workflow_channels"
    )
    connected_handlers = {
        node.args[0].attr
        for node in ast.walk(analysis_wiring)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "connect"
        and node.args
        and isinstance(node.args[0], ast.Attribute)
    }

    assert "automatic_analysis_policy" in keywords
    assert analysis_controller_calls
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_wire_analysis_workflow_channels"
        for node in ast.walk(init)
    )
    assert "handle_analysis_requested" in connected_handlers
    assert "_handle_legacy_analysis_requested" not in connected_handlers


def _snapshot(
    tmp_path: Path,
    *,
    replay: bool = False,
    sample_rate: int | float = 48_000,
) -> RecordingSessionSnapshot:
    return RecordingSessionSnapshot.create(
        session_id="session-1",
        workflow_generation=7,
        configuration_generation=3,
        mode="RECORD_ONLY",
        sample_rate=sample_rate,
        bit_depth=32,
        input_channels=[0, 2],
        input_device={"name": "mic", "rates": [48_000]},
        output_device=None,
        stimulus_snapshot={"samples": [0.1, 0.2]},
        target_samples=2,
        output_path=tmp_path / "record.wav",
        temp_path=tmp_path / "record.session.tmp.wav",
        backup_path=(tmp_path / "record.replay.backup.wav") if replay else None,
        record_id="record-1",
        label="not_labeled",
        analysis_config={"auto_analysis": True},
        configuration_snapshot={"name": "frozen"},
        pending_count=None if replay else 12,
        replay=replay,
    )


def _staged(snapshot: RecordingSessionSnapshot) -> StagedRecording:
    multi = np.asarray([[0.1, 0.3], [0.2, 0.4]], dtype=np.float32)
    return StagedRecording.create(
        snapshot=snapshot,
        sample_count=2,
        data_struct_fields={
            "store_wave_data": multi.mean(axis=1),
            "store_wave_data_multi": multi,
            "sample_rate": 48_000,
            "audio_lenth": 2,
        },
        recorded_signal_info={"file_path": str(snapshot.output_path)},
        stimulus_info=None,
    )


def test_recording_session_snapshot_preserves_accepted_sample_rate_numeric_kind(
    tmp_path,
):
    integer_snapshot = _snapshot(tmp_path, sample_rate=48_000)
    historical_float_snapshot = _snapshot(tmp_path, sample_rate=48_000.0)

    assert type(integer_snapshot.sample_rate) is int
    assert integer_snapshot.sample_rate == 48_000
    assert type(historical_float_snapshot.sample_rate) is float
    assert historical_float_snapshot.sample_rate == 48_000.0


def test_recording_session_snapshot_is_frozen_and_detached(tmp_path):
    channels = [0, 2]
    device = {"name": "mic", "rates": [48_000]}
    analysis = {"items": ["FFT"]}

    snapshot = RecordingSessionSnapshot.create(
        session_id="session-1",
        workflow_generation=7,
        configuration_generation=3,
        mode="RECORD_ONLY",
        sample_rate=48_000,
        bit_depth=32,
        input_channels=channels,
        input_device=device,
        output_device=None,
        stimulus_snapshot=None,
        target_samples=96_000,
        output_path=tmp_path / "record.wav",
        temp_path=tmp_path / "record.tmp.wav",
        backup_path=None,
        record_id="record-1",
        label="not_labeled",
        analysis_config=analysis,
        configuration_snapshot={"mode": "RECORD_ONLY"},
        pending_count=4,
        replay=False,
    )

    channels.append(9)
    device["rates"].append(44_100)
    analysis["items"].append("SPL")

    assert snapshot.input_channels == (0, 2)
    assert snapshot.input_device["rates"] == (48_000,)
    assert snapshot.analysis_config["items"] == ("FFT",)
    with pytest.raises(TypeError):
        snapshot.analysis_config["new"] = True


@pytest.mark.parametrize(
    ("field_name", "hostile"),
    [
        ("input_channels", type("HostileChannels", (list,), {})([0])),
        ("output_path", type("HostilePath", (str,), {})("hostile.wav")),
    ],
)
def test_recording_session_snapshot_rejects_non_exact_session_fields(
    tmp_path, field_name, hostile
):
    values = _snapshot(tmp_path).as_dict()
    values[field_name] = hostile

    with pytest.raises(TypeError):
        RecordingSessionSnapshot.create(**values)


def test_recording_session_snapshot_public_constructor_cannot_bypass_factory(
    tmp_path,
):
    with pytest.raises(TypeError, match="create"):
        RecordingSessionSnapshot(**_snapshot(tmp_path).as_dict())


@pytest.mark.parametrize("sealed_type", [RecordingSessionSnapshot, StagedRecording])
def test_recording_boundary_values_reject_hostile_subclasses(sealed_type):
    with pytest.raises(TypeError, match="sealed"):

        class HostileRecordingValue(sealed_type):
            def __init__(self, **values):
                self.__dict__.update(values)


def test_prepared_session_rebuilds_even_a_forged_snapshot_without_aliases(tmp_path):
    values = _snapshot(tmp_path).as_dict()
    device = {"nested": ["original"]}
    stimulus = np.arange(4, dtype=np.float32)
    values["input_device"] = device
    values["stimulus_snapshot"] = {"data": stimulus}
    forged = object.__new__(RecordingSessionSnapshot)
    for name in RecordingSessionSnapshot.__dataclass_fields__:
        if name != "_canonical_marker":
            object.__setattr__(forged, name, values[name])

    prepared = PreparedRecordingSession(forged, {})
    device["nested"].append("mutated")
    stimulus[:] = -1

    assert prepared.snapshot is not forged
    assert prepared.snapshot.input_device["nested"] == ("original",)
    assert prepared.snapshot.stimulus_snapshot["data"].tolist() == [0, 1, 2, 3]
    assert prepared.snapshot.stimulus_snapshot["data"].flags.writeable is False


@pytest.mark.parametrize("hostile_kind", ["mapping", "list", "ndarray"])
def test_prepared_session_strictly_revalidates_poisoned_context_without_hooks(
    tmp_path,
    hostile_kind,
):
    hook_calls = []

    class HostileMapping(dict):
        def items(self):
            hook_calls.append("items")
            return super().items()

    class HostileList(list):
        pass

    class HostileArray(np.ndarray):
        pass

    context = RecordingAcquisitionContext.create({})
    if hostile_kind == "mapping":
        object.__setattr__(context, "detail", HostileMapping({"value": 1}))
    elif hostile_kind == "list":
        object.__setattr__(context, "stimulus_info", HostileList([1, 2]))
    else:
        object.__setattr__(
            context,
            "stimulus_data",
            np.asarray([1.0, 2.0]).view(HostileArray),
        )

    with pytest.raises(TypeError, match="exact|unsupported"):
        PreparedRecordingSession(_snapshot(tmp_path), context)

    assert hook_calls == []


def test_recording_context_rejects_hostile_schema_key_without_hash_or_eq_hooks():
    hook_calls = []

    class HostileKey:
        def __hash__(self):
            hook_calls.append("hash")
            return 1

        def __eq__(self, _other):
            hook_calls.append("eq")
            return False

    hostile_key = HostileKey()
    values = {hostile_key: {"nested": [1]}}
    hook_calls.clear()

    with pytest.raises(TypeError, match="field names|exact text"):
        RecordingAcquisitionContext.create(values)

    assert hook_calls == []


@pytest.mark.parametrize(
    "source",
    [
        np.empty(0, dtype=np.float64),
        np.asarray([object()], dtype=object),
        np.asarray([(1, 2.0)], dtype=[("left", "i4"), ("right", "f8")]),
    ],
    ids=["zero-sized", "object", "structured"],
)
def test_recording_value_stable_snapshot_validates_numpy_dtype_edges(source):
    if source.dtype.hasobject or source.dtype.fields is not None:
        with pytest.raises(TypeError, match="dtype|numeric"):
            freeze_recording_session_value({"source": source})
        return

    frozen = freeze_recording_session_value({"source": source})

    assert frozen["source"].shape == (0,)
    assert frozen["source"].flags.writeable is False


def test_recording_value_rejects_concurrent_array_mutation_barrier_without_torn_result(
    monkeypatch,
):
    source = np.zeros(4_000_000, dtype=np.int64)
    original_freeze = recording_model_module._freeze_payload
    captures = []

    def freeze_then_cross_mutation_barrier(*args, **kwargs):
        frozen = original_freeze(*args, **kwargs)
        captures.append(frozen)
        if len(captures) == 1:
            source.fill(1)
        return frozen

    monkeypatch.setattr(
        recording_model_module,
        "_freeze_payload",
        freeze_then_cross_mutation_barrier,
    )

    with pytest.raises(ValueError, match="changed|stable"):
        freeze_recording_session_value(
            {"first": source, "same-source-alias": source}
        )

    assert len(captures) == 2
    assert np.count_nonzero(captures[0]["first"]) == 0
    assert np.count_nonzero(captures[1]["first"]) == source.size


def test_recording_context_captures_related_array_aliases_as_one_stable_graph(
    monkeypatch,
):
    source = np.zeros(1_000_000, dtype=np.int32)
    original_freeze = recording_model_module._freeze_payload
    captures = []

    def mutate_between_independent_field_captures(*args, **kwargs):
        frozen = original_freeze(*args, **kwargs)
        captures.append(frozen)
        if len(captures) == 2:
            source.fill(1)
        return frozen

    monkeypatch.setattr(
        recording_model_module,
        "_freeze_payload",
        mutate_between_independent_field_captures,
    )

    context = RecordingAcquisitionContext.create(
        {
            "detail": {"related": source},
            "stimulus_data": source,
        }
    )

    np.testing.assert_array_equal(
        context["detail"]["related"],
        context["stimulus_data"],
    )


def test_recording_snapshot_captures_related_array_aliases_as_one_stable_graph(
    tmp_path,
    monkeypatch,
):
    source = np.zeros(1_000_000, dtype=np.int32)
    input_device = {"related": source}
    original_freeze = recording_model_module._freeze_payload
    input_device_captures = []

    def mutate_after_input_device_capture(*args, **kwargs):
        frozen = original_freeze(*args, **kwargs)
        if args[0] is input_device:
            input_device_captures.append(frozen)
            if len(input_device_captures) == 2:
                source.fill(1)
        return frozen

    monkeypatch.setattr(
        recording_model_module,
        "_freeze_payload",
        mutate_after_input_device_capture,
    )
    values = _snapshot(tmp_path).as_dict()
    values["input_device"] = input_device
    values["stimulus_snapshot"] = {"related": source}

    snapshot = RecordingSessionSnapshot.create(**values)

    np.testing.assert_array_equal(
        snapshot.input_device["related"],
        snapshot.stimulus_snapshot["related"],
    )


def test_recording_snapshot_detaches_poisoned_configuration_snapshot_graph(
    tmp_path,
):
    source = np.arange(6, dtype=np.float32)
    mutable_configuration = {
        "nested": {"labels": ["original"], "samples": source}
    }
    configuration = ConfigurationSnapshot({}, {})
    object.__setattr__(
        configuration,
        "sequence_config",
        mutable_configuration,
    )
    values = _snapshot(tmp_path).as_dict()
    values["configuration_snapshot"] = configuration

    snapshot = RecordingSessionSnapshot.create(**values)
    frozen_configuration = snapshot.configuration_snapshot
    mutable_configuration["nested"]["labels"].append("mutated")
    source[:] = -1
    object.__setattr__(configuration, "analysis_config", {"poisoned": True})

    assert type(frozen_configuration) is ConfigurationSnapshot
    assert frozen_configuration is not configuration
    assert frozen_configuration.sequence_config["nested"]["labels"] == (
        "original",
    )
    np.testing.assert_array_equal(
        frozen_configuration.sequence_config["nested"]["samples"],
        np.arange(6, dtype=np.float32),
    )
    assert (
        frozen_configuration.sequence_config["nested"]["samples"].flags.writeable
        is False
    )
    assert frozen_configuration.analysis_config == {}


def test_recording_configuration_capture_preserves_detached_array_alias_graph(
    tmp_path,
):
    source = np.arange(8, dtype=np.int32)
    configuration = ConfigurationSnapshot({}, {})
    object.__setattr__(configuration, "sequence_config", {"samples": source})
    object.__setattr__(configuration, "analysis_config", {"samples": source})
    values = _snapshot(tmp_path).as_dict()
    values["input_device"] = {"samples": source}
    values["configuration_snapshot"] = configuration

    snapshot = RecordingSessionSnapshot.create(**values)
    frozen_configuration = snapshot.configuration_snapshot
    sequence_samples = frozen_configuration.sequence_config["samples"]
    analysis_samples = frozen_configuration.analysis_config["samples"]
    source[:] = -1

    assert sequence_samples is analysis_samples
    assert sequence_samples is snapshot.input_device["samples"]
    assert sequence_samples is not source
    assert sequence_samples.flags.writeable is False
    np.testing.assert_array_equal(sequence_samples, np.arange(8, dtype=np.int32))


def test_recording_capture_detaches_immutable_array_shell_and_shares_only_bytes():
    source_configuration = ConfigurationSnapshot(
        {"samples": np.arange(12, dtype=np.int32)},
        {},
    )
    source = source_configuration.sequence_config["samples"]

    frozen = freeze_recording_session_value(
        {"first": source, "same-source-alias": source}
    )
    detached = frozen["first"]

    assert detached is frozen["same-source-alias"]
    assert detached is not source
    assert np.ndarray.base.__get__(detached) is np.ndarray.base.__get__(source)
    np.ndarray.shape.__set__(detached, (3, 4))
    np.ndarray.dtype.__set__(detached, np.dtype("int16"))
    assert detached.shape == (3, 8)
    assert detached.dtype == np.dtype("int16")
    assert source.shape == (12,)
    assert source.dtype == np.dtype("int32")


@pytest.mark.parametrize(
    ("source_factory", "expected"),
    [
        (
            lambda root: root,
            np.arange(12, dtype=np.float64).reshape(3, 4),
        ),
        (
            lambda root: root[:, ::2],
            np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2],
        ),
        (
            lambda root: root.T,
            np.arange(12, dtype=np.float64).reshape(3, 4).T,
        ),
        (
            lambda _root: ConfigurationSnapshot(
                {"samples": np.empty((0, 3), dtype=np.int16)}, {}
            ).sequence_config["samples"],
            np.empty((0, 3), dtype=np.int16),
        ),
    ],
    ids=["root", "strided-view", "transposed-view", "empty"],
)
def test_recording_array_shell_detachment_handles_views_empty_and_dtypes(
    source_factory,
    expected,
):
    root = ConfigurationSnapshot(
        {"samples": np.arange(12, dtype=np.float64).reshape(3, 4)},
        {},
    ).sequence_config["samples"]
    source = source_factory(root)

    detached = freeze_recording_session_value({"samples": source})["samples"]

    assert detached is not source
    assert detached.flags.writeable is False
    assert detached.flags.c_contiguous is True
    np.testing.assert_array_equal(detached, expected)
    original_shape = source.shape
    np.ndarray.shape.__set__(detached, (detached.size,))
    assert source.shape == original_shape


def test_recording_array_detachment_is_bounded_and_does_not_copy_immutable_bytes(
    monkeypatch,
):
    source = ConfigurationSnapshot(
        {"samples": np.arange(1_000_000, dtype=np.float32)},
        {},
    ).sequence_config["samples"]
    built_shells = []
    copied_arrays = []
    original_build = sequence_messages_module._build_immutable_numpy_array
    original_copy = sequence_messages_module._copy_numpy_array_to_immutable

    def track_build(**kwargs):
        result = original_build(**kwargs)
        built_shells.append(result)
        return result

    def track_copy(value, *, validate_provenance):
        copied_arrays.append(value)
        return original_copy(value, validate_provenance=validate_provenance)

    monkeypatch.setattr(
        sequence_messages_module,
        "_build_immutable_numpy_array",
        track_build,
    )
    monkeypatch.setattr(
        sequence_messages_module,
        "_copy_numpy_array_to_immutable",
        track_copy,
    )

    frozen = freeze_recording_session_value(
        {"first": source, "alias": source, "nested": [source]}
    )

    assert len(built_shells) == 2
    assert copied_arrays == []
    assert frozen["first"] is frozen["alias"]
    assert frozen["first"] is frozen["nested"][0]
    assert np.ndarray.base.__get__(frozen["first"]) is np.ndarray.base.__get__(
        source
    )


def test_detached_array_shell_keeps_shared_bytes_alive_after_other_owner_gc():
    source = ConfigurationSnapshot(
        {"samples": np.arange(32, dtype=np.int64)},
        {},
    ).sequence_config["samples"]
    source_ref = weakref.ref(source)
    detached = freeze_recording_session_value(source)
    shared_bytes = np.ndarray.base.__get__(detached)

    source = None
    gc.collect()

    assert source_ref() is None
    assert np.ndarray.base.__get__(detached) is shared_bytes
    np.testing.assert_array_equal(detached, np.arange(32, dtype=np.int64))


@pytest.mark.parametrize(
    "poisoned_field",
    ["mapping", "list", "ndarray", "cycle"],
)
def test_recording_snapshot_strictly_revalidates_configuration_subtree_without_hooks(
    tmp_path,
    poisoned_field,
):
    hook_calls = []

    class HostileMapping(dict):
        def items(self):
            hook_calls.append("items")
            return super().items()

    class HostileList(list):
        pass

    class HostileArray(np.ndarray):
        pass

    if poisoned_field == "mapping":
        poisoned = HostileMapping({"value": 1})
    elif poisoned_field == "list":
        poisoned = {"value": HostileList([1, 2])}
    elif poisoned_field == "ndarray":
        poisoned = {"value": np.arange(2).view(HostileArray)}
    else:
        poisoned = {}
        poisoned["self"] = poisoned
    configuration = ConfigurationSnapshot({}, {})
    object.__setattr__(configuration, "sequence_config", poisoned)
    values = _snapshot(tmp_path).as_dict()
    values["configuration_snapshot"] = configuration

    with pytest.raises(TypeError, match="exact|unsupported|cyclic"):
        RecordingSessionSnapshot.create(**values)

    assert hook_calls == []


@pytest.mark.parametrize(
    "round_trip",
    [copy.copy, copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))],
    ids=["copy", "deepcopy", "pickle"],
)
def test_recording_snapshot_rebuilds_serialized_configuration_without_aliases(
    tmp_path,
    round_trip,
):
    configuration = ConfigurationSnapshot({}, {})
    object.__setattr__(
        configuration,
        "sequence_config",
        {"nested": {"labels": ["serialized"]}},
    )
    object.__setattr__(configuration, "analysis_config", {"threshold": 1.0})
    configuration = round_trip(configuration)
    poisoned = configuration.sequence_config
    values = _snapshot(tmp_path).as_dict()
    values["configuration_snapshot"] = configuration

    snapshot = RecordingSessionSnapshot.create(**values)
    poisoned["nested"]["labels"].append("mutated")

    assert snapshot.configuration_snapshot is not configuration
    assert snapshot.configuration_snapshot.sequence_config["nested"]["labels"] == (
        "serialized",
    )


def test_recording_configuration_stable_capture_rejects_concurrent_mutation(
    monkeypatch,
):
    source = np.zeros(1_000_000, dtype=np.int32)
    configuration = ConfigurationSnapshot({}, {})
    object.__setattr__(configuration, "sequence_config", {"samples": source})
    original_freeze = recording_model_module._freeze_payload
    captures = []

    def mutate_between_bounded_whole_graph_captures(*args, **kwargs):
        result = original_freeze(*args, **kwargs)
        captures.append(result)
        if len(captures) == 1:
            source.fill(1)
        return result

    monkeypatch.setattr(
        recording_model_module,
        "_freeze_payload",
        mutate_between_bounded_whole_graph_captures,
    )

    with pytest.raises(ValueError, match="changed|stable"):
        freeze_recording_session_value({"configuration": configuration})

    assert len(captures) == 2


def test_recording_configuration_capture_releases_original_mutable_graph():
    source = np.arange(16, dtype=np.float64)
    source_ref = weakref.ref(source)
    mutable_configuration = {"samples": source}
    configuration = ConfigurationSnapshot({}, {})
    object.__setattr__(
        configuration,
        "sequence_config",
        mutable_configuration,
    )

    frozen = freeze_recording_session_value({"configuration": configuration})
    source = None
    mutable_configuration = None
    configuration = None
    gc.collect()

    assert source_ref() is None
    np.testing.assert_array_equal(
        frozen["configuration"].sequence_config["samples"],
        np.arange(16, dtype=np.float64),
    )


def test_recording_snapshot_message_payload_detaches_configuration_shell(
    tmp_path,
):
    values = _snapshot(tmp_path).as_dict()
    values["configuration_snapshot"] = ConfigurationSnapshot(
        {"mode": "RECORD_ONLY"},
        {
            "nested": {"labels": ["original"]},
            "samples": np.arange(8, dtype=np.int32),
        },
    )
    snapshot = RecordingSessionSnapshot.create(**values)

    payload = snapshot.as_message_payload()
    payload_configuration = payload["configuration_snapshot"]
    payload_samples = payload_configuration.analysis_config["samples"]
    model_samples = snapshot.configuration_snapshot.analysis_config["samples"]
    object.__setattr__(
        payload_configuration,
        "analysis_config",
        {"nested": {"labels": ["event-poison"]}},
    )

    assert payload_configuration is not snapshot.configuration_snapshot
    assert payload_samples is not model_samples
    assert np.ndarray.base.__get__(payload_samples) is np.ndarray.base.__get__(
        model_samples
    )
    np.ndarray.shape.__set__(payload_samples, (2, 4))
    np.ndarray.dtype.__set__(payload_samples, np.dtype("int16"))
    assert model_samples.shape == (8,)
    assert model_samples.dtype == np.dtype("int32")
    assert snapshot.configuration_snapshot.analysis_config["nested"][
        "labels"
    ] == ("original",)


def test_prepared_session_canonicalize_rebuilds_copy_and_pickled_state(tmp_path):
    state = pickle.loads(
        pickle.dumps(
            {
                "detail": {"nested": ["original"]},
                "stimulus_data": np.asarray([0.1, 0.2], dtype=np.float32),
            }
        )
    )
    context = copy.copy(RecordingAcquisitionContext.create({}))
    object.__setattr__(context, "detail", state["detail"])
    object.__setattr__(context, "stimulus_data", state["stimulus_data"])
    forged = object.__new__(PreparedRecordingSession)
    object.__setattr__(forged, "snapshot", copy.copy(_snapshot(tmp_path)))
    object.__setattr__(forged, "acquisition_context", context)

    rebuilt = PreparedRecordingSession.canonicalize(forged)
    state["detail"]["nested"].append("mutated")
    state["stimulus_data"][:] = -1

    assert rebuilt is not forged
    assert rebuilt.acquisition_context["detail"]["nested"] == ("original",)
    np.testing.assert_allclose(
        rebuilt.acquisition_context["stimulus_data"], [0.1, 0.2]
    )


def test_controller_canonicalizes_prepared_before_deferred_blocking_worker(
    tmp_path,
):
    snapshot = _snapshot(tmp_path)
    snapshot.temp_path.write_bytes(b"pending")
    device = {"nested": ["original"]}
    detail = {"nested": ["original"]}
    poisoned_snapshot = copy.copy(snapshot)
    object.__setattr__(poisoned_snapshot, "input_device", device)
    context = RecordingAcquisitionContext.create({})
    object.__setattr__(context, "detail", detail)
    returned = object.__new__(PreparedRecordingSession)
    object.__setattr__(returned, "snapshot", poisoned_snapshot)
    object.__setattr__(returned, "acquisition_context", context)
    entered = Event()
    release = Event()
    observed = []

    def acquire(prepared):
        entered.set()
        assert release.wait(5)
        observed.append(prepared)
        return _staged(prepared.snapshot)

    bus = SequenceEventBus()
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda _command: returned,
        blocking_acquirer=acquire,
        transaction_factory=lambda _prepared: RecordingTransaction(
            data_struct=SimpleNamespace()
        ),
        workflow_generation_provider=lambda: 7,
        connect_queued=True,
    )
    command = BeginRecordingRequested(
        "command-prepared-copy",
        snapshot.session_id,
        False,
        {"workflow_generation": 7},
    )

    assert controller.handle_begin_recording(command) is True
    assert entered.wait(5)
    assert controller._session.prepared is not returned
    device["nested"].append("mutated")
    detail["nested"].append("mutated")
    object.__setattr__(returned, "snapshot", _snapshot(tmp_path, replay=True))
    release.set()
    worker = controller._session.blocking_job
    worker._thread.join(5)
    assert worker.is_alive() is False
    _wait_for_qt_signal(bus.events.recording_completed)

    prepared = observed[0]
    assert prepared.snapshot.input_device["nested"] == ("original",)
    assert prepared.acquisition_context["detail"]["nested"] == ("original",)


def test_controller_rejects_hostile_prepared_before_model_or_started_event(tmp_path):
    hook_calls = []

    class HostileMapping(dict):
        def items(self):
            hook_calls.append("items")
            return super().items()

    context = RecordingAcquisitionContext.create({})
    object.__setattr__(context, "detail", HostileMapping({"value": 1}))
    returned = object.__new__(PreparedRecordingSession)
    object.__setattr__(returned, "snapshot", _snapshot(tmp_path))
    object.__setattr__(returned, "acquisition_context", context)
    bus = SequenceEventBus()
    started = []
    failed = []
    bus.events.recording_started.connect(started.append)
    bus.events.recording_failed.connect(failed.append)
    model = RecordingModel()
    controller = SequenceRecordingController(
        model,
        bus,
        prepare_session=lambda _command: returned,
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )

    assert controller.handle_begin_recording(
        BeginRecordingRequested(
            "command-hostile-prepared",
            "session-1",
            False,
            {"workflow_generation": 7},
        )
    ) is False
    assert hook_calls == []
    assert started == []
    assert len(failed) == 1
    assert model.state is RecordingState.IDLE
    assert controller._session is None


def test_controller_canonicalizes_prepared_before_streaming_adapter_capture(
    tmp_path,
):
    snapshot = _snapshot(tmp_path)
    device = {"nested": ["original"]}
    detail = {"nested": ["original"]}
    poisoned_snapshot = copy.copy(snapshot)
    object.__setattr__(poisoned_snapshot, "input_device", device)
    context = RecordingAcquisitionContext.create({})
    object.__setattr__(context, "detail", detail)
    returned = object.__new__(PreparedRecordingSession)
    object.__setattr__(returned, "snapshot", poisoned_snapshot)
    object.__setattr__(returned, "acquisition_context", context)
    observed = []
    controller = SequenceRecordingController(
        RecordingModel(),
        SequenceEventBus(),
        prepare_session=lambda _command: returned,
        use_streaming=lambda _prepared: True,
        streaming_adapter=lambda prepared, _terminal: observed.append(prepared)
        or True,
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )

    assert controller.handle_begin_recording(
        BeginRecordingRequested(
            "command-prepared-stream",
            snapshot.session_id,
            False,
            {"workflow_generation": 7},
        )
    ) is True
    assert controller._session.prepared is not returned
    device["nested"].append("mutated")
    detail["nested"].append("mutated")
    object.__setattr__(returned, "snapshot", _snapshot(tmp_path, replay=True))

    prepared = observed[0]
    assert prepared is not returned
    assert prepared.snapshot.input_device["nested"] == ("original",)
    assert prepared.acquisition_context["detail"]["nested"] == ("original",)
    controller.disconnect()


@pytest.mark.parametrize("streaming", [False, True])
def test_acquisition_policy_cannot_poison_worker_owned_prepared(tmp_path, streaming):
    values = _snapshot(tmp_path).as_dict()
    values["configuration_snapshot"] = ConfigurationSnapshot(
        {"mode": "RECORD_ONLY"},
        {
            "nested": {"labels": ["original"]},
            "samples": np.arange(8, dtype=np.int32),
        },
    )
    snapshot = RecordingSessionSnapshot.create(**values)
    snapshot.temp_path.write_bytes(b"pending")
    returned = PreparedRecordingSession(
        snapshot,
        RecordingAcquisitionContext.create(
            {"detail": {"nested": ["original"]}}
        ),
    )
    policy_values = []
    worker_values = []
    worker_entered = Event()
    bus = SequenceEventBus()

    def use_streaming(policy_prepared):
        policy_values.append(policy_prepared)
        policy_samples = (
            policy_prepared.snapshot.configuration_snapshot.analysis_config[
                "samples"
            ]
        )
        object.__setattr__(
            policy_prepared.snapshot,
            "input_device",
            {"nested": ["policy-poison"]},
        )
        object.__setattr__(
            policy_prepared.acquisition_context,
            "detail",
            {"nested": ["policy-poison"]},
        )
        object.__setattr__(
            policy_prepared.snapshot.configuration_snapshot,
            "analysis_config",
            {"nested": {"labels": ["policy-poison"]}},
        )
        np.ndarray.shape.__set__(policy_samples, (2, 4))
        np.ndarray.dtype.__set__(policy_samples, np.dtype("int16"))
        return streaming

    def acquire(worker_prepared):
        worker_values.append(worker_prepared)
        worker_entered.set()
        return _staged(worker_prepared.snapshot)

    def start_streaming(worker_prepared, terminal):
        worker_values.append(worker_prepared)
        worker_entered.set()
        terminal.staged_recording_ready(_staged(worker_prepared.snapshot))
        return True

    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda _command: returned,
        use_streaming=use_streaming,
        streaming_adapter=start_streaming,
        blocking_acquirer=acquire,
        transaction_factory=lambda _prepared: RecordingTransaction(
            data_struct=SimpleNamespace()
        ),
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )

    assert controller.handle_begin_recording(
        BeginRecordingRequested(
            f"command-policy-{streaming}",
            snapshot.session_id,
            False,
            {"workflow_generation": 7},
        )
    ) is True
    assert worker_entered.wait(5)

    assert len(policy_values) == 1
    assert len(worker_values) == 1
    assert policy_values[0] is not worker_values[0]
    assert worker_values[0] is not returned
    assert (
        policy_values[0].snapshot.configuration_snapshot
        is not worker_values[0].snapshot.configuration_snapshot
    )
    assert worker_values[0].snapshot.input_device["name"] == "mic"
    assert worker_values[0].snapshot.configuration_snapshot.analysis_config[
        "nested"
    ]["labels"] == ("original",)
    worker_samples = worker_values[0].snapshot.configuration_snapshot.analysis_config[
        "samples"
    ]
    assert worker_samples.shape == (8,)
    assert worker_samples.dtype == np.dtype("int32")
    assert worker_values[0].acquisition_context["detail"]["nested"] == (
        "original",
    )
    controller.disconnect()


def test_qthread_prepared_copy_is_detached_from_policy_configuration_poisoning(
    tmp_path,
):
    values = _snapshot(tmp_path).as_dict()
    values["configuration_snapshot"] = ConfigurationSnapshot(
        {"mode": "RECORD_ONLY"},
        {
            "nested": {"labels": ["original"]},
            "samples": np.arange(8, dtype=np.float32),
        },
    )
    prepared = PreparedRecordingSession(
        RecordingSessionSnapshot.create(**values),
        {},
    )
    policy_copy = PreparedRecordingSession.canonicalize(prepared)
    policy_samples = policy_copy.snapshot.configuration_snapshot.analysis_config[
        "samples"
    ]
    object.__setattr__(
        policy_copy.snapshot.configuration_snapshot,
        "analysis_config",
        {"nested": {"labels": ["policy-poison"]}},
    )
    np.ndarray.shape.__set__(policy_samples, (2, 4))

    class Canonicalizer(QObject):
        finished = pyqtSignal(object, object)

        def run(self):
            try:
                self.finished.emit(
                    PreparedRecordingSession.canonicalize(prepared),
                    None,
                )
            except BaseException as error:
                self.finished.emit(None, error)

    worker = Canonicalizer()
    thread = QThread()
    worker.moveToThread(thread)
    observed = []
    worker.finished.connect(
        lambda result, error: observed.append((result, error)),
        Qt.DirectConnection,
    )
    worker.finished.connect(thread.quit, Qt.DirectConnection)
    thread.started.connect(worker.run)
    thread.start()
    assert thread.wait(5_000)

    assert len(observed) == 1
    worker_copy, error = observed[0]
    assert error is None
    assert (
        worker_copy.snapshot.configuration_snapshot
        is not policy_copy.snapshot.configuration_snapshot
    )
    assert worker_copy.snapshot.configuration_snapshot.analysis_config[
        "nested"
    ]["labels"] == ("original",)
    worker_samples = worker_copy.snapshot.configuration_snapshot.analysis_config[
        "samples"
    ]
    assert worker_samples is not policy_samples
    assert worker_samples.shape == (8,)


def test_staged_recording_public_constructor_cannot_bypass_factory(tmp_path):
    with pytest.raises(TypeError, match="create"):
        StagedRecording(
            snapshot=_snapshot(tmp_path),
            sample_count=1,
            data_struct_fields={"store_wave_data": np.ones(1)},
            recorded_signal_info={},
        )


def test_staged_recording_rejects_unsupported_mutable_leaf(tmp_path):
    with pytest.raises(TypeError, match="unsupported"):
        StagedRecording.create(
            snapshot=_snapshot(tmp_path),
            sample_count=1,
            data_struct_fields={"unsupported": object()},
            recorded_signal_info={},
        )


def test_staged_canonicalize_rebuilds_marker_and_serialized_state_forgery(
    tmp_path,
):
    canonical = _staged(_snapshot(tmp_path))
    serialized_fields = thaw_recording_session_value(
        canonical.data_struct_fields
    )
    source = np.asarray([0.1, 0.2], dtype=np.float32)
    serialized_fields["store_wave_data"] = source
    copied = copy.copy(canonical)
    copied_snapshot = copy.copy(canonical.snapshot)
    device = {"nested": ["original"]}
    object.__setattr__(copied_snapshot, "input_device", device)
    object.__setattr__(copied, "snapshot", copied_snapshot)
    object.__setattr__(copied, "data_struct_fields", serialized_fields)
    object.__setattr__(
        copied,
        "_canonical_marker",
        object.__getattribute__(canonical, "_canonical_marker"),
    )

    rebuilt = StagedRecording.canonicalize(copied)
    source[:] = -1
    device["nested"].append("mutated")
    serialized_fields["audio_lenth"] = 999

    assert rebuilt is not copied
    np.testing.assert_allclose(
        rebuilt.data_struct_fields["store_wave_data"], [0.1, 0.2]
    )
    assert rebuilt.data_struct_fields["audio_lenth"] == 2
    assert rebuilt.snapshot.input_device["nested"] == ("original",)


@pytest.mark.parametrize("hostile_kind", ["mapping", "list", "ndarray"])
def test_staged_canonicalize_rejects_hostile_container_subclasses(
    tmp_path,
    hostile_kind,
):
    hook_calls = []

    class HostileMapping(dict):
        def items(self):
            hook_calls.append("items")
            return super().items()

    class HostileList(list):
        pass

    class HostileArray(np.ndarray):
        pass

    staged = _staged(_snapshot(tmp_path))
    if hostile_kind == "mapping":
        hostile_fields = HostileMapping(staged.data_struct_fields)
    elif hostile_kind == "list":
        hostile_fields = {"hostile": HostileList([1, 2])}
    else:
        hostile_fields = {
            "hostile": np.asarray([1.0, 2.0]).view(HostileArray)
        }
    object.__setattr__(staged, "data_struct_fields", hostile_fields)

    with pytest.raises(TypeError, match="exact|unsupported"):
        StagedRecording.canonicalize(staged)

    assert hook_calls == []


def test_queued_blocking_result_is_copied_before_qt_delivery_and_releases_aliases(
    tmp_path,
):
    snapshot = _snapshot(tmp_path)
    snapshot.temp_path.write_bytes(b"pending")
    source = np.asarray([0.1, 0.2], dtype=np.float32)
    source_ref = weakref.ref(source)
    fields = {
        "store_wave_data": source,
        "store_wave_data_multi": np.broadcast_to(source.reshape(-1, 1), (2, 2)),
        "audio_lenth": 2,
        "nested": {"labels": ["original"]},
    }
    info = {"metadata": {"serial": "SN-1"}}
    poisoned = StagedRecording.create(
        snapshot=snapshot,
        sample_count=2,
        data_struct_fields={
            "store_wave_data": np.zeros(2),
            "store_wave_data_multi": np.zeros((2, 2)),
            "audio_lenth": 2,
        },
        recorded_signal_info={},
    )
    object.__setattr__(poisoned, "data_struct_fields", fields)
    object.__setattr__(poisoned, "recorded_signal_info", info)
    pending_result = [poisoned]
    presented = []
    terminals = []

    def acquire(_prepared):
        return pending_result.pop()

    bus = SequenceEventBus()
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(commit_identity=presented.append),
        prepare_session=lambda _command: PreparedRecordingSession(snapshot, {}),
        blocking_acquirer=acquire,
        transaction_factory=lambda _prepared: RecordingTransaction(
            data_struct=SimpleNamespace()
        ),
        workflow_generation_provider=lambda: 7,
        connect_queued=True,
    )
    bus.events.recording_completed.connect(
        lambda event: terminals.append(("completed", event))
    )
    bus.events.recording_failed.connect(
        lambda event: terminals.append(("failed", event))
    )
    bus.events.recording_cancelled.connect(
        lambda event: terminals.append(("cancelled", event))
    )
    command = BeginRecordingRequested(
        "command-queued-copy",
        snapshot.session_id,
        False,
        {"workflow_generation": 7},
    )

    assert controller.handle_begin_recording(command) is True
    worker = controller._session.blocking_job
    worker._thread.join(5)
    assert worker.is_alive() is False
    source[:] = -1
    fields["nested"]["labels"].append("mutated")
    info["metadata"]["serial"] = "MUTATED"
    loop = QEventLoop()
    bus.events.recording_completed.connect(loop.quit)
    bus.events.recording_failed.connect(loop.quit)
    bus.events.recording_cancelled.connect(loop.quit)
    QTimer.singleShot(5_000, loop.quit)
    if not terminals:
        loop.exec()
    assert terminals and terminals[0][0] == "completed", terminals

    staged = presented[0]
    np.testing.assert_allclose(
        staged.data_struct_fields["store_wave_data"], [0.1, 0.2]
    )
    assert staged.data_struct_fields["nested"]["labels"] == ("original",)
    assert staged.recorded_signal_info["metadata"]["serial"] == "SN-1"
    poisoned = None
    source = None
    fields = None
    info = None
    gc.collect()
    assert source_ref() is None


def test_recording_snapshot_rejects_divergent_count_persistence_identity(tmp_path):
    values = _snapshot(tmp_path).as_dict()
    values.update(
        product_model="MODEL-A",
        serial_number="SN-A",
        scanner_enabled=True,
        count_persistence={
            "product_model": "MODEL-A",
            "serial_number": "SN-OTHER",
            "scanner_enabled": True,
        },
    )

    with pytest.raises(ValueError, match="count_persistence"):
        RecordingSessionSnapshot.create(**values)


def test_prepared_session_rejects_divergent_count_context(tmp_path):
    values = _snapshot(tmp_path).as_dict()
    values.update(
        product_model="MODEL-A",
        serial_number="SN-A",
        count_persistence={
            "product_model": "MODEL-A",
            "serial_number": "SN-A",
            "scanner_enabled": False,
        },
    )
    snapshot = RecordingSessionSnapshot.create(**values)

    with pytest.raises(ValueError, match="count_persistence"):
        PreparedRecordingSession(
            snapshot,
            {
                "count_persistence": {
                    "product_model": "MODEL-A",
                    "serial_number": "SN-B",
                    "scanner_enabled": False,
                }
            },
        )


def test_recording_freeze_thaw_preserves_hashable_composite_keys_and_members():
    original = {
        ("channel", 1): {("tag", 2), frozenset({("nested", 3)})},
    }

    frozen = freeze_recording_session_value(original)
    thawed = thaw_recording_session_value(frozen)

    assert thawed == original
    assert ("channel", 1) in thawed


def test_prepared_session_deep_freezes_typed_worker_context(tmp_path):
    snapshot_values = _snapshot(tmp_path).as_dict()
    snapshot_values.update(
        product_model="MODEL-1",
        serial_number="SN-1",
        scanner_enabled=True,
        count_persistence={
            "product_model": "MODEL-1",
            "serial_number": "SN-1",
            "scanner_enabled": True,
        },
    )
    snapshot = RecordingSessionSnapshot.create(**snapshot_values)
    samples = np.arange(8, dtype=np.float32)[::2]
    nested = {
        "recorded_dict": {
            "device": {"rates": [48_000]},
            "tags": {"primary", "recording"},
        },
        "stimulus_dict": {"data": samples, "steps": [{"hz": 1_000}]},
        "recorded_signal_info": {"metadata": {"serial": "SN-1"}},
        "count_persistence": {
            "product_model": "MODEL-1",
            "serial_number": "SN-1",
            "scanner_enabled": True,
        },
        "detail": {"callback_block_size": 256},
    }

    prepared = PreparedRecordingSession(snapshot, nested)
    samples[0] = 99
    nested["recorded_dict"]["device"]["rates"].append(44_100)
    nested["recorded_dict"]["tags"].add("mutated")
    nested["stimulus_dict"]["steps"][0]["hz"] = 2_000
    nested["count_persistence"]["serial_number"] = "MUTATED"

    assert type(prepared.acquisition_context) is RecordingAcquisitionContext
    assert prepared.acquisition_context["recorded_dict"]["device"]["rates"] == (
        48_000,
    )
    assert prepared.acquisition_context["recorded_dict"]["tags"] == frozenset(
        {"primary", "recording"}
    )
    assert prepared.acquisition_context["stimulus_dict"]["steps"][0]["hz"] == 1_000
    frozen_samples = prepared.acquisition_context["stimulus_dict"]["data"]
    assert frozen_samples.tolist() == [0.0, 2.0, 4.0, 6.0]
    assert frozen_samples.flags.c_contiguous is True
    assert frozen_samples.flags.writeable is False
    assert not np.shares_memory(frozen_samples, samples)
    assert prepared.acquisition_context["count_persistence"]["serial_number"] == "SN-1"
    with pytest.raises(TypeError):
        prepared.acquisition_context["recorded_dict"]["new"] = True
    with pytest.raises(ValueError):
        frozen_samples[0] = 5


@pytest.mark.parametrize(
    "hostile",
    [
        pytest.param(type("HostileDict", (dict,), {})(), id="dict-subclass"),
        pytest.param(type("HostileList", (list,), {})(), id="list-subclass"),
        pytest.param(
            np.asarray([object()], dtype=object),
            id="object-array",
        ),
        pytest.param(
            np.arange(2, dtype=np.float32).view(
                type("HostileArray", (np.ndarray,), {})
            ),
            id="ndarray-subclass",
        ),
        pytest.param(object(), id="arbitrary-object"),
    ],
)
def test_prepared_session_rejects_non_exact_or_hostile_worker_values(
    tmp_path, hostile
):
    with pytest.raises(TypeError):
        PreparedRecordingSession(
            _snapshot(tmp_path),
            {"recorded_dict": {"hostile": hostile}},
        )


def test_recording_model_rejects_invalid_and_competing_sessions(tmp_path):
    model = RecordingModel()
    valid = _snapshot(tmp_path)
    model.begin(valid)

    with pytest.raises(RuntimeError, match="already active"):
        model.begin(
            RecordingSessionSnapshot.create(
                **{**valid.as_dict(), "session_id": "session-2"}
            )
        )

    with pytest.raises(ValueError, match="sample_rate"):
        RecordingSessionSnapshot.create(
            **{**valid.as_dict(), "sample_rate": 0}
        )


def test_transaction_commits_in_exact_order_and_reports_nonfatal_warnings(tmp_path):
    snapshot = _snapshot(tmp_path)
    snapshot.temp_path.write_bytes(b"new-audio")
    data_struct = SimpleNamespace(sample_rate=44_100, old_field="keep")
    order = []

    tx = RecordingTransaction(
        data_struct=data_struct,
        finalize_output=lambda staged: order.append("finalize"),
        alignment_handoff=lambda staged: order.append("alignment") or {},
        finalize_metadata=lambda staged: order.append("metadata"),
        promote_output=lambda staged: order.append("promote")
        or Path(staged.snapshot.temp_path).replace(staged.snapshot.output_path),
        save_database=lambda info, stimulus: order.append("database")
        or (_ for _ in ()).throw(OSError("db unavailable")),
        commit_count=lambda count: order.append("count-memory"),
        persist_count=lambda count: order.append("count-json")
        or (_ for _ in ()).throw(OSError("count json unavailable")),
        cleanup=lambda staged: order.append("cleanup")
        or (_ for _ in ()).throw(OSError("backup busy")),
    )

    result = tx.commit(_staged(snapshot))

    assert order == [
        "finalize",
        "alignment",
        "metadata",
        "promote",
        "database",
        "count-memory",
        "count-json",
        "cleanup",
    ]
    assert result.audio_committed is True
    assert result.completed is True
    assert result.analysis_snapshot["record_id"] == "record-1"
    assert [warning.stage for warning in result.warnings] == [
        "database",
        "count-persistence",
        "cleanup",
    ]
    assert data_struct.sample_rate == 48_000


def test_precommit_failure_rolls_back_temp_and_restores_replay_backup(tmp_path):
    snapshot = _snapshot(tmp_path, replay=True)
    snapshot.temp_path.write_bytes(b"incomplete")
    snapshot.backup_path.write_bytes(b"old-audio")
    tx = RecordingTransaction(
        data_struct=SimpleNamespace(),
        alignment_handoff=lambda staged: (_ for _ in ()).throw(
            ValueError("alignment failed")
        ),
    )

    result = tx.commit(_staged(snapshot))

    assert result.completed is False
    assert result.audio_committed is False
    assert snapshot.output_path.read_bytes() == b"old-audio"
    assert not snapshot.temp_path.exists()
    assert not snapshot.backup_path.exists()


def test_controller_transaction_setup_failure_rolls_back_replay_before_failure(
    tmp_path,
):
    snapshot = _snapshot(tmp_path, replay=True)
    snapshot.backup_path.write_bytes(b"old-audio")
    snapshot.temp_path.write_bytes(b"partial")
    adapter = BlockingRecordingAdapter(
        data_struct=SimpleNamespace(),
        save_database=lambda info, stimulus: None,
        commit_count=lambda count: None,
        persist_count=lambda count, data: None,
    )
    failed = []
    controller, bus, terminals, command = _streaming_controller(
        tmp_path,
        cancel_adapter=adapter.cancel,
    )
    controller.prepare_session = lambda command: PreparedRecordingSession(snapshot, {})
    controller.transaction_factory = lambda prepared: (_ for _ in ()).throw(
        RuntimeError("transaction setup failed")
    )
    bus.events.recording_failed.connect(failed.append)
    replay_command = BeginRecordingRequested(
        command.command_id,
        command.session_id,
        True,
        {"workflow_generation": 7},
    )
    assert controller.handle_begin_recording(replay_command) is True

    assert terminals[0].staged_recording_ready(_staged(snapshot)) is False

    assert len(failed) == 1
    assert failed[0].audio_committed is False
    assert failed[0].rollback_outcome["restored"] is True
    assert snapshot.output_path.read_bytes() == b"old-audio"
    assert not snapshot.temp_path.exists()
    assert not snapshot.backup_path.exists()


class _FailingDataStruct:
    def __init__(self):
        object.__setattr__(self, "store_wave_data", "old")

    def __setattr__(self, name, value):
        if name == "sample_rate" and value == 48_000:
            raise RuntimeError("data struct rejected snapshot")
        object.__setattr__(self, name, value)


def test_data_struct_failure_after_promotion_keeps_audio_and_blocks_analysis(tmp_path):
    snapshot = _snapshot(tmp_path, replay=True)
    snapshot.output_path.write_bytes(b"old-audio")
    snapshot.temp_path.write_bytes(b"new-audio")
    tx = RecordingTransaction(data_struct=_FailingDataStruct())

    result = tx.commit(_staged(snapshot))

    assert result.audio_committed is True
    assert result.completed is False
    assert Path(result.recovery_path).read_bytes() == b"new-audio"
    assert snapshot.backup_path.read_bytes() == b"old-audio"
    assert result.analysis_snapshot is None


def test_controller_publishes_completion_only_after_required_commits(tmp_path):
    snapshot = _snapshot(tmp_path)
    snapshot.temp_path.write_bytes(b"new-audio")
    data_struct = SimpleNamespace()
    order = []
    bus = SequenceEventBus()
    completed = []
    analysis = []
    bus.events.recording_completed.connect(
        lambda event: (order.append("completed"), completed.append(event))
    )
    bus.commands.analysis_requested.connect(analysis.append)

    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda command: PreparedRecordingSession(snapshot, None),
        blocking_acquirer=lambda prepared: _staged(prepared.snapshot),
        transaction_factory=lambda prepared: RecordingTransaction(
            data_struct=data_struct,
            save_database=lambda info, stimulus: order.append("database"),
            commit_count=lambda count: order.append("count-memory"),
            persist_count=lambda count: order.append("count-json"),
            cleanup=lambda staged: order.append("cleanup"),
        ),
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )

    accepted = controller.handle_begin_recording(
        BeginRecordingRequested(
            "command-1",
            "session-1",
            False,
            {"workflow_generation": 7},
        )
    )

    assert accepted is True
    assert order == [
        "database",
        "count-memory",
        "count-json",
        "cleanup",
        "completed",
    ]
    assert len(completed) == 1
    assert completed[0].sample_count == 2
    assert analysis == []


def test_controller_rejects_stale_admission_without_starting(tmp_path):
    started = []
    failed = []
    bus = SequenceEventBus()
    bus.events.recording_failed.connect(failed.append)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda command: started.append(command),
        workflow_generation_provider=lambda: 8,
        connect_queued=False,
    )

    accepted = controller.handle_begin_recording(
        BeginRecordingRequested(
            "command-1", "session-1", False, {"workflow_generation": 7}
        )
    )

    assert accepted is False
    assert started == []
    assert len(failed) == 1
    assert "stale" in failed[0].reason


def _streaming_controller(tmp_path, *, view=None, cancel_adapter=None, connect_queued=False):
    snapshot = _snapshot(tmp_path)
    bus = SequenceEventBus()
    terminals = []
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=view,
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        use_streaming=lambda prepared: True,
        streaming_adapter=lambda prepared, terminal: terminals.append(terminal) or True,
        cancel_adapter=cancel_adapter,
        workflow_generation_provider=lambda: 7,
        connect_queued=connect_queued,
    )
    command = BeginRecordingRequested(
        "command-1", "session-1", False, {"workflow_generation": 7}
    )
    return controller, bus, terminals, command


def _queued_blocking_controller(
    snapshot,
    transaction_factory,
    *,
    blocking_acquirer=None,
    request_blocking_cancel=None,
    cancel_adapter=None,
):
    bus = SequenceEventBus()
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        blocking_acquirer=blocking_acquirer
        or (lambda prepared: _staged(prepared.snapshot)),
        transaction_factory=transaction_factory,
        request_blocking_cancel=request_blocking_cancel,
        cancel_adapter=cancel_adapter,
        workflow_generation_provider=lambda: 7,
        connect_queued=True,
    )
    command = BeginRecordingRequested(
        "command-1", "session-1", False, {"workflow_generation": 7}
    )
    return controller, bus, command


@pytest.mark.parametrize("phase", ["recording", "finalizing"])
def test_cancellation_is_validated_idempotent_and_first_terminal_wins(tmp_path, phase):
    cancellations = []
    rollback_calls = []
    controller, bus, terminals, command = _streaming_controller(
        tmp_path,
        cancel_adapter=lambda prepared, reason: rollback_calls.append(reason)
        or {"restored": True},
    )
    bus.events.recording_cancelled.connect(cancellations.append)
    assert controller.handle_begin_recording(command) is True
    if phase == "finalizing":
        controller.model.mark_finalizing()

    cancellation = CancelRecordingRequested("session-1", 7, "operator stop")
    assert controller.handle_cancel_recording(cancellation) is True
    assert controller.handle_cancel_recording(cancellation) is False
    assert terminals[0].recording_failed("late failure") is False

    assert rollback_calls == ["operator stop"]
    assert len(cancellations) == 1
    assert cancellations[0].rollback_outcome["restored"] is True
    assert controller.model.active_session_id is None


def test_cancellation_rejects_stale_session_and_generation(tmp_path):
    controller, bus, _terminals, command = _streaming_controller(tmp_path)
    cancelled = []
    bus.events.recording_cancelled.connect(cancelled.append)
    assert controller.handle_begin_recording(command) is True

    assert controller.handle_cancel_recording(
        CancelRecordingRequested("other-session", 7, "stale")
    ) is False
    assert controller.handle_cancel_recording(
        CancelRecordingRequested("session-1", 8, "stale")
    ) is False
    assert cancelled == []


def test_cancellation_can_win_while_transaction_is_finalizing(tmp_path):
    snapshot = _snapshot(tmp_path)
    snapshot.temp_path.write_bytes(b"pending")
    entered = _QtProbe()
    release = Event()
    cancelled = []
    completed = []
    failed = []
    cleanup_calls = []

    def block_before_promotion(_staged):
        entered.reached.emit()
        assert release.wait(5)

    def cancel_adapter(prepared, reason):
        cleanup_calls.append(reason)
        return {"restored": True, "errors": ()}

    controller, bus, command = _queued_blocking_controller(
        snapshot,
        lambda prepared: RecordingTransaction(
            data_struct=SimpleNamespace(),
            finalize_output=block_before_promotion,
        ),
        cancel_adapter=cancel_adapter,
    )
    bus.events.recording_cancelled.connect(cancelled.append)
    bus.events.recording_completed.connect(completed.append)
    bus.events.recording_failed.connect(failed.append)
    starts = []
    _wait_for_qt_signal(
        entered.reached,
        lambda: starts.append(controller.handle_begin_recording(command)),
    )
    assert starts == [True]
    assert controller.handle_cancel_recording(
        CancelRecordingRequested("session-1", 7, "operator stop")
    ) is True
    assert cancelled == []
    assert cleanup_calls == []
    _wait_for_qt_signal(bus.events.recording_cancelled, release.set)

    assert cleanup_calls == ["operator stop"]
    assert len(cancelled) == 1
    assert completed == []
    assert failed == []
    assert not snapshot.output_path.exists()


def test_cancellation_after_durable_commit_barrier_waits_for_truthful_completion(
    tmp_path,
):
    snapshot = _snapshot(tmp_path)
    snapshot.temp_path.write_bytes(b"pending")
    promotion_entered = _QtProbe()
    release_promotion = Event()
    completed = []
    cancelled = []
    failed = []
    cleanup_calls = []

    def blocking_promotion(staged):
        promotion_entered.reached.emit()
        assert release_promotion.wait(5)
        staged.snapshot.temp_path.replace(staged.snapshot.output_path)

    controller, bus, command = _queued_blocking_controller(
        snapshot,
        lambda prepared: RecordingTransaction(
            data_struct=SimpleNamespace(),
            promote_output=blocking_promotion,
        ),
        cancel_adapter=lambda prepared, reason: cleanup_calls.append(reason)
        or {"restored": True},
    )
    bus.events.recording_completed.connect(completed.append)
    bus.events.recording_cancelled.connect(cancelled.append)
    bus.events.recording_failed.connect(failed.append)
    starts = []
    _wait_for_qt_signal(
        promotion_entered.reached,
        lambda: starts.append(controller.handle_begin_recording(command)),
    )
    assert starts == [True]
    cancellation = CancelRecordingRequested("session-1", 7, "too late")
    assert controller.handle_cancel_recording(cancellation) is True
    assert controller.handle_cancel_recording(cancellation) is False
    assert completed == []
    assert cancelled == []
    assert failed == []
    assert cleanup_calls == []
    _wait_for_qt_signal(bus.events.recording_completed, release_promotion.set)

    assert len(completed) == 1
    assert cancelled == []
    assert failed == []
    assert cleanup_calls == []
    assert snapshot.output_path.read_bytes() == b"pending"


def test_cancellation_after_durable_barrier_reports_truthful_commit_failure(tmp_path):
    snapshot = _snapshot(tmp_path)
    snapshot.temp_path.write_bytes(b"pending")
    promotion_entered = _QtProbe()
    release_promotion = Event()
    failed = []
    cancelled = []

    def blocking_promotion(staged):
        promotion_entered.reached.emit()
        assert release_promotion.wait(5)
        staged.snapshot.temp_path.replace(staged.snapshot.output_path)

    controller, bus, command = _queued_blocking_controller(
        snapshot,
        lambda prepared: RecordingTransaction(
            data_struct=_FailingDataStruct(),
            promote_output=blocking_promotion,
        ),
    )
    bus.events.recording_failed.connect(failed.append)
    bus.events.recording_cancelled.connect(cancelled.append)
    starts = []
    _wait_for_qt_signal(
        promotion_entered.reached,
        lambda: starts.append(controller.handle_begin_recording(command)),
    )
    assert starts == [True]
    assert controller.handle_cancel_recording(
        CancelRecordingRequested("session-1", 7, "too late")
    ) is True
    assert failed == []
    assert cancelled == []
    _wait_for_qt_signal(bus.events.recording_failed, release_promotion.set)

    assert len(failed) == 1
    assert failed[0].audio_committed is True
    assert failed[0].recovery_path == str(snapshot.output_path)
    assert cancelled == []
    assert snapshot.output_path.read_bytes() == b"pending"


def test_precommit_cancellation_wins_if_interrupted_stage_then_raises(tmp_path):
    snapshot = _snapshot(tmp_path)
    snapshot.temp_path.write_bytes(b"pending")
    entered = _QtProbe()
    release = Event()
    cancelled = []
    failed = []

    def interrupted_finalize(_staged):
        entered.reached.emit()
        assert release.wait(5)
        raise RuntimeError("stage noticed shutdown")

    controller, bus, command = _queued_blocking_controller(
        snapshot,
        lambda prepared: RecordingTransaction(
            data_struct=SimpleNamespace(),
            finalize_output=interrupted_finalize,
        ),
    )
    bus.events.recording_cancelled.connect(cancelled.append)
    bus.events.recording_failed.connect(failed.append)
    starts = []
    _wait_for_qt_signal(
        entered.reached,
        lambda: starts.append(controller.handle_begin_recording(command)),
    )
    assert starts == [True]
    assert controller.handle_cancel_recording(
        CancelRecordingRequested("session-1", 7, "operator stop")
    ) is True
    _wait_for_qt_signal(bus.events.recording_cancelled, release.set)

    assert len(cancelled) == 1
    assert cancelled[0].reason == "operator stop"
    assert failed == []
    assert not snapshot.temp_path.exists()


def test_disconnect_returns_before_durable_worker_quiesces_and_suppresses_late_ui(
    tmp_path,
):
    snapshot = _snapshot(tmp_path)
    snapshot.temp_path.write_bytes(b"pending")
    promotion_entered = _QtProbe()
    release_promotion = Event()
    cleanup_calls = []
    completed = []
    failed = []
    cancelled = []

    def blocking_promotion(staged):
        promotion_entered.reached.emit()
        assert release_promotion.wait(5)
        staged.snapshot.temp_path.replace(staged.snapshot.output_path)

    controller, bus, command = _queued_blocking_controller(
        snapshot,
        lambda prepared: RecordingTransaction(
            data_struct=SimpleNamespace(),
            promote_output=blocking_promotion,
        ),
        cancel_adapter=lambda prepared, reason: cleanup_calls.append(reason)
        or {"restored": True},
    )
    bus.events.recording_completed.connect(completed.append)
    bus.events.recording_failed.connect(failed.append)
    bus.events.recording_cancelled.connect(cancelled.append)
    starts = []
    _wait_for_qt_signal(
        promotion_entered.reached,
        lambda: starts.append(controller.handle_begin_recording(command)),
    )
    assert starts == [True]

    controller.disconnect()

    assert controller.model.active_session_id == "session-1"
    assert cleanup_calls == []
    assert completed == []
    _wait_for_qt_signal(controller.disconnect_quiesced, release_promotion.set)

    assert controller.model.active_session_id is None
    assert completed == []
    assert failed == []
    assert cancelled == []
    assert snapshot.output_path.read_bytes() == b"pending"


def test_queued_cancel_remains_responsive_during_blocking_acquisition(tmp_path):
    QCoreApplication.instance() or QCoreApplication([])
    snapshot = _snapshot(tmp_path)
    entered = _QtProbe()
    release = Event()
    stop_calls = []
    cancelled = []
    completed = []

    def blocking_acquirer(prepared):
        entered.reached.emit()
        assert release.wait(5)
        prepared.snapshot.temp_path.write_bytes(b"producer-finished")
        return _staged(prepared.snapshot)

    def request_cancel(prepared, reason):
        stop_calls.append(reason)
        release.set()

    def cleanup_cancel(prepared, reason):
        prepared.snapshot.temp_path.unlink(missing_ok=True)
        return {"restored": True, "errors": ()}

    bus = SequenceEventBus()
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        blocking_acquirer=blocking_acquirer,
        transaction_factory=lambda prepared: RecordingTransaction(
            data_struct=SimpleNamespace()
        ),
        request_blocking_cancel=request_cancel,
        cancel_adapter=cleanup_cancel,
        workflow_generation_provider=lambda: 7,
        connect_queued=True,
    )
    bus.events.recording_cancelled.connect(cancelled.append)
    bus.events.recording_completed.connect(completed.append)
    command = BeginRecordingRequested(
        "command-1", "session-1", False, {"workflow_generation": 7}
    )

    _wait_for_qt_signal(
        entered.reached,
        lambda: bus.commands.begin_recording_requested.emit(command),
    )
    assert controller.model.active_session_id == "session-1"

    _wait_for_qt_signal(
        bus.events.recording_cancelled,
        lambda: bus.commands.cancel_recording_requested.emit(
            CancelRecordingRequested("session-1", 7, "queued stop")
        ),
    )

    assert stop_calls == ["queued stop"]
    assert len(cancelled) == 1
    assert completed == []
    assert not snapshot.temp_path.exists()
    controller.disconnect()


def test_queued_recording_pipeline_keeps_worker_io_off_qt_thread(tmp_path):
    app = QCoreApplication.instance() or QCoreApplication([])
    main_thread = current_thread().ident
    snapshot = _snapshot(tmp_path)
    snapshot.temp_path.write_bytes(b"pending")
    worker_threads = []
    main_projection_threads = []
    completed = []

    class ThreadTrackingDataStruct:
        def __setattr__(self, name, value):
            main_projection_threads.append(("data", current_thread().ident))
            object.__setattr__(self, name, value)

    def worker_step(name, result=None):
        def callback(*_args):
            worker_threads.append((name, current_thread().ident))
            return result

        return callback

    def promote(staged):
        worker_threads.append(("promote", current_thread().ident))
        staged.snapshot.temp_path.replace(staged.snapshot.output_path)

    data_struct = ThreadTrackingDataStruct()
    bus = SequenceEventBus()
    loop = QEventLoop()

    def on_completed(event):
        main_projection_threads.append(("event", current_thread().ident))
        completed.append(event)
        loop.quit()

    bus.events.recording_completed.connect(on_completed)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(
            set_finished=lambda successful: main_projection_threads.append(
                ("view", current_thread().ident)
            )
        ),
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        blocking_acquirer=lambda prepared: _staged(prepared.snapshot),
        transaction_factory=lambda prepared: RecordingTransaction(
            data_struct=data_struct,
            alignment_handoff=worker_step("alignment", {}),
            finalize_metadata=worker_step("metadata"),
            promote_output=promote,
            save_database=worker_step("database"),
            commit_count=lambda count: main_projection_threads.append(
                ("count", current_thread().ident)
            ),
            persist_count=worker_step("persist"),
            cleanup=worker_step("cleanup"),
        ),
        workflow_generation_provider=lambda: 7,
        connect_queued=True,
    )

    assert controller.handle_begin_recording(
        BeginRecordingRequested(
            "command-1", "session-1", False, {"workflow_generation": 7}
        )
    ) is True
    QTimer.singleShot(5_000, loop.quit)
    loop.exec()

    assert len(completed) == 1
    assert {name for name, _ in worker_threads} == {
        "alignment",
        "metadata",
        "promote",
        "database",
        "persist",
        "cleanup",
    }
    assert all(thread_id != main_thread for _, thread_id in worker_threads)
    assert main_projection_threads
    assert all(thread_id == main_thread for _, thread_id in main_projection_threads)
    assert app.thread() == controller.thread()
    controller.disconnect()


def test_reentrant_disconnect_during_completed_publication_skips_late_view_projection(
    tmp_path,
):
    snapshot = _snapshot(tmp_path)
    snapshot.temp_path.write_bytes(b"pending")
    completed = []
    view_calls = []
    bus = SequenceEventBus()
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(
            set_finished=lambda successful: view_calls.append(successful)
        ),
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        blocking_acquirer=lambda prepared: _staged(prepared.snapshot),
        transaction_factory=lambda prepared: RecordingTransaction(
            data_struct=SimpleNamespace()
        ),
        workflow_generation_provider=lambda: 7,
        connect_queued=True,
    )

    def disconnect_reentrantly(event):
        completed.append(event)
        controller.disconnect()

    bus.events.recording_completed.connect(disconnect_reentrantly)
    _wait_for_qt_signal(
        controller.disconnect_quiesced,
        lambda: controller.handle_begin_recording(
            BeginRecordingRequested(
                "command-1", "session-1", False, {"workflow_generation": 7}
            )
        ),
    )

    assert len(completed) == 1
    assert view_calls == []
    assert controller.model.active_session_id is None


def test_late_worker_relay_after_disconnect_and_destruction_is_bounded(tmp_path):
    snapshot = _snapshot(tmp_path)
    entered = _QtProbe()
    release = Event()
    terminal_events = []

    def blocking_acquirer(prepared):
        entered.reached.emit()
        assert release.wait(5)
        return _staged(prepared.snapshot)

    controller, bus, command = _queued_blocking_controller(
        snapshot,
        lambda prepared: RecordingTransaction(data_struct=SimpleNamespace()),
        blocking_acquirer=blocking_acquirer,
        request_blocking_cancel=lambda prepared, reason: release.set(),
        cancel_adapter=lambda prepared, reason: {
            "restored": True,
            "errors": (),
        },
    )
    bus.events.recording_completed.connect(terminal_events.append)
    bus.events.recording_failed.connect(terminal_events.append)
    bus.events.recording_cancelled.connect(terminal_events.append)
    _wait_for_qt_signal(
        entered.reached,
        lambda: controller.handle_begin_recording(command),
    )
    job = controller._session.blocking_job

    controller.disconnect()
    _wait_for_qt_signal(controller.disconnect_quiesced)

    assert not job.is_alive()
    assert job.completion_delivery_error is None
    assert terminal_events == []
    destroyed = []
    controller.destroyed.connect(lambda *_args: destroyed.append(True))
    controller.deleteLater()
    QCoreApplication.sendPostedEvents(controller, QEvent.DeferredDelete)
    assert destroyed == [True]

    drained = _QtProbe()
    job._relay.finished.emit(
        recording_controller_module._AcquisitionOutcome(
            error=RuntimeError("late producer relay")
        )
    )
    _wait_for_qt_signal(
        drained.reached,
        lambda: QTimer.singleShot(0, drained.reached.emit),
    )

    assert terminal_events == []
    assert job.late_delivery_count == 1


def test_disconnect_defers_cleanup_until_deferred_job_quiesces_when_cancel_fails(
    tmp_path,
):
    snapshot = _snapshot(tmp_path)
    cleanup_calls = []
    cancel_calls = []
    joined = []
    terminal_events = []

    class DeferredJob:
        def __init__(self, work, deliver, request_cancel):
            self.work = work
            self.deliver = deliver
            self.request_cancel = request_cancel

        def start(self):
            return True

        def cancel(self, reason):
            cancel_calls.append(reason)
            raise RuntimeError("device cancel failed")

        def join(self):
            joined.append(True)
            raise AssertionError("Qt teardown must not join")

        def complete(self):
            self.deliver(
                recording_controller_module._AcquisitionOutcome(
                    staged=self.work()
                )
            )

    class DeferredExecutor:
        def create(self, work, deliver, request_cancel):
            self.job = DeferredJob(work, deliver, request_cancel)
            return self.job

    executor = DeferredExecutor()
    bus = SequenceEventBus()
    bus.events.recording_cancelled.connect(terminal_events.append)
    bus.events.recording_failed.connect(terminal_events.append)
    bus.events.recording_completed.connect(terminal_events.append)
    model = RecordingModel()
    controller = SequenceRecordingController(
        model,
        bus,
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        blocking_acquirer=lambda prepared: _staged(prepared.snapshot),
        transaction_factory=lambda prepared: RecordingTransaction(
            data_struct=SimpleNamespace()
        ),
        request_blocking_cancel=lambda prepared, reason: None,
        cancel_adapter=lambda prepared, reason: cleanup_calls.append(reason)
        or {"restored": True, "errors": ()},
        blocking_executor=executor,
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    assert controller.handle_begin_recording(
        BeginRecordingRequested(
            "command-1", "session-1", False, {"workflow_generation": 7}
        )
    ) is True

    controller.disconnect()

    assert cancel_calls == ["disconnect"]
    assert joined == []
    assert cleanup_calls == []
    assert model.active_session_id == "session-1"

    executor.job.complete()
    executor.job.complete()

    assert cleanup_calls == ["disconnect"]
    assert terminal_events == []
    assert model.active_session_id is None


def test_external_terminal_callback_is_marshaled_to_controller_qt_thread(tmp_path):
    app = QCoreApplication.instance() or QCoreApplication([])
    main_thread = current_thread().ident
    view_threads = []
    event_threads = []
    terminals = []
    bus = SequenceEventBus()
    loop = QEventLoop()
    bus.events.recording_failed.connect(
        lambda event: (
            event_threads.append(current_thread().ident),
            loop.quit(),
        )
    )
    snapshot = _snapshot(tmp_path)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(
            set_finished=lambda successful: view_threads.append(
                current_thread().ident
            )
        ),
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        use_streaming=lambda prepared: True,
        streaming_adapter=lambda prepared, terminal: terminals.append(terminal) or True,
        workflow_generation_provider=lambda: 7,
        connect_queued=True,
    )
    assert controller.handle_begin_recording(
        BeginRecordingRequested(
            "command-1", "session-1", False, {"workflow_generation": 7}
        )
    ) is True

    returns = []
    producer = Thread(
        target=lambda: returns.append(terminals[0].recording_failed("worker failure"))
    )
    producer.start()
    producer.join(5)
    QTimer.singleShot(5_000, loop.quit)
    loop.exec()

    assert returns == [True]
    assert event_threads == [main_thread]
    assert view_threads == [main_thread]
    assert app.thread() == controller.thread()
    controller.disconnect()


def test_queued_cancel_subscription_publishes_one_canonical_event(tmp_path):
    app = QCoreApplication.instance() or QCoreApplication([])
    rollback_calls = []
    controller, bus, _terminals, command = _streaming_controller(
        tmp_path,
        cancel_adapter=lambda prepared, reason: rollback_calls.append(reason)
        or {"restored": True},
        connect_queued=True,
    )
    cancelled = []
    bus.events.recording_cancelled.connect(cancelled.append)

    bus.commands.begin_recording_requested.emit(command)
    app.processEvents()
    cancellation = CancelRecordingRequested("session-1", 7, "queued stop")
    _wait_for_qt_signal(
        bus.events.recording_cancelled,
        lambda: (
            bus.commands.cancel_recording_requested.emit(cancellation),
            bus.commands.cancel_recording_requested.emit(cancellation),
        ),
    )

    assert rollback_calls == ["queued stop"]
    assert len(cancelled) == 1
    controller.disconnect()


def test_disconnect_invalidates_queued_start_cancel_and_late_terminal(tmp_path):
    app = QCoreApplication.instance() or QCoreApplication([])
    cancel_calls = []
    controller, bus, terminals, command = _streaming_controller(
        tmp_path,
        cancel_adapter=lambda prepared, reason: cancel_calls.append(reason),
        connect_queued=True,
    )
    started = []
    cancelled = []
    bus.events.recording_started.connect(started.append)
    bus.events.recording_cancelled.connect(cancelled.append)

    bus.commands.begin_recording_requested.emit(command)
    app.processEvents()
    assert len(terminals) == 1
    controller.disconnect()
    bus.commands.cancel_recording_requested.emit(
        CancelRecordingRequested("session-1", 7, "late")
    )
    app.processEvents()

    assert cancel_calls == ["disconnect"]
    assert terminals[0].recording_failed("late") is False
    assert len(started) == 1
    assert cancelled == []


class _RaisingView:
    def __init__(self, failing):
        self.failing = failing

    def recording_started(self):
        if self.failing == "started":
            raise RuntimeError("started view failed")

    def recording_completed(self, staged):
        if self.failing == "completed":
            raise RuntimeError("completed view failed")

    def recording_failed(self, reason):
        if self.failing == "failed":
            raise RuntimeError("failed view failed")

    def recording_cancelled(self, reason):
        return None


@pytest.mark.parametrize("failing", ["started", "completed"])
def test_raising_view_cannot_block_started_or_completed_events(tmp_path, failing):
    snapshot = _snapshot(tmp_path)
    snapshot.temp_path.write_bytes(b"audio")
    bus = SequenceEventBus()
    started = []
    completed = []
    bus.events.recording_started.connect(started.append)
    bus.events.recording_completed.connect(completed.append)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=_RaisingView(failing),
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        blocking_acquirer=lambda prepared: _staged(prepared.snapshot),
        transaction_factory=lambda prepared: RecordingTransaction(
            data_struct=SimpleNamespace()
        ),
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )

    assert controller.handle_begin_recording(
        BeginRecordingRequested(
            "command-1", "session-1", False, {"workflow_generation": 7}
        )
    ) is True
    assert len(started) == 1
    assert len(completed) == 1


def test_raising_failure_view_cannot_block_failed_event(tmp_path):
    snapshot = _snapshot(tmp_path)
    bus = SequenceEventBus()
    failed = []
    bus.events.recording_failed.connect(failed.append)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=_RaisingView("failed"),
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        blocking_acquirer=lambda prepared: (_ for _ in ()).throw(
            RuntimeError("device failed")
        ),
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )

    assert controller.handle_begin_recording(
        BeginRecordingRequested(
            "command-1", "session-1", False, {"workflow_generation": 7}
        )
    ) is False
    assert len(failed) == 1
    assert failed[0].reason == "device failed"


@pytest.mark.parametrize(
    "sample_count,mono_length,multi_length,audio_length",
    [(1, 2, 2, 2), (3, 3, 3, 3), (2, 1, 2, 2), (2, 2, 1, 2), (2, 2, 2, 1)],
)
def test_transaction_rejects_sample_target_and_array_mismatches_before_promotion(
    tmp_path, sample_count, mono_length, multi_length, audio_length
):
    snapshot = _snapshot(tmp_path)
    snapshot.temp_path.write_bytes(b"incomplete")
    staged = StagedRecording.create(
        snapshot=snapshot,
        sample_count=sample_count,
        data_struct_fields={
            "store_wave_data": np.zeros(mono_length, dtype=np.float32),
            "store_wave_data_multi": np.zeros((multi_length, 2), dtype=np.float32),
            "audio_lenth": audio_length,
        },
        recorded_signal_info={"file_path": str(snapshot.output_path)},
    )
    promoted = []
    result = RecordingTransaction(
        data_struct=SimpleNamespace(),
        promote_output=lambda value: promoted.append(value),
    ).commit(staged)

    assert result.completed is False
    assert result.audio_committed is False
    assert promoted == []
    assert not snapshot.temp_path.exists()
    assert "sample" in result.reason


def test_transaction_revalidates_alignment_fields_before_promotion(tmp_path):
    snapshot = _snapshot(tmp_path)
    snapshot.temp_path.write_bytes(b"incomplete")
    promoted = []

    result = RecordingTransaction(
        data_struct=SimpleNamespace(),
        alignment_handoff=lambda staged: {
            "store_wave_data": np.zeros(1, dtype=np.float32)
        },
        promote_output=lambda staged: promoted.append(staged),
    ).commit(_staged(snapshot))

    assert result.completed is False
    assert result.audio_committed is False
    assert promoted == []
    assert not snapshot.temp_path.exists()
    assert "store_wave_data" in result.reason


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("store_wave_data", "xx"),
        ("store_wave_data", np.asarray([object(), object()], dtype=object)),
        ("store_wave_data", np.asarray([np.nan, 0.0], dtype=np.float32)),
        ("store_wave_data", np.asarray([np.inf, 0.0], dtype=np.float32)),
        ("store_wave_data", np.zeros((2, 1), dtype=np.float32)),
        ("store_wave_data_multi", np.zeros(2, dtype=np.float32)),
        ("store_wave_data_multi", np.zeros((2, 1, 1), dtype=np.float32)),
        ("store_wave_data_multi", np.zeros((2, 1), dtype=np.float32)),
    ],
)
def test_transaction_rejects_non_numeric_nonfinite_dimensional_and_channel_data(
    tmp_path, field_name, value
):
    snapshot = _snapshot(tmp_path)
    snapshot.temp_path.write_bytes(b"invalid")
    fields = dict(_staged(snapshot).data_struct_fields)
    fields[field_name] = value
    if isinstance(value, np.ndarray) and value.dtype.hasobject:
        with pytest.raises(TypeError, match="dtype|object"):
            StagedRecording.create(
                snapshot=snapshot,
                sample_count=2,
                data_struct_fields=fields,
                recorded_signal_info={"file_path": str(snapshot.output_path)},
            )
        assert not snapshot.output_path.exists()
        return
    staged = StagedRecording.create(
        snapshot=snapshot,
        sample_count=2,
        data_struct_fields=fields,
        recorded_signal_info={"file_path": str(snapshot.output_path)},
    )
    promoted = []

    result = RecordingTransaction(
        data_struct=SimpleNamespace(),
        promote_output=lambda recording: promoted.append(recording),
    ).commit(staged)

    assert result.completed is False
    assert result.audio_committed is False
    assert promoted == []
    assert not snapshot.temp_path.exists()


def test_staged_recording_detaches_numeric_arrays_from_producer(tmp_path):
    snapshot = _snapshot(tmp_path)
    mono = np.asarray([0.1, 0.2], dtype=np.float32)
    multi = np.asarray([[0.1, 0.3], [0.2, 0.4]], dtype=np.float32)

    staged = StagedRecording.create(
        snapshot=snapshot,
        sample_count=2,
        data_struct_fields={
            "store_wave_data": mono,
            "store_wave_data_multi": multi,
            "audio_lenth": 2,
        },
        recorded_signal_info={"file_path": str(snapshot.output_path)},
    )
    mono[:] = 9
    multi[:] = 9

    assert np.allclose(staged.data_struct_fields["store_wave_data"], [0.1, 0.2])
    assert np.allclose(
        staged.data_struct_fields["store_wave_data_multi"],
        [[0.1, 0.3], [0.2, 0.4]],
    )
    assert staged.data_struct_fields["store_wave_data"].flags.writeable is False


def test_blocking_adapter_rejects_unsupported_recorded_dimensions():
    with pytest.raises(ValueError, match="dimension"):
        BlockingRecordingAdapter._normalize_recorded_data(
            np.zeros((2, 1, 1), dtype=np.float32),
            (0,),
        )


class _RestoreFailingDataStruct:
    def __init__(self):
        object.__setattr__(self, "store_wave_data", "old")
        object.__setattr__(self, "reject_restore", False)

    def __setattr__(self, name, value):
        if name == "sample_rate" and value == 48_000:
            object.__setattr__(self, "reject_restore", True)
            raise RuntimeError("apply failed")
        if (
            name == "store_wave_data"
            and isinstance(value, str)
            and value == "old"
            and self.reject_restore
        ):
            raise RuntimeError("restore failed")
        object.__setattr__(self, name, value)


def test_data_struct_rollback_reports_actual_restore_failure(tmp_path):
    snapshot = _snapshot(tmp_path)
    snapshot.temp_path.write_bytes(b"new")
    result = RecordingTransaction(data_struct=_RestoreFailingDataStruct()).commit(
        _staged(snapshot)
    )

    assert result.completed is False
    assert result.audio_committed is True
    assert result.rollback_outcome["data_struct_restored"] is False
    assert result.rollback_outcome["data_struct_restore_errors"]


def test_completed_event_and_model_include_immutable_normalized_warnings(tmp_path):
    snapshot = _snapshot(tmp_path)
    snapshot.temp_path.write_bytes(b"audio")
    bus = SequenceEventBus()
    completed = []
    bus.events.recording_completed.connect(completed.append)
    model = RecordingModel()
    controller = SequenceRecordingController(
        model,
        bus,
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        blocking_acquirer=lambda prepared: _staged(prepared.snapshot),
        transaction_factory=lambda prepared: RecordingTransaction(
            data_struct=SimpleNamespace(),
            finalize_metadata=lambda staged: (_ for _ in ()).throw(
                OSError("metadata")
            ),
            save_database=lambda info, stimulus: (_ for _ in ()).throw(OSError("db")),
            persist_count=lambda count: (_ for _ in ()).throw(OSError("json")),
            cleanup=lambda staged: (_ for _ in ()).throw(OSError("cleanup")),
        ),
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )

    assert controller.handle_begin_recording(
        BeginRecordingRequested(
            "command-1", "session-1", False, {"workflow_generation": 7}
        )
    ) is True
    assert [warning["stage"] for warning in completed[0].result_snapshot["warnings"]] == [
        "metadata",
        "database",
        "count-persistence",
        "cleanup",
    ]
    assert [warning["stage"] for warning in model.final_result["warnings"]] == [
        "metadata",
        "database",
        "count-persistence",
        "cleanup",
    ]
    with pytest.raises(TypeError):
        model.final_result["warnings"][0]["stage"] = "changed"


def _adapter_command(
    tmp_path,
    *,
    mode="RECORD_ONLY",
    replay=False,
    hostapis=(1, 1),
    use_streaming_recording: bool = False,
):
    detail = {
        "total_time": 2 / 48_000,
        "use_streaming_recording": use_streaming_recording,
    }
    if mode == "RECORD_ONLY":
        detail["monitor_playback"] = hostapis[1] is not None
    configuration = ConfigurationSnapshot(
        sequence_config=[{"seq1": {"acq": {"mode": mode, "detail": detail}}}],
        analysis_config={"auto_analysis": False},
        mic={"name": "mic", "hostapi": hostapis[0], "samplerate": 48_000},
        speaker=(
            None
            if hostapis[1] is None
            else {"name": "speaker", "hostapi": hostapis[1], "samplerate": 48_000}
        ),
        mic_channels=(0,),
    )
    output = tmp_path / "adapter.wav"
    return BeginRecordingRequested(
        "command-adapter",
        "session-1",
        replay,
        {
            "workflow_generation": 7,
            "configuration_generation": 3,
            "configuration": configuration,
            "recorded_path": str(output),
            "recorded_signal_info": {"file_path": str(output)},
            "record_id": "adapter-record",
            "pending_count": None if replay else 2,
            "label": "not_labeled",
            "product_model": "MODEL-A",
            "serial_number": "SN-0007",
            "scanner_enabled": True,
        },
    )


def test_blocking_adapter_rejects_mismatched_hostapi_before_device_open(tmp_path):
    opened = []
    adapter = BlockingRecordingAdapter(
        data_struct=SimpleNamespace(),
        save_database=lambda info, stimulus: None,
        commit_count=lambda count: None,
        persist_count=lambda count, data: None,
        audio_processor_factory=lambda: opened.append(True),
    )

    with pytest.raises(RuntimeError, match="同一驱动类型"):
        adapter.prepare(
            _adapter_command(tmp_path, mode="PLAY_AND_RECORD", hostapis=(1, 2))
        )
    assert opened == []


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_blocking_adapter_rejects_invalid_duration_before_started_or_device_open(
    tmp_path, duration
):
    command = _adapter_command(tmp_path, hostapis=(1, None))
    admission = dict(command.session_snapshot)
    configuration = admission["configuration"]
    admission["configuration"] = ConfigurationSnapshot(
        sequence_config=[
            {
                "seq1": {
                    "acq": {
                        "mode": "RECORD_ONLY",
                        "detail": {
                            "total_time": duration,
                            "use_streaming_recording": False,
                        },
                    }
                }
            }
        ],
        analysis_config=configuration.analysis_config,
        mic=configuration.mic,
        speaker=None,
        mic_channels=configuration.mic_channels,
    )
    invalid = BeginRecordingRequested(
        command.command_id, command.session_id, False, admission
    )
    opened = []
    adapter = BlockingRecordingAdapter(
        data_struct=SimpleNamespace(),
        save_database=lambda info, stimulus: None,
        commit_count=lambda count: None,
        persist_count=lambda count, data: None,
        audio_processor_factory=lambda: opened.append(True),
    )

    with pytest.raises(ValueError, match="duration|target"):
        adapter.prepare(invalid)
    assert opened == []


def test_blocking_adapter_rejects_nonpositive_derived_target_before_device_open(
    tmp_path, monkeypatch
):
    opened = []
    adapter = BlockingRecordingAdapter(
        data_struct=SimpleNamespace(),
        save_database=lambda info, stimulus: None,
        commit_count=lambda count: None,
        persist_count=lambda count, data: None,
        audio_processor_factory=lambda: opened.append(True),
    )
    monkeypatch.setattr(
        recording_controller_module.LoadUiConfig,
        "get_rec_and_play_dict_base_sequence_dict",
        lambda *args, **kwargs: ({}, {"num_frames": 0, "prolong_frames": 0}),
    )

    with pytest.raises(ValueError, match="target"):
        adapter.prepare(_adapter_command(tmp_path, hostapis=(1, None)))
    assert opened == []


def test_formal_controller_presents_hostapi_failure_without_started_event(tmp_path):
    opened = []
    errors = []
    started = []
    failed = []
    adapter = BlockingRecordingAdapter(
        data_struct=SimpleNamespace(),
        save_database=lambda info, stimulus: None,
        commit_count=lambda count: None,
        persist_count=lambda count, data: None,
        audio_processor_factory=lambda: opened.append(True),
    )
    bus = SequenceEventBus()
    bus.events.recording_started.connect(started.append)
    bus.events.recording_failed.connect(failed.append)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(present_error=errors.append),
        prepare_session=adapter.prepare,
        blocking_acquirer=adapter.acquire,
        transaction_factory=adapter.transaction,
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )

    assert controller.handle_begin_recording(
        _adapter_command(tmp_path, mode="PLAY_AND_RECORD", hostapis=(1, 2))
    ) is False

    assert started == []
    assert opened == []
    assert len(failed) == 1
    assert errors == [failed[0].reason]
    assert "同一驱动类型" in failed[0].reason


def test_blocking_play_record_target_uses_frozen_alignment_sample_count(tmp_path):
    base_command = _adapter_command(
        tmp_path, mode="PLAY_AND_RECORD", hostapis=(1, 1)
    )
    admission = dict(base_command.session_snapshot)
    admission.update(
        {
            "stimulus_data": np.arange(4, dtype=np.float32),
            "stimulus_info": {
                "amplitude": 1.0,
                "stimulus_method": "frequency_stepped",
            },
            "alignment_sample_count": 2,
        }
    )
    command = BeginRecordingRequested(
        base_command.command_id,
        base_command.session_id,
        False,
        admission,
    )
    adapter = BlockingRecordingAdapter(
        data_struct=SimpleNamespace(),
        save_database=lambda info, stimulus: None,
        commit_count=lambda count: None,
        persist_count=lambda count, data: None,
    )

    prepared = adapter.prepare(command)

    assert prepared.snapshot.target_samples == 2


def test_recording_adapter_prepares_streaming_policy_with_integer_sample_rate(
    tmp_path,
):
    adapter = BlockingRecordingAdapter(
        data_struct=SimpleNamespace(),
        save_database=lambda info, stimulus: None,
        commit_count=lambda count: None,
        persist_count=lambda count, data: None,
    )

    prepared = adapter.prepare(
        _adapter_command(tmp_path, use_streaming_recording=True)
    )

    assert prepared.acquisition_context["use_streaming"] is True
    assert type(prepared.snapshot.sample_rate) is int
    assert type(prepared.acquisition_context["recorded_dict"]["sr"]) is int
    assert type(
        prepared.acquisition_context["recorded_dict"]["sample_rate"]
    ) is int


def test_required_repeat_split_failure_is_precommit_and_rolls_back(
    tmp_path, monkeypatch
):
    snapshot = _snapshot(tmp_path)
    snapshot.temp_path.write_bytes(b"pending")
    prepared = PreparedRecordingSession(
        snapshot,
        {
            "wav_calibration_metadata": None,
            "count_persistence": {},
        },
    )
    staged = StagedRecording.create(
        snapshot=snapshot,
        sample_count=2,
        data_struct_fields=thaw_recording_session_value(
            _staged(snapshot).data_struct_fields
        ),
        recorded_signal_info={"file_path": str(snapshot.output_path)},
        stimulus_info={"repeat_times": 2},
    )
    monkeypatch.setattr(
        recording_controller_module.SplitRepeatSignal,
        "split_repeat_signal",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("split required")
        ),
    )
    adapter = BlockingRecordingAdapter(
        data_struct=SimpleNamespace(),
        save_database=lambda info, stimulus: None,
        commit_count=lambda count: None,
        persist_count=lambda count, data: None,
    )

    result = adapter.transaction(prepared).commit(staged)

    assert result.completed is False
    assert result.audio_committed is False
    assert "split required" in result.reason
    assert not snapshot.temp_path.exists()
    assert not snapshot.output_path.exists()


def test_blocking_adapter_prepares_and_acquires_frozen_record_only_session(
    tmp_path,
):
    processor_inputs = []

    class Processor:
        def sd_rec(self, recorded_dict):
            processor_inputs.append(recorded_dict)
            recorded_dict["device"]["name"] = "mutated-by-worker"
            recorded_dict["input_channels"].append(9)
            return 0, np.asarray([[0.1], [0.2]], dtype=np.float32)

    adapter = BlockingRecordingAdapter(
        data_struct=SimpleNamespace(),
        save_database=lambda info, stimulus: None,
        commit_count=lambda count: None,
        persist_count=lambda count, data: None,
        audio_processor_factory=Processor,
    )
    command = _adapter_command(tmp_path, hostapis=(1, None))
    prepared = adapter.prepare(command)
    staged = adapter.acquire(prepared)

    assert prepared.snapshot.target_samples == 2
    assert prepared.snapshot.product_model == "MODEL-A"
    assert prepared.snapshot.serial_number == "SN-0007"
    assert prepared.snapshot.scanner_enabled is True
    assert dict(prepared.snapshot.count_persistence) == {
        "product_model": "MODEL-A",
        "serial_number": "SN-0007",
        "scanner_enabled": True,
    }
    assert prepared.acquisition_context["recorded_dict"]["device"]["name"] == "mic"
    assert prepared.acquisition_context["recorded_dict"]["input_channels"] == (0,)
    assert processor_inputs[0] is not prepared.acquisition_context["recorded_dict"]
    assert type(prepared.snapshot.sample_rate) is int
    assert type(prepared.acquisition_context["recorded_dict"]["sr"]) is int
    assert type(
        prepared.acquisition_context["recorded_dict"]["sample_rate"]
    ) is int
    wav_rate, wav_data = wavfile.read(prepared.snapshot.temp_path)
    assert wav_rate == 48_000
    assert wav_data.shape[0] == 2
    assert staged.sample_count == 2
    assert staged.data_struct_fields["store_wave_data_multi"].shape == (2, 1)


def test_blocking_adapter_cancel_quiesces_processor_before_cleanup(tmp_path):
    entered = Event()
    release = Event()
    stop_calls = []

    class Processor:
        def sd_rec(self, recorded_dict):
            entered.set()
            assert release.wait(5)
            return 0, np.asarray([[0.1], [0.2]], dtype=np.float32)

        def cancel(self):
            stop_calls.append("cancel")
            release.set()

    adapter = BlockingRecordingAdapter(
        data_struct=SimpleNamespace(),
        save_database=lambda info, stimulus: None,
        commit_count=lambda count: None,
        persist_count=lambda count, data: None,
        audio_processor_factory=Processor,
    )
    prepared = adapter.prepare(_adapter_command(tmp_path, hostapis=(1, None)))
    errors = []

    def run_producer():
        try:
            adapter.acquire(prepared)
        except BaseException as error:
            errors.append(error)

    producer = Thread(target=run_producer)
    producer.start()
    assert entered.wait(5)

    adapter.request_cancel(prepared, "operator stop")
    producer.join(5)
    outcome = adapter.cancel(prepared, "operator stop")

    assert not producer.is_alive()
    assert stop_calls == ["cancel"]
    assert len(errors) == 1
    assert "cancelled" in str(errors[0])
    assert outcome["restored"] is True
    assert not prepared.snapshot.temp_path.exists()


def test_sequence_recording_view_projects_success_and_isolates_callback_failures():
    calls = []

    def fail_lock(value):
        calls.append(("lock", value))
        raise RuntimeError("lock presentation failed")

    view = SequenceRecordingView(
        set_recording_locked=fail_lock,
        set_started=lambda: calls.append(("started",)),
        set_finished=lambda success: calls.append(("finished", success)),
        present_error=lambda reason: calls.append(("error", reason)),
    )
    view.recording_started()
    view.recording_failed("device")

    assert ("started",) in calls
    assert ("finished", False) in calls
    assert ("error", "device") in calls


class _FailOnceLineEdit:
    def __init__(self, text):
        self._text = text
        self.fail = True

    def text(self):
        return self._text

    def setText(self, value):
        if self.fail:
            self.fail = False
            raise RuntimeError("line edit failed")
        self._text = str(value)


def test_count_projection_restores_partial_ui_commit_on_failure():
    from ui.sequence.sequence_recording_view import RecordingCountProjection

    model = SimpleNamespace(
        current_recorded_count=4,
        last_play_count=4,
    )
    line_edit = _FailOnceLineEdit("4")
    projection = RecordingCountProjection(model, line_edit)

    with pytest.raises(RuntimeError, match="line edit failed"):
        projection(5)
    assert model.current_recorded_count == 4
    assert line_edit.text() == "4"
    assert model.last_play_count == 4


class _FailOnceCountProjection:
    def __init__(self):
        self.current_recorded_count = 4
        self._last_play_count = 4
        self.fail_last = True

    @property
    def last_play_count(self):
        return self._last_play_count

    @last_play_count.setter
    def last_play_count(self, value):
        if value == 5 and self.fail_last:
            self.fail_last = False
            raise RuntimeError("last play count failed")
        self._last_play_count = value

def test_count_projection_restores_line_and_count_after_setter_failure():
    from ui.sequence.sequence_recording_view import RecordingCountProjection

    model = _FailOnceCountProjection()
    line_edit = _FailOnceLineEdit("4")
    line_edit.fail = False
    projection = RecordingCountProjection(model, line_edit)

    with pytest.raises(RuntimeError, match="last play count failed"):
        projection(5)

    assert model.current_recorded_count == 4
    assert line_edit.text() == "4"
    assert model.last_play_count == 4


class _Button:
    def __init__(self, enabled):
        self.enabled = enabled

    def isEnabled(self):
        return self.enabled

    def setDisabled(self, disabled):
        self.enabled = not disabled

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)


def test_replay_failure_restores_pre_attempt_data_and_replay_button_states():
    from ui.sequence.sequence_widget import SequenceWindow

    window = SimpleNamespace(
        data_btn=_Button(False),
        replayer_btn=_Button(True),
        streaming_buffer_multi=[1],
        streaming_plot_item=object(),
        _sn_clear_on_next_scan=True,
        barcode_scanner_box=SimpleNamespace(isChecked=lambda: False),
    )
    window._clear_plot_area = lambda: None
    window._cleanup_streaming_resources = lambda: None
    window.update_player_btn_is_playing = lambda: None
    window._set_sn_input_recording_read_only = lambda value: None
    window.update_player_btn_is_paused = lambda: None

    SequenceWindow._recording_view_started(window)
    assert window.data_btn.enabled is False
    assert window.replayer_btn.enabled is False
    SequenceWindow._recording_view_finished(window, False)
    assert window.data_btn.enabled is False
    assert window.replayer_btn.enabled is True


def test_prepare_failure_leaves_existing_button_states_unchanged():
    from ui.sequence.sequence_widget import SequenceWindow

    window = SimpleNamespace(
        data_btn=_Button(False),
        replayer_btn=_Button(True),
        _recording_button_state_before_recording=None,
        _sn_clear_on_next_scan=False,
        barcode_scanner_box=SimpleNamespace(isChecked=lambda: False),
    )
    window._set_sn_input_recording_read_only = lambda value: None
    window.update_player_btn_is_paused = lambda: None

    SequenceWindow._recording_view_finished(window, False)

    assert window.data_btn.enabled is False
    assert window.replayer_btn.enabled is True


def test_production_recording_view_state_initialization_preserves_buttons_before_start():
    script = r'''
import ui.sequence.sequence_widget as module
from types import SimpleNamespace
from PyQt5.QtWidgets import QApplication
from functools import partial
from ui.sequence.sequence_event_bus import SequenceEventBus as _SequenceEventBus
SequenceEventBus = partial(
    _SequenceEventBus,
    standalone_recording_admission=True,
)
from ui.sequence.sequence_messages import BeginRecordingRequested
from ui.sequence.sequence_recording_controller import SequenceRecordingController
from ui.sequence.sequence_recording_model import RecordingModel
from ui.sequence.sequence_recording_view import SequenceRecordingView

class StopAfterRecordingState(BaseException):
    pass

class StopDataStruct:
    def __init__(self):
        raise StopAfterRecordingState()

module.DataDealStruct = StopDataStruct
app = QApplication.instance() or QApplication([])
window = module.SequenceWindow.__new__(module.SequenceWindow)
try:
    module.SequenceWindow.__init__(window)
except StopAfterRecordingState:
    pass
else:
    raise AssertionError("constructor did not reach the controlled boundary")
assert window._recording_button_state_before_recording is module._RECORDING_BUTTON_STATE_NOT_CAPTURED

class Button:
    def __init__(self, enabled):
        self.enabled = enabled
    def isEnabled(self):
        return self.enabled
    def setEnabled(self, value):
        self.enabled = bool(value)

data_button = Button(False)
replay_button = Button(True)
window.toolsbar = SimpleNamespace(
    data_btn=data_button,
    replayer_btn=replay_button,
    barcode_scanner_box=SimpleNamespace(isChecked=lambda: False),
)
window._sn_clear_on_next_scan = False
window._set_sn_input_recording_read_only = lambda value: None
window.update_player_btn_is_paused = lambda: None
view = SequenceRecordingView(
    set_finished=window._recording_view_finished,
)
bus = SequenceEventBus()
started = []
bus.events.recording_started.connect(started.append)
controller = SequenceRecordingController(
    RecordingModel(),
    bus,
    view=view,
    prepare_session=lambda command: (_ for _ in ()).throw(ValueError("prepare failed")),
    workflow_generation_provider=lambda: 7,
    connect_queued=False,
)
assert controller.handle_begin_recording(
    BeginRecordingRequested("command-1", "session-1", False, {"workflow_generation": 7})
) is False
assert started == []
assert data_button.enabled is False
assert replay_button.enabled is True
'''
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_precommit_replay_failure_restores_file_and_button_state(tmp_path):
    from ui.sequence.sequence_widget import SequenceWindow

    snapshot = _snapshot(tmp_path, replay=True)
    snapshot.backup_path.write_bytes(b"old-audio")
    snapshot.temp_path.write_bytes(b"partial-new-audio")
    window = SimpleNamespace(
        data_btn=_Button(False),
        replayer_btn=_Button(True),
        streaming_buffer_multi=[1],
        streaming_plot_item=object(),
        _sn_clear_on_next_scan=True,
        barcode_scanner_box=SimpleNamespace(isChecked=lambda: False),
    )
    window._clear_plot_area = lambda: None
    window._cleanup_streaming_resources = lambda: None
    window.update_player_btn_is_playing = lambda: None
    window._set_sn_input_recording_read_only = lambda value: None
    window.update_player_btn_is_paused = lambda: None
    view = SequenceRecordingView(
        set_started=lambda: SequenceWindow._recording_view_started(window),
        set_finished=lambda successful: SequenceWindow._recording_view_finished(
            window, successful
        ),
    )
    bus = SequenceEventBus()
    failed = []
    bus.events.recording_failed.connect(failed.append)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=view,
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        blocking_acquirer=lambda prepared: _staged(prepared.snapshot),
        transaction_factory=lambda prepared: RecordingTransaction(
            data_struct=SimpleNamespace(),
            finalize_output=lambda staged: (_ for _ in ()).throw(
                OSError("finalize failed")
            ),
        ),
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )

    assert controller.handle_begin_recording(
        BeginRecordingRequested(
            "command-1", "session-1", True, {"workflow_generation": 7}
        )
    ) is False

    assert len(failed) == 1
    assert failed[0].audio_committed is False
    assert snapshot.output_path.read_bytes() == b"old-audio"
    assert not snapshot.backup_path.exists()
    assert not snapshot.temp_path.exists()
    assert window.data_btn.enabled is False
    assert window.replayer_btn.enabled is True


def test_acquisition_keyboard_interrupt_restores_controller_and_replay(tmp_path):
    snapshot = _snapshot(tmp_path, replay=True)
    snapshot.backup_path.write_bytes(b"old-audio")
    snapshot.temp_path.write_bytes(b"partial")
    bus = SequenceEventBus()
    failed = []
    cancelled = []
    bus.events.recording_failed.connect(failed.append)
    bus.events.recording_cancelled.connect(cancelled.append)
    adapter = BlockingRecordingAdapter(
        data_struct=SimpleNamespace(),
        save_database=lambda info, stimulus: None,
        commit_count=lambda count: None,
        persist_count=lambda count, data: None,
    )
    model = RecordingModel()
    controller = SequenceRecordingController(
        model,
        bus,
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        blocking_acquirer=lambda prepared: (_ for _ in ()).throw(
            KeyboardInterrupt("acquisition interrupted")
        ),
        transaction_factory=lambda prepared: RecordingTransaction(
            data_struct=SimpleNamespace()
        ),
        cancel_adapter=adapter.cancel,
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )

    with pytest.raises(KeyboardInterrupt, match="acquisition interrupted"):
        controller.handle_begin_recording(
            BeginRecordingRequested(
                "command-1", "session-1", True, {"workflow_generation": 7}
            )
        )

    assert model.active_session_id is None
    assert snapshot.output_path.read_bytes() == b"old-audio"
    assert not snapshot.temp_path.exists()
    assert not snapshot.backup_path.exists()
    assert failed == []
    assert cancelled == []
    assert controller.recording_failed("late") is False


def test_executor_creation_keyboard_interrupt_restores_controller_and_replay(tmp_path):
    snapshot = _snapshot(tmp_path, replay=True)
    snapshot.backup_path.write_bytes(b"old-audio")
    snapshot.temp_path.write_bytes(b"partial")
    adapter = BlockingRecordingAdapter(
        data_struct=SimpleNamespace(),
        save_database=lambda info, stimulus: None,
        commit_count=lambda count: None,
        persist_count=lambda count, data: None,
    )

    class InterruptingExecutor:
        def create(self, *args, **kwargs):
            raise KeyboardInterrupt("executor interrupted")

    model = RecordingModel()
    controller = SequenceRecordingController(
        model,
        SequenceEventBus(),
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        blocking_acquirer=lambda prepared: _staged(prepared.snapshot),
        transaction_factory=lambda prepared: RecordingTransaction(
            data_struct=SimpleNamespace()
        ),
        cancel_adapter=adapter.cancel,
        blocking_executor=InterruptingExecutor(),
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )

    with pytest.raises(KeyboardInterrupt, match="executor interrupted"):
        controller.handle_begin_recording(
            BeginRecordingRequested(
                "command-1", "session-1", True, {"workflow_generation": 7}
            )
        )

    assert model.active_session_id is None
    assert snapshot.output_path.read_bytes() == b"old-audio"
    assert not snapshot.temp_path.exists()
    assert not snapshot.backup_path.exists()


def test_executor_start_failure_rolls_back_replay_and_retires_controller(tmp_path):
    snapshot = _snapshot(tmp_path, replay=True)
    snapshot.backup_path.write_bytes(b"old-audio")
    snapshot.temp_path.write_bytes(b"partial")
    adapter = BlockingRecordingAdapter(
        data_struct=SimpleNamespace(),
        save_database=lambda info, stimulus: None,
        commit_count=lambda count: None,
        persist_count=lambda count, data: None,
    )

    class FailingJob:
        def start(self):
            raise RuntimeError("thread start failed")

        def cancel(self, reason):
            return None

        def join(self):
            return None

    class FailingExecutor:
        def create(self, *args, **kwargs):
            return FailingJob()

    failed = []
    bus = SequenceEventBus()
    bus.events.recording_failed.connect(failed.append)
    model = RecordingModel()
    controller = SequenceRecordingController(
        model,
        bus,
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        blocking_acquirer=lambda prepared: _staged(prepared.snapshot),
        transaction_factory=lambda prepared: RecordingTransaction(
            data_struct=SimpleNamespace()
        ),
        cancel_adapter=adapter.cancel,
        blocking_executor=FailingExecutor(),
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )

    assert controller.handle_begin_recording(
        BeginRecordingRequested(
            "command-1", "session-1", True, {"workflow_generation": 7}
        )
    ) is False

    assert model.active_session_id is None
    assert len(failed) == 1
    assert failed[0].rollback_outcome["restored"] is True
    assert snapshot.output_path.read_bytes() == b"old-audio"
    assert not snapshot.temp_path.exists()
    assert not snapshot.backup_path.exists()


def test_replay_promotion_keyboard_interrupt_restores_original_and_temp(
    tmp_path, monkeypatch
):
    snapshot = _snapshot(tmp_path, replay=True)
    snapshot.output_path.write_bytes(b"old-audio")
    snapshot.temp_path.write_bytes(b"new-audio")
    real_replace = recording_controller_module.os.replace
    calls = []

    def interrupt_second_replace(source, target):
        calls.append((Path(source), Path(target)))
        if len(calls) == 2:
            raise KeyboardInterrupt("promotion interrupted")
        return real_replace(source, target)

    monkeypatch.setattr(
        "ui.sequence.sequence_recording_transaction.os.replace",
        interrupt_second_replace,
    )

    with pytest.raises(KeyboardInterrupt, match="promotion interrupted"):
        RecordingTransaction(data_struct=SimpleNamespace()).commit(_staged(snapshot))

    assert snapshot.output_path.read_bytes() == b"old-audio"
    assert not snapshot.temp_path.exists()
    assert not snapshot.backup_path.exists()


def test_controller_ordinary_promotion_failure_keeps_original_reason_and_rolls_back(
    tmp_path,
):
    snapshot = _snapshot(tmp_path)
    snapshot.temp_path.write_bytes(b"new-audio")
    promotion_error = OSError("original promotion failure")
    cancel_calls = []
    failed = []
    bus = SequenceEventBus()
    bus.events.recording_failed.connect(failed.append)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        blocking_acquirer=lambda prepared: _staged(prepared.snapshot),
        transaction_factory=lambda prepared: RecordingTransaction(
            data_struct=SimpleNamespace(),
            promote_output=lambda staged: (_ for _ in ()).throw(promotion_error),
        ),
        cancel_adapter=lambda prepared, reason: cancel_calls.append(reason),
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )

    assert controller.handle_begin_recording(
        BeginRecordingRequested(
            "command-1", "session-1", False, {"workflow_generation": 7}
        )
    ) is False

    assert len(failed) == 1
    assert failed[0].reason == str(promotion_error)
    assert failed[0].audio_committed is False
    assert failed[0].recovery_path is None
    assert failed[0].rollback_outcome["restored"] is True
    assert cancel_calls == []
    assert not snapshot.temp_path.exists()
    assert not snapshot.output_path.exists()


def test_controller_replay_promotion_restore_failure_reports_original_error(
    tmp_path, monkeypatch
):
    snapshot = _snapshot(tmp_path, replay=True)
    snapshot.output_path.write_bytes(b"old-audio")
    snapshot.temp_path.write_bytes(b"new-audio")
    real_replace = recording_controller_module.os.replace
    promotion_error = OSError("original promotion failure")

    def fail_promotion_and_restore(source, target):
        source_path = Path(source)
        target_path = Path(target)
        if source_path == snapshot.temp_path and target_path == snapshot.output_path:
            raise promotion_error
        if source_path == snapshot.backup_path and target_path == snapshot.output_path:
            raise OSError("replay restore failed")
        return real_replace(source, target)

    monkeypatch.setattr(
        "ui.sequence.sequence_recording_transaction.os.replace",
        fail_promotion_and_restore,
    )
    failed = []
    bus = SequenceEventBus()
    bus.events.recording_failed.connect(failed.append)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        blocking_acquirer=lambda prepared: _staged(prepared.snapshot),
        transaction_factory=lambda prepared: RecordingTransaction(
            data_struct=SimpleNamespace()
        ),
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )

    assert controller.handle_begin_recording(
        BeginRecordingRequested(
            "command-1", "session-1", True, {"workflow_generation": 7}
        )
    ) is False

    assert len(failed) == 1
    assert failed[0].reason == str(promotion_error)
    assert failed[0].audio_committed is False
    assert failed[0].rollback_outcome["restored"] is False
    assert any(
        "replay restore failed" in error
        for error in failed[0].rollback_outcome["errors"]
    )
    assert snapshot.backup_path.read_bytes() == b"old-audio"
    assert not snapshot.output_path.exists()
    assert not snapshot.temp_path.exists()


def test_finalization_system_exit_retires_controller_and_preserves_recoverable_audio(
    tmp_path,
):
    snapshot = _snapshot(tmp_path, replay=True)
    snapshot.output_path.write_bytes(b"old-audio")
    snapshot.temp_path.write_bytes(b"new-audio")
    bus = SequenceEventBus()
    completed = []
    failed = []
    cancelled = []
    bus.events.recording_completed.connect(completed.append)
    bus.events.recording_failed.connect(failed.append)
    bus.events.recording_cancelled.connect(cancelled.append)
    model = RecordingModel()
    controller = SequenceRecordingController(
        model,
        bus,
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        blocking_acquirer=lambda prepared: _staged(prepared.snapshot),
        transaction_factory=lambda prepared: RecordingTransaction(
            data_struct=SimpleNamespace(),
            commit_count=lambda count: (_ for _ in ()).throw(
                SystemExit("finalization interrupted")
            ),
        ),
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )

    with pytest.raises(SystemExit, match="finalization interrupted"):
        controller.handle_begin_recording(
            BeginRecordingRequested(
                "command-1", "session-1", True, {"workflow_generation": 7}
            )
        )

    assert model.active_session_id is None
    assert snapshot.output_path.read_bytes() == b"new-audio"
    assert snapshot.backup_path.read_bytes() == b"old-audio"
    assert not snapshot.temp_path.exists()
    assert completed == []
    assert failed == []
    assert cancelled == []


class _StageControlledJob:
    def __init__(self, stage, fail_stage, failure_mode, error, work, deliver):
        self.stage = stage
        self.fail_stage = fail_stage
        self.failure_mode = failure_mode
        self.error = error
        self.work = work
        self.deliver = deliver

    def start(self):
        if self.stage == self.fail_stage and self.failure_mode == "start":
            raise self.error
        try:
            outcome = recording_controller_module._AcquisitionOutcome(
                staged=self.work()
            )
        except BaseException as error:
            outcome = recording_controller_module._AcquisitionOutcome(error=error)
        return self.deliver(outcome)

    def cancel(self, _reason):
        return None

    def is_alive(self):
        return False


class _StageControlledExecutor:
    def __init__(self, fail_stage, failure_mode, error):
        self.fail_stage = fail_stage
        self.failure_mode = failure_mode
        self.error = error
        self.create_count = 0

    def create(self, work, deliver, _request_cancel):
        self.create_count += 1
        if (
            self.create_count == self.fail_stage
            and self.failure_mode == "factory"
        ):
            raise self.error
        return _StageControlledJob(
            self.create_count,
            self.fail_stage,
            self.failure_mode,
            self.error,
            work,
            deliver,
        )


def _stage_failure_controller(
    tmp_path, fail_stage, failure_mode, *, replay=False
):
    snapshot = _snapshot(tmp_path, replay=replay)
    if replay:
        snapshot.output_path.write_bytes(b"old-audio")
    snapshot.temp_path.write_bytes(b"new-audio")
    error = RuntimeError(f"stage-{fail_stage}-{failure_mode}")
    data_struct = SimpleNamespace(original="kept")
    counts = []
    persisted = []
    cleanup = []
    bus = SequenceEventBus()
    completed = []
    failed = []
    bus.events.recording_completed.connect(completed.append)
    bus.events.recording_failed.connect(failed.append)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        blocking_acquirer=lambda prepared: _staged(prepared.snapshot),
        transaction_factory=lambda prepared: RecordingTransaction(
            data_struct=data_struct,
            save_database=lambda info, stimulus: None,
            commit_count=counts.append,
            persist_count=persisted.append,
            cleanup=lambda staged: cleanup.append(staged.snapshot.session_id),
        ),
        finalization_executor=_StageControlledExecutor(
            fail_stage, failure_mode, error
        ),
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    accepted = controller.handle_begin_recording(
        BeginRecordingRequested(
            "command-1", "session-1", replay, {"workflow_generation": 7}
        )
    )
    return SimpleNamespace(
        accepted=accepted,
        snapshot=snapshot,
        error=error,
        data_struct=data_struct,
        counts=counts,
        persisted=persisted,
        cleanup=cleanup,
        completed=completed,
        failed=failed,
    )


@pytest.mark.parametrize("failure_mode", ["factory", "start"])
@pytest.mark.parametrize("replay", [False, True])
def test_precommit_worker_launch_failure_rolls_back_without_false_audio_commit(
    tmp_path, failure_mode, replay
):
    outcome = _stage_failure_controller(
        tmp_path, 1, failure_mode, replay=replay
    )

    assert outcome.accepted is False
    assert len(outcome.failed) == 1
    assert outcome.failed[0].reason == str(outcome.error)
    assert outcome.failed[0].audio_committed is False
    assert outcome.failed[0].recovery_path is None
    assert not outcome.snapshot.temp_path.exists()
    if replay:
        assert outcome.snapshot.output_path.read_bytes() == b"old-audio"
    else:
        assert not outcome.snapshot.output_path.exists()
    assert not hasattr(outcome.data_struct, "store_wave_data")
    assert outcome.counts == []


@pytest.mark.parametrize("failure_mode", ["factory", "start"])
def test_database_worker_launch_failure_warns_and_continues_required_commits(
    tmp_path, failure_mode
):
    outcome = _stage_failure_controller(tmp_path, 2, failure_mode)

    assert outcome.accepted is True
    assert outcome.failed == []
    assert len(outcome.completed) == 1
    warnings = outcome.completed[0].result_snapshot["warnings"]
    assert [(item["stage"], item["message"]) for item in warnings] == [
        ("database", str(outcome.error))
    ]
    assert outcome.snapshot.output_path.read_bytes() == b"new-audio"
    assert isinstance(outcome.data_struct.store_wave_data, np.ndarray)
    assert outcome.counts == [12]
    assert outcome.persisted == [12]
    assert outcome.cleanup == ["session-1"]


@pytest.mark.parametrize("failure_mode", ["factory", "start"])
def test_persistence_worker_launch_failure_completes_with_immutable_warnings(
    tmp_path, failure_mode
):
    outcome = _stage_failure_controller(tmp_path, 3, failure_mode)

    assert outcome.accepted is True
    assert outcome.failed == []
    assert len(outcome.completed) == 1
    warnings = outcome.completed[0].result_snapshot["warnings"]
    assert [(item["stage"], item["message"]) for item in warnings] == [
        ("count-persistence", str(outcome.error)),
        ("cleanup", str(outcome.error)),
    ]
    with pytest.raises(TypeError):
        outcome.completed[0].result_snapshot["warnings"][0]["stage"] = "changed"
    assert outcome.snapshot.output_path.read_bytes() == b"new-audio"
    assert isinstance(outcome.data_struct.store_wave_data, np.ndarray)
    assert outcome.counts == [12]
    assert outcome.persisted == []
    assert outcome.cleanup == []


def test_terminal_port_concurrent_calls_reserve_one_bounded_queued_terminal(
    tmp_path,
):
    main_thread = current_thread().ident
    provider_threads = []
    terminals = []
    terminal_events = []
    terminal_observed = _QtProbe()
    bus = SequenceEventBus()

    def main_thread_provider():
        provider_threads.append(current_thread().ident)
        if current_thread().ident != main_thread:
            raise AssertionError("workflow generation provider left Qt thread")
        return 7

    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda command: PreparedRecordingSession(
            _snapshot(tmp_path), {}
        ),
        use_streaming=lambda prepared: True,
        streaming_adapter=lambda prepared, terminal: terminals.append(terminal)
        or True,
        workflow_generation_provider=main_thread_provider,
        connect_queued=True,
    )
    for signal in (
        bus.events.recording_failed,
        bus.events.recording_completed,
        bus.events.recording_cancelled,
    ):
        signal.connect(terminal_events.append)
        signal.connect(lambda _event: terminal_observed.reached.emit())
    assert controller.handle_begin_recording(
        BeginRecordingRequested(
            "command-1", "session-1", False, {"workflow_generation": 7}
        )
    ) is True
    provider_threads.clear()

    barrier = Barrier(17)
    returns = []
    errors = []

    def submit(index):
        barrier.wait()
        try:
            if index % 2:
                accepted = terminals[0].recording_failed(f"failure-{index}")
            else:
                accepted = terminals[0].recording_completed(
                    sample_count=2,
                    result_snapshot={"producer": index},
                )
            returns.append(accepted)
        except BaseException as error:
            errors.append(error)

    workers = [Thread(target=submit, args=(index,)) for index in range(16)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(5)

    assert errors == []
    assert returns.count(True) == 1
    assert returns.count(False) == 15
    assert provider_threads == []
    assert terminal_events == []
    _wait_for_qt_signal(terminal_observed.reached)
    assert len(terminal_events) == 1
    assert provider_threads == []


def test_terminal_port_snapshot_is_immutable_and_independent_of_live_model(tmp_path):
    snapshot = _snapshot(tmp_path)
    terminals = []
    controller, _bus, _command = _queued_blocking_controller(
        snapshot,
        lambda prepared: RecordingTransaction(data_struct=SimpleNamespace()),
    )
    controller.use_streaming = lambda prepared: True
    controller.streaming_adapter = (
        lambda prepared, terminal: terminals.append(terminal) or True
    )
    assert controller.handle_begin_recording(
        BeginRecordingRequested(
            "command-1", "session-1", False, {"workflow_generation": 7}
        )
    ) is True
    controller.model.active_snapshot = None
    observed = []
    worker = Thread(target=lambda: observed.append(terminals[0].session_snapshot))
    worker.start()
    worker.join(5)

    assert observed[0]["session_id"] == "session-1"
    assert observed[0]["mode"] == "RECORD_ONLY"
    with pytest.raises(TypeError):
        observed[0]["mode"] = "changed"
    controller.disconnect()


def test_retained_terminal_port_is_native_safe_after_controller_deletion_and_race(
    tmp_path,
):
    script = r'''
import sys
from pathlib import Path
from threading import Barrier, Thread

from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtWidgets import QApplication

from functools import partial
from ui.sequence.sequence_event_bus import SequenceEventBus as _SequenceEventBus
SequenceEventBus = partial(
    _SequenceEventBus,
    standalone_recording_admission=True,
)
from ui.sequence.sequence_messages import BeginRecordingRequested
from ui.sequence.sequence_recording_controller import (
    PreparedRecordingSession,
    SequenceRecordingController,
)
from ui.sequence.sequence_recording_model import RecordingModel, RecordingSessionSnapshot

app = QApplication.instance() or QApplication([])
root = Path(sys.argv[1])

def build(index):
    session_id = f"session-{index}"
    snapshot = RecordingSessionSnapshot.create(
        session_id=session_id,
        workflow_generation=7,
        configuration_generation=3,
        mode="RECORD_ONLY",
        sample_rate=48_000,
        bit_depth=32,
        input_channels=[0],
        input_device={"name": "mic"},
        output_device=None,
        stimulus_snapshot=None,
        target_samples=2,
        output_path=root / f"record-{index}.wav",
        temp_path=root / f"record-{index}.tmp.wav",
        backup_path=None,
        record_id=f"record-{index}",
        label="not_labeled",
        analysis_config={},
        configuration_snapshot={},
        pending_count=1,
        replay=False,
    )
    terminals = []
    controller = SequenceRecordingController(
        RecordingModel(),
        SequenceEventBus(),
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        use_streaming=lambda prepared: True,
        streaming_adapter=lambda prepared, terminal: terminals.append(terminal) or True,
        workflow_generation_provider=lambda: 7,
        connect_queued=True,
    )
    command = BeginRecordingRequested(
        f"command-{index}", session_id, False, {"workflow_generation": 7}
    )
    assert controller.handle_begin_recording(command) is True
    return controller, terminals[0]

# The retained port must be harmless after native QObject destruction even when
# disconnect() was never called.
controller, terminal = build(1)
destroyed = []
controller.destroyed.connect(lambda *_args: destroyed.append(True))
controller.deleteLater()
QCoreApplication.sendPostedEvents(controller, QEvent.DeferredDelete)
assert destroyed == [True]
returns = []
worker = Thread(target=lambda: returns.append(terminal.recording_failed("late")))
worker.start()
worker.join(5)
assert not worker.is_alive()
assert returns == [False]

# Exercise the emit/destruction race repeatedly.  A producer may win before the
# destruction gate closes, but neither side may touch a deleted native QObject.
for index in range(2, 18):
    controller, terminal = build(index)
    destroyed = []
    controller.destroyed.connect(lambda *_args: destroyed.append(True))
    barrier = Barrier(2)
    errors = []
    def submit():
        try:
            barrier.wait()
            terminal.recording_failed("racing")
        except BaseException as error:
            errors.append(error)
    worker = Thread(target=submit)
    worker.start()
    barrier.wait()
    controller.deleteLater()
    QCoreApplication.sendPostedEvents(controller, QEvent.DeferredDelete)
    assert destroyed == [True]
    worker.join(5)
    assert not worker.is_alive()
    assert errors == []
    app.processEvents()
'''
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def _streaming_quiescence_controller(
    tmp_path,
    *,
    close_streaming_admission,
    quiesce_streaming,
    cancel_adapter,
):
    snapshot = _snapshot(tmp_path)
    terminals = []
    bus = SequenceEventBus()
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        use_streaming=lambda prepared: True,
        streaming_adapter=lambda prepared, terminal: terminals.append(terminal)
        or True,
        close_streaming_admission=close_streaming_admission,
        quiesce_streaming=quiesce_streaming,
        cancel_adapter=cancel_adapter,
        workflow_generation_provider=lambda: 7,
        connect_queued=True,
    )
    command = BeginRecordingRequested(
        "command-1", "session-1", False, {"workflow_generation": 7}
    )
    assert controller.handle_begin_recording(command) is True
    return controller, bus, terminals[0]


def test_streaming_cancel_waits_for_worker_quiescence_before_cleanup_and_terminal(
    tmp_path,
):
    entered = _QtProbe()
    release = Event()
    close_threads = []
    cleanup = []
    cancelled = []

    def close_admission(prepared):
        close_threads.append(current_thread().ident)
        return {"producer": prepared.snapshot.session_id}

    def quiesce(prepared, reason, handle):
        assert handle == {"producer": "session-1"}
        entered.reached.emit()
        assert release.wait(5)
        return {"quiesced": True}

    controller, bus, terminal = _streaming_quiescence_controller(
        tmp_path,
        close_streaming_admission=close_admission,
        quiesce_streaming=quiesce,
        cancel_adapter=lambda prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
    )
    bus.events.recording_cancelled.connect(cancelled.append)
    returns = []
    cancellation = CancelRecordingRequested("session-1", 7, "operator stop")
    _wait_for_qt_signal(
        entered.reached,
        lambda: returns.append(controller.handle_cancel_recording(cancellation)),
    )

    assert returns == [True]
    assert close_threads == [current_thread().ident]
    assert cleanup == []
    assert cancelled == []
    assert terminal.recording_failed("late producer terminal") is False
    assert controller.handle_cancel_recording(cancellation) is False
    assert controller.handle_cancel_recording(
        CancelRecordingRequested("other", 7, "stale")
    ) is False

    _wait_for_qt_signal(bus.events.recording_cancelled, release.set)
    assert cleanup == ["operator stop"]
    assert len(cancelled) == 1


def test_streaming_stop_failure_stays_pending_without_cleanup_or_false_terminal(
    tmp_path,
):
    cleanup = []
    cancelled = []
    failed = []
    stop_error = RuntimeError("producer did not quiesce")

    def fail_stop(_prepared, _reason, _handle):
        raise stop_error

    controller, bus, terminal = _streaming_quiescence_controller(
        tmp_path,
        close_streaming_admission=lambda prepared: object(),
        quiesce_streaming=fail_stop,
        cancel_adapter=lambda prepared, reason: cleanup.append(reason),
    )
    bus.events.recording_cancelled.connect(cancelled.append)
    bus.events.recording_failed.connect(failed.append)

    _wait_for_qt_signal(
        controller.streaming_quiescence_failed,
        lambda: controller.handle_cancel_recording(
            CancelRecordingRequested("session-1", 7, "operator stop")
        ),
    )

    assert cleanup == []
    assert cancelled == []
    assert failed == []
    assert controller.model.active_session_id == "session-1"
    assert controller.streaming_cancellation_pending is True
    assert terminal.recording_completed(sample_count=2, result_snapshot={}) is False
    retry_returns = []
    _wait_for_qt_signal(
        controller.streaming_quiescence_failed,
        lambda: retry_returns.append(
            controller.handle_cancel_recording(
                CancelRecordingRequested("session-1", 7, "duplicate")
            )
        ),
    )
    assert retry_returns == [True]
    assert cleanup == []
    assert cancelled == []
    assert failed == []
    assert controller.streaming_cancellation_pending is True


def test_streaming_consumer_failure_uses_stop_failed_pending_then_retry_cleanup(
    tmp_path,
):
    stop_outcomes = iter(
        [
            {"quiesced": False, "diagnostic": "close failed"},
            {"quiesced": True},
        ]
    )
    cleanup = []
    failed = []
    close_calls = []
    controller, bus, terminal = _streaming_quiescence_controller(
        tmp_path,
        close_streaming_admission=lambda prepared: close_calls.append(
            prepared.snapshot.session_id
        )
        or "closed",
        quiesce_streaming=lambda _prepared, _reason, _handle: next(
            stop_outcomes
        ),
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
    )
    bus.events.recording_failed.connect(failed.append)

    _wait_for_qt_signal(
        controller.streaming_quiescence_failed,
        lambda: terminal.streaming_consumer_failed(
            "writer failed",
            {"restored": False, "errors": ("writer rollback",)},
            False,
            "close failed",
        ),
    )

    assert failed == []
    assert cleanup == []
    assert close_calls == ["session-1"]
    assert controller.streaming_cancellation_pending is True
    _wait_for_qt_signal(
        bus.events.recording_failed,
        lambda: controller.handle_cancel_recording(
            CancelRecordingRequested("session-1", 7, "retry stop")
        ),
    )

    assert cleanup == ["writer failed"]
    assert len(failed) == 1
    assert failed[0].reason == "writer failed"
    assert failed[0].rollback_outcome["restored"] is False
    assert failed[0].rollback_outcome["errors"] == ("writer rollback",)
    assert controller.streaming_cancellation_pending is False


def test_quiesced_streaming_consumer_failure_still_routes_through_task5_cleanup(
    tmp_path,
):
    order = []
    failed = []
    controller, bus, terminal = _streaming_quiescence_controller(
        tmp_path,
        close_streaming_admission=lambda _prepared: order.append("close")
        or "closed",
        quiesce_streaming=lambda _prepared, _reason, _handle: order.append(
            "quiesce"
        )
        or {"quiesced": True},
        cancel_adapter=lambda _prepared, _reason: order.append("cleanup")
        or {"restored": True, "errors": ()},
    )
    bus.events.recording_failed.connect(
        lambda event: (order.append("failed"), failed.append(event))
    )

    _wait_for_qt_signal(
        bus.events.recording_failed,
        lambda: terminal.streaming_consumer_failed(
            "protocol gap",
            {"restored": True, "errors": ()},
            True,
            "",
        ),
    )

    assert order == ["close", "quiesce", "cleanup", "failed"]
    assert failed[0].reason == "protocol gap"


def test_streaming_stop_baseexception_re_raises_while_remaining_pending(tmp_path):
    snapshot = _snapshot(tmp_path)
    cleanup = []
    diagnostics = []
    interruption = KeyboardInterrupt("stop interrupted")
    bus = SequenceEventBus()
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        use_streaming=lambda prepared: True,
        streaming_adapter=lambda prepared, terminal: True,
        close_streaming_admission=lambda prepared: "closed",
        quiesce_streaming=lambda prepared, reason, handle: (_ for _ in ()).throw(
            interruption
        ),
        cancel_adapter=lambda prepared, reason: cleanup.append(reason),
        workflow_generation_provider=lambda: 7,
        logger=SimpleNamespace(
            error=lambda _message: (_ for _ in ()).throw(
                SystemExit("logger must not replace interruption")
            )
        ),
        connect_queued=False,
    )
    controller.streaming_quiescence_failed.connect(
        lambda session_id, reason: diagnostics.append((session_id, reason))
    )
    assert controller.handle_begin_recording(
        BeginRecordingRequested(
            "command-1", "session-1", False, {"workflow_generation": 7}
        )
    ) is True

    with pytest.raises(KeyboardInterrupt, match="stop interrupted") as raised:
        controller.handle_cancel_recording(
            CancelRecordingRequested("session-1", 7, "cancel")
        )
    assert raised.value is interruption

    assert controller.streaming_cancellation_pending is True
    assert controller.model.active_session_id == "session-1"
    assert cleanup == []
    assert diagnostics == [("session-1", "stop interrupted")]


def test_streaming_disconnect_returns_before_ack_and_retires_only_after_quiescence(
    tmp_path,
):
    entered = _QtProbe()
    release = Event()
    cleanup = []
    terminal_events = []

    def quiesce(_prepared, _reason, _handle):
        entered.reached.emit()
        assert release.wait(5)
        return {"quiesced": True}

    controller, bus, terminal = _streaming_quiescence_controller(
        tmp_path,
        close_streaming_admission=lambda prepared: "closed",
        quiesce_streaming=quiesce,
        cancel_adapter=lambda prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
    )
    bus.events.recording_cancelled.connect(terminal_events.append)
    bus.events.recording_failed.connect(terminal_events.append)
    bus.events.recording_completed.connect(terminal_events.append)

    _wait_for_qt_signal(entered.reached, controller.disconnect)

    assert controller.model.active_session_id == "session-1"
    assert cleanup == []
    assert terminal_events == []
    assert terminal.recording_failed("late") is False

    _wait_for_qt_signal(controller.disconnect_quiesced, release.set)
    assert controller.model.active_session_id is None
    assert cleanup == ["disconnect"]
    assert terminal_events == []


def test_production_streaming_handshake_detaches_qt_state_before_worker_stop():
    from ui.sequence.sequence_widget import SequenceWindow

    main_thread = current_thread().ident
    calls = []

    class Service:
        def close_admission(self, prepared):
            calls.append(("gate", current_thread().ident))
            return {"prepared": prepared}

        def quiesce(self, prepared, reason, handle):
            assert handle == {"prepared": prepared}
            assert reason == "cancel"
            calls.append(("producer", current_thread().ident))
            calls.append(("consumer", current_thread().ident))
            return {"quiesced": True}

    window = SimpleNamespace(streaming_recording_service=Service())
    prepared = SimpleNamespace()
    handle = SequenceWindow._close_streaming_recording_admission(
        window, prepared
    )

    assert calls == [("gate", main_thread)]
    acknowledgement = []
    worker = Thread(
        target=lambda: acknowledgement.append(
            SequenceWindow._quiesce_streaming_recording(
                window, prepared, "cancel", handle
            )
        )
    )
    worker.start()
    worker.join(5)

    assert acknowledgement == [{"quiesced": True}]
    assert [name for name, _thread in calls] == ["gate", "producer", "consumer"]
    assert all(
        thread != main_thread
        for name, thread in calls
        if name in {"producer", "consumer"}
    )


def test_full_threaded_recording_pipeline_never_reads_workflow_provider_off_qt_thread(
    tmp_path,
):
    snapshot = _snapshot(tmp_path)
    snapshot.temp_path.write_bytes(b"new-audio")
    qt_thread = current_thread().ident
    provider_threads = []
    completed = []
    failed = []
    terminal_observed = _QtProbe()
    bus = SequenceEventBus()

    def workflow_generation():
        provider_threads.append(current_thread().ident)
        if current_thread().ident != qt_thread:
            raise AssertionError("workflow generation provider left Qt thread")
        return 7

    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        blocking_acquirer=lambda prepared: _staged(prepared.snapshot),
        transaction_factory=lambda prepared: RecordingTransaction(
            data_struct=SimpleNamespace()
        ),
        workflow_generation_provider=workflow_generation,
        connect_queued=True,
    )
    bus.events.recording_completed.connect(completed.append)
    bus.events.recording_failed.connect(failed.append)
    bus.events.recording_completed.connect(lambda _event: terminal_observed.reached.emit())
    bus.events.recording_failed.connect(lambda _event: terminal_observed.reached.emit())
    accepted = []

    _wait_for_qt_signal(
        terminal_observed.reached,
        lambda: accepted.append(
            controller.handle_begin_recording(
                BeginRecordingRequested(
                    "command-threaded",
                    "session-1",
                    False,
                    {"workflow_generation": 7},
                )
            )
        ),
    )

    assert accepted == [True]
    assert len(completed) == 1
    assert failed == []
    assert provider_threads
    assert set(provider_threads) == {qt_thread}
    controller.disconnect()


def test_streaming_stop_failure_can_be_retried_by_later_cancel(tmp_path):
    close_calls = []
    stop_calls = []
    cleanup = []
    cancelled = []

    def stop(_prepared, _reason, _handle):
        stop_calls.append(True)
        assert _handle is close_calls[0]
        if len(stop_calls) == 1:
            raise RuntimeError("first stop failed")
        return {"quiesced": True}

    def close(prepared):
        handle = {"producer": prepared.snapshot.session_id}
        close_calls.append(handle)
        return handle

    controller, bus, _terminal = _streaming_quiescence_controller(
        tmp_path,
        close_streaming_admission=close,
        quiesce_streaming=stop,
        cancel_adapter=lambda prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
    )
    bus.events.recording_cancelled.connect(cancelled.append)
    command = CancelRecordingRequested("session-1", 7, "operator stop")
    _wait_for_qt_signal(
        controller.streaming_quiescence_failed,
        lambda: controller.handle_cancel_recording(command),
    )
    retried = []
    _wait_for_qt_signal(
        bus.events.recording_cancelled,
        lambda: retried.append(controller.handle_cancel_recording(command)),
    )

    assert retried == [True]
    assert len(close_calls) == 1
    assert len(stop_calls) == 2
    assert cleanup == ["operator stop"]
    assert len(cancelled) == 1
    assert controller.streaming_cancellation_pending is False
    assert controller.model.active_session_id is None
    assert controller.handle_cancel_recording(command) is False


def test_streaming_stop_failure_disconnect_retries_and_suppresses_terminal(tmp_path):
    close_calls = []
    stop_calls = []
    cleanup = []
    terminal_events = []

    def stop(_prepared, _reason, _handle):
        stop_calls.append(True)
        if len(stop_calls) == 1:
            raise RuntimeError("first stop failed")
        return {"quiesced": True}

    controller, bus, _terminal = _streaming_quiescence_controller(
        tmp_path,
        close_streaming_admission=lambda prepared: close_calls.append("closed") or "closed",
        quiesce_streaming=stop,
        cancel_adapter=lambda prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
    )
    for signal in (
        bus.events.recording_completed,
        bus.events.recording_failed,
        bus.events.recording_cancelled,
    ):
        signal.connect(terminal_events.append)
    _wait_for_qt_signal(
        controller.streaming_quiescence_failed,
        lambda: controller.handle_cancel_recording(
            CancelRecordingRequested("session-1", 7, "operator stop")
        ),
    )

    _wait_for_qt_signal(controller.disconnect_quiesced, controller.disconnect)

    assert len(stop_calls) == 2
    assert close_calls == ["closed"]
    assert cleanup == ["operator stop"]
    assert terminal_events == []
    assert controller.model.active_session_id is None


def test_late_streaming_quiesced_acknowledgement_continues_cleanup(tmp_path):
    cleanup = []
    cancelled = []
    controller, bus, _terminal = _streaming_quiescence_controller(
        tmp_path,
        close_streaming_admission=lambda prepared: "closed",
        quiesce_streaming=lambda prepared, reason, handle: (_ for _ in ()).throw(
            RuntimeError("stop acknowledgement delayed")
        ),
        cancel_adapter=lambda prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
    )
    bus.events.recording_cancelled.connect(cancelled.append)
    _wait_for_qt_signal(
        controller.streaming_quiescence_failed,
        lambda: controller.handle_cancel_recording(
            CancelRecordingRequested("session-1", 7, "operator stop")
        ),
    )
    acknowledged = []

    _wait_for_qt_signal(
        bus.events.recording_cancelled,
        lambda: acknowledged.append(
            controller.acknowledge_streaming_quiesced("session-1")
        ),
    )

    assert acknowledged == [True]
    assert cleanup == ["operator stop"]
    assert len(cancelled) == 1
    assert controller.acknowledge_streaming_quiesced("session-1") is False
    assert controller.acknowledge_streaming_quiesced("stale-session") is False


def test_streaming_cleanup_failure_after_confirmed_stop_retires_truthfully(tmp_path):
    cancelled = []
    failed = []
    controller, bus, _terminal = _streaming_quiescence_controller(
        tmp_path,
        close_streaming_admission=lambda prepared: "closed",
        quiesce_streaming=lambda prepared, reason, handle: {"quiesced": True},
        cancel_adapter=lambda prepared, reason: (_ for _ in ()).throw(
            OSError("cleanup failed")
        ),
    )
    bus.events.recording_cancelled.connect(cancelled.append)
    bus.events.recording_failed.connect(failed.append)

    _wait_for_qt_signal(
        bus.events.recording_cancelled,
        lambda: controller.handle_cancel_recording(
            CancelRecordingRequested("session-1", 7, "operator stop")
        ),
    )

    assert len(cancelled) == 1
    assert failed == []
    assert cancelled[0].rollback_outcome["restored"] is False
    assert cancelled[0].rollback_outcome["errors"] == ("cleanup failed",)
    assert controller.streaming_cancellation_pending is False
    assert controller.model.active_session_id is None


class _DeferredMalformedJob:
    def __init__(self, start_callback):
        self._start_callback = start_callback

    def start(self):
        return self._start_callback()

    def cancel(self, _reason):
        return None

    def is_alive(self):
        return False


class _DeferredMalformedStageExecutor:
    def __init__(self, malformed_stage):
        self.malformed_stage = malformed_stage
        self.create_count = 0
        self.pending_delivery = None

    def create(self, work, deliver, _request_cancel):
        self.create_count += 1
        if self.create_count == self.malformed_stage:
            def defer():
                self.pending_delivery = deliver
                return True

            return _DeferredMalformedJob(defer)

        def run():
            try:
                outcome = recording_controller_module._AcquisitionOutcome(
                    staged=work()
                )
            except BaseException as error:
                outcome = recording_controller_module._AcquisitionOutcome(error=error)
            return deliver(outcome)

        return _DeferredMalformedJob(run)

    def deliver_malformed(self):
        delivery = self.pending_delivery
        assert delivery is not None
        self.pending_delivery = None
        return delivery(
            recording_controller_module._AcquisitionOutcome(staged=object())
        )


@pytest.mark.parametrize("malformed_stage", [2, 3], ids=["database", "persistence"])
def test_malformed_worker_result_uses_single_terminal_retirement_and_allows_next_session(
    tmp_path, malformed_stage
):
    snapshots = {}

    def prepare(command):
        base = _snapshot(tmp_path)
        suffix = command.session_id
        snapshot = RecordingSessionSnapshot.create(
            **{
                **base.as_dict(),
                "session_id": suffix,
                "output_path": tmp_path / f"{suffix}.wav",
                "temp_path": tmp_path / f"{suffix}.tmp.wav",
                "record_id": suffix,
            }
        )
        snapshot.temp_path.write_bytes(b"new-audio")
        snapshots[suffix] = snapshot
        return PreparedRecordingSession(snapshot, {})

    executor = _DeferredMalformedStageExecutor(malformed_stage)
    bus = SequenceEventBus()
    model = RecordingModel()
    failed = []
    completed = []
    bus.events.recording_failed.connect(failed.append)
    bus.events.recording_completed.connect(completed.append)
    controller = SequenceRecordingController(
        model,
        bus,
        prepare_session=prepare,
        blocking_acquirer=lambda prepared: _staged(prepared.snapshot),
        transaction_factory=lambda prepared: RecordingTransaction(
            data_struct=SimpleNamespace()
        ),
        finalization_executor=executor,
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    first = BeginRecordingRequested(
        "command-1", "session-1", False, {"workflow_generation": 7}
    )
    assert controller.handle_begin_recording(first) is True
    retained_port = controller._session.terminal

    assert executor.deliver_malformed() is False

    assert len(failed) == 1
    assert completed == []
    assert failed[0].audio_committed is True
    assert failed[0].recovery_path == str(snapshots["session-1"].output_path)
    assert snapshots["session-1"].output_path.read_bytes() == b"new-audio"
    assert model.active_session_id is None
    assert controller._session is None
    assert retained_port.recording_failed("late duplicate") is False
    assert len(failed) == 1

    second = BeginRecordingRequested(
        "command-2", "session-2", False, {"workflow_generation": 7}
    )
    assert controller.handle_begin_recording(second) is True
    assert len(completed) == 1
    assert completed[0].session_id == "session-2"
    assert model.active_session_id is None


def test_native_worker_relays_drop_late_results_after_controller_destruction(
    tmp_path,
):
    script = r'''
import gc
import sys
import weakref
from pathlib import Path
from threading import Event

from PyQt5.QtCore import QCoreApplication, QEvent, QEventLoop, QObject, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication

from functools import partial
from ui.sequence.sequence_event_bus import SequenceEventBus as _SequenceEventBus
SequenceEventBus = partial(
    _SequenceEventBus,
    standalone_recording_admission=True,
)
from ui.sequence.sequence_messages import BeginRecordingRequested
from ui.sequence.sequence_recording_controller import PreparedRecordingSession, SequenceRecordingController
from ui.sequence.sequence_recording_model import RecordingModel, RecordingSessionSnapshot, StagedRecording
from ui.sequence.sequence_recording_transaction import RecordingTransaction

app = QApplication.instance() or QApplication([])
root = Path(sys.argv[1])

class Probe(QObject):
    reached = pyqtSignal()

def wait(signal, trigger=lambda: None):
    loop = QEventLoop()
    seen = []
    signal.connect(lambda *_args: (seen.append(True), loop.quit()))
    QTimer.singleShot(5_000, loop.quit)
    trigger()
    if not seen:
        loop.exec()
    assert seen

def snapshot(name):
    return RecordingSessionSnapshot.create(
        session_id=name,
        workflow_generation=7,
        configuration_generation=1,
        mode="RECORD_ONLY",
        sample_rate=48_000,
        bit_depth=32,
        input_channels=[0],
        input_device={"name": "mic"},
        output_device=None,
        stimulus_snapshot=None,
        target_samples=2,
        output_path=root / f"{name}.wav",
        temp_path=root / f"{name}.tmp.wav",
        backup_path=None,
        record_id=name,
        label="not_labeled",
        analysis_config={},
        configuration_snapshot={},
        pending_count=1,
        replay=False,
    )

def staged(item):
    import numpy as np
    mono = np.asarray([0.1, 0.2], dtype=np.float32)
    return StagedRecording.create(
        snapshot=item,
        sample_count=2,
        data_struct_fields={
            "store_wave_data": mono,
            "store_wave_data_multi": mono.reshape(-1, 1),
            "sample_rate": 48_000,
            "audio_lenth": 2,
        },
        recorded_signal_info={"file_path": str(item.output_path)},
        stimulus_info=None,
    )

def destroy_while_blocking():
    item = snapshot("blocking")
    entered = Probe()
    release = Event()
    model = RecordingModel()
    events = []
    bus = SequenceEventBus()
    bus.events.recording_completed.connect(events.append)
    bus.events.recording_failed.connect(events.append)
    bus.events.recording_cancelled.connect(events.append)
    def acquire(_prepared):
        entered.reached.emit()
        assert release.wait(5)
        return staged(item)
    controller = SequenceRecordingController(
        model,
        bus,
        prepare_session=lambda command: PreparedRecordingSession(item, {}),
        blocking_acquirer=acquire,
        transaction_factory=lambda prepared: RecordingTransaction(data_struct=object()),
        workflow_generation_provider=lambda: 7,
        connect_queued=True,
    )
    wait(
        entered.reached,
        lambda: controller.handle_begin_recording(
            BeginRecordingRequested("command-blocking", "blocking", False, {"workflow_generation": 7})
        ),
    )
    job = controller._session.blocking_job
    before = (model.state, model.active_session_id, model.final_result)
    destroyed = []
    controller.destroyed.connect(lambda *_args: destroyed.append(True))
    controller_ref = weakref.ref(controller)
    controller.deleteLater()
    QCoreApplication.sendPostedEvents(controller, QEvent.DeferredDelete)
    assert destroyed == [True]
    del controller
    gc.collect()
    assert controller_ref() is None
    release.set()
    job._thread.join(5)
    assert not job.is_alive()
    app.processEvents()
    assert events == []
    assert (model.state, model.active_session_id, model.final_result) == before

def destroy_during_stage(name):
    item = snapshot(name)
    item.temp_path.write_bytes(b"new-audio")
    entered = Probe()
    release = Event()
    model = RecordingModel()
    events = []
    bus = SequenceEventBus()
    bus.events.recording_completed.connect(events.append)
    bus.events.recording_failed.connect(events.append)
    bus.events.recording_cancelled.connect(events.append)
    def block(*_args):
        entered.reached.emit()
        assert release.wait(5)
    transaction_options = {"data_struct": type("Data", (), {})()}
    if name == "durable":
        transaction_options["finalize_output"] = block
    elif name == "database":
        transaction_options["save_database"] = block
    else:
        transaction_options["persist_count"] = block
    controller = SequenceRecordingController(
        model,
        bus,
        prepare_session=lambda command: PreparedRecordingSession(item, {}),
        blocking_acquirer=lambda prepared: staged(item),
        transaction_factory=lambda prepared: RecordingTransaction(**transaction_options),
        workflow_generation_provider=lambda: 7,
        connect_queued=True,
    )
    wait(
        entered.reached,
        lambda: controller.handle_begin_recording(
            BeginRecordingRequested(f"command-{name}", name, False, {"workflow_generation": 7})
        ),
    )
    job = controller._session.finalization_job
    before = (model.state, model.active_session_id, model.final_result)
    destroyed = []
    controller.destroyed.connect(lambda *_args: destroyed.append(True))
    controller_ref = weakref.ref(controller)
    controller.deleteLater()
    QCoreApplication.sendPostedEvents(controller, QEvent.DeferredDelete)
    assert destroyed == [True]
    del controller
    gc.collect()
    assert controller_ref() is None
    release.set()
    job._thread.join(5)
    assert not job.is_alive()
    app.processEvents()
    assert events == []
    assert (model.state, model.active_session_id, model.final_result) == before

destroy_while_blocking()
for stage in ("durable", "database", "persistence"):
    destroy_during_stage(stage)
'''
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=40,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("promoted", [False, True], ids=["pre-promotion", "post-promotion"])
def test_malformed_durable_result_uses_protected_promotion_truth(tmp_path, promoted):
    snapshot = _snapshot(tmp_path, replay=promoted)
    if promoted:
        snapshot.output_path.write_bytes(b"old-audio")
    snapshot.temp_path.write_bytes(b"new-audio")
    failed = []

    class MalformedDurableTransaction(RecordingTransaction):
        def prepare_durable(self, staged):
            if promoted:
                self.begin_durable_commit()
                self.promote_output(staged)
                self.promotion_succeeded()
            return object()

    def rollback(prepared, _reason):
        prepared.snapshot.temp_path.unlink(missing_ok=True)
        return {"restored": True, "errors": ()}

    bus = SequenceEventBus()
    bus.events.recording_failed.connect(failed.append)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        blocking_acquirer=lambda prepared: _staged(prepared.snapshot),
        transaction_factory=lambda prepared: MalformedDurableTransaction(
            data_struct=SimpleNamespace()
        ),
        cancel_adapter=rollback,
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )

    assert controller.handle_begin_recording(
        BeginRecordingRequested(
            "command-malformed", "session-1", promoted, {"workflow_generation": 7}
        )
    ) is False

    assert len(failed) == 1
    assert failed[0].audio_committed is promoted
    assert failed[0].recovery_path == (
        str(snapshot.output_path) if promoted else None
    )
    assert controller.model.active_session_id is None
    if promoted:
        assert snapshot.output_path.read_bytes() == b"new-audio"
        assert snapshot.backup_path.read_bytes() == b"old-audio"
    else:
        assert not snapshot.temp_path.exists()
        assert not snapshot.output_path.exists()


def test_streaming_quiesced_ack_wins_against_late_stop_worker_failure(tmp_path):
    entered = _QtProbe()
    release = Event()
    cleanup = []
    cancelled = []
    diagnostics = []

    def stop(_prepared, _reason, _handle):
        entered.reached.emit()
        assert release.wait(5)
        raise RuntimeError("late stop worker failure")

    controller, bus, _terminal = _streaming_quiescence_controller(
        tmp_path,
        close_streaming_admission=lambda prepared: "closed",
        quiesce_streaming=stop,
        cancel_adapter=lambda prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
    )
    bus.events.recording_cancelled.connect(cancelled.append)
    controller.streaming_quiescence_failed.connect(
        lambda session_id, reason: diagnostics.append((session_id, reason))
    )
    _wait_for_qt_signal(
        entered.reached,
        lambda: controller.handle_cancel_recording(
            CancelRecordingRequested("session-1", 7, "operator stop")
        ),
    )

    acknowledged = []
    acknowledged.append(
        controller.acknowledge_streaming_quiesced("session-1")
    )
    assert acknowledged == [True]
    assert cleanup == []
    assert cancelled == []

    _wait_for_qt_signal(bus.events.recording_cancelled, release.set)
    drained = _QtProbe()
    _wait_for_qt_signal(
        drained.reached,
        lambda: QTimer.singleShot(0, drained.reached.emit),
    )

    assert cleanup == ["operator stop"]
    assert len(cancelled) == 1
    assert diagnostics == []
    assert controller.model.active_session_id is None


def test_disconnect_during_inflight_stop_failure_starts_one_bounded_retry(tmp_path):
    entered = _QtProbe()
    release = Event()
    stop_calls = []
    cleanup = []

    def stop(_prepared, _reason, _handle):
        stop_calls.append(True)
        if len(stop_calls) == 1:
            entered.reached.emit()
            assert release.wait(5)
            raise RuntimeError("inflight stop failed")
        return {"quiesced": True}

    controller, _bus, _terminal = _streaming_quiescence_controller(
        tmp_path,
        close_streaming_admission=lambda prepared: "closed",
        quiesce_streaming=stop,
        cancel_adapter=lambda prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
    )
    _wait_for_qt_signal(
        entered.reached,
        lambda: controller.handle_cancel_recording(
            CancelRecordingRequested("session-1", 7, "operator stop")
        ),
    )
    controller.disconnect()

    _wait_for_qt_signal(controller.disconnect_quiesced, release.set)

    assert len(stop_calls) == 2
    assert cleanup == ["operator stop"]
    assert controller.model.active_session_id is None


def test_repeated_disconnect_retries_only_after_previous_stop_attempt_failed(tmp_path):
    stop_calls = []
    cleanup = []

    def stop(_prepared, _reason, _handle):
        stop_calls.append(True)
        if len(stop_calls) < 3:
            raise RuntimeError(f"stop attempt {len(stop_calls)} failed")
        return {"quiesced": True}

    controller, _bus, _terminal = _streaming_quiescence_controller(
        tmp_path,
        close_streaming_admission=lambda prepared: "closed",
        quiesce_streaming=stop,
        cancel_adapter=lambda prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
    )
    _wait_for_qt_signal(
        controller.streaming_quiescence_failed,
        lambda: controller.handle_cancel_recording(
            CancelRecordingRequested("session-1", 7, "operator stop")
        ),
    )
    _wait_for_qt_signal(
        controller.streaming_quiescence_failed,
        controller.disconnect,
    )

    _wait_for_qt_signal(controller.disconnect_quiesced, controller.disconnect)

    assert len(stop_calls) == 3
    assert cleanup == ["operator stop"]
    assert controller.model.active_session_id is None


def test_concurrent_duplicate_streaming_retry_and_ack_resolve_once(tmp_path):
    retry_entered = _QtProbe()
    release_retry = Event()
    stop_calls = []
    cleanup = []
    cancelled = []
    diagnostics = []

    def stop(_prepared, _reason, _handle):
        stop_calls.append(True)
        if len(stop_calls) == 1:
            raise RuntimeError("initial stop failure")
        retry_entered.reached.emit()
        assert release_retry.wait(5)
        raise RuntimeError("late retry failure")

    controller, bus, _terminal = _streaming_quiescence_controller(
        tmp_path,
        close_streaming_admission=lambda prepared: "closed",
        quiesce_streaming=stop,
        cancel_adapter=lambda prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
    )
    bus.events.recording_cancelled.connect(cancelled.append)
    controller.streaming_quiescence_failed.connect(
        lambda session_id, reason: diagnostics.append((session_id, reason))
    )
    _wait_for_qt_signal(
        controller.streaming_quiescence_failed,
        lambda: controller.handle_cancel_recording(
            CancelRecordingRequested("session-1", 7, "operator stop")
        ),
    )

    class AckEmitter(QObject):
        requested = pyqtSignal(str)

    emitter = AckEmitter()
    emitter.requested.connect(
        controller.acknowledge_streaming_quiesced, Qt.QueuedConnection
    )
    barrier = Barrier(17)
    workers = []
    for index in range(16):
        if index % 2:
            callback = lambda: emitter.requested.emit("session-1")
        else:
            callback = lambda: bus.commands.cancel_recording_requested.emit(
                CancelRecordingRequested("session-1", 7, "duplicate")
            )
        worker = Thread(target=lambda callback=callback: (barrier.wait(), callback()))
        workers.append(worker)
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(5)

    queued_drained = _QtProbe()
    _wait_for_qt_signal(
        queued_drained.reached,
        lambda: QTimer.singleShot(0, queued_drained.reached.emit),
    )
    release_retry.set()
    _wait_for_qt_signal(bus.events.recording_cancelled)
    drained = _QtProbe()
    _wait_for_qt_signal(
        drained.reached,
        lambda: QTimer.singleShot(0, drained.reached.emit),
    )

    assert len(cancelled) == 1
    assert cleanup == ["operator stop"]
    assert len(stop_calls) <= 2
    assert len(diagnostics) == 1
    assert controller.model.active_session_id is None


def test_native_threaded_stop_baseexceptions_are_diagnostic_not_qt_fatal(tmp_path):
    script = r'''
import sys
from pathlib import Path

from PyQt5.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer
from PyQt5.QtWidgets import QApplication

from functools import partial
from ui.sequence.sequence_event_bus import SequenceEventBus as _SequenceEventBus
SequenceEventBus = partial(
    _SequenceEventBus,
    standalone_recording_admission=True,
)
from ui.sequence.sequence_messages import BeginRecordingRequested, CancelRecordingRequested
from ui.sequence.sequence_recording_controller import PreparedRecordingSession, SequenceRecordingController
from ui.sequence.sequence_recording_model import RecordingModel, RecordingSessionSnapshot

app = QApplication.instance() or QApplication([])
root = Path(sys.argv[1])

def run(index, error):
    session_id = f"session-{index}"
    snapshot = RecordingSessionSnapshot.create(
        session_id=session_id,
        workflow_generation=7,
        configuration_generation=1,
        mode="RECORD_ONLY",
        sample_rate=48_000,
        bit_depth=32,
        input_channels=[0],
        input_device={"name": "mic"},
        output_device=None,
        stimulus_snapshot=None,
        target_samples=2,
        output_path=root / f"{session_id}.wav",
        temp_path=root / f"{session_id}.tmp.wav",
        backup_path=None,
        record_id=session_id,
        label="not_labeled",
        analysis_config={},
        configuration_snapshot={},
        pending_count=1,
        replay=False,
    )
    bus = SequenceEventBus()
    diagnostics = []
    terminal_events = []
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        use_streaming=lambda prepared: True,
        streaming_adapter=lambda prepared, terminal: True,
        close_streaming_admission=lambda prepared: "closed",
        quiesce_streaming=lambda prepared, reason, handle: (_ for _ in ()).throw(error),
        workflow_generation_provider=lambda: 7,
        connect_queued=True,
    )
    bus.events.recording_completed.connect(terminal_events.append)
    bus.events.recording_failed.connect(terminal_events.append)
    bus.events.recording_cancelled.connect(terminal_events.append)
    loop = QEventLoop()
    controller.streaming_quiescence_failed.connect(
        lambda session, reason: (diagnostics.append((session, reason)), loop.quit())
    )
    assert controller.handle_begin_recording(
        BeginRecordingRequested(f"command-{index}", session_id, False, {"workflow_generation": 7})
    ) is True
    assert controller.handle_cancel_recording(
        CancelRecordingRequested(session_id, 7, "stop")
    ) is True
    QTimer.singleShot(5_000, loop.quit)
    loop.exec()
    assert len(diagnostics) == 1
    assert terminal_events == []
    assert controller.streaming_cancellation_pending is True
    controller.deleteLater()
    QCoreApplication.sendPostedEvents(controller, QEvent.DeferredDelete)

run(1, KeyboardInterrupt("keyboard stop"))
run(2, SystemExit("system stop"))
'''
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "stage",
    ["acquisition", "durable", "database", "persistence", "cancellation-cleanup"],
)
def test_native_threaded_worker_baseexceptions_never_escape_queued_delivery(
    tmp_path, stage
):
    script = r'''
import os
import sys
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
from PyQt5.QtCore import QEventLoop, QObject, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication

from functools import partial
from ui.sequence.sequence_event_bus import SequenceEventBus as _SequenceEventBus
SequenceEventBus = partial(
    _SequenceEventBus,
    standalone_recording_admission=True,
)
from ui.sequence.sequence_messages import BeginRecordingRequested, CancelRecordingRequested
from ui.sequence.sequence_recording_controller import PreparedRecordingSession, SequenceRecordingController
from ui.sequence.sequence_recording_model import RecordingModel, RecordingSessionSnapshot, StagedRecording
from ui.sequence.sequence_recording_transaction import RecordingTransaction

app = QApplication.instance() or QApplication([])
root = Path(sys.argv[1])
stage = sys.argv[2]


class Probe(QObject):
    reached = pyqtSignal()


def wait(signal, trigger=lambda: None):
    loop = QEventLoop()
    seen = []
    signal.connect(lambda *_args: (seen.append(True), loop.quit()))
    QTimer.singleShot(5_000, loop.quit)
    trigger()
    if not seen:
        loop.exec()
    assert seen


def snapshot(case_name):
    return RecordingSessionSnapshot.create(
        session_id=case_name,
        workflow_generation=7,
        configuration_generation=1,
        mode="RECORD_ONLY",
        sample_rate=48_000,
        bit_depth=32,
        input_channels=[0],
        input_device={"name": "mic"},
        output_device=None,
        stimulus_snapshot=None,
        target_samples=2,
        output_path=root / f"{case_name}.wav",
        temp_path=root / f"{case_name}.tmp.wav",
        backup_path=None,
        record_id=case_name,
        label="not_labeled",
        analysis_config={},
        configuration_snapshot={},
        pending_count=1,
        replay=False,
    )


def staged(item):
    mono = np.asarray([0.1, 0.2], dtype=np.float32)
    return StagedRecording.create(
        snapshot=item,
        sample_count=2,
        data_struct_fields={
            "store_wave_data": mono,
            "store_wave_data_multi": mono.reshape(-1, 1),
            "sample_rate": 48_000,
            "audio_lenth": 2,
        },
        recorded_signal_info={"file_path": str(item.output_path)},
        stimulus_info=None,
    )


def run(case_index, error):
    case_name = f"{stage}-{case_index}"
    item = snapshot(case_name)
    item.temp_path.write_bytes(b"new-audio")
    target_entered = Probe()
    release_target = Event()
    acquisition_entered = Probe()
    release_acquisition = Event()
    rollback_calls = []
    persistence_calls = []
    transaction_cleanup_calls = []
    events = []
    bus = SequenceEventBus()
    model = RecordingModel()
    for signal in (
        bus.events.recording_completed,
        bus.events.recording_failed,
        bus.events.recording_cancelled,
    ):
        signal.connect(events.append)

    def fail_target():
        target_entered.reached.emit()
        assert release_target.wait(5)
        raise error

    def acquire(_prepared):
        if stage == "acquisition":
            fail_target()
        if stage == "cancellation-cleanup":
            acquisition_entered.reached.emit()
            assert release_acquisition.wait(5)
        return staged(item)

    def promote(_staged):
        if stage == "durable" and case_index == 1:
            fail_target()
        os.replace(item.temp_path, item.output_path)

    def save_database(_info, _stimulus):
        if stage == "database":
            fail_target()

    def persist_count(_count):
        persistence_calls.append(True)
        if stage == "persistence":
            fail_target()

    def transaction_cleanup(_staged):
        transaction_cleanup_calls.append(True)
        item.temp_path.unlink(missing_ok=True)

    def rollback(_prepared, reason):
        rollback_calls.append(reason)
        if stage == "cancellation-cleanup":
            fail_target()
        item.temp_path.unlink(missing_ok=True)
        return {"restored": True, "errors": ()}

    class Transaction(RecordingTransaction):
        def prepare_durable(self, item_staged):
            if stage == "durable" and case_index == 2:
                self.begin_durable_commit()
                self.promote_output(item_staged)
                self.promotion_succeeded()
                fail_target()
            return super().prepare_durable(item_staged)

    controller = SequenceRecordingController(
        model,
        bus,
        prepare_session=lambda command: PreparedRecordingSession(item, {}),
        blocking_acquirer=acquire,
        transaction_factory=lambda prepared: Transaction(
            data_struct=SimpleNamespace(),
            promote_output=promote,
            save_database=save_database,
            commit_count=lambda _count: None,
            persist_count=persist_count,
            cleanup=transaction_cleanup,
        ),
        request_blocking_cancel=lambda prepared, reason: release_acquisition.set(),
        cancel_adapter=rollback,
        workflow_generation_provider=lambda: 7,
        connect_queued=True,
    )
    command = BeginRecordingRequested(
        f"command-{case_name}", case_name, False, {"workflow_generation": 7}
    )

    if stage == "cancellation-cleanup":
        wait(acquisition_entered.reached, lambda: controller.handle_begin_recording(command))
        assert controller.handle_cancel_recording(
            CancelRecordingRequested(case_name, 7, "operator stop")
        ) is True
        wait(target_entered.reached)
    else:
        wait(target_entered.reached, lambda: controller.handle_begin_recording(command))

    retained_port = controller._session.terminal
    terminal = Probe()
    for signal in (
        bus.events.recording_completed,
        bus.events.recording_failed,
        bus.events.recording_cancelled,
    ):
        signal.connect(lambda _event: terminal.reached.emit())
    wait(terminal.reached, release_target.set)
    drained = Probe()
    wait(drained.reached, lambda: QTimer.singleShot(0, drained.reached.emit))

    assert len(events) == 1
    assert model.active_session_id is None
    assert controller._session is None
    assert retained_port.recording_failed("late duplicate") is False
    app.processEvents()
    assert len(events) == 1

    event = events[0]
    if stage in {"acquisition", "durable"}:
        assert type(event).__name__ == "RecordingFailed"
        committed = stage == "durable" and case_index == 2
        assert event.audio_committed is committed
        assert event.recovery_path == (str(item.output_path) if committed else None)
        assert rollback_calls == ([] if committed else [str(error)])
    elif stage == "database":
        assert type(event).__name__ == "RecordingCompleted"
        warnings = event.result_snapshot["warnings"]
        assert [(entry["stage"], entry["message"]) for entry in warnings] == [
            ("database", str(error))
        ]
        assert transaction_cleanup_calls == [True]
    elif stage == "persistence":
        assert type(event).__name__ == "RecordingCompleted"
        warnings = event.result_snapshot["warnings"]
        assert [entry["message"] for entry in warnings] == [str(error), str(error)]
        assert persistence_calls == [True]
        assert transaction_cleanup_calls == []
    else:
        assert type(event).__name__ == "RecordingCancelled"
        assert rollback_calls == ["operator stop"]
        assert event.rollback_outcome["restored"] is False
        assert event.rollback_outcome["errors"] == (str(error),)


run(1, KeyboardInterrupt(f"{stage} keyboard interruption"))
run(2, SystemExit(f"{stage} system interruption"))
'''
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), stage],
        cwd=Path(__file__).parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )

    assert result.returncode == 0, result.stderr


def test_threaded_worker_queue_payload_detaches_original_exception_and_traceback():
    class TraceMarker:
        pass

    class TrackedInterruption(BaseException):
        pass

    marker = TraceMarker()
    error = TrackedInterruption("tracked interruption")
    error.marker = marker
    marker_ref = weakref.ref(marker)
    error_ref = weakref.ref(error)
    holder = [error]
    del marker
    del error
    direct_payloads = []
    queued_payloads = []

    def interrupt():
        current_error = holder.pop()
        raise current_error

    job = recording_controller_module.ThreadedBlockingAcquisitionJob(
        interrupt,
        queued_payloads.append,
        lambda _reason: None,
        None,
    )
    job._relay.finished.connect(direct_payloads.append, Qt.DirectConnection)
    assert job.start() is True
    job._thread.join(5)
    assert not job.is_alive()
    gc.collect()

    assert len(direct_payloads) == 1
    descriptor = direct_payloads[0].error
    assert not isinstance(descriptor, BaseException)
    assert descriptor.category == "base-exception"
    assert descriptor.module == "builtins"
    assert descriptor.name == "BaseException"
    assert descriptor.message == "tracked interruption"
    assert descriptor.diagnostic == "worker error: tracked interruption"
    assert error_ref() is None
    assert marker_ref() is None

    drained = _QtProbe()
    _wait_for_qt_signal(
        drained.reached,
        lambda: QTimer.singleShot(0, drained.reached.emit),
    )
    assert len(queued_payloads) == 1
    assert queued_payloads[0].error == descriptor


def test_threaded_worker_descriptor_survives_hostile_exception_stringification():
    class HostileInterruption(BaseException):
        def __str__(self):
            raise SystemExit("stringification failed")

    holder = [HostileInterruption()]
    payloads = []

    def interrupt():
        raise holder.pop()

    job = recording_controller_module.ThreadedBlockingAcquisitionJob(
        interrupt,
        lambda _outcome: None,
        lambda _reason: None,
        None,
    )
    job._relay.finished.connect(payloads.append, Qt.DirectConnection)
    assert job.start() is True
    job._thread.join(5)

    assert not job.is_alive()
    assert len(payloads) == 1
    assert payloads[0].error.message == "worker interrupted"
    assert payloads[0].error.diagnostic == "worker error: worker interrupted"


@pytest.mark.parametrize(
    "case",
    ["prepare", "executor-setup", "streaming-adapter", "cancel"],
)
def test_native_queued_commands_contain_handler_baseexceptions(tmp_path, case):
    script = r'''
import sys
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
from PyQt5.QtCore import QEventLoop, QObject, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication

from functools import partial
from ui.sequence.sequence_event_bus import SequenceEventBus as _SequenceEventBus
SequenceEventBus = partial(
    _SequenceEventBus,
    standalone_recording_admission=True,
)
from ui.sequence.sequence_messages import BeginRecordingRequested, CancelRecordingRequested
from ui.sequence.sequence_recording_controller import PreparedRecordingSession, SequenceRecordingController
from ui.sequence.sequence_recording_model import RecordingModel, RecordingSessionSnapshot, StagedRecording
from ui.sequence.sequence_recording_transaction import RecordingTransaction

app = QApplication.instance() or QApplication([])
root = Path(sys.argv[1])
case = sys.argv[2]


class Probe(QObject):
    reached = pyqtSignal()


def wait(signal, trigger=lambda: None):
    loop = QEventLoop()
    seen = []
    signal.connect(lambda *_args: (seen.append(True), loop.quit()))
    QTimer.singleShot(5_000, loop.quit)
    trigger()
    if not seen:
        loop.exec()
    assert seen


def drain(trigger=lambda: None):
    probe = Probe()
    wait(probe.reached, lambda: (trigger(), QTimer.singleShot(0, probe.reached.emit)))


def run(index, error):
    session_id = f"{case}-{index}"
    snapshot = RecordingSessionSnapshot.create(
        session_id=session_id,
        workflow_generation=7,
        configuration_generation=1,
        mode="RECORD_ONLY",
        sample_rate=48_000,
        bit_depth=32,
        input_channels=[0],
        input_device={"name": "mic"},
        output_device=None,
        stimulus_snapshot=None,
        target_samples=2,
        output_path=root / f"{session_id}.wav",
        temp_path=root / f"{session_id}.tmp.wav",
        backup_path=None,
        record_id=session_id,
        label="not_labeled",
        analysis_config={},
        configuration_snapshot={},
        pending_count=1,
        replay=False,
    )
    snapshot.temp_path.write_bytes(b"partial")
    events = []
    cleanup = []
    terminal_ports = []
    acquisition_entered = Probe()
    release_acquisition = Event()
    bus = SequenceEventBus()
    model = RecordingModel()
    for signal in (
        bus.events.recording_completed,
        bus.events.recording_failed,
        bus.events.recording_cancelled,
    ):
        signal.connect(events.append)

    mono = np.asarray([0.1, 0.2], dtype=np.float32)
    staged = StagedRecording.create(
        snapshot=snapshot,
        sample_count=2,
        data_struct_fields={
            "store_wave_data": mono,
            "store_wave_data_multi": mono.reshape(-1, 1),
            "sample_rate": 48_000,
            "audio_lenth": 2,
        },
        recorded_signal_info={"file_path": str(snapshot.output_path)},
        stimulus_info=None,
    )

    def prepare(_command):
        if case == "prepare":
            raise error
        return PreparedRecordingSession(snapshot, {})

    def acquire(_prepared):
        if case == "cancel":
            acquisition_entered.reached.emit()
            assert release_acquisition.wait(5)
        return staged

    class InterruptingExecutor:
        def create(self, *_args):
            raise error

    def streaming_adapter(_prepared, terminal):
        terminal_ports.append(terminal)
        if case == "streaming-adapter":
            raise error
        return True

    def rollback(_prepared, reason):
        cleanup.append(reason)
        snapshot.temp_path.unlink(missing_ok=True)
        return {"restored": True, "errors": ()}

    controller = SequenceRecordingController(
        model,
        bus,
        prepare_session=prepare,
        blocking_acquirer=acquire,
        transaction_factory=lambda prepared: RecordingTransaction(
            data_struct=SimpleNamespace()
        ),
        use_streaming=lambda prepared: case == "streaming-adapter",
        streaming_adapter=streaming_adapter,
        request_blocking_cancel=lambda prepared, reason: (_ for _ in ()).throw(error),
        cancel_adapter=rollback,
        blocking_executor=InterruptingExecutor() if case == "executor-setup" else None,
        workflow_generation_provider=lambda: 7,
        connect_queued=True,
    )
    command = BeginRecordingRequested(
        f"command-{session_id}", session_id, False, {"workflow_generation": 7}
    )

    if case == "cancel":
        wait(acquisition_entered.reached, lambda: bus.commands.begin_recording_requested.emit(command))
        retained_port = controller._session.terminal
        drain(lambda: bus.commands.cancel_recording_requested.emit(
            CancelRecordingRequested(session_id, 7, "operator stop")
        ))
        assert controller.model.active_session_id == session_id
        terminal = Probe()
        bus.events.recording_cancelled.connect(lambda _event: terminal.reached.emit())
        wait(terminal.reached, release_acquisition.set)
        assert type(events[0]).__name__ == "RecordingCancelled"
        assert cleanup == ["operator stop"]
        assert retained_port.recording_failed("late") is False
    else:
        terminal = Probe()
        bus.events.recording_failed.connect(lambda _event: terminal.reached.emit())
        wait(terminal.reached, lambda: bus.commands.begin_recording_requested.emit(command))
        assert type(events[0]).__name__ == "RecordingFailed"
        assert cleanup == ([] if case == "prepare" else [str(error)])
        if terminal_ports:
            assert terminal_ports[0].recording_failed("late") is False

    assert len(events) == 1
    assert model.active_session_id is None
    drain(lambda: bus.commands.begin_recording_requested.emit(command))
    assert len(events) == 1


run(1, KeyboardInterrupt(f"{case} keyboard interruption"))
run(2, SystemExit(f"{case} system interruption"))
'''
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), case],
        cwd=Path(__file__).parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )

    assert result.returncode == 0, result.stderr


def test_native_worker_terminal_relay_contains_main_handler_baseexceptions(tmp_path):
    script = r'''
import sys
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import numpy as np
from PyQt5.QtCore import QEventLoop, QObject, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication

from functools import partial
from ui.sequence.sequence_event_bus import SequenceEventBus as _SequenceEventBus
SequenceEventBus = partial(
    _SequenceEventBus,
    standalone_recording_admission=True,
)
from ui.sequence.sequence_messages import BeginRecordingRequested
from ui.sequence.sequence_recording_controller import PreparedRecordingSession, SequenceRecordingController
from ui.sequence.sequence_recording_model import RecordingModel, RecordingSessionSnapshot, StagedRecording

app = QApplication.instance() or QApplication([])
root = Path(sys.argv[1])


class Probe(QObject):
    reached = pyqtSignal()


def wait(signal, trigger=lambda: None):
    loop = QEventLoop()
    seen = []
    signal.connect(lambda *_args: (seen.append(True), loop.quit()))
    QTimer.singleShot(5_000, loop.quit)
    trigger()
    if not seen:
        loop.exec()
    assert seen


def run(index, error):
    session_id = f"terminal-{index}"
    snapshot = RecordingSessionSnapshot.create(
        session_id=session_id,
        workflow_generation=7,
        configuration_generation=1,
        mode="RECORD_ONLY",
        sample_rate=48_000,
        bit_depth=32,
        input_channels=[0],
        input_device={"name": "mic"},
        output_device=None,
        stimulus_snapshot=None,
        target_samples=2,
        output_path=root / f"{session_id}.wav",
        temp_path=root / f"{session_id}.tmp.wav",
        backup_path=None,
        record_id=session_id,
        label="not_labeled",
        analysis_config={},
        configuration_snapshot={},
        pending_count=1,
        replay=False,
    )
    snapshot.temp_path.write_bytes(b"audio")
    mono = np.asarray([0.1, 0.2], dtype=np.float32)
    staged = StagedRecording.create(
        snapshot=snapshot,
        sample_count=2,
        data_struct_fields={
            "store_wave_data": mono,
            "store_wave_data_multi": mono.reshape(-1, 1),
            "sample_rate": 48_000,
            "audio_lenth": 2,
        },
        recorded_signal_info={"file_path": str(snapshot.output_path)},
        stimulus_info=None,
    )
    ports = []
    events = []
    cleanup = []
    bus = SequenceEventBus()
    model = RecordingModel()
    for signal in (
        bus.events.recording_completed,
        bus.events.recording_failed,
        bus.events.recording_cancelled,
    ):
        signal.connect(events.append)
    controller = SequenceRecordingController(
        model,
        bus,
        prepare_session=lambda command: PreparedRecordingSession(snapshot, {}),
        use_streaming=lambda prepared: True,
        streaming_adapter=lambda prepared, terminal: ports.append(terminal) or True,
        transaction_factory=lambda prepared: (_ for _ in ()).throw(error),
        cancel_adapter=lambda prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=True,
    )
    assert controller.handle_begin_recording(
        BeginRecordingRequested(
            f"command-{session_id}", session_id, False, {"workflow_generation": 7}
        )
    ) is True
    results = []
    worker = Thread(target=lambda: results.append(ports[0].staged_recording_ready(staged)))
    worker.start()
    worker.join(5)
    assert results == [True]
    terminal = Probe()
    bus.events.recording_failed.connect(lambda _event: terminal.reached.emit())
    wait(terminal.reached)

    assert len(events) == 1
    assert type(events[0]).__name__ == "RecordingFailed"
    assert cleanup == [str(error)]
    assert model.active_session_id is None
    assert ports[0].recording_failed("late duplicate") is False
    app.processEvents()
    assert len(events) == 1


run(1, KeyboardInterrupt("terminal keyboard interruption"))
run(2, SystemExit("terminal system interruption"))
'''
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_threaded_worker_descriptor_is_total_bounded_and_graph_free():
    script = r'''
import gc
import weakref

from PyQt5.QtCore import Qt

from ui.sequence.sequence_recording_controller import ThreadedBlockingAcquisitionJob


class AttachedGraph:
    pass


class GraphStr(str):
    def __new__(cls, value, graph):
        instance = super().__new__(cls, value)
        instance.graph = graph
        return instance

    def __format__(self, _spec):
        raise SystemExit("format hook must not run")


class HostileExceptionMeta(type):
    def __getattribute__(cls, name):
        if name in {"__module__", "__name__"}:
            raise SystemExit(f"metadata hook must not run: {name}")
        return super().__getattribute__(name)


class HostileInterruption(BaseException, metaclass=HostileExceptionMeta):
    def __str__(self):
        return GraphStr("x" * 20_000, self.graph)


graph = AttachedGraph()
error = HostileInterruption(graph)
error.graph = graph
graph_ref = weakref.ref(graph)
error_ref = weakref.ref(error)
holder = [error]
del graph
del error
payloads = []


def interrupt():
    current_error = holder.pop()
    local_graph = current_error.graph
    assert local_graph is not None
    raise current_error


job = ThreadedBlockingAcquisitionJob(
    interrupt,
    lambda _outcome: None,
    lambda _reason: None,
    None,
)
job._relay.finished.connect(payloads.append, Qt.DirectConnection)
assert job.start() is True
job._thread.join(5)
gc.collect()

assert not job.is_alive()
assert len(payloads) == 1
descriptor = payloads[0].error
assert descriptor.category == "base-exception"
assert descriptor.module == "builtins"
assert descriptor.name == "BaseException"
assert 0 < len(descriptor.message) <= 512
assert 0 < len(descriptor.diagnostic) <= 640
for value in (
    descriptor.category,
    descriptor.module,
    descriptor.name,
    descriptor.message,
    descriptor.diagnostic,
):
    assert type(value) is str
assert error_ref() is None
assert graph_ref() is None

system_payloads = []
system_job = ThreadedBlockingAcquisitionJob(
    lambda: (_ for _ in ()).throw(SystemExit(GraphStr("y" * 20_000, AttachedGraph()))),
    lambda _outcome: None,
    lambda _reason: None,
    None,
)
system_job._relay.finished.connect(system_payloads.append, Qt.DirectConnection)
assert system_job.start() is True
system_job._thread.join(5)

assert not system_job.is_alive()
assert len(system_payloads) == 1
assert system_payloads[0].error.category == "base-exception"
assert type(system_payloads[0].error.message) is str
assert len(system_payloads[0].error.message) <= 512
'''
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("boundary", ["command", "worker", "terminal"])
def test_native_queued_recovery_ignores_logger_baseexceptions(tmp_path, boundary):
    script = r'''
import os
import sys
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
from PyQt5.QtCore import QEventLoop, QObject, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication

from functools import partial
from ui.sequence.sequence_event_bus import SequenceEventBus as _SequenceEventBus
SequenceEventBus = partial(
    _SequenceEventBus,
    standalone_recording_admission=True,
)
from ui.sequence.sequence_messages import BeginRecordingRequested
from ui.sequence.sequence_recording_controller import PreparedRecordingSession, SequenceRecordingController
from ui.sequence.sequence_recording_model import RecordingModel, RecordingSessionSnapshot, StagedRecording
from ui.sequence.sequence_recording_transaction import RecordingTransaction

app = QApplication.instance() or QApplication([])
root = Path(sys.argv[1])
boundary = sys.argv[2]


class Probe(QObject):
    reached = pyqtSignal()


def wait(signal, trigger=lambda: None):
    loop = QEventLoop()
    seen = []
    signal.connect(lambda *_args: (seen.append(True), loop.quit()))
    QTimer.singleShot(5_000, loop.quit)
    trigger()
    if not seen:
        loop.exec()
    assert seen


def snapshot(session_id):
    item = RecordingSessionSnapshot.create(
        session_id=session_id,
        workflow_generation=7,
        configuration_generation=1,
        mode="RECORD_ONLY",
        sample_rate=48_000,
        bit_depth=32,
        input_channels=[0],
        input_device={"name": "mic"},
        output_device=None,
        stimulus_snapshot=None,
        target_samples=2,
        output_path=root / f"{session_id}.wav",
        temp_path=root / f"{session_id}.tmp.wav",
        backup_path=None,
        record_id=session_id,
        label="not_labeled",
        analysis_config={},
        configuration_snapshot={},
        pending_count=1,
        replay=False,
    )
    item.temp_path.write_bytes(b"audio")
    return item


def staged(item):
    mono = np.asarray([0.1, 0.2], dtype=np.float32)
    return StagedRecording.create(
        snapshot=item,
        sample_count=2,
        data_struct_fields={
            "store_wave_data": mono,
            "store_wave_data_multi": mono.reshape(-1, 1),
            "sample_rate": 48_000,
            "audio_lenth": 2,
        },
        recorded_signal_info={"file_path": str(item.output_path)},
        stimulus_info=None,
    )


def run(index, logger_error):
    first = snapshot(f"{boundary}-{index}-first")
    second = snapshot(f"{boundary}-{index}-second")
    snapshots = {first.session_id: first, second.session_id: second}
    events = []
    cleanup = []
    ports = []
    worker_entered = Probe()
    release_worker = Event()
    first_job = []

    class Logger:
        def error(self, _message):
            raise logger_error

    bus = SequenceEventBus()
    model = RecordingModel()
    for signal in (
        bus.events.recording_completed,
        bus.events.recording_failed,
        bus.events.recording_cancelled,
    ):
        signal.connect(events.append)

    def prepare(command):
        return PreparedRecordingSession(snapshots[command.session_id], {})

    def acquire(prepared):
        if boundary == "worker" and prepared.snapshot.session_id == first.session_id:
            worker_entered.reached.emit()
            assert release_worker.wait(5)
            raise KeyboardInterrupt("worker interrupted")
        return staged(prepared.snapshot)

    def streaming_adapter(prepared, terminal):
        ports.append(terminal)
        if boundary == "command" and prepared.snapshot.session_id == first.session_id:
            raise KeyboardInterrupt("command interrupted")
        return True

    def transaction_factory(prepared):
        if boundary == "terminal" and prepared.snapshot.session_id == first.session_id:
            raise KeyboardInterrupt("terminal interrupted")
        return RecordingTransaction(data_struct=SimpleNamespace())

    def rollback(prepared, reason):
        cleanup.append((prepared.snapshot.session_id, reason))
        prepared.snapshot.temp_path.unlink(missing_ok=True)
        return {"restored": True, "errors": ()}

    controller = SequenceRecordingController(
        model,
        bus,
        prepare_session=prepare,
        blocking_acquirer=acquire,
        transaction_factory=transaction_factory,
        use_streaming=lambda prepared: boundary in {"command", "terminal"},
        streaming_adapter=streaming_adapter,
        cancel_adapter=rollback,
        workflow_generation_provider=lambda: 7,
        logger=Logger(),
        connect_queued=True,
    )

    first_command = BeginRecordingRequested(
        f"command-{first.session_id}", first.session_id, False, {"workflow_generation": 7}
    )
    terminal_probe = Probe()
    bus.events.recording_failed.connect(lambda event: terminal_probe.reached.emit())

    if boundary == "worker":
        wait(worker_entered.reached, lambda: bus.commands.begin_recording_requested.emit(first_command))
        first_job.append(controller._session.blocking_job)
        wait(terminal_probe.reached, release_worker.set)
        delivery_error = first_job[0]._relay.queued_delivery_error
    elif boundary == "terminal":
        assert controller.handle_begin_recording(first_command) is True
        worker = Thread(target=lambda: ports[0].staged_recording_ready(staged(first)))
        worker.start()
        worker.join(5)
        wait(terminal_probe.reached)
        delivery_error = ports[0]._boundary._relay.queued_delivery_error
    else:
        wait(terminal_probe.reached, lambda: bus.commands.begin_recording_requested.emit(first_command))
        delivery_error = controller._begin_guard.queued_delivery_error

    assert len(events) == 1
    assert type(events[0]).__name__ == "RecordingFailed"
    assert cleanup == [(first.session_id, f"{boundary} interrupted")]
    assert model.active_session_id is None
    assert delivery_error is None or (
        type(delivery_error) is str and len(delivery_error) <= 512
    )

    second_command = BeginRecordingRequested(
        f"command-{second.session_id}", second.session_id, False, {"workflow_generation": 7}
    )
    if boundary in {"command", "terminal"}:
        assert controller.handle_begin_recording(second_command) is True
        assert ports[-1].recording_failed("finish next session") is False
    else:
        completed = Probe()
        bus.events.recording_completed.connect(lambda _event: completed.reached.emit())
        wait(completed.reached, lambda: controller.handle_begin_recording(second_command))
    assert model.active_session_id is None
    assert len(events) == 2


run(1, RuntimeError("logger ordinary failure"))
run(2, SystemExit("logger system failure"))
'''
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), boundary],
        cwd=Path(__file__).parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )

    assert result.returncode == 0, result.stderr


def test_native_pre_admission_failures_keep_identity_history_strictly_bounded(
    tmp_path,
):
    script = r'''
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PyQt5.QtCore import QEventLoop, QObject, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication

from functools import partial
from ui.sequence.sequence_event_bus import SequenceEventBus as _SequenceEventBus
SequenceEventBus = partial(
    _SequenceEventBus,
    standalone_recording_admission=True,
)
from ui.sequence.sequence_messages import BeginRecordingRequested
from ui.sequence.sequence_recording_controller import (
    InlineBlockingAcquisitionExecutor,
    PreparedRecordingSession,
    SequenceRecordingController,
)
from ui.sequence.sequence_recording_model import (
    RecordingModel,
    RecordingSessionSnapshot,
    StagedRecording,
)
from ui.sequence.sequence_recording_transaction import RecordingTransaction

app = QApplication.instance() or QApplication([])
root = Path(sys.argv[1])
identity_limit = 64
unique_count = 600
provider_fails = [True]
failures = []
completed = []


class Probe(QObject):
    reached = pyqtSignal()


failure_probe = Probe()
completion_probe = Probe()
bus = SequenceEventBus()
model = RecordingModel()
bus.events.recording_failed.connect(
    lambda event: (
        failures.append(event),
        failure_probe.reached.emit() if len(failures) == unique_count else None,
    )
)
bus.events.recording_completed.connect(
    lambda event: (completed.append(event), completion_probe.reached.emit())
)


def wait(signal, trigger):
    loop = QEventLoop()
    seen = []
    signal.connect(lambda *_args: (seen.append(True), loop.quit()))
    QTimer.singleShot(5_000, loop.quit)
    trigger()
    if not seen:
        loop.exec()
    assert seen


def workflow_generation():
    if provider_fails[0]:
        raise RuntimeError("generation provider unavailable")
    return 7


def prepare(command):
    session_id = command.session_id
    snapshot = RecordingSessionSnapshot.create(
        session_id=session_id,
        workflow_generation=7,
        configuration_generation=1,
        mode="RECORD_ONLY",
        sample_rate=48_000,
        bit_depth=32,
        input_channels=[0],
        input_device={"name": "mic"},
        output_device=None,
        stimulus_snapshot=None,
        target_samples=2,
        output_path=root / f"{session_id}.wav",
        temp_path=root / f"{session_id}.tmp.wav",
        backup_path=None,
        record_id=session_id,
        label="not_labeled",
        analysis_config={},
        configuration_snapshot={},
        pending_count=1,
        replay=False,
    )
    snapshot.temp_path.write_bytes(b"audio")
    return PreparedRecordingSession(snapshot, {})


def acquire(prepared):
    mono = np.asarray([0.1, 0.2], dtype=np.float32)
    return StagedRecording.create(
        snapshot=prepared.snapshot,
        sample_count=2,
        data_struct_fields={
            "store_wave_data": mono,
            "store_wave_data_multi": mono.reshape(-1, 1),
            "sample_rate": 48_000,
            "audio_lenth": 2,
        },
        recorded_signal_info={"file_path": str(prepared.snapshot.output_path)},
        stimulus_info=None,
    )


inline = InlineBlockingAcquisitionExecutor()
controller = SequenceRecordingController(
    model,
    bus,
    prepare_session=prepare,
    blocking_acquirer=acquire,
    transaction_factory=lambda _prepared: RecordingTransaction(
        data_struct=SimpleNamespace()
    ),
    blocking_executor=inline,
    finalization_executor=inline,
    workflow_generation_provider=workflow_generation,
    recent_identity_limit=identity_limit,
    connect_queued=True,
)
commands = [
    BeginRecordingRequested(
        f"command-{index}",
        f"session-{index}",
        False,
        {"workflow_generation": 7},
    )
    for index in range(unique_count)
]


def emit_failed_commands_with_duplicates():
    for command in commands:
        bus.commands.begin_recording_requested.emit(command)
        bus.commands.begin_recording_requested.emit(command)


wait(failure_probe.reached, emit_failed_commands_with_duplicates)

expected_recent = {
    (command.command_id, command.session_id)
    for command in commands[-identity_limit:]
}
assert len(failures) == unique_count
assert {event.session_id for event in failures} == {
    command.session_id for command in commands
}
assert controller.recent_identity_count == identity_limit
assert len(controller._recent_identity_order) == identity_limit
assert controller._recent_identities == expected_recent
assert controller._queued_command_failure_identities == expected_recent
assert len(controller._published_terminal_identities) <= identity_limit

# Once the provider recovers, an identity evicted from every failure-history
# collection is a normal admission and cannot interfere with the next session.
evicted = commands[0]
assert (evicted.command_id, evicted.session_id) not in expected_recent
provider_fails[0] = False
wait(
    completion_probe.reached,
    lambda: bus.commands.begin_recording_requested.emit(evicted),
)
assert [event.session_id for event in completed] == [evicted.session_id]
assert model.active_session_id is None

fresh = BeginRecordingRequested(
    "command-fresh", "session-fresh", False, {"workflow_generation": 7}
)
wait(
    completion_probe.reached,
    lambda: bus.commands.begin_recording_requested.emit(fresh),
)
assert [event.session_id for event in completed] == [
    evicted.session_id,
    fresh.session_id,
]
assert len(failures) == unique_count
assert model.active_session_id is None
assert controller.recent_identity_count <= identity_limit
assert len(controller._recent_identity_order) <= identity_limit
assert len(controller._queued_command_failure_identities) <= identity_limit
assert len(controller._published_terminal_identities) <= identity_limit
'''
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_native_active_command_identity_survives_failure_history_eviction(tmp_path):
    script = r'''
import sys
from pathlib import Path

from PyQt5.QtCore import QEventLoop, QObject, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication

from functools import partial
from ui.sequence.sequence_event_bus import SequenceEventBus as _SequenceEventBus
SequenceEventBus = partial(
    _SequenceEventBus,
    standalone_recording_admission=True,
)
from ui.sequence.sequence_messages import BeginRecordingRequested
from ui.sequence.sequence_recording_controller import PreparedRecordingSession, SequenceRecordingController
from ui.sequence.sequence_recording_model import RecordingModel, RecordingSessionSnapshot

app = QApplication.instance() or QApplication([])
root = Path(sys.argv[1])
identity_limit = 8
provider_fails = [False]
ports = []
started = []
failed = []


class Probe(QObject):
    reached = pyqtSignal()


started_probe = Probe()
failed_probe = Probe()
completed_probe = Probe()
bus = SequenceEventBus()
bus.events.recording_started.connect(
    lambda event: (started.append(event), started_probe.reached.emit())
)
bus.events.recording_failed.connect(
    lambda event: (
        failed.append(event),
        failed_probe.reached.emit() if len(failed) == 24 else None,
    )
)
bus.events.recording_completed.connect(lambda _event: completed_probe.reached.emit())


def wait(signal, trigger):
    loop = QEventLoop()
    seen = []
    signal.connect(lambda *_args: (seen.append(True), loop.quit()))
    QTimer.singleShot(5_000, loop.quit)
    trigger()
    if not seen:
        loop.exec()
    assert seen


def workflow_generation():
    if provider_fails[0]:
        raise RuntimeError("generation provider unavailable")
    return 7


def prepare(command):
    snapshot = RecordingSessionSnapshot.create(
        session_id=command.session_id,
        workflow_generation=7,
        configuration_generation=1,
        mode="RECORD_ONLY",
        sample_rate=48_000,
        bit_depth=32,
        input_channels=[0],
        input_device={"name": "mic"},
        output_device=None,
        stimulus_snapshot=None,
        target_samples=2,
        output_path=root / f"{command.session_id}.wav",
        temp_path=root / f"{command.session_id}.tmp.wav",
        backup_path=None,
        record_id=command.session_id,
        label="not_labeled",
        analysis_config={},
        configuration_snapshot={},
        pending_count=1,
        replay=False,
    )
    return PreparedRecordingSession(snapshot, {})


controller = SequenceRecordingController(
    RecordingModel(),
    bus,
    prepare_session=prepare,
    use_streaming=lambda _prepared: True,
    streaming_adapter=lambda _prepared, terminal: ports.append(terminal) or True,
    workflow_generation_provider=workflow_generation,
    recent_identity_limit=identity_limit,
    connect_queued=True,
)
original = BeginRecordingRequested(
    "command-original", "session-original", False, {"workflow_generation": 7}
)
original_identity = (original.command_id, original.session_id)
wait(started_probe.reached, lambda: bus.commands.begin_recording_requested.emit(original))
assert len(started) == 1

provider_fails[0] = True
flood = [
    BeginRecordingRequested(
        f"command-flood-{index}",
        f"session-flood-{index}",
        False,
        {"workflow_generation": 7},
    )
    for index in range(24)
]
wait(
    failed_probe.reached,
    lambda: [
        controller._recover_queued_command_escape(
            "handle_begin_recording",
            command,
            RuntimeError("generation provider unavailable"),
        )
        for command in flood
    ],
)
assert len(failed) == len(flood)
assert original_identity not in controller._recent_identities
assert controller._active_command_identity == original_identity
assert len(controller._recent_identities) == identity_limit
assert len(controller._recent_identity_order) == identity_limit
assert len(controller._queued_command_failure_identities) == identity_limit
assert len(controller._recent_identities | {controller._active_command_identity}) == identity_limit + 1

# Even if the generation provider is currently failing, a delayed duplicate of
# the protected active command is not a new failure and cannot abort the session.
assert controller._deliver_queued_command("handle_begin_recording", original) is False
assert len(failed) == len(flood)
assert controller.model.active_session_id == original.session_id
assert controller._active_command_identity == original_identity

provider_fails[0] = False
wait(
    completed_probe.reached,
    lambda: ports[0].recording_completed(sample_count=2, result_snapshot={}),
)
assert controller._active_command_identity is None
assert controller._recent_identity_order[-1] == original_identity
assert list(controller._recent_identity_order).count(original_identity) == 1
assert original_identity in controller._recent_identities
assert len(controller._recent_identities) <= identity_limit
assert len(controller._queued_command_failure_identities) <= identity_limit
assert len(controller._published_terminal_identities) <= identity_limit

assert controller._deliver_queued_command("handle_begin_recording", original) is False
assert len(started) == 1

fresh = BeginRecordingRequested(
    "command-fresh", "session-fresh", False, {"workflow_generation": 7}
)
wait(started_probe.reached, lambda: bus.commands.begin_recording_requested.emit(fresh))
assert [event.session_id for event in started] == ["session-original", "session-fresh"]
assert ports[1].recording_failed("finish fresh session") is False
assert controller._active_command_identity is None
assert len(controller._recent_identities) <= identity_limit
assert len(controller._recent_identity_order) <= identity_limit
assert len(controller._queued_command_failure_identities) <= identity_limit
assert len(controller._published_terminal_identities) <= identity_limit
'''
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "retirement", ["failure", "cancel", "abort", "disconnect"]
)
def test_active_command_identity_is_reinserted_on_every_retirement_path(
    tmp_path, retirement
):
    identity_limit = 4
    provider_fails = [False]
    ports = []
    bus = SequenceEventBus()
    snapshot = _snapshot(tmp_path)
    original = BeginRecordingRequested(
        "command-original", snapshot.session_id, False, {"workflow_generation": 7}
    )
    original_identity = (original.command_id, original.session_id)

    def workflow_generation():
        if provider_fails[0]:
            raise RuntimeError("generation provider unavailable")
        return 7

    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda _command: PreparedRecordingSession(snapshot, {}),
        use_streaming=lambda _prepared: True,
        streaming_adapter=lambda _prepared, terminal: ports.append(terminal) or True,
        close_streaming_admission=lambda _prepared: "closed",
        quiesce_streaming=lambda _prepared, _reason, _handle: {"quiesced": True},
        cancel_adapter=lambda _prepared, _reason: {
            "restored": True,
            "errors": (),
        },
        workflow_generation_provider=workflow_generation,
        recent_identity_limit=identity_limit,
        connect_queued=False,
    )
    assert controller.handle_begin_recording(original) is True

    provider_fails[0] = True
    for index in range(identity_limit * 3):
        flood_command = BeginRecordingRequested(
            f"command-flood-{index}",
            f"session-flood-{index}",
            False,
            {"workflow_generation": 7},
        )
        controller._recover_queued_command_escape(
            "handle_begin_recording",
            flood_command,
            RuntimeError("generation provider unavailable"),
        )
    assert original_identity not in controller._recent_identities
    assert controller._active_command_identity == original_identity
    assert len(controller._recent_identities) == identity_limit
    assert len(controller._recent_identities | {controller._active_command_identity}) == identity_limit + 1

    provider_fails[0] = False
    if retirement == "failure":
        assert ports[0].recording_failed("device failed") is False
    elif retirement == "cancel":
        assert controller.handle_cancel_recording(
            CancelRecordingRequested(snapshot.session_id, 7, "operator stop")
        ) is True
    elif retirement == "abort":
        controller._interrupt_active_session(
            controller._session.token,
            KeyboardInterrupt("recording interrupted"),
            cleanup=True,
        )
    else:
        controller.disconnect()

    assert controller.model.active_session_id is None
    assert controller._active_command_identity is None
    assert controller._recent_identity_order[-1] == original_identity
    assert list(controller._recent_identity_order).count(original_identity) == 1
    assert original_identity in controller._recent_identities
    assert len(controller._recent_identities) <= identity_limit
    assert len(controller._recent_identity_order) <= identity_limit
    assert len(controller._queued_command_failure_identities) <= identity_limit
    assert len(controller._published_terminal_identities) <= identity_limit
    assert ports[0].recording_failed("late duplicate") is False
    assert list(controller._recent_identity_order).count(original_identity) == 1
    assert controller.handle_begin_recording(original) is False


def _prepared_admission_for_command(tmp_path, command):
    base = _snapshot(tmp_path)
    values = base.as_dict()
    values.update(
        session_id=command.session_id,
        output_path=tmp_path / f"{command.session_id}.wav",
        temp_path=tmp_path / f"{command.session_id}.tmp.wav",
        record_id=command.session_id,
        replay=command.replay,
    )
    return PreparedRecordingSession(
        RecordingSessionSnapshot.create(**values),
        {},
    )


def test_reentrant_disconnect_during_preparation_discards_admission_atomically(
    tmp_path,
):
    bus = SequenceEventBus()
    model = RecordingModel()
    controller_holder = []
    cleanup = []
    started = []
    terminals = []
    bus.events.recording_started.connect(started.append)
    command = BeginRecordingRequested(
        "command-reentrant", "session-reentrant", False, {"workflow_generation": 7}
    )

    def prepare(admitted):
        controller_holder[0].disconnect()
        return _prepared_admission_for_command(tmp_path, admitted)

    controller = SequenceRecordingController(
        model,
        bus,
        prepare_session=prepare,
        use_streaming=lambda _prepared: True,
        streaming_adapter=lambda _prepared, terminal: terminals.append(terminal)
        or True,
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    controller_holder.append(controller)

    assert controller.handle_begin_recording(command) is False
    assert controller._active is False
    assert controller._disconnecting is True
    assert controller._pending_admission is None
    assert controller._session is None
    assert controller._active_command_identity is None
    assert model.state is RecordingState.IDLE
    assert model.active_session_id is None
    assert started == []
    assert terminals == []
    assert cleanup == ["recording admission invalidated"]
    assert controller._recent_identity_order[-1] == (
        command.command_id,
        command.session_id,
    )
    assert controller.recent_identity_count <= controller._recent_identity_limit
    assert controller.recording_failed("late invalidated preparation") is False

    # The discarded preparation did not poison the shared model for a new
    # controller lifecycle.
    next_ports = []
    next_controller = SequenceRecordingController(
        model,
        SequenceEventBus(),
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        use_streaming=lambda _prepared: True,
        streaming_adapter=lambda _prepared, terminal: next_ports.append(terminal)
        or True,
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    next_command = BeginRecordingRequested(
        "command-next", "session-next", False, {"workflow_generation": 7}
    )
    assert next_controller.handle_begin_recording(next_command) is True
    assert next_ports[0].recording_failed("finish next lifecycle") is False


def test_pending_admission_rejects_competitor_before_provider_or_prepare(tmp_path):
    first_entered = Event()
    release_first = Event()
    provider_calls = []
    prepare_calls = []
    started = []
    started_probe = _QtProbe()
    ports = []
    bus = SequenceEventBus()
    bus.events.recording_started.connect(
        lambda event: (started.append(event), started_probe.reached.emit())
    )
    first = BeginRecordingRequested(
        "command-first", "session-first", False, {"workflow_generation": 7}
    )
    second = BeginRecordingRequested(
        "command-second", "session-second", False, {"workflow_generation": 7}
    )

    def provider():
        provider_calls.append(current_thread().ident)
        return 7

    def prepare(command):
        prepare_calls.append(command.command_id)
        if command == first:
            first_entered.set()
            assert release_first.wait(5)
        return _prepared_admission_for_command(tmp_path, command)

    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=prepare,
        use_streaming=lambda _prepared: True,
        streaming_adapter=lambda _prepared, terminal: ports.append(terminal) or True,
        workflow_generation_provider=provider,
        connect_queued=False,
    )
    first_results = []
    first_thread = Thread(
        target=lambda: first_results.append(controller.handle_begin_recording(first))
    )
    first_thread.start()
    assert first_entered.wait(5)
    try:
        second_result = controller.handle_begin_recording(second)
    finally:
        release_first.set()
        first_thread.join(5)

    assert not first_thread.is_alive()
    _wait_for_qt_signal(started_probe.reached)
    assert second_result is False
    assert first_results == [True]
    assert len(provider_calls) == 2
    assert prepare_calls == [first.command_id]
    assert [event.session_id for event in started] == [first.session_id]
    assert controller.model.active_session_id == first.session_id
    assert controller._pending_admission is None
    assert controller._active_command_identity == (first.command_id, first.session_id)
    assert controller.handle_cancel_recording(
        CancelRecordingRequested(first.session_id, 7, "test cleanup")
    ) is True


def test_disconnect_invalidates_delayed_preparation_and_cleans_once(tmp_path):
    prepare_entered = Event()
    release_prepare = Event()
    cleanup = []
    started = []
    ports = []
    bus = SequenceEventBus()
    bus.events.recording_started.connect(started.append)
    command = BeginRecordingRequested(
        "command-delayed", "session-delayed", False, {"workflow_generation": 7}
    )

    def prepare(admitted):
        prepare_entered.set()
        assert release_prepare.wait(5)
        return _prepared_admission_for_command(tmp_path, admitted)

    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=prepare,
        use_streaming=lambda _prepared: True,
        streaming_adapter=lambda _prepared, terminal: ports.append(terminal) or True,
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    results = []
    worker = Thread(
        target=lambda: results.append(controller.handle_begin_recording(command))
    )
    worker.start()
    assert prepare_entered.wait(5)
    reservation = controller._pending_admission
    assert reservation is not None
    assert reservation.gate.is_open()
    controller.disconnect()
    assert not reservation.gate.is_open()
    release_prepare.set()
    worker.join(5)

    assert not worker.is_alive()
    assert results == [False]
    assert cleanup == ["recording admission invalidated"]
    assert started == []
    assert ports == []
    assert controller._pending_admission is None
    assert controller._session is None
    assert controller._active_command_identity is None
    assert controller.model.state is RecordingState.IDLE
    assert controller.model.active_session_id is None
    assert controller._recent_identity_order[-1] == (
        command.command_id,
        command.session_id,
    )
    assert controller.recent_identity_count <= controller._recent_identity_limit


def test_stale_preparation_exception_cannot_clear_newer_pending_reservation(tmp_path):
    first_entered = Event()
    release_first = Event()
    second_entered = Event()
    release_second = Event()
    first = BeginRecordingRequested(
        "command-first", "session-first", False, {"workflow_generation": 7}
    )
    second = BeginRecordingRequested(
        "command-second", "session-second", False, {"workflow_generation": 7}
    )

    def prepare(command):
        if command == first:
            first_entered.set()
            assert release_first.wait(5)
            raise ValueError("first preparation failed late")
        second_entered.set()
        assert release_second.wait(5)
        return _prepared_admission_for_command(tmp_path, command)

    ports = []
    controller = SequenceRecordingController(
        RecordingModel(),
        SequenceEventBus(),
        prepare_session=prepare,
        use_streaming=lambda _prepared: True,
        streaming_adapter=lambda _prepared, terminal: ports.append(terminal) or True,
        workflow_generation_provider=lambda: 7,
        recent_identity_limit=4,
        connect_queued=False,
    )
    first_results = []
    second_results = []
    first_thread = Thread(
        target=lambda: first_results.append(controller.handle_begin_recording(first))
    )
    first_thread.start()
    assert first_entered.wait(5)
    first_reservation = controller._pending_admission
    with controller._lock:
        assert controller._clear_pending_admission_locked(first_reservation) is True

    second_thread = Thread(
        target=lambda: second_results.append(controller.handle_begin_recording(second))
    )
    second_thread.start()
    assert second_entered.wait(5)
    second_reservation = controller._pending_admission
    assert second_reservation is not None
    assert second_reservation is not first_reservation

    release_first.set()
    first_thread.join(5)
    assert not first_thread.is_alive()
    assert first_results == [False]
    assert controller._pending_admission is second_reservation
    assert second_reservation.gate.is_open()

    release_second.set()
    second_thread.join(5)
    assert not second_thread.is_alive()
    assert second_results == [True]
    assert controller._pending_admission is None
    assert not second_reservation.gate.is_open()
    assert controller.model.active_session_id == second.session_id
    assert controller._active_command_identity == (second.command_id, second.session_id)
    assert controller.recent_identity_count <= controller._recent_identity_limit
    assert controller.handle_cancel_recording(
        CancelRecordingRequested(second.session_id, 7, "finish replacement")
    ) is True


def test_pending_reservation_linearizes_before_simultaneous_provider_call(tmp_path):
    provider_entered = Event()
    release_provider = Event()
    provider_calls = []
    prepare_calls = []
    first = BeginRecordingRequested(
        "command-first", "session-first", False, {"workflow_generation": 7}
    )
    second = BeginRecordingRequested(
        "command-second", "session-second", False, {"workflow_generation": 7}
    )

    def provider():
        provider_calls.append(current_thread().ident)
        if len(provider_calls) == 1:
            provider_entered.set()
            assert release_provider.wait(5)
        return 7

    def prepare(command):
        prepare_calls.append(command.command_id)
        return _prepared_admission_for_command(tmp_path, command)

    ports = []
    controller = SequenceRecordingController(
        RecordingModel(),
        SequenceEventBus(),
        prepare_session=prepare,
        use_streaming=lambda _prepared: True,
        streaming_adapter=lambda _prepared, terminal: ports.append(terminal) or True,
        workflow_generation_provider=provider,
        connect_queued=False,
    )
    first_results = []
    first_thread = Thread(
        target=lambda: first_results.append(controller.handle_begin_recording(first))
    )
    first_thread.start()
    assert provider_entered.wait(5)
    try:
        second_result = controller.handle_begin_recording(second)
    finally:
        release_provider.set()
        first_thread.join(5)

    assert not first_thread.is_alive()
    assert second_result is False
    assert first_results == [True]
    assert len(provider_calls) == 2
    assert prepare_calls == [first.command_id]
    assert controller.model.active_session_id == first.session_id
    assert controller.handle_cancel_recording(
        CancelRecordingRequested(first.session_id, 7, "test cleanup")
    ) is True


def test_reentrant_generation_provider_cannot_admit_competing_begin(tmp_path):
    first = BeginRecordingRequested(
        "command-first", "session-first", False, {"workflow_generation": 7}
    )
    competitor = BeginRecordingRequested(
        "command-competitor", "session-competitor", False, {"workflow_generation": 7}
    )
    provider_calls = []
    prepare_calls = []
    competitor_results = []
    controller_holder = []

    def provider():
        provider_calls.append(True)
        if len(provider_calls) == 1:
            competitor_results.append(
                controller_holder[0].handle_begin_recording(competitor)
            )
        return 7

    def prepare(command):
        prepare_calls.append(command.command_id)
        return _prepared_admission_for_command(tmp_path, command)

    ports = []
    controller = SequenceRecordingController(
        RecordingModel(),
        SequenceEventBus(),
        prepare_session=prepare,
        use_streaming=lambda _prepared: True,
        streaming_adapter=lambda _prepared, terminal: ports.append(terminal) or True,
        workflow_generation_provider=provider,
        connect_queued=False,
    )
    controller_holder.append(controller)

    assert controller.handle_begin_recording(first) is True
    assert competitor_results == [False]
    assert len(provider_calls) == 2
    assert prepare_calls == [first.command_id]
    assert controller.model.active_session_id == first.session_id
    assert controller.handle_cancel_recording(
        CancelRecordingRequested(first.session_id, 7, "test cleanup")
    ) is True


def test_generation_change_during_prepare_discards_and_cleans_once(tmp_path):
    generations = [7, 8]
    cleanup = []
    started = []
    producer_started = []
    bus = SequenceEventBus()
    bus.events.recording_started.connect(started.append)
    command = BeginRecordingRequested(
        "command-generation", "session-generation", False, {"workflow_generation": 7}
    )
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        use_streaming=lambda _prepared: True,
        streaming_adapter=lambda _prepared, _terminal: producer_started.append(True)
        or True,
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: generations.pop(0),
        connect_queued=False,
    )

    assert controller.handle_begin_recording(command) is False
    assert generations == []
    assert cleanup == ["recording admission invalidated"]
    assert started == []
    assert producer_started == []
    assert controller._pending_admission is None
    assert controller._session is None
    assert controller._active_command_identity is None
    assert controller.model.state is RecordingState.IDLE


@pytest.mark.parametrize("callback", ["event", "view", "policy"])
@pytest.mark.parametrize("streaming", [False, True])
def test_start_callbacks_cannot_start_producer_after_reentrant_disconnect(
    tmp_path, callback, streaming
):
    producer_calls = []
    cleanup = []
    controller_holder = []
    bus = SequenceEventBus()

    class Job:
        def start(self):
            producer_calls.append("blocking-start")
            return True

        def cancel(self, _reason):
            return None

    class Executor:
        queued_delivery = False

        def create(self, *_args):
            producer_calls.append("blocking-create")
            return Job()

    def disconnect():
        controller_holder[0].disconnect()

    view = SequenceRecordingView(
        set_started=disconnect if callback == "view" else lambda: None,
    )
    if callback == "event":
        bus.events.recording_started.connect(lambda _event: disconnect())

    def use_streaming(_prepared):
        if callback == "policy":
            disconnect()
        return streaming

    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=view,
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        blocking_acquirer=lambda _prepared: producer_calls.append(
            "blocking-acquire"
        ),
        blocking_executor=Executor(),
        use_streaming=use_streaming,
        streaming_adapter=lambda _prepared, _terminal: producer_calls.append(
            "streaming-start"
        )
        or True,
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    controller_holder.append(controller)
    command = BeginRecordingRequested(
        f"command-{callback}-{streaming}",
        f"session-{callback}-{streaming}",
        False,
        {"workflow_generation": 7},
    )

    assert controller.handle_begin_recording(command) is False
    assert producer_calls == []
    assert cleanup == ["disconnect"]
    assert controller._session is None
    assert controller._active_command_identity is None
    assert controller.model.state is RecordingState.IDLE


@pytest.mark.parametrize(
    "stage",
    ["relay", "boundary", "port", "session", "model_begin", "model_mark", "model_fail"],
)
def test_admission_construction_interruption_rolls_back_transactionally(
    tmp_path, monkeypatch, stage
):
    original_gate = recording_controller_module._TerminalLifecycleGate
    created_gates = []

    class TrackingGate(original_gate):
        def __init__(self):
            super().__init__()
            created_gates.append(self)

    monkeypatch.setattr(
        recording_controller_module, "_TerminalLifecycleGate", TrackingGate
    )

    construction_targets = {
        "relay": "_TerminalRelay",
        "boundary": "_TerminalSubmissionBoundary",
        "port": "RecordingTerminalPort",
        "session": "_ActiveRecordingSession",
    }
    if stage in construction_targets:
        name = construction_targets[stage]
        original = getattr(recording_controller_module, name)
        interrupted = [False]

        def construct(*args, **kwargs):
            if not interrupted[0]:
                interrupted[0] = True
                raise KeyboardInterrupt(f"{stage} interrupted")
            return original(*args, **kwargs)

        monkeypatch.setattr(recording_controller_module, name, construct)

    class InterruptingModel(RecordingModel):
        def __init__(self):
            super().__init__()
            self.armed = True
            self.rolling_back = False

        def begin(self, snapshot):
            super().begin(snapshot)
            if stage == "model_begin" and self.armed:
                self.armed = False
                raise KeyboardInterrupt("model_begin interrupted")

        def mark_recording(self):
            super().mark_recording()
            if stage in {"model_mark", "model_fail"} and self.armed:
                self.armed = False
                self.rolling_back = True
                raise KeyboardInterrupt(f"{stage} interrupted")

        def fail(self):
            if stage == "model_fail" and self.rolling_back:
                self.rolling_back = False
                raise SystemExit("model rollback interrupted")
            return super().fail()

    cleanup = []
    ports = []
    started = []
    bus = SequenceEventBus()
    bus.events.recording_started.connect(started.append)
    model = InterruptingModel()
    controller = SequenceRecordingController(
        model,
        bus,
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        use_streaming=lambda _prepared: True,
        streaming_adapter=lambda _prepared, terminal: ports.append(terminal) or True,
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    first = BeginRecordingRequested(
        f"command-{stage}",
        f"session-{stage}",
        False,
        {"workflow_generation": 7},
    )

    with pytest.raises(KeyboardInterrupt, match=f"{stage} interrupted"):
        controller.handle_begin_recording(first)

    assert cleanup == [f"{stage} interrupted"]
    assert started == []
    assert ports == []
    assert controller._pending_admission is None
    assert controller._session is None
    assert controller._active_command_identity is None
    assert controller._validated_workflow_generation is None
    assert model.state is RecordingState.IDLE
    assert model.active_session_id is None
    assert model.active_snapshot is None
    assert model.final_result is None
    assert created_gates[2:]
    assert all(not gate.is_open() for gate in created_gates[2:])

    model.armed = False
    next_command = BeginRecordingRequested(
        f"command-next-{stage}",
        f"session-next-{stage}",
        False,
        {"workflow_generation": 7},
    )
    assert controller.handle_begin_recording(next_command) is True
    assert ports and ports[0].recording_failed("finish next") is False


def test_workflow_generation_provider_is_never_called_under_controller_lock(tmp_path):
    lock_states = []
    controller_holder = []

    def provider():
        controller = controller_holder[0]
        owned = getattr(controller._lock, "_is_owned", lambda: False)()
        lock_states.append(owned)
        return 7

    ports = []
    controller = SequenceRecordingController(
        RecordingModel(),
        SequenceEventBus(),
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        use_streaming=lambda _prepared: True,
        streaming_adapter=lambda _prepared, terminal: ports.append(terminal) or True,
        workflow_generation_provider=provider,
        connect_queued=False,
    )
    controller_holder.append(controller)
    command = BeginRecordingRequested(
        "command-lock-probe",
        "session-lock-probe",
        False,
        {"workflow_generation": 7},
    )

    assert controller.handle_begin_recording(command) is True
    assert lock_states == [False, False]
    assert ports[0].recording_failed("finish") is False
    assert lock_states == [False, False]


@pytest.mark.parametrize("streaming", [False, True])
def test_started_cancellation_reentry_cleans_without_starting_producer(
    tmp_path, streaming
):
    producer_calls = []
    cleanup = []
    cancelled = []
    controller_holder = []
    bus = SequenceEventBus()
    bus.events.recording_cancelled.connect(cancelled.append)

    def cancel_on_started(event):
        assert controller_holder[0].handle_cancel_recording(
            CancelRecordingRequested(event.session_id, 7, "callback cancel")
        ) is True

    bus.events.recording_started.connect(cancel_on_started)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        blocking_acquirer=lambda _prepared: producer_calls.append("blocking"),
        use_streaming=lambda _prepared: streaming,
        streaming_adapter=lambda _prepared, _terminal: producer_calls.append(
            "streaming"
        )
        or True,
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    controller_holder.append(controller)
    command = BeginRecordingRequested(
        f"command-cancel-{streaming}",
        f"session-cancel-{streaming}",
        False,
        {"workflow_generation": 7},
    )

    assert controller.handle_begin_recording(command) is False
    assert producer_calls == []
    assert cleanup == ["callback cancel"]
    assert [event.session_id for event in cancelled] == [command.session_id]
    assert controller._session is None
    assert controller.model.state is RecordingState.IDLE


def test_blocking_setup_disconnect_reentry_never_starts_created_job(tmp_path):
    calls = []
    cleanup = []
    controller_holder = []

    class Job:
        def cancel(self, reason):
            calls.append(("cancel", reason))

        def start(self):
            calls.append(("start", None))
            return True

    class Executor:
        queued_delivery = False

        def create(self, *_args):
            calls.append(("create", None))
            controller_holder[0].disconnect()
            return Job()

    controller = SequenceRecordingController(
        RecordingModel(),
        SequenceEventBus(),
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        blocking_acquirer=lambda _prepared: calls.append(("acquire", None)),
        blocking_executor=Executor(),
        use_streaming=lambda _prepared: False,
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    controller_holder.append(controller)
    command = BeginRecordingRequested(
        "command-blocking-setup",
        "session-blocking-setup",
        False,
        {"workflow_generation": 7},
    )

    assert controller.handle_begin_recording(command) is False
    assert calls == [("create", None), ("cancel", "disconnect")]
    assert cleanup == ["disconnect"]
    assert controller._session is None
    assert controller.model.state is RecordingState.IDLE


def test_queued_construction_interruption_publishes_once_and_next_begin_works(
    tmp_path, monkeypatch
):
    original_relay = recording_controller_module._TerminalRelay
    interrupted = [False]

    def relay(*args, **kwargs):
        if not interrupted[0]:
            interrupted[0] = True
            raise KeyboardInterrupt("queued relay interrupted")
        return original_relay(*args, **kwargs)

    monkeypatch.setattr(recording_controller_module, "_TerminalRelay", relay)
    failures = []
    cleanup = []
    ports = []
    bus = SequenceEventBus()
    bus.events.recording_failed.connect(failures.append)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        use_streaming=lambda _prepared: True,
        streaming_adapter=lambda _prepared, terminal: ports.append(terminal) or True,
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    interrupted_command = BeginRecordingRequested(
        "command-queued-interrupt",
        "session-queued-interrupt",
        False,
        {"workflow_generation": 7},
    )

    assert (
        controller._deliver_queued_command(
            "handle_begin_recording", interrupted_command
        )
        is False
    )
    assert cleanup == ["queued relay interrupted"]
    assert [event.session_id for event in failures] == [
        interrupted_command.session_id
    ]
    assert (
        controller._deliver_queued_command(
            "handle_begin_recording", interrupted_command
        )
        is False
    )
    assert len(failures) == 1

    next_command = BeginRecordingRequested(
        "command-after-queued-interrupt",
        "session-after-queued-interrupt",
        False,
        {"workflow_generation": 7},
    )
    assert controller.handle_begin_recording(next_command) is True
    assert ports[0].recording_failed("finish") is False


@pytest.mark.parametrize("stage", ["begin", "mark"])
@pytest.mark.parametrize("action", ["disconnect", "cancel"])
def test_model_transition_reentrant_disconnect_never_installs_zombie_session(
    tmp_path, stage, action
):
    controller_holder = []
    lock_states = []
    cleanup = []

    class ReentrantModel(RecordingModel):
        reenter = True

        def _invalidate(self):
            if not self.reenter:
                return
            self.reenter = False
            controller = controller_holder[0]
            lock_states.append(controller._lock._is_owned())
            if action == "disconnect":
                controller.disconnect()
            else:
                assert controller.handle_cancel_recording(
                    CancelRecordingRequested(
                        self.active_session_id,
                        7,
                        "model transition cancellation",
                    )
                ) is True

        def begin(self, snapshot):
            super().begin(snapshot)
            if stage == "begin":
                self._invalidate()

        def mark_recording(self):
            super().mark_recording()
            if stage == "mark":
                self._invalidate()

    model = ReentrantModel()
    controller = SequenceRecordingController(
        model,
        SequenceEventBus(),
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        use_streaming=lambda _prepared: True,
        streaming_adapter=lambda _prepared, _terminal: True,
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    controller_holder.append(controller)
    command = BeginRecordingRequested(
        f"command-model-{stage}-{action}",
        f"session-model-{stage}-{action}",
        False,
        {"workflow_generation": 7},
    )

    assert controller.handle_begin_recording(command) is False
    assert lock_states == [False]
    assert cleanup == ["recording admission invalidated"]
    assert controller._pending_admission is None
    assert controller._session is None
    assert controller._active_command_identity is None
    assert model.state is RecordingState.IDLE
    assert model.active_session_id is None

    next_ports = []
    next_controller = SequenceRecordingController(
        model,
        SequenceEventBus(),
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        use_streaming=lambda _prepared: True,
        streaming_adapter=lambda _prepared, terminal: next_ports.append(terminal)
        or True,
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    next_command = BeginRecordingRequested(
        f"command-next-{stage}-{action}",
        f"session-next-{stage}-{action}",
        False,
        {"workflow_generation": 7},
    )
    assert next_controller.handle_begin_recording(next_command) is True
    assert next_ports[0].recording_failed("finish next") is False


@pytest.mark.parametrize(
    ("stage", "interrupt_type"),
    [("begin", SystemExit), ("mark", KeyboardInterrupt)],
)
def test_model_transition_interruption_restores_outside_lock_even_if_fail_interrupts(
    tmp_path, stage, interrupt_type
):
    controller_holder = []
    lock_states = []

    class InterruptingModel(RecordingModel):
        def begin(self, snapshot):
            super().begin(snapshot)
            if stage == "begin":
                lock_states.append(controller_holder[0]._lock._is_owned())
                raise interrupt_type(f"{stage} transition interrupted")

        def mark_recording(self):
            super().mark_recording()
            if stage == "mark":
                lock_states.append(controller_holder[0]._lock._is_owned())
                raise interrupt_type(f"{stage} transition interrupted")

        def fail(self):
            lock_states.append(controller_holder[0]._lock._is_owned())
            raise SystemExit("model fail interrupted")

    model = InterruptingModel()
    cleanup = []
    controller = SequenceRecordingController(
        model,
        SequenceEventBus(),
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    controller_holder.append(controller)
    command = BeginRecordingRequested(
        "command-model-interrupt",
        "session-model-interrupt",
        False,
        {"workflow_generation": 7},
    )

    with pytest.raises(interrupt_type, match=f"{stage} transition interrupted"):
        controller.handle_begin_recording(command)

    assert lock_states == [False, False]
    assert cleanup == [f"{stage} transition interrupted"]
    assert controller._pending_admission is None
    assert controller._session is None
    assert model.state is RecordingState.IDLE
    assert model.active_session_id is None


@pytest.mark.parametrize("callback", ["event", "view"])
def test_started_callback_base_exception_retires_unstarted_session_once(
    tmp_path, monkeypatch, callback
):
    created_ports = []
    original_port = recording_controller_module.RecordingTerminalPort

    def tracking_port(*args, **kwargs):
        port = original_port(*args, **kwargs)
        created_ports.append(port)
        return port

    monkeypatch.setattr(
        recording_controller_module, "RecordingTerminalPort", tracking_port
    )
    cleanup = []

    class RaisingSignal:
        def emit(self, _event):
            raise KeyboardInterrupt("started event interrupted")

    class CanonicalTestBus:
        def __init__(self):
            self.events = SimpleNamespace(recording_started=RaisingSignal())
            self._capability = object()
            self._claimed = None

        def _bind_canonical_recording_consumer(self, _owner):
            return self._capability

        def _claim_canonical_recording_begin(self, capability, command):
            assert capability is self._capability
            assert self._claimed is None
            self._claimed = command
            return SimpleNamespace(
                cancellation=None,
                workflow_admitted=False,
            )

        def _retire_canonical_recording_admission(self, capability, command):
            assert capability is self._capability
            assert command is self._claimed
            self._claimed = None
            return True

        def _release_canonical_recording_consumer(self, capability):
            return capability is self._capability

    if callback == "event":
        bus = CanonicalTestBus()
        view = SequenceRecordingView()
        expected = "started event interrupted"
    else:
        bus = SequenceEventBus()

        def interrupt_view():
            raise KeyboardInterrupt("started view interrupted")

        view = SequenceRecordingView(set_started=interrupt_view)
        expected = "started view interrupted"

    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=view,
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        use_streaming=lambda _prepared: True,
        streaming_adapter=lambda _prepared, _terminal: True,
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    command = BeginRecordingRequested(
        f"command-started-{callback}",
        f"session-started-{callback}",
        False,
        {"workflow_generation": 7},
    )

    with pytest.raises(KeyboardInterrupt, match=expected):
        controller.handle_begin_recording(command)

    assert cleanup == [expected]
    assert controller._session is None
    assert controller._active_command_identity is None
    assert controller.model.state is RecordingState.IDLE
    assert len(created_ports) == 1
    assert not created_ports[0]._gate.is_open()


@pytest.mark.parametrize("action", ["cancel", "disconnect"])
@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_streaming_policy_invalidation_then_error_uses_cancellation_cleanup(
    tmp_path, action, error_type
):
    controller_holder = []
    cleanup = []
    producer_calls = []

    def policy(_prepared):
        controller = controller_holder[0]
        if action == "cancel":
            assert controller.handle_cancel_recording(
                CancelRecordingRequested(
                    f"session-policy-{action}-{error_type.__name__}",
                    7,
                    "policy cancellation",
                )
            ) is True
        else:
            controller.disconnect()
        raise error_type("policy interrupted after invalidation")

    controller = SequenceRecordingController(
        RecordingModel(),
        SequenceEventBus(),
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        use_streaming=policy,
        streaming_adapter=lambda _prepared, _terminal: producer_calls.append(True)
        or True,
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    controller_holder.append(controller)
    command = BeginRecordingRequested(
        f"command-policy-{action}-{error_type.__name__}",
        f"session-policy-{action}-{error_type.__name__}",
        False,
        {"workflow_generation": 7},
    )

    if error_type is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt, match="policy interrupted"):
            controller.handle_begin_recording(command)
    else:
        assert controller.handle_begin_recording(command) is False

    assert producer_calls == []
    assert cleanup == [
        "policy cancellation" if action == "cancel" else "disconnect"
    ]
    assert controller._session is None
    assert controller._active_command_identity is None
    assert controller.model.state is RecordingState.IDLE


def test_streaming_start_handshake_defers_disconnect_cleanup_until_start_returns(
    tmp_path
):
    start_entered = Event()
    release_start = Event()
    cleanup = []
    controller_holder = []

    def start_streaming(_prepared, _terminal):
        start_entered.set()
        assert release_start.wait(5)
        return True

    controller = SequenceRecordingController(
        RecordingModel(),
        SequenceEventBus(),
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        use_streaming=lambda _prepared: True,
        streaming_adapter=start_streaming,
        close_streaming_admission=lambda _prepared: "closed",
        quiesce_streaming=lambda _prepared, _reason, _handle: {"quiesced": True},
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    controller_holder.append(controller)
    command = BeginRecordingRequested(
        "command-streaming-handshake",
        "session-streaming-handshake",
        False,
        {"workflow_generation": 7},
    )
    results = []
    worker = Thread(
        target=lambda: results.append(controller.handle_begin_recording(command))
    )
    worker.start()
    assert start_entered.wait(5)

    controller.disconnect()
    assert cleanup == []
    assert controller._session is not None

    release_start.set()
    worker.join(5)
    assert not worker.is_alive()
    assert results == [False]
    assert cleanup == ["disconnect"]
    assert controller._session is None
    assert controller.model.state is RecordingState.IDLE


def test_blocking_start_handshake_cleans_when_disconnect_wins_during_no_start(
    tmp_path
):
    start_entered = Event()
    release_start = Event()
    calls = []
    cleanup = []

    class Job:
        def start(self):
            start_entered.set()
            assert release_start.wait(5)
            return False

        def cancel(self, reason):
            calls.append(("cancel", reason))

        def is_alive(self):
            return False

    class Executor:
        queued_delivery = False

        def create(self, *_args):
            return Job()

    controller = SequenceRecordingController(
        RecordingModel(),
        SequenceEventBus(),
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        blocking_acquirer=lambda _prepared: None,
        blocking_executor=Executor(),
        use_streaming=lambda _prepared: False,
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    command = BeginRecordingRequested(
        "command-blocking-handshake",
        "session-blocking-handshake",
        False,
        {"workflow_generation": 7},
    )
    results = []
    worker = Thread(
        target=lambda: results.append(controller.handle_begin_recording(command))
    )
    worker.start()
    assert start_entered.wait(5)

    controller.disconnect()
    assert cleanup == []
    assert calls == []

    release_start.set()
    worker.join(5)
    assert not worker.is_alive()
    assert results == [False]
    assert calls == []
    assert cleanup == ["disconnect"]
    assert controller._session is None
    assert controller.model.state is RecordingState.IDLE


def test_blocking_started_handshake_defers_cancel_until_start_returns_and_ack(
    tmp_path
):
    start_entered = Event()
    release_start = Event()
    deliver_holder = []
    calls = []
    cleanup = []

    class Job:
        def start(self):
            start_entered.set()
            assert release_start.wait(5)
            return True

        def cancel(self, reason):
            calls.append(("cancel", reason))

        def is_alive(self):
            return True

    class Executor:
        queued_delivery = False

        def create(self, _work, deliver, _cancel):
            deliver_holder.append(deliver)
            return Job()

    controller = SequenceRecordingController(
        RecordingModel(),
        SequenceEventBus(),
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        blocking_acquirer=lambda _prepared: None,
        blocking_executor=Executor(),
        use_streaming=lambda _prepared: False,
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    command = BeginRecordingRequested(
        "command-blocking-started-handshake",
        "session-blocking-started-handshake",
        False,
        {"workflow_generation": 7},
    )
    results = []
    worker = Thread(
        target=lambda: results.append(controller.handle_begin_recording(command))
    )
    worker.start()
    assert start_entered.wait(5)

    controller.disconnect()
    assert calls == []
    assert cleanup == []
    release_start.set()
    worker.join(5)

    assert not worker.is_alive()
    assert results == [False]
    assert calls == [("cancel", "disconnect")]
    assert cleanup == []
    assert controller._session is not None

    assert (
        deliver_holder[0](
            recording_controller_module._AcquisitionOutcome(
                error=RuntimeError("producer acknowledged cancellation")
            )
        )
        is False
    )
    assert cleanup == ["disconnect"]
    assert controller._session is None
    assert controller.model.state is RecordingState.IDLE


def test_blocking_start_reentrant_disconnect_and_completion_ack_returns_false(
    tmp_path
):
    controller_holder = []
    cleanup = []
    deliveries = []

    class Job:
        def __init__(self, deliver):
            self.deliver = deliver

        def start(self):
            controller_holder[0].disconnect()
            deliveries.append(
                self.deliver(
                    recording_controller_module._AcquisitionOutcome(
                        error=RuntimeError("producer stopped")
                    )
                )
            )
            return True

        def cancel(self, _reason):
            return None

        def is_alive(self):
            return False

    class Executor:
        queued_delivery = False

        def create(self, _work, deliver, _cancel):
            return Job(deliver)

    controller = SequenceRecordingController(
        RecordingModel(),
        SequenceEventBus(),
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        blocking_acquirer=lambda _prepared: None,
        blocking_executor=Executor(),
        use_streaming=lambda _prepared: False,
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    controller_holder.append(controller)
    command = BeginRecordingRequested(
        "command-blocking-reentrant-ack",
        "session-blocking-reentrant-ack",
        False,
        {"workflow_generation": 7},
    )

    assert controller.handle_begin_recording(command) is False
    # The producer acknowledgement is reserved while start() is still on the
    # stack, then reconciled against the already-winning disconnect.
    assert deliveries == [True]
    assert cleanup == ["disconnect"]
    assert controller._session is None
    assert controller.model.state is RecordingState.IDLE


def test_blocking_start_raise_waits_for_live_producer_completion_ack(tmp_path):
    controller_holder = []
    deliver_holder = []
    cleanup = []

    class Job:
        def start(self):
            controller_holder[0].disconnect()
            raise RuntimeError("start interrupted after launch")

        def cancel(self, _reason):
            return None

        def is_alive(self):
            return True

    class Executor:
        queued_delivery = False

        def create(self, _work, deliver, _cancel):
            deliver_holder.append(deliver)
            return Job()

    controller = SequenceRecordingController(
        RecordingModel(),
        SequenceEventBus(),
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        blocking_acquirer=lambda _prepared: None,
        blocking_executor=Executor(),
        use_streaming=lambda _prepared: False,
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    controller_holder.append(controller)
    command = BeginRecordingRequested(
        "command-blocking-live-after-raise",
        "session-blocking-live-after-raise",
        False,
        {"workflow_generation": 7},
    )

    assert controller.handle_begin_recording(command) is False
    assert cleanup == []
    assert controller._session is not None
    assert controller._session.cancellation_requested

    assert (
        deliver_holder[0](
            recording_controller_module._AcquisitionOutcome(
                error=RuntimeError("producer acknowledged stop")
            )
        )
        is False
    )
    assert cleanup == ["disconnect"]
    assert controller._session is None
    assert controller.model.state is RecordingState.IDLE


def test_disconnect_before_start_claim_closes_gate_and_producer_is_never_called(
    tmp_path
):
    policy_entered = Event()
    release_policy = Event()
    producer_calls = []
    cleanup = []

    def policy(_prepared):
        policy_entered.set()
        assert release_policy.wait(5)
        return True

    controller = SequenceRecordingController(
        RecordingModel(),
        SequenceEventBus(),
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        use_streaming=policy,
        streaming_adapter=lambda _prepared, _terminal: producer_calls.append(True)
        or True,
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    command = BeginRecordingRequested(
        "command-before-start-claim",
        "session-before-start-claim",
        False,
        {"workflow_generation": 7},
    )
    results = []
    worker = Thread(
        target=lambda: results.append(controller.handle_begin_recording(command))
    )
    worker.start()
    assert policy_entered.wait(5)

    controller.disconnect()
    assert producer_calls == []
    assert cleanup == []
    release_policy.set()
    worker.join(5)

    assert not worker.is_alive()
    assert results == [False]
    assert producer_calls == []
    assert cleanup == ["disconnect"]
    assert controller._session is None
    assert controller.model.state is RecordingState.IDLE


def test_queued_started_view_interruption_publishes_one_failure_after_cleanup(
    tmp_path
):
    failures = []
    cleanup = []
    bus = SequenceEventBus()
    bus.events.recording_failed.connect(failures.append)

    def interrupt_view():
        raise KeyboardInterrupt("queued started view interrupted")

    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        view=SequenceRecordingView(set_started=interrupt_view),
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        use_streaming=lambda _prepared: True,
        streaming_adapter=lambda _prepared, _terminal: True,
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    command = BeginRecordingRequested(
        "command-queued-started-view",
        "session-queued-started-view",
        False,
        {"workflow_generation": 7},
    )

    assert controller._deliver_queued_command("handle_begin_recording", command) is False
    assert cleanup == ["queued started view interrupted"]
    assert [event.session_id for event in failures] == [command.session_id]
    assert controller._session is None
    assert controller.model.state is RecordingState.IDLE
    assert controller._deliver_queued_command("handle_begin_recording", command) is False
    assert len(failures) == 1


def test_queued_blocking_start_interruption_defers_one_failure_until_ack(tmp_path):
    deliver_holder = []
    failures = []
    cleanup = []

    class Job:
        def start(self):
            raise KeyboardInterrupt("queued blocking start interrupted")

        def cancel(self, _reason):
            return None

        def is_alive(self):
            return True

    class Executor:
        queued_delivery = False

        def create(self, _work, deliver, _cancel):
            deliver_holder.append(deliver)
            return Job()

    bus = SequenceEventBus()
    bus.events.recording_failed.connect(failures.append)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        blocking_acquirer=lambda _prepared: None,
        blocking_executor=Executor(),
        use_streaming=lambda _prepared: False,
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    command = BeginRecordingRequested(
        "command-queued-blocking-start",
        "session-queued-blocking-start",
        False,
        {"workflow_generation": 7},
    )

    assert controller._deliver_queued_command("handle_begin_recording", command) is False
    assert failures == []
    assert cleanup == []
    assert controller._session is not None

    assert (
        deliver_holder[0](
            recording_controller_module._AcquisitionOutcome(
                error=RuntimeError("producer stopped after interruption")
            )
        )
        is False
    )
    assert cleanup == ["queued blocking start interrupted"]
    assert [event.session_id for event in failures] == [command.session_id]
    assert controller._session is None
    assert controller.model.state is RecordingState.IDLE


@pytest.mark.parametrize(
    ("operation", "expected_result", "event_name"),
    [
        ("completed", True, "recording_completed"),
        ("failed", False, "recording_failed"),
        ("cancelled", False, "recording_cancelled"),
    ],
)
def test_streaming_start_defers_first_synchronous_terminal_until_reconciled(
    tmp_path, operation, expected_result, event_name
):
    bus = SequenceEventBus()
    observed = []
    getattr(bus.events, event_name).connect(observed.append)
    submissions = []

    def start_streaming(_prepared, terminal):
        if operation == "completed":
            submissions.append(
                terminal.recording_completed(
                    sample_count=2, result_snapshot={"record_id": "sync"}
                )
            )
            submissions.append(
                terminal.recording_completed(sample_count=3, result_snapshot={})
            )
        elif operation == "failed":
            submissions.append(terminal.recording_failed("sync failure"))
            submissions.append(terminal.recording_failed("duplicate failure"))
        else:
            submissions.append(terminal.recording_cancelled("sync cancellation"))
            submissions.append(terminal.recording_cancelled("duplicate cancellation"))
        return True

    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        use_streaming=lambda _prepared: True,
        streaming_adapter=start_streaming,
        cancel_adapter=lambda _prepared, _reason: {
            "restored": True,
            "errors": (),
        },
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    command = BeginRecordingRequested(
        f"command-sync-{operation}",
        f"session-sync-{operation}",
        False,
        {"workflow_generation": 7},
    )

    assert controller.handle_begin_recording(command) is expected_result
    assert submissions == [True, False]
    assert len(observed) == 1
    assert observed[0].session_id == command.session_id
    assert controller._session is None
    assert controller.model.state is RecordingState.IDLE


def test_streaming_start_keeps_deferred_finalization_phase_after_adapter_returns(
    tmp_path
):
    submissions = []

    class DeferredJob:
        def start(self):
            return True

        def cancel(self, _reason):
            return None

    class DeferredExecutor:
        queued_delivery = False

        def create(self, work, deliver, _cancel):
            self.work = work
            self.deliver = deliver
            return DeferredJob()

    executor = DeferredExecutor()

    def start_streaming(prepared, terminal):
        submissions.append(terminal.staged_recording_ready(_staged(prepared.snapshot)))
        submissions.append(terminal.recording_failed("late conflicting terminal"))
        return True

    controller = SequenceRecordingController(
        RecordingModel(),
        SequenceEventBus(),
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        transaction_factory=lambda _prepared: RecordingTransaction(
            data_struct=SimpleNamespace()
        ),
        use_streaming=lambda _prepared: True,
        streaming_adapter=start_streaming,
        finalization_executor=executor,
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    command = BeginRecordingRequested(
        "command-sync-staged",
        "session-sync-staged",
        False,
        {"workflow_generation": 7},
    )

    assert controller.handle_begin_recording(command) is True
    assert submissions == [True, False]
    assert controller._session is not None
    assert controller._session.phase.name == "FINALIZING_PRECOMMIT"
    assert controller.model.state is RecordingState.FINALIZING


@pytest.mark.parametrize("action", ["cancel", "disconnect"])
def test_first_deferred_terminal_wins_later_starting_invalidation(tmp_path, action):
    controller_holder = []
    completed = []
    cancellation_results = []
    bus = SequenceEventBus()
    bus.events.recording_completed.connect(completed.append)

    def start_streaming(_prepared, terminal):
        assert terminal.recording_completed(
            sample_count=2, result_snapshot={"record_id": "first-terminal"}
        ) is True
        controller = controller_holder[0]
        if action == "cancel":
            cancellation_results.append(
                controller.handle_cancel_recording(
                    CancelRecordingRequested(
                        "session-first-terminal",
                        7,
                        "late cancellation",
                    )
                )
            )
        else:
            controller.disconnect()
        return True

    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        use_streaming=lambda _prepared: True,
        streaming_adapter=start_streaming,
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    controller_holder.append(controller)
    command = BeginRecordingRequested(
        "command-first-terminal",
        "session-first-terminal",
        False,
        {"workflow_generation": 7},
    )

    assert controller.handle_begin_recording(command) is True
    assert cancellation_results == ([False] if action == "cancel" else [])
    assert len(completed) == 1
    assert completed[0].session_id == command.session_id
    assert controller._session is None
    assert controller.model.state is RecordingState.IDLE


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_streaming_start_replays_first_terminal_even_when_adapter_raises(
    tmp_path, error_type
):
    completed = []
    bus = SequenceEventBus()
    bus.events.recording_completed.connect(completed.append)

    def start_streaming(_prepared, terminal):
        assert terminal.recording_completed(
            sample_count=2, result_snapshot={"record_id": "terminal-first"}
        ) is True
        raise error_type("adapter raised after terminal")

    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        use_streaming=lambda _prepared: True,
        streaming_adapter=start_streaming,
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    command = BeginRecordingRequested(
        f"command-terminal-before-{error_type.__name__}",
        f"session-terminal-before-{error_type.__name__}",
        False,
        {"workflow_generation": 7},
    )

    if error_type is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt, match="adapter raised after terminal"):
            controller.handle_begin_recording(command)
    else:
        assert controller.handle_begin_recording(command) is True

    assert len(completed) == 1
    assert completed[0].session_id == command.session_id
    assert controller._session is None
    assert controller.model.state is RecordingState.IDLE


def test_streaming_terminal_truth_wins_false_adapter_return(tmp_path):
    completed = []
    bus = SequenceEventBus()
    bus.events.recording_completed.connect(completed.append)

    def start_streaming(_prepared, terminal):
        assert terminal.recording_completed(
            sample_count=2, result_snapshot={"record_id": "terminal-before-false"}
        ) is True
        return False

    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        use_streaming=lambda _prepared: True,
        streaming_adapter=start_streaming,
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    command = BeginRecordingRequested(
        "command-terminal-before-false",
        "session-terminal-before-false",
        False,
        {"workflow_generation": 7},
    )

    assert controller.handle_begin_recording(command) is True
    assert len(completed) == 1
    assert completed[0].session_id == command.session_id
    assert controller._session is None
    assert controller.model.state is RecordingState.IDLE


def test_blocking_synchronous_failure_overrides_truthy_job_start_return(tmp_path):
    failures = []

    class Job:
        def __init__(self, deliver):
            self.deliver = deliver

        def start(self):
            assert self.deliver(
                recording_controller_module._AcquisitionOutcome(
                    error=RuntimeError("synchronous producer failure")
                )
            ) is True
            return True

        def cancel(self, _reason):
            return None

        def is_alive(self):
            return False

    class Executor:
        queued_delivery = False

        def create(self, _work, deliver, _cancel):
            return Job(deliver)

    bus = SequenceEventBus()
    bus.events.recording_failed.connect(failures.append)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        blocking_acquirer=lambda _prepared: None,
        blocking_executor=Executor(),
        use_streaming=lambda _prepared: False,
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    command = BeginRecordingRequested(
        "command-sync-blocking-failure",
        "session-sync-blocking-failure",
        False,
        {"workflow_generation": 7},
    )

    assert controller.handle_begin_recording(command) is False
    assert len(failures) == 1
    assert failures[0].reason == "synchronous producer failure"
    assert controller._session is None
    assert controller.model.state is RecordingState.IDLE


def test_blocking_synchronous_data_ack_does_not_override_starting_cancel(tmp_path):
    controller_holder = []
    cleanup = []
    deliveries = []

    class Job:
        def __init__(self, deliver):
            self.deliver = deliver

        def start(self):
            assert controller_holder[0].handle_cancel_recording(
                CancelRecordingRequested(
                    "session-sync-blocking-cancel",
                    7,
                    "synchronous cancellation wins",
                )
            ) is True
            prepared = controller_holder[0]._session.prepared
            deliveries.append(
                self.deliver(
                    recording_controller_module._AcquisitionOutcome(
                        staged=_staged(prepared.snapshot)
                    )
                )
            )
            return True

        def cancel(self, _reason):
            return None

        def is_alive(self):
            return False

    class Executor:
        queued_delivery = False

        def create(self, _work, deliver, _cancel):
            return Job(deliver)

    controller = SequenceRecordingController(
        RecordingModel(),
        SequenceEventBus(),
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        blocking_acquirer=lambda _prepared: None,
        blocking_executor=Executor(),
        use_streaming=lambda _prepared: False,
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    controller_holder.append(controller)
    command = BeginRecordingRequested(
        "command-sync-blocking-cancel",
        "session-sync-blocking-cancel",
        False,
        {"workflow_generation": 7},
    )

    assert controller.handle_begin_recording(command) is False
    assert deliveries == [True]
    assert cleanup == ["synchronous cancellation wins"]
    assert controller._session is None
    assert controller.model.state is RecordingState.IDLE


@pytest.mark.parametrize("action", ["cancel", "disconnect"])
@pytest.mark.parametrize("liveness", ["live", "ambiguous"])
def test_queued_blocking_start_interruption_waits_for_cancellation_ack(
    tmp_path, action, liveness
):
    controller_holder = []
    deliveries = []
    cleanup = []
    cancelled = []

    class Job:
        def start(self):
            controller = controller_holder[0]
            if action == "cancel":
                assert controller.handle_cancel_recording(
                    CancelRecordingRequested(
                        "session-queued-start-cancel",
                        7,
                        "operator cancellation wins",
                    )
                ) is True
            else:
                controller.disconnect()
            raise KeyboardInterrupt("queued start interruption diagnostic")

        def cancel(self, _reason):
            return None

        def is_alive(self):
            if liveness == "ambiguous":
                raise SystemExit("liveness unavailable")
            return True

    class Executor:
        queued_delivery = False

        def create(self, _work, deliver, _cancel):
            deliveries.append(deliver)
            return Job()

    bus = SequenceEventBus()
    bus.events.recording_cancelled.connect(cancelled.append)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        blocking_acquirer=lambda _prepared: None,
        blocking_executor=Executor(),
        use_streaming=lambda _prepared: False,
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    controller_holder.append(controller)
    command = BeginRecordingRequested(
        "command-queued-start-cancel",
        "session-queued-start-cancel",
        False,
        {"workflow_generation": 7},
    )

    assert controller._deliver_queued_command("handle_begin_recording", command) is False
    assert cleanup == []
    assert controller._session is not None
    expected_reason = "operator cancellation wins" if action == "cancel" else "disconnect"
    assert controller._session.cancellation_reason == expected_reason
    assert "queued start interruption diagnostic" in (
        controller._session.deferred_start_failure_reason
    )

    outcome = recording_controller_module._AcquisitionOutcome(
        error=RuntimeError("producer acknowledged cancellation")
    )
    assert deliveries[0](outcome) is (action == "cancel")
    assert deliveries[0](outcome) is False
    assert cleanup == [expected_reason]
    assert len(cancelled) == (1 if action == "cancel" else 0)
    assert controller._session is None
    assert controller.model.state is RecordingState.IDLE


@pytest.mark.parametrize("action", ["cancel", "disconnect"])
def test_queued_streaming_start_interruption_waits_for_quiescence_ack(
    tmp_path, action
):
    controller_holder = []
    cleanup = []
    cancelled = []

    class Job:
        def __init__(self, owner, work, deliver):
            self.owner = owner
            self.work = work
            self.deliver = deliver

        def start(self):
            if self.owner.create_count == 1:
                return True
            try:
                outcome = recording_controller_module._AcquisitionOutcome(
                    staged=self.work()
                )
            except BaseException as error:
                outcome = recording_controller_module._AcquisitionOutcome(error=error)
            return self.deliver(outcome)

        def cancel(self, _reason):
            return None

    class AckExecutor:
        queued_delivery = False

        def __init__(self):
            self.create_count = 0
            self.first = None

        def create(self, work, deliver, _cancel):
            self.create_count += 1
            job = Job(self, work, deliver)
            if self.create_count == 1:
                self.first = job
            return job

    executor = AckExecutor()

    def start_streaming(_prepared, _terminal):
        controller = controller_holder[0]
        if action == "cancel":
            assert controller.handle_cancel_recording(
                CancelRecordingRequested(
                    "session-streaming-start-interrupt",
                    7,
                    "streaming operator cancellation",
                )
            ) is True
        else:
            controller.disconnect()
        raise KeyboardInterrupt("streaming start interruption diagnostic")

    bus = SequenceEventBus()
    bus.events.recording_cancelled.connect(cancelled.append)
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        use_streaming=lambda _prepared: True,
        streaming_adapter=start_streaming,
        close_streaming_admission=lambda _prepared: "closed",
        quiesce_streaming=lambda _prepared, _reason, _handle: {"quiesced": True},
        finalization_executor=executor,
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    controller_holder.append(controller)
    command = BeginRecordingRequested(
        "command-streaming-start-interrupt",
        "session-streaming-start-interrupt",
        False,
        {"workflow_generation": 7},
    )

    assert controller._deliver_queued_command("handle_begin_recording", command) is False
    assert cleanup == []
    assert controller._session is not None
    expected_reason = (
        "streaming operator cancellation" if action == "cancel" else "disconnect"
    )
    assert controller._session.cancellation_reason == expected_reason
    assert "streaming start interruption diagnostic" in (
        controller._session.deferred_start_failure_reason
    )

    ack = recording_controller_module._AcquisitionOutcome(
        staged={"quiesced": True}
    )
    assert executor.first.deliver(ack) is (action == "cancel")
    assert executor.first.deliver(ack) is False
    assert cleanup == [expected_reason]
    assert len(cancelled) == (1 if action == "cancel" else 0)
    assert controller._session is None
    assert controller.model.state is RecordingState.IDLE


def test_terminal_failure_model_transition_and_reentrant_disconnect_are_outside_lock(
    tmp_path
):
    controller_holder = []
    lock_states = []
    calls = []

    class ReentrantInterruptingModel(RecordingModel):
        def fail(self):
            lock_states.append(controller_holder[0]._lock._is_owned())
            calls.append("fail")
            if len(calls) == 1:
                controller_holder[0].disconnect()
                raise KeyboardInterrupt("terminal model transition interrupted")
            super().fail()

    ports = []
    model = ReentrantInterruptingModel()
    controller = SequenceRecordingController(
        model,
        SequenceEventBus(),
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        use_streaming=lambda _prepared: True,
        streaming_adapter=lambda _prepared, terminal: ports.append(terminal) or True,
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    controller_holder.append(controller)
    command = BeginRecordingRequested(
        "command-terminal-lock-probe",
        "session-terminal-lock-probe",
        False,
        {"workflow_generation": 7},
    )
    assert controller.handle_begin_recording(command) is True

    with pytest.raises(KeyboardInterrupt, match="terminal model transition interrupted"):
        ports[0].recording_failed("device failed")

    assert lock_states == [False, False]
    assert controller._session is None
    assert model.state is RecordingState.IDLE


def test_post_durable_disconnect_model_transition_is_outside_lock_and_retires(
    tmp_path
):
    controller_holder = []
    lock_states = []

    class InterruptingModel(RecordingModel):
        def fail(self):
            lock_states.append(controller_holder[0]._lock._is_owned())
            controller_holder[0].disconnect()
            raise KeyboardInterrupt("durable disconnect model interrupted")

    model = InterruptingModel()
    controller = SequenceRecordingController(
        model,
        SequenceEventBus(),
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        use_streaming=lambda _prepared: True,
        streaming_adapter=lambda _prepared, _terminal: True,
        workflow_generation_provider=lambda: 7,
        connect_queued=False,
    )
    controller_holder.append(controller)
    command = BeginRecordingRequested(
        "command-durable-disconnect-lock-probe",
        "session-durable-disconnect-lock-probe",
        False,
        {"workflow_generation": 7},
    )
    assert controller.handle_begin_recording(command) is True
    session = controller._session
    token = session.token
    staged = _staged(session.prepared.snapshot)
    session.phase = recording_controller_module._RecordingSessionPhase.PROMOTION_SUCCEEDED
    session.transaction = RecordingTransaction(data_struct=SimpleNamespace())
    controller.disconnect()
    durable = recording_controller_module.RecordingDurableResult(
        staged=staged,
        fields={},
        audio_committed=True,
        recovery_path=staged.snapshot.output_path,
        warnings=(),
        rollback_outcome={},
    )

    with pytest.raises(KeyboardInterrupt, match="durable disconnect model interrupted"):
        controller._deliver_durable_result(
            token,
            recording_controller_module._AcquisitionOutcome(staged=durable),
        )

    assert lock_states == [False]
    assert controller._session is None
    assert controller._active is False


def test_native_queued_start_interruption_keeps_live_cancel_ack_owned(tmp_path):
    controller_holder = []
    deliveries = []
    cleanup = []

    class Job:
        def start(self):
            assert controller_holder[0].handle_cancel_recording(
                CancelRecordingRequested(
                    "session-native-queued-start",
                    7,
                    "native queued cancellation",
                )
            ) is True
            raise KeyboardInterrupt("native queued start interruption")

        def cancel(self, _reason):
            return None

        def is_alive(self):
            return True

    class Executor:
        queued_delivery = False

        def create(self, _work, deliver, _cancel):
            deliveries.append(deliver)
            return Job()

    bus = SequenceEventBus()
    controller = SequenceRecordingController(
        RecordingModel(),
        bus,
        prepare_session=lambda admitted: _prepared_admission_for_command(
            tmp_path, admitted
        ),
        blocking_acquirer=lambda _prepared: None,
        blocking_executor=Executor(),
        finalization_executor=recording_controller_module.InlineBlockingAcquisitionExecutor(),
        use_streaming=lambda _prepared: False,
        cancel_adapter=lambda _prepared, reason: cleanup.append(reason)
        or {"restored": True, "errors": ()},
        workflow_generation_provider=lambda: 7,
        connect_queued=True,
    )
    controller_holder.append(controller)
    command = BeginRecordingRequested(
        "command-native-queued-start",
        "session-native-queued-start",
        False,
        {"workflow_generation": 7},
    )
    drained = _QtProbe()

    _wait_for_qt_signal(
        drained.reached,
        lambda: (
            bus.commands.begin_recording_requested.emit(command),
            QTimer.singleShot(0, drained.reached.emit),
        ),
    )

    assert cleanup == []
    assert controller._session is not None
    assert controller._session.cancellation_reason == "native queued cancellation"
    assert "native queued start interruption" in (
        controller._session.deferred_start_failure_reason
    )
    assert deliveries[0](
        recording_controller_module._AcquisitionOutcome(
            error=RuntimeError("native producer acknowledgement")
        )
    ) is True
    assert cleanup == ["native queued cancellation"]
    assert controller._session is None


def test_recording_controller_has_no_direct_model_transition_inside_its_lock():
    source_path = (
        Path(__file__).parents[2]
        / "ui"
        / "sequence"
        / "sequence_recording_controller.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    controller = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "SequenceRecordingController"
    )
    transitions = {"begin", "mark_recording", "mark_finalizing", "finish", "fail"}
    locked_transitions = []
    for node in ast.walk(controller):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        owns_controller_lock = any(
            isinstance(item.context_expr, ast.Attribute)
            and isinstance(item.context_expr.value, ast.Name)
            and item.context_expr.value.id == "self"
            and item.context_expr.attr == "_lock"
            for item in node.items
        )
        if not owns_controller_lock:
            continue
        for descendant in ast.walk(node):
            if (
                isinstance(descendant, ast.Call)
                and isinstance(descendant.func, ast.Attribute)
                and descendant.func.attr in transitions
                and isinstance(descendant.func.value, ast.Attribute)
                and isinstance(descendant.func.value.value, ast.Name)
                and descendant.func.value.value.id == "self"
                and descendant.func.value.attr == "model"
            ):
                locked_transitions.append(descendant.func.attr)

    assert locked_transitions == []

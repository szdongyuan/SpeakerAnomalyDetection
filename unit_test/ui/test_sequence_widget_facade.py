from __future__ import annotations

import ast
import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt5.QtWidgets import QApplication, QWidget


try:
    import base.analysis_warning_preferences  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "base.analysis_warning_preferences":
        raise
    warning_preferences = types.ModuleType("base.analysis_warning_preferences")
    warning_preferences.is_uncalibrated_microphone_warning_suppressed = (
        lambda logger=None: False
    )
    warning_preferences.save_uncalibrated_microphone_warning_suppressed = (
        lambda logger=None: None
    )
    sys.modules[warning_preferences.__name__] = warning_preferences


from ui.sequence.sequence_messages import ConfigurationSnapshot
from ui.sequence.sequence_widget import SequenceWindow
from ui.sequence.sequence_workflow_model import WorkflowPhase


ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_WIDGET = ROOT / "ui" / "sequence" / "sequence_widget.py"
MAIN_WINDOW = ROOT / "main_window.py"
RESOURCE_LIFECYCLE = (
    ROOT / "ui" / "sequence" / "sequence_resource_lifecycle_controller.py"
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def facade(qapp):
    from base.load_config import LoadUiConfig

    patch = pytest.MonkeyPatch()
    patch.setattr(
        LoadUiConfig,
        "get_tcp_config",
        staticmethod(lambda: ("127.0.0.1", 0)),
    )
    window = SequenceWindow()
    yield window
    window.close()
    qapp.processEvents()
    patch.undo()


def test_explicit_facade_properties_are_owned_by_composed_models(facade):
    mic = {"name": "input", "samplerate": 48_000}
    speaker = {"name": "output", "samplerate": 48_000}
    channels = [1, 3]
    sequence = [{"seq1": {"acq": {"mode": "RECORD_ONLY", "detail": {}}}}]
    analysis = {"display_sequence": ["SPL"]}

    facade.mic = mic
    facade.speaker = speaker
    facade.mic_channels = channels
    facade.sequence_config = sequence
    facade.analysis_config = analysis
    facade.using_config_path = "configs/product.json"
    facade.streaming_stimulus_data = [0.25, -0.25]

    assert facade.mic == facade.configuration_model.mic == mic
    assert facade.speaker == facade.configuration_model.speaker == speaker
    assert facade.mic_channels == facade.configuration_model.mic_channels == channels
    assert facade.sequence_config == facade.configuration_model.sequence_config == sequence
    assert facade.analysis_config == facade.configuration_model.analysis_config == analysis
    assert facade.using_config_path == facade.configuration_model.using_config_path
    assert facade.data_struct is facade.configuration_model.data_struct
    assert facade.streaming_stimulus_data is facade.configuration_model.streaming_stimulus_data


def test_runtime_v2pa_device_and_channel_refresh_are_explicit_delegates(
    facade, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        facade.configuration_controller,
        "init_data_struct_stimulus_config",
        lambda: calls.append(("runtime",)) or "runtime-result",
    )
    monkeypatch.setattr(
        facade.configuration_controller,
        "on_sequence_config_updated",
        lambda *args: calls.append(("configuration", args)) or "configuration-result",
    )
    monkeypatch.setattr(
        facade.configuration_controller,
        "set_audio_devices_available",
        lambda available, message="": calls.append(("devices", available, message)),
    )
    monkeypatch.setattr(
        facade.analysis_controller,
        "update_v2pa_factor",
        lambda: calls.append(("v2pa",)) or "v2pa-result",
    )
    monkeypatch.setattr(
        facade,
        "_refresh_channel_workspace",
        lambda: calls.append(("channels",)) or "channel-result",
    )

    assert facade.init_data_struct_stimulus_config() == "runtime-result"
    assert facade.on_sequence_config_updated("dialog") is None
    assert facade.set_audio_devices_available(False, "missing") is None
    assert facade.update_v2pa_factor() == "v2pa-result"
    assert facade.refresh_channel_windows() == "channel-result"
    assert calls == [
        ("runtime",),
        ("configuration", ("dialog",)),
        ("devices", False, "missing"),
        ("v2pa",),
        ("channels",),
    ]


def test_busy_flags_are_read_only_canonical_workflow_projections(facade):
    facade.workflow_model.phase = WorkflowPhase.IDLE
    assert facade.player_status_flag is False
    assert facade._record_workflow_busy is False
    assert facade.is_workflow_active() is False

    facade.workflow_model.phase = WorkflowPhase.RECORDING
    assert facade.player_status_flag is True
    assert facade._record_workflow_busy is True
    assert facade.is_workflow_active() is True

    with pytest.raises(AttributeError):
        facade.player_status_flag = False
    with pytest.raises(AttributeError):
        facade._record_workflow_busy = False
    assert "_legacy_player_status_flag" not in facade.__dict__
    assert "_legacy_record_workflow_busy" not in facade.__dict__


def test_tcp_class_mirror_and_synchronous_flush_contract(facade):
    server = SimpleNamespace(stop=lambda: True)
    assert facade._set_tcp_mirror_identity(server) is True
    assert SequenceWindow.tcp_server is server
    SequenceWindow.tcp_server = None
    assert facade._get_tcp_mirror_identity() is None

    calls = []
    facade.export_model = SimpleNamespace(tracked_spool_targets=lambda: ("target",))
    facade.export_service = SimpleNamespace(
        flush_spool_targets=lambda targets, **kwargs: (
            calls.append((targets, kwargs)) or [("excel", "failed")]
        )
    )
    facade.configuration_model.analysis_config = {"display_sequence": []}

    signature = inspect.signature(SequenceWindow.flush_excel_spool_build)
    assert tuple(signature.parameters) == ("self", "on_close")
    assert signature.parameters["on_close"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["on_close"].default is False
    assert facade.flush_excel_spool_build(on_close=True) == [("excel", "failed")]
    assert calls[0][0] == ("target",)
    assert calls[0][1]["on_close"] is True


def test_sequence_window_remains_qt_native_at_original_import_path(facade):
    assert SequenceWindow.__module__ == "ui.sequence.sequence_widget"
    assert isinstance(facade, QWidget)
    for method in ("show", "close", "setMouseTracking", "isVisible"):
        assert method not in SequenceWindow.__dict__
        assert callable(getattr(facade, method))


def test_production_composition_owns_exact_resource_lifecycle_registrations(
    facade,
):
    registrations = (
        facade.resource_lifecycle_model.resource_lifecycle_registrations
    )
    assert {
        (registration.operation, registration.name)
        for registration in registrations
    } == {
        ("disconnect-domains", "trigger"),
        ("disconnect-domains", "analysis-transport"),
        ("disconnect-domains", "analysis"),
        ("disconnect-domains", "workflow"),
        ("disconnect-domains", "recording"),
        ("disconnect-domains", "export"),
    }
    assert len({id(item.token) for item in registrations}) == 6
    assert all(item.owner() is not None for item in registrations)
    assert all(item.registration_generation > 0 for item in registrations)


def test_legacy_product_entries_are_thin_command_or_controller_adapters():
    tree = ast.parse(SEQUENCE_WIDGET.read_text(encoding="utf-8"))
    sequence_window = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    methods = {
        node.name: node
        for node in sequence_window.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    migrated_entries = {
        "import_audio_and_analyze",
        "reset_work_pram",
        "_start_streaming_recording",
        "_start_blocking_recording",
        "run",
        "_maybe_export_excel_results",
    }
    assert migrated_entries <= methods.keys()
    for name in migrated_entries:
        node = methods[name]
        assert not any(isinstance(item, (ast.For, ast.While, ast.Try)) for item in ast.walk(node)), name
        assert len(node.body) <= 3, name


def test_static_sequence_architecture_contract():
    widget_source = SEQUENCE_WIDGET.read_text(encoding="utf-8")
    main_source = MAIN_WINDOW.read_text(encoding="utf-8")
    widget_tree = ast.parse(widget_source)
    main_tree = ast.parse(main_source)

    sequence_window = next(
        node for node in widget_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    assert sequence_window is not None
    for prohibited in (
        "streaming_poll_timer",
        "_poll_streaming_queue",
        "_active_instance_ref",
    ):
        assert prohibited not in widget_source
        assert prohibited not in main_source

    for path, tree in ((SEQUENCE_WIDGET, widget_tree), (MAIN_WINDOW, main_tree)):
        assert not any(
            isinstance(node, ast.While)
            and isinstance(node.test, ast.Constant)
            and node.test.value is True
            for node in ast.walk(tree)
        ), path.name
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "processEvents"
            for node in ast.walk(tree)
        ), path.name

    controller_paths = sorted((ROOT / "ui" / "sequence").glob("sequence_*_controller.py"))
    for path in controller_paths:
        own_domain = path.stem.removeprefix("sequence_").removesuffix("_controller")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if not (
                node.module.startswith("ui.sequence.sequence_")
                and node.module.endswith("_controller")
            ):
                continue
            imported_domain = node.module.rsplit(".", 1)[-1].removeprefix(
                "sequence_"
            ).removesuffix("_controller")
            assert imported_domain == own_domain, (path.name, node.module)


def test_facade_delegates_reusable_and_shutdown_resource_ownership():
    assert RESOURCE_LIFECYCLE.exists()
    widget_tree = ast.parse(SEQUENCE_WIDGET.read_text(encoding="utf-8"))
    lifecycle_tree = ast.parse(RESOURCE_LIFECYCLE.read_text(encoding="utf-8"))
    sequence_window = next(
        node
        for node in widget_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    lifecycle_owner = next(
        node
        for node in lifecycle_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "SequenceResourceLifecycleController"
    )
    facade_methods = {
        node.name
        for node in sequence_window.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    owner_methods = {
        node.name
        for node in lifecycle_owner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    migrated = {
        "_set_reusable_resource_identity",
        "_suspend_reusable_child_resources",
        "_resume_reusable_child_resources",
        "_disconnect_trigger_inputs",
        "_prepare_application_shutdown_resources",
        "_complete_application_shutdown_delivery",
    }
    assert facade_methods.isdisjoint(migrated)
    assert migrated <= owner_methods
    facade_assignments = {
        target.attr
        for node in ast.walk(sequence_window)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else (node.target,)
        )
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }
    assert not any(
        name.startswith("_reusable_")
        or name.startswith("_shutdown_cleanup")
        or name.startswith("_shutdown_prepared")
        for name in facade_assignments
    )


def test_resource_lifecycle_owner_has_no_cross_controller_escape_hatch():
    source_text = RESOURCE_LIFECYCLE.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    lifecycle_owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "SequenceResourceLifecycleController"
    )
    methods = {
        node.name
        for node in lifecycle_owner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "__getattr__" not in methods

    forbidden_names = {
        "trigger_controller",
        "workflow_controller",
        "analysis_controller",
        "analysis_transport_controller",
        "recording_controller",
        "legacy_recording_bridge",
        "export_controller",
        "shutdown_coordinator",
    }
    accessed = {
        node.attr
        for node in ast.walk(lifecycle_owner)
        if isinstance(node, ast.Attribute)
    }
    string_lookups = {
        node.value
        for node in ast.walk(lifecycle_owner)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert forbidden_names.isdisjoint(accessed | string_lookups)

    owner_source = ast.get_source_segment(source_text, lifecycle_owner)
    assert "trigger.disconnect" not in owner_source
    assert "coordinator.disconnect" not in owner_source
    assert not any(isinstance(node, ast.While) for node in ast.walk(lifecycle_owner))


def test_facade_composition_registers_all_required_lifecycle_recipients_directly():
    source_text = SEQUENCE_WIDGET.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    sequence_window = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    registration = next(
        node
        for node in sequence_window.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_register_resource_lifecycle_recipients"
    )
    registration_source = ast.get_source_segment(source_text, registration)
    for required in (
        "trigger_controller",
        "analysis_transport_controller",
        "analysis_controller",
        "workflow_controller",
        "recording_controller",
        "export_controller",
    ):
        assert required in registration_source
    assert "disconnect-domains" in registration_source
    assert "disconnect-coordinator" not in registration_source
    assert not any(isinstance(node, ast.Lambda) for node in ast.walk(registration))


def test_facade_retains_no_migrated_recording_or_export_fallback_algorithms():
    tree = ast.parse(SEQUENCE_WIDGET.read_text(encoding="utf-8"))
    sequence_window = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    methods = {
        node.name
        for node in sequence_window.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    summary_delegate = next(
        node
        for node in sequence_window.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_maybe_show_analysis_result_summary"
    )

    assert methods.isdisjoint(
        {
            "_can_output_ok_ng",
            "on_barcode_received",
            "_clear_barcode_input_safely",
            "_reset_barcode_dedup_state",
            "swap_tcp_status",
            "_schedule_excel_spool_build",
            "_on_excel_spool_build_timeout",
            "_show_channel_mismatch_warning",
            "_load_analysis_window_geometry",
            "_flush_analysis_window_geometry",
            "_normalize_geometry",
            "_is_geometry_on_any_screen",
            "del_geometry_config",
            "_set_sequence_config_available_state",
            "on_audio_chunk_received",
            "_start_transitional_streaming_recording",
            "_unwire_workflow_continuation_ports",
            "_finish_recording_success",
            "_finish_recording_failure",
            "_on_streaming_complete",
            "_cleanup_streaming_resources",
            "_should_use_streaming_recording",
            "_normalize_blocking_recorded_data",
            "_build_current_wav_calibration_metadata",
            "_current_metadata_input_channels",
            "_validate_mes_summary_input",
            "_maybe_write_mes_result",
            "_capture_excel_export_cache",
            "_handle_legacy_analysis_export_requested",
        }
    )
    assert not any(
        isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
        for decorator in summary_delegate.decorator_list
    )


def test_facade_has_no_duplicate_model_state_or_bound_controller_injection():
    source_text = SEQUENCE_WIDGET.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    sequence_window = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    methods = {
        node.name: node
        for node in sequence_window.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "_ensure_configuration_projection_hooks" not in methods
    assert "_ensure_recording_model" not in methods
    assert "_legacy_" not in source_text
    assert "_configuration_data_struct" not in source_text
    assert "_clear_data_struct_stimulus_runtime_state" not in source_text
    assert "_clear_sequence_stimulus_runtime_state" not in source_text
    assert "_excel_spool_build_timer" not in source_text
    assert "_excel_spool_build_delay_ms" not in source_text
    assert "legacy_recording_bridge" not in source_text
    assert "clicked_player_flag" not in source_text
    assert "self.ip_format" not in source_text
    assert "self.port_format" not in source_text
    assert "_is_positive_runtime_integer" not in methods
    assert "_has_runtime_samples" not in methods

    for node in ast.walk(sequence_window):
        assert not (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("_")
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "self"
            and (
                node.value.attr.endswith("_controller")
                or node.value.attr.endswith("_service")
            )
        ), ast.unparse(node)
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        for target in targets:
            assert not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "controller"
                and target.attr.startswith("_")
            )

    assert len(
        [node for node in sequence_window.body if isinstance(node, ast.FunctionDef) and node.name == "showEvent"]
    ) == 1
    assert len(
        [node for node in sequence_window.body if isinstance(node, ast.FunctionDef) and node.name == "closeEvent"]
    ) == 1

    for name, operation in (
        ("_get_tcp_mirror_identity", "read_tcp_mirror_identity"),
        ("_set_tcp_mirror_identity", "write_tcp_mirror_identity"),
    ):
        delegate = methods[name]
        assert len(delegate.body) == 1
        assert operation in ast.get_source_segment(source_text, delegate)
        assert "resource_lifecycle_controller" in ast.get_source_segment(
            source_text, delegate
        )
        assert "_CANONICAL_TCP_MIRROR_STATE" not in ast.get_source_segment(
            source_text, delegate
        )

    for name in (
        "init_result_files",
        "reset_test_reord",
        "on_reset_statistics_clicked",
    ):
        delegate = methods[name]
        assert len(delegate.body) == 1
        assert not any(
            isinstance(item, (ast.If, ast.Try, ast.For, ast.While))
            for item in ast.walk(delegate)
        )

    snapshot_lookup = methods["_workflow_analysis_snapshot_lookup"]
    assert len(snapshot_lookup.body) == 1
    assert "recording_model.retained_analysis_inputs" in ast.get_source_segment(
        source_text, snapshot_lookup
    )
    assert not any(
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == "deepcopy"
        for item in ast.walk(snapshot_lookup)
    )

    broad_catch_methods = {
        name
        for name, method in methods.items()
        if any(
            isinstance(item, ast.ExceptHandler)
            and isinstance(item.type, ast.Name)
            and item.type.id == "Exception"
            for item in ast.walk(method)
        )
    }
    assert broad_catch_methods <= {"eventFilter"}

    awaiting_label = methods["_awaiting_ok_ng"]
    awaiting_source = ast.get_source_segment(source_text, awaiting_label)
    assert "workflow_model.awaiting_label" in awaiting_source
    assert "workflow_view" not in awaiting_source
    assert not any(
        isinstance(decorator, ast.Attribute) and decorator.attr == "setter"
        for decorator in awaiting_label.decorator_list
    )

    manual_label = methods["clicked_ok_or_ng"]
    assert len(manual_label.body) == 1
    assert "recording_controller.request_manual_label" in ast.get_source_segment(
        source_text, manual_label
    )

    initial_splitter = methods["_apply_initial_waveform_splitter_sizes"]
    assert not any(
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and isinstance(item.func.value, ast.Name)
        and item.func.value.id == "QTimer"
        and item.func.attr == "singleShot"
        for item in ast.walk(initial_splitter)
    )


def test_protected_configuration_surfaces_are_direct_model_projections():
    source_text = SEQUENCE_WIDGET.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    sequence_window = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    methods = [
        node
        for node in sequence_window.body
        if isinstance(node, ast.FunctionDef)
    ]

    protected = {
        "mic",
        "speaker",
        "mic_channels",
        "sequence_config",
        "analysis_config",
        "using_config_path",
        "streaming_stimulus_data",
    }
    for name in protected:
        accessors = [node for node in methods if node.name == name]
        assert len(accessors) == 2, name
        assert all(
            not any(isinstance(item, (ast.If, ast.Try)) for item in ast.walk(node))
            for node in accessors
        ), name

    data_struct_accessors = [node for node in methods if node.name == "data_struct"]
    assert len(data_struct_accessors) == 1
    assert not any(
        isinstance(item, (ast.If, ast.Try))
        for item in ast.walk(data_struct_accessors[0])
    )

    for name in (
        "on_sequence_config_updated",
        "init_data_struct_stimulus_config",
        "set_audio_devices_available",
    ):
        delegate = next(node for node in methods if node.name == name)
        assert not any(
            isinstance(item, (ast.If, ast.Try, ast.For, ast.While))
            for item in ast.walk(delegate)
        ), name
        assert len(delegate.body) <= 2, name


def test_sequence_window_has_no_analysis_calibration_policy_or_dead_warning_helper():
    source = SEQUENCE_WIDGET.read_text(encoding="utf-8")
    tree = ast.parse(source)
    window = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWindow"
    )
    assigned_attributes = {
        target.attr
        for node in ast.walk(window)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            tuple(node.targets)
            if isinstance(node, ast.Assign)
            else (node.target,)
        )
        if isinstance(target, ast.Attribute)
    }
    top_level_functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    assert "analysis_types_requiring_v2pa" not in assigned_attributes
    assert "_show_uncalibrated_microphone_warning" not in top_level_functions
    assert "analysis_warning_preferences" not in source

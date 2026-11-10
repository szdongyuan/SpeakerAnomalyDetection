import builtins
import io
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

from base.workbench_configuration_loader import WorkbenchConfigurationError

import pytest

from base.motor_workbench_config import MotorWorkbenchConfigRepository


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def condition(condition_id, queue, station_ids):
    return {
        "condition_id": condition_id,
        "condition_name": f"工况 {condition_id}",
        "test_queue": queue,
        "applicable_station_ids": station_ids,
        "trigger_state": "",
        "segmented_analysis": {"mode": "time", "settings": {"intervals": [1, 2]}},
    }


def acquisition():
    return {"detail": {"sample_rate": 44100, "total_time": 2.5, "streaming": True}}


@pytest.fixture
def config_files(tmp_path):
    programs = tmp_path / "products" / "program_registry.json"
    queues = tmp_path / "queues" / "sequence_config_registry.json"
    product = programs.parent / "motor.json"
    shared = queues.parent / "nested" / "shared.json"
    empty = tmp_path / "absolute" / "empty.json"
    layout = tmp_path / "layout.json"
    unused = queues.parent / "unused-queue.json"
    write_json(layout, {
        "rows": 1,
        "columns": 4,
        "cells": [[
            {"title": "响度一", "analysis_item": "B", "analysis_type": "SPL"},
            {"title": "频谱二", "analysis_item": "A", "analysis_type": "FFT"},
            {"title": "响度三", "analysis_item": "B", "analysis_type": "SPL"},
            {"title": "频谱四", "analysis_item": "A", "analysis_type": "FFT"},
        ]],
    })
    write_json(programs, {
        "active_file": product.name,
        "configs": [{"file": product.name, "project_name": "电机"}],
    })
    write_json(queues, {
        "using_config_path": "unused.json",
        "Shared": "nested/shared.json",
        "Empty": str(empty),
        "Unused": unused.name,
    })
    write_json(product, {
        "project_id": "motor",
        "project_name": "电机",
        "result_root_directory": str(tmp_path / "unused-results"),
        "stations": [
            {
                "station_id": "s3",
                "station_name": "工位三",
                "channels": [{
                    "channel_id": "ch",
                    "channel_name": "声音",
                    "signal_type": "audio",
                    "hardware_channel": {
                        "device_ref": "unavailable-device",
                        "physical_channel_index": 0,
                    },
                }],
            },
            {"station_id": "s1", "station_name": "工位一", "channels": []},
            {"station_id": "s2", "station_name": "工位二", "channels": []},
        ],
        "conditions": [
            condition("c2", "Shared", ["s3", "s1"]),
            condition("c1", "Shared", ["s1"]),
            condition("c3", "Empty", []),
        ],
    })
    write_json(shared, [{
        "seq1": {
            "acq": {**acquisition(), "mode": " record_only "},
            "analysis_list": {
                "display_sequence": ["B", "A"],
                "auto_analysis": True,
                "A": {},
                "B": {"type": "SPL", "analysis_channels": [2, 0], "limits": {"values": [3, 5]}},
            },
            "analysis_channel_basis": "wav",
        },
    }, {"seq2": {"custom": {"values": [4, 6]}}}])
    write_json(empty, [{"seq1": {"acq": acquisition(), "analysis_list": {"display_sequence": []}}}])
    write_json(unused, [{"seq1": {"analysis_list": {"display_sequence": []}}}])
    return {
        "programs": programs,
        "queues": queues,
        "product": product,
        "shared": shared,
        "empty": empty,
        "layout": layout,
        "unused": unused,
    }


@pytest.fixture
def repository(config_files):
    return MotorWorkbenchConfigRepository(
        config_files["programs"], config_files["queues"], plot_layout_path=config_files["layout"]
    )


def test_load_active_preserves_order_and_parameters(repository, config_files):
    config = repository.load_active()

    assert config.recording.context.project_id == "motor"
    assert config.recording.context.project_name == "电机"
    assert Path(config.recording.context.result_root_directory) == config_files["product"].parent.parent / "unused-results"
    assert not Path(config.recording.context.result_root_directory).exists()
    assert tuple(station.station_id for station in config.recording.context.stations) == ("s3", "s1", "s2")
    assert tuple(station.station_name for station in config.recording.context.stations) == ("工位三", "工位一", "工位二")
    channel, = config.recording.context.stations[0].channels
    assert (channel.channel_id, channel.channel_name, channel.signal_type) == ("ch", "声音", "audio")
    assert channel.hardware_channel.device_ref == "unavailable-device"
    assert channel.hardware_channel.physical_channel_index == 0
    assert config.recording.context.stations[1].channels == ()
    assert tuple(item.condition_id for item in config.recording.context.conditions) == ("c2", "c1", "c3")
    assert tuple(item.condition_name for item in config.recording.context.conditions) == ("工况 c2", "工况 c1", "工况 c3")
    assert config.recording.context.conditions[0].applicable_station_ids == ("s3", "s1")
    assert config.recording.context.conditions[0].test_queue == "Shared"
    assert config.recording.context.conditions[0].trigger_state == ""
    assert config.recording.context.conditions[0].segmented_analysis["mode"] == "time"
    assert config.recording.context.conditions[0].segmented_analysis["settings"]["intervals"] == (1, 2)
    assert tuple(config.analysis.test_queues[item.test_queue].analysis_config["display_sequence"] for item in config.recording.context.conditions) == (("B", "A"), ("B", "A"), ())
    assert repository.load_active() == config

    with pytest.raises(FrozenInstanceError):
        config.recording.context.project_name = "changed"
    with pytest.raises(TypeError):
        config.recording.context.conditions[0].segmented_analysis["settings"]["intervals"] = (3,)


def test_load_active_preserves_layout_and_exports_independent_process_configs(repository, config_files):
    config = repository.load_active()
    layout = config.ui.layout
    assert (layout.rows, layout.columns) == (1, 4)
    assert isinstance(layout.cells, tuple)
    assert isinstance(layout.cells[0], tuple)
    cell = layout.cells[0][1]
    assert (cell.title, cell.analysis_item, cell.analysis_type) == ("频谱二", "A", "FFT")

    first, second, empty = config.recording.context.conditions
    assert first.test_queue == second.test_queue
    assert config.recording.context is config.analysis.context
    assert config.recording.parameters_for(first.condition_id) is config.recording.parameters_for(second.condition_id)
    assert "sample_rate" not in config.recording.parameters_for(first.condition_id)
    assert config.analysis.test_queues[first.test_queue].analysis_config["B"]["analysis_channels"] == (2, 0)
    assert config.recording.context.get_condition("c2") is first
    assert config.recording.context.get_station("s3").channels is config.recording.context.stations[0].channels
    with pytest.raises(TypeError):
        config.recording.parameters_for(first.condition_id)["total_time"] = 48000
    raw_sequence = json.loads(config_files["shared"].read_text(encoding="utf-8"))
    raw_seq1 = raw_sequence[0]["seq1"]
    exported = config.analysis.describe()
    assert exported["test_queues"][first.test_queue]["analysis_config"] == raw_seq1["analysis_list"]
    exported["test_queues"][first.test_queue]["analysis_config"]["B"]["analysis_channels"].append(99)
    exported["conditions"][0]["segmented_analysis"]["settings"]["intervals"].clear()
    assert config.analysis.describe()["test_queues"][first.test_queue]["analysis_config"]["B"]["analysis_channels"] == [2, 0]
    assert first.segmented_analysis["settings"]["intervals"] == (1, 2)
    empty_export = config.analysis.describe()["test_queues"][empty.test_queue]
    assert empty_export["analysis_config"]["display_sequence"] == []
    assert empty_export["analysis_channel_basis"] is None
    assert set(exported) == {"stations", "conditions", "test_queues", "tasks"}


def test_default_layout_is_loaded_from_the_application_root(config_files, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repository = MotorWorkbenchConfigRepository(config_files["programs"], config_files["queues"])

    layout = repository.load_active().ui.layout

    assert (layout.rows, layout.columns) == (2, 4)
    assert tuple((cell.title, cell.analysis_item, cell.analysis_type) for cell in layout.cells[0]) == (
        ("声压级（SPL）1", "声压级", "SPL"),
    ) * 4
    assert tuple((cell.title, cell.analysis_item, cell.analysis_type) for cell in layout.cells[1]) == (
        ("快速傅里叶变换 (FFT) 1", "快速傅里叶变换 (FFT) 1", "FFT"),
    ) * 4


def test_same_queue_is_read_once_per_load_and_load_is_read_only(
    repository, config_files, monkeypatch,
):
    reads = []
    before = {path: path.read_bytes() for path in config_files.values()}

    def watch_open(original_open):
        def read_only_open(file, mode="r", *args, **kwargs):
            assert not any(flag in mode for flag in "wax+")
            reads.append(Path(file))
            return original_open(file, mode, *args, **kwargs)
        return read_only_open

    with monkeypatch.context() as patch:
        patch.setattr(io, "open", watch_open(io.open))
        patch.setattr(builtins, "open", watch_open(builtins.open))
        repository.load_active()
        assert reads.count(config_files["programs"]) == 1
        assert reads.count(config_files["shared"]) == 1
        assert reads.count(config_files["layout"]) == 1
        assert config_files["unused"] not in reads

    assert before == {path: path.read_bytes() for path in config_files.values()}
    write_json(config_files["shared"], [{
        "seq1": {"acq": acquisition(), "analysis_list": {"display_sequence": ["C"], "C": {}}},
    }])
    assert repository.load_active().analysis.test_queues["Shared"].analysis_config["display_sequence"] == ("C",)


@pytest.mark.parametrize("saved_path", ["C:shared.json", r"\shared.json"])
def test_partially_qualified_windows_queue_paths_are_rejected_before_reading(
    repository, config_files, monkeypatch, saved_path,
):
    registry = config_files["queues"]
    write_json(registry, {"Shared": saved_path, "Empty": str(config_files["empty"])})
    allowed_reads = {config_files[key] for key in ("programs", "product", "queues", "layout")}
    original_open = Path.open

    def guard_open(path, *args, **kwargs):
        assert path in allowed_reads, f"attempted to read queue file {path}"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard_open)
    with pytest.raises(WorkbenchConfigurationError) as caught:
        repository.load_active()
    assert str(registry) in str(caught.value)
    assert "Shared" in str(caught.value)


def test_no_active_product_does_not_read_layout_or_queue_registry(repository, config_files):
    write_json(config_files["programs"], {"active_file": None, "configs": []})
    config_files["queues"].unlink()
    config_files["layout"].unlink()

    assert repository.load_active() is None


def test_active_product_accepts_empty_conditions(repository, config_files):
    product = json.loads(config_files["product"].read_text(encoding="utf-8"))
    product["conditions"] = []
    write_json(config_files["product"], product)

    assert repository.load_active().recording.context.conditions == ()


@pytest.mark.parametrize(("file_key", "breakage", "cause"), [
    ("programs", "json", json.JSONDecodeError),
    ("programs", "oversized integer", ValueError),
    ("programs", "decoder nesting", RecursionError),
    ("queues", "encoding", UnicodeDecodeError),
    ("product", "missing", FileNotFoundError),
    ("layout", "json", json.JSONDecodeError),
    ("layout", "missing", FileNotFoundError),
    ("shared", "missing", FileNotFoundError),
])
def test_file_errors_have_path_and_cause(repository, config_files, file_key, breakage, cause):
    path = config_files[file_key]
    if breakage == "json":
        path.write_text("{", encoding="utf-8")
    elif breakage == "oversized integer":
        limit = sys.get_int_max_str_digits()
        if not limit:
            pytest.skip("Python integer string conversion limit is disabled")
        path.write_text("1" * (limit + 1), encoding="utf-8")
    elif breakage == "decoder nesting":
        # CPython 3.12 uses a separate recursion limit for the C decoder.
        depth = max(10_000, sys.getrecursionlimit() + 1)
        path.write_text("[" * depth + "0" + "]" * depth, encoding="utf-8")
    elif breakage == "encoding":
        path.write_bytes(b"\xff")
    else:
        path.unlink()

    with pytest.raises(WorkbenchConfigurationError) as caught:
        repository.load_active()
    assert str(path) in str(caught.value)
    assert isinstance(caught.value.__cause__, cause)
    if file_key == "shared":
        assert "Shared" in str(caught.value)


def test_deep_segmented_analysis_reports_its_field_and_cause(repository, config_files):
    path = config_files["product"]
    product = json.loads(path.read_text(encoding="utf-8"))
    product["conditions"][0]["segmented_analysis"]["settings"] = "nested_placeholder"
    depth = sys.getrecursionlimit() * 3 // 4
    nested = "[" * depth + "0" + "]" * depth
    payload = json.dumps(product).replace('"nested_placeholder"', nested)
    json.loads(payload)  # The decoder can read this depth.
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(WorkbenchConfigurationError) as caught:
        repository.load_active()

    assert str(path) in str(caught.value)
    assert "conditions[0].segmented_analysis" in str(caught.value)
    assert isinstance(caught.value.__cause__, RecursionError)


def test_deepest_loadable_condition_can_be_exported_and_compared(repository, config_files):
    path = config_files["product"]
    product = json.loads(path.read_text(encoding="utf-8"))
    product["conditions"][0]["segmented_analysis"]["settings"] = "nested_placeholder"
    template = json.dumps(product)
    # The highest loadable depth depends on the Python version and the call stack.
    for depth in range(sys.getrecursionlimit(), 0, -1):
        nested = '{"child":' * depth + "42" + "}" * depth
        path.write_text(template.replace('"nested_placeholder"', nested), encoding="utf-8")
        try:
            config = repository.load_active()
        except WorkbenchConfigurationError as error:
            assert isinstance(error.__cause__, RecursionError)
        else:
            break
    else:
        pytest.fail("No depth could be loaded")

    exported = config.analysis.describe()
    settings = exported["conditions"][0]["segmented_analysis"]["settings"]
    for _ in range(depth):
        assert type(settings) is dict
        settings = settings["child"]
    assert settings == 42
    assert repository.load_active() == config


def test_snapshot_equality_compares_channel_types_but_ignores_object_key_order(repository, config_files):
    path = config_files["shared"]
    sequence = json.loads(path.read_text(encoding="utf-8"))
    metric = sequence[0]["seq1"]["analysis_list"]["B"]
    metric["analysis_channels"] = [1, 0]
    write_json(path, sequence)
    original = repository.load_active()

    path.write_text(json.dumps(sequence, sort_keys=True), encoding="utf-8")
    assert repository.load_active() == original

    metric["analysis_channels"][0] = True
    write_json(path, sequence)
    updated = repository.load_active()

    assert updated != original
    assert updated.analysis.describe()["test_queues"]["Shared"]["analysis_config"]["B"]["analysis_channels"][0] is True


@pytest.mark.parametrize(("file_key", "keys", "replacement", "diagnostic"), [
    ("programs", (), [], "registry"),
    ("programs", ("active_file",), "other.json", "active_file"),
    ("programs", ("configs",), [{"file": "../motor.json", "project_name": "电机"}], "configs"),
    ("programs", ("configs",), [{"file": "bad\u0000.json", "project_name": "电机"}], "configs"),
    ("programs", ("configs",), [{"file": "motor.json", "project_name": ""}], "project_name"),
    ("programs", ("configs",), [
        {"file": "motor.json", "project_name": "电机"},
        {"file": "motor.json", "project_name": "另一电机"},
    ], "configs"),
    ("product", ("project_id",), "", "project_id"),
    ("product", ("stations",), [], "stations"),
    ("product", ("stations",), [
        {"station_id": f"s{i}", "station_name": str(i), "channels": []} for i in range(5)
    ], "stations"),
    ("product", ("stations", 1, "station_id"), "s3", "station_id"),
    ("product", ("stations", 0, "channels"), [
        {"channel_id": "ch", "channel_name": "A", "signal_type": "audio",
         "hardware_channel": {"device_ref": "d", "physical_channel_index": 0}},
        {"channel_id": "ch", "channel_name": "B", "signal_type": "audio",
         "hardware_channel": {"device_ref": "d", "physical_channel_index": 1}},
    ], "channel_id"),
    ("product", ("stations", 0, "channels", 0, "hardware_channel", "physical_channel_index"),
     True, "physical_channel_index"),
    ("product", ("stations", 0, "channels", 0, "hardware_channel", "physical_channel_index"),
     -1, "physical_channel_index"),
    ("product", ("conditions", 1, "condition_id"), "c2", "condition_id"),
    ("product", ("conditions", 0, "applicable_station_ids"), ["s3", "s3"], "applicable_station_ids"),
    ("product", ("conditions", 0, "applicable_station_ids"), ["unknown"], "unknown"),
    ("product", ("conditions", 0, "test_queue"), "unknown", "unknown"),
    ("product", ("conditions", 0, "test_queue"), "using_config_path", "using_config_path"),
    ("product", ("conditions", 0, "trigger_state"), None, "trigger_state"),
    ("product", ("conditions", 0, "segmented_analysis"), {}, "mode"),
    ("queues", ("Shared",), {}, "Shared"),
    ("queues", ("Shared",), "shared\u0000.json", "Shared"),
    ("layout", ("rows",), True, "rows"),
    ("layout", ("rows",), 0, "rows"),
    ("layout", ("rows",), 4, "rows"),
    ("layout", ("columns",), 4.0, "columns"),
    ("layout", ("columns",), 5, "columns"),
    ("layout", ("rows",), 2, "cells"),
    ("layout", ("cells", 0), [], "cells[0]"),
    ("layout", ("cells", 0, 0, "analysis_type"), "STFT", "cells[0][0].analysis_type"),
    ("layout", ("cells", 0, 0, "title"), " ", "cells[0][0].title"),
    ("layout", ("cells", 0, 0, "analysis_item"), "", "cells[0][0].analysis_item"),
    ("shared", (), [], "seq1"),
    ("shared", (0, "seq1"), [], "seq1"),
    ("shared", (0, "seq1"), {"analysis_list": {"display_sequence": []}}, "seq1.acq"),
    ("shared", (0, "seq1", "acq"), {}, "seq1.acq.detail"),
    ("shared", (0, "seq1", "acq", "mode"), "PLAY_ONLY", "seq1.acq.mode"),
    ("shared", (0, "seq1", "acq", "mode"), " \t ", "seq1.acq.mode"),
    ("shared", (0, "seq1", "acq", "mode"), False, "seq1.acq.mode"),
    ("shared", (0, "seq1", "acq", "detail"), {"sample_rate": 100}, "acq.detail.total_time"),
    ("shared", (0, "seq1", "acq", "detail", "total_time"), 0, "acq.detail.total_time"),
    ("shared", (0, "seq1", "acq", "detail", "total_time"), True, "acq.detail.total_time"),
    ("shared", (0, "seq1", "acq", "detail"), {
        "total_time": 3, "sample_rate": 100, "recording_preview_time_mode": "bad",
    }, "acq.detail.recording_preview_time_mode"),
    ("shared", (0, "seq1", "analysis_list"), {}, "display_sequence"),
    ("shared", (0, "seq1", "analysis_list", "display_sequence"), ["A", "A"], "display_sequence"),
    ("shared", (0, "seq1", "analysis_list", "display_sequence"), [0], "display_sequence"),
    ("shared", (0, "seq1", "analysis_list", "display_sequence"), ["missing-metric"], "missing-metric"),
    ("shared", (0, "seq1", "analysis_list", "A"), [], "A"),
])
def test_invalid_schema_has_path_and_diagnostic(
    repository, config_files, file_key, keys, replacement, diagnostic,
):
    path = config_files[file_key]
    data = json.loads(path.read_text(encoding="utf-8"))
    if keys:
        target = data
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = replacement
    else:
        data = replacement
    write_json(path, data)

    with pytest.raises(WorkbenchConfigurationError) as caught:
        repository.load_active()
    assert str(path) in str(caught.value)
    assert diagnostic in str(caught.value)


def test_nonfinite_duration_is_rejected(repository, config_files):
    field = "total_time"
    path = config_files["shared"]
    queue = json.loads(path.read_text(encoding="utf-8"))
    queue[0]["seq1"]["acq"]["detail"][field] = "number_placeholder"
    path.write_text(json.dumps(queue).replace('"number_placeholder"', "1e999"), encoding="utf-8")

    with pytest.raises(WorkbenchConfigurationError) as caught:
        repository.load_active()

    assert str(path) in str(caught.value)
    assert f"acq.detail.{field}" in str(caught.value)


def test_station_count_must_fit_layout(repository, config_files):
    path = config_files["layout"]
    layout = json.loads(path.read_text(encoding="utf-8"))
    layout["columns"] = 2
    layout["cells"][0] = layout["cells"][0][:2]
    write_json(path, layout)

    with pytest.raises(WorkbenchConfigurationError) as caught:
        repository.load_active()

    assert str(path) in str(caught.value)
    assert str(config_files["product"]) in str(caught.value)
    assert "columns" in str(caught.value)


@pytest.mark.parametrize("sample_rate", [None, 96000, -1, True, "legacy"])
def test_saved_sample_rate_is_excluded_without_capture_validation(repository, config_files, sample_rate):
    path = config_files["shared"]
    queue = json.loads(path.read_text(encoding="utf-8"))
    detail = queue[0]["seq1"]["acq"]["detail"]
    if sample_rate is None:
        del detail["sample_rate"]
    else:
        detail["sample_rate"] = sample_rate
    write_json(path, queue)

    config = repository.load_active()
    parameters = config.recording.parameters_for("c2")
    assert dict(parameters) == {key: value for key, value in detail.items() if key in {"total_time", "startup_trim_ms", "audio_validation"}}


def test_source_identity_changes_only_after_successful_load(repository, config_files):
    assert repository.source_identity == ""
    repository.load_active()
    source = repository.source_identity
    assert Path(source) == config_files["product"]
    registry = json.loads(config_files["programs"].read_text(encoding="utf-8"))
    registry["configs"].append({"file": "missing.json", "project_name": "missing"})
    registry["active_file"] = "missing.json"
    write_json(config_files["programs"], registry)
    with pytest.raises(WorkbenchConfigurationError):
        repository.load_active()
    assert repository.source_identity == source
    registry["active_file"] = None
    write_json(config_files["programs"], registry)
    assert repository.load_active() is None
    assert repository.source_identity == ""

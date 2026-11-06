import builtins
import io
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from base.motor_workbench_config import (
    MotorWorkbenchConfigRepository,
    WorkbenchConfigurationError,
)


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


@pytest.fixture
def config_files(tmp_path):
    programs = tmp_path / "products" / "program_registry.json"
    queues = tmp_path / "queues" / "sequence_config_registry.json"
    product = programs.parent / "motor.json"
    shared = queues.parent / "nested" / "shared.json"
    empty = tmp_path / "absolute" / "empty.json"
    write_json(programs, {
        "active_file": product.name,
        "configs": [{"file": product.name, "project_name": "电机"}],
    })
    write_json(queues, {
        "using_config_path": "unused.json",
        "Shared": "nested/shared.json",
        "Empty": str(empty),
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
        "seq1": {"analysis_list": {"display_sequence": ["B", "A"], "A": {}, "B": {}}},
    }])
    write_json(empty, [{"seq1": {"analysis_list": {"display_sequence": []}}}])
    return {
        "programs": programs,
        "queues": queues,
        "product": product,
        "shared": shared,
        "empty": empty,
    }


@pytest.fixture
def repository(config_files):
    return MotorWorkbenchConfigRepository(config_files["programs"], config_files["queues"])


def test_load_active_preserves_order_and_parameters(repository, config_files):
    config = repository.load_active()

    assert config.project_id == "motor"
    assert config.project_name == "电机"
    assert Path(config.result_root_directory) == config_files["product"].parent.parent / "unused-results"
    assert not Path(config.result_root_directory).exists()
    assert tuple(station.station_id for station in config.stations) == ("s3", "s1", "s2")
    assert tuple(station.station_name for station in config.stations) == ("工位三", "工位一", "工位二")
    channel, = config.stations[0].channels
    assert (channel.channel_id, channel.channel_name, channel.signal_type) == ("ch", "声音", "audio")
    assert channel.hardware_channel.device_ref == "unavailable-device"
    assert channel.hardware_channel.physical_channel_index == 0
    assert config.stations[1].channels == ()
    assert tuple(item.condition_id for item in config.conditions) == ("c2", "c1", "c3")
    assert tuple(item.condition_name for item in config.conditions) == ("工况 c2", "工况 c1", "工况 c3")
    assert config.conditions[0].applicable_station_ids == ("s3", "s1")
    assert config.conditions[0].test_queue == "Shared"
    assert config.conditions[0].trigger_state == ""
    assert config.conditions[0].segmented_analysis["mode"] == "time"
    assert config.conditions[0].segmented_analysis["settings"]["intervals"] == (1, 2)
    assert tuple(item.metric_names for item in config.conditions) == (("B", "A"), ("B", "A"), ())
    assert repository.load_active() == config

    with pytest.raises(FrozenInstanceError):
        config.project_name = "changed"
    with pytest.raises(TypeError):
        config.conditions[0].segmented_analysis["settings"]["intervals"] = (3,)


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
        assert reads.count(config_files["shared"]) == 1

    assert before == {path: path.read_bytes() for path in config_files.values()}
    write_json(config_files["shared"], [{
        "seq1": {"analysis_list": {"display_sequence": ["C"], "C": {}}},
    }])
    assert repository.load_active().conditions[0].metric_names == ("C",)


@pytest.mark.parametrize("saved_path", ["C:shared.json", r"\shared.json"])
def test_partially_qualified_windows_queue_paths_are_rejected_before_reading(
    repository, config_files, monkeypatch, saved_path,
):
    registry = config_files["queues"]
    write_json(registry, {"Shared": saved_path, "Empty": str(config_files["empty"])})
    allowed_reads = {config_files[key] for key in ("programs", "product", "queues")}
    original_open = Path.open

    def guard_open(path, *args, **kwargs):
        assert path in allowed_reads, f"attempted to read queue file {path}"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard_open)
    with pytest.raises(WorkbenchConfigurationError) as caught:
        repository.load_active()
    assert str(registry) in str(caught.value)
    assert "Shared" in str(caught.value)


def test_no_active_product_does_not_read_queue_registry(repository, config_files):
    write_json(config_files["programs"], {"active_file": None, "configs": []})
    config_files["queues"].unlink()

    assert repository.load_active() is None


def test_active_product_accepts_empty_conditions(repository, config_files):
    product = json.loads(config_files["product"].read_text(encoding="utf-8"))
    product["conditions"] = []
    write_json(config_files["product"], product)

    assert repository.load_active().conditions == ()


@pytest.mark.parametrize(("file_key", "breakage", "cause"), [
    ("programs", "json", json.JSONDecodeError),
    ("programs", "oversized integer", ValueError),
    ("programs", "decoder nesting", RecursionError),
    ("queues", "encoding", UnicodeDecodeError),
    ("product", "missing", FileNotFoundError),
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
    ("shared", (), [], "seq1"),
    ("shared", (0, "seq1"), [], "seq1"),
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

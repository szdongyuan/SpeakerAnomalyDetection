"""Real-file tests for assembling separate workbench configurations."""

import json
import sys
from pathlib import Path

import pytest

from base.workbench_configuration_loader import (
    WorkbenchConfiguration,
    WorkbenchConfigurationError,
    load_workbench_configuration,
)
from unit_test.base.workbench_config_fixtures import write_json, write_workbench_files


def load(files):
    return load_workbench_configuration(files["programs"], files["queues"], files["layout"])


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_one_read_assembles_shared_context_and_independent_complete_test_queues(tmp_path, monkeypatch):
    files = write_workbench_files(tmp_path)
    product = read(files["product"])
    product["conditions"].append(dict(product["conditions"][0], condition_id="c3"))
    write_json(files["product"], product)
    before = {path: path.read_bytes() for path in files.values()}
    reads = []
    original_open = Path.open

    def read_only_open(path, mode="r", *args, **kwargs):
        assert not any(flag in mode for flag in "wax+")
        reads.append(path)
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", read_only_open)
        config = load(files)

    assert isinstance(config, WorkbenchConfiguration)
    assert set(vars(config)) == {"recording", "analysis", "ui"}
    assert config.recording.context is config.analysis.context
    assert all(reads.count(path) == 1 for path in files.values())
    assert before == {path: path.read_bytes() for path in files.values()}
    assert not Path(product["result_root_directory"]).exists()
    assert config.recording.parameters_for("c1") is config.recording.parameters_for("c3")
    assert config.ui.layout.rows == 2
    assert config.ui.layout.columns == 4
    exported = config.analysis.describe()
    assert exported["tasks"] == []
    for name in ("测试队列A", "测试队列B"):
        seq1 = read(files[name])[0]["seq1"]
        assert exported["test_queues"][name] == {
            "analysis_channel_basis": seq1["analysis_channel_basis"],
            "analysis_config": seq1["analysis_list"],
        }
        assert dict(config.recording.parameters_by_queue[name]) == {
            "total_time": seq1["acq"]["detail"]["total_time"],
            "startup_trim_ms": 0,
            "audio_validation": seq1["acq"]["detail"]["audio_validation"],
        }
        assert config.ui.preview_time_modes[name] == "cumulative"
    exported["test_queues"]["测试队列A"]["analysis_config"]["声压级"]["analysis_channels"].append(99)
    assert config.analysis.describe()["test_queues"]["测试队列A"]["analysis_config"]["声压级"]["analysis_channels"] == [0]
    with pytest.raises(TypeError):
        config.analysis.test_queues["测试队列A"].analysis_config["声压级"]["limits"]["values"][0][0] = 99


def test_no_active_product_returns_none_without_reading_other_files(tmp_path):
    files = write_workbench_files(tmp_path)
    write_json(files["programs"], {"active_file": None, "configs": []})
    for key in ("product", "queues", "layout"):
        files[key].unlink()
    assert load(files) is None


def test_mixed_device_references_in_one_station_are_rejected(tmp_path):
    files = write_workbench_files(tmp_path)
    product = read(files["product"])
    product["stations"][0]["channels"][1]["hardware_channel"]["device_ref"] = "other-device"
    write_json(files["product"], product)
    with pytest.raises(WorkbenchConfigurationError) as caught:
        load(files)
    assert str(files["product"]) in str(caught.value)
    assert "stations[0]" in str(caught.value)
    assert "device_ref" in str(caught.value)


def test_empty_channels_and_unassociated_conditions_remain_queryable(tmp_path):
    files = write_workbench_files(tmp_path)
    product = read(files["product"])
    product["stations"][0]["channels"] = []
    product["conditions"][1]["applicable_station_ids"] = []
    write_json(files["product"], product)
    config = load(files)
    assert config.recording.context.get_station("s1").channels == ()
    assert config.analysis.stations_for("c2") == ()
    assert config.analysis.stations_for("c1")[0] is config.recording.context.get_station("s1")
    assert [station["station_id"] for station in config.analysis.describe()["stations"]] == ["s2"]
    assert [condition["condition_id"] for condition in config.analysis.describe()["conditions"]] == ["c1", "c2"]


def test_equal_queue_contents_keep_distinct_test_queues_and_seq2_is_not_retained(tmp_path):
    files = write_workbench_files(tmp_path)
    sequence = read(files["测试队列A"])
    write_json(files["测试队列B"], sequence)
    config = load(files)
    assert tuple(config.analysis.test_queues) == ("测试队列A", "测试队列B")
    assert config.analysis.test_queues["测试队列A"] is not config.analysis.test_queues["测试队列B"]
    assert config.analysis.test_queues["测试队列A"] == config.analysis.test_queues["测试队列B"]
    sequence[1] = {"seq2": {"acq": {"mode": "PLAY_ONLY"}, "analysis_list": {"extra": [1, 2]}}}
    write_json(files["测试队列A"], sequence)
    assert load(files) == config
    assert config.analysis.describe()["tasks"] == []


def test_missing_optional_fields_preserve_defaults_at_their_existing_boundaries(tmp_path):
    files = write_workbench_files(tmp_path)
    sequence = read(files["测试队列A"])
    seq1 = sequence[0]["seq1"]
    del seq1["analysis_channel_basis"]
    seq1["acq"] = {"detail": {"total_time": 3}}
    write_json(files["测试队列A"], sequence)
    config = load(files)
    assert dict(config.recording.parameters_for("c1")) == {"total_time": 3}
    assert config.analysis.test_queues["测试队列A"].analysis_channel_basis is None
    assert config.analysis.describe()["test_queues"]["测试队列A"]["analysis_channel_basis"] is None
    assert config.ui.preview_time_modes["测试队列A"] == "relative_latest"


@pytest.mark.parametrize("basis", [{"mutable": [1]}, ["wav"], True, False, 0, 1, 1.5])
def test_analysis_channel_basis_rejects_non_string_values_with_queue_location(tmp_path, basis):
    files = write_workbench_files(tmp_path)
    path = files["测试队列A"]
    sequence = read(path)
    sequence[0]["seq1"]["analysis_channel_basis"] = basis
    write_json(path, sequence)

    with pytest.raises(WorkbenchConfigurationError) as caught:
        load(files)

    assert str(path) in str(caught.value)
    assert "[0].seq1.analysis_channel_basis" in str(caught.value)


@pytest.mark.parametrize("basis", [None, "wav", "physical", "", " custom basis "])
def test_analysis_channel_basis_preserves_null_and_arbitrary_strings(tmp_path, basis):
    files = write_workbench_files(tmp_path)
    path = files["测试队列A"]
    sequence = read(path)
    sequence[0]["seq1"]["analysis_channel_basis"] = basis
    write_json(path, sequence)

    config = load(files)

    assert config.analysis.test_queues["测试队列A"].analysis_channel_basis == basis
    assert config.analysis.describe()["test_queues"]["测试队列A"]["analysis_channel_basis"] == basis


def test_audio_thresholds_are_preserved_without_request_validation(tmp_path):
    files = write_workbench_files(tmp_path)
    sequence = read(files["测试队列A"])
    detail = sequence[0]["seq1"]["acq"]["detail"]
    detail["startup_trim_ms"] = "invalid until recording starts"
    detail["audio_validation"] = {"min_rms_dbfs": "invalid until recording starts"}
    detail["sample_rate"] = "ignored"
    write_json(files["测试队列A"], sequence)
    parameters = load(files).recording.parameters_for("c1")
    assert parameters["startup_trim_ms"] == detail["startup_trim_ms"]
    assert parameters["audio_validation"] == detail["audio_validation"]


def test_no_conditions_does_not_need_queue_registry(tmp_path):
    files = write_workbench_files(tmp_path)
    product = read(files["product"])
    product["conditions"] = []
    write_json(files["product"], product)
    files["queues"].unlink()
    config = load(files)
    assert config.recording.context.conditions == ()
    assert dict(config.recording.parameters_by_queue) == {}
    assert dict(config.ui.preview_time_modes) == {}
    assert config.analysis.describe() == {"stations": [], "conditions": [], "test_queues": {}, "tasks": []}


@pytest.mark.parametrize(("file_key", "keys", "value", "diagnostic"), [
    ("programs", (), [], "registry"),
    ("programs", ("active_file",), "unknown.json", "active_file"),
    ("programs", ("configs", 0, "file"), "../motor.json", "configs[0].file"),
    ("programs", ("configs", 0, "project_name"), "", "project_name"),
    ("product", ("project_id",), "", "project_id"),
    ("product", ("result_root_directory",), "", "result_root_directory"),
    ("product", ("stations",), [], "stations"),
    ("product", ("stations", 1, "station_id"), "s1", "station_id"),
    ("product", ("stations", 0, "channels", 1, "channel_id"), "ch1", "channel_id"),
    ("product", ("stations", 0, "channels", 0, "hardware_channel", "physical_channel_index"), True, "physical_channel_index"),
    ("product", ("conditions", 1, "condition_id"), "c1", "condition_id"),
    ("product", ("conditions", 0, "applicable_station_ids"), ["s1", "s1"], "applicable_station_ids"),
    ("product", ("conditions", 0, "applicable_station_ids"), ["unknown"], "unknown station"),
    ("product", ("conditions", 0, "test_queue"), "unknown", "unknown queue"),
    ("product", ("conditions", 0, "test_queue"), "using_config_path", "unknown queue"),
    ("product", ("conditions", 0, "trigger_state"), None, "trigger_state"),
    ("product", ("conditions", 0, "segmented_analysis"), {}, "segmented_analysis.mode"),
    ("queues", ("测试队列A",), {}, "测试队列A"),
    ("queues", ("测试队列A",), "bad\0.json", "测试队列A"),
    ("queues", ("测试队列A",), "C:queue.json", "测试队列A"),
    ("queues", ("测试队列A",), "\\queue.json", "测试队列A"),
    ("layout", ("rows",), True, "rows"),
    ("layout", ("rows",), 4, "rows"),
    ("layout", ("columns",), 0, "columns"),
    ("layout", ("columns",), 5, "columns"),
    ("layout", ("cells", 0), [], "cells[0]"),
    ("layout", ("cells", 0, 0, "analysis_type"), "STFT", "analysis_type"),
    ("layout", ("cells", 0, 0, "title"), "", "title"),
    ("测试队列A", (), [], "seq1"),
    ("测试队列A", (0, "seq1"), [], "seq1"),
    ("测试队列A", (0, "seq1", "acq", "mode"), "PLAY_ONLY", "acq.mode"),
    ("测试队列A", (0, "seq1", "acq", "mode"), False, "acq.mode"),
    ("测试队列A", (0, "seq1", "acq", "detail", "total_time"), True, "detail.total_time"),
    ("测试队列A", (0, "seq1", "acq", "detail", "total_time"), 0, "detail.total_time"),
    ("测试队列A", (0, "seq1", "acq", "detail", "total_time"), -1, "detail.total_time"),
    ("测试队列A", (0, "seq1", "acq", "detail", "recording_preview_time_mode"), "bad", "recording_preview_time_mode"),
    ("测试队列A", (0, "seq1", "analysis_list"), {}, "display_sequence"),
    ("测试队列A", (0, "seq1", "analysis_list", "display_sequence"), ["声压级", "声压级"], "display_sequence"),
    ("测试队列A", (0, "seq1", "analysis_list", "display_sequence"), ["missing"], "missing"),
    ("测试队列A", (0, "seq1", "analysis_list", "声压级"), [], "声压级"),
])
def test_invalid_inputs_have_file_and_field_diagnostics(tmp_path, file_key, keys, value, diagnostic):
    files = write_workbench_files(tmp_path)
    path = files[file_key]
    data = read(path)
    if keys:
        target = data
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    else:
        data = value
    write_json(path, data)
    with pytest.raises(WorkbenchConfigurationError) as caught:
        load(files)
    assert str(path) in str(caught.value)
    assert diagnostic in str(caught.value)


@pytest.mark.parametrize(("file_key", "breakage", "cause"), [
    ("programs", "json", json.JSONDecodeError),
    ("queues", "encoding", UnicodeDecodeError),
    ("product", "missing", FileNotFoundError),
    ("layout", "missing", FileNotFoundError),
    ("测试队列A", "missing", FileNotFoundError),
])
def test_file_errors_include_path_and_cause(tmp_path, file_key, breakage, cause):
    files = write_workbench_files(tmp_path)
    path = files[file_key]
    if breakage == "json":
        path.write_text("{", encoding="utf-8")
    elif breakage == "encoding":
        path.write_bytes(b"\xff")
    else:
        path.unlink()
    with pytest.raises(WorkbenchConfigurationError) as caught:
        load(files)
    assert str(path) in str(caught.value)
    assert isinstance(caught.value.__cause__, cause)


@pytest.mark.parametrize("number", ["1e999", "NaN", "Infinity"])
def test_nonfinite_duration_is_rejected(tmp_path, number):
    files = write_workbench_files(tmp_path)
    path = files["测试队列A"]
    path.write_text(path.read_text(encoding="utf-8").replace('"total_time": 2.5', f'"total_time": {number}'), encoding="utf-8")
    with pytest.raises(WorkbenchConfigurationError) as caught:
        load(files)
    assert str(path) in str(caught.value)


def test_station_count_must_fit_layout(tmp_path):
    files = write_workbench_files(tmp_path)
    layout = read(files["layout"])
    layout["columns"] = 2
    layout["cells"] = [row[:2] for row in layout["cells"]]
    write_json(files["layout"], layout)
    with pytest.raises(WorkbenchConfigurationError) as caught:
        load(files)
    assert str(files["product"]) in str(caught.value)
    assert str(files["layout"]) in str(caught.value)
    assert "columns" in str(caught.value)


@pytest.mark.parametrize(("file_key", "field"), [
    ("product", "conditions[0].segmented_analysis"),
    ("测试队列A", "[0].seq1.analysis_list"),
])
def test_deep_snapshots_have_input_location_and_recursion_cause(tmp_path, file_key, field):
    files = write_workbench_files(tmp_path)
    path = files[file_key]
    data = read(path)
    target = data["conditions"][0]["segmented_analysis"] if file_key == "product" else data[0]["seq1"]["analysis_list"]
    target["extra"] = "nested_placeholder"
    depth = sys.getrecursionlimit() * 3 // 4
    payload = json.dumps(data).replace('"nested_placeholder"', "[" * depth + "0" + "]" * depth)
    json.loads(payload)
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(WorkbenchConfigurationError) as caught:
        load(files)
    assert str(path) in str(caught.value)
    assert field in str(caught.value)
    assert isinstance(caught.value.__cause__, RecursionError)


def test_deep_recording_parameters_report_queue_file_and_detail_field(tmp_path):
    files = write_workbench_files(tmp_path)
    path = files["测试队列A"]
    sequence = read(path)
    sequence[0]["seq1"]["acq"]["detail"]["audio_validation"] = "nested_placeholder"
    depth = sys.getrecursionlimit() * 3 // 4
    payload = json.dumps(sequence).replace('"nested_placeholder"', "[" * depth + "0" + "]" * depth)
    json.loads(payload)
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(WorkbenchConfigurationError) as caught:
        load(files)

    assert str(path) in str(caught.value)
    assert "[0].seq1.acq.detail" in str(caught.value)
    assert str(files["queues"]) not in str(caught.value)
    assert isinstance(caught.value.__cause__, RecursionError)

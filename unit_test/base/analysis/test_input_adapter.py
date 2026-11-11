from copy import deepcopy
from pathlib import Path

import pytest

from base.analysis import AnalysisInputError, build_analysis_batch


@pytest.fixture
def referenced_payload(batch_payload):
    config = batch_payload["tasks"][0]["analysis_config"]
    batch_payload["test_queues"] = {"queue": {"analysis_channel_basis": "wav", "analysis_config": config}}
    for condition in batch_payload["conditions"]:
        condition["test_queue"] = "queue"
    for task in batch_payload["tasks"]:
        del task["analysis_config"]
    return batch_payload


def test_queue_references_expand_to_independent_wav_channel_tasks(referenced_payload):
    batch = build_analysis_batch(referenced_payload)
    assert len(batch.tasks) == 8
    assert len(batch.tasks[0].instances) == 5
    assert batch.tasks[1].instances[0].channel_index == 0
    assert batch.tasks[1].channels[0].physical_channel_index == 4
    referenced_payload["test_queues"]["queue"]["analysis_config"]["SPL 1"]["weighting"] = "Z"
    assert batch.tasks[0].instances[0].parameters["weighting"] == "A"


@pytest.mark.parametrize("basis", ["hardware", "device", "unknown", 0])
def test_queue_channel_basis_must_be_wav(referenced_payload, basis):
    referenced_payload["test_queues"]["queue"]["analysis_channel_basis"] = basis
    with pytest.raises(AnalysisInputError) as caught:
        build_analysis_batch(referenced_payload)
    assert caught.value.issue.field_path == 'test_queues["queue"].analysis_channel_basis'


def test_missing_referenced_queue_is_not_replaced_by_inline_config(referenced_payload):
    referenced_payload["tasks"][0]["analysis_config"] = {"display_sequence": []}
    referenced_payload["test_queues"] = {}
    with pytest.raises(AnalysisInputError) as caught:
        build_analysis_batch(referenced_payload)
    assert caught.value.issue.error_type == "UnknownReference"


def test_conflicting_inline_and_referenced_parameters_rejected(referenced_payload):
    referenced_payload["tasks"][0]["analysis_config"] = {"display_sequence": []}
    with pytest.raises(AnalysisInputError) as caught:
        build_analysis_batch(referenced_payload)
    assert caught.value.issue.error_type == "ConflictingAnalysisConfig"


def test_matching_inline_and_reference_allowed(referenced_payload):
    referenced_payload["tasks"][0]["analysis_config"] = deepcopy(
        referenced_payload["test_queues"]["queue"]["analysis_config"])
    assert len(build_analysis_batch(referenced_payload).tasks) == 8


def test_four_stations_two_conditions_expand_in_requested_order(batch_payload):
    batch = build_analysis_batch(batch_payload)
    assert len(batch.tasks) == 8
    assert sum(len(task.instances) for task in batch.tasks) == 40
    assert [(item.config_key, item.channel_index) for item in batch.tasks[0].instances] == [
        ("SPL 1", 0), ("FFT 1", 0), ("FFT 1", 1), ("FFT 1", 2), ("FFT 1", 3),
    ]
    station_two = batch.tasks[1]
    assert station_two.station_id == "station-2"
    assert station_two.condition_name == "工况1"
    assert station_two.channels[0].channel_index == 0
    assert station_two.channels[0].physical_channel_index == 4
    assert station_two.channels[0].device_ref == "demo-device"
    assert station_two.instances[0].channel_index == 0
    assert all(task.analysis_result_directory is None for task in batch.tasks)
    issues = [issue for issue in batch.preparation_issues if issue.task_id == station_two.task_id]
    assert {issue.error_type for issue in issues} == {
        "MissingOutputDirectory", "CalibrationNotProvided", "UnsupportedSignalType",
    }
    assert sum(issue.error_type == "UnsupportedSignalType" for issue in issues) == 3


def test_subset_and_different_condition_parameters(batch_payload):
    payload = deepcopy(batch_payload)
    payload["tasks"] = [payload["tasks"][1], payload["tasks"][5]]
    payload["stations"] = [payload["stations"][1]]
    payload["tasks"][1]["analysis_config"] = {
        "display_sequence": ["SPL Z"],
        "SPL Z": {"type": "SPL", "analysis_channel": 0, "weighting": "Z"},
    }
    tasks = build_analysis_batch(payload).tasks
    assert len(tasks) == 2
    assert [len(task.instances) for task in tasks] == [5, 1]
    assert [task.instances[0].parameters["weighting"] for task in tasks] == ["A", "Z"]


@pytest.mark.parametrize("collection,id_field", [
    ("stations", "station_id"), ("conditions", "condition_id"), ("tasks", "task_id"),
])
def test_duplicate_identity_rejected(batch_payload, collection, id_field):
    batch_payload[collection][1][id_field] = batch_payload[collection][0][id_field]
    with pytest.raises(AnalysisInputError) as caught:
        build_analysis_batch(batch_payload)
    assert caught.value.issue.error_type == "DuplicateId"
    assert caught.value.issue.field_path == f"{collection}[1].{id_field}"


@pytest.mark.parametrize("field", ["station_id", "condition_id"])
def test_unknown_reference_identifies_task_and_field(batch_payload, field):
    batch_payload["tasks"][2][field] = "missing"
    with pytest.raises(AnalysisInputError) as caught:
        build_analysis_batch(batch_payload)
    issue = caught.value.issue
    assert issue.task_id == "analysis-1-3"
    assert issue.field_path == f"tasks[2].{field}"
    assert issue.error_type == "UnknownReference"
    assert issue.message


@pytest.mark.parametrize("channels", [[], None, "0", [True], [-1], [4], [0, 0], [0.0], ["0"]])
def test_invalid_channel_selection_never_falls_back(batch_payload, channels):
    config = batch_payload["tasks"][0]["analysis_config"]["SPL 1"]
    config.update(analysis_channel=0, analysis_channels=channels)
    with pytest.raises(AnalysisInputError) as caught:
        build_analysis_batch(batch_payload)
    assert ".analysis_channels" in caught.value.issue.field_path


def test_plural_selection_is_authoritative_and_preserves_order(batch_payload):
    config = batch_payload["tasks"][0]["analysis_config"]
    config["FFT 1"].update(analysis_channels=[3, 0], analysis_channel=999)
    instances = build_analysis_batch(batch_payload).tasks[0].instances
    assert [item.channel_index for item in instances] == [0, 3, 0]


def test_legacy_single_channel_requires_an_explicit_selection(batch_payload):
    config = batch_payload["tasks"][0]["analysis_config"]["SPL 1"]
    del config["analysis_channels"]
    with pytest.raises(AnalysisInputError):
        build_analysis_batch(batch_payload)
    config["analysis_channel"] = 0
    assert build_analysis_batch(batch_payload).tasks[0].instances[0].channel_index == 0


@pytest.mark.parametrize("sequence", [["missing"], ["SPL 1", "SPL 1"], "SPL 1", None])
def test_invalid_analysis_sequence_is_not_silently_skipped(batch_payload, sequence):
    batch_payload["tasks"][0]["analysis_config"]["display_sequence"] = sequence
    with pytest.raises(AnalysisInputError):
        build_analysis_batch(batch_payload)


def test_profile_is_never_used_to_fill_missing_analysis_config(batch_payload):
    task = batch_payload["tasks"][0]
    config = task.pop("analysis_config")
    batch_payload["conditions"][0]["profile_id"] = "recording-profile"
    batch_payload["profiles"] = {"recording-profile": {"analysis_config": config}}
    with pytest.raises(AnalysisInputError) as caught:
        build_analysis_batch(batch_payload)
    assert caught.value.issue.field_path == "tasks[0].analysis_config"


@pytest.mark.parametrize("start,end", [(-1, 2), (2, 1), (2, 2), (False, 2), (0, float("inf"))])
def test_invalid_enabled_time_range(batch_payload, start, end):
    batch_payload["tasks"][0]["analysis_config"]["SPL 1"].update(
        analysis_time_range_enabled=True,
        analysis_start_time_sec=start, analysis_end_time_sec=end,
    )
    with pytest.raises(AnalysisInputError):
        build_analysis_batch(batch_payload)


def test_zero_end_means_wav_end_and_no_file_is_read(batch_payload, monkeypatch):
    config = batch_payload["tasks"][0]["analysis_config"]["SPL 1"]
    config.update(analysis_time_range_enabled=True,
                  analysis_start_time_sec=0.5, analysis_end_time_sec=0.0)

    def forbidden(*args, **kwargs):
        pytest.fail("input construction must not read files or create directories")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    task = build_analysis_batch(batch_payload).tasks[0]
    assert task.instances[0].parameters["analysis_start_time_sec"] == 0.5
    assert task.instances[0].parameters["analysis_end_time_sec"] == 0


def test_empty_analysis_is_valid_and_produces_no_preparation_issues(batch_payload):
    for task in batch_payload["tasks"]:
        task["analysis_config"] = {"display_sequence": []}
    batch = build_analysis_batch(batch_payload)
    assert all(task.instances == () for task in batch.tasks)
    assert batch.preparation_issues == ()


def test_non_none_segment_mode_is_explicitly_rejected(batch_payload):
    batch_payload["conditions"][0]["segmented_analysis"] = {"mode": "time"}
    with pytest.raises(AnalysisInputError) as caught:
        build_analysis_batch(batch_payload)
    assert caught.value.issue.error_type == "UnsupportedSegmentation"


def test_unknown_algorithm_is_retained_with_preparation_issue(batch_payload):
    config = batch_payload["tasks"][0]["analysis_config"]
    config["SPL 1"]["type"] = "FUTURE_ALGORITHM"
    batch = build_analysis_batch(batch_payload)
    assert batch.tasks[0].instances[0].analysis_type == "FUTURE_ALGORITHM"
    assert any(issue.error_type == "UnsupportedAnalysisType" for issue in batch.preparation_issues)


@pytest.mark.parametrize("field,value", [
    ("wav_path", "relative.wav"), ("wav_path", ""),
    ("analysis_result_directory", "relative"),
    ("analysis_result_directory", 123),
])
def test_invalid_path_format(batch_payload, field, value):
    batch_payload["tasks"][0][field] = value
    with pytest.raises(AnalysisInputError) as caught:
        build_analysis_batch(batch_payload)
    assert caught.value.issue.field_path == f"tasks[0].{field}"


def test_calibration_is_not_invented_or_applied(batch_payload, tmp_path):
    batch_payload["stations"][0]["channels"][0]["v2pa_factor"] = 2.5
    batch_payload["tasks"][0]["analysis_result_directory"] = str(tmp_path / "results")
    batch = build_analysis_batch(batch_payload)
    assert batch.tasks[0].channels[0].v2pa_factor == 2.5
    assert batch.tasks[1].channels[0].v2pa_factor is None
    assert not (tmp_path / "results").exists()
    issues = [issue.error_type for issue in batch.preparation_issues if issue.task_id == "analysis-1-1"]
    assert "CalibrationNotProvided" not in issues
    assert "MissingOutputDirectory" not in issues


def test_disconnected_hardware_and_recording_config_are_not_consulted(batch_payload):
    batch_payload["profiles"] = {"bad": object()}
    batch_payload["acq"] = {"sample_rate": -1}
    batch_payload["conditions"][0]["test_queue"] = "missing queue file"
    batch_payload["conditions"][0]["profile_id"] = "missing profile"
    assert len(build_analysis_batch(batch_payload).tasks) == 8


@pytest.mark.parametrize("field,value", [
    ("stations", None), ("conditions", {}), ("tasks", "wrong"),
])
def test_malformed_batch_collections(batch_payload, field, value):
    batch_payload[field] = value
    with pytest.raises(AnalysisInputError) as caught:
        build_analysis_batch(batch_payload)
    assert caught.value.issue.field_path == field


@pytest.mark.parametrize("value", [True, 0, -1, "1.0", float("nan"), float("inf")])
def test_invalid_provided_calibration_is_not_replaced_with_one(batch_payload, value):
    batch_payload["stations"][0]["channels"][0]["v2pa_factor"] = value
    with pytest.raises(AnalysisInputError) as caught:
        build_analysis_batch(batch_payload)
    assert caught.value.issue.field_path == "stations[0].channels[0].v2pa_factor"


@pytest.mark.parametrize("value", [object(), {1, 2}, float("nan")])
def test_non_json_parameters_fail_at_input_boundary(batch_payload, value):
    batch_payload["tasks"][0]["analysis_config"]["SPL 1"]["extra_parameter"] = value
    with pytest.raises(AnalysisInputError) as caught:
        build_analysis_batch(batch_payload)
    assert caught.value.issue.task_id == "analysis-1-1"
    assert "analysis_config" in caught.value.issue.field_path


def test_no_four_channel_or_four_station_limit(batch_payload):
    station = batch_payload["stations"][0]
    station["channels"] = [
        {"channel_name": f"CH{i + 1}", "signal_type": "sound_pressure"}
        for i in range(8)
    ]
    batch_payload["stations"] = [station]
    batch_payload["conditions"] = batch_payload["conditions"][:1]
    batch_payload["tasks"] = batch_payload["tasks"][:1]
    batch_payload["tasks"][0]["analysis_config"]["SPL 1"]["analysis_channels"] = [7]
    task = build_analysis_batch(batch_payload).tasks[0]
    assert len(task.channels) == 8
    assert task.instances[0].channel_index == 7


def test_disabled_time_and_threshold_settings_are_not_evaluated(batch_payload):
    config = batch_payload["tasks"][0]["analysis_config"]["SPL 1"]
    config.update(analysis_start_time_sec="unused", analysis_end_time_sec="unused",
                  scalar_upper_value="unused", limit_checked=False)
    assert build_analysis_batch(batch_payload).tasks[0].instances[0].parameters == config


def test_invalid_later_task_returns_no_partial_batch_and_can_be_corrected(batch_payload):
    batch_payload["tasks"][-1]["condition_id"] = "missing"
    with pytest.raises(AnalysisInputError) as caught:
        build_analysis_batch(batch_payload)
    assert caught.value.issue.task_id == "analysis-2-4"
    batch_payload["tasks"][-1]["condition_id"] = "condition-2"
    assert len(build_analysis_batch(batch_payload).tasks) == 8


def test_missing_labels_and_channel_ids_do_not_change_identity(batch_payload):
    station = batch_payload["stations"][0]
    del station["station_name"]
    del batch_payload["conditions"][0]["condition_name"]
    for channel in station["channels"]:
        del channel["channel_id"]
        channel["channel_name"] = "同名通道"
    task = build_analysis_batch(batch_payload).tasks[0]
    assert task.station_name is None
    assert task.condition_name is None
    assert task.station_id == "station-1"
    assert [channel.channel_index for channel in task.channels] == list(range(4))


def test_unsupported_single_channel_diagnostic_points_to_supplied_field(batch_payload):
    config = batch_payload["tasks"][0]["analysis_config"]
    config["display_sequence"] = ["FFT 1"]
    del config["FFT 1"]["analysis_channels"]
    config["FFT 1"]["analysis_channel"] = 2
    issues = build_analysis_batch(batch_payload).preparation_issues
    assert issues[0].error_type == "UnsupportedSignalType"
    assert issues[0].field_path.endswith(".analysis_channel")

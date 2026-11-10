"""Recording requests from the separately loaded recording configuration."""

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from base.recording_process_protocol import FrozenConfig
from base.workbench_configuration_loader import load_workbench_configuration
from ui.workbench_recording_device import WorkbenchRecordingDeviceSnapshot
from unit_test.base.test_workbench_recording_request import ChangingCalibrationStore
from unit_test.base.vkinging_fakes import device_info, input_config
from unit_test.base.workbench_config_fixtures import write_workbench_files


@pytest.fixture
def loaded(tmp_path):
    files = write_workbench_files(tmp_path)
    return load_workbench_configuration(files["programs"], files["queues"], files["layout"])


@pytest.fixture
def device():
    channels = tuple(range(8))
    info = device_info(physical_channels=list(channels), input_config=input_config(48000))
    return WorkbenchRecordingDeviceSnapshot("probe", 3, FrozenConfig.snapshot(info), channels, 48000)


def test_device_rate_and_fixed_streaming_from_loaded_recording(loaded, device):
    from base.workbench_recording_request import build_recording_request

    recording = loaded.recording
    detail = recording.parameters_for("c1")
    assert "sample_rate" not in detail and "use_streaming_recording" not in detail
    assert not hasattr(recording, "sample_rate") and not hasattr(recording, "use_streaming_recording")
    request = build_recording_request(
        recording, "c1", device, "take-1", recorded_at=datetime(2026, 11, 9, 13, 14, 15, 123999),
        ve_calibration_store=ChangingCalibrationStore(),
    )
    assert request.sample_rate == 48000
    assert request.streaming is True
    assert request.target_samples == 120000
    assert request.trim_samples == 0
    assert [station.station_id for station in request.station_group.stations] == ["s1", "s2"]


def with_parameters(recording, **changes):
    return replace(recording, parameters_by_queue={
        **recording.parameters_by_queue,
        "测试队列A": dict(recording.parameters_for("c1")) | changes,
    })


def test_recording_key_compares_effective_options_and_source_identity(loaded):
    from base.workbench_recording_request import recording_config_key

    recording = loaded.recording
    original = with_parameters(recording, startup_trim_ms="100", audio_validation={
        "enabled": True, "min_peak": "0.25",
    })
    equivalent = with_parameters(recording, startup_trim_ms=100.0, audio_validation={
        "enabled": True, "min_peak": 0.25, "unused_threshold": 9,
    })
    condition = equivalent.context.conditions[0]
    equivalent = replace(equivalent, context=replace(equivalent.context, conditions=(
        replace(condition, applicable_station_ids=tuple(reversed(condition.applicable_station_ids)),
                segmented_analysis={"mode": "changed"}), equivalent.context.conditions[1],
    )))
    key = recording_config_key(original, source_identity="product.json")
    assert key == recording_config_key(equivalent, source_identity="product.json")
    assert key != recording_config_key(original, source_identity="second-product.json")
    assert key != recording_config_key(with_parameters(original, total_time=9), source_identity="product.json")
    assert key != recording_config_key(with_parameters(original, startup_trim_ms=0), source_identity="product.json")


@pytest.mark.parametrize("field,value", [
    ("project_id", "other-product"), ("project_name", "新产品"),
    ("result_root_directory", "other-root"),
])
def test_recording_key_tracks_product_identity_and_output(loaded, field, value):
    from base.workbench_recording_request import recording_config_key

    recording = loaded.recording
    changed = replace(recording, context=replace(recording.context, **{field: value}))
    assert recording_config_key(recording, source_identity="same") != recording_config_key(changed, source_identity="same")


@pytest.mark.parametrize("change", ["condition-name", "station-name", "channel", "association", "condition-order"])
def test_recording_key_tracks_ordered_work_and_channel_mapping(loaded, change):
    from base.workbench_recording_request import recording_config_key

    recording = loaded.recording
    context = recording.context
    first, second = context.conditions
    station, *others = context.stations
    if change == "condition-name":
        context = replace(context, conditions=(replace(first, condition_name="新工况"), second))
    elif change == "association":
        context = replace(context, conditions=(replace(first, applicable_station_ids=("s1",)), second))
    elif change == "condition-order":
        context = replace(context, conditions=(second, first))
    else:
        station = (replace(station, station_name="新工位") if change == "station-name" else
                   replace(station, channels=tuple(reversed(station.channels))))
        context = replace(context, stations=(station, *others))
    assert recording_config_key(recording, source_identity="same") != recording_config_key(
        replace(recording, context=context), source_identity="same",
    )


def test_preview_mode_is_separate_ui_input_with_request_default(loaded, device):
    from base.workbench_recording_request import build_recording_request

    kwargs = {"recorded_at": datetime(2026, 11, 9), "ve_calibration_store": ChangingCalibrationStore()}
    default = build_recording_request(loaded.recording, "c1", device, "default", **kwargs)
    supplied = build_recording_request(
        loaded.recording, "c1", device, "supplied", **kwargs,
        preview_time_mode=loaded.ui.preview_time_modes["测试队列A"],
    )
    assert default.preview_time_mode == "relative_latest"
    assert supplied.preview_time_mode == "cumulative"


def test_readable_paths_collision_allocation_and_one_calibration_snapshot(loaded, device):
    from base.workbench_recording_request import build_recording_request

    context = loaded.recording.context
    stations = tuple(replace(station, station_name="工位", channels=tuple(reversed(station.channels)))
                     for station in context.stations)
    recording = replace(loaded.recording, context=replace(context, stations=stations))
    calibration = ChangingCalibrationStore()
    now = datetime(2026, 11, 9, 13, 14, 15, 123999)
    first, second = (
        build_recording_request(recording, "c2", device, take, recorded_at=now, ve_calibration_store=calibration)
        for take in ("take-1", "take-2")
    )
    assert calibration.reads == 2
    group = first.station_group
    assert Path(group.final_path) == Path(context.result_root_directory) / "电机" / "工况2" / "2026-11-09_13-14-15-123"
    assert first.path != second.path
    assert Path(first.path).parent == Path(group.final_path).parent
    assert Path(first.path).name.endswith("-take-1.staging")
    assert group.final_path == second.station_group.final_path
    assert group.allocate_final_path is True
    assert group.prepared_generation == 3
    assert not Path(context.result_root_directory).exists()
    assert [station.filename for station in group.stations] == ["工位.wav", "工位_2.wav"]
    assert [[(c.physical_channel_index, c.wav_channel_index) for c in station.channels]
            for station in group.stations] == [[(3, 0), (2, 1)], [(7, 0), (6, 1)]]
    metadata = group.stations[1].calibration_metadata.to_dict()
    assert metadata["acquisition"]["sample_rate"] == 48000
    assert metadata["recorded_channels"][0]["v2pa_factor"] == 10
    assert second.station_group.stations[1].calibration_metadata["recorded_channels"][0]["v2pa_factor"] == 20


@pytest.mark.parametrize("trim, expected", [(0, 0), (12.5, 600), (None, 4800)])
def test_trim_and_disabled_quality_keep_existing_resolver_semantics(loaded, device, trim, expected):
    from base.workbench_recording_request import build_recording_request, recording_config_key

    recording = with_parameters(loaded.recording, startup_trim_ms=trim,
                                audio_validation={"enabled": False, "min_peak": "bad"})
    request = build_recording_request(recording, "c1", device, "take", recorded_at=datetime(2026, 11, 9),
                                      ve_calibration_store=ChangingCalibrationStore())
    assert request.trim_samples == expected
    assert request.target_samples == 120000 + expected
    assert request.validation_thresholds.to_dict() == {"enabled": False}
    equivalent = with_parameters(recording, audio_validation={"enabled": False, "min_peak": 0.9})
    assert recording_config_key(recording, source_identity="same") == recording_config_key(equivalent, source_identity="same")


@pytest.mark.parametrize("threshold", ["bad", float("inf")])
def test_invalid_quality_is_comparable_but_diagnosed_on_start(loaded, device, threshold):
    from base.workbench_recording_request import (
        WorkbenchRecordingRequestError, build_recording_request, recording_config_key,
    )

    recording = with_parameters(loaded.recording, audio_validation={"enabled": True, "min_peak": threshold})
    key = recording_config_key(recording, source_identity="same")
    assert key == recording_config_key(recording, source_identity="same")
    assert key != recording_config_key(loaded.recording, source_identity="same")
    calibration = ChangingCalibrationStore()
    with pytest.raises(WorkbenchRecordingRequestError, match=r"audio_validation\.min_peak") as caught:
        build_recording_request(recording, "c1", device, "take", recorded_at=datetime(2026, 11, 9),
                                ve_calibration_store=calibration)
    assert "motor" in str(caught.value) and "工况1" in str(caught.value)
    assert calibration.reads == 0


@pytest.mark.parametrize("problem, diagnostic", [
    ("unknown", "missing"), ("no-station", "关联录制工位"), ("empty-channel", "s1"),
    ("device-channel", "物理索引 99"), ("root", "根目录"), ("id", "录制 ID"),
])
def test_request_boundary_diagnoses_identity_mapping_and_paths(loaded, device, problem, diagnostic):
    from base.workbench_recording_request import WorkbenchRecordingRequestError, build_recording_request

    recording = loaded.recording
    context = recording.context
    condition, other = context.conditions
    if problem == "no-station":
        context = replace(context, conditions=(replace(condition, applicable_station_ids=()), other))
    elif problem in ("empty-channel", "device-channel"):
        station, *others = context.stations
        channel = station.channels[0]
        channels = () if problem == "empty-channel" else (replace(
            channel, hardware_channel=replace(channel.hardware_channel, physical_channel_index=99)),)
        context = replace(context, stations=(replace(station, channels=channels), *others))
    elif problem == "root":
        context = replace(context, result_root_directory="")
    calibration = ChangingCalibrationStore()
    with pytest.raises(WorkbenchRecordingRequestError, match=diagnostic) as caught:
        build_recording_request(
            replace(recording, context=context), "missing" if problem == "unknown" else "c1", device,
            "../unsafe" if problem == "id" else "take", recorded_at=datetime(2026, 11, 9),
            ve_calibration_store=calibration,
        )
    assert "motor" in str(caught.value)
    assert calibration.reads == 0

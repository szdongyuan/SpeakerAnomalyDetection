import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

import pytest

from base.recording_process_protocol import FrozenConfig
from consts import model_consts
from ui.workbench_recording_device import WorkbenchRecordingDeviceSnapshot
from unit_test.base.test_motor_workbench_config import config_files, repository
from unit_test.base.vkinging_fakes import device_info


@pytest.fixture
def recording_config(repository):
    recording = repository.load_active().recording
    config = recording.context
    condition = replace(config.conditions[0], applicable_station_ids=("NUL", "../s:3"))
    channel = config.stations[0].channels[0]
    first = replace(channel, hardware_channel=replace(channel.hardware_channel, physical_channel_index=7))
    second = replace(channel, channel_id="second",
                     hardware_channel=replace(channel.hardware_channel, physical_channel_index=1))
    stations = (
        replace(config.stations[0], station_id="../s:3", channels=(second, first)),
        replace(config.stations[1], station_id="NUL", channels=(first,)),
    )
    device = WorkbenchRecordingDeviceSnapshot("probe", 1, FrozenConfig.snapshot(device_info()), (7, 1), 51200)
    recording = replace(recording, context=replace(recording.context, conditions=(condition,), stations=stations))
    return recording, stations, device


class ChangingCalibrationStore:
    def __init__(self):
        self.reads = 0

    def observe(self, device):
        self.reads += 1
        assert device["machine_id"] == "test-machine-1"
        return {7: {
            "status": "valid", "v2pa_factor": self.reads * 10, "standard_spl": 94.0,
            "calibrated_at": "2026-11-07T00:00:00+00:00",
            "calibration_sample_rate": 51200, "calibration_duration_seconds": 2.0,
        }}


def with_detail(recording, **detail):
    condition = recording.context.conditions[0]
    return replace(recording, parameters_by_queue={condition.test_queue: detail})


def recording_with(recording, **changes):
    context_changes = {key: changes.pop(key) for key in ("result_root_directory",) if key in changes}
    condition = replace(recording.context.conditions[0], **changes)
    return replace(recording, context=replace(recording.context, conditions=(condition,), **context_changes))


def request_for(recording, stations, device, request_id, *, project_name=None, recorded_at=None, **kwargs):
    from base.workbench_recording_request import build_recording_request
    context = replace(recording.context, stations=stations,
                      project_name=recording.context.project_id if project_name is None else project_name)
    return build_recording_request(replace(recording, context=context), context.conditions[0].condition_id,
                                   device, request_id, recorded_at=recorded_at or datetime(2026, 11, 9), **kwargs)


def test_uses_device_rate_and_one_full_calibration_snapshot(recording_config, tmp_path, monkeypatch):
    recording, stations, device = recording_config
    monkeypatch.chdir(tmp_path)
    recording = with_detail(recording, total_time=0.75, sample_rate="legacy", startup_trim_ms=12.5,
                            recording_root="nested/../configured",
                            audio_validation=MappingProxyType({"enabled": True, "min_peak": 0.7}))
    calibration = ChangingCalibrationStore()
    request = request_for(recording, stations, device, "take-1", ve_calibration_store=calibration,
                                                project_name="产品甲", recorded_at=datetime(2026, 11, 9, 13, 14, 15, 123999))

    assert (request.channels, request.sample_rate) == ((7, 1), 51200)
    assert (request.target_samples, request.trim_samples) == (38400 + 640, 640)
    assert request.streaming and not request.effective_streaming
    assert request.validation_thresholds["enabled"] is True
    assert request.validation_thresholds["min_peak"] == 0.7
    group = request.station_group
    assert group.prepared_generation == device.generation
    assert Path(request.path).parent == Path(group.final_path).parent == Path(recording.context.result_root_directory) / "产品甲" / recording.context.conditions[0].condition_name
    assert Path(group.final_path).name == "2026-11-09_13-14-15-123"
    assert request.path != group.final_path
    assert not Path(request.path).parent.exists()
    first, second = group.stations
    assert (first.station_id, second.station_id) == ("../s:3", "NUL")
    assert first.filename == stations[0].station_name + ".wav"
    assert second.filename == stations[1].station_name + ".wav"
    assert [(c.physical_channel_index, c.wav_channel_index) for c in first.channels] == [(1, 0), (7, 1)]
    assert [(c.physical_channel_index, c.wav_channel_index) for c in second.channels] == [(7, 0)]
    assert calibration.reads == 1
    first_meta = first.calibration_metadata.to_dict()
    second_meta = second.calibration_metadata.to_dict()
    assert first_meta["backend"] == second_meta["backend"] == "vkinging"
    assert first_meta["schema_version"] == second_meta["schema_version"]
    assert first_meta["acquisition"] == second_meta["acquisition"]
    assert first_meta["acquisition"]["sample_rate"] == 51200
    assert first_meta["recorded_channels"][0]["physical_input_channel"] == 1
    assert first_meta["recorded_channels"][1] == {
        **second_meta["recorded_channels"][0], "wav_channel_index": 1,
    }
    assert second_meta["recorded_channels"][0]["v2pa_factor"] == 10


def test_repeated_timestamp_has_unique_staging_and_product_root(recording_config, tmp_path, monkeypatch):
    recording, stations, device = recording_config
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(model_consts, "STORED_RECORDED_PATH", "default")
    now = datetime(2026, 11, 9, 13, 14, 15, 123999)
    first, retry = (
        request_for(recording, stations, device, take,
            project_name="产品甲", recorded_at=now, ve_calibration_store=ChangingCalibrationStore())
        for take in ("take-1", "take-2")
    )
    assert first.path != retry.path
    assert first.station_group.final_path == retry.station_group.final_path
    assert Path(first.path).parent == Path(recording.context.result_root_directory) / "产品甲" / recording.context.conditions[0].condition_name
    assert first.station_group.allocate_final_path is True
    assert not (tmp_path / "default").exists()


@pytest.mark.parametrize("root", ["", "   ", None, 42, "bad\0root", "NUL", "bad?root"])
def test_invalid_product_root_never_falls_back(recording_config, root):
    from base.workbench_recording_request import WorkbenchRecordingRequestError

    recording, stations, device = recording_config
    recording = recording_with(recording, result_root_directory=root)
    with pytest.raises(WorkbenchRecordingRequestError):
        request_for(recording, stations, device, "take", project_name="产品")


def test_windows_safe_names_and_case_insensitive_suffix_clashes(recording_config):
    recording, stations, device = recording_config
    names = ["工位", "工位", "工位_2", "A", "a", "a_2", "x/y", "x:y", "NUL", "COM1.txt", "...", "尾部. "]
    stations = tuple(replace(stations[0], station_id=str(index), station_name=name)
                     for index, name in enumerate(names))
    recording = recording_with(recording, condition_name="CON", applicable_station_ids=tuple(s.station_id for s in stations))
    request = request_for(recording, stations, device, "take", project_name="../产品:甲",
                                               ve_calibration_store=ChangingCalibrationStore())
    assert Path(request.station_group.final_path).parent.parts[-2:] == ("_产品_甲", "_CON")
    assert [s.filename for s in request.station_group.stations] == [
        "工位.wav", "工位_2.wav", "工位_2_2.wav", "A.wav", "a_2.wav", "a_2_2.wav",
        "x_y.wav", "x_y_2.wav", "_NUL.wav", "_COM1.txt.wav", "工位_3.wav", "尾部.wav",
    ]


@pytest.mark.parametrize(("invalid", "diagnostic"), [
    ("no-station", "工位"),
    ("empty", "NUL"),
    ("duplicate", "7"),
    ("unavailable", "99"),
    ("mixed", "other-device"),
    ("short", "帧"),
    ("unsafe-id", "录制 ID"),
])
def test_request_errors_identify_condition_before_calibration(recording_config, invalid, diagnostic):
    from base.workbench_recording_request import WorkbenchRecordingRequestError

    recording, stations, device = recording_config
    first, second = stations
    channel = second.channels[0]
    request_id = "take-1"
    if invalid == "no-station":
        recording = recording_with(recording, applicable_station_ids=())
    elif invalid == "empty":
        second = replace(second, channels=())
    elif invalid == "duplicate":
        second = replace(second, channels=(channel, channel))
    elif invalid == "unavailable":
        channel = replace(channel, hardware_channel=replace(channel.hardware_channel, physical_channel_index=99))
        second = replace(second, channels=(channel,))
    elif invalid == "mixed":
        channel = replace(channel, hardware_channel=replace(channel.hardware_channel, device_ref="other-device"))
        second = replace(second, channels=(channel,))
    elif invalid == "short":
        recording = with_detail(recording, total_time=0.0000001)
    else:
        request_id = "../outside"

    calibration = ChangingCalibrationStore()
    with pytest.raises(WorkbenchRecordingRequestError, match=diagnostic) as caught:
        request_for(recording, (first, second), device, request_id,
                                          ve_calibration_store=calibration)
    assert recording.context.conditions[0].condition_name in str(caught.value)
    assert recording.context.project_id in str(caught.value)
    assert calibration.reads == 0



def test_invalid_enabled_quality_threshold_names_field_before_calibration(recording_config):
    from base.workbench_recording_request import WorkbenchRecordingRequestError

    recording, stations, device = recording_config
    recording = with_detail(recording, total_time=1, audio_validation={"enabled": True, "min_peak": "bad"})
    calibration = ChangingCalibrationStore()
    with pytest.raises(WorkbenchRecordingRequestError, match=r"audio_validation\.min_peak") as caught:
        request_for(recording, stations, device, "take-1", ve_calibration_store=calibration)
    assert recording.context.conditions[0].condition_name in str(caught.value)
    assert calibration.reads == 0


def test_quality_limit_accepts_infinity_string_but_rejects_overflowed_json_number(recording_config):
    from base.recording_settings import validate_recorded_audio_metrics
    from base.workbench_recording_request import WorkbenchRecordingRequestError

    recording, stations, device = recording_config
    calibration = ChangingCalibrationStore()
    limits = {"enabled": True, "min_peak": "-inf", "min_rms_dbfs": -100, "min_variance": 0}
    valid = with_detail(recording, total_time=1, audio_validation=limits)
    request = request_for(valid, stations, device, "valid", ve_calibration_store=calibration)
    assert request.validation_thresholds["min_peak"] == "-inf"
    assert validate_recorded_audio_metrics(10, 0.5, 0.25, 0.1, request.validation_thresholds.to_dict())[0]

    overflowed = json.loads('{"min_peak": -1e400}')
    invalid = with_detail(recording, total_time=1, audio_validation=limits | overflowed)
    with pytest.raises(WorkbenchRecordingRequestError, match=r"audio_validation\.min_peak"):
        request_for(invalid, stations, device, "invalid", ve_calibration_store=calibration)
    assert calibration.reads == 1


def test_disabled_quality_ignores_overflowed_threshold(recording_config):
    recording, stations, device = recording_config
    limits = json.loads('{"enabled": false, "min_peak": -1e400}')
    recording = with_detail(recording, total_time=1, audio_validation=limits)
    request = request_for(
        recording, stations, device, "disabled", ve_calibration_store=ChangingCalibrationStore(),
    )
    assert request.validation_thresholds.to_dict() == {"enabled": False}

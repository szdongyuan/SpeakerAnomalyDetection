from dataclasses import FrozenInstanceError, replace
from enum import IntEnum
import os
import pickle

import pytest

from base.recording_process_protocol import FrozenConfig, RecordingRequest, RecordingResult
from base.workbench_recording_protocol import (
    StationRecordingGroup,
    StationRecordingPlan,
    StationWav,
    StationWavChannel,
    WorkbenchRecordingCompletion,
)
from consts.recording_preview_consts import PREVIEW_TIME_MODE_CUMULATIVE
from unit_test.base.vkinging_fakes import device_info, wav_metadata


SOUNDCARD = {"index": 0, "hostapi": 0, "name": "Test input", "max_input_channels": 4}


def channel(physical, wav_index, device_ref="input"):
    return StationWavChannel(
        f"ch-{physical}", f"Input {physical}", "sound", device_ref, physical, wav_index,
    )


def soundcard_metadata(physical_channels):
    return {"recorded_channels": [
        {"wav_channel_index": index, "physical_input_channel": physical,
         "v2pa_factor": None, "standard_spl": None, "calibrated": False}
        for index, physical in enumerate(physical_channels)
    ]}


def ve_metadata(physical_channels):
    metadata = wav_metadata(
        ("none",) * len(physical_channels), 51200, physical_channels,
    )
    metadata["acquisition"]["machine_id"] = "test-machine-1"
    return metadata


@pytest.fixture
def grouped_request(tmp_path):
    stations = (
        StationRecordingPlan("left", "01-left.wav", (channel(2, 0), channel(0, 1)),
                             soundcard_metadata((2, 0))),
        StationRecordingPlan("right", "02-right.wav", (channel(2, 0),)),
    )
    group = StationRecordingGroup("project", "condition", str(tmp_path / "final"), stations)
    return RecordingRequest(
        "request", "main", 8000, 800, (0, 1, 2, 3), SOUNDCARD,
        str(tmp_path / "staging"), True, 0, {}, None, {}, station_group=group,
    )


def test_group_request_pickles_station_order_and_owns_its_inputs(grouped_request):
    group = grouped_request.station_group
    metadata = soundcard_metadata((2, 0))
    channels = list(group.stations[0].channels)
    first = replace(group.stations[0], channels=channels, calibration_metadata=metadata)
    stations = [first, group.stations[1]]
    request = replace(grouped_request, station_group=replace(group, stations=stations))

    channels.reverse()
    stations.clear()
    metadata["recorded_channels"][0]["physical_input_channel"] = 3
    restored = pickle.loads(pickle.dumps(request))

    assert restored == request
    assert restored.channels == (0, 1, 2, 3)
    assert restored.streaming is True
    assert restored.effective_streaming is False
    restored_group = restored.station_group
    assert (restored_group.project_id, restored_group.condition_id) == ("project", "condition")
    assert restored_group.final_path == group.final_path
    assert tuple(station.station_id for station in restored_group.stations) == ("left", "right")
    first, second = restored_group.stations
    assert first.channels == (channel(2, 0), channel(0, 1))
    assert second.channels == (channel(2, 0),)
    assert isinstance(first.calibration_metadata, FrozenConfig)
    assert first.calibration_metadata.to_dict() == soundcard_metadata((2, 0))
    assert second.calibration_metadata is None
    with pytest.raises(FrozenInstanceError):
        restored_group.stations = ()
    with pytest.raises(FrozenInstanceError):
        first.channels[0].wav_channel_index = 1
    with pytest.raises(TypeError):
        first.calibration_metadata["recorded_channels"][0]["physical_input_channel"] = 3


@pytest.mark.parametrize(("changes", "message"), [
    ({"channels": ()}, "channels"),
    ({"channels": (channel(2, 0), channel(2, 1))}, "physical"),
    ({"channels": (channel(2, 1), channel(0, 0))}, "WAV channel index"),
    ({"filename": "../escape.wav"}, "filename"),
    ({"filename": r"..\escape.wav"}, "filename"),
    ({"filename": "subdir/child.wav"}, "filename"),
    ({"filename": r"C:\escape.wav"}, "filename"),
    ({"filename": "NUL.wav"}, "filename"),
    ({"filename": "stream:other.wav"}, "filename"),
    ({"filename": "other.txt"}, "filename"),
])
def test_station_rejects_invalid_output_plan(grouped_request, changes, message):
    with pytest.raises(ValueError, match=message):
        replace(grouped_request.station_group.stations[0], **changes)


@pytest.mark.parametrize("problem", ["no stations", "duplicate station", "mixed device",
                                    "path collision", "case-insensitive path collision"])
def test_group_rejects_conflicting_stations(grouped_request, problem):
    group = grouped_request.station_group
    first, second = group.stations
    candidates = {
        "no stations": (),
        "duplicate station": (first, replace(second, station_id=first.station_id)),
        "mixed device": (first, replace(second, channels=(channel(2, 0, "other-input"),))),
        "path collision": (first, replace(second, filename=first.filename)),
        "case-insensitive path collision": (first, replace(second, filename=first.filename.upper())),
    }
    with pytest.raises(ValueError, match="stations|station_id|device_ref|filename"):
        replace(group, stations=candidates[problem])


@pytest.mark.parametrize(("problem", "message"), [
    ("relative staging", "absolute"),
    ("relative final", "absolute"),
    ("same directory", "distinct siblings"),
    ("different parent", "distinct siblings"),
    ("unavailable channel", "unavailable physical channels"),
    ("calibration purpose", "main recording"),
    ("request-level metadata", "per-station calibration metadata"),
])
def test_request_rejects_invalid_group_placement(grouped_request, tmp_path, problem, message):
    group = grouped_request.station_group
    first, second = group.stations
    with pytest.raises(ValueError, match=message):
        if problem == "relative staging":
            replace(grouped_request, path="relative-staging")
        elif problem == "relative final":
            replace(group, final_path="relative-final")
        elif problem == "same directory":
            replace(grouped_request, station_group=replace(
                group, final_path=str(tmp_path / "unused" / ".." / "staging"),
            ))
        elif problem == "different parent":
            replace(grouped_request, station_group=replace(
                group, final_path=str(tmp_path / "other" / "final"),
            ))
        elif problem == "unavailable channel":
            second = replace(second, channels=(channel(4, 0),))
            replace(grouped_request, station_group=replace(group, stations=(first, second)))
        elif problem == "calibration purpose":
            replace(grouped_request, purpose="calibration", channels=(2,))
        else:
            replace(grouped_request, calibration_metadata=soundcard_metadata((0, 1, 2, 3)))


def test_soundcard_metadata_matches_station_physical_order(grouped_request):
    group = grouped_request.station_group
    first, second = group.stations
    with pytest.raises(ValueError, match="metadata channel order"):
        first = replace(first, calibration_metadata=soundcard_metadata((0, 2)))
        replace(grouped_request, station_group=replace(group, stations=(first, second)))


def test_ve_metadata_is_validated_against_each_station_and_device(grouped_request):
    group = grouped_request.station_group
    first, second = group.stations
    first = replace(first, calibration_metadata=ve_metadata((2, 0)))
    second = replace(second, calibration_metadata=ve_metadata((2,)))
    request = replace(
        grouped_request, sample_rate=51200, device=device_info(physical_channels=(0, 1, 2, 3)),
        station_group=replace(group, stations=(first, second)),
    )
    assert request.station_group.stations == (first, second)

    with pytest.raises(ValueError, match="sample_rate must match"):
        replace(request, sample_rate=44100)
    with pytest.raises(ValueError, match="metadata channel order"):
        replace(request, station_group=replace(group, stations=(
            replace(first, calibration_metadata=ve_metadata((0, 2))), second,
        )))
    with pytest.raises(ValueError, match="VE metadata"):
        replace(request, station_group=replace(group, stations=(
            first, replace(second, calibration_metadata=None),
        )))
    mismatched_metadata = ve_metadata((2,))
    mismatched_metadata["acquisition"]["machine_id"] = "another-device"
    with pytest.raises(ValueError, match="metadata acquisition"):
        replace(request, station_group=replace(group, stations=(
            first, replace(second, calibration_metadata=mismatched_metadata),
        )))


@pytest.mark.parametrize(("purpose", "channels", "effective_streaming"), [
    ("main", (2, 0), True),
    ("calibration", (1,), False),
])
def test_legacy_positional_requests_and_results(tmp_path, purpose, channels, effective_streaming):
    request = RecordingRequest(
        "legacy", purpose, 8000, 800, channels, SOUNDCARD, str(tmp_path / "legacy.wav"),
        True, 0, {"historical": True}, None, {}, PREVIEW_TIME_MODE_CUMULATIVE,
    )
    result = RecordingResult("legacy", purpose, request.path, 8000, channels, 800, 800, False)

    assert request.station_group is None
    assert request.channels == channels
    assert request.effective_streaming is effective_streaming
    assert request.preview_time_mode == PREVIEW_TIME_MODE_CUMULATIVE
    assert not request.monitor
    assert result.station_files == ()


def test_legacy_ve_requests_still_validate_main_metadata_and_calibration(tmp_path):
    path = str(tmp_path / "ve.wav")
    device = device_info(physical_channels=(0, 1, 2, 3))
    main = RecordingRequest(
        "main", "main", 51200, 9, (2, 0), device, path, True, 0, {}, ve_metadata((2, 0)), {},
    )
    calibration = RecordingRequest(
        "calibration", "calibration", 51200, 512000, (2,), device, path, True, 0, {}, None, {},
    )
    assert main.effective_streaming is True
    assert calibration.effective_streaming is False
    with pytest.raises(ValueError, match="VE metadata"):
        replace(main, calibration_metadata=None)
    with pytest.raises(ValueError, match="ten seconds"):
        replace(calibration, target_samples=9)


def test_staged_result_and_published_completion_pickle_absolute_wav_paths(grouped_request, tmp_path):
    req = grouped_request
    group = req.station_group
    staged = tuple(
        StationWav(station.station_id, str(tmp_path / "staging" / station.filename), station.channels)
        for station in group.stations
    )
    result = RecordingResult(
        req.request_id, req.purpose, req.path, req.sample_rate, req.channels,
        req.target_samples, req.target_samples, False, station_files=staged,
    )
    assert pickle.loads(pickle.dumps(result)).station_files == staged

    published = tuple(
        replace(wav, path=str(tmp_path / "final" / station.filename))
        for wav, station in zip(staged, group.stations)
    )
    completion = WorkbenchRecordingCompletion(
        req.request_id, group.project_id, group.condition_id, req.sample_rate,
        req.target_samples, published,
    )
    assert pickle.loads(pickle.dumps(completion)) == completion
    assert completion.stations == published
    with pytest.raises(FrozenInstanceError):
        completion.stations[0].path = staged[0].path


@pytest.mark.parametrize("name", ["staging.", "staging ", "NUL", "staging:stream"])
def test_group_rejects_unsafe_final_directory_components(grouped_request, tmp_path, name):
    with pytest.raises(ValueError, match="unsafe Windows path component"):
        replace(grouped_request.station_group, final_path=str(tmp_path / name))


def test_group_rejects_unsafe_staging_ancestor(grouped_request, tmp_path):
    request = replace(grouped_request, path=str(tmp_path / "parent" / "staging"),
                      station_group=replace(
                          grouped_request.station_group,
                          final_path=str(tmp_path / "parent" / "final"),
                      ))
    with pytest.raises(ValueError, match="unsafe Windows path component"):
        replace(request, path=str(tmp_path / "parent " / "staging"))


@pytest.mark.skipif(os.name != "nt", reason="Windows UNC paths")
@pytest.mark.parametrize("root", [r"\\server\share", r"\\?\UNC\server\share"])
def test_group_preserves_absolute_unc_directories(grouped_request, root):
    group = replace(grouped_request.station_group, final_path=root + r"\recordings\final")
    request = replace(grouped_request, path=root + r"\recordings\stage", station_group=group)
    assert request.station_group == group
    with pytest.raises(ValueError, match="unsafe Windows path component"):
        replace(group, final_path=root + r".\recordings\final")


def test_group_rejects_unordered_station_collection(grouped_request):
    group = grouped_request.station_group
    with pytest.raises(ValueError, match="ordered"):
        replace(group, stations={group.stations[1]})


def test_station_wav_and_result_snapshot_ordered_caller_collections(grouped_request, tmp_path):
    first, second = grouped_request.station_group.stations
    mapped = list(first.channels)
    wav = StationWav(first.station_id, str(tmp_path / first.filename), mapped)
    other_wav = StationWav(second.station_id, str(tmp_path / second.filename), second.channels)
    stations = [wav, other_wav]
    result = RecordingResult(
        "request", "main", grouped_request.path, 8000, (0, 1, 2, 3), 800, 800,
        False, station_files=stations,
    )
    completion = WorkbenchRecordingCompletion("request", "project", "condition", 8000, 800, stations)

    mapped.reverse()
    stations.clear()

    assert wav.channels == first.channels
    assert result.station_files == completion.stations == (wav, other_wav)
    assert pickle.loads(pickle.dumps(completion)) == completion


@pytest.mark.parametrize(("value", "message"), [
    ({"path": "relative.wav"}, "absolute"),
    ({"path": "wrong-suffix.txt"}, "WAV filename"),
    ({"channels": ()}, "channels"),
    ({"channels": ({"physical_channel_index": 2},)}, "channels"),
    ({"channels": (channel(2, 1), channel(0, 0))}, "WAV channel index"),
    ({"channels": (channel(2, 0), channel(2, 1))}, "duplicate physical"),
    ({"channels": {channel(2, 0)}}, "ordered"),
])
def test_station_wav_rejects_invalid_path_and_channel_contract(tmp_path, value, message):
    fields = {"station_id": "left", "path": str(tmp_path / "left.wav"), "channels": (channel(2, 0),)}
    if value.get("path") == "wrong-suffix.txt":
        value = {"path": str(tmp_path / value["path"])}
    with pytest.raises(ValueError, match=message):
        StationWav(**(fields | value))


@pytest.mark.parametrize("malformed", [None, {}, ({"station_id": "left"},)])
def test_completion_and_result_reject_malformed_station_collections(grouped_request, malformed):
    with pytest.raises(ValueError, match="station WAV"):
        RecordingResult(
            "request", "main", grouped_request.path, 8000, (0, 1, 2, 3), 800, 800,
            False, station_files=malformed,
        )
    with pytest.raises(ValueError, match="station WAV"):
        WorkbenchRecordingCompletion("request", "project", "condition", 8000, 800, malformed)



@pytest.mark.parametrize(("changes", "message"), [
    ({"request_id": ["request"]}, "request_id"),
    ({"sample_rate": 8000.0}, "sample_rate"),
    ({"final_frames": True}, "final_frames"),
    ({"stations": ()}, "at least one station WAV"),
])
def test_completion_rejects_invalid_identity_counts_or_empty_stations(tmp_path, changes, message):
    wav = StationWav("left", str(tmp_path / "left.wav"), (channel(2, 0),))
    fields = {
        "request_id": "request", "project_id": "project", "condition_id": "condition",
        "sample_rate": 8000, "final_frames": 800, "stations": (wav,),
    }
    with pytest.raises(ValueError, match=message):
        WorkbenchRecordingCompletion(**(fields | changes))



@pytest.mark.skipif(os.name != "nt", reason="Windows path namespaces")
@pytest.mark.parametrize(("path", "message"), [
    (r"\\.\pipe\recording", "drive or UNC"),
    (r"//./pipe/recording", "drive or UNC"),
    (r"\\?\GLOBALROOT\Device\recording", "drive or UNC"),
    (r"\\?\Volume{abc}\recording", "drive or UNC"),
    (r"\\?\Volume{12345678-1234-1234-1234-123456789abz}\recording", "drive or UNC"),
    (r"\\?\Volume{12345678123412341234123456789abc}\recording", "drive or UNC"),
    (r"\\server\.\recording", "unsafe Windows path component"),
    (r"\\server\..\recording", "unsafe Windows path component"),
    (r"\\?\UNC\server\..\recording", "extended Windows path"),
    (r"\\server\pipe\recording", "non-filesystem UNC share"),
    (r"\\server\MaIlSlOt\recording", "non-filesystem UNC share"),
    (r"\\server\IPC$\recording", "non-filesystem UNC share"),
    (r"\\?\UNC\server\PiPe\recording", "non-filesystem UNC share"),
    (r"\\?\UNC\server\MAILSLOT\recording", "non-filesystem UNC share"),
    (r"\\?\unc\server\ipc$\recording", "non-filesystem UNC share"),
    (r"1:\recording", "drive or UNC"),
    (r"\\?\1:\recording", "drive or UNC"),
    (r"\\?\C:\folder\.\recording", "extended Windows path"),
    (r"\\?\C:\folder\extra\..\recording", "extended Windows path"),
    (r"\\?\UNC\server\share\folder\..\recording", "extended Windows path"),
    (r"\\?\C:\folder/recording", "extended Windows path"),
    (r"\\?\C:\folder\\recording", "extended Windows path"),
])
def test_windows_directory_and_wav_reject_unsupported_paths(grouped_request, path, message):
    with pytest.raises(ValueError, match=message):
        replace(grouped_request.station_group, final_path=path)
    with pytest.raises(ValueError, match=message):
        StationWav("left", path + r"\left.wav", (channel(2, 0),))


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespaces")
def test_group_rejects_raw_extended_staging_traversal(grouped_request):
    group = replace(grouped_request.station_group, final_path=r"\\?\C:\recordings\final")
    with pytest.raises(ValueError, match="extended Windows path"):
        replace(grouped_request, path=r"\\?\C:\recordings\extra\..\stage", station_group=group)


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespaces")
@pytest.mark.parametrize(("staging", "final"), [
    (r"C:/recordings/extra/.././stage", r"C:\recordings\final"),
    (r"\\?\C:\recordings\stage", r"\\?\C:\recordings\final"),
    (r"\\server.\share\recordings\stage", r"\\server.\share\recordings\final"),
    (r"\\server\C$\pipe\stage", r"\\server\C$\pipe\final"),
    (r"\\?\UNC\server\C$\recordings\stage", r"\\?\UNC\server\C$\recordings\final"),
    (r"\\?\unc\server\share\recordings\stage", r"\\?\unc\server\share\recordings\final"),
    (
        r"\\?\Volume{12345678-1234-1234-1234-123456789abc}\recordings\stage",
        r"\\?\VOLUME{12345678-1234-1234-1234-123456789ABC}\recordings\final",
    ),
])
def test_windows_groups_and_wavs_preserve_safe_anchors_and_short_shaped_names(
    grouped_request, staging, final,
):
    first, second = grouped_request.station_group.stations
    first = replace(first, filename="RECOR~1.WAV")
    group = replace(grouped_request.station_group, final_path=final, stations=(first, second))
    request = replace(grouped_request, path=staging, station_group=group)
    wav = StationWav("left", os.path.join(final, first.filename), first.channels)
    assert request.path == staging
    assert wav.path == os.path.join(final, "RECOR~1.WAV")



@pytest.mark.parametrize("role", ["staging", "final"])
def test_group_rejects_native_root_and_paths_normalized_to_it(grouped_request, tmp_path, role):
    root = type(tmp_path)(tmp_path.anchor)
    group = grouped_request.station_group
    path = str(root / "recordings" / "..")
    with pytest.raises(ValueError, match="filesystem root"):
        if role == "staging":
            group = replace(group, final_path=str(root / "final"))
            replace(grouped_request, path=path, station_group=group)
        else:
            group = replace(group, final_path=path)
            replace(grouped_request, path=str(root / "staging"), station_group=group)


@pytest.mark.skipif(os.name != "nt", reason="Windows filesystem roots")
@pytest.mark.parametrize("root", [
    "C:\\", r"\\server\share", "\\\\server\\share\\",
    "\\\\?\\C:\\", r"\\?\UNC\server\share",
    "\\\\?\\Volume{12345678-1234-1234-1234-123456789abc}\\",
])
def test_group_rejects_windows_roots(grouped_request, root):
    with pytest.raises(ValueError, match="filesystem root"):
        replace(grouped_request.station_group, final_path=root)


@pytest.mark.parametrize("backend", ["sounddevice", "vkinging"])
@pytest.mark.parametrize("selection", ["partial", "reordered"])
def test_group_requires_every_device_channel_in_physical_order(grouped_request, backend, selection):
    request = grouped_request
    if backend == "vkinging":
        group = request.station_group
        first, second = group.stations
        stations = (
            replace(first, calibration_metadata=ve_metadata((2, 0))),
            replace(second, calibration_metadata=ve_metadata((2,))),
        )
        expected = (7, 2, 0, 5)
        request = replace(
            request, sample_rate=51200, channels=expected,
            device=device_info(physical_channels=expected),
            station_group=replace(group, stations=stations),
        )
        assert request.channels == expected

    channels = (2, 0) if selection == "partial" else request.channels[::-1]
    with pytest.raises(ValueError, match="all input channels in device order"):
        replace(request, channels=channels)



class LegacyRequestId(str):
    pass


class LegacySampleRate(IntEnum):
    INPUT_RATE = 8000


@pytest.mark.parametrize(("field", "value"), [
    ("request_id", " \t "),
    ("request_id", LegacyRequestId("legacy")),
    ("sample_rate", LegacySampleRate.INPUT_RATE),
])
def test_group_admission_requires_completion_identity_and_rate_types(grouped_request, field, value):
    with pytest.raises(ValueError, match=field):
        replace(grouped_request, **{field: value})

    legacy = replace(grouped_request, station_group=None, **{field: value})
    assert getattr(legacy, field) is value

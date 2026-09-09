"""VE request boundary, with legacy compatibility and no device/file I/O."""
import pickle

import pytest

from unit_test.base.recording_process_fakes import device_info as legacy_device
from unit_test.base.ve3668n_fakes import capture_request, device_info, input_config, wav_metadata


@pytest.mark.parametrize("rate", [44100, 48000, 51200])
@pytest.mark.parametrize("streaming", [False, True])
def test_ve_main_freezes_valid_none_snapshot(tmp_path, rate, streaming):
    req = capture_request(tmp_path / "v.wav", sample_rate=rate, streaming=streaming)
    assert req.effective_streaming is streaming
    assert req.device["input_config"]["sample_rate"] == rate
    assert req.channels == (7, 1)
    assert pickle.loads(pickle.dumps(req)) == req
    with pytest.raises(TypeError):
        req.device["input_config"]["sample_rate"] = 44100


@pytest.mark.parametrize("backend", [None, "sounddevice"])
@pytest.mark.parametrize("rate", [1, 100, 96000, 102400])
def test_legacy_backend_and_ranges_are_unchanged(tmp_path, backend, rate):
    device = legacy_device()
    if backend is not None:
        device["backend"] = backend
    req = capture_request(tmp_path / "old.wav", device=device, channels=(2, 0),
                          sample_rate=rate, calibration_metadata=None)
    assert req.device.to_dict() == device
    assert req.sample_rate == rate


@pytest.mark.parametrize("backend", ["other", "", None, True])
def test_unknown_explicit_backend_rejects_legacy_shaped_device(tmp_path, backend):
    with pytest.raises(ValueError, match="backend"):
        capture_request(tmp_path / "bad.wav", device={**legacy_device(), "backend": backend},
                        channels=(0,))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("rate", [None, True, False, 44100.0, "48000", 1, 96000, 102400,
                                  44099, 44101, 47999, 48001, 51199, 51201])
def test_ve_request_rejects_illegal_rate(tmp_path, rate):
    with pytest.raises(ValueError, match="sample_rate"):
        capture_request(tmp_path / "bad.wav", sample_rate=rate)


@pytest.mark.parametrize("change", [
    {"available": False}, {"available": 1}, {"index": 0}, {"hostapi": 0},
    {"sensitivity": 1000}, {"physical_channels": [True, 7]},
    {"physical_channels": [1]}, {"physical_channels": [7, 7]},
    {"max_input_channels": True}, {"model": "OTHER"}, {"machine_id": ""},
])
def test_ve_closed_available_device_contract(tmp_path, change):
    with pytest.raises(ValueError):
        capture_request(tmp_path / "bad.wav", device=device_info(**change))


@pytest.mark.parametrize("change", [
    {"sample_rate": 48000}, {"input_mode": "voltage"}, {"unit": "g"},
    {"range_min": -1}, {"range_max": 1}, {"sensitivity": 1000},
])
def test_ve_frozen_profile_must_match_request_and_fixed_input(tmp_path, change):
    with pytest.raises(ValueError):
        capture_request(tmp_path / "bad.wav", device=device_info(input_config=input_config(**change)))


@pytest.mark.parametrize("channels", [(True,), (8,), (-1,), (7, 7), (), (7.0,), (0,)])
def test_invalid_selected_channels_rejected(tmp_path, channels):
    with pytest.raises(ValueError):
        capture_request(tmp_path / "bad.wav", channels=channels)


@pytest.mark.parametrize("channels", [{7, 1}, {7: "AIN8", 1: "AIN2"}, None, iter([7, 1])])
def test_ve_selected_channels_must_be_an_ordered_sequence(tmp_path, channels):
    with pytest.raises(ValueError, match="ordered sequence"):
        capture_request(tmp_path / "bad.wav", channels=channels)


@pytest.mark.parametrize("monitor", [{"enabled": True}, {"enabled": 1},
                                      {"enabled": "false"}, {"sensitivity": 2}])
def test_ve_rejects_monitoring_before_output_device_validation(tmp_path, monitor):
    with pytest.raises(ValueError, match="monitor"):
        capture_request(tmp_path / "bad.wav", monitor=monitor)


@pytest.mark.parametrize("metadata", [None, {}, {"recorded_channels": []}])
def test_main_requires_ve_v1_snapshot(tmp_path, metadata):
    with pytest.raises(ValueError, match="metadata"):
        capture_request(tmp_path / "bad.wav", calibration_metadata=metadata)


@pytest.mark.parametrize("field,value", [
    ("model", "OTHER"), ("machine_id", "other"), ("input_mode", "voltage"),
    ("unit", "g"), ("range_min", -1), ("range_max", 1), ("sample_rate", 48000),
])
def test_main_acquisition_metadata_must_match_device_and_current_rate(tmp_path, field, value):
    payload = wav_metadata(sample_rate=51200)
    payload["acquisition"]["machine_id"] = "test-machine-1"
    payload["acquisition"][field] = value
    with pytest.raises(ValueError):
        capture_request(tmp_path / "bad.wav", calibration_metadata=payload)


@pytest.mark.parametrize("change", ["physical", "wav_index", "bool_index", "extra", "version"])
def test_main_metadata_has_exact_channel_order_and_closed_schema(tmp_path, change):
    payload = wav_metadata(sample_rate=51200)
    payload["acquisition"]["machine_id"] = "test-machine-1"
    if change == "physical":
        payload["recorded_channels"][0]["physical_input_channel"] = 0
    elif change == "wav_index":
        payload["recorded_channels"].reverse()
    elif change == "bool_index":
        payload["recorded_channels"][0]["wav_channel_index"] = False
    elif change == "extra":
        payload["sensitivity"] = 1000
    else:
        payload["schema_version"] = 2
    with pytest.raises(ValueError):
        capture_request(tmp_path / "bad.wav", calibration_metadata=payload)


def test_measured_provenance_rate_can_differ_and_mutations_cannot_leak(tmp_path):
    device = device_info(input_config=input_config(44100))
    payload = wav_metadata(sample_rate=44100)
    payload["acquisition"]["machine_id"] = device["machine_id"]
    req = capture_request(tmp_path / "v.wav", device=device, sample_rate=44100,
                          calibration_metadata=payload)
    device["input_config"]["sample_rate"] = 48000
    payload["recorded_channels"][0]["v2pa_factor"] = 99
    assert req.calibration_metadata["recorded_channels"][0]["v2pa_factor"] == 10
    assert req.calibration_metadata["recorded_channels"][0]["calibration"]["sample_rate"] == 51200
    assert req.sample_rate == 44100


@pytest.mark.parametrize("rate", [44100, 48000, 51200])
def test_calibration_requires_no_existing_metadata(tmp_path, rate):
    req = capture_request(tmp_path / "c.wav", purpose="calibration", channels=(7,),
                          sample_rate=rate, target_samples=rate * 10, trim_samples=0,
                          streaming=True, calibration_metadata=None)
    assert not req.effective_streaming
    assert req.target_samples == 10 * rate


@pytest.mark.parametrize("change", [{"channels": (7, 1)}, {"target_samples": 511999},
                                      {"target_samples": 512001}, {"trim_samples": 1}])
def test_calibration_exact_ten_seconds_one_channel_no_trim(tmp_path, change):
    values = dict(purpose="calibration", channels=(7,), target_samples=512000,
                  trim_samples=0, calibration_metadata=None)
    values.update(change)
    with pytest.raises(ValueError):
        capture_request(tmp_path / "bad.wav", **values)


def test_progress_event_is_typed_frozen_and_spawn_serializable():
    from base import recording_process_protocol as protocol

    assert hasattr(protocol, "RecordingProgress"), "missing VE progress wire contract"
    progress = protocol.RecordingProgress("ve", 2, 11, 100.5)
    event = protocol.RecordingEvent(2, "ve", "progress", progress)
    assert pickle.loads(pickle.dumps(event)) == event
    with pytest.raises(AttributeError):
        progress.frames = 12
    for payload in (None, {}, 11):
        with pytest.raises(ValueError):
            protocol.RecordingEvent(2, "ve", "progress", payload)
    with pytest.raises(ValueError):
        protocol.RecordingEvent(2, "other", "progress", progress)
    with pytest.raises(ValueError):
        protocol.RecordingEvent(3, "ve", "progress", progress)


@pytest.mark.parametrize("field,value", [
    ("request_id", ""), ("request_id", 1), ("generation", True), ("generation", 0),
    ("generation", 1.0), ("frames", True), ("frames", -1), ("frames", 1.0),
    ("last_frame_at", True), ("last_frame_at", -1), ("last_frame_at", float("nan")),
    ("last_frame_at", float("inf")), ("last_frame_at", None), ("last_frame_at", "100"),
])
def test_progress_scalar_contract(field, value):
    from base import recording_process_protocol as protocol

    assert hasattr(protocol, "RecordingProgress"), "missing VE progress wire contract"
    values = dict(request_id="ve", generation=1, frames=0, last_frame_at=100.0)
    values[field] = value
    with pytest.raises(ValueError):
        protocol.RecordingProgress(**values)


def test_started_allows_native_time_and_legacy_none():
    from base.recording_process_protocol import RecordingEvent

    assert RecordingEvent(1, "legacy", "started").payload is None
    assert RecordingEvent(1, "ve", "started", 100.0).payload == 100.0
    for value in (True, -1, float("inf"), float("nan"), "100"):
        with pytest.raises(ValueError):
            RecordingEvent(1, "ve", "started", value)


def _lifecycle_counts(**changes):
    from base.recording_process_protocol import VeLifecycleCounts

    values = dict(sdk_open=1, task_create=1, task_start=1, task_stop=0,
                  task_clear=0, sdk_close=0)
    values.update(changes)
    return VeLifecycleCounts(**values)


def test_capture_slot_payload_is_frozen_spawn_serializable_and_event_matched():
    from base import recording_process_protocol as protocol

    assert hasattr(protocol, "CaptureSlotReleased"), "missing slot-release wire contract"
    payload = protocol.CaptureSlotReleased(
        request_id="ve", generation=2, target_reached_at=100.5, raw_frames=512,
        adapter_released=True, writer_released=True,
        lifecycle_counts=_lifecycle_counts(),
    )
    event = protocol.RecordingEvent(2, "ve", "capture_slot_released", payload)

    assert pickle.loads(pickle.dumps(event)) == event
    assert payload.target_reached_at == protocol.RecordingProgress(
        "ve", 2, 512, 100.5).last_frame_at
    with pytest.raises(AttributeError):
        payload.raw_frames = 1
    with pytest.raises(ValueError):
        protocol.RecordingEvent(2, "other", "capture_slot_released", payload)
    with pytest.raises(ValueError):
        protocol.RecordingEvent(3, "ve", "capture_slot_released", payload)
    for invalid_payload in (None, {}, _lifecycle_counts()):
        with pytest.raises(ValueError):
            protocol.RecordingEvent(2, "ve", "capture_slot_released", invalid_payload)


@pytest.mark.parametrize("field,value", [
    ("request_id", ""), ("request_id", 1),
    ("generation", True), ("generation", 0), ("generation", 1.0),
    ("target_reached_at", True), ("target_reached_at", -1),
    ("target_reached_at", float("nan")), ("target_reached_at", float("inf")),
    ("raw_frames", True), ("raw_frames", 0), ("raw_frames", -1),
    ("raw_frames", 1.0), ("adapter_released", 1),
    ("adapter_released", None), ("writer_released", 1),
    ("writer_released", None), ("lifecycle_counts", None),
])
def test_capture_slot_scalar_contract(field, value):
    from base.recording_process_protocol import CaptureSlotReleased

    values = dict(request_id="ve", generation=1, target_reached_at=100.0,
                  raw_frames=1, adapter_released=True, writer_released=True,
                  lifecycle_counts=_lifecycle_counts())
    values[field] = value
    with pytest.raises(ValueError):
        CaptureSlotReleased(**values)


def test_capture_slot_revalidates_mutated_nested_lifecycle_counts():
    from base.recording_process_protocol import CaptureSlotReleased, RecordingEvent

    counts = _lifecycle_counts()
    payload = CaptureSlotReleased(
        request_id="ve", generation=1, target_reached_at=100.0, raw_frames=1,
        adapter_released=True, writer_released=True, lifecycle_counts=counts,
    )
    event = RecordingEvent(1, "ve", "capture_slot_released", payload)
    object.__setattr__(counts, "task_stop", -1)

    with pytest.raises(ValueError, match="task_stop"):
        payload.__post_init__()
    with pytest.raises(ValueError, match="task_stop"):
        event.__post_init__()
    with pytest.raises(ValueError, match="task_stop"):
        RecordingEvent(1, "ve", "capture_slot_released", payload)


@pytest.mark.parametrize("field", [
    "sdk_open", "task_create", "task_start", "task_stop", "task_clear", "sdk_close",
])
@pytest.mark.parametrize("value", [True, -1, 1.0, None])
def test_lifecycle_counts_are_nonnegative_strict_integers(field, value):
    from base.recording_process_protocol import VeLifecycleCounts

    with pytest.raises(ValueError, match=field):
        _lifecycle_counts(**{field: value})
    counts = _lifecycle_counts(**{field: 0})
    assert pickle.loads(pickle.dumps(counts)) == counts
    assert isinstance(counts, VeLifecycleCounts)


def test_release_payloads_and_private_event_kinds_are_typed_and_serializable():
    from base import recording_process_protocol as protocol
    from consts.ve3668n_consts import VE_BACKEND

    assert hasattr(protocol, "VeReleaseOutcome"), "missing VE release wire contract"
    signature = (VE_BACKEND, "test-machine-1", (7, 1), 51200)
    outcome = protocol.VeReleaseOutcome(2, signature, ("released",))
    fatal = protocol.WorkerFatal(2, "native_read", "device lost")
    events = (
        protocol.RecordingEvent(2, "", "release_ve"),
        protocol.RecordingEvent(2, "", "ve_released", outcome),
        protocol.RecordingEvent(2, "", "ve_release_failed", outcome),
        protocol.RecordingEvent(2, "", "worker_fatal", fatal),
    )

    assert pickle.loads(pickle.dumps(events)) == events
    for kind in ("ve_released", "ve_release_failed"):
        with pytest.raises(ValueError):
            protocol.RecordingEvent(3, "", kind, outcome)
        with pytest.raises(ValueError):
            protocol.RecordingEvent(2, "", kind)
    with pytest.raises(ValueError):
        protocol.RecordingEvent(3, "", "worker_fatal", fatal)
    with pytest.raises(ValueError):
        protocol.RecordingEvent(2, "", "worker_fatal")
    with pytest.raises(ValueError):
        protocol.RecordingEvent(2, "", "release_ve", outcome)


@pytest.mark.parametrize("request_id", ["ve", "wrong", " "])
@pytest.mark.parametrize("kind", [
    "release_ve", "ve_released", "ve_release_failed", "worker_fatal",
])
def test_generation_lifecycle_events_require_canonical_empty_request_id(kind, request_id):
    from base import recording_process_protocol as protocol
    from consts.ve3668n_consts import VE_BACKEND

    payloads = {
        "release_ve": None,
        "ve_released": protocol.VeReleaseOutcome(
            2, (VE_BACKEND, "test-machine-1", (7, 1), 51200)),
        "ve_release_failed": protocol.VeReleaseOutcome(
            2, (VE_BACKEND, "test-machine-1", (7, 1), 51200), ("failure",)),
        "worker_fatal": protocol.WorkerFatal(2, "native_read", "device lost"),
    }

    assert protocol.RecordingEvent(2, "", kind, payloads[kind]).request_id == ""
    with pytest.raises(ValueError, match="request_id"):
        protocol.RecordingEvent(2, request_id, kind, payloads[kind])


@pytest.mark.parametrize("values", [
    dict(generation=True), dict(generation=0), dict(generation=1.0),
    dict(released_signature=[]),
    dict(released_signature=("vkinging", "machine", (7, 1), 96000)),
    dict(released_signature=("vkinging", "machine", (7, 7), 51200)),
    dict(released_signature=("vkinging", " machine ", (7, 1), 51200)),
    dict(diagnostics=["failed"]), dict(diagnostics=(1,)),
])
def test_release_outcome_scalar_and_signature_contract(values):
    from base.recording_process_protocol import VeReleaseOutcome
    from consts.ve3668n_consts import VE_BACKEND

    valid = dict(generation=1,
                 released_signature=(VE_BACKEND, "machine", (7, 1), 51200),
                 diagnostics=())
    valid.update(values)
    with pytest.raises(ValueError):
        VeReleaseOutcome(**valid)

    assert VeReleaseOutcome(1, None).released_signature is None


@pytest.mark.parametrize("field,value", [
    ("generation", True), ("generation", 0), ("generation", 1.0),
    ("stage", ""), ("stage", 1), ("message", ""), ("message", 1),
])
def test_worker_fatal_scalar_contract(field, value):
    from base.recording_process_protocol import WorkerFatal

    values = dict(generation=1, stage="native_read", message="device lost")
    values[field] = value
    with pytest.raises(ValueError):
        WorkerFatal(**values)

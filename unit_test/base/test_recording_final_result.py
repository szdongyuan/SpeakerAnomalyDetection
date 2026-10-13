"""Parent preparation validates bounded evidence without opening recorded files."""
from dataclasses import replace
import pickle

import numpy as np
import pytest

from base.recording_final_result import PreparedRecordingResult, prepare_main_recording_result
from base.recording_process_protocol import (
    RecordingFinalEnvelope, RecordingQualityEvidence, RecordingRequest, RecordingResult,
)
from consts.recording_result_consts import RECORDING_FINAL_ENVELOPE_VERSION
from unit_test.base.recording_process_fakes import device_info


def request(tmp_path, **changes):
    values = dict(request_id="final", purpose="main", sample_rate=100,
                  target_samples=9, channels=(2, 0), device=device_info(),
                  path=str(tmp_path / "not-created.wav"), streaming=False, trim_samples=2,
                  monitor={}, calibration_metadata=None,
                  validation_thresholds={"enabled": True})
    values.update(changes)
    return RecordingRequest(**values)


def descriptor(req):
    trim = req.trim_samples if req.trim_samples < req.target_samples else 0
    count = req.target_samples - trim
    columns = tuple((np.array([0, count - 1], dtype=np.int64),
                     np.array([-.25, .5], dtype=np.float32) + column)
                    for column in range(len(req.channels)))
    quality = RecordingQualityEvidence(count, "passed", .5, .3, .09, None)
    envelope = RecordingFinalEnvelope(
        RECORDING_FINAL_ENVELOPE_VERSION, count, req.channels, columns, None, quality)
    return RecordingResult(req.request_id, req.purpose, req.path, req.sample_rate,
                           req.channels, req.target_samples, count, False,
                           final_envelope=envelope)


def test_prepares_sparse_channel_order_without_raw_audio(tmp_path):
    req = request(tmp_path)
    result = descriptor(req)
    prepared = prepare_main_recording_result(result, req)
    assert prepared.descriptor is result
    assert prepared.is_prepared_for(req)
    assert prepared.display_error is None
    assert not hasattr(prepared, "multi") and not hasattr(prepared, "mono")
    for column, (time_axis, amplitude) in enumerate(prepared.waveforms):
        np.testing.assert_array_equal(time_axis, np.array([0., .06]))
        np.testing.assert_array_equal(amplitude, result.final_envelope.columns[column][1])
        assert time_axis.dtype == np.float64
        assert amplitude.dtype == np.float32


@pytest.mark.parametrize("changes", [
    {"request_id": "other"}, {"purpose": "calibration"}, {"path": "other.wav"},
    {"sample_rate": 200}, {"sample_rate": 100.0}, {"channels": (0, 2)},
    {"channels": [2, 0]}, {"raw_frames": 8}, {"raw_frames": 9.0},
    {"final_frames": 8}, {"final_frames": 7.0}, {"handles_released": False},
    {"handles_released": 1}, {"metadata_appended": 0},
    {"digest_algorithm": "sha256-float32-le"}, {"sample_digest": "a" * 64},
    {"final_envelope": None},
])
def test_rejects_descriptor_mismatch(tmp_path, changes):
    req = request(tmp_path)
    with pytest.raises(ValueError):
        prepare_main_recording_result(replace(descriptor(req), **changes), req)


@pytest.mark.parametrize("changes", [
    {"version": 2}, {"version": True}, {"frame_count": 8}, {"frame_count": 7.0},
    {"channels": (0, 2)}, {"channels": [2, 0]}, {"quality": None},
    {"columns": ()}, {"columns": []}, {"display_error": ""},
    {"display_error": " "}, {"display_error": "failed"},
])
def test_rejects_malformed_envelope(tmp_path, changes):
    req = request(tmp_path)
    result = descriptor(req)
    envelope = replace(result.final_envelope, **changes)
    with pytest.raises(ValueError):
        prepare_main_recording_result(replace(result, final_envelope=envelope), req)


@pytest.mark.parametrize("indices,values", [
    (np.array([0, 6], dtype=np.int32), np.ones(2, dtype=np.float32)),
    (np.array([0, 6], dtype=np.int64), np.ones(2, dtype=np.float64)),
    (np.array([[0, 6]], dtype=np.int64), np.ones(2, dtype=np.float32)),
    (np.array([0, 6], dtype=np.int64), np.ones((2, 1), dtype=np.float32)),
    (np.array([0, 6], dtype=np.int64), np.ones(1, dtype=np.float32)),
    (np.array([], dtype=np.int64), np.ones(0, dtype=np.float32)),
    (np.array([0, 0, 6], dtype=np.int64), np.ones(3, dtype=np.float32)),
    (np.array([0, 5, 4, 6], dtype=np.int64), np.ones(4, dtype=np.float32)),
    (np.array([1, 6], dtype=np.int64), np.ones(2, dtype=np.float32)),
    (np.array([0, 5], dtype=np.int64), np.ones(2, dtype=np.float32)),
    (np.array([-1, 0, 6], dtype=np.int64), np.ones(3, dtype=np.float32)),
    (np.array([0, 6, 7], dtype=np.int64), np.ones(3, dtype=np.float32)),
    (np.array([0, 6], dtype=np.int64), np.array([0, np.nan], dtype=np.float32)),
    (np.array([0, 6], dtype=np.int64), np.array([0, np.inf], dtype=np.float32)),
    ([0, 6], np.ones(2, dtype=np.float32)),
])
def test_rejects_malformed_columns(tmp_path, indices, values):
    req = request(tmp_path)
    result = descriptor(req)
    envelope = replace(result.final_envelope, columns=((indices, values),) * 2)
    with pytest.raises(ValueError):
        prepare_main_recording_result(replace(result, final_envelope=envelope), req)


@pytest.mark.parametrize("changes", [
    {"count": 8}, {"count": 7.0}, {"status": "failed"}, {"status": "skipped"},
    {"peak": None}, {"peak": np.inf}, {"rms": np.nan}, {"variance": -1.},
    {"peak": True}, {"peak": "1"}, {"rms": 0.}, {"variance": 0.},
    {"voltage_finite": True},
])
def test_rejects_invalid_quality_evidence(tmp_path, changes):
    req = request(tmp_path)
    result = descriptor(req)
    with pytest.raises(ValueError):
        evidence = replace(result.final_envelope.quality, **changes)
        envelope = replace(result.final_envelope, quality=evidence)
        prepare_main_recording_result(replace(result, final_envelope=envelope), req)


def test_explicit_display_error_preserves_success_without_columns(tmp_path):
    req = request(tmp_path)
    result = descriptor(req)
    envelope = replace(result.final_envelope, columns=(), display_error="accumulator failed")
    prepared = prepare_main_recording_result(replace(result, final_envelope=envelope), req)
    assert prepared.is_prepared_for(req)
    assert prepared.waveforms == ()
    assert prepared.display_error == "accumulator failed"


def test_disabled_quality_requires_skipped_but_allows_no_metrics(tmp_path):
    req = request(tmp_path, validation_thresholds={"enabled": False})
    result = descriptor(req)
    with pytest.raises(ValueError):
        prepare_main_recording_result(result, req)
    quality = RecordingQualityEvidence(7, "skipped", None, None, None, None)
    envelope = replace(result.final_envelope, quality=quality)
    assert prepare_main_recording_result(replace(result, final_envelope=envelope), req).is_prepared_for(req)


def test_published_buffers_have_no_writable_alias_and_cannot_be_unfrozen(tmp_path):
    req = request(tmp_path)
    result = descriptor(req)
    prepared = prepare_main_recording_result(result, req)
    result.final_envelope.columns[0][1][:] = 9
    np.testing.assert_array_equal(prepared.waveforms[0][1], [-.25, .5])
    for pair in prepared.waveforms:
        for array in pair:
            assert not array.flags.writeable
            with pytest.raises(ValueError):
                array.setflags(write=True)
    assert prepared.is_prepared_for(req)


def test_preparation_cannot_be_forged_or_transferred(tmp_path):
    req = request(tmp_path)
    prepared = prepare_main_recording_result(descriptor(req), req)
    assert not prepared.is_prepared_for(replace(req))
    assert not replace(prepared).is_prepared_for(req)
    assert not PreparedRecordingResult(prepared.descriptor, prepared.waveforms).is_prepared_for(req)
    restored_req, restored = pickle.loads(pickle.dumps((req, prepared)))
    assert not restored.is_prepared_for(restored_req)
    assert not restored.is_prepared_for(req)


def test_factory_reestablishes_readonly_ownership_after_payload_pickle(tmp_path):
    req = request(tmp_path)
    restored = pickle.loads(pickle.dumps(descriptor(req)))
    prepared = prepare_main_recording_result(restored, req)
    assert prepared.is_prepared_for(req)
    for pair in prepared.waveforms:
        for array in pair:
            with pytest.raises(ValueError):
                array.setflags(write=True)


def test_admission_checks_array_identity_and_shape_without_scanning(tmp_path, monkeypatch):
    req = request(tmp_path)
    prepared = prepare_main_recording_result(descriptor(req), req)
    def forbidden(*args, **kwargs):
        raise AssertionError("admission must not scan samples")
    monkeypatch.setattr(np, "isfinite", forbidden)
    monkeypatch.setattr(np, "diff", forbidden)
    assert prepared.is_prepared_for(req)
    prepared.waveforms[0][0].shape = (1, 2)
    assert not prepared.is_prepared_for(req)


def test_factory_performs_no_file_io(tmp_path, monkeypatch):
    import builtins
    import io
    import os
    import soundfile as sf
    req = request(tmp_path)
    result = descriptor(req)
    def forbidden(*args, **kwargs):
        raise AssertionError("preparation must not access files")
    with monkeypatch.context() as patch:
        for owner, name in ((builtins, "open"), (io, "open"), (os, "open"),
                            (os, "stat"), (sf, "SoundFile"), (sf, "read"), (sf, "info")):
            patch.setattr(owner, name, forbidden)
        prepared = prepare_main_recording_result(result, req)
    assert prepared.is_prepared_for(req)


def test_quality_completion_protocol_rejects_failed_status():
    with pytest.raises(ValueError, match="passed or skipped"):
        RecordingQualityEvidence(7, "failed", .5, .3, .09, None)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("metadata_appended,voltage_finite,valid", [
    (True, True, True), (False, True, False), (True, None, False),
    (True, False, False), (True, 1, False),
])
def test_ve_requires_metadata_and_finite_voltage_even_when_disabled(
        tmp_path, enabled, metadata_appended, voltage_finite, valid):
    from unit_test.base.ve3668n_fakes import capture_request
    req = replace(capture_request(tmp_path / "ve.wav"),
                  validation_thresholds={"enabled": enabled})
    result = descriptor(req)
    evidence = replace(result.final_envelope.quality,
                       status="passed" if enabled else "skipped", voltage_finite=voltage_finite)
    result = replace(result, metadata_appended=metadata_appended,
                     final_envelope=replace(result.final_envelope, quality=evidence))
    if valid:
        assert prepare_main_recording_result(result, req).is_prepared_for(req)
    else:
        with pytest.raises(ValueError):
            prepare_main_recording_result(result, req)


@pytest.mark.parametrize("trim", [0, 9, 10])
def test_effective_trim_uses_existing_target_rule(tmp_path, trim):
    req = request(tmp_path, trim_samples=trim)
    prepared = prepare_main_recording_result(descriptor(req), req)
    assert prepared.descriptor.final_frames == 9
    assert prepared.waveforms[0][0][-1] == .08


def test_one_frame_and_point_limit_are_supported_and_enforced(tmp_path):
    from consts.recording_preview_consts import MAIN_RECORDING_FINAL_MAX_POINTS
    req = request(tmp_path, target_samples=1, trim_samples=0)
    result = descriptor(req)
    envelope = replace(result.final_envelope, columns=tuple(
        (np.array([0], dtype=np.int64), np.array([.5], dtype=np.float32)) for _ in req.channels))
    prepared = prepare_main_recording_result(replace(result, final_envelope=envelope), req)
    assert prepared.is_prepared_for(req)
    req = request(tmp_path, target_samples=10**9, trim_samples=0)
    result = descriptor(req)
    for count in (MAIN_RECORDING_FINAL_MAX_POINTS, MAIN_RECORDING_FINAL_MAX_POINTS + 1):
        indices = np.arange(count, dtype=np.int64)
        indices[-1] = req.target_samples - 1
        values = np.ones(count, dtype=np.float32)
        envelope = replace(result.final_envelope, columns=((indices, values),) * 2)
        result = replace(result, final_envelope=envelope)
        if count == MAIN_RECORDING_FINAL_MAX_POINTS:
            prepared = prepare_main_recording_result(result, req)
            assert sum(array.nbytes for pair in prepared.waveforms for array in pair) == count * 12 * 2
        else:
            with pytest.raises(ValueError, match="point count"):
                prepare_main_recording_result(result, req)


def test_frozen_thresholds_recheck_metrics_and_preserve_gate_diagnostic(tmp_path):
    req = request(tmp_path, validation_thresholds={"enabled": True, "min_peak": .6})
    with pytest.raises(ValueError, match="weak_signal"):
        prepare_main_recording_result(descriptor(req), req)


def test_plain_construction_and_replacement_do_not_change_legacy_positions(tmp_path):
    req = request(tmp_path)
    legacy = RecordingResult(req.request_id, req.purpose, req.path, req.sample_rate,
                             req.channels, 9, 7, False, (), True, (), "sha256", "digest", None)
    assert legacy.digest_algorithm == "sha256"
    assert legacy.sample_digest == "digest"
    assert legacy.final_envelope is None


def test_factory_rejects_non_main_and_non_protocol_objects(tmp_path):
    req = request(tmp_path)
    for invalid in (None, object()):
        with pytest.raises(ValueError):
            prepare_main_recording_result(invalid, req)
        with pytest.raises(ValueError):
            prepare_main_recording_result(descriptor(req), invalid)
    calibration = replace(req, purpose="calibration", channels=(0,))
    with pytest.raises(ValueError):
        prepare_main_recording_result(descriptor(calibration), calibration)

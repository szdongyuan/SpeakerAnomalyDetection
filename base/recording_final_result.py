"""Parent-only preparation of bounded main recording display results; no file I/O."""
from dataclasses import dataclass, field
import math
from numbers import Real

import numpy as np

from base.recording_process_protocol import (
    RecordingFinalEnvelope, RecordingQualityEvidence, RecordingRequest, RecordingResult,
)
from base.recording_settings import validate_recorded_audio_metrics
from consts.recording_preview_consts import MAIN_RECORDING_FINAL_MAX_POINTS
from consts.recording_result_consts import RECORDING_FINAL_ENVELOPE_VERSION
from consts.ve3668n_consts import VE_BACKEND


@dataclass(frozen=True)
class PreparedRecordingResult:
    descriptor: RecordingResult
    waveforms: tuple
    display_error: str | None = None
    _preparation: object = field(default=None, init=False, repr=False, compare=False)

    def is_prepared_for(self, request):
        """Check identity and O(channels) structure, never scan display samples.

        Ordinary construction/replacement and pickle cannot inherit preparation.
        Deliberate private attribute tampering is outside this ownership contract.
        """
        evidence = self._preparation
        if (evidence is None or evidence[0] != id(self) or evidence[1] is not request
                or evidence[2] is not self.descriptor or evidence[3] is not self.waveforms
                or evidence[4] != self.display_error):
            return False
        return all(pair[0] is time_axis and pair[1] is amplitude
                   and time_axis.shape == amplitude.shape == shape
                   and time_axis.dtype == np.float64 and amplitude.dtype == np.float32
                   and time_axis.flags.c_contiguous and amplitude.flags.c_contiguous
                   and not time_axis.flags.writeable and not amplitude.flags.writeable
                   for pair, (time_axis, amplitude, shape) in zip(self.waveforms, evidence[5]))

    def __reduce__(self):
        # Preparation is parent-local evidence, never transferable via pickle.
        return (type(self), (self.descriptor, self.waveforms, self.display_error))


def _validate_descriptor(descriptor, request):
    if not isinstance(request, RecordingRequest) or request.purpose != "main":
        raise ValueError("main result requires a frozen main request")
    if not isinstance(descriptor, RecordingResult):
        raise ValueError("main result requires a RecordingResult descriptor")
    trim = request.trim_samples if request.trim_samples < request.target_samples else 0
    if (descriptor.request_id != request.request_id or descriptor.path != request.path
            or descriptor.purpose != request.purpose
            or type(descriptor.sample_rate) is not int or descriptor.sample_rate != request.sample_rate
            or type(descriptor.channels) is not tuple or descriptor.channels != request.channels
            or any(type(channel) is not int for channel in descriptor.channels)
            or type(descriptor.raw_frames) is not int or descriptor.raw_frames != request.target_samples
            or type(descriptor.final_frames) is not int
            or descriptor.final_frames != request.target_samples - trim
            or descriptor.handles_released is not True
            or type(descriptor.metadata_appended) is not bool):
        raise ValueError("final envelope descriptor differs from frozen request")
    if descriptor.digest_algorithm is not None or descriptor.sample_digest is not None:
        raise ValueError("main final envelope must not carry sample digest evidence")
    if request.device.get("backend") == VE_BACKEND and not descriptor.metadata_appended:
        raise ValueError("VE final result requires appended calibration metadata")


def _validate_quality(quality, request, count):
    if (not isinstance(quality, RecordingQualityEvidence) or type(quality.count) is not int
            or quality.count != count):
        raise ValueError("final quality count must match saved frames")
    thresholds = request.validation_thresholds.to_dict()
    expected_status = "passed" if thresholds.get("enabled", True) else "skipped"
    if quality.status != expected_status:
        raise ValueError("final quality status differs from frozen validation settings")
    voltage_finite = True if request.device.get("backend") == VE_BACKEND else None
    if quality.voltage_finite is not voltage_finite:
        raise ValueError("final quality voltage finite evidence differs from backend requirement")
    metrics = (quality.peak, quality.rms, quality.variance)
    if expected_status == "skipped" and all(metric is None for metric in metrics):
        return
    if any(isinstance(metric, (bool, np.bool_)) or not isinstance(metric, Real)
           or not math.isfinite(metric) or metric < 0 for metric in metrics):
        raise ValueError("final quality metrics must be finite nonnegative numbers")
    ok, reason, detail = validate_recorded_audio_metrics(count, *metrics, thresholds)
    if not ok:
        raise ValueError(f"final quality gate rejected evidence: {reason} {detail}")


def _validate_envelope(envelope, descriptor, request):
    if (not isinstance(envelope, RecordingFinalEnvelope)
            or type(envelope.version) is not int or envelope.version != RECORDING_FINAL_ENVELOPE_VERSION
            or type(envelope.frame_count) is not int or envelope.frame_count != descriptor.final_frames
            or type(envelope.channels) is not tuple or envelope.channels != request.channels
            or any(type(channel) is not int for channel in envelope.channels)
            or type(envelope.columns) is not tuple):
        raise ValueError("invalid final envelope version, frames or channels")
    _validate_quality(envelope.quality, request, descriptor.final_frames)
    if envelope.display_error is not None:
        if (not isinstance(envelope.display_error, str) or not envelope.display_error.strip()
                or envelope.columns):
            raise ValueError("display failure requires a nonempty error and no columns")
        return
    if len(envelope.columns) != len(request.channels):
        raise ValueError("final envelope requires one column per requested channel")
    for pair in envelope.columns:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("final envelope column must be an index/amplitude pair")
        indices, values = pair
        if (not isinstance(indices, np.ndarray) or not isinstance(values, np.ndarray)
                or indices.ndim != 1 or values.ndim != 1
                or indices.dtype != np.int64 or values.dtype != np.float32
                or indices.shape != values.shape
                or not 1 <= len(indices) <= MAIN_RECORDING_FINAL_MAX_POINTS):
            raise ValueError("invalid final envelope column shape, dtype or point count")
        if (indices[0] != 0 or indices[-1] != descriptor.final_frames - 1
                or np.any(indices[1:] <= indices[:-1]) or not np.isfinite(values).all()):
            raise ValueError("final envelope indices or amplitudes are invalid")


def _immutable_array(array):
    # Bytes owns the bounded copy; even setflags(write=True) cannot expose a
    # writable alias, including after incoming arrays have crossed pickle.
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype)


def prepare_main_recording_result(descriptor, request):
    """Validate child evidence and publish request-bound, immutable display data.

    Work and allocations are bounded by channel count times the display limit.
    Malformed payloads raise ValueError; explicit display failures remain valid
    saved recordings with empty waveforms. This module never opens the WAV.
    """
    _validate_descriptor(descriptor, request)
    envelope = descriptor.final_envelope
    _validate_envelope(envelope, descriptor, request)
    waveforms = tuple((_immutable_array(indices.astype(np.float64) / request.sample_rate),
                       _immutable_array(values))
                      for indices, values in envelope.columns)
    result = PreparedRecordingResult(descriptor, waveforms, envelope.display_error)
    object.__setattr__(result, "_preparation", (
        id(result), request, descriptor, waveforms, envelope.display_error,
        tuple((time_axis, values, time_axis.shape) for time_axis, values in waveforms)))
    return result

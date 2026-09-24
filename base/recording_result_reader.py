"""Parent-only chunked WAV reader. Its outcome never contains mapped arrays."""
from dataclasses import dataclass, field
import hashlib
import threading
import time

from base.log_manager import LogManager

import numpy as np
import soundfile as sf

from base.recording_process_protocol import RecordingResult
from base.recording_waveform_preparation import prepare_waveform_display_data
from consts.recording_preview_consts import MAIN_RECORDING_FINAL_MAX_POINTS
from base.recording_settings import validate_recorded_audio
from base.wav_calibration_metadata import (
    WavCalibrationMetadataReadStatus, inspect_wav_calibration_metadata,
    normalize_wav_calibration_metadata,
)
from consts.recording_result_consts import RECORDING_SAMPLE_DIGEST_ALGORITHM
from consts.wav_format_consts import WAV_PCM24_SUBTYPE
from consts.ve3668n_consts import VE_BACKEND


@dataclass(frozen=True)
class RecordingAudio:
    descriptor: RecordingResult
    multi: np.ndarray
    mono: np.ndarray
    waveforms: tuple = ()
    _preparation: object = field(default=None, init=False, repr=False, compare=False)

    def is_prepared_for(self, request):
        """Constant-size provenance checks; never scan the retained recording.

        Only ResultReader attaches this evidence after request validation. Identity
        binding means dataclass replacement/direct construction cannot inherit it.
        Published arrays wrap read-only buffers, with no writable array aliases
        escaping the reader. Deliberate private-object tampering is outside this
        in-process ownership contract; ordinary consumers must copy to mutate.
        """
        evidence = self._preparation
        return (evidence is not None and request is not None and request.purpose == "main"
                and evidence[0] is request
                and evidence[1] is self.descriptor and evidence[2] is self.multi
                and evidence[3] is self.mono and evidence[4] is self.waveforms
                and self.multi.dtype == np.float32 and self.mono.dtype == np.float32
                and self.multi.flags.c_contiguous and self.mono.flags.c_contiguous
                and self.multi.shape == (self.descriptor.final_frames, len(request.channels))
                and self.mono.shape == (self.descriptor.final_frames,)
                and len(self.waveforms) == len(request.channels)
                and all(time_axis.ndim == amplitude.ndim == 1
                        and time_axis.dtype == np.float64 and amplitude.dtype == np.float32
                        and time_axis.shape == amplitude.shape
                        and 0 < len(time_axis) <= MAIN_RECORDING_FINAL_MAX_POINTS
                        and time_axis.flags.c_contiguous and amplitude.flags.c_contiguous
                        for time_axis, amplitude in self.waveforms)
                and all(not array.flags.writeable for array in (
                    self.multi, self.mono, *(a for pair in self.waveforms for a in pair))))


def _freeze_owned(array):
    """Publish a no-copy read-only buffer view; owner never leaves the reader."""
    array.setflags(write=False)
    return np.frombuffer(memoryview(array).toreadonly(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class ReadOutcome:
    audio: RecordingAudio | None
    error: str | None
    handles_released: bool


class ResultReader:
    def __init__(self, descriptor, completed, *, request=None, block_frames=65536, opener=sf.SoundFile):
        """Production supplies the frozen request and sample evidence.

        Only legacy direct callers with no evidence may omit request context.
        File access and request validation remain on the reader thread.
        """
        if type(block_frames) is not int or block_frames <= 0:
            raise ValueError("reader block_frames must be a positive integer")
        self._logger = LogManager.set_log_handler("core")
        self.descriptor = descriptor
        self.request = request
        self.cancelled = threading.Event()
        self.exited = threading.Event()
        self._completed = completed
        self._block_frames = block_frames
        self._opener = opener
        self._retained = None
        self._metadata_retained = ()
        self._metadata_handles_released = True
        self.thread = threading.Thread(target=self._run, name=f"recording-reader-{descriptor.request_id}",
                                       daemon=True)

    def start(self):
        self.thread.start()

    def cancel(self):
        self.cancelled.set()

    def _validate_request(self):
        descriptor, request = self.descriptor, self.request
        if request is None:
            if descriptor.sample_digest is not None or descriptor.digest_algorithm is not None:
                raise ValueError("sample evidence requires frozen request context")
            return
        trim = request.trim_samples if request.purpose == "main" else 0
        if trim >= request.target_samples:
            trim = 0
        if (descriptor.request_id != request.request_id or descriptor.path != request.path
                or descriptor.purpose != request.purpose or descriptor.sample_rate != request.sample_rate
                or descriptor.channels != request.channels or descriptor.raw_frames != request.target_samples
                or descriptor.final_frames != request.target_samples - trim
                or descriptor.handles_released is not True):
            raise ValueError("final WAV descriptor differs from frozen request")
        if (descriptor.digest_algorithm != RECORDING_SAMPLE_DIGEST_ALGORITHM
                or type(descriptor.sample_digest) is not str or len(descriptor.sample_digest) != 64
                or any(char not in "0123456789abcdef" for char in descriptor.sample_digest)):
            raise ValueError("final WAV requires valid capture sample digest evidence")

    def _validate_metadata(self):
        request = self.request
        if request is None or request.purpose != "main":
            return
        required = request.device.get("backend") == VE_BACKEND
        if not required and not self.descriptor.metadata_appended:
            return
        diagnostic = inspect_wav_calibration_metadata(self.descriptor.path)
        self._metadata_retained = diagnostic.retained_handles
        self._metadata_handles_released = diagnostic.handles_released and not diagnostic.retained_handles
        if (not diagnostic.handles_released or diagnostic.retained_handles
                or diagnostic.primary_error is not None):
            raise ValueError(diagnostic.primary_error or "; ".join(diagnostic.close_errors)
                             or "final WAV metadata reader handle was not released")
        expected = normalize_wav_calibration_metadata(
            request.calibration_metadata.to_dict() if request.calibration_metadata is not None else None)
        if (not self.descriptor.metadata_appended
                or diagnostic.status is not WavCalibrationMetadataReadStatus.VALID
                or expected is None or diagnostic.metadata != expected
                or (required and diagnostic.declared_backend != VE_BACKEND)):
            raise ValueError("final WAV calibration metadata differs from frozen request")

    def _run(self):
        started = time.monotonic()
        source = None
        audio = error = None
        released = True
        try:
            descriptor = self.descriptor
            self._validate_request()
            source = self._opener(descriptor.path, mode="r")
            if (source.subtype != WAV_PCM24_SUBTYPE or source.samplerate != descriptor.sample_rate
                    or source.channels != len(descriptor.channels) or len(source) != descriptor.final_frames):
                raise ValueError("final WAV frame/channel/rate/PCM24 contract mismatch")
            multi = np.empty((descriptor.final_frames, len(descriptor.channels)), dtype=np.float32)
            offset = 0
            digest = hashlib.sha256() if self.request is not None else None
            while offset < len(multi) and not self.cancelled.is_set():
                block = source.read(min(self._block_frames, len(multi) - offset),
                                    dtype="float32", always_2d=True)
                if not len(block):
                    raise ValueError("final WAV ended before expected frame count")
                if block.shape != (len(block), len(descriptor.channels)) or len(block) > len(multi) - offset:
                    raise ValueError("final WAV read block violates frame/channel contract")
                if self.request is not None and self.request.device.get("backend") == VE_BACKEND:
                    if not np.isfinite(block).all():
                        raise ValueError("final WAV contains non-finite voltage samples")
                if digest is not None:
                    digest.update(block.astype("<f4", copy=False).tobytes(order="C"))
                multi[offset:offset + len(block)] = block
                offset += len(block)
            if not self.cancelled.is_set():
                if digest is not None and digest.hexdigest() != descriptor.sample_digest:
                    raise ValueError("final WAV sample digest differs from captured samples")
                self._validate_metadata()
                read_finished = time.monotonic()
                mono = multi.mean(axis=1, dtype=np.float32)
                mono_finished = time.monotonic()
                if (self.request is not None and self.request.purpose == "main"
                        and self.request.device.get("backend") == VE_BACKEND
                        and not np.isfinite(mono).all()):
                    # Finite inputs can overflow float32 channel reduction. This
                    # replaces the main GUI's former full-length mono check.
                    raise ValueError("final WAV contains non-finite mono voltage samples")
                if self.request is not None and self.request.purpose == "main":
                    thresholds = self.request.validation_thresholds.to_dict()
                    if thresholds.get("enabled", True):
                        # Preserve division-before-channel-mean rounding for VE.
                        quality_audio = (multi / self.request.device["input_config"]["range_max"]
                                         if self.request.device.get("backend") == VE_BACKEND else mono)
                        ok, reason, detail = validate_recorded_audio(quality_audio, thresholds)
                        if not ok:
                            raise ValueError(f"{reason} {detail}")
                quality_finished = time.monotonic()
                self._logger.info("Recording timing request=%s process=parent stage=read_validation seconds=%.6f includes=read_digest_metadata_quality",
                    descriptor.request_id, read_finished - started + quality_finished - mono_finished)
                if not self.cancelled.is_set():
                    waveforms = ()
                    if self.request is not None and self.request.purpose == "main":
                        waveforms = tuple(tuple(_freeze_owned(array) for array in
                            prepare_waveform_display_data(multi[:, column], descriptor.sample_rate,
                                max_points=MAIN_RECORDING_FINAL_MAX_POINTS))
                            for column in range(multi.shape[1]))
                    if self.request is not None and self.request.purpose == "main":
                        audio = RecordingAudio(descriptor, _freeze_owned(multi), _freeze_owned(mono), waveforms)
                        object.__setattr__(audio, "_preparation", (
                            self.request, descriptor, audio.multi, audio.mono, audio.waveforms))
                    else:
                        audio = RecordingAudio(descriptor, multi, mono)
                    self._logger.info("Recording timing request=%s process=parent stage=mono_display seconds=%.6f display_applicable=%s",
                        descriptor.request_id, mono_finished - read_finished + time.monotonic() - quality_finished,
                        self.request is not None and self.request.purpose == "main")
        except Exception as exc:
            # File/library boundary includes injected/custom readers. Normalize
            # once, retain ownership through close, and never deliver partial data.
            error = f"result read failed for {self.descriptor.path}: {exc}"
        finally:
            if not self._metadata_handles_released:
                released = False
            if source is not None:
                try:
                    source.close()
                except Exception as exc:
                    error = f"result reader close failed for {self.descriptor.path}: {exc}"
                    released = False
                    self._retained = source
            if released:
                self.exited.set()
            self._logger.info("Recording timing request=%s process=parent stage=reader_complete seconds=%.6f handles_released=%s cancelled=%s error=%s",
                self.descriptor.request_id, time.monotonic() - started, released, self.cancelled.is_set(), error)
            self._completed(ReadOutcome(audio if released else None, error, released))

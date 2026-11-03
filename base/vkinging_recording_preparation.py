"""One request-owned file reader; no SDK, Qt callbacks or service audio events."""
from dataclasses import dataclass
import os
from pathlib import Path
from queue import Empty, Queue
import threading
import uuid

import numpy as np

from base.wav_calibration_metadata import inspect_wav_calibration_metadata, normalize_vkinging_metadata
from base.vkinging_service import _exception_text


def format_recording_error(error):
    """Keep external exception formatting nonempty and safe for UI delivery."""
    text = _exception_text(error)
    if not text.strip():
        text = type(error).__name__
    notes = [note for note in getattr(error, '__notes__', ()) if isinstance(note, str)]
    return '; '.join((text, *notes))


@dataclass(frozen=True)
class PreparedRecording:
    request_id: str
    samples: object = None
    mono: object = None
    inspection: object = None
    error: str | None = None
    resources_released: bool = True


class VkingingRecordingPreparation:
    """Bounded single-result handoff, separate from the unique native owner.

    Closing revokes delivery without joining or touching files held by this
    reader. The daemon keeps only its own request and closes its own handles.
    """
    def __init__(self, request, descriptor):
        self.request = request
        self._result = Queue(maxsize=1)
        self._cancelled = threading.Event()
        self._thread = threading.Thread(target=self._read, args=(descriptor,),
                                        name='vkinging-file-preparation', daemon=True)
        self._thread.start()

    def close(self):
        self._cancelled.set()

    def poll(self):
        try:
            return self._result.get_nowait()
        except Empty:
            return None

    def _read(self, descriptor):
        request = self.request
        handle = None
        released = True
        error = None
        samples = mono = inspection = None
        # soundfile/provenance parsing are external file-format boundaries.
        # Always attempt reader close, preserve primary and close diagnostics,
        # and never publish prepared audio while file release is uncertain.
        try:
            import soundfile as sf
            if not descriptor.get('resources_released'):
                raise ValueError('capture resources are not released')
            if (Path(descriptor['path']).absolute() != Path(request.output_path).absolute()
                    or descriptor['frames'] != request.target_samples
                    or descriptor['sample_rate'] != request.sample_rate
                    or descriptor['bit_depth'] != request.bit_depth
                    or tuple(descriptor['physical_channels']) != request.channels):
                raise ValueError('recording descriptor does not match the frozen request')
            if self._cancelled.is_set():
                return
            inspection = inspect_wav_calibration_metadata(request.output_path)
            if (inspection.source != 'vkinging'
                    or inspection.status not in ('vkinging_valid', 'vkinging_uncalibrated')
                    or inspection.metadata != normalize_vkinging_metadata(descriptor['metadata'])):
                raise ValueError(f'WAV inspection failed: {inspection.reason or inspection.status}')
            released = False
            try:
                handle = sf.SoundFile(request.output_path, 'r')
            except sf.LibsndfileError:
                # This read-only path open raises LibsndfileError when
                # sf_open returns NULL, so no handle was acquired. Unexpected
                # constructor failures cannot provide that release proof.
                released = True
                raise
            if (handle.frames != request.target_samples or handle.samplerate != request.sample_rate
                    or handle.channels != len(request.channels)
                    or handle.subtype != ('FLOAT' if request.bit_depth == 32 else 'DOUBLE')):
                raise ValueError('prepared WAV frames or floating format mismatch')
            samples = handle.read(dtype='float32' if request.bit_depth == 32 else 'float64', always_2d=True)
            if samples.shape != (request.target_samples, len(request.channels)) or not np.isfinite(samples).all():
                raise ValueError('prepared WAV contains invalid samples')
            mono = samples.mean(axis=1)
        except Exception as exc:
            error = format_recording_error(exc)
            if not released and handle is None:
                error += f'; WAV reader open/release uncertain; owned path: {request.output_path}'
        finally:
            if handle is not None:
                try:
                    handle.close()
                except Exception as exc:
                    released = False
                    diagnostic = (f'WAV reader close: {format_recording_error(exc)}; '
                                  f'owned path: {request.output_path}')
                    error = f'{error}; {diagnostic}' if error is not None else diagnostic
                else:
                    released = True
            if not self._cancelled.is_set():
                self._result.put_nowait(PreparedRecording(request.request_id,
                    samples=samples if error is None else None, mono=mono if error is None else None,
                    inspection=inspection, error=error, resources_released=released))


def publish_replay(attempt_path, canonical_path):
    """Explicit replay replacement with an exclusive, request-owned backup."""
    backup = str(Path(canonical_path).with_name(Path(canonical_path).name + '.' + uuid.uuid4().hex + '.backup'))
    had_prior = os.path.exists(canonical_path)
    if had_prior:
        os.link(canonical_path, backup)  # never overwrite a user's backup
    try:
        os.replace(attempt_path, canonical_path)
    except OSError as primary:
        if had_prior:
            try:
                os.unlink(backup)  # canonical was not replaced; this link is ours
            except OSError as cleanup:
                primary.add_note(f'replay backup cleanup failed: {format_recording_error(cleanup)}; '
                                 f'owned backup: {backup}')
        raise
    return backup if had_prior else None


def restore_replay(attempt_path, canonical_path, backup_path):
    """Keep the valid new recording before restoring a previous canonical file."""
    os.link(canonical_path, attempt_path)
    if backup_path:
        os.replace(backup_path, canonical_path)
    else:
        os.unlink(canonical_path)
    return attempt_path

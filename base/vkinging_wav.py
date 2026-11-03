"""Task-owned floating voltage WAVs; no publication before validation/closure."""
import importlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile
from contextlib import contextmanager

import numpy as np

from base.vkinging_input import validate_bit_depth, validate_physical_channels, validate_sample_rate
from base.wav_calibration_metadata import (
    _build_chunk, _build_info_comment_list_chunk, inspect_wav_calibration_metadata,
    normalize_vkinging_metadata, scan_riff_layout,
)
from consts.audio_consts import MAX_RIFF_SIZE
from consts.vkinging_consts import VKINGING_WAV_CHUNK_ID, VKINGING_WAV_MARKER


def _soundfile():
    # Missing dependency is an explicit error; never label floats as integer PCM.
    return importlib.import_module('soundfile')


def _metadata_chunks(metadata):
    normalized = normalize_vkinging_metadata(metadata)
    comment = 'mic_calibration=' + json.dumps(normalized, ensure_ascii=True, allow_nan=False, separators=(',', ':'))
    return (_build_chunk(VKINGING_WAV_CHUNK_ID, VKINGING_WAV_MARKER),
            _build_info_comment_list_chunk(comment), normalized)


def preflight_vkinging_wav_size(*, sample_rate, bit_depth, channels, target_frames, metadata):
    """Exact final file bytes using the installed writer's serialized empty header."""
    validate_sample_rate(sample_rate)
    validate_bit_depth(bit_depth)
    if type(channels) is not int or channels < 1 or channels > 65535:
        raise ValueError('channels must fit the WAV channel field')
    if type(target_frames) is not int or target_frames < 1:
        raise ValueError('target_frames must be positive')
    marker, calibration, normalized = _metadata_chunks(metadata)
    acquisition = normalized['acquisition']
    if (len(normalized['recorded_channels']) != channels or acquisition['sample_rate'] != sample_rate
            or acquisition['bit_depth'] != bit_depth):
        raise ValueError('metadata differs from requested WAV format')
    buffer = io.BytesIO()
    with _soundfile().SoundFile(buffer, mode='w', samplerate=sample_rate, channels=channels,
                               format='WAV', subtype='FLOAT' if bit_depth == 32 else 'DOUBLE'):
        pass
    total = len(buffer.getvalue()) + target_frames * channels * (bit_depth // 8) + len(marker) + len(calibration)
    if total - 8 > MAX_RIFF_SIZE:
        raise ValueError('RIFF size exceeds 32-bit limit; RF64 is not supported')
    return total


class VkingingVoltageWriter:
    """Single-owner streaming writer with retained, uniquely owned partials.

    close releases the capture handle; finalize assembles a separate partial and
    validates it; commit atomically creates the destination without overwriting.
    abort closes only: failed/cancelled artifacts remain available for diagnosis.
    A failed close is uncertain even if the third-party handle reports closed.
    """
    def __init__(self, output_path, *, sample_rate, bit_depth, physical_channels,
                 target_frames, metadata):
        self.output_path = Path(output_path).absolute()
        self.physical_channels = validate_physical_channels(physical_channels)
        self.sample_rate, self.bit_depth = sample_rate, bit_depth
        self.target_frames = target_frames
        self.metadata = normalize_vkinging_metadata(metadata)
        if tuple(ch['physical_channel_index'] for ch in self.metadata['recorded_channels']) != self.physical_channels:
            raise ValueError('metadata physical channels differ from requested channels')
        self.expected_size = preflight_vkinging_wav_size(
            sample_rate=sample_rate, bit_depth=bit_depth, channels=len(self.physical_channels),
            target_frames=target_frames, metadata=self.metadata)
        self.frames_written = 0
        self._handle = None
        self._closed = True
        self._uncertain = False
        self._failed = False
        self._prepared = None
        self._committed = False
        self._owned_paths = []
        self._open_streams = []
        self.raw_path = self._new_partial('raw')
        # Third-party open is an external boundary: a failed construction may
        # still leave a partial, whose path is attached to the primary error.
        try:
            self._handle = _soundfile().SoundFile(
                self.raw_path, mode='w', samplerate=sample_rate,
                channels=len(self.physical_channels), format='WAV',
                subtype='FLOAT' if bit_depth == 32 else 'DOUBLE')
            self._closed = False
        except Exception as exc:
            exc.add_note(f'owned partial: {self.raw_path}')
            raise

    def _new_partial(self, phase):
        fd, path = tempfile.mkstemp(prefix=f'.{self.output_path.name}.{phase}.',
                                    suffix='.partial.wav', dir=self.output_path.parent)
        self._owned_paths.append(path)
        try:
            os.close(fd)
        except OSError as exc:
            self._uncertain = True
            exc.add_note(f'temporary fd close uncertain: fd={fd}; owned partial: {path}')
            raise
        return path

    @property
    def owned_paths(self):
        return tuple(self._owned_paths)

    @property
    def resources_released(self):
        return self._closed and not self._uncertain and not self._open_streams

    @contextmanager
    def _owned_stream(self, stream):
        """Close every assembly/read handle without masking a primary error."""
        import sys
        self._open_streams.append(stream)
        try:
            yield stream
        finally:
            primary = sys.exception()
            try:
                stream.close()
            except Exception as exc:
                # File/soundfile close is a third-party I/O boundary. A failed
                # close leaves ownership uncertain; preserve primary failure.
                self._uncertain = True
                if primary is not None:
                    primary.add_note(f'WAV close failed: {exc}; owned paths: {self.owned_paths}')
                else:
                    raise
            else:
                self._open_streams.remove(stream)

    def write_chunk(self, samples):
        if self._closed or self._failed:
            raise RuntimeError('writer is closed or failed')
        values = np.asarray(samples)
        if (values.ndim != 2 or values.shape[1] != len(self.physical_channels)
                or values.dtype.kind not in 'fi' or not np.isfinite(values).all()):
            raise ValueError('samples must be finite frames by channels')
        if self.frames_written + len(values) > self.target_frames:
            raise ValueError('frames exceed target_frames')
        # Reject values that cannot be represented in the requested float type.
        if np.any(np.abs(values) > np.finfo('float32' if self.bit_depth == 32 else 'float64').max):
            raise ValueError('samples exceed floating point range')
        try:
            self._handle.write(values)
        except Exception as exc:
            self._failed = True  # Partial writes cannot prove an exact frame count.
            exc.add_note(f'owned partial: {self.raw_path}')
            raise
        self.frames_written += len(values)

    def close(self):
        if self._closed:
            return self.descriptor(self.raw_path)
        try:
            self._handle.close()
        except Exception as exc:
            self._uncertain = self._failed = True
            exc.add_note(f'WAV close uncertain; owned partial: {self.raw_path}')
            raise
        self._closed = True
        return self.descriptor(self.raw_path)

    def abort(self):
        return self.close()

    def descriptor(self, path):
        return {'path': str(path), 'frames': self.frames_written, 'sample_rate': self.sample_rate,
                'bit_depth': self.bit_depth, 'physical_channels': list(self.physical_channels),
                'resources_released': self.resources_released, 'owned_paths': list(self.owned_paths),
                'metadata': normalize_vkinging_metadata(self.metadata)}

    def _validate_audio(self, path, *, provenance):
        with self._owned_stream(_soundfile().SoundFile(path, mode='r')) as info:
            if (info.frames != self.target_frames or info.samplerate != self.sample_rate
                    or info.channels != len(self.physical_channels)
                    or info.subtype != ('FLOAT' if self.bit_depth == 32 else 'DOUBLE') or info.format != 'WAV'):
                raise ValueError('WAV frames or format mismatch')
        if provenance:
            inspection = inspect_wav_calibration_metadata(path)
            if (inspection.status not in ('vkinging_valid', 'vkinging_uncalibrated')
                    or inspection.metadata != self.metadata):
                raise ValueError(f'WAV metadata validation failed: {inspection.reason}')

    def finalize(self, *, checkpoint=None):
        self.close()
        if self._failed or not self.resources_released:
            raise RuntimeError('writer failure or uncertain resource release')
        if self.frames_written != self.target_frames:
            raise ValueError('recorded frames differ from target_frames')
        if self._prepared is not None:
            return self.descriptor(self._prepared)
        self._validate_audio(self.raw_path, provenance=False)
        marker, calibration, _ = _metadata_chunks(self.metadata)
        prepared = self._new_partial('assembled')
        # Both context managers attempt closure; any assembly/close failure
        # prevents validation and publication. Handles whose close failed are
        # conservatively uncertain and their paths are always retained.
        with self._owned_stream(open(self.raw_path, 'rb')) as source:
            layout = scan_riff_layout(source)
            raw_size = os.fstat(source.fileno()).st_size
            total = raw_size + len(marker) + len(calibration)
            if total != self.expected_size or total - 8 > MAX_RIFF_SIZE:
                raise ValueError('RIFF size differs from preflight')
            with self._owned_stream(open(prepared, 'wb')) as destination:
                destination.write(b'RIFF' + struct.pack('<I', total - 8) + b'WAVE')
                for name, offset, size in layout:
                    if checkpoint:
                        checkpoint()
                    if name == b'data':
                        destination.write(marker)
                    source.seek(offset - 8)
                    remaining = 8 + size + size % 2
                    while remaining:
                        if checkpoint:
                            checkpoint()
                        block = source.read(min(65536, remaining))
                        if not block:
                            raise ValueError('truncated WAV during assembly')
                        destination.write(block)
                        remaining -= len(block)
                destination.write(calibration)
        if self._uncertain:
            raise RuntimeError('assembly close was not confirmed')
        self._validate_audio(prepared, provenance=True)
        self._prepared = prepared
        return self.descriptor(prepared)

    def commit(self, *, checkpoint=None):
        if self._prepared is None or self._failed or not self.resources_released:
            raise RuntimeError('finalize and resource release required before commit')
        if self._committed:
            raise RuntimeError('recording already committed')
        if checkpoint:
            checkpoint()
        self._validate_audio(self._prepared, provenance=True)
        if checkpoint:
            checkpoint()
        result = self.descriptor(self.output_path)
        # Hard-link creation is atomic and fails if the target exists on Windows
        # and POSIX. Keep the owned partial instead of risking post-publish
        # cleanup errors being misreported as a failed publication.
        os.link(self._prepared, self.output_path)
        self._committed = True
        return result

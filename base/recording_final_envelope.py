"""Bounded final waveform summaries of already quantized, frames-first samples."""

import numpy as np

from consts.recording_preview_consts import MAIN_RECORDING_FINAL_MAX_POINTS


class FinalEnvelopeAccumulator:
    """Match the batch display buckets without retaining long recordings.

    Inputs must be finite float32 arrays, already trimmed and PCM24 quantized
    by the caller. Appends are consecutive; empty blocks are allowed. Persistent
    storage is O(channel_count * max_points), and each append uses only block
    sized temporaries. No file, Qt, or transport state is owned here.
    """

    def __init__(
        self, total_frames, channel_count, max_points=MAIN_RECORDING_FINAL_MAX_POINTS
    ):
        for name, value, minimum in (
            ("total_frames", total_frames, 1),
            ("channel_count", channel_count, 1),
            ("max_points", max_points, 4),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        total_frames = int(total_frames)
        channel_count = int(channel_count)
        max_points = int(max_points)
        self._total_frames = total_frames
        self._channel_count = channel_count
        self._frame_count = 0
        self._samples = None
        if total_frames <= max_points:
            self._samples = np.empty((total_frames, channel_count), dtype=np.float32)
            return

        bucket_budget = (max_points - 2) // 2
        self._bucket_width = (total_frames + bucket_budget - 1) // bucket_budget
        bucket_count = (total_frames + self._bucket_width - 1) // self._bucket_width
        shape = (bucket_count, channel_count)
        self._min_values = np.full(shape, np.inf, dtype=np.float32)
        self._max_values = np.full(shape, -np.inf, dtype=np.float32)
        self._min_indices = np.zeros(shape, dtype=np.int64)
        self._max_indices = np.zeros(shape, dtype=np.int64)
        self._first = np.empty(channel_count, dtype=np.float32)
        self._last = np.empty(channel_count, dtype=np.float32)

    def append(self, block):
        """Consume a consecutive block without modifying or retaining it."""
        if (
            not isinstance(block, np.ndarray)
            or block.dtype != np.float32
            or block.ndim != 2
            or block.shape[1] != self._channel_count
        ):
            raise ValueError("block must be a frames-first float32 array with matching channels")
        row_count = block.shape[0]
        if self._frame_count + row_count > self._total_frames:
            raise ValueError("block exceeds total_frames")
        if not np.isfinite(block).all():
            raise ValueError("block samples must be finite")
        if not row_count:
            return
        start = self._frame_count
        end = start + row_count
        if self._samples is not None:
            self._samples[start:end] = block
        else:
            if start == 0:
                self._first[:] = block[0]
            self._last[:] = block[-1]
            offset = 0
            # Only the two boundary fragments need merging; all complete buckets
            # in between are reduced together by NumPy.
            if start % self._bucket_width:
                offset = min(row_count, self._bucket_width - start % self._bucket_width)
                self._merge_fragment(block[:offset], start)
            full_count = (row_count - offset) // self._bucket_width
            if full_count:
                stop = offset + full_count * self._bucket_width
                buckets = block[offset:stop].reshape(
                    full_count, self._bucket_width, self._channel_count
                )
                min_offsets = buckets.argmin(axis=1)
                max_offsets = buckets.argmax(axis=1)
                first_bucket = (start + offset) // self._bucket_width
                target = slice(first_bucket, first_bucket + full_count)
                starts = (
                    np.arange(full_count, dtype=np.int64) * self._bucket_width
                    + start + offset
                )[:, None]
                self._min_indices[target] = starts + min_offsets
                self._max_indices[target] = starts + max_offsets
                self._min_values[target] = np.take_along_axis(
                    buckets, min_offsets[:, None, :], axis=1
                )[:, 0, :]
                self._max_values[target] = np.take_along_axis(
                    buckets, max_offsets[:, None, :], axis=1
                )[:, 0, :]
                offset = stop
            if offset < row_count:
                self._merge_fragment(block[offset:], start + offset)
        self._frame_count = end

    def _merge_fragment(self, fragment, start):
        bucket = start // self._bucket_width
        columns = np.arange(self._channel_count)
        min_offsets = fragment.argmin(axis=0)
        max_offsets = fragment.argmax(axis=0)
        minima = fragment[min_offsets, columns]
        maxima = fragment[max_offsets, columns]
        lower = minima < self._min_values[bucket]
        higher = maxima > self._max_values[bucket]
        self._min_values[bucket, lower] = minima[lower]
        self._max_values[bucket, higher] = maxima[higher]
        self._min_indices[bucket, lower] = start + min_offsets[lower]
        self._max_indices[bucket, higher] = start + max_offsets[higher]

    def finish(self):
        """Return per-channel (int64 positions, float32 amplitudes) copies."""
        if self._frame_count != self._total_frames:
            raise ValueError("received frames do not match total_frames")
        columns = []
        for column in range(self._channel_count):
            if self._samples is not None:
                indices = np.arange(self._total_frames, dtype=np.int64)
                values = self._samples[:, column].copy()
            else:
                positions = np.concatenate((
                    np.array([0, self._total_frames - 1], dtype=np.int64),
                    self._min_indices[:, column], self._max_indices[:, column],
                ))
                amplitudes = np.concatenate((
                    np.array([self._first[column], self._last[column]], dtype=np.float32),
                    self._min_values[:, column], self._max_values[:, column],
                ))
                indices, selected = np.unique(positions, return_index=True)
                values = amplitudes[selected]
            columns.append((indices, values))
        return tuple(columns)

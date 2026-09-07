from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from base.streaming_waveform_accumulator import StreamingWaveformSnapshot


@dataclass
class _Bucket:
    start: int
    stop: int
    minimum_index: int
    minimum_value: np.float32
    maximum_index: int
    maximum_value: np.float32


class RollingWaveformAccumulator:
    def __init__(self, *, max_points: int, window_seconds: int):
        if isinstance(max_points, (bool, np.bool_)) or not isinstance(
            max_points, (int, np.integer)
        ):
            raise ValueError("max_points must be an integer")
        if max_points < 4:
            raise ValueError("max_points must be at least 4")
        if isinstance(window_seconds, (bool, np.bool_)) or not isinstance(
            window_seconds, (int, np.integer)
        ):
            raise ValueError("window_seconds must be a positive integer")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be a positive integer")

        self._max_points = int(max_points)
        self._window_seconds = int(window_seconds)
        self._bucket_count = (self._max_points - 2) // 2
        self.clear()

    @property
    def raw_sample_count(self) -> int:
        return self._raw_sample_count

    @property
    def display_sample_count(self) -> int:
        return max(0, self._raw_sample_count - self._startup_trim_samples)

    @property
    def retained_bucket_count(self) -> int:
        return len(self._buckets)

    def begin(self, *, sample_rate: int, startup_trim_samples: int) -> None:
        if isinstance(sample_rate, (bool, np.bool_)) or not isinstance(
            sample_rate, (int, np.integer)
        ):
            raise ValueError("sample_rate must be a positive integer")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        if isinstance(startup_trim_samples, (bool, np.bool_)) or not isinstance(
            startup_trim_samples, (int, np.integer)
        ):
            raise ValueError(
                "startup_trim_samples must be a non-negative integer"
            )
        if startup_trim_samples < 0:
            raise ValueError(
                "startup_trim_samples must be a non-negative integer"
            )

        self.clear()
        self._sample_rate = int(sample_rate)
        self._startup_trim_samples = int(startup_trim_samples)
        self._window_samples = self._sample_rate * self._window_seconds
        self._bucket_width = max(
            1,
            math.ceil(self._window_samples / self._bucket_count),
        )

    def append(self, mono_chunk: np.ndarray) -> tuple[int, int]:
        if (
            not isinstance(mono_chunk, np.ndarray)
            or mono_chunk.dtype != np.float32
            or mono_chunk.ndim != 1
            or mono_chunk.size == 0
        ):
            raise ValueError(
                "mono_chunk must be a non-empty one-dimensional float32 array"
            )
        if self._sample_rate is None:
            raise RuntimeError("begin must be called before append")

        start = self._raw_sample_count
        stop = start + int(mono_chunk.size)
        post_trim_start = max(start, self._startup_trim_samples)
        if post_trim_start < stop:
            chunk_offset = post_trim_start - start
            display_start = post_trim_start - self._startup_trim_samples
            display_values = mono_chunk[chunk_offset:]
            self._update_fixed_buckets(display_values, display_start)
            self._latest_display_sample = (
                display_start + int(display_values.size) - 1,
                np.float32(display_values[-1]),
            )

        self._raw_sample_count = stop
        return start, stop

    def snapshot(self) -> StreamingWaveformSnapshot:
        if self._latest_display_sample is None:
            return self._empty_snapshot()

        latest_index = self.display_sample_count - 1
        cutoff = max(0, latest_index - self._window_samples + 1)
        by_index: dict[int, np.float32] = {}
        for bucket in self._buckets:
            if bucket.start < cutoff:
                continue
            by_index[bucket.minimum_index] = bucket.minimum_value
            by_index[bucket.maximum_index] = bucket.maximum_value

        newest_index, newest_value = self._latest_display_sample
        by_index[newest_index] = newest_value
        records = sorted(by_index.items())
        if len(records) > self._max_points:
            raise AssertionError("rolling waveform snapshot exceeds max_points")

        indices = np.fromiter(
            (record[0] for record in records),
            dtype=np.int64,
            count=len(records),
        )
        amplitude = np.fromiter(
            (record[1] for record in records),
            dtype=np.float32,
            count=len(records),
        )
        time = (indices - latest_index).astype(np.float64) / self._sample_rate
        time[-1] = 0.0
        time.setflags(write=False)
        amplitude.setflags(write=False)
        return StreamingWaveformSnapshot(
            time=time,
            amplitude=amplitude,
            sample_stop=self.display_sample_count,
        )

    def clear(self) -> None:
        self._raw_sample_count = 0
        self._sample_rate: int | None = None
        self._startup_trim_samples = 0
        self._window_samples = 0
        self._bucket_width = 1
        self._buckets: deque[_Bucket] = deque()
        self._latest_display_sample: tuple[int, np.float32] | None = None

    def _empty_snapshot(self) -> StreamingWaveformSnapshot:
        time = np.empty(0, dtype=np.float64)
        amplitude = np.empty(0, dtype=np.float32)
        time.setflags(write=False)
        amplitude.setflags(write=False)
        return StreamingWaveformSnapshot(
            time=time,
            amplitude=amplitude,
            sample_stop=self.display_sample_count,
        )

    def _update_fixed_buckets(
        self,
        values: np.ndarray,
        display_start: int,
    ) -> None:
        offset = 0
        while offset < values.size:
            segment_start = display_start + offset
            bucket_id = segment_start // self._bucket_width
            bucket_stop = (bucket_id + 1) * self._bucket_width
            segment_size = min(
                int(values.size) - offset,
                bucket_stop - segment_start,
            )
            segment = values[offset : offset + segment_size]
            minimum_offset = int(np.argmin(segment))
            maximum_offset = int(np.argmax(segment))
            self._merge_bucket_summary(
                start=segment_start,
                stop=segment_start + segment_size,
                minimum_index=segment_start + minimum_offset,
                minimum_value=np.float32(segment[minimum_offset]),
                maximum_index=segment_start + maximum_offset,
                maximum_value=np.float32(segment[maximum_offset]),
            )
            offset += segment_size
            self._evict_expired_buckets(segment_start + segment_size - 1)

    def _merge_bucket_summary(
        self,
        *,
        start: int,
        stop: int,
        minimum_index: int,
        minimum_value: np.float32,
        maximum_index: int,
        maximum_value: np.float32,
    ) -> None:
        bucket_id = start // self._bucket_width
        if (
            self._buckets
            and self._buckets[-1].start // self._bucket_width == bucket_id
        ):
            bucket = self._buckets[-1]
            bucket.stop = stop
            if minimum_value < bucket.minimum_value:
                bucket.minimum_index = minimum_index
                bucket.minimum_value = minimum_value
            if maximum_value > bucket.maximum_value:
                bucket.maximum_index = maximum_index
                bucket.maximum_value = maximum_value
            return

        self._buckets.append(
            _Bucket(
                start=start,
                stop=stop,
                minimum_index=minimum_index,
                minimum_value=minimum_value,
                maximum_index=maximum_index,
                maximum_value=maximum_value,
            )
        )

    def _evict_expired_buckets(self, latest_index: int) -> None:
        cutoff = max(0, latest_index - self._window_samples + 1)
        while self._buckets and self._buckets[0].stop <= cutoff:
            self._buckets.popleft()

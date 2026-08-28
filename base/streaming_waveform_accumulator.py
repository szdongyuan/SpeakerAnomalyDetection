from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class StreamingWaveformSnapshot:
    time: np.ndarray
    amplitude: np.ndarray
    sample_stop: int


class StreamingWaveformAccumulator:
    def __init__(self, *, max_points: int, retain_raw: bool = True):
        if isinstance(max_points, (bool, np.bool_)) or not isinstance(
            max_points, (int, np.integer)
        ):
            raise ValueError("max_points must be an integer")
        if max_points < 4:
            raise ValueError("max_points must be at least 4")
        self._max_points = int(max_points)
        self._retain_raw = bool(retain_raw)
        self.clear()

    @property
    def capacity(self) -> int:
        return int(self._raw.size)

    @property
    def raw_sample_count(self) -> int:
        return self._raw_sample_count

    @property
    def display_sample_count(self) -> int:
        return max(0, self._raw_sample_count - self._startup_trim_samples)

    def begin(self, *, sample_rate: float, startup_trim_samples: int) -> None:
        if isinstance(sample_rate, (bool, np.bool_)) or not isinstance(
            sample_rate, (int, float, np.integer, np.floating)
        ):
            raise ValueError("sample_rate must be positive and finite")
        if not math.isfinite(float(sample_rate)) or sample_rate <= 0:
            raise ValueError("sample_rate must be positive and finite")
        if isinstance(startup_trim_samples, (bool, np.bool_)) or not isinstance(
            startup_trim_samples, (int, np.integer)
        ):
            raise ValueError("startup_trim_samples must be a non-negative integer")
        if startup_trim_samples < 0:
            raise ValueError("startup_trim_samples must be a non-negative integer")

        self.clear()
        self._sample_rate = float(sample_rate)
        self._startup_trim_samples = int(startup_trim_samples)

    def append(self, mono_chunk: np.ndarray) -> tuple[int, int]:
        if (
            not isinstance(mono_chunk, np.ndarray)
            or mono_chunk.dtype != np.float32
            or mono_chunk.ndim != 1
            or mono_chunk.size == 0
        ):
            raise ValueError("mono_chunk must be a non-empty one-dimensional float32 array")
        if self._sample_rate is None:
            raise RuntimeError("begin must be called before append")

        start = self._raw_sample_count
        stop = start + int(mono_chunk.size)
        if self._retain_raw:
            target = self._expanded_buffer(stop)
            target[start:stop] = mono_chunk
        else:
            target = self._raw

        post_trim_start = max(start, self._startup_trim_samples)
        if post_trim_start < stop:
            chunk_offset = post_trim_start - start
            display_start = post_trim_start - self._startup_trim_samples
            self._accept_display_samples(mono_chunk[chunk_offset:], display_start)

        self._raw = target
        self._raw_sample_count = stop
        return start, stop

    def snapshot(self) -> StreamingWaveformSnapshot:
        records = self._envelope_records()
        indices = np.fromiter((record[0] for record in records), dtype=np.int64)
        amplitude = np.fromiter(
            (record[1] for record in records), dtype=np.float32, count=len(records)
        )
        if indices.size:
            time = indices.astype(np.float64) / self._sample_rate
        else:
            time = np.empty(0, dtype=np.float64)
        time.setflags(write=False)
        amplitude.setflags(write=False)
        return StreamingWaveformSnapshot(
            time=time,
            amplitude=amplitude,
            sample_stop=self.display_sample_count,
        )

    def raw_view(self) -> np.ndarray:
        view = self._raw[: self._raw_sample_count].view()
        view.setflags(write=False)
        return view

    def clear(self) -> None:
        self._raw = np.empty(0, dtype=np.float32)
        self._raw_sample_count = 0
        self._sample_rate = None
        self._startup_trim_samples = 0
        self._bucket_width = 1
        self._bucket_starts = np.empty(0, dtype=np.int64)
        self._bucket_stops = np.empty(0, dtype=np.int64)
        self._minimum_indices = np.empty(0, dtype=np.int64)
        self._minimum_values = np.empty(0, dtype=np.float32)
        self._maximum_indices = np.empty(0, dtype=np.int64)
        self._maximum_values = np.empty(0, dtype=np.float32)
        self._nan_indices = np.empty(0, dtype=np.int64)
        self._nan_values = np.empty(0, dtype=np.float32)
        self._first_display_sample: tuple[int, np.float32] | None = None
        self._latest_display_sample: tuple[int, np.float32] | None = None

    def _expanded_buffer(self, required_capacity: int) -> np.ndarray:
        if required_capacity <= self.capacity:
            return self._raw
        capacity = 1 if self.capacity == 0 else self.capacity
        while capacity < required_capacity:
            capacity *= 2
        expanded = np.empty(capacity, dtype=np.float32)
        expanded[: self._raw_sample_count] = self._raw[: self._raw_sample_count]
        return expanded

    def _accept_display_samples(
        self, values: np.ndarray, display_start: int
    ) -> None:
        if self._first_display_sample is None:
            self._first_display_sample = (display_start, values[0])
        accepted = 0
        while accepted < values.size:
            remaining = values[accepted:]
            remaining_start = display_start + accepted
            point_counts = self._prefix_materialized_point_counts(
                remaining, remaining_start
            )
            over_limit = np.flatnonzero(point_counts > self._max_points)
            batch_size = (
                int(over_limit[0]) + 1 if over_limit.size else int(remaining.size)
            )
            batch = remaining[:batch_size]
            summaries = self._summarize_groups(batch, remaining_start)
            summary_start = 0
            if self._tail_accepts_more():
                self._merge_tail_group(summaries)
                summary_start = 1
            self._append_group_summaries(summaries, summary_start)
            accepted += batch_size
            self._latest_display_sample = (
                display_start + accepted - 1,
                values[accepted - 1],
            )

            while self._materialized_point_count() > self._max_points:
                if self._bucket_starts.size == 1:
                    self._drop_auxiliary_nan_marker()
                    break
                self._compact_envelope()

    def _tail_accepts_more(self) -> bool:
        return bool(
            self._bucket_starts.size
            and self._bucket_stops[-1] - self._bucket_starts[-1]
            < self._bucket_width
        )

    def _prefix_materialized_point_counts(
        self, values: np.ndarray, display_start: int
    ) -> np.ndarray:
        """Return exact legacy envelope size after every prefix of ``values``."""
        indices = np.arange(
            display_start,
            display_start + values.size,
            dtype=np.int64,
        )
        group_ids = indices // self._bucket_width
        group_starts = np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.flatnonzero(group_ids[1:] != group_ids[:-1]).astype(
                    np.int64, copy=False
                )
                + 1,
            )
        )
        group_stops = np.concatenate(
            (group_starts[1:], np.asarray([values.size], dtype=np.int64))
        )
        lengths = group_stops - group_starts
        row_count = group_starts.size
        column_count = int(np.max(lengths))
        rows = np.repeat(np.arange(row_count), lengths)
        columns = np.arange(values.size) - np.repeat(group_starts, lengths)

        assigned = np.zeros((row_count, column_count), dtype=bool)
        assigned[rows, columns] = True
        matrix = np.zeros((row_count, column_count), dtype=np.float32)
        matrix[rows, columns] = values
        nan_mask = assigned & np.isnan(matrix)
        valid_mask = assigned & ~nan_mask

        merge_tail = self._tail_accepts_more()
        initial_valid = bool(merge_tail and self._minimum_indices[-1] >= 0)
        initial_nan = bool(merge_tail and self._nan_indices[-1] >= 0)
        initial_minimum = (
            self._minimum_values[-1] if initial_valid else np.float32(np.inf)
        )
        initial_maximum = (
            self._maximum_values[-1] if initial_valid else np.float32(-np.inf)
        )

        running_minimum = np.minimum.accumulate(
            np.where(valid_mask, matrix, np.float32(np.inf)), axis=1
        )
        running_maximum = np.maximum.accumulate(
            np.where(valid_mask, matrix, np.float32(-np.inf)), axis=1
        )
        running_minimum[0] = np.minimum(running_minimum[0], initial_minimum)
        running_maximum[0] = np.maximum(running_maximum[0], initial_maximum)
        has_valid = np.logical_or.accumulate(valid_mask, axis=1)
        has_nan = np.logical_or.accumulate(nan_mask, axis=1)
        if initial_valid:
            has_valid[0] = True
        if initial_nan:
            has_nan[0] = True

        prior_valid = np.zeros_like(has_valid)
        prior_nan = np.zeros_like(has_nan)
        prior_valid[:, 1:] = has_valid[:, :-1]
        prior_nan[:, 1:] = has_nan[:, :-1]
        prior_valid[0, 0] = initial_valid
        prior_nan[0, 0] = initial_nan
        prior_minimum = np.full_like(matrix, np.float32(np.inf))
        prior_maximum = np.full_like(matrix, np.float32(-np.inf))
        prior_minimum[:, 1:] = running_minimum[:, :-1]
        prior_maximum[:, 1:] = running_maximum[:, :-1]
        prior_minimum[0, 0] = initial_minimum
        prior_maximum[0, 0] = initial_maximum

        latest_is_bucket_point = np.where(
            nan_mask,
            ~prior_nan,
            valid_mask
            & (
                ~prior_valid
                | (matrix < prior_minimum)
                | (matrix > prior_maximum)
            ),
        )
        latest_extra = assigned & ~latest_is_bucket_point
        latest_extra[rows, columns] &= indices != self._first_display_sample[0]

        bucket_counts = has_valid.astype(np.int64)
        bucket_counts += (
            has_valid & (running_minimum != running_maximum)
        ).astype(np.int64)
        bucket_counts += has_nan.astype(np.int64)

        existing_counts = self._bucket_point_counts()
        base_count = int(np.sum(existing_counts))
        if merge_tail:
            base_count -= int(existing_counts[-1])
        final_bucket_counts = bucket_counts[
            np.arange(row_count), lengths - 1
        ]
        points_before_row = base_count + np.concatenate(
            (np.asarray([0], dtype=np.int64), np.cumsum(final_bucket_counts[:-1]))
        )

        first_extra = np.zeros_like(bucket_counts)
        first_bucket_is_row_zero = (
            self._bucket_starts.size == 0
            or (merge_tail and self._bucket_starts.size == 1)
        )
        if first_bucket_is_row_zero:
            first_value = self._first_display_sample[1]
            if np.isnan(first_value):
                first_retained = has_nan[0]
            else:
                first_retained = (running_minimum[0] == first_value) | (
                    running_maximum[0] == first_value
                )
            first_extra[0] = ~first_retained
            if row_count > 1:
                first_extra[1:] = int(not bool(first_retained[lengths[0] - 1]))
        else:
            fixed_first_extra = int(
                self._first_display_sample[0] not in self._point_indices_at(0)
            )
            first_extra[:] = fixed_first_extra

        materialized = (
            points_before_row[:, None]
            + bucket_counts
            + first_extra
            + latest_extra.astype(np.int64)
        )
        return materialized[rows, columns]

    def _bucket_point_counts(self) -> np.ndarray:
        return (
            (self._minimum_indices >= 0).astype(np.int64)
            + (
                (self._maximum_indices >= 0)
                & (self._maximum_indices != self._minimum_indices)
            ).astype(np.int64)
            + (self._nan_indices >= 0).astype(np.int64)
        )

    def _summarize_groups(
        self, values: np.ndarray, display_start: int
    ) -> tuple[np.ndarray, ...]:
        indices = np.arange(
            display_start,
            display_start + values.size,
            dtype=np.int64,
        )
        group_ids = indices // self._bucket_width
        group_starts = np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.flatnonzero(group_ids[1:] != group_ids[:-1]).astype(
                    np.int64, copy=False
                )
                + 1,
            )
        )
        group_stops = np.concatenate(
            (group_starts[1:], np.asarray([values.size], dtype=np.int64))
        )
        group_lengths = group_stops - group_starts
        nan_mask = np.isnan(values)
        sentinel = np.iinfo(np.int64).max

        safe_minimum = np.where(nan_mask, np.float32(np.inf), values)
        minimums = np.minimum.reduceat(safe_minimum, group_starts)
        minimum_indices = np.minimum.reduceat(
            np.where(
                ~nan_mask & (values == np.repeat(minimums, group_lengths)),
                indices,
                sentinel,
            ),
            group_starts,
        )
        minimum_indices[minimum_indices == sentinel] = -1
        minimum_values = self._values_at_indices(
            values, minimum_indices, display_start
        )

        safe_maximum = np.where(nan_mask, np.float32(-np.inf), values)
        maximums = np.maximum.reduceat(safe_maximum, group_starts)
        maximum_indices = np.minimum.reduceat(
            np.where(
                ~nan_mask & (values == np.repeat(maximums, group_lengths)),
                indices,
                sentinel,
            ),
            group_starts,
        )
        maximum_indices[maximum_indices == sentinel] = -1
        maximum_values = self._values_at_indices(
            values, maximum_indices, display_start
        )

        nan_indices = np.minimum.reduceat(
            np.where(nan_mask, indices, sentinel), group_starts
        )
        nan_indices[nan_indices == sentinel] = -1
        nan_values = self._values_at_indices(values, nan_indices, display_start)

        return (
            indices[group_starts],
            indices[group_stops - 1] + 1,
            minimum_indices,
            minimum_values,
            maximum_indices,
            maximum_values,
            nan_indices,
            nan_values,
        )

    @staticmethod
    def _values_at_indices(
        values: np.ndarray, indices: np.ndarray, display_start: int
    ) -> np.ndarray:
        result = np.zeros(indices.size, dtype=np.float32)
        valid = indices >= 0
        result[valid] = values[indices[valid] - display_start]
        return result

    def _merge_tail_group(self, summaries: tuple[np.ndarray, ...]) -> None:
        self._bucket_stops[-1] = summaries[1][0]
        new_minimum_index = summaries[2][0]
        if new_minimum_index >= 0 and (
            self._minimum_indices[-1] < 0
            or summaries[3][0] < self._minimum_values[-1]
        ):
            self._minimum_indices[-1] = new_minimum_index
            self._minimum_values[-1] = summaries[3][0]
        new_maximum_index = summaries[4][0]
        if new_maximum_index >= 0 and (
            self._maximum_indices[-1] < 0
            or summaries[5][0] > self._maximum_values[-1]
        ):
            self._maximum_indices[-1] = new_maximum_index
            self._maximum_values[-1] = summaries[5][0]
        if self._nan_indices[-1] < 0 and summaries[6][0] >= 0:
            self._nan_indices[-1] = summaries[6][0]
            self._nan_values[-1] = summaries[7][0]

    def _append_group_summaries(
        self, summaries: tuple[np.ndarray, ...], summary_start: int
    ) -> None:
        if summary_start >= summaries[0].size:
            return
        names = (
            "_bucket_starts",
            "_bucket_stops",
            "_minimum_indices",
            "_minimum_values",
            "_maximum_indices",
            "_maximum_values",
            "_nan_indices",
            "_nan_values",
        )
        for name, summary in zip(names, summaries):
            current = getattr(self, name)
            setattr(self, name, np.concatenate((current, summary[summary_start:])))

    def _materialized_point_count(self) -> int:
        minimum_exists = self._minimum_indices >= 0
        count = int(np.count_nonzero(minimum_exists))
        count += int(
            np.count_nonzero(
                (self._maximum_indices >= 0)
                & (self._maximum_indices != self._minimum_indices)
            )
        )
        count += int(np.count_nonzero(self._nan_indices >= 0))
        first_index = self._first_display_sample[0]
        if first_index not in self._point_indices_at(0):
            count += 1

        if self._latest_display_sample is not None:
            latest_index = self._latest_display_sample[0]
            if latest_index not in self._point_indices_at(-1) and latest_index != first_index:
                count += 1
        return count

    def _point_indices_at(self, position: int) -> set[int]:
        return {
            int(indices[position])
            for indices in (
                self._minimum_indices,
                self._maximum_indices,
                self._nan_indices,
            )
            if indices[position] >= 0
        }

    def _compact_envelope(self) -> None:
        left = np.arange(0, self._bucket_starts.size, 2)
        paired_left = left[left + 1 < self._bucket_starts.size]
        right = paired_left + 1

        starts = self._bucket_starts[left].copy()
        stops = self._bucket_stops[left].copy()
        stops[: right.size] = self._bucket_stops[right]
        minimum_indices = self._minimum_indices[left].copy()
        minimum_values = self._minimum_values[left].copy()
        maximum_indices = self._maximum_indices[left].copy()
        maximum_values = self._maximum_values[left].copy()
        nan_indices = self._nan_indices[left].copy()
        nan_values = self._nan_values[left].copy()

        choose_right_minimum = (minimum_indices[: right.size] < 0) | (
            (self._minimum_indices[right] >= 0)
            & (self._minimum_values[right] < minimum_values[: right.size])
        )
        minimum_indices[: right.size][choose_right_minimum] = self._minimum_indices[
            right
        ][choose_right_minimum]
        minimum_values[: right.size][choose_right_minimum] = self._minimum_values[
            right
        ][choose_right_minimum]

        choose_right_maximum = (maximum_indices[: right.size] < 0) | (
            (self._maximum_indices[right] >= 0)
            & (self._maximum_values[right] > maximum_values[: right.size])
        )
        maximum_indices[: right.size][choose_right_maximum] = self._maximum_indices[
            right
        ][choose_right_maximum]
        maximum_values[: right.size][choose_right_maximum] = self._maximum_values[
            right
        ][choose_right_maximum]

        choose_right_nan = (nan_indices[: right.size] < 0) & (
            self._nan_indices[right] >= 0
        )
        nan_indices[: right.size][choose_right_nan] = self._nan_indices[right][
            choose_right_nan
        ]
        nan_values[: right.size][choose_right_nan] = self._nan_values[right][
            choose_right_nan
        ]

        self._bucket_starts = starts
        self._bucket_stops = stops
        self._minimum_indices = minimum_indices
        self._minimum_values = minimum_values
        self._maximum_indices = maximum_indices
        self._maximum_values = maximum_values
        self._nan_indices = nan_indices
        self._nan_values = nan_values
        self._bucket_width *= 2

    def _drop_auxiliary_nan_marker(self) -> None:
        if self._nan_indices[0] < 0:
            return
        endpoint_indices = {
            self._first_display_sample[0],
            self._latest_display_sample[0],
        }
        if self._nan_indices[0] in endpoint_indices:
            return
        self._nan_indices[0] = -1
        self._nan_values[0] = np.float32(0)

    def _envelope_records(self) -> list[tuple[int, np.float32]]:
        by_index = {}
        for indices, values in (
            (self._minimum_indices, self._minimum_values),
            (self._maximum_indices, self._maximum_values),
            (self._nan_indices, self._nan_values),
        ):
            by_index.update(
                (int(index), value)
                for index, value in zip(indices, values)
                if index >= 0
            )
        for point in (self._first_display_sample, self._latest_display_sample):
            if point is not None:
                by_index[point[0]] = point[1]
        return sorted(by_index.items())

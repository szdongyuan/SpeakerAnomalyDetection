import math

import numpy as np
import pytest

from base.rolling_waveform_accumulator import RollingWaveformAccumulator
from consts.recording_preview_consts import (
    MAIN_RECORDING_LIVE_MAX_POINTS,
    MAIN_RECORDING_LIVE_WINDOW_SECONDS,
    PREVIEW_TIME_LOWER_BOUND_TOLERANCE,
)


def _expected_fixed_bucket_records(
    values: np.ndarray,
    *,
    cutoff: int,
    latest_index: int,
    bucket_width: int,
) -> list[tuple[int, np.float32]]:
    first_complete_start = math.ceil(cutoff / bucket_width) * bucket_width
    by_index: dict[int, np.float32] = {}
    for bucket_start in range(
        first_complete_start,
        latest_index + 1,
        bucket_width,
    ):
        bucket_stop = min(bucket_start + bucket_width, latest_index + 1)
        bucket = values[bucket_start:bucket_stop]
        minimum_index = bucket_start + int(np.argmin(bucket))
        maximum_index = bucket_start + int(np.argmax(bucket))
        by_index[minimum_index] = np.float32(values[minimum_index])
        by_index[maximum_index] = np.float32(values[maximum_index])
    by_index[latest_index] = np.float32(values[latest_index])
    return sorted(by_index.items())


def test_rolling_accumulator_validates_boundary_inputs():
    for invalid in (True, 0, 3, 3.5):
        with pytest.raises(ValueError, match="max_points"):
            RollingWaveformAccumulator(max_points=invalid, window_seconds=10)
    with pytest.raises(ValueError, match="window_seconds"):
        RollingWaveformAccumulator(max_points=8, window_seconds=0)


@pytest.mark.parametrize("sample_rate", (True, 0, -1, 2.5))
def test_begin_requires_a_positive_integer_sample_rate(sample_rate):
    accumulator = RollingWaveformAccumulator(max_points=8, window_seconds=10)

    with pytest.raises(ValueError, match="sample_rate"):
        accumulator.begin(sample_rate=sample_rate, startup_trim_samples=0)


@pytest.mark.parametrize("startup_trim_samples", (True, -1, 1.5))
def test_begin_requires_a_non_negative_integer_trim(startup_trim_samples):
    accumulator = RollingWaveformAccumulator(max_points=8, window_seconds=10)

    with pytest.raises(ValueError, match="startup_trim_samples"):
        accumulator.begin(
            sample_rate=2,
            startup_trim_samples=startup_trim_samples,
        )


def test_append_requires_begin_and_float32_mono():
    accumulator = RollingWaveformAccumulator(max_points=8, window_seconds=10)
    with pytest.raises(RuntimeError, match="begin"):
        accumulator.append(np.ones(1, dtype=np.float32))
    accumulator.begin(sample_rate=2, startup_trim_samples=0)
    with pytest.raises(ValueError, match="one-dimensional float32"):
        accumulator.append(np.ones(2, dtype=np.float64))


@pytest.mark.parametrize(
    "chunk",
    (
        np.empty(0, dtype=np.float32),
        np.ones((1, 1), dtype=np.float32),
        [1.0],
    ),
)
def test_append_rejects_non_normalized_or_empty_chunks(chunk):
    accumulator = RollingWaveformAccumulator(max_points=8, window_seconds=10)
    accumulator.begin(sample_rate=2, startup_trim_samples=0)

    with pytest.raises(ValueError, match="one-dimensional float32"):
        accumulator.append(chunk)


def test_counts_are_cumulative_and_startup_trim_is_not_displayed():
    accumulator = RollingWaveformAccumulator(max_points=8, window_seconds=10)
    accumulator.begin(sample_rate=2, startup_trim_samples=2)

    assert accumulator.append(np.array([90.0], dtype=np.float32)) == (0, 1)
    assert accumulator.append(np.array([91.0, 1.0, 2.0], dtype=np.float32)) == (
        1,
        4,
    )

    assert accumulator.raw_sample_count == 4
    assert accumulator.display_sample_count == 2
    assert accumulator.snapshot().sample_stop == 2


def test_empty_snapshots_are_owned_read_only_arrays_and_clear_resets_lifecycle():
    accumulator = RollingWaveformAccumulator(max_points=8, window_seconds=10)

    snapshot = accumulator.snapshot()
    assert snapshot.sample_stop == 0
    assert snapshot.time.dtype == np.float64
    assert snapshot.amplitude.dtype == np.float32
    assert snapshot.time.size == 0
    assert snapshot.amplitude.size == 0
    assert not snapshot.time.flags.writeable
    assert not snapshot.amplitude.flags.writeable

    accumulator.begin(sample_rate=2, startup_trim_samples=0)
    accumulator.append(np.arange(5, dtype=np.float32))
    accumulator.clear()

    snapshot = accumulator.snapshot()
    assert accumulator.raw_sample_count == 0
    assert accumulator.display_sample_count == 0
    assert accumulator.retained_bucket_count == 0
    assert snapshot.sample_stop == 0
    assert not snapshot.time.flags.writeable
    assert not snapshot.amplitude.flags.writeable
    with pytest.raises(RuntimeError, match="begin"):
        accumulator.append(np.ones(1, dtype=np.float32))


def test_snapshot_emits_every_in_window_bucket_extremum_in_temporal_order():
    sample_rate = 2
    window_seconds = 10
    max_points = 10
    bucket_count = (max_points - 2) // 2
    window_samples = sample_rate * window_seconds
    bucket_width = math.ceil(window_samples / bucket_count)
    values = np.arange(28, dtype=np.float32)
    values[10:15] = np.array([3, 9, -2, 7, 1], dtype=np.float32)
    values[15:20] = np.array([-8, 0, 5, 2, 1], dtype=np.float32)
    values[20:25] = np.float32(4)
    values[25:28] = np.array([-6, 8, 2], dtype=np.float32)
    accumulator = RollingWaveformAccumulator(
        max_points=max_points,
        window_seconds=window_seconds,
    )
    accumulator.begin(sample_rate=sample_rate, startup_trim_samples=0)

    accumulator.append(values)
    snapshot = accumulator.snapshot()

    latest_index = values.size - 1
    cutoff = latest_index - window_samples + 1
    expected = _expected_fixed_bucket_records(
        values,
        cutoff=cutoff,
        latest_index=latest_index,
        bucket_width=bucket_width,
    )
    actual_indices = np.rint(
        snapshot.time * sample_rate + latest_index
    ).astype(np.int64)
    np.testing.assert_array_equal(
        actual_indices,
        np.asarray([record[0] for record in expected], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        snapshot.amplitude,
        np.asarray([record[1] for record in expected], dtype=np.float32),
    )
    assert list(actual_indices).index(11) < list(actual_indices).index(12)
    assert list(actual_indices).index(15) < list(actual_indices).index(17)
    assert np.count_nonzero(actual_indices == 20) == 1
    assert snapshot.amplitude[-1] == values[-1]
    first_complete_start = math.ceil(cutoff / bucket_width) * bucket_width
    assert 0 <= first_complete_start - cutoff < bucket_width
    assert snapshot.time[-1] == 0.0
    assert np.all(snapshot.time <= 0.0)
    assert snapshot.time[0] >= -window_seconds
    assert len(snapshot.time) <= max_points
    assert np.all(np.diff(snapshot.time) > 0)


def test_long_chunked_run_keeps_bounded_state_and_cumulative_progress(monkeypatch):
    sample_rate = 1_000
    trim = 137
    window_samples = sample_rate * MAIN_RECORDING_LIVE_WINDOW_SECONDS
    bucket_count = (MAIN_RECORDING_LIVE_MAX_POINTS - 2) // 2
    total = trim + window_samples * 12 + 431
    values = np.sin(np.arange(total, dtype=np.float32) / 17.0)
    accumulator = RollingWaveformAccumulator(
        max_points=MAIN_RECORDING_LIVE_MAX_POINTS,
        window_seconds=MAIN_RECORDING_LIVE_WINDOW_SECONDS,
    )
    accumulator.begin(
        sample_rate=sample_rate,
        startup_trim_samples=trim,
    )

    sample_stop = 0
    for start in range(0, total, 1_013):
        sample_stop = accumulator.append(values[start : start + 1_013])[1]
        assert accumulator.retained_bucket_count <= bucket_count + 2

    assert sample_stop == total
    assert accumulator.raw_sample_count == total
    assert accumulator.display_sample_count == total - trim
    assert accumulator.display_sample_count > window_samples
    assert not hasattr(accumulator, "_raw")
    assert not hasattr(accumulator, "_chunks")
    assert not any(
        isinstance(value, np.ndarray)
        and value.size > MAIN_RECORDING_LIVE_MAX_POINTS
        for value in vars(accumulator).values()
    )

    def fail_if_acceptance_runs_during_snapshot(*args, **kwargs):
        raise AssertionError("snapshot inspected samples through the update helper")

    monkeypatch.setattr(
        accumulator,
        "_update_fixed_buckets",
        fail_if_acceptance_runs_during_snapshot,
    )
    snapshot = accumulator.snapshot()

    assert snapshot.sample_stop == total - trim
    assert snapshot.time[-1] == 0.0
    assert np.all(snapshot.time <= 0.0)
    assert snapshot.time[0] >= (
        -MAIN_RECORDING_LIVE_WINDOW_SECONDS
        - PREVIEW_TIME_LOWER_BOUND_TOLERANCE
    )
    assert len(snapshot.time) <= MAIN_RECORDING_LIVE_MAX_POINTS
    assert np.all(np.diff(snapshot.time) > 0)
    assert not snapshot.time.flags.writeable
    assert not snapshot.amplitude.flags.writeable

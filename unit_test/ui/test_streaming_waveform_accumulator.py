import numpy as np
import pytest

from ui.sequence.streaming_waveform_accumulator import StreamingWaveformAccumulator


class _ScalarEnvelopeReference:
    """Small copy of the original sample-at-a-time envelope semantics."""

    def __init__(self, *, max_points, startup_trim_samples):
        self.max_points = max_points
        self.startup_trim_samples = startup_trim_samples
        self.raw_sample_count = 0
        self.bucket_width = 1
        self.buckets = []
        self.first = None
        self.latest = None

    @staticmethod
    def _select(left, right, *, minimum):
        if left is None:
            return right
        if right is None:
            return left
        if minimum:
            return right if right[1] < left[1] else left
        return right if right[1] > left[1] else left

    def _materialized_point_count(self):
        by_index = {
            point[0]
            for bucket in self.buckets
            for point in (bucket["minimum"], bucket["maximum"], bucket["nan"])
            if point is not None
        }
        by_index.add(self.first[0])
        by_index.add(self.latest[0])
        return len(by_index)

    def _compact(self):
        compacted = []
        for position in range(0, len(self.buckets), 2):
            left = self.buckets[position]
            if position + 1 == len(self.buckets):
                compacted.append(left)
                continue
            right = self.buckets[position + 1]
            compacted.append(
                {
                    "start": left["start"],
                    "stop": right["stop"],
                    "minimum": self._select(
                        left["minimum"], right["minimum"], minimum=True
                    ),
                    "maximum": self._select(
                        left["maximum"], right["maximum"], minimum=False
                    ),
                    "nan": left["nan"] if left["nan"] is not None else right["nan"],
                }
            )
        self.buckets = compacted
        self.bucket_width *= 2

    def _drop_auxiliary_nan_marker(self):
        bucket = self.buckets[0]
        nan = bucket["nan"]
        if nan is not None and nan[0] not in {self.first[0], self.latest[0]}:
            bucket["nan"] = None

    def append(self, values):
        start = self.raw_sample_count
        stop = start + values.size
        post_trim_start = max(start, self.startup_trim_samples)
        for raw_index in range(post_trim_start, stop):
            index = raw_index - self.startup_trim_samples
            value = values[raw_index - start]
            point = (index, value)
            if self.first is None:
                self.first = point
            self.latest = point
            if not self.buckets or (
                self.buckets[-1]["stop"] - self.buckets[-1]["start"]
                >= self.bucket_width
            ):
                is_nan = np.isnan(value)
                self.buckets.append(
                    {
                        "start": index,
                        "stop": index + 1,
                        "minimum": None if is_nan else point,
                        "maximum": None if is_nan else point,
                        "nan": point if is_nan else None,
                    }
                )
            else:
                bucket = self.buckets[-1]
                bucket["stop"] = index + 1
                if np.isnan(value):
                    if bucket["nan"] is None:
                        bucket["nan"] = point
                else:
                    bucket["minimum"] = self._select(
                        bucket["minimum"], point, minimum=True
                    )
                    bucket["maximum"] = self._select(
                        bucket["maximum"], point, minimum=False
                    )
            while self._materialized_point_count() > self.max_points:
                if len(self.buckets) == 1:
                    self._drop_auxiliary_nan_marker()
                    break
                self._compact()
        self.raw_sample_count = stop

    def records(self):
        by_index = {}
        for bucket in self.buckets:
            for point in (bucket["minimum"], bucket["maximum"], bucket["nan"]):
                if point is not None:
                    by_index[point[0]] = point[1]
        for point in (self.first, self.latest):
            if point is not None:
                by_index[point[0]] = point[1]
        return sorted(by_index.items())


def test_append_grows_geometrically_and_preserves_samples():
    accumulator = StreamingWaveformAccumulator(max_points=8)
    accumulator.begin(sample_rate=4.0, startup_trim_samples=0)

    accumulator.append(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    first_capacity = accumulator.capacity
    accumulator.append(np.array([4.0, 5.0, 6.0], dtype=np.float32))

    assert first_capacity >= 3
    assert accumulator.capacity >= 6
    assert accumulator.capacity & (accumulator.capacity - 1) == 0
    assert accumulator.raw_sample_count == 6
    np.testing.assert_array_equal(
        accumulator.raw_view(),
        np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
    )


def test_startup_trim_changes_display_count_not_raw_count():
    accumulator = StreamingWaveformAccumulator(max_points=8)
    accumulator.begin(sample_rate=2.0, startup_trim_samples=2)
    accumulator.append(np.arange(8, dtype=np.float32))

    snapshot = accumulator.snapshot()

    assert accumulator.raw_sample_count == 8
    assert accumulator.display_sample_count == 6
    assert snapshot.sample_stop == 6
    assert snapshot.time[-1] == pytest.approx(2.5)


@pytest.mark.parametrize(
    "samples, earlier, later",
    [
        ([0, 5, -4, 1, -7, 6, 0, 1, 2], 5.0, -4.0),
        ([0, -5, 4, 1, 7, -6, 0, 1, 2], -5.0, 4.0),
    ],
)
def test_snapshot_preserves_min_max_temporal_order(samples, earlier, later):
    accumulator = StreamingWaveformAccumulator(max_points=8)
    accumulator.begin(sample_rate=1.0, startup_trim_samples=0)
    accumulator.append(np.array(samples, dtype=np.float32))

    snapshot = accumulator.snapshot()

    assert len(snapshot.time) <= 8
    assert snapshot.time[0] == 0.0
    assert snapshot.time[-1] == 8.0
    earlier_position = np.flatnonzero(snapshot.amplitude == earlier)[0]
    later_position = np.flatnonzero(snapshot.amplitude == later)[0]
    assert earlier_position < later_position
    assert list(snapshot.time) == sorted(snapshot.time)


def test_many_appends_never_emit_more_than_the_cap():
    accumulator = StreamingWaveformAccumulator(max_points=16)
    accumulator.begin(sample_rate=10.0, startup_trim_samples=0)
    for start in range(0, 10_000, 7):
        accumulator.append(
            np.arange(start, min(start + 7, 10_000), dtype=np.float32)
        )
        assert len(accumulator.snapshot().time) <= 16


def test_snapshot_keeps_first_and_latest_samples_across_compaction():
    accumulator = StreamingWaveformAccumulator(max_points=6)
    accumulator.begin(sample_rate=2.0, startup_trim_samples=0)
    accumulator.append(np.arange(33, dtype=np.float32))

    snapshot = accumulator.snapshot()

    assert snapshot.time[0] == 0.0
    assert snapshot.amplitude[0] == 0.0
    assert snapshot.time[-1] == 16.0
    assert snapshot.amplitude[-1] == 32.0


def test_snapshot_keeps_every_final_bucket_extremum_and_both_endpoints():
    accumulator = StreamingWaveformAccumulator(max_points=4)
    accumulator.begin(sample_rate=1.0, startup_trim_samples=0)
    accumulator.append(
        np.array([1.0, 0.0, -19.0, 8.0, 14.0, 15.0], dtype=np.float32)
    )

    snapshot = accumulator.snapshot()
    actual = dict(zip(snapshot.time.astype(int), snapshot.amplitude))
    expected = dict(accumulator._envelope_records())

    assert len(snapshot.time) <= 4
    assert actual.keys() == expected.keys()
    for index, amplitude in expected.items():
        assert actual[index] == amplitude


def test_identical_bucket_extrema_are_emitted_only_once():
    accumulator = StreamingWaveformAccumulator(max_points=4)
    accumulator.begin(sample_rate=1.0, startup_trim_samples=0)
    accumulator.append(np.full(20, 3.0, dtype=np.float32))

    snapshot = accumulator.snapshot()

    assert len(snapshot.time) <= 4
    assert len(snapshot.time) == len(np.unique(snapshot.time))


def test_odd_tail_bucket_absorbs_later_samples_at_compacted_width():
    accumulator = StreamingWaveformAccumulator(max_points=4)
    accumulator.begin(sample_rate=1.0, startup_trim_samples=0)
    accumulator.append(np.array([0, 1, 2, 3, 4], dtype=np.float32))
    assert len(accumulator.snapshot().time) <= 4

    accumulator.append(np.array([-20, 30, 7], dtype=np.float32))
    snapshot = accumulator.snapshot()

    assert -20.0 in snapshot.amplitude
    assert 30.0 in snapshot.amplitude
    assert snapshot.time[-1] == 7.0


def test_non_finite_values_are_preserved_in_the_envelope():
    accumulator = StreamingWaveformAccumulator(max_points=8)
    accumulator.begin(sample_rate=1.0, startup_trim_samples=0)
    values = np.concatenate(
        (
            np.array([0.0, np.nan, 2.0, np.inf, -np.inf, 5.0], dtype=np.float32),
            np.arange(6, 100, dtype=np.float32),
        )
    )
    accumulator.append(values)

    amplitude = accumulator.snapshot().amplitude

    assert np.isnan(amplitude).any()
    assert np.isposinf(amplitude).any()
    assert np.isneginf(amplitude).any()


def test_minimum_cap_never_discards_bucket_extrema_for_auxiliary_nan():
    accumulator = StreamingWaveformAccumulator(max_points=4)
    accumulator.begin(sample_rate=1.0, startup_trim_samples=0)
    accumulator.append(
        np.array([5.0, np.nan, -np.inf, np.inf, 6.0], dtype=np.float32)
    )

    snapshot = accumulator.snapshot()

    assert len(snapshot.time) <= 4
    assert snapshot.amplitude[0] == 5.0
    assert snapshot.amplitude[-1] == 6.0
    assert np.isneginf(snapshot.amplitude).any()
    assert np.isposinf(snapshot.amplitude).any()


def test_snapshot_uses_only_envelope_state(monkeypatch):
    accumulator = StreamingWaveformAccumulator(max_points=8)
    accumulator.begin(sample_rate=1.0, startup_trim_samples=0)
    accumulator.append(np.arange(100, dtype=np.float32))

    def fail_if_raw_history_is_read():
        raise AssertionError("snapshot rescanned raw history")

    monkeypatch.setattr(accumulator, "raw_view", fail_if_raw_history_is_read)

    assert len(accumulator.snapshot().time) <= 8


def test_raw_view_is_read_only_and_clear_releases_capacity():
    accumulator = StreamingWaveformAccumulator(max_points=8)
    accumulator.begin(sample_rate=1.0, startup_trim_samples=0)
    accumulator.append(np.arange(5, dtype=np.float32))

    assert not accumulator.raw_view().flags.writeable

    accumulator.clear()

    assert accumulator.capacity == 0
    assert accumulator.raw_sample_count == 0
    assert accumulator.display_sample_count == 0
    assert accumulator.raw_view().size == 0


def test_display_only_mode_keeps_bounded_snapshot_without_raw_history():
    accumulator = StreamingWaveformAccumulator(max_points=8, retain_raw=False)
    accumulator.begin(sample_rate=2.0, startup_trim_samples=2)

    accumulator.append(np.arange(12, dtype=np.float32))

    snapshot = accumulator.snapshot()
    assert accumulator.capacity == 0
    assert accumulator.raw_sample_count == 12
    assert accumulator.display_sample_count == 10
    assert len(snapshot.time) <= 8
    assert snapshot.sample_stop == 10
    assert snapshot.time[0] == 0.0
    assert snapshot.amplitude[0] == 2.0
    assert snapshot.time[-1] == 4.5
    assert snapshot.amplitude[-1] == 11.0


def test_display_only_raw_view_is_explicitly_empty_and_read_only():
    accumulator = StreamingWaveformAccumulator(max_points=8, retain_raw=False)
    accumulator.begin(sample_rate=1.0, startup_trim_samples=0)
    accumulator.append(np.arange(5, dtype=np.float32))

    raw = accumulator.raw_view()

    assert raw.size == 0
    assert not raw.flags.writeable
    assert accumulator.capacity == 0
    assert accumulator.raw_sample_count == 5


def test_default_mode_still_retains_raw_history():
    accumulator = StreamingWaveformAccumulator(max_points=8)
    accumulator.begin(sample_rate=1.0, startup_trim_samples=0)
    values = np.arange(5, dtype=np.float32)

    accumulator.append(values)

    assert accumulator.capacity >= values.size
    np.testing.assert_array_equal(accumulator.raw_view(), values)


def test_600_second_curve_reaches_complete_post_trim_duration():
    sample_rate = 1_000
    trim = 137
    display_samples = sample_rate * 600
    accumulator = StreamingWaveformAccumulator(max_points=4_000)
    accumulator.begin(
        sample_rate=sample_rate,
        startup_trim_samples=trim,
    )

    total = display_samples + trim
    for start in range(0, total, 2_048):
        stop = min(start + 2_048, total)
        chunk = np.sin(np.arange(start, stop, dtype=np.float32) / 17.0)
        accumulator.append(chunk)

    class _UnreadableRawStorage:
        def __init__(self, size):
            self.size = size

        def __getitem__(self, key):
            raise AssertionError(f"snapshot read raw history with key {key!r}")

        def __array__(self, dtype=None, copy=None):
            raise AssertionError("snapshot materialized raw history")

        def __iter__(self):
            raise AssertionError("snapshot iterated over raw history")

    unreadable_raw = _UnreadableRawStorage(accumulator.capacity)
    accumulator._raw = unreadable_raw
    with pytest.raises(AssertionError, match="read raw history"):
        unreadable_raw[:]
    with pytest.raises(AssertionError, match="materialized raw history"):
        np.asarray(unreadable_raw)
    snapshot = accumulator.snapshot()

    assert accumulator.raw_sample_count == total
    assert accumulator.display_sample_count == display_samples
    assert len(snapshot.time) <= 4_000
    assert snapshot.sample_stop == display_samples
    assert snapshot.time[-1] == pytest.approx(
        (display_samples - 1) / sample_rate
    )
    assert not hasattr(accumulator, "streaming_buffer_multi")
    assert not hasattr(accumulator, "_chunks")


@pytest.mark.parametrize("max_points", [True, 3, 4.5])
def test_max_points_must_be_an_integer_of_at_least_four(max_points):
    with pytest.raises(ValueError):
        StreamingWaveformAccumulator(max_points=max_points)


@pytest.mark.parametrize(
    "sample_rate, startup_trim_samples",
    [(0.0, 0), (np.inf, 0), (1.0, -1), (1.0, 1.5), (True, 0)],
)
def test_begin_validates_session_parameters(sample_rate, startup_trim_samples):
    accumulator = StreamingWaveformAccumulator(max_points=8)

    with pytest.raises(ValueError):
        accumulator.begin(
            sample_rate=sample_rate,
            startup_trim_samples=startup_trim_samples,
        )


@pytest.mark.parametrize(
    "chunk",
    [
        np.array([], dtype=np.float32),
        np.array([1.0], dtype=np.float64),
        np.array([[1.0]], dtype=np.float32),
        [1.0],
    ],
)
def test_append_requires_normalized_nonempty_float32_vector(chunk):
    accumulator = StreamingWaveformAccumulator(max_points=8)
    accumulator.begin(sample_rate=1.0, startup_trim_samples=0)

    with pytest.raises(ValueError):
        accumulator.append(chunk)


@pytest.mark.parametrize("retain_raw", [True, False])
@pytest.mark.parametrize(
    "max_points, trim, values, chunk_sizes",
    [
        (
            4,
            3,
            np.asarray(
                [
                    90,
                    91,
                    92,
                    5,
                    np.nan,
                    5,
                    -2,
                    -2,
                    np.inf,
                    -np.inf,
                    np.nan,
                    7,
                    7,
                    1,
                    4,
                ],
                dtype=np.float32,
            ),
            (2, 1, 4, 3, 5),
        ),
        (
            9,
            5,
            np.where(
                np.arange(257) % 31 == 0,
                np.nan,
                np.round(np.sin(np.arange(257) / 3.0) * 4),
            ).astype(np.float32),
            (1, 17, 2, 64, 3, 89, 81),
        ),
    ],
)
def test_chunk_batched_envelope_matches_scalar_reference_after_every_append(
    retain_raw, max_points, trim, values, chunk_sizes
):
    values = values.copy()
    if values.size > 200:
        values[111] = np.inf
        values[199] = -np.inf
    accumulator = StreamingWaveformAccumulator(
        max_points=max_points, retain_raw=retain_raw
    )
    accumulator.begin(sample_rate=8.0, startup_trim_samples=trim)
    reference = _ScalarEnvelopeReference(
        max_points=max_points, startup_trim_samples=trim
    )

    start = 0
    for chunk_size in chunk_sizes:
        chunk = values[start : start + chunk_size]
        accumulator.append(chunk)
        reference.append(chunk)
        start += chunk_size

        snapshot = accumulator.snapshot()
        expected = reference.records()
        np.testing.assert_array_equal(
            snapshot.time * 8.0,
            np.asarray([point[0] for point in expected], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            snapshot.amplitude,
            np.asarray([point[1] for point in expected], dtype=np.float32),
        )
        assert snapshot.sample_stop == max(0, start - trim)
        assert not snapshot.time.flags.writeable
        assert not snapshot.amplitude.flags.writeable

    assert start == values.size

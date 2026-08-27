import re
import statistics
import time

import numpy as np
import pytest

from ui.sequence.multichannel_waveform_session import MultichannelWaveformSession


def test_session_projects_startup_trim_independently_in_channel_order():
    session = MultichannelWaveformSession(max_points=4_000)
    session.begin(channels=(0, 2), sample_rate=48_000, startup_trim_samples=4)

    session.append(
        np.asarray(
            [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]],
            dtype=np.float32,
        )
    )

    snapshots = session.snapshots()
    assert tuple(snapshots) == (0, 2)
    np.testing.assert_array_equal(snapshots[0].amplitude, [5])
    np.testing.assert_array_equal(snapshots[2].amplitude, [50])
    assert snapshots[0].sample_stop == 1
    assert snapshots[2].sample_stop == 1


@pytest.mark.parametrize(
    "chunk, actual_shape",
    [
        (np.ones((3, 1), dtype=np.float32), (3, 1)),
        (np.ones((3, 3), dtype=np.float32), (3, 3)),
        (np.ones(3, dtype=np.float32), (3,)),
        (np.ones((1, 2, 1), dtype=np.float32), (1, 2, 1)),
    ],
)
def test_session_rejects_shapes_that_do_not_exactly_match_channels(
    chunk, actual_shape
):
    session = MultichannelWaveformSession(max_points=8)
    session.begin(channels=(0, 2), sample_rate=48_000, startup_trim_samples=0)

    with pytest.raises(
        ValueError,
        match=rf"expected 2 channels.*actual shape {re.escape(str(actual_shape))}",
    ):
        session.append(chunk)


def test_session_accepts_one_dimensional_chunk_only_for_one_channel():
    session = MultichannelWaveformSession(max_points=8)
    session.begin(channels=(3,), sample_rate=2, startup_trim_samples=0)

    session.append(np.asarray([1, 2, 3], dtype=np.float32))

    snapshots = session.snapshots()
    assert tuple(snapshots) == (3,)
    np.testing.assert_array_equal(snapshots[3].amplitude, [1, 2, 3])


def test_session_requires_normalized_nonempty_float32_chunks():
    session = MultichannelWaveformSession(max_points=8)
    session.begin(channels=(0, 2), sample_rate=48_000, startup_trim_samples=0)

    for chunk in (
        np.empty((0, 2), dtype=np.float32),
        np.ones((3, 2), dtype=np.float64),
        [[1.0, 2.0]],
    ):
        with pytest.raises(ValueError, match="expected 2 channels"):
            session.append(chunk)


def test_session_keeps_channel_envelopes_independent_across_chunks():
    session = MultichannelWaveformSession(max_points=8)
    session.begin(channels=(0, 2), sample_rate=10, startup_trim_samples=0)

    session.append(np.asarray([[1, 100], [2, 200]], dtype=np.float32))
    session.append(np.asarray([[3, 300], [4, 400]], dtype=np.float32))

    snapshots = session.snapshots()
    np.testing.assert_array_equal(snapshots[0].amplitude, [1, 2, 3, 4])
    np.testing.assert_array_equal(snapshots[2].amplitude, [100, 200, 300, 400])
    np.testing.assert_array_equal(snapshots[0].time, snapshots[2].time)


def test_clear_releases_every_display_accumulator_and_session_mapping():
    session = MultichannelWaveformSession(max_points=8)
    session.begin(channels=(0, 2), sample_rate=10, startup_trim_samples=0)
    session.append(np.asarray([[1, 10], [2, 20]], dtype=np.float32))
    accumulators = tuple(session._accumulators.values())

    session.clear()

    assert session.snapshots() == {}
    for accumulator in accumulators:
        assert accumulator.capacity == 0
        assert accumulator.raw_sample_count == 0
        assert accumulator.raw_view().size == 0


def test_600_second_multichannel_projection_is_bounded_and_retains_no_raw_history():
    sample_rate = 1_000
    trim = 137
    display_samples = sample_rate * 600
    session = MultichannelWaveformSession(max_points=4_000)
    session.begin(channels=(0, 2), sample_rate=sample_rate, startup_trim_samples=trim)

    total = display_samples + trim
    for start in range(0, total, 2_048):
        stop = min(start + 2_048, total)
        indices = np.arange(start, stop, dtype=np.float32)
        session.append(
            np.column_stack((np.sin(indices / 17.0), np.cos(indices / 19.0))).astype(
                np.float32,
                copy=False,
            )
        )

    snapshots = session.snapshots()
    for snapshot in snapshots.values():
        assert len(snapshot.time) <= 4_000
        assert snapshot.sample_stop == display_samples
        assert snapshot.time[-1] == pytest.approx((display_samples - 1) / sample_rate)
    for accumulator in session._accumulators.values():
        assert accumulator.capacity == 0
        assert accumulator.raw_sample_count == total
        assert accumulator.raw_view().size == 0
    assert not hasattr(session, "streaming_buffer_multi")
    assert not hasattr(session, "_chunks")


def test_append_before_begin_fails_without_implicit_channel_state():
    session = MultichannelWaveformSession(max_points=8)

    with pytest.raises(RuntimeError, match="begin"):
        session.append(np.ones((1, 1), dtype=np.float32))


def test_16_channel_48khz_live_chunks_leave_processing_headroom():
    """Envelope work leaves at least 40% of a 2048-frame block for WAV/Qt."""
    sample_rate = 48_000
    frames = 2_048
    channel_count = 16
    session = MultichannelWaveformSession(max_points=4_000)
    session.begin(
        channels=tuple(range(channel_count)),
        sample_rate=sample_rate,
        startup_trim_samples=0,
    )
    indices = np.arange(frames, dtype=np.float32)
    chunk = np.column_stack(
        [np.sin(indices / (17.0 + channel)) for channel in range(channel_count)]
    ).astype(np.float32, copy=False)

    durations = []
    for _ in range(9):
        started = time.perf_counter()
        session.append(chunk)
        durations.append(time.perf_counter() - started)

    block_period = frames / sample_rate
    assert statistics.median(durations) < block_period * 0.60

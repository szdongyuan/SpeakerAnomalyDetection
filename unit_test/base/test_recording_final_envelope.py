import numpy as np
import pytest

from base.recording_final_envelope import FinalEnvelopeAccumulator
from base.recording_waveform_preparation import prepare_waveform_display_data
from base.wav_pcm24 import quantize_pcm24
from consts.recording_preview_consts import MAIN_RECORDING_FINAL_MAX_POINTS


def test_arbitrary_chunks_match_batch_display():
    rng = np.random.default_rng(71)
    data = quantize_pcm24(rng.uniform(-1, 1, (113, 3)).astype(np.float32))
    acc = FinalEnvelopeAccumulator(total_frames=113, channel_count=3, max_points=12)
    for block in np.split(data, [1, 8, 49, 50, 110]):
        acc.append(block)
    for column, (indices, values) in enumerate(acc.finish()):
        expected_time, expected_values = prepare_waveform_display_data(
            data[:, column], 44100, max_points=12
        )
        np.testing.assert_array_equal(indices.astype(np.float64) / 44100, expected_time)
        np.testing.assert_array_equal(values, expected_values)
        assert len(indices) <= 12


@pytest.mark.parametrize("field", ["total_frames", "channel_count", "max_points"])
@pytest.mark.parametrize("invalid", [True, np.bool_(False), 1.5, "12", None, 0, -1])
def test_invalid_constructor_arguments(field, invalid):
    kwargs = dict(total_frames=20, channel_count=2, max_points=12)
    kwargs[field] = invalid
    with pytest.raises(ValueError, match=field):
        FinalEnvelopeAccumulator(**kwargs)


@pytest.mark.parametrize("max_points", [1, 2, 3])
def test_point_budget_requires_four_points(max_points):
    with pytest.raises(ValueError, match="max_points"):
        FinalEnvelopeAccumulator(20, 2, max_points)


@pytest.mark.parametrize("total_frames", [4, 20])
@pytest.mark.parametrize("invalid", [
    np.zeros((2, 2), dtype=np.float64),
    np.zeros(2, dtype=np.float32),
    np.zeros((2, 1), dtype=np.float32),
    np.zeros((1, 2, 1), dtype=np.float32),
    np.full((1, 2), np.nan, dtype=np.float32),
    np.full((1, 2), np.inf, dtype=np.float32),
    np.full((1, 2), -np.inf, dtype=np.float32),
    [[0.0, 0.0]],
])
def test_invalid_blocks_are_rejected_without_consuming_frames(total_frames, invalid):
    acc = FinalEnvelopeAccumulator(total_frames, 2, 4)
    with pytest.raises(ValueError):
        acc.append(invalid)
    data = np.zeros((total_frames, 2), dtype=np.float32)
    acc.append(data)
    assert len(acc.finish()) == 2


@pytest.mark.parametrize("total_frames", [4, 20])
def test_incomplete_and_excess_frames_cannot_finish_successfully(total_frames):
    acc = FinalEnvelopeAccumulator(total_frames, 2, 4)
    with pytest.raises(ValueError, match="frames"):
        acc.finish()
    acc.append(np.zeros((total_frames - 1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="frames"):
        acc.finish()
    with pytest.raises(ValueError, match="frames"):
        acc.append(np.zeros((2, 2), dtype=np.float32))
    acc.append(np.zeros((1, 2), dtype=np.float32))
    result = acc.finish()
    with pytest.raises(ValueError, match="frames"):
        acc.append(np.zeros((1, 2), dtype=np.float32))
    for before, after in zip(result, acc.finish()):
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])


def assert_matches_batch(data, max_points, chunks):
    original = data.copy()
    acc = FinalEnvelopeAccumulator(len(data), data.shape[1], max_points)
    for chunk in chunks:
        acc.append(chunk)
    result = acc.finish()
    assert isinstance(result, tuple)
    assert len(result) == data.shape[1]
    for column, (indices, values) in enumerate(result):
        expected_time, expected_values = prepare_waveform_display_data(
            data[:, column], 44100, max_points=max_points
        )
        assert indices.dtype == np.int64
        assert values.dtype == np.float32
        assert 1 <= len(indices) <= max_points
        assert indices[0] == 0
        assert indices[-1] == len(data) - 1
        assert np.all(np.diff(indices) > 0)
        np.testing.assert_array_equal(indices.astype(np.float64) / 44100, expected_time)
        np.testing.assert_array_equal(values, expected_values)
    np.testing.assert_array_equal(data, original)
    return result


@pytest.mark.parametrize("count,point_limit", [(1, 12), (12, 12), (13, 12), (113, 12), (113, 4), (113, 13), (25031, 480)])
@pytest.mark.parametrize("pattern", ["random", "constant", "impulses", "tied_extrema"])
def test_edge_cases_and_different_chunkings_match_batch(count, point_limit, pattern):
    rng = np.random.default_rng(712)
    if pattern == "random":
        samples = rng.uniform(-1, 1, (count, 5)).astype(np.float32)
    elif pattern == "constant":
        samples = np.full((count, 5), 0.25, dtype=np.float32)
    elif pattern == "impulses":
        samples = np.zeros((count, 5), dtype=np.float32)
        for column in range(5):
            samples[(column * 3) % count, column] = 0.75
            samples[(count - 1 - column * 2) % count, column] = -0.875
    else:
        samples = np.resize(np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32), (count, 5))
    data = quantize_pcm24(samples)
    data.setflags(write=False)
    chunk_size = max(1, count // 17)
    chunks = [data[start:start + chunk_size] for start in range(0, count, chunk_size)]
    split = assert_matches_batch(data, point_limit, chunks)
    whole = assert_matches_batch(data, point_limit, [data])
    for split_column, whole_column in zip(split, whole):
        np.testing.assert_array_equal(split_column[0], whole_column[0])
        np.testing.assert_array_equal(split_column[1], whole_column[1])


def test_tied_extrema_across_fragments_keep_the_earliest_index():
    # N=31, P=8 gives width=11. Repeated extrema straddle append boundaries.
    data = np.zeros((31, 1), dtype=np.float32)
    data[[2, 5, 10], 0] = -0.75
    data[[3, 6, 9], 0] = 0.75
    result = assert_matches_batch(data, 8, np.split(data, [4, 7, 11, 22, 30]))
    np.testing.assert_array_equal(result[0][0], [0, 2, 3, 11, 22, 30])


@pytest.mark.parametrize("count", [4, 30])
def test_empty_blocks_and_noncontiguous_inputs_are_supported(count):
    rng = np.random.default_rng(1)
    data = quantize_pcm24(rng.uniform(-1, 1, (count, 4)).astype(np.float32))[:, ::2]
    assert not data.flags.c_contiguous
    empty = np.empty((0, 2), dtype=np.float32)
    assert_matches_batch(data, 4, [empty, data[:3], empty, data[3:], empty])


@pytest.mark.parametrize("count", [4, 30])
def test_input_and_finished_arrays_have_independent_ownership(count):
    data = np.full((count, 2), 0.25, dtype=np.float32)
    acc = FinalEnvelopeAccumulator(count, 2, 4)
    acc.append(data)
    data[:] = -0.25
    result = acc.finish()
    assert all(np.all(values == 0.25) for _, values in result)
    result[0][0][:] = 0
    result[0][1][:] = 0
    repeated = acc.finish()
    assert repeated[0][0][-1] == count - 1
    assert np.all(repeated[0][1] == 0.25)


def test_numpy_integer_arguments_are_accepted():
    acc = FinalEnvelopeAccumulator(np.int64(1), np.int32(2), np.uint32(4))
    acc.append(np.zeros((1, 2), dtype=np.float32))
    assert len(acc.finish()) == 2


def test_large_recording_has_bounded_storage_from_creation():
    channels = 5
    point_limit = MAIN_RECORDING_FINAL_MAX_POINTS
    byte_budget = ((point_limit - 2) // 2) * channels * 24 + channels * 8
    for count in (44_100 * 600, 44_100 * 60_000):
        acc = FinalEnvelopeAccumulator(count, channels)
        arrays = {key: value for key, value in vars(acc).items() if isinstance(value, np.ndarray)}
        initial_bytes = sum(value.nbytes for value in arrays.values())
        assert initial_bytes <= byte_budget
        for _ in range(3):
            acc.append(np.zeros((1024, channels), dtype=np.float32))
        assert sum(value.nbytes for value in vars(acc).values() if isinstance(value, np.ndarray)) == initial_bytes
        assert all(vars(acc)[key] is value for key, value in arrays.items())
        assert not any(isinstance(value, (list, tuple, dict)) for value in vars(acc).values())

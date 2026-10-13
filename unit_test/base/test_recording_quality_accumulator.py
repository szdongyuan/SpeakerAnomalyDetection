"""Streaming quality uses saved samples and the existing validation gate."""

import numpy as np
import pytest

from base.recording_quality_accumulator import RecordingQualityAccumulator
from base.recording_settings import validate_recorded_audio, validate_recorded_audio_metrics
from base.wav_pcm24 import quantize_pcm24


def thresholds(**overrides):
    return dict(dict(enabled=True, min_peak=1e-4, min_rms_dbfs=-65.0,
                     min_variance=1e-10), **overrides)


def test_normal_quantized_chunks_match_batch():
    data = quantize_pcm24(np.random.default_rng(71).uniform(-1, 1, (113, 3)).astype(np.float32))
    acc = RecordingQualityAccumulator(channel_count=3, range_max=None)
    for block in np.split(data, [1, 8, 49, 50, 110]):
        acc.append(block)
    result = acc.finish(thresholds())
    mono = data.mean(axis=1, dtype=np.float32)
    assert result.count == len(data)
    assert result.status == "passed"
    assert (True, result.reason, result.detail) == validate_recorded_audio(data, thresholds())
    assert result.peak == float(np.max(np.abs(mono)))
    assert result.rms == pytest.approx(np.sqrt(np.mean(np.square(mono, dtype=np.float64))), rel=1e-14)
    assert result.variance == pytest.approx(np.var(mono, dtype=np.float64), rel=1e-14)
    assert result.voltage_finite is None


@pytest.mark.parametrize("data", [
    np.zeros((9, 2), dtype=np.float32),
    np.full((9, 2), 0.5, dtype=np.float32),
    np.array([[-1e-6, -2e-6], [1e-6, 2e-6]] * 5, dtype=np.float32),
    np.array([[0.5, -0.5], [-0.25, 0.25]] * 5, dtype=np.float32),
    np.array([[0.25, 0.5], [-0.25, -0.5]] * 5, dtype=np.float32),
])
def test_signal_failures_and_messages_match_batch(data):
    data = quantize_pcm24(data)
    acc = RecordingQualityAccumulator(2)
    for block in np.array_split(data, 4):
        acc.append(block)
    result = acc.finish(thresholds())
    ok, reason, detail = validate_recorded_audio(data, thresholds())
    assert (result.status, result.reason, result.detail) == (
        "passed" if ok else "failed", reason, detail)


def test_ve_divides_before_channel_mean():
    data = quantize_pcm24(np.random.default_rng(73).uniform(-1, 1, (113, 3)).astype(np.float32))
    mono = (data / 3.0).mean(axis=1, dtype=np.float32)
    wrong_order = data.mean(axis=1, dtype=np.float32) / 3.0
    assert np.any(mono != wrong_order)
    acc = RecordingQualityAccumulator(3, range_max=3.0)
    for block in np.split(data, [1, 8, 49, 50, 110]):
        acc.append(block)
    result = acc.finish(thresholds())
    assert result.count == len(data)
    assert result.voltage_finite is True
    assert result.peak == float(np.max(np.abs(mono)))
    assert result.rms == pytest.approx(np.sqrt(np.mean(np.square(mono, dtype=np.float64))), rel=1e-14)
    assert result.variance == pytest.approx(np.var(mono, dtype=np.float64), rel=1e-14)
    assert (result.status == "passed", result.reason, result.detail) == validate_recorded_audio(data / 3.0, thresholds())


@pytest.mark.parametrize("range_max", [None, 2.0])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_block_is_rejected_without_changing_statistics(range_max, value):
    acc = RecordingQualityAccumulator(2, range_max=range_max)
    good = np.array([[0.5, 0.25], [-0.5, -0.25]], dtype=np.float32)
    acc.append(good)
    before = acc.finish(thresholds(enabled=False))
    with pytest.raises(ValueError, match="finite"):
        acc.append(np.array([[value, 0.5]], dtype=np.float32))
    assert acc.finish(thresholds(enabled=False)) == before


def test_ve_checks_voltage_mono_even_when_normalized_mean_is_finite():
    data = np.full((2, 3), np.finfo(np.float32).max, dtype=np.float32)
    assert np.isfinite((data / 4.0).mean(axis=1, dtype=np.float32)).all()
    acc = RecordingQualityAccumulator(3, range_max=4.0)
    with pytest.raises(ValueError, match="non-finite mono voltage"):
        acc.append(data)
    assert acc.finish(thresholds(enabled=False)).count == 0


def test_normalized_mono_overflow_is_rejected():
    acc = RecordingQualityAccumulator(2, range_max=1e-39)
    with pytest.raises(ValueError, match="finite"):
        acc.append(np.full((2, 2), 0.75, dtype=np.float32))
    assert acc.finish(thresholds(enabled=False)).count == 0


@pytest.mark.parametrize("range_max", [None, 2.0])
def test_disabled_quality_is_explicitly_skipped_but_counts_samples(range_max):
    acc = RecordingQualityAccumulator(2, range_max=range_max)
    acc.append(np.zeros((5, 2), dtype=np.float32))
    result = acc.finish(thresholds(enabled=False))
    assert (result.count, result.status, result.reason, result.detail) == (5, "skipped", "", "")
    assert result.voltage_finite is (None if range_max is None else True)


def test_empty_block_and_empty_finish_preserve_batch_gate_behavior():
    acc = RecordingQualityAccumulator(2)
    acc.append(np.empty((0, 2), dtype=np.float32))
    result = acc.finish(thresholds())
    assert (result.status == "passed", result.reason, result.detail) == validate_recorded_audio([], thresholds())
    assert result.count == 0
    assert acc.finish(thresholds(enabled=False)).status == "skipped"


def test_only_caller_trimmed_samples_are_measured_and_inputs_are_not_retained():
    data = np.array([[0.875], [0.75], [0.25], [-0.25]], dtype=np.float32)
    original = data.copy()
    acc = RecordingQualityAccumulator(1)
    acc.append(data[2:])
    np.testing.assert_array_equal(data, original)
    result = acc.finish(thresholds())
    assert (result.count, result.peak, result.rms, result.variance) == (2, 0.25, 0.25, 0.0625)
    data[:] = 0
    assert acc.finish(thresholds()) == result
    assert not any(isinstance(value, (np.ndarray, list, tuple, dict)) for value in vars(acc).values())


@pytest.mark.parametrize("field, boundary", [("min_peak", 0.5), ("min_rms_dbfs", 0.0), ("min_variance", 0.25)])
@pytest.mark.parametrize("direction, passed", [(-np.inf, True), (0, True), (np.inf, False)])
def test_exact_threshold_comparison_direction(field, boundary, direction, passed):
    # +/-1 gives exactly 0 dBFS; +/-0.5 gives exact peak=0.5, variance=0.25.
    amplitude = 1.0 if field == "min_rms_dbfs" else 0.5
    data = np.array([[amplitude], [-amplitude]], dtype=np.float32)
    settings = thresholds(**{field: boundary if direction == 0 else np.nextafter(boundary, direction)})
    acc = RecordingQualityAccumulator(1)
    acc.append(data[:1])
    acc.append(data[1:])
    result = acc.finish(settings)
    assert (result.status == "passed") is passed
    assert (passed, result.reason, result.detail) == validate_recorded_audio(data, settings)


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True, np.bool_(True), None])
def test_invalid_channel_count(invalid):
    with pytest.raises(ValueError, match="channel_count"):
        RecordingQualityAccumulator(invalid)


@pytest.mark.parametrize("invalid", [0.0, -1, np.nan, np.inf, True, "2"])
def test_invalid_range_max(invalid):
    with pytest.raises(ValueError, match="range_max"):
        RecordingQualityAccumulator(1, range_max=invalid)


@pytest.mark.parametrize("invalid", [
    [[0.0, 0.0]], np.zeros((2, 2), dtype=np.float64),
    np.zeros((2,), dtype=np.float32), np.zeros((2, 1), dtype=np.float32),
    np.zeros((0, 3), dtype=np.float32),
])
def test_invalid_blocks(invalid):
    with pytest.raises(ValueError, match="frames-first float32"):
        RecordingQualityAccumulator(2).append(invalid)


def test_shared_gate_preserves_disabled_short_circuit_and_empty_messages():
    assert validate_recorded_audio(object(), thresholds(enabled=False)) == (True, "", "")
    assert validate_recorded_audio_metrics(0, None, None, None, thresholds(enabled=False)) == (True, "", "")
    assert validate_recorded_audio_metrics(0, None, None, None, thresholds()) == (
        False, "录音缓冲区为空，未采集到任何样本。", "samples=0")
    assert validate_recorded_audio(object(), None) == (True, "", "")
    assert validate_recorded_audio_metrics(0, None, None, None, None) == (True, "", "")


def test_large_dc_small_variance_uses_stable_m2_and_fixed_size_state():
    # Cancellation in E[x*x] - E[x]**2 would lose this variance.
    data = np.array([[0.75], [np.nextafter(np.float32(0.75), np.float32(1.0))]], dtype=np.float32)
    acc = RecordingQualityAccumulator(1)
    for _ in range(300):
        acc.append(data[:1])
        acc.append(data[1:])
    result = acc.finish(thresholds())
    assert result.count == 600
    assert result.variance == pytest.approx(np.var(data[:, 0], dtype=np.float64), rel=1e-9, abs=0)
    assert not any(isinstance(value, (np.ndarray, list, tuple, dict)) for value in vars(acc).values())


def test_roundoff_boundary_uses_each_measured_variance_without_threshold_tolerance():
    """Record a real reduction-order difference, not a promise of bit equality.

    NumPy 1.26 / Windows: seed 71 gives batch=0.10092705095462548,
    stream=0.1009270509546255. At the latter threshold batch fails and
    streaming passes. Other reduction implementations may round identically;
    in every case each gate must compare its own measured value strictly.
    """
    data = quantize_pcm24(np.random.default_rng(71).uniform(-1, 1, (113, 3)).astype(np.float32))
    acc = RecordingQualityAccumulator(3)
    for block in np.split(data, [1, 8, 49, 50, 110]):
        acc.append(block)
    batch_variance = float(np.var(data.mean(axis=1, dtype=np.float32), dtype=np.float64))
    stream_variance = acc.finish(thresholds()).variance
    assert stream_variance == pytest.approx(batch_variance, rel=1e-14)
    for boundary in (batch_variance, stream_variance,
                     np.nextafter(max(batch_variance, stream_variance), np.inf)):
        settings = thresholds(min_variance=boundary)
        assert validate_recorded_audio(data, settings)[0] == (batch_variance >= boundary)
        assert (acc.finish(settings).status == "passed") == (stream_variance >= boundary)

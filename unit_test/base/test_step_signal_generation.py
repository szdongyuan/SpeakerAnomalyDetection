import numpy as np
import pytest

from base.pre_processing.swept_sine_chirps import StimulusSignal


def test_generate_steps_uses_half_open_discrete_time_for_first_segment():
    sample_rate = 48000
    waveform, returned_rate = StimulusSignal.generate_steps(
        start_freq=1000,
        stop_freq=2000,
        total_time=0.006,
        sample_rate=sample_rate,
        num_steps=2,
        repeat_times=1,
        stimulus_type="linear",
    )

    assert returned_rate == sample_rate
    assert len(waveform) == 288
    n = np.arange(144, dtype=float)
    expected_first = np.sin(2.0 * np.pi * 1000.0 * n / sample_rate)
    np.testing.assert_allclose(waveform[:144], expected_first, rtol=1e-12, atol=1e-12)


def test_generate_steps_advances_boundary_phase_by_full_sample_count():
    sample_rate = 48000
    waveform, _ = StimulusSignal.generate_steps(
        start_freq=1000,
        stop_freq=2000,
        total_time=0.006,
        sample_rate=sample_rate,
        num_steps=2,
        repeat_times=1,
        stimulus_type="linear",
    )

    n = np.arange(144, dtype=float)
    boundary_phase = 2.0 * np.pi * 1000.0 * 144 / sample_rate
    expected_second = np.sin(boundary_phase + 2.0 * np.pi * 2000.0 * n / sample_rate)
    np.testing.assert_allclose(waveform[144:288], expected_second, rtol=1e-12, atol=1e-12)


def test_generate_steps_floors_non_integer_step_sample_count_and_preserves_length():
    sample_rate = 48000
    waveform, returned_rate = StimulusSignal.generate_steps(
        start_freq=750,
        stop_freq=1250,
        total_time=0.0051,
        sample_rate=sample_rate,
        num_steps=2,
        repeat_times=1,
        stimulus_type="linear",
    )

    assert returned_rate == sample_rate
    assert len(waveform) == 244
    n = np.arange(122, dtype=float)
    expected_first = np.sin(2.0 * np.pi * 750.0 * n / sample_rate)
    boundary_phase = 2.0 * np.pi * 750.0 * 122 / sample_rate
    expected_second = np.sin(boundary_phase + 2.0 * np.pi * 1250.0 * n / sample_rate)
    np.testing.assert_allclose(waveform[:122], expected_first, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(waveform[122:], expected_second, rtol=1e-12, atol=1e-12)


def test_generate_steps_preserves_log_spaced_frequency_order():
    sample_rate = 48000
    waveform, _ = StimulusSignal.generate_steps(
        start_freq=100,
        stop_freq=10000,
        total_time=0.03,
        sample_rate=sample_rate,
        num_steps=3,
        repeat_times=1,
        stimulus_type="log",
    )

    num_samples = 480
    frequencies = np.logspace(np.log10(100.0), np.log10(10000.0), 3)
    phase_position = 0.0
    sample_offsets = np.arange(num_samples, dtype=float)
    for index, frequency in enumerate(frequencies):
        start = index * num_samples
        end = start + num_samples
        expected = np.sin(phase_position + 2.0 * np.pi * frequency * sample_offsets / sample_rate)
        np.testing.assert_allclose(waveform[start:end], expected, rtol=1e-12, atol=1e-12)
        phase_position = (phase_position + 2.0 * np.pi * frequency * num_samples / sample_rate) % (
            2.0 * np.pi
        )


def test_generate_steps_repetitions_are_verbatim_copies():
    waveform, _ = StimulusSignal.generate_steps(
        start_freq=1000,
        stop_freq=2000,
        total_time=0.012,
        sample_rate=48000,
        num_steps=2,
        repeat_times=2,
        stimulus_type="linear",
    )

    assert len(waveform) == 576
    first_repetition = waveform[:288]
    second_repetition = waveform[288:]
    np.testing.assert_array_equal(second_repetition, first_repetition)


def test_generate_steps_zero_total_time_contract_is_preserved():
    waveform, sample_rate = StimulusSignal.generate_steps(total_time=0, sample_rate=48000)

    assert waveform == []
    assert sample_rate == 48000


def test_generate_steps_invalid_step_type_message_is_preserved():
    with pytest.raises(Exception, match="Invalid step type\\."):
        StimulusSignal.generate_steps(total_time=1.0, stimulus_type="bad")

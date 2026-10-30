import numpy as np
import pytest

from base.data_struct.golden_sample_context import (
    create_golden_analysis_input,
    create_golden_context,
)


def test_new_golden_contexts_start_empty_and_are_independent():
    first = create_golden_context(48000)
    second = create_golden_context(44100)

    assert first is not second
    for context, sample_rate in ((first, 48000), (second, 44100)):
        assert context.sample_rate == sample_rate
        assert context.golden_recording is None
        assert context.stimulus_data is None
        assert context.stimulus_info is None
        assert context.wav_calibration_metadata is None
        assert not hasattr(context, "store_wave_data")
        assert not hasattr(context, "store_wave_data_multi")
        assert not hasattr(context, "alignment_sample_count")

    first.golden_recording = np.array([0.25], dtype=np.float32)
    first.stimulus_info = {"channels": [1]}
    first.wav_calibration_metadata = {"recorded_channels": [1]}
    assert second.golden_recording is None
    assert second.stimulus_info is None
    assert second.wav_calibration_metadata is None


def test_golden_adapter_uses_only_new_recording():
    golden_context = create_golden_context(48000)
    golden_recording = np.array([0.25, -0.5], dtype=np.float32)
    golden_context.golden_recording = golden_recording
    golden_context.stimulus_data = np.array([1.0, -1.0], dtype=np.float32)
    golden_context.stimulus_info = {"sample_rate": 48000}
    golden_context.wav_calibration_metadata = {"recorded_channels": []}

    golden_analysis_input = create_golden_analysis_input(golden_context)

    assert golden_analysis_input is not golden_context
    assert golden_analysis_input.store_wave_data is golden_recording
    assert golden_analysis_input.store_wave_data_multi is None
    assert not hasattr(golden_context, "store_wave_data")
    assert not hasattr(golden_context, "store_wave_data_multi")
    assert golden_analysis_input.stimulus_data is golden_context.stimulus_data
    assert golden_analysis_input.sample_rate == 48000
    assert golden_analysis_input.wav_calibration_metadata_authoritative is False
    assert golden_analysis_input.wav_calibration_warning_shown is False


def test_empty_context_produces_empty_analysis_input():
    analysis_input = create_golden_analysis_input(create_golden_context(48000))

    assert analysis_input.store_wave_data is None
    assert analysis_input.store_wave_data_multi is None
    assert analysis_input.stimulus_data is None
    assert analysis_input.stimulus_info is None
    assert analysis_input.wav_calibration_metadata is None
    assert analysis_input.analysis_result_dict == {}


def test_adapters_have_independent_results_and_calibration_flags():
    context = create_golden_context(48000)
    first = create_golden_analysis_input(context)
    second = create_golden_analysis_input(context)
    other_operation = create_golden_analysis_input(create_golden_context(48000))

    assert first is not second
    first.analysis_result_dict["FR"] = [True, 1.0]
    first.wav_calibration_warning_shown = True
    first.wav_calibration_metadata_authoritative = True

    for untouched in (second, other_operation):
        assert untouched.analysis_result_dict == {}
        assert untouched.analysis_result_dict is not first.analysis_result_dict
        assert untouched.wav_calibration_warning_shown is False
        assert untouched.wav_calibration_metadata_authoritative is False
    assert not hasattr(context, "analysis_result_dict")


def test_adapter_metadata_is_deep_copied_for_each_adapter():
    context = create_golden_context(48000)
    context.stimulus_info = {"segments": [{"frequencies": [80, 6000]}]}
    context.wav_calibration_metadata = {
        "recorded_channels": [{"channel": 1, "factors": [2.0]}]
    }
    first = create_golden_analysis_input(context)
    second = create_golden_analysis_input(context)

    first.stimulus_info["segments"][0]["frequencies"].append(7000)
    first.wav_calibration_metadata["recorded_channels"][0]["factors"].append(3.0)

    assert context.stimulus_info == {"segments": [{"frequencies": [80, 6000]}]}
    assert second.stimulus_info == context.stimulus_info
    assert context.wav_calibration_metadata == {
        "recorded_channels": [{"channel": 1, "factors": [2.0]}]
    }
    assert second.wav_calibration_metadata == context.wav_calibration_metadata

    context.stimulus_info["segments"][0]["frequencies"].append(9000)
    context.wav_calibration_metadata["recorded_channels"][0]["factors"].append(4.0)
    assert second.stimulus_info == {"segments": [{"frequencies": [80, 6000]}]}
    assert second.wav_calibration_metadata == {
        "recorded_channels": [{"channel": 1, "factors": [2.0]}]
    }


@pytest.mark.parametrize("sample_count", [0, 256])
def test_adapter_transfers_alignment_only_from_its_context(sample_count):
    aligned_context = create_golden_context(48000)
    aligned_context.alignment_sample_count = sample_count

    aligned_input = create_golden_analysis_input(aligned_context)
    unaligned_input = create_golden_analysis_input(create_golden_context(48000))

    assert aligned_input.alignment_sample_count == sample_count
    assert not hasattr(unaligned_input, "alignment_sample_count")

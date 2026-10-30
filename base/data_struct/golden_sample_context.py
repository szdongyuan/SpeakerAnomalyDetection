"""Operation-local golden capture data and its legacy analysis interface."""

from copy import deepcopy
from types import SimpleNamespace


def create_golden_context(sample_rate):
    return SimpleNamespace(
        sample_rate=sample_rate,
        stimulus_data=None,
        stimulus_info=None,
        golden_recording=None,
        wav_calibration_metadata=None,
    )


def create_golden_analysis_input(golden_context):
    golden_analysis_input = SimpleNamespace(
        sample_rate=golden_context.sample_rate,
        stimulus_data=golden_context.stimulus_data,
        stimulus_info=deepcopy(golden_context.stimulus_info),
        store_wave_data=golden_context.golden_recording,
        store_wave_data_multi=None,
        wav_calibration_metadata=deepcopy(golden_context.wav_calibration_metadata),
        wav_calibration_metadata_authoritative=False,
        wav_calibration_warning_shown=False,
        analysis_result_dict={},
    )
    if hasattr(golden_context, "alignment_sample_count"):
        golden_analysis_input.alignment_sample_count = golden_context.alignment_sample_count
    return golden_analysis_input

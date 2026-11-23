"""Shared loudness uses calculation identity, never presentation configuration."""

import numpy as np
import pytest

from base.analysis_compute_context import TaskComputeContext
from base.core_algorithm.sound_quality import service
from base.core_algorithm.sound_quality.loudness import LoudnessAnalyzer
from consts.acoustic_analysis.specific_consts.loudness_consts import (
    LOUDNESS_DEFAULT_OUTPUT_TIME_RESOLUTION_S,
    LOUDNESS_DEFAULT_STATIONARY_FRAME_DURATION_S,
    LOUDNESS_DEFAULT_STATIONARY_HOP_DURATION_S,
)


@pytest.fixture
def counted_core(monkeypatch):
    original = LoudnessAnalyzer.compute
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LoudnessAnalyzer, "compute", counted)
    return calls


def run(config=None, *, context=None, identity="CH1", rate=48000, factor=2.0):
    signal = 0.02 * np.sin(2 * np.pi * 1000 * np.arange(int(rate * 0.6)) / rate)
    return service.run_sound_quality(
        signal, rate, factor, {"enabled": True, "items": {"LOUD": {
            "enabled": True, **(config or {}),
        }}}, compute_context=context, input_identity=identity,
    ).loudness


@pytest.mark.parametrize("method", ["per_segment", "time_varying_iso532_1"])
def test_defaults_aliases_and_presentation_share_readonly_arrays(method, counted_core):
    context = TaskComputeContext()
    first = run({"method": method}, context=context)
    config = {
        "method": method,
        "advanced": {
            "frame_duration_s": LOUDNESS_DEFAULT_STATIONARY_FRAME_DURATION_S,
            "hop_duration_s": LOUDNESS_DEFAULT_STATIONARY_HOP_DURATION_S,
            "output_time_resolution_ms": 1000 * LOUDNESS_DEFAULT_OUTPUT_TIME_RESOLUTION_S,
            "curve_y_unit": "phon", "specific_loudness_colormap": "plasma",
        },
        "display": {"summary_metrics": ["max_transient_loudness"]},
        "save": {"summary": True},
    }
    second = run(config, context=context)
    direct = run(config)
    assert len(counted_core) == 2 and context.hits == 1
    assert first.raw_result.specific_loudness is second.raw_result.specific_loudness
    assert first.summary is not second.summary
    assert second.summary == direct.summary
    for name in ("time_s", "loudness_sone", "loudness_level_phon", "specific_loudness", "bark_axis"):
        values = getattr(second.raw_result, name)
        assert not values.flags.writeable
        np.testing.assert_array_equal(values, getattr(direct.raw_result, name))
    first.raw_result.metadata["field_type"] = "changed by consumer"
    third = run(config, context=context)
    assert third.raw_result.metadata["field_type"] == "free"
    assert direct.raw_result.loudness_sone.flags.writeable


@pytest.mark.parametrize("change", ["channel", "rate", "factor", "method", "frame", "hop", "resolution", "range"])
def test_different_inputs_and_computation_parameters_do_not_share(change, counted_core):
    context = TaskComputeContext()
    config = {"method": "per_segment", "advanced": {}}
    assert run(config, context=context).enabled
    args = {}
    if change == "channel":
        args["identity"] = "CH2"
    elif change == "rate":
        args["rate"] = 44100
    elif change == "factor":
        args["factor"] = 3.0
    elif change == "method":
        config["method"] = "time_varying_iso532_1"
    elif change == "frame":
        config["advanced"]["stationary_frame_duration_s"] = 0.2
    elif change == "hop":
        config["advanced"]["stationary_hop_duration_s"] = 0.03
    elif change == "resolution":
        config["advanced"]["output_time_resolution_ms"] = 4.0
    else:
        config["advanced"].update(analysis_time_range_enabled=True,
                                  analysis_start_time_sec=0.1, analysis_end_time_sec=0.5)
    assert run(config, context=context, **args).enabled
    assert len(counted_core) == 2 and context.hits == 0


def test_equivalent_sample_windows_keep_each_call_source_metadata(counted_core):
    context = TaskComputeContext()
    first = run(context=context)
    second = run({"advanced": {"analysis_time_range_enabled": True,
                               "analysis_end_time_sec": 0.6}}, context=context)
    assert len(counted_core) == 1
    assert "analysis_source_end_s" not in first.raw_result.metadata
    assert second.raw_result.metadata["analysis_source_end_s"] == 0.6


def test_external_reference_path_bypasses_cache_even_when_missing(counted_core, tmp_path):
    config = {"advanced": {"background_noise_processing_enabled": True,
                           "background_noise_file_path": str(tmp_path / "missing.wav")}}
    context = TaskComputeContext()
    assert run(config, context=context).enabled
    assert run(config, context=context).enabled
    assert len(counted_core) == 2 and len(context) == 0


def test_missing_identity_fails_without_caching_or_computing(counted_core):
    context = TaskComputeContext()
    result = run(context=context, identity=None)
    assert not result.enabled and "input_identity" in result.skipped_reason
    assert not counted_core and len(context) == 0


def test_failure_reused_but_output_error_does_not_poison_success(monkeypatch):
    calls = []

    def fail(self, *args, **kwargs):
        calls.append(1)
        raise FloatingPointError("frontend failed")

    with monkeypatch.context() as patch:
        patch.setattr(LoudnessAnalyzer, "compute", fail)
        context = TaskComputeContext()
        first, second = run(context=context), run(context=context)
        assert first.skipped_reason == second.skipped_reason == "loudness computation failed: frontend failed"
        assert len(calls) == 1
        run(context=TaskComputeContext())
        assert len(calls) == 2

    context = TaskComputeContext()
    with monkeypatch.context() as patch:
        def output_failure(*args, **kwargs):
            raise ValueError("output failed")
        patch.setattr(service, "_build_loudness_output", output_failure)
        with pytest.raises(ValueError, match="output failed"):
            run(context=context)
    assert run(context=context).enabled
    assert context.compute_count == 1 and context.hits == 1

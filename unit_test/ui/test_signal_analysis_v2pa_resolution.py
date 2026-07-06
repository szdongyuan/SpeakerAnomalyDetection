import os
import types

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from ui import signal_analysis_window as saw


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def _recording_widget(widget, *, analysis_channel=2):
    widget.analysis_config = {
        "analysis_channel": analysis_channel,
        "stimulus_method": "steps",
        "stimulus_type": "linear",
        "start_freq": 1000,
        "stop_freq": 1000,
        "num_steps": 1,
        "total_time": 0.01,
        "repeat_times": 1,
    }
    widget.data_struct.sample_rate = 48000
    widget.data_struct.stimulus_info = dict(widget.analysis_config)
    widget.data_struct.store_wave_data = np.ones(32, dtype=np.float64)
    widget.data_struct.store_wave_data_multi = np.ones((32, 8), dtype=np.float64)
    return widget


def _install_resolver(monkeypatch, *, factor=6.25, warning=None):
    calls = []

    def resolve(raw_channel, warn_callback=None):
        calls.append(raw_channel)
        if warning and warn_callback:
            warn_callback(warning)
        return factor

    monkeypatch.setattr(saw, "resolve_analysis_v2pa_factor_for_channel", resolve)
    return calls


def _install_resolver_outcomes(monkeypatch, outcomes):
    calls = []
    pending = list(outcomes)

    def resolve(raw_channel, warn_callback=None):
        calls.append(raw_channel)
        outcome = pending.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(saw, "resolve_analysis_v2pa_factor_for_channel", resolve)
    return calls


def _empty_thd_result():
    return {"freq_value": [], "harmonic": [], "thd": [], "thd_raw": []}


def test_prb_direct_analysis_resolves_v2pa_before_calculation(qapp, monkeypatch):
    resolver_calls = _install_resolver(monkeypatch, factor=7.5)
    captured = {}

    def calculate_perceptual(self, recorded_signal, sample_rate, thd_kwargs, v2pa_factor=None):
        captured["v2pa_factor"] = v2pa_factor
        return np.array([1000.0]), np.array([[1000.0]]), np.array([12.0])

    widget = _recording_widget(saw.PerceptualRubAndBuzz("PRB"), analysis_channel=2)
    monkeypatch.setattr(
        saw.AudioThdFrequencyResponseAnalysis,
        "calculate_perceptual_thd_three_phase",
        calculate_perceptual,
    )
    monkeypatch.setattr(widget, "plot_graph", lambda *args, **kwargs: None)

    result = widget.calculate_thd()

    assert resolver_calls == [2]
    assert captured["v2pa_factor"] == 7.5
    assert widget.v2pa_factor == 7.5
    assert result["thd"] == [12.0]


def test_missing_microphone_calibration_warns_and_prb_continues_with_temporary_factor(qapp, monkeypatch):
    warnings = []
    _install_resolver(monkeypatch, factor=1.0, warning="麦克风未进行校准，结果仅供参考。")
    captured = {}

    def calculate_perceptual(self, recorded_signal, sample_rate, thd_kwargs, v2pa_factor=None):
        captured["v2pa_factor"] = v2pa_factor
        return np.array([1000.0]), np.array([[1000.0]]), np.array([3.0])

    widget = _recording_widget(saw.PerceptualRubAndBuzz("PRB"), analysis_channel=1)
    monkeypatch.setattr(
        saw.AudioThdFrequencyResponseAnalysis,
        "calculate_perceptual_thd_three_phase",
        calculate_perceptual,
    )
    monkeypatch.setattr(widget, "plot_graph", lambda *args, **kwargs: None)
    monkeypatch.setattr(saw.MessageBox, "warning", lambda parent, title, message: warnings.append(message))

    result = widget.calculate_thd()

    assert captured["v2pa_factor"] == 1.0
    assert result["thd"] == [3.0]
    assert warnings == ["麦克风未进行校准，结果仅供参考。"]


def test_non_pd_pre_resolved_v2pa_is_reused_once_before_resolving_again(qapp, monkeypatch):
    warnings = []
    resolver_calls = _install_resolver(monkeypatch, factor=1.0, warning="麦克风未进行校准，结果仅供参考。")
    captured_factors = []

    def spl_calculation(self, recorded_signal, reference_pressure=20e-6, **kwargs):
        captured_factors.append(kwargs.get("v2pa_factor"))
        return np.array([42.0])

    widget = _recording_widget(saw.Spl("SPL"), analysis_channel=3)
    widget.v2pa_factor = 1.0
    widget._use_pre_resolved_v2pa_factor = True
    monkeypatch.setattr(saw.AudioThdFrequencyResponseAnalysis, "spl_calculation", spl_calculation)
    monkeypatch.setattr(widget, "plot_spl", lambda *args, **kwargs: None)
    monkeypatch.setattr(saw.MessageBox, "warning", lambda parent, title, message: warnings.append(message))

    first = widget.calculate_spl()
    second = widget.calculate_spl()

    assert first["signal_spl"] == [42.0]
    assert second["signal_spl"] == [42.0]
    assert captured_factors == [1.0, 1.0]
    assert resolver_calls == [3]
    assert warnings == ["麦克风未进行校准，结果仅供参考。"]
    assert widget._use_pre_resolved_v2pa_factor is False


def test_sequence_pre_resolved_v2pa_re_resolves_preserved_raw_channel_after_record_only_mapping(
    qapp, monkeypatch
):
    warnings = []
    resolver_calls = _install_resolver(monkeypatch, factor=2.0, warning="resolved again")
    captured_factors = []

    def spl_calculation(self, recorded_signal, reference_pressure=20e-6, **kwargs):
        captured_factors.append(kwargs.get("v2pa_factor"))
        return np.array([42.0])

    widget = _recording_widget(saw.Spl("SPL"), analysis_channel=1)
    widget.v2pa_factor = 9.0
    widget._use_pre_resolved_v2pa_factor = True
    widget._v2pa_raw_analysis_channel = 5
    monkeypatch.setattr(saw.AudioThdFrequencyResponseAnalysis, "spl_calculation", spl_calculation)
    monkeypatch.setattr(widget, "plot_spl", lambda *args, **kwargs: None)
    monkeypatch.setattr(saw.MessageBox, "warning", lambda parent, title, message: warnings.append(message))

    first = widget.calculate_spl()
    second = widget.calculate_spl()

    assert first["signal_spl"] == [42.0]
    assert second["signal_spl"] == [42.0]
    assert captured_factors == [9.0, 2.0]
    assert resolver_calls == [5]
    assert warnings == ["resolved again"]
    assert widget.analysis_config["analysis_channel"] == 1
    assert widget._use_pre_resolved_v2pa_factor is False


def test_direct_missing_microphone_calibration_warns_on_each_calculation(qapp, monkeypatch):
    warnings = []
    resolver_calls = _install_resolver(monkeypatch, factor=1.0, warning="麦克风未进行校准，结果仅供参考。")

    def spl_calculation(self, recorded_signal, reference_pressure=20e-6, **kwargs):
        return np.array([42.0])

    widget = _recording_widget(saw.Spl("SPL"), analysis_channel=4)
    monkeypatch.setattr(saw.AudioThdFrequencyResponseAnalysis, "spl_calculation", spl_calculation)
    monkeypatch.setattr(widget, "plot_spl", lambda *args, **kwargs: None)
    monkeypatch.setattr(saw.MessageBox, "warning", lambda parent, title, message: warnings.append(message))

    widget.calculate_spl()
    widget.calculate_spl()

    assert resolver_calls == [4, 4]
    assert warnings == ["麦克风未进行校准，结果仅供参考。", "麦克风未进行校准，结果仅供参考。"]


def test_resolver_failure_does_not_reuse_stale_v2pa_factor(qapp, monkeypatch):
    warnings = []

    def fail_resolver(raw_channel, warn_callback=None):
        raise ValueError("registered microphone hardware_id is required for calibration")

    widget = _recording_widget(saw.PerceptualRubAndBuzz("PRB"), analysis_channel=1)
    widget.v2pa_factor = 9.9
    monkeypatch.setattr(saw, "resolve_analysis_v2pa_factor_for_channel", fail_resolver)
    monkeypatch.setattr(saw.MessageBox, "warning", lambda parent, title, message: warnings.append(message))

    assert widget._resolve_v2pa_factor_for_analysis() is False
    assert widget.v2pa_factor is None
    assert warnings == ["registered microphone hardware_id is required for calibration"]


def test_distortion_standard_thd_does_not_resolve_v2pa(qapp, monkeypatch):
    warnings = []
    plot_calls = []
    captured = {}

    def fail_resolver(*args, **kwargs):
        raise AssertionError("standard HD/RB THD should not resolve microphone calibration")

    def calculate_thd(self, recorded_signal, sample_rate, thd_kwargs):
        captured["sample_rate"] = sample_rate
        captured["thd_kwargs"] = thd_kwargs
        return np.array([1000.0]), np.array([[2.0]]), np.array([10.0])

    widget = _recording_widget(saw.Distortion("HD"), analysis_channel=2)
    widget.analysis_config["selected_labels"] = [2]
    monkeypatch.setattr(saw, "resolve_analysis_v2pa_factor_for_channel", fail_resolver)
    monkeypatch.setattr(saw.AudioThdFrequencyResponseAnalysis, "calculate_thd_three_phase", calculate_thd)
    monkeypatch.setattr(widget, "plot_graph", lambda freq, thd, *args: plot_calls.append((freq, thd)))
    monkeypatch.setattr(saw.MessageBox, "warning", lambda parent, title, message: warnings.append(message))

    result = widget.calculate_thd()

    assert captured["sample_rate"] == 48000
    assert captured["thd_kwargs"]["harmonic_orders"] == [2]
    assert result["thd"] == [10.0]
    assert widget.result["thd"] == [10.0]
    assert widget.v2pa_factor is None
    assert plot_calls[-1][0].tolist() == [1000.0]
    assert warnings == []


def test_prb_resolver_failure_after_success_clears_stale_result(qapp, monkeypatch):
    resolver_calls = _install_resolver_outcomes(
        monkeypatch,
        [3.0, RuntimeError("database unavailable")],
    )
    warnings = []
    plot_calls = []

    def calculate_perceptual(self, recorded_signal, sample_rate, thd_kwargs, v2pa_factor=None):
        return np.array([1000.0]), np.array([[2.0]]), np.array([4.0])

    widget = _recording_widget(saw.PerceptualRubAndBuzz("PRB"), analysis_channel=3)
    monkeypatch.setattr(
        saw.AudioThdFrequencyResponseAnalysis,
        "calculate_perceptual_thd_three_phase",
        calculate_perceptual,
    )
    monkeypatch.setattr(widget, "plot_graph", lambda freq, thd, *args: plot_calls.append((freq, thd)))
    monkeypatch.setattr(saw.MessageBox, "warning", lambda parent, title, message: warnings.append(message))

    previous_result = widget.calculate_thd()
    failed_result = widget.calculate_thd()

    assert resolver_calls == [3, 3]
    assert previous_result["thd"] == [4.0]
    assert failed_result == _empty_thd_result()
    assert widget.result == _empty_thd_result()
    assert failed_result is not previous_result
    assert plot_calls[-1] == ([], [])
    assert warnings == ["database unavailable"]


def test_spl_direct_analysis_passes_resolved_v2pa(qapp, monkeypatch):
    resolver_calls = _install_resolver(monkeypatch, factor=2.5)
    captured = {}

    def spl_calculation(self, recorded_signal, reference_pressure=20e-6, **kwargs):
        captured["v2pa_factor"] = kwargs.get("v2pa_factor")
        return np.array([42.0, 43.0])

    widget = _recording_widget(saw.Spl("SPL"), analysis_channel=3)
    monkeypatch.setattr(saw.AudioThdFrequencyResponseAnalysis, "spl_calculation", spl_calculation)
    monkeypatch.setattr(widget, "plot_spl", lambda *args, **kwargs: None)

    result = widget.calculate_spl()

    assert resolver_calls == [3]
    assert captured["v2pa_factor"] == 2.5
    assert result["signal_spl"] == [42.0, 43.0]


def test_spl_frequency_direct_analysis_passes_resolved_v2pa(qapp, monkeypatch):
    resolver_calls = _install_resolver(monkeypatch, factor=3.5)
    captured = {}

    class FakeSplFrequencyAnalyzer:
        def __init__(self, sample_rate):
            captured["sample_rate"] = sample_rate

        def compute(self, recorded_signal, *, stimulus_metadata, v2pa_factor, splf_calc_mode):
            captured["v2pa_factor"] = v2pa_factor
            return types.SimpleNamespace(
                frequencies_hz=np.array([1000.0]),
                spl_db=np.array([55.0]),
            )

    widget = _recording_widget(saw.SplFrequency("SPLF"), analysis_channel=4)
    monkeypatch.setattr(saw, "SplFrequencyAnalyzer", FakeSplFrequencyAnalyzer)
    monkeypatch.setattr(widget, "plot_spl_frequency", lambda *args, **kwargs: None)

    result = widget.calculate_spl()

    assert resolver_calls == [4]
    assert captured["v2pa_factor"] == 3.5
    assert result["spl_db"] == [55.0]


def test_loose_particle_direct_analysis_passes_resolved_v2pa(qapp, monkeypatch):
    resolver_calls = _install_resolver(monkeypatch, factor=4.5)
    captured = {}

    def calculate_loose_particle_spl(recorded_signal, cutoff_freq, sample_rate, reference_spl, v2pa_factor):
        captured["v2pa_factor"] = v2pa_factor
        return np.array([1.0, 2.0]), np.array([0.1, 0.2])

    widget = _recording_widget(saw.LooseParticle("LP"), analysis_channel=5)
    widget.analysis_config.update(
        {
            "cutoff_freq": 1000,
            "trigger_threshold": 1,
            "hysterests_threshold": 0.5,
            "min_check_duration": 0,
            "max_check_duration": 1,
            "loose_particle_num": 1,
        }
    )
    monkeypatch.setattr(
        saw.AudioThdFrequencyResponseAnalysis,
        "calculate_loose_particle_spl",
        staticmethod(calculate_loose_particle_spl),
    )
    monkeypatch.setattr(widget, "plot_graph", lambda *args, **kwargs: setattr(widget, "result", 0))

    widget.calculate_loose_particle()

    assert resolver_calls == [5]
    assert captured["v2pa_factor"] == 4.5


def test_peak_detection_direct_analysis_passes_resolved_v2pa(qapp, monkeypatch):
    resolver_calls = _install_resolver(monkeypatch, factor=5.5)
    captured = {}

    def fake_peak_detection(recorded_signal, sample_rate, analysis_config, *, v2pa_factor):
        captured["v2pa_factor"] = v2pa_factor
        return {
            "peaks_index": [],
            "peaks_time_sec": [],
            "spl_db_series": np.array([60.0, 61.0]),
        }

    widget = _recording_widget(saw.PeakDetection("PD"), analysis_channel=6)
    monkeypatch.setattr(saw, "peak_detection", fake_peak_detection)

    result = widget.calculate_peak_detection()

    assert resolver_calls == [6]
    assert captured["v2pa_factor"] == 5.5
    assert result["spl_db_series"].tolist() == [60.0, 61.0]


@pytest.mark.parametrize(
    "factor, resolver_warning, expected_warnings",
    [
        (8.5, None, []),
        (1.0, "麦克风未进行校准，结果仅供参考。", ["麦克风未进行校准，结果仅供参考。"]),
    ],
)
def test_pipeline_pd_pm_passes_resolved_v2pa_to_embedded_peak_detection(
    qapp, monkeypatch, factor, resolver_warning, expected_warnings
):
    resolver_calls = _install_resolver(monkeypatch, factor=factor, warning=resolver_warning)
    warnings = []
    captured = {}

    class FakePatternMatch:
        pass

    def fake_peak_detection(recorded_signal, sample_rate, analysis_config, *, v2pa_factor):
        captured["v2pa_factor"] = v2pa_factor
        captured["analysis_config"] = analysis_config
        return {
            "peaks_index": [],
            "peaks_time_sec": [],
            "spl_db_series": np.array([60.0, 61.0]),
        }

    widget = saw.PipelinePdPm("PD+PM")
    widget.data_struct.sample_rate = 48000
    widget.data_struct.store_wave_data = np.ones(32, dtype=np.float64)
    widget.analysis_config = {
        "head": {"config": {"analysis_channel": 8}},
        "tail": {"config": {}},
        "pass_condition": {},
    }

    monkeypatch.setattr(saw, "peak_detection", fake_peak_detection)
    monkeypatch.setattr(saw, "get_class_mapping", lambda: {"PD": saw.PeakDetection, "PM": FakePatternMatch})
    monkeypatch.setattr(saw.MessageBox, "warning", lambda parent, title, message: warnings.append(message))

    result = widget.calculate_pipeline_pd_pm()

    assert resolver_calls == [8]
    assert captured["v2pa_factor"] == factor
    assert captured["analysis_config"] == {"analysis_channel": 8}
    assert warnings == expected_warnings
    assert result["total"] == 0


def test_frequency_band_analysis_direct_analysis_passes_resolved_v2pa(qapp, monkeypatch):
    resolver_calls = _install_resolver(monkeypatch, factor=6.5)
    captured = {}

    band = types.SimpleNamespace(label="1k", f_center=1000.0)

    class FakeFrequencyBandAnalyzer:
        def __init__(self, **kwargs):
            pass

        def analyze(self, recorded_signal, *, fs, v2pa_factor):
            captured["v2pa_factor"] = v2pa_factor
            return saw.BandAnalysisResult(
                bands=[band],
                band_levels_db=np.array([70.0]),
                band_levels_weighted_db=np.array([71.0]),
                overall_db=70.0,
                overall_weighted_db=71.0,
                weighting="A",
            )

    widget = _recording_widget(saw.FrequencyBandAnalysis("FBA"), analysis_channel=7)
    monkeypatch.setattr(saw, "FrequencyBandAnalyzer", FakeFrequencyBandAnalyzer)
    monkeypatch.setattr(widget, "_plot_bar_chart", lambda *args, **kwargs: None)

    result = widget.calculate_fba()

    assert resolver_calls == [7]
    assert captured["v2pa_factor"] == 6.5
    assert result["band_levels_db"] == [70.0]

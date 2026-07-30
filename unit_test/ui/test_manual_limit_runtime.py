import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PyQt5.QtWidgets import QApplication

from ui import signal_analysis_window as saw


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def captured_limit_utils(monkeypatch):
    captured = {"setup": [], "check": [], "compare": [], "out": []}
    original_check_interp_limits = saw.LimitPlotUtils.check_interp_limits
    original_compare_with_limits = saw.LimitPlotUtils.compare_with_limits

    def setup_limit_plot(
        plot_widget,
        data_x,
        data_y,
        csv_x,
        csv_upper,
        csv_lower,
        *args,
        **kwargs,
    ):
        captured["setup"].append(
            {
                "data_x": np.asarray(data_x, dtype=float),
                "data_y": np.asarray(data_y, dtype=float),
                "csv_x": np.asarray(csv_x, dtype=float),
                "csv_upper": np.asarray(csv_upper, dtype=float),
                "csv_lower": np.asarray(csv_lower, dtype=float),
                "kwargs": kwargs,
            }
        )

    def check_interp_limits(data_x, data_y, csv_x, csv_upper, csv_lower):
        result = original_check_interp_limits(data_x, data_y, csv_x, csv_upper, csv_lower)
        captured["check"].append(
            {
                "data_x": np.asarray(data_x, dtype=float),
                "data_y": np.asarray(data_y, dtype=float),
                "csv_x": np.asarray(csv_x, dtype=float),
                "csv_upper": np.asarray(csv_upper, dtype=float),
                "csv_lower": np.asarray(csv_lower, dtype=float),
                "result": result,
            }
        )
        return result

    def compare_with_limits(plot_y, upper_limits, lower_limits, valid_mask=None):
        result = original_compare_with_limits(plot_y, upper_limits, lower_limits, valid_mask)
        captured["compare"].append(
            {
                "plot_y": np.asarray(plot_y, dtype=float),
                "upper_limits": np.asarray(upper_limits, dtype=float),
                "lower_limits": np.asarray(lower_limits, dtype=float),
                "valid_mask": np.asarray(valid_mask, dtype=bool),
                "result": result,
            }
        )
        return result

    def plot_out_segments(plot_widget, x_data, y_data, out_mask, *args, **kwargs):
        captured["out"].append(
            {
                "x_data": np.asarray(x_data, dtype=float),
                "y_data": np.asarray(y_data, dtype=float),
                "out_mask": np.asarray(out_mask, dtype=bool),
            }
        )

    monkeypatch.setattr(saw.LimitPlotUtils, "setup_limit_plot", staticmethod(setup_limit_plot))
    monkeypatch.setattr(saw.LimitPlotUtils, "check_interp_limits", staticmethod(check_interp_limits))
    monkeypatch.setattr(saw.LimitPlotUtils, "compare_with_limits", staticmethod(compare_with_limits))
    monkeypatch.setattr(saw.LimitPlotUtils, "plot_out_segments", staticmethod(plot_out_segments))
    return captured


def _manual_upper_config(segments, *, scalar_upper=999.0):
    return {
        "limit_checked": True,
        "limit_mode": "manual",
        "limit_data": None,
        "manual_upper_enabled": True,
        "manual_lower_enabled": False,
        "manual_upper": scalar_upper,
        "manual_lower": -999.0,
        "manual_upper_segments": segments,
        "manual_lower_segments": [],
    }


def _exact_gap_segments():
    return [
        {"start_x": 50.0, "start_y": 4.0, "end_x": 100.0, "end_y": 4.0},
        {"start_x": 100.1, "start_y": 3.0, "end_x": 1000.0, "end_y": 3.0},
    ]


def _capture_highlight_judgment(widget):
    captured = {}
    original_highlight = widget._highlight_out_of_range_curve

    def capture(freq_value, y_data, limit_x, upper, lower, **kwargs):
        captured.update(
            {
                "freq_value": np.asarray(freq_value, dtype=float),
                "y_data": np.asarray(y_data, dtype=float),
                "limit_x": np.asarray(limit_x, dtype=float),
                "upper": np.asarray(upper, dtype=float),
                "lower": np.asarray(lower, dtype=float),
            }
        )
        return original_highlight(
            freq_value,
            y_data,
            limit_x,
            upper,
            lower,
            **kwargs,
        )

    widget._highlight_out_of_range_curve = capture
    return captured


def _assert_upper_with_gap(actual, expected_middle):
    assert np.isnan(actual[0])
    assert actual[1] == pytest.approx(expected_middle)
    assert np.isnan(actual[2])


def _recording_widget(widget, analysis_config):
    widget.analysis_config = analysis_config
    widget.data_struct.sample_rate = 10
    widget.data_struct.store_wave_data = np.ones(3, dtype=np.float32)
    widget.data_struct.store_wave_data_multi = None
    return widget


def _frequency_analysis_config(config):
    config.update(
        {
            "analysis_channel": 0,
            "splf_calc_mode": "fundamental",
            "stimulus_method": "steps",
            "stimulus_type": "linear",
            "start_freq": 50,
            "stop_freq": 1000,
            "num_steps": 5,
            "total_time": 0.5,
            "repeat_times": 1,
        }
    )
    return config


def _frequency_analysis_widget(monkeypatch, analysis_kind, config, measured_x, measured_y):
    if analysis_kind == "splf":
        widget = _recording_widget(saw.SplFrequency("SPLF"), config)
        widget.data_struct.sample_rate = 48000
        widget.data_struct.stimulus_info = dict(config)
        monkeypatch.setattr(widget, "_resolve_v2pa_factor_for_analysis", lambda: True)

        class FakeSplFrequencyAnalyzer:
            def __init__(self, sample_rate):
                pass

            def compute(self, recorded_signal, *, stimulus_metadata, v2pa_factor, splf_calc_mode):
                return types.SimpleNamespace(
                    frequencies_hz=np.asarray(measured_x, dtype=float),
                    spl_db=np.asarray(measured_y, dtype=float),
                )

        monkeypatch.setattr(saw, "SplFrequencyAnalyzer", FakeSplFrequencyAnalyzer)
        return widget, widget.calculate_spl

    widget = _recording_widget(saw.Frequency("FR"), config)
    widget.data_struct.sample_rate = 48000
    widget.data_struct.stimulus_data = np.ones(3, dtype=np.float32)
    widget.data_struct.stimulus_info = dict(config)

    class FakeFrequencyResponseAnalyzer:
        def __init__(self, sample_rate):
            pass

        def compute(self, stimulus_signal, recorded_signal, *, stimulus_metadata, method):
            return types.SimpleNamespace(
                frequencies_hz=np.asarray(measured_x, dtype=float),
                magnitude_db=np.asarray(measured_y, dtype=float),
            )

    monkeypatch.setattr(saw, "FrequencyResponseAnalyzer", FakeFrequencyResponseAnalyzer)
    return widget, widget.calculate_fr


@pytest.mark.parametrize("golden_case", ["missing_file", "missing_field", "no_overlap"])
def test_splf_invalid_golden_envelope_skips_limit_judgment_and_clears_result(
    qapp,
    monkeypatch,
    captured_limit_utils,
    tmp_path,
    golden_case,
):
    golden_path = tmp_path / f"{golden_case}.json"
    if golden_case == "missing_field":
        payload = {
            "items": {
                "SPLF": {
                    "result": {
                        "frequency_list": [100.0, 200.0, 300.0],
                    }
                }
            }
        }
        golden_path.write_text(json.dumps(payload), encoding="utf-8")
    elif golden_case == "no_overlap":
        payload = {
            "items": {
                "SPLF": {
                    "result": {
                        "frequency_list": [1000.0, 2000.0],
                        "spl_db": [80.0, 81.0],
                    }
                }
            }
        }
        golden_path.write_text(json.dumps(payload), encoding="utf-8")

    config = {
        "analysis_channel": 0,
        "splf_calc_mode": "fundamental",
        "stimulus_method": "steps",
        "stimulus_type": "linear",
        "start_freq": 100,
        "stop_freq": 300,
        "num_steps": 3,
        "total_time": 0.3,
        "repeat_times": 1,
        "golden_sample_checked": True,
        "golden_sample_display_mode": "envelope",
        "golden_sample_result_path": str(golden_path),
        "limit_checked": True,
        "limit_mode": "csv",
        "limit_data": (
            np.array([100.0, 200.0, 300.0]),
            np.array([3.0, 3.0, 3.0]),
            np.array([-5.0, -5.0, -5.0]),
        ),
    }
    widget = _recording_widget(saw.SplFrequency("SPLF"), config)
    widget.data_struct.analysis_result_dict["SPLF"] = (True, 0.0)
    widget.data_struct.sample_rate = 48000
    widget.data_struct.stimulus_info = dict(config)
    monkeypatch.setattr(widget, "_resolve_v2pa_factor_for_analysis", lambda: True)

    class FakeSplFrequencyAnalyzer:
        def __init__(self, sample_rate):
            pass

        def compute(self, recorded_signal, *, stimulus_metadata, v2pa_factor, splf_calc_mode):
            return types.SimpleNamespace(
                frequencies_hz=np.array([100.0, 200.0, 300.0]),
                spl_db=np.array([80.0, 81.0, 82.0]),
            )

    warnings = []
    monkeypatch.setattr(saw, "SplFrequencyAnalyzer", FakeSplFrequencyAnalyzer)
    monkeypatch.setattr(saw.MessageBox, "warning", lambda *args: warnings.append(args[-1]))

    result = widget.calculate_spl()

    assert result is not False
    assert captured_limit_utils["setup"] == []
    assert captured_limit_utils["check"] == []
    assert "SPLF" not in widget.data_struct.analysis_result_dict
    assert warnings == [
        "黄金样本无效或与测试曲线没有有效频率重叠，已仅显示测试曲线，本次未进行上下框线判定。"
    ]


@pytest.mark.parametrize(
    ("widget_class", "title"),
    [
        (saw.Distortion, "HD"),
        (saw.PerceptualRubAndBuzz, "PRB"),
    ],
)
def test_nearest_envelope_without_comparable_limit_point_does_not_write_ok(
    qapp,
    monkeypatch,
    captured_limit_utils,
    widget_class,
    title,
):
    widget = widget_class(title)
    widget.data_struct.analysis_result_dict[title] = (True, 0.0)
    config = {
        "golden_sample_checked": True,
        "golden_sample_display_mode": "envelope",
        "limit_checked": True,
        "limit_mode": "csv",
        "limit_data": (
            np.array([1000.0, 2000.0]),
            np.array([3.0, 3.0]),
            np.array([-5.0, -5.0]),
        ),
    }
    warnings = []
    monkeypatch.setattr(saw.MessageBox, "warning", lambda *args: warnings.append(args[-1]))

    widget.plot_graph(
        np.array([100.0, 200.0]),
        np.array([0.0, 0.0]),
        config,
        raw_y=np.array([100.0, 100.0]),
        baseline_aligned=np.array([100.0, 100.0]),
    )

    assert title not in widget.data_struct.analysis_result_dict
    assert captured_limit_utils["out"] == []
    assert warnings == [
        "测试曲线、黄金样本与上下限没有共同的有效频率范围，已仅显示测试曲线，本次未进行上下框线判定。"
    ]


@pytest.mark.parametrize(
    ("widget_class", "title"),
    [
        (saw.Distortion, "HD"),
        (saw.PerceptualRubAndBuzz, "PRB"),
    ],
)
def test_nearest_envelope_single_comparable_point_is_still_judged(
    qapp,
    monkeypatch,
    captured_limit_utils,
    widget_class,
    title,
):
    widget = widget_class(title)
    config = {
        "golden_sample_checked": True,
        "golden_sample_display_mode": "envelope",
        "limit_checked": True,
        "limit_mode": "csv",
        "limit_data": (
            np.array([100.0, 200.0]),
            np.array([3.0, 3.0]),
            np.array([-5.0, -5.0]),
        ),
    }
    warnings = []
    monkeypatch.setattr(saw.MessageBox, "warning", lambda *args: warnings.append(args[-1]))

    widget.plot_graph(
        np.array([100.0]),
        np.array([10.0]),
        config,
        raw_y=np.array([110.0]),
        baseline_aligned=np.array([100.0]),
    )

    assert widget.data_struct.analysis_result_dict[title] == (False, 7.0)
    assert captured_limit_utils["out"][-1]["out_mask"].tolist() == [True]
    assert warnings == []


@pytest.mark.parametrize(
    ("widget_class", "title"),
    [
        (saw.Distortion, "HD"),
        (saw.PerceptualRubAndBuzz, "PRB"),
    ],
)
@pytest.mark.parametrize(
    ("upper", "lower", "deviation_curve", "expected_result", "expected_out_mask"),
    [
        ([5.0, 5.0], [np.nan, np.nan], [1.0, 2.0], (True, 3.0), [False, False]),
        ([5.0, 5.0], [np.nan, np.nan], [5.0, 2.0], (True, 0.0), [False, False]),
        ([np.nan, np.nan], [-5.0, -5.0], [-1.0, -2.0], (True, 3.0), [False, False]),
        ([3.0, 3.0], [-5.0, -5.0], [1.0, -2.0], (True, 2.0), [False, False]),
        ([np.nan, np.nan], [-5.0, -5.0], [-6.0, -2.0], (False, 1.0), [True, False]),
    ],
)
def test_nearest_envelope_margin_uses_only_configured_sides(
    qapp,
    monkeypatch,
    captured_limit_utils,
    widget_class,
    title,
    upper,
    lower,
    deviation_curve,
    expected_result,
    expected_out_mask,
):
    widget = widget_class(title)
    config = {
        "golden_sample_checked": True,
        "golden_sample_display_mode": "envelope",
        "limit_checked": True,
        "limit_mode": "csv",
        "limit_data": (
            np.array([100.0, 200.0]),
            np.asarray(upper, dtype=float),
            np.asarray(lower, dtype=float),
        ),
    }
    baseline = np.array([100.0, 100.0])
    deviation = np.asarray(deviation_curve, dtype=float)
    warnings = []
    monkeypatch.setattr(saw.MessageBox, "warning", lambda *args: warnings.append(args[-1]))

    widget.plot_graph(
        np.array([100.0, 200.0]),
        deviation,
        config,
        raw_y=baseline + deviation,
        baseline_aligned=baseline,
    )

    assert widget.data_struct.analysis_result_dict[title] == expected_result
    assert captured_limit_utils["out"][-1]["out_mask"].tolist() == expected_out_mask
    assert warnings == []


@pytest.mark.parametrize(
    ("widget_class", "method_name", "title"),
    [
        (saw.SplFrequency, "plot_spl_frequency_with_limits", "SPLF"),
        (saw.Frequency, "plot_fr_with_limits", "FR"),
    ],
)
def test_interpolated_envelope_without_comparable_limit_point_does_not_write_ok(
    qapp,
    monkeypatch,
    captured_limit_utils,
    widget_class,
    method_name,
    title,
):
    widget = widget_class(title)
    widget.data_struct.analysis_result_dict[title] = (True, 0.0)
    warnings = []
    monkeypatch.setattr(saw.MessageBox, "warning", lambda *args: warnings.append(args[-1]))

    getattr(widget, method_name)(
        np.array([100.0, 200.0]),
        np.array([0.0, 0.0]),
        np.array([1000.0, 2000.0]),
        np.array([3.0, 3.0]),
        np.array([-5.0, -5.0]),
        raw_y=np.array([100.0, 100.0]),
        baseline_aligned=np.array([100.0, 100.0]),
        display_mode="envelope",
    )

    assert title not in widget.data_struct.analysis_result_dict
    assert captured_limit_utils["out"] == []
    assert warnings == [
        "测试曲线、黄金样本与上下限没有共同的有效频率范围，已仅显示测试曲线，本次未进行上下框线判定。"
    ]


@pytest.mark.parametrize(
    ("widget_class", "method_name", "title"),
    [
        (saw.SplFrequency, "plot_spl_frequency_with_limits", "SPLF"),
        (saw.Frequency, "plot_fr_with_limits", "FR"),
    ],
)
def test_interpolated_envelope_single_comparable_point_is_still_judged(
    qapp,
    monkeypatch,
    captured_limit_utils,
    widget_class,
    method_name,
    title,
):
    widget = widget_class(title)
    warnings = []
    monkeypatch.setattr(saw.MessageBox, "warning", lambda *args: warnings.append(args[-1]))

    getattr(widget, method_name)(
        np.array([100.0]),
        np.array([10.0]),
        np.array([100.0, 200.0]),
        np.array([3.0, 3.0]),
        np.array([-5.0, -5.0]),
        raw_y=np.array([110.0]),
        baseline_aligned=np.array([100.0]),
        display_mode="envelope",
    )

    assert widget.data_struct.analysis_result_dict[title] == (False, 7.0)
    assert captured_limit_utils["out"][-1]["out_mask"].tolist() == [True]
    assert warnings == []


def test_spl_time_domain_manual_segments_drive_limit_arrays_and_out_of_range(
    qapp, monkeypatch, captured_limit_utils
):
    config = _manual_upper_config(
        [{"start_x": 0.0, "start_y": 8.0, "end_x": 0.2, "end_y": 8.0}]
    )
    config.update({"analysis_channel": 0, "weighting": "Z", "smooth_checked": False})
    widget = _recording_widget(saw.Spl("SPL"), config)
    monkeypatch.setattr(widget, "_resolve_v2pa_factor_for_analysis", lambda: True)

    def spl_calculation(self, recorded_signal, reference_pressure=20e-6, **kwargs):
        return np.array([5.0, 12.0, 4.0])

    monkeypatch.setattr(saw.AudioThdFrequencyResponseAnalysis, "spl_calculation", spl_calculation)

    result = widget.calculate_spl()

    assert result is not False
    setup = captured_limit_utils["setup"][-1]
    np.testing.assert_allclose(setup["csv_x"], [0.0, 0.2])
    np.testing.assert_allclose(setup["csv_upper"], [8.0, 8.0])
    assert np.all(np.isnan(setup["csv_lower"]))
    comparison = captured_limit_utils["compare"][-1]
    np.testing.assert_allclose(comparison["upper_limits"][1:], [8.0, 8.0])
    assert np.isnan(comparison["upper_limits"][0])
    assert np.isnan(comparison["lower_limits"]).all()
    out = captured_limit_utils["out"][-1]
    assert out["out_mask"].tolist() == [False, True, False]
    assert widget.data_struct.analysis_result_dict["SPL"] == (False, 4.0)


def test_time_spl_exact_plot_endpoints_keep_measured_judgment_and_linear_geometry(
    qapp, monkeypatch, captured_limit_utils
):
    config = _manual_upper_config(
        [
            {"start_x": 0.09, "start_y": 8.0, "end_x": 0.100, "end_y": 8.0},
            {"start_x": 0.101, "start_y": 7.0, "end_x": 0.120, "end_y": 7.0},
        ]
    )
    config.update({"analysis_channel": 0, "weighting": "Z", "smooth_checked": False})
    widget = _recording_widget(saw.Spl("SPL"), config)
    widget.data_struct.sample_rate = 2000
    widget.data_struct.store_wave_data = np.ones(41, dtype=np.float32)
    monkeypatch.setattr(widget, "_resolve_v2pa_factor_for_analysis", lambda: True)
    monkeypatch.setattr(
        widget,
        "_apply_spl_analysis_time_range",
        lambda recorded_signal, sample_rate: (recorded_signal, 180),
    )

    def spl_calculation(self, recorded_signal, reference_pressure=20e-6, **kwargs):
        spl_values = np.full(41, 6.0)
        spl_values[21] = 9.0
        return spl_values

    manual_plot_calls = []
    original_manual_limit_plot_data = saw.manual_limit_plot_data

    def capture_manual_limit_plot_data(config_arg, **kwargs):
        manual_plot_calls.append(kwargs)
        return original_manual_limit_plot_data(config_arg, **kwargs)

    monkeypatch.setattr(saw.AudioThdFrequencyResponseAnalysis, "spl_calculation", spl_calculation)
    monkeypatch.setattr(saw, "manual_limit_plot_data", capture_manual_limit_plot_data)

    result = widget.calculate_spl()

    assert result is not False
    setup = captured_limit_utils["setup"][-1]
    np.testing.assert_allclose(setup["csv_x"], [0.09, 0.100, 0.101, 0.120])
    np.testing.assert_allclose(setup["csv_upper"], [8.0, 8.0, 7.0, 7.0])
    assert np.isnan(setup["csv_lower"]).all()
    assert setup["kwargs"]["log_x"] is False

    comparison = captured_limit_utils["compare"][-1]
    measured_time = np.asarray(result["signal_duration"], dtype=float)
    expected_spl = np.full(41, 6.0)
    expected_spl[21] = 9.0
    np.testing.assert_allclose(comparison["plot_y"], expected_spl)
    assert comparison["upper_limits"].shape == measured_time.shape
    gap_index = int(np.flatnonzero(np.isclose(measured_time, 0.1005))[0])
    assert gap_index == 21
    assert comparison["upper_limits"][gap_index] == pytest.approx(7.5)
    assert comparison["valid_mask"][gap_index]
    assert captured_limit_utils["out"][-1]["out_mask"][gap_index]
    assert widget.data_struct.analysis_result_dict["SPL"] == pytest.approx((False, 1.5))
    assert manual_plot_calls[-1]["positive_x_support"] is None


def test_spl_frequency_manual_segments_with_gaps_drive_interp_check_and_out_of_range(
    qapp, monkeypatch, captured_limit_utils
):
    config = _manual_upper_config(
        [{"start_x": 100.0, "start_y": 10.0, "end_x": 250.0, "end_y": 10.0}]
    )
    config.update(
        {
            "analysis_channel": 0,
            "splf_calc_mode": "fundamental",
            "stimulus_method": "steps",
            "stimulus_type": "linear",
            "start_freq": 100,
            "stop_freq": 300,
            "num_steps": 3,
            "total_time": 0.3,
            "repeat_times": 1,
        }
    )
    widget = _recording_widget(saw.SplFrequency("SPLF"), config)
    widget.data_struct.sample_rate = 48000
    widget.data_struct.stimulus_info = dict(config)
    monkeypatch.setattr(widget, "_resolve_v2pa_factor_for_analysis", lambda: True)

    class FakeSplFrequencyAnalyzer:
        def __init__(self, sample_rate):
            pass

        def compute(self, recorded_signal, *, stimulus_metadata, v2pa_factor, splf_calc_mode):
            return types.SimpleNamespace(
                frequencies_hz=np.array([100.0, 200.0, 300.0]),
                spl_db=np.array([1.0, 20.0, 1.0]),
            )

    monkeypatch.setattr(saw, "SplFrequencyAnalyzer", FakeSplFrequencyAnalyzer)

    result = widget.calculate_spl()

    assert result is not False
    check = captured_limit_utils["check"][-1]
    np.testing.assert_allclose(check["csv_x"], [100.0, 200.0, 300.0])
    _assert_upper_with_gap(check["csv_upper"], 10.0)
    out_mask, _plot_x, _plot_y, deviation, is_ok = check["result"]
    assert out_mask.tolist() == [False, True, False]
    assert deviation == 10.0
    assert is_ok is False


def test_spl_frequency_manual_segments_stay_aligned_for_unsorted_analyzer_output(
    qapp, monkeypatch, captured_limit_utils
):
    config = _manual_upper_config(
        [{"start_x": 100.0, "start_y": 10.0, "end_x": 250.0, "end_y": 10.0}]
    )
    config.update(
        {
            "analysis_channel": 0,
            "splf_calc_mode": "fundamental",
            "stimulus_method": "steps",
            "stimulus_type": "linear",
            "start_freq": 100,
            "stop_freq": 300,
            "num_steps": 3,
            "total_time": 0.3,
            "repeat_times": 1,
        }
    )
    widget = _recording_widget(saw.SplFrequency("SPLF"), config)
    widget.data_struct.sample_rate = 48000
    widget.data_struct.stimulus_info = dict(config)
    monkeypatch.setattr(widget, "_resolve_v2pa_factor_for_analysis", lambda: True)

    class FakeSplFrequencyAnalyzer:
        def __init__(self, sample_rate):
            pass

        def compute(self, recorded_signal, *, stimulus_metadata, v2pa_factor, splf_calc_mode):
            return types.SimpleNamespace(
                frequencies_hz=np.array([300.0, 200.0, 100.0]),
                spl_db=np.array([1.0, 20.0, 1.0]),
            )

    monkeypatch.setattr(saw, "SplFrequencyAnalyzer", FakeSplFrequencyAnalyzer)

    result = widget.calculate_spl()

    assert result is not False
    check = captured_limit_utils["check"][-1]
    np.testing.assert_allclose(check["data_x"], [100.0, 200.0, 300.0])
    np.testing.assert_allclose(check["data_y"], [1.0, 20.0, 1.0])
    np.testing.assert_allclose(check["csv_x"], [100.0, 200.0, 300.0])
    _assert_upper_with_gap(check["csv_upper"], 10.0)
    out_mask, _plot_x, _plot_y, deviation, is_ok = check["result"]
    assert out_mask.tolist() == [False, True, False]
    assert deviation == 10.0
    assert is_ok is False
    assert widget.data_struct.analysis_result_dict["SPLF"] == (False, 10.0)


def test_frequency_response_manual_segments_with_gaps_drive_interp_check_and_out_of_range(
    qapp, monkeypatch, captured_limit_utils
):
    config = _manual_upper_config(
        [{"start_x": 100.0, "start_y": 10.0, "end_x": 250.0, "end_y": 10.0}]
    )
    config.update(
        {
            "analysis_channel": 0,
            "stimulus_method": "steps",
            "stimulus_type": "linear",
            "start_freq": 100,
            "stop_freq": 300,
            "num_steps": 3,
            "total_time": 0.3,
            "repeat_times": 1,
        }
    )
    widget = _recording_widget(saw.Frequency("FR"), config)
    widget.data_struct.sample_rate = 48000
    widget.data_struct.stimulus_data = np.ones(3, dtype=np.float32)
    widget.data_struct.stimulus_info = dict(config)

    class FakeFrequencyResponseAnalyzer:
        def __init__(self, sample_rate):
            pass

        def compute(self, stimulus_signal, recorded_signal, *, stimulus_metadata, method):
            return types.SimpleNamespace(
                frequencies_hz=np.array([100.0, 200.0, 300.0]),
                magnitude_db=np.array([1.0, 12.0, 1.0]),
            )

    monkeypatch.setattr(saw, "FrequencyResponseAnalyzer", FakeFrequencyResponseAnalyzer)

    result = widget.calculate_fr()

    assert result is not False
    check = captured_limit_utils["check"][-1]
    np.testing.assert_allclose(check["csv_x"], [100.0, 200.0, 300.0])
    _assert_upper_with_gap(check["csv_upper"], 10.0)
    out_mask, _plot_x, _plot_y, deviation, is_ok = check["result"]
    assert out_mask.tolist() == [False, True, False]
    assert deviation == 2.0
    assert is_ok is False


def test_frequency_response_manual_segments_stay_aligned_for_unsorted_analyzer_output(
    qapp, monkeypatch, captured_limit_utils
):
    config = _manual_upper_config(
        [{"start_x": 100.0, "start_y": 10.0, "end_x": 250.0, "end_y": 10.0}]
    )
    config.update(
        {
            "analysis_channel": 0,
            "stimulus_method": "steps",
            "stimulus_type": "linear",
            "start_freq": 100,
            "stop_freq": 300,
            "num_steps": 3,
            "total_time": 0.3,
            "repeat_times": 1,
        }
    )
    widget = _recording_widget(saw.Frequency("FR"), config)
    widget.data_struct.sample_rate = 48000
    widget.data_struct.stimulus_data = np.ones(3, dtype=np.float32)
    widget.data_struct.stimulus_info = dict(config)

    class FakeFrequencyResponseAnalyzer:
        def __init__(self, sample_rate):
            pass

        def compute(self, stimulus_signal, recorded_signal, *, stimulus_metadata, method):
            return types.SimpleNamespace(
                frequencies_hz=np.array([300.0, 200.0, 100.0]),
                magnitude_db=np.array([1.0, 12.0, 1.0]),
            )

    monkeypatch.setattr(saw, "FrequencyResponseAnalyzer", FakeFrequencyResponseAnalyzer)

    result = widget.calculate_fr()

    assert result is not False
    check = captured_limit_utils["check"][-1]
    np.testing.assert_allclose(check["data_x"], [100.0, 200.0, 300.0])
    np.testing.assert_allclose(check["data_y"], [1.0, 12.0, 1.0])
    np.testing.assert_allclose(check["csv_x"], [100.0, 200.0, 300.0])
    _assert_upper_with_gap(check["csv_upper"], 10.0)
    out_mask, _plot_x, _plot_y, deviation, is_ok = check["result"]
    assert out_mask.tolist() == [False, True, False]
    assert deviation == 2.0
    assert is_ok is False
    assert widget.data_struct.analysis_result_dict["FR"] == (False, 2.0)


@pytest.mark.parametrize(
    "display_config",
    [
        {},
        {
            "golden_sample_checked": True,
            "golden_sample_display_mode": "deviation",
        },
    ],
    ids=["normal", "deviation"],
)
def test_splf_exact_plot_endpoints_keep_connector_judgment(
    qapp,
    monkeypatch,
    captured_limit_utils,
    tmp_path,
    display_config,
):
    measured_x = np.array([50.0, 100.0, 100.05, 110.0, 1000.0])
    config = _frequency_analysis_config(_manual_upper_config(_exact_gap_segments()))
    config.update(display_config)
    if display_config:
        golden_path = tmp_path / "splf_deviation_golden.json"
        golden_path.write_text(
            json.dumps(
                {
                    "items": {
                        "SPLF": {
                            "result": {
                                "frequency_list": measured_x.tolist(),
                                "spl_db": np.zeros(measured_x.size).tolist(),
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        config["golden_sample_result_path"] = str(golden_path)
    monkeypatch.setattr(saw.MessageBox, "warning", lambda *args: None)
    widget, calculate = _frequency_analysis_widget(
        monkeypatch,
        "splf",
        config,
        measured_x,
        np.array([0.0, 0.0, 100.0, 0.0, 0.0]),
    )

    result = calculate()

    assert result is not False
    setup = captured_limit_utils["setup"][-1]
    np.testing.assert_allclose(setup["csv_x"], [50.0, 100.0, 100.1, 1000.0])
    np.testing.assert_allclose(setup["csv_upper"], [4.0, 4.0, 3.0, 3.0])
    assert np.isnan(setup["csv_lower"]).all()
    check = captured_limit_utils["check"][-1]
    np.testing.assert_allclose(check["csv_x"], measured_x)
    np.testing.assert_allclose(
        check["csv_upper"],
        [np.nan, 4.0, 3.5, 3.0, 3.0],
        equal_nan=True,
    )
    out_mask, _plot_x, _plot_y, deviation, is_ok = check["result"]
    assert out_mask.tolist() == [False, False, True, False, False]
    assert deviation == pytest.approx(96.5)
    assert is_ok is False
    assert widget.data_struct.analysis_result_dict["SPLF"] == pytest.approx((False, 96.5))


@pytest.mark.parametrize(
    "display_config",
    [
        {},
        {
            "golden_sample_checked": True,
            "golden_sample_display_mode": "deviation",
        },
    ],
    ids=["normal", "deviation"],
)
def test_fr_exact_plot_endpoints_keep_connector_judgment(
    qapp,
    monkeypatch,
    captured_limit_utils,
    tmp_path,
    display_config,
):
    measured_x = np.array([50.0, 100.0, 100.05, 110.0, 1000.0])
    config = _frequency_analysis_config(_manual_upper_config(_exact_gap_segments()))
    config.update(display_config)
    if display_config:
        golden_path = tmp_path / "fr_deviation_golden.json"
        golden_path.write_text(
            json.dumps(
                {
                    "items": {
                        "FR": {
                            "result": {
                                "frequency_list": measured_x.tolist(),
                                "fr": np.zeros(measured_x.size).tolist(),
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        config["golden_sample_result_path"] = str(golden_path)
    monkeypatch.setattr(saw.MessageBox, "warning", lambda *args: None)
    widget, calculate = _frequency_analysis_widget(
        monkeypatch,
        "fr",
        config,
        measured_x,
        np.array([0.0, 0.0, 100.0, 0.0, 0.0]),
    )

    result = calculate()

    assert result is not False
    setup = captured_limit_utils["setup"][-1]
    np.testing.assert_allclose(setup["csv_x"], [50.0, 100.0, 100.1, 1000.0])
    np.testing.assert_allclose(setup["csv_upper"], [4.0, 4.0, 3.0, 3.0])
    assert np.isnan(setup["csv_lower"]).all()
    check = captured_limit_utils["check"][-1]
    np.testing.assert_allclose(check["csv_x"], measured_x)
    np.testing.assert_allclose(
        check["csv_upper"],
        [np.nan, 4.0, 3.5, 3.0, 3.0],
        equal_nan=True,
    )
    out_mask, _plot_x, _plot_y, deviation, is_ok = check["result"]
    assert out_mask.tolist() == [False, False, True, False, False]
    assert deviation == pytest.approx(96.5)
    assert is_ok is False
    assert widget.data_struct.analysis_result_dict["FR"] == pytest.approx((False, 96.5))


@pytest.mark.parametrize("analysis_kind", ["splf", "fr"])
def test_splf_fr_manual_envelope_uses_exact_geometry_without_extrapolation(
    qapp,
    monkeypatch,
    captured_limit_utils,
    tmp_path,
    analysis_kind,
):
    measured_x = np.array([50.0, 100.0, 100.05, 110.0, 1000.0])
    baseline = np.array([10.0, 12.0, 12.5, 13.0, 14.0])
    title = "SPLF" if analysis_kind == "splf" else "FR"
    baseline_key = "spl_db" if analysis_kind == "splf" else "fr"
    golden_path = tmp_path / f"{analysis_kind}_golden.json"
    golden_path.write_text(
        json.dumps(
            {
                "items": {
                    title: {
                        "result": {
                            "frequency_list": measured_x.tolist(),
                            baseline_key: baseline.tolist(),
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    segments = [
        {"start_x": 40.0, "start_y": 4.0, "end_x": 100.0, "end_y": 4.0},
        {"start_x": 100.1, "start_y": 3.0, "end_x": 1100.0, "end_y": 3.0},
    ]
    config = _frequency_analysis_config(_manual_upper_config(segments))
    config.update(
        {
            "golden_sample_checked": True,
            "golden_sample_display_mode": "envelope",
            "golden_sample_result_path": str(golden_path),
        }
    )
    measured_y = baseline.copy()
    measured_y[2] += 100.0
    widget, calculate = _frequency_analysis_widget(
        monkeypatch,
        analysis_kind,
        config,
        measured_x,
        measured_y,
    )

    result = calculate()

    assert result is not False
    setup = captured_limit_utils["setup"][-1]
    np.testing.assert_allclose(
        setup["csv_x"],
        [40.0, 50.0, 100.0, 100.05, 100.1, 110.0, 1000.0, 1100.0],
    )
    assert np.isnan(setup["csv_upper"][[0, -1]]).all()
    endpoint_index = setup["csv_x"].tolist().index(100.1)
    assert setup["csv_upper"][endpoint_index] == pytest.approx(
        np.interp(100.1, measured_x, baseline) + 3.0
    )
    comparison = captured_limit_utils["compare"][-1]
    np.testing.assert_allclose(
        comparison["upper_limits"],
        [4.0, 4.0, 3.5, 3.0, 3.0],
        equal_nan=True,
    )
    out_mask, deviation, is_ok = comparison["result"]
    assert out_mask.tolist() == [False, False, True, False, False]
    assert deviation == pytest.approx(96.5)
    assert is_ok is False
    assert captured_limit_utils["out"][-1]["out_mask"].tolist() == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert widget.data_struct.analysis_result_dict[title] == pytest.approx((False, 96.5))


@pytest.mark.parametrize("analysis_kind", ["splf", "fr"])
def test_splf_fr_zero_start_clips_plot_only_and_keeps_measured_judgment(
    qapp,
    monkeypatch,
    captured_limit_utils,
    analysis_kind,
):
    measured_x = np.array([0.0, 10.0, 20.0, 30.0])
    config = _frequency_analysis_config(
        _manual_upper_config(
            [{"start_x": 0.0, "start_y": 0.0, "end_x": 20.0, "end_y": 20.0}]
        )
    )
    manual_plot_calls = []
    original_manual_limit_plot_data = saw.manual_limit_plot_data

    def capture_manual_limit_plot_data(config_arg, **kwargs):
        manual_plot_calls.append(kwargs)
        return original_manual_limit_plot_data(config_arg, **kwargs)

    monkeypatch.setattr(saw, "manual_limit_plot_data", capture_manual_limit_plot_data)
    widget, calculate = _frequency_analysis_widget(
        monkeypatch,
        analysis_kind,
        config,
        measured_x,
        np.array([100.0, 0.0, 0.0, 100.0]),
    )

    result = calculate()

    assert result is not False
    setup = captured_limit_utils["setup"][-1]
    np.testing.assert_allclose(setup["csv_x"], [10.0, 20.0])
    np.testing.assert_allclose(setup["csv_upper"], [10.0, 20.0])
    assert np.isnan(setup["csv_lower"]).all()
    check = captured_limit_utils["check"][-1]
    np.testing.assert_allclose(check["csv_x"], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(
        check["csv_upper"],
        [10.0, 20.0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        manual_plot_calls[-1]["positive_x_support"],
        [10.0, 20.0, 30.0],
    )


@pytest.mark.parametrize("analysis_kind", ["splf", "fr"])
def test_splf_fr_zero_start_plot_support_keeps_x_with_nonfinite_y(
    qapp,
    monkeypatch,
    captured_limit_utils,
    analysis_kind,
):
    measured_x = np.array([10.0, 20.0, 40.0])
    config = _frequency_analysis_config(
        _manual_upper_config(
            [{"start_x": 0.0, "start_y": 0.0, "end_x": 30.0, "end_y": 30.0}]
        )
    )
    widget, calculate = _frequency_analysis_widget(
        monkeypatch,
        analysis_kind,
        config,
        measured_x,
        np.array([np.nan, 1.0, 1.0]),
    )

    result = calculate()

    assert result is not False
    setup = captured_limit_utils["setup"][-1]
    np.testing.assert_allclose(setup["csv_x"], [10.0, 30.0])
    np.testing.assert_allclose(setup["csv_upper"], [10.0, 30.0])
    assert np.isnan(setup["csv_lower"]).all()
    check = captured_limit_utils["check"][-1]
    np.testing.assert_allclose(check["csv_x"], [20.0, 40.0])
    np.testing.assert_allclose(
        check["csv_upper"],
        [20.0, np.nan],
        equal_nan=True,
    )


def test_distortion_plot_graph_uses_manual_segments_for_out_of_range_highlight(
    qapp, captured_limit_utils
):
    widget = saw.Distortion("HD")
    config = _manual_upper_config(
        [{"start_x": 100.0, "start_y": 10.0, "end_x": 250.0, "end_y": 10.0}]
    )

    widget.plot_graph(
        np.array([100.0, 200.0, 300.0]),
        np.array([1.0, 12.0, 1.0]),
        config,
    )

    setup = captured_limit_utils["setup"][-1]
    np.testing.assert_allclose(setup["csv_x"], [100.0, 250.0])
    np.testing.assert_allclose(setup["csv_upper"], [10.0, 10.0])
    assert np.isnan(setup["csv_lower"]).all()
    out = captured_limit_utils["out"][-1]
    assert out["out_mask"].tolist() == [False, True, False]
    assert widget.data_struct.analysis_result_dict["HD"] == (False, 2.0)


@pytest.mark.parametrize(
    "display_config",
    [
        {},
        {
            "golden_sample_checked": True,
            "golden_sample_display_mode": "deviation",
        },
    ],
    ids=["normal", "deviation"],
)
@pytest.mark.parametrize(
    ("widget_class", "title"),
    [
        (saw.Distortion, "HD"),
        (saw.RubAndBuzz, "RB"),
    ],
)
def test_hd_rb_exact_plot_endpoints_keep_connector_judgment(
    qapp,
    captured_limit_utils,
    display_config,
    widget_class,
    title,
):
    widget = widget_class(title)
    config = _manual_upper_config(_exact_gap_segments())
    config.update(display_config)
    judgment = _capture_highlight_judgment(widget)
    measured_x = np.array([50.0, 100.0, 100.05, 110.0, 1000.0])

    widget.plot_graph(
        measured_x,
        np.array([0.0, 0.0, 100.0, 0.0, 0.0]),
        config,
    )

    setup = captured_limit_utils["setup"][-1]
    np.testing.assert_allclose(setup["csv_x"], [50.0, 100.0, 100.1, 1000.0])
    np.testing.assert_allclose(setup["csv_upper"], [4.0, 4.0, 3.0, 3.0])
    assert np.isnan(setup["csv_lower"]).all()
    np.testing.assert_allclose(judgment["limit_x"], measured_x)
    np.testing.assert_allclose(
        judgment["upper"],
        [np.nan, 4.0, 3.5, 3.0, 3.0],
        equal_nan=True,
    )
    assert captured_limit_utils["out"][-1]["out_mask"].tolist() == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert widget.data_struct.analysis_result_dict[title] == pytest.approx((False, 96.5))


def test_distortion_plot_graph_csv_passing_margin_preserves_legacy_limit_behavior(
    qapp, captured_limit_utils
):
    widget = saw.Distortion("HD")
    config = {
        "limit_checked": True,
        "limit_mode": "csv",
        "limit_data": (
            np.array([100.0, 200.0]),
            np.array([10.0, 100.0]),
            np.array([0.0, 0.0]),
        ),
    }

    widget.plot_graph(
        np.array([100.0, 200.0]),
        np.array([8.0, 9.0]),
        config,
    )

    out = captured_limit_utils["out"][-1]
    assert out["out_mask"].tolist() == [False, False]
    assert widget.data_struct.analysis_result_dict["HD"] == (True, 1.0)


def test_distortion_plot_graph_csv_upper_only_margin_preserves_legacy_nan_side(
    qapp, captured_limit_utils
):
    widget = saw.Distortion("HD")
    config = {
        "limit_checked": True,
        "limit_mode": "csv",
        "limit_data": (
            np.array([100.0, 200.0]),
            np.array([10.0, 10.0]),
            np.array([np.nan, np.nan]),
        ),
    }

    widget.plot_graph(
        np.array([100.0, 200.0]),
        np.array([8.0, 9.0]),
        config,
    )

    out = captured_limit_utils["out"][-1]
    assert out["out_mask"].tolist() == [False, False]
    assert widget.data_struct.analysis_result_dict["HD"] == (True, 1.0)


def test_distortion_plot_graph_csv_lower_only_margin_preserves_legacy_nan_side(
    qapp, captured_limit_utils
):
    widget = saw.Distortion("HD")
    config = {
        "limit_checked": True,
        "limit_mode": "csv",
        "limit_data": (
            np.array([100.0, 200.0]),
            np.array([np.nan, np.nan]),
            np.array([0.0, 0.0]),
        ),
    }

    widget.plot_graph(
        np.array([100.0, 200.0]),
        np.array([1.0, 2.0]),
        config,
    )

    out = captured_limit_utils["out"][-1]
    assert out["out_mask"].tolist() == [False, False]
    is_ok, deviation = widget.data_struct.analysis_result_dict["HD"]
    assert is_ok is True
    assert np.isnan(deviation)


def test_distortion_plot_graph_connector_sample_sets_passing_margin(
    qapp, captured_limit_utils
):
    widget = saw.Distortion("HD")
    config = _manual_upper_config(_exact_gap_segments())
    judgment = _capture_highlight_judgment(widget)

    widget.plot_graph(
        np.array([100.0, 100.05, 110.0]),
        np.array([0.0, 3.25, 0.0]),
        config,
    )

    setup = captured_limit_utils["setup"][-1]
    np.testing.assert_allclose(setup["csv_x"], [50.0, 100.0, 100.1, 1000.0])
    np.testing.assert_allclose(setup["csv_upper"], [4.0, 4.0, 3.0, 3.0])
    out = captured_limit_utils["out"][-1]
    assert judgment["upper"][1] == pytest.approx(3.5)
    assert out["out_mask"].tolist() == [False, False, False]
    assert widget.data_struct.analysis_result_dict["HD"] == pytest.approx((True, 0.25))


@pytest.mark.parametrize("nonfinite_y", [np.nan, np.inf])
def test_distortion_plot_graph_manual_margins_ignore_nonfinite_measured_values(
    qapp, captured_limit_utils, nonfinite_y
):
    widget = saw.Distortion("HD")
    config = _manual_upper_config(
        [{"start_x": 100.0, "start_y": 10.0, "end_x": 250.0, "end_y": 10.0}]
    )

    widget.plot_graph(
        np.array([100.0, 200.0]),
        np.array([1.0, nonfinite_y]),
        config,
    )

    out = captured_limit_utils["out"][-1]
    assert out["out_mask"].tolist() == [False, False]
    is_ok, deviation = widget.data_struct.analysis_result_dict["HD"]
    assert is_ok is True
    assert deviation == 0.0
    assert np.isfinite(deviation)


@pytest.mark.parametrize("uncovered_y", [10.0, 10.2])
def test_distortion_plot_graph_passing_margin_ignores_uncovered_manual_points(
    qapp, captured_limit_utils, uncovered_y
):
    widget = saw.Distortion("HD")
    config = _manual_upper_config(
        [{"start_x": 100.0, "start_y": 10.0, "end_x": 250.0, "end_y": 10.0}]
    )

    widget.plot_graph(
        np.array([100.0, 200.0]),
        np.array([uncovered_y, 8.0]),
        config,
    )

    setup = captured_limit_utils["setup"][-1]
    np.testing.assert_allclose(setup["csv_x"], [100.0, 250.0])
    np.testing.assert_allclose(setup["csv_upper"], [10.0, 10.0])
    out = captured_limit_utils["out"][-1]
    assert out["out_mask"].tolist() == [False, False]
    assert widget.data_struct.analysis_result_dict["HD"] == (True, 2.0)


def test_perceptual_rub_and_buzz_plot_graph_uses_manual_segments_for_out_of_range_highlight(
    qapp, captured_limit_utils
):
    widget = saw.PerceptualRubAndBuzz("PRB")
    config = _manual_upper_config(
        [{"start_x": 100.0, "start_y": 10.0, "end_x": 250.0, "end_y": 10.0}]
    )

    widget.plot_graph(
        np.array([100.0, 200.0, 300.0]),
        np.array([1.0, 12.0, 1.0]),
        config,
    )

    setup = captured_limit_utils["setup"][-1]
    np.testing.assert_allclose(setup["csv_x"], [100.0, 250.0])
    np.testing.assert_allclose(setup["csv_upper"], [10.0, 10.0])
    assert np.isnan(setup["csv_lower"]).all()
    out = captured_limit_utils["out"][-1]
    assert out["out_mask"].tolist() == [False, True, False]
    assert widget.data_struct.analysis_result_dict["PRB"] == (False, 2.0)


@pytest.mark.parametrize(
    "display_config",
    [
        {},
        {
            "golden_sample_checked": True,
            "golden_sample_display_mode": "deviation",
        },
    ],
    ids=["normal", "deviation"],
)
def test_perceptual_exact_plot_endpoints_keep_connector_judgment(
    qapp,
    captured_limit_utils,
    display_config,
):
    widget = saw.PerceptualRubAndBuzz("PRB")
    config = _manual_upper_config(_exact_gap_segments())
    config.update(display_config)
    judgment = _capture_highlight_judgment(widget)
    measured_x = np.array([50.0, 100.0, 100.05, 110.0, 1000.0])

    widget.plot_graph(
        measured_x,
        np.array([0.0, 0.0, 100.0, 0.0, 0.0]),
        config,
    )

    setup = captured_limit_utils["setup"][-1]
    np.testing.assert_allclose(setup["csv_x"], [50.0, 100.0, 100.1, 1000.0])
    np.testing.assert_allclose(setup["csv_upper"], [4.0, 4.0, 3.0, 3.0])
    assert np.isnan(setup["csv_lower"]).all()
    np.testing.assert_allclose(judgment["limit_x"], measured_x)
    np.testing.assert_allclose(
        judgment["upper"],
        [np.nan, 4.0, 3.5, 3.0, 3.0],
        equal_nan=True,
    )
    assert captured_limit_utils["out"][-1]["out_mask"].tolist() == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert widget.data_struct.analysis_result_dict["PRB"] == pytest.approx((False, 96.5))


@pytest.mark.parametrize(
    ("widget_class", "title"),
    [
        (saw.Distortion, "HD"),
        (saw.RubAndBuzz, "RB"),
        (saw.PerceptualRubAndBuzz, "PRB"),
    ],
)
def test_hd_rb_manual_envelope_uses_endpoint_geometry_but_measured_judgment(
    qapp,
    captured_limit_utils,
    widget_class,
    title,
):
    widget = widget_class(title)
    config = _manual_upper_config(_exact_gap_segments())
    config.update(
        {
            "golden_sample_checked": True,
            "golden_sample_display_mode": "envelope",
        }
    )
    judgment = _capture_highlight_judgment(widget)
    measured_x = np.array([50.0, 100.0, 100.05, 110.0, 1000.0])
    baseline = np.array([10.0, 12.0, 12.5, 13.0, 14.0])

    widget.plot_graph(
        measured_x,
        np.array([0.0, 0.0, 100.0, 0.0, 0.0]),
        config,
        raw_y=baseline,
        baseline_aligned=baseline,
    )

    setup = captured_limit_utils["setup"][-1]
    np.testing.assert_allclose(
        setup["csv_x"],
        [50.0, 100.0, 100.05, 100.1, 110.0, 1000.0],
    )
    endpoint_index = setup["csv_x"].tolist().index(100.1)
    assert setup["csv_upper"][endpoint_index] == pytest.approx(
        np.interp(100.1, measured_x, baseline) + 3.0
    )
    np.testing.assert_allclose(judgment["limit_x"], measured_x)
    np.testing.assert_allclose(
        judgment["upper"],
        [np.nan, 4.0, 3.5, 3.0, 3.0],
        equal_nan=True,
    )
    assert captured_limit_utils["out"][-1]["out_mask"].tolist() == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert widget.data_struct.analysis_result_dict[title] == pytest.approx((False, 96.5))


@pytest.mark.parametrize(
    ("widget_class", "title"),
    [
        (saw.Distortion, "HD"),
        (saw.PerceptualRubAndBuzz, "PRB"),
    ],
)
def test_hd_rb_zero_start_clips_plot_only_and_keeps_measured_judgment(
    qapp,
    captured_limit_utils,
    widget_class,
    title,
):
    widget = widget_class(title)
    config = _manual_upper_config(
        [{"start_x": 0.0, "start_y": 0.0, "end_x": 20.0, "end_y": 20.0}]
    )
    judgment = _capture_highlight_judgment(widget)
    measured_x = np.array([0.0, 10.0, 20.0, 30.0])

    widget.plot_graph(
        measured_x,
        np.array([100.0, 0.0, 0.0, 100.0]),
        config,
    )

    setup = captured_limit_utils["setup"][-1]
    np.testing.assert_allclose(setup["csv_x"], [10.0, 20.0])
    np.testing.assert_allclose(setup["csv_upper"], [10.0, 20.0])
    assert np.isnan(setup["csv_lower"]).all()
    np.testing.assert_allclose(judgment["limit_x"], measured_x)
    np.testing.assert_allclose(
        judgment["upper"],
        [np.nan, 10.0, 20.0, np.nan],
        equal_nan=True,
    )
    assert captured_limit_utils["out"][-1]["out_mask"].tolist() == [
        False,
        False,
        False,
        False,
    ]
    assert widget.data_struct.analysis_result_dict[title] == (True, 10.0)


def test_perceptual_rub_and_buzz_plot_graph_connector_sample_sets_passing_margin(
    qapp, captured_limit_utils
):
    widget = saw.PerceptualRubAndBuzz("PRB")
    config = _manual_upper_config(_exact_gap_segments())
    judgment = _capture_highlight_judgment(widget)

    widget.plot_graph(
        np.array([100.0, 100.05, 110.0]),
        np.array([0.0, 3.25, 0.0]),
        config,
    )

    setup = captured_limit_utils["setup"][-1]
    np.testing.assert_allclose(setup["csv_x"], [50.0, 100.0, 100.1, 1000.0])
    np.testing.assert_allclose(setup["csv_upper"], [4.0, 4.0, 3.0, 3.0])
    out = captured_limit_utils["out"][-1]
    assert judgment["upper"][1] == pytest.approx(3.5)
    assert out["out_mask"].tolist() == [False, False, False]
    assert widget.data_struct.analysis_result_dict["PRB"] == pytest.approx((True, 0.25))


@pytest.mark.parametrize("nonfinite_y", [np.nan, np.inf])
def test_perceptual_rub_and_buzz_manual_margins_ignore_nonfinite_measured_values(
    qapp, captured_limit_utils, nonfinite_y
):
    widget = saw.PerceptualRubAndBuzz("PRB")
    config = _manual_upper_config(
        [{"start_x": 100.0, "start_y": 10.0, "end_x": 250.0, "end_y": 10.0}]
    )

    widget.plot_graph(
        np.array([100.0, 200.0]),
        np.array([1.0, nonfinite_y]),
        config,
    )

    out = captured_limit_utils["out"][-1]
    assert out["out_mask"].tolist() == [False, False]
    is_ok, deviation = widget.data_struct.analysis_result_dict["PRB"]
    assert is_ok is True
    assert deviation == 0.0
    assert np.isfinite(deviation)


@pytest.mark.parametrize("uncovered_y", [10.0, 10.2])
def test_perceptual_rub_and_buzz_passing_margin_ignores_uncovered_manual_points(
    qapp, captured_limit_utils, uncovered_y
):
    widget = saw.PerceptualRubAndBuzz("PRB")
    config = _manual_upper_config(
        [{"start_x": 100.0, "start_y": 10.0, "end_x": 250.0, "end_y": 10.0}]
    )

    widget.plot_graph(
        np.array([100.0, 200.0]),
        np.array([uncovered_y, 8.0]),
        config,
    )

    setup = captured_limit_utils["setup"][-1]
    np.testing.assert_allclose(setup["csv_x"], [100.0, 250.0])
    np.testing.assert_allclose(setup["csv_upper"], [10.0, 10.0])
    out = captured_limit_utils["out"][-1]
    assert out["out_mask"].tolist() == [False, False]
    assert widget.data_struct.analysis_result_dict["PRB"] == (True, 2.0)


def test_scalar_only_manual_config_warns_and_returns_invalid_boundary(
    qapp, monkeypatch, captured_limit_utils
):
    warnings = []
    config = {
        "limit_checked": True,
        "limit_mode": "manual",
        "manual_upper_enabled": True,
        "manual_lower_enabled": False,
        "manual_upper": 8.0,
        "manual_lower": -999.0,
        "analysis_channel": 0,
        "weighting": "Z",
        "smooth_checked": False,
    }
    widget = _recording_widget(saw.Spl("SPL"), config)
    monkeypatch.setattr(widget, "_resolve_v2pa_factor_for_analysis", lambda: True)
    monkeypatch.setattr(
        saw.AudioThdFrequencyResponseAnalysis,
        "spl_calculation",
        lambda self, *args, **kwargs: np.array([12.0, 12.0, 12.0]),
    )
    monkeypatch.setattr(
        saw.MessageBox,
        "warning",
        lambda parent, title, message: warnings.append((title, message)),
    )

    result = widget.calculate_spl()

    assert result is False
    assert captured_limit_utils["setup"] == []
    assert warnings
    assert warnings[-1][0] == "提示"
    assert "上限" in warnings[-1][1]

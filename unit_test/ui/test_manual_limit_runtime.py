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
    captured = {"setup": [], "check": [], "out": []}
    original_check_interp_limits = saw.LimitPlotUtils.check_interp_limits

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
    np.testing.assert_allclose(setup["csv_x"], [0.0, 0.1, 0.2])
    assert np.isnan(setup["csv_upper"][0])
    np.testing.assert_allclose(setup["csv_upper"][1:], [8.0, 8.0])
    assert np.all(np.isnan(setup["csv_lower"]))
    out = captured_limit_utils["out"][-1]
    assert out["out_mask"].tolist() == [False, True, False]
    assert widget.data_struct.analysis_result_dict["SPL"] == (False, 4.0)


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
    np.testing.assert_allclose(setup["csv_x"], [100.0, 200.0, 300.0])
    _assert_upper_with_gap(setup["csv_upper"], 10.0)
    out = captured_limit_utils["out"][-1]
    assert out["out_mask"].tolist() == [False, True, False]
    assert widget.data_struct.analysis_result_dict["HD"] == (False, 2.0)


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


def test_distortion_plot_graph_manual_gap_keeps_passing_deviation_finite(
    qapp, captured_limit_utils
):
    widget = saw.Distortion("HD")
    config = _manual_upper_config(
        [{"start_x": 400.0, "start_y": 10.0, "end_x": 500.0, "end_y": 10.0}]
    )

    widget.plot_graph(
        np.array([100.0, 200.0, 300.0]),
        np.array([1.0, 2.0, 3.0]),
        config,
    )

    setup = captured_limit_utils["setup"][-1]
    assert np.all(np.isnan(setup["csv_upper"]))
    out = captured_limit_utils["out"][-1]
    assert out["out_mask"].tolist() == [False, False, False]
    is_ok, deviation = widget.data_struct.analysis_result_dict["HD"]
    assert is_ok is True
    assert deviation == 0.0
    assert np.isfinite(deviation)


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
    assert np.isnan(setup["csv_upper"][0])
    assert setup["csv_upper"][1] == pytest.approx(10.0)
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
    np.testing.assert_allclose(setup["csv_x"], [100.0, 200.0, 300.0])
    _assert_upper_with_gap(setup["csv_upper"], 10.0)
    out = captured_limit_utils["out"][-1]
    assert out["out_mask"].tolist() == [False, True, False]
    assert widget.data_struct.analysis_result_dict["PRB"] == (False, 2.0)


def test_perceptual_rub_and_buzz_plot_graph_manual_gap_keeps_passing_deviation_finite(
    qapp, captured_limit_utils
):
    widget = saw.PerceptualRubAndBuzz("PRB")
    config = _manual_upper_config(
        [{"start_x": 400.0, "start_y": 10.0, "end_x": 500.0, "end_y": 10.0}]
    )

    widget.plot_graph(
        np.array([100.0, 200.0, 300.0]),
        np.array([1.0, 2.0, 3.0]),
        config,
    )

    setup = captured_limit_utils["setup"][-1]
    assert np.all(np.isnan(setup["csv_upper"]))
    out = captured_limit_utils["out"][-1]
    assert out["out_mask"].tolist() == [False, False, False]
    is_ok, deviation = widget.data_struct.analysis_result_dict["PRB"]
    assert is_ok is True
    assert deviation == 0.0
    assert np.isfinite(deviation)


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
    assert np.isnan(setup["csv_upper"][0])
    assert setup["csv_upper"][1] == pytest.approx(10.0)
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

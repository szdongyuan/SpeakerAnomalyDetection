import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PyQt5.QtWidgets import QApplication
from pyqtgraph import ViewBox

from consts.acoustic_analysis.common_consts import (
    GOLDEN_SAMPLE_DISPLAY_DEVIATION,
    GOLDEN_SAMPLE_DISPLAY_ENVELOPE,
)
from ui import signal_analysis_window as saw


DUAL_MODES = (
    GOLDEN_SAMPLE_DISPLAY_DEVIATION,
    GOLDEN_SAMPLE_DISPLAY_ENVELOPE,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class GoldenRangeWidget(saw.AnalysisGraphWidget):
    plot_view_allow_x = True
    plot_view_allow_y = True

    def __init__(self, analysis_config=None):
        self.analysis_config = analysis_config or {}
        super().__init__()


def _dual_config(**overrides):
    config = {
        "golden_sample_checked": True,
        "golden_sample_display_modes": list(DUAL_MODES),
        "limit_checked": False,
    }
    config.update(overrides)
    return config


def _manual_limit_config(**overrides):
    config = _dual_config(
        limit_checked=True,
        limit_mode="manual",
        limit_data=None,
        manual_upper_enabled=True,
        manual_lower_enabled=True,
        manual_upper_segments=[
            {"start_x": 80.0, "start_y": 2.0, "end_x": 2000.0, "end_y": 2.0}
        ],
        manual_lower_segments=[
            {"start_x": 80.0, "start_y": -2.0, "end_x": 2000.0, "end_y": -2.0}
        ],
    )
    config.update(overrides)
    return config


def _plots(widget):
    return (
        widget.plot_for_golden_mode(GOLDEN_SAMPLE_DISPLAY_ENVELOPE),
        widget.plot_for_golden_mode(GOLDEN_SAMPLE_DISPLAY_DEVIATION),
    )


def _render_response_views(widget, analysis_config, frequency):
    widget.analysis_config = analysis_config
    selected_modes = saw._prepare_golden_response_plots(widget, analysis_config)
    frequency = np.asarray(frequency, dtype=float)
    deviation = np.linspace(-1.0, 1.0, frequency.size)
    baseline = np.full(frequency.size, 10.0)
    measured = baseline + deviation
    result = saw._plot_golden_response_views(
        widget,
        frequency,
        deviation,
        measured,
        baseline,
        analysis_config,
        selected_modes,
        "SPL (dB)" if widget.title_name == "SPLF" else "Amplitude (dB)",
    )
    assert result is not False
    return result


def _fixed_x_config(config, x_min=4500.0, x_max=7500.0):
    return {
        **config,
        "display": {
            "plot_view": {
                "x_enabled": True,
                "x_min": x_min,
                "x_max": x_max,
                "y_enabled": False,
            }
        },
    }


def _assert_single_x_auto_range(widget, expected_min, expected_max):
    view_box = widget._primary_analysis_plot.getViewBox()
    x_range = view_box.viewRange()[0]
    assert bool(view_box.autoRangeEnabled()[0]) is True
    assert x_range[0] <= math.log10(expected_min)
    assert x_range[1] >= math.log10(expected_max)
    return x_range


def _assert_single_fixed_x_range(widget, expected_min=4500.0, expected_max=7500.0):
    view_box = widget._primary_analysis_plot.getViewBox()
    assert bool(view_box.autoRangeEnabled()[0]) is False
    assert view_box.viewRange()[0] == pytest.approx(
        [math.log10(expected_min), math.log10(expected_max)]
    )


def _assert_finalized_dual(widget, expected_min=80.0, expected_max=2000.0):
    envelope, deviation = _plots(widget)
    envelope_view = envelope.getViewBox()
    deviation_view = deviation.getViewBox()
    x_range = envelope_view.viewRange()[0]

    assert envelope.getAxis("bottom").logMode is True
    assert deviation.getAxis("bottom").logMode is True
    assert deviation_view.linkedView(ViewBox.XAxis) is envelope_view
    assert envelope_view.linkedView(ViewBox.YAxis) is None
    assert deviation_view.linkedView(ViewBox.YAxis) is None
    assert bool(envelope_view.autoRangeEnabled()[1]) is True
    assert bool(deviation_view.autoRangeEnabled()[1]) is True
    assert x_range[0] <= math.log10(expected_min)
    assert x_range[1] >= math.log10(expected_max)
    assert deviation_view.viewRange()[0] == pytest.approx(x_range)
    assert "27" not in str(envelope.getAxis("bottom").labelUnitPrefix)
    assert "27" not in str(deviation.getAxis("bottom").labelUnitPrefix)
    return x_range


def test_dual_configuration_unlinks_follower_until_post_render_finalization(qapp):
    widget = GoldenRangeWidget()
    plots = widget.configure_golden_sample_plots(DUAL_MODES)
    primary = plots[GOLDEN_SAMPLE_DISPLAY_ENVELOPE]
    follower = plots[GOLDEN_SAMPLE_DISPLAY_DEVIATION]
    primary.setLogMode(x=False, y=False)
    primary.setXRange(-1.0e27, 1.0e27, padding=0.0)
    follower.setLogMode(x=True, y=False)
    follower.setXRange(1.0, 2.0, padding=0.0)

    plots = widget.configure_golden_sample_plots(DUAL_MODES)

    assert plots[GOLDEN_SAMPLE_DISPLAY_DEVIATION].getViewBox().linkedView(
        ViewBox.XAxis
    ) is None
    widget.close()


def test_dual_finalizer_auto_ranges_each_finite_plot_before_union_and_link(
    qapp, monkeypatch
):
    widget = GoldenRangeWidget()
    plots = widget.configure_golden_sample_plots(DUAL_MODES)
    envelope = plots[GOLDEN_SAMPLE_DISPLAY_ENVELOPE]
    deviation = plots[GOLDEN_SAMPLE_DISPLAY_DEVIATION]
    envelope.plot([100.0, 1000.0], [10.0, 12.0])
    deviation.plot([80.0, 2000.0], [-1.0, 1.0])
    envelope.setLogMode(x=True, y=False)
    deviation.setLogMode(x=True, y=False)
    calls = []

    for name, plot in (("envelope", envelope), ("deviation", deviation)):
        view_box = plot.getViewBox()
        original_auto_range = view_box.autoRange

        def record_auto_range(*args, _name=name, _original=original_auto_range, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(view_box, "autoRange", record_auto_range)

    original_set_x_link = deviation.setXLink

    def record_set_x_link(target):
        if target is not None:
            calls.append("link")
        return original_set_x_link(target)

    monkeypatch.setattr(deviation, "setXLink", record_set_x_link)

    widget._finalize_plot_view_ranges_after_render()

    initial_range = _assert_finalized_dual(widget)
    assert calls == ["envelope", "deviation", "link"]

    envelope.getViewBox().autoRange()
    after_auto_range = envelope.getViewBox().viewRange()[0]
    assert initial_range[0] <= after_auto_range[0]
    assert initial_range[1] >= after_auto_range[1]
    widget.close()


def test_dual_finalizer_preserves_configured_log_x_range(qapp, monkeypatch):
    config = _dual_config(
        display={
            "plot_view": {
                "x_enabled": True,
                "x_min": 100.0,
                "x_max": 1000.0,
                "y_enabled": False,
            }
        }
    )
    widget = GoldenRangeWidget(config)
    plots = widget.configure_golden_sample_plots(DUAL_MODES)
    for plot in plots.values():
        plot.plot([80.0, 2000.0], [1.0, 2.0])
        plot.setLogMode(x=True, y=False)
        monkeypatch.setattr(
            plot.getViewBox(),
            "autoRange",
            lambda: pytest.fail("configured X range must not auto-range"),
        )

    widget._finalize_plot_view_ranges_after_render()

    envelope, deviation = _plots(widget)
    assert envelope.getViewBox().viewRange()[0] == pytest.approx([2.0, 3.0])
    assert deviation.getViewBox().viewRange()[0] == pytest.approx([2.0, 3.0])
    assert envelope.getViewBox().autoRangeEnabled()[0] is False
    assert deviation.getViewBox().autoRangeEnabled()[0] is False
    assert deviation.getViewBox().linkedView(ViewBox.XAxis) is envelope.getViewBox()
    widget.close()


@pytest.mark.parametrize("finite_mode", [None, "envelope", "deviation"])
def test_empty_or_one_sided_dual_finalization_does_not_use_empty_peer_range(
    qapp, finite_mode
):
    widget = GoldenRangeWidget()
    plots = widget.configure_golden_sample_plots(DUAL_MODES)
    envelope = plots[GOLDEN_SAMPLE_DISPLAY_ENVELOPE]
    deviation = plots[GOLDEN_SAMPLE_DISPLAY_DEVIATION]
    envelope.setXRange(7.0, 8.0, padding=0.0)
    initial_primary_range = envelope.getViewBox().viewRange()[0]
    if finite_mode is not None:
        plots[finite_mode].plot([80.0, 2000.0], [1.0, 2.0])

    widget._finalize_plot_view_ranges_after_render()

    assert envelope.getAxis("bottom").logMode is True
    assert deviation.getAxis("bottom").logMode is True
    assert deviation.getViewBox().linkedView(ViewBox.XAxis) is envelope.getViewBox()
    if finite_mode is None:
        assert envelope.getViewBox().viewRange()[0] == pytest.approx(
            initial_primary_range
        )
    else:
        x_range = envelope.getViewBox().viewRange()[0]
        assert x_range[0] <= math.log10(80.0)
        assert x_range[1] >= math.log10(2000.0)
        assert x_range[0] > 0.0
    widget.close()


@pytest.mark.parametrize(
    ("widget_class", "title"),
    [
        (saw.Distortion, "HD"),
        (saw.RubAndBuzz, "RB"),
        (saw.PerceptualRubAndBuzz, "PRB"),
    ],
)
@pytest.mark.parametrize("analysis_config", [_dual_config(), _manual_limit_config()])
def test_distortion_family_dual_render_is_synchronously_finalized(
    qapp, widget_class, title, analysis_config
):
    widget = widget_class(title)
    freq = np.array([80.0, 100.0, 1000.0, 2000.0])

    widget.plot_graph(
        freq,
        np.array([-1.0, 0.0, 1.0, 0.5]),
        analysis_config,
        raw_y=np.array([9.0, 10.0, 11.0, 10.5]),
        baseline_aligned=np.array([10.0, 10.0, 10.0, 10.0]),
    )

    assert widget.isVisible() is False
    initial_range = _assert_finalized_dual(widget)
    widget.show()
    qapp.processEvents()
    shown_range = widget._primary_analysis_plot.getViewBox().viewRange()[0]
    assert shown_range[0] <= initial_range[0] + 1e-9
    assert shown_range[1] >= initial_range[1] - 1e-9
    widget.close()


@pytest.mark.parametrize(
    ("widget_class", "title"),
    [
        (saw.Distortion, "HD"),
        (saw.RubAndBuzz, "RB"),
        (saw.PerceptualRubAndBuzz, "PRB"),
    ],
)
def test_distortion_family_repeated_render_and_single_dual_transitions_use_fresh_range(
    qapp, widget_class, title
):
    widget = widget_class(title)
    dual = _dual_config()
    widget.plot_graph(
        np.array([80.0, 2000.0]),
        np.array([-1.0, 1.0]),
        dual,
        raw_y=np.array([9.0, 11.0]),
        baseline_aligned=np.array([10.0, 10.0]),
    )
    widget.show()
    qapp.processEvents()
    assert widget.isVisible() is True

    single = _dual_config(golden_sample_display_modes=[GOLDEN_SAMPLE_DISPLAY_ENVELOPE])
    widget.analysis_config = single
    widget.plot_graph(
        np.array([4000.0, 8000.0]),
        np.array([-1.0, 1.0]),
        single,
        raw_y=np.array([9.0, 11.0]),
        baseline_aligned=np.array([10.0, 10.0]),
    )
    secondary = widget._secondary_analysis_plot
    assert secondary.getViewBox().linkedView(ViewBox.XAxis) is None
    assert widget._primary_analysis_plot.getViewBox().linkedView(ViewBox.XAxis) is None
    _assert_single_x_auto_range(widget, 4000.0, 8000.0)

    widget.plot_graph(
        np.array([100.0, 1000.0]),
        np.array([-1.0, 1.0]),
        dual,
        raw_y=np.array([9.0, 11.0]),
        baseline_aligned=np.array([10.0, 10.0]),
    )
    _assert_finalized_dual(widget, 100.0, 1000.0)

    single_deviation = _fixed_x_config(
        _dual_config(golden_sample_display_modes=[GOLDEN_SAMPLE_DISPLAY_DEVIATION])
    )
    widget.analysis_config = single_deviation
    widget.plot_graph(
        np.array([4000.0, 8000.0]),
        np.array([-1.0, 1.0]),
        single_deviation,
        raw_y=np.array([9.0, 11.0]),
        baseline_aligned=np.array([10.0, 10.0]),
    )
    assert secondary.getViewBox().linkedView(ViewBox.XAxis) is None
    assert widget._primary_analysis_plot.getViewBox().linkedView(ViewBox.XAxis) is None
    _assert_single_fixed_x_range(widget)
    widget.close()


@pytest.mark.parametrize(
    ("widget_class", "title"),
    [(saw.SplFrequency, "SPLF"), (saw.Frequency, "FR")],
)
def test_response_family_hidden_visible_repeated_and_single_dual_transitions(
    qapp, widget_class, title
):
    widget = widget_class(title)
    dual = _dual_config()

    _render_response_views(widget, dual, [80.0, 2000.0])
    assert widget.isVisible() is False
    hidden_range = _assert_finalized_dual(widget)

    widget.show()
    qapp.processEvents()
    assert widget.isVisible() is True
    shown_range = widget._primary_analysis_plot.getViewBox().viewRange()[0]
    assert shown_range == pytest.approx(hidden_range)

    _render_response_views(widget, dual, [4000.0, 8000.0])
    repeated_range = _assert_finalized_dual(widget, 4000.0, 8000.0)
    assert repeated_range[0] > math.log10(2000.0)

    envelope_only = _dual_config(
        golden_sample_display_modes=[GOLDEN_SAMPLE_DISPLAY_ENVELOPE]
    )
    _render_response_views(widget, envelope_only, [10000.0, 16000.0])
    secondary = widget._secondary_analysis_plot
    assert secondary.getViewBox().linkedView(ViewBox.XAxis) is None
    assert widget._primary_analysis_plot.getViewBox().linkedView(ViewBox.XAxis) is None
    _assert_single_x_auto_range(widget, 10000.0, 16000.0)

    _render_response_views(widget, dual, [100.0, 1000.0])
    _assert_finalized_dual(widget, 100.0, 1000.0)

    deviation_only = _fixed_x_config(
        _dual_config(golden_sample_display_modes=[GOLDEN_SAMPLE_DISPLAY_DEVIATION])
    )
    _render_response_views(widget, deviation_only, [4000.0, 8000.0])
    assert secondary.getViewBox().linkedView(ViewBox.XAxis) is None
    assert widget._primary_analysis_plot.getViewBox().linkedView(ViewBox.XAxis) is None
    _assert_single_fixed_x_range(widget)
    widget.close()


@pytest.mark.parametrize(
    ("widget_class", "title", "family"),
    [
        (saw.SplFrequency, "SPLF", "response"),
        (saw.Frequency, "FR", "response"),
        (saw.Distortion, "HD", "distortion"),
        (saw.RubAndBuzz, "RB", "distortion"),
        (saw.PerceptualRubAndBuzz, "PRB", "distortion"),
    ],
)
@pytest.mark.parametrize("range_policy", ["automatic", "invalid", "fixed"])
def test_supported_analysis_dual_to_non_golden_refreshes_x_range_after_render(
    qapp, widget_class, title, family, range_policy
):
    widget = widget_class(title)
    dual = _dual_config()
    if family == "response":
        _render_response_views(widget, dual, [80.0, 2000.0])
    else:
        widget.plot_graph(
            np.array([80.0, 2000.0]),
            np.array([-1.0, 1.0]),
            dual,
            raw_y=np.array([9.0, 11.0]),
            baseline_aligned=np.array([10.0, 10.0]),
        )
    widget.show()
    qapp.processEvents()
    _assert_finalized_dual(widget)

    non_golden = {
        "golden_sample_checked": False,
        "limit_checked": False,
    }
    if range_policy == "invalid":
        non_golden = _fixed_x_config(non_golden, x_min=0.0, x_max=1000.0)
    elif range_policy == "fixed":
        non_golden = _fixed_x_config(non_golden)
    frequency = np.array([4000.0, 8000.0])
    values = np.array([1.0, 2.0])
    if family == "response":
        widget.analysis_config = non_golden
        assert saw._prepare_golden_response_plots(widget, non_golden) == ()
        if title == "SPLF":
            widget.plot_spl_frequency(frequency, values)
        else:
            widget.plot_fr(frequency, values)
    else:
        widget.analysis_config = non_golden
        widget.plot_graph(frequency, values, non_golden)

    assert widget.golden_plot_widgets == {}
    assert widget._primary_analysis_plot.getViewBox().linkedView(ViewBox.XAxis) is None
    if range_policy == "fixed":
        _assert_single_fixed_x_range(widget)
    else:
        _assert_single_x_auto_range(widget, 4000.0, 8000.0)
    widget.close()


@pytest.mark.parametrize(
    ("widget_class", "title", "family"),
    [
        (saw.SplFrequency, "SPLF", "response"),
        (saw.Frequency, "FR", "response"),
        (saw.Distortion, "HD", "distortion"),
        (saw.RubAndBuzz, "RB", "distortion"),
        (saw.PerceptualRubAndBuzz, "PRB", "distortion"),
    ],
)
def test_supported_analysis_one_shot_auto_range_reveals_no_missing_x_extent(
    qapp, widget_class, title, family
):
    widget = widget_class(title)
    frequency = np.array([80.0, 100.0, 1000.0, 2000.0])
    config = _dual_config()
    if family == "response":
        _render_response_views(widget, config, frequency)
    else:
        widget.plot_graph(
            frequency,
            np.array([-1.0, 0.0, 1.0, 0.5]),
            config,
            raw_y=np.array([9.0, 10.0, 11.0, 10.5]),
            baseline_aligned=np.array([10.0, 10.0, 10.0, 10.0]),
        )

    widget.show()
    qapp.processEvents()
    initial_range = _assert_finalized_dual(widget)

    widget._primary_analysis_plot.getViewBox().autoRange()
    after_auto_range = widget._primary_analysis_plot.getViewBox().viewRange()[0]

    assert after_auto_range[0] <= math.log10(frequency.min())
    assert after_auto_range[1] >= math.log10(frequency.max())
    data_span = math.log10(frequency.max()) - math.log10(frequency.min())
    ordinary_padding_tolerance = data_span * 0.02
    assert after_auto_range[0] >= initial_range[0] - ordinary_padding_tolerance
    assert after_auto_range[1] <= initial_range[1] + ordinary_padding_tolerance
    widget.close()


@pytest.mark.parametrize(
    ("widget_class", "title"),
    [(saw.SplFrequency, "SPLF"), (saw.Frequency, "FR")],
)
@pytest.mark.parametrize("analysis_config", [_dual_config(), _manual_limit_config()])
def test_response_family_dual_render_is_synchronously_finalized(
    qapp, widget_class, title, analysis_config
):
    widget = widget_class(title)
    widget.analysis_config = analysis_config
    selected_modes = saw._prepare_golden_response_plots(widget, analysis_config)
    freq = np.array([80.0, 100.0, 1000.0, 2000.0])

    result = saw._plot_golden_response_views(
        widget,
        freq,
        np.array([-1.0, 0.0, 1.0, 0.5]),
        np.array([9.0, 10.0, 11.0, 10.5]),
        np.array([10.0, 10.0, 10.0, 10.0]),
        analysis_config,
        selected_modes,
        "SPL (dB)" if title == "SPLF" else "Amplitude (dB)",
    )

    assert result is not False
    _assert_finalized_dual(widget)
    widget.close()


@pytest.mark.parametrize(
    ("widget_class", "title"),
    [
        (saw.Distortion, "HD"),
        (saw.RubAndBuzz, "RB"),
        (saw.PerceptualRubAndBuzz, "PRB"),
    ],
)
@pytest.mark.parametrize("failure_kind", ["manual", "csv"])
def test_distortion_limit_early_return_clears_and_relinks_dual_plots(
    qapp, monkeypatch, widget_class, title, failure_kind
):
    widget = widget_class(title)
    warnings = []
    monkeypatch.setattr(
        saw.MessageBox,
        "warning",
        lambda parent, warning_title, message: warnings.append(message),
    )
    config = (
        _manual_limit_config(manual_upper_segments=[], manual_lower_segments=[])
        if failure_kind == "manual"
        else _dual_config(limit_checked=True, limit_mode="csv", limit_data=None)
    )

    payload = widget.plot_graph(
        np.array([80.0, 2000.0]),
        np.array([-1.0, 1.0]),
        config,
        raw_y=np.array([9.0, 11.0]),
        baseline_aligned=np.array([10.0, 10.0]),
    )

    envelope, deviation = _plots(widget)
    assert payload["series"] == {
        "deviation": {"available": False},
        "envelope": {"available": False},
    }
    assert envelope.listDataItems() == []
    assert deviation.listDataItems() == []
    assert envelope.getAxis("bottom").logMode is True
    assert deviation.getAxis("bottom").logMode is True
    assert deviation.getViewBox().linkedView(ViewBox.XAxis) is envelope.getViewBox()
    assert bool(warnings) is (failure_kind == "manual")
    widget.close()


@pytest.mark.parametrize("failure_kind", ["manual", "csv"])
def test_response_limit_early_return_clears_and_relinks_dual_plots(
    qapp, monkeypatch, failure_kind
):
    widget = saw.SplFrequency("SPLF")
    warnings = []
    monkeypatch.setattr(
        saw.MessageBox,
        "warning",
        lambda parent, warning_title, message: warnings.append(message),
    )
    config = (
        _manual_limit_config(manual_upper_segments=[], manual_lower_segments=[])
        if failure_kind == "manual"
        else _dual_config(limit_checked=True, limit_mode="csv", limit_data=None)
    )
    widget.analysis_config = config
    selected_modes = saw._prepare_golden_response_plots(widget, config)

    result = saw._plot_golden_response_views(
        widget,
        np.array([80.0, 2000.0]),
        np.array([-1.0, 1.0]),
        np.array([9.0, 11.0]),
        np.array([10.0, 10.0]),
        config,
        selected_modes,
        "SPL (dB)",
    )

    envelope, deviation = _plots(widget)
    assert result is False
    assert envelope.listDataItems() == []
    assert deviation.listDataItems() == []
    assert envelope.getAxis("bottom").logMode is True
    assert deviation.getAxis("bottom").logMode is True
    assert deviation.getViewBox().linkedView(ViewBox.XAxis) is envelope.getViewBox()
    assert bool(warnings) is (failure_kind == "manual")
    widget.close()


@pytest.mark.parametrize(
    ("widget_class", "title", "calculate_name"),
    [
        (saw.SplFrequency, "SPLF", "calculate_spl"),
        (saw.Frequency, "FR", "calculate_fr"),
    ],
)
def test_response_missing_analysis_data_relinks_empty_dual_layout(
    qapp, monkeypatch, widget_class, title, calculate_name
):
    widget = widget_class(title)
    widget.analysis_config = _dual_config()
    monkeypatch.setattr(saw.MessageBox, "warning", lambda *args: None)

    result = getattr(widget, calculate_name)()

    envelope, deviation = _plots(widget)
    assert result["frequency_list"] == []
    assert envelope.listDataItems() == []
    assert deviation.listDataItems() == []
    assert deviation.getViewBox().linkedView(ViewBox.XAxis) is envelope.getViewBox()
    widget.close()


@pytest.mark.parametrize(
    ("widget_class", "title"),
    [
        (saw.Distortion, "HD"),
        (saw.RubAndBuzz, "RB"),
        (saw.PerceptualRubAndBuzz, "PRB"),
    ],
)
def test_distortion_empty_data_relinks_without_fabricating_curves(
    qapp, widget_class, title
):
    widget = widget_class(title)

    payload = widget.plot_graph(
        [], [], _dual_config(), raw_y=[], baseline_aligned=None
    )

    envelope, deviation = _plots(widget)
    assert payload["series"] == {
        "deviation": {"available": False},
        "envelope": {"available": False},
    }
    assert envelope.listDataItems() == []
    assert deviation.listDataItems() == []
    assert deviation.getViewBox().linkedView(ViewBox.XAxis) is envelope.getViewBox()
    widget.close()

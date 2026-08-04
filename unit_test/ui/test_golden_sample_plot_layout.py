import json
import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication
from pyqtgraph import ViewBox

import ui.signal_analysis_window as signal_analysis_window
from consts.acoustic_analysis.common_consts import (
    GOLDEN_SAMPLE_DISPLAY_DEVIATION,
    GOLDEN_SAMPLE_DISPLAY_ENVELOPE,
)
from ui.signal_analysis_window import AnalysisGraphWidget


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class GoldenPlotWidget(AnalysisGraphWidget):
    plot_view_allow_x = True
    plot_view_allow_y = True


def _close(*widgets):
    for widget in widgets:
        widget.close()
        widget.deleteLater()
    QApplication.processEvents()


@pytest.mark.parametrize(
    "mode",
    [GOLDEN_SAMPLE_DISPLAY_DEVIATION, GOLDEN_SAMPLE_DISPLAY_ENVELOPE],
)
def test_single_mode_uses_existing_plot_as_the_only_splitter_plot(qapp, mode):
    widget = GoldenPlotWidget()
    try:
        primary = widget.analysis_plot

        plots = widget.configure_golden_sample_plots((mode,))

        assert plots == {mode: primary}
        assert widget.analysis_plot is primary
        assert widget.plot_splitter.count() == 1
        assert widget.plot_splitter.widget(0) is primary
        assert tuple(widget.iter_analysis_plots()) == (primary,)
        assert widget.plot_for_golden_mode(mode) is primary
    finally:
        _close(widget)


def test_dual_mode_orders_equal_height_linked_x_only_plots(qapp):
    widget = GoldenPlotWidget()
    try:
        widget.resize(640, 480)
        plots = widget.configure_golden_sample_plots(
            (GOLDEN_SAMPLE_DISPLAY_DEVIATION, GOLDEN_SAMPLE_DISPLAY_ENVELOPE)
        )
        widget.show()
        QApplication.processEvents()

        envelope = plots[GOLDEN_SAMPLE_DISPLAY_ENVELOPE]
        deviation = plots[GOLDEN_SAMPLE_DISPLAY_DEVIATION]
        assert tuple(plots) == (
            GOLDEN_SAMPLE_DISPLAY_DEVIATION,
            GOLDEN_SAMPLE_DISPLAY_ENVELOPE,
        )
        assert widget.plot_splitter.widget(0) is envelope
        assert widget.plot_splitter.widget(1) is deviation
        assert widget.plot_splitter.orientation() == signal_analysis_window.Qt.Vertical
        assert widget.analysis_plot is envelope
        assert tuple(widget.iter_analysis_plots()) == (envelope, deviation)

        sizes = widget.plot_splitter.sizes()
        assert len(sizes) == 2
        assert min(sizes) > 0
        assert abs(sizes[0] - sizes[1]) <= 1
        assert deviation.getViewBox().linkedView(ViewBox.XAxis) is envelope.getViewBox()
        assert deviation.getViewBox().linkedView(ViewBox.YAxis) is None
    finally:
        _close(widget)


def test_secondary_plot_is_lazy_and_instance_scoped(qapp):
    first = GoldenPlotWidget()
    second = GoldenPlotWidget()
    try:
        assert first._secondary_analysis_plot is None
        assert second._secondary_analysis_plot is None

        first.configure_golden_sample_plots(
            (GOLDEN_SAMPLE_DISPLAY_DEVIATION, GOLDEN_SAMPLE_DISPLAY_ENVELOPE)
        )

        assert first._secondary_analysis_plot is not None
        assert second._secondary_analysis_plot is None
        assert first._secondary_analysis_plot is not second.analysis_plot
    finally:
        _close(first, second)


def test_set_plot_font_size_styles_both_active_plots(qapp, monkeypatch):
    widget = GoldenPlotWidget()
    try:
        monkeypatch.setattr(signal_analysis_window.ui_style_const, "scale_size_px", lambda value: value)
        plots = widget.configure_golden_sample_plots(
            (GOLDEN_SAMPLE_DISPLAY_DEVIATION, GOLDEN_SAMPLE_DISPLAY_ENVELOPE)
        )

        widget.set_plot_font_size(17)

        for plot in plots.values():
            assert plot.getAxis("bottom").style["tickFont"].pixelSize() == 17
            assert plot.getAxis("left").style["tickFont"].pixelSize() == 17
            assert plot.backgroundBrush().color().name() == "#ffffff"
    finally:
        _close(widget)


def test_reactivated_secondary_receives_current_plot_style(qapp, monkeypatch):
    widget = GoldenPlotWidget()
    try:
        monkeypatch.setattr(signal_analysis_window.ui_style_const, "scale_size_px", lambda value: value)
        widget.configure_golden_sample_plots(
            (GOLDEN_SAMPLE_DISPLAY_DEVIATION, GOLDEN_SAMPLE_DISPLAY_ENVELOPE)
        )
        widget.configure_golden_sample_plots((GOLDEN_SAMPLE_DISPLAY_ENVELOPE,))
        widget.set_plot_font_size(17)

        plots = widget.configure_golden_sample_plots(
            (GOLDEN_SAMPLE_DISPLAY_DEVIATION, GOLDEN_SAMPLE_DISPLAY_ENVELOPE)
        )

        secondary = plots[GOLDEN_SAMPLE_DISPLAY_DEVIATION]
        assert secondary.getAxis("bottom").style["tickFont"].pixelSize() == 17
        assert secondary.getAxis("left").style["tickFont"].pixelSize() == 17
        assert secondary.backgroundBrush().color().name() == "#ffffff"
    finally:
        _close(widget)


def test_show_event_applies_configured_ranges_to_single_selected_plot(qapp, monkeypatch):
    widget = GoldenPlotWidget()
    calls = []
    try:
        widget.analysis_config = {"display": {"plot_view": {"x_enabled": True, "y_enabled": True}}}
        selected = widget.configure_golden_sample_plots((GOLDEN_SAMPLE_DISPLAY_DEVIATION,))
        monkeypatch.setattr(
            signal_analysis_window,
            "apply_plot_view_range",
            lambda plot, config, allow_x, allow_y: calls.append(
                (plot, config, allow_x, allow_y)
            ),
        )

        widget.show()
        QApplication.processEvents()

        assert calls == [
            (
                selected[GOLDEN_SAMPLE_DISPLAY_DEVIATION],
                widget.analysis_config,
                True,
                True,
            )
        ]
    finally:
        _close(widget)


def test_show_event_applies_shared_x_but_auto_ranges_each_dual_y(qapp, monkeypatch):
    widget = GoldenPlotWidget()
    calls = []
    auto_range_calls = []
    try:
        widget.analysis_config = {"display": {"plot_view": {"x_enabled": True, "y_enabled": True}}}
        plots = widget.configure_golden_sample_plots(
            (GOLDEN_SAMPLE_DISPLAY_DEVIATION, GOLDEN_SAMPLE_DISPLAY_ENVELOPE)
        )
        for mode, plot in plots.items():
            monkeypatch.setattr(
                plot,
                "enableAutoRange",
                lambda axis=None, _mode=mode: auto_range_calls.append((_mode, axis)),
            )
        monkeypatch.setattr(
            signal_analysis_window,
            "apply_plot_view_range",
            lambda plot, config, allow_x, allow_y: calls.append(
                (plot, config, allow_x, allow_y)
            ),
        )

        widget.show()
        QApplication.processEvents()

        assert calls == [
            (plots[GOLDEN_SAMPLE_DISPLAY_ENVELOPE], widget.analysis_config, True, False),
            (plots[GOLDEN_SAMPLE_DISPLAY_DEVIATION], widget.analysis_config, True, False),
        ]
        assert auto_range_calls == [
            (GOLDEN_SAMPLE_DISPLAY_ENVELOPE, ViewBox.YAxis),
            (GOLDEN_SAMPLE_DISPLAY_DEVIATION, ViewBox.YAxis),
        ]
    finally:
        _close(widget)


def test_visible_single_fixed_y_reconfiguration_enables_dual_y_auto_range(qapp):
    widget = GoldenPlotWidget()
    try:
        widget.analysis_config = {
            "display": {
                "plot_view": {
                    "x_enabled": True,
                    "x_min": 100.0,
                    "x_max": 1000.0,
                    "y_enabled": True,
                    "y_min": -5.0,
                    "y_max": 5.0,
                }
            }
        }
        single = widget.configure_golden_sample_plots(
            (GOLDEN_SAMPLE_DISPLAY_DEVIATION,)
        )
        widget.show()
        QApplication.processEvents()
        assert single[GOLDEN_SAMPLE_DISPLAY_DEVIATION].getViewBox().autoRangeEnabled()[1] is False

        dual = widget.configure_golden_sample_plots(
            (GOLDEN_SAMPLE_DISPLAY_DEVIATION, GOLDEN_SAMPLE_DISPLAY_ENVELOPE)
        )
        QApplication.processEvents()

        assert bool(
            dual[GOLDEN_SAMPLE_DISPLAY_ENVELOPE].getViewBox().autoRangeEnabled()[1]
        )
        assert bool(
            dual[GOLDEN_SAMPLE_DISPLAY_DEVIATION].getViewBox().autoRangeEnabled()[1]
        )
        dual[GOLDEN_SAMPLE_DISPLAY_ENVELOPE].plot(
            [100.0, 200.0],
            [20.0, 30.0],
        )
        dual[GOLDEN_SAMPLE_DISPLAY_DEVIATION].plot(
            [100.0, 200.0],
            [-30.0, -20.0],
        )
        QApplication.processEvents()
        envelope_y_range = dual[
            GOLDEN_SAMPLE_DISPLAY_ENVELOPE
        ].getViewBox().viewRange()[1]
        deviation_y_range = dual[
            GOLDEN_SAMPLE_DISPLAY_DEVIATION
        ].getViewBox().viewRange()[1]
        assert envelope_y_range[0] <= 20.0 <= 30.0 <= envelope_y_range[1]
        assert deviation_y_range[0] <= -30.0 <= -20.0 <= deviation_y_range[1]
    finally:
        _close(widget)


def test_visible_dual_to_single_reconfiguration_restores_configured_xy_ranges(qapp):
    widget = GoldenPlotWidget()
    try:
        widget.analysis_config = {
            "display": {
                "plot_view": {
                    "x_enabled": True,
                    "x_min": 100.0,
                    "x_max": 1000.0,
                    "y_enabled": True,
                    "y_min": -5.0,
                    "y_max": 5.0,
                }
            }
        }
        dual = widget.configure_golden_sample_plots(
            (GOLDEN_SAMPLE_DISPLAY_DEVIATION, GOLDEN_SAMPLE_DISPLAY_ENVELOPE)
        )
        widget.show()
        QApplication.processEvents()
        assert bool(
            dual[GOLDEN_SAMPLE_DISPLAY_ENVELOPE].getViewBox().autoRangeEnabled()[1]
        )

        single = widget.configure_golden_sample_plots(
            (GOLDEN_SAMPLE_DISPLAY_ENVELOPE,)
        )
        QApplication.processEvents()

        primary_view = single[GOLDEN_SAMPLE_DISPLAY_ENVELOPE].getViewBox()
        assert primary_view.autoRangeEnabled()[1] is False
        assert primary_view.viewRange()[0] == pytest.approx([100.0, 1000.0])
        assert primary_view.viewRange()[1] == pytest.approx([-5.0, 5.0])
    finally:
        _close(widget)


def _fixed_xy_plot_config(*, golden_checked, modes=None):
    config = {
        "golden_sample_checked": golden_checked,
        "limit_checked": False,
        "display": {
            "plot_view": {
                "x_enabled": True,
                "x_min": 100.0,
                "x_max": 1000.0,
                "y_enabled": True,
                "y_min": -5.0,
                "y_max": 5.0,
            }
        },
    }
    if modes is not None:
        config["golden_sample_display_modes"] = list(modes)
    return config


@pytest.mark.parametrize(
    ("widget_class", "title"),
    [
        (signal_analysis_window.Distortion, "HD"),
        (signal_analysis_window.RubAndBuzz, "RB"),
        (signal_analysis_window.PerceptualRubAndBuzz, "PRB"),
    ],
)
def test_visible_distortion_family_first_dual_render_restores_log_x_range(
    qapp,
    widget_class,
    title,
):
    widget = widget_class(title)
    try:
        single_config = _fixed_xy_plot_config(golden_checked=False)
        widget.analysis_config = single_config
        widget.plot_graph([100.0, 1000.0], [0.0, 1.0], single_config)
        widget.show()
        QApplication.processEvents()

        dual_config = _fixed_xy_plot_config(
            golden_checked=True,
            modes=(
                GOLDEN_SAMPLE_DISPLAY_DEVIATION,
                GOLDEN_SAMPLE_DISPLAY_ENVELOPE,
            ),
        )
        widget.analysis_config = dual_config
        widget.plot_graph(
            [100.0, 1000.0],
            [-1.0, 1.0],
            dual_config,
            raw_y=[9.0, 11.0],
            baseline_aligned=[10.0, 10.0],
        )
        QApplication.processEvents()

        for plot in widget.iter_analysis_plots():
            assert plot.getAxis("bottom").logMode is True
            assert plot.getViewBox().viewRange()[0] == pytest.approx(
                [2.0, 3.0],
                abs=1e-6,
            )
            assert bool(plot.getViewBox().autoRangeEnabled()[1])
    finally:
        _close(widget)


def test_visible_response_first_dual_render_restores_log_x_range(qapp):
    widget = signal_analysis_window.SplFrequency("SPLF")
    try:
        single_config = _fixed_xy_plot_config(golden_checked=False)
        widget.analysis_config = single_config
        widget.plot_spl_frequency([100.0, 1000.0], [9.0, 11.0])
        widget.show()
        QApplication.processEvents()

        dual_config = _fixed_xy_plot_config(
            golden_checked=True,
            modes=(
                GOLDEN_SAMPLE_DISPLAY_DEVIATION,
                GOLDEN_SAMPLE_DISPLAY_ENVELOPE,
            ),
        )
        widget.analysis_config = dual_config
        selected_modes = signal_analysis_window._prepare_golden_response_plots(
            widget,
            dual_config,
        )
        signal_analysis_window._plot_golden_response_views(
            widget,
            [100.0, 1000.0],
            [-1.0, 1.0],
            [9.0, 11.0],
            [10.0, 10.0],
            dual_config,
            selected_modes,
            "SPL (dB)",
        )
        QApplication.processEvents()

        for plot in widget.iter_analysis_plots():
            assert plot.getAxis("bottom").logMode is True
            assert plot.getViewBox().viewRange()[0] == pytest.approx(
                [2.0, 3.0],
                abs=1e-6,
            )
            assert bool(plot.getViewBox().autoRangeEnabled()[1])
    finally:
        _close(widget)


def test_reconfiguration_removes_stale_secondary_and_restores_primary(qapp):
    widget = GoldenPlotWidget()
    try:
        primary = widget.analysis_plot
        dual = widget.configure_golden_sample_plots(
            (GOLDEN_SAMPLE_DISPLAY_DEVIATION, GOLDEN_SAMPLE_DISPLAY_ENVELOPE)
        )
        secondary = dual[GOLDEN_SAMPLE_DISPLAY_DEVIATION]

        single = widget.configure_golden_sample_plots((GOLDEN_SAMPLE_DISPLAY_ENVELOPE,))

        assert single == {GOLDEN_SAMPLE_DISPLAY_ENVELOPE: primary}
        assert widget.analysis_plot is primary
        assert widget.plot_splitter.count() == 1
        assert widget.plot_splitter.widget(0) is primary
        assert widget.plot_splitter.indexOf(secondary) == -1
        assert secondary.parent() is widget
        assert not secondary.isVisible()
        assert secondary.getViewBox().linkedView(ViewBox.XAxis) is None
    finally:
        _close(widget)


def test_non_golden_widget_retains_single_plot_behavior(qapp):
    widget = AnalysisGraphWidget()
    try:
        assert widget.golden_plot_widgets == {}
        assert widget.plot_splitter.count() == 1
        assert widget.plot_splitter.widget(0) is widget.analysis_plot
        assert tuple(widget.iter_analysis_plots()) == (widget.analysis_plot,)
    finally:
        _close(widget)


def test_distortion_dual_envelope_plot_includes_measured_and_golden_curves(qapp):
    widget = signal_analysis_window.Distortion("HD")
    try:
        widget.plot_graph(
            [100.0, 200.0],
            [-1.0, 3.0],
            {
                "golden_sample_checked": True,
                "golden_sample_display_modes": ["deviation", "envelope"],
                "limit_checked": False,
            },
            raw_y=[9.0, 13.0],
            baseline_aligned=[10.0, 10.0],
        )

        envelope = widget.plot_for_golden_mode(GOLDEN_SAMPLE_DISPLAY_ENVELOPE)
        plotted_y = [
            item.getData()[1].tolist()
            for item in envelope.listDataItems()
        ]
        assert [9.0, 13.0] in plotted_y
        assert [10.0, 10.0] in plotted_y
    finally:
        _close(widget)


def test_distortion_switch_from_dual_to_non_golden_restores_clean_primary_plot(qapp):
    widget = signal_analysis_window.Distortion("HD")
    try:
        widget.plot_graph(
            [100.0, 200.0],
            [-1.0, 3.0],
            {
                "golden_sample_checked": True,
                "golden_sample_display_modes": ["deviation", "envelope"],
                "limit_checked": False,
            },
            raw_y=[9.0, 13.0],
            baseline_aligned=[10.0, 10.0],
        )
        secondary = widget.plot_for_golden_mode(GOLDEN_SAMPLE_DISPLAY_DEVIATION)
        assert secondary.listDataItems()

        widget.plot_graph(
            [100.0, 200.0],
            [4.0, 5.0],
            {
                "golden_sample_checked": False,
                "limit_checked": False,
            },
        )

        assert widget.golden_plot_widgets == {}
        assert widget.analysis_plot is widget._primary_analysis_plot
        assert widget.plot_splitter.count() == 1
        assert widget.plot_splitter.widget(0) is widget._primary_analysis_plot
        assert widget.plot_splitter.indexOf(secondary) == -1
        assert not secondary.isVisible()
        assert secondary.getViewBox().linkedView(ViewBox.XAxis) is None
        assert secondary.listDataItems() == []
        assert widget._last_golden_curve_exports is None
    finally:
        _close(widget)


@pytest.mark.parametrize(
    ("widget_class", "title", "result_key", "analyzer_name"),
    [
        (signal_analysis_window.SplFrequency, "SPLF", "spl_db", "SplFrequencyAnalyzer"),
        (signal_analysis_window.Frequency, "FR", "fr", "FrequencyResponseAnalyzer"),
    ],
)
def test_response_dual_envelope_includes_measured_and_golden_curves(
    qapp,
    monkeypatch,
    tmp_path,
    widget_class,
    title,
    result_key,
    analyzer_name,
):
    golden_path = tmp_path / f"{title.lower()}-golden.json"
    golden_path.write_text(
        json.dumps(
            {
                "items": {
                    title: {
                        "result": {
                            "frequency_list": [100.0, 200.0],
                            result_key: [10.0, 10.0],
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    config = {
        "analysis_channel": 0,
        "splf_calc_mode": "fundamental",
        "stimulus_method": "steps",
        "stimulus_type": "linear",
        "start_freq": 100,
        "stop_freq": 200,
        "num_steps": 2,
        "total_time": 0.2,
        "repeat_times": 1,
        "golden_sample_checked": True,
        "golden_sample_display_modes": ["deviation", "envelope"],
        "golden_sample_result_path": str(golden_path),
        "limit_checked": False,
    }
    widget = widget_class(title)
    try:
        widget.analysis_config = config
        widget.data_struct.sample_rate = 48000
        widget.data_struct.store_wave_data = np.ones(3, dtype=np.float32)
        widget.data_struct.stimulus_info = dict(config)
        if title == "SPLF":
            monkeypatch.setattr(widget, "_resolve_v2pa_factor_for_analysis", lambda: True)

            class Analyzer:
                def __init__(self, sample_rate):
                    pass

                def compute(self, *args, **kwargs):
                    return types.SimpleNamespace(
                        frequencies_hz=np.array([100.0, 200.0]),
                        spl_db=np.array([9.0, 13.0]),
                    )

            calculate = widget.calculate_spl
        else:
            widget.data_struct.stimulus_data = np.ones(3, dtype=np.float32)

            class Analyzer:
                def __init__(self, sample_rate):
                    pass

                def compute(self, *args, **kwargs):
                    return types.SimpleNamespace(
                        frequencies_hz=np.array([100.0, 200.0]),
                        magnitude_db=np.array([9.0, 13.0]),
                    )

            calculate = widget.calculate_fr
        monkeypatch.setattr(signal_analysis_window, analyzer_name, Analyzer)

        calculate()

        envelope = widget.plot_for_golden_mode(GOLDEN_SAMPLE_DISPLAY_ENVELOPE)
        deviation = widget.plot_for_golden_mode(GOLDEN_SAMPLE_DISPLAY_DEVIATION)
        envelope_y = [item.getData()[1].tolist() for item in envelope.listDataItems()]
        deviation_y = [item.getData()[1].tolist() for item in deviation.listDataItems()]
        assert [9.0, 13.0] in envelope_y
        assert [10.0, 10.0] in envelope_y
        assert [-1.0, 3.0] in deviation_y
        assert envelope.getPlotItem().getAxis("bottom").logMode is True
        assert deviation.getPlotItem().getAxis("bottom").logMode is True
    finally:
        _close(widget)

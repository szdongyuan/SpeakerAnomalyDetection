import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from pyqtgraph import ViewBox

from consts.acoustic_analysis.common_consts import (
    GOLDEN_SAMPLE_CHECKED_KEY,
    GOLDEN_SAMPLE_DISPLAY_DEVIATION,
    GOLDEN_SAMPLE_DISPLAY_ENVELOPE,
)
from ui.signal_analysis_window import (
    AnalysisGraphWidget,
    Distortion,
    PerceptualRubAndBuzz,
    RubAndBuzz,
)


DEVIATION_TITLE = "偏差曲线（测试 − 黄金）"
ENVELOPE_TITLE = "测试曲线 + 黄金样本上下框线"


class GoldenPlotWidget(AnalysisGraphWidget):
    pass


class SelectedLabel:
    def __init__(self, value):
        self.value = value

    def text(self):
        return self.value


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _show(widget, qapp):
    widget.resize(640, 480)
    widget.show()
    qapp.processEvents()


def _close(widget, qapp):
    widget.close()
    qapp.processEvents()


def _assert_title(plot_widget, expected):
    title_label = plot_widget.plotItem.titleLabel
    assert title_label.text == expected
    assert title_label.isVisible()


@pytest.mark.parametrize(
    ("mode", "expected_title"),
    [
        (GOLDEN_SAMPLE_DISPLAY_DEVIATION, DEVIATION_TITLE),
        (GOLDEN_SAMPLE_DISPLAY_ENVELOPE, ENVELOPE_TITLE),
    ],
)
def test_single_golden_plot_title_matches_selected_config(
    qapp,
    mode,
    expected_title,
):
    widget = GoldenPlotWidget()
    try:
        _show(widget, qapp)

        configured = widget.configure_golden_sample_plots((mode,))

        assert configured == {mode: widget.analysis_plot}
        assert widget.plot_splitter.count() == 1
        _assert_title(widget.analysis_plot, expected_title)
    finally:
        _close(widget, qapp)


def test_dual_golden_plot_titles_match_existing_layout_contract(qapp):
    widget = GoldenPlotWidget()
    try:
        _show(widget, qapp)

        configured = widget.configure_golden_sample_plots(
            (
                GOLDEN_SAMPLE_DISPLAY_DEVIATION,
                GOLDEN_SAMPLE_DISPLAY_ENVELOPE,
            )
        )
        envelope_plot = configured[GOLDEN_SAMPLE_DISPLAY_ENVELOPE]
        deviation_plot = configured[GOLDEN_SAMPLE_DISPLAY_DEVIATION]
        qapp.processEvents()

        assert widget.plot_splitter.orientation() == Qt.Vertical
        assert widget.plot_splitter.count() == 2
        assert widget.plot_splitter.widget(0) is envelope_plot
        assert widget.plot_splitter.widget(1) is deviation_plot
        first_size, second_size = widget.plot_splitter.sizes()
        assert first_size == second_size
        assert first_size > 0
        assert (
            deviation_plot.getViewBox().linkedView(ViewBox.XAxis)
            is envelope_plot.getViewBox()
        )
        assert deviation_plot.getViewBox().linkedView(ViewBox.YAxis) is None
        _assert_title(envelope_plot, ENVELOPE_TITLE)
        _assert_title(deviation_plot, DEVIATION_TITLE)
    finally:
        _close(widget, qapp)


def test_switching_dual_to_single_modes_refreshes_primary_title(qapp):
    widget = GoldenPlotWidget()
    try:
        _show(widget, qapp)
        dual = widget.configure_golden_sample_plots(
            (
                GOLDEN_SAMPLE_DISPLAY_DEVIATION,
                GOLDEN_SAMPLE_DISPLAY_ENVELOPE,
            )
        )
        secondary = dual[GOLDEN_SAMPLE_DISPLAY_DEVIATION]

        deviation_only = widget.configure_golden_sample_plots(
            (GOLDEN_SAMPLE_DISPLAY_DEVIATION,)
        )
        qapp.processEvents()

        assert deviation_only == {
            GOLDEN_SAMPLE_DISPLAY_DEVIATION: widget._primary_analysis_plot
        }
        _assert_title(widget._primary_analysis_plot, DEVIATION_TITLE)
        assert widget.plot_splitter.indexOf(secondary) == -1
        assert not secondary.isVisible()
        assert not secondary.plotItem.titleLabel.isVisible()

        envelope_only = widget.configure_golden_sample_plots(
            (GOLDEN_SAMPLE_DISPLAY_ENVELOPE,)
        )
        qapp.processEvents()

        assert envelope_only == {
            GOLDEN_SAMPLE_DISPLAY_ENVELOPE: widget._primary_analysis_plot
        }
        _assert_title(widget._primary_analysis_plot, ENVELOPE_TITLE)
        assert widget.plot_splitter.indexOf(secondary) == -1
        assert not secondary.isVisible()
        assert not secondary.plotItem.titleLabel.isVisible()
    finally:
        _close(widget, qapp)


def test_reset_golden_sample_plots_hides_all_golden_titles(qapp):
    widget = GoldenPlotWidget()
    try:
        _show(widget, qapp)
        configured = widget.configure_golden_sample_plots(
            (
                GOLDEN_SAMPLE_DISPLAY_DEVIATION,
                GOLDEN_SAMPLE_DISPLAY_ENVELOPE,
            )
        )
        secondary = configured[GOLDEN_SAMPLE_DISPLAY_DEVIATION]

        widget.reset_golden_sample_plots()
        qapp.processEvents()

        assert widget.golden_plot_widgets == {}
        assert widget.analysis_plot is widget._primary_analysis_plot
        assert widget.plot_splitter.count() == 1
        assert not widget._primary_analysis_plot.plotItem.titleLabel.isVisible()
        assert not secondary.plotItem.titleLabel.isVisible()
    finally:
        _close(widget, qapp)


def test_reactivating_secondary_reuses_plot_and_refreshes_title(qapp):
    widget = GoldenPlotWidget()
    try:
        _show(widget, qapp)
        first_dual = widget.configure_golden_sample_plots(
            (
                GOLDEN_SAMPLE_DISPLAY_DEVIATION,
                GOLDEN_SAMPLE_DISPLAY_ENVELOPE,
            )
        )
        secondary = first_dual[GOLDEN_SAMPLE_DISPLAY_DEVIATION]

        widget.configure_golden_sample_plots((GOLDEN_SAMPLE_DISPLAY_ENVELOPE,))
        assert not secondary.plotItem.titleLabel.isVisible()

        second_dual = widget.configure_golden_sample_plots(
            (
                GOLDEN_SAMPLE_DISPLAY_ENVELOPE,
                GOLDEN_SAMPLE_DISPLAY_DEVIATION,
            )
        )
        qapp.processEvents()

        assert second_dual[GOLDEN_SAMPLE_DISPLAY_DEVIATION] is secondary
        assert widget.plot_splitter.widget(1) is secondary
        _assert_title(secondary, DEVIATION_TITLE)
    finally:
        _close(widget, qapp)


@pytest.mark.parametrize(
    ("widget_type", "title_name"),
    [
        (Distortion, "HD"),
        (RubAndBuzz, "RB"),
        (PerceptualRubAndBuzz, "PRB"),
    ],
)
def test_golden_distortion_render_keeps_mode_title_authoritative(
    qapp,
    widget_type,
    title_name,
):
    widget = widget_type(title_name)
    widget.selected_label = SelectedLabel("2nd")
    try:
        _show(widget, qapp)

        widget.plot_graph(
            np.array([100.0, 200.0]),
            np.array([1.0, 2.0]),
            {
                GOLDEN_SAMPLE_CHECKED_KEY: True,
                "golden_sample_display_modes": [
                    GOLDEN_SAMPLE_DISPLAY_DEVIATION
                ],
                "limit_checked": False,
            },
        )

        _assert_title(widget.analysis_plot, DEVIATION_TITLE)
    finally:
        _close(widget, qapp)


@pytest.mark.parametrize(
    ("widget_type", "title_name", "expected_title"),
    [
        (Distortion, "HD", "The Distortion of 2nd order"),
        (RubAndBuzz, "RB", "The Distortion of 2nd order"),
        (
            PerceptualRubAndBuzz,
            "PRB",
            "Perceived Loudness of 2nd order",
        ),
    ],
)
def test_non_golden_distortion_render_retains_existing_order_title(
    qapp,
    widget_type,
    title_name,
    expected_title,
):
    widget = widget_type(title_name)
    widget.selected_label = SelectedLabel("2nd")
    try:
        _show(widget, qapp)

        widget.plot_graph(
            np.array([100.0, 200.0]),
            np.array([1.0, 2.0]),
            {"limit_checked": False},
        )

        _assert_title(widget.analysis_plot, expected_title)
    finally:
        _close(widget, qapp)

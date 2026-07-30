import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from consts.acoustic_analysis.curve_style_consts import DEFAULT_CURVE_COLORS
from ui.curve_style import (
    build_curve_color_config,
    normalize_curve_color,
    resolve_curve_colors,
)
from ui.plot_view import (
    apply_plot_view_range,
    build_plot_view_config,
    normalize_plot_view_config,
    resolve_plot_view_config,
)


class _FakeAxis:
    def __init__(self, log_mode=False):
        self.logMode = log_mode


class _FakeViewBox:
    def __init__(self):
        self.x_range = None
        self.y_range = None

    def setXRange(self, lower, upper, padding):
        self.x_range = (lower, upper, padding)

    def setYRange(self, lower, upper, padding):
        self.y_range = (lower, upper, padding)


class _FakePlotWidget:
    def __init__(self, *, x_log=False, y_log=False):
        self.axes = {
            "bottom": _FakeAxis(x_log),
            "left": _FakeAxis(y_log),
        }
        self.view_box = _FakeViewBox()

    def getAxis(self, name):
        return self.axes[name]

    def getViewBox(self):
        return self.view_box


def test_normalize_plot_view_disables_incomplete_or_non_finite_ranges():
    normalized = normalize_plot_view_config(
        {
            "x_enabled": True,
            "x_min": 20,
            "x_max": "20000",
            "y_enabled": True,
            "y_min": float("nan"),
            "y_max": 100,
        }
    )

    assert normalized["x_enabled"] is True
    assert normalized["x_min"] == 20.0
    assert normalized["x_max"] == 20000.0
    assert normalized["y_enabled"] is False
    assert normalized["y_min"] is None
    assert normalized["y_max"] is None


def test_build_plot_view_preserves_other_display_fields_and_future_keys():
    result = build_plot_view_config(
        {
            "display": {
                "main_curve_color": "#123456",
                "plot_view": {"future_key": "keep", "y_enabled": False},
            }
        },
        {"x_enabled": True, "x_min": 10, "x_max": 100},
    )

    assert result["display"]["main_curve_color"] == "#123456"
    assert result["display"]["plot_view"]["future_key"] == "keep"
    assert result["display"]["plot_view"]["x_enabled"] is True
    assert result["display"]["plot_view"]["x_min"] == 10.0
    assert result["display"]["plot_view"]["x_max"] == 100.0


def test_apply_plot_view_uses_physical_values_and_converts_log_axes():
    plot_widget = _FakePlotWidget(x_log=True)
    config = {
        "display": {
            "plot_view": {
                "x_enabled": True,
                "x_min": 10,
                "x_max": 1000,
                "y_enabled": True,
                "y_min": -20,
                "y_max": 80,
            }
        }
    }

    assert apply_plot_view_range(plot_widget, config) is True
    assert plot_widget.view_box.x_range == (1.0, 3.0, 0.0)
    assert plot_widget.view_box.y_range == (-20.0, 80.0, 0.0)


def test_apply_plot_view_rejects_reversed_range_without_touching_view_box():
    plot_widget = _FakePlotWidget()
    config = {
        "display": {
            "plot_view": {
                "x_enabled": True,
                "x_min": 100,
                "x_max": 10,
            }
        }
    }

    assert apply_plot_view_range(plot_widget, config) is False
    assert plot_widget.view_box.x_range is None
    assert plot_widget.view_box.y_range is None


def test_resolve_plot_view_requires_nested_display_configuration():
    assert resolve_plot_view_config(None) is None
    assert resolve_plot_view_config({}) is None
    assert resolve_plot_view_config({"display": {"plot_view": "invalid"}}) is None


def test_curve_color_normalization_and_resolution_use_stable_defaults():
    assert normalize_curve_color(" #a1b2c3 ", "#000000") == "#A1B2C3"
    assert normalize_curve_color("red", "#000000") == "#000000"

    resolved = resolve_curve_colors(
        {
            "display": {
                "main_curve_color": "#112233",
                "upper_limit_color": "invalid",
            }
        }
    )

    assert resolved["main_curve_color"] == "#112233"
    assert resolved["upper_limit_color"] == DEFAULT_CURVE_COLORS["upper_limit_color"]
    assert resolved["lower_limit_color"] == DEFAULT_CURVE_COLORS["lower_limit_color"]


def test_build_curve_colors_preserves_unrelated_display_configuration():
    result = build_curve_color_config(
        {"display": {"plot_view": {"x_enabled": False}, "future_key": 7}},
        {
            "main_curve_color": "#abcdef",
            "upper_limit_color": "#010203",
            "lower_limit_color": "#040506",
        },
    )

    assert result["display"]["plot_view"] == {"x_enabled": False}
    assert result["display"]["future_key"] == 7
    assert result["display"]["main_curve_color"] == "#ABCDEF"
    assert result["display"]["upper_limit_color"] == "#010203"
    assert result["display"]["lower_limit_color"] == "#040506"

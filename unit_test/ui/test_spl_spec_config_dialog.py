import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from PyQt5.QtWidgets import QApplication
import pyqtgraph as pg

from ui.graph_widget import LimitPlotUtils
from ui.ui_analysis_config.spec_config_dialog import SpecConfigWindow
from ui.ui_analysis_config.spl_config_dialog import SplConfigWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _ConfigManager:
    def __init__(self, config):
        self.config = config

    def load_config(self):
        return self.config

    def save_default_config(self, model_type, config):
        self.config[model_type] = config
        return True


def test_spl_uses_semantic_sections_and_shared_display_config(qapp):
    manager = _ConfigManager(
        {
            "SPL": {
                "analysis_channel": 1,
                "weighting": "A",
                "smooth_checked": True,
                "limit_checked": False,
                "limit_data": None,
            }
        }
    )
    dialog = SplConfigWindow(
        manager,
        "SPL",
        available_channels=[0, 1],
    )

    assert dialog.semantic_group_keys() == [
        "input",
        "compute",
        "display",
        "judgment",
    ]
    assert dialog.channel_selector.current_channel() == 1
    assert dialog.threshold_widget.allow_manual_limits is False
    assert dialog.curve_color_widget is not None

    plot_view = dialog.plot_view_config_widget
    plot_view.x_enabled_checkbox.setChecked(True)
    plot_view.x_min_spinbox.setValue(0.1)
    plot_view.x_max_spinbox.setValue(1.5)
    config = dialog.get_default_config()

    assert config["analysis_channel"] == 1
    assert config["weighting"] == "A"
    assert config["smooth_checked"] is True
    assert config["limit_checked"] is False
    assert "limit_mode" not in config
    assert config["display"]["main_curve_color"].startswith("#")
    assert config["display"]["plot_view"]["x_enabled"] is True
    assert config["display"]["plot_view"]["x_min"] == 0.1
    assert config["display"]["plot_view"]["x_max"] == 1.5
    dialog.close()


def test_splf_keeps_existing_analysis_fields_in_semantic_layout(qapp):
    manager = _ConfigManager(
        {
            "SPLF": {
                "analysis_channel": 0,
                "splf_calc_mode": "total",
                "octave_smoothing": 3,
                "golden_sample_checked": True,
                "limit_checked": False,
                "limit_data": None,
            }
        }
    )
    dialog = SplConfigWindow(
        manager,
        "SPLF",
        available_channels=[0],
    )

    assert dialog.semantic_group_keys() == [
        "input",
        "compute",
        "reference",
        "display",
        "judgment",
    ]
    config = dialog.get_default_config()
    assert config["splf_calc_mode"] == "total"
    assert config["octave_smoothing"] == 3
    assert config["golden_sample_checked"] is True
    dialog.close()


def test_spec_uses_semantic_sections_without_changing_analysis_contract(
    qapp,
):
    manager = _ConfigManager(
        {
            "Spec": {
                "analysis_channel": 1,
                "freq_scale_type": "log",
                "n_fft": 4096,
                "hop_length": 512,
                "window_func": "blackman",
                "color_map": "magma",
                "custom_limit": True,
                "top_limit": 80,
                "bottom_limit": 40,
            }
        }
    )
    dialog = SpecConfigWindow(
        manager,
        "Spec",
        available_channels=[0, 1],
    )

    assert dialog.semantic_group_keys() == [
        "input",
        "compute",
        "display",
    ]
    config = dialog.get_default_config()
    assert config == {
        "n_fft": 4096,
        "hop_length": 512,
        "window_func": "blackman",
        "color_map": "magma",
        "freq_scale_type": "log",
        "top_limit": 80,
        "bottom_limit": 40,
        "custom_limit": True,
        "analysis_channel": 1,
    }
    dialog.close()


def test_limit_plot_uses_configured_main_and_limit_colors(qapp):
    plot_widget = pg.PlotWidget()
    LimitPlotUtils.setup_limit_plot(
        plot_widget,
        [1, 2],
        [3, 4],
        [1, 2],
        [5, 5],
        [1, 1],
        curve_colors={
            "main_curve_color": "#112233",
            "upper_limit_color": "#445566",
            "lower_limit_color": "#778899",
        },
    )

    colors = [
        item.opts["pen"].color().name().upper()
        for item in plot_widget.listDataItems()
    ]
    assert colors == ["#112233", "#445566", "#778899"]
    plot_widget.close()

import os
import sys
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication

from consts import error_code


class FakeConfigManager:
    def __init__(self, config):
        self.config = config
        self.saved = []

    def load_config(self):
        return self.config

    def save_default_config(self, model_type, config_data):
        self.saved.append((model_type, config_data))
        return True


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_ai_dialog_uses_shared_channel_selector_without_changing_config(qapp, monkeypatch):
    class FakeTrainingModelManagement:
        def get_all_model_name_from_db(self):
            return error_code.OK, [("model_a", "1024 samples")]

    monkeypatch.setitem(
        sys.modules,
        "base.training_model_management",
        types.SimpleNamespace(TrainingModelManagement=FakeTrainingModelManagement),
    )
    from ui.ui_analysis_config import ai_config_dialog

    monkeypatch.setattr(ai_config_dialog, "TrainingModelManagement", FakeTrainingModelManagement)
    config_manager = FakeConfigManager({"AI": {"analyse_model_name": "model_a", "analysis_channel": 2}})

    window = ai_config_dialog.AIConfigWindow(config_manager, "AI", available_channels=[0, 2])

    assert window.get_default_config() == {
        "analyse_model_name": "model_a",
        "analysis_channel": 2,
    }


def test_lp_dialog_preserves_saved_keys_after_channel_migration(qapp):
    from ui.ui_analysis_config.lp_config_dialog import LPConfigWindow

    config_manager = FakeConfigManager(
        {
            "LP": {
                "trigger_threshold": 12,
                "hysterests_threshold": 3,
                "min_check_duration": 20,
                "max_check_duration": 80,
                "loose_particle_num": 2,
                "cutoff_freq": 15000,
                "analysis_channel": 1,
            }
        }
    )

    window = LPConfigWindow(config_manager, "LP", available_channels=[0, 1])

    assert window.get_default_config() == {
        "trigger_threshold": 12,
        "hysterests_threshold": 3,
        "min_check_duration": 20,
        "max_check_duration": 80,
        "loose_particle_num": 2,
        "cutoff_freq": 15000,
        "analysis_channel": 1,
    }


def test_spec_dialog_preserves_saved_keys_after_channel_migration(qapp):
    from ui.ui_analysis_config.spec_config_dialog import SpecConfigWindow

    config_manager = FakeConfigManager(
        {
            "Spec": {
                "n_fft": 1024,
                "hop_length": 128,
                "window_func": "hamming",
                "color_map": "magma",
                "freq_scale_type": "log",
                "top_limit": 75,
                "bottom_limit": 35,
                "custom_limit": True,
                "analysis_channel": 3,
            }
        }
    )

    window = SpecConfigWindow(config_manager, "Spec", available_channels=[1, 3])

    assert window.get_default_config() == {
        "n_fft": 1024,
        "hop_length": 128,
        "window_func": "hamming",
        "color_map": "magma",
        "freq_scale_type": "log",
        "top_limit": 75,
        "bottom_limit": 35,
        "custom_limit": True,
        "analysis_channel": 3,
    }

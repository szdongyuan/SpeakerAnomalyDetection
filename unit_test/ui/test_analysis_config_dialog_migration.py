import os
import sys
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import QApplication, QTableView

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


def stub_pattern_match_import_dependencies(monkeypatch):
    class FakeLoadUiConfig:
        @staticmethod
        def load_data_from_json(_file_path):
            return 0, {}

    class FakeFileOps:
        @staticmethod
        def get_relative_path(file_path, _base_path):
            return file_path

    class FakeDataView(QTableView):
        def __init__(self, rows, columns, _data):
            super().__init__()
            self._model = QStandardItemModel(rows, columns)
            self.setModel(self._model)

        def set_h_header(self, labels):
            self._model.setHorizontalHeaderLabels(labels)

    class FakeAudioClipExtractionDialog:
        def __init__(self, *args, **kwargs):
            pass

        def on_exec(self):
            return None, None, None

    class FakeGenericFeatureParamsDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec_(self):
            return 0

        def get_params(self):
            return {}

    monkeypatch.setitem(sys.modules, "base.load_config", types.SimpleNamespace(LoadUiConfig=FakeLoadUiConfig))
    monkeypatch.setitem(sys.modules, "base.file_ops", types.SimpleNamespace(FileOps=FakeFileOps))
    monkeypatch.setitem(sys.modules, "base.load_audio", types.SimpleNamespace(load_audio_simple=lambda *_: ([], None)))
    monkeypatch.setitem(
        sys.modules,
        "ui.custom_ui_widget.custom_table_widget",
        types.SimpleNamespace(DataView=FakeDataView),
    )
    monkeypatch.setitem(
        sys.modules,
        "ui.custom_ui_widget.audio_clip_extraction_dialog",
        types.SimpleNamespace(AudioClipExtractionDialog=FakeAudioClipExtractionDialog),
    )
    monkeypatch.setitem(
        sys.modules,
        "ui.generic_feature_params_dialog",
        types.SimpleNamespace(GenericFeatureParamsDialog=FakeGenericFeatureParamsDialog),
    )


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def assert_vertical_golden_size(window):
    assert window.minimumWidth() == 630
    assert window.minimumHeight() == 840
    assert round(window.minimumWidth() / window.minimumHeight(), 2) == 0.75


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

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["input", "compute"]
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

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["input", "detection"]
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

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["input", "compute", "display"]
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


def test_spl_dialog_preserves_saved_keys_after_shared_control_migration(qapp):
    from ui.ui_analysis_config.spl_config_dialog import SplConfigWindow

    config_manager = FakeConfigManager(
        {
            "SPL": {
                "smooth_checked": True,
                "limit_checked": False,
                "limit_data": None,
                "weighting": "C",
                "analysis_channel": 2,
            }
        }
    )

    window = SplConfigWindow(config_manager, "SPL", available_channels=[0, 2])

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["input", "compute", "judgment"]
    assert window.get_default_config() == {
        "smooth_checked": True,
        "limit_checked": False,
        "limit_data": None,
        "weighting": "C",
        "analysis_channel": 2,
    }


def test_splf_dialog_preserves_saved_keys_after_shared_control_migration(qapp):
    from ui.ui_analysis_config.spl_config_dialog import SplConfigWindow

    config_manager = FakeConfigManager(
        {
            "SPLF": {
                "splf_calc_mode": "total",
                "octave_smoothing": 3,
                "golden_sample_checked": True,
                "limit_checked": False,
                "limit_data": None,
                "weighting": "Z（None）",
                "analysis_channel": 1,
            }
        }
    )

    window = SplConfigWindow(config_manager, "SPLF", available_channels=[0, 1])

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["input", "compute", "reference", "judgment"]
    assert window.get_default_config() == {
        "splf_calc_mode": "total",
        "octave_smoothing": 3,
        "golden_sample_checked": True,
        "limit_checked": False,
        "limit_data": None,
        "weighting": "Z",
        "analysis_channel": 1,
    }


def test_fr_dialog_uses_shared_octave_smoothing_legacy_fallback(qapp):
    from ui.ui_analysis_config.fr_config_dialog import FrConfigWindow

    config_manager = FakeConfigManager(
        {
            "FR": {
                "smooth_checked": True,
                "golden_sample_checked": True,
                "limit_checked": False,
                "limit_data": None,
            }
        }
    )

    window = FrConfigWindow(config_manager, "FR")

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["compute", "reference", "judgment"]
    assert window.get_default_config() == {
        "octave_smoothing": 6,
        "golden_sample_checked": True,
        "limit_checked": False,
        "limit_data": None,
    }


def test_spl_dialog_restore_default_reloads_semantic_sections(qapp):
    from ui.ui_analysis_config.spl_config_dialog import SplConfigWindow

    config_manager = FakeConfigManager(
        {
            "SPL": {
                "smooth_checked": True,
                "limit_checked": False,
                "limit_data": None,
                "weighting": "A",
                "analysis_channel": 0,
            }
        }
    )
    window = SplConfigWindow(config_manager, "SPL", available_channels=[0, 1])

    config_manager.config = {
        "SPL": {
            "smooth_checked": False,
            "limit_checked": False,
            "limit_data": None,
            "weighting": "C",
            "analysis_channel": 1,
        }
    }
    window.on_restore_default_btn_clicked()

    assert window.semantic_group_keys() == ["input", "compute", "judgment"]
    assert window.get_default_config() == {
        "smooth_checked": False,
        "limit_checked": False,
        "limit_data": None,
        "weighting": "C",
        "analysis_channel": 1,
    }


def test_fba_dialog_preserves_saved_keys_after_shared_control_migration(qapp):
    from ui.ui_analysis_config.fba_config_dialog import FbaConfigWindow

    config_manager = FakeConfigManager(
        {
            "FBA": {
                "band_strategy": "1/3 倍频程",
                "f_min": 50,
                "f_max": 16000,
                "bandwidth": 200,
                "weighting": "Z（None）",
                "analysis_channel": 3,
                "limit_checked": False,
                "limit_data": None,
            }
        }
    )

    window = FbaConfigWindow(config_manager, "FBA", available_channels=[1, 3])

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["input", "compute", "judgment"]
    assert window.get_default_config() == {
        "band_strategy": "1/3 倍频程",
        "f_min": 50,
        "f_max": 16000,
        "bandwidth": 200,
        "analysis_channel": 3,
        "weighting": "Z",
        "limit_checked": False,
        "limit_data": None,
    }


def test_hd_dialog_uses_shared_harmonic_and_golden_widgets(qapp):
    from ui.ui_analysis_config.hd_config_dialog import HdConfigWindow

    config_manager = FakeConfigManager(
        {
            "HD": {
                "selected_labels": [2, "3", 40],
                "all_checked": False,
                "golden_sample_checked": True,
                "limit_checked": False,
                "limit_data": None,
            }
        }
    )

    window = HdConfigWindow(config_manager, "HD")

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["detection", "reference", "judgment"]
    assert window.get_default_config() == {
        "selected_labels": [2, 3],
        "all_checked": False,
        "golden_sample_checked": True,
        "limit_checked": False,
        "limit_data": None,
    }


def test_rb_dialog_filters_harmonics_to_rub_buzz_range(qapp):
    from ui.ui_analysis_config.rb_config_dialog import RbConfigWindow

    config_manager = FakeConfigManager(
        {
            "RB": {
                "selected_labels": [2, 10, "12", 40],
                "all_checked": False,
                "golden_sample_checked": False,
                "limit_checked": False,
                "limit_data": None,
            }
        }
    )

    window = RbConfigWindow(config_manager, "RB")

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["detection", "reference", "judgment"]
    assert window.get_default_config() == {
        "selected_labels": [10, 12],
        "all_checked": False,
        "golden_sample_checked": False,
        "limit_checked": False,
        "limit_data": None,
    }


def test_prb_dialog_preserves_metric_fallback_and_golden_sample(qapp):
    from ui.ui_analysis_config.perceptual_rb_config_dialog import PerceptualRbConfigWindow

    config_manager = FakeConfigManager(
        {
            "PRB": {
                "masking_config": {"sc_metric": "totalnl_phons", "keep": "value"},
                "golden_sample_checked": True,
                "limit_checked": False,
                "limit_data": None,
            }
        }
    )

    window = PerceptualRbConfigWindow(config_manager, "PRB")
    config = window.get_default_config()

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["compute", "reference", "judgment"]
    assert config["prb_method"] == "sc"
    assert config["masking_config"] == {"sc_metric": "totalnl_x_ehs", "keep": "value"}
    assert config["golden_sample_checked"] is True
    assert config["limit_checked"] is False
    assert config["limit_data"] is None


def test_rsc_dialog_keeps_smoothing_key_after_shared_selector_migration(qapp, monkeypatch):
    from ui.ui_analysis_config import reference_spectrum_config_dialog as rsc_dialog

    monkeypatch.setattr(rsc_dialog, "get_reference_data_state", lambda **_: "not_generated")
    config_manager = FakeConfigManager(
        {
            "RSC": {
                "reference_source_path": "",
                "reference_data_path": "",
                "use_custom_band": False,
                "start_freq_hz": 100,
                "end_freq_hz": 1000,
                "highlight_analysis_band": False,
                "enable_threshold_judgment": True,
                "lower_offset_db": -2.0,
                "upper_offset_db": 2.0,
                "channel_labels": {},
                "window": "hann",
                "nperseg": 4096,
                "overlap_ratio": 0.5,
                "smoothing": 3,
            }
        }
    )

    window = rsc_dialog.ReferenceSpectrumConfigWindow(config_manager, "RSC")
    config = window.get_default_config()

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["reference", "compute", "judgment", "display"]
    assert config["smoothing"] == 3
    assert "octave_smoothing" not in config
    assert config["enable_threshold_judgment"] is True
    assert config["lower_offset_db"] == -2.0
    assert config["upper_offset_db"] == 2.0


def test_excel_dialog_preserves_output_config_after_semantic_migration(qapp):
    from ui.ui_analysis_config.excel_config_dialog import ExcelConfigWindow

    config_manager = FakeConfigManager(
        {
            "Excel": {
                "save_dir": "",
                "file_base": "result",
                "add_date": False,
                "add_model_dir": True,
                "lock_files": False,
                "max_points": 500,
                "save_items": ["SPL", "FBA"],
                "save_mes_enabled": True,
                "mes_file_base": "D:/dataMES",
                "mes_file_name": "MES_Result",
            },
            "SPL": {"type": "SPL"},
            "FBA": {"type": "FBA"},
            "Spec": {"type": "Spec"},
        }
    )

    window = ExcelConfigWindow(config_manager, "Excel")

    assert_vertical_golden_size(window)
    assert window.semantic_group_keys() == ["output"]
    assert window.get_default_config() == {
        "enabled": True,
        "save_dir": None,
        "file_base": "result",
        "add_date": False,
        "add_model_dir": True,
        "lock_files": False,
        "date_format": "%Y%m%d",
        "max_points": 500,
        "save_items": ["FBA", "SPL"],
        "save_mes_enabled": True,
        "mes_file_base": "D:/dataMES",
        "mes_file_name": "MES_Result",
    }


def test_pd_dialog_preserves_time_smoothing_keys_after_widget_migration(qapp):
    from ui.ui_analysis_config.pd_config_dialog import PDConfigWindow

    config_manager = FakeConfigManager(
        {
            "PD": {
                "smooth_enabled": True,
                "smooth_unit": "points",
                "smooth_time_sec": 0.125,
                "smooth_points": 0,
                "smooth_algo": 3,
            }
        }
    )

    window = PDConfigWindow(config_manager, "PD")
    config = window.get_default_config()

    assert config["smooth_enabled"] is True
    assert config["smooth_unit"] == "points"
    assert config["smooth_time_sec"] == 0.125
    assert config["smooth_points"] == 0
    assert config["smooth_algo"] == 3


def test_pm_dialog_preserves_fixed_threshold_config_after_button_migration(qapp, monkeypatch):
    stub_pattern_match_import_dependencies(monkeypatch)
    from ui.ui_analysis_config import pattern_match_config_dialog as pm_dialog

    monkeypatch.setattr(
        pm_dialog.PatternMatchConfigWindow,
        "load_features_param_config",
        staticmethod(
            lambda: (
                True,
                {
                    "mfcc": {
                        "display_name": "MFCC",
                        "params": {"n_mfcc": {"default": 13}},
                    }
                },
            )
        ),
    )
    config_manager = FakeConfigManager(
        {
            "PM": {
                "pattern_list": [],
                "sample_rate": 48000,
                "feature_type": "mfcc",
                "feature_params": {"n_mfcc": 20},
                "apply_filter": True,
                "filter_range_hz": [100, 5000],
                "similarity_metric": "cosine",
                "threshold_strategy": "fixed_threshold",
                "threshold_value": 0.75,
            }
        }
    )

    window = pm_dialog.PatternMatchConfigWindow(config_manager, "PM")

    assert window.get_config() == {
        "pattern_list": [],
        "sample_rate": 48000,
        "feature_type": "mfcc",
        "feature_params": {"n_mfcc": 20},
        "apply_filter": True,
        "filter_range_hz": (100, 5000),
        "algorithm": "dtw",
        "similarity_metric": "cosine",
        "threshold_strategy": "fixed_threshold",
        "threshold_value": 0.75,
    }


def test_pipeline_pd_pm_dialog_preserves_nested_config_after_base_migration(qapp, monkeypatch):
    stub_pattern_match_import_dependencies(monkeypatch)
    from ui.ui_analysis_config.pipeline_pd_pm_config import PipelinePdPmConfigWindow

    config_manager = FakeConfigManager(
        {
            "ED": {
                "type": "ED",
                "head": {"type": "PD", "config": {"peak_count": 2}},
                "tail": {"type": "PM", "config": {"threshold_value": 0.8}},
                "auto_equal_length": False,
                "left_grid": 10,
                "right_grid": 20,
                "pass_condition": {"n1": 2, "n2": 5},
            }
        }
    )

    window = PipelinePdPmConfigWindow(config_manager, "ED")

    assert window.get_default_config() == {
        "type": "ED",
        "head": {"type": "PD", "config": {"peak_count": 2}},
        "tail": {"type": "PM", "config": {"threshold_value": 0.8}},
        "auto_equal_length": False,
        "left_grid": 10,
        "right_grid": 20,
        "pass_condition": {"n1": 2, "n2": 5},
    }

import os
import time
import types
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_modulation_algorithm_detects_synthetic_am_tone():
    from base.core_algorithm.modulation_map import compute_modulation_map, default_modulation_config

    sample_rate = 48000
    duration_s = 0.5
    t = np.arange(int(sample_rate * duration_s), dtype=np.float64) / sample_rate
    audio = (1.0 + 0.35 * np.sin(2.0 * np.pi * 75.0 * t)) * np.sin(2.0 * np.pi * 1200.0 * t)

    cfg = default_modulation_config()
    cfg.update(
        {
            "signal_freq_range_hz": [500.0, 4000.0],
            "mod_freq_range_hz": [0.0, 160.0],
            "signal_freq_display_step_hz": 20.0,
            "mod_freq_bin_hz": 2.0,
            "smoothing_points": 1,
        }
    )

    result = compute_modulation_map(audio, sample_rate, cfg)

    matrix = result["mod_depth_matrix"]
    signal_axis = result["signal_freq_axis_hz"]
    mod_axis = result["mod_freq_axis_hz"]
    assert matrix.shape == (len(signal_axis), len(mod_axis))
    assert np.all(np.isfinite(matrix))
    assert [tone["freq_hz"] for tone in result["input_main_tones"]] == [1200.0, 3500.0]

    tone_1200 = next(item for item in result["main_tone_results"] if item["target_signal_freq_hz"] == 1200.0)
    assert tone_1200["mod_depth_percent"] > 10.0
    assert tone_1200["mod_freq_hz"] == pytest.approx(75.0, abs=4.0)


def test_modulation_main_tone_search_width_selects_peak_within_window():
    from base.core_algorithm.modulation_map import _evaluate_main_tones

    mod_depth = np.array(
        [
            [0.0, 2.0],
            [0.0, 5.0],
            [0.0, 20.0],
        ],
        dtype=np.float64,
    )
    signal_freqs = np.array([1160.0, 1200.0, 1240.0], dtype=np.float64)
    mod_freqs = np.array([0.0, 50.0], dtype=np.float64)
    main_tones = [{"id": 1, "label": "tone", "freq_hz": 1200.0}]

    narrow = _evaluate_main_tones(
        mod_depth,
        signal_freqs,
        mod_freqs,
        main_tones,
        threshold_percent=10.0,
        mechanical_refs=[],
        mechanical_match_tolerance_hz=20.0,
        main_tone_search_width_hz=40.0,
        min_modulation_depth_percent=1.0,
    )[0]
    wide = _evaluate_main_tones(
        mod_depth,
        signal_freqs,
        mod_freqs,
        main_tones,
        threshold_percent=10.0,
        mechanical_refs=[],
        mechanical_match_tolerance_hz=20.0,
        main_tone_search_width_hz=100.0,
        min_modulation_depth_percent=1.0,
    )[0]

    assert narrow["analysis_signal_freq_hz"] == 1200.0
    assert narrow["mod_depth_percent"] == 5.0
    assert wide["analysis_signal_freq_hz"] == 1240.0
    assert wide["mod_depth_percent"] == 20.0


def test_modulation_algorithm_does_not_assign_frequency_to_unmodulated_main_tone():
    from base.core_algorithm.modulation_map import compute_modulation_map, default_modulation_config

    sample_rate = 48000
    duration_s = 2.0
    t = np.arange(int(sample_rate * duration_s), dtype=np.float64) / sample_rate
    audio = (
        (1.0 + 0.20 * np.sin(2.0 * np.pi * 75.0 * t)) * np.sin(2.0 * np.pi * 1200.0 * t)
        + 0.6 * np.sin(2.0 * np.pi * 3500.0 * t)
    )

    cfg = default_modulation_config()
    cfg.update(
        {
            "signal_freq_range_hz": [500.0, 4000.0],
            "mod_freq_range_hz": [0.0, 160.0],
            "main_tones_hz": [1200.0, 3500.0],
            "show_global_hotspots": True,
            "signal_freq_display_step_hz": 2.0,
            "mod_freq_bin_hz": 1.0,
            "smoothing_points": 1,
        }
    )

    result = compute_modulation_map(audio, sample_rate, cfg)

    tone_3500 = next(item for item in result["main_tone_results"] if item["target_signal_freq_hz"] == 3500.0)
    assert tone_3500["has_modulation_peak"] is False
    assert tone_3500["mod_freq_hz"] is None
    assert tone_3500["is_valid"] is True
    assert tone_3500["reason"] == "no AM peak"
    assert not [
        hotspot
        for hotspot in result["global_hotspots"]
        if abs(float(hotspot["signal_freq_hz"]) - 3500.0) <= cfg["main_tone_search_width_hz"] / 2.0
    ]


def test_modulation_algorithm_limits_work_to_main_tone_rois():
    from base.core_algorithm.modulation_map import compute_modulation_map, default_modulation_config

    sample_rate = 48000
    duration_s = 2.0
    t = np.arange(int(sample_rate * duration_s), dtype=np.float64) / sample_rate
    audio = (
        (1.0 + 0.08 * np.sin(2.0 * np.pi * 50.0 * t)) * np.sin(2.0 * np.pi * 1200.0 * t)
        + 0.6 * np.sin(2.0 * np.pi * 3500.0 * t)
    )

    base_cfg = default_modulation_config()
    base_cfg.update(
        {
            "signal_freq_range_hz": [500.0, 4000.0],
            "mod_freq_range_hz": [0.0, 200.0],
            "show_global_hotspots": True,
            "signal_freq_display_step_hz": 1.0,
            "mod_freq_bin_hz": 1.0,
            "smoothing_points": 3,
            "main_tone_search_width_hz": 100.0,
            "min_modulation_depth_percent": 1.0,
        }
    )

    full_cfg = dict(base_cfg)
    full_cfg["main_tones_hz"] = None
    full_start = time.perf_counter()
    full_result = compute_modulation_map(audio, sample_rate, full_cfg)
    full_elapsed = time.perf_counter() - full_start

    roi_cfg = dict(base_cfg)
    roi_cfg.update({"main_tones_hz": [1200.0, 3500.0], "tone_band_hz": 50.0})
    roi_start = time.perf_counter()
    roi_result = compute_modulation_map(audio, sample_rate, roi_cfg)
    roi_elapsed = time.perf_counter() - roi_start

    assert full_result["stft_params"]["analysis_scope"] == "full"
    assert roi_result["stft_params"]["analysis_scope"] == "main_tone_roi"
    assert roi_result["stft_params"]["computed_signal_freq_count"] < (
        full_result["stft_params"]["computed_signal_freq_count"] / 4
    )
    assert roi_elapsed < full_elapsed

    signal_axis = np.asarray(roi_result["signal_freq_axis_hz"], dtype=float)
    in_1200_roi = (signal_axis >= 1200.0 - 50.0) & (signal_axis <= 1200.0 + 50.0)
    in_3500_roi = (signal_axis >= 3500.0 - 50.0) & (signal_axis <= 3500.0 + 50.0)
    assert np.all(in_1200_roi | in_3500_roi)
    assert np.any(in_1200_roi)
    assert np.any(in_3500_roi)

    tone_3500 = next(item for item in roi_result["main_tone_results"] if item["target_signal_freq_hz"] == 3500.0)
    assert tone_3500["has_modulation_peak"] is False
    assert tone_3500["mod_freq_hz"] is None
    assert not [
        hotspot
        for hotspot in roi_result["global_hotspots"]
        if abs(float(hotspot["signal_freq_hz"]) - 3500.0) <= 50.0
        and float(hotspot["mod_depth_percent"]) >= roi_cfg["threshold_percent"]
    ]


def test_modulation_queue_item_uses_builtin_defaults_when_json_missing(monkeypatch):
    from base.data_struct.sequence_data import SequenceData
    from ui.operation_sequence import LoadUiConfig, OptionList

    monkeypatch.setattr(LoadUiConfig, "load_data_from_json", staticmethod(lambda _path: (1, "missing")))

    option_list = OptionList.__new__(OptionList)
    option_list.config = [SequenceData("seq1")]
    option_list.default_logger = types.SimpleNamespace(error=lambda *_args, **_kwargs: None)

    option_list.get_item_default_config("调制 (Modulation) ", "调制 (Modulation) 1")

    cfg = option_list.config[0].analysis_list["调制 (Modulation) 1"]
    assert cfg["type"] == "Modulation"
    assert cfg["main_tones_hz"] == [1200.0, 3500.0]
    assert cfg["fan_rpm"] == 4500.0
    assert cfg["blade_count"] == 2


def test_sequence_run_dispatches_modulation_without_ok_ng_summary():
    from ui.sequence.sequence_widget import SequenceWindow

    class FakeSize:
        def width(self):
            return 1200

        def height(self):
            return 900

    class FakeScreen:
        def size(self):
            return FakeSize()

    class FakeModulation:
        def __init__(self):
            self.called = False
            self.shown = False
            self._sequence_analysis_key = "调制 (Modulation) 1"

        def calculate_modulation(self):
            self.called = True
            return {"mod_depth_matrix": [[1.0]]}

        def show(self):
            self.shown = True

        def setGeometry(self, *_args):
            return None

        def setMinimumSize(self, *_args):
            return None

        def installEventFilter(self, *_args):
            return None

    fake_instance = FakeModulation()
    data_struct = types.SimpleNamespace(
        analysis_result_dict={},
        stimulus_info={},
        audio_lenth=0,
    )
    window = types.SimpleNamespace(
        data_struct=data_struct,
        sequence_config=[{"seq1": {"acq": {"mode": "RECORD_ONLY"}}}],
        analysis_window=[],
        _analysis_result_summary_window=None,
        analysis_config={
            "display_sequence": ["调制 (Modulation) 1"],
            "调制 (Modulation) 1": {"type": "Modulation"},
        },
        screen=lambda: FakeScreen(),
        instance_analysis_class=lambda _key, _type, _params: window.analysis_window.append(fake_instance),
        _show_channel_mismatch_warning=lambda *_args, **_kwargs: None,
        _get_analysis_window_geometry=lambda _key: None,
        _set_analysis_window_geometry=lambda *_args, **_kwargs: None,
        _analysis_window_key_by_obj={},
        _begin_analysis_export_run=lambda: None,
        _handle_post_analysis_exports=lambda: None,
        count_board=types.SimpleNamespace(mode="analysis"),
        _maybe_show_analysis_result_summary=lambda *_args, **_kwargs: None,
    )

    SequenceWindow.run(window)

    assert fake_instance.called is True
    assert fake_instance.shown is True
    assert data_struct.analysis_result_dict == {}


@pytest.fixture
def qapp():
    from PyQt5.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_modulation_ui_smoke_renders_pyqtgraph_without_matplotlib(qapp, monkeypatch):
    from base.data_struct.data_deal_struct import DataDealStruct
    from ui.signal_analysis_window import Modulation
    from ui.ui_analysis_config.modulation_config_dialog import ModulationConfigWindow

    fake_manager = types.SimpleNamespace(
        load_config=lambda: {},
        save_default_config=lambda *_args, **_kwargs: True,
    )
    dialog = ModulationConfigWindow(fake_manager, "调制 (Modulation) 1", available_channels=[0, 1])
    dialog_config = dialog.get_default_config()
    assert dialog_config["main_tones_hz"] == [1200.0, 3500.0]
    assert dialog_config["tone_band_hz"] == 80.0

    fake_result = {
        "mod_depth_matrix": np.array([[0.0, 12.0, 8.0], [1.0, 20.0, 5.0]], dtype=float),
        "signal_freq_axis_hz": np.array([1000.0, 1200.0], dtype=float),
        "mod_freq_axis_hz": np.array([0.0, 75.0, 150.0], dtype=float),
        "main_tone_results": [
            {
                "target_signal_freq_hz": 1200.0,
                "signal_freq_khz": 1.2,
                "mod_freq_hz": 75.0,
                "mod_depth_percent": 20.0,
                "mechanical_match": True,
                "is_valid": False,
                "reason": "AM depth above threshold",
            }
        ],
        "hotspots": [],
        "global_hotspots": [],
        "mechanical_references": [{"freq_hz": 75.0, "label": "rotation 1x"}],
        "mechanical_mod_freqs_hz": np.array([75.0]),
        "stft_params": {},
        "threshold_percent": 10.0,
        "core_freq_lines_khz": [1.0],
        "main_tone_search_width_hz": 160.0,
        "mechanical_match_tolerance_hz": 20.0,
    }
    monkeypatch.setattr("ui.signal_analysis_window.compute_modulation_map", lambda *_args, **_kwargs: fake_result)

    widget = Modulation("调制 (Modulation) 1")
    data_struct = DataDealStruct()
    old_wave = data_struct.store_wave_data
    old_multi = data_struct.store_wave_data_multi
    old_sample_rate = data_struct.sample_rate
    try:
        data_struct.store_wave_data = np.zeros(512, dtype=np.float32)
        data_struct.store_wave_data_multi = None
        data_struct.sample_rate = 48000
        widget.data_struct = data_struct
        widget.analysis_config = {}

        result = widget.calculate_modulation()

        assert result["mod_depth_matrix"] == fake_result["mod_depth_matrix"].tolist()
        assert widget.table_widget.rowCount() == 1
        assert widget.img_item is not None
        assert widget.pdf_summary_exclude_fields == ("main_tone_results",)
        assert widget.export_pdf_tables() == [
            {
                "title": "分析表格",
                "headers": ["主音(Hz)", "分析频率(kHz)", "调制频率(Hz)", "深度(%)", "机械匹配", "原因"],
                "rows": [["1200.0", "1.200", "75.0", "20.00", "Yes", "AM depth above threshold"]],
            }
        ]
    finally:
        data_struct.store_wave_data = old_wave
        data_struct.store_wave_data_multi = old_multi
        data_struct.sample_rate = old_sample_rate

    algorithm_source = Path(__file__).parents[2] / "base" / "core_algorithm" / "modulation_map.py"
    assert "matplotlib" not in algorithm_source.read_text(encoding="utf-8")

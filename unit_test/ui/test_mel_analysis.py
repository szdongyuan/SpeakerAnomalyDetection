import os
import types
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_mel_algorithm_detects_synthetic_tone():
    from base.core_algorithm.mel_spectrogram import compute_mel_spectrogram, default_mel_config, hz_to_mel

    sample_rate = 48000
    duration_s = 0.25
    t = np.arange(int(sample_rate * duration_s), dtype=np.float64) / sample_rate
    audio = 0.5 * np.sin(2.0 * np.pi * 2000.0 * t)

    cfg = default_mel_config()
    cfg.update(
        {
            "fmin_hz": 100.0,
            "fmax_hz": 20000.0,
            "n_mels": 48,
            "stft_nfft": 2048,
            "main_tones_hz": [],
        }
    )

    result = compute_mel_spectrogram(audio, sample_rate, cfg, v2pa_factor=1.0)

    mel_db_a = result["mel_db_a"]
    assert mel_db_a.shape[0] == 48
    assert mel_db_a.shape[1] == len(result["times_s"])
    assert np.all(np.isfinite(mel_db_a))
    assert np.isfinite(result["overall_spl_dba"])
    assert 1000.0 <= result["hotspot"]["freq_hz"] <= 3500.0
    assert result["hotspot"]["kind"] == "mel_band"
    assert result["hotspot"]["aggregation"] == "mean_over_time"
    assert "time_s" not in result["hotspot"]
    assert result["hotspot"]["mel_low"] <= result["hotspot"]["mel"] <= result["hotspot"]["mel_high"]
    assert result["global_hotspot"] == result["hotspot"]
    assert result["main_tone_hotspots"] == []
    assert result["main_tone_hotspot_count"] == 0
    assert np.isclose(result["mel_axis_edges"][0], hz_to_mel(100.0))
    assert np.isclose(result["mel_axis_edges"][-1], hz_to_mel(20000.0))
    assert result["params"]["mel_scale_range"] == [0.0, 8000.0]
    assert result["params"]["mel_display_range"] == [0.0, 8000.0]
    assert np.allclose(result["params"]["analysis_mel_range"], [hz_to_mel(100.0), hz_to_mel(20000.0)])
    assert np.allclose(result["params"]["core_mel_range"], [hz_to_mel(2000.0), hz_to_mel(5000.0)])
    assert result["params"]["log_compression"] == "10*log10(power_pa2/reference_pressure_pa^2)"


def test_mel_algorithm_finds_hotspots_from_configured_main_tones():
    from base.core_algorithm.mel_spectrogram import compute_mel_spectrogram, default_mel_config

    sample_rate = 48000
    duration_s = 0.35
    t = np.arange(int(sample_rate * duration_s), dtype=np.float64) / sample_rate
    audio = (
        0.45 * np.sin(2.0 * np.pi * 1200.0 * t)
        + 0.35 * np.sin(2.0 * np.pi * 3200.0 * t)
    )

    cfg = default_mel_config()
    cfg.update(
        {
            "fmin_hz": 100.0,
            "fmax_hz": 8000.0,
            "n_mels": 64,
            "stft_nfft": 4096,
            "main_tones_hz": [1200.0, 3200.0],
            "main_tone_search_width_hz": 800.0,
        }
    )

    result = compute_mel_spectrogram(audio, sample_rate, cfg, v2pa_factor=1.0)

    hotspots = result["main_tone_hotspots"]
    assert result["main_tone_hotspot_count"] == 2
    assert len(hotspots) == 2
    assert result["hotspot"] == hotspots[0]
    assert result["global_hotspot"]["kind"] == "mel_band"
    assert result["params"]["main_tone_source"] == "configured_main_tones"
    assert result["params"]["main_tones_hz"] == [1200.0, 3200.0]
    assert result["params"]["main_tone_search_width_hz"] == 800.0
    assert [item["main_tone_frequency_hz"] for item in hotspots] == [1200.0, 3200.0]
    assert [item["main_tone_label"] for item in hotspots] == ["Tone 1", "Tone 2"]
    for item in hotspots:
        assert item["kind"] == "main_tone_mel_band"
        assert item["aggregation"] == "mean_over_time"
        assert item["source"] == "configured_main_tone"
        assert item["search_freq_low_hz"] <= item["freq_hz"] <= item["search_freq_high_hz"]


def test_mel_algorithm_requires_and_uses_v2pa_factor():
    from base.core_algorithm.mel_spectrogram import compute_mel_spectrogram, default_mel_config, hz_to_mel

    sample_rate = 48000
    duration_s = 0.25
    t = np.arange(int(sample_rate * duration_s), dtype=np.float64) / sample_rate
    audio = 0.5 * np.sin(2.0 * np.pi * 2000.0 * t)

    cfg = default_mel_config()
    cfg.update(
        {
            "fmin_hz": 100.0,
            "fmax_hz": 20000.0,
            "n_mels": 48,
            "stft_nfft": 2048,
            "main_tones_hz": [],
        }
    )

    with pytest.raises(ValueError, match="v2pa_factor"):
        compute_mel_spectrogram(audio, sample_rate, cfg)

    low_result = compute_mel_spectrogram(audio, sample_rate, cfg, v2pa_factor=0.5)
    high_result = compute_mel_spectrogram(audio, sample_rate, cfg, v2pa_factor=1.0)

    expected_delta_db = 20.0 * np.log10(1.0 / 0.5)
    actual_delta_db = high_result["overall_spl_dba"] - low_result["overall_spl_dba"]
    assert np.isclose(actual_delta_db, expected_delta_db, atol=1e-6)
    assert low_result["params"]["calibration_source"] == "v2pa_factor"
    assert low_result["params"]["pressure_scale_pa_per_sample"] == 0.5
    assert low_result["params"]["v2pa_factor"] == 0.5


def test_mel_queue_item_uses_builtin_defaults_when_json_missing(monkeypatch):
    from base.data_struct.sequence_data import SequenceData
    from ui.operation_sequence import LoadUiConfig, OptionList

    monkeypatch.setattr(LoadUiConfig, "load_data_from_json", staticmethod(lambda _path: (1, "missing")))

    option_list = OptionList.__new__(OptionList)
    option_list.config = [SequenceData("seq1")]
    option_list.default_logger = types.SimpleNamespace(error=lambda *_args, **_kwargs: None)

    option_list.get_item_default_config("梅尔频谱 (Mel) ", "梅尔频谱 (Mel) 1")

    cfg = option_list.config[0].analysis_list["梅尔频谱 (Mel) 1"]
    assert cfg["type"] == "Mel"
    assert cfg["main_tones_hz"] == [1200.0, 3500.0]
    assert cfg["main_tone_search_width_hz"] == 160.0
    assert cfg["fmin_hz"] == 100.0
    assert cfg["n_mels"] == 128
    assert cfg["window"] == "hamming"
    assert cfg["mel_scale_range"] == [0.0, 8000.0]
    assert "sample_to_pa" not in cfg
    assert "mic_sensitivity_v_per_pa" not in cfg


@pytest.fixture
def qapp():
    from PyQt5.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_mel_ui_smoke_renders_pyqtgraph_without_matplotlib(qapp, monkeypatch):
    from base.data_struct.data_deal_struct import DataDealStruct
    from ui.signal_analysis_window import Mel, get_class_mapping
    from ui.ui_analysis_config.mel_config_dialog import MelConfigWindow

    fake_manager = types.SimpleNamespace(
        load_config=lambda: {},
        save_default_config=lambda *_args, **_kwargs: True,
    )
    dialog = MelConfigWindow(fake_manager, "梅尔频谱 (Mel) 1", available_channels=[0, 1])
    dialog_config = dialog.get_default_config()
    assert dialog_config["main_tones_hz"] == [1200.0, 3500.0]
    assert dialog_config["main_tone_search_width_hz"] == 160.0
    assert dialog_config["fmin_hz"] == 100.0
    assert dialog_config["n_mels"] == 128
    assert dialog_config["mel_scale_range"] == [0.0, 8000.0]
    assert "sample_to_pa" not in dialog_config
    assert "mic_sensitivity_v_per_pa" not in dialog_config
    assert get_class_mapping()["Mel"] is Mel

    fake_result = {
        "mel_db_a": np.array([[20.0, 30.0], [25.0, 35.0], [22.0, 28.0]], dtype=float),
        "times_s": np.array([0.01, 0.02], dtype=float),
        "mel_axis": np.array([500.0, 900.0, 1300.0], dtype=float),
        "mel_axis_edges": np.array([100.0, 500.0, 1000.0, 2000.0], dtype=float),
        "mel_true_axis": np.array([500.0, 900.0, 1300.0], dtype=float),
        "mel_center_freqs_hz": np.array([400.0, 1000.0, 2000.0], dtype=float),
        "mel_freq_edges_hz": np.array([100.0, 500.0, 1000.0, 2000.0, 5000.0], dtype=float),
        "overall_spl_dba": 42.5,
        "hotspot": {
            "kind": "main_tone_mel_band",
            "mel": 900.0,
            "mel_low": 800.0,
            "mel_high": 1000.0,
            "freq_hz": 1000.0,
            "freq_low_hz": 900.0,
            "freq_high_hz": 1100.0,
            "level_dba": 35.0,
            "peak_level_dba": 38.0,
            "aggregation": "mean_over_time",
            "main_tone_frequency_hz": 1000.0,
            "main_tone_band_low_hz": 800.0,
            "main_tone_band_high_hz": 1200.0,
        },
        "global_hotspot": {
            "kind": "mel_band",
            "mel": 1300.0,
            "mel_low": 1000.0,
            "mel_high": 2000.0,
            "freq_hz": 2000.0,
            "freq_low_hz": 1500.0,
            "freq_high_hz": 2500.0,
            "level_dba": 34.0,
            "peak_level_dba": 36.0,
            "aggregation": "mean_over_time",
        },
        "main_tone_hotspots": [
            {
                "kind": "main_tone_mel_band",
                "mel": 900.0,
                "mel_low": 800.0,
                "mel_high": 1000.0,
                "freq_hz": 1000.0,
                "freq_low_hz": 900.0,
                "freq_high_hz": 1100.0,
                "level_dba": 35.0,
                "peak_level_dba": 38.0,
                "aggregation": "mean_over_time",
                "main_tone_frequency_hz": 1000.0,
                "main_tone_band_low_hz": 800.0,
                "main_tone_band_high_hz": 1200.0,
            },
            {
                "kind": "main_tone_mel_band",
                "mel": 1300.0,
                "mel_low": 1200.0,
                "mel_high": 1500.0,
                "freq_hz": 2000.0,
                "freq_low_hz": 1800.0,
                "freq_high_hz": 2200.0,
                "level_dba": 32.0,
                "peak_level_dba": 34.0,
                "aggregation": "mean_over_time",
                "main_tone_frequency_hz": 2000.0,
                "main_tone_band_low_hz": 1800.0,
                "main_tone_band_high_hz": 2200.0,
            },
        ],
        "main_tone_hotspot_count": 2,
        "params": {
            "sample_rate_hz": 48000,
            "n_mels": 3,
            "fmin_hz": 100.0,
            "fmax_hz": 20000.0,
            "mel_display_range": [0.0, 8000.0],
            "analysis_mel_range": [100.0, 2000.0],
            "core_range_hz": [2000.0, 5000.0],
            "core_mel_range": [1521.3595541555756, 2363.4658366331187],
        },
    }
    captured_kwargs = {}
    captured_args = {}

    def fake_compute_mel_spectrogram(*args, **kwargs):
        captured_args["config"] = args[2]
        captured_kwargs.update(kwargs)
        return fake_result

    monkeypatch.setattr("ui.signal_analysis_window.compute_mel_spectrogram", fake_compute_mel_spectrogram)

    widget = Mel("梅尔频谱 (Mel) 1")
    data_struct = DataDealStruct()
    old_wave = data_struct.store_wave_data
    old_multi = data_struct.store_wave_data_multi
    old_sample_rate = data_struct.sample_rate
    try:
        data_struct.store_wave_data = np.zeros(512, dtype=np.float32)
        data_struct.store_wave_data_multi = None
        data_struct.sample_rate = 48000
        widget.data_struct = data_struct
        widget.analysis_config = {
            "color_map": "magma",
            "main_tones_hz": [1000.0, 2000.0],
            "main_tone_search_width_hz": 400.0,
        }
        widget.v2pa_factor = 2.5

        result = widget.calculate_mel()

        assert result["mel_db_a"] == fake_result["mel_db_a"].tolist()
        assert captured_kwargs["v2pa_factor"] == 2.5
        assert captured_args["config"]["main_tones_hz"] == [1000.0, 2000.0]
        assert captured_args["config"]["main_tone_search_width_hz"] == 400.0
        assert widget.table_widget.rowCount() == 2
        assert widget.img_item is not None
        assert widget.plot_widget.getAxis("left").width() >= widget.LEFT_AXIS_MIN_WIDTH
        assert np.allclose(widget.plot_widget.viewRange()[1], [0.0, 8000.0])
        assert widget.analysis_region is not None
        assert widget.analysis_label is None
        assert widget.core_region is not None
        assert widget.core_label is None
        assert widget.hotspot_region is not None
        assert widget.hotspot_region.brush.color().getRgb()[:3] == (34, 197, 94)
        assert len(widget.hotspot_regions) == 2
        assert widget.hotspot_label is None
        assert widget.overall_spl_label is not None
        assert not widget.status_label.isVisible()
        assert widget.pdf_summary_exclude_fields == ("overall_spl_dba", "hotspot")
        assert widget.export_pdf_tables() == [
            {
                "title": "分析表格",
                "headers": [
                    "Main tone (Hz)",
                    "Hotspot band (kHz)",
                    "Hotspot Mel band",
                ],
                "rows": [
                    ["1000.0", "0.900-1.100", "800.0-1000.0"],
                    ["2000.0", "1.800-2.200", "1200.0-1500.0"],
                ],
            }
        ]
    finally:
        data_struct.store_wave_data = old_wave
        data_struct.store_wave_data_multi = old_multi
        data_struct.sample_rate = old_sample_rate

    algorithm_source = Path(__file__).parents[2] / "base" / "core_algorithm" / "mel_spectrogram.py"
    assert "matplotlib" not in algorithm_source.read_text(encoding="utf-8")

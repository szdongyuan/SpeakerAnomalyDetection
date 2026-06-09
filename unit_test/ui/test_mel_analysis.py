import os
import types
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_mel_algorithm_detects_synthetic_tone():
    from base.core_algorithm.mel_spectrogram import compute_mel_spectrogram, default_mel_config

    sample_rate = 48000
    duration_s = 0.25
    t = np.arange(int(sample_rate * duration_s), dtype=np.float64) / sample_rate
    audio = 0.5 * np.sin(2.0 * np.pi * 2000.0 * t)

    cfg = default_mel_config()
    cfg.update(
        {
            "fmin_hz": 100.0,
            "fmax_hz": 8000.0,
            "n_mels": 48,
            "stft_nfft": 2048,
        }
    )

    result = compute_mel_spectrogram(audio, sample_rate, cfg)

    mel_db_a = result["mel_db_a"]
    assert mel_db_a.shape[0] == 48
    assert mel_db_a.shape[1] == len(result["times_s"])
    assert np.all(np.isfinite(mel_db_a))
    assert np.isfinite(result["overall_spl_dba"])
    assert 1000.0 <= result["hotspot"]["freq_hz"] <= 3500.0


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
    assert cfg["n_mels"] == 128
    assert cfg["window"] == "hamming"


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
    assert dialog.get_default_config()["n_mels"] == 128
    assert get_class_mapping()["Mel"] is Mel

    fake_result = {
        "mel_db_a": np.array([[20.0, 30.0], [25.0, 35.0], [22.0, 28.0]], dtype=float),
        "times_s": np.array([0.01, 0.02], dtype=float),
        "mel_axis": np.array([500.0, 900.0, 1300.0], dtype=float),
        "mel_freq_edges_hz": np.array([100.0, 500.0, 1000.0, 2000.0, 5000.0], dtype=float),
        "overall_spl_dba": 42.5,
        "hotspot": {"time_s": 0.02, "mel": 900.0, "freq_hz": 1000.0, "level_dba": 35.0},
        "params": {"sample_rate_hz": 48000, "n_mels": 3},
    }
    monkeypatch.setattr("ui.signal_analysis_window.compute_mel_spectrogram", lambda *_args, **_kwargs: fake_result)

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
        widget.analysis_config = {"color_map": "magma"}

        result = widget.calculate_mel()

        assert result["mel_db_a"] == fake_result["mel_db_a"].tolist()
        assert widget.table_widget.rowCount() == 1
        assert widget.img_item is not None
    finally:
        data_struct.store_wave_data = old_wave
        data_struct.store_wave_data_multi = old_multi
        data_struct.sample_rate = old_sample_rate

    algorithm_source = Path(__file__).parents[2] / "base" / "core_algorithm" / "mel_spectrogram.py"
    assert "matplotlib" not in algorithm_source.read_text(encoding="utf-8")

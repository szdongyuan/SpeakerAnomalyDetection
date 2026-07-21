import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_QAPP = None


@pytest.fixture(scope="session")
def qapp():
    from PyQt5.QtWidgets import QApplication

    global _QAPP
    _QAPP = QApplication.instance() or QApplication([])
    _QAPP.setQuitOnLastWindowClosed(False)
    yield _QAPP


def test_spec_mel_mode_renders_without_calibration(qapp, monkeypatch):
    import pyqtgraph as pg
    from PyQt5.QtCore import QCoreApplication, QEvent

    from ui import signal_analysis_window as signal_window

    def fail_if_calibration_is_requested(raw_channel, warn_callback=None, hardware_id=None, db_path=None):
        raise AssertionError("Generic Mel spectrogram must not request acoustic calibration")

    analyze_calls = []

    def fake_analyze(
        _self,
        signal,
        fs,
        n_fft=2048,
        hop_length=256,
        n_mels=128,
        fmin_hz=0.0,
        fmax_hz=None,
        window="hann",
    ):
        analyze_calls.append(
            (
                np.asarray(signal),
                fs,
                {
                    "n_fft": n_fft,
                    "hop_length": hop_length,
                    "n_mels": n_mels,
                    "fmin_hz": fmin_hz,
                    "fmax_hz": fmax_hz,
                    "window": window,
                },
            )
        )
        return (
            np.array([0.0, 0.1, 0.2]),
            np.array([100.0, 1000.0, 5000.0]),
            np.array(
                [
                    [-40.0, -35.0, -30.0],
                    [-30.0, -20.0, -25.0],
                    [-50.0, -45.0, -40.0],
                ]
            ),
        )

    monkeypatch.setattr(
        signal_window,
        "resolve_analysis_v2pa_factor_for_channel",
        fail_if_calibration_is_requested,
    )
    monkeypatch.setattr(signal_window.MelSpectrogramAnalyzer, "analyze", fake_analyze)
    plot_calls = []
    real_plot_2d_image = signal_window.plot_2d_image

    def capture_plot_2d_image(*args, **kwargs):
        plot_calls.append(kwargs)
        return real_plot_2d_image(*args, **kwargs)

    monkeypatch.setattr(signal_window, "plot_2d_image", capture_plot_2d_image)

    sample_rate = 16000
    time_s = np.arange(sample_rate // 2, dtype=np.float64) / sample_rate
    recorded_signal = 0.2 * np.sin(2.0 * np.pi * 1000.0 * time_s)
    assert signal_window.get_class_mapping()["Spec"] is signal_window.Spectrogram
    assert "MEL" not in signal_window.get_class_mapping()

    widget = signal_window.Spectrogram("频谱分析 (Spec) 1")
    widget.data_struct.store_wave_data = recorded_signal
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = sample_rate
    widget.analysis_config = {
        "analysis_channel": 0,
        "n_fft": 1024,
        "hop_length": 256,
        "window_func": "hann",
        "freq_scale_type": "mel",
        "mel_n_mels": 40,
        "mel_fmin_hz": 0,
        "mel_fmax_mode": "manual",
        "mel_fmax_hz": 8000,
        "color_map": "magma",
        "custom_limit": False,
    }

    try:
        assert widget.calculate_spec() is True

        assert analyze_calls[-1][1] == sample_rate
        assert analyze_calls[-1][2]["n_mels"] == 40
        assert analyze_calls[-1][2]["fmax_hz"] == 8000
        assert widget.current_plot_widget is not None
        assert plot_calls[-1]["z_range"] is None
        assert plot_calls[-1]["z"].shape[1] == 3
        plot_widget = widget.current_plot_widget.findChild(pg.PlotWidget)
        assert plot_widget is not None
        assert "Mel" in plot_widget.plotItem.titleLabel.text
        assert plot_widget.getAxis("left").labelText == "频率 (Hz)"
        assert not hasattr(widget, "v2pa_factor")

        first_plot_container = widget.current_plot_widget
        widget.analysis_config["mel_fmax_mode"] = "nyquist"
        widget.analysis_config["custom_limit"] = True
        widget.analysis_config["bottom_limit"] = -80
        widget.analysis_config["top_limit"] = 0
        assert widget.calculate_spec() is True

        assert analyze_calls[-1][2]["fmax_hz"] is None
        assert plot_calls[-1]["z_range"] == (-80, 0)
        assert widget.current_plot_widget is not first_plot_container
        assert widget.plot_container_layout.count() == 1
        assert widget.current_plot_widget is not widget.stft_plot_widget
    finally:
        widget.close()
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.processEvents()


def test_spec_linear_mode_applies_custom_color_limits_verbatim(qapp):
    from PyQt5.QtCore import QCoreApplication, QEvent

    from ui import signal_analysis_window as signal_window

    sample_rate = 8000
    time_s = np.arange(sample_rate // 4, dtype=np.float64) / sample_rate
    recorded_signal = 0.2 * np.sin(2.0 * np.pi * 1000.0 * time_s)

    widget = signal_window.Spectrogram("频谱分析 (Spec) 1")
    widget.data_struct.store_wave_data = recorded_signal
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = sample_rate
    widget.analysis_config = {
        "analysis_channel": 0,
        "n_fft": 256,
        "hop_length": 64,
        "window_func": "hann",
        "freq_scale_type": "linear",
        "color_map": "viridis",
        "custom_limit": True,
        "bottom_limit": -80,
        "top_limit": 0,
    }

    try:
        assert widget.calculate_spec() is True

        np.testing.assert_allclose(widget.img_item.getLevels(), [-80, 0])
    finally:
        widget.close()
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.processEvents()


def test_spec_mel_mode_reports_calculation_errors(qapp, monkeypatch):
    from PyQt5.QtCore import QCoreApplication, QEvent

    from ui import signal_analysis_window as signal_window

    log_calls = []

    def make_logger(thread_holder):
        assert thread_holder == "core"
        return SimpleNamespace(
            exception=lambda message, *args: log_calls.append((message, args))
        )

    def fail_analysis(_self, signal, fs, **kwargs):
        raise signal_window.InvalidMelBandConfigurationError(
            "invalid Mel band configuration"
        )

    monkeypatch.setattr(signal_window.LogManager, "set_log_handler", make_logger)
    monkeypatch.setattr(signal_window.MelSpectrogramAnalyzer, "analyze", fail_analysis)
    warning_calls = []
    monkeypatch.setattr(
        signal_window.MessageBox,
        "warning",
        lambda *args, **kwargs: warning_calls.append((args, kwargs)),
    )

    widget = signal_window.Spectrogram("频谱分析 (Spec) 1")
    widget.data_struct.store_wave_data = np.ones(1024, dtype=np.float64)
    widget.data_struct.store_wave_data_multi = None
    widget.data_struct.sample_rate = 48000
    widget.analysis_config = {
        "analysis_channel": 0,
        "freq_scale_type": "mel",
        "mel_fmax_mode": "nyquist",
    }

    try:
        assert widget.calculate_spec() is False

        assert widget.current_plot_widget is None
        assert warning_calls == [
            (
                (widget, "提示", "Mel 频谱计算失败，请减少 Mel 频带数量或增大 FFT 窗长。"),
                {},
            )
        ]
        assert len(log_calls) == 1
        assert "频谱分析失败" in log_calls[0][0]
        assert log_calls[0][1] == (
            "mel",
            48000,
            2048,
            256,
            128,
            0,
            "nyquist",
            None,
        )
    finally:
        widget.close()
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.processEvents()

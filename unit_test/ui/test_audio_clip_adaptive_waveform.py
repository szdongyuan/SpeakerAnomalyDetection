import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from ui.adaptive_waveform import AdaptiveWaveformItem
from ui.custom_ui_widget import audio_clip_extraction_dialog as clip_module


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(qapp, monkeypatch):
    source = np.sin(np.arange(48001))
    monkeypatch.setattr(clip_module.librosa, "load", lambda path, *, sr, mono: (source, 48000))
    window = clip_module.AudioClipExtractionDialog(source_audio=source, sample_rate=48000)
    window.resize(800, 500)
    window.show()
    window.load_and_display_waveform("source.wav")
    qapp.processEvents()
    yield window, source
    window.close()


def test_clip_adaptive_waveform_and_exact_sample_times(dialog, qapp):
    window, source = dialog
    item = window.plot_curve
    assert isinstance(item, AdaptiveWaveformItem)
    assert item.opts["autoDownsample"] is True
    assert item.opts["clipToView"] is True
    assert item.opts["downsampleMethod"] == "peak"
    window.plot_widget.setXRange(0, 1, padding=0)
    qapp.processEvents()
    assert len(item.getData()[0]) < len(source)
    np.testing.assert_array_equal(item.xData, np.arange(len(source)) / 48000)
    assert item.xData[-1] == (len(source) - 1) / 48000
    np.testing.assert_array_equal(item.yData, source)


@pytest.mark.parametrize("zoom", [False, True])
def test_clip_saves_original_selected_samples_after_zoom(dialog, qapp, monkeypatch, zoom):
    window, source = dialog
    saved = []
    monkeypatch.setattr(clip_module, "save_audio_simple", lambda path, data, fs: saved.append((data.copy(), fs)))
    monkeypatch.setattr(clip_module.FileOps, "get_relative_path", lambda *args: "clip.wav")
    window.save_clip = True
    window.save_clip_path = "clip.wav"
    window.region.show()
    window.region.setRegion((0.25, 0.5))
    if zoom:
        window.plot_widget.setXRange(0.25, 0.2502, padding=0)
        qapp.processEvents()
    window.on_click_ok_btn()
    np.testing.assert_array_equal(saved[0][0], source[12000:24000])
    np.testing.assert_array_equal(window.return_value[0], source[12000:24000])
    np.testing.assert_array_equal(window.audio_data, source)
    np.testing.assert_array_equal(window.source_audio, source)
    np.testing.assert_array_equal(window.plot_curve.yData, source)
    assert saved[0][1] == 48000
    assert window.return_value[1:] == ("clip.wav", 12000)


def test_clip_cancel_and_fixed_length_controls(dialog):
    window, _ = dialog
    window.region.show()
    window.region.setRegion((0.25, 0.5))
    window.on_selection_cancelled()
    assert window.selected_region_time == (None, None)
    window.fixed_len_checkbox.setChecked(True)
    window.clip_len_spinbox.setValue(0.2)
    assert window.plot_widget.drag_mode == "click"
    assert window.plot_widget.region_len == 0.2
    assert not window.region.isVisible()
    window.fixed_len_checkbox.setChecked(False)
    assert window.plot_widget.drag_mode == "click_drag"
    window.reject()
    assert window.return_value == (None, None, None)


@pytest.mark.parametrize("empty", [False, True])
def test_clip_failed_or_empty_load_clears_display(dialog, monkeypatch, empty):
    window, _ = dialog
    errors = []
    monkeypatch.setattr(clip_module.MessageBox, "critical", lambda *args: errors.append(args))
    def load(*args, **kwargs):
        if empty:
            return np.array([]), 48000
        raise ValueError("unreadable audio")
    monkeypatch.setattr(clip_module.librosa, "load", load)
    window.load_and_display_waveform("bad.wav")
    assert len(errors) == 1
    assert window.audio_data is None
    assert window.sample_rate is None
    assert window.plot_curve.xData is None
    assert not window.plot_curve.curve.isVisible()


@pytest.mark.parametrize("start_sample", [0, 9], ids=["full-range", "final-sample"])
def test_clip_selection_duration_includes_final_original_sample(qapp, monkeypatch, start_sample):
    source = np.arange(10, dtype=float)
    monkeypatch.setattr(clip_module.librosa, "load", lambda path, *, sr, mono: (source, 10))
    saved = []
    monkeypatch.setattr(clip_module, "save_audio_simple", lambda path, data, fs: saved.append(data.copy()))
    monkeypatch.setattr(clip_module.FileOps, "get_relative_path", lambda *args: "clip.wav")
    window = clip_module.AudioClipExtractionDialog(sample_rate=10, save_clip=True, save_clip_path="clip.wav")
    try:
        window.show()
        window.load_and_display_waveform("source.wav")
        window.region.show()
        window.region.setRegion((start_sample / 10, 1.0))
        qapp.processEvents()
        assert window.region.getRegion() == (start_sample / 10, 1.0)
        np.testing.assert_array_equal(window.plot_curve.xData, np.arange(10) / 10)
        assert window.plot_curve.xData[-1] == 0.9
        window.on_click_ok_btn()
        np.testing.assert_array_equal(window.return_value[0], source[start_sample:])
        np.testing.assert_array_equal(saved[0], source[start_sample:])
        assert window.return_value[2] == 10 - start_sample
    finally:
        window.close()

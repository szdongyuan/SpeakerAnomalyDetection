import inspect
import os
import tempfile

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "speaker_anomaly_pdf_graph_numba_cache"),
)
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

from PyQt5.QtWidgets import QApplication, QWidget

from base import pdf_result_exporter
from base.pdf_result_exporter import export_plot_widget_image


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


def test_export_plot_widget_image_writes_png(qapp, tmp_path):
    import pyqtgraph as pg

    plot = pg.PlotWidget()
    plot.plot([1, 2, 3], [2, 3, 4])

    out = export_plot_widget_image(plot, str(tmp_path), "plot")

    assert out.endswith(".png")
    assert os.path.isabs(out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_plot_export_helper_does_not_use_whole_window_grab():
    src = inspect.getsource(pdf_result_exporter.export_plot_widget_image)
    assert ".grab(" not in src


def test_analysis_graph_widget_exports_analysis_plot(qapp, tmp_path, monkeypatch):
    from ui import signal_analysis_window
    from ui.signal_analysis_window import AnalysisGraphWidget

    widget = AnalysisGraphWidget()
    calls = []

    def fake_export(plot_widget, output_dir, filename_base):
        calls.append((plot_widget, output_dir, filename_base))
        return os.path.join(output_dir, f"{filename_base}.png")

    monkeypatch.setattr(signal_analysis_window, "export_plot_widget_image", fake_export)

    images = widget.export_pdf_images(str(tmp_path))

    assert calls == [(widget.analysis_plot, str(tmp_path), "analysis_graph")]
    assert images == [{"title": "", "path": os.path.join(str(tmp_path), "analysis_graph.png")}]


def test_analysis_graph_widget_export_failure_propagates(qapp, tmp_path, monkeypatch):
    from ui import signal_analysis_window
    from ui.signal_analysis_window import AnalysisGraphWidget

    widget = AnalysisGraphWidget()

    def fake_export(plot_widget, output_dir, filename_base):
        raise RuntimeError("analysis export failed")

    monkeypatch.setattr(signal_analysis_window, "export_plot_widget_image", fake_export)

    with pytest.raises(RuntimeError, match="analysis export failed"):
        widget.export_pdf_images(str(tmp_path))


def test_spectrogram_exports_child_plot_widget_from_container(qapp, tmp_path, monkeypatch):
    import pyqtgraph as pg
    from ui import signal_analysis_window
    from ui.signal_analysis_window import Spectrogram

    widget = Spectrogram("Spec1")
    container = QWidget()
    plot = pg.PlotWidget(container)
    widget.current_plot_widget = container
    calls = []

    def fake_export(plot_widget, output_dir, filename_base):
        calls.append((plot_widget, output_dir, filename_base))
        return os.path.join(output_dir, f"{filename_base}.png")

    monkeypatch.setattr(signal_analysis_window, "export_plot_widget_image", fake_export)

    images = widget.export_pdf_images(str(tmp_path))

    assert calls == [(plot, str(tmp_path), "spectrogram")]
    assert images == [{"title": "Spec1", "path": os.path.join(str(tmp_path), "spectrogram.png")}]


def test_spectrogram_export_sets_left_axis_width_guard(qapp, tmp_path, monkeypatch):
    import pyqtgraph as pg
    from ui import signal_analysis_window
    from ui.signal_analysis_window import Spectrogram

    widget = Spectrogram("Spec1")
    plot = pg.PlotWidget()
    widget.current_plot_widget = plot
    left_axis = plot.getAxis("left")
    calls = []

    def fake_export(plot_widget, output_dir, filename_base):
        calls.append((plot_widget, output_dir, filename_base, left_axis.width()))
        return os.path.join(output_dir, f"{filename_base}.png")

    monkeypatch.setattr(signal_analysis_window, "export_plot_widget_image", fake_export)

    images = widget.export_pdf_images(str(tmp_path))

    assert calls[0][0] is plot
    assert calls[0][3] >= 110
    assert images == [{"title": "Spec1", "path": os.path.join(str(tmp_path), "spectrogram.png")}]


def test_spectrogram_export_failure_propagates(qapp, tmp_path, monkeypatch):
    import pyqtgraph as pg
    from ui import signal_analysis_window
    from ui.signal_analysis_window import Spectrogram

    widget = Spectrogram("Spec1")
    widget.current_plot_widget = pg.PlotWidget()

    def fake_export(plot_widget, output_dir, filename_base):
        raise RuntimeError("spectrogram export failed")

    monkeypatch.setattr(signal_analysis_window, "export_plot_widget_image", fake_export)

    with pytest.raises(RuntimeError, match="spectrogram export failed"):
        widget.export_pdf_images(str(tmp_path))


def test_reference_spectrum_exports_each_channel_plot(qapp, tmp_path, monkeypatch):
    import pyqtgraph as pg
    from ui import reference_spectrum_analysis_window
    from ui.reference_spectrum_analysis_window import ReferenceSpectrumCompareWindow

    widget = ReferenceSpectrumCompareWindow("RSC1")
    plot_a = pg.PlotWidget()
    plot_b = pg.PlotWidget()
    widget.channel_plots = [plot_a, plot_b]
    calls = []

    def fake_export(plot_widget, output_dir, filename_base):
        calls.append((plot_widget, output_dir, filename_base))
        return os.path.join(output_dir, f"{filename_base}.png")

    monkeypatch.setattr(reference_spectrum_analysis_window, "export_plot_widget_image", fake_export)

    images = widget.export_pdf_images(str(tmp_path))

    assert calls == [
        (plot_a, str(tmp_path), "reference_spectrum_channel_1"),
        (plot_b, str(tmp_path), "reference_spectrum_channel_2"),
    ]
    assert images == [
        {"title": "RSC1 CH1", "path": os.path.join(str(tmp_path), "reference_spectrum_channel_1.png")},
        {"title": "RSC1 CH2", "path": os.path.join(str(tmp_path), "reference_spectrum_channel_2.png")},
    ]


def test_reference_spectrum_export_failure_propagates(qapp, tmp_path, monkeypatch):
    import pyqtgraph as pg
    from ui import reference_spectrum_analysis_window
    from ui.reference_spectrum_analysis_window import ReferenceSpectrumCompareWindow

    widget = ReferenceSpectrumCompareWindow("RSC1")
    widget.channel_plots = [pg.PlotWidget()]

    def fake_export(plot_widget, output_dir, filename_base):
        raise RuntimeError("reference spectrum export failed")

    monkeypatch.setattr(reference_spectrum_analysis_window, "export_plot_widget_image", fake_export)

    with pytest.raises(RuntimeError, match="reference spectrum export failed"):
        widget.export_pdf_images(str(tmp_path))

from types import SimpleNamespace

from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget

from unit_test.ui.test_spl_runtime import (
    _load_signal_analysis_module_without_heavy_optional_imports,
)


class _Axis:
    def __init__(self, label_text=""):
        self.labelText = label_text
        self.style = {}

    def setTickFont(self, font):
        self.tick_font = font

    def setStyle(self, **style):
        self.style.update(style)

    def setTextPen(self, pen):
        self.text_pen = pen

    def setLabel(self, text, **style):
        self.labelText = text
        self.label_style = style


class _PlotWidget:
    def __init__(self):
        self.axes = {
            "bottom": _Axis("Time (s)"),
            "left": _Axis("Frequency (Hz)"),
        }
        self.plotItem = SimpleNamespace(
            titleLabel=SimpleNamespace(text="Spectrogram (Linear Scale)")
        )

    def getAxis(self, orientation):
        return self.axes[orientation]

    def setTitle(self, title, **style):
        self.plotItem.titleLabel.text = title
        self.title_style = style


def test_spectrogram_reserves_space_for_large_axis_fonts():
    signal_module = _load_signal_analysis_module_without_heavy_optional_imports()
    plot_widget = _PlotWidget()
    color_bar = SimpleNamespace(axis=_Axis())
    window = SimpleNamespace(
        plot_container=SimpleNamespace(
            findChildren=lambda _widget_type: [plot_widget]
        ),
        stft_colorbar=color_bar,
    )

    signal_module.Spectrogram.set_color_font_size(window)

    bottom_style = plot_widget.getAxis("bottom").style
    left_style = plot_widget.getAxis("left").style
    color_bar_style = color_bar.axis.style
    assert bottom_style["autoExpandTextSpace"] is False
    assert bottom_style["tickTextHeight"] == 26
    assert left_style["autoExpandTextSpace"] is False
    assert left_style["tickTextWidth"] == 72
    assert "hideOverlappingLabels" not in left_style
    assert color_bar_style["autoExpandTextSpace"] is False
    assert color_bar_style["tickTextWidth"] == 44


def test_report_plot_providers_only_return_known_business_plots():
    app = QApplication.instance() or QApplication([])
    signal_module = _load_signal_analysis_module_without_heavy_optional_imports()

    spl = signal_module.Spl("SPL")
    fft = signal_module.FftAnalysis("FFT")
    fba = signal_module.FrequencyBandAnalysis("FBA")
    spectrogram = signal_module.Spectrogram("Spec")
    loudness = signal_module.LoudnessAnalysis("LOUD")
    try:
        assert spl.get_report_plot_widgets() == [spl.analysis_plot]
        assert fft.get_report_plot_widgets() == [fft.analysis_plot]
        assert fba.get_report_plot_widgets() == [fba.analysis_plot]

        spectrogram.current_plot_widget = spectrogram.stft_plot_widget
        assert spectrogram.get_report_plot_widgets() == [
            spectrogram.stft_plot_widget
        ]

        logarithmic_plot = signal_module.pg.PlotWidget()
        logarithmic_container = QWidget()
        logarithmic_layout = QVBoxLayout(logarithmic_container)
        logarithmic_layout.addWidget(logarithmic_plot)
        spectrogram.current_plot_widget = logarithmic_container
        assert spectrogram.get_report_plot_widgets() == [logarithmic_plot]

        profile_plot = signal_module.pg.PlotWidget()
        heatmap_plot = signal_module.pg.PlotWidget()
        heatmap_container = QWidget()
        heatmap_layout = QVBoxLayout(heatmap_container)
        heatmap_layout.addWidget(heatmap_plot)
        loudness.specific_loudness_profile_widget = profile_plot
        loudness.specific_loudness_widget = heatmap_container

        assert loudness.get_report_plot_widgets() == [
            loudness.analysis_plot,
            profile_plot,
            heatmap_plot,
        ]
    finally:
        for widget in (spl, fft, fba, spectrogram, loudness):
            widget.close()
        app.processEvents()

from types import SimpleNamespace

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

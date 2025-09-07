import os

import librosa
import numpy as np
import pyqtgraph as pg
from librosa.core import spectrum
from librosa.feature import spectral
from librosa.sequence import dtw
from pyqtgraph import mkPen
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import QTextEdit, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QTableWidget, QTableWidgetItem, QHeaderView
from scipy.signal import find_peaks
from base.utils.data_alignment import align_signals_by_peaks, align_signals_with_peak
from scipy.spatial.distance import cosine, euclidean, cityblock
from scipy.stats import pearsonr
from skimage.feature import local_binary_pattern, match_template
from base.pre_processing.audio_feature_extraction import AudioFeatureExtraction

from base.data_struct.data_deal_struct import DataDealStruct
from base.load_audio import load_audio_simple
from base.log_manager import LogManager
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from base.pre_processing.audio_peak_detection import peak_detection, filter_peaks_by_spectral_flux
from base.pre_processing.audio_equalizer import AudioEqualizer
from base.utils.custom_signals import sign
from base.utils.smooth import smooth
from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.graph_widget import plot_2d_image, custom_log_tick_strings


def get_class_mapping():
    """
    Retrieves the class mapping dictionary.

    This method returns a dictionary where the keys are string identifiers and the values are the corresponding classes.
    This mapping is typically used to dynamically retrieve the appropriate class based on an identifier.

    Returns:
        dict: A dictionary containing the class mapping, in the format {"identifier": class}.
    """
    class_mapping = {
        "SPL": Spl,
        "Spec": Spectrogram,
        "LP": LooseParticle,
        "PD": PeakDetection,
        "PM": PatternMatch,
        "ED": PipelinePdPm,
    }
    return class_mapping


class AnalysisGraphWidget(QWidget):

    def __init__(self):
        super().__init__()

        self.analysis_plot_top = pg.PlotWidget()
        self.analysis_plot_bottom = pg.PlotWidget()

        self.plot_list = [self.analysis_plot_top, self.analysis_plot_bottom]

        self.set_plot_font_size(20)
        self.init_ui()

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))

        self.analysis_plot_top.setBackground("white")
        self.analysis_plot_bottom.setBackground("white")

        layout = QVBoxLayout()
        layout.addWidget(self.analysis_plot_top)
        layout.addWidget(self.analysis_plot_bottom)
        self.setLayout(layout)

    def set_plot_channel_flag(self):
        font_top = QFont()
        font_top.setPixelSize(20)

        self.text_top = pg.TextItem(
            text="Channel_1", 
            color=(0, 0, 0),  # 黑色
            fill=(255, 255, 255, 150)  # 半透明背景
        )
        self.text_top.setFont(font_top)
        self.text_top.setPos(0, 0)  # 设置左上角位置
        self.text_top.setParentItem(self.analysis_plot_top.getViewBox())  # 绑定到视图框
        self.text_top.setZValue(1000)

        font_bottom = QFont()
        font_bottom.setPixelSize(20)

        self.text_bottom = pg.TextItem(
            text="Channel_2", 
            color=(0, 0, 0),  # 黑色
            fill=(255, 255, 255, 150)  # 半透明背景
        )
        self.text_bottom.setFont(font_bottom)
        self.text_bottom.setPos(0, 0)  # 设置左上角位置
        self.text_bottom.setParentItem(self.analysis_plot_bottom.getViewBox())  # 绑定到视图框
        self.text_bottom.setZValue(1000)
        

    def set_plot_font_size(self, font_size: int):
        for i in self.plot_list:
            if not hasattr(i, "getPlotItem"):
                continue
            try:
                plot_item = i.getPlotItem()
                b_axis = plot_item.getAxis("bottom")
                l_axis = plot_item.getAxis("left")
                font = QFont()
                font.setPixelSize(font_size)
                b_axis.logTickStrings = custom_log_tick_strings
                b_axis.setTickFont(font)
                l_axis.setTickFont(font)
                b_axis.setTextPen("black")
                l_axis.setTextPen("black")
                b_axis.setLabel(b_axis.labelText, **{"font-size": f"{font_size}px"})
                l_axis.setLabel(l_axis.labelText, **{"font-size": f"{font_size}px"})
            except Exception as e:
                print(f"Plot font size set error: {e}")


class Spl(AnalysisGraphWidget):
    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.deviation_value = None
        self.analysis_config = None
        self.result = {}
        self.setWindowTitle(title_name)
        self.set_plot_channel_flag()

    def calculate_spl(self):
        # calculate Sound Pressure Level according to recorded_signal
        recorded_signal = self.data_struct.store_wave_data
        sample_rate = self.data_struct.sample_rate
        signal_duration = np.linspace(0, len(recorded_signal[0]) / sample_rate, len(recorded_signal[0]))
        reference_pressure = 20e-6
        signal_spl_0 = AudioThdFrequencyResponseAnalysis().spl_calculation(
            recorded_signal[0], reference_pressure, deviation=self.deviation_value
        )
        signal_spl_1 = AudioThdFrequencyResponseAnalysis().spl_calculation(
            recorded_signal[1], reference_pressure, deviation=self.deviation_value
        )
        if self.analysis_config["smooth_checked"]:
            signal_spl_0 = smooth(signal_spl_0, window_size=1102, method="rms")
            signal_spl_1 = smooth(signal_spl_1, window_size=1102, method="rms")
        limit_checked = self.analysis_config.get("limit_checked")
        self_defined = self.analysis_config.get("self_defined")
        if limit_checked:
            if self_defined:
                upper_limit = self.analysis_config.get("upper_limit")
                lower_limit = self.analysis_config.get("lower_limit")
                self.plot_spl(
                    signal_duration, signal_spl_0, self.analysis_plot_top, upper_limit=upper_limit, lower_limit=lower_limit
                )
                self.plot_spl(
                    signal_duration, signal_spl_1, self.analysis_plot_bottom, upper_limit=upper_limit, lower_limit=lower_limit
                )
            else:
                self.plot_spl(signal_duration, signal_spl_0, self.analysis_plot_top)
                self.plot_spl(signal_duration, signal_spl_1, self.analysis_plot_bottom)
        else:
            self.plot_spl(signal_duration, signal_spl_0, self.analysis_plot_top)
            self.plot_spl(signal_duration, signal_spl_1, self.analysis_plot_bottom)
        self.result = {
            "signal_duration": signal_duration.tolist(),
            "recorded_signal": recorded_signal.tolist(),
            "signal_spl": signal_spl_0.tolist(),
        }
        return self.result

    def plot_spl(self, signal_duration, signal_spl, analysis_plot, upper_limit="", lower_limit=""):
        analysis_plot.clear()
        analysis_plot.plot(signal_duration, signal_spl, pen=mkPen(color=(51, 196, 77)))
        if lower_limit and upper_limit:
            upper_limit = float(upper_limit)
            lower_limit = float(lower_limit)
            out_range_points = []
            current_out_range = []
            for i in range(len(signal_spl)):
                if signal_spl[i] <= lower_limit or signal_spl[i] >= upper_limit:
                    current_out_range.append((signal_duration[i], signal_spl[i]))
                else:
                    if current_out_range:
                        out_range_points.append(current_out_range)
                        current_out_range = []
            if current_out_range:
                out_range_points.append(current_out_range)

            for points in out_range_points:
                x = [point[0] for point in points]
                y = [point[1] for point in points]
                out_range_plot = pg.PlotDataItem(x, y, pen="r")
                analysis_plot.addItem(out_range_plot)
            dashed_pen = mkPen(color=(128, 0, 128), width=1, style=Qt.DashLine)
            lower_limit1 = pg.InfiniteLine(angle=0, pos=lower_limit, pen=dashed_pen)
            analysis_plot.addItem(lower_limit1)
            upper_limit1 = pg.InfiniteLine(angle=0, pos=upper_limit, pen=dashed_pen)
            analysis_plot.addItem(upper_limit1)
        analysis_plot.setLabel("left", "SPL (dB)")
        analysis_plot.setLabel("bottom", "Time (s)")
        analysis_plot.showGrid(x=True, y=True)


class Frequency(AnalysisGraphWidget):

    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.smooth_flag = False
        self.temp_frequency_list = None
        self.deviation_value = None
        self.analysis_config = None
        self.result = {}
        self.setWindowTitle(title_name)

    def calculate_fr(self):
        stimulus_signal = self.data_struct.stimulus_data
        recorded_signal = self.data_struct.store_wave_data
        sr = self.data_struct.sample_rate
        fr, frequency_list = AudioThdFrequencyResponseAnalysis().calculate_fr(
            stimulus_signal, recorded_signal, sr, is_smooth=self.analysis_config["smooth_checked"]
        )
        limit_checked = self.analysis_config.get("limit_checked")
        self_defined = self.analysis_config.get("self_defined")
        if limit_checked:
            if self_defined:
                upper_limit = self.analysis_config.get("upper_limit")
                lower_limit = self.analysis_config.get("lower_limit")
                self.plot_fr(frequency_list, fr, upper_limit=upper_limit, lower_limit=lower_limit)
            else:
                self.plot_fr(frequency_list, fr)
        else:
            self.plot_fr(frequency_list, fr)
        self.result = {"fr": fr.tolist(), "frequency_list": frequency_list.tolist()}
        return self.result

    def plot_fr(self, frequency_list, fr, upper_limit="", lower_limit=""):
        self.analysis_plot.clear()
        fr = fr + 94 + self.deviation_value
        self.analysis_plot.plot(frequency_list, fr, pen=mkPen(color=(51, 196, 77)))
        if lower_limit and upper_limit:
            upper_limit = float(upper_limit)
            lower_limit = float(lower_limit)
            out_range_points = []
            current_out_range = []
            for i in range(len(fr)):
                if fr[i] <= lower_limit or fr[i] >= upper_limit:
                    current_out_range.append((frequency_list[i], fr[i]))
                else:
                    if current_out_range:
                        out_range_points.append(current_out_range)
                        current_out_range = []
            if current_out_range:
                out_range_points.append(current_out_range)

            for points in out_range_points:
                x = [point[0] for point in points]
                y = [point[1] for point in points]
                x_min, x_max = min(x), max(x)
                padding_x = (x_max - x_min) * 0.05
                self.analysis_plot.setXRange(x_min - padding_x, x_max + padding_x)
                out_range_plot = pg.PlotDataItem(x, y, pen="r")
                self.analysis_plot.addItem(out_range_plot)
            dashed_pen = mkPen(color=(128, 0, 128), width=1, style=Qt.DashLine)
            lower_limit1 = pg.InfiniteLine(angle=0, pos=lower_limit, pen=dashed_pen)
            self.analysis_plot.addItem(lower_limit1)
            upper_limit1 = pg.InfiniteLine(angle=0, pos=upper_limit, pen=dashed_pen)
            self.analysis_plot.addItem(upper_limit1)
        self.analysis_plot.setLabel("left", "Amplitude (dB)")
        self.analysis_plot.setLabel("bottom", "Frequency (Hz)")
        self.analysis_plot.showGrid(x=True, y=True)


class Spectrogram(QWidget):
    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.deviation_value = None
        self.analysis_config = None
        self.current_plot_widget = None
        self.stft_plot_widget_top = None
        self.stft_plot_widget_bottom = None
        self.img_item_top = None
        self.stft_colorbar_top = None
        self.stft_colorbar_top = None
        self.init_ui()
        self.setWindowTitle(title_name)

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.main_layout = QVBoxLayout(self)

        self.plot_container = QWidget()
        self.plot_container_layout = QVBoxLayout(self.plot_container)
        self.plot_container_layout.setContentsMargins(0, 0, 0, 0)

        self.main_layout.addWidget(self.plot_container)

        self._init_stft_plot_components()

    def _init_stft_plot_components(self):
        self.stft_plot_widget_top = pg.PlotWidget()
        self.stft_plot_widget_top.setBackground("white")
        self.img_item_top = pg.ImageItem()
        self.stft_plot_widget_top.addItem(self.img_item_top)

        self.stft_plot_widget_bottom = pg.PlotWidget()
        self.stft_plot_widget_bottom.setBackground("white")
        self.img_item_bottom = pg.ImageItem()
        self.stft_plot_widget_bottom.addItem(self.img_item_bottom)

    def set_color_font_size(self):
        plot_widgets = self.plot_container.findChildren(pg.PlotWidget)
        for plot_widget in plot_widgets:
            font = QFont()
            font.setPixelSize(20)
            b_axis = plot_widget.getAxis("bottom")
            l_axis = plot_widget.getAxis("left")
            b_axis.setTickFont(font)
            l_axis.setTickFont(font)
            b_axis.setTextPen("black")
            l_axis.setTextPen("black")
            b_axis.setLabel(b_axis.labelText, **{"font-size": "20px"})
            l_axis.setLabel(l_axis.labelText, **{"font-size": "20px"})

            current_title = plot_widget.plotItem.titleLabel.text  # 获取当前标题
            plot_widget.setTitle(current_title, size="20px", color="black")

        if self.stft_colorbar_top:
            color_bar_axis_top = self.stft_colorbar_top.axis
            color_bar_font_top = QFont()
            color_bar_font_top.setPixelSize(20)
            color_bar_axis_top.setTickFont(color_bar_font_top)
            color_bar_axis_top.setTextPen("black")
            # color_bar_axis.setStyle(tickTextOffset=10)
        if self.stft_colorbar_bottom:
            color_bar_axis_bottom = self.stft_colorbar_bottom.axis
            color_bar_font_bottom = QFont()
            color_bar_font_bottom.setPixelSize(20)
            color_bar_axis_bottom.setTickFont(color_bar_font_bottom)
            color_bar_axis_bottom.setTextPen("black")
            # color_bar_axis.setStyle(tickTextOffset=10)

    def calculate_spec(self):
        recorded_signal = self.data_struct.store_wave_data
        sample_rate = self.data_struct.sample_rate

        n_fft = self.analysis_config.get("n_fft", 2048)
        hop_length = self.analysis_config.get("hop_length", 256)
        color_map = self.analysis_config.get("color_map", "viridis")
        window_func = self.analysis_config.get("window_func", "hann")
        freq_scale_type = self.analysis_config.get("freq_scale_type", "linear")
        top_limit = self.analysis_config.get("top_limit", 70)
        bottom_limit = self.analysis_config.get("bottom_limit", 50)
        custom_limit_flag = self.analysis_config.get("custom_limit", False)

        mid_value = (top_limit - bottom_limit) / 2
        max_value = top_limit + mid_value
        min_value = bottom_limit - mid_value

        if freq_scale_type == "log":
            fmin_cqt = librosa.note_to_hz("C1")
            CQT_complex_top, freqs_top, times_top = AudioThdFrequencyResponseAnalysis().compute_cqt(
                y=recorded_signal[0], sr=sample_rate, hop_length=hop_length, n_fft=n_fft, fmin=fmin_cqt
            )
            CQT_complex_bottom, freqs_bottom, times_bottom = AudioThdFrequencyResponseAnalysis().compute_cqt(
                y=recorded_signal[1], sr=sample_rate, hop_length=hop_length, n_fft=n_fft, fmin=fmin_cqt
            )

            CQT_mag_top = np.abs(CQT_complex_top)
            CQT_db_top = librosa.amplitude_to_db(CQT_mag_top, ref=20e-6)
            Z_top = CQT_db_top.T
            CQT_mag_bottom = np.abs(CQT_complex_top)
            CQT_db_bottom = librosa.amplitude_to_db(CQT_mag_bottom, ref=20e-6)
            Z_bottom = CQT_db_bottom.T

            target_ticks_hz = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
            major_ticks_top = []
            custom_y_ticks_top = None
            major_ticks_bottom = []
            custom_y_ticks_bottom = None

            y_top_min_hz, y_top_max_hz = freqs_top.min(), freqs_top.max()
            for freq in target_ticks_hz:
                if y_top_min_hz <= freq <= y_top_max_hz:
                    label_top = f"{freq} Hz" if freq < 1000 else f"{freq/1000:.0f} kHz"
                    major_ticks_top.append((freq, label_top))
            y_bottom_min_hz, y_bottom_max_hz = freqs_bottom.min(), freqs_bottom.max()
            for freq in target_ticks_hz:
                if y_bottom_min_hz <= freq <= y_bottom_max_hz:
                    label_bottom = f"{freq} Hz" if freq < 1000 else f"{freq/1000:.0f} kHz"
                    major_ticks_bottom.append((freq, label_bottom))

            custom_y_ticks_top = [major_ticks_top, []] if major_ticks_top else None
            custom_y_ticks_bottom = [major_ticks_bottom, []] if major_ticks_bottom else None

            cqt_plot_widget_top, self.stft_colorbar_top = plot_2d_image(
                x=times_top,
                y=freqs_top,
                z=Z_top,
                title="Spectrogram(Log Scale)(Channel_1)",
                xlabel="Time (s)",
                ylabel="Frequency (Hz)",
                colormap=color_map,
                x_range=(times_top.min(), times_top.max()),
                y_range=(freqs_top.min(), freqs_top.max()),
                y_ticks=custom_y_ticks_top,
                background_color="white",
            )
            cqt_plot_widget_bottom, self.stft_colorbar_bottom = plot_2d_image(
                x=times_bottom,
                y=freqs_bottom,
                z=Z_bottom,
                title="Spectrogram(Log Scale)(Channel_2)",
                xlabel="Time (s)",
                ylabel="Frequency (Hz)",
                colormap=color_map,
                x_range=(times_bottom.min(), times_bottom.max()),
                y_range=(freqs_bottom.min(), freqs_bottom.max()),
                y_ticks=custom_y_ticks_bottom,
                background_color="white",
            )
            self.plot_container_layout.addWidget(cqt_plot_widget_top)
            self.plot_container_layout.addWidget(cqt_plot_widget_bottom)
            self.current_plot_widget = cqt_plot_widget_top

        else:
            spec_top = np.abs(librosa.stft(y=recorded_signal[0], n_fft=n_fft, hop_length=hop_length, window=window_func))
            spec_dB_top = librosa.amplitude_to_db(spec_top, ref=20e-6)
            times_top = librosa.times_like(spec_dB_top, sr=sample_rate, hop_length=hop_length)

            spec_bottom = np.abs(librosa.stft(y=recorded_signal[1], n_fft=n_fft, hop_length=hop_length, window=window_func))
            spec_dB_bottom = librosa.amplitude_to_db(spec_bottom, ref=20e-6)
            times_bottom = librosa.times_like(spec_dB_bottom, sr=sample_rate, hop_length=hop_length)

            freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

            if self.stft_plot_widget_top is None or self.img_item_top is None:
                self._init_stft_plot_components()

            self.img_item_top.setImage(spec_dB_top.T)
            self.img_item_bottom.setImage(spec_dB_bottom.T)

            times_top_min, times_top_max = times_top.min(), times_top.max()
            width_top = times_top_max - times_top_min
            times_bottom_min, times_bottom_max = times_bottom.min(), times_bottom.max()
            width_bottom = times_bottom_max - times_bottom_min
            freqs_min, freqs_max = freqs.min(), freqs.max()
            height = freqs_max - freqs_min

            self.img_item_top.setRect(pg.QtCore.QRectF(times_top_min, freqs_min, width_top, height))
            self.img_item_bottom.setRect(pg.QtCore.QRectF(times_bottom_min, freqs_min, width_bottom, height))

            self.stft_plot_widget_top.setTitle("Spectrogram (Linear Scale)(Channel_1)")
            self.stft_plot_widget_top.setLabel("bottom", "Time (s)")
            self.stft_plot_widget_top.setLabel("left", "Frequency (Hz)")
            self.stft_plot_widget_top.setLogMode(x=False, y=False)

            self.stft_plot_widget_bottom.setTitle("Spectrogram (Linear Scale)(Channel_2)")
            self.stft_plot_widget_bottom.setLabel("bottom", "Time (s)")
            self.stft_plot_widget_bottom.setLabel("left", "Frequency (Hz)")
            self.stft_plot_widget_bottom.setLogMode(x=False, y=False)

            pos = np.linspace(0.0, 1.0, 256)

            colors = pg.colormap.get(color_map).getLookupTable(nPts=256)
            cmap = pg.ColorMap(pos, colors)
            db_top_min, db_top_max = np.nanmin(spec_dB_top), np.nanmax(spec_dB_top)
            db_bottom_min, db_bottom_max = np.nanmin(spec_dB_bottom), np.nanmax(spec_dB_bottom)

            lut = cmap.getLookupTable(nPts=256)
            self.img_item_top.setLookupTable(lut)
            self.img_item_top.setLevels([db_top_min, db_top_max])
            self.img_item_bottom.setLevels([db_bottom_min, db_bottom_max])

            view_box_top = self.stft_plot_widget_top.getViewBox()
            if view_box_top:
                view_box_top.setDefaultPadding(0.0)

            view_box_bottom = self.stft_plot_widget_bottom.getViewBox()
            if view_box_bottom:
                view_box_bottom.setDefaultPadding(0.0)

            self.stft_plot_widget_top.setXRange(times_top_min, times_top_max, padding=0)
            self.stft_plot_widget_top.setYRange(freqs_min, freqs_max, padding=0)
            self.stft_plot_widget_bottom.setXRange(times_bottom_min, times_bottom_max, padding=0)
            self.stft_plot_widget_bottom.setYRange(freqs_min, freqs_max, padding=0)

            plot_item_top = self.stft_plot_widget_top.getPlotItem()
            if plot_item_top:
                self.stft_colorbar_top = pg.ColorBarItem(values=(db_top_min, db_top_max), width=25, colorMap=cmap)
                self.stft_colorbar_top.setImageItem(self.img_item_top, insert_in=plot_item_top)
            else:
                self.stft_colorbar_top = None

            plot_item_bottom = self.stft_plot_widget_bottom.getPlotItem()
            if plot_item_bottom:
                self.stft_colorbar_bottom = pg.ColorBarItem(values=(db_bottom_min, db_bottom_max), width=25, colorMap=cmap)
                self.stft_colorbar_bottom.setImageItem(self.img_item_bottom, insert_in=plot_item_bottom)
            else:
                self.stft_colorbar_bottom = None

            self.plot_container_layout.addWidget(self.stft_plot_widget_top)
            self.plot_container_layout.addWidget(self.stft_plot_widget_bottom)
            self.current_plot_widget = self.stft_plot_widget_top

        if custom_limit_flag:
            self.stft_colorbar_top.setLevels((min_value, max_value))
            self.stft_colorbar_bottom.setLevels((min_value, max_value))

        self.set_color_font_size()


class LooseParticle(AnalysisGraphWidget):
    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.analysis_plot_bottom.close()
        self.result = None
        self.analysis_config = None
        self.lp_num_label = QLabel("LP 数量: %s" % self.result)
        self.status_label = QLabel()
        self.threshould = None
        self.setWindowTitle(title_name)
        self.add_label_to_layout()
        self.setStyleSheet("font-size: 20px;")

    def add_label_to_layout(self):
        lp_num_layout = QHBoxLayout()
        lp_num_layout.addStretch()
        lp_num_layout.addWidget(self.status_label)
        lp_num_layout.addWidget(self.lp_num_label)
        lp_num_layout.setSpacing(20)

        self.layout().insertLayout(0, lp_num_layout)

    def calculate_loose_particle(self):
        recorded_signal = self.data_struct.store_wave_data[0]
        filtered_spl, deviation = AudioThdFrequencyResponseAnalysis.calculate_loose_particle_spl(
            recorded_signal, self.analysis_config.get("cutoff_freq"), self.data_struct.sample_rate, 67
        )
        self.plot_graph(filtered_spl, deviation)
        self.lp_num_label.setText("LP 数量: %s" % self.result)
        if self.result > self.analysis_config.get("loose_particle_num"):
            self.status_label.setText("状态: 异常")
        else:
            self.status_label.setText("状态: 正常")

    def plot_graph(self, amplitude, deviation):
        signal_duration = np.linspace(0, len(amplitude) / (self.data_struct.sample_rate), len(amplitude))
        self.result = self.detect_peaks(
            amplitude,
            self.analysis_config.get("trigger_threshold"),
            self.analysis_config.get("hysterests_threshold"),
            self.analysis_config.get("min_check_duration"),
            self.analysis_config.get("max_check_duration"),
            self.data_struct.sample_rate,
        )
        amplitude = amplitude - deviation
        self.analysis_plot_top.plot(signal_duration, amplitude, pen=mkPen(color=(51, 196, 77)))
        self.plot_loose_particle_waveform(self.threshould, signal_duration, deviation)
        self.analysis_plot_top.setLabel("left", "Amplitude (dB)")
        self.analysis_plot_top.setLabel("bottom", "Time (s)")
        self.analysis_plot_top.showGrid(x=True, y=True)

    def detect_peaks(
        self, filtered_db, max_threshold, hysterests_threshold, min_check_duration, max_check_duration, sampling_rate
    ):
        num = 0
        peaks, _ = find_peaks(filtered_db, max_threshold)
        first_iteration = True
        last_peak = None
        current_out_range = []
        self.threshould = np.full_like(
            filtered_db, self.analysis_config.get("trigger_threshold") - float(hysterests_threshold) / 2
        )
        for peak in peaks:
            if first_iteration:
                first_iteration = False
            else:
                if (peak - last_peak) * 4 * 1000 / sampling_rate < min_check_duration or end_index > peak:
                    continue
            start_index = peak
            last_peak = peak
            end_index = start_index
            iterator_index_flag = False
            while end_index + 1 < len(filtered_db) and filtered_db[end_index] >= max_threshold - hysterests_threshold:
                if filtered_db[end_index] < max_threshold:
                    iterator_index_flag = True
                elif iterator_index_flag and filtered_db[end_index] > max_threshold:
                    current_out_range = []
                    iterator_index_flag = False
                    break
                current_out_range.append((end_index, filtered_db[end_index]))
                end_index += 1

            peak_duration = (end_index - start_index + 1) / (sampling_rate // 2)
            if peak_duration * 1000 >= min_check_duration and peak_duration * 1000 < max_check_duration:
                if filtered_db[end_index] > max_threshold - hysterests_threshold:
                    current_out_range = []
                    continue
                if current_out_range:
                    self.get_peak_duration(current_out_range, max_threshold, hysterests_threshold)
                num += 1
            current_out_range = []
        return num

    def get_peak_duration(self, signal_duration, max_threshold, hysteresis):
        for key, value in signal_duration:
            if value < max_threshold - float(hysteresis) / 2:
                self.threshould[key] = max_threshold - hysteresis
            else:
                self.threshould[key] = value

    def plot_loose_particle_waveform(self, out_range_points, signal_duration, deviation):
        pen = pg.mkPen(color="orange", width=3)
        out_range_points = np.array(out_range_points) - deviation
        out_range_plot = pg.PlotDataItem(signal_duration, out_range_points, pen=pen)
        self.analysis_plot_top.addItem(out_range_plot)


class PeakDetection(AnalysisGraphWidget):
    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.analysis_plot_bottom.close()
        self.analysis_config = None
        self.result = None
        self.deviation_value = None
        self.setWindowTitle(title_name)

        # top status bar
        self.status_label = QLabel()
        self.PD_num_label = QLabel("PD 数量: -")
        pd_num_layout = QHBoxLayout()
        pd_num_layout.addStretch()
        pd_num_layout.addWidget(self.status_label)
        pd_num_layout.addWidget(self.PD_num_label)
        pd_num_layout.setSpacing(20)
        self.layout().insertLayout(0, pd_num_layout)

        self.setStyleSheet("font-size: 16px;")

    def _update_fonts(self):
        # only adjust the font size of the upper time series plot
        self.set_plot_font_size(20)
    
    def _is_spectral_flux_only_mode(self):
        """
        Check if only spectral flux is enabled for peak detection
        """
        ch_configs = self.analysis_config.get("channels", {})
        config_ch1 = ch_configs.get("channel_1", {})
        
        # Check if other main detection methods are disabled
        peak_size_enabled = config_ch1.get("peak_size_enabled", True)
        peak_slope_enabled = config_ch1.get("peak_slope_enabled", False)
        duration_enabled = config_ch1.get("duration_enabled", False)
        
        # If peak size is disabled or has very low/no threshold, consider this spectral flux only mode
        if not peak_size_enabled:
            return True
            
        # Check if threshold is effectively disabled (very low value)
        peak_min_value = float(config_ch1.get("peak_min_value", 0.0))
        if peak_min_value <= 0.0:
            return True
            
        return False
    
    def _spectral_flux_peak_detection(self, audio_signal, sample_rate, threshold, window_s=0.1, n_fft=1024, hop_length=128):
        """
        Perform peak detection directly on spectral flux
        """
        try:
            # Compute spectral flux
            flux, flux_times = AudioFeatureExtraction.spectral_flux(
                audio_signal, sample_rate, n_fft=n_fft, hop_length=hop_length
            )
            
            if flux is None or flux.size == 0:
                return []
            
            # Find peaks in spectral flux
            min_distance_samples = max(1, int(window_s * sample_rate / hop_length))
            peaks_flux_idx, _ = find_peaks(flux, height=threshold, distance=min_distance_samples)
            
            # Convert flux peak indices to audio sample indices
            if len(peaks_flux_idx) == 0:
                return []
                
            # Map flux time indices back to audio sample indices
            peak_times = flux_times[peaks_flux_idx]
            peak_sample_indices = (peak_times * sample_rate).astype(int)
            
            # Ensure indices are within bounds
            peak_sample_indices = np.clip(peak_sample_indices, 0, len(audio_signal) - 1)
            
            return peak_sample_indices.tolist()
            
        except Exception:
            # If spectral flux fails, return empty list
            return []

    def _merge_peak_results(self, result_ch1, result_ch2, config):
        peaks1_indices = np.array(result_ch1.get("peaks_index", []))
        peaks2_indices = np.array(result_ch2.get("peaks_index", []))

        sample_rate = self.data_struct.sample_rate
        dual_channel_window_ms = config.get("dual_channel_window_ms", 20)
        matching_window_samples = int(dual_channel_window_ms * sample_rate / 1000.0)

        # 双通道峰值匹配：只有两个通道都检测到峰值时才算有效峰值
        # 单通道的峰值被视为噪声并删除
        merged_indices = []
        matched_ch1_indices = []
        matched_ch2_indices = []
        matched_ch1_times = []
        matched_ch2_times = []
        
        if len(peaks1_indices) > 0 and len(peaks2_indices) > 0:
            # 记录已匹配的通道2峰值索引，避免重复匹配
            used_ch2_indices = set()
            
            for p1 in peaks1_indices:
                distances = np.abs(peaks2_indices - p1)
                min_distance_idx = np.argmin(distances)
                min_distance = distances[min_distance_idx]
                
                # 只保留在时间窗口内且未被匹配的峰值对
                if min_distance <= matching_window_samples and min_distance_idx not in used_ch2_indices:
                    p2 = peaks2_indices[min_distance_idx]
                    # 使用两个峰值的平均位置作为最终峰值位置
                    merged_indices.append(int((p1 + p2) / 2))
                    # 保存原始通道峰值信息
                    matched_ch1_indices.append(int(p1))
                    matched_ch2_indices.append(int(p2))
                    matched_ch1_times.append(float(p1) / sample_rate)
                    matched_ch2_times.append(float(p2) / sample_rate)
                    used_ch2_indices.add(min_distance_idx)
                    
        # 去重并排序
        merged_indices = sorted(list(set(merged_indices)))

        merged_times_sec = (np.array(merged_indices) / sample_rate).tolist()

        return {
            "peaks_index": merged_indices,
            "peaks_time_sec": merged_times_sec,
            "num_peaks": len(merged_indices),
            "spl_db_series": result_ch1.get("spl_db_series"), # keep ch1's spl series for plotting
            # 新增：保存双通道原始峰值信息和SPL数据
            "dual_channel_peaks": {
                "channel_1": {
                    "peaks_index": matched_ch1_indices,
                    "peaks_time_sec": matched_ch1_times,
                    "all_peaks_index": peaks1_indices.tolist(),
                    "all_peaks_time_sec": (peaks1_indices / sample_rate).tolist(),
                    "spl_db_series": result_ch1.get("spl_db_series", [])  # RMS-windowed SPL data
                },
                "channel_2": {
                    "peaks_index": matched_ch2_indices,
                    "peaks_time_sec": matched_ch2_times,
                    "all_peaks_index": peaks2_indices.tolist(),
                    "all_peaks_time_sec": (peaks2_indices / sample_rate).tolist(),
                    "spl_db_series": result_ch2.get("spl_db_series", [])  # RMS-windowed SPL data
                }
            }
        }


    def calculate_peak_detection(self):
        """
        calculate and plot PD analysis: the upper plot is SPL time series with peak annotation;
        """
        recorded_signal_all_channels = self.data_struct.store_wave_data
        sample_rate = self.data_struct.sample_rate
        if recorded_signal_all_channels is None or len(recorded_signal_all_channels) == 0 or sample_rate is None:
            return None

        # two channels by default
        recorded_signal_ch1 = recorded_signal_all_channels[0]
        recorded_signal_ch2 = recorded_signal_all_channels[1] if len(recorded_signal_all_channels) > 1 else None

        ch_configs = self.analysis_config.get("channels", {})
        config_ch1 = ch_configs.get("channel_1", {})
        config_ch2 = ch_configs.get("channel_2", {})

        result_ch1, result_ch2 = None, None
        try:
            if recorded_signal_ch1 is not None:
                result_ch1 = peak_detection(
                    np.asarray(recorded_signal_ch1, dtype=np.float64),
                    int(sample_rate),
                    config_ch1,
                    deviation=self.deviation_value,
                )
            if recorded_signal_ch2 is not None:
                result_ch2 = peak_detection(
                    np.asarray(recorded_signal_ch2, dtype=np.float64),
                    int(sample_rate),
                    config_ch2,
                    deviation=self.deviation_value,
                )
        except Exception as e:
            self.status_label.setText(f"状态: 异常({e.__class__.__name__})")
            self.PD_num_label.setText("PD 数量: -")
            self.analysis_plot_top.clear()
            return None

        # Apply spectral flux filtering/detection if configured
        specflux_min = float(self.analysis_config.get("specflux_min_value", 0.0))
        window_s = 0.1
        
        # Check if spectral flux should be used as primary or secondary detection method
        use_spectral_flux_primary = specflux_min > 0.0 and self._is_spectral_flux_only_mode()
        
        if use_spectral_flux_primary:
            # Use spectral flux as the primary peak detection method
            if result_ch1:
                pidx1_f = self._spectral_flux_peak_detection(
                    recorded_signal_ch1, sample_rate, specflux_min, window_s
                )
                result_ch1["peaks_index"] = pidx1_f
                result_ch1["peaks_time_sec"] = (np.array(pidx1_f, dtype=float) / float(sample_rate)).tolist()
                result_ch1["num_peaks"] = len(pidx1_f)
                
            if result_ch2:
                pidx2_f = self._spectral_flux_peak_detection(
                    recorded_signal_ch2, sample_rate, specflux_min, window_s
                )
                result_ch2["peaks_index"] = pidx2_f
                result_ch2["peaks_time_sec"] = (np.array(pidx2_f, dtype=float) / float(sample_rate)).tolist()
                result_ch2["num_peaks"] = len(pidx2_f)
        else:
            # Use spectral flux as a filter on existing peaks
            if result_ch1 and specflux_min > 0.0:
                pidx1 = result_ch1.get("peaks_index", [])
                pidx1_f = filter_peaks_by_spectral_flux(
                    pidx1, sample_rate, recorded_signal_ch1, 
                    window_s=window_s, threshold=specflux_min, 
                    n_fft=1024, hop_length=128
                )
                result_ch1["peaks_index"] = pidx1_f
                result_ch1["peaks_time_sec"] = (np.array(pidx1_f, dtype=float) / float(sample_rate)).tolist()
                result_ch1["num_peaks"] = len(pidx1_f)
                
            if result_ch2 and specflux_min > 0.0:
                pidx2 = result_ch2.get("peaks_index", [])
                pidx2_f = filter_peaks_by_spectral_flux(
                    pidx2, sample_rate, recorded_signal_ch2, 
                    window_s=window_s, threshold=specflux_min, 
                    n_fft=1024, hop_length=128
                )
                result_ch2["peaks_index"] = pidx2_f
                result_ch2["peaks_time_sec"] = (np.array(pidx2_f, dtype=float) / float(sample_rate)).tolist()
                result_ch2["num_peaks"] = len(pidx2_f)

        merged_results = self._merge_peak_results(result_ch1, result_ch2, self.analysis_config)
        self.result = merged_results

        # save the grid points (sample point indices) corresponding to the peaks
        peak_indices = self.result.get("peaks_index", []) if isinstance(self.result, dict) else []
        indices_list = [int(i) for i in peak_indices] if len(peak_indices) > 0 else []
        analysis_key = self.windowTitle()
        self.data_struct.pd_peak_grid_points_map[analysis_key] = indices_list

        # --- Plotting ---
        self.analysis_plot_top.clear()
        self.analysis_plot_top.getPlotItem().addLegend()

        # 1. & 2. Get/Calculate and Plot SPL series for both channels
        spl_series_ch1 = []
        if result_ch1:
            spl_series_ch1 = np.asarray(result_ch1.get("spl_db_series", []), dtype=float)
        if len(spl_series_ch1) == 0 and recorded_signal_ch1 is not None and len(recorded_signal_ch1) > 0:
            ref_p = 20e-6
            spl_series_ch1 = 20.0 * np.log10(np.maximum(np.abs(recorded_signal_ch1), 1e-30) / ref_p)
            if self.deviation_value is not None:
                spl_series_ch1 = spl_series_ch1 + float(self.deviation_value)

        spl_series_ch2 = []
        if result_ch2:
            spl_series_ch2 = np.asarray(result_ch2.get("spl_db_series", []), dtype=float)
        if len(spl_series_ch2) == 0 and recorded_signal_ch2 is not None and len(recorded_signal_ch2) > 0:
            ref_p = 20e-6
            spl_series_ch2 = 20.0 * np.log10(np.maximum(np.abs(recorded_signal_ch2), 1e-30) / ref_p)
            if self.deviation_value is not None:
                spl_series_ch2 = spl_series_ch2 + float(self.deviation_value)

        # Plot if data exists
        if len(spl_series_ch1) > 0:
            time_axis = np.linspace(0, len(spl_series_ch1) / sample_rate, len(spl_series_ch1))
            self.analysis_plot_top.plot(time_axis, spl_series_ch1, pen=mkPen(color='g', width=1), name='Channel 1')

        if len(spl_series_ch2) > 0:
            time_axis_ch2 = np.linspace(0, len(spl_series_ch2) / sample_rate, len(spl_series_ch2))
            self.analysis_plot_top.plot(time_axis_ch2, spl_series_ch2, pen=mkPen(color='b', width=1), name='Channel 2')

        # 3. Plot original peaks as vertical lines
        if result_ch1:
            pen_ch1_peak = mkPen(color=(0, 255, 0, 150), style=Qt.DashLine)
            for peak_time in result_ch1.get("peaks_time_sec", []):
                line = pg.InfiniteLine(angle=90, movable=False, pos=peak_time, pen=pen_ch1_peak)
                self.analysis_plot_top.addItem(line)

        if result_ch2:
            pen_ch2_peak = mkPen(color=(0, 0, 255, 150), style=Qt.DashLine)
            for peak_time in result_ch2.get("peaks_time_sec", []):
                line = pg.InfiniteLine(angle=90, movable=False, pos=peak_time, pen=pen_ch2_peak)
                self.analysis_plot_top.addItem(line)

        # 4. Highlight merged peaks with scatter plot
        merged_peak_times = self.result.get("peaks_time_sec", [])
        if merged_peak_times and (len(spl_series_ch1) > 0 or len(spl_series_ch2) > 0):
            max_len = 0
            if len(spl_series_ch1) > 0:
                max_len = len(spl_series_ch1)
            if len(spl_series_ch2) > 0:
                max_len = max(max_len, len(spl_series_ch2))

            merged_peak_indices = np.clip((np.array(merged_peak_times) * sample_rate).astype(int), 0, max_len - 1)
            
            peak_values_ch1 = spl_series_ch1[merged_peak_indices] if len(spl_series_ch1) > 0 else np.zeros_like(merged_peak_indices, dtype=float)
            peak_values_ch2 = spl_series_ch2[merged_peak_indices] if len(spl_series_ch2) > 0 else np.zeros_like(merged_peak_indices, dtype=float)
            peak_values = np.maximum(peak_values_ch1, peak_values_ch2)

            scatter = pg.ScatterPlotItem(
                x=np.array(merged_peak_times), y=peak_values, pen=pg.mkPen(None), brush=pg.mkBrush(200, 0, 0, 200), size=10, name="Merged Peak"
            )
            self.analysis_plot_top.addItem(scatter)

        # Clear bottom plot (spectral flux plotting removed)
        try:
            if self.analysis_plot_bottom is not None:
                self.analysis_plot_bottom.clear()
        except:
            pass
        
        self.analysis_plot_top.setLabel("left", "SPL (dB)")
        self.analysis_plot_top.setLabel("bottom", "Time (s)")
        self.analysis_plot_top.showGrid(x=True, y=True)

        # Update the number and status based on merged results
        num_peaks = int(self.result.get("num_peaks", 0))
        final_peak_min = self.analysis_config.get("final_test_peak_min", 0)
        final_peak_max = self.analysis_config.get("final_test_peak_max", 1000000)
        passed = final_peak_min <= num_peaks <= final_peak_max

        self.PD_num_label.setText(f"PD 数量: {num_peaks}")
        self.status_label.setText("状态: 正常" if passed else "状态: 异常")

        # self._update_fonts()
        return self.result


class PatternMatch(QWidget):
    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.target_data = self.data_struct.store_wave_data
        self.analysis_config = None
        self.sample_rate = self.data_struct.sample_rate
        self.init_ui()
        self.setWindowTitle(title_name)

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.main_layout = QVBoxLayout(self)
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)

        self.main_layout.addWidget(self.result_display)
        self.setLayout(self.main_layout)
        self.setStyleSheet(
            ui_style_const.qlabel_style
            + ui_style_const.qlineedit_style
            + ui_style_const.qtextedit_style
        )

    def calculate_pattern_match(self, target_data=None, peak=None, analysis_config=None):
        if target_data is not None:
            self.target_data = target_data
        if analysis_config is not None:
            self.analysis_config = analysis_config
        if self.target_data is None or self.analysis_config is None:
            self.result_display.setText("错误：数据或配置不完整")
            return None
        if self.target_data.ndim != 2 or self.target_data.shape[0] != 2:
            self.result_display.setText("错误：目标数据必须是双通道")
            return None
        config_ch1 = self.analysis_config.get("channel_1_config")
        config_ch2 = self.analysis_config.get("channel_2_config")
        if not config_ch1 or not config_ch2:
            self.result_display.setText("错误：配置不完整，缺少通道配置")
            return None

        # Add auto_equal_length configuration to channel configs
        auto_equal_length = self.analysis_config.get("auto_equal_length", False)
        config_ch1 = config_ch1.copy()
        config_ch2 = config_ch2.copy()
        config_ch1["auto_equal_length"] = auto_equal_length
        config_ch2["auto_equal_length"] = auto_equal_length
        
        # Add external peak parameter if provided
        if peak is not None:
            config_ch1["peak"] = peak
            config_ch2["peak"] = peak

        target_ch1, target_ch2 = self.target_data[0], self.target_data[1]

        results_ch1 = self.run_matching_for_channel(target_ch1, config_ch1)
        results_ch2 = self.run_matching_for_channel(target_ch2, config_ch2)

        final_success = results_ch1["success"] and results_ch2["success"]

        ch1_score_str = f"{results_ch1['score'] * 100:.2f}" if results_ch1['score'] is not None else "N/A"
        ch2_score_str = f"{results_ch2['score'] * 100:.2f}" if results_ch2['score'] is not None else "N/A"
        ch1_status_str = '成功' if results_ch1["success"] else '失败'
        ch2_status_str = '成功' if results_ch2["success"] else '失败'
        ch1_threshold_str = f"{results_ch1.get('threshold', 'N/A'):.2f}" if results_ch1.get(
            'threshold') is not None else "N/A"
        ch2_threshold_str = f"{results_ch2.get('threshold', 'N/A'):.2f}" if results_ch2.get(
            'threshold') is not None else "N/A"

        final_text = f"最终匹配结果: {'成功' if final_success else '失败'}\n"
        final_text += "------------------------------------\n"
        final_text += f"通道 1 -> 得分: {ch1_score_str}, 状态: {ch1_status_str}, 阈值: {ch1_threshold_str}\n"
        final_text += f"通道 2 -> 得分: {ch2_score_str}, 状态: {ch2_status_str}, 阈值: {ch2_threshold_str}"

        self.result_display.setPlainText(final_text)

        return {
            "final_success": final_success,
            "channel_1": results_ch1,
            "channel_2": results_ch2,
        }

    def run_matching_for_channel(self, target_data, channel_config):

        pattern_list = channel_config.get("pattern_list", [])

        if not pattern_list:
            return {"score": None, "success": False, "threshold": None}

        threshold_strategy = channel_config.get("threshold_strategy", "fixed_threshold")

        all_scores = []
        for pattern_info in pattern_list:
            pattern_path = os.path.join(DEFAULT_DIR, pattern_info["clip_path"])
            if not os.path.exists(pattern_path):
                continue

            pattern_data, _ = load_audio_simple(pattern_path, self.sample_rate)
            result = self.process_single_channel(target_data, pattern_data, channel_config)

            if result and result['score'] is not None:
                all_scores.append(result['score'])

        if not all_scores:
            return {"score": None, "success": False, "threshold": None}

        best_score = np.max(all_scores)
        channel_is_successful = False
        if threshold_strategy == "fixed_threshold":
            threshold = channel_config.get("threshold_value", 50.0) / 100.0
            channel_is_successful = best_score >= threshold

        elif threshold_strategy == "adaptive_threshold":
            if len(all_scores) < 3:
                threshold = 0.5
                channel_is_successful = best_score >= threshold
            else:
                scores_array = np.array(all_scores)
                best_score_idx = np.argmax(scores_array)
                background_scores = np.delete(scores_array, best_score_idx)
                mean_bg = np.mean(background_scores)
                std_bg = np.std(background_scores)
                k = 3.0
                threshold = mean_bg + k * std_bg
                channel_is_successful = best_score >= threshold

        return {"score": best_score,
                "success": channel_is_successful,
                "threshold": threshold * 100
                }


    def process_single_channel(self, target_data, pattern_data, config):
        # Check if external peak parameter is provided
        peak = config.get("peak")
        
        if config.get("apply_filter", False):
            start_freq, end_freq = config.get("filter_range_hz", [None, None])
            if start_freq is not None and end_freq is not None:
                target_data = AudioEqualizer.apply_equalizer(target_data, self.sample_rate, start_freq=start_freq,
                                                             end_freq=end_freq)
                pattern_data = AudioEqualizer.apply_equalizer(pattern_data, self.sample_rate, start_freq=start_freq,
                                                              end_freq=end_freq)

        # Handle auto equal length peak alignment
        auto_equal = config.get("auto_equal_length", False)
        if auto_equal:
            # If external peak is provided, use it; otherwise calculate peak alignment
            if peak is not None:
                target_data, pattern_data = align_signals_with_peak(target_data, pattern_data, peak)
            else:
                target_data, pattern_data = align_signals_by_peaks(target_data, pattern_data)

        target_features, pattern_features = self.extract_features(target_data, pattern_data, config)
        if target_features is None or pattern_features is None:
            return None

        result = self.execute_match_algorithm(target_features, pattern_features, config)
        return result


    @staticmethod
    def execute_match_algorithm(target_features, pattern_features, config):
        algorithm = config.get("algorithm", "distance")
        params = config.get("algorithm_params", {})
        dispatcher = {
            "distance": PatternMatch.match_distance,
            "dtw": PatternMatch.match_dtw,
            "lbp": PatternMatch.match_lbp,
            "ncc": PatternMatch.match_ncc,
        }
        if algorithm not in dispatcher:
            return None

        score_value = dispatcher[algorithm](target_features, pattern_features, params)
        return {"score": score_value}


    @staticmethod
    def match_distance(target, pattern, params):
        metric = params.get("metric", "euclidean")
        target, pattern = target.flatten(), pattern.flatten()
        if len(target) != len(pattern):
            min_len = min(len(target), len(pattern))
            target, pattern = target[:min_len], pattern[:min_len]
        metric_map = {"euclidean": euclidean, "cosine": cosine, "manhattan": cityblock}
        if metric not in metric_map:
            return None

        distance = metric_map[metric](target, pattern)

        if metric == 'cosine':
            score = 1 - distance
            score = (score + 1) / 2
        else:
            score = 1 / (1 + distance)
        return score

    @staticmethod
    def match_dtw(target, pattern, params):
        metric = params.get("metric", "euclidean")
        normalization = params.get("normalization", "none")
        global_constraints = params.get("global_constraints", False)
        band_rad = params.get("band_rad", 0.25)
        try:
            D, wp = dtw(X=target, Y=pattern, metric=metric, backtrack=True, global_constraints=global_constraints,
                        band_rad=band_rad)
            distance = D[-1, -1]
            if normalization == 'path_length':
                if len(wp) > 0:
                    distance /= len(wp)
            elif normalization == 'sum_length':
                total_len = target.shape[1] + pattern.shape[1]
                if total_len > 0:
                    distance /= total_len

            score = 1 / (1 + distance)
            return score
        except Exception as e:
            return None

    @staticmethod
    def match_lbp(target_spec, pattern_spec, params):
        try:
            target_hist = PatternMatch.extract_lbp_histogram(target_spec, params)
            pattern_hist = PatternMatch.extract_lbp_histogram(pattern_spec, params)
        except Exception as e:
            return None

        metric = params.get("metric", "chi2")

        if metric in ['chi2', 'euclidean']:
            distance = euclidean(target_hist, pattern_hist) if metric == 'euclidean' else PatternMatch.hist_compare_chi2(
                target_hist, pattern_hist)
            score = 1 / (1 + distance)
        elif metric in ['intersection', 'correlation', 'bhattacharyya']:
            if metric == 'intersection':
                score = PatternMatch.hist_compare_intersection(target_hist, pattern_hist)
            elif metric == 'correlation':
                score = PatternMatch.hist_compare_correlation(target_hist, pattern_hist)
                score = (score + 1) / 2
            else:
                score = PatternMatch.hist_compare_bhattacharyya(target_hist, pattern_hist)
        elif metric == 'cosine':
            distance = cosine(target_hist, pattern_hist)
            score = 1 - distance
            score = (score + 1) / 2
        else:
            return None

        return score

    @staticmethod
    def match_ncc(target, pattern, params):
        try:
            if pattern.shape[0] > target.shape[0] or pattern.shape[1] > target.shape[1]:
                return None
            result = match_template(target, pattern)
            score_raw = np.max(result)
            score = (score_raw + 1) / 2
            return score
        except Exception as e:
            return None

    @staticmethod
    def extract_lbp_histogram(features, params):
        n_points = params.get("n_points", 8)
        radius = params.get("radius", 1)
        method = params.get("method", "uniform")

        feature_normal = (features - features.min()) / (features.max() - features.min())
        lbp_feature = local_binary_pattern(feature_normal, P=n_points, R=radius, method=method)
        if method == 'uniform':
            n_bins = n_points + 2
        elif method == 'nri_uniform':
            n_bins = n_points * (n_points - 1) + 3
        else:
            n_bins = 2 ** n_points

        hist, _ = np.histogram(lbp_feature.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype("float32")
        hist /= (hist.sum() + 1e-6)
        return hist

    @staticmethod
    def extract_features(target_data, pattern_data, config):
        feature_type = config.get("feature_type", "mfcc")
        sample_rate = config.get("sample_rate")
        params = config.get("feature_params")
        convert_to_db = params.pop("power_to_db", False)
        try:
            if feature_type == "waveform":
                return target_data, pattern_data
            elif feature_type == "mfcc":
                target_f = spectral.mfcc(y=target_data, sr=sample_rate, **params)
                pattern_f = spectral.mfcc(y=pattern_data, sr=sample_rate, **params)
                return target_f, pattern_f
            elif feature_type == "fft":
                target_len = len(target_data)
                pattern_len = len(pattern_data)
                target_f = np.abs(np.fft.fft(target_data) / target_len)[:target_len // 2]
                pattern_f = np.abs(np.fft.fft(pattern_data) / pattern_len)[:pattern_len // 2]
                return target_f, pattern_f
            elif feature_type == "spec" or feature_type == "melspec":
                if feature_type == "spec":
                    target_f = np.abs(spectrum.stft(y=target_data, **params))
                    pattern_f = np.abs(spectrum.stft(y=pattern_data, **params))
                else:
                    target_f = spectral.melspectrogram(y=target_data, sr=sample_rate, **params)
                    pattern_f = spectral.melspectrogram(y=pattern_data, sr=sample_rate, **params)
                if convert_to_db:
                    target_f = librosa.amplitude_to_db(target_f, ref=np.max)
                    pattern_f = librosa.amplitude_to_db(pattern_f, ref=np.max)
                return target_f, pattern_f
            else:
                return None, None
        except Exception as e:
            return None, None

    @staticmethod
    def calculate_loocv_threshold(channel_config, sample_rate):
        SIMILARITY_SAFETY_FACTOR = 0.95

        pattern_list = channel_config.get("pattern_list", [])
        if len(pattern_list) < 2:
            return None

        all_features = []
        for pattern_info in pattern_list:
            try:
                pattern_path = os.path.join(DEFAULT_DIR, pattern_info["clip_path"])
                pattern_data, _ = load_audio_simple(pattern_path, sample_rate)

                _, features = PatternMatch.extract_features(pattern_data, pattern_data, channel_config)
                if features is None:
                    return None
                all_features.append(features)
            except Exception as e:
                return None

        internal_best_scores = []
        for i, test_features in enumerate(all_features):
            committee_features = all_features[:i] + all_features[i + 1:]

            scores_to_committee = []
            for ref_features in committee_features:
                result = PatternMatch.execute_match_algorithm(test_features, ref_features, channel_config)
                if result and result.get('score') is not None:
                    scores_to_committee.append(result['score'])

            if not scores_to_committee:
                continue

            best_score = np.max(scores_to_committee)
            internal_best_scores.append(best_score)

        if not internal_best_scores:
            return None

        min_internal_score = np.min(internal_best_scores)
        suggested_threshold_score = min_internal_score * SIMILARITY_SAFETY_FACTOR

        return suggested_threshold_score

    @staticmethod
    def hist_compare_chi2(h1, h2):
        return np.sum(np.square(h1 - h2) / (h1 + h2 + 1e-10))

    @staticmethod
    def hist_compare_intersection(h1, h2):
        h1_norm = h1 / (h1.sum() + 1e-10)
        h2_norm = h2 / (h2.sum() + 1e-10)
        return np.sum(np.minimum(h1_norm, h2_norm))

    @staticmethod
    def hist_compare_correlation(h1, h2):
        corr, _ = pearsonr(h1, h2)
        return corr if not np.isnan(corr) else 0

    @staticmethod
    def hist_compare_bhattacharyya(h1, h2):
        h1_norm = h1 / (h1.sum() + 1e-10)
        h2_norm = h2 / (h2.sum() + 1e-10)
        return np.sum(np.sqrt(h1_norm * h2_norm))



class PipelinePdPm(AnalysisGraphWidget):
    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.analysis_config = None  # structure: {"head": {...}, "tail": {...}}
        self.deviation_value = None
        self.is_default_flag = False
        self.default_logger = LogManager.set_log_handler("core")
        self.analysis_plot_bottom.close()  # Use single plot like PeakDetection
        self._init_ui()
        self.setWindowTitle(title_name)


    def _init_ui(self):
        # Configure single plot for dual-channel visualization
        self.analysis_plot_top.setBackground("white")
        self.analysis_plot_top.showGrid(x=True, y=True, alpha=0.5)
        self.analysis_plot_top.setLabel("left", "SPL (dB)")
        self.analysis_plot_top.setLabel("bottom", "Time (s)")

        # Create additional UI components
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        
        # match result table with dual-channel columns
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(7)  # Increased for dual-channel info
        self.table_widget.setHorizontalHeaderLabels([
            "序号", "时间(s)", "长度(ms)", "Ch1分数(%)", "Ch2分数(%)", "总分数(%)", "匹配状态"
        ])
        header = self.table_widget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        
        # Modify the existing layout from AnalysisGraphWidget instead of creating new one
        existing_layout = self.layout()
        if existing_layout:
            # Clear existing layout and rebuild
            while existing_layout.count():
                child = existing_layout.takeAt(0)
                if child.widget():
                    child.widget().setParent(None)
        else:
            # Create new layout if none exists
            existing_layout = QVBoxLayout(self)
        
        # Build new layout structure
        content_layout = QHBoxLayout()
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.addWidget(self.result_display)
        left_layout.addWidget(self.analysis_plot_top)  # Only use top plot
        
        content_layout.addWidget(left_container)
        content_layout.addWidget(self.table_widget)
        content_layout.setStretch(0, 3)
        content_layout.setStretch(1, 2)
        
        existing_layout.addLayout(content_layout)
        
        self.setStyleSheet(ui_style_const.qlabel_style + ui_style_const.qlineedit_style + ui_style_const.qtextedit_style)
        self.result_display.setStyleSheet("font-size:20px;")
        
        # Single plot dual-axis setup for similarity visualization
        self._right_view = None
        self._bars_item_ch1 = None
        self._bars_item_ch2 = None
        self._last_spl_series_ch1 = None
        self._last_spl_series_ch2 = None

    def _setup_dual_axis_if_needed(self):
        # Setup dual axis for single plot (similarity visualization)
        if self._right_view is None:
            plot_item = self.analysis_plot_top.getPlotItem()
            plot_item.showAxis("right")
            right_axis = plot_item.getAxis("right")
            right_axis.setLabel("相似度")
            self._right_view = pg.ViewBox()
            self._right_view.setXLink(plot_item.vb)
            plot_item.scene().addItem(self._right_view)
            right_axis.linkToView(self._right_view)

            def _update_views_geometry():
                self._right_view.setGeometry(plot_item.vb.sceneBoundingRect())
                self._right_view.linkedViewChanged(plot_item.vb, self._right_view.XAxis)

            plot_item.vb.sigResized.connect(_update_views_geometry)
            _update_views_geometry()

    def _prepare_pipeline_context(self):
        """
        prepare the input data for the pipeline, including the recorded signal, sample rate, and the analysis config
        """
        recorded_signal = self.data_struct.store_wave_data
        sample_rate = self.data_struct.sample_rate
        if recorded_signal is None or sample_rate is None:
            return None

        cfg = self.analysis_config or {}
        head = cfg.get("head", {}) or {}
        tail = cfg.get("tail", {}) or {}
        if not head or not tail:
            return None

        class_mapping = get_class_mapping()
        pd_cls = class_mapping.get("PD")
        pm_cls = class_mapping.get("PM")
        if not pd_cls or not pm_cls:
            return None

        return recorded_signal, sample_rate, cfg, head, tail, pd_cls, pm_cls

    def _validate_dual_channel_pm_config(self, pm_cfg):
        """
        Validate that the pattern matching configuration has proper dual-channel structure
        """
        if not isinstance(pm_cfg, dict):
            return False
        
        # Check for required dual-channel configuration keys
        channel_1_config = pm_cfg.get("channel_1_config")
        channel_2_config = pm_cfg.get("channel_2_config")
        
        if not channel_1_config or not channel_2_config:
            self.default_logger.warning("PM config missing channel_1_config or channel_2_config")
            return False
        
        # Validate that each channel config has required fields
        for ch_name, ch_config in [("channel_1", channel_1_config), ("channel_2", channel_2_config)]:
            if not isinstance(ch_config, dict):
                self.default_logger.warning(f"PM config {ch_name}_config is not a dictionary")
                return False
            
            # Check for pattern list
            pattern_list = ch_config.get("pattern_list", [])
            if not isinstance(pattern_list, list) or len(pattern_list) == 0:
                self.default_logger.warning(f"PM config {ch_name}_config missing or empty pattern_list")
                return False
            
            # Check for threshold strategy  
            threshold_strategy = ch_config.get("threshold_strategy")
            if threshold_strategy not in ["fixed_threshold", "adaptive_threshold"]:
                self.default_logger.warning(f"PM config {ch_name}_config has invalid threshold_strategy: {threshold_strategy}")
                return False
        
        return True

    def _execute_pd(self, pd_cls, head_cfg):
        pd_instance = pd_cls(f"{self.windowTitle()}-PD")
        pd_instance.data_struct = self.data_struct
        pd_instance.deviation_value = self.deviation_value
        pd_instance.analysis_config = head_cfg.get("config", {})
        pd_result = pd_instance.calculate_peak_detection()

        peak_indices = []
        if isinstance(pd_result, dict):
            peak_indices = [int(i) for i in pd_result.get("peaks_index", [])]
        return pd_result, peak_indices


    def _execute_pm(self, pm_cls, sample_rate, recorded_signal_all_channels, peak_indices, pm_cfg):
        # Handle multi-channel data
        if len(recorded_signal_all_channels) < 2:
            return []
        
        recorded_signal_ch1 = recorded_signal_all_channels[0]
        recorded_signal_ch2 = recorded_signal_all_channels[1]
        
        pm_instance = pm_cls(f"{self.windowTitle()}-PM")
        pm_instance.sample_rate = sample_rate

        # Pass full dual-channel data to PatternMatch
        dual_channel_data = np.array([recorded_signal_ch1, recorded_signal_ch2])

        results = []
        for pk in peak_indices:
            peak_index = int(pk)
            
            # Pass full signal data + peak index to PatternMatch for alignment
            result_dict = pm_instance.calculate_pattern_match(
                target_data=dual_channel_data, 
                peak=peak_index,
                analysis_config=pm_cfg
            )

            if result_dict:
                # Handle dual-channel results structure
                final_success = result_dict.get("final_success", False)
                ch1_result = result_dict.get("channel_1", {})
                ch2_result = result_dict.get("channel_2", {})
                
                # Calculate combined score (average of both channels)
                ch1_score = ch1_result.get("score", 0.0) or 0.0
                ch2_score = ch2_result.get("score", 0.0) or 0.0
                combined_score = (ch1_score + ch2_score) / 2.0
                
                results.append(
                    {
                        "peak_index": peak_index,
                        "time_sec": peak_index / float(sample_rate),
                        "is_match": bool(final_success),
                        "score": float(combined_score),
                        "threshold": float(ch1_result.get("threshold", 0.0)),  # Use ch1 threshold as reference
                        # Add dual-channel specific information
                        "dual_channel_results": {
                            "channel_1": {
                                "score": float(ch1_score),
                                "success": bool(ch1_result.get("success", False)),
                                "threshold": float(ch1_result.get("threshold", 0.0))
                            },
                            "channel_2": {
                                "score": float(ch2_score), 
                                "success": bool(ch2_result.get("success", False)),
                                "threshold": float(ch2_result.get("threshold", 0.0))
                            },
                            "final_success": bool(final_success)
                        }
                    }
                )
        return results

    def _render_plots(self, pd_result, recorded_signal, sample_rate, peak_indices, results):
        # Clear single plot
        self.analysis_plot_top.clear()
        plot_item = self.analysis_plot_top.getPlotItem()

        # Handle multi-channel data
        if not isinstance(recorded_signal, (list, np.ndarray)) or len(recorded_signal) < 2:
            return
        
        recorded_signal_ch1 = recorded_signal[0]
        recorded_signal_ch2 = recorded_signal[1]
        
        # Extract proper RMS-windowed SPL data from PeakDetection dual_channel_peaks
        spl_series_ch1 = None
        spl_series_ch2 = None
        
        if isinstance(pd_result, dict):
            dual_channel_data = pd_result.get("dual_channel_peaks", {})
            if dual_channel_data:
                # Get RMS-windowed SPL data from individual channel results
                ch1_data = dual_channel_data.get("channel_1", {})
                ch2_data = dual_channel_data.get("channel_2", {})
                
                ch1_spl = ch1_data.get("spl_db_series", [])
                ch2_spl = ch2_data.get("spl_db_series", [])
                
                if len(ch1_spl) > 0:
                    spl_series_ch1 = np.asarray(ch1_spl, dtype=float)
                if len(ch2_spl) > 0:
                    spl_series_ch2 = np.asarray(ch2_spl, dtype=float)
        
        # Fallback: Use AudioThdFrequencyResponseAnalysis for proper RMS-windowed SPL if not available
        if spl_series_ch1 is None or len(spl_series_ch1) == 0:
            spl_series_ch1 = AudioThdFrequencyResponseAnalysis().spl_calculation(
                recorded_signal_ch1, 
                reference_pressure=20e-6,
                window_size=1201,
                method="rms",
                deviation=self.deviation_value
            )
        
        if spl_series_ch2 is None or len(spl_series_ch2) == 0:
            spl_series_ch2 = AudioThdFrequencyResponseAnalysis().spl_calculation(
                recorded_signal_ch2,
                reference_pressure=20e-6, 
                window_size=1201,
                method="rms",
                deviation=self.deviation_value
            )
        
        # Create time axis
        time_axis_ch1 = np.linspace(0, len(spl_series_ch1) / sample_rate, len(spl_series_ch1))
        time_axis_ch2 = np.linspace(0, len(spl_series_ch2) / sample_rate, len(spl_series_ch2))
        
        # Plot both SPL curves on same plot (like PeakDetection)
        plot_item.plot(time_axis_ch1, spl_series_ch1, pen=mkPen(color='g', width=1), name='SPL CH1')
        plot_item.plot(time_axis_ch2, spl_series_ch2, pen=mkPen(color='b', width=1), name='SPL CH2')
        
        # Add legend for main plot
        if not hasattr(plot_item, 'legend') or plot_item.legend is None:
            plot_item.addLegend(offset=(10, 10))
        
        self._last_spl_series_ch1 = np.asarray(spl_series_ch1)
        self._last_spl_series_ch2 = np.asarray(spl_series_ch2)

        # Plot individual channel peaks if available in dual_channel_peaks
        if isinstance(pd_result, dict) and "dual_channel_peaks" in pd_result:
            dual_peaks = pd_result["dual_channel_peaks"]
            
            # Channel 1 peaks (green scatter points)
            ch1_peak_indices = dual_peaks.get("channel_1", {}).get("peaks_index", [])
            if ch1_peak_indices:
                ch1_peak_indices_arr = np.clip(np.asarray(ch1_peak_indices, dtype=int), 0, len(spl_series_ch1) - 1)
                ch1_peak_times = ch1_peak_indices_arr / float(sample_rate)
                ch1_peak_values = np.asarray(spl_series_ch1)[ch1_peak_indices_arr]
                scatter_ch1 = pg.ScatterPlotItem(
                    x=ch1_peak_times, y=ch1_peak_values, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0, 200), size=8
                )
                plot_item.addItem(scatter_ch1)
                # Add to legend
                if hasattr(plot_item, 'legend') and plot_item.legend is not None:
                    plot_item.legend.addItem(scatter_ch1, 'Peaks CH1')
            
            # Channel 2 peaks (blue scatter points)
            ch2_peak_indices = dual_peaks.get("channel_2", {}).get("peaks_index", [])
            if ch2_peak_indices:
                ch2_peak_indices_arr = np.clip(np.asarray(ch2_peak_indices, dtype=int), 0, len(spl_series_ch2) - 1)
                ch2_peak_times = ch2_peak_indices_arr / float(sample_rate)
                ch2_peak_values = np.asarray(spl_series_ch2)[ch2_peak_indices_arr]
                scatter_ch2 = pg.ScatterPlotItem(
                    x=ch2_peak_times, y=ch2_peak_values, pen=pg.mkPen(None), brush=pg.mkBrush(0, 0, 255, 200), size=8
                )
                plot_item.addItem(scatter_ch2)
                # Add to legend
                if hasattr(plot_item, 'legend') and plot_item.legend is not None:
                    plot_item.legend.addItem(scatter_ch2, 'Peaks CH2')
        elif peak_indices:
            # Fallback: use merged peak indices for both channels
            peak_indices_arr = np.clip(np.asarray(peak_indices, dtype=int), 0, min(len(spl_series_ch1), len(spl_series_ch2)) - 1)
            
            # Ch1 peaks (green)
            if len(peak_indices_arr) > 0:
                ch1_peak_times = peak_indices_arr / float(sample_rate)
                ch1_peak_values = np.asarray(spl_series_ch1)[peak_indices_arr]
                scatter_ch1 = pg.ScatterPlotItem(
                    x=ch1_peak_times, y=ch1_peak_values, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0, 200), size=8
                )
                plot_item.addItem(scatter_ch1)
                # Add to legend
                if hasattr(plot_item, 'legend') and plot_item.legend is not None:
                    plot_item.legend.addItem(scatter_ch1, 'Peaks CH1')
                
                # Ch2 peaks (blue)  
                ch2_peak_times = peak_indices_arr / float(sample_rate)
                ch2_peak_values = np.asarray(spl_series_ch2)[peak_indices_arr]
                scatter_ch2 = pg.ScatterPlotItem(
                    x=ch2_peak_times, y=ch2_peak_values, pen=pg.mkPen(None), brush=pg.mkBrush(0, 0, 255, 200), size=8
                )
                plot_item.addItem(scatter_ch2)
                # Add to legend
                if hasattr(plot_item, 'legend') and plot_item.legend is not None:
                    plot_item.legend.addItem(scatter_ch2, 'Peaks CH2')

        # Setup dual axis for pattern matching similarity scores
        self._setup_dual_axis_if_needed()
        
        # Clear existing similarity bars
        if self._bars_item_ch1 is not None:
            self._right_view.removeItem(self._bars_item_ch1)
            self._bars_item_ch1 = None
        if self._bars_item_ch2 is not None:
            self._right_view.removeItem(self._bars_item_ch2)
            self._bars_item_ch2 = None

        # Plot pattern matching similarity results with dual colors
        if results:
            times = np.array([r.get("time_sec", 0.0) for r in results], dtype=float)
            
            # Extract dual channel scores
            ch1_scores = []
            ch2_scores = []
            
            for r in results:
                dual_results = r.get("dual_channel_results", {})
                ch1_scores.append(dual_results.get("channel_1", {}).get("score", 0.0))
                ch2_scores.append(dual_results.get("channel_2", {}).get("score", 0.0))
            
            ch1_scores = np.array(ch1_scores, dtype=float)
            ch2_scores = np.array(ch2_scores, dtype=float)
            
            # Note: Cannot add legend to ViewBox directly - similarity bars will be distinguishable by color
            
            if times.size > 0:
                duration = max(max(time_axis_ch1[-1] - time_axis_ch1[0], time_axis_ch2[-1] - time_axis_ch2[0]), 1e-6)
                bar_width = max(duration * 0.002, duration / 1000.0)
                
                # Channel 1 similarity bars (orange-tinted)
                if np.any(ch1_scores > 0):
                    bars_ch1 = pg.BarGraphItem(
                        x=times - bar_width/4,  # Slight offset to avoid overlap
                        height=ch1_scores,
                        width=bar_width/2,
                        brush=pg.mkBrush(255, 140, 0, 180),  # Orange-tinted
                        pen=pg.mkPen(255, 100, 0, 220),
                    )
                    self._right_view.addItem(bars_ch1)
                    self._bars_item_ch1 = bars_ch1
                
                # Channel 2 similarity bars (purple-tinted)
                if np.any(ch2_scores > 0):
                    bars_ch2 = pg.BarGraphItem(
                        x=times + bar_width/4,  # Slight offset to avoid overlap
                        height=ch2_scores,
                        width=bar_width/2,
                        brush=pg.mkBrush(200, 0, 255, 180),  # Purple-tinted
                        pen=pg.mkPen(150, 0, 200, 220),
                    )
                    self._right_view.addItem(bars_ch2)
                    self._bars_item_ch2 = bars_ch2
                    
                # Set similarity axis range based on all scores
                max_score = max(np.max(ch1_scores) if np.any(ch1_scores > 0) else 0,
                              np.max(ch2_scores) if np.any(ch2_scores > 0) else 0)
                if max_score > 0:
                    self._right_view.setYRange(0.0, max_score, padding=0.05)

    def _update_table(self, sample_rate, results):
        # update the PM result table with dual-channel information
        if not hasattr(self, "table_widget"):
            return
        rows = len(results) if isinstance(results, list) else 0
        self.table_widget.setRowCount(rows)
        for idx, r in enumerate(results or []):
            time_sec = float(r.get("time_sec", 0.0))
            start_idx = int(r.get("start_index", 0))
            stop_idx = int(r.get("stop_index", start_idx))
            seg_len = max(0, stop_idx - start_idx)
            length_ms = seg_len / float(sample_rate) * 1000.0
            
            # Extract dual-channel scores
            dual_results = r.get("dual_channel_results", {})
            ch1_result = dual_results.get("channel_1", {})
            ch2_result = dual_results.get("channel_2", {})
            final_success = dual_results.get("final_success", False)
            
            ch1_score = float(ch1_result.get("score", 0.0)) * 100.0
            ch2_score = float(ch2_result.get("score", 0.0)) * 100.0
            combined_score = float(r.get("score", 0.0)) * 100.0  # This is the average score from _execute_pm
            
            # Determine match status
            ch1_success = ch1_result.get("success", False)
            ch2_success = ch2_result.get("success", False)
            if final_success:
                match_status = "双通道匹配"
            elif ch1_success and ch2_success:
                match_status = "双通道成功"
            elif ch1_success:
                match_status = "仅Ch1匹配"
            elif ch2_success:
                match_status = "仅Ch2匹配"
            else:
                match_status = "无匹配"

            # Table columns: ["序号", "时间(s)", "长度(ms)", "Ch1分数(%)", "Ch2分数(%)", "总分数(%)", "匹配状态"]
            items = [
                QTableWidgetItem(str(idx + 1)),
                QTableWidgetItem(f"{time_sec:.3f}"),
                QTableWidgetItem(f"{length_ms:.1f}"),
                QTableWidgetItem(f"{ch1_score:.2f}"),
                QTableWidgetItem(f"{ch2_score:.2f}"),
                QTableWidgetItem(f"{combined_score:.2f}"),
                QTableWidgetItem(match_status),
            ]
            
            # Color code the match status
            status_item = items[-1]
            if final_success:
                status_item.setBackground(pg.mkBrush(200, 255, 200))  # Light green
            elif ch1_success or ch2_success:
                status_item.setBackground(pg.mkBrush(255, 255, 200))  # Light yellow
            else:
                status_item.setBackground(pg.mkBrush(255, 200, 200))  # Light red
            
            for col, it in enumerate(items):
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.table_widget.setItem(idx, col, it)

    def _summarize_and_notify(self, results, pass_condition=None):
        # statistics
        total = len(results) if isinstance(results, list) else 0
        matched = sum(1 for r in (results or []) if r.get("is_match"))

        # pass condition: n1 <= matched points <= n2; if not provided, return to "if there is a match, pass"
        passed = False
        if isinstance(pass_condition, dict) and pass_condition:
            try:
                n1 = int(pass_condition.get("n1", 1))
                n2 = int(pass_condition.get("n2", 1))
                if n2 < n1:
                    n2 = n1
                passed = (matched >= n1) and (matched <= n2)
            except Exception:
                passed = matched > 0
        else:
            passed = matched > 0

        status_text = "OK" if passed else "NG"
        color = "#2e7d32" if passed else "#c62828"
        summary_line = (
            f"<span style='color:{color};font-weight:bold'>{status_text}</span>  检测到峰值数: {total}，匹配片段数: {matched}"
        )
        self.result_display.setHtml(summary_line)

        if self.is_default_flag:
            sign.set_result_file_sign.emit(0, status_text)
            sign.get_result_file_sign.emit(0)
            sign.test_insert_data_into_db_sign.emit(status_text)
        return {"results": results, "matched": matched, "total": total, "passed": passed}

    def calculate_pipeline_pd_pm(self):
        context = self._prepare_pipeline_context()
        if context is None:
            return None

        recorded_signal, sample_rate, cfg, head, tail, pd_cls, pm_cls = context

        pd_result, peak_indices = self._execute_pd(pd_cls, head)

        if not peak_indices:
            # no peak is also a valid result: no matching needed, the peak number and matching number are both 0
            self._render_plots(pd_result, recorded_signal, sample_rate, peak_indices, [])
            self._update_table(sample_rate, [])
            return self._summarize_and_notify([], cfg.get("pass_condition", {}))

        pm_cfg = tail.get("config", {})
        
        # Validate dual-channel PM configuration
        if not self._validate_dual_channel_pm_config(pm_cfg):
            self.default_logger.warning("Invalid or missing dual-channel Pattern Match configuration")
            return self._summarize_and_notify([], cfg.get("pass_condition", {}))
        

        results = self._execute_pm(
            pm_cls=pm_cls,
            sample_rate=sample_rate,
            recorded_signal_all_channels=recorded_signal,
            peak_indices=peak_indices,
            pm_cfg=pm_cfg,
        )

        self._render_plots(pd_result, recorded_signal, sample_rate, peak_indices, results)
        self._update_table(sample_rate, results)

        return self._summarize_and_notify(results, cfg.get("pass_condition", {}))

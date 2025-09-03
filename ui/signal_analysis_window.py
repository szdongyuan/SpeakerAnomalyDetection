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
from scipy.spatial.distance import cosine, euclidean, cityblock
from scipy.stats import pearsonr
from skimage.feature import local_binary_pattern, match_template

from base.data_struct.data_deal_struct import DataDealStruct
from base.load_audio import load_audio_simple
from base.log_manager import LogManager
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from base.pre_processing.audio_peak_detection import peak_detection
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

        layout = QVBoxLayout()
        layout.addWidget(self.analysis_plot_top)
        layout.addWidget(self.analysis_plot_bottom)
        self.setLayout(layout)

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

    def calculate_peak_detection(self):
        """
        calculate and plot PD analysis: the upper plot is SPL time series with peak annotation;
        """
        recorded_signal = self.data_struct.store_wave_data[0]
        sample_rate = self.data_struct.sample_rate
        if recorded_signal is None or sample_rate is None:
            return None

        try:
            self.result = peak_detection(
                np.asarray(recorded_signal, dtype=np.float64),
                int(sample_rate),
                self.analysis_config,
                deviation=self.deviation_value,
            )
        except Exception as e:
            self.status_label.setText(f"状态: 异常({e.__class__.__name__})")
            self.PD_num_label.setText("PD 数量: -")
            # clear the image and return
            self.analysis_plot_top.clear()
            return None

        # save the grid points (sample point indices) corresponding to the peaks
        peak_indices = self.result.get("peaks_index", []) if isinstance(self.result, dict) else []
        indices_list = [int(i) for i in peak_indices] if len(peak_indices) > 0 else []
        analysis_key = self.windowTitle()
        self.data_struct.pd_peak_grid_points_map[analysis_key] = indices_list
        # SPL time series + peak annotation
        self.analysis_plot_top.clear()
        spl_series = np.asarray(self.result.get("spl_db_series", []), dtype=float)
        if spl_series.size == 0:
            ref_p = 20e-6
            spl_series = 20.0 * np.log10(np.maximum(np.abs(recorded_signal), 1e-30) / ref_p)
            if self.deviation_value is not None:
                spl_series = spl_series + float(self.deviation_value)
        time_axis = np.linspace(0, len(spl_series) / sample_rate, len(spl_series))
        self.analysis_plot_top.plot(time_axis, spl_series, pen=mkPen(color=(51, 196, 77)))

        peak_times = self.result.get("peaks_time_sec", [])
        if peak_times:
            peak_indices = np.clip((np.array(peak_times) * sample_rate).astype(int), 0, len(spl_series) - 1)
            peak_values = spl_series[peak_indices]
            scatter = pg.ScatterPlotItem(
                x=np.array(peak_times), y=peak_values, pen=pg.mkPen(None), brush=pg.mkBrush(200, 0, 0, 200), size=8
            )
            self.analysis_plot_top.addItem(scatter)

        self.analysis_plot_top.setLabel("left", "SPL (dB)")
        self.analysis_plot_top.setLabel("bottom", "Time (s)")
        self.analysis_plot_top.showGrid(x=True, y=True)

        # update the number and status
        num_peaks = int(self.result.get("num_peaks", 0))
        self.PD_num_label.setText(f"PD 数量: {num_peaks}")
        self.status_label.setText("状态: 正常" if self.result.get("passed", False) else "状态: 异常")

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

    def calculate_pattern_match(self, target_data=None, analysis_config=None):
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

        target_ch1, target_ch2 = self.target_data[0], self.target_data[1]

        results_ch1 = self.run_matching_for_channel(target_ch1, config_ch1)
        results_ch2 = self.run_matching_for_channel(target_ch2, config_ch2)

        final_success = results_ch1["success"] and results_ch2["success"]

        ch1_score_str = f"{results_ch1['score'] * 100:.2f}" if results_ch1['score'] is not None else "N/A"
        ch2_score_str = f"{results_ch2['score'] * 100:.2f}" if results_ch2['score'] is not None else "N/A"
        ch1_status_str = '成功' if results_ch1["success"] else '失败'
        ch2_status_str = '成功' if results_ch2["success"] else '失败'

        final_text = f"最终匹配结果: {'成功' if final_success else '失败'}\n"
        final_text += "------------------------------------\n"
        final_text += f"通道 1 -> 得分: {ch1_score_str}, 状态: {ch1_status_str}\n"
        final_text += f"通道 2 -> 得分: {ch2_score_str}, 状态: {ch2_status_str}"

        self.result_display.setPlainText(final_text)

        return {
            "final_success": final_success,
            "channel_1": results_ch1,
            "channel_2": results_ch2,
        }

    def run_matching_for_channel(self, target_data, channel_config):

        pattern_list = channel_config.get("pattern_list", [])

        if not pattern_list:
            return {"score": None, "success": False}

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
            return {"score": None, "success": False}

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
        if config.get("apply_filter", False):
            start_freq, end_freq = config.get("filter_range_hz", [None, None])
            if start_freq is not None and end_freq is not None:
                target_data = AudioEqualizer.apply_equalizer(target_data, self.sample_rate, start_freq=start_freq,
                                                             end_freq=end_freq)
                pattern_data = AudioEqualizer.apply_equalizer(pattern_data, self.sample_rate, start_freq=start_freq,
                                                              end_freq=end_freq)

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



class PipelinePdPm(QWidget):
    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.analysis_config = None  # structure: {"head": {...}, "tail": {...}}
        self.deviation_value = None
        self.is_default_flag = False
        self.default_logger = LogManager.set_log_handler("core")
        self._init_ui()
        self.setWindowTitle(title_name)

    def _calc_left_right_from_array(self, pattern_segment: np.ndarray):
        """
        calculate the left and right grid points from the array
        """
        pattern_segment = np.asarray(pattern_segment).astype(float)
        seg_len = int(pattern_segment.size)
        if seg_len <= 0:
            return 0, 0, 0
        abs_seg = np.abs(pattern_segment)
        peaks, _ = find_peaks(abs_seg)
        if isinstance(peaks, (list, np.ndarray)) and len(peaks) > 0:
            try:
                peak_idx = int(peaks[int(np.argmax(abs_seg[peaks]))])
            except Exception:
                peak_idx = int(np.argmax(abs_seg))
        else:
            peak_idx = int(np.argmax(abs_seg))
        left_point = max(0, min(peak_idx, seg_len - 1))
        right_point = max(0, seg_len - peak_idx - 1)
        return seg_len, left_point, right_point

    def _init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.main_layout = QVBoxLayout(self)
        # plot area for summary
        self.plot_widget = pg.PlotWidget(background="white")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.5)
        self.plot_widget.setLabel("left", "SPL (dB)")
        self.plot_widget.setLabel("bottom", "Time (s)")

        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        # match result table
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(5)
        self.table_widget.setHorizontalHeaderLabels(["序号", "时间(s)", "长度(ms)", "相似度", "SPL(dB)"])
        header = self.table_widget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        #  vertical stacking, right: table; overall left and right side by side, right about 2/5
        content_layout = QHBoxLayout()
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.addWidget(self.result_display)
        left_layout.addWidget(self.plot_widget)
        content_layout.addWidget(left_container)
        content_layout.addWidget(self.table_widget)
        content_layout.setStretch(0, 3)
        content_layout.setStretch(1, 2)
        self.main_layout.addLayout(content_layout)
        self.setLayout(self.main_layout)
        self.setStyleSheet(ui_style_const.qlabel_style + ui_style_const.qlineedit_style + ui_style_const.qtextedit_style)

        self.result_display.setStyleSheet("font-size:20px;")
        self._right_view = None
        self._bars_item = None
        self._last_spl_series = None

    def _setup_dual_axis_if_needed(self):
        if self._right_view is not None:
            return
        plot_item = self.plot_widget.getPlotItem()
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

    def _compute_segment_window(self, cfg, pm_cfg):
        """
        for auto length mode, calculate the length, left and right grid points from the peak of the pattern
        """
        auto_equal = bool(cfg.get("auto_equal_length", False))
        seg_len, left_point, right_point = 0, 0, 0

        if auto_equal:
            rel_path = pm_cfg.get("pattern_save_path")
            pattern_data_path = os.path.join(DEFAULT_DIR, rel_path) if rel_path else None
            pattern_data, _ = load_audio_simple(pattern_data_path)
            segment = np.asarray(pattern_data)
            seg_len, left_point, right_point = self._calc_left_right_from_array(segment)
        else:
            left_point = int(cfg.get("left_grid", 0) or 0)
            right_point = int(cfg.get("right_grid", 0) or 0)
            seg_len = max(0, int(left_point + right_point))
        return seg_len, left_point, right_point

    def _execute_pm(self, pm_cls, sample_rate, recorded_signal, peak_indices, seg_len, left_point, right_point, pm_cfg):
        n = len(recorded_signal)
        pm_instance = pm_cls(f"{self.windowTitle()}-PM")
        pm_instance.sample_rate = sample_rate

        results = []
        for pk in peak_indices:
            center = int(pk)
            start = max(0, center - left_point)
            stop = min(n, start + seg_len)
            if stop - start <= 0:
                continue
            segment = np.asarray(recorded_signal[start:stop])
            result_dict = pm_instance.calculate_pattern_match(target_data=segment, analysis_config=pm_cfg)

            if result_dict:
                results.append(
                    {
                        "peak_index": center,
                        "time_sec": center / float(sample_rate),
                        "is_match": bool(result_dict.get("is_match", False)),
                        "score": float(result_dict.get("score", 0.0)),
                        "threshold": float(result_dict.get("threshold", 0.0)),
                        "segment_len": int(stop - start),
                        "start_index": int(start),
                        "stop_index": int(stop),
                    }
                )
        return results

    def _render_plots(self, pd_result, recorded_signal, sample_rate, peak_indices, results):
        self.plot_widget.clear()
        plot_item = self.plot_widget.getPlotItem()
        spl_series = []
        if isinstance(pd_result, dict):
            spl_series = np.asarray(pd_result.get("spl_db_series", []), dtype=float)
        if spl_series is None or len(spl_series) == 0:
            ref_p = 20e-6
            spl_series = 20.0 * np.log10(np.maximum(np.abs(recorded_signal), 1e-30) / ref_p)
            if self.deviation_value is not None:
                spl_series = spl_series + float(self.deviation_value)
        time_axis = np.linspace(0, len(spl_series) / sample_rate, len(spl_series))
        plot_item.plot(time_axis, spl_series, pen=mkPen(color=(51, 196, 77)))
        self._last_spl_series = np.asarray(spl_series)

        if peak_indices:
            peak_indices_arr = np.clip(np.asarray(peak_indices, dtype=int), 0, len(spl_series) - 1)
            peak_times = peak_indices_arr / float(sample_rate)
            peak_values = np.asarray(spl_series)[peak_indices_arr]
            scatter = pg.ScatterPlotItem(
                x=peak_times, y=peak_values, pen=pg.mkPen(None), brush=pg.mkBrush(200, 0, 0, 200), size=8
            )
            plot_item.addItem(scatter)

        self._setup_dual_axis_if_needed()
        if self._bars_item is not None:
            self._right_view.removeItem(self._bars_item)
            self._bars_item = None

        if results:
            times = np.array([r.get("time_sec", 0.0) for r in results], dtype=float)
            scores = np.array([r.get("score", 0.0) for r in results], dtype=float)
            if times.size > 0:
                duration = max(time_axis[-1] - time_axis[0], 1e-6)
                bar_width = max(duration * 0.002, duration / 1000.0)
                bars = pg.BarGraphItem(
                    x=times,
                    height=scores,
                    width=bar_width,
                    brush=pg.mkBrush(100, 149, 237, 180),
                    pen=pg.mkPen(100, 149, 237, 220),
                )
                self._right_view.addItem(bars)
                self._bars_item = bars
                self._right_view.setYRange(0.0, np.max(scores), padding=0.05)

    def _update_table(self, sample_rate, results):
        # update the PM result table
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
            score = float(r.get("score", 0.0)) * 100.0
            # get the maximum SPL in the segment
            spl_db = float("nan")
            try:
                if isinstance(self._last_spl_series, np.ndarray) and seg_len > 0:
                    seg = self._last_spl_series[start_idx:stop_idx]
                    if seg.size > 0:
                        spl_db = float(np.max(seg))
            except Exception:
                spl_db = float("nan")

            items = [
                QTableWidgetItem(str(idx + 1)),
                QTableWidgetItem(f"{time_sec:.3f}"),
                QTableWidgetItem(f"{length_ms:.1f}"),
                QTableWidgetItem(f"{score:.2f}%"),
                QTableWidgetItem(f"{spl_db:.2f}" if not np.isnan(spl_db) else "-"),
            ]
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
        seg_len, left_point, right_point = self._compute_segment_window(cfg, pm_cfg)

        results = self._execute_pm(
            pm_cls=pm_cls,
            sample_rate=sample_rate,
            recorded_signal=recorded_signal,
            peak_indices=peak_indices,
            seg_len=seg_len,
            left_point=left_point,
            right_point=right_point,
            pm_cfg=pm_cfg,
        )

        self._render_plots(pd_result, recorded_signal, sample_rate, peak_indices, results)
        self._update_table(sample_rate, results)

        return self._summarize_and_notify(results, cfg.get("pass_condition", {}))

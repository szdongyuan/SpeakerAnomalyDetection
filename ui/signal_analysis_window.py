import json
import os
import re
import sys

import librosa
import numpy as np
import pyqtgraph as pg
from pyqtgraph import mkPen
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QTextCursor, QTextCharFormat, QColor, QFont
from PyQt5.QtWidgets import QApplication, QTextEdit, QHBoxLayout, QVBoxLayout, QWidget, QLabel
from scipy.signal import find_peaks

from base.data_struct.data_deal_struct import DataDealStruct
from base.log_manager import LogManager
from base.predict_model import predict_from_audio
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from base.pre_processing.audio_peak_detection import peak_detection
from base.training_model_management import TrainingModelManagement
from base.utils.custom_signals import sign
from consts import error_code, ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.graph_widget import plot_2d_image


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
        "FR": Frequency,
        "HD": Distortion,
        "AI": AI,
        "Spec": Spectrogram,
        "LP": LooseParticle,
        "PD": PD,
    }
    return class_mapping


def custom_log_tick_strings(values, scale, spacing):
    estrings = ["%0.1g" % x for x in 10 ** np.array(values).astype(float) * np.array(scale)]
    convdict = {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
    }
    dstrings = []
    for i, e in enumerate(estrings):
        if "e" in e:
            v, p = e.split("e")
            sign = "⁻" if p[0] == "-" else ""
            pot = "".join([convdict[pp] for pp in p[1:].lstrip("0")])
            if v == "1":
                v = ""
                dstrings.append(v + "10" + sign + pot)
            elif v == "2" or v == "5":
                v = v + "·"
                dstrings.append(v + "10" + sign + pot)
            else:
                dstrings.append("")
        else:
            dstrings.append(e)
    return dstrings


class AnalysisGraphWidget(QWidget):

    def __init__(self):
        super().__init__()

        self.analysis_plot = pg.PlotWidget()

        self.set_plot_font_size(20)
        self.init_ui()

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))

        self.analysis_plot.setBackground("white")

        layout = QVBoxLayout()
        layout.addWidget(self.analysis_plot)
        self.setLayout(layout)

    def set_plot_font_size(self, font_size: int):
        font = QFont()
        font.setPixelSize(font_size)

        b_axis = self.analysis_plot.getAxis("bottom")
        l_axis = self.analysis_plot.getAxis("left")

        b_axis.logTickStrings = custom_log_tick_strings

        b_axis.setTickFont(font)
        l_axis.setTickFont(font)
        b_axis.setTextPen("black")
        l_axis.setTextPen("black")
        b_axis.setLabel(b_axis.labelText, **{"font-size": f"{font_size}px"})
        l_axis.setLabel(l_axis.labelText, **{"font-size": f"{font_size}px"})


class Distortion(AnalysisGraphWidget):
    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.refresh_stimulus_flag = None
        self.selected_label = None
        self.freq_dict = None
        self.base_freq_list = None
        self.analysis_config = None
        self.selected_harmonics = []
        self.result = {}

        self.setWindowTitle(title_name)

    def calculate_thd(self):
        freq_value, harmonic, thd = [], [], []
        self.selected_harmonics = self.analysis_config["selected_labels"]
        self.selected_harmonics = [i - 1 for i in self.selected_harmonics]
        if self.selected_harmonics:
            kwargs = {"harmonics": self.selected_harmonics}
            stimulus_signal = self.data_struct.stimulus_data
            recorded_signal = self.data_struct.store_wave_data
            sample_rate = self.data_struct.sample_rate
            atfra = AudioThdFrequencyResponseAnalysis()
            if self.refresh_stimulus_flag or (self.freq_dict is None or self.base_freq_list is None):
                self.freq_dict, self.base_freq_list = atfra.calculate_spectrum(stimulus_signal, sample_rate)
                self.refresh_stimulus_flag = False
            freq_value, harmonic, thd = atfra.calculate_thd(
                self.freq_dict, self.base_freq_list, recorded_signal, sample_rate, **kwargs
            )
        self.plot_graph(freq_value, thd)
        if isinstance("harmonic", np.ndarray):
            harmonic = harmonic.tolist()
        self.result = {"freq_value": freq_value, "harmonic": harmonic, "thd": thd}
        return self.result

    def plot_graph(self, freq_value, thd):
        # Draw a graph based on the calculated thd
        self.analysis_plot.clear()
        if self.check_valid_data(freq_value) and self.check_valid_data(thd):
            self.analysis_plot.plot(freq_value, thd, pen="b", name="THD")
        if self.selected_label is not None:
            self.analysis_plot.setTitle(f"The Distortion of {self.selected_label.text()} order")
        self.analysis_plot.setLabel("left", "Distortion(%)")
        self.analysis_plot.setLabel("bottom", "Frequency")
        self.analysis_plot.setLogMode(x=True, y=False)
        self.analysis_plot.showGrid(x=True, y=True)

    @staticmethod
    def check_valid_data(data):
        return isinstance(data, (list, np.ndarray)) and len(data) > 0


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
        signal_duration = np.linspace(0, len(recorded_signal) / sample_rate, len(recorded_signal))
        reference_pressure = 20e-6
        signal_spl = AudioThdFrequencyResponseAnalysis().spl_calculation(
            recorded_signal, reference_pressure, is_smooth=self.analysis_config["smooth_checked"]
        )
        signal_spl = signal_spl + self.deviation_value
        limit_checked = self.analysis_config.get("limit_checked")
        self_defined = self.analysis_config.get("self_defined")
        if limit_checked:
            if self_defined:
                upper_limit = self.analysis_config.get("upper_limit")
                lower_limit = self.analysis_config.get("lower_limit")
                self.plot_spl(signal_duration, signal_spl, upper_limit=upper_limit, lower_limit=lower_limit)
            else:
                self.plot_spl(signal_duration, signal_spl)
        else:
            self.plot_spl(signal_duration, signal_spl)
        self.result = {
            "signal_duration": signal_duration.tolist(),
            "recorded_signal": recorded_signal.tolist(),
            "signal_spl": signal_spl.tolist(),
        }
        return self.result

    def plot_spl(self, signal_duration, signal_spl, upper_limit="", lower_limit=""):
        self.analysis_plot.clear()
        self.analysis_plot.plot(signal_duration, signal_spl, pen=mkPen(color=(51, 196, 77)))
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
                self.analysis_plot.addItem(out_range_plot)
            dashed_pen = mkPen(color=(128, 0, 128), width=1, style=Qt.DashLine)
            lower_limit1 = pg.InfiniteLine(angle=0, pos=lower_limit, pen=dashed_pen)
            self.analysis_plot.addItem(lower_limit1)
            upper_limit1 = pg.InfiniteLine(angle=0, pos=upper_limit, pen=dashed_pen)
            self.analysis_plot.addItem(upper_limit1)
        self.analysis_plot.setLabel("left", "SPL (dB)")
        self.analysis_plot.setLabel("bottom", "Time (s)")
        self.analysis_plot.showGrid(x=True, y=True)


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


class AI(QWidget):
    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.analysis_config = None
        self.result = None
        self.default_logger = LogManager.set_log_handler("core")
        self.init_ui()
        self.setWindowTitle(title_name)

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        ai_analyse_layout = self.create_ai_analyse_layout()
        self.setLayout(ai_analyse_layout)

    def create_ai_analyse_layout(self):
        ai_analyse_layout = QVBoxLayout()
        analyse_score_layout = QHBoxLayout()
        self.ai_analyse_score_textedit = QTextEdit()
        self.ai_analyse_score_textedit.setAlignment(Qt.AlignCenter)
        self.ai_analyse_score_textedit.setDisabled(True)

        self.ai_analyse_score_textedit.setStyleSheet(ui_style_const.qtextedit_style)
        analyse_score_layout.addWidget(self.ai_analyse_score_textedit)
        analyse_score_layout.setContentsMargins(20, 0, 20, 0)

        ai_analyse_layout.addLayout(analyse_score_layout)

        return ai_analyse_layout

    def highlight_keywords(self, keyword, text_edit):
        cursor = text_edit.textCursor()
        format = QTextCharFormat()
        format.setForeground(QColor("red"))

        matches = []
        cursor.movePosition(QTextCursor.Start)
        while True:
            cursor = text_edit.document().find(keyword, cursor)
            if cursor.isNull():
                break
            matches.append(cursor)

        if len(matches) == 2:
            first_match = matches[0]
            first_match.mergeCharFormat(format)

    def calculate_ai_scores(self, mode, analysis_config):
        model_name = self.analysis_config["analyse_model_name"]
        code, result = self.get_model_info(model_name, self.default_logger)
        if code != error_code.OK or not os.path.exists(result[0]):
            self.ai_analyse_score_textedit.setPlainText("模型不存在，请重新选择！")
        else:
            model_path, config_path = result
            kwargs = {"config_path": config_path}
            result_text = self.model_predict(model_path, model_name, **kwargs)
            default_ai_model = analysis_config["default_ai"]
            if mode == "test" and default_ai_model:
                analyse_model_name = analysis_config.get(default_ai_model, None).get("analyse_model_name", None)
                match_object = re.search(r"评分结果:\s*(\S+)", result_text)
                if match_object:
                    match_result = match_object.group(1)
                    if match_result == "OK":
                        sign.set_result_file_sign.emit(0, "OK", analyse_model_name)
                        sign.get_result_file_sign.emit(0)
                        sign.test_insert_data_into_db_sign.emit("OK")
                    elif match_result == "NG":
                        sign.set_result_file_sign.emit(0, "NG", analyse_model_name)
                        sign.get_result_file_sign.emit(0)
                        sign.test_insert_data_into_db_sign.emit("NG")
            self.ai_analyse_score_textedit.setPlainText(result_text)
            self.highlight_keywords("ng", self.ai_analyse_score_textedit)

    def model_predict(self, model_path, model_name, **kwargs):
        ret_str = predict_from_audio(
            signals=[np.array(self.data_struct.store_wave_data, dtype=np.float32)],
            file_names=["modelpredict.wav"],
            fs=[self.data_struct.sample_rate],
            load_model_path=model_path,
            **kwargs,
        )
        ret_dict = json.loads(ret_str)
        predict_result = ret_dict["result"]
        predict_label = predict_result[0][1]
        ok_scores = float(predict_result[0][2]) * 100
        ng_scores = 100 - ok_scores
        self.result = predict_label
        result_text = (
            f"评分结果: {predict_label} \n \n"
            f"\xa0\xa0评分模型: {model_name}\n"
            f"\xa0\xa0OK Score: {ok_scores:.2f}%\n"
            f"\xa0\xa0NG Score: {ng_scores:.2f}%"
        )
        return result_text

    @staticmethod
    def get_model_info(selected_model, logger: LogManager):
        query_code, query_result = TrainingModelManagement().get_model_path_from_db(selected_model)
        if query_code == error_code.OK:
            model_path, config_path = query_result[0]
            really_model_path = DEFAULT_DIR + model_path
            really_config_path = DEFAULT_DIR + config_path
            return error_code.OK, (really_model_path, really_config_path)
        else:
            logger.error(f"Failed to get the model {selected_model} information.")
            return error_code.INVALID_QUERY, "Failed to get the model information."


class Spectrogram(QWidget):
    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.deviation_value = None
        self.analysis_config = None
        self.current_plot_widget = None
        self.stft_plot_widget = None
        self.img_item = None
        self.stft_colorbar = None
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
        self.stft_plot_widget = pg.PlotWidget()
        self.stft_plot_widget.setBackground("white")
        self.img_item = pg.ImageItem()
        self.stft_plot_widget.addItem(self.img_item)

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

        if self.stft_colorbar:
            color_bar_axis = self.stft_colorbar.axis
            color_bar_font = QFont()
            color_bar_font.setPixelSize(20)  # 设置颜色条字体大小为 14px
            color_bar_axis.setTickFont(color_bar_font)
            color_bar_axis.setTextPen("black")
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
            CQT_complex, freqs, times = AudioThdFrequencyResponseAnalysis().compute_cqt(
                y=recorded_signal, sr=sample_rate, hop_length=hop_length, n_fft=n_fft, fmin=fmin_cqt
            )

            CQT_mag = np.abs(CQT_complex)
            CQT_db = librosa.amplitude_to_db(CQT_mag, ref=20e-6)
            Z = CQT_db.T

            target_ticks_hz = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
            major_ticks = []
            custom_y_ticks = None

            y_min_hz, y_max_hz = freqs.min(), freqs.max()
            for freq in target_ticks_hz:
                if y_min_hz <= freq <= y_max_hz:
                    label = f"{freq} Hz" if freq < 1000 else f"{freq/1000:.0f} kHz"
                    major_ticks.append((freq, label))

            custom_y_ticks = [major_ticks, []] if major_ticks else None

            cqt_plot_widget, self.stft_colorbar = plot_2d_image(
                x=times,
                y=freqs,
                z=Z,
                title="Spectrogram(Log Scale)",
                xlabel="Time (s)",
                ylabel="Frequency (Hz)",
                colormap=color_map,
                x_range=(times.min(), times.max()),
                y_range=(freqs.min(), freqs.max()),
                y_ticks=custom_y_ticks,
                background_color="white",
            )
            self.plot_container_layout.addWidget(cqt_plot_widget)
            self.current_plot_widget = cqt_plot_widget

        else:
            spec = np.abs(librosa.stft(y=recorded_signal, n_fft=n_fft, hop_length=hop_length, window=window_func))
            spec_dB = librosa.amplitude_to_db(spec, ref=20e-6)
            freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
            times = librosa.times_like(spec_dB, sr=sample_rate, hop_length=hop_length)

            if self.stft_plot_widget is None or self.img_item is None:
                self._init_stft_plot_components()

            self.img_item.setImage(spec_dB.T)

            times_min, times_max = times.min(), times.max()
            freqs_min, freqs_max = freqs.min(), freqs.max()
            width = times_max - times_min
            height = freqs_max - freqs_min

            self.img_item.setRect(pg.QtCore.QRectF(times_min, freqs_min, width, height))

            self.stft_plot_widget.setTitle("Spectrogram (Linear Scale)")
            self.stft_plot_widget.setLabel("bottom", "Time (s)")
            self.stft_plot_widget.setLabel("left", "Frequency (Hz)")
            self.stft_plot_widget.setLogMode(x=False, y=False)

            pos = np.linspace(0.0, 1.0, 256)

            colors = pg.colormap.get(color_map).getLookupTable(nPts=256)
            cmap = pg.ColorMap(pos, colors)
            db_min, db_max = np.nanmin(spec_dB), np.nanmax(spec_dB)

            lut = cmap.getLookupTable(nPts=256)
            self.img_item.setLookupTable(lut)
            self.img_item.setLevels([db_min, db_max])

            view_box = self.stft_plot_widget.getViewBox()
            if view_box:
                view_box.setDefaultPadding(0.0)

            self.stft_plot_widget.setXRange(times_min, times_max, padding=0)
            self.stft_plot_widget.setYRange(freqs_min, freqs_max, padding=0)
            plot_item = self.stft_plot_widget.getPlotItem()
            if plot_item:
                self.stft_colorbar = pg.ColorBarItem(values=(db_min, db_max), width=25, colorMap=cmap)
                self.stft_colorbar.setImageItem(self.img_item, insert_in=plot_item)
            else:
                self.stft_colorbar = None
            self.plot_container_layout.addWidget(self.stft_plot_widget)
            self.current_plot_widget = self.stft_plot_widget
        self.set_color_font_size()


class LooseParticle(AnalysisGraphWidget):
    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
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
        recorded_signal = self.data_struct.store_wave_data
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
        signal_duration = np.linspace(
            0, len(amplitude) / (self.data_struct.sample_rate), len(amplitude)
        )
        self.result = self.detect_peaks(
            amplitude,
            self.analysis_config.get("trigger_threshold"),
            self.analysis_config.get("hysterests_threshold"),
            self.analysis_config.get("min_check_duration"),
            self.analysis_config.get("max_check_duration"),
            self.data_struct.sample_rate,
        )
        amplitude = amplitude - deviation
        self.analysis_plot.plot(signal_duration, amplitude, pen=mkPen(color=(51, 196, 77)))
        self.plot_loose_particle_waveform(self.threshould, signal_duration, deviation)
        self.analysis_plot.setLabel("left", "Amplitude (dB)")
        self.analysis_plot.setLabel("bottom", "Time (s)")
        self.analysis_plot.showGrid(x=True, y=True)

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
        self.analysis_plot.addItem(out_range_plot)


class PD(AnalysisGraphWidget):
    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.analysis_config = None
        self.result = None
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

    def calculate_spec(self):
        """
        calculate and plot PD analysis: the upper plot is SPL time series with peak annotation; 
        """
        recorded_signal = self.data_struct.store_wave_data
        sample_rate = self.data_struct.sample_rate
        if recorded_signal is None or sample_rate is None:
            return None

        try:
            detection_result = peak_detection(
                np.asarray(recorded_signal, dtype=np.float64), int(sample_rate), self.analysis_config
            )
        except Exception as e:
            self.status_label.setText(f"状态: 异常({e.__class__.__name__})")
            self.PD_num_label.setText("PD 数量: -")
            # clear the image and return
            self.analysis_plot.clear()
            return None
        self.result = detection_result

        # save the grid points (sample point indices) corresponding to the peaks
        peak_indices = detection_result.get("peaks_index", []) if isinstance(detection_result, dict) else []
        indices_list = [int(i) for i in peak_indices] if len(peak_indices) > 0 else []
        analysis_key = self.windowTitle()
        self.data_struct.pd_peak_grid_points_map[analysis_key] = indices_list

        # SPL time series + peak annotation
        self.analysis_plot.clear()
        spl_series = np.asarray(detection_result.get("spl_db_series", []), dtype=float)
        if spl_series.size == 0:
            ref_p = 20e-6
            spl_series = 20.0 * np.log10(np.maximum(np.abs(recorded_signal), 1e-30) / ref_p)
        time_axis = np.linspace(0, len(spl_series) / sample_rate, len(spl_series))
        self.analysis_plot.plot(time_axis, spl_series, pen=mkPen(color=(51, 196, 77)))

        peak_times = detection_result.get("peaks_time_sec", [])
        if peak_times:
            peak_indices = np.clip((np.array(peak_times) * sample_rate).astype(int), 0, len(spl_series) - 1)
            peak_values = spl_series[peak_indices]
            scatter = pg.ScatterPlotItem(x=np.array(peak_times), y=peak_values, pen=pg.mkPen(None), brush=pg.mkBrush(200, 0, 0, 200), size=8)
            self.analysis_plot.addItem(scatter)

        self.analysis_plot.setLabel("left", "SPL (dB)")
        self.analysis_plot.setLabel("bottom", "Time (s)")
        self.analysis_plot.showGrid(x=True, y=True)

        # update the number and status
        num_peaks = int(detection_result.get("num_peaks", 0))
        self.PD_num_label.setText(f"PD 数量: {num_peaks}")
        self.status_label.setText("状态: 正常" if detection_result.get("passed", False) else "状态: 异常")

        self._update_fonts()
        return detection_result


if __name__ == "__main__":
    stimulus, sr = librosa.load("../audio_data/analysis_samples/stimulus.wav", sr=44100)
    recorded, _ = librosa.load("../audio_data/analysis_samples/recording.wav", sr=44100)
    signal_info = {"stimulus_signal": stimulus, "recorded_signal": recorded, "sample_rate": sr}
    app = QApplication(sys.argv)
    # window = Spl(signal_info)
    # window = AnalyseWindow()
    window = AI()
    window.show()
    app.exec_()

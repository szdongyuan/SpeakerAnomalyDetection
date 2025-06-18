import json
import os
import re
import sys

from scipy.signal import find_peaks

import librosa
import numpy as np
import pyqtgraph as pg
from pyqtgraph import mkPen
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QTextCursor, QTextCharFormat, QColor
from PyQt5.QtWidgets import QApplication, QTextEdit, QHBoxLayout
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLabel

from base.data_struct.data_deal_struct import DataDealStruct
from base.log_manager import LogManager
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from base.training_model_management import TrainingModelManagement
from base.utils.custom_signals import sign
from consts import error_code, ui_style_const
from consts.running_consts import DEFAULT_DIR
from main import predict_from_audio
from ui.graph_widget import plot_2d_image

class Distortion(QWidget):
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
        self.init_ui()
        self.setWindowTitle(title_name)

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        # layout have a child's plotwidget, use to display the thd analysis results
        layout = QVBoxLayout()

        self.thd_plot = pg.PlotWidget()
        self.thd_plot.setBackground('white')

        # layout.addWidget(harmonic_group_box)
        layout.addWidget(self.thd_plot)
        self.setLayout(layout)

    def calculate_thd(self):
        freq_value, harmonic, thd = [], [], []
        self.selected_harmonics = self.analysis_config["selected_labels"]
        self.selected_harmonics = [i - 1 for i in self.selected_harmonics]
        if self.selected_harmonics:
            kwargs = {"harmonics": self.selected_harmonics}
            stimulus_signal = self.data_struct.stimulus_data
            recorded_signal = self.data_struct.store_wave_data
            sample_rate = self.data_struct.stimulus_info["sample_rate"]
            atfra = AudioThdFrequencyResponseAnalysis()
            if self.refresh_stimulus_flag or (self.freq_dict is None or self.base_freq_list is None):
                self.freq_dict, self.base_freq_list = atfra.calculate_spectrum(stimulus_signal, sample_rate)
                self.refresh_stimulus_flag = False
            freq_value, harmonic, thd = atfra.calculate_thd(self.freq_dict, self.base_freq_list,
                                                            recorded_signal, sample_rate, **kwargs)
        self.plot_graph(freq_value, thd)
        if isinstance("harmonic", np.ndarray):
            harmonic = harmonic.tolist()
        self.result = {"freq_value": freq_value,
                       "harmonic": harmonic,
                       "thd": thd}
        return self.result

    def plot_graph(self, freq_value, thd):
        # Draw a graph based on the calculated thd
        self.thd_plot.clear()
        if self.check_valid_data(freq_value) and self.check_valid_data(thd):
            self.thd_plot.plot(freq_value, thd, pen='b', name="THD")
        if self.selected_label is not None:
            self.thd_plot.setTitle(f"The Distortion of {self.selected_label.text()} order")
        self.thd_plot.setLabel('left', 'Distortion(%)')
        self.thd_plot.setLabel('bottom', 'Frequency')
        self.thd_plot.setLogMode(x=True, y=False)
        self.thd_plot.showGrid(x=True, y=True)

    @staticmethod
    def check_valid_data(data):
        return isinstance(data, (list, np.ndarray)) and len(data) > 0


class Spl(QWidget):
    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.deviation_value = None
        self.analysis_config = None
        self.result = {}
        self.init_ui()
        self.setWindowTitle(title_name)

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.spl_plot = pg.PlotWidget(title='Sound Pressure Level')
        self.spl_plot.setBackground('white')
        layout = QVBoxLayout()
        layout.addWidget(self.spl_plot)
        self.setLayout(layout)

    def calculate_spl(self):
        # calculate Sound Pressure Level according to recorded_signal
        recorded_signal = self.data_struct.store_wave_data
        sample_rate = self.data_struct.stimulus_info["sample_rate"]
        signal_duration = np.linspace(0, len(recorded_signal) / sample_rate, len(recorded_signal))
        reference_pressure = 20e-6
        signal_spl = AudioThdFrequencyResponseAnalysis().spl_calculation(recorded_signal, reference_pressure,
                                                                         is_smooth=self.analysis_config["smooth_checked"])
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
        self.result = {"signal_duration": signal_duration.tolist(),
                       "recorded_signal": recorded_signal.tolist(),
                       "signal_spl": signal_spl.tolist(),
                       }
        return self.result

    def plot_spl(self, signal_duration, signal_spl, upper_limit="", lower_limit=""):
        self.spl_plot.clear()
        self.spl_plot.plot(signal_duration, signal_spl, pen=mkPen(color=(51, 196, 77)))
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
                out_range_plot = pg.PlotDataItem(x, y, pen='r')
                self.spl_plot.addItem(out_range_plot)
            dashed_pen = mkPen(color=(128, 0, 128), width=1, style=Qt.DashLine)
            lower_limit1 = pg.InfiniteLine(angle=0, pos=lower_limit, pen=dashed_pen)
            self.spl_plot.addItem(lower_limit1)
            upper_limit1 = pg.InfiniteLine(angle=0, pos=upper_limit, pen=dashed_pen)
            self.spl_plot.addItem(upper_limit1)
        self.spl_plot.setLabel('left', 'SPL (dB)')
        self.spl_plot.setLabel('bottom', 'Time (s)')
        self.spl_plot.showGrid(x=True, y=True)


class Frequency(QWidget):

    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.smooth_flag = False
        self.temp_frequency_list = None
        self.deviation_value = None
        self.analysis_config = None
        self.result = {}
        self.init_ui()
        self.setWindowTitle(title_name)

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.fr_plot = pg.PlotWidget(title='Frequency Response')
        # self.fr_plot.setFixedSize(400, 320)
        self.fr_plot.setBackground('white')
        layout = QVBoxLayout()
        layout.addWidget(self.fr_plot)
        self.setLayout(layout)

    def calculate_fr(self):
        stimulus_signal = self.data_struct.stimulus_data
        recorded_signal = self.data_struct.store_wave_data
        sr = self.data_struct.stimulus_info["sample_rate"]
        fr, frequency_list = AudioThdFrequencyResponseAnalysis().calculate_fr(stimulus_signal, recorded_signal, sr,
                                                                              is_smooth=self.analysis_config["smooth_checked"])
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
        self.result = {"fr": fr.tolist(),
                       "frequency_list": frequency_list.tolist()
                       }
        return self.result

    def plot_fr(self, frequency_list, fr, upper_limit="", lower_limit=""):
        self.fr_plot.clear()
        fr = fr + 94 + self.deviation_value
        self.fr_plot.plot(frequency_list, fr, pen=mkPen(color=(51, 196, 77)))
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
                out_range_plot = pg.PlotDataItem(x, y, pen='r')
                self.fr_plot.addItem(out_range_plot)
            dashed_pen = mkPen(color=(128, 0, 128), width=1, style=Qt.DashLine)
            lower_limit1 = pg.InfiniteLine(angle=0, pos=lower_limit, pen=dashed_pen)
            self.fr_plot.addItem(lower_limit1)
            upper_limit1 = pg.InfiniteLine(angle=0, pos=upper_limit, pen=dashed_pen)
            self.fr_plot.addItem(upper_limit1)
        self.fr_plot.setLabel('left', 'Amplitude (dB)')
        self.fr_plot.setLabel('bottom', 'Frequency (Hz)')
        self.fr_plot.showGrid(x=True, y=True)


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

        self.ai_analyse_score_textedit.setStyleSheet(ui_style_const.qtextedit_stytle)
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

    def calculate_ai_scores(self, mode):
        model_name = self.analysis_config["analyse_model_name"]
        code, result = self.get_model_info(model_name, self.default_logger)
        if code != error_code.OK or not os.path.exists(result[0]):
            self.ai_analyse_score_textedit.setPlainText("模型不存在，请重新选择！")
        else:
            model_path, config_path = result
            kwargs = {"config_path": config_path}
            result_text = self.model_predict(model_path, model_name, **kwargs)
            config_file_path = DEFAULT_DIR + "ui/ui_config/analysis_temp_config.json"
            with open(config_file_path, 'r') as f:
                default_config = json.load(f)
                default_ai_model = default_config["default_ai"]
            if mode == "test" and default_ai_model:
                analyse_model_name = default_config.get(default_ai_model, None).get("analyse_model_name", None)
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
        ret_str = predict_from_audio(signals=[np.array(self.data_struct.store_wave_data, dtype=np.float32)],
                                     file_names=["modelpredict.wav"],
                                     fs=[self.data_struct.stimulus_info["sample_rate"]],
                                     load_model_path=model_path,
                                     **kwargs)
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
        self.stft_plot_widget.setBackground('white')
        self.img_item = pg.ImageItem()
        self.stft_plot_widget.addItem(self.img_item)

    def calculate_spec(self):
        recorded_signal = self.data_struct.store_wave_data
        sample_rate = self.data_struct.stimulus_info["sample_rate"]

        n_fft = self.analysis_config.get("n_fft", 2048)
        hop_length = self.analysis_config.get("hop_length", 256)
        color_map = self.analysis_config.get("color_map", "viridis")
        window_func = self.analysis_config.get("window_func", "hann")
        freq_scale_type = self.analysis_config.get("freq_scale_type", "linear")

        if freq_scale_type == "log":
            fmin_cqt = librosa.note_to_hz('C1')
            CQT_complex, freqs, times = AudioThdFrequencyResponseAnalysis().compute_cqt(
                y=recorded_signal,
                sr=sample_rate,
                hop_length=hop_length,
                n_fft=n_fft,
                fmin=fmin_cqt
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

            cqt_plot_widget = plot_2d_image(
                x=times, y=freqs, z=Z,
                title="Spectrogram(Log Scale)",
                xlabel="Time (s)", ylabel="Frequency (Hz)",
                colormap=color_map,
                x_range=(times.min(), times.max()),
                y_range=(freqs.min(), freqs.max()),
                y_ticks=custom_y_ticks,
                background_color='white'
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
            self.stft_plot_widget.setLabel('bottom', "Time (s)")
            self.stft_plot_widget.setLabel('left', "Frequency (Hz)")
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


class LooseParticle(QWidget):
    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.result = None
        self.analysis_config = None
        self.lp_graph = pg.PlotWidget()
        self.lp_graph.setBackground('white')
        self.lp_num_label = QLabel("LP 数量: %s"% self.result)
        self.status_label = QLabel()
        self.threshould = None
        self.init_ui()
        self.setWindowTitle(title_name)

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        lp_num_layout = QHBoxLayout()
        lp_num_layout.addStretch()
        lp_num_layout.addWidget(self.status_label)
        lp_num_layout.addWidget(self.lp_num_label)
        layout = QVBoxLayout()
        layout.addLayout(lp_num_layout)
        layout.addWidget(self.lp_graph)
        self.setLayout(layout)

    def calculate_loose_particle(self):
        recorded_signal = self.data_struct.store_wave_data
        filtered_spl, deviation = AudioThdFrequencyResponseAnalysis.calculate_loose_particle_spl(recorded_signal,
                                                                                                 self.analysis_config.get("cutoff_freq"),
                                                                                                 self.data_struct.stimulus_info["sample_rate"],
                                                                                                 67)
        self.plot_graph(filtered_spl, deviation)
        self.lp_num_label.setText("LP 数量: %s" % self.result)
        if self.result > self.analysis_config.get("loose_particle_num"):
            self.status_label.setText("状态: 异常")
        else:
            self.status_label.setText("状态: 正常")

    def plot_graph(self, amplitude, deviation):
        signal_duration = np.linspace(0, len(amplitude) / (self.data_struct.stimulus_info["sample_rate"]), len(amplitude))
        self.result = self.detect_peaks(amplitude,
                                        self.analysis_config.get("trigger_threshold"),
                                        self.analysis_config.get("hysterests_threshold"),
                                        self.analysis_config.get("min_check_duration"),
                                        self.analysis_config.get("max_check_duration"),
                                        self.data_struct.stimulus_info["sample_rate"])
        amplitude = amplitude - deviation
        self.lp_graph.plot(signal_duration, amplitude, pen=mkPen(color=(51, 196, 77)))
        self.plot_loose_particle_waveform(self.threshould, signal_duration, deviation)
        self.lp_graph.setLabel('left', 'Amplitude (dB)')
        self.lp_graph.setLabel('bottom', 'time (s)')

    def detect_peaks(self, filtered_db, max_threshold, hysterests_threshold , min_check_duration, max_check_duration, sampling_rate):
        num = 0
        peaks, _ = find_peaks(filtered_db, max_threshold)
        first_iteration = True
        last_peak = None
        current_out_range = []
        self.threshould = np.full_like(filtered_db, self.analysis_config.get("trigger_threshold") - float(hysterests_threshold) / 2)
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
            if value  < max_threshold - float(hysteresis) / 2:
                self.threshould[key] = max_threshold - hysteresis
            else:
                self.threshould[key] = value

    def plot_loose_particle_waveform(self, out_range_points, signal_duration, deviation):
        pen = pg.mkPen(color='orange', width=3)
        out_range_points = np.array(out_range_points) - deviation
        out_range_plot = pg.PlotDataItem(signal_duration, out_range_points, pen=pen)
        self.lp_graph.addItem(out_range_plot)


if __name__ == "__main__":
    stimulus, sr = librosa.load("../audio_data/analysis_samples/stimulus.wav", sr=44100)
    recorded, _ = librosa.load("../audio_data/analysis_samples/recording.wav", sr=44100)
    signal_info = {"stimulus_signal": stimulus,
                   "recorded_signal": recorded,
                   "sample_rate": sr}
    app = QApplication(sys.argv)
    # window = Spl(signal_info)
    # window = AnalyseWindow()
    window = AI()
    window.show()
    app.exec_()

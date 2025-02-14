import json
import os
import sys

import librosa
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QTextEdit, QHBoxLayout
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from base.training_model_management import TrainingModelManagement
from consts import error_code
from consts.running_consts import DEFAULT_DIR
from main import predict


class Distortion(QWidget):
    def __init__(self):
        super().__init__()
        self.signal_info = None
        self.refresh_stimulus_flag = None
        self.selected_label = None
        self.freq_dict = None
        self.base_freq_list = None
        self.analysis_config = None
        self.selected_harmonics = []
        self.result = {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("谐波分析")
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
            stimulus_signal = self.signal_info["stimulus_signal"]
            recorded_signal = self.signal_info["recorded_signal"]
            sample_rate = self.signal_info["sample_rate"]
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
    def __init__(self):
        super().__init__()
        self.signal_info = None
        self.deviation_value = None
        self.analysis_config = None
        self.result = {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("声压分析")
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.spl_plot = pg.PlotWidget(title='Sound Pressure Level')
        self.spl_plot.setBackground('white')
        layout = QVBoxLayout()
        layout.addWidget(self.spl_plot)
        self.setLayout(layout)

    def calculate_spl(self):
        # calculate Sound Pressure Level according to recorded_signal
        recorded_signal = self.signal_info["recorded_signal"]
        sample_rate = self.signal_info["sample_rate"]
        signal_duration = np.linspace(0, len(recorded_signal) / sample_rate, len(recorded_signal))
        reference_pressure = 20e-6
        signal_spl = AudioThdFrequencyResponseAnalysis().spl_calculation(recorded_signal, reference_pressure,
                                                                         is_smooth=self.analysis_config["smooth_checked"])
        signal_spl = signal_spl + self.deviation_value
        if self.analysis_config["limit_checked"]:
            if self.analysis_config["radio_button_1_checked"]:
                self.plot_spl(signal_duration, signal_spl, upper_limit=self.analysis_config["upper_limit"],
                              lower_limit=self.analysis_config["lower_limit"])
            else:
                self.plot_spl(signal_duration, recorded_signal, signal_spl)
        self.result = {"signal_duration": signal_duration.tolist(),
                       "recorded_signal": recorded_signal.tolist(),
                       "signal_spl": signal_spl.tolist(),
                       }
        return self.result

    def plot_spl(self, signal_duration, signal_spl, upper_limit=None, lower_limit=None):
        self.spl_plot.clear()
        self.spl_plot.plot(signal_duration, signal_spl, pen='r')
        self.spl_plot.setLabel('left', 'SPL (dB)')
        self.spl_plot.setLabel('bottom', 'Time (s)')
        dashed_pen = pg.mkPen(color='gray', width=1, style=Qt.DashLine)
        lower_limit = pg.InfiniteLine(angle=0, pos=lower_limit, pen=dashed_pen)
        self.spl_plot.addItem(lower_limit)
        upper_limit = pg.InfiniteLine(angle=0, pos=upper_limit, pen=dashed_pen)
        self.spl_plot.addItem(upper_limit)


class Frequency(QWidget):

    def __init__(self):
        super().__init__()
        self.signal_info = None
        self.smooth_flag = False
        self.temp_frequency_list = None
        self.deviation_value = None
        self.analysis_config = None
        self.result = {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("频响分析")
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.fr_plot = pg.PlotWidget(title='Frequency Response')
        # self.fr_plot.setFixedSize(400, 320)
        self.fr_plot.setBackground('white')
        layout = QVBoxLayout()
        layout.addWidget(self.fr_plot)
        self.setLayout(layout)

    def calculate_fr(self):
        stimulus_signal = self.signal_info["stimulus_signal"]
        recorded_signal = self.signal_info["recorded_signal"]
        sr = self.signal_info["sample_rate"]
        fr, frequency_list = AudioThdFrequencyResponseAnalysis().calculate_fr(stimulus_signal, recorded_signal, sr)
        if self.analysis_config["limit_checked"]:
            if self.analysis_config["radio_button_1_checked"]:
                upper_limit = self.analysis_config["upper_limit"]
                lower_limit = self.analysis_config["lower_limit"]
                self.plot_fr(frequency_list, fr, upper_limit=upper_limit, lower_limit=lower_limit)
            else:
                self.plot_fr(frequency_list, fr)
        self.result = {"fr": fr.tolist(),
                       "frequency_list": frequency_list.tolist()
                       }
        return self.result

    def plot_fr(self, frequency_list, fr, upper_limit=None, lower_limit=None):
        # drawing the Frequency Response
        self.fr_plot.clear()
        fr = fr + self.deviation_value
        self.fr_plot.plot(frequency_list, fr, pen='b')
        self.fr_plot.setLabel('left', 'Amplitude (dB)')
        self.fr_plot.setLabel('bottom', 'Frequency (Hz)')
        self.fr_plot.setLogMode(x=True, y=False)
        dashed_pen = pg.mkPen(color='gray', width=1, style=Qt.DashLine)
        upper_limit = pg.InfiniteLine(angle=0, pos=upper_limit, pen=dashed_pen)
        self.fr_plot.addItem(upper_limit)
        lower_limit = pg.InfiniteLine(angle=0, pos=lower_limit, pen=dashed_pen)
        self.fr_plot.addItem(lower_limit)


class AI(QWidget):
    def __init__(self):
        super().__init__()
        self.signal_info = None
        self.analysis_config = None
        self.init_ui()

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setWindowTitle("AI 分析")
        ai_analyse_layout = self.create_ai_analyse_layout()
        self.setLayout(ai_analyse_layout)

    def create_ai_analyse_layout(self):
        ai_analyse_layout = QVBoxLayout()
        analyse_score_layout = QHBoxLayout()
        self.ai_analyse_score_lineedit = QTextEdit()
        self.ai_analyse_score_lineedit.setAlignment(Qt.AlignCenter)
        self.ai_analyse_score_lineedit.setDisabled(True)
        self.ai_analyse_score_lineedit.setMaximumSize(600, 800)
        self.ai_analyse_score_lineedit.setMinimumSize(550, 500)
        self.ai_analyse_score_lineedit.setStyleSheet("font-size: 23pt;")
        analyse_score_layout.addWidget(self.ai_analyse_score_lineedit)
        analyse_score_layout.setContentsMargins(20, 0, 20, 0)

        ai_analyse_layout.addLayout(analyse_score_layout)

        return ai_analyse_layout

    def calculate_ai_scores(self):
        model_name = self.analysis_config["analyse_model_name"]
        code, result = self.get_model_info(model_name)
        if code != error_code.OK or not os.path.exists(result[0]):
            self.ai_analyse_score_lineedit.setPlainText("模型不存在，请重新选择！")
        else:
            model_path, config_path = result
            kwargs = {"config_path": config_path}
            result_text = self.model_predict(model_path, **kwargs)
            self.ai_analyse_score_lineedit.setPlainText(result_text)

    def model_predict(self, model_path, **kwargs):
        ret_str = predict(self.signal_info["recorded_path"], load_model_path=model_path, **kwargs)
        ret_dict = json.loads(ret_str)
        predict_result = ret_dict["result"]
        predict_label = predict_result[0][1]
        ok_scores = float(predict_result[0][2]) * 100
        ng_scores = 100 - ok_scores
        result_text = (
            f"评分：\n"
            f"OK Score: {ok_scores:.2f}%\n"
            f"NG Score: {ng_scores:.2f}%\n"
            f"评分结果: {predict_label}"
        )
        return result_text

    def get_model_info(self, selected_model):
        query_code, query_result = TrainingModelManagement().get_model_path_from_db(selected_model)
        if query_code == error_code.OK:
            model_path, config_path = query_result[0]
            return error_code.OK, (model_path, config_path)
        else:
            self.default_logger.error(f"Failed to get the model {selected_model} information.")
            return error_code.INVALID_QUERY, "Failed to get the model information."


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

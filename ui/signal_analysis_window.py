import json
import os
import re
import sys

import librosa
import numpy as np
import pyqtgraph as pg
from librosa.core import spectrum
from librosa.feature import spectral
from librosa.sequence import dtw
from pyqtgraph import mkPen
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QTextCursor, QTextCharFormat, QColor, QFont
from PyQt5.QtWidgets import QApplication, QTextEdit, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView
from scipy.signal import find_peaks

from base.data_struct.data_deal_struct import DataDealStruct
from base.load_audio import load_audio_simple
from base.log_manager import LogManager
from base.predict_model import predict_from_audio
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from base.pre_processing.audio_peak_detection import peak_detection
from base.pre_processing.audio_equalizer import AudioEqualizer
from base.training_model_management import TrainingModelManagement
from base.utils.custom_signals import sign
from base.utils.smooth import smooth
from consts import error_code, ui_style_const
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
        "FR": Frequency,
        "HD": Distortion,
        "RB": RubAndBuzz,  # Rub & Buzz (high-order 10th-35th harmonic distortion)
        "PRB": PerceptualRubAndBuzz,  # Perceptual Rub & Buzz (2nd-35th harmonics, psychoacoustic loudness in phons)
        "AI": AI,
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
        """
        Calculate THD using the new three-phase architecture.

        Retrieves stimulus metadata from data_struct and calls the modern THD calculation pipeline.
        For mirror chirps, averages the forward and backward sweeps into a single curve.
        """
        # Get selected harmonics from analysis config
        # UI config stores harmonic orders directly (2, 3, 4, 10, 15, etc.)
        # Handle case where config might not have selected_labels (e.g., during initialization)
        if self.analysis_config is None:
            self.plot_graph([], [])
            self.result = {"freq_value": [], "harmonic": [], "thd": []}
            return self.result

        self.selected_harmonics = self.analysis_config.get("selected_labels", [])

        if not self.selected_harmonics:
            # No harmonics selected, nothing to calculate
            self.plot_graph([], [])
            self.result = {"freq_value": [], "harmonic": [], "thd": []}
            return self.result
        
        # Get signals and metadata from data_struct
        recorded_signal = self.data_struct.store_wave_data
        sample_rate = self.data_struct.sample_rate
        stimulus_info = self.data_struct.stimulus_info
        
        if recorded_signal is None or sample_rate is None or stimulus_info is None:
            raise ValueError("Missing required data: recorded_signal, sample_rate, or stimulus_info")
        
        # Convert stimulus_info to stimulus_metadata format
        # Handle naming differences: "chirp" -> "chirps", normalize method names
        stimulus_method = stimulus_info.get("stimulus_method", "steps")
        if stimulus_method == "chirp":
            stimulus_method = "chirps"
        elif stimulus_method == "step":
            stimulus_method = "steps"
        
        stimulus_metadata = {
            'stimulus_method': stimulus_method,
            'stimulus_type': stimulus_info.get("stimulus_type", "linear"),
            'start_freq': stimulus_info.get("start_freq"),
            'stop_freq': stimulus_info.get("stop_freq"),
            'num_steps': stimulus_info.get("num_steps"),
            'total_time': stimulus_info.get("total_time"),
            'repeat_times': stimulus_info.get("repeat_times"),
            'sample_rate': sample_rate
        }
        
        # Call the new three-phase architecture
        atfra = AudioThdFrequencyResponseAnalysis()
        thd_kwargs = {
            'stimulus_metadata': stimulus_metadata,
            'harmonic_orders': self.selected_harmonics
        }
        
        freq_value, harmonic, thd = atfra._calculate_thd_three_phase(
            recorded_signal, sample_rate, thd_kwargs
        )
        
        # Handle mirror chirps: average forward and backward sweeps
        if stimulus_method == "chirps" and "mirror" in stimulus_metadata['stimulus_type']:
            # Split data in half (first half = backward sweep, second half = forward sweep)
            mid_point = len(thd) // 2
            thd_backward = thd[:mid_point]
            thd_forward = thd[mid_point:]
            freq_backward = freq_value[:mid_point]
            freq_forward = freq_value[mid_point:]
            
            # Reverse backward sweep to align frequencies with forward sweep
            thd_backward_reversed = thd_backward[::-1]
            
            # Average the two sweeps (handle potential length mismatch from odd/even split)
            min_len = min(len(thd_forward), len(thd_backward_reversed))
            thd = (thd_forward[:min_len] + thd_backward_reversed[:min_len]) / 2.0
            freq_value = freq_forward[:min_len]  # Use forward frequencies (ascending order)
        
        # Plot the results
        self.plot_graph(freq_value, thd)
        
        # Convert to list format for result storage
        if isinstance(harmonic, np.ndarray):
            harmonic = harmonic.tolist()
        if isinstance(freq_value, np.ndarray):
            freq_value = freq_value.tolist()
        if isinstance(thd, np.ndarray):
            thd = thd.tolist()
        
        self.result = {"freq_value": freq_value, "harmonic": harmonic, "thd": thd}
        return self.result

        # OLD CODE (REMOVED - kept for reference):
        # freq_value, harmonic, thd = [], [], []
        # self.selected_harmonics = self.analysis_config["selected_labels"]
        # self.selected_harmonics = [i - 1 for i in self.selected_harmonics]
        # if self.selected_harmonics:
        #     kwargs = {"harmonics": self.selected_harmonics}
        #     stimulus_signal = self.data_struct.stimulus_data
        #     recorded_signal = self.data_struct.store_wave_data
        #     sample_rate = self.data_struct.sample_rate
        #     atfra = AudioThdFrequencyResponseAnalysis()
        #     if self.refresh_stimulus_flag or (self.freq_dict is None or self.base_freq_list is None):
        #         self.freq_dict, self.base_freq_list = atfra.calculate_spectrum(stimulus_signal, sample_rate)
        #         self.refresh_stimulus_flag = False
        #     freq_value, harmonic, thd = atfra.calculate_thd(
        #         self.freq_dict, self.base_freq_list, recorded_signal, sample_rate, **kwargs
        #     )
        # self.plot_graph(freq_value, thd)
        # if isinstance("harmonic", np.ndarray):
        #     harmonic = harmonic.tolist()
        # self.result = {"freq_value": freq_value, "harmonic": harmonic, "thd": thd}
        # return self.result

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


class RubAndBuzz(Distortion):
    """
    Rub & Buzz analysis widget - displays high-order harmonic distortion (10th+ harmonics).

    Inherits from Distortion and reuses all calculation methods.
    The only difference is the harmonic range enforced by RbConfigWindow (10-35 instead of 2-35).
    """

    def __init__(self, title_name):
        super().__init__(title_name)
        # Inherits all attributes and methods from Distortion
        # No additional state or overrides needed - harmonic range is controlled by config dialog


class PerceptualRubAndBuzz(RubAndBuzz):
    """
    Perceptual Rub & Buzz analysis widget - displays perceived loudness (phons) of harmonics (2nd-35th).

    Inherits from RubAndBuzz but uses psychoacoustic models (ISO 226 equal-loudness + masking).
    Y-axis shows phons instead of THD percentage. Unlike standard RB (10th-35th), PRB analyzes
    the full harmonic range (2nd-35th).
    """

    def __init__(self, title_name):
        super().__init__(title_name)

    def calculate_thd(self):
        """
        Calculate perceptual loudness using three-phase architecture with psychoacoustic models.

        Overrides parent method to use _calculate_perceptual_thd_three_phase instead of
        _calculate_thd_three_phase.
        """
        # Get selected harmonics from analysis config
        # Handle case where config might not have selected_labels (e.g., during initialization)
        if self.analysis_config is None:
            self.plot_graph([], [])
            self.result = {"freq_value": [], "harmonic": [], "thd": []}
            return self.result

        # PRB uses a fixed harmonic range (2nd-35th). The config dialog no longer exposes harmonic selection.
        self.selected_harmonics = list(range(2, 36))

        # PRB loudness model selection:
        #   - "sc": Listen/SoundCheck simplified model (paper path)
        #   - "iso": ISO 532 (mosqito) path
        prb_method = str(self.analysis_config.get("prb_method", "iso")).strip().lower()
        if prb_method not in {"sc", "iso"}:
            prb_method = "sc" if "sc" in prb_method else "iso"

        # Get signals and metadata from data_struct
        recorded_signal = self.data_struct.store_wave_data
        sample_rate = self.data_struct.sample_rate
        stimulus_info = self.data_struct.stimulus_info

        if recorded_signal is None or sample_rate is None or stimulus_info is None:
            raise ValueError("Missing required data: recorded_signal, sample_rate, or stimulus_info")

        # Convert stimulus_info to stimulus_metadata format
        stimulus_method = stimulus_info.get("stimulus_method", "steps")
        if stimulus_method == "chirp":
            stimulus_method = "chirps"
        elif stimulus_method == "step":
            stimulus_method = "steps"

        stimulus_metadata = {
            'stimulus_method': stimulus_method,
            'stimulus_type': stimulus_info.get("stimulus_type", "linear"),
            'start_freq': stimulus_info.get("start_freq"),
            'stop_freq': stimulus_info.get("stop_freq"),
            'num_steps': stimulus_info.get("num_steps"),
            'total_time': stimulus_info.get("total_time"),
            'repeat_times': stimulus_info.get("repeat_times"),
            'sample_rate': sample_rate
        }

        # Call the PERCEPTUAL three-phase architecture
        # Get microphone calibration deviation
        from base.soundcard_calibration_manager import get_mic_deviation_value
        mic_deviation_db = get_mic_deviation_value()

        atfra = AudioThdFrequencyResponseAnalysis()
        masking_config = {}
        cfg_masking = self.analysis_config.get("masking_config")
        if isinstance(cfg_masking, dict):
            masking_config.update(cfg_masking)
        masking_config["prb_method"] = prb_method
        thd_kwargs = {
            'stimulus_metadata': stimulus_metadata,
            'harmonic_orders': self.selected_harmonics,
            'masking_config': masking_config,
        }

        freq_value, harmonic, perceptual_loudness = atfra._calculate_perceptual_thd_three_phase(
            recorded_signal, sample_rate, thd_kwargs, spl_calibration_db=mic_deviation_db
        )

        # Handle mirror chirps: average forward and backward sweeps
        if stimulus_method == "chirps" and "mirror" in stimulus_metadata['stimulus_type']:
            # Split data in half
            mid_point = len(perceptual_loudness) // 2
            loudness_backward = perceptual_loudness[:mid_point]
            loudness_forward = perceptual_loudness[mid_point:]
            freq_backward = freq_value[:mid_point]
            freq_forward = freq_value[mid_point:]

            # Reverse backward sweep
            loudness_backward_reversed = loudness_backward[::-1]

            # Average the two sweeps
            min_len = min(len(loudness_forward), len(loudness_backward_reversed))
            perceptual_loudness = (loudness_forward[:min_len] + loudness_backward_reversed[:min_len]) / 2.0
            freq_value = freq_forward[:min_len]

        # Plot the results (Y-axis will be in phons)
        self.plot_graph(freq_value, perceptual_loudness)

        # Convert to list format for result storage
        if isinstance(harmonic, np.ndarray):
            harmonic = harmonic.tolist()
        if isinstance(freq_value, np.ndarray):
            freq_value = freq_value.tolist()
        if isinstance(perceptual_loudness, np.ndarray):
            perceptual_loudness = perceptual_loudness.tolist()

        # Note: "thd" key name kept for backward compatibility, but contains phons
        self.result = {"freq_value": freq_value, "harmonic": harmonic, "thd": perceptual_loudness}
        return self.result

    def plot_graph(self, freq_value, perceptual_loudness):
        """Plot perceptual loudness with correct phons label."""
        self.analysis_plot.clear()
        if self.check_valid_data(freq_value) and self.check_valid_data(perceptual_loudness):
            self.analysis_plot.plot(freq_value, perceptual_loudness, pen="b", name="Perceived Loudness")
        if self.selected_label is not None:
            self.analysis_plot.setTitle(f"Perceived Loudness of {self.selected_label.text()} order")
        self.analysis_plot.setLabel("left", "Perceived Loudness (phons)")
        self.analysis_plot.setLabel("bottom", "Frequency")
        self.analysis_plot.setLogMode(x=True, y=False)
        self.analysis_plot.showGrid(x=True, y=True)


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
            recorded_signal, reference_pressure, deviation=self.deviation_value
        )
        if self.analysis_config["smooth_checked"]:
            signal_spl = smooth(signal_spl, window_size=1102, method="rms")
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
        top_limit = self.analysis_config.get("top_limit", 70)
        bottom_limit = self.analysis_config.get("bottom_limit", 50)
        custom_limit_flag = self.analysis_config.get("custom_limit", False)
        
        mid_value = (top_limit - bottom_limit) / 2
        max_value = top_limit + mid_value
        min_value = bottom_limit - mid_value

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

            self.img_item.setImage(spec_dB.T, autoLevels=False)

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
            
        if custom_limit_flag:
            self.stft_colorbar.setLevels((min_value, max_value))

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


class PeakDetection(AnalysisGraphWidget):
    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
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
        recorded_signal = self.data_struct.store_wave_data
        sample_rate = self.data_struct.sample_rate
        if recorded_signal is None or sample_rate is None:
            return None

        try:
            self.result = peak_detection(
                np.asarray(recorded_signal, dtype=np.float64), int(sample_rate), self.analysis_config, deviation=self.deviation_value
            )
        except Exception as e:
            self.status_label.setText(f"状态: 异常({e.__class__.__name__})")
            self.PD_num_label.setText("PD 数量: -")
            # clear the image and return
            self.analysis_plot.clear()
            return None

        # save the grid points (sample point indices) corresponding to the peaks
        peak_indices = self.result.get("peaks_index", []) if isinstance(self.result, dict) else []
        indices_list = [int(i) for i in peak_indices] if len(peak_indices) > 0 else []
        analysis_key = self.windowTitle()
        self.data_struct.pd_peak_grid_points_map[analysis_key] = indices_list
        # SPL time series + peak annotation
        self.analysis_plot.clear()
        spl_series = np.asarray(self.result.get("spl_db_series", []), dtype=float)
        if spl_series.size == 0:
            ref_p = 20e-6
            spl_series = 20.0 * np.log10(np.maximum(np.abs(recorded_signal), 1e-30) / ref_p)
            if self.deviation_value is not None:
                spl_series = spl_series + float(self.deviation_value)
        time_axis = np.linspace(0, len(spl_series) / sample_rate, len(spl_series))
        self.analysis_plot.plot(time_axis, spl_series, pen=mkPen(color=(51, 196, 77)))

        peak_times = self.result.get("peaks_time_sec", [])
        if peak_times:
            peak_indices = np.clip((np.array(peak_times) * sample_rate).astype(int), 0, len(spl_series) - 1)
            peak_values = spl_series[peak_indices]
            scatter = pg.ScatterPlotItem(x=np.array(peak_times), y=peak_values, pen=pg.mkPen(None), brush=pg.mkBrush(200, 0, 0, 200), size=8)
            self.analysis_plot.addItem(scatter)

        self.analysis_plot.setLabel("left", "SPL (dB)")
        self.analysis_plot.setLabel("bottom", "Time (s)")
        self.analysis_plot.showGrid(x=True, y=True)

        # update the number and status
        num_peaks = int(self.result.get("num_peaks", 0))
        self.PD_num_label.setText(f"PD 数量: {num_peaks}")
        self.status_label.setText("状态: 正常" if self.result.get("passed", False) else "状态: 异常")

        self._update_fonts()
        return self.result



class PatternMatch(QWidget):
    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.target_data = self.data_struct.store_wave_data
        self.pattern_data = None
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
        self.pattern_data = self.load_pattern_data()
        if self.pattern_data is None:
            self.result_display.setText("错误：模版数据不存在")
            return

        algorithm_name = self.analysis_config.get("algorithm", "dtw")
        threshold_method = self.analysis_config.get("threshold_strategy")

        threshold = 0.9
        if threshold_method == "fixed_threshold":
            threshold = self.analysis_config.get("threshold_value", 0.9)

        similarity_metric = self.analysis_config.get("similarity_metric", "euclidean")
        apply_filter = self.analysis_config.get("apply_filter", False)

        if apply_filter:
            filter_range_hz = self.analysis_config.get("filter_range_hz")
            start_freq, end_freq = filter_range_hz
            self.target_data = AudioEqualizer.apply_equalizer(self.target_data, self.sample_rate, start_freq=start_freq,
                                                              end_freq=end_freq)
            self.pattern_data = AudioEqualizer.apply_equalizer(self.pattern_data, self.sample_rate,
                                                               start_freq=start_freq, end_freq=end_freq)
        feature_type = self.analysis_config.get("feature_type", "mfcc")
        target_features, pattern_features = self.feature_extraction_handle(self.target_data, self.pattern_data,
                                                                           feature_type)

        result_dict = self.algorithm_handle(algorithm_name, target_features, pattern_features,
                                            distance_measure_method=similarity_metric,
                                            threshold=threshold
                                            )
        if result_dict:
            is_match = result_dict['is_match']
            score = result_dict['score']
            used_threshold = result_dict['threshold']

            if is_match:
                match_status = "匹配成功"
            else:
                match_status = "匹配失败"

            result_text = (
                f"\xa0\xa0匹配结果: {match_status}\n\n"
                f"\xa0\xa0相似度评分: {score * 100:.2f}%\n"
                f"\xa0\xa0判定阈值: {used_threshold * 100:.2f}%"
            )
            self.result_display.setPlainText(result_text)
            return result_dict
        else:
            self.result_display.setText("分析执行时发生错误!")
            return None

    @staticmethod
    def algorithm_handle(algorithm_name, target_data, pattern_data, distance_measure_method=None, threshold=None):
        if algorithm_name == "dtw":
            if threshold is None:
                raise ValueError("A threshold must be provided to determine similarity.")

            target_data = target_data
            pattern_data = pattern_data

            D, wp = dtw(X=target_data, Y=pattern_data, metric=distance_measure_method, backtrack=True)

            distance = D[-1, -1]
            similarity = 1 / (1 + distance)
            is_match = similarity >= threshold

            return {
                "is_match": is_match,
                "score": similarity,
                "threshold": threshold
            }
        return None

    def feature_extraction_handle(self, target_data, pattern_data, feature_type):
        if feature_type != "waveform":
            feature_params = self.analysis_config["feature_params"]
            if feature_type == "mfcc":
                target_data = spectral.mfcc(y=target_data, sr=self.sample_rate, **feature_params)
                pattern_data = spectral.mfcc(y=pattern_data, sr=self.sample_rate, **feature_params)
            elif feature_params == "spec":
                target_data = np.abs(spectrum.stft(y=target_data, **feature_params))
                pattern_data = np.abs(spectrum.stft(y=pattern_data, **feature_params))
            elif feature_params == "fft":
                target_len = len(target_data)
                pattern_len = len(pattern_data)
                target_data = np.abs(np.fft.fft(target_data) / target_len)[:target_len // 2]
                pattern_data = np.abs(np.fft.fft(pattern_data) / pattern_len)[:pattern_len // 2]
        return target_data, pattern_data

    def load_pattern_data(self):
        re_path = self.analysis_config.get("pattern_save_path")
        pattern_data_path = None
        if re_path:
            pattern_data_path = os.path.join(DEFAULT_DIR, re_path)
        if not pattern_data_path or not os.path.exists(pattern_data_path):
            self.result_display.setText("错误：模版数据不存在")
            return
        pattern_data, _ = load_audio_simple(pattern_data_path)
        return pattern_data



class PipelinePdPm(QWidget):
    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.analysis_config = None  # structure: {"head": {...}, "tail": {...}}
        self.deviation_value = None
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
        self.setStyleSheet(
            ui_style_const.qlabel_style
            + ui_style_const.qlineedit_style
            + ui_style_const.qtextedit_style
        )

        self.result_display.setStyleSheet("font-size:20px;")
        self._right_view = None
        self._bars_item = None
        self._last_spl_series = None

    def _setup_dual_axis_if_needed(self):
        if self._right_view is not None:
            return
        plot_item = self.plot_widget.getPlotItem()
        plot_item.showAxis('right')
        right_axis = plot_item.getAxis('right')
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
            scatter = pg.ScatterPlotItem(x=peak_times, y=peak_values, pen=pg.mkPen(None),
                                         brush=pg.mkBrush(200, 0, 0, 200), size=8)
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
                bars = pg.BarGraphItem(x=times, height=scores, width=bar_width,
                                       brush=pg.mkBrush(100, 149, 237, 180), pen=pg.mkPen(100, 149, 237, 220))
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
        summary_line = f"<span style='color:{color};font-weight:bold'>{status_text}</span>  检测到峰值数: {total}，匹配片段数: {matched}"
        self.result_display.setHtml(summary_line)
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

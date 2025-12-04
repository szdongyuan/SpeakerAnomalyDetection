import csv
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
from base.load_config import load_config
from base.log_manager import LogManager
from base.predict_model import predict_from_audio, predict_multichannel_from_audio
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from base.training_model_management import TrainingModelManagement
from base.utils.custom_signals import sign
from base.utils.smooth import smooth
from consts import error_code, ui_style_const, model_consts
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
        "AI": AI,
        "Spec": Spectrogram,
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



class Spl(AnalysisGraphWidget):
    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.deviation_value = None
        self.analysis_config = None
        self.result = {}
        self.setWindowTitle(title_name)

    def calculate_spl(self):
        """Calculate Sound Pressure Level for single or multi-channel audio."""
        recorded_signal = self.data_struct.store_wave_data
        sample_rate = self.data_struct.sample_rate
        num_channels = self.data_struct.num_channels
        reference_pressure = 20e-6

        # Handle single vs multi-channel
        if num_channels == 1:
            # Single channel (backward compatible)
            signal_1d = recorded_signal if recorded_signal.ndim == 1 else recorded_signal[:, 0]
            signal_duration = np.linspace(0, len(signal_1d) / sample_rate, len(signal_1d))

            signal_spl = AudioThdFrequencyResponseAnalysis().spl_calculation(
                signal_1d, reference_pressure, deviation=self.deviation_value
            )

            if self.analysis_config.get("smooth_checked", False):
                signal_spl = smooth(signal_spl, window_size=1102, method="rms")

            self.plot_spl(signal_duration, signal_spl)

            self.result = {
                "signal_duration": signal_duration.tolist(),
                "recorded_signal": signal_1d.tolist(),
                "signal_spl": signal_spl.tolist(),
            }
        else:
            # Multi-channel
            signal_duration = np.linspace(0, recorded_signal.shape[0] / sample_rate, recorded_signal.shape[0])

            spl_channels = []
            for ch_idx in range(num_channels):
                signal_ch = recorded_signal[:, ch_idx]
                spl_ch = AudioThdFrequencyResponseAnalysis().spl_calculation(
                    signal_ch, reference_pressure, deviation=self.deviation_value
                )
                if self.analysis_config.get("smooth_checked", False):
                    spl_ch = smooth(spl_ch, window_size=1102, method="rms")
                spl_channels.append(spl_ch)

            self.plot_multi_channel_spl(signal_duration, spl_channels, num_channels)

            self.result = {
                "signal_duration": signal_duration.tolist(),
                "recorded_signal": recorded_signal.tolist(),
                "signal_spl_channels": [spl.tolist() for spl in spl_channels],
            }

        return self.result

    def plot_spl(self, signal_duration, signal_spl):
        """Plot SPL for single channel."""
        self.analysis_plot.clear()
        self.analysis_plot.plot(signal_duration, signal_spl, pen=mkPen(color=(51, 196, 77), width=2))
        self.analysis_plot.setLabel("left", "SPL (dB)")
        self.analysis_plot.setLabel("bottom", "Time (s)")
        self.analysis_plot.showGrid(x=True, y=True)

    def plot_multi_channel_spl(self, signal_duration, spl_channels, num_channels):
        """Plot SPL for multiple channels on the same axes with different colors."""
        self.analysis_plot.clear()

        # Define colors for up to 4 channels
        channel_colors = [
            (51, 196, 77),    # Green - Channel 1
            (77, 144, 255),   # Blue - Channel 2
            (255, 153, 51),   # Orange - Channel 3
            (204, 51, 255),   # Purple - Channel 4
        ]

        # Plot each channel with different color
        for ch_idx in range(num_channels):
            color = channel_colors[ch_idx]
            pen = mkPen(color=color, width=2)
            self.analysis_plot.plot(
                signal_duration,
                spl_channels[ch_idx],
                pen=pen,
                name=f'Channel {ch_idx + 1}'
            )

        # Add legend
        self.analysis_plot.addLegend()

        # Labels and grid
        self.analysis_plot.setLabel("left", "SPL (dB)")
        self.analysis_plot.setLabel("bottom", "Time (s)")
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
        # 读取配置判断单/多通道
        config_path = kwargs.get("config_path", model_consts.CONFIG_PATH)
        full_config_path = config_path
        data_load_config = load_config(config_path=full_config_path, module_name="data_load")
        multichannel_config = data_load_config.get("multichannel", {})
        if multichannel_config.get("enabled"):
            # 多通道预测
            raw_data = np.array(self.data_struct.store_wave_data, dtype=np.float32)
            if raw_data.ndim == 2 and raw_data.shape[0] > raw_data.shape[1]:
                raw_data = raw_data.T
            ret_str = predict_multichannel_from_audio(
                multichannel_signal=np.array(raw_data, dtype=np.float32),
                file_name="modelpredict.wav",
                sr=self.data_struct.sample_rate,
                load_model_path=model_path,
                **kwargs,
            )
        else:
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
        # 如果有通道详情（多通道），则显示
        channel_details = predict_result[0][3] if len(predict_result[0]) > 3 else None
        self.result = predict_label
        result_text = (
            f"评分结果: {predict_label} \n \n"
            f"\xa0\xa0评分模型: {model_name}\n"
            f"\xa0\xa0OK Score: {ok_scores:.2f}%\n"
            f"\xa0\xa0NG Score: {ng_scores:.2f}%"
        )
        if channel_details:
            result_text += f"\n\xa0\xa0通道详情: {channel_details}"

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

    def _clear_plot_container(self):
        """Remove all widgets from plot container."""
        while self.plot_container_layout.count():
            item = self.plot_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

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

            current_title = plot_widget.plotItem.titleLabel.text
            plot_widget.setTitle(current_title, size="20px", color="black")

        if self.stft_colorbar:
            color_bar_axis = self.stft_colorbar.axis
            color_bar_font = QFont()
            color_bar_font.setPixelSize(20)
            color_bar_axis.setTickFont(color_bar_font)
            color_bar_axis.setTextPen("black")

    def calculate_spec(self):
        """Calculate spectrogram for single or multi-channel audio."""
        recorded_signal = self.data_struct.store_wave_data
        sample_rate = self.data_struct.sample_rate
        num_channels = self.data_struct.num_channels

        # Clear existing plots
        self._clear_plot_container()

        # Handle single vs multi-channel
        if num_channels == 1:
            # Single channel (backward compatible)
            signal_1d = recorded_signal if recorded_signal.ndim == 1 else recorded_signal[:, 0]
            self._plot_single_spectrogram(signal_1d, sample_rate, channel_index=0)
        else:
            # Multi-channel: create grid of spectrograms
            self._plot_multi_channel_spectrogram(recorded_signal, sample_rate, num_channels)

        self.set_color_font_size()

    def _plot_single_spectrogram(self, signal, sample_rate, channel_index):
        """Plot spectrogram for a single channel."""
        n_fft = self.analysis_config.get("n_fft", 2048)
        hop_length = self.analysis_config.get("hop_length", 256)
        color_map = self.analysis_config.get("color_map", "viridis")
        window_func = self.analysis_config.get("window_func", "hann")
        freq_scale_type = self.analysis_config.get("freq_scale_type", "linear")

        if freq_scale_type == "log":
            # CQT (log scale)
            fmin_cqt = librosa.note_to_hz("C1")
            CQT_complex, freqs, times = AudioThdFrequencyResponseAnalysis().compute_cqt(
                y=signal, sr=sample_rate, hop_length=hop_length, n_fft=n_fft, fmin=fmin_cqt
            )
            CQT_db = librosa.amplitude_to_db(np.abs(CQT_complex), ref=20e-6)
            Z = CQT_db.T

            target_ticks_hz = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
            major_ticks = []
            y_min_hz, y_max_hz = freqs.min(), freqs.max()
            for freq in target_ticks_hz:
                if y_min_hz <= freq <= y_max_hz:
                    label = f"{freq} Hz" if freq < 1000 else f"{freq/1000:.0f} kHz"
                    major_ticks.append((freq, label))
            custom_y_ticks = [major_ticks, []] if major_ticks else None

            cqt_plot_widget, self.stft_colorbar = plot_2d_image(
                x=times, y=freqs, z=Z,
                title="Spectrogram (Log Scale)",
                xlabel="Time (s)", ylabel="Frequency (Hz)",
                colormap=color_map,
                x_range=(times.min(), times.max()),
                y_range=(freqs.min(), freqs.max()),
                y_ticks=custom_y_ticks,
                background_color="white"
            )
            self.plot_container_layout.addWidget(cqt_plot_widget)
            self.current_plot_widget = cqt_plot_widget
        else:
            # STFT (linear scale)
            spec = np.abs(librosa.stft(y=signal, n_fft=n_fft, hop_length=hop_length, window=window_func))
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

    def _plot_multi_channel_spectrogram(self, multi_signal, sample_rate, num_channels):
        """Plot spectrograms for multiple channels in a grid layout using PyQtGraph."""
        from PyQt5.QtWidgets import QGridLayout

        n_fft = self.analysis_config.get("n_fft", 2048)
        hop_length = self.analysis_config.get("hop_length", 256)
        color_map = self.analysis_config.get("color_map", "viridis")
        window_func = self.analysis_config.get("window_func", "hann")
        freq_scale_type = self.analysis_config.get("freq_scale_type", "linear")

        # Determine grid layout: 2x2 for 3-4 channels, 2x1 for 2 channels
        if num_channels <= 2:
            nrows, ncols = num_channels, 1
        else:
            nrows, ncols = 2, 2

        # Create grid widget
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setContentsMargins(5, 5, 5, 5)
        grid_layout.setSpacing(10)

        for ch_idx in range(num_channels):
            row = ch_idx // ncols
            col = ch_idx % ncols

            # Extract channel signal
            signal_ch = multi_signal[:, ch_idx]

            # Create PlotWidget for this channel
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground("white")
            img_item = pg.ImageItem()
            plot_widget.addItem(img_item)

            # Calculate spectrogram
            spec = np.abs(librosa.stft(y=signal_ch, n_fft=n_fft, hop_length=hop_length, window=window_func))
            spec_dB = librosa.amplitude_to_db(spec, ref=20e-6)
            freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
            times = librosa.times_like(spec_dB, sr=sample_rate, hop_length=hop_length)

            # Set image data
            img_item.setImage(spec_dB.T, autoLevels=False)

            times_min, times_max = times.min(), times.max()
            freqs_min, freqs_max = freqs.min(), freqs.max()
            width = times_max - times_min
            height = freqs_max - freqs_min

            img_item.setRect(pg.QtCore.QRectF(times_min, freqs_min, width, height))

            # Setup plot
            scale_label = "Log Scale" if freq_scale_type == "log" else "Linear Scale"
            plot_widget.setTitle(f"Channel {ch_idx + 1} ({scale_label})")
            plot_widget.setLabel("bottom", "Time (s)")
            plot_widget.setLabel("left", "Frequency (Hz)")
            plot_widget.setLogMode(x=False, y=freq_scale_type == "log")

            # Setup colormap
            pos = np.linspace(0.0, 1.0, 256)
            colors = pg.colormap.get(color_map).getLookupTable(nPts=256)
            cmap = pg.ColorMap(pos, colors)
            db_min, db_max = np.nanmin(spec_dB), np.nanmax(spec_dB)

            lut = cmap.getLookupTable(nPts=256)
            img_item.setLookupTable(lut)
            img_item.setLevels([db_min, db_max])

            view_box = plot_widget.getViewBox()
            if view_box:
                view_box.setDefaultPadding(0.0)

            plot_widget.setXRange(times_min, times_max, padding=0)
            plot_widget.setYRange(freqs_min, freqs_max, padding=0)

            # Add colorbar
            plot_item = plot_widget.getPlotItem()
            if plot_item:
                colorbar = pg.ColorBarItem(values=(db_min, db_max), width=15, colorMap=cmap)
                colorbar.setImageItem(img_item, insert_in=plot_item)

            # Add to grid
            grid_layout.addWidget(plot_widget, row, col)

        self.plot_container_layout.addWidget(grid_widget)
        self.current_plot_widget = grid_widget



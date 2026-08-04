import csv
import json
import os
import sys

import librosa
import numpy as np
import pyqtgraph as pg
from librosa.core import spectrum
from librosa.feature import spectral
from librosa.sequence import dtw
from pyqtgraph import mkPen
from PyQt5.QtCore import Qt, QModelIndex
from PyQt5.QtGui import QIcon, QTextCursor, QTextCharFormat, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QTextEdit,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QLabel,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)
from scipy.signal import find_peaks

from base.core_algorithm.harmonic_distortion.weighted import apply_weighting_filter
from base.core_algorithm.sound_quality import run_sound_quality
from base.data_struct.data_deal_struct import DataDealStruct
from base.load_audio import load_audio_simple
from base.log_manager import LogManager
from base.model_runtime_validation import (
    build_blocked_ai_export_detail,
    should_validate_model_duration,
    validate_model_duration,
)
from base.predict_model import predict_from_audio
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from base.pre_processing.audio_peak_detection import peak_detection
from base.pre_processing.audio_equalizer import AudioEqualizer
from base.pre_processing.spl_runtime_config import (
    calculate_overall_spl,
    resolve_spl_unit,
)
from base.core_algorithm.response import (
    BandAnalysisResult,
    FftAnalyzer,
    FrequencyBandAnalyzer,
    FrequencyResponseAnalyzer,
    SplFrequencyAnalyzer,
)
from base.training_model_management import TrainingModelManagement
from base.utils.smooth import smooth
from base.utils.octave_smoothing import smooth_to_octave_grid
from consts import error_code, ui_style_const
from consts.acoustic_analysis.curve_style_consts import (
    LOWER_LIMIT_COLOR,
    MAIN_CURVE_COLOR,
    UPPER_LIMIT_COLOR,
)
from consts.running_consts import DEFAULT_DIR
from ui.curve_style import resolve_curve_colors
from ui.graph_widget import plot_2d_image, custom_log_tick_strings, LimitPlotUtils
from ui.plot_view import apply_plot_view_range
from ui.reference_spectrum_analysis_window import ReferenceSpectrumCompareWindow
from ui.ui_analysis_config.manual_limit_segments import (
    ManualLimitValidationError,
    limits_from_manual_segments,
)


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
        "SPLF": SplFrequency,
        "FR": Frequency,
        "RSC": ReferenceSpectrumCompareWindow,
        "HD": Distortion,
        "RB": RubAndBuzz,  # Rub & Buzz (high-order 10th-35th harmonic distortion)
        "PRB": PerceptualRubAndBuzz,  # Perceptual Rub & Buzz (2nd-35th harmonics, psychoacoustic loudness in phons)
        "AI": AI,
        "Spec": Spectrogram,
        "LP": LooseParticle,
        "PD": PeakDetection,
        "PM": PatternMatch,
        "ED": PipelinePdPm,
        "FBA": FrequencyBandAnalysis,
        "FFT": FftAnalysis,
        "LOUD": LoudnessAnalysis,
    }
    return class_mapping


def _resolve_golden_baseline_path(path: str):
    if not path or not isinstance(path, str):
        return None
    p = path.replace("\\", "/").strip()
    if not p:
        return None
    if os.path.isabs(p):
        return p
    return os.path.join(DEFAULT_DIR, p).replace("\\", "/")


def _load_golden_baseline_result(analysis_config: dict, title_name: str):
    """
    Load baseline result for a specific analysis item from golden baseline JSON.

    Expected JSON schema:
      {"items": {"<title_name>": {"type": "...", "result": {...}}}}
    """
    if not isinstance(analysis_config, dict):
        return None
    path = analysis_config.get("golden_sample_result_path")
    resolved = _resolve_golden_baseline_path(path)
    if not resolved or (not os.path.exists(resolved)):
        return None
    try:
        with open(resolved, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    items = payload.get("items")
    if not isinstance(items, dict):
        return None
    item = items.get(title_name)
    if not isinstance(item, dict):
        return None
    result = item.get("result")
    return result if isinstance(result, dict) else None


def resolve_analysis_channel_signal(data_struct: DataDealStruct, analysis_config: dict, title_name: str, strict: bool = True):
    cfg = analysis_config if isinstance(analysis_config, dict) else {}
    channel = int(cfg.get("analysis_channel", 0) or 0)

    multi = getattr(data_struct, "store_wave_data_multi", None)
    if multi is not None:
        arr = np.asarray(multi)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"[{title_name}] invalid multi-channel data shape: {arr.shape}")
        n_channels = int(arr.shape[1])
        if channel < 0 or channel >= n_channels:
            if strict:
                raise ValueError(
                    f"[{title_name}] analysis_channel={channel} out of range; recorded channels={n_channels}"
                )
            channel = 0
        return np.asarray(arr[:, channel], dtype=np.float32)

    mono = getattr(data_struct, "store_wave_data", None)
    if mono is None:
        raise ValueError(f"[{title_name}] missing recorded signal")
    arr = np.asarray(mono)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if strict and channel != 0:
        raise ValueError(
            f"[{title_name}] analysis_channel={channel} requires multi-channel recording, but only mono data available"
        )
    return np.asarray(arr, dtype=np.float32)


def _abs_deviation_curve(x_current, y_current, x_base, y_base):
    """
    Compute absolute deviation curve: abs(current - interp(base->current_x)).
    Points outside baseline x-range are set to NaN.
    """
    x_c = np.asarray(x_current, dtype=float)
    y_c = np.asarray(y_current, dtype=float)
    x_b = np.asarray(x_base, dtype=float)
    y_b = np.asarray(y_base, dtype=float)

    if x_c.size == 0 or y_c.size == 0 or x_b.size == 0 or y_b.size == 0:
        return y_c

    m = np.isfinite(x_b) & np.isfinite(y_b)
    x_b = x_b[m]
    y_b = y_b[m]
    if x_b.size < 2:
        return y_c

    sort_idx = np.argsort(x_b)
    x_b = x_b[sort_idx]
    y_b = y_b[sort_idx]
    x_b, uniq_idx = np.unique(x_b, return_index=True)
    y_b = y_b[uniq_idx]
    if x_b.size < 2:
        return y_c

    interp = np.interp(x_c, x_b, y_b)
    in_range = (x_c >= float(np.min(x_b))) & (x_c <= float(np.max(x_b)))
    interp = np.where(in_range, interp, np.nan)
    return (y_c - interp)


class AnalysisResultSummaryWindow(QWidget):
    """
    Summary window for DataDealStruct.analysis_result_dict.

    Displays a simple table: Analysis Item / Result(OK/NG).
    """

    def __init__(self, result_dict: dict[str, bool], title: str = "分析结果汇总"):
        super().__init__()
        self.setWindowTitle(title)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))

        self._overall_label = QLabel(self)
        overall_font = QFont()
        overall_font.setPixelSize(22)
        self._overall_label.setFont(overall_font)
        self._overall_label.setAlignment(Qt.AlignCenter)

        self._table = QTableWidget(self)
        font = QFont()
        font.setPixelSize(20)
        self._table.setFont(font)
        self._table.horizontalHeader().setFont(font)
        self._table.verticalHeader().setFont(font)
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["分析项", "偏差值", "结果"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        # Avoid default focus/selection highlight on show

        layout = QVBoxLayout()
        layout.addWidget(self._overall_label)
        layout.addWidget(self._table)
        self.setLayout(layout)

        self.set_results(result_dict)
        # Ensure no default selected cell/row
        self._table.clearSelection()
        self._table.setCurrentIndex(QModelIndex())

    def set_results(self, result_dict: dict[str, (bool, float)]):
        items = list(result_dict.items())
        # Stable order (alphabetical by name) for readability
        items.sort(key=lambda kv: kv[0])

        overall_ok = True
        judged_count = 0
        for _, (ok, _dev) in items:
            if ok is None:
                continue
            judged_count += 1
            if not bool(ok):
                overall_ok = False
                break
        overall_text = "N/A" if judged_count == 0 else ("OK" if overall_ok else "NG")
        self._overall_label.setText(f"最终结果：{overall_text}")
        if judged_count == 0:
            self._overall_label.setStyleSheet("color: rgb(90, 103, 120);")
        else:
            self._overall_label.setStyleSheet(
                "color: rgb(0, 128, 0);" if overall_ok else "color: rgb(200, 0, 0);"
            )

        self._table.setRowCount(len(items))
        for row, (name, (ok, deviation)) in enumerate(items):
            name_item = QTableWidgetItem(str(name))
            if "SPL" in name:
                deviation = f"{deviation:.2f} dB"
            elif "FR" in name:
                deviation = f"{deviation:.2f} dB"
            elif "RSC" in name:
                deviation = f"{deviation:.2f} dB"
            elif "FBA" in name:
                deviation = f"{deviation:.2f} dB"
            elif "FFT" in name:
                deviation = f"{deviation:.2f} dB"
            elif "PRB" in name:
                deviation = f"{deviation:.2f} phon"
            elif "HD" in name or "RB" in name:
                deviation = f"{deviation:.2f} %"
            deviation_item = QTableWidgetItem(str(deviation))
            result_text = "N/A" if ok is None else ("OK" if ok else "NG")
            result_item = QTableWidgetItem(result_text)
            result_item.setTextAlignment(Qt.AlignCenter)

            # color hint
            if ok is None:
                deviation_item.setForeground(QColor(90, 103, 120))
                result_item.setForeground(QColor(90, 103, 120))
            elif ok:
                deviation_item.setForeground(QColor(0, 128, 0))
                result_item.setForeground(QColor(0, 128, 0))
            else:
                deviation_item.setForeground(QColor(200, 0, 0))
                result_item.setForeground(QColor(200, 0, 0))

            self._table.setItem(row, 0, name_item)
            self._table.setItem(row, 1, deviation_item)
            self._table.setItem(row, 2, result_item)
        # Ensure no default selected cell/row after refreshing data
        self._table.clearSelection()
        self._table.setCurrentIndex(QModelIndex())


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

    @staticmethod
    def apply_plot_font_style(plot_widget, font_size: int = 20):
        font_size = ui_style_const.scale_size_px(font_size)
        font = QFont()
        font.setPixelSize(font_size)
        for axis_name in ("bottom", "left"):
            axis = plot_widget.getAxis(axis_name)
            axis.setTickFont(font)
            axis.setTextPen("black")
            axis.setLabel(
                axis.labelText,
                **{"font-size": f"{font_size}px"},
            )

    def _valid_v2pa_factor(self):
        try:
            factor = float(getattr(self, "v2pa_factor", None))
        except (TypeError, ValueError):
            factor = float("nan")
        if not np.isfinite(factor) or factor <= 0.0:
            QMessageBox.warning(
                self,
                "提示",
                "麦克风校准系数无效，请先完成声卡校准。",
            )
            return None
        return factor


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
        self.v2pa_factor = None
        self.title_name = title_name

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
            self.result = {"freq_value": [], "harmonic": [], "thd": [], "thd_raw": []}
            return self.result

        self.selected_harmonics = self.analysis_config.get("selected_labels", [])

        if not self.selected_harmonics:
            # No harmonics selected, nothing to calculate
            self.plot_graph([], [])
            self.result = {"freq_value": [], "harmonic": [], "thd": [], "thd_raw": []}
            return self.result

        # Get signals and metadata from data_struct
        try:
            recorded_signal = resolve_analysis_channel_signal(self.data_struct, self.analysis_config, self.title_name)
        except Exception as e:
            QMessageBox.warning(self, "提示", str(e))
            self.plot_graph([], [])
            self.result = {"freq_value": [], "harmonic": [], "thd": [], "thd_raw": []}
            return self.result
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
            "stimulus_method": stimulus_method,
            "stimulus_type": stimulus_info.get("stimulus_type", "linear"),
            "start_freq": stimulus_info.get("start_freq"),
            "stop_freq": stimulus_info.get("stop_freq"),
            "num_steps": stimulus_info.get("num_steps"),
            "total_time": stimulus_info.get("total_time"),
            "repeat_times": stimulus_info.get("repeat_times"),
            "sample_rate": sample_rate,
        }

        # Call the new three-phase architecture
        atfra = AudioThdFrequencyResponseAnalysis()
        thd_kwargs = {"stimulus_metadata": stimulus_metadata, "harmonic_orders": self.selected_harmonics}

        freq_value, harmonic, thd = atfra._calculate_thd_three_phase(recorded_signal, sample_rate, thd_kwargs)

        # Handle mirror chirps: average forward and backward sweeps
        if stimulus_method == "chirps" and "mirror" in stimulus_metadata["stimulus_type"]:
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

        # Apply 1/6 octave smoothing for chirp signals only
        if stimulus_method == "chirps":
            freq_value, thd = smooth_to_octave_grid(freq_value, thd, fraction=6, method="log")

        # Keep the absolute curve for export/saving (do not subtract golden baseline).
        thd_raw = thd

        # Golden sample baseline: use abs(current - golden) deviation curve
        if isinstance(self.analysis_config, dict) and self.analysis_config.get("golden_sample_checked"):
            baseline = _load_golden_baseline_result(self.analysis_config, self.title_name)
            if baseline:
                base_freq = baseline.get("freq_value")
                base_thd = baseline.get("thd")
                if base_freq is not None and base_thd is not None:
                    try:
                        thd = _abs_deviation_curve(freq_value, thd_raw, base_freq, base_thd)
                    except Exception:
                        pass
            else:
                QMessageBox.warning(self, "提示", "未找到黄金样本基准文件或基准数据，已按原始曲线分析")

        # Plot the results with threshold support
        self.plot_graph(freq_value, thd, self.analysis_config)

        # Convert to list format for result storage
        if isinstance(harmonic, np.ndarray):
            harmonic = harmonic.tolist()
        if isinstance(freq_value, np.ndarray):
            freq_value = freq_value.tolist()
        if isinstance(thd, np.ndarray):
            thd = thd.tolist()
        if isinstance(thd_raw, np.ndarray):
            thd_raw = thd_raw.tolist()

        self.result = {"freq_value": freq_value, "harmonic": harmonic, "thd": thd, "thd_raw": thd_raw}
        return self.result

    def plot_graph(self, freq_value, thd, analysis_config=None):
        """
        绑制 THD 曲线，支持可选的阈值限制。

        重构说明：
        - 有限制配置时：使用公共函数 setup_limit_plot() 统一绑图设置
        - 无限制配置时：保持原有逻辑（只绘制主曲线）
        - 超限段绘制：使用公共函数 plot_out_segments()

        Args:
            freq_value: 频率数组
            thd: THD 值数组
            analysis_config: 分析配置（可选，包含限制数据）
        """
        # Validate data
        valid_data = self.check_valid_data(freq_value) and self.check_valid_data(thd)

        # === With limit config: use setup_limit_plot() ===
        if analysis_config and analysis_config.get("limit_checked"):
            result = analysis_config.get("limit_data")
            if result and valid_data:
                csv_freq_list, csv_upper_list, csv_lower_list = result
                # Use common function for plot setup
                LimitPlotUtils.setup_limit_plot(
                    self.analysis_plot,
                    freq_value, thd,
                    csv_freq_list, csv_upper_list, csv_lower_list,
                    x_label="Frequency (Hz)",
                    y_label="Distortion(%)",
                    log_x=True,
                    curve_name="THD",
                )
                # THD specific title
                if self.selected_label is not None:
                    self.analysis_plot.setTitle(f"The Distortion of {self.selected_label.text()} order")
                # Highlight out-of-limit segments
                self._highlight_out_of_range_curve(freq_value, thd, csv_freq_list, csv_upper_list, csv_lower_list)
                return

        # === Without limit config: original logic ===
        self.analysis_plot.clear()
        if valid_data:
            self.analysis_plot.plot(freq_value, thd, pen=mkPen(color=(51, 196, 77), width=2), name="THD")
        if self.selected_label is not None:
            self.analysis_plot.setTitle(f"The Distortion of {self.selected_label.text()} order")
        self.analysis_plot.setLabel("left", "Distortion(%)")
        self.analysis_plot.setLabel("bottom", "Frequency (Hz)")
        self.analysis_plot.setLogMode(x=True, y=False)
        self.analysis_plot.showGrid(x=True, y=True)

    def _highlight_out_of_range_curve(self, freq_value, y_data, csv_freq_list, csv_upper_list, csv_lower_list):
        """
        Highlight out-of-limit segments in THD curve.

        Note:
        - Matching logic: kept here (nearest neighbor + index boundary filter)
        - Deviation calculation: kept here (THD has special logic for compatibility)
        - Out-of-limit plotting: uses LimitPlotUtils.plot_out_segments()

        Args:
            freq_value: Frequency array
            y_data: THD value array
            csv_freq_list: CSV frequency list
            csv_upper_list: Upper limit list
            csv_lower_list: Lower limit list
        """
        freq_arr = np.asarray(freq_value)
        y_arr = np.asarray(y_data)
        csv_freq_arr = np.asarray(csv_freq_list)

        max_csv_freq = max(csv_freq_list)
        min_csv_freq = min(csv_freq_list)
        freq_arr_capacity = freq_arr.size

        # === 1. THD specific matching and deviation calculation ===
        # Note: original logic retained due to THD's special deviation requirements
        deviation: float = 0.0
        no_deviation_flag = True
        out_mask = np.zeros(freq_arr_capacity, dtype=bool)

        for i, f in enumerate(freq_arr):
            # Find nearest CSV frequency point
            table_index = int(np.argmin(np.abs(csv_freq_arr - f)))
            if (i + 1) != freq_arr_capacity:
                next_table_index = int(np.argmin(np.abs(csv_freq_arr - freq_arr[i + 1])))
            else:
                next_table_index = table_index

            # Index boundary filter: skip if current and next point map to same boundary index
            if f < min_csv_freq and table_index == next_table_index:
                continue
            if f > max_csv_freq and table_index == next_table_index:
                continue

            upper_val = csv_upper_list[table_index]
            lower_val = csv_lower_list[table_index]

            is_out = False
            if not np.isnan(upper_val) and y_arr[i] > upper_val:
                if no_deviation_flag:
                    deviation = 0.0
                    no_deviation_flag = False
                deviation = max(deviation, abs(y_arr[i] - upper_val))
                is_out = True

            if not np.isnan(lower_val) and y_arr[i] < lower_val:
                if no_deviation_flag:
                    deviation = 0.0
                    no_deviation_flag = False
                deviation = max(deviation, abs(y_arr[i] - lower_val))
                is_out = True

            out_mask[i] = is_out

            if not is_out and no_deviation_flag:
                # THD special logic: calculate distance using current point's limits
                deviation_value = min(min(np.abs(y_arr - upper_val)), min(np.abs(y_arr - lower_val)))
                deviation = min(deviation, deviation_value) if deviation > 0 else deviation_value

        # === 2. Save result ===
        deviation = round(deviation, 2)
        is_ok = not np.any(out_mask)
        self.data_struct.analysis_result_dict[self.title_name] = (is_ok, deviation)

        # === 3. Plot out-of-limit segments using LimitPlotUtils ===
        LimitPlotUtils.plot_out_segments(self.analysis_plot, freq_arr, y_arr, out_mask)

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
    Perceptual Rub & Buzz analysis widget - displays SC-based perceptual indicator of harmonics (2nd-35th).

    Inherits from RubAndBuzz but uses the SoundCheck/Listen (SC) perceptual model.
    Y-axis shows TotalNL / EHS / TotalNL×EHS (PRB Index or PRB Loudness). Unlike standard RB (10th-35th),
    PRB analyzes the full harmonic range (2nd-35th).
    """

    def __init__(self, title_name):
        super().__init__(title_name)
        self._prb_curve_label = "感知失真指数"
        self._prb_y_label = "感知失真指数 (phon)"

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
            self.result = {"freq_value": [], "harmonic": [], "thd": [], "thd_raw": []}
            return self.result

        # PRB uses a fixed harmonic range (2nd-35th). The config dialog no longer exposes harmonic selection.
        self.selected_harmonics = list(range(2, 36))

        # PRB supports only the SoundCheck/Listen ("sc") method.
        prb_method = "sc"

        # Get signals and metadata from data_struct
        try:
            recorded_signal = resolve_analysis_channel_signal(self.data_struct, self.analysis_config, self.title_name)
        except Exception as e:
            QMessageBox.warning(self, "提示", str(e))
            self.plot_graph([], [])
            self.result = {"freq_value": [], "harmonic": [], "thd": [], "thd_raw": []}
            return self.result
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
            "stimulus_method": stimulus_method,
            "stimulus_type": stimulus_info.get("stimulus_type", "linear"),
            "start_freq": stimulus_info.get("start_freq"),
            "stop_freq": stimulus_info.get("stop_freq"),
            "num_steps": stimulus_info.get("num_steps"),
            "total_time": stimulus_info.get("total_time"),
            "repeat_times": stimulus_info.get("repeat_times"),
            "sample_rate": sample_rate,
        }

        # Call the PERCEPTUAL three-phase architecture
        v2pa_factor = self.v2pa_factor

        atfra = AudioThdFrequencyResponseAnalysis()
        masking_config = {}
        cfg_masking = self.analysis_config.get("masking_config")
        if isinstance(cfg_masking, dict):
            masking_config.update(cfg_masking)
        masking_config["prb_method"] = prb_method

        # Paper-aligned default: plot the combined indicator TotalNL×EHS unless explicitly overridden.
        sc_metric = str(masking_config.get("sc_metric", "totalnl_x_ehs")).strip().lower()
        if sc_metric not in {"totalnl", "totalnl_phons", "ehs", "totalnl_x_ehs"}:
            sc_metric = "totalnl_x_ehs"
        masking_config["sc_metric"] = sc_metric
        if sc_metric == "ehs":
            self._prb_curve_label = "EHS"
            self._prb_y_label = "EHS"
        elif sc_metric == "totalnl_x_ehs":
            self._prb_curve_label = "感知失真指数"
            self._prb_y_label = "感知失真指数 (phon)"
        else:
            self._prb_curve_label = "感知失真响度"
            self._prb_y_label = "感知失真响度 (phon)"
        thd_kwargs = {
            "stimulus_metadata": stimulus_metadata,
            "harmonic_orders": self.selected_harmonics,
            "masking_config": masking_config,
        }

        freq_value, harmonic, perceptual_loudness = atfra._calculate_perceptual_thd_three_phase(
            recorded_signal, sample_rate, thd_kwargs, v2pa_factor=v2pa_factor
        )

        # Handle mirror chirps: average forward and backward sweeps
        if stimulus_method == "chirps" and "mirror" in stimulus_metadata["stimulus_type"]:
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

        # Apply 1/6 octave smoothing for chirp signals only
        if stimulus_metadata["stimulus_method"] == "chirps":
            freq_value, perceptual_loudness = smooth_to_octave_grid(
                freq_value, perceptual_loudness, fraction=6, method="log"
            )

        # Keep the absolute curve for export/saving (do not subtract golden baseline).
        perceptual_loudness_raw = perceptual_loudness

        # Golden sample baseline: use abs(current - golden) deviation curve
        if isinstance(self.analysis_config, dict) and self.analysis_config.get("golden_sample_checked"):
            baseline = _load_golden_baseline_result(self.analysis_config, self.title_name)
            if baseline:
                base_freq = baseline.get("freq_value")
                base_y = baseline.get("thd")  # stored under 'thd' for backward compatibility
                if base_freq is not None and base_y is not None:
                    try:
                        perceptual_loudness = _abs_deviation_curve(
                            freq_value, perceptual_loudness_raw, base_freq, base_y
                        )
                    except Exception:
                        pass
            else:
                QMessageBox.warning(self, "提示", "未找到黄金样本基准文件或基准数据，已按原始曲线分析")

        # Plot the results with threshold support (Y-axis will be in phons)
        self.plot_graph(freq_value, perceptual_loudness, self.analysis_config)

        # Convert to list format for result storage
        if isinstance(harmonic, np.ndarray):
            harmonic = harmonic.tolist()
        if isinstance(freq_value, np.ndarray):
            freq_value = freq_value.tolist()
        if isinstance(perceptual_loudness, np.ndarray):
            perceptual_loudness = perceptual_loudness.tolist()
        if isinstance(perceptual_loudness_raw, np.ndarray):
            perceptual_loudness_raw = perceptual_loudness_raw.tolist()

        # Note: "thd" key name kept for backward compatibility, but contains phons
        self.result = {
            "freq_value": freq_value,
            "harmonic": harmonic,
            "thd": perceptual_loudness,
            "thd_raw": perceptual_loudness_raw,
        }
        return self.result

    def plot_graph(self, freq_value, perceptual_loudness, analysis_config=None):
        """
        Plot perceptual loudness (phons) and optionally apply limit curves.

        Note:
        - PRB inherits from Distortion, so it uses the same limit logic as THD:
          setup_limit_plot() + _highlight_out_of_range_curve() (nearest-neighbor matching).
        - This differs from SPLF/FR which use interpolation (check_interp_limits).
        """
        valid_data = self.check_valid_data(freq_value) and self.check_valid_data(perceptual_loudness)

        # === With limit config: use THD-style limit handling (nearest-neighbor) ===
        if analysis_config and analysis_config.get("limit_checked"):
            result = analysis_config.get("limit_data")
            if result and valid_data:
                csv_freq_list, csv_upper_list, csv_lower_list = result

                # 1) Plot main curve + limit curves (same as THD)
                LimitPlotUtils.setup_limit_plot(
                    self.analysis_plot,
                    freq_value, perceptual_loudness,
                    csv_freq_list, csv_upper_list, csv_lower_list,
                    x_label="Frequency (Hz)",
                    y_label=self._prb_y_label,
                    log_x=True,
                    curve_name=self._prb_curve_label,
                )

                if self.selected_label is not None:
                    self.analysis_plot.setTitle(
                        f"Perceived Loudness of {self.selected_label.text()} order"
                    )

                # 2) Use parent's _highlight_out_of_range_curve() for limit check + highlight
                #    This uses nearest-neighbor matching and highlights on original data points
                self._highlight_out_of_range_curve(
                    freq_value, perceptual_loudness,
                    csv_freq_list, csv_upper_list, csv_lower_list
                )
                return

        # === Without limit config (or missing limit_data): original plot logic ===
        self.analysis_plot.clear()
        if valid_data:
            self.analysis_plot.plot(
                freq_value,
                perceptual_loudness,
                pen=mkPen(color=(51, 196, 77), width=2),
                name=self._prb_curve_label,
            )
        if self.selected_label is not None:
            self.analysis_plot.setTitle(f"Perceived Loudness of {self.selected_label.text()} order")
        self.analysis_plot.setLabel("left", self._prb_y_label)
        self.analysis_plot.setLabel("bottom", "Frequency (Hz)")
        self.analysis_plot.setLogMode(x=True, y=False)
        self.analysis_plot.showGrid(x=True, y=True)


class Spl(AnalysisGraphWidget):
    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.v2pa_factor = None
        self.analysis_config = None
        self.result = {}
        self.title_name = title_name

        self.setWindowTitle(title_name)

    def _get_spl_label(self):
        """Get SPL y-axis label based on weighting type."""
        return f"SPL ({self._get_spl_unit()})"

    def _get_spl_unit(self):
        """Get the display unit for the configured frequency weighting."""
        weighting = self.analysis_config.get("weighting", "Z") if self.analysis_config else "Z"
        return resolve_spl_unit(weighting)

    def _set_overall_spl_title(self, overall_spl):
        if overall_spl is None:
            self.analysis_plot.setTitle("")
            return
        value_text = (
            "--"
            if not np.isfinite(overall_spl)
            else f"{float(overall_spl):.2f}"
        )
        self.analysis_plot.setTitle(
            f"总体声压级：{value_text} {self._get_spl_unit()}",
            size="14px",
            color="k",
        )

    def calculate_spl(self):
        # calculate Sound Pressure Level according to recorded_signal
        try:
            recorded_signal = resolve_analysis_channel_signal(self.data_struct, self.analysis_config, self.title_name)
        except Exception as e:
            QMessageBox.warning(self, "提示", str(e))
            return False
        sample_rate = self.data_struct.sample_rate
        reference_pressure = 20e-6
        window_size = 1201
        weighting = self.analysis_config.get("weighting", "Z") if self.analysis_config else "Z"
        if weighting and weighting.upper() not in ["NONE", "Z"]:
            recorded_signal = apply_weighting_filter(recorded_signal, sample_rate, weighting=weighting, zero_phase=False)
        show_overall_spl = bool(
            (self.analysis_config or {}).get(
                "show_overall_spl",
                False,
            )
        )
        overall_spl = None
        if show_overall_spl:
            overall_spl = calculate_overall_spl(
                recorded_signal,
                reference_pressure,
                v2pa_factor=self.v2pa_factor,
            )
        signal_spl = AudioThdFrequencyResponseAnalysis().spl_calculation(
            recorded_signal,
            reference_pressure,
            window_size=window_size,
            v2pa_factor=self.v2pa_factor,
            trim_edges=True,
        )
        start_index = 0 if len(signal_spl) == len(recorded_signal) else window_size // 2
        signal_duration = (np.arange(len(signal_spl), dtype=float) + float(start_index)) / float(sample_rate)

        if self.analysis_config and self.analysis_config.get("smooth_checked"):
            # NOTE: Do not apply RMS smoothing on dB values (squaring negatives turns silence into ~100 dB).
            signal_spl = smooth(signal_spl, window_size=1102, method="savgol")
        limit_checked = self.analysis_config.get("limit_checked")
        if limit_checked:
            result = self.analysis_config.get("limit_data")
            if not result:
                return False
            csv_time_list, csv_upper_list, csv_lower_list = result
            self.plot_spl_with_limits(signal_duration, signal_spl, csv_time_list, csv_upper_list, csv_lower_list)
        else:
            self.plot_spl(signal_duration, signal_spl)
        apply_plot_view_range(
            self.analysis_plot,
            self.analysis_config or {},
        )
        self._set_overall_spl_title(overall_spl)
        self.result = {
            "signal_duration": signal_duration.tolist(),
            "recorded_signal": recorded_signal.tolist(),
            "signal_spl": signal_spl.tolist(),
        }
        if show_overall_spl:
            self.result["overall_spl"] = overall_spl
        return self.result

    def plot_spl_with_limits(self, signal_duration, signal_spl, csv_time_list, csv_upper_list, csv_lower_list):
        """
        Plot SPL time-domain curve and highlight out-of-limit segments.

        Note:
        - Uses LimitPlotUtils.setup_limit_plot() for canvas, curves, and axis setup
        - Matching logic: kept here (nearest neighbor + time threshold, SPL specific)
        - Limit comparison: uses LimitPlotUtils.compare_with_limits()
        - Out-of-limit plotting: uses LimitPlotUtils.plot_out_segments()

        Args:
            signal_duration: Time axis array
            signal_spl: SPL value array
            csv_time_list: CSV time point list
            csv_upper_list: Upper limit list
            csv_lower_list: Lower limit list
        """
        # === 1. Common plot setup (clear, draw main curve and limit curves, set axes) ===
        # Note: SPL time-domain uses linear scale (log_x=False), Y label is dynamic
        LimitPlotUtils.setup_limit_plot(
            self.analysis_plot,
            signal_duration, signal_spl,
            csv_time_list, csv_upper_list, csv_lower_list,
            x_label="Time (s)",
            y_label=self._get_spl_label(),
            log_x=False,
            curve_colors=resolve_curve_colors(
                self.analysis_config or {}
            ),
        )

        # === 2. Matching: nearest neighbor + time threshold filter (SPL specific) ===
        max_time_diff = 0.01  # 10 ms threshold
        sig_t = np.asarray(signal_duration, dtype=float)
        sig_spl = np.asarray(signal_spl, dtype=float)
        csv_t = np.asarray(csv_time_list, dtype=float)
        csv_u = np.asarray(csv_upper_list, dtype=float)
        csv_l = np.asarray(csv_lower_list, dtype=float)

        # Use searchsorted for vectorized nearest index lookup (O(N*logM) instead of O(N*M))
        insert_idx = np.searchsorted(csv_t, sig_t)
        insert_idx = np.clip(insert_idx, 0, len(csv_t) - 1)
        left_idx = np.clip(insert_idx - 1, 0, len(csv_t) - 1)
        dist_right = np.abs(csv_t[insert_idx] - sig_t)
        dist_left = np.abs(csv_t[left_idx] - sig_t)
        nearest_idx = np.where(dist_left < dist_right, left_idx, insert_idx)

        # Time threshold filter: points exceeding threshold are invalid
        nearest_time = csv_t[nearest_idx]
        valid_mask = np.abs(nearest_time - sig_t) <= max_time_diff

        # Get upper/lower limits for each signal point
        upper_at = csv_u[nearest_idx]
        lower_at = csv_l[nearest_idx]

        # === 5. Limit comparison using LimitPlotUtils ===
        out_mask, deviation, is_ok = LimitPlotUtils.compare_with_limits(sig_spl, upper_at, lower_at, valid_mask)

        # === 6. Save result ===
        self.data_struct.analysis_result_dict[self.title_name] = (is_ok, deviation)

        # === 7. Plot out-of-limit segments using LimitPlotUtils ===
        LimitPlotUtils.plot_out_segments(self.analysis_plot, sig_t, sig_spl, out_mask)

    def plot_spl(self, signal_duration, signal_spl):
        self.analysis_plot.clear()
        main_color = resolve_curve_colors(
            self.analysis_config or {}
        )["main_curve_color"]
        self.analysis_plot.plot(
            signal_duration,
            signal_spl,
            pen=mkPen(color=main_color, width=2),
        )
        self.analysis_plot.setLabel("left", self._get_spl_label())
        self.analysis_plot.setLabel("bottom", "Time (s)")
        self.analysis_plot.showGrid(x=True, y=True)


class SplFrequency(AnalysisGraphWidget):
    """
    SPL vs Frequency analysis (output level curve for the current drive level).

    Requires stimulus_info (step/chirp) to map each segment/frame to a frequency.
    """

    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.v2pa_factor = None
        self.analysis_config = None
        self.result = {}
        self.title_name = title_name
        self.setWindowTitle(title_name)

    def calculate_spl(self):
        try:
            recorded_signal = resolve_analysis_channel_signal(self.data_struct, self.analysis_config, self.title_name)
        except Exception as e:
            QMessageBox.warning(self, "提示", str(e))
            self.plot_spl_frequency([], [])
            self.result = {"frequency_list": [], "spl_db": [], "spl_db_raw": []}
            return self.result
        sample_rate = self.data_struct.sample_rate
        stimulus_info = self.data_struct.stimulus_info or {}
        analysis_config = self.analysis_config or {}

        if recorded_signal is None or sample_rate is None or not stimulus_info:
            self.plot_spl_frequency([], [])
            self.result = {"frequency_list": [], "spl_db": [], "spl_db_raw": []}
            return self.result

        stimulus_method = stimulus_info.get("stimulus_method", "steps")
        if stimulus_method == "chirp":
            stimulus_method = "chirps"
        elif stimulus_method == "step":
            stimulus_method = "steps"

        stimulus_metadata = {
            "stimulus_method": stimulus_method,
            "stimulus_type": stimulus_info.get("stimulus_type", "linear"),
            "start_freq": stimulus_info.get("start_freq"),
            "stop_freq": stimulus_info.get("stop_freq"),
            "num_steps": stimulus_info.get("num_steps"),
            "total_time": stimulus_info.get("total_time"),
            "repeat_times": stimulus_info.get("repeat_times"),
            "sample_rate": sample_rate,
        }

        try:
            analyzer = SplFrequencyAnalyzer(sample_rate=int(sample_rate))
            result = analyzer.compute(
                recorded_signal,
                stimulus_metadata=stimulus_metadata,
                v2pa_factor=self.v2pa_factor,
                splf_calc_mode=analysis_config.get("splf_calc_mode", "fundamental"),
            )
        except Exception as e:
            QMessageBox.warning(self, "提示", f"声压级-频率计算失败: {str(e)[:200]}")
            self.plot_spl_frequency([], [])
            self.result = {"frequency_list": [], "spl_db": [], "spl_db_raw": []}
            return self.result

        frequency_list = np.asarray(result.frequencies_hz, dtype=float)
        spl_db = np.asarray(result.spl_db, dtype=float)

        # Optional octave smoothing (frequency-domain).
        octave_smoothing = analysis_config.get("octave_smoothing", None)
        if octave_smoothing is None and analysis_config.get("smooth_checked"):
            octave_smoothing = 6
        try:
            octave_smoothing = int(octave_smoothing) if octave_smoothing is not None else 0
        except Exception:
            octave_smoothing = 0

        if octave_smoothing in {1, 3, 6, 12, 24, 48} and spl_db.size > 1:
            try:


                freq = np.asarray(frequency_list, dtype=float)
                val = np.asarray(spl_db, dtype=float)
                mask = np.isfinite(freq) & np.isfinite(val) & (freq > 0.0)
                freq = freq[mask]
                val = val[mask]
                if freq.size > 1:
                    sort_idx = np.argsort(freq)
                    freq = freq[sort_idx]
                    val = val[sort_idx]
                    freq, unique_idx = np.unique(freq, return_index=True)
                    val = val[unique_idx]

                    frequency_list, spl_db = smooth_to_octave_grid(freq, val, fraction=octave_smoothing, method="log")
            except Exception:
                pass

        # Keep the absolute curve for export/saving (do not subtract golden baseline).
        spl_db_raw = spl_db

        # Golden sample baseline: use abs(current - golden) deviation curve
        if analysis_config.get("golden_sample_checked"):
            baseline = _load_golden_baseline_result(analysis_config, self.title_name)
            if baseline:
                base_freq = baseline.get("frequency_list")
                base_spl = baseline.get("spl_db")
                if base_freq is not None and base_spl is not None:
                    try:
                        spl_db = _abs_deviation_curve(frequency_list, spl_db_raw, base_freq, base_spl)
                    except Exception:
                        pass
            else:
                QMessageBox.warning(self, "提示", "未找到黄金样本基准文件或基准数据，已按原始曲线分析")

        limit_checked = analysis_config.get("limit_checked")
        if limit_checked:
            result = analysis_config.get("limit_data")
            if not result:
                return False
            csv_freq_list, csv_upper_list, csv_lower_list = result
            self.plot_spl_frequency_with_limits(frequency_list, spl_db, csv_freq_list, csv_upper_list, csv_lower_list)
        else:
            self.plot_spl_frequency(frequency_list, spl_db)

        apply_plot_view_range(
            self.analysis_plot,
            self.analysis_config or {},
        )
        self.result = {
            "frequency_list": frequency_list.tolist(),
            "spl_db": spl_db.tolist(),
            "spl_db_raw": spl_db_raw.tolist() if isinstance(spl_db_raw, np.ndarray) else spl_db_raw,
        }
        return self.result

    def plot_spl_frequency_with_limits(self, frequency_list, spl_db, csv_freq_list, csv_upper_list, csv_lower_list):
        """
        Plot SPLF (SPL-Frequency) curve and highlight out-of-limit segments.

        Note:
        - Uses LimitPlotUtils.setup_limit_plot() for canvas, curves, and axis setup
        - Uses LimitPlotUtils.check_interp_limits() for interpolation and limit check
        - Uses LimitPlotUtils.plot_out_segments() for out-of-limit plotting

        Args:
            frequency_list: Frequency array
            spl_db: SPL value array
            csv_freq_list: CSV frequency list
            csv_upper_list: Upper limit list
            csv_lower_list: Lower limit list
        """
        # === 1. Preprocess: sort data by frequency ===
        freq_arr = np.asarray(frequency_list, dtype=float)
        spl_arr = np.asarray(spl_db, dtype=float)
        mask = np.isfinite(freq_arr) & np.isfinite(spl_arr) & (freq_arr > 0)
        freq_valid = freq_arr[mask]
        spl_valid = spl_arr[mask]
        if freq_valid.size > 1:
            sort_idx = np.argsort(freq_valid)
            freq_valid = freq_valid[sort_idx]
            spl_valid = spl_valid[sort_idx]

        # === 2. Common plot setup (use sorted data for both green and red curves) ===
        LimitPlotUtils.setup_limit_plot(
            self.analysis_plot,
            freq_valid, spl_valid,
            csv_freq_list, csv_upper_list, csv_lower_list,
            x_label="Frequency (Hz)",
            y_label="SPL (dB)",
            log_x=True,
            curve_colors=resolve_curve_colors(
                self.analysis_config or {}
            ),
        )

        # === 3. Limit check using LimitPlotUtils ===
        try:
            out_mask, plot_x, plot_y, deviation, is_ok = LimitPlotUtils.check_interp_limits(
                freq_valid,
                spl_valid,
                np.asarray(csv_freq_list, dtype=float),
                np.asarray(csv_upper_list, dtype=float),
                np.asarray(csv_lower_list, dtype=float),
            )
        except Exception:
            is_ok, deviation = False, 0.0
            out_mask = np.zeros(len(freq_valid), dtype=bool)
            plot_x, plot_y = freq_valid, spl_valid

        # === 4. Save result and plot out-of-limit segments ===
        self.data_struct.analysis_result_dict[self.title_name] = (is_ok, deviation)
        LimitPlotUtils.plot_out_segments(self.analysis_plot, plot_x, plot_y, out_mask)

    def plot_spl_frequency(self, frequency_list, spl_db):
        self.analysis_plot.clear()
        main_color = resolve_curve_colors(
            self.analysis_config or {}
        )["main_curve_color"]
        self.analysis_plot.plot(
            frequency_list,
            spl_db,
            pen=mkPen(color=main_color, width=2),
        )
        self.analysis_plot.setLabel("left", "SPL (dB)")
        self.analysis_plot.setLabel("bottom", "Frequency (Hz)")
        self.analysis_plot.setLogMode(x=True, y=False)
        self.analysis_plot.showGrid(x=True, y=True)


class Frequency(AnalysisGraphWidget):

    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.smooth_flag = False
        self.temp_frequency_list = None
        self.v2pa_factor = None
        self.analysis_config = None
        self.result = {}
        self.title_name = title_name

        self.setWindowTitle(title_name)

    def calculate_fr(self):
        stimulus_signal = self.data_struct.stimulus_data
        try:
            recorded_signal = resolve_analysis_channel_signal(self.data_struct, self.analysis_config, self.title_name)
        except Exception as e:
            QMessageBox.warning(self, "提示", str(e))
            self.plot_fr([], [])
            self.result = {"fr": [], "frequency_list": [], "fr_raw": []}
            return self.result
        sr = self.data_struct.sample_rate
        stimulus_info = self.data_struct.stimulus_info or {}
        analysis_config = self.analysis_config or {}

        if stimulus_signal is None or recorded_signal is None or sr is None:
            self.plot_fr([], [])
            self.result = {"fr": [], "frequency_list": [], "fr_raw": []}
            return self.result

        # Convert stimulus_info to metadata (shared convention with harmonic distortion pipeline).
        stimulus_method = stimulus_info.get("stimulus_method", "steps")
        if stimulus_method == "chirp":
            stimulus_method = "chirps"
        elif stimulus_method == "step":
            stimulus_method = "steps"

        stimulus_metadata = {
            "stimulus_method": stimulus_method,
            "stimulus_type": stimulus_info.get("stimulus_type", "linear"),
            "start_freq": stimulus_info.get("start_freq"),
            "stop_freq": stimulus_info.get("stop_freq"),
            "num_steps": stimulus_info.get("num_steps"),
            "total_time": stimulus_info.get("total_time"),
            "repeat_times": stimulus_info.get("repeat_times"),
            "sample_rate": sr,
        }

        try:
            analyzer = FrequencyResponseAnalyzer(sample_rate=int(sr))
            fr_result = analyzer.compute(
                stimulus_signal,
                recorded_signal,
                stimulus_metadata=stimulus_metadata,
                method="sweep_wiener",
            )

            frequency_list = np.asarray(fr_result.frequencies_hz, dtype=float)
            fr = np.asarray(fr_result.magnitude_db, dtype=float)

            # Optional octave smoothing (frequency-domain).
            octave_smoothing = analysis_config.get("octave_smoothing", None)
            if octave_smoothing is None and analysis_config.get("smooth_checked"):
                octave_smoothing = 6
            try:
                octave_smoothing = int(octave_smoothing) if octave_smoothing is not None else 0
            except Exception:
                octave_smoothing = 0

            if octave_smoothing in {1, 3, 6, 12, 24, 48} and fr.size > 1:
                try:
                    freq = np.asarray(frequency_list, dtype=float)
                    val = np.asarray(fr, dtype=float)
                    mask = np.isfinite(freq) & np.isfinite(val) & (freq > 0.0)
                    freq = freq[mask]
                    val = val[mask]
                    if freq.size > 1:
                        sort_idx = np.argsort(freq)
                        freq = freq[sort_idx]
                        val = val[sort_idx]
                        freq, unique_idx = np.unique(freq, return_index=True)
                        val = val[unique_idx]

                        frequency_list, fr = smooth_to_octave_grid(freq, val, fraction=octave_smoothing, method="log")
                except Exception:
                    pass
        except Exception as e:
            QMessageBox.warning(self, "提示", f"频响计算失败: {str(e)[:200]}")
            self.plot_fr([], [])
            self.result = {"fr": [], "frequency_list": [], "fr_raw": []}
            return self.result

        # Keep the absolute curve for export/saving (do not subtract golden baseline).
        fr_raw = fr

        # Golden sample baseline: use abs(current - golden) deviation curve
        if analysis_config.get("golden_sample_checked"):
            baseline = _load_golden_baseline_result(analysis_config, self.title_name)
            if baseline:
                base_freq = baseline.get("frequency_list")
                base_fr = baseline.get("fr")
                if base_freq is not None and base_fr is not None:
                    try:
                        fr = _abs_deviation_curve(frequency_list, fr_raw, base_freq, base_fr)
                    except Exception:
                        pass
            else:
                QMessageBox.warning(self, "提示", "未找到黄金样本基准文件或基准数据，已按原始曲线分析")
        limit_checked = analysis_config.get("limit_checked")
        if limit_checked:
            result = self.analysis_config.get("limit_data")
            if not result:
                return False
            csv_freq_list, csv_upper_list, csv_lower_list = result
            self.plot_fr_with_limits(frequency_list, fr, csv_freq_list, csv_upper_list, csv_lower_list)
        else:
            self.plot_fr(frequency_list, fr)
        self.result = {
            "fr": fr.tolist(),
            "frequency_list": frequency_list.tolist(),
            "fr_raw": fr_raw.tolist() if isinstance(fr_raw, np.ndarray) else fr_raw,
        }
        return self.result

    @staticmethod
    def load_excel_limit(excel_path):
        if not excel_path:
            QMessageBox.warning(None, "提示", f"Excel路径为空, 请选择一个Excel文件路径！")
            return None
        ext = os.path.splitext(excel_path)[1].lower()
        if ext == ".csv":
            with open(excel_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
        else:
            QMessageBox.warning(None, "提示", f"不支持对这种Excel格式的分析:\n{excel_path}")
            return None

        if not rows or len(rows) == 0:
            QMessageBox.warning(None, "提示", f"CSV文件为空或格式不正确:\n{excel_path}")
            return None

        csv_freq_list, csv_upper_list, csv_lower_list = [], [], []
        lenth = len(rows[0])
        if lenth == 3 and rows[0][1] == "upperbound":
            upperbound = True
        elif lenth == 3 and rows[0][1] == "lowerbound":
            upperbound = False
        elif lenth == 2 and rows[0][1] == "upperbound":
            upperbound = True
        elif lenth == 2 and rows[0][1] == "lowerbound":
            upperbound = False
        else:
            QMessageBox.warning(None, "提示", "Excel/CSV 格式不符合要求!")
            return None
        for index, row in enumerate(rows[1:], start=2):
            csv_line_no = index
            if lenth == 3 and upperbound:
                try:
                    fval = float(row[0])
                    uval = float(row[1])
                    lval = float(row[2])
                except ValueError:
                    QMessageBox.warning(None, "提示", f"CSV 数据错误:第 {csv_line_no} 行存在空值或非数字,无法解析\n")
                    return None
                csv_freq_list.append(fval)
                csv_upper_list.append(uval)
                csv_lower_list.append(lval)
            elif lenth == 3 and not upperbound:
                try:
                    fval = float(row[0])
                    uval = float(row[2])
                    lval = float(row[1])
                except ValueError:
                    QMessageBox.warning(None, "提示", f"CSV 数据错误:第 {csv_line_no} 行存在空值或非数字,无法解析\n")
                    return None
                csv_freq_list.append(fval)
                csv_upper_list.append(uval)
                csv_lower_list.append(lval)
            elif lenth == 2 and upperbound:
                try:
                    fval = float(row[0])
                    uval = float(row[1])
                except ValueError:
                    QMessageBox.warning(None, "提示", f"CSV 数据错误:第 {csv_line_no} 行存在空值或非数字,无法解析\n")
                    return None
                csv_freq_list.append(fval)
                csv_upper_list.append(uval)
                csv_lower_list.append(np.nan)
            elif lenth == 2 and not upperbound:
                try:
                    fval = float(row[0])
                    lval = float(row[1])
                except ValueError:
                    QMessageBox.warning(None, "提示", f"CSV 数据错误:第 {csv_line_no} 行存在空值或非数字,无法解析\n")
                    return None
                csv_freq_list.append(fval)
                csv_upper_list.append(np.nan)
                csv_lower_list.append(lval)
        for i, (x, u, l) in enumerate(zip(csv_freq_list, csv_upper_list, csv_lower_list)):
            if (u is not None) and (l is not None) and (not np.isnan(u)) and (not np.isnan(l)):
                if l > u:
                    QMessageBox.warning(
                        None,
                        "提示",
                        f"CSV 上下限配置错误：下限不能大于上限。\n"
                        f"位置: 第{i+2}条数据, X={x}\n"
                        f"lower={l}, upper={u}\n"
                        f"文件: {excel_path}",
                    )
                    return None
        return (
            np.asarray(csv_freq_list, dtype=float),
            np.asarray(csv_upper_list, dtype=float),
            np.asarray(csv_lower_list, dtype=float),
        )

    def plot_fr_with_limits(self, frequency_list, fr, csv_freq_list, csv_upper_list, csv_lower_list):
        """
        Plot Frequency Response (FR) curve and highlight out-of-limit segments.

        Note:
        - Uses LimitPlotUtils.setup_limit_plot() for canvas, curves, and axis setup
        - Uses LimitPlotUtils.check_interp_limits() for interpolation and limit check
        - Uses LimitPlotUtils.plot_out_segments() for out-of-limit plotting

        Args:
            frequency_list: Frequency array
            fr: Frequency response value array
            csv_freq_list: CSV frequency list
            csv_upper_list: Upper limit list
            csv_lower_list: Lower limit list
        """
        # fr_disp = fr + 94 + self.v2pa_factor  # Todo: modify later
        fr_disp = fr

        # === 1. Preprocess: sort data by frequency ===
        freq_arr = np.asarray(frequency_list, dtype=float)
        fr_arr = np.asarray(fr_disp, dtype=float)
        mask = np.isfinite(freq_arr) & np.isfinite(fr_arr) & (freq_arr > 0)
        freq_valid = freq_arr[mask]
        fr_valid = fr_arr[mask]
        if freq_valid.size > 1:
            sort_idx = np.argsort(freq_valid)
            freq_valid = freq_valid[sort_idx]
            fr_valid = fr_valid[sort_idx]

        # === 2. Common plot setup (use sorted data for both green and red curves) ===
        LimitPlotUtils.setup_limit_plot(
            self.analysis_plot,
            freq_valid, fr_valid,
            csv_freq_list, csv_upper_list, csv_lower_list,
            x_label="Frequency (Hz)",
            y_label="Amplitude (dB)",
            log_x=True,
        )

        # === 3. Limit check using LimitPlotUtils ===
        try:
            out_mask, plot_x, plot_y, deviation, is_ok = LimitPlotUtils.check_interp_limits(
                freq_valid,
                fr_valid,
                np.asarray(csv_freq_list, dtype=float),
                np.asarray(csv_upper_list, dtype=float),
                np.asarray(csv_lower_list, dtype=float),
            )
        except Exception:
            is_ok, deviation = False, 0.0
            out_mask = np.zeros(len(freq_valid), dtype=bool)
            plot_x, plot_y = freq_valid, fr_valid

        # === 4. Save result and plot out-of-limit segments ===
        self.data_struct.analysis_result_dict[self.title_name] = (is_ok, deviation)
        LimitPlotUtils.plot_out_segments(self.analysis_plot, plot_x, plot_y, out_mask)

    def plot_fr(self, frequency_list, fr):
        self.analysis_plot.clear()
        # fr = fr + 94 + self.v2pa_factor  # Todo: modify later
        self.analysis_plot.plot(frequency_list, fr, pen=mkPen(color=(51, 196, 77), width=2))
        self.analysis_plot.setLabel("left", "Amplitude (dB)")
        self.analysis_plot.setLabel("bottom", "Frequency (Hz)")
        self.analysis_plot.setLogMode(x=True, y=False)
        self.analysis_plot.showGrid(x=True, y=True)


class AI(QWidget):
    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.analysis_config = None
        self.result = None
        self.export_detail = None
        self.default_logger = LogManager.set_log_handler("core")
        self.title_name = title_name

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

    def calculate_ai_scores(self, mode, analysis_config, acq_mode=None):
        model_name = self.analysis_config["analyse_model_name"]
        self.result = None
        self.export_detail = {}
        code, result = self.get_model_info(model_name, self.default_logger)
        if code != error_code.OK or not os.path.exists(result[0]):
            self.ai_analyse_score_textedit.setPlainText("模型不存在，请重新选择")
            return
        model_path, config_path = result
        try:
            ai_signal = resolve_analysis_channel_signal(self.data_struct, self.analysis_config, self.title_name)
        except Exception as e:
            self.ai_analyse_score_textedit.setPlainText(str(e))
            return

        if should_validate_model_duration(mode, acq_mode=acq_mode):
            matched, message = validate_model_duration(
                model_name,
                len(ai_signal),
                sample_rate=getattr(self.data_struct, "sample_rate", 0),
                config_path=config_path,
            )
            if not matched:
                self.ai_analyse_score_textedit.setPlainText(message)
                self.export_detail = build_blocked_ai_export_detail(
                    model_name,
                    reason="duration_mismatch",
                    message=message,
                )
                QMessageBox.information(self, "提示", message)
                return
            self.default_logger.info("The model matches the audio duration. Starting analysis...")
        kwargs = {"config_path": config_path}
        result_text = self.model_predict(model_path, model_name, signal_data=ai_signal, **kwargs)
        self.ai_analyse_score_textedit.setPlainText(result_text)
        self.highlight_keywords("ng", self.ai_analyse_score_textedit)

    def model_predict(self, model_path, model_name, signal_data=None, **kwargs):
        if signal_data is None:
            signal_data = resolve_analysis_channel_signal(self.data_struct, self.analysis_config, self.title_name)
        ret_str, pred_config = predict_from_audio(
            signals=[np.array(signal_data, dtype=np.float32)],
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
        deviation = round(abs(float(predict_result[0][2]) - float(pred_config.get("acc_req", 0.5))), 2)
        is_passed_bool = True if predict_label == "OK" else False
        self.data_struct.analysis_result_dict[self.title_name] = (is_passed_bool, deviation)
        self.result = predict_label
        self.export_detail = {
            "label": predict_label,
            "ok_score": round(ok_scores, 2),
            "ng_score": round(ng_scores, 2),
            "model_name": model_name,
        }
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
        self.v2pa_factor = None
        self.analysis_config = None
        self.title_name = title_name
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
        recorded_signal = resolve_analysis_channel_signal(self.data_struct, self.analysis_config, self.title_name)
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
        self.title_name = title_name
        self.lp_num_label = QLabel("LP 数量: %s" % self.result)
        self.status_label = QLabel()
        self.v2pa_factor = None
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
        recorded_signal = resolve_analysis_channel_signal(self.data_struct, self.analysis_config, self.title_name)
        filtered_spl, deviation = AudioThdFrequencyResponseAnalysis.calculate_loose_particle_spl(
            recorded_signal, self.analysis_config.get("cutoff_freq"), self.data_struct.sample_rate, 67, self.v2pa_factor
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
        self.analysis_plot.plot(signal_duration, amplitude, pen=mkPen(color=(51, 196, 77), width=2))
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
        pen = pg.mkPen(color="orange", width=2)
        out_range_points = np.array(out_range_points) - deviation
        out_range_plot = pg.PlotDataItem(signal_duration, out_range_points, pen=pen)
        self.analysis_plot.addItem(out_range_plot)


class PeakDetection(AnalysisGraphWidget):
    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.analysis_config = None
        self.result = None
        self.v2pa_factor = None
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
                np.asarray(recorded_signal, dtype=np.float64),
                int(sample_rate),
                self.analysis_config,
                v2pa_factor=self.v2pa_factor,
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
            spl_series = AudioThdFrequencyResponseAnalysis.spl_calculation(
                recorded_signal, v2pa_factor=self.v2pa_factor
            )
        time_axis = np.linspace(0, len(spl_series) / sample_rate, len(spl_series))
        self.analysis_plot.plot(time_axis, spl_series, pen=mkPen(color=(51, 196, 77), width=2))

        peak_times = self.result.get("peaks_time_sec", [])
        if peak_times:
            peak_indices = np.clip((np.array(peak_times) * sample_rate).astype(int), 0, len(spl_series) - 1)
            peak_values = spl_series[peak_indices]
            scatter = pg.ScatterPlotItem(
                x=np.array(peak_times), y=peak_values, pen=pg.mkPen(None), brush=pg.mkBrush(200, 0, 0, 200), size=8
            )
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
            ui_style_const.qlabel_style + ui_style_const.qlineedit_style + ui_style_const.qtextedit_style
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
            self.target_data = AudioEqualizer.apply_equalizer(
                self.target_data, self.sample_rate, start_freq=start_freq, end_freq=end_freq
            )
            self.pattern_data = AudioEqualizer.apply_equalizer(
                self.pattern_data, self.sample_rate, start_freq=start_freq, end_freq=end_freq
            )
        feature_type = self.analysis_config.get("feature_type", "mfcc")
        target_features, pattern_features = self.feature_extraction_handle(
            self.target_data, self.pattern_data, feature_type
        )

        result_dict = self.algorithm_handle(
            algorithm_name,
            target_features,
            pattern_features,
            distance_measure_method=similarity_metric,
            threshold=threshold,
        )
        if result_dict:
            is_match = result_dict["is_match"]
            score = result_dict["score"]
            used_threshold = result_dict["threshold"]

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

            return {"is_match": is_match, "score": similarity, "threshold": threshold}
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
                target_data = np.abs(np.fft.fft(target_data) / target_len)[: target_len // 2]
                pattern_data = np.abs(np.fft.fft(pattern_data) / pattern_len)[: pattern_len // 2]
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
        self.v2pa_factor = None
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
            ui_style_const.qlabel_style + ui_style_const.qlineedit_style + ui_style_const.qtextedit_style
        )

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
        pd_instance.v2pa_factor = self.v2pa_factor
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
            spl_series = AudioThdFrequencyResponseAnalysis.spl_calculation(
                recorded_signal, v2pa_factor=self.v2pa_factor
            )
        time_axis = np.linspace(0, len(spl_series) / sample_rate, len(spl_series))
        plot_item.plot(time_axis, spl_series, pen=mkPen(color=(51, 196, 77), width=2))
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


class FftAnalysis(AnalysisGraphWidget):
    """Welch FFT 频谱分析窗口。"""

    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.v2pa_factor = None
        self.analysis_config = None
        self.result = {}
        self.title_name = title_name
        self.setWindowTitle(title_name)

    def calculate_fft(self):
        """执行 FFT 分析、可选背景对比和阈值判定。"""
        config = self.analysis_config or {}
        self.data_struct.analysis_result_dict.pop(self.title_name, None)
        try:
            recorded_signal = resolve_analysis_channel_signal(
                self.data_struct,
                config,
                self.title_name,
            )
        except Exception as exc:
            QMessageBox.warning(self, "提示", str(exc))
            self._plot_fft(np.array([]), np.array([]), config=config)
            self.result = {}
            return False

        sample_rate = self.data_struct.sample_rate
        if sample_rate is None or int(sample_rate) <= 0:
            QMessageBox.warning(self, "提示", "缺少采样率，无法执行 FFT 分析。")
            return False

        v2pa_factor = self._valid_v2pa_factor()
        if v2pa_factor is None:
            return False

        try:
            n_fft = int(config.get("n_fft", 4096))
            window = str(config.get("window", "hann") or "hann")
            overlap_ratio = float(config.get("overlap_ratio", 0.5))
            weighting = str(config.get("weighting", "Z") or "Z")
            analyzer = FftAnalyzer()
            main_result = analyzer.analyze(
                recorded_signal,
                fs=int(sample_rate),
                n_fft=n_fft,
                window=window,
                overlap_ratio=overlap_ratio,
                weighting=weighting,
                v2pa_factor=v2pa_factor,
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "提示",
                f"FFT 分析失败: {str(exc)[:200]}",
            )
            self._plot_fft(np.array([]), np.array([]), config=config)
            self.result = {}
            return False

        frequency = np.asarray(
            main_result.frequencies_hz,
            dtype=np.float64,
        )
        fft_db = np.asarray(
            main_result.spectrum_db,
            dtype=np.float64,
        )
        weighting = main_result.weighting
        baseline_db = self._load_baseline(
            config,
            analyzer,
            frequency,
            sample_rate=int(sample_rate),
            n_fft=n_fft,
            window=window,
            overlap_ratio=overlap_ratio,
            weighting=weighting,
            v2pa_factor=v2pa_factor,
        )

        requested_display_mode = str(
            config.get("baseline_display_mode", "overlay") or "overlay"
        )
        if requested_display_mode == "delta" and baseline_db is None:
            QMessageBox.warning(
                self,
                "提示",
                "FFT 差值显示需要可用的背景音频基线。",
            )
            return False
        display_mode = (
            "delta"
            if requested_display_mode == "delta"
            else "overlay"
        )
        curves = self._build_display_curves(
            fft_db,
            baseline_db,
            display_mode,
        )
        plot_y = curves["plot_y"]
        delta_db = curves["delta_db"]

        x_axis_scale = str(
            config.get("x_axis_scale", "log") or "log"
        ).lower()
        if x_axis_scale not in {"linear", "log"}:
            x_axis_scale = "log"
        focus_enabled = bool(config.get("focus_range_enabled", True))
        focus_min_hz = float(config.get("focus_min_hz", 100))
        focus_max_hz = float(config.get("focus_max_hz", 20000))
        if (
            focus_enabled
            and (
                not np.isfinite(focus_min_hz)
                or not np.isfinite(focus_max_hz)
                or focus_min_hz < 0.0
                or focus_max_hz <= focus_min_hz
            )
        ):
            QMessageBox.warning(self, "提示", "FFT 频率聚焦范围配置无效。")
            return False

        frequency_mask = self._build_frequency_mask(
            frequency,
            focus_enabled,
            focus_min_hz,
            focus_max_hz,
            x_axis_scale,
        )
        plot_x = frequency[frequency_mask]
        display_y = plot_y[frequency_mask]
        display_fft = fft_db[frequency_mask]
        display_baseline = (
            baseline_db[frequency_mask]
            if baseline_db is not None
            else None
        )
        display_delta = (
            delta_db[frequency_mask]
            if isinstance(delta_db, np.ndarray)
            else None
        )
        if plot_x.size == 0:
            QMessageBox.warning(
                self,
                "提示",
                "当前频率聚焦范围内没有可显示的 FFT 数据。",
            )
            return False

        y_label = (
            f"FFT Spectrum [dB({weighting}) SPL]"
            if weighting != "Z"
            else "FFT Spectrum [dB SPL]"
        )
        if display_mode == "delta":
            y_label = "FFT - Baseline [dB]"

        upper_limits = None
        lower_limits = None
        out_mask = None
        if bool(config.get("limit_checked", False)):
            try:
                upper_limits, lower_limits = self._resolve_limits(
                    config,
                    plot_x,
                )
            except (ManualLimitValidationError, TypeError, ValueError) as exc:
                QMessageBox.warning(
                    self,
                    "提示",
                    f"FFT 阈值配置无效: {str(exc)[:200]}",
                )
                return False

            valid_mask = np.isfinite(display_y) & (
                np.isfinite(upper_limits) | np.isfinite(lower_limits)
            )
            if not np.any(valid_mask):
                QMessageBox.warning(
                    self,
                    "提示",
                    "当前 FFT 结果没有可用于阈值判定的有效频点。",
                )
                return False
            out_mask, deviation, is_ok = LimitPlotUtils.compare_with_limits(
                display_y,
                upper_limits,
                lower_limits,
                valid_mask=valid_mask,
            )
            self.data_struct.analysis_result_dict[self.title_name] = (
                bool(is_ok),
                float(deviation),
            )

        self._plot_fft(
            plot_x,
            display_y,
            config=config,
            y_label=y_label,
            baseline_y=(
                display_baseline
                if display_mode == "overlay"
                else None
            ),
            upper_limits=upper_limits,
            lower_limits=lower_limits,
            out_mask=out_mask,
        )
        self.result = {
            "frequency_bins": plot_x.tolist(),
            "fft_db": display_fft.tolist(),
            "baseline_db": (
                display_baseline.tolist()
                if isinstance(display_baseline, np.ndarray)
                else []
            ),
            "delta_db": (
                display_delta.tolist()
                if isinstance(display_delta, np.ndarray)
                else []
            ),
            "plot_db": display_y.tolist(),
            "weighting": weighting,
            "display_mode": display_mode,
            "baseline_smooth_third_octave": bool(
                config.get("baseline_smooth_third_octave", False)
            ),
            "n_fft": n_fft,
            "window": window,
            "overlap_ratio": overlap_ratio,
            "x_axis_scale": x_axis_scale,
        }
        return self.result

    def _load_baseline(
        self,
        config,
        analyzer,
        frequency,
        *,
        sample_rate,
        n_fft,
        window,
        overlap_ratio,
        weighting,
        v2pa_factor,
    ):
        baseline_file_path = str(
            config.get("baseline_file_path", "") or ""
        ).strip()
        if not baseline_file_path:
            return None
        try:
            baseline_signal, _ = librosa.load(
                baseline_file_path,
                sr=sample_rate,
                mono=True,
            )
            baseline_result = analyzer.analyze(
                baseline_signal,
                fs=sample_rate,
                n_fft=n_fft,
                window=window,
                overlap_ratio=overlap_ratio,
                weighting=weighting,
                v2pa_factor=v2pa_factor,
            )
            baseline_db = np.interp(
                frequency,
                np.asarray(
                    baseline_result.frequencies_hz,
                    dtype=np.float64,
                ),
                np.asarray(
                    baseline_result.spectrum_db,
                    dtype=np.float64,
                ),
                left=np.nan,
                right=np.nan,
            )
            if bool(
                config.get("baseline_smooth_third_octave", False)
            ):
                baseline_db = self._smooth_baseline_third_octave(
                    frequency,
                    baseline_db,
                )
            return baseline_db
        except Exception as exc:
            QMessageBox.warning(
                self,
                "提示",
                f"背景噪声基线加载失败: {str(exc)[:200]}",
            )
            return None

    @staticmethod
    def _build_display_curves(spectrum_db, baseline_db, display_mode):
        spectrum = np.asarray(spectrum_db, dtype=np.float64)
        baseline = (
            None
            if baseline_db is None
            else np.asarray(baseline_db, dtype=np.float64)
        )
        delta = spectrum - baseline if baseline is not None else None
        plot_y = (
            delta
            if display_mode == "delta" and delta is not None
            else spectrum
        )
        return {
            "plot_y": plot_y,
            "fft_db": spectrum,
            "baseline_db": baseline,
            "delta_db": delta,
        }

    @staticmethod
    def _smooth_baseline_third_octave(frequency, baseline_db):
        frequency = np.asarray(frequency, dtype=np.float64)
        baseline = np.asarray(baseline_db, dtype=np.float64)
        smoothed = np.full_like(baseline, np.nan, dtype=np.float64)
        factor = 2.0 ** (1.0 / 6.0)

        valid_points = np.isfinite(frequency) & np.isfinite(baseline)
        if not np.any(valid_points):
            return smoothed
        order = np.argsort(frequency[valid_points])
        sorted_frequency = frequency[valid_points][order]
        sorted_power = np.power(
            10.0,
            baseline[valid_points][order] / 10.0,
        )
        prefix_power = np.concatenate(
            ([0.0], np.cumsum(sorted_power))
        )

        valid_centers = np.isfinite(frequency) & (frequency > 0.0)
        lower_frequency = frequency[valid_centers] / factor
        upper_frequency = frequency[valid_centers] * factor
        left_indices = np.searchsorted(
            sorted_frequency,
            lower_frequency,
            side="left",
        )
        right_indices = np.searchsorted(
            sorted_frequency,
            upper_frequency,
            side="right",
        )
        counts = right_indices - left_indices
        power_sum = (
            prefix_power[right_indices] - prefix_power[left_indices]
        )
        center_values = np.full(
            counts.shape,
            np.nan,
            dtype=np.float64,
        )
        non_empty = counts > 0
        center_values[non_empty] = 10.0 * np.log10(
            np.maximum(
                power_sum[non_empty] / counts[non_empty],
                1e-30,
            )
        )
        smoothed[valid_centers] = center_values
        return smoothed

    @staticmethod
    def _build_frequency_mask(
        frequency,
        focus_enabled,
        focus_min_hz,
        focus_max_hz,
        x_axis_scale,
    ):
        frequency = np.asarray(frequency, dtype=np.float64)
        mask = np.isfinite(frequency)
        if x_axis_scale == "log":
            mask &= frequency > 0.0
        if focus_enabled:
            mask &= (
                (frequency >= focus_min_hz)
                & (frequency <= focus_max_hz)
            )
        return mask

    @classmethod
    def _resolve_limits(cls, config, target_x):
        limit_mode = str(
            config.get("limit_mode", "csv") or "csv"
        ).lower()
        if limit_mode == "manual":
            _, upper_values, lower_values = limits_from_manual_segments(
                config,
                target_x,
            )
            upper_limits = np.asarray(
                upper_values,
                dtype=np.float64,
            )
            lower_limits = np.asarray(
                lower_values,
                dtype=np.float64,
            )
        elif limit_mode == "csv":
            limit_data = config.get("limit_data")
            if not limit_data:
                raise ValueError("已启用阈值，但未加载 CSV 配置文件")
            try:
                csv_x, csv_upper, csv_lower = limit_data
            except (TypeError, ValueError) as exc:
                raise ValueError("CSV 阈值数据格式不正确") from exc
            upper_limits = cls._interpolate_limit_side(
                target_x,
                csv_x,
                csv_upper,
            )
            lower_limits = cls._interpolate_limit_side(
                target_x,
                csv_x,
                csv_lower,
            )
        else:
            raise ValueError(f"不支持的阈值模式: {limit_mode}")

        overlap = np.isfinite(upper_limits) & np.isfinite(lower_limits)
        if np.any(lower_limits[overlap] > upper_limits[overlap]):
            raise ValueError("下限不能大于上限")
        if not np.any(np.isfinite(upper_limits)) and not np.any(
            np.isfinite(lower_limits)
        ):
            raise ValueError("当前频率范围内没有可用的上下限")
        return upper_limits, lower_limits

    @staticmethod
    def _interpolate_limit_side(target_x, raw_x, raw_values):
        x_values = np.asarray(list(raw_x), dtype=np.float64)
        side_values = np.asarray(list(raw_values), dtype=np.float64)
        if x_values.size != side_values.size:
            raise ValueError("CSV 阈值数据长度不一致")

        finite = np.isfinite(x_values) & np.isfinite(side_values)
        output = np.full(
            np.asarray(target_x).shape,
            np.nan,
            dtype=np.float64,
        )
        if not np.any(finite):
            return output

        points = {}
        for x_value, side_value in zip(
            x_values[finite],
            side_values[finite],
        ):
            points[float(x_value)] = float(side_value)
        sorted_keys = sorted(points)
        sorted_x = np.asarray(sorted_keys, dtype=np.float64)
        sorted_values = np.asarray(
            [points[x_value] for x_value in sorted_keys],
            dtype=np.float64,
        )
        target_values = np.asarray(target_x, dtype=np.float64)
        in_range = (
            (target_values >= sorted_x[0])
            & (target_values <= sorted_x[-1])
        )
        if np.any(in_range):
            output[in_range] = np.interp(
                target_values[in_range],
                sorted_x,
                sorted_values,
            )
        return output

    def _plot_fft(
        self,
        frequency,
        spectrum_db,
        *,
        config,
        y_label="FFT Spectrum [dB SPL]",
        baseline_y=None,
        upper_limits=None,
        lower_limits=None,
        out_mask=None,
    ):
        self.analysis_plot.clear()
        colors = resolve_curve_colors(config)
        frequency = np.asarray(frequency, dtype=np.float64)
        spectrum_db = np.asarray(spectrum_db, dtype=np.float64)
        self.analysis_plot.plot(
            frequency,
            spectrum_db,
            pen=mkPen(color=colors[MAIN_CURVE_COLOR], width=2),
            name="FFT",
        )
        if baseline_y is not None:
            self.analysis_plot.plot(
                frequency,
                np.asarray(baseline_y, dtype=np.float64),
                pen=mkPen(color="#808080", width=2),
                name="Baseline",
            )
        self._plot_limit_curve(
            frequency,
            upper_limits,
            colors[UPPER_LIMIT_COLOR],
        )
        self._plot_limit_curve(
            frequency,
            lower_limits,
            colors[LOWER_LIMIT_COLOR],
        )
        if out_mask is not None:
            LimitPlotUtils.plot_out_segments(
                self.analysis_plot,
                frequency,
                spectrum_db,
                np.asarray(out_mask, dtype=bool),
                pen_color="#F44336",
                pen_width=3,
            )

        x_axis_scale = str(
            config.get("x_axis_scale", "log") or "log"
        ).lower()
        self.analysis_plot.setLabel("left", y_label)
        self.analysis_plot.setLabel("bottom", "Frequency (Hz)")
        self.analysis_plot.setLogMode(
            x=x_axis_scale == "log",
            y=False,
        )
        self.analysis_plot.showGrid(x=True, y=True)
        apply_plot_view_range(
            self.analysis_plot,
            config,
            allow_x=True,
            allow_y=True,
        )

    def _plot_limit_curve(self, frequency, limits, color):
        if limits is None:
            return
        values = np.asarray(limits, dtype=np.float64)
        if values.size != frequency.size or not np.any(np.isfinite(values)):
            return
        self.analysis_plot.plot(
            frequency,
            values,
            pen=mkPen(
                color=color,
                width=2,
                style=Qt.DashLine,
            ),
        )


class FrequencyBandAnalysis(AnalysisGraphWidget):
    """频段能量分析窗口。"""

    STRATEGY_LABELS = {
        "1/1 倍频程": ("octave", {"fraction": 1}),
        "1/3 倍频程": ("octave", {"fraction": 3}),
        "1/6 倍频程": ("octave", {"fraction": 6}),
        "1/12 倍频程": ("octave", {"fraction": 12}),
        "Bark": ("bark", {}),
        "等宽": ("equal_width", {}),
        "自定义": ("custom", {}),
    }

    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.v2pa_factor = None
        self.analysis_config = None
        self.result = {}
        self.title_name = title_name
        self.setWindowTitle(title_name)

    def calculate_fba(self):
        """执行频段能量分析、阈值判定并绘制结果。"""
        config = self.analysis_config or {}
        self.data_struct.analysis_result_dict.pop(self.title_name, None)
        try:
            recorded_signal = resolve_analysis_channel_signal(
                self.data_struct,
                config,
                self.title_name,
            )
        except Exception as exc:
            QMessageBox.warning(self, "提示", str(exc))
            return False

        sample_rate = self.data_struct.sample_rate
        if sample_rate is None or int(sample_rate) <= 0:
            QMessageBox.warning(self, "提示", "缺少采样率，无法执行频段能量分析。")
            return False

        v2pa_factor = self._valid_v2pa_factor()
        if v2pa_factor is None:
            return False

        strategy_label = config.get("band_strategy", "1/3 倍频程")
        strategy_name, strategy_kwargs = self.STRATEGY_LABELS.get(
            strategy_label,
            ("octave", {"fraction": 3}),
        )
        weighting = str(config.get("weighting", "A") or "A")
        if weighting in ("None", "Z（None）"):
            weighting = "Z"

        f_min = float(config.get("f_min", 20))
        f_max = float(config.get("f_max", 20000))
        if (
            not np.isfinite(f_min)
            or not np.isfinite(f_max)
            or f_min <= 0.0
            or f_max <= f_min
        ):
            QMessageBox.warning(self, "提示", "FBA 分析频率范围配置无效。")
            return False

        custom_edges = None
        if strategy_label == "自定义":
            try:
                custom_edges = self._parse_custom_bands_text(
                    config.get("custom_bands_text", "")
                )
            except ValueError as exc:
                QMessageBox.warning(
                    self,
                    "提示",
                    f"自定义频段解析失败: {str(exc)[:200]}",
                )
                return False

        try:
            analyzer = FrequencyBandAnalyzer(
                strategy=strategy_name,
                weighting=weighting,
                f_min=f_min,
                f_max=f_max,
                fraction=strategy_kwargs.get("fraction", 3),
                n_bands=int(config.get("n_bands", 40)),
                bandwidth=float(config.get("bandwidth", 100)),
                custom_edges=custom_edges,
            )
            analysis_result = analyzer.analyze(
                recorded_signal,
                fs=int(sample_rate),
                v2pa_factor=v2pa_factor,
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "提示",
                f"频段能量分析失败: {str(exc)[:200]}",
            )
            return False

        upper_limits = None
        lower_limits = None
        if bool(config.get("limit_checked", False)):
            centers = np.asarray(
                [band.f_center for band in analysis_result.bands],
                dtype=np.float64,
            )
            try:
                upper_limits, lower_limits = self._resolve_limits(
                    config,
                    centers,
                )
            except (ManualLimitValidationError, TypeError, ValueError) as exc:
                QMessageBox.warning(
                    self,
                    "提示",
                    f"FBA 阈值配置无效: {str(exc)[:200]}",
                )
                return False

            levels = np.asarray(
                analysis_result.band_levels_weighted_db,
                dtype=np.float64,
            )
            valid_mask = np.isfinite(levels) & (
                np.isfinite(upper_limits) | np.isfinite(lower_limits)
            )
            if not np.any(valid_mask):
                QMessageBox.warning(
                    self,
                    "提示",
                    "当前 FBA 结果没有可用于阈值判定的有效频段。",
                )
                return False
            out_mask, deviation, is_ok = LimitPlotUtils.compare_with_limits(
                levels,
                upper_limits,
                lower_limits,
                valid_mask=valid_mask,
            )
            analysis_result.exceeded_bands = (
                np.where(out_mask)[0].astype(int).tolist()
            )
            self.data_struct.analysis_result_dict[self.title_name] = (
                bool(is_ok),
                float(deviation),
            )

        self._plot_bar_chart(
            analysis_result,
            weighting,
            upper_limits=upper_limits,
            lower_limits=lower_limits,
        )
        self.result = {
            "bands": [band.label for band in analysis_result.bands],
            "band_centers": [
                band.f_center for band in analysis_result.bands
            ],
            "band_levels_db": analysis_result.band_levels_db.tolist(),
            "band_levels_weighted_db": (
                analysis_result.band_levels_weighted_db.tolist()
            ),
            "overall_db": analysis_result.overall_db,
            "overall_weighted_db": analysis_result.overall_weighted_db,
            "weighting": analysis_result.weighting,
            "exceeded_bands": list(analysis_result.exceeded_bands),
        }
        return self.result

    @classmethod
    def _resolve_limits(cls, config, centers):
        limit_mode = str(config.get("limit_mode", "csv") or "csv").lower()
        if limit_mode == "manual":
            _, upper_values, lower_values = limits_from_manual_segments(
                config,
                centers,
            )
            upper_limits = np.asarray(upper_values, dtype=np.float64)
            lower_limits = np.asarray(lower_values, dtype=np.float64)
        elif limit_mode == "csv":
            limit_data = config.get("limit_data")
            if not limit_data:
                raise ValueError("已启用阈值，但未加载 CSV 配置文件")
            try:
                csv_x, csv_upper, csv_lower = limit_data
            except (TypeError, ValueError) as exc:
                raise ValueError("CSV 阈值数据格式不正确") from exc
            upper_limits = cls._interpolate_limit_side(
                centers,
                csv_x,
                csv_upper,
            )
            lower_limits = cls._interpolate_limit_side(
                centers,
                csv_x,
                csv_lower,
            )
        else:
            raise ValueError(f"不支持的阈值模式: {limit_mode}")

        overlap = np.isfinite(upper_limits) & np.isfinite(lower_limits)
        if np.any(lower_limits[overlap] > upper_limits[overlap]):
            raise ValueError("下限不能大于上限")
        if not np.any(np.isfinite(upper_limits)) and not np.any(
            np.isfinite(lower_limits)
        ):
            raise ValueError("当前频段范围内没有可用的上下限")
        return upper_limits, lower_limits

    @staticmethod
    def _interpolate_limit_side(target_x, raw_x, raw_values):
        x_values = np.asarray(list(raw_x), dtype=np.float64)
        side_values = np.asarray(list(raw_values), dtype=np.float64)
        if x_values.size != side_values.size:
            raise ValueError("CSV 阈值数据长度不一致")

        finite = np.isfinite(x_values) & np.isfinite(side_values)
        if not np.any(finite):
            return np.full(np.asarray(target_x).shape, np.nan, dtype=np.float64)

        points = {}
        for x_value, side_value in zip(
            x_values[finite],
            side_values[finite],
        ):
            points[float(x_value)] = float(side_value)
        sorted_x = np.asarray(sorted(points), dtype=np.float64)
        sorted_values = np.asarray(
            [points[x_value] for x_value in sorted(points)],
            dtype=np.float64,
        )
        return np.interp(
            np.asarray(target_x, dtype=np.float64),
            sorted_x,
            sorted_values,
            left=sorted_values[0],
            right=sorted_values[-1],
        )

    @staticmethod
    def _parse_custom_bands_text(text: str):
        edges = []
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if "," in line:
                parts = [part.strip() for part in line.split(",") if part.strip()]
            else:
                parts = [
                    part.strip()
                    for part in line.replace("\t", " ").split(" ")
                    if part.strip()
                ]

            label = None
            try:
                if len(parts) == 1 and "-" in parts[0]:
                    lower, upper = [
                        part.strip() for part in parts[0].split("-", 1)
                    ]
                    f_low, f_high = float(lower), float(upper)
                elif len(parts) >= 2:
                    f_low, f_high = float(parts[0]), float(parts[1])
                    if len(parts) >= 3:
                        label = " ".join(parts[2:]).strip() or None
                else:
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise ValueError(f"格式错误: {raw!r}") from exc

            if f_low <= 0 or f_high <= 0:
                raise ValueError(f"频率必须为正数: {raw!r}")
            if f_high <= f_low:
                raise ValueError(f"频段上限必须大于下限: {raw!r}")
            edges.append((f_low, f_high, label))

        edges.sort(key=lambda item: item[0])
        if not edges:
            raise ValueError("请至少输入一个频段")
        for index in range(1, len(edges)):
            if edges[index][0] < edges[index - 1][1]:
                raise ValueError("自定义频段不允许重叠，请检查相邻频段边界")
        return edges

    def _plot_bar_chart(
        self,
        result: BandAnalysisResult,
        weighting: str,
        *,
        upper_limits=None,
        lower_limits=None,
    ):
        self.analysis_plot.clear()

        levels = np.asarray(
            result.band_levels_weighted_db,
            dtype=np.float64,
        )
        labels = [band.label for band in result.bands]
        if not labels:
            return

        x_values = np.arange(len(labels))
        colors = resolve_curve_colors(self.analysis_config or {})
        exceeded_color = "#F44336"
        missing_color = "#BDBDBD"
        finite_mask = np.isfinite(levels)
        plot_levels = levels.copy()
        plot_levels[~finite_mask] = 0.0

        brushes = [
            pg.mkBrush(colors[MAIN_CURVE_COLOR])
            if finite
            else pg.mkBrush(missing_color)
            for finite in finite_mask
        ]
        for index in result.exceeded_bands:
            if 0 <= int(index) < len(brushes) and finite_mask[int(index)]:
                brushes[int(index)] = pg.mkBrush(exceeded_color)

        self.analysis_plot.addItem(
            pg.BarGraphItem(
                x=x_values,
                height=plot_levels,
                width=0.7,
                brushes=brushes,
                pen=pg.mkPen("w", width=0.5),
            )
        )

        self._plot_limit_curve(
            x_values,
            upper_limits,
            colors[UPPER_LIMIT_COLOR],
            "o",
            "Upper Limit",
        )
        self._plot_limit_curve(
            x_values,
            lower_limits,
            colors[LOWER_LIMIT_COLOR],
            "t",
            "Lower Limit",
        )

        for index in result.exceeded_bands:
            index = int(index)
            if 0 <= index < len(plot_levels) and finite_mask[index]:
                text_item = pg.TextItem(
                    "NG",
                    color=exceeded_color,
                    anchor=(0.5, 1.0),
                )
                text_item.setPos(index, plot_levels[index])
                self.analysis_plot.addItem(text_item)

        self.analysis_plot.getAxis("bottom").setTicks(
            [[(index, label) for index, label in enumerate(labels)]]
        )
        weight_label = f"dB({weighting})" if weighting != "Z" else "dB"
        self.analysis_plot.setLabel(
            "left",
            f"Sound Pressure Level [{weight_label}]",
        )
        self.analysis_plot.setLabel("bottom", "Frequency (Hz)")
        overall = (
            result.overall_weighted_db
            if weighting != "Z"
            else result.overall_db
        )
        self.analysis_plot.setTitle(
            f"Overall: {overall:.1f} {weight_label} [SPL]",
            size="14px",
            color="k",
        )
        self.analysis_plot.showGrid(x=False, y=True)
        apply_plot_view_range(
            self.analysis_plot,
            self.analysis_config or {},
            allow_x=False,
            allow_y=True,
        )

    def _plot_limit_curve(
        self,
        x_values,
        limits,
        color,
        symbol,
        name,
    ):
        if limits is None:
            return
        values = np.asarray(limits, dtype=np.float64)
        if values.size != len(x_values) or not np.any(np.isfinite(values)):
            return
        self.analysis_plot.plot(
            x_values,
            values,
            pen=mkPen(color=color, width=2),
            symbol=symbol,
            symbolSize=5,
            symbolBrush=color,
            name=name,
        )


class LoudnessAnalysis(AnalysisGraphWidget):
    """LOUD analysis window backed by the sound-quality loudness service."""

    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.v2pa_factor = None
        self.analysis_config = None
        self.recorded_path = None
        self.result = {}
        self.export_detail = {}
        self.specific_loudness_widget = None
        self.specific_loudness_colorbar = None
        self.specific_loudness_profile_widget = None
        self.sharpness_plot = None
        self.roughness_plot = None
        self.title_name = title_name
        self.setWindowTitle(title_name)

    def calculate_loudness(self):
        config = self.analysis_config or {}
        try:
            recorded_signal = resolve_analysis_channel_signal(self.data_struct, config, self.title_name)
        except ValueError as exc:
            QMessageBox.warning(self, "提示", f"响度分析失败：{exc}")
            return False
        sample_rate = self.data_struct.sample_rate

        if recorded_signal is None or sample_rate is None:
            QMessageBox.warning(self, "提示", "响度分析失败：没有可用录音数据。")
            return False

        v2pa_factor = self._valid_v2pa_factor()
        if v2pa_factor is None:
            return False

        sq_config = self._build_sq_config(config)
        run_result = run_sound_quality(
            np.asarray(recorded_signal, dtype=np.float64),
            int(sample_rate),
            project_v2pa_factor=v2pa_factor,
            sq_config=sq_config,
        )
        loud_result = run_result.loudness
        if loud_result is None or not loud_result.enabled or loud_result.raw_result is None:
            reason = getattr(loud_result, "skipped_reason", None) or run_result.skipped_reason or "unknown"
            QMessageBox.warning(self, "提示", f"响度分析跳过：{reason}")
            return False

        summary = loud_result.summary or {}
        self._plot_loudness_curve(loud_result, config)
        self._apply_loudness_limits(loud_result, config)

        raw = loud_result.raw_result
        self.result = {
            "summary": summary,
            "time_s": self._loudness_display_time_axis(raw).tolist(),
            "loudness_sone": np.asarray(raw.loudness_sone, dtype=np.float64).tolist(),
            "loudness_level_phon": np.asarray(raw.loudness_level_phon, dtype=np.float64).tolist(),
            "metadata": dict(raw.metadata or {}),
        }
        self.export_detail = {
            "specific_loudness_sum_sone": summary.get("specific_loudness_sum_sone"),
            "specific_loudness_summed_exceedance": summary.get("specific_loudness_summed_exceedance"),
            "steady_state_average_sone": summary.get("steady_state_average_sone"),
            "steady_state_average_phon": summary.get("steady_state_average_phon"),
            "max_transient_sone": summary.get("max_transient_sone"),
            "max_transient_phon": summary.get("max_transient_phon"),
            "nmax_sone": summary.get("nmax_sone"),
            "lnmax_phon": summary.get("lnmax_phon"),
            "mean_sone": summary.get("mean_sone"),
            "mean_phon": summary.get("mean_phon"),
        }
        return self.result

    @staticmethod
    def _build_sq_config(config: dict) -> dict:
        display_cfg = config.get("display", {}) or {}
        save_cfg = config.get("save", {}) or {}
        advanced_cfg = config.get("advanced", {}) or {}
        required_summary_metrics = []
        if bool(config.get("limit_checked", False)):
            limit_metric = str(config.get("limit_metric", "curve_y") or "curve_y").lower()
            limit_unit = str(
                config.get("curve_limit_unit")
                or advanced_cfg.get("curve_y_unit", "sone")
                or "sone"
            ).lower()
            if limit_metric == "steady_state_average":
                required_metric = (
                    "steady_state_average_phon"
                    if limit_unit == "phon"
                    else "steady_state_average_sone"
                )
            elif limit_metric == "max_transient":
                required_metric = (
                    "max_transient_phon"
                    if limit_unit == "phon"
                    else "max_transient_sone"
                )
            elif limit_metric == "specific_loudness_summed_exceedance":
                required_metric = "specific_loudness_summed_exceedance"
            else:
                required_metric = None
            if required_metric:
                required_summary_metrics.append(required_metric)
        return {
            "enabled": True,
            "shared": {"field_type": config.get("field_type", "free")},
            "items": {
                "LOUD": {
                    "enabled": bool(config.get("enabled", True)),
                    "method": config.get("method", "time_varying_iso532_1"),
                    "display": display_cfg,
                    "save": save_cfg,
                    "advanced": advanced_cfg,
                    "required_summary_metrics": required_summary_metrics,
                },
                "SHRP": {"enabled": False},
                "ROUGH": {"enabled": False},
                "FLUC": {"enabled": False},
                "TON": {"enabled": False},
                "PR": {"enabled": False},
                "TNR": {"enabled": False},
            },
        }

    @staticmethod
    def _loudness_display_time_axis(raw) -> np.ndarray:
        time_s = np.asarray(raw.time_s, dtype=np.float64)
        metadata = dict(getattr(raw, "metadata", None) or {})
        if not metadata.get("analysis_time_range_enabled", False):
            return time_s
        try:
            source_start_s = float(metadata.get("analysis_source_start_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            return time_s
        if not np.isfinite(source_start_s) or source_start_s <= 0.0:
            return time_s
        return time_s + source_start_s

    def _plot_loudness_curve(self, loud_result, config=None):
        raw = loud_result.raw_result
        advanced_cfg = (config or {}).get("advanced", {}) or {}
        metadata = dict(getattr(raw, "metadata", None) or {})
        plot_start_s = 0.0
        if metadata.get("analysis_time_range_enabled", False):
            try:
                plot_start_s = max(0.0, float(metadata.get("analysis_source_start_s", 0.0) or 0.0))
            except (TypeError, ValueError):
                plot_start_s = 0.0
        time_s = self._loudness_display_time_axis(raw)
        curve_y_unit = str(advanced_cfg.get("curve_y_unit", "sone") or "sone").lower()
        if curve_y_unit == "phon":
            loudness = np.asarray(raw.loudness_level_phon, dtype=np.float64)
            y_label = "响度级 (phon)"
        else:
            loudness = np.asarray(raw.loudness_sone, dtype=np.float64)
            y_label = "响度 (sone)"

        plot_time_s = time_s
        plot_loudness = loudness
        method = str((config or {}).get("method", "") or "").lower()
        if (
            method == "per_segment"
            and time_s.size
            and loudness.size
            and np.isfinite(time_s[0])
            and np.isfinite(loudness[0])
            and time_s[0] > 0.0
        ):
            if time_s.size > 1 and loudness.size > 1 and np.isfinite(loudness[1]):
                plot_time_s = np.insert(time_s[1:], 0, plot_start_s)
                plot_loudness = np.insert(loudness[1:], 0, loudness[1])
            else:
                plot_time_s = np.insert(time_s, 0, plot_start_s)
                plot_loudness = np.insert(loudness, 0, loudness[0])
            end_time_s = None
            if metadata.get("analysis_time_range_enabled", False):
                try:
                    end_time_s = float(metadata.get("analysis_source_end_s"))
                except (TypeError, ValueError):
                    end_time_s = None
            sample_rate = getattr(self.data_struct, "sample_rate", None)
            recorded_signal = getattr(self.data_struct, "store_wave_data", None)
            try:
                if end_time_s is None and sample_rate and recorded_signal is not None:
                    end_time_s = len(recorded_signal) / float(sample_rate)
            except (TypeError, ValueError, ZeroDivisionError):
                end_time_s = None
            if end_time_s is None:
                try:
                    frame_duration_s = float(metadata.get("frame_duration_s"))
                    end_time_s = float(time_s[-1]) + frame_duration_s / 2.0
                except (TypeError, ValueError):
                    end_time_s = None
            if (
                end_time_s is not None
                and np.isfinite(end_time_s)
                and plot_time_s.size
                and end_time_s > float(plot_time_s[-1])
            ):
                plot_time_s = np.append(plot_time_s, float(end_time_s))
                plot_loudness = np.append(plot_loudness, float(plot_loudness[-1]))

        self.analysis_plot.clear()
        if plot_time_s.size and plot_loudness.size:
            self.analysis_plot.plot(
                plot_time_s,
                plot_loudness,
                pen=mkPen(color=(51, 196, 77), width=2),
                name="Loudness",
            )
            self._apply_loudness_y_axis(loudness, advanced_cfg, curve_y_unit)
        self.analysis_plot.setLabel("left", y_label)
        self.analysis_plot.setLabel("bottom", "时间 (s)")
        self.analysis_plot.showGrid(x=True, y=True)

        payload = loud_result.display_payload or {}
        title_parts = []
        for card in payload.get("summary_cards", []) or []:
            # The summed exceedance is shown on the N'(z) profile plot (in cSones),
            # so keep it off the loudness-time curve title to avoid duplication.
            if card.get("key") == "specific_loudness_summed_exceedance":
                continue
            value = card.get("value")
            try:
                finite_value = value is not None and np.isfinite(float(value))
            except (TypeError, ValueError):
                finite_value = False
            if finite_value:
                title_parts.append(f"{card.get('label', card.get('key'))}: {float(value):.3g} {card.get('unit', '')}")
        if title_parts:
            self.analysis_plot.setTitle(" | ".join(title_parts), size="14px", color="k")

        if config:
            self._draw_loudness_limit_lines(self.analysis_plot, config)

        self._plot_specific_loudness_profile(loud_result, config or {})
        self._plot_specific_loudness_heatmap(loud_result, config or {})

    def _apply_loudness_y_axis(self, loudness: np.ndarray, advanced_cfg: dict, curve_y_unit: str):
        finite = np.asarray(loudness, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return
        y_max = float(np.max(finite))
        if y_max <= 0.0:
            return
        unit = str(curve_y_unit or "sone").lower()
        min_y_range = max(5.0, y_max * 0.08) if unit == "phon" else max(2.0, y_max * 0.12)
        self.analysis_plot.getViewBox().setLimits(minYRange=min_y_range)
        if not bool(advanced_cfg.get("curve_y_axis_zero_based", True)):
            return
        upper = y_max * 1.03
        if upper <= y_max:
            upper = y_max + 1.0
        self.analysis_plot.getViewBox().setYRange(0.0, upper, padding=0.0)

    def _apply_sharpness_y_axis(self, plot_widget, sharpness: np.ndarray):
        finite = np.asarray(sharpness, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return
        y_max = float(np.max(finite))
        upper = max(2.0, y_max * 1.3)
        view_box = plot_widget.getViewBox()
        view_box.setYRange(0.0, upper, padding=0.0)

    def _apply_roughness_y_axis(self, plot_widget, roughness: np.ndarray):
        finite = np.asarray(roughness, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return
        y_max = float(np.max(finite))
        upper = max(0.5, y_max * 1.3)
        view_box = plot_widget.getViewBox()
        view_box.setYRange(0.0, upper, padding=0.0)

    def _plot_specific_loudness_profile(self, loud_result, config=None):
        self._remove_specific_loudness_profile()

        curves = (loud_result.display_payload or {}).get("curves", []) or []
        curve = next((item for item in curves if item.get("key") == "specific_loudness_profile"), None)
        if not curve:
            return

        bark_axis = np.asarray(curve.get("x"), dtype=np.float64)
        profile = np.asarray(curve.get("y"), dtype=np.float64)
        if bark_axis.size == 0 or profile.size == 0:
            return

        plot_widget = pg.PlotWidget(background="white")
        legend = plot_widget.addLegend(offset=(-10, 10))
        legend.setParentItem(plot_widget.getPlotItem().getViewBox())
        legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 10))
        plot_widget.plot(
            bark_axis,
            profile,
            pen=mkPen(color=(238, 126, 33), width=2),
            name="N'(z) measured",
        )
        ref_curve = self._draw_specific_loudness_ref_line(plot_widget, bark_axis, profile, config or {})
        mode = str(curve.get("profile_mode", "steady_average") or "steady_average")
        title = "特征响度曲线 N'(z)"
        if mode == "max_loudness":
            title += " - 最大响度时刻"
        else:
            title += " - 稳态平均"
        exceedance_csones = self._specific_loudness_exceedance_csones(loud_result)
        if exceedance_csones is not None:
            title += f"  |  超限总量: {exceedance_csones:.2f} cSones"
        plot_widget.setTitle(title, size="14px", color="k")
        plot_widget.setLabel("left", "N' (sone/Bark)")
        plot_widget.setLabel("bottom", "Bark")
        self.apply_plot_font_style(plot_widget, 20)
        plot_widget.showGrid(x=True, y=True)
        finite = profile[np.isfinite(profile)]
        candidates = [float(np.max(finite))] if finite.size else []
        if ref_curve is not None and ref_curve.size:
            ref_finite = ref_curve[np.isfinite(ref_curve)]
            if ref_finite.size:
                candidates.append(float(np.max(ref_finite)))
        if candidates:
            y_max = max(candidates)
            plot_widget.getViewBox().setYRange(0.0, max(0.1, y_max * 1.15), padding=0.0)
        self.specific_loudness_profile_widget = plot_widget
        self.layout().addWidget(plot_widget)

    @staticmethod
    def _specific_loudness_exceedance_csones(loud_result):
        """Return the summed specific-loudness exceedance in cSones (0.01 sone), or None.

        The value is computed by the service layer in sone; centi-sones makes the
        small exceedance numbers easier to read on the plot title.
        """
        payload = loud_result.display_payload or {}
        for card in payload.get("summary_cards", []) or []:
            if card.get("key") != "specific_loudness_summed_exceedance":
                continue
            value = card.get("value")
            try:
                value_sone = float(value)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(value_sone):
                return None
            return value_sone * 100.0
        return None

    @staticmethod
    def _draw_specific_loudness_ref_line(plot_widget, bark_axis, profile, config):
        advanced_cfg = (config or {}).get("advanced", {}) or {}
        ref_key = str(advanced_cfg.get("specific_loudness_exceedance_ref_line", "") or "").lower()
        try:
            from base.core_algorithm.sound_quality.psychoacoustic_constants import (
                SSTS_SPECIFIC_LOUDNESS_REF_LINES,
            )
            from base.core_algorithm.sound_quality.service import interpolate_ref_line
        except ImportError:
            return None
        if ref_key not in SSTS_SPECIFIC_LOUDNESS_REF_LINES:
            return None
        ref_curve = interpolate_ref_line(bark_axis, SSTS_SPECIFIC_LOUDNESS_REF_LINES[ref_key])
        ref_label = f"Ref {ref_key[-1]} limit"
        plot_widget.plot(
            bark_axis,
            ref_curve,
            pen=mkPen(color=(214, 39, 40), width=2, style=Qt.DashLine),
            name=ref_label,
        )
        excess = np.maximum(profile - ref_curve, 0.0)
        if np.any(excess > 0.0):
            fill_top = pg.PlotDataItem(bark_axis, np.maximum(profile, ref_curve))
            fill_bottom = pg.PlotDataItem(bark_axis, ref_curve)
            fill = pg.FillBetweenItem(fill_top, fill_bottom, brush=(214, 39, 40, 70))
            plot_widget.addItem(fill)
        return ref_curve

    def _remove_specific_loudness_profile(self):
        if self.specific_loudness_profile_widget is None:
            return
        try:
            self.layout().removeWidget(self.specific_loudness_profile_widget)
            self.specific_loudness_profile_widget.deleteLater()
        finally:
            self.specific_loudness_profile_widget = None

    def _plot_specific_loudness_heatmap(self, loud_result, config: dict):
        self._remove_specific_loudness_heatmap()

        heatmaps = (loud_result.display_payload or {}).get("heatmaps", []) or []
        heatmap = next((item for item in heatmaps if item.get("key") == "specific_loudness"), None)
        if not heatmap:
            return

        time_s = np.asarray(heatmap.get("x"), dtype=np.float64)
        bark_axis = np.asarray(heatmap.get("y"), dtype=np.float64)
        specific = np.asarray(heatmap.get("z"), dtype=np.float64)
        if time_s.size < 2 or bark_axis.size < 2 or specific.size == 0:
            return

        advanced_cfg = config.get("advanced", {}) or {}
        colormap = str(advanced_cfg.get("specific_loudness_colormap", "viridis") or "viridis")
        z = specific.T if specific.shape[0] == bark_axis.size else specific
        self.specific_loudness_widget, self.specific_loudness_colorbar = plot_2d_image(
            x=time_s,
            y=bark_axis,
            z=z,
            title="特征响度 N'(z, t) [sone/Bark]",
            xlabel="时间 (s)",
            ylabel="Bark",
            colormap=colormap,
            x_range=(float(time_s.min()), float(time_s.max())),
            y_range=(float(bark_axis.min()), float(bark_axis.max())),
            background_color="white",
        )
        heatmap_plot = self.specific_loudness_widget.findChild(pg.PlotWidget)
        if heatmap_plot is not None:
            heatmap_plot.setTitle("特征响度 N'(z, t) [sone/Bark]", size="14px", color="k")
            self.apply_plot_font_style(heatmap_plot, 20)
        self.layout().addWidget(self.specific_loudness_widget)

    def _remove_specific_loudness_heatmap(self):
        if self.specific_loudness_widget is None:
            return
        try:
            self.layout().removeWidget(self.specific_loudness_widget)
            self.specific_loudness_widget.deleteLater()
        finally:
            self.specific_loudness_widget = None
            self.specific_loudness_colorbar = None

    def _apply_loudness_limits(self, loud_result, config: dict):
        if not bool(config.get("limit_checked", False)):
            return

        raw = getattr(loud_result, "raw_result", None)
        if raw is None:
            self.data_struct.analysis_result_dict[self.title_name] = (False, float("nan"))
            return

        advanced_cfg = (config or {}).get("advanced", {}) or {}
        limit_metric = str(config.get("limit_metric", "curve_y") or "curve_y").lower()

        if limit_metric == "steady_state_average":
            self._apply_loudness_scalar_limit(loud_result, config, metric="steady_state_average")
        elif limit_metric == "max_transient":
            self._apply_loudness_scalar_limit(loud_result, config, metric="max_transient")
        elif limit_metric == "specific_loudness_summed_exceedance":
            self._apply_loudness_scalar_limit(loud_result, config, metric="specific_loudness_summed_exceedance")
        else:
            self._apply_loudness_curve_limit(raw, config, advanced_cfg)

    def _apply_loudness_scalar_limit(self, loud_result, config: dict, metric: str):
        """Apply limit check on a single scalar metric value."""
        summary = getattr(loud_result, "summary", None) or {}
        advanced_cfg = (config or {}).get("advanced", {}) or {}
        curve_unit = str(
            config.get("curve_limit_unit")
            or advanced_cfg.get("curve_y_unit", "sone")
            or "sone"
        ).lower()
        if metric == "steady_state_average":
            if curve_unit == "phon":
                value = summary.get("steady_state_average_phon", summary.get("mean_phon"))
            else:
                value = summary.get("steady_state_average_sone", summary.get("mean_sone"))
        elif metric == "max_transient":
            if curve_unit == "phon":
                value = summary.get("max_transient_phon", summary.get("lnmax_phon"))
            else:
                value = summary.get("max_transient_sone", summary.get("nmax_sone"))
        elif metric == "specific_loudness_summed_exceedance":
            value = summary.get("specific_loudness_summed_exceedance")
        else:
            value = None

        if value is None or not np.isfinite(float(value)):
            self.data_struct.analysis_result_dict[self.title_name] = (False, float("nan"))
            return

        value = float(value)
        upper_enabled = bool(config.get("curve_upper_enabled", False))
        upper_limit = float(config.get("curve_upper_value", 0.0) or 0.0)
        lower_enabled = bool(config.get("curve_lower_enabled", False))
        lower_limit = float(config.get("curve_lower_value", 0.0) or 0.0)

        deviations = []
        if upper_enabled and value > upper_limit:
            deviations.append(value - upper_limit)
        if lower_enabled and value < lower_limit:
            deviations.append(lower_limit - value)

        deviation = float(max(deviations)) if deviations else 0.0
        self._merge_analysis_limit_result(deviation == 0.0, deviation)

    def _apply_loudness_curve_limit(self, raw, config: dict, advanced_cfg: dict):
        """Apply per-point limit check on the loudness time curve."""
        curve_y_unit = str(
            config.get("curve_limit_unit")
            or advanced_cfg.get("curve_y_unit", "sone")
            or "sone"
        ).lower()
        if curve_y_unit == "phon":
            values = np.asarray(raw.loudness_level_phon, dtype=np.float64)
        else:
            values = np.asarray(raw.loudness_sone, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            self.data_struct.analysis_result_dict[self.title_name] = (False, float("nan"))
            return

        upper_enabled = bool(
            config.get(
                "curve_upper_enabled",
                config.get("mean_upper_enabled", config.get("nmax_upper_enabled", False)),
            )
        )
        upper_limit = float(
            config.get(
                "curve_upper_value",
                config.get("mean_upper_sone", config.get("nmax_upper_sone", 0.0)),
            )
            or 0.0
        )
        lower_enabled = bool(
            config.get(
                "curve_lower_enabled",
                config.get("mean_lower_enabled", config.get("nmax_lower_enabled", False)),
            )
        )
        lower_limit = float(
            config.get(
                "curve_lower_value",
                config.get("mean_lower_sone", config.get("nmax_lower_sone", 0.0)),
            )
            or 0.0
        )

        deviations = []
        if upper_enabled:
            upper_deviation = float(np.max(values - upper_limit))
            if upper_deviation > 0.0:
                deviations.append(upper_deviation)
        if lower_enabled:
            lower_deviation = float(np.max(lower_limit - values))
            if lower_deviation > 0.0:
                deviations.append(lower_deviation)

        deviation = float(max(deviations)) if deviations else 0.0
        self._merge_analysis_limit_result(deviation == 0.0, deviation)

    @staticmethod
    def _draw_loudness_limit_lines(plot_widget, config: dict):
        """Draw horizontal limit lines for loudness plot."""
        if not bool(config.get("limit_checked", False)):
            return
        upper_enabled = bool(
            config.get("curve_upper_enabled",
                        config.get("mean_upper_enabled", config.get("nmax_upper_enabled", False)))
        )
        lower_enabled = bool(
            config.get("curve_lower_enabled",
                        config.get("mean_lower_enabled", config.get("nmax_lower_enabled", False)))
        )
        dashed_pen = mkPen(color=(128, 0, 128), width=2, style=Qt.DashLine)
        if upper_enabled:
            val = float(
                config.get("curve_upper_value",
                            config.get("mean_upper_sone", config.get("nmax_upper_sone", 0.0))) or 0.0
            )
            plot_widget.addLine(y=val, pen=dashed_pen)
        if lower_enabled:
            val = float(
                config.get("curve_lower_value",
                            config.get("mean_lower_sone", config.get("nmax_lower_sone", 0.0))) or 0.0
            )
            plot_widget.addLine(y=val, pen=dashed_pen)

    def _merge_analysis_limit_result(self, is_ok: bool, deviation: float):
        existing = self.data_struct.analysis_result_dict.get(self.title_name)
        if not isinstance(existing, (tuple, list)) or len(existing) < 2:
            self.data_struct.analysis_result_dict[self.title_name] = (bool(is_ok), float(deviation))
            return

        existing_ok = bool(existing[0])
        try:
            existing_deviation = float(existing[1])
        except (TypeError, ValueError):
            existing_deviation = float("nan")

        if np.isfinite(existing_deviation) and np.isfinite(float(deviation)):
            merged_deviation = max(existing_deviation, float(deviation))
        elif np.isfinite(existing_deviation):
            merged_deviation = existing_deviation
        else:
            merged_deviation = float(deviation)
        self.data_struct.analysis_result_dict[self.title_name] = (existing_ok and bool(is_ok), merged_deviation)


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

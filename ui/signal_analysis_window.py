import copy
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
from PyQt5.QtCore import Qt, QModelIndex, QRectF, QTimer
from PyQt5.QtGui import QIcon, QTextCursor, QTextCharFormat, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QTableWidgetItem,
    QHeaderView,
    QToolTip,
)
from scipy.signal import find_peaks

from base.pdf_result_exporter import export_plot_widget_image
from base.core_algorithm.harmonic_distortion.weighted import apply_weighting_filter
from base.data_struct.data_deal_struct import DataDealStruct
from base.load_audio import load_audio_simple
from base.log_manager import LogManager
from base.predict_model import predict_from_audio
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from base.pre_processing.audio_peak_detection import peak_detection
from base.pre_processing.audio_equalizer import AudioEqualizer
from base.core_algorithm.response import FrequencyResponseAnalyzer, SplFrequencyAnalyzer, FftAnalyzer
from base.core_algorithm.response.dominant_tone_analyzer import (
    FrequencyInterval,
    find_dominant_fba_bands,
    find_dominant_fft_peaks,
    parse_frequency_intervals,
)
from base.stimulus_signal.methods import analysis_stimulus_method
from base.core_algorithm.response.frequency_band_analyzer import (
    FrequencyBandAnalyzer,
    BandAnalysisResult,
    Threshold as BandThreshold,
)
from base.core_algorithm.mel_spectrogram import compute_mel_spectrogram, hz_to_mel
from base.core_algorithm.modulation_map import compute_modulation_map
from base.core_algorithm.response.prominence_ratio_analyzer import (
    ProminenceRatioAnalyzer,
    ProminenceRatioParams,
)
from base.training_model_management import TrainingModelManagement
from base.utils.smooth import smooth
from base.utils.octave_smoothing import smooth_to_octave_grid
from consts import error_code, ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.widgets import MessageBox, TextEdit, Label, TableWidget
from ui.graph_widget import plot_2d_image, custom_log_tick_strings, LimitPlotUtils
from ui.reference_spectrum_analysis_window import ReferenceSpectrumCompareWindow
from ui.ui_src import ui_resources


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
        "FFT": FftAnalysis,
        "FR": Frequency,
        "RSC": ReferenceSpectrumCompareWindow,
        "HD": Distortion,
        "RB": RubAndBuzz,  # Rub & Buzz (high-order 10th-35th harmonic distortion)
        "PRB": PerceptualRubAndBuzz,  # Perceptual Rub & Buzz (2nd-35th harmonics, psychoacoustic loudness in phons)
        "AI": AI,
        "Spec": Spectrogram,
        "Mel": Mel,
        "Modulation": Modulation,
        "LP": LooseParticle,
        "PD": PeakDetection,
        "PM": PatternMatch,
        "ED": PipelinePdPm,
        "FBA": FrequencyBandAnalysis,
        "PR": ProminenceRatioAnalysis,
        "LOUD": LoudnessAnalysis,
        "SHRP": SharpnessAnalysis,
        "ROUGH": RoughnessAnalysis,
    }
    return class_mapping


def _export_table_widget_for_pdf(table_widget, title="分析表格"):
    if table_widget is None:
        return []
    row_count = table_widget.rowCount()
    column_count = table_widget.columnCount()
    if row_count <= 0 or column_count <= 0:
        return []

    headers = []
    for col in range(column_count):
        header_item = table_widget.horizontalHeaderItem(col)
        headers.append(header_item.text() if header_item is not None else "")

    rows = []
    for row in range(row_count):
        row_values = []
        for col in range(column_count):
            item = table_widget.item(row, col)
            row_values.append(item.text() if item is not None else "")
        rows.append(row_values)

    return [{"title": title, "headers": headers, "rows": rows}]


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


def _load_default_analysis_item_config(item_type: str) -> dict:
    default_config_path = os.path.join(DEFAULT_DIR, "ui/ui_config/analysis_default_config.json")
    try:
        with open(default_config_path, "r", encoding="utf-8") as file:
            defaults = json.load(file)
    except Exception:
        return {}
    item_cfg = defaults.get(item_type, {})
    return copy.deepcopy(item_cfg) if isinstance(item_cfg, dict) else {}


def _merge_loudness_config_with_defaults(loud_cfg: dict) -> dict:
    merged = _load_default_analysis_item_config("LOUD")
    source = copy.deepcopy(loud_cfg) if isinstance(loud_cfg, dict) else {}
    for section in ("display", "save", "advanced"):
        section_defaults = dict(merged.get(section, {}) or {})
        section_source = source.pop(section, {}) or {}
        if isinstance(section_source, dict):
            section_defaults.update(section_source)
        merged[section] = section_defaults
    merged.update(source)
    merged.pop("type", None)
    return merged


def resolve_analysis_channel_signal(
    data_struct: DataDealStruct, analysis_config: dict, title_name: str, strict: bool = True
):
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

    if np.unique(x_b).size != x_b.size and x_c.size == y_c.size:
        x_c_flat = np.ravel(x_c)
        y_c_flat = np.ravel(y_c)
        current_mask = np.isfinite(x_c_flat) & np.isfinite(y_c_flat)
        if int(np.count_nonzero(current_mask)) == x_b.size:
            current_order = np.argsort(x_c_flat[current_mask], kind="stable")
            baseline_order = np.argsort(x_b, kind="stable")
            current_sorted_x = x_c_flat[current_mask][current_order]
            baseline_sorted_x = x_b[baseline_order]
            if current_sorted_x.shape == baseline_sorted_x.shape and np.allclose(
                current_sorted_x,
                baseline_sorted_x,
                rtol=1e-9,
                atol=1e-9,
            ):
                paired_deviation = y_c_flat[current_mask][current_order] - y_b[baseline_order]
                deviation = np.asarray(y_c_flat, dtype=float).copy()
                current_indices = np.flatnonzero(current_mask)
                deviation[current_indices[current_order]] = paired_deviation
                return deviation.reshape(y_c.shape)

    sort_idx = np.argsort(x_b, kind="stable")
    x_b = x_b[sort_idx]
    y_b = y_b[sort_idx]
    x_b, uniq_idx = np.unique(x_b, return_index=True)
    y_b = y_b[uniq_idx]
    if x_b.size < 2:
        return y_c

    interp = np.interp(x_c, x_b, y_b)
    in_range = (x_c >= float(np.min(x_b))) & (x_c <= float(np.max(x_b)))
    interp = np.where(in_range, interp, np.nan)
    return y_c - interp


def _analysis_stimulus_metadata(stimulus_info: dict, sample_rate):
    stimulus_info = stimulus_info or {}
    stimulus_method = analysis_stimulus_method(stimulus_info.get("stimulus_method", "steps"))
    metadata = {
        "stimulus_method": stimulus_method,
        "stimulus_type": stimulus_info.get("stimulus_type", "linear"),
        "start_freq": stimulus_info.get("start_freq"),
        "stop_freq": stimulus_info.get("stop_freq"),
        "num_steps": stimulus_info.get("num_steps"),
        "total_time": stimulus_info.get("total_time"),
        "repeat_times": stimulus_info.get("repeat_times"),
        "sample_rate": sample_rate,
    }
    if stimulus_method != "frequency_stepped":
        return metadata

    passthrough_keys = (
        "frequency_mode",
        "frequencies",
        "segments",
        "step_durations",
        "min_duration",
        "min_cycles",
        "repeat_times",
        "schedule_sample_rate",
        "schedule_provenance",
        "schedule_algorithm",
        "per_repetition_sample_count",
        "alignment_sample_count",
        "playback_sample_count",
        "effective_start_freq",
        "effective_stop_freq",
        "resolution",
        "transition_hz",
        "safe_max_freq",
        "frequency_clamped",
    )
    for key in passthrough_keys:
        if key in stimulus_info:
            metadata[key] = stimulus_info[key]
    metadata["stimulus_type"] = stimulus_info.get("stimulus_type", metadata.get("frequency_mode", "custom_linear"))
    metadata["frequency_mode"] = stimulus_info.get("frequency_mode", metadata["stimulus_type"])
    return metadata


def _has_duplicate_finite_frequency_points(frequencies) -> bool:
    freq = np.asarray(frequencies, dtype=float)
    freq = freq[np.isfinite(freq)]
    if freq.size <= 1:
        return False
    return np.unique(freq).size != freq.size


class AnalysisResultSummaryWindow(QWidget):
    """
    Summary window for DataDealStruct.analysis_result_dict.

    Displays a simple table: Analysis Item / Result(OK/NG).
    """

    def __init__(self, result_dict: dict[str, bool], title: str = "分析结果汇总"):
        super().__init__()
        self.setWindowTitle(title)
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))

        self._overall_label = Label(self)
        self._overall_label.setObjectName("overallResultLabel")
        overall_font = QFont()
        overall_font.setPixelSize(22)
        self._overall_label.setFont(overall_font)
        self._overall_label.setAlignment(Qt.AlignCenter)

        self._table = TableWidget(self)
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["分析项", "偏差值", "结果"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(TableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(TableWidget.SelectRows)
        self._table.setSelectionMode(TableWidget.SingleSelection)
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

        # Overall judgment: all OK -> OK else NG
        overall_ok = True
        for _, (ok, _dev) in items:
            if not bool(ok):
                overall_ok = False
                break
        # set_results 里
        overall_text = "OK" if overall_ok else "NG"
        self._overall_label.setText(f"最终结果：{overall_text}")
        self._overall_label.setProperty("resultState", "ok" if overall_ok else "ng")

        # 动态属性变更后，触发重新应用 QSS
        self._overall_label.style().unpolish(self._overall_label)
        self._overall_label.style().polish(self._overall_label)
        self._overall_label.update()

        self._table.setRowCount(len(items))
        for row, (name, (ok, deviation)) in enumerate(items):
            name_item = QTableWidgetItem(str(name))
            if "SPL" in name:
                deviation = f"{deviation:.2f} dB"
            elif "FR" in name:
                deviation = f"{deviation:.2f} dB"
            elif "RSC" in name:
                deviation = f"{deviation:.2f} dB"
            elif "PRB" in name:
                deviation = f"{deviation:.2f} phon"
            elif "HD" in name or "RB" in name:
                deviation = f"{deviation:.2f} %"
            deviation_item = QTableWidgetItem(str(deviation))
            result_text = "OK" if ok else "NG"
            result_item = QTableWidgetItem(result_text)
            result_item.setTextAlignment(Qt.AlignCenter)

            # color hint
            if ok:
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
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))

        self.analysis_plot.setBackground("white")

        layout = QVBoxLayout()
        layout.addWidget(self.analysis_plot)
        self.setLayout(layout)

    def set_plot_font_size(self, font_size: int):
        font_size = ui_style_const.scale_size_px(font_size)
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

    def export_pdf_images(self, output_dir):
        image_path = export_plot_widget_image(self.analysis_plot, output_dir, "analysis_graph")
        return [{"title": self.windowTitle(), "path": image_path}]

    @staticmethod
    def apply_plot_font_style(plot_widget, font_size: int = 20):
        font_size = ui_style_const.scale_size_px(font_size)
        font = QFont()
        font.setPixelSize(font_size)
        for axis_name in ("bottom", "left"):
            axis = plot_widget.getAxis(axis_name)
            axis.setTickFont(font)
            axis.setTextPen("black")
            axis.setLabel(axis.labelText, **{"font-size": f"{font_size}px"})


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
            MessageBox.warning(self, "提示", str(e))
            self.plot_graph([], [])
            self.result = {"freq_value": [], "harmonic": [], "thd": [], "thd_raw": []}
            return self.result
        sample_rate = self.data_struct.sample_rate
        stimulus_info = self.data_struct.stimulus_info

        if recorded_signal is None or sample_rate is None or stimulus_info is None:
            raise ValueError("Missing required data: recorded_signal, sample_rate, or stimulus_info")

        # Convert stimulus_info to stimulus_metadata format.
        stimulus_metadata = _analysis_stimulus_metadata(stimulus_info, sample_rate)
        stimulus_method = stimulus_metadata["stimulus_method"]

        # Call the new three-phase architecture
        atfra = AudioThdFrequencyResponseAnalysis()
        thd_kwargs = {"stimulus_metadata": stimulus_metadata, "harmonic_orders": self.selected_harmonics}

        freq_value, harmonic, thd = atfra.calculate_thd_three_phase(recorded_signal, sample_rate, thd_kwargs)

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
                MessageBox.warning(self, "提示", "未找到黄金样本基准文件或基准数据，已按原始曲线分析")

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
            limit_mode = str(analysis_config.get("limit_mode", "csv") or "csv").lower()
            if limit_mode == "manual" and valid_data:
                n = len(freq_value)
                upper_ok = bool(analysis_config.get("manual_upper_enabled", True))
                lower_ok = bool(analysis_config.get("manual_lower_enabled", False))
                upper = float(analysis_config.get("manual_upper", 0.0) or 0.0)
                lower = float(analysis_config.get("manual_lower", 0.0) or 0.0)
                csv_freq_list = freq_value
                csv_upper_list = (np.full(n, upper) if upper_ok else np.full(n, np.nan)).tolist()
                csv_lower_list = (np.full(n, lower) if lower_ok else np.full(n, np.nan)).tolist()
            else:
                result = analysis_config.get("limit_data")
                if not result:
                    return
                csv_freq_list, csv_upper_list, csv_lower_list = result

            if valid_data:
                # Use common function for plot setup
                LimitPlotUtils.setup_limit_plot(
                    self.analysis_plot,
                    freq_value,
                    thd,
                    csv_freq_list,
                    csv_upper_list,
                    csv_lower_list,
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

        Overrides parent method to use calculate_perceptual_thd_three_phase instead of
        calculate_thd_three_phase.
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
            MessageBox.warning(self, "提示", str(e))
            self.plot_graph([], [])
            self.result = {"freq_value": [], "harmonic": [], "thd": [], "thd_raw": []}
            return self.result
        sample_rate = self.data_struct.sample_rate
        stimulus_info = self.data_struct.stimulus_info

        if recorded_signal is None or sample_rate is None or stimulus_info is None:
            raise ValueError("Missing required data: recorded_signal, sample_rate, or stimulus_info")

        # Convert stimulus_info to stimulus_metadata format.
        stimulus_metadata = _analysis_stimulus_metadata(stimulus_info, sample_rate)
        stimulus_method = stimulus_metadata["stimulus_method"]

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

        freq_value, harmonic, perceptual_loudness = atfra.calculate_perceptual_thd_three_phase(
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
                MessageBox.warning(self, "提示", "未找到黄金样本基准文件或基准数据，已按原始曲线分析")

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
            limit_mode = str(analysis_config.get("limit_mode", "csv") or "csv").lower()
            if limit_mode == "manual" and valid_data:
                n = len(freq_value)
                upper_ok = bool(analysis_config.get("manual_upper_enabled", True))
                lower_ok = bool(analysis_config.get("manual_lower_enabled", False))
                upper = float(analysis_config.get("manual_upper", 0.0) or 0.0)
                lower = float(analysis_config.get("manual_lower", 0.0) or 0.0)
                csv_freq_list = freq_value
                csv_upper_list = (np.full(n, upper) if upper_ok else np.full(n, np.nan)).tolist()
                csv_lower_list = (np.full(n, lower) if lower_ok else np.full(n, np.nan)).tolist()
            else:
                result = analysis_config.get("limit_data")
                if not result:
                    return
                csv_freq_list, csv_upper_list, csv_lower_list = result

            if valid_data:

                # 1) Plot main curve + limit curves (same as THD)
                LimitPlotUtils.setup_limit_plot(
                    self.analysis_plot,
                    freq_value,
                    perceptual_loudness,
                    csv_freq_list,
                    csv_upper_list,
                    csv_lower_list,
                    x_label="Frequency (Hz)",
                    y_label=self._prb_y_label,
                    log_x=True,
                    curve_name=self._prb_curve_label,
                )

                if self.selected_label is not None:
                    self.analysis_plot.setTitle(f"Perceived Loudness of {self.selected_label.text()} order")

                # 2) Use parent's _highlight_out_of_range_curve() for limit check + highlight
                #    This uses nearest-neighbor matching and highlights on original data points
                self._highlight_out_of_range_curve(
                    freq_value, perceptual_loudness, csv_freq_list, csv_upper_list, csv_lower_list
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
        weighting = self.analysis_config.get("weighting", "Z") if self.analysis_config else "Z"
        weighting = weighting.upper()
        if weighting == "A":
            return "SPL (dBA)"
        elif weighting == "B":
            return "SPL (dBB)"
        elif weighting == "C":
            return "SPL (dBC)"
        elif weighting == "D":
            return "SPL (dBD)"
        else:  # Z or None
            return "SPL (dB)"

    def calculate_spl(self):
        # calculate Sound Pressure Level according to recorded_signal
        try:
            recorded_signal = resolve_analysis_channel_signal(self.data_struct, self.analysis_config, self.title_name)
        except Exception as e:
            MessageBox.warning(self, "提示", str(e))
            return False
        sample_rate = self.data_struct.sample_rate
        reference_pressure = 20e-6
        window_size = 1201
        weighting = self.analysis_config.get("weighting", "Z") if self.analysis_config else "Z"
        if weighting and weighting.upper() not in ["NONE", "Z"]:
            recorded_signal = apply_weighting_filter(
                recorded_signal, sample_rate, weighting=weighting, zero_phase=False
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
            limit_mode = str(self.analysis_config.get("limit_mode", "csv") or "csv").lower()
            if limit_mode == "manual":
                n = len(signal_duration)
                upper_ok = bool(self.analysis_config.get("manual_upper_enabled", True))
                lower_ok = bool(self.analysis_config.get("manual_lower_enabled", False))
                upper = float(self.analysis_config.get("manual_upper", 0.0) or 0.0)
                lower = float(self.analysis_config.get("manual_lower", 0.0) or 0.0)
                csv_time_list = signal_duration
                csv_upper_list = (np.full(n, upper) if upper_ok else np.full(n, np.nan)).tolist()
                csv_lower_list = (np.full(n, lower) if lower_ok else np.full(n, np.nan)).tolist()
            else:
                result = self.analysis_config.get("limit_data")
                if not result:
                    return False
                csv_time_list, csv_upper_list, csv_lower_list = result
            self.plot_spl_with_limits(signal_duration, signal_spl, csv_time_list, csv_upper_list, csv_lower_list)
        else:
            self.plot_spl(signal_duration, signal_spl)
        self.result = {
            "signal_duration": signal_duration.tolist(),
            "recorded_signal": recorded_signal.tolist(),
            "signal_spl": signal_spl.tolist(),
        }
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
            signal_duration,
            signal_spl,
            csv_time_list,
            csv_upper_list,
            csv_lower_list,
            x_label="Time (s)",
            y_label=self._get_spl_label(),
            log_x=False,
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
        self.analysis_plot.plot(signal_duration, signal_spl, pen=mkPen(color=(51, 196, 77), width=2))
        self.analysis_plot.setLabel("left", "SPL (dB)")
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
            MessageBox.warning(self, "提示", str(e))
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

        stimulus_metadata = _analysis_stimulus_metadata(stimulus_info, sample_rate)
        stimulus_method = stimulus_metadata["stimulus_method"]

        try:
            analyzer = SplFrequencyAnalyzer(sample_rate=int(sample_rate))
            result = analyzer.compute(
                recorded_signal,
                stimulus_metadata=stimulus_metadata,
                v2pa_factor=self.v2pa_factor,
                splf_calc_mode=analysis_config.get("splf_calc_mode", "fundamental"),
            )
        except Exception as e:
            MessageBox.warning(self, "提示", f"声压级-频率计算失败: {str(e)[:200]}")
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

        skip_smoothing_for_duplicates = (
            stimulus_method == "frequency_stepped" and _has_duplicate_finite_frequency_points(frequency_list)
        )
        if octave_smoothing in {1, 3, 6, 12, 24, 48} and spl_db.size > 1 and not skip_smoothing_for_duplicates:
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
                MessageBox.warning(self, "提示", "未找到黄金样本基准文件或基准数据，已按原始曲线分析")

        limit_checked = analysis_config.get("limit_checked")
        if limit_checked:
            limit_mode = str(analysis_config.get("limit_mode", "csv") or "csv").lower()
            if limit_mode == "manual":
                n = len(frequency_list)
                upper_ok = bool(analysis_config.get("manual_upper_enabled", True))
                lower_ok = bool(analysis_config.get("manual_lower_enabled", False))
                upper = float(analysis_config.get("manual_upper", 0.0) or 0.0)
                lower = float(analysis_config.get("manual_lower", 0.0) or 0.0)
                csv_freq_list = frequency_list
                csv_upper_list = (np.full(n, upper) if upper_ok else np.full(n, np.nan)).tolist()
                csv_lower_list = (np.full(n, lower) if lower_ok else np.full(n, np.nan)).tolist()
            else:
                result = analysis_config.get("limit_data")
                if not result:
                    return False
                csv_freq_list, csv_upper_list, csv_lower_list = result
            self.plot_spl_frequency_with_limits(frequency_list, spl_db, csv_freq_list, csv_upper_list, csv_lower_list)
        else:
            self.plot_spl_frequency(frequency_list, spl_db)

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
            freq_valid,
            spl_valid,
            csv_freq_list,
            csv_upper_list,
            csv_lower_list,
            x_label="Frequency (Hz)",
            y_label="SPL (dB)",
            log_x=True,
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
        self.analysis_plot.plot(frequency_list, spl_db, pen=mkPen(color=(51, 196, 77), width=2))
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
            MessageBox.warning(self, "提示", str(e))
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
        stimulus_metadata = _analysis_stimulus_metadata(stimulus_info, sr)

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
            MessageBox.warning(self, "提示", f"频响计算失败: {str(e)[:200]}")
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
                MessageBox.warning(self, "提示", "未找到黄金样本基准文件或基准数据，已按原始曲线分析")
        limit_checked = analysis_config.get("limit_checked")
        if limit_checked:
            limit_mode = str(self.analysis_config.get("limit_mode", "csv") or "csv").lower()
            if limit_mode == "manual":
                n = len(frequency_list)
                upper_ok = bool(self.analysis_config.get("manual_upper_enabled", True))
                lower_ok = bool(self.analysis_config.get("manual_lower_enabled", False))
                upper = float(self.analysis_config.get("manual_upper", 0.0) or 0.0)
                lower = float(self.analysis_config.get("manual_lower", 0.0) or 0.0)
                csv_freq_list = frequency_list
                csv_upper_list = (np.full(n, upper) if upper_ok else np.full(n, np.nan)).tolist()
                csv_lower_list = (np.full(n, lower) if lower_ok else np.full(n, np.nan)).tolist()
            else:
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
            MessageBox.warning(None, "提示", f"Excel路径为空, 请选择一个Excel文件路径！")
            return None
        ext = os.path.splitext(excel_path)[1].lower()
        if ext == ".csv":
            with open(excel_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
        else:
            MessageBox.warning(None, "提示", f"不支持对这种Excel格式的分析:\n{excel_path}")
            return None

        if not rows or len(rows) == 0:
            MessageBox.warning(None, "提示", f"CSV文件为空或格式不正确:\n{excel_path}")
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
            MessageBox.warning(None, "提示", "Excel/CSV 格式不符合要求!")
            return None
        for index, row in enumerate(rows[1:], start=2):
            csv_line_no = index
            if lenth == 3 and upperbound:
                try:
                    fval = float(row[0])
                    uval = float(row[1])
                    lval = float(row[2])
                except ValueError:
                    MessageBox.warning(None, "提示", f"CSV 数据错误:第 {csv_line_no} 行存在空值或非数字,无法解析\n")
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
                    MessageBox.warning(None, "提示", f"CSV 数据错误:第 {csv_line_no} 行存在空值或非数字,无法解析\n")
                    return None
                csv_freq_list.append(fval)
                csv_upper_list.append(uval)
                csv_lower_list.append(lval)
            elif lenth == 2 and upperbound:
                try:
                    fval = float(row[0])
                    uval = float(row[1])
                except ValueError:
                    MessageBox.warning(None, "提示", f"CSV 数据错误:第 {csv_line_no} 行存在空值或非数字,无法解析\n")
                    return None
                csv_freq_list.append(fval)
                csv_upper_list.append(uval)
                csv_lower_list.append(np.nan)
            elif lenth == 2 and not upperbound:
                try:
                    fval = float(row[0])
                    lval = float(row[1])
                except ValueError:
                    MessageBox.warning(None, "提示", f"CSV 数据错误:第 {csv_line_no} 行存在空值或非数字,无法解析\n")
                    return None
                csv_freq_list.append(fval)
                csv_upper_list.append(np.nan)
                csv_lower_list.append(lval)
        for i, (x, u, l) in enumerate(zip(csv_freq_list, csv_upper_list, csv_lower_list)):
            if (u is not None) and (l is not None) and (not np.isnan(u)) and (not np.isnan(l)):
                if l > u:
                    MessageBox.warning(
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
            freq_valid,
            fr_valid,
            csv_freq_list,
            csv_upper_list,
            csv_lower_list,
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


class FftAnalysis(AnalysisGraphWidget):
    """Welch FFT spectrum analysis window."""

    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.v2pa_factor = None
        self.analysis_config = None
        self.result = {}
        self.title_name = title_name
        self.setWindowTitle(title_name)

    def calculate_fft(self):
        config = self.analysis_config or {}
        try:
            recorded_signal = resolve_analysis_channel_signal(self.data_struct, config, self.title_name)
        except Exception as e:
            MessageBox.warning(self, "提示", str(e))
            self.plot_fft(np.array([]), np.array([]), x_axis_scale=config.get("x_axis_scale", "log"))
            self.result = {}
            return False

        sample_rate = self.data_struct.sample_rate
        if sample_rate is None:
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
                v2pa_factor=self.v2pa_factor or 1.0,
            )
        except Exception as e:
            MessageBox.warning(self, "提示", f"FFT 分析失败: {str(e)[:200]}")
            self.plot_fft(np.array([]), np.array([]), x_axis_scale=config.get("x_axis_scale", "log"))
            self.result = {}
            return False

        frequency = np.asarray(main_result.frequencies_hz, dtype=float)
        fft_db = np.asarray(main_result.spectrum_db, dtype=float)
        baseline_db = None

        baseline_file_path = str(config.get("baseline_file_path", "") or "").strip()
        if baseline_file_path:
            try:
                baseline_signal, _baseline_sr = librosa.load(baseline_file_path, sr=int(sample_rate), mono=True)
                baseline_result = analyzer.analyze(
                    baseline_signal,
                    fs=int(sample_rate),
                    n_fft=n_fft,
                    window=window,
                    overlap_ratio=overlap_ratio,
                    weighting=weighting,
                    v2pa_factor=self.v2pa_factor or 1.0,
                )
                baseline_db = np.interp(
                    frequency,
                    np.asarray(baseline_result.frequencies_hz, dtype=float),
                    np.asarray(baseline_result.spectrum_db, dtype=float),
                    left=np.nan,
                    right=np.nan,
                )
                if bool(config.get("baseline_smooth_third_octave", False)):
                    baseline_db = self._smooth_baseline_third_octave(frequency, baseline_db)
            except Exception as e:
                MessageBox.warning(self, "提示", f"背景噪声基线加载失败: {str(e)[:200]}")
                baseline_db = None

        display_mode = str(config.get("baseline_display_mode", "overlay") or "overlay")
        curves = self._build_display_curves(frequency, fft_db, baseline_db, display_mode)
        plot_y = np.asarray(curves["plot_y"], dtype=float)
        delta_db = curves["delta_db"]

        x_axis_scale = str(config.get("x_axis_scale", "log") or "log")
        focus_enabled = bool(config.get("focus_range_enabled", True))
        focus_min_hz = float(config.get("focus_min_hz", 100) or 100)
        focus_max_hz = float(config.get("focus_max_hz", 20000) or 20000)
        mask = self._build_frequency_mask(frequency, focus_enabled, focus_min_hz, focus_max_hz, x_axis_scale)

        plot_x = frequency[mask]
        display_y = plot_y[mask]
        display_baseline = baseline_db[mask] if baseline_db is not None else None
        display_fft = fft_db[mask]
        display_delta = delta_db[mask] if isinstance(delta_db, np.ndarray) else None
        dominant_y = display_y if bool(config.get("dominant_tone_use_display_curve", True)) else display_fft
        dominant_tones = self._detect_dominant_tones(
            plot_x,
            dominant_y,
            config,
            fallback_low_hz=float(np.nanmin(plot_x)) if plot_x.size else 0.0,
            fallback_high_hz=float(np.nanmax(plot_x)) if plot_x.size else 0.0,
        )

        limit_checked = bool(config.get("limit_checked", False))
        y_label = f"FFT Spectrum [dB({weighting}) SPL]" if weighting != "Z" else "FFT Spectrum [dB SPL]"
        if display_mode == "delta" and display_delta is not None:
            y_label = "FFT - Baseline [dB]"

        if limit_checked:
            limit_data = config.get("limit_data")
            if not limit_data:
                MessageBox.warning(self, "提示", "已启用阈值，但未加载 CSV 配置文件。")
                return False
            csv_x, csv_upper, csv_lower = limit_data
            self.plot_fft_with_limits(
                plot_x,
                display_y,
                csv_x,
                csv_upper,
                csv_lower,
                x_axis_scale=x_axis_scale,
                y_label=y_label,
                baseline_y=display_baseline if display_mode == "overlay" else None,
                dominant_tones=dominant_tones,
            )
        else:
            self.plot_fft(
                plot_x,
                display_y,
                x_axis_scale=x_axis_scale,
                y_label=y_label,
                baseline_y=display_baseline if display_mode == "overlay" else None,
                dominant_tones=dominant_tones,
            )

        self.result = {
            "frequency_bins": plot_x.tolist(),
            "fft_db": display_fft.tolist(),
            "baseline_db": display_baseline.tolist() if isinstance(display_baseline, np.ndarray) else [],
            "delta_db": display_delta.tolist() if isinstance(display_delta, np.ndarray) else [],
            "plot_db": display_y.tolist(),
            "weighting": weighting,
            "display_mode": display_mode,
            "baseline_smooth_third_octave": bool(config.get("baseline_smooth_third_octave", False)),
            "n_fft": n_fft,
            "window": window,
            "overlap_ratio": overlap_ratio,
            "x_axis_scale": x_axis_scale,
            "dominant_tones": dominant_tones,
        }
        return self.result

    @staticmethod
    def _build_display_curves(frequency, spectrum_db, baseline_db, display_mode):
        spectrum = np.asarray(spectrum_db, dtype=float)
        baseline = None if baseline_db is None else np.asarray(baseline_db, dtype=float)
        delta = None
        plot_y = spectrum
        if str(display_mode or "overlay") == "delta" and baseline is not None:
            delta = spectrum - baseline
            plot_y = delta
        elif baseline is not None:
            delta = spectrum - baseline
        return {
            "frequency": frequency,
            "plot_y": plot_y,
            "fft_db": spectrum,
            "baseline_db": baseline,
            "delta_db": delta,
        }

    @staticmethod
    def _smooth_baseline_third_octave(frequency, baseline_db):
        freq = np.asarray(frequency, dtype=float)
        baseline = np.asarray(baseline_db, dtype=float)
        smoothed = np.full_like(baseline, np.nan, dtype=float)
        factor = 2.0 ** (1.0 / 6.0)

        valid_points = np.isfinite(freq) & np.isfinite(baseline)
        if not np.any(valid_points):
            return smoothed

        sorted_idx = np.argsort(freq[valid_points])
        sorted_freq = freq[valid_points][sorted_idx]
        sorted_power = np.power(10.0, baseline[valid_points][sorted_idx] / 10.0)
        prefix_power = np.concatenate(([0.0], np.cumsum(sorted_power)))
        prefix_count = np.arange(sorted_power.size + 1, dtype=float)

        valid_centers = np.isfinite(freq) & (freq > 0)
        if not np.any(valid_centers):
            return smoothed

        f_low = freq[valid_centers] / factor
        f_high = freq[valid_centers] * factor
        left = np.searchsorted(sorted_freq, f_low, side="left")
        right = np.searchsorted(sorted_freq, f_high, side="right")
        counts = prefix_count[right] - prefix_count[left]
        power_sum = prefix_power[right] - prefix_power[left]

        center_values = np.full(counts.shape, np.nan, dtype=float)
        non_empty = counts > 0
        center_values[non_empty] = 10.0 * np.log10(np.maximum(power_sum[non_empty] / counts[non_empty], 1e-30))
        smoothed[valid_centers] = center_values
        return smoothed

    @staticmethod
    def _build_frequency_mask(frequency, focus_enabled, focus_min_hz, focus_max_hz, x_axis_scale):
        freq = np.asarray(frequency, dtype=float)
        mask = np.isfinite(freq)
        if str(x_axis_scale or "linear").lower() == "log":
            mask &= freq > 0
        if focus_enabled:
            mask &= (freq >= float(focus_min_hz)) & (freq <= float(focus_max_hz))
        return mask

    @staticmethod
    def _apply_frequency_focus(frequency, values, focus_enabled, focus_min_hz, focus_max_hz, x_axis_scale):
        freq = np.asarray(frequency, dtype=float)
        val = np.asarray(values, dtype=float)
        mask = np.isfinite(freq)
        if str(x_axis_scale or "linear").lower() == "log":
            mask &= freq > 0
        if focus_enabled:
            mask &= (freq >= float(focus_min_hz)) & (freq <= float(focus_max_hz))
        return freq[mask], val[mask]

    @staticmethod
    def _detect_dominant_tones(frequency, values, config, *, fallback_low_hz, fallback_high_hz):
        if not bool(config.get("dominant_tone_enabled", False)):
            return []
        intervals = parse_frequency_intervals(config.get("dominant_tone_intervals_text", ""))
        if not intervals and fallback_high_hz > fallback_low_hz:
            intervals = [FrequencyInterval(float(fallback_low_hz), float(fallback_high_hz), "Overall")]
        return find_dominant_fft_peaks(
            frequency,
            values,
            intervals,
            min_prominence_db=float(config.get("dominant_tone_min_prominence_db", 3.0) or 0.0),
        )

    @staticmethod
    def _dominant_annotation_x(freq, x_axis_scale):
        freq = float(freq)
        if str(x_axis_scale or "linear").lower() == "log":
            if not (np.isfinite(freq) and freq > 0):
                return np.nan
            return float(np.log10(freq))
        return freq

    def _draw_fft_dominant_tones(self, dominant_tones, *, x_axis_scale="linear"):
        for tone in dominant_tones or []:
            freq = float(tone.get("frequency_hz", np.nan))
            level = float(tone.get("level_db", np.nan))
            plot_x = self._dominant_annotation_x(freq, x_axis_scale)
            if not (np.isfinite(plot_x) and np.isfinite(level)):
                continue
            line = pg.InfiniteLine(pos=plot_x, angle=90, pen=mkPen(color=(255, 152, 0), width=1))
            self.analysis_plot.addItem(line)
            text = pg.TextItem(
                f"{tone.get('interval_label', '')}\n{freq:.1f} Hz",
                color=(255, 152, 0),
                anchor=(0.0, 1.0),
            )
            text.setPos(plot_x, level)
            self.analysis_plot.addItem(text)

    def plot_fft(
        self,
        frequency,
        spectrum_db,
        *,
        x_axis_scale="log",
        y_label="FFT Spectrum [dB SPL]",
        baseline_y=None,
        dominant_tones=None,
    ):
        self.analysis_plot.clear()
        self.analysis_plot.plot(frequency, spectrum_db, pen=mkPen(color=(51, 196, 77), width=2), name="FFT")
        if baseline_y is not None:
            self.analysis_plot.plot(frequency, baseline_y, pen=mkPen(color=(128, 128, 128), width=2), name="Baseline")
        self.analysis_plot.setLabel("left", y_label)
        self.analysis_plot.setLabel("bottom", "Frequency (Hz)")
        self.analysis_plot.setLogMode(x=str(x_axis_scale).lower() == "log", y=False)
        self._draw_fft_dominant_tones(dominant_tones, x_axis_scale=x_axis_scale)
        self.analysis_plot.showGrid(x=True, y=True)

    def plot_fft_with_limits(
        self,
        frequency,
        spectrum_db,
        csv_freq_list,
        csv_upper_list,
        csv_lower_list,
        *,
        x_axis_scale="log",
        y_label="FFT Spectrum [dB SPL]",
        baseline_y=None,
        dominant_tones=None,
    ):
        freq_arr = np.asarray(frequency, dtype=float)
        y_arr = np.asarray(spectrum_db, dtype=float)
        mask = np.isfinite(freq_arr) & np.isfinite(y_arr)
        if str(x_axis_scale).lower() == "log":
            mask &= freq_arr > 0
        freq_valid = freq_arr[mask]
        y_valid = y_arr[mask]
        sort_idx = None
        if freq_valid.size > 1:
            sort_idx = np.argsort(freq_valid)
            freq_valid = freq_valid[sort_idx]
            y_valid = y_valid[sort_idx]

        LimitPlotUtils.setup_limit_plot(
            self.analysis_plot,
            freq_valid,
            y_valid,
            np.asarray(csv_freq_list, dtype=float),
            np.asarray(csv_upper_list, dtype=float),
            np.asarray(csv_lower_list, dtype=float),
            x_label="Frequency (Hz)",
            y_label=y_label,
            log_x=str(x_axis_scale).lower() == "log",
            curve_name="FFT",
        )
        if baseline_y is not None:
            base = np.asarray(baseline_y, dtype=float)[mask]
            if sort_idx is not None:
                base = base[sort_idx]
            self.analysis_plot.plot(freq_valid, base, pen=mkPen(color=(128, 128, 128), width=2), name="Baseline")
        self._draw_fft_dominant_tones(dominant_tones, x_axis_scale=x_axis_scale)

        try:
            out_mask, plot_x, plot_y, deviation, is_ok = LimitPlotUtils.check_interp_limits(
                freq_valid,
                y_valid,
                np.asarray(csv_freq_list, dtype=float),
                np.asarray(csv_upper_list, dtype=float),
                np.asarray(csv_lower_list, dtype=float),
            )
        except Exception:
            is_ok, deviation = False, 0.0
            out_mask = np.zeros(len(freq_valid), dtype=bool)
            plot_x, plot_y = freq_valid, y_valid
        self.data_struct.analysis_result_dict[self.title_name] = (is_ok, deviation)
        LimitPlotUtils.plot_out_segments(self.analysis_plot, plot_x, plot_y, out_mask)


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
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        ai_analyse_layout = self.create_ai_analyse_layout()
        self.setLayout(ai_analyse_layout)

    def create_ai_analyse_layout(self):
        ai_analyse_layout = QVBoxLayout()
        analyse_score_layout = QHBoxLayout()
        self.ai_analyse_score_textedit = TextEdit()
        self.ai_analyse_score_textedit.setAlignment(Qt.AlignCenter)
        self.ai_analyse_score_textedit.setDisabled(True)

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
        try:
            ai_signal = resolve_analysis_channel_signal(self.data_struct, self.analysis_config, self.title_name)
        except Exception as e:
            self.ai_analyse_score_textedit.setPlainText(str(e))
            return

        if acq_mode in ["IMPORT_AUDIO", "IMPORT_STIMULUS_AUDIO"]:
            query_code, query_result = TrainingModelManagement().get_input_dim_info_by_name(model_name)
            if query_code == error_code.OK:
                input_dim = str(query_result).split("x")[0].strip()
                if input_dim != str(len(ai_signal)):
                    self.ai_analyse_score_textedit.setPlainText("模型与音频时长不匹配")
                    return
                else:
                    self.default_logger.info("The model matches the audio duration. Starting analysis...")
            else:
                self.ai_analyse_score_textedit.setPlainText("查询数据库模型时长失败")
                return
        code, result = self.get_model_info(model_name, self.default_logger)
        if code != error_code.OK or not os.path.exists(result[0]):
            self.ai_analyse_score_textedit.setPlainText("模型不存在，请重新选择！")
        else:
            model_path, config_path = result
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
    PDF_EXPORT_LEFT_AXIS_MIN_WIDTH = 110

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
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
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

    def _prepare_plot_for_pdf_export(self, plot_widget):
        if plot_widget is None:
            return
        try:
            left_axis = plot_widget.getAxis("left")
            if left_axis is not None:
                left_axis.setWidth(max(left_axis.width(), self.PDF_EXPORT_LEFT_AXIS_MIN_WIDTH))
                app = QApplication.instance()
                if app is not None:
                    app.processEvents()
        except Exception:
            pass

    def export_pdf_images(self, output_dir):
        plot_widget = self.current_plot_widget
        if isinstance(plot_widget, pg.PlotWidget):
            export_widget = plot_widget
        else:
            export_widget = None
            if plot_widget is not None:
                children = plot_widget.findChildren(pg.PlotWidget)
                export_widget = children[0] if children else None
            if export_widget is None:
                export_widget = self.stft_plot_widget

        if export_widget is None:
            return []

        self._prepare_plot_for_pdf_export(export_widget)
        image_path = export_plot_widget_image(export_widget, output_dir, "spectrogram")
        return [{"title": self.title_name, "path": image_path}]


class Mel(QWidget):
    HOTSPOT_MARKER_SIZE = 12
    LEFT_AXIS_MIN_WIDTH = 120
    pdf_summary_exclude_fields = ("overall_spl_dba", "hotspot")

    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.analysis_config = None
        self.result = None
        self.title_name = title_name
        self.img_item = None
        self.colorbar = None
        self.analysis_region = None
        self.analysis_label = None
        self.core_region = None
        self.core_label = None
        self.hotspot_region = None
        self.hotspot_regions = []
        self.hotspot_label = None
        self.overall_spl_label = None
        self.plot_widget = None
        self.table_widget = None
        self.status_label = None
        self.v2pa_factor = None
        self.init_ui()
        self.setWindowTitle(title_name)

    def init_ui(self):
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        self.main_layout = QVBoxLayout(self)

        self.status_label = Label("Mel")
        self.status_label.setVisible(False)
        self.main_layout.addWidget(self.status_label)

        self.plot_widget = pg.PlotWidget(background="white")
        self.plot_widget.setLabel("bottom", "Time (s)")
        self.plot_widget.setLabel("left", "Mel frequency (Mel)")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self._reserve_left_axis_space()
        self.main_layout.addWidget(self.plot_widget, 3)

        self.table_widget = TableWidget()
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(
            [
                "Main tone (Hz)",
                "Hotspot band (kHz)",
                "Hotspot Mel band",
            ]
        )
        header = self.table_widget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.table_widget.verticalHeader().setVisible(False)
        self.table_widget.setEditTriggers(TableWidget.NoEditTriggers)
        self.main_layout.addWidget(self.table_widget, 1)
        self.setLayout(self.main_layout)

    def showEvent(self, event):
        super().showEvent(event)
        self._schedule_plot_layout_refresh()

    def _reserve_left_axis_space(self):
        if self.plot_widget is None:
            return
        left_axis = self.plot_widget.getAxis("left")
        if left_axis is not None:
            left_axis.setWidth(self.LEFT_AXIS_MIN_WIDTH)

    def _schedule_plot_layout_refresh(self):
        self._reserve_left_axis_space()
        QTimer.singleShot(0, self._refresh_plot_layout)

    def _refresh_plot_layout(self):
        self._reserve_left_axis_space()
        if self.plot_widget is None:
            return
        plot_item = self.plot_widget.getPlotItem()
        if plot_item is not None:
            plot_item.updateGeometry()
        self.plot_widget.updateGeometry()
        self.plot_widget.update()

    def calculate_mel(self):
        try:
            recorded_signal = resolve_analysis_channel_signal(self.data_struct, self.analysis_config, self.title_name)
        except Exception as e:
            MessageBox.warning(self, "提示", str(e))
            return None
        sample_rate = self.data_struct.sample_rate
        if recorded_signal is None or sample_rate is None:
            MessageBox.warning(self, "提示", "Mel 分析缺少录音数据或采样率。")
            return None

        try:
            analysis_result = compute_mel_spectrogram(
                np.asarray(recorded_signal, dtype=np.float64),
                int(sample_rate),
                self.analysis_config or {},
                v2pa_factor=self.v2pa_factor,
            )
        except Exception as e:
            MessageBox.warning(self, "提示", f"Mel 分析失败: {str(e)[:200]}")
            return None

        self._render_result(analysis_result)
        self.result = self._to_exportable_result(analysis_result)
        return self.result

    def _render_result(self, analysis_result):
        mel_db_a = np.asarray(analysis_result.get("mel_db_a", []), dtype=np.float64)
        times_s = np.asarray(analysis_result.get("times_s", []), dtype=np.float64)
        mel_axis = np.asarray(analysis_result.get("mel_axis", []), dtype=np.float64)
        mel_axis_edges = np.asarray(analysis_result.get("mel_axis_edges", []), dtype=np.float64)
        if mel_db_a.size == 0 or times_s.size == 0 or mel_axis.size == 0:
            return

        self._clear_plot()
        self.img_item = pg.ImageItem()
        self.plot_widget.addItem(self.img_item)

        display_matrix = mel_db_a.T
        self.img_item.setImage(display_matrix, autoLevels=False)

        x_min, x_max = float(times_s[0]), float(times_s[-1])
        if mel_axis_edges.size == mel_axis.size + 1:
            y_min, y_max = float(mel_axis_edges[0]), float(mel_axis_edges[-1])
        else:
            y_min, y_max = float(mel_axis[0]), float(mel_axis[-1])
        self.img_item.setRect(QRectF(x_min, y_min, max(x_max - x_min, 1e-9), max(y_max - y_min, 1e-9)))

        finite_values = mel_db_a[np.isfinite(mel_db_a)]
        if finite_values.size:
            vmax = float(np.nanpercentile(finite_values, 99.0))
            vmin = float(np.nanpercentile(finite_values, 5.0))
        else:
            vmax, vmin = 1.0, 0.0
        dynamic_range = float((self.analysis_config or {}).get("dynamic_range_db", 65.0) or 65.0)
        vmin = max(vmin, vmax - dynamic_range)
        if not np.isfinite(vmax) or not np.isfinite(vmin) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0

        color_map_name = (self.analysis_config or {}).get("color_map", "magma")
        color_map = self._build_color_map(color_map_name)
        self.img_item.setLookupTable(color_map.getLookupTable(nPts=256))
        self.img_item.setLevels([vmin, vmax])
        self.colorbar = pg.ColorBarItem(values=(vmin, vmax), width=25, colorMap=color_map)
        self.colorbar.setImageItem(self.img_item, insert_in=self.plot_widget.getPlotItem())

        self.plot_widget.setXRange(x_min, x_max, padding=0)
        display_y_min, display_y_max = self._display_y_range(analysis_result, y_min, y_max)
        self.plot_widget.setYRange(display_y_min, display_y_max, padding=0)
        self.plot_widget.setTitle("A-weighted Mel spectrogram", size="20px", color="black")

        self._plot_analysis_mel_band(display_y_min, display_y_max, analysis_result, x_min)
        self._plot_core_band(display_y_min, display_y_max, analysis_result, x_min)
        self._plot_overall_spl_label(analysis_result, x_min, x_max, display_y_min, display_y_max)
        self._plot_hotspot(analysis_result)
        self._update_table(analysis_result)
        self._update_fonts()
        self._schedule_plot_layout_refresh()

    def _clear_plot(self):
        self.plot_widget.clear()
        if self.colorbar is not None:
            try:
                self.colorbar.close()
            except Exception:
                pass
        self.img_item = None
        self.colorbar = None
        self.analysis_region = None
        self.analysis_label = None
        self.core_region = None
        self.core_label = None
        self.hotspot_region = None
        self.hotspot_regions = []
        self.hotspot_label = None
        self.overall_spl_label = None

    @staticmethod
    def _build_color_map(color_map_name):
        try:
            colors = pg.colormap.get(str(color_map_name or "magma")).getLookupTable(nPts=256)
        except Exception:
            colors = pg.colormap.get("magma").getLookupTable(nPts=256)
        positions = np.linspace(0.0, 1.0, 256)
        return pg.ColorMap(positions, colors)

    @staticmethod
    def _display_y_range(analysis_result, fallback_min, fallback_max):
        params = analysis_result.get("params", {}) if isinstance(analysis_result, dict) else {}
        display_range = params.get("mel_display_range", params.get("mel_scale_range", None))
        try:
            display_min, display_max = list(display_range)[:2]
            display_min = float(display_min)
            display_max = float(display_max)
        except Exception:
            display_min, display_max = float(fallback_min), float(fallback_max)
        if not np.isfinite(display_min) or not np.isfinite(display_max) or display_max <= display_min:
            return float(fallback_min), float(fallback_max)
        return display_min, display_max

    @staticmethod
    def _clipped_band_range(raw_range, y_min, y_max):
        try:
            first, second = list(raw_range)[:2]
            raw_low = min(float(first), float(second))
            raw_high = max(float(first), float(second))
            low = max(raw_low, float(y_min))
            high = min(raw_high, float(y_max))
        except Exception:
            return None
        if high <= low:
            return None
        return low, high

    def _plot_band_region(self, low, high, x_min, label, brush, pen, text_color, z_value):
        region = None
        try:
            region = pg.LinearRegionItem(
                values=(low, high),
                orientation="horizontal",
                brush=brush,
                movable=False,
            )
            region.setZValue(z_value)
            self.plot_widget.addItem(region)
        except Exception:
            pass

        for value in (low, high):
            line = pg.InfiniteLine(pos=value, angle=0, pen=pen, movable=False)
            line.setZValue(z_value + 8)
            self.plot_widget.addItem(line)

        text_item = None
        if label:
            try:
                text_item = pg.TextItem(
                    text=label,
                    color=text_color,
                    anchor=(0.0, 1.0),
                    fill=pg.mkBrush(255, 255, 255, 185),
                )
                text_item.setPos(float(x_min), high)
                text_item.setZValue(z_value + 12)
                self.plot_widget.addItem(text_item)
            except Exception:
                pass

        return region, text_item

    def _plot_analysis_mel_band(self, y_min, y_max, analysis_result, x_min):
        params = analysis_result.get("params", {}) if isinstance(analysis_result, dict) else {}
        mel_range = params.get("analysis_mel_range", params.get("filter_mel_range", None))
        clipped = self._clipped_band_range(mel_range, y_min, y_max)
        if clipped is None:
            return
        low, high = clipped
        self.analysis_region, self.analysis_label = self._plot_band_region(
            low,
            high,
            x_min,
            None,
            pg.mkBrush(255, 193, 7, 24),
            pg.mkPen(color=(245, 158, 11, 230), width=1.5, style=Qt.DashLine),
            (120, 75, 0),
            5,
        )

    def _plot_core_band(self, y_min, y_max, analysis_result=None, x_min=None):
        params = analysis_result.get("params", {}) if isinstance(analysis_result, dict) else {}
        cfg = self.analysis_config if isinstance(self.analysis_config, dict) else {}
        core_range_hz = params.get("core_range_hz", cfg.get("core_range_hz", [2000.0, 5000.0]))
        core_mel_range = params.get("core_mel_range")
        try:
            core_low_hz, core_high_hz = list(core_range_hz)[:2]
            if core_mel_range is None:
                core_low_mel = float(hz_to_mel(float(core_low_hz)))
                core_high_mel = float(hz_to_mel(float(core_high_hz)))
            else:
                core_low_mel, core_high_mel = list(core_mel_range)[:2]
        except Exception:
            return
        clipped = self._clipped_band_range([core_low_mel, core_high_mel], y_min, y_max)
        if clipped is None:
            return
        low, high = clipped
        if x_min is None:
            try:
                x_min = self.plot_widget.viewRange()[0][0]
            except Exception:
                x_min = 0.0
        self.core_region, self.core_label = self._plot_band_region(
            low,
            high,
            x_min,
            None,
            pg.mkBrush(0, 188, 212, 46),
            pg.mkPen(color=(0, 131, 143, 235), width=1.6, style=Qt.DotLine),
            (0, 90, 105),
            16,
        )

    def _plot_hotspot(self, analysis_result):
        hotspots = analysis_result.get("main_tone_hotspots") or []
        if not hotspots:
            fallback_hotspot = analysis_result.get("hotspot")
            hotspots = [fallback_hotspot] if isinstance(fallback_hotspot, dict) else []
        if not hotspots:
            return
        y_range = self.plot_widget.viewRange()[1]
        x_min = self.plot_widget.viewRange()[0][0]
        for index, hotspot in enumerate(hotspots):
            if not isinstance(hotspot, dict):
                continue
            clipped = self._clipped_band_range(
                [hotspot.get("mel_low", hotspot.get("mel")), hotspot.get("mel_high", hotspot.get("mel"))],
                y_range[0],
                y_range[1],
            )
            if clipped is None:
                continue
            low, high = clipped
            is_main_tone_hotspot = bool(hotspot.get("main_tone_frequency_hz") is not None) or (
                str(hotspot.get("kind", "")) == "main_tone_mel_band"
            )
            if is_main_tone_hotspot:
                hotspot_brush = pg.mkBrush(34, 197, 94, 88)
                hotspot_pen = pg.mkPen(color=(22, 101, 52, 235), width=2.4, style=Qt.SolidLine)
                hotspot_text_color = (22, 101, 52)
            else:
                hotspot_brush = pg.mkBrush(236, 72, 153, 88)
                hotspot_pen = pg.mkPen(color=(157, 23, 77, 235), width=2.4, style=Qt.SolidLine)
                hotspot_text_color = (157, 23, 77)
            region, label = self._plot_band_region(
                low,
                high,
                x_min,
                None,
                hotspot_brush,
                hotspot_pen,
                hotspot_text_color,
                32 + index,
            )
            self.hotspot_regions.append(region)
            if index == 0:
                self.hotspot_region = region
                self.hotspot_label = label

    def _plot_overall_spl_label(self, analysis_result, x_min, x_max, y_min, y_max):
        try:
            overall_spl = float(analysis_result.get("overall_spl_dba", 0.0))
        except (TypeError, ValueError):
            return
        if not np.isfinite(overall_spl):
            return

        x_margin = max((float(x_max) - float(x_min)) * 0.012, 1e-6)
        y_margin = max((float(y_max) - float(y_min)) * 0.025, 1e-6)
        label = pg.TextItem(
            text=f"Overall SPL: {overall_spl:.1f} dB(A)",
            color=(255, 255, 255),
            anchor=(0.0, 0.0),
            fill=pg.mkBrush(0, 0, 0, 150),
        )
        label.setPos(float(x_min) + x_margin, float(y_max) - y_margin)
        label.setZValue(80)
        self.plot_widget.addItem(label)
        self.overall_spl_label = label

    @staticmethod
    def _format_khz_range(low_hz, high_hz):
        try:
            low = float(low_hz) / 1000.0
            high = float(high_hz) / 1000.0
        except (TypeError, ValueError):
            return "-"
        if not np.isfinite(low) or not np.isfinite(high):
            return "-"
        if abs(high - low) < 1e-9:
            return f"{low:.3f}"
        return f"{low:.3f}-{high:.3f}"

    @staticmethod
    def _format_mel_range(low_mel, high_mel):
        try:
            low = float(low_mel)
            high = float(high_mel)
        except (TypeError, ValueError):
            return "-"
        if not np.isfinite(low) or not np.isfinite(high):
            return "-"
        if abs(high - low) < 1e-9:
            return f"{low:.1f}"
        return f"{low:.1f}-{high:.1f}"

    def _table_row_for_hotspot(self, hotspot):
        tone_freq = hotspot.get("main_tone_frequency_hz")
        try:
            tone_text = f"{float(tone_freq):.1f}" if tone_freq is not None else "-"
        except (TypeError, ValueError):
            tone_text = "-"
        hotspot_band = self._format_khz_range(hotspot.get("freq_low_hz"), hotspot.get("freq_high_hz"))
        mel_band = self._format_mel_range(
            hotspot.get("mel_low", hotspot.get("mel")),
            hotspot.get("mel_high", hotspot.get("mel")),
        )
        return [tone_text, hotspot_band, mel_band]

    def _update_table(self, analysis_result):
        hotspots = analysis_result.get("main_tone_hotspots") or []
        if hotspots:
            self.table_widget.setRowCount(len(hotspots))
            for row, hotspot in enumerate(hotspots):
                values = self._table_row_for_hotspot(hotspot)
                for col, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.table_widget.setItem(row, col, item)
            status_text = "Mel"
        else:
            hotspot = analysis_result.get("global_hotspot") or analysis_result.get("hotspot")
            self.table_widget.setRowCount(1)
            if isinstance(hotspot, dict):
                values = self._table_row_for_hotspot(hotspot)
                status_text = "Mel"
            else:
                values = ["-", "-", "-"]
                status_text = "Mel"
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table_widget.setItem(0, col, item)
        self.status_label.setText(status_text)

    def _update_fonts(self):
        font = QFont()
        font.setPixelSize(20)
        for axis_name in ("bottom", "left"):
            axis = self.plot_widget.getAxis(axis_name)
            axis.setTickFont(font)
            axis.setTextPen("black")
            axis.setLabel(axis.labelText, **{"font-size": "20px"})
        self._reserve_left_axis_space()

    @staticmethod
    def _to_plain_value(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, float)):
            return float(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, dict):
            return {k: Mel._to_plain_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Mel._to_plain_value(v) for v in value]
        return value

    @staticmethod
    def _to_exportable_result(analysis_result):
        keys = (
            "mel_db_a",
            "times_s",
            "mel_axis",
            "mel_axis_edges",
            "mel_true_axis",
            "mel_center_freqs_hz",
            "mel_freq_edges_hz",
            "overall_spl_dba",
            "hotspot",
            "global_hotspot",
            "main_tone_hotspots",
            "main_tone_hotspot_count",
            "params",
        )
        return {key: Mel._to_plain_value(analysis_result.get(key)) for key in keys}

    def export_pdf_images(self, output_dir):
        if self.plot_widget is None:
            return []
        image_path = export_plot_widget_image(self.plot_widget, output_dir, "mel_spectrogram")
        return [{"title": self.title_name, "path": image_path}]

    def export_pdf_tables(self):
        return _export_table_widget_for_pdf(self.table_widget)


class Modulation(QWidget):
    MODULATION_COLOR_CAP_PERCENT = 20.0
    HOTSPOT_MARKER_SIZE = 20
    MAIN_TONE_LABEL_FONT_PX = 9
    pdf_summary_exclude_fields = ("main_tone_results",)

    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.analysis_config = None
        self.result = None
        self.title_name = title_name
        self.img_item = None
        self.colorbar = None
        self.threshold_contour = None
        self.plot_widget = None
        self.table_widget = None
        self.status_label = None
        self.init_ui()
        self.setWindowTitle(title_name)

    def init_ui(self):
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        self.main_layout = QVBoxLayout(self)

        self.status_label = Label("Modulation: -")
        self.main_layout.addWidget(self.status_label)

        self.plot_widget = pg.PlotWidget(background="white")
        self.plot_widget.setLabel("bottom", "Modulation frequency (Hz)")
        self.plot_widget.setLabel("left", "Signal frequency (kHz)")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.35)
        self.main_layout.addWidget(self.plot_widget, 3)

        self.table_widget = TableWidget()
        self.table_widget.setColumnCount(5)
        self.table_widget.setHorizontalHeaderLabels(
            ["主音(Hz)", "调制频率(Hz)", "深度(%)", "机械匹配", "原因"]
        )
        header = self.table_widget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.table_widget.verticalHeader().setVisible(False)
        self.table_widget.setEditTriggers(TableWidget.NoEditTriggers)
        self.main_layout.addWidget(self.table_widget, 1)
        self.setLayout(self.main_layout)

    def calculate_modulation(self):
        try:
            recorded_signal = resolve_analysis_channel_signal(self.data_struct, self.analysis_config, self.title_name)
        except Exception as e:
            MessageBox.warning(self, "提示", str(e))
            return None
        sample_rate = self.data_struct.sample_rate
        if recorded_signal is None or sample_rate is None:
            MessageBox.warning(self, "提示", "Modulation 分析缺少录音数据或采样率。")
            return None

        try:
            analysis_result = compute_modulation_map(
                np.asarray(recorded_signal, dtype=np.float64),
                int(sample_rate),
                self.analysis_config or {},
            )
        except Exception as e:
            MessageBox.warning(self, "提示", f"Modulation 分析失败: {str(e)[:200]}")
            return None

        self._render_result(analysis_result)
        self.result = self._to_exportable_result(analysis_result)
        return self.result

    def _render_result(self, analysis_result):
        mod_depth = np.asarray(analysis_result.get("mod_depth_matrix", []), dtype=np.float64)
        signal_freqs = np.asarray(analysis_result.get("signal_freq_axis_hz", []), dtype=np.float64)
        mod_freqs = np.asarray(analysis_result.get("mod_freq_axis_hz", []), dtype=np.float64)
        if mod_depth.size == 0 or signal_freqs.size == 0 or mod_freqs.size == 0:
            return

        self._clear_plot()
        self.img_item = pg.ImageItem()
        self.plot_widget.addItem(self.img_item)

        display_matrix = mod_depth.T
        self.img_item.setImage(display_matrix, autoLevels=False)

        x_min, x_max = float(mod_freqs[0]), float(mod_freqs[-1])
        y_axis_khz = signal_freqs / 1000.0
        y_min, y_max = float(y_axis_khz[0]), float(y_axis_khz[-1])
        self.img_item.setRect(QRectF(x_min, y_min, max(x_max - x_min, 1e-9), max(y_max - y_min, 1e-9)))

        threshold = float(analysis_result.get("threshold_percent", 10.0))
        vmax = self.MODULATION_COLOR_CAP_PERCENT
        color_map = self._build_modulation_color_map()
        self.img_item.setLookupTable(color_map.getLookupTable(nPts=256))
        self.img_item.setLevels([0.0, vmax])
        self.colorbar = pg.ColorBarItem(values=(0.0, vmax), width=25, colorMap=color_map)
        self.colorbar.setImageItem(self.img_item, insert_in=self.plot_widget.getPlotItem())

        self.plot_widget.setXRange(x_min, x_max, padding=0)
        self.plot_widget.setYRange(y_min, y_max, padding=0)
        self.plot_widget.setTitle("Modulation map", size="20px", color="black")

        self._plot_core_lines(analysis_result, y_min, y_max)
        self._plot_threshold_contour(display_matrix, threshold)
        self._plot_main_tone_results(analysis_result, signal_freqs, mod_freqs)
        self._plot_hotspots(analysis_result)
        self._update_table(analysis_result)
        self._update_fonts()

    def _clear_plot(self):
        self.plot_widget.clear()
        if self.colorbar is not None:
            try:
                self.colorbar.close()
            except Exception:
                pass
        self.colorbar = None
        self.img_item = None
        self.threshold_contour = None

    @staticmethod
    def _build_modulation_color_map():
        positions = np.array([0.0, 0.35, 0.5, 1.0], dtype=np.float64)
        colors = np.array(
            [
                [8, 24, 72, 255],
                [30, 64, 175, 255],
                [250, 204, 21, 255],
                [220, 38, 38, 255],
            ],
            dtype=np.ubyte,
        )
        return pg.ColorMap(positions, colors)

    def _plot_threshold_contour(self, display_matrix, threshold):
        if self.img_item is None or display_matrix.size == 0 or not np.isfinite(threshold):
            return
        finite_values = display_matrix[np.isfinite(display_matrix)]
        if finite_values.size == 0 or float(np.nanmax(finite_values)) < threshold:
            return

        low_fill = min(float(np.nanmin(finite_values)), threshold) - 1.0
        contour_data = np.asarray(display_matrix, dtype=np.float64)
        contour_data = np.nan_to_num(contour_data, nan=low_fill, posinf=float(np.nanmax(finite_values)), neginf=low_fill)
        self.threshold_contour = pg.IsocurveItem(
            data=contour_data,
            level=threshold,
            pen=pg.mkPen(color=(0, 0, 0, 220), width=1.2, style=Qt.DashLine),
        )
        self.threshold_contour.setParentItem(self.img_item)
        self.threshold_contour.setZValue(20)

    def _plot_core_lines(self, analysis_result, y_min, y_max):
        for line_khz in analysis_result.get("core_freq_lines_khz", []) or []:
            try:
                value = float(line_khz)
            except (TypeError, ValueError):
                continue
            if y_min <= value <= y_max:
                line = pg.InfiniteLine(
                    pos=value,
                    angle=0,
                    pen=pg.mkPen(color=(255, 255, 255, 210), width=1, style=Qt.DotLine),
                    movable=False,
                )
                self.plot_widget.addItem(line)

    def _plot_main_tone_results(self, analysis_result, signal_freqs, mod_freqs):
        tone_results = analysis_result.get("main_tone_results", []) or []
        if not tone_results:
            return

        label_font = QFont()
        label_font.setPixelSize(self.MAIN_TONE_LABEL_FONT_PX)
        for item in tone_results:
            if not bool(item.get("is_valid", False)) or not bool(item.get("has_modulation_peak", True)):
                continue
            if item.get("mod_freq_hz") is None:
                continue
            target_hz = float(item.get("target_signal_freq_hz", item.get("analysis_signal_freq_hz", 0.0)))
            mod_hz = float(item.get("mod_freq_hz", 0.0))
            depth = float(item.get("mod_depth_percent", 0.0))
            label = pg.TextItem(
                text=f"{target_hz:.0f} Hz\n{mod_hz:.1f} Hz / {depth:.1f}%",
                color=(20, 20, 20),
                anchor=(0.0, 1.0),
                fill=pg.mkBrush(255, 255, 255, 170),
            )
            label.setFont(label_font)
            label.setPos(mod_hz, float(item.get("signal_freq_khz", 0.0)))
            label.setZValue(30)
            self.plot_widget.addItem(label)

    def _plot_hotspots(self, analysis_result):
        hotspots = analysis_result.get("hotspots", []) or analysis_result.get("global_hotspots", []) or []
        if not hotspots:
            return
        spots = []
        for item in hotspots:
            spots.append(
                {
                    "pos": (float(item.get("mod_freq_hz", 0.0)), float(item.get("signal_freq_khz", 0.0))),
                    "brush": pg.mkBrush(0, 229, 255, 235),
                    "pen": pg.mkPen("black", width=1.0),
                    "size": self.HOTSPOT_MARKER_SIZE,
                    "symbol": "star",
                }
            )
        self.plot_widget.addItem(pg.ScatterPlotItem(spots=spots))

    def _update_table(self, analysis_result):
        rows = analysis_result.get("main_tone_results", []) or []
        self.table_widget.setRowCount(len(rows))
        for row, item in enumerate(rows):
            has_modulation_peak = bool(item.get("has_modulation_peak", True))
            mod_freq_hz = item.get("mod_freq_hz")
            mechanical_text = "-"
            if has_modulation_peak:
                mechanical_text = "Yes" if item.get("mechanical_match", False) else "No"
            values = [
                f"{float(item.get('target_signal_freq_hz', 0.0)):.1f}",
                "-" if mod_freq_hz is None else f"{float(mod_freq_hz):.1f}",
                f"{float(item.get('mod_depth_percent', 0.0)):.2f}",
                "是" if item.get("mechanical_match", False) else "否",
                str(item.get("reason", "")),
            ]
            values[3] = mechanical_text
            for col, value in enumerate(values):
                table_item = QTableWidgetItem(value)
                table_item.setFlags(table_item.flags() & ~Qt.ItemIsEditable)
                self.table_widget.setItem(row, col, table_item)

        hotspot_count = len(analysis_result.get("global_hotspots", []) or [])
        self.status_label.setText(f"Modulation: 主音 {len(rows)} 个, 全局热点 {hotspot_count} 个")

    def _update_fonts(self):
        font = QFont()
        font.setPixelSize(20)
        for axis_name in ("bottom", "left"):
            axis = self.plot_widget.getAxis(axis_name)
            axis.setTickFont(font)
            axis.setTextPen("black")
            axis.setLabel(axis.labelText, **{"font-size": "20px"})

    @staticmethod
    def _to_plain_value(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, float)):
            return float(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, dict):
            return {k: Modulation._to_plain_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Modulation._to_plain_value(v) for v in value]
        return value

    @staticmethod
    def _to_exportable_result(analysis_result):
        keys = (
            "mod_depth_matrix",
            "signal_freq_axis_hz",
            "mod_freq_axis_hz",
            "main_tone_results",
            "global_hotspots",
            "mechanical_references",
            "mechanical_mod_freqs_hz",
            "stft_params",
            "threshold_percent",
            "analysis_scope",
            "main_tones_hz",
            "tone_band_hz",
            "computed_signal_freq_count",
        )
        return {key: Modulation._to_plain_value(analysis_result.get(key)) for key in keys}

    def export_pdf_images(self, output_dir):
        if self.plot_widget is None:
            return []
        image_path = export_plot_widget_image(self.plot_widget, output_dir, "modulation_map")
        return [{"title": self.title_name, "path": image_path}]

    def export_pdf_tables(self):
        return _export_table_widget_for_pdf(self.table_widget)


class LooseParticle(AnalysisGraphWidget):
    def __init__(self, title_name):
        super().__init__()
        self.setObjectName("LooseParticle")
        self.data_struct = DataDealStruct()
        self.result = None
        self.analysis_config = None
        self.title_name = title_name
        self.lp_num_label = Label("LP 数量: %s" % self.result)
        self.status_label = Label()
        self.v2pa_factor = None
        self.threshould = None
        self.setWindowTitle(title_name)
        self.add_label_to_layout()

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

    @staticmethod
    def downsample_min_max(x_data, y_data, window_size=8):
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
        if x_data.size <= 2 or y_data.size <= 2 or x_data.size != y_data.size or window_size <= 1:
            return x_data, y_data

        downsampled_x = []
        downsampled_y = []
        for start in range(0, y_data.size, window_size):
            end = min(start + window_size, y_data.size)
            window_x = x_data[start:end]
            window_y = y_data[start:end]
            if window_y.size == 0:
                continue

            max_idx = int(np.argmax(window_y))
            min_idx = int(np.argmin(window_y))
            selected_indices = sorted({min_idx, max_idx})
            for idx in selected_indices:
                downsampled_x.append(window_x[idx])
                downsampled_y.append(window_y[idx])

        return np.asarray(downsampled_x), np.asarray(downsampled_y)

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
        plot_x, plot_y = self.downsample_min_max(signal_duration, amplitude)
        self.analysis_plot.plot(plot_x, plot_y, pen=mkPen(color=(51, 196, 77), width=2))
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
        plot_x, plot_y = self.downsample_min_max(signal_duration, out_range_points)
        out_range_plot = pg.PlotDataItem(plot_x, plot_y, pen=pen)
        self.analysis_plot.addItem(out_range_plot)


class PeakDetection(AnalysisGraphWidget):
    def __init__(self, title_name):
        super().__init__()
        self.setObjectName("PeakDetection")
        self.data_struct = DataDealStruct()
        self.analysis_config = None
        self.result = None
        self.v2pa_factor = None
        self.setWindowTitle(title_name)

        # top status bar
        self.status_label = Label()
        self.PD_num_label = Label("PD 数量: -")
        pd_num_layout = QHBoxLayout()
        pd_num_layout.addStretch()
        pd_num_layout.addWidget(self.status_label)
        pd_num_layout.addWidget(self.PD_num_label)
        pd_num_layout.setSpacing(20)
        self.layout().insertLayout(0, pd_num_layout)

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
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        self.main_layout = QVBoxLayout(self)
        self.result_display = TextEdit()
        self.result_display.setReadOnly(True)

        self.main_layout.addWidget(self.result_display)
        self.setLayout(self.main_layout)

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
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        self.main_layout = QVBoxLayout(self)
        # plot area for summary
        self.plot_widget = pg.PlotWidget(background="white")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.5)
        self.plot_widget.setLabel("left", "SPL (dB)")
        self.plot_widget.setLabel("bottom", "Time (s)")

        self.result_display = TextEdit()
        self.result_display.set_font_size(20)
        self.result_display.setObjectName("resultDisplay")
        self.result_display.setReadOnly(True)
        # match result table
        self.table_widget = TableWidget()
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
        recorded_signal = self.data_struct.store_wave_data
        sample_rate = self.data_struct.sample_rate
        config = self.analysis_config or {}

        if recorded_signal is None or sample_rate is None:
            MessageBox.warning(self, "提示", "响度分析失败：没有可用录音数据。")
            return False

        sq_config = self._build_sq_config(config)
        run_result = run_sound_quality(
            np.asarray(recorded_signal, dtype=np.float64),
            int(sample_rate),
            project_v2pa_factor=float(self.v2pa_factor or 1.0),
            sq_config=sq_config,
        )
        loud_result = run_result.loudness
        if loud_result is None or not loud_result.enabled or loud_result.raw_result is None:
            reason = getattr(loud_result, "skipped_reason", None) or run_result.skipped_reason or "unknown"
            MessageBox.warning(self, "提示", f"响度分析跳过：{reason}")
            return False

        summary = loud_result.summary or {}
        self._plot_loudness_curve(loud_result, config)
        self._apply_loudness_limits(loud_result, config)

        raw = loud_result.raw_result
        self.result = {
            "summary": summary,
            "time_s": np.asarray(raw.time_s, dtype=np.float64).tolist(),
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
            "n5_sone": summary.get("n5_sone"),
            "lnmax_phon": summary.get("lnmax_phon"),
            "ln5_phon": summary.get("ln5_phon"),
            "mean_sone": summary.get("mean_sone"),
            "mean_phon": summary.get("mean_phon"),
        }
        return self.result

    @staticmethod
    def _build_sq_config(config: dict) -> dict:
        display_cfg = config.get("display", {}) or {}
        save_cfg = config.get("save", {}) or {}
        advanced_cfg = config.get("advanced", {}) or {}
        return {
            "enabled": True,
            "shared": {"field_type": config.get("field_type", "free")},
            "items": {
                "LOUD": {
                    "enabled": bool(config.get("enabled", True)),
                    "method": config.get("method", "per_segment"),
                    "display": display_cfg,
                    "save": save_cfg,
                    "advanced": advanced_cfg,
                },
                "SHRP": {"enabled": False},
                "ROUGH": {"enabled": False},
                "FLUC": {"enabled": False},
                "TON": {"enabled": False},
                "PR": {"enabled": False},
                "TNR": {"enabled": False},
            },
        }

    def _plot_loudness_curve(self, loud_result, config=None):
        raw = loud_result.raw_result
        time_s = np.asarray(raw.time_s, dtype=np.float64)
        advanced_cfg = (config or {}).get("advanced", {}) or {}
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
        metadata = dict(getattr(raw, "metadata", None) or {})
        if (
            method == "per_segment"
            and time_s.size
            and loudness.size
            and np.isfinite(time_s[0])
            and np.isfinite(loudness[0])
            and time_s[0] > 0.0
        ):
            if time_s.size > 1 and loudness.size > 1 and np.isfinite(loudness[1]):
                plot_time_s = np.insert(time_s[1:], 0, 0.0)
                plot_loudness = np.insert(loudness[1:], 0, loudness[1])
            else:
                plot_time_s = np.insert(time_s, 0, 0.0)
                plot_loudness = np.insert(loudness, 0, loudness[0])
            end_time_s = None
            sample_rate = getattr(self.data_struct, "sample_rate", None)
            recorded_signal = getattr(self.data_struct, "store_wave_data", None)
            try:
                if sample_rate and recorded_signal is not None:
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
            from base.core_algorithm.sound_quality.service import _interpolate_ref_line
        except ImportError:
            return None
        if ref_key not in SSTS_SPECIFIC_LOUDNESS_REF_LINES:
            return None
        ref_curve = _interpolate_ref_line(bark_axis, SSTS_SPECIFIC_LOUDNESS_REF_LINES[ref_key])
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
        if metric == "steady_state_average":
            value = summary.get("steady_state_average_sone",
                    summary.get("steady_state_average_phon",
                    summary.get("mean_sone", summary.get("mean_phon"))))
        elif metric == "max_transient":
            value = summary.get("max_transient_sone",
                    summary.get("max_transient_phon",
                    summary.get("nmax_sone")))
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
        curve_y_unit = str(advanced_cfg.get("curve_y_unit", config.get("curve_limit_unit", "sone")) or "sone").lower()
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

    def _apply_curve_limits(self, values: np.ndarray, config: dict):
        if not bool(config.get("limit_checked", False)):
            return

        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            self._merge_analysis_limit_result(False, float("nan"))
            return

        upper_enabled = bool(config.get("curve_upper_enabled", False))
        upper_limit = float(config.get("curve_upper_value", 0.0) or 0.0)
        lower_enabled = bool(config.get("curve_lower_enabled", False))
        lower_limit = float(config.get("curve_lower_value", 0.0) or 0.0)

        deviations = []
        if upper_enabled:
            upper_deviation = float(np.max(finite - upper_limit))
            if upper_deviation > 0.0:
                deviations.append(upper_deviation)
        if lower_enabled:
            lower_deviation = float(np.max(lower_limit - finite))
            if lower_deviation > 0.0:
                deviations.append(lower_deviation)

        deviation = float(max(deviations)) if deviations else 0.0
        self._merge_analysis_limit_result(deviation == 0.0, deviation)

    @staticmethod
    def _draw_limit_lines(plot_widget, config: dict, x_range=None):
        """Draw horizontal limit lines on a plot if configured."""
        if not bool(config.get("limit_checked", False)):
            return
        upper_enabled = bool(config.get("curve_upper_enabled", False))
        lower_enabled = bool(config.get("curve_lower_enabled", False))
        dashed_pen = mkPen(color=(128, 0, 128), width=2, style=Qt.DashLine)
        if upper_enabled:
            val = float(config.get("curve_upper_value", 0.0) or 0.0)
            plot_widget.addLine(y=val, pen=dashed_pen)
        if lower_enabled:
            val = float(config.get("curve_lower_value", 0.0) or 0.0)
            plot_widget.addLine(y=val, pen=dashed_pen)

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


class RoughnessAnalysis(LoudnessAnalysis):
    """Standalone ROUGH analysis backed by Daniel-Weber / Aures roughness."""

    def calculate_roughness(self):
        recorded_signal = self.data_struct.store_wave_data
        sample_rate = self.data_struct.sample_rate
        config = self.analysis_config or {}

        if recorded_signal is None or sample_rate is None:
            MessageBox.warning(self, "提示", "粗糙度分析失败：没有可用录音数据。")
            return False

        sq_config = self._build_sq_config(config)
        run_result = run_sound_quality(
            np.asarray(recorded_signal, dtype=np.float64),
            int(sample_rate),
            project_v2pa_factor=float(self.v2pa_factor or 1.0),
            sq_config=sq_config,
        )

        roughness_result = run_result.roughness
        if roughness_result is None or not roughness_result.enabled or roughness_result.raw_result is None:
            reason = getattr(roughness_result, "skipped_reason", None) or run_result.skipped_reason or "unknown"
            MessageBox.warning(self, "提示", f"粗糙度分析跳过：{reason}")
            return False

        self._plot_roughness_curve(roughness_result, config)
        self._apply_curve_limits(roughness_result.raw_result.roughness_asper, config)

        raw = roughness_result.raw_result
        summary = roughness_result.summary or {}
        self.result = {
            "summary": summary,
            "time_s": np.asarray(raw.time_s, dtype=np.float64).tolist(),
            "roughness_asper": np.asarray(raw.roughness_asper, dtype=np.float64).tolist(),
            "bark_axis": np.asarray(raw.bark_axis, dtype=np.float64).tolist(),
            "metadata": dict(raw.metadata or {}),
        }
        self.export_detail = {
            "rmean_asper": summary.get("r_mean_asper"),
        }
        return self.result

    @staticmethod
    def _build_sq_config(config: dict) -> dict:
        display_cfg = config.get("display", {}) or {}
        save_cfg = config.get("save", {}) or {}
        advanced_cfg = config.get("advanced", {}) or {}
        return {
            "enabled": True,
            "shared": {"field_type": "free"},
            "items": {
                "LOUD": {"enabled": False},
                "SHRP": {"enabled": False},
                "ROUGH": {
                    "enabled": bool(config.get("enabled", True)),
                    "display": display_cfg,
                    "save": save_cfg,
                    "advanced": advanced_cfg,
                },
                "FLUC": {"enabled": False},
                "TON": {"enabled": False},
                "PR": {"enabled": False},
                "TNR": {"enabled": False},
            },
        }

    def _plot_roughness_curve(self, roughness_result, config=None):
        payload = roughness_result.display_payload or {}
        curves = payload.get("curves", []) or []
        curve = next((item for item in curves if item.get("key") == "roughness_time"), None)

        self.analysis_plot.clear()
        if curve:
            time_s = np.asarray(curve.get("x"), dtype=np.float64)
            roughness = np.asarray(curve.get("y"), dtype=np.float64)
            if time_s.size and roughness.size:
                self.analysis_plot.plot(
                    time_s,
                    roughness,
                    pen=mkPen(color=(105, 85, 190), width=2),
                    name="Roughness",
                    connect="finite",
                )
                self._apply_roughness_y_axis(self.analysis_plot, roughness)

        self.analysis_plot.setLabel("left", "Roughness R (asper)")
        self.analysis_plot.setLabel("bottom", "Time (s)")
        self.analysis_plot.showGrid(x=True, y=True)

        if config:
            self._draw_limit_lines(self.analysis_plot, config)

        title_parts = []
        for card in payload.get("summary_cards", []) or []:
            value = card.get("value")
            try:
                finite_value = value is not None and np.isfinite(float(value))
            except (TypeError, ValueError):
                finite_value = False
            if finite_value:
                title_parts.append(f"{card.get('label', card.get('key'))}: {float(value):.3g} {card.get('unit', '')}")
        if title_parts:
            self.analysis_plot.setTitle(" | ".join(title_parts), size="14px", color="k")


class SharpnessAnalysis(LoudnessAnalysis):
    """Standalone SHRP analysis backed by DIN 45692 sharpness."""

    def calculate_sharpness(self):
        recorded_signal = self.data_struct.store_wave_data
        sample_rate = self.data_struct.sample_rate
        config = self.analysis_config or {}

        if recorded_signal is None or sample_rate is None:
            MessageBox.warning(self, "提示", "尖锐度分析失败：没有可用录音数据。")
            return False

        sq_config = self._build_sq_config(config)
        run_result = run_sound_quality(
            np.asarray(recorded_signal, dtype=np.float64),
            int(sample_rate),
            project_v2pa_factor=float(self.v2pa_factor or 1.0),
            sq_config=sq_config,
        )

        sharpness_result = run_result.sharpness
        if sharpness_result is None or not sharpness_result.enabled or sharpness_result.raw_result is None:
            reason = getattr(sharpness_result, "skipped_reason", None) or run_result.skipped_reason or "unknown"
            MessageBox.warning(self, "提示", f"尖锐度分析跳过：{reason}")
            return False

        self._plot_sharpness_curve(sharpness_result, config)
        self._apply_curve_limits(sharpness_result.raw_result.sharpness_acum, config)

        raw = sharpness_result.raw_result
        summary = sharpness_result.summary or {}
        self.result = {
            "summary": summary,
            "time_s": np.asarray(raw.time_s, dtype=np.float64).tolist(),
            "sharpness_acum": np.asarray(raw.sharpness_acum, dtype=np.float64).tolist(),
            "metadata": dict(raw.metadata or {}),
        }
        self.export_detail = {
            "smean_acum": summary.get("s_mean_acum"),
            "sstationary_acum": summary.get("s_stationary_acum"),
        }
        return self.result

    @staticmethod
    def _build_sq_config(config: dict) -> dict:
        display_cfg = config.get("display", {}) or {}
        save_cfg = config.get("save", {}) or {}
        advanced_cfg = config.get("advanced", {}) or {}

        loud_cfg = _merge_loudness_config_with_defaults(
            config.get("upstream_loudness", {}) or config.get("loudness", {}) or {}
        )
        loud_display = dict(loud_cfg.get("display", {}) or {})
        loud_display.setdefault("summary_metrics", [])
        loud_display.setdefault("curves", [])
        loud_display.setdefault("heatmaps", [])

        loud_save = dict(loud_cfg.get("save", {}) or {})
        loud_save.setdefault("summary", False)
        loud_save.setdefault("curve", False)
        loud_save.setdefault("specific_loudness", False)

        loud_advanced = dict(loud_cfg.get("advanced", {}) or {})

        return {
            "enabled": True,
            "shared": {"field_type": "free"},
            "items": {
                "LOUD": {
                    "enabled": True,
                    "method": loud_cfg.get("method", config.get("loudness_method", "per_segment")),
                    "display": loud_display,
                    "save": loud_save,
                    "advanced": loud_advanced,
                },
                "SHRP": {
                    "enabled": bool(config.get("enabled", True)),
                    "display": display_cfg,
                    "save": save_cfg,
                    "advanced": advanced_cfg,
                },
                "ROUGH": {"enabled": False},
                "FLUC": {"enabled": False},
                "TON": {"enabled": False},
                "PR": {"enabled": False},
                "TNR": {"enabled": False},
            },
        }

    def _plot_sharpness_curve(self, sharpness_result, config=None):
        payload = sharpness_result.display_payload or {}
        curves = payload.get("curves", []) or []
        curve = next((item for item in curves if item.get("key") == "sharpness_time"), None)

        self.analysis_plot.clear()
        if curve:
            time_s = np.asarray(curve.get("x"), dtype=np.float64)
            sharpness = np.asarray(curve.get("y"), dtype=np.float64)
            if time_s.size and sharpness.size:
                self.analysis_plot.plot(
                    time_s,
                    sharpness,
                    pen=mkPen(color=(225, 122, 39), width=2),
                    name="Sharpness",
                    connect="finite",
                )
                self._apply_sharpness_y_axis(self.analysis_plot, sharpness)

        self.analysis_plot.setLabel("left", "Sharpness S (acum)")
        self.analysis_plot.setLabel("bottom", "Time (s)")
        self.analysis_plot.showGrid(x=True, y=True)

        if config:
            self._draw_limit_lines(self.analysis_plot, config)

        title_parts = []
        for card in payload.get("summary_cards", []) or []:
            value = card.get("value")
            try:
                finite_value = value is not None and np.isfinite(float(value))
            except (TypeError, ValueError):
                finite_value = False
            if finite_value:
                title_parts.append(f"{card.get('label', card.get('key'))}: {float(value):.3g} {card.get('unit', '')}")
        if title_parts:
            self.analysis_plot.setTitle(" | ".join(title_parts), size="14px", color="k")


class FrequencyBandAnalysis(AnalysisGraphWidget):
    """
    频段能量分析 (Frequency Band Analysis) 窗口。

    将音频频谱按指定策略拆分为有限个频段，计算各频段声压级，
    以柱状图形式展示，并可与阈值比较。
    """

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
        self._fba_hover_rows = []
        self._fba_hover_bar_width = 0.7
        self._fba_hover_connected = False
        self._fba_last_hover_index = None
        self.setWindowTitle(title_name)

    def calculate_fba(self):
        """执行频段能量分析并绘图"""
        config = self.analysis_config or {}
        try:
            recorded_signal = resolve_analysis_channel_signal(self.data_struct, self.analysis_config, self.title_name)
        except Exception as e:
            MessageBox.warning(self, "提示", str(e))
            return False
        sample_rate = self.data_struct.sample_rate
        if sample_rate is None:
            return False

        strategy_label = config.get("band_strategy", "1/3 倍频程")
        strategy_info = self.STRATEGY_LABELS.get(strategy_label)
        if not strategy_info:
            strategy_name, strategy_kwargs = "octave", {"fraction": 3}
        else:
            strategy_name, strategy_kwargs = strategy_info

        weighting = config.get("weighting", "A")
        if weighting in ("None", "Z（None）"):
            weighting = "Z"
        f_min = config.get("f_min", 20)
        f_max = config.get("f_max", 20000)
        n_bands = config.get("n_bands", 40)
        bandwidth = config.get("bandwidth", 100)
        custom_bands_text = config.get("custom_bands_text", "")
        custom_edges = None
        if strategy_label == "自定义":
            try:
                custom_edges = self._parse_custom_bands_text(custom_bands_text)
            except Exception as e:
                MessageBox.warning(self, "提示", f"自定义频段解析失败: {str(e)[:200]}")
                return False

        try:
            analyzer = FrequencyBandAnalyzer(
                strategy=strategy_name,
                weighting=weighting,
                f_min=f_min,
                f_max=f_max,
                fraction=strategy_kwargs.get("fraction", 3),
                n_bands=n_bands,
                bandwidth=bandwidth,
                custom_edges=custom_edges,
            )
            analysis_result = analyzer.analyze(
                recorded_signal,
                fs=int(sample_rate),
                v2pa_factor=self.v2pa_factor or 1.0,
            )
        except Exception as e:
            MessageBox.warning(self, "提示", f"频段能量分析失败: {str(e)[:200]}")
            return False

        baseline_result = None
        baseline_file_path = str(config.get("baseline_file_path", "") or "").strip()
        if baseline_file_path:
            try:
                baseline_signal, _baseline_sr = librosa.load(baseline_file_path, sr=int(sample_rate), mono=True)
                baseline_result = analyzer.analyze(
                    baseline_signal,
                    fs=int(sample_rate),
                    v2pa_factor=self.v2pa_factor or 1.0,
                )
            except Exception as e:
                MessageBox.warning(self, "提示", f"背景噪声基线加载失败: {str(e)[:200]}")
                baseline_result = None

        baseline_levels = (
            np.asarray(baseline_result.band_levels_weighted_db, dtype=np.float64)
            if baseline_result is not None
            else None
        )
        display_mode = str(config.get("baseline_display_mode", "overlay") or "overlay")
        display_curves = self._build_fba_display_levels(
            np.asarray(analysis_result.band_levels_weighted_db, dtype=np.float64),
            baseline_levels,
            display_mode,
        )
        plot_levels = np.asarray(display_curves["plot_levels"], dtype=np.float64)
        delta_levels = display_curves["delta_levels"]
        dominant_tones = self._detect_dominant_tones(
            analysis_result.bands,
            plot_levels,
            config,
            fallback_low_hz=float(f_min),
            fallback_high_hz=float(f_max),
        )

        limit_checked = config.get("limit_checked", False)
        limit_mode = str(config.get("limit_mode", "csv") or "csv").lower()
        upper_limits = None
        lower_limits = None

        if limit_checked:
            levels = np.asarray(plot_levels, dtype=np.float64)
            n = int(levels.size)
            centers = np.array([b.f_center for b in analysis_result.bands], dtype=np.float64)

            if limit_mode == "manual":
                upper_ok = bool(config.get("manual_upper_enabled", True))
                lower_ok = bool(config.get("manual_lower_enabled", False))
                upper = float(config.get("manual_upper", 0.0) or 0.0)
                lower = float(config.get("manual_lower", 0.0) or 0.0)
                upper_limits = np.full(n, upper, dtype=np.float64) if upper_ok else np.full(n, np.nan, dtype=np.float64)
                lower_limits = np.full(n, lower, dtype=np.float64) if lower_ok else np.full(n, np.nan, dtype=np.float64)
            else:
                limit_data = config.get("limit_data", None)
                if not limit_data:
                    MessageBox.warning(self, "提示", "已启用阈值，但未加载 CSV 配置文件。")
                    return False
                csv_x_list, csv_upper_list, csv_lower_list = limit_data
                csv_x = np.asarray(csv_x_list, dtype=np.float64)
                csv_u = np.asarray(csv_upper_list, dtype=np.float64)
                csv_l = np.asarray(csv_lower_list, dtype=np.float64)

                sort_idx = np.argsort(csv_x)
                csv_x = csv_x[sort_idx]
                csv_u = csv_u[sort_idx]
                csv_l = csv_l[sort_idx]

                upper_limits = (
                    np.interp(centers, csv_x, csv_u, left=csv_u[0], right=csv_u[-1])
                    if csv_x.size
                    else np.full(n, np.nan)
                )
                # lower 可能是全 NaN（仅上限），保持 NaN 即可
                if np.all(np.isnan(csv_l)) or (not np.any(np.isfinite(csv_l))):
                    lower_limits = np.full(n, np.nan, dtype=np.float64)
                else:
                    # 对 lower 的 NaN 先用最近的有限值填充后再插值，避免整段变 NaN
                    finite = np.isfinite(csv_l)
                    if np.any(finite):
                        filled = np.interp(csv_x, csv_x[finite], csv_l[finite])
                        lower_limits = np.interp(centers, csv_x, filled, left=filled[0], right=filled[-1])
                    else:
                        lower_limits = np.full(n, np.nan, dtype=np.float64)

            # 用公共逻辑计算 OK/NG 与偏差
            out_mask, deviation, is_ok = LimitPlotUtils.compare_with_limits(
                levels,
                np.asarray(upper_limits, dtype=np.float64),
                np.asarray(lower_limits, dtype=np.float64),
                valid_mask=np.isfinite(levels),
            )
            analysis_result.exceeded_bands = np.where(out_mask)[0].astype(int).tolist()
            self.data_struct.analysis_result_dict[self.title_name] = (bool(is_ok), float(deviation))

        self._plot_bar_chart(
            analysis_result,
            weighting,
            upper_limits=upper_limits,
            lower_limits=lower_limits,
            plot_levels=plot_levels,
            baseline_levels=baseline_levels if display_mode == "overlay" else None,
            display_mode=display_mode,
            dominant_tones=dominant_tones,
        )

        self.result = {
            "bands": [b.label for b in analysis_result.bands],
            "band_centers": [b.f_center for b in analysis_result.bands],
            "band_levels_db": analysis_result.band_levels_db.tolist(),
            "band_levels_weighted_db": analysis_result.band_levels_weighted_db.tolist(),
            "baseline_band_levels_weighted_db": baseline_levels.tolist() if isinstance(baseline_levels, np.ndarray) else [],
            "delta_band_levels_weighted_db": delta_levels.tolist() if isinstance(delta_levels, np.ndarray) else [],
            "plot_band_levels_weighted_db": plot_levels.tolist(),
            "overall_db": analysis_result.overall_db,
            "overall_weighted_db": analysis_result.overall_weighted_db,
            "weighting": analysis_result.weighting,
            "baseline_display_mode": display_mode,
            "dominant_tones": dominant_tones,
            "exceeded_bands": analysis_result.exceeded_bands,
        }
        return self.result

    @staticmethod
    def _parse_custom_bands_text(text: str):
        edges = []
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if "," in line:
                parts = [p.strip() for p in line.split(",") if p.strip()]
            else:
                parts = [p.strip() for p in line.replace("\t", " ").split(" ") if p.strip()]

            label = None
            if len(parts) == 1 and "-" in parts[0]:
                a, b = [p.strip() for p in parts[0].split("-", 1)]
                fl, fh = float(a), float(b)
            elif len(parts) >= 2:
                fl, fh = float(parts[0]), float(parts[1])
                if len(parts) >= 3:
                    label = " ".join(parts[2:]).strip() or None
            else:
                raise ValueError(f"无法解析行: {raw!r}")

            if not (fl > 0 and fh > 0):
                raise ValueError(f"频率必须为正数: {raw!r}")
            if not (fh > fl):
                raise ValueError(f"频段上限必须大于下限: {raw!r}")
            edges.append((fl, fh, label))

        edges.sort(key=lambda x: x[0])
        if not edges:
            raise ValueError("请至少输入一个频段。")
        for i in range(1, len(edges)):
            if edges[i][0] < edges[i - 1][1]:
                raise ValueError("自定义频段不允许重叠，请检查相邻频段边界。")
        return edges

    @staticmethod
    def _build_sparse_x_ticks(labels, max_ticks=10):
        n = len(labels)
        if n == 0:
            return []
        if n <= max_ticks:
            return [(i, lbl) for i, lbl in enumerate(labels)]

        max_ticks = max(2, int(max_ticks))
        step = int(np.ceil((n - 1) / float(max_ticks - 1)))
        indices = list(range(0, n, step))
        if indices[-1] != n - 1:
            indices.append(n - 1)
        while len(indices) > max_ticks:
            indices.pop(-2)
        return [(i, labels[i]) for i in indices if 0 <= i < n]

    @staticmethod
    def _build_fba_hover_rows(bands, levels, exceeded_indices):
        rows = []
        exceeded = {int(idx) for idx in exceeded_indices}
        for i, band in enumerate(bands):
            level = float(levels[i]) if i < len(levels) and np.isfinite(levels[i]) else None
            rows.append(
                {
                    "index": i,
                    "label": str(getattr(band, "label", "")),
                    "f_low": float(getattr(band, "f_low", 0.0)),
                    "f_high": float(getattr(band, "f_high", 0.0)),
                    "f_center": float(getattr(band, "f_center", 0.0)),
                    "level": level,
                    "level_text": f"{level:.2f} dB" if level is not None else "N/A",
                    "status": "NG" if i in exceeded else "OK",
                }
            )
        return rows

    @staticmethod
    def _format_fba_freq(freq):
        if freq >= 1000:
            return f"{freq / 1000:.4g} kHz"
        return f"{freq:.4g} Hz"

    @staticmethod
    def _build_fba_display_levels(levels, baseline_levels, display_mode):
        plot_levels = np.asarray(levels, dtype=np.float64)
        baseline = None if baseline_levels is None else np.asarray(baseline_levels, dtype=np.float64)
        delta = None
        if baseline is not None:
            if baseline.shape != plot_levels.shape:
                baseline = None
            else:
                delta = plot_levels - baseline
                if str(display_mode or "overlay") == "delta":
                    plot_levels = delta
        return {
            "plot_levels": plot_levels,
            "baseline_levels": baseline,
            "delta_levels": delta,
        }

    @staticmethod
    def _detect_dominant_tones(bands, levels, config, *, fallback_low_hz, fallback_high_hz):
        if not bool(config.get("dominant_tone_enabled", False)):
            return []
        intervals = parse_frequency_intervals(config.get("dominant_tone_intervals_text", ""))
        if not intervals and fallback_high_hz > fallback_low_hz:
            intervals = [FrequencyInterval(float(fallback_low_hz), float(fallback_high_hz), "Overall")]
        return find_dominant_fba_bands(bands, levels, intervals)

    def _connect_fba_hover_tooltip(self):
        if self._fba_hover_connected:
            return
        try:
            self.analysis_plot.scene().sigMouseMoved.connect(self._on_fba_mouse_moved)
            self._fba_hover_connected = True
        except Exception:
            self._fba_hover_connected = False

    def _on_fba_mouse_moved(self, scene_pos):
        if not self._fba_hover_rows:
            QToolTip.hideText()
            self._fba_last_hover_index = None
            return
        if not self.analysis_plot.sceneBoundingRect().contains(scene_pos):
            QToolTip.hideText()
            self._fba_last_hover_index = None
            return

        view_pos = self.analysis_plot.getViewBox().mapSceneToView(scene_pos)
        x_value = float(view_pos.x())
        index = int(round(x_value))
        if index < 0 or index >= len(self._fba_hover_rows):
            QToolTip.hideText()
            self._fba_last_hover_index = None
            return
        if abs(x_value - index) > (self._fba_hover_bar_width / 2.0):
            QToolTip.hideText()
            self._fba_last_hover_index = None
            return

        row = self._fba_hover_rows[index]
        if self._fba_last_hover_index == index and QToolTip.isVisible():
            return
        self._fba_last_hover_index = index

        text = (
            f"Frequency: {row['label']}\n"
            f"Band: {self._format_fba_freq(row['f_low'])} - {self._format_fba_freq(row['f_high'])}\n"
            f"Center: {self._format_fba_freq(row['f_center'])}\n"
            f"Level: {row['level_text']}\n"
            f"Status: {row['status']}"
        )
        views = self.analysis_plot.scene().views()
        global_pos = views[0].mapToGlobal(views[0].mapFromScene(scene_pos)) if views else self.mapToGlobal(self.rect().center())
        QToolTip.showText(global_pos, text, self.analysis_plot)

    def _plot_bar_chart(
        self,
        result: BandAnalysisResult,
        weighting: str,
        *,
        upper_limits: np.ndarray | None = None,
        lower_limits: np.ndarray | None = None,
        plot_levels: np.ndarray | None = None,
        baseline_levels: np.ndarray | None = None,
        display_mode: str = "overlay",
        dominant_tones: list | None = None,
    ):
        """使用 pyqtgraph 绘制频段能量柱状图"""
        self.analysis_plot.clear()
        self._fba_hover_rows = []
        self._fba_last_hover_index = None

        levels = (
            np.asarray(plot_levels, dtype=np.float64)
            if plot_levels is not None
            else np.asarray(result.band_levels_weighted_db, dtype=np.float64)
        )
        original_levels = levels.copy()
        labels = [b.label for b in result.bands]
        n = len(labels)
        if n == 0:
            return

        x = np.arange(n)
        bar_width = 0.7

        normal_color = (76, 175, 80)
        exceed_color = (244, 67, 54)
        missing_color = (189, 189, 189)
        tone_color = (255, 152, 0)

        finite_mask = np.isfinite(levels)
        if not np.all(finite_mask):
            # pyqtgraph 对 NaN/inf 的柱高兼容性不稳定，这里统一降级成 0 并用灰色标记缺失段
            levels = levels.copy()
            levels[~finite_mask] = 0.0

        brushes = [pg.mkBrush(normal_color) for _ in range(n)]
        for i in range(n):
            if not finite_mask[i]:
                brushes[i] = pg.mkBrush(missing_color)
        dominant_indices = self._dominant_tone_band_indices(result.bands, dominant_tones)
        for idx in dominant_indices:
            if 0 <= int(idx) < n and finite_mask[int(idx)]:
                brushes[int(idx)] = pg.mkBrush(tone_color)
        for idx in result.exceeded_bands:
            if 0 <= int(idx) < n and finite_mask[int(idx)]:
                brushes[int(idx)] = pg.mkBrush(exceed_color)

        bar_item = pg.BarGraphItem(
            x=x,
            height=levels,
            width=bar_width,
            brushes=brushes,
            pen=pg.mkPen("w", width=0.5),
        )
        self.analysis_plot.addItem(bar_item)

        if baseline_levels is not None:
            baseline = np.asarray(baseline_levels, dtype=np.float64)
            if baseline.size == n and np.any(np.isfinite(baseline)):
                self.analysis_plot.plot(
                    x,
                    baseline,
                    pen=mkPen(color=(128, 128, 128), width=2),
                    symbol="x",
                    symbolSize=5,
                    symbolBrush=(128, 128, 128),
                    name="Baseline",
                )

        if upper_limits is not None:
            u = np.asarray(upper_limits, dtype=np.float64)
            if u.size == n and np.any(np.isfinite(u)):
                self.analysis_plot.plot(
                    x,
                    u,
                    pen=mkPen(color=(0, 188, 212), width=2),
                    symbol="o",
                    symbolSize=4,
                    symbolBrush=(0, 188, 212),
                    name="Upper Limit",
                )
        if lower_limits is not None:
            l = np.asarray(lower_limits, dtype=np.float64)
            if l.size == n and np.any(np.isfinite(l)):
                self.analysis_plot.plot(
                    x,
                    l,
                    pen=mkPen(color=(63, 81, 181), width=2),
                    symbol="t",
                    symbolSize=6,
                    symbolBrush=(63, 81, 181),
                    name="Lower Limit",
                )

        # 超限标注（仅提示超上限的幅度；下限超限同样标红柱子）
        if result.exceeded_bands:
            for idx in result.exceeded_bands:
                if 0 <= int(idx) < n and np.isfinite(levels[int(idx)]):
                    text = pg.TextItem("NG", color=exceed_color, anchor=(0.5, 1.0))
                    text.setPos(int(idx), levels[int(idx)])
                    self.analysis_plot.addItem(text)

        for idx in dominant_indices:
            if 0 <= int(idx) < n and np.isfinite(levels[int(idx)]):
                text = pg.TextItem("Tone", color=tone_color, anchor=(0.5, 1.0))
                text.setPos(int(idx), levels[int(idx)])
                self.analysis_plot.addItem(text)

        x_axis = self.analysis_plot.getAxis("bottom")
        x_axis.setTicks([self._build_sparse_x_ticks(labels, max_ticks=10)])
        self._fba_hover_bar_width = bar_width
        self._fba_hover_rows = self._build_fba_hover_rows(result.bands, original_levels, result.exceeded_bands)
        self._connect_fba_hover_tooltip()

        weight_label = f"dB({weighting})" if weighting != "Z" else "dB"
        y_label = f"Sound Pressure Level [{weight_label}]"
        if str(display_mode or "overlay") == "delta":
            y_label = "FBA - Baseline [dB]"
        self.analysis_plot.setLabel("left", y_label)
        self.analysis_plot.setLabel("bottom", "Frequency (Hz)")

        overall = result.overall_weighted_db if weighting != "Z" else result.overall_db
        self.analysis_plot.setTitle(f"Overall: {overall:.1f} {weight_label} [SPL]", size="14px", color="k")

        self.analysis_plot.showGrid(x=False, y=True)

    @staticmethod
    def _dominant_tone_band_indices(bands, dominant_tones):
        indices = []
        for tone in dominant_tones or []:
            tone_label = str(tone.get("band_label", ""))
            tone_center = float(tone.get("frequency_hz", np.nan))
            for index, band in enumerate(bands):
                if tone_label and str(getattr(band, "label", "")) == tone_label:
                    indices.append(index)
                    break
                if np.isfinite(tone_center) and np.isclose(float(getattr(band, "f_center", np.nan)), tone_center):
                    indices.append(index)
                    break
        return sorted(set(indices))


def _decimate_peak_envelope(x: np.ndarray, y: np.ndarray, max_points: int):
    """峰值包络降采样：把密集曲线压到约 max_points 点，保留每段的最小/最大值。

    频谱有数万点时 pyqtgraph 宽线渲染很慢；普通抽稀会丢掉窄带峰，这里按桶取
    min/max 两点并按 x 顺序排列，视觉上峰谷与原曲线一致。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if max_points <= 0 or n <= max_points:
        return x, y
    buckets = max(1, max_points // 2)
    step = n // buckets
    if step < 2:
        return x, y
    usable = step * buckets
    xr = x[:usable].reshape(buckets, step)
    yr = y[:usable].reshape(buckets, step)
    cols = np.arange(buckets)
    imin = yr.argmin(axis=1)
    imax = yr.argmax(axis=1)
    first_is_min = imin <= imax
    out_x = np.empty(buckets * 2, dtype=float)
    out_y = np.empty(buckets * 2, dtype=float)
    out_x[0::2] = np.where(first_is_min, xr[cols, imin], xr[cols, imax])
    out_y[0::2] = np.where(first_is_min, yr[cols, imin], yr[cols, imax])
    out_x[1::2] = np.where(first_is_min, xr[cols, imax], xr[cols, imin])
    out_y[1::2] = np.where(first_is_min, yr[cols, imax], yr[cols, imin])
    if usable < n:
        out_x = np.concatenate([out_x, x[usable:]])
        out_y = np.concatenate([out_y, y[usable:]])
    return out_x, out_y


def _merge_curve_anchor_points(x: np.ndarray, y: np.ndarray, anchor_x: np.ndarray, anchor_y: np.ndarray):
    """把关键标注点并回降采样曲线，避免绘图抽稀后曲线跳过业务关注点。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    anchor_x = np.asarray(anchor_x, dtype=float)
    anchor_y = np.asarray(anchor_y, dtype=float)
    anchor_valid = np.isfinite(anchor_x) & np.isfinite(anchor_y)
    if not np.any(anchor_valid):
        return x, y
    out_x = np.concatenate([x, anchor_x[anchor_valid]])
    out_y = np.concatenate([y, anchor_y[anchor_valid]])
    order = np.argsort(out_x, kind="mergesort")
    return out_x[order], out_y[order]


class ProminenceRatioAnalysis(AnalysisGraphWidget):
    """突出比 (PR / Prominence Ratio) 分析窗口（双轨图）。

    需求书《笔记本电脑风扇噪音核心测试要求书》PR 频谱模块：
    - 上轨：线性功率谱曲线，红/蓝临界频带框内标注频带积分功率 dB。
    - 下轨：PR 值分布曲线 dB；横轴 Frequency (kHz)。
    - 默认计算口径：ECMA-74 Annex D / ECMA-418-1 标准临界带 + 线性(Z)/不计权功率。
      15% 固定比例仅按需求书作为图面划分线显示，不参与默认 PR 功率积分。
    - 图面按需求书展示名称标注，不显示 mode/df/FFT/weighting 等内部调试参数。
    """

    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.v2pa_factor = None
        self.analysis_config = None
        self.result = {}
        self.export_detail = {}
        self.title_name = title_name
        # 背景噪声基线谱（已插值对齐到主谱 fft_freq_hz；无基线时为 None）
        self._pr_baseline_db = None
        self.setWindowTitle(title_name)

        # 下轨 PR 曲线（上轨复用基类 self.analysis_plot 作为线性功率谱）
        self.pr_plot = pg.PlotWidget()
        self.pr_plot.setBackground("white")
        self.apply_plot_font_style(self.pr_plot, 20)
        self.layout().addWidget(self.pr_plot)
        # 双轨共用同一频率横轴：与下轨联动，并隐藏上轨重复的 X 刻度，仅下轨保留一条频率轴
        self.analysis_plot.setXLink(self.pr_plot)
        self.analysis_plot.getAxis("bottom").setStyle(showValues=False)

    def calculate_pr(self):
        """执行 PR 分析并绘制双轨图。"""
        config = self.analysis_config or {}
        try:
            recorded_signal = resolve_analysis_channel_signal(self.data_struct, self.analysis_config, self.title_name)
        except Exception as e:
            MessageBox.warning(self, "提示", str(e))
            return False
        sample_rate = self.data_struct.sample_rate
        if sample_rate is None:
            return False

        params, cfg_warnings = ProminenceRatioParams.from_config(config)
        fan_pr_limits = config.get("fan_pr_limits") or [[100, 2000, 4], [2000, 5000, 2], [5000, 20000, 4]]

        try:
            analyzer = ProminenceRatioAnalyzer(int(sample_rate))
            res = analyzer.compute(
                recorded_signal,
                self.v2pa_factor or 1.0,
                params,
                fan_pr_limits,
                initial_warnings=cfg_warnings,
            )
        except Exception as e:
            MessageBox.warning(self, "提示", f"PR 分析失败: {str(e)[:200]}")
            return False

        if res.decision_status == "invalid" and (res.frequency_hz is None or len(res.frequency_hz) == 0):
            MessageBox.warning(self, "提示", "PR 分析无有效数据:\n" + "\n".join(res.warnings[:6]))
            return False

        # 背景噪声基线：用与主信号完全相同的 PR 频谱管线计算背景谱，再插值对齐到主谱频点。
        # 仅用于上轨频谱显示（叠加/差值），不改变 PR 计算与判定。
        self._pr_baseline_db = None
        baseline_path = str(config.get("baseline_file_path", "") or "").strip()
        if baseline_path:
            try:
                baseline_signal, _baseline_sr = librosa.load(baseline_path, sr=int(sample_rate), mono=True)
                baseline_res = analyzer.compute(
                    baseline_signal,
                    self.v2pa_factor or 1.0,
                    params,
                    fan_pr_limits,
                )
                baseline_db = np.interp(
                    np.asarray(res.fft_freq_hz, dtype=float),
                    np.asarray(baseline_res.fft_freq_hz, dtype=float),
                    np.asarray(baseline_res.fft_magnitude_db, dtype=float),
                    left=np.nan,
                    right=np.nan,
                )
                if bool(config.get("baseline_smooth_third_octave", False)):
                    baseline_db = FftAnalysis._smooth_baseline_third_octave(
                        np.asarray(res.fft_freq_hz, dtype=float), baseline_db
                    )
                self._pr_baseline_db = baseline_db
            except Exception as e:
                MessageBox.warning(self, "提示", f"背景噪声基线加载失败: {str(e)[:200]}")
                self._pr_baseline_db = None

        self._plot_dual_track(res, params, fan_pr_limits)

        import logging
        _pr_logger = logging.getLogger("pr_analysis")
        for w in res.warnings:
            _pr_logger.warning(f"PR: {w}")

        self.result = {
            "frequency_hz": np.asarray(res.frequency_hz, dtype=float).tolist(),
            "band_power_db": np.asarray(res.band_power_db, dtype=float).tolist(),
            "pr_db": np.asarray(res.pr_db, dtype=float).tolist(),
            "fft_freq_hz": np.asarray(res.fft_freq_hz, dtype=float).tolist(),
            "fft_magnitude_db": np.asarray(res.fft_magnitude_db, dtype=float).tolist(),
            "baseline_db": (
                np.asarray(self._pr_baseline_db, dtype=float).tolist()
                if isinstance(self._pr_baseline_db, np.ndarray)
                else []
            ),
            "baseline_display_mode": str(config.get("baseline_display_mode", "overlay") or "overlay"),
            "max_pr_db": res.max_pr_db,
            "max_pr_frequency_hz": res.max_pr_frequency_hz,
            "decision_status": res.decision_status,
            "overall_ok": res.overall_ok,
            "max_exceed_db": res.max_exceed_db,
            "no_valid_main_tone": res.no_valid_main_tone,
            "main_tones": [self._tone_to_dict(t) for t in res.main_tones],
            "warnings": list(res.warnings),
            "metadata": dict(res.metadata),
        }
        self.export_detail = {
            "max_pr_db": res.max_pr_db,
            "max_pr_frequency_hz": res.max_pr_frequency_hz,
            "max_exceed_db": res.max_exceed_db,
            "decision_status": res.decision_status,
            "tone_count": int(sum(1 for t in res.main_tones if t.valid_main_tone)),
        }
        self.pdf_export_result = None
        return self.result

    def export_pdf_images(self, output_dir):
        """导出双轨图（上轨 + 下轨拼合为一张 PNG）。"""
        import os
        from PyQt5.QtGui import QPixmap, QPainter

        os.makedirs(str(output_dir), exist_ok=True)
        top_path = export_plot_widget_image(self.analysis_plot, output_dir, "_pr_upper")
        bot_path = export_plot_widget_image(self.pr_plot, output_dir, "_pr_lower")

        top_pix = QPixmap(top_path)
        bot_pix = QPixmap(bot_path)
        w = max(top_pix.width(), bot_pix.width())
        combined = QPixmap(w, top_pix.height() + bot_pix.height())
        combined.fill(Qt.white)
        painter = QPainter(combined)
        painter.drawPixmap(0, 0, top_pix)
        painter.drawPixmap(0, top_pix.height(), bot_pix)
        painter.end()

        combined_path = os.path.join(str(output_dir), "pr_dual_track.png")
        combined.save(combined_path)

        try:
            os.remove(top_path)
            os.remove(bot_path)
        except OSError:
            pass

        return [{"title": self.windowTitle(), "path": combined_path}]

    @staticmethod
    def _tone_to_dict(t):
        return {
            "frequency_hz": t.frequency_hz,
            "peak_db": t.peak_db,
            "target_band_hz": list(t.target_band_hz),
            "lower_adjacent_band_hz": list(t.lower_adjacent_band_hz),
            "upper_adjacent_band_hz": list(t.upper_adjacent_band_hz),
            "pr_db": t.pr_db,
            "limit_db": t.limit_db,
            "margin_db": t.margin_db,
            "ecma_prominent": t.ecma_prominent,
            "customer_ok": t.customer_ok,
            "is_ok": t.is_ok,
            "valid_main_tone": t.valid_main_tone,
            "user_specified": t.user_specified,
            "same_band_group_id": t.same_band_group_id,
            "same_band_representative": t.same_band_representative,
            "harmonic_order": t.harmonic_order,
            "bpf_verified": t.bpf_verified,
            "invalid_reasons": list(t.invalid_reasons),
        }

    @staticmethod
    def _finite_float(value, default=float("nan")):
        try:
            out = float(value)
        except (TypeError, ValueError):
            return default
        return out if np.isfinite(out) else default

    @staticmethod
    def _tone_sort_key(t):
        pr_db = ProminenceRatioAnalysis._finite_float(getattr(t, "pr_db", float("nan")), -1e9)
        limit_db = ProminenceRatioAnalysis._finite_float(getattr(t, "limit_db", float("nan")), 0.0)
        exceed_db = pr_db - limit_db
        is_ng = getattr(t, "customer_ok", None) is False
        is_rep = bool(getattr(t, "same_band_representative", True))
        return (1 if is_ng else 0, 1 if is_rep else 0, exceed_db, pr_db)

    def _select_pr_plot_tones(self, tones):
        cfg = self.analysis_config or {}
        max_labels = int(cfg.get("max_pr_annotation_tones", 8) or 8)
        max_labels = max(1, min(max_labels, 20))
        valid = [
            t for t in tones
            if getattr(t, "valid_main_tone", False)
            and np.isfinite(self._finite_float(getattr(t, "pr_db", float("nan"))))
        ]
        if not valid:
            return []
        user = [t for t in valid if getattr(t, "user_specified", False)]
        if user:
            return user
        primary = [
            t for t in valid
            if getattr(t, "customer_ok", None) is False
            or bool(getattr(t, "same_band_representative", True))
        ]
        pool = primary or valid
        return sorted(pool, key=self._tone_sort_key, reverse=True)[:max_labels]

    @staticmethod
    def _reset_pr_legend(plot, offset=(-10, 10)):
        plot_item = plot.getPlotItem()
        legend = getattr(plot_item, "legend", None)
        if legend is None:
            legend = plot.addLegend(offset=offset)
            legend.setParentItem(plot_item.getViewBox())
        else:
            legend.clear()
        legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=offset)
        try:
            legend.setBrush(QColor(255, 255, 255, 220))
            legend.setPen(mkPen(color=(180, 180, 180), width=1))
        except Exception:
            pass
        return legend

    @staticmethod
    def _add_legend_sample(plot, name, *, color, width=2, style=Qt.SolidLine, symbol=None, line=True, fill=None):
        """向图例添加一个不参与绘图的样本条目。

        fill 给定 RGBA 时，图例样本渲染为“半透明填充块 + 同色细边”，与频带色块
        （LinearRegionItem 填充 + InfiniteLine 边线）一致，避免图例颜色比实际更鲜艳。
        """
        if fill is not None:
            # RGBA 预合成到白底，与画面上半透明色块视觉一致
            r, g, b = fill[:3]
            a = fill[3] / 255.0
            premul = (int(255 * (1 - a) + r * a),
                      int(255 * (1 - a) + g * a),
                      int(255 * (1 - a) + b * a))
            plot.plot([], [], pen=mkPen(color=premul, width=8), name=name)
            return

        if symbol and not line:
            # pyqtgraph 对空 PlotDataItem 的 legend symbol 可能回退默认颜色；
            # 直接给 legend 一个 ScatterPlotItem 样本，确保颜色与图上主音点一致。
            scatter = pg.ScatterPlotItem(
                [],
                [],
                symbol=symbol,
                size=9,
                brush=pg.mkBrush(color) if color is not None else None,
                pen=mkPen(color=color, width=1) if color is not None else None,
            )
            legend = getattr(plot.getPlotItem(), "legend", None)
            if legend is not None:
                legend.addItem(scatter, name)
                return

        kwargs = {
            "pen": mkPen(color=color, width=width, style=style) if (line and color is not None) else mkPen(None),
            "name": name,
        }
        if symbol:
            kwargs.update({
                "symbol": symbol,
                "symbolSize": 9,
                "symbolBrush": pg.mkBrush(color) if color is not None else None,
                "symbolPen": mkPen(color=color, width=1) if color is not None else None,
            })
        plot.plot([], [], **kwargs)

    @staticmethod
    def _pr_axis_x(x_khz, is_log_x):
        """横轴坐标变换。

        log 模式下，pyqtgraph 仅自动变换 plot() 曲线数据；手动添加的图元
        (InfiniteLine / TextItem / LinearRegionItem) 需自行换算到 log10(kHz) 坐标。
        线性模式下原样返回 kHz。
        """
        x = ProminenceRatioAnalysis._finite_float(x_khz)
        if is_log_x:
            return float(np.log10(x)) if (np.isfinite(x) and x > 0) else float("nan")
        return x

    @staticmethod
    def _add_pr_band_region(plot, band_hz, color, is_log_x=False):
        f1, f2 = [ProminenceRatioAnalysis._finite_float(v) for v in band_hz]
        if not (np.isfinite(f1) and np.isfinite(f2)) or f2 <= f1:
            return
        x1 = ProminenceRatioAnalysis._pr_axis_x(f1 / 1000.0, is_log_x)
        x2 = ProminenceRatioAnalysis._pr_axis_x(f2 / 1000.0, is_log_x)
        if not (np.isfinite(x1) and np.isfinite(x2)) or x2 <= x1:
            return
        region = pg.LinearRegionItem(
            values=(x1, x2),
            orientation=pg.LinearRegionItem.Vertical,
            movable=False,
            brush=QColor(color[0], color[1], color[2], color[3]),
            pen=mkPen(color=color[:3], width=1),
        )
        region.setZValue(-10)
        plot.addItem(region)

    @staticmethod
    def _vline_label_font():
        """竖线标签（如 15%）用较大字体。"""
        font = QFont()
        font.setPixelSize(ui_style_const.scale_size_px(13))
        return font

    @staticmethod
    def _add_pr_vertical_line(plot, x_hz, pen, label=None, label_y=None, is_log_x=False):
        x = ProminenceRatioAnalysis._finite_float(x_hz)
        if not np.isfinite(x):
            return
        xpos = ProminenceRatioAnalysis._pr_axis_x(x / 1000.0, is_log_x)
        if not np.isfinite(xpos):
            return
        plot.addItem(pg.InfiniteLine(pos=xpos, angle=90, pen=pen))
        if label and label_y is not None and np.isfinite(label_y):
            text = pg.TextItem(label, color=(90, 90, 90), anchor=(0.5, 1.0))
            text.setFont(ProminenceRatioAnalysis._vline_label_font())
            text.setPos(xpos, label_y)
            plot.addItem(text)

    @staticmethod
    def _format_band_line(name, band_hz, power_db):
        """格式化单行频带信息：名称 频率范围 临界频带功率。"""
        f1 = ProminenceRatioAnalysis._finite_float(band_hz[0])
        f2 = ProminenceRatioAnalysis._finite_float(band_hz[1])
        power = ProminenceRatioAnalysis._finite_float(power_db)
        if not (np.isfinite(f1) and np.isfinite(f2)):
            return ""
        pwr = f"{power:.1f} dB" if np.isfinite(power) else "-- dB"
        return f"{name} {f1:.0f}-{f2:.0f}Hz  临界频带功率 {pwr}"

    def _plot_pr_band_annotations(self, tones, params, y_top, y_span, is_log_x=False):
        """上轨频带色块 + 信息卡片（白底框，放在色块右侧）。"""
        ratio = self._finite_float(getattr(params, "customer_band_ratio", 0.15), 0.15)
        dashed_15pct = mkPen(color=(120, 120, 120), width=1, style=Qt.DashLine)
        for idx, t in enumerate(tones):
            xm = self._finite_float(getattr(t, "target_power_db", float("nan")))
            xl = self._finite_float(getattr(t, "lower_adjacent_power_db", float("nan")))
            xu = self._finite_float(getattr(t, "upper_adjacent_power_db", float("nan")))
            lower_band = getattr(t, "lower_adjacent_band_hz", (np.nan, np.nan))
            target_band = getattr(t, "target_band_hz", (np.nan, np.nan))
            upper_band = getattr(t, "upper_adjacent_band_hz", (np.nan, np.nan))

            self._add_pr_band_region(self.analysis_plot, lower_band, (33, 102, 172, 34), is_log_x)
            self._add_pr_band_region(self.analysis_plot, upper_band, (33, 102, 172, 34), is_log_x)
            self._add_pr_band_region(self.analysis_plot, target_band, (214, 39, 40, 42), is_log_x)

            # 信息卡片：三行汇总，白底半透明框，放在上邻带右侧
            lines = [
                self._format_band_line("下邻带", lower_band, xl),
                self._format_band_line("目标带", target_band, xm),
                self._format_band_line("上邻带", upper_band, xu),
            ]
            card_text = "\n".join(ln for ln in lines if ln)
            if card_text:
                upper_f2 = self._finite_float(upper_band[1])
                card_x_khz = (upper_f2 / 1000.0 + 0.05) if np.isfinite(upper_f2) else 0.0
                card_x = self._pr_axis_x(card_x_khz, is_log_x)
                card_y = y_top - (0.03 + 0.30 * idx) * y_span
                if np.isfinite(card_x):
                    card = pg.TextItem(
                        card_text, color=(50, 50, 50), anchor=(0.0, 0.0),
                        border=mkPen(color=(160, 160, 160), width=1),
                        fill=QColor(255, 255, 255, 200),
                    )
                    card.setPos(card_x, card_y)
                    self.analysis_plot.addItem(card)

            ft = self._finite_float(getattr(t, "frequency_hz", float("nan")))
            if np.isfinite(ft):
                half_15 = 0.5 * ratio * ft
                line_label = f"{ratio * 100:.0f}%" if idx == 0 else None
                self._add_pr_vertical_line(self.analysis_plot, ft - half_15, dashed_15pct,
                                           line_label, y_top + 0.14 * y_span, is_log_x)
                self._add_pr_vertical_line(self.analysis_plot, ft + half_15, dashed_15pct,
                                           is_log_x=is_log_x)
                self._add_pr_vertical_line(
                    self.analysis_plot,
                    ft,
                    mkPen(color=(45, 45, 45), width=1, style=Qt.DotLine),
                    is_log_x=is_log_x,
                )

    def _plot_dual_track(self, res, params, fan_pr_limits):
        # 批量绘图期间冻结视图更新，避免每次 addItem 都触发重绘
        for pw in (self.analysis_plot, self.pr_plot):
            pw.getPlotItem().getViewBox().blockSignals(True)
            pw.getPlotItem().getViewBox().disableAutoRange()
        try:
            self._plot_dual_track_inner(res, params, fan_pr_limits)
        finally:
            for pw in (self.analysis_plot, self.pr_plot):
                pw.getPlotItem().getViewBox().blockSignals(False)
            # 上轨已手动 setYRange；下轨需要解冻后自动适配
            self.pr_plot.getPlotItem().getViewBox().enableAutoRange()
            self.pr_plot.getPlotItem().getViewBox().autoRange()

    def _plot_dual_track_inner(self, res, params, fan_pr_limits):
        self.analysis_plot.clear()
        self.pr_plot.clear()
        self._reset_pr_legend(self.analysis_plot)
        self._reset_pr_legend(self.pr_plot)

        freq = np.asarray(res.frequency_hz, dtype=float)
        if freq.size == 0:
            return
        f_khz = freq / 1000.0
        # 横轴标度：linear=线性(kHz)，log=对数(kHz)。双轨 X 轴联动，必须同步设置。
        x_axis_scale = str((self.analysis_config or {}).get("x_axis_scale", "linear") or "linear").lower()
        is_log_x = x_axis_scale == "log"
        self.analysis_plot.setLogMode(x=is_log_x, y=False)
        self.pr_plot.setLogMode(x=is_log_x, y=False)
        pr = np.asarray(res.pr_db, dtype=float)
        spectrum_freq = np.asarray(getattr(res, "fft_freq_hz", []), dtype=float)
        spectrum_power = np.asarray(getattr(res, "fft_magnitude_db", []), dtype=float)
        spectrum_weighting = str(
            (getattr(res, "metadata", {}) or {}).get("spectrum_display_weighting", "Z")
        ).upper()
        spectrum_unit = f"dB({spectrum_weighting})" if spectrum_weighting in ("A", "C") else "dB"

        # ---- 上轨：线性功率谱 + 频带功率标注 ----
        # 谱可达数万点，pyqtgraph 宽线渲染极慢；先做峰值包络降采样（保留峰，渲染快约 10×）
        max_pts = int((self.analysis_config or {}).get("max_plot_points", 2000))
        f_lo_hz = self._finite_float(getattr(params, "f_min", 0.0), 0.0)
        configured_f_hi_hz = self._finite_float(getattr(params, "f_max", freq[-1]), float(freq[-1]))
        f_hi_hz = configured_f_hi_hz
        if spectrum_freq.size and spectrum_freq.size == spectrum_power.size:
            spectrum_valid = (
                np.isfinite(spectrum_freq)
                & np.isfinite(spectrum_power)
                & (spectrum_freq >= f_lo_hz)
                & (spectrum_freq <= f_hi_hz)
            )
        else:
            spectrum_valid = np.zeros(0, dtype=bool)
        upper_curve_name = "线性功率谱" if spectrum_weighting == "Z" else f"{spectrum_weighting}计权功率谱"
        upper_label = f"{upper_curve_name} [{spectrum_unit}]"
        # 背景噪声基线（已对齐到 spectrum_freq）：叠加=灰色第二条曲线；差值=主谱减背景
        baseline_db = getattr(self, "_pr_baseline_db", None)
        has_baseline = (
            isinstance(baseline_db, np.ndarray)
            and baseline_db.size == spectrum_freq.size
            and spectrum_freq.size > 0
        )
        baseline_mode = str((self.analysis_config or {}).get("baseline_display_mode", "overlay") or "overlay")
        if spectrum_freq.size and np.any(spectrum_valid):
            sf_khz = spectrum_freq[spectrum_valid] / 1000.0
            main_power = spectrum_power[spectrum_valid]
            base_power = baseline_db[spectrum_valid] if has_baseline else None
            if has_baseline and baseline_mode == "delta":
                delta = main_power - base_power
                dx, dy = _decimate_peak_envelope(sf_khz, delta, max_pts)
                self.analysis_plot.plot(dx, dy, pen=mkPen(color=(51, 196, 77), width=2), name=upper_curve_name)
                finite_upper = delta
                upper_label = f"{upper_curve_name}-背景 [{spectrum_unit}]"
            else:
                sx, sy = _decimate_peak_envelope(sf_khz, main_power, max_pts)
                self.analysis_plot.plot(sx, sy, pen=mkPen(color=(51, 196, 77), width=2), name=upper_curve_name)
                finite_upper = main_power
                if has_baseline:
                    bx, by = _decimate_peak_envelope(sf_khz, base_power, max_pts)
                    self.analysis_plot.plot(bx, by, pen=mkPen(color=(128, 128, 128), width=2), name="背景噪声")
                    finite_upper = np.concatenate([main_power, base_power])
        else:
            band_power = np.asarray(res.band_power_db, dtype=float)
            band_valid = np.isfinite(band_power) & (freq >= f_lo_hz) & (freq <= f_hi_hz)
            bx, by = _decimate_peak_envelope(f_khz[band_valid], band_power[band_valid], max_pts)
            self.analysis_plot.plot(bx, by, pen=mkPen(color=(51, 196, 77), width=2), name="临界频带功率")
            finite_upper = band_power[band_valid]
            upper_label = "临界频带功率 [dB]"
        self.analysis_plot.setLabel("left", upper_label)
        self.analysis_plot.showGrid(x=True, y=True)
        self.analysis_plot.setTitle("线性功率谱 + PR值分布曲线", size="12px", color="k")

        if finite_upper.size:
            y_top = float(np.nanmax(finite_upper))
            y_bottom = float(np.nanmin(finite_upper))
            y_span = max(y_top - y_bottom, 1.0)
            # 顶部留白，给抬高的 15% 标签和信息卡片腾出空间
            self.analysis_plot.setYRange(y_bottom - 0.02 * y_span, y_top + 0.18 * y_span, padding=0)
        else:
            y_top, y_span = 1.0, 1.0
        plot_tones = self._select_pr_plot_tones(res.main_tones)
        self._plot_pr_band_annotations(plot_tones, params, y_top, y_span, is_log_x)
        if plot_tones:
            # 图例色块直接用画色块时的颜色（取 RGB 实色，便于辨认）
            self._add_legend_sample(self.analysis_plot, "目标频带", color=(214, 39, 40), fill=(214, 39, 40, 42))
            self._add_legend_sample(self.analysis_plot, "相邻频带", color=(33, 102, 172), fill=(33, 102, 172, 34))
            self._add_legend_sample(
                self.analysis_plot,
                f"{self._finite_float(getattr(params, 'customer_band_ratio', 0.15), 0.15) * 100:.0f}%划分线",
                color=(120, 120, 120),
                width=1,
                style=Qt.DashLine,
            )
        # ---- 下轨：PR 曲线 + 限值线 + 主音标注 ----
        valid = np.isfinite(pr)
        px, py = _decimate_peak_envelope(f_khz[valid], pr[valid], max_pts)
        tone_anchor_x = np.array(
            [self._finite_float(getattr(t, "frequency_hz", float("nan"))) / 1000.0 for t in plot_tones],
            dtype=float,
        )
        tone_anchor_y = np.array(
            [self._finite_float(getattr(t, "pr_db", float("nan"))) for t in plot_tones],
            dtype=float,
        )
        px, py = _merge_curve_anchor_points(px, py, tone_anchor_x, tone_anchor_y)
        self.pr_plot.plot(px, py, pen=mkPen(color=(33, 102, 172), width=2), name="PR")
        self.pr_plot.setLabel("left", "PR值 [dB]")
        self.pr_plot.setLabel("bottom", "频率 [kHz]")
        self.pr_plot.showGrid(x=True, y=True)

        x_min = max(0.0, f_lo_hz / 1000.0)
        x_max = f_hi_hz / 1000.0
        if np.isfinite(x_max) and x_max > x_min:
            if is_log_x:
                # log 模式下视图坐标为 log10(kHz)；下限需为正，缺省以 1Hz(=1e-3 kHz) 兜底
                lo_khz = x_min if x_min > 0 else 1e-3
                v_lo = float(np.log10(lo_khz))
                v_hi = float(np.log10(x_max))
            else:
                v_lo = x_min
                v_hi = x_max
            v_pad = (v_hi - v_lo) * 0.02
            self.analysis_plot.setXRange(v_lo, v_hi + v_pad, padding=0)
            self.pr_plot.setXRange(v_lo, v_hi + v_pad, padding=0)
            self.analysis_plot.getViewBox().setLimits(xMin=v_lo, xMax=v_hi + v_pad)
            self.pr_plot.getViewBox().setLimits(xMin=v_lo, xMax=v_hi + v_pad)

        # 分段限值线（仅在启用限值时绘制）
        if bool((self.analysis_config or {}).get("limit_checked", True)):
            self._add_legend_sample(
                self.pr_plot,
                "PR限值",
                color=(214, 39, 40),
                width=2,
                style=Qt.DashLine,
            )
            for band in fan_pr_limits:
                f_lo, f_hi, lim = float(band[0]), float(band[1]), float(band[2])
                self.pr_plot.plot(
                    [f_lo / 1000.0, f_hi / 1000.0], [lim, lim],
                    pen=mkPen(color=(214, 39, 40), width=2, style=Qt.DashLine),
                )

        # 主音标注：只标代表峰/NG 峰，避免候选峰文本铺满图面
        if plot_tones:
            tone_legend = "指定主音PR点" if any(getattr(t, "user_specified", False) for t in plot_tones) else "有效主音"
            self._add_legend_sample(self.pr_plot, tone_legend, color=(76, 175, 80), symbol="o", line=False)
        for idx, t in enumerate(plot_tones):
            is_ng = (t.customer_ok is False)
            color = (214, 39, 40) if is_ng else (76, 175, 80)
            self.pr_plot.plot(
                [t.frequency_hz / 1000.0], [t.pr_db],
                pen=mkPen(None),
                symbol="o",
                symbolSize=9,
                symbolBrush=pg.mkBrush(color),
                symbolPen=mkPen(color=color, width=1),
            )
            prefix = "NG " if is_ng else ""
            if getattr(t, "user_specified", False):
                prefix += "指定 "
            label = f"{prefix}{t.frequency_hz:.0f}Hz  PR={t.pr_db:.1f} dB"
            dev = getattr(t, "cross_deviation_db", None)
            if dev is not None:
                label += f", ΔECMA={dev:.2f} dB"
            text_color = color
            if dev is not None and dev > float((self.analysis_config or {}).get("ecma_cross_check_threshold_db", 0.5)):
                text_color = (200, 80, 0)
            text = pg.TextItem(label, color=text_color, anchor=(0.5, 1.15 + 0.25 * (idx % 3)))
            text.setPos(self._pr_axis_x(t.frequency_hz / 1000.0, is_log_x), t.pr_db + 0.15 * (idx % 3))
            self.pr_plot.addItem(text)


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

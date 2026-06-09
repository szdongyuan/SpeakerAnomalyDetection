"""
Runtime window for reference spectrum comparison.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph import mkPen
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from base.pdf_result_exporter import export_plot_widget_image
from base.core_algorithm.response import ReferenceSpectrumAnalyzer, ReferenceSpectrumParams
from base.data_struct.data_deal_struct import DataDealStruct
from base.reference_spectrum_cache import (
    REFERENCE_DATA_READY,
    extract_reference_channel_results,
    get_reference_data_state,
    load_reference_data,
)
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.widgets import MessageBox, PushButton, Label
from ui.graph_widget import custom_log_tick_strings

REFERENCE_CURVE_COLOR = "#5d6875"
CURRENT_CURVE_COLOR = "#00b8ab"
LIMIT_CURVE_COLOR = "#7158b4"
OUT_OF_RANGE_CURVE_COLOR = "#cc4b37"
HIGHLIGHT_REGION_COLOR = "#7158b4"


class ReferenceSpectrumCompareWindow(QWidget):
    def __init__(self, title_name):
        super().__init__()
        self.data_struct = DataDealStruct()
        self.analysis_config = None
        self.result = {}
        self.export_detail = {}
        self.title_name = title_name
        self._reference_payload = None
        self._compare_results = []
        self._plot_font_size = 16
        self._use_log_x_axis = False
        self.channel_cards: list[dict] = []
        self.channel_plots: list[pg.PlotWidget] = []
        self.summary_label = Label("等待分析")
        self.summary_label.setWordWrap(True)
        self.axis_mode_button = PushButton()
        self.axis_mode_button.setCheckable(True)
        self.axis_mode_button.setChecked(self._use_log_x_axis)
        self.axis_mode_button.clicked.connect(self._on_axis_mode_button_clicked)
        self._refresh_axis_mode_button_text()
        self.plot_scroll_area = QScrollArea()
        self.plot_scroll_area.setWidgetResizable(True)
        self.plot_scroll_area.setFrameShape(QFrame.NoFrame)
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout()
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(12)
        self.plot_container.setLayout(self.plot_layout)
        self.plot_layout.addStretch(1)
        self.plot_scroll_area.setWidget(self.plot_container)

        self.setWindowTitle(title_name)
        self.init_ui()

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))

        layout = QVBoxLayout()
        top_row = QHBoxLayout()
        top_row.addWidget(self.summary_label, 1)
        top_row.addWidget(self.axis_mode_button, 0, Qt.AlignRight)
        layout.addLayout(top_row)
        layout.addWidget(self.plot_scroll_area)
        self.setLayout(layout)

    def _refresh_axis_mode_button_text(self):
        self.axis_mode_button.setText("频率轴：对数" if self._use_log_x_axis else "频率轴：线性")

    def _axis_mode_view_text(self) -> str:
        return "对数坐标 / 监测视图" if self._use_log_x_axis else "线性坐标 / 监测视图"

    def _on_axis_mode_button_clicked(self, checked: bool):
        self._use_log_x_axis = bool(checked)
        self._refresh_axis_mode_button_text()
        if self._compare_results:
            self._render_all_channel_results()

    def export_pdf_images(self, output_dir):
        images = []
        for index, plot_widget in enumerate(self.channel_plots):
            image_path = export_plot_widget_image(
                plot_widget,
                output_dir,
                f"reference_spectrum_channel_{index + 1}",
            )
            channel_title = self._channel_export_title(index)
            images.append({"title": channel_title, "path": image_path})
        return images

    def _channel_export_title(self, index: int) -> str:
        if index < len(self.channel_cards):
            title_label = self.channel_cards[index].get("title_label")
            title_text = title_label.text() if title_label is not None else ""
            title_text = title_text.replace("通道：", "").strip()
            if title_text:
                return f"{self.title_name} {title_text}"
        return f"{self.title_name} CH{index + 1}"

    def _apply_plot_style(self, plot_widget: pg.PlotWidget, font_size: int | None = None):
        effective_font_size = int(font_size or self._plot_font_size)
        font = QFont()
        font.setPixelSize(effective_font_size)
        axis_text_color = (78, 86, 98)
        b_axis = plot_widget.getAxis("bottom")
        l_axis = plot_widget.getAxis("left")
        b_axis.logTickStrings = custom_log_tick_strings
        b_axis.setTickFont(font)
        l_axis.setTickFont(font)
        b_axis.setTextPen(axis_text_color)
        l_axis.setTextPen(axis_text_color)
        b_axis.setPen(pg.mkPen(color=(210, 215, 223), width=1))
        l_axis.setPen(pg.mkPen(color=(210, 215, 223), width=1))
        b_axis.setLabel(
            b_axis.labelText,
            color="#5a6270",
            **{"font-size": f"{effective_font_size}px"},
        )
        l_axis.setLabel(
            l_axis.labelText,
            color="#5a6270",
            **{"font-size": f"{effective_font_size}px"},
        )

    @staticmethod
    def _create_legend_item(text: str, color: str, *, dashed: bool = False, alert: bool = False) -> QWidget:
        legend_item = QWidget()
        legend_layout = QHBoxLayout()
        legend_layout.setContentsMargins(0, 0, 0, 0)
        legend_layout.setSpacing(8)
        legend_item.setLayout(legend_layout)

        swatch = QFrame()
        swatch.setFixedWidth(22)
        swatch.setFixedHeight(10)
        line_width = 3 if alert else 2
        line_style = "dashed" if dashed else "solid"
        swatch.setStyleSheet(f"border-top: {line_width}px {line_style} {color};")

        label = Label(text)
        label.setStyleSheet("color: #5f6f81; font-size: 13px;")

        legend_layout.addWidget(swatch)
        legend_layout.addWidget(label)
        return legend_item

    def _create_channel_card(self) -> dict:
        card = QFrame()
        card.setObjectName("rscChannelCard")
        card.setStyleSheet(
            """
            QFrame#rscChannelCard {
                background: #f8fafc;
                border: 1px solid #d7dde5;
                border-radius: 16px;
            }
            """
        )

        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(18, 16, 18, 14)
        card_layout.setSpacing(10)
        card.setLayout(card_layout)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(12)

        title_label = Label("通道：CH1")
        title_label.setStyleSheet("color: #203245; font-size: 18px; font-weight: 700;")

        mode_label = Label(self._axis_mode_view_text())
        mode_label.setStyleSheet("color: #607285; font-size: 13px;")
        mode_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        header_layout.addWidget(title_label, 1)
        header_layout.addWidget(mode_label, 0)
        card_layout.addLayout(header_layout)

        plot_widget = pg.PlotWidget()
        plot_widget.setBackground("#fcfdff")
        plot_widget.setMinimumHeight(260)
        plot_widget.getPlotItem().hideButtons()
        view_box = plot_widget.getViewBox()
        if view_box is not None:
            view_box.setDefaultPadding(0.0)
        self._apply_plot_style(plot_widget)
        self._get_or_create_plot_cache(plot_widget)
        card_layout.addWidget(plot_widget)

        legend_layout = QHBoxLayout()
        legend_layout.setContentsMargins(0, 2, 0, 0)
        legend_layout.setSpacing(18)
        reference_legend = self._create_legend_item("参考", REFERENCE_CURVE_COLOR)
        current_legend = self._create_legend_item("当前", CURRENT_CURVE_COLOR)
        upper_legend = self._create_legend_item("上限", LIMIT_CURVE_COLOR, dashed=True)
        lower_legend = self._create_legend_item("下限", LIMIT_CURVE_COLOR, dashed=True)
        out_of_range_legend = self._create_legend_item("超差", OUT_OF_RANGE_CURVE_COLOR, alert=True)
        legend_layout.addWidget(reference_legend)
        legend_layout.addWidget(current_legend)
        legend_layout.addWidget(upper_legend)
        legend_layout.addWidget(lower_legend)
        legend_layout.addWidget(out_of_range_legend)
        legend_layout.addStretch(1)
        card_layout.addLayout(legend_layout)

        return {
            "container": card,
            "plot_widget": plot_widget,
            "title_label": title_label,
            "mode_label": mode_label,
            "reference_legend": reference_legend,
            "current_legend": current_legend,
            "upper_legend": upper_legend,
            "lower_legend": lower_legend,
            "out_of_range_legend": out_of_range_legend,
        }

    def _create_plot_widget(self) -> pg.PlotWidget:
        card_info = self._create_channel_card()
        return card_info["plot_widget"]

    @staticmethod
    def _get_or_create_plot_cache(plot_widget: pg.PlotWidget) -> dict:
        cache = getattr(plot_widget, "_rsc_plot_cache", None)
        if cache is not None:
            return cache

        highlight_region = pg.LinearRegionItem(
            values=[1.0, 1.0],
            movable=False,
            brush=(113, 88, 180, 20),
            pen=pg.mkPen(color=HIGHLIGHT_REGION_COLOR, width=1),
        )
        highlight_region.setVisible(False)
        plot_widget.addItem(highlight_region)

        cache = {
            "highlight_region": highlight_region,
            "reference_curve": plot_widget.plot([], [], pen=mkPen(color=REFERENCE_CURVE_COLOR, width=2.7)),
            "current_curve": plot_widget.plot([], [], pen=mkPen(color=CURRENT_CURVE_COLOR, width=3.4)),
            "upper_curve": plot_widget.plot([], [], pen=mkPen(color=LIMIT_CURVE_COLOR, width=1.6, style=Qt.DashLine)),
            "lower_curve": plot_widget.plot([], [], pen=mkPen(color=LIMIT_CURVE_COLOR, width=1.6, style=Qt.DashLine)),
            "out_of_range_curve": plot_widget.plot([], [], pen=mkPen(color=OUT_OF_RANGE_CURVE_COLOR, width=3.0)),
        }
        setattr(plot_widget, "_rsc_plot_cache", cache)
        return cache

    def _clear_channel_plots(self):
        while self.channel_cards:
            card_info = self.channel_cards.pop()
            self.plot_layout.removeWidget(card_info["container"])
            card_info["container"].deleteLater()
        self.channel_plots = []

    def _ensure_channel_plot_count(self, target_count: int):
        while len(self.channel_cards) > target_count:
            card_info = self.channel_cards.pop()
            self.plot_layout.removeWidget(card_info["container"])
            card_info["container"].deleteLater()

        while len(self.channel_cards) < target_count:
            card_info = self._create_channel_card()
            insert_index = max(self.plot_layout.count() - 1, 0)
            self.plot_layout.insertWidget(insert_index, card_info["container"])
            self.channel_cards.append(card_info)

        self.channel_plots = [card_info["plot_widget"] for card_info in self.channel_cards]

    def _render_all_channel_results(self):
        self._ensure_channel_plot_count(len(self._compare_results))
        for card_info, compare_result in zip(self.channel_cards, self._compare_results):
            card_info["title_label"].setText(f"通道：{self._resolve_channel_label(compare_result.channel_index)}")
            card_info["mode_label"].setText(self._axis_mode_view_text())
            threshold_enabled = bool(getattr(compare_result, "threshold_enabled", True))
            card_info["upper_legend"].setVisible(threshold_enabled)
            card_info["lower_legend"].setVisible(threshold_enabled)
            card_info["out_of_range_legend"].setVisible(threshold_enabled)
            self._plot_channel_result(card_info["plot_widget"], compare_result)

    @staticmethod
    def _normalize_current_signal(signal_multi, signal_mono):
        if signal_multi is not None:
            arr = np.asarray(signal_multi, dtype=np.float64)
            if arr.ndim == 1:
                return arr.reshape(-1, 1)
            if arr.ndim == 2:
                return arr
            raise ValueError(f"Unsupported multi-channel signal shape: {arr.shape}")
        if signal_mono is None:
            raise ValueError("Missing current recording signal")
        mono_arr = np.asarray(signal_mono, dtype=np.float64)
        if mono_arr.ndim != 1:
            mono_arr = mono_arr.reshape(-1)
        return mono_arr.reshape(-1, 1)

    def _current_params(self) -> ReferenceSpectrumParams:
        cfg = self.analysis_config or {}
        return ReferenceSpectrumParams(
            window=str(cfg.get("window", "hann")),
            nperseg=int(cfg.get("nperseg", 4096)),
            overlap_ratio=float(cfg.get("overlap_ratio", 0.5)),
            smoothing=int(cfg.get("smoothing", 0)),
        )

    def _resolve_channel_label(self, channel_index: int) -> str:
        cfg = self.analysis_config or {}
        labels = cfg.get("channel_labels") or {}
        if channel_index in labels:
            return str(labels[channel_index])
        if str(channel_index) in labels:
            return str(labels[str(channel_index)])

        if isinstance(self._reference_payload, dict):
            channels = self._reference_payload.get("channels") or []
            for channel in channels:
                if int(channel.get("channel_index", -1)) == int(channel_index):
                    label = str(channel.get("label") or "").strip()
                    if label:
                        return label
        return f"CH{int(channel_index) + 1}"

    def _load_reference_payload(self):
        cfg = self.analysis_config or {}
        source_path = str(cfg.get("reference_source_path") or "")
        data_path = str(cfg.get("reference_data_path") or "")
        params = self._current_params()
        signal_multi = self._normalize_current_signal(
            getattr(self.data_struct, "store_wave_data_multi", None),
            getattr(self.data_struct, "store_wave_data", None),
        )
        sample_rate = int(self.data_struct.sample_rate or 0)
        if sample_rate <= 0:
            raise ValueError("Missing current sample rate")

        state = get_reference_data_state(
            reference_source_path=source_path,
            reference_data_path=data_path,
            params=params,
            current_sample_rate=sample_rate,
            current_channel_count=int(signal_multi.shape[1]),
        )
        if state != REFERENCE_DATA_READY:
            if state == "outdated":
                raise ValueError("参考数据与当前分析参数、采样率或通道数不一致，请重新生成参考数据")
            raise ValueError("参考数据不存在，请先生成参考数据")

        payload = load_reference_data(data_path)
        if payload is None:
            raise ValueError("参考数据读取失败")
        return payload, signal_multi, sample_rate

    def calculate_reference_spectrum(self):
        try:
            payload, signal_multi, sample_rate = self._load_reference_payload()
            analyzer = ReferenceSpectrumAnalyzer(sample_rate=sample_rate)
            reference_results = extract_reference_channel_results(payload)
            if int(signal_multi.shape[1]) != len(reference_results):
                raise ValueError("当前样本通道数与参考样本通道数不一致，无法分析")

            cfg = self.analysis_config or {}
            use_custom_band = bool(cfg.get("use_custom_band", True))
            threshold_enabled = bool(cfg.get("enable_threshold_judgment", True))
            start_freq_hz = float(cfg.get("start_freq_hz", 500)) if use_custom_band else None
            end_freq_hz = float(cfg.get("end_freq_hz", 8000)) if use_custom_band else None
            lower_offset_db = float(cfg.get("lower_offset_db", -3.0)) if threshold_enabled else None
            upper_offset_db = float(cfg.get("upper_offset_db", 3.0)) if threshold_enabled else None
            params = self._current_params()

            compare_results = []
            runtime_channel_results = []
            for reference_result in reference_results:
                channel_index = int(reference_result.channel_index)
                compare_result = analyzer.compare_channel_to_reference(
                    signal_multi[:, channel_index],
                    reference_result,
                    params=params,
                    start_freq_hz=start_freq_hz,
                    end_freq_hz=end_freq_hz,
                    lower_offset_db=lower_offset_db,
                    upper_offset_db=upper_offset_db,
                )
                compare_results.append(compare_result)
                runtime_channel_results.append(
                    {
                        "channel_index": channel_index,
                        "channel_name": self._resolve_channel_label(channel_index),
                        "max_over_upper_db": float(compare_result.max_over_upper_db),
                        "max_under_lower_db": float(compare_result.max_under_lower_db),
                        "max_exceed_db": float(compare_result.max_exceed_db),
                        "out_of_range_point_count": int(compare_result.out_of_range_point_count),
                        "out_of_range_ratio": float(compare_result.out_of_range_ratio),
                        "channel_ok": compare_result.channel_ok,
                        "threshold_enabled": bool(compare_result.threshold_enabled),
                    }
                )

            overall_ok = all(item["channel_ok"] for item in runtime_channel_results) if threshold_enabled else None
            overall_deviation = max((item["max_exceed_db"] for item in runtime_channel_results), default=0.0)
            self.data_struct.analysis_result_dict[self.title_name] = (overall_ok, float(overall_deviation))

            self._reference_payload = payload
            self._compare_results = compare_results
            self._refresh_summary_label(overall_ok, runtime_channel_results)
            self._render_all_channel_results()
            export_channel_curves = self._build_export_channel_curves()
            export_curve_payload = self._build_export_curve_payload(runtime_channel_results)

            self.result = {
                "overall_ok": overall_ok,
                "analysis_band_hz": [start_freq_hz, end_freq_hz] if use_custom_band else [],
                "use_custom_band": use_custom_band,
                "threshold_enabled": threshold_enabled,
                "per_channel_results": runtime_channel_results,
                "summary": self.summary_label.text(),
                "export_channel_curves": export_channel_curves,
                **export_curve_payload,
            }
            self.export_detail = {
                "reference_source_path": str(cfg.get("reference_source_path") or ""),
                "reference_data_path": str(cfg.get("reference_data_path") or ""),
                "analysis_band_hz": [start_freq_hz, end_freq_hz] if use_custom_band else [],
                "use_custom_band": use_custom_band,
                "threshold_enabled": threshold_enabled,
                "per_channel_results": runtime_channel_results,
                "overall_ok": overall_ok,
                "summary": self.summary_label.text(),
            }
            return self.result
        except Exception as e:
            MessageBox.warning(self, "提示", str(e)[:200])
            self._clear_channel_plots()
            self.summary_label.setText("分析失败")
            self.summary_label.setStyleSheet("color: rgb(200, 0, 0); font-size: 16px; font-weight: 700;")
            self.result = {
                "overall_ok": False,
                "analysis_band_hz": [],
                "per_channel_results": [],
                "summary": str(e)[:200],
            }
            self.export_detail = {
                "overall_ok": False,
                "summary": str(e)[:200],
            }
            return self.result

    def _refresh_summary_label(self, overall_ok: bool | None, runtime_channel_results: list[dict]):
        if overall_ok is None:
            summary_parts = ["整体结果：仅对比（未启用阈值判定）"]
        else:
            status_text = "OK" if overall_ok else "NG"
            summary_parts = [f"整体结果：{status_text}"]
        for item in runtime_channel_results:
            if item.get("channel_ok") is None:
                summary_parts.append(f"{item['channel_name']}: 仅对比 (最大偏差 {item['max_exceed_db']:.2f} dB)")
            else:
                summary_parts.append(
                    f"{item['channel_name']}: {'OK' if item['channel_ok'] else 'NG'} "
                    f"(最大超差 {item['max_exceed_db']:.2f} dB)"
                )
        self.summary_label.setText(" | ".join(summary_parts))
        if overall_ok is None:
            self.summary_label.setStyleSheet("color: rgb(90, 103, 120); font-size: 16px; font-weight: 700;")
        else:
            self.summary_label.setStyleSheet(
                "color: rgb(0, 128, 0); font-size: 16px; font-weight: 700;"
                if overall_ok
                else "color: rgb(200, 0, 0); font-size: 16px; font-weight: 700;"
            )

    def _build_export_curve_payload(self, runtime_channel_results: list[dict]) -> dict:
        if not self._compare_results or not runtime_channel_results:
            return {}

        worst_index = 0
        worst_exceed_db = -1.0
        for index, item in enumerate(runtime_channel_results):
            current_exceed_db = float(item.get("max_exceed_db", 0.0) or 0.0)
            if current_exceed_db > worst_exceed_db:
                worst_exceed_db = current_exceed_db
                worst_index = index

        export_result = self._compare_results[worst_index]
        export_channel_name = self._resolve_channel_label(export_result.channel_index)
        return {
            "export_channel_index": int(export_result.channel_index),
            "export_channel_name": export_channel_name,
            "frequencies_hz": np.asarray(export_result.frequencies_hz, dtype=np.float64).tolist(),
            "current_db": np.asarray(export_result.current_db, dtype=np.float64).tolist(),
            "reference_db": np.asarray(export_result.reference_db, dtype=np.float64).tolist(),
            "upper_limit_db": np.asarray(export_result.upper_limit_db, dtype=np.float64).tolist(),
            "lower_limit_db": np.asarray(export_result.lower_limit_db, dtype=np.float64).tolist(),
        }

    def _build_export_channel_curves(self) -> list[dict]:
        export_curves = []
        for compare_result in self._compare_results:
            channel_index = int(compare_result.channel_index)
            export_curves.append(
                {
                    "channel_index": channel_index,
                    "channel_name": self._resolve_channel_label(channel_index),
                    "frequencies_hz": np.asarray(compare_result.frequencies_hz, dtype=np.float64).tolist(),
                    "current_db": np.asarray(compare_result.current_db, dtype=np.float64).tolist(),
                    "reference_db": np.asarray(compare_result.reference_db, dtype=np.float64).tolist(),
                    "upper_limit_db": np.asarray(compare_result.upper_limit_db, dtype=np.float64).tolist(),
                    "lower_limit_db": np.asarray(compare_result.lower_limit_db, dtype=np.float64).tolist(),
                }
            )
        return export_curves

    def _plot_channel_result(self, plot_widget: pg.PlotWidget, compare_result):
        plot_cache = self._get_or_create_plot_cache(plot_widget)
        freq_hz = np.asarray(compare_result.frequencies_hz, dtype=np.float64)
        reference_db = np.asarray(compare_result.reference_db, dtype=np.float64)
        current_db = np.asarray(compare_result.current_db, dtype=np.float64)
        lower_limit_db = np.asarray(compare_result.lower_limit_db, dtype=np.float64)
        upper_limit_db = np.asarray(compare_result.upper_limit_db, dtype=np.float64)
        band_mask = np.asarray(compare_result.band_mask, dtype=bool)
        out_of_range_mask = np.asarray(compare_result.out_of_range_mask, dtype=bool)
        finite_freq_mask = np.isfinite(freq_hz)
        use_log_x_axis = bool(self._use_log_x_axis)
        plot_freq_mask = finite_freq_mask & (freq_hz > 0.0 if use_log_x_axis else freq_hz >= 0.0)

        if not np.any(plot_freq_mask):
            raise ValueError("频率轴缺少可用于显示的有效频率点")

        band_plot_mask = band_mask & plot_freq_mask
        highlight_region = plot_cache["highlight_region"]
        cfg = self.analysis_config or {}
        show_band_highlight = bool(cfg.get("use_custom_band", True)) and bool(cfg.get("highlight_analysis_band", True))
        if show_band_highlight and np.any(band_plot_mask):
            band_min = float(np.min(freq_hz[band_plot_mask]))
            band_max = float(np.max(freq_hz[band_plot_mask]))
            if use_log_x_axis:
                region_values = [float(np.log10(band_min)), float(np.log10(band_max))]
            else:
                region_values = [band_min, band_max]
            highlight_region.setRegion(region_values)
            highlight_region.setVisible(True)
        else:
            highlight_region.setVisible(False)

        def _set_curve_data(curve_item, y_values):
            plot_mask = plot_freq_mask & np.isfinite(y_values)
            if np.any(plot_mask):
                curve_item.setData(freq_hz[plot_mask], y_values[plot_mask])
            else:
                curve_item.setData([], [])

        _set_curve_data(plot_cache["reference_curve"], reference_db)
        _set_curve_data(plot_cache["current_curve"], current_db)
        if bool(compare_result.threshold_enabled):
            _set_curve_data(plot_cache["upper_curve"], upper_limit_db)
            _set_curve_data(plot_cache["lower_curve"], lower_limit_db)
        else:
            plot_cache["upper_curve"].setData([], [])
            plot_cache["lower_curve"].setData([], [])

        out_plot_mask = out_of_range_mask & plot_freq_mask & np.isfinite(current_db)
        if bool(compare_result.threshold_enabled) and np.any(out_plot_mask):
            out_x = np.where(out_plot_mask, freq_hz, np.nan)
            out_y = np.where(out_plot_mask, current_db, np.nan)
            plot_cache["out_of_range_curve"].setData(out_x, out_y)
        else:
            plot_cache["out_of_range_curve"].setData([], [])

        plot_widget.setTitle("")
        plot_widget.setLabel("left", "Spectrum (dB)")
        plot_widget.setLabel("bottom", "Frequency (Hz)")
        plot_widget.setLogMode(x=use_log_x_axis, y=False)
        plot_freq_hz = freq_hz[plot_freq_mask]
        if use_log_x_axis:
            plot_widget.setLimits(xMin=-1e307, xMax=1e307)
            plot_widget.setXRange(
                float(np.log10(np.min(plot_freq_hz))),
                float(np.log10(np.max(plot_freq_hz))),
                padding=0.02,
            )
        else:
            plot_widget.setLimits(xMin=0.0, xMax=float(np.max(plot_freq_hz)))
            plot_widget.setXRange(0.0, float(np.max(plot_freq_hz)), padding=0.0)
        plot_widget.showGrid(x=True, y=True, alpha=0.18)
        self._apply_plot_style(plot_widget)

"""
Reference spectrum comparison configuration dialog.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import List, Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from base.core_algorithm.response import ReferenceSpectrumAnalyzer, ReferenceSpectrumParams
from base.reference_spectrum_cache import (
    REFERENCE_DATA_NOT_GENERATED,
    REFERENCE_DATA_OUTDATED,
    REFERENCE_DATA_READY,
    build_reference_data_payload,
    get_reference_data_state,
    save_reference_data,
)
from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils


class ReferenceSpectrumConfigWindow(QDialog):
    WINDOW_OPTIONS = ["hann", "hamming", "blackman"]
    NPERSEG_OPTIONS = ["1024", "2048", "4096", "8192"]
    OVERLAP_OPTIONS = [("25%", 0.25), ("50%", 0.5), ("75%", 0.75)]
    SMOOTHING_OPTIONS = [
        ("不平滑", 0),
        ("1/3 Oct", 3),
        ("1/6 Oct", 6),
    ]

    def __init__(self, config_manager, model_type, available_channels: Optional[List[int]] = None):
        super().__init__()
        self.config_manager = config_manager
        self.model_type = model_type
        self.available_channels = list(available_channels or [])
        self.load_config = self.config_manager.load_config().get(model_type, {})
        self._reference_audio_meta = None
        self._reference_data_path = ""
        self._reference_data_last_generated_at = self.load_config.get("reference_data_last_generated_at")
        self._channel_name_inputs: dict[int, QLineEdit] = {}
        self.default_btn = None
        self.ok_btn = None
        self.init_ui()
        self._bind_initial_values()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumSize(520, 680)
        self.resize(560, 720)

        layout = QVBoxLayout()
        layout.addWidget(self._create_reference_group())
        layout.addWidget(self._create_band_group())
        layout.addWidget(self._create_threshold_group())
        layout.addWidget(self._create_channel_name_group())
        layout.addWidget(self._create_advanced_group())
        layout.addStretch()
        layout.addLayout(self.create_btn())
        self.setLayout(layout)

        self.setStyleSheet(
            ui_style_const.qgroupbox_style
            + ui_style_const.qlabel_style
            + ui_style_const.qlineedit_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qcombobox_style
            + ui_style_const.qspinbox_style
            + ui_style_const.qdoublespinbox_style
            + ui_style_const.qcheckbox_style
        )

    def _create_reference_group(self):
        group = QGroupBox("参考样本")
        layout = QVBoxLayout()

        source_path_row = QHBoxLayout()
        source_path_row.addWidget(QLabel("参考样本"))
        self.reference_source_path_edit = QLineEdit()
        self.reference_source_path_edit.setReadOnly(True)
        self.reference_source_path_edit.setPlaceholderText("请导入参考样本")
        browse_btn = QPushButton("选择")
        browse_btn.clicked.connect(self._on_reference_file_browse)
        source_path_row.addWidget(self.reference_source_path_edit)
        source_path_row.addWidget(browse_btn)

        layout.addLayout(source_path_row)
        group.setLayout(layout)
        return group

    def _create_band_group(self):
        group = QGroupBox("分析频段（可选）")
        layout = QVBoxLayout()

        self.custom_band_checkbox = QCheckBox("自定义分析频段")
        self.custom_band_checkbox.setChecked(False)
        self.custom_band_checkbox.stateChanged.connect(self._on_band_visibility_changed)
        layout.addWidget(self.custom_band_checkbox)

        self.band_container = QWidget()
        band_layout = QVBoxLayout()
        band_layout.setContentsMargins(0, 0, 0, 0)

        start_row = QHBoxLayout()
        start_row.addWidget(QLabel("起始频率 (Hz)"))
        self.start_freq_spinbox = QSpinBox()
        self.start_freq_spinbox.setRange(1, 200000)
        self.start_freq_spinbox.setValue(500)
        start_row.addWidget(self.start_freq_spinbox)

        end_row = QHBoxLayout()
        end_row.addWidget(QLabel("结束频率 (Hz)"))
        self.end_freq_spinbox = QSpinBox()
        self.end_freq_spinbox.setRange(1, 200000)
        self.end_freq_spinbox.setValue(8000)
        end_row.addWidget(self.end_freq_spinbox)

        self.highlight_band_checkbox = QCheckBox("图中高亮分析频段")
        self.highlight_band_checkbox.setChecked(True)

        band_layout.addLayout(start_row)
        band_layout.addLayout(end_row)
        band_layout.addWidget(self.highlight_band_checkbox)
        self.band_container.setLayout(band_layout)
        self.band_container.setVisible(False)
        layout.addWidget(self.band_container)
        group.setLayout(layout)
        return group

    def _create_threshold_group(self):
        group = QGroupBox("阈值判定（可选）")
        layout = QVBoxLayout()

        self.enable_threshold_checkbox = QCheckBox("启用阈值判定")
        self.enable_threshold_checkbox.setChecked(True)
        self.enable_threshold_checkbox.stateChanged.connect(self._on_threshold_visibility_changed)
        layout.addWidget(self.enable_threshold_checkbox)

        self.threshold_container = QWidget()
        threshold_layout = QVBoxLayout()
        threshold_layout.setContentsMargins(0, 0, 0, 0)

        lower_row = QHBoxLayout()
        lower_row.addWidget(QLabel("下偏移 (dB)"))
        self.lower_offset_spinbox = QDoubleSpinBox()
        self.lower_offset_spinbox.setDecimals(2)
        self.lower_offset_spinbox.setRange(-120.0, 120.0)
        self.lower_offset_spinbox.setValue(-3.0)
        lower_row.addWidget(self.lower_offset_spinbox)

        upper_row = QHBoxLayout()
        upper_row.addWidget(QLabel("上偏移 (dB)"))
        self.upper_offset_spinbox = QDoubleSpinBox()
        self.upper_offset_spinbox.setDecimals(2)
        self.upper_offset_spinbox.setRange(-120.0, 120.0)
        self.upper_offset_spinbox.setValue(3.0)
        upper_row.addWidget(self.upper_offset_spinbox)

        threshold_layout.addLayout(lower_row)
        threshold_layout.addLayout(upper_row)
        self.threshold_container.setLayout(threshold_layout)
        layout.addWidget(self.threshold_container)
        group.setLayout(layout)
        return group

    def _create_channel_name_group(self):
        self.channel_name_group = QGroupBox("通道名称（可选）")
        group_layout = QVBoxLayout()
        self.custom_channel_name_checkbox = QCheckBox("自定义通道名称")
        self.custom_channel_name_checkbox.setChecked(False)
        self.custom_channel_name_checkbox.stateChanged.connect(self._on_channel_name_visibility_changed)
        group_layout.addWidget(self.custom_channel_name_checkbox)

        self.channel_name_container = QWidget()
        self.channel_name_layout = QVBoxLayout()
        self.channel_name_layout.setContentsMargins(0, 0, 0, 0)
        self.channel_name_container.setLayout(self.channel_name_layout)
        self.channel_name_container.setVisible(False)
        group_layout.addWidget(self.channel_name_container)

        self.channel_name_group.setLayout(group_layout)
        self._rebuild_channel_name_rows(self._resolved_channel_count())
        return self.channel_name_group

    def _create_advanced_group(self):
        group = QGroupBox("高级设置（可选）")
        layout = QVBoxLayout()

        self.show_advanced_checkbox = QCheckBox("调整高级参数")
        self.show_advanced_checkbox.setChecked(False)
        self.show_advanced_checkbox.stateChanged.connect(self._on_advanced_visibility_changed)
        layout.addWidget(self.show_advanced_checkbox)

        self.advanced_container = QWidget()
        advanced_layout = QVBoxLayout()

        window_row = QHBoxLayout()
        window_row.addWidget(QLabel("窗函数"))
        self.window_combo_box = QComboBox()
        self.window_combo_box.addItems(self.WINDOW_OPTIONS)
        self.window_combo_box.currentIndexChanged.connect(self._on_generation_relevant_value_changed)
        window_row.addWidget(self.window_combo_box)

        nperseg_row = QHBoxLayout()
        nperseg_row.addWidget(QLabel("分段长度"))
        self.nperseg_combo_box = QComboBox()
        self.nperseg_combo_box.addItems(self.NPERSEG_OPTIONS)
        self.nperseg_combo_box.currentIndexChanged.connect(self._on_generation_relevant_value_changed)
        nperseg_row.addWidget(self.nperseg_combo_box)

        overlap_row = QHBoxLayout()
        overlap_row.addWidget(QLabel("重叠比例"))
        self.overlap_combo_box = QComboBox()
        for text, value in self.OVERLAP_OPTIONS:
            self.overlap_combo_box.addItem(text, float(value))
        self.overlap_combo_box.currentIndexChanged.connect(self._on_generation_relevant_value_changed)
        overlap_row.addWidget(self.overlap_combo_box)

        smoothing_row = QHBoxLayout()
        smoothing_row.addWidget(QLabel("平滑"))
        self.smoothing_combo_box = QComboBox()
        for text, value in self.SMOOTHING_OPTIONS:
            self.smoothing_combo_box.addItem(text, int(value))
        self.smoothing_combo_box.currentIndexChanged.connect(self._on_generation_relevant_value_changed)
        smoothing_row.addWidget(self.smoothing_combo_box)

        advanced_layout.addLayout(window_row)
        advanced_layout.addLayout(nperseg_row)
        advanced_layout.addLayout(overlap_row)
        advanced_layout.addLayout(smoothing_row)
        self.advanced_container.setLayout(advanced_layout)
        self.advanced_container.setVisible(False)

        layout.addWidget(self.advanced_container)
        group.setLayout(layout)
        return group

    def create_btn(self):
        btn_layout = QHBoxLayout()
        self.default_btn = QPushButton(" 设为默认 ")
        self.default_btn.clicked.connect(self.on_default_btn_clicked)
        self.ok_btn = QPushButton(" 确  认 ")
        self.ok_btn.clicked.connect(self.on_click_ok_btn)
        btn_layout.addWidget(self.default_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.ok_btn)
        return btn_layout

    def _bind_initial_values(self):
        cfg = self.load_config or {}
        source_path = str(cfg.get("reference_source_path") or "")
        self.reference_source_path_edit.setText(source_path)
        self._reference_data_path = str(
            cfg.get("reference_data_path") or self._build_default_reference_data_path(source_path)
        )

        self.start_freq_spinbox.setValue(int(cfg.get("start_freq_hz", 500)))
        self.end_freq_spinbox.setValue(int(cfg.get("end_freq_hz", 8000)))
        self.highlight_band_checkbox.setChecked(bool(cfg.get("highlight_analysis_band", True)))
        self.custom_band_checkbox.setChecked(bool(cfg.get("use_custom_band", True)))
        self.lower_offset_spinbox.setValue(float(cfg.get("lower_offset_db", -3.0)))
        self.upper_offset_spinbox.setValue(float(cfg.get("upper_offset_db", 3.0)))
        self.enable_threshold_checkbox.setChecked(bool(cfg.get("enable_threshold_judgment", True)))

        self.window_combo_box.setCurrentText(str(cfg.get("window", "hann")))
        self.nperseg_combo_box.setCurrentText(str(int(cfg.get("nperseg", 4096))))
        overlap_value = float(cfg.get("overlap_ratio", 0.5))
        overlap_index = self.overlap_combo_box.findData(overlap_value)
        self.overlap_combo_box.setCurrentIndex(overlap_index if overlap_index >= 0 else 1)
        smoothing_value = int(cfg.get("smoothing", 0))
        smoothing_index = self.smoothing_combo_box.findData(smoothing_value)
        self.smoothing_combo_box.setCurrentIndex(smoothing_index if smoothing_index >= 0 else 0)
        self.show_advanced_checkbox.setChecked(self._has_non_default_advanced_params(cfg))

        if source_path:
            try:
                self._probe_and_refresh_reference_meta(source_path)
            except Exception:
                self._reference_audio_meta = None
                self._refresh_reference_info_labels(None)
                self._rebuild_channel_name_rows(self._resolved_channel_count())
        else:
            self._refresh_reference_info_labels(None)
            self._rebuild_channel_name_rows(self._resolved_channel_count())
        self._load_channel_labels_into_inputs()
        self.custom_channel_name_checkbox.setChecked(bool(cfg.get("channel_labels") or {}))
        self._refresh_reference_data_status()

    def _resolved_channel_count(self) -> int:
        if self._reference_audio_meta and self._reference_audio_meta.get("channel_count"):
            return int(self._reference_audio_meta["channel_count"])
        load_labels = self.load_config.get("channel_labels") or {}
        return max(1, len(load_labels))

    def _create_path_hash(self, text: str) -> str:
        normalized = os.path.abspath(text).replace("\\", "/").lower().encode("utf-8")
        return hashlib.md5(normalized).hexdigest()[:8]

    @staticmethod
    def _has_non_default_advanced_params(cfg: dict) -> bool:
        if not isinstance(cfg, dict):
            return False
        return (
            str(cfg.get("window", "hann")) != "hann"
            or int(cfg.get("nperseg", 4096)) != 4096
            or float(cfg.get("overlap_ratio", 0.5)) != 0.5
            or int(cfg.get("smoothing", 0)) != 0
        )

    def _build_default_reference_data_path(self, source_path: str) -> str:
        if not source_path:
            return ""
        source_abs = os.path.abspath(source_path)
        stem = os.path.splitext(os.path.basename(source_abs))[0]
        suffix = self._create_path_hash(source_abs)
        target_dir = os.path.join(DEFAULT_DIR, "audio_data", "reference_cache")
        return os.path.join(target_dir, f"{stem}_{suffix}.rsc.json").replace("\\", "/")

    def _current_params(self) -> ReferenceSpectrumParams:
        return ReferenceSpectrumParams(
            window=self.window_combo_box.currentText(),
            nperseg=int(self.nperseg_combo_box.currentText()),
            overlap_ratio=float(self.overlap_combo_box.currentData()),
            smoothing=int(self.smoothing_combo_box.currentData()),
        )

    def _probe_audio_file(self, file_path: str):
        if not file_path:
            raise ValueError("未选择参考音频文件")
        try:
            import soundfile as sf

            audio_data, sample_rate = sf.read(file_path, dtype="float32", always_2d=True)
            audio_data = np.asarray(audio_data, dtype=np.float32)
        except Exception:
            import librosa

            audio_data, sample_rate = librosa.load(file_path, sr=None, mono=False)
            audio_data = np.asarray(audio_data, dtype=np.float32)
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(-1, 1)
            elif audio_data.ndim == 2:
                audio_data = audio_data.T
            else:
                raise ValueError(f"不支持的音频形状: {audio_data.shape}")

        if audio_data.ndim != 2:
            raise ValueError(f"参考音频需为单通道或多通道波形，当前形状: {audio_data.shape}")
        if audio_data.shape[0] <= 0:
            raise ValueError("参考音频为空")
        return audio_data, int(sample_rate)

    def _probe_and_refresh_reference_meta(self, file_path: str):
        audio_data, sample_rate = self._probe_audio_file(file_path)
        self._reference_audio_meta = {
            "sample_rate": int(sample_rate),
            "channel_count": int(audio_data.shape[1]),
            "frame_count": int(audio_data.shape[0]),
            "duration_sec": float(audio_data.shape[0] / sample_rate),
        }
        self._refresh_reference_info_labels(self._reference_audio_meta)
        self._rebuild_channel_name_rows(int(audio_data.shape[1]))

    def _refresh_reference_info_labels(self, meta: Optional[dict]):
        _ = meta

    def _load_channel_labels_into_inputs(self):
        saved_labels = self.load_config.get("channel_labels") or {}
        for channel_index, line_edit in self._channel_name_inputs.items():
            if channel_index in saved_labels:
                line_edit.setText(str(saved_labels[channel_index]))
            elif str(channel_index) in saved_labels:
                line_edit.setText(str(saved_labels[str(channel_index)]))

    def _rebuild_channel_name_rows(self, channel_count: int):
        channel_count_value = max(1, int(channel_count))
        previous_values = {idx: widget.text() for idx, widget in self._channel_name_inputs.items()}
        while self.channel_name_layout.count():
            item = self.channel_name_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._channel_name_inputs = {}

        for channel_index in range(channel_count_value):
            row_widget = QWidget()
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(QLabel(f"CH{channel_index + 1} 名称"))
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(f"默认显示为 CH{channel_index + 1}")
            if channel_index in previous_values:
                line_edit.setText(previous_values[channel_index])
            row_layout.addWidget(line_edit)
            row_widget.setLayout(row_layout)
            self.channel_name_layout.addWidget(row_widget)
            self._channel_name_inputs[channel_index] = line_edit

    def _collect_channel_labels(self) -> dict:
        if not getattr(self, "custom_channel_name_checkbox", None) or not self.custom_channel_name_checkbox.isChecked():
            return {}
        labels = {}
        for channel_index, line_edit in self._channel_name_inputs.items():
            text = line_edit.text().strip()
            if text:
                labels[str(channel_index)] = text
        return labels

    def _current_reference_state(self) -> str:
        source_path = self.reference_source_path_edit.text().strip()
        target_path = self._reference_data_path or self._build_default_reference_data_path(source_path)
        return get_reference_data_state(
            reference_source_path=source_path,
            reference_data_path=target_path,
            params=self._current_params(),
        )

    def _refresh_reference_data_status(self):
        state = self._current_reference_state()
        self._update_action_button_states(state)

    def _update_action_button_states(self, state: Optional[str] = None):
        has_source = bool(self.reference_source_path_edit.text().strip())
        if self.ok_btn is not None:
            self.ok_btn.setEnabled(has_source)
        if self.default_btn is not None:
            self.default_btn.setEnabled(has_source)

    def _on_reference_file_browse(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择参考音频文件",
            DEFAULT_DIR + "audio_data/stored_data",
            "WAV Files (*.wav)",
        )
        if not file_path:
            return
        file_path = file_path.replace("\\", "/")
        self.reference_source_path_edit.setText(file_path)
        self._reference_data_path = self._build_default_reference_data_path(file_path)
        self._reference_data_last_generated_at = None
        try:
            self._probe_and_refresh_reference_meta(file_path)
        except Exception as e:
            self._reference_audio_meta = None
            self._refresh_reference_info_labels(None)
            QMessageBox.warning(self, "提示", f"参考音频读取失败：{str(e)[:200]}")
        self._refresh_reference_data_status()

    def _on_generation_relevant_value_changed(self):
        self._refresh_reference_data_status()

    def _on_advanced_visibility_changed(self, state):
        self.advanced_container.setVisible(state == Qt.Checked)

    def _on_band_visibility_changed(self, state):
        self.band_container.setVisible(state == Qt.Checked)

    def _on_threshold_visibility_changed(self, state):
        self.threshold_container.setVisible(state == Qt.Checked)

    def _on_channel_name_visibility_changed(self, state):
        self.channel_name_container.setVisible(state == Qt.Checked)

    def _generate_reference_data(self):
        source_path = self.reference_source_path_edit.text().strip()
        if not source_path:
            raise ValueError("请先导入参考样本")
        audio_data, sample_rate = self._probe_audio_file(source_path)
        analyzer = ReferenceSpectrumAnalyzer(sample_rate=sample_rate)
        params = self._current_params()
        channel_results = analyzer.build_multi_channel_spectrum(audio_data, params=params)
        payload = build_reference_data_payload(
            reference_source_path=source_path,
            sample_rate=sample_rate,
            channel_results=channel_results,
            params=params,
            channel_labels=self._collect_channel_labels(),
            frame_count=int(audio_data.shape[0]),
        )
        target_path = self._reference_data_path or self._build_default_reference_data_path(source_path)
        self._reference_data_path = target_path
        if not save_reference_data(target_path, payload):
            raise ValueError("参考样本内部数据保存失败")
        self._reference_audio_meta = {
            "sample_rate": int(sample_rate),
            "channel_count": int(audio_data.shape[1]),
            "frame_count": int(audio_data.shape[0]),
            "duration_sec": float(audio_data.shape[0] / sample_rate),
        }
        self._reference_data_last_generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._refresh_reference_info_labels(self._reference_audio_meta)

    def _ensure_reference_data_ready(self, *, show_error_message: bool = False) -> bool:
        state = self._current_reference_state()
        if state == REFERENCE_DATA_READY:
            return True
        try:
            self._generate_reference_data()
            self._refresh_reference_data_status()
            return True
        except Exception as e:
            self._refresh_reference_data_status()
            if show_error_message:
                QMessageBox.warning(self, "提示", f"参考样本准备失败：{str(e)[:200]}")
            return False

    def validate(self, *, save_reference_sample_data: bool = False) -> bool:
        source_path = self.reference_source_path_edit.text().strip()
        if not source_path:
            QMessageBox.warning(self, "提示", "请先导入参考样本")
            return False
        if not os.path.exists(source_path):
            QMessageBox.warning(self, "提示", "参考样本文件不存在，请重新选择")
            return False
        if self.custom_band_checkbox.isChecked() and self.end_freq_spinbox.value() <= self.start_freq_spinbox.value():
            QMessageBox.warning(self, "提示", "分析频段配置错误，请检查起始频率和结束频率")
            return False
        if self.enable_threshold_checkbox.isChecked() and self.lower_offset_spinbox.value() > self.upper_offset_spinbox.value():
            QMessageBox.warning(self, "提示", "阈值范围配置错误，请检查上下偏移")
            return False
        if save_reference_sample_data and not self._ensure_reference_data_ready(show_error_message=True):
            return False
        return True

    def get_default_config(self):
        config = {
            "reference_source_path": self.reference_source_path_edit.text().strip(),
            "reference_data_path": self._reference_data_path
            or self._build_default_reference_data_path(self.reference_source_path_edit.text().strip()),
            "reference_data_state": self._current_reference_state(),
            "reference_data_last_generated_at": self._reference_data_last_generated_at,
            "use_custom_band": self.custom_band_checkbox.isChecked(),
            "start_freq_hz": int(self.start_freq_spinbox.value()),
            "end_freq_hz": int(self.end_freq_spinbox.value()),
            "highlight_analysis_band": self.highlight_band_checkbox.isChecked(),
            "enable_threshold_judgment": self.enable_threshold_checkbox.isChecked(),
            "lower_offset_db": float(self.lower_offset_spinbox.value()),
            "upper_offset_db": float(self.upper_offset_spinbox.value()),
            "channel_labels": self._collect_channel_labels(),
            "window": self.window_combo_box.currentText(),
            "nperseg": int(self.nperseg_combo_box.currentText()),
            "overlap_ratio": float(self.overlap_combo_box.currentData()),
            "smoothing": int(self.smoothing_combo_box.currentData()),
        }
        return config

    def on_default_btn_clicked(self):
        if not self.validate(save_reference_sample_data=False):
            return
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config("RSC", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        if not self.validate(save_reference_sample_data=True):
            return
        config_data = self.get_default_config()
        self.accept()
        return config_data

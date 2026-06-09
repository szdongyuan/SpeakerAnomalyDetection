"""突出比 (PR / Prominence Ratio) 分析配置对话框。

需求书《笔记本电脑风扇噪音核心测试要求书》要点：
- PR 模块适配 ECMA-74 Annex D，默认按 ECMA/ECMA-418-1 标准临界带计算 PR，
  15% 固定比例带宽保留为图面划分线和兼容对比模式。
  （PR 为带功率比值，需求书 4.4.3 把 PR 模块幅度标度写为 dB 而非 dB(A)；计权为隐藏高级项。）
- 临界带功率积分与 PR 频谱共用同一段 FFT 频谱，临界带宽与 FFT 参数天然同步
  （对应 3.3"确认临界带宽与 FFT 参数同步"，截图本配置界面即可留存）。

`PRConfigWindow` 与 SPL/FBA 等同级，通过 `operation_sequence.create_config_dialog` 路由，
保存的配置由 `ProminenceRatioParams.from_config()` 读取。
"""

import re
from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QGridLayout, QSizePolicy, QWidget

from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import (
    PushButton,
    ComboBox,
    Label,
    GroupBox,
    SpinBox,
    DoubleSpinBox,
    CheckBox,
    MessageBox,
)


# 临界带定义：UI 文本 ↔ 配置值
MODE_LABELS = [
    ("ECMA 标准临界带", "ecma"),
    ("固定比例带宽", "customer_15pct"),
]

# 需求书默认三段 PR 限值 [f_lo, f_hi, limit_db]
DEFAULT_FAN_PR_LIMITS = [[100, 2000, 4], [2000, 5000, 2], [5000, 20000, 4]]


def default_pr_config() -> dict:
    """PR 顶层默认配置（代码内可靠默认源，不依赖被 gitignore 的 JSON）。

    与需求书校核口径一致：ECMA-74 Annex D / ECMA-418-1 标准临界带
    + 线性(不计权) + 20Hz 高通；15% 固定比例仅作显示划分线/兼容模式。
    （weighting=auto→线性(Z)，为隐藏高级项，界面不暴露。）
    """
    return {
        "analysis_channel": 0,
        "critical_band_mode": "ecma",
        "standard": "ecma74_annexD",
        "mode_profile": "ecma2022",
        "f_min": 100,
        "f_max": 20000,
        "window": "hann",
        "window_samples": 65536,
        "overlap_ratio": 0.75,
        "target_resolution_hz": 1.0,
        "weighting": "auto",
        "customer_band_ratio": 0.15,
        "highpass_hz": 20.0,
        "dc_removal": "mean",
        "bpf_enabled": False,
        "blade_count": 0,
        "rpm": 0,
        "bpf_tolerance_percent": 5.0,
        # 需求书 1.4 第7条 / 4.3：谐波分量一律不纳入判定，固定 False，不开放 UI 配置。
        "include_harmonics_in_customer_judgement": False,
        "limit_checked": True,
        "fan_pr_limits": [list(b) for b in DEFAULT_FAN_PR_LIMITS],
    }


class PRConfigWindow(QDialog):
    """突出比 (PR) 分析配置窗口。"""

    def __init__(self, config_manager, model_type, available_channels: Optional[List[int]] = None):
        super().__init__()
        self.config_manager = config_manager
        self.model_type_str = "".join(re.findall(r"[A-Za-z]", str(model_type))) or "PR"
        self.show_channel_selector = available_channels is not None
        self.available_channels = self._normalize_available_channels(available_channels)

        full_config = self.config_manager.load_config()
        saved = full_config.get(model_type, {}) if isinstance(full_config, dict) else {}
        # 以默认配置兜底，保证旧配置/缺字段也能完整加载
        self.load_config = {**default_pr_config(), **(saved or {})}

        self.init_ui()

    @staticmethod
    def _normalize_available_channels(available_channels):
        try:
            channels = sorted({int(ch) for ch in (available_channels or [])})
        except Exception:
            channels = []
        return channels or [0]

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------
    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        self.setWindowTitle("PR (突出比) 分析配置")
        self.setMinimumSize(640, 700)
        self.resize(640, 700)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(18, 18, 18, 18)

        if self.show_channel_selector:
            main_layout.addLayout(self._create_channel_layout())

        main_layout.addWidget(self._create_mode_group())
        main_layout.addWidget(self._create_fft_group())
        main_layout.addWidget(self._create_limit_group())

        main_layout.addStretch(1)
        main_layout.addLayout(self.create_btn())
        self.setLayout(main_layout)

        self._on_mode_changed()

    def _create_channel_layout(self):
        layout = QHBoxLayout()
        layout.addWidget(Label("通道:"))
        self.channel_combo_box = ComboBox()
        for ch in self.available_channels:
            self.channel_combo_box.addItem(f"In{int(ch) + 1}", int(ch))
        saved_channel = self.load_config.get("analysis_channel", None)
        if saved_channel is None or int(saved_channel) not in self.available_channels:
            saved_channel = int(self.available_channels[0])
        idx = self.channel_combo_box.findData(int(saved_channel))
        self.channel_combo_box.setCurrentIndex(idx if idx >= 0 else 0)
        layout.addWidget(self.channel_combo_box, 1)
        return layout

    def _create_mode_group(self):
        group = GroupBox("临界带定义 / 频率范围")
        layout = QVBoxLayout()

        row_mode = QHBoxLayout()
        row_mode.addWidget(Label("临界带定义:"))
        self.mode_combo = ComboBox()
        for text, _val in MODE_LABELS:
            self.mode_combo.addItem(text)
        saved_mode = str(self.load_config.get("critical_band_mode", "ecma"))
        mode_idx = next((i for i, (_t, v) in enumerate(MODE_LABELS) if v == saved_mode), 0)
        self.mode_combo.setCurrentIndex(mode_idx)
        self.mode_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mode_combo.setMinimumWidth(200)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        row_mode.addWidget(self.mode_combo, 1)

        # 固定带宽比例与“临界带定义”同一行（仅固定比例带宽模式可见）；百分比显示（如 15 %）
        self.band_ratio_spin = DoubleSpinBox()
        self.band_ratio_spin.setRange(5, 50)
        self.band_ratio_spin.setDecimals(0)
        self.band_ratio_spin.setSingleStep(1)
        self.band_ratio_spin.setSuffix(" %")
        self.band_ratio_spin.setValue(round(float(self.load_config.get("customer_band_ratio", 0.15)) * 100))
        self.ratio_container = QWidget()
        row_ratio = QHBoxLayout(self.ratio_container)
        row_ratio.setContentsMargins(12, 0, 0, 0)
        row_ratio.addWidget(Label("固定带宽比例:"))
        row_ratio.addWidget(self.band_ratio_spin)
        row_mode.addWidget(self.ratio_container)
        layout.addLayout(row_mode)

        row_range = QHBoxLayout()
        row_range.addWidget(Label("频率下限:"))
        self.f_min_spin = DoubleSpinBox()
        self.f_min_spin.setRange(1, 20000)
        self.f_min_spin.setDecimals(0)
        self.f_min_spin.setValue(float(self.load_config.get("f_min", 100)))
        self.f_min_spin.setSuffix(" Hz")
        row_range.addWidget(self.f_min_spin)
        row_range.addSpacing(16)
        row_range.addWidget(Label("频率上限:"))
        self.f_max_spin = DoubleSpinBox()
        self.f_max_spin.setRange(100, 96000)
        self.f_max_spin.setDecimals(0)
        self.f_max_spin.setValue(float(self.load_config.get("f_max", 20000)))
        self.f_max_spin.setSuffix(" Hz")
        row_range.addWidget(self.f_max_spin)
        layout.addLayout(row_range)

        group.setLayout(layout)
        return group

    def _create_fft_group(self):
        group = GroupBox("FFT 参数")
        layout = QGridLayout()

        layout.addWidget(Label("窗函数:"), 0, 0)
        self.window_combo = ComboBox()
        self.window_combo.addItems(["hann", "hamming", "blackman"])
        widx = self.window_combo.findText(str(self.load_config.get("window", "hann")))
        self.window_combo.setCurrentIndex(widx if widx >= 0 else 0)
        layout.addWidget(self.window_combo, 0, 1)

        layout.addWidget(Label("窗长 (点):"), 1, 0)
        self.window_samples_combo = ComboBox()
        for n in (8192, 16384, 32768, 65536, 131072):
            self.window_samples_combo.addItem(str(n), n)
        nidx = self.window_samples_combo.findData(int(self.load_config.get("window_samples", 65536)))
        self.window_samples_combo.setCurrentIndex(nidx if nidx >= 0 else 3)
        layout.addWidget(self.window_samples_combo, 1, 1)

        layout.addWidget(Label("重叠率:"), 2, 0)
        self.overlap_spin = DoubleSpinBox()
        self.overlap_spin.setRange(0.0, 0.95)
        self.overlap_spin.setDecimals(2)
        self.overlap_spin.setSingleStep(0.05)
        self.overlap_spin.setValue(float(self.load_config.get("overlap_ratio", 0.75)))
        layout.addWidget(self.overlap_spin, 2, 1)

        saved_hp = float(self.load_config.get("highpass_hz", 20.0))
        self.highpass_check = CheckBox(" 高通去直流")
        self.highpass_check.setChecked(saved_hp > 0)
        layout.addWidget(self.highpass_check, 3, 0)
        self.highpass_spin = DoubleSpinBox()
        self.highpass_spin.setRange(1.0, 200.0)
        self.highpass_spin.setDecimals(0)
        self.highpass_spin.setValue(saved_hp if saved_hp > 0 else 20.0)
        self.highpass_spin.setSuffix(" Hz")
        self.highpass_spin.setEnabled(saved_hp > 0)
        self.highpass_check.stateChanged.connect(
            lambda *_: self.highpass_spin.setEnabled(self.highpass_check.isChecked())
        )
        layout.addWidget(self.highpass_spin, 3, 1)

        group.setLayout(layout)
        return group

    def _create_limit_group(self):
        group = GroupBox("PR 限值对比")
        layout = QVBoxLayout()

        self.limit_check = CheckBox(" 启用 PR 限值对比")
        self.limit_check.setChecked(bool(self.load_config.get("limit_checked", True)))
        layout.addWidget(self.limit_check)

        saved_limits = self.load_config.get("fan_pr_limits") or DEFAULT_FAN_PR_LIMITS
        header = QHBoxLayout()
        header.addWidget(Label("频率下限 (Hz)"))
        header.addWidget(Label("频率上限 (Hz)"))
        header.addWidget(Label("PR 限值 (dB)"))
        layout.addLayout(header)

        self.limit_rows = []
        for band in saved_limits:
            row = QHBoxLayout()
            lo = DoubleSpinBox(); lo.setRange(1, 96000); lo.setDecimals(0); lo.setValue(float(band[0]))
            hi = DoubleSpinBox(); hi.setRange(1, 96000); hi.setDecimals(0); hi.setValue(float(band[1]))
            lim = DoubleSpinBox(); lim.setRange(0, 60); lim.setDecimals(1); lim.setValue(float(band[2]))
            row.addWidget(lo); row.addWidget(hi); row.addWidget(lim)
            layout.addLayout(row)
            self.limit_rows.append((lo, hi, lim))

        group.setLayout(layout)
        return group

    # ------------------------------------------------------------------
    # 模式切换
    # ------------------------------------------------------------------
    def _current_mode(self) -> str:
        idx = self.mode_combo.currentIndex()
        return MODE_LABELS[idx][1] if 0 <= idx < len(MODE_LABELS) else "ecma"

    def _on_mode_changed(self, *_args):
        """固定比例带宽才需要比例输入；ECMA 临界带由公式确定，隐藏比例框。"""
        if hasattr(self, "ratio_container"):
            self.ratio_container.setVisible(self._current_mode() == "customer_15pct")

    # ------------------------------------------------------------------
    # 按钮 / 配置导出
    # ------------------------------------------------------------------
    def create_btn(self):
        btn_layout = QHBoxLayout()
        default_btn = PushButton(" 设为默认 ")
        default_btn.clicked.connect(self.on_default_btn_clicked)
        ok_btn = PushButton(" 确  认 ")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(default_btn)
        btn_layout.addSpacing(10)
        btn_layout.addWidget(ok_btn)
        return btn_layout

    def _collect_fan_pr_limits(self):
        limits = []
        for lo, hi, lim in self.limit_rows:
            limits.append([int(lo.value()), int(hi.value()), float(lim.value())])
        return limits

    def _validate(self) -> bool:
        if float(self.f_max_spin.value()) <= float(self.f_min_spin.value()):
            MessageBox.warning(self, "提示", "频率上限必须大于下限。")
            return False
        for lo, hi, _lim in self.limit_rows:
            if float(hi.value()) <= float(lo.value()):
                MessageBox.warning(self, "提示", "PR 限值频段的结束频率必须大于起始频率。")
                return False
        return True

    def get_default_config(self):
        config = dict(default_pr_config())
        config.update({
            "critical_band_mode": self._current_mode(),
            "customer_band_ratio": float(self.band_ratio_spin.value()) / 100.0,
            "f_min": int(self.f_min_spin.value()),
            "f_max": int(self.f_max_spin.value()),
            "window": self.window_combo.currentText(),
            "window_samples": int(self.window_samples_combo.currentData() or 65536),
            "overlap_ratio": float(self.overlap_spin.value()),
            "target_resolution_hz": float(self.load_config.get("target_resolution_hz", 1.0)),
            "highpass_hz": float(self.highpass_spin.value()) if self.highpass_check.isChecked() else 0.0,
            "limit_checked": bool(self.limit_check.isChecked()),
            "fan_pr_limits": self._collect_fan_pr_limits(),
        })
        # 计权为隐藏高级项：界面不暴露，默认 auto→线性(Z)；如配置里显式写了 A/C 则保留。
        config["weighting"] = str(self.load_config.get("weighting", "auto"))
        if self.show_channel_selector and hasattr(self, "channel_combo_box"):
            config["analysis_channel"] = int(self.channel_combo_box.currentData())
        else:
            config["analysis_channel"] = int(self.load_config.get("analysis_channel", 0) or 0)
        return config

    def on_default_btn_clicked(self):
        if not self._validate():
            return
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config(self.model_type_str, config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        if not self._validate():
            return
        self.accept()
        return self.get_default_config()

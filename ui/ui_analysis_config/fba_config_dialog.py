import re
from typing import List, Optional
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QFileDialog, QDialog, QHBoxLayout, QScrollArea, QVBoxLayout, QSizePolicy, QWidget

from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils

# 确保这里导入的是你项目中正确的 ThresholdConfigWidget 路径
from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget
from ui.custom_ui_widget.widgets import PushButton, ComboBox, Label, GroupBox, DoubleSpinBox, PlainTextEdit, MessageBox, LineEdit, CheckBox
from ui.ui_src import ui_resources


class FbaConfigWindow(QDialog):
    """
    FBA 频段能量分析配置窗口 (修复布局版)
    """

    STRATEGY_LABELS = [
        "1/1 倍频程",
        "1/3 倍频程",
        "1/6 倍频程",
        "1/12 倍频程",
        "Bark",
        "等宽",
        "自定义",
    ]
    BASELINE_DISPLAY_MODES = {
        "overlay": "叠加显示",
        "delta": "差值显示",
    }

    def __init__(self, config_manager, model_type, available_channels: Optional[List[int]] = None):
        super().__init__()
        self.config_manager = config_manager
        # 提取模型类型中的字母部分，例如 "FBA"
        self.model_type_str = "".join(re.findall(r"[A-Za-z]", str(model_type))) or "FBA"
        self.show_channel_selector = available_channels is not None
        self.available_channels = self._normalize_available_channels(available_channels)

        # 加载配置
        full_config = self.config_manager.load_config()
        self.load_config = full_config.get(model_type, {})

        self.init_ui()

    @staticmethod
    def _normalize_available_channels(available_channels):
        channels = []
        try:
            channels = sorted({int(ch) for ch in (available_channels or [])})
        except Exception:
            channels = []
        if not channels:
            channels = [0]
        return channels

    def _create_channel_layout(self):
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(Label("通道:"))
        self.channel_combo_box = ComboBox()
        for ch in self.available_channels:
            self.channel_combo_box.addItem(f"In{int(ch) + 1}", int(ch))
        saved_channel = self.load_config.get("analysis_channel", None)
        if saved_channel is None or int(saved_channel) not in self.available_channels:
            saved_channel = int(self.available_channels[0])
        idx = self.channel_combo_box.findData(int(saved_channel))
        self.channel_combo_box.setCurrentIndex(idx if idx >= 0 else 0)
        channel_layout.addWidget(self.channel_combo_box, 1)
        return channel_layout

    def init_ui(self):
        # --- 窗口基本设置 ---
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        self.setWindowTitle("FBA 分析配置")
        # 该对话框主要是纵向排布；过大的默认宽度会让界面“显得很宽”
        # 同时避免默认高度过大把“阈值展示区域”撑得太高
        self.setMinimumSize(420, 620)
        self.resize(420, 620)

        # --- 主布局：垂直流式布局 ---
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(15)
        content_layout.setContentsMargins(0, 0, 0, 0)

        if self.show_channel_selector:
            content_layout.addLayout(self._create_channel_layout())

        # =========================================================
        # 1. 频段划分策略 (GroupBox)
        # =========================================================
        strategy_group = GroupBox("频段划分策略")
        strategy_layout = QVBoxLayout()

        # 1.1 策略选择下拉框
        row_strat = QHBoxLayout()
        row_strat.addWidget(Label("划分策略:"))
        self.strategy_combo = ComboBox()
        self.strategy_combo.addItems(self.STRATEGY_LABELS)
        self._init_strategy_combo()
        self.strategy_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row_strat.addWidget(self.strategy_combo, 1)  # 1代表拉伸因子，让下拉框尽可能宽
        strategy_layout.addLayout(row_strat)

        # 1.2 等宽模式参数 (默认隐藏，动态显示)
        self.bandwidth_widget = QWidget()
        bw_layout = QHBoxLayout(self.bandwidth_widget)
        bw_layout.setContentsMargins(0, 0, 0, 0)
        bw_layout.addWidget(Label("频段宽度:"))
        self.bandwidth_spin = DoubleSpinBox()
        self.bandwidth_spin.setRange(10, 5000)
        self.bandwidth_spin.setDecimals(0)
        self.bandwidth_spin.setValue(self.load_config.get("bandwidth", 100))
        self.bandwidth_spin.setSuffix(" Hz")
        bw_layout.addWidget(self.bandwidth_spin, 1)
        strategy_layout.addWidget(self.bandwidth_widget)

        # 1.3 自定义模式参数 (默认隐藏，动态显示)
        self.custom_widget = QWidget()
        cust_layout = QVBoxLayout(self.custom_widget)
        cust_layout.setContentsMargins(0, 0, 0, 0)
        cust_help = Label("格式: f_low, f_high [, label] (每行一条)")
        cust_help.setObjectName("custhelp")
        cust_help.set_font_size(12)
        cust_layout.addWidget(cust_help)
        self.custom_bands_edit = PlainTextEdit()
        self.custom_bands_edit.setPlaceholderText("示例:\n20, 200, Low\n200, 1000, Mid")
        self.custom_bands_edit.setPlainText(self.load_config.get("custom_bands_text", ""))
        self.custom_bands_edit.setMaximumHeight(80)  # 限制高度，防止占用过多空间
        cust_layout.addWidget(self.custom_bands_edit)
        strategy_layout.addWidget(self.custom_widget)

        strategy_group.setLayout(strategy_layout)
        content_layout.addWidget(strategy_group)

        # =========================================================
        # 2. 频率范围 (GroupBox)
        # =========================================================
        range_group = GroupBox("分析频率范围")
        range_layout = QHBoxLayout()

        range_layout.addWidget(Label("最低:"))
        self.f_min_spin = DoubleSpinBox()
        self.f_min_spin.setRange(1, 20000)
        self.f_min_spin.setDecimals(0)
        self.f_min_spin.setValue(self.load_config.get("f_min", 20))
        self.f_min_spin.setSuffix(" Hz")
        range_layout.addWidget(self.f_min_spin)

        range_layout.addSpacing(20)  # 中间加点间距

        range_layout.addWidget(Label("最高:"))
        self.f_max_spin = DoubleSpinBox()
        self.f_max_spin.setRange(100, 48000)
        self.f_max_spin.setDecimals(0)
        self.f_max_spin.setValue(self.load_config.get("f_max", 20000))
        self.f_max_spin.setSuffix(" Hz")
        range_layout.addWidget(self.f_max_spin)

        range_group.setLayout(range_layout)
        content_layout.addWidget(range_group)

        # =========================================================
        # 3. 计权方式 (水平布局) - 关键修复点
        # =========================================================
        weight_layout = QHBoxLayout()

        weight_label = Label("计权方式:")
        weight_layout.addWidget(weight_label)

        self.weighting_combo = ComboBox()
        self.weighting_combo.addItems(["Z（None）", "A", "C"])
        self._init_weighting_combo()
        # 避免下拉框“参与抢占”横向空间导致对话框默认宽度变大
        self.weighting_combo.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        weight_layout.addWidget(self.weighting_combo)

        # 加一个弹簧，让它靠左显示，不要拉得太长
        weight_layout.addStretch()

        # 将这一行加入主布局
        content_layout.addLayout(weight_layout)

        baseline_group = GroupBox("背景噪声基线")
        baseline_group.setLayout(self._create_baseline_layout())
        content_layout.addWidget(baseline_group)

        dominant_group = GroupBox("主音识别")
        dominant_group.setLayout(self._create_dominant_tone_layout())
        content_layout.addWidget(dominant_group)

        # =========================================================
        # 4. 阈值配置组件 (ThresholdConfigWidget)
        # =========================================================
        # 传递 model_type，让它自己决定坐标轴标签
        self.threshold_widget = ThresholdConfigWidget(
            parent=self, load_config=self.load_config, model_type=self.model_type_str
        )
        # 不让阈值组件吃掉所有剩余高度（否则内部 addStretch 会形成很大空白）
        self.threshold_widget.setMaximumHeight(360)
        self.threshold_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        content_layout.addWidget(self.threshold_widget)

        # =========================================================
        # 5. 底部按钮
        # =========================================================
        content_layout.addStretch(1)

        scroll_area = QScrollArea()
        scroll_area.setObjectName("fba_config_scroll_area")
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area, 1)
        main_layout.addLayout(self.create_btn())

        self.setLayout(main_layout)

        # --- 样式设置 ---

        # --- 信号连接 ---
        self.strategy_combo.currentTextChanged.connect(self._on_strategy_changed)

        # --- 初始化界面状态 ---
        self._on_strategy_changed(self.strategy_combo.currentText())

    def _create_baseline_layout(self):
        layout = QVBoxLayout()

        file_layout = QHBoxLayout()
        file_layout.addWidget(Label("背景音频:"))
        self.baseline_path_edit = LineEdit()
        self.baseline_path_edit.setReadOnly(True)
        self.baseline_path_edit.setText(str(self.load_config.get("baseline_file_path", "") or ""))
        icon = QIcon(":/ui/icon/folder-s.png")
        action = self.baseline_path_edit.addAction(icon, LineEdit.TrailingPosition)
        action.setToolTip("选择背景噪声音频")
        action.triggered.connect(self._on_baseline_file_clicked)
        file_layout.addWidget(self.baseline_path_edit, 1)
        layout.addLayout(file_layout)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(Label("显示方式:"))
        self.baseline_mode_combo = ComboBox()
        for value, label in self.BASELINE_DISPLAY_MODES.items():
            self.baseline_mode_combo.addItem(label, value)
        saved_mode = str(self.load_config.get("baseline_display_mode", "overlay"))
        idx = self.baseline_mode_combo.findData(saved_mode)
        self.baseline_mode_combo.setCurrentIndex(idx if idx >= 0 else 0)
        mode_layout.addWidget(self.baseline_mode_combo, 1)
        layout.addLayout(mode_layout)
        return layout

    def _on_baseline_file_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择背景噪声音频",
            DEFAULT_DIR,
            filter="音频文件 (*.wav *.flac *.mp3);;所有文件 (*.*)",
        )
        if file_path:
            self.baseline_path_edit.setText(file_path)

    def _create_dominant_tone_layout(self):
        layout = QVBoxLayout()

        self.dominant_tone_checkbox = CheckBox("启用主音识别")
        self.dominant_tone_checkbox.setChecked(bool(self.load_config.get("dominant_tone_enabled", False)))
        layout.addWidget(self.dominant_tone_checkbox)

        prominence_layout = QHBoxLayout()
        prominence_layout.addWidget(Label("最小 prominence:"))
        self.dominant_prominence_spin = DoubleSpinBox()
        self.dominant_prominence_spin.setRange(0, 100)
        self.dominant_prominence_spin.setDecimals(1)
        self.dominant_prominence_spin.setSuffix(" dB")
        self.dominant_prominence_spin.setValue(float(self.load_config.get("dominant_tone_min_prominence_db", 3.0)))
        prominence_layout.addWidget(self.dominant_prominence_spin, 1)
        layout.addLayout(prominence_layout)

        self.dominant_use_display_curve_checkbox = CheckBox("使用当前显示曲线识别")
        self.dominant_use_display_curve_checkbox.setChecked(
            bool(self.load_config.get("dominant_tone_use_display_curve", True))
        )
        layout.addWidget(self.dominant_use_display_curve_checkbox)

        layout.addWidget(Label("频率区间:"))
        self.dominant_intervals_edit = PlainTextEdit()
        self.dominant_intervals_edit.setPlaceholderText("示例:\n100, 500, Low\n500, 2000, Mid")
        self.dominant_intervals_edit.setPlainText(str(self.load_config.get("dominant_tone_intervals_text", "") or ""))
        self.dominant_intervals_edit.setMaximumHeight(80)
        layout.addWidget(self.dominant_intervals_edit)
        return layout

    def _init_strategy_combo(self):
        val = self.load_config.get("band_strategy", "1/3 倍频程")
        idx = self.strategy_combo.findText(val)
        self.strategy_combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _init_weighting_combo(self):
        val = self.load_config.get("weighting", "A")
        # 兼容旧配置中的 None
        if val in ("None", "Z"):
            val = "Z（None）"
        idx = self.weighting_combo.findText(val)
        # 默认选 A (索引通常是 1，取决于 addItems 的顺序)
        self.weighting_combo.setCurrentIndex(idx if idx >= 0 else 1)

    def _on_strategy_changed(self, text):
        """根据选择的策略，显隐对应的参数控件"""
        is_equal = text == "等宽"
        is_custom = text == "自定义"

        self.bandwidth_widget.setVisible(is_equal)
        self.custom_widget.setVisible(is_custom)

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

    # -----------------------------------------------------------
    # 以下逻辑方法保持原有不变
    # -----------------------------------------------------------

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
            try:
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
            except Exception:
                raise ValueError(f"格式错误: {raw!r}")

        edges.sort(key=lambda x: x[0])
        for i in range(1, len(edges)):
            if edges[i][0] < edges[i - 1][1]:
                raise ValueError("自定义频段不允许重叠，请检查相邻频段边界。")
        return edges

    def _validate_custom_bands_if_needed(self) -> bool:
        if self.strategy_combo.currentText() != "自定义":
            return True
        try:
            edges = self._parse_custom_bands_text(self.custom_bands_edit.toPlainText())
            if not edges:
                raise ValueError("请至少输入一个频段。")
            return True
        except Exception as e:
            MessageBox.warning(self, "提示", f"自定义频段格式错误：{str(e)[:200]}")
            return False

    def get_default_config(self):
        config = {
            "band_strategy": self.strategy_combo.currentText(),
            "f_min": int(self.f_min_spin.value()),
            "f_max": int(self.f_max_spin.value()),
            "bandwidth": int(self.bandwidth_spin.value()),
            "analysis_channel": int(self.channel_combo_box.currentData())
            if self.show_channel_selector and hasattr(self, "channel_combo_box")
            else int(self.load_config.get("analysis_channel", 0) or 0),
            "baseline_file_path": self.baseline_path_edit.text().strip(),
            "baseline_display_mode": self.baseline_mode_combo.currentData(),
            "dominant_tone_enabled": self.dominant_tone_checkbox.isChecked(),
            "dominant_tone_intervals_text": self.dominant_intervals_edit.toPlainText(),
            "dominant_tone_min_prominence_db": float(self.dominant_prominence_spin.value()),
            "dominant_tone_use_display_curve": self.dominant_use_display_curve_checkbox.isChecked(),
        }
        if self.strategy_combo.currentText() == "自定义":
            config["custom_bands_text"] = self.custom_bands_edit.toPlainText()

        weighting_value = self.weighting_combo.currentText()
        config["weighting"] = "Z" if weighting_value == "Z（None）" else weighting_value

        # 获取阈值组件的配置
        config.update(self.threshold_widget.get_config())
        return config

    def on_default_btn_clicked(self):
        if not self._validate_custom_bands_if_needed():
            return
        if not self.threshold_widget.validate():
            return

        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config(self.model_type_str, config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        if not self._validate_custom_bands_if_needed():
            return
        if not self.threshold_widget.validate():
            return

        self.accept()
        return self.get_default_config()

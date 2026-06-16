import re
from typing import List, Optional
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QSizePolicy, QWidget

from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils

# 确保这里导入的是你项目中正确的 ThresholdConfigWidget 路径
from ui.ui_analysis_config.common_widgets import (
    ChannelSelectorWidget,
    SemanticAnalysisConfigDialogBase,
    WeightingSelectorWidget,
)
from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget
from ui.custom_ui_widget.widgets import ComboBox, Label, GroupBox, DoubleSpinBox, PlainTextEdit, MessageBox
from ui.ui_src import ui_resources


class FbaConfigWindow(SemanticAnalysisConfigDialogBase):
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

    def __init__(self, config_manager, model_type, available_channels: Optional[List[int]] = None):
        super().__init__(disable_close_button=True)
        self.config_manager = config_manager
        self.config_key = model_type
        # 提取模型类型中的字母部分，例如 "FBA"
        self.model_type_str = "".join(re.findall(r"[A-Za-z]", str(model_type))) or "FBA"
        self.show_channel_selector = available_channels is not None
        self.available_channels = available_channels

        # 加载配置
        full_config = self.config_manager.load_config()
        self.load_config = full_config.get(self.config_key, {})

        self.init_ui()

    def init_ui(self):
        # --- 窗口基本设置 ---
        self.setWindowTitle("FBA 分析配置")
        self.apply_semantic_dialog_size()
        self.set_semantic_button_callbacks(
            default_callback=self.on_default_btn_clicked,
            restore_callback=self.on_restore_default_btn_clicked,
            ok_callback=self.on_click_ok_btn,
        )
        self._build_semantic_sections()

    def _build_semantic_sections(self):
        if self.show_channel_selector:
            self.channel_selector = ChannelSelectorWidget(self.load_config, self.available_channels, self)
            self.add_semantic_section("input", widget=self.channel_selector)

        compute_widget = QWidget(self)
        compute_layout = QVBoxLayout(compute_widget)
        compute_layout.setContentsMargins(0, 0, 0, 0)
        compute_layout.setSpacing(12)

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
        compute_layout.addWidget(strategy_group)

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
        compute_layout.addWidget(range_group)

        self.weighting_selector = WeightingSelectorWidget(
            self.load_config,
            allowed_options=("Z", "A", "C"),
            default="A",
            parent=self,
        )
        compute_layout.addWidget(self.weighting_selector)

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
        self.add_semantic_section("compute", widget=compute_widget)
        self.add_semantic_section("judgment", widget=self.threshold_widget)

        # --- 信号连接 ---
        self.strategy_combo.currentTextChanged.connect(self._on_strategy_changed)

        # --- 初始化界面状态 ---
        self._on_strategy_changed(self.strategy_combo.currentText())

    def _init_strategy_combo(self):
        val = self.load_config.get("band_strategy", "1/3 倍频程")
        idx = self.strategy_combo.findText(val)
        self.strategy_combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _on_strategy_changed(self, text):
        """根据选择的策略，显隐对应的参数控件"""
        is_equal = text == "等宽"
        is_custom = text == "自定义"

        self.bandwidth_widget.setVisible(is_equal)
        self.custom_widget.setVisible(is_custom)

    def create_btn(self):
        return self.create_standard_button_layout(self.on_default_btn_clicked, self.on_click_ok_btn)

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
            "analysis_channel": self.channel_selector.current_channel()
            if self.show_channel_selector and hasattr(self, "channel_selector")
            else int(self.load_config.get("analysis_channel", 0) or 0),
        }
        if self.strategy_combo.currentText() == "自定义":
            config["custom_bands_text"] = self.custom_bands_edit.toPlainText()

        config.update(self.weighting_selector.get_config())

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

    def on_restore_default_btn_clicked(self):
        self.load_config = self.config_manager.load_config().get(self.config_key, {})
        self.clear_semantic_sections()
        self._build_semantic_sections()

    def on_click_ok_btn(self):
        if not self._validate_custom_bands_if_needed():
            return
        if not self.threshold_widget.validate():
            return

        self.accept()
        return self.get_default_config()

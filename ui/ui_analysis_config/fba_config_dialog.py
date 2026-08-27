import re
from typing import List, Optional

from PyQt5.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import (
    ComboBox,
    DoubleSpinBox,
    GroupBox,
    Label,
    MessageBox,
    PlainTextEdit,
)
from ui.ui_analysis_config.common_widgets import (
    AnalysisChannelSpinBoxWidget,
    SemanticAnalysisConfigDialogBase,
    WeightingSelectorWidget,
)
from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget


class FbaConfigWindow(SemanticAnalysisConfigDialogBase):
    """FBA 频段能量分析配置窗口。"""

    DEFAULT_CONFIG = {
        "band_strategy": "1/3 倍频程",
        "f_min": 20,
        "f_max": 20000,
        "bandwidth": 100,
        "weighting": "A",
        "analysis_channel": 0,
        "limit_checked": False,
        "limit_data": None,
        "limit_mode": "csv",
        "manual_input_mode": "segments",
        "constant_upper_enabled": True,
        "constant_lower_enabled": False,
        "constant_upper_value": 100.0,
        "constant_lower_value": 0.0,
        "manual_upper_enabled": True,
        "manual_lower_enabled": False,
        "manual_upper_segments": [],
        "manual_lower_segments": [],
    }

    STRATEGY_LABELS = [
        "1/1 倍频程",
        "1/3 倍频程",
        "1/6 倍频程",
        "1/12 倍频程",
        "Bark",
        "等宽",
        "自定义",
    ]

    def __init__(
        self,
        config_manager,
        model_type,
        available_channels: Optional[List[int]] = None,
        restrict_analysis_channel: bool = False,
    ):
        super().__init__(disable_close_button=True)
        self.config_manager = config_manager
        self.config_key = model_type
        self.model_type_str = "".join(re.findall(r"[A-Za-z]", str(model_type))) or "FBA"
        self.show_channel_selector = available_channels is not None
        self.available_channels = available_channels
        self.restrict_analysis_channel = restrict_analysis_channel

        full_config = self.config_manager.load_config()
        self.load_config = dict(self.DEFAULT_CONFIG)
        self.load_config.update(full_config.get(self.config_key, {}))
        self.init_ui()

    def init_ui(self):
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
            self.channel_selector = AnalysisChannelSpinBoxWidget(
                self.load_config,
                self.available_channels,
                self,
                restrict_to_available_channels=(
                    self.restrict_analysis_channel
                ),
            )
            self.add_semantic_section("input", widget=self.channel_selector)

        compute_widget = QWidget(self)
        compute_layout = QVBoxLayout(compute_widget)
        compute_layout.setContentsMargins(0, 0, 0, 0)
        compute_layout.setSpacing(12)

        strategy_group = GroupBox("频段划分策略")
        strategy_layout = QVBoxLayout()

        strategy_row = QHBoxLayout()
        strategy_row.addWidget(Label("划分策略:"))
        self.strategy_combo = ComboBox()
        self.strategy_combo.addItems(self.STRATEGY_LABELS)
        self._init_strategy_combo()
        self.strategy_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        strategy_row.addWidget(self.strategy_combo, 1)
        strategy_layout.addLayout(strategy_row)

        self.bandwidth_widget = QWidget()
        bandwidth_layout = QHBoxLayout(self.bandwidth_widget)
        bandwidth_layout.setContentsMargins(0, 0, 0, 0)
        bandwidth_layout.addWidget(Label("频段宽度:"))
        self.bandwidth_spin = DoubleSpinBox()
        self.bandwidth_spin.setRange(10, 5000)
        self.bandwidth_spin.setDecimals(0)
        self.bandwidth_spin.setValue(self.load_config.get("bandwidth", 100))
        self.bandwidth_spin.setSuffix(" Hz")
        bandwidth_layout.addWidget(self.bandwidth_spin, 1)
        strategy_layout.addWidget(self.bandwidth_widget)

        self.custom_widget = QWidget()
        custom_layout = QVBoxLayout(self.custom_widget)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        custom_help = Label("格式: f_low, f_high [, label] (每行一条)")
        custom_help.setObjectName("custhelp")
        custom_help.set_font_size(12)
        custom_layout.addWidget(custom_help)
        self.custom_bands_edit = PlainTextEdit()
        self.custom_bands_edit.setPlaceholderText(
            "示例:\n20, 200, Low\n200, 1000, Mid"
        )
        self.custom_bands_edit.setPlainText(
            self.load_config.get("custom_bands_text", "")
        )
        self.custom_bands_edit.setMaximumHeight(80)
        custom_layout.addWidget(self.custom_bands_edit)
        strategy_layout.addWidget(self.custom_widget)

        strategy_group.setLayout(strategy_layout)
        compute_layout.addWidget(strategy_group)

        range_group = GroupBox("分析频率范围")
        range_layout = QHBoxLayout()
        range_layout.addWidget(Label("最低:"))
        self.f_min_spin = DoubleSpinBox()
        self.f_min_spin.setRange(1, 20000)
        self.f_min_spin.setDecimals(0)
        self.f_min_spin.setValue(self.load_config.get("f_min", 20))
        self.f_min_spin.setSuffix(" Hz")
        range_layout.addWidget(self.f_min_spin)
        range_layout.addSpacing(20)
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

        self.threshold_widget = ThresholdConfigWidget(
            parent=self,
            load_config=self.load_config,
            model_type=self.model_type_str,
            allow_manual_limits=True,
            allow_constant_limits=True,
            allow_csv_limit_offsets=True,
        )
        self.threshold_widget.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )

        self.add_semantic_section("compute", widget=compute_widget)
        self.enable_plot_view_config(
            self.load_config,
            "",
            "dB",
            False,
            True,
            False,
        )
        self.add_threshold_curve_sections(
            self.threshold_widget,
            self.load_config,
        )

        self.strategy_combo.currentTextChanged.connect(
            self._on_strategy_changed
        )
        self._on_strategy_changed(self.strategy_combo.currentText())

    def _init_strategy_combo(self):
        value = self.load_config.get("band_strategy", "1/3 倍频程")
        index = self.strategy_combo.findText(value)
        self.strategy_combo.setCurrentIndex(index if index >= 0 else 0)

    def _on_strategy_changed(self, text):
        self.bandwidth_widget.setVisible(text == "等宽")
        self.custom_widget.setVisible(text == "自定义")
        self.bandwidth_widget.updateGeometry()
        self.custom_widget.updateGeometry()
        self._refresh_section_container_minimum_height()

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

            try:
                label = None
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
        for index in range(1, len(edges)):
            if edges[index][0] < edges[index - 1][1]:
                raise ValueError("自定义频段不允许重叠，请检查相邻频段边界。")
        return edges

    def _validate_form(self) -> bool:
        if self.f_min_spin.value() >= self.f_max_spin.value():
            MessageBox.warning(self, "提示", "最高频率必须大于最低频率。")
            return False
        if self.strategy_combo.currentText() == "自定义":
            try:
                edges = self._parse_custom_bands_text(
                    self.custom_bands_edit.toPlainText()
                )
                if not edges:
                    raise ValueError("请至少输入一个频段。")
            except ValueError as exc:
                MessageBox.warning(
                    self,
                    "提示",
                    f"自定义频段格式错误：{str(exc)[:200]}",
                )
                return False
        if not self.validate_plot_view_config():
            return False
        return self.threshold_widget.validate()

    def get_default_config(self):
        config = {
            "band_strategy": self.strategy_combo.currentText(),
            "f_min": int(self.f_min_spin.value()),
            "f_max": int(self.f_max_spin.value()),
            "bandwidth": int(self.bandwidth_spin.value()),
            "analysis_channel": (
                self.channel_selector.current_channel()
                if self.show_channel_selector
                and hasattr(self, "channel_selector")
                else int(self.load_config.get("analysis_channel", 0) or 0)
            ),
        }
        if self.strategy_combo.currentText() == "自定义":
            config["custom_bands_text"] = (
                self.custom_bands_edit.toPlainText()
            )
        config.update(self.weighting_selector.get_config())
        config.update(self.threshold_widget.get_config())
        return self.merge_plot_view_config(config)

    def on_default_btn_clicked(self):
        if not self._validate_form():
            return
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config(
            self.model_type_str,
            config_data,
        )
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_restore_default_btn_clicked(self):
        self.load_config = dict(self.DEFAULT_CONFIG)
        self.load_config.update(
            self.config_manager.load_config().get(self.config_key, {})
        )
        self.clear_semantic_sections()
        self._build_semantic_sections()

    def on_click_ok_btn(self):
        if not self._validate_form():
            return
        self.accept()
        return self.get_default_config()

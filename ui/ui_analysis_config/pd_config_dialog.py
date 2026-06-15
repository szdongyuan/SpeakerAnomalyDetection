from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout

from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import (
    GroupBox,
    Label,
    PushButton,
    ComboBox,
    LineEdit,
    SpinBox,
    DoubleSpinBox,
    CheckBox,
)
from ui.ui_analysis_config.common_widgets import AnalysisConfigDialogBase, TimeSmoothingWidget
from ui.ui_src import ui_resources


class PDConfigWindow(AnalysisConfigDialogBase):
    def __init__(self, config_manager, model_type):
        super().__init__(disable_close_button=True)
        self.config_manager = config_manager
        self.load_config = self.config_manager.load_config().get(model_type, {})

        self.init_ui()

    def init_ui(self):
        self.setMinimumSize(940, 520)
        self.resize(1000, 560)

        root_layout = QHBoxLayout()

        # left layout
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.create_detect_group())
        left_layout.addWidget(self.create_test_group())
        left_layout.addStretch()

        # advanced mode: always hidden when entering PD config
        self.advanced_visible = False
        self.btn_toggle_advanced = PushButton("高级模式 >>>")
        self.btn_toggle_advanced.clicked.connect(self.on_toggle_advanced_mode)
        left_layout.addWidget(self.btn_toggle_advanced)

        left_layout.addLayout(self.create_btn_layout())
        left_layout.setSpacing(10)

        # right layout
        self.advanced_panel = self.create_advanced_group()
        # set the minimum width of the advanced panel to be larger, to avoid being compressed after opening
        try:
            self.advanced_panel.setMinimumWidth(360)
        except Exception:
            pass
        self.advanced_panel.setVisible(self.advanced_visible)

        root_layout.addLayout(left_layout)
        root_layout.addWidget(self.advanced_panel)
        root_layout.setSpacing(12)
        self.setLayout(root_layout)

        # adapt the size according to the visibility of the panel
        self.adjustSize()

    def create_preprocess_group(self):
        group_box = GroupBox("预处理选项")
        vbox = QVBoxLayout()

        # filter (two rows: main parameters + order)
        row_filter_main = QHBoxLayout()
        self.chk_filter = CheckBox("滤波")
        self.chk_filter.setChecked(self.load_config.get("filter_enabled", False))
        self.chk_filter.stateChanged.connect(self.get_default_config)
        row_filter_main.addWidget(self.chk_filter)
        row_filter_main.addStretch()
        row_filter_main.addWidget(Label("范围(Hz):"))
        self.edit_filter_ranges = LineEdit()
        self.edit_filter_ranges.setPlaceholderText("0,300; 700,1000;")
        self.edit_filter_ranges.setText(self.load_config.get("filter_ranges", ""))
        self.edit_filter_ranges.textChanged.connect(lambda _: self.get_default_config())
        row_filter_main.addWidget(self.edit_filter_ranges)
        row_filter_main.addWidget(Label("类型:"))
        self.combo_filter_type = ComboBox()
        self.combo_filter_type.addItems(["带通", "带阻"])
        self.combo_filter_type.setCurrentIndex(
            0 if self.load_config.get("filter_type", "bandpass") == "bandpass" else 1
        )
        self.combo_filter_type.currentIndexChanged.connect(lambda _: self.get_default_config())
        row_filter_main.addWidget(self.combo_filter_type)

        row_filter_order = QHBoxLayout()
        row_filter_order.addStretch()
        row_filter_order.addWidget(Label("阶数"))
        self.spin_filter_order = SpinBox()
        self.spin_filter_order.setRange(1, 20)
        self.spin_filter_order.setValue(int(self.load_config.get("filter_order", 4)))
        self.spin_filter_order.valueChanged.connect(lambda _: self.get_default_config())
        row_filter_order.addWidget(self.spin_filter_order)

        self.smoothing_widget = TimeSmoothingWidget(
            self.load_config,
            defaults={"enabled": False, "unit": "time", "time_sec": 0.02, "points": 0, "algo": 1},
            min_points=0,
            parent=self,
        )
        self.smoothing_widget.enabled_checkbox.stateChanged.connect(self.get_default_config)
        self.smoothing_widget.unit_combo.currentIndexChanged.connect(lambda _: self.get_default_config())
        self.smoothing_widget.time_spin.valueChanged.connect(lambda _: self.get_default_config())
        self.smoothing_widget.points_spin.valueChanged.connect(lambda _: self.get_default_config())
        self.smoothing_widget.algo_group.buttonClicked.connect(lambda _: self.get_default_config())

        # SPL calculation window length (no check box, default enabled; support time/grid point number)
        row_splwin = QHBoxLayout()
        row_splwin.addWidget(Label("SPL计算窗长"))
        row_splwin.addStretch()
        row_splwin.addWidget(Label("单位:"))
        self.combo_spl_window_unit = ComboBox()
        self.combo_spl_window_unit.addItems(["时间(秒)", "格点数"])
        self.combo_spl_window_unit.setCurrentIndex(
            0 if self.load_config.get("spl_window_unit", "time") == "time" else 1
        )
        self.combo_spl_window_unit.currentIndexChanged.connect(
            lambda _: (self._update_spl_window_unit_enabled(), self.get_default_config())
        )
        row_splwin.addWidget(self.combo_spl_window_unit)
        self.spin_spl_window_time = DoubleSpinBox()
        self.spin_spl_window_time.setRange(0.000, 999.000)
        self.spin_spl_window_time.setDecimals(4)
        self.spin_spl_window_time.setSingleStep(0.001)
        self.spin_spl_window_time.setValue(float(self.load_config.get("spl_window_time_sec", 0.050)))
        self.spin_spl_window_time.valueChanged.connect(lambda _: self.get_default_config())
        row_splwin.addWidget(self.spin_spl_window_time)
        self.spin_spl_window_points = SpinBox()
        self.spin_spl_window_points.setRange(1, 99999)
        self.spin_spl_window_points.setValue(int(self.load_config.get("spl_window_points", 0)))
        self.spin_spl_window_points.valueChanged.connect(lambda _: self.get_default_config())
        row_splwin.addWidget(self.spin_spl_window_points)

        vbox.addLayout(row_filter_main)
        vbox.addLayout(row_filter_order)
        # place the SPL calculation window length between the filter and smooth
        vbox.addLayout(row_splwin)
        vbox.addWidget(self.smoothing_widget)
        vbox.setSpacing(8)
        vbox.setContentsMargins(10, 12, 10, 12)
        group_box.setLayout(vbox)
        # initialize the display state
        self._update_spl_window_unit_enabled()
        return group_box

    def create_detect_group(self):
        group_box = GroupBox("峰值提取参数")
        vbox = QVBoxLayout()

        # peak count
        row_count = QHBoxLayout()
        self.chk_peak_count = CheckBox("峰值个数")
        self.chk_peak_count.setChecked(self.load_config.get("peak_count_enabled", True))
        self.chk_peak_count.stateChanged.connect(self.get_default_config)
        row_count.addWidget(self.chk_peak_count)
        row_count.addStretch()
        row_count.addWidget(Label("最大峰数目:"))
        self.spin_peak_count = SpinBox()
        self.spin_peak_count.setRange(1, 9999)
        self.spin_peak_count.setValue(int(self.load_config.get("peak_count", 3)))
        self.spin_peak_count.valueChanged.connect(lambda _: self.get_default_config())
        row_count.addWidget(self.spin_peak_count)
        # row_count.addWidget(Label("个"))

        # peak size
        row_size = QHBoxLayout()
        self.chk_peak_size = CheckBox("峰值大小")
        self.chk_peak_size.setChecked(self.load_config.get("peak_size_enabled", True))
        self.chk_peak_size.stateChanged.connect(self.get_default_config)
        row_size.addWidget(self.chk_peak_size)
        row_size.addStretch()
        row_size.addWidget(Label("单位:"))
        self.combo_peak_size_unit = ComboBox()
        self.combo_peak_size_unit.addItems(["rmsV", "dBL"])
        peak_size_unit_saved = self.load_config.get("peak_size_unit", "db")
        self.combo_peak_size_unit.setCurrentIndex(0 if peak_size_unit_saved == "rms" else 1)
        self.combo_peak_size_unit.currentIndexChanged.connect(
            lambda _: (self._update_peak_units(), self.get_default_config())
        )
        row_size.addWidget(self.combo_peak_size_unit)
        self.spin_peak_size = DoubleSpinBox()
        self.spin_peak_size.setRange(-100.0, 200.0)
        self.spin_peak_size.setDecimals(2)
        self.spin_peak_size.setSingleStep(1.0)
        self.spin_peak_size.setValue(float(self.load_config.get("peak_min_value", 100.0)))
        self.spin_peak_size.valueChanged.connect(lambda _: self.get_default_config())
        row_size.addWidget(self.spin_peak_size)

        # peak slope
        row_slope = QHBoxLayout()
        self.chk_peak_slope = CheckBox("峰凸起度")
        self.chk_peak_slope.setChecked(self.load_config.get("peak_slope_enabled", False))
        self.chk_peak_slope.stateChanged.connect(self.get_default_config)
        row_slope.addWidget(self.chk_peak_slope)
        row_slope.addStretch()
        row_slope.addWidget(Label("单位:"))
        self.combo_peak_slope_unit = ComboBox()
        self.combo_peak_slope_unit.addItems(["rmsV", "dBL"])
        peak_slope_unit_saved = self.load_config.get("peak_slope_unit", "db")
        self.combo_peak_slope_unit.setCurrentIndex(0 if peak_slope_unit_saved == "rms" else 1)
        self.combo_peak_slope_unit.currentIndexChanged.connect(
            lambda _: (self._update_peak_units(), self.get_default_config())
        )
        row_slope.addWidget(self.combo_peak_slope_unit)
        self.spin_peak_slope = DoubleSpinBox()
        self.spin_peak_slope.setRange(0.0, 200.0)
        self.spin_peak_slope.setDecimals(3)
        self.spin_peak_slope.setSingleStep(1.0)
        self.spin_peak_slope.setValue(float(self.load_config.get("peak_min_slope", 100.0)))
        self.spin_peak_slope.valueChanged.connect(lambda _: self.get_default_config())
        row_slope.addWidget(self.spin_peak_slope)

        # minimum peak distance (support time/grid point number)
        row_nms = QHBoxLayout()
        self.chk_nms = CheckBox("最小峰间距")
        self.chk_nms.setChecked(self.load_config.get("nms_enabled", False))
        self.chk_nms.stateChanged.connect(self.get_default_config)
        row_nms.addWidget(self.chk_nms)
        row_nms.addStretch()
        row_nms.addWidget(Label("单位:"))
        self.combo_nms_unit = ComboBox()
        self.combo_nms_unit.addItems(["时间(秒)", "格点数"])
        self.combo_nms_unit.setCurrentIndex(0 if self.load_config.get("nms_unit", "time") == "time" else 1)
        self.combo_nms_unit.currentIndexChanged.connect(
            lambda _: (self._update_nms_unit_enabled(), self.get_default_config())
        )
        row_nms.addWidget(self.combo_nms_unit)
        self.spin_nms_time = DoubleSpinBox()
        self.spin_nms_time.setRange(0.00, 100.00)
        self.spin_nms_time.setDecimals(3)
        self.spin_nms_time.setSingleStep(0.01)
        self.spin_nms_time.setValue(float(self.load_config.get("nms_time_sec", 0.50)))
        self.spin_nms_time.valueChanged.connect(lambda _: self.get_default_config())
        row_nms.addWidget(self.spin_nms_time)
        self.spin_nms_points = SpinBox()
        self.spin_nms_points.setRange(1, 99999)
        self.spin_nms_points.setValue(int(self.load_config.get("nms_points", 0)))
        self.spin_nms_points.valueChanged.connect(lambda _: self.get_default_config())
        row_nms.addWidget(self.spin_nms_points)

        # duration
        row_duration = QHBoxLayout()
        self.chk_duration = CheckBox("峰持续时间")
        self.chk_duration.setChecked(self.load_config.get("duration_enabled", False))
        self.chk_duration.stateChanged.connect(self.get_default_config)
        row_duration.addWidget(self.chk_duration)
        row_duration.addStretch()
        row_duration.addWidget(Label("最短"))
        self.spin_duration_min = DoubleSpinBox()
        self.spin_duration_min.setRange(0.0, 1000.0)
        self.spin_duration_min.setDecimals(3)
        self.spin_duration_min.setSingleStep(0.001)
        self.spin_duration_min.setValue(float(self.load_config.get("duration_min", 0.0)))
        self.spin_duration_min.valueChanged.connect(lambda _: self.get_default_config())
        row_duration.addWidget(self.spin_duration_min)
        row_duration.addWidget(Label("最长"))
        self.spin_duration_max = DoubleSpinBox()
        self.spin_duration_max.setRange(0.0, 1000.0)
        self.spin_duration_max.setDecimals(3)
        self.spin_duration_max.setSingleStep(0.001)
        self.spin_duration_max.setValue(float(self.load_config.get("duration_max", 0.0)))
        self.spin_duration_max.valueChanged.connect(lambda _: self.get_default_config())
        row_duration.addWidget(self.spin_duration_max)

        vbox.addLayout(row_count)
        vbox.addLayout(row_size)
        vbox.addLayout(row_slope)
        vbox.addLayout(row_nms)
        vbox.addLayout(row_duration)
        vbox.setSpacing(8)
        vbox.setContentsMargins(10, 12, 10, 12)
        group_box.setLayout(vbox)
        # initialize the display state
        self._update_nms_unit_enabled()
        self._update_peak_units()
        return group_box

    def create_test_group(self):
        group_box = GroupBox("测试选项")
        vbox = QVBoxLayout()

        row_peak_condition = QHBoxLayout()
        row_peak_condition.addWidget(Label("峰值点数目"))
        row_peak_condition.addStretch()
        self.combo_test_peak_op = ComboBox()
        self.combo_test_peak_op.addItems([">", "<", "=", "≥", "≤"])
        saved_op = self.load_config.get("test_peak_op", "≥")
        try:
            idx = [">", "<", "=", "≥", "≤"].index(saved_op)
        except ValueError:
            idx = 3
        self.combo_test_peak_op.setCurrentIndex(idx)
        self.combo_test_peak_op.currentIndexChanged.connect(lambda _: self.get_default_config())
        row_peak_condition.addWidget(self.combo_test_peak_op)
        self.spin_test_peak_value = SpinBox()
        self.spin_test_peak_value.setRange(0, 1000000)
        self.spin_test_peak_value.setValue(int(self.load_config.get("test_peak_value", 3)))
        self.spin_test_peak_value.valueChanged.connect(lambda _: self.get_default_config())
        row_peak_condition.addWidget(self.spin_test_peak_value)

        vbox.addLayout(row_peak_condition)
        vbox.setSpacing(8)
        vbox.setContentsMargins(10, 12, 10, 12)
        group_box.setLayout(vbox)
        return group_box

    def create_advanced_group(self):
        adv_group = GroupBox("高级选项")
        adv_layout = QVBoxLayout()
        adv_layout.addWidget(self.create_preprocess_group())

        row_convex_len = QHBoxLayout()
        row_convex_len.addWidget(Label("峰凸起度计算窗口"))
        row_convex_len.addStretch(1)
        row_convex_len.addWidget(Label("单位:"))
        self.combo_convex_unit = ComboBox()
        self.combo_convex_unit.addItems(["音频长度", "格点数", "时长(秒)"])
        self.combo_convex_unit.setCurrentIndex(
            {"audio": 0, "points": 1, "time": 2}.get(self.load_config.get("convex_unit", "audio"), 0)
        )
        self.combo_convex_unit.currentIndexChanged.connect(
            lambda _: (self._update_convex_unit_enabled(), self.get_default_config())
        )
        row_convex_len.addWidget(self.combo_convex_unit)
        self.spin_convex_audio_ratio = DoubleSpinBox()
        self.spin_convex_audio_ratio.setRange(0.001, 1.000)
        self.spin_convex_audio_ratio.setDecimals(4)
        self.spin_convex_audio_ratio.setSingleStep(0.01)
        self.spin_convex_audio_ratio.setValue(float(self.load_config.get("convex_audio_ratio", 1.0)))
        self.spin_convex_audio_ratio.valueChanged.connect(lambda _: self.get_default_config())
        row_convex_len.addWidget(self.spin_convex_audio_ratio)
        self.spin_convex_points = SpinBox()
        self.spin_convex_points.setRange(1, 100000000)
        self.spin_convex_points.setValue(int(self.load_config.get("convex_points", 1024)))
        self.spin_convex_points.valueChanged.connect(lambda _: self.get_default_config())
        row_convex_len.addWidget(self.spin_convex_points)
        self.spin_convex_time = DoubleSpinBox()
        self.spin_convex_time.setRange(0.000, 999.000)
        self.spin_convex_time.setDecimals(3)
        self.spin_convex_time.setSingleStep(0.1)
        self.spin_convex_time.setValue(float(self.load_config.get("convex_time_sec", 0.0)))
        self.spin_convex_time.valueChanged.connect(lambda _: self.get_default_config())
        row_convex_len.addWidget(self.spin_convex_time)

        # 峰持续时间参考点（始终启用）
        row_dmode = QHBoxLayout()
        row_dmode.addWidget(Label("峰持续时间参考点"))
        row_dmode.addStretch(1)
        row_dmode.addWidget(Label("单位:"))

        ref_unit_saved = self.load_config.get("duration_ref_unit", "peak")
        ref_value_saved = float(self.load_config.get("duration_ref_value", 0.50 if ref_unit_saved == "peak" else 100.0))

        self.combo_duration_ref_unit = ComboBox()
        self.combo_duration_ref_unit.addItems(["Vpeak", "dBL"])
        self.combo_duration_ref_unit.setCurrentIndex(0 if ref_unit_saved == "peak" else 1)
        self.combo_duration_ref_unit.currentIndexChanged.connect(
            lambda _: (self._update_duration_ref_unit(), self.get_default_config())
        )
        row_dmode.addWidget(self.combo_duration_ref_unit)

        self.spin_duration_ref = DoubleSpinBox()
        self.spin_duration_ref.setDecimals(2)
        self.spin_duration_ref.setSingleStep(0.01)
        row_dmode.addWidget(self.spin_duration_ref)
        # set the range/step according to the unit, and fill the value
        self._update_duration_ref_unit()
        self.spin_duration_ref.setValue(float(ref_value_saved))
        self.spin_duration_ref.valueChanged.connect(lambda _: self.get_default_config())

        adv_layout.addLayout(row_convex_len)
        adv_layout.addLayout(row_dmode)
        adv_layout.addStretch()
        adv_layout.setSpacing(10)
        adv_layout.setContentsMargins(10, 20, 10, 20)
        adv_group.setLayout(adv_layout)
        adv_group.setMinimumWidth(260)

        self._update_convex_unit_enabled()
        return adv_group

    def on_toggle_advanced_mode(self):
        # switch only affects the display, not the configuration selection semantics
        self.advanced_visible = not getattr(self, "advanced_visible", False)
        self.advanced_panel.setVisible(self.advanced_visible)
        self.btn_toggle_advanced.setText("高级模式 <<<" if self.advanced_visible else "高级模式 >>>")
        if self.advanced_visible:
            self.setMinimumWidth(940)
        else:
            self.setMinimumWidth(820)
        self.adjustSize()

    def create_btn_layout(self):
        return self.create_standard_button_layout(self.on_click_default_btn, self.on_click_ok_btn)

    def get_default_config(self):
        default_config = {
            # preprocess(advanced)
            "filter_enabled": self.chk_filter.isChecked(),
            "filter_ranges": self.edit_filter_ranges.text().strip(),
            "filter_type": "bandpass" if self.combo_filter_type.currentIndex() == 0 else "bandstop",
            # based parameter
            "peak_count_enabled": self.chk_peak_count.isChecked(),
            "peak_count": int(self.spin_peak_count.value()),
            "peak_size_enabled": self.chk_peak_size.isChecked(),
            "peak_size_unit": ("rms" if self.combo_peak_size_unit.currentIndex() == 0 else "db"),
            "peak_min_value": float(self.spin_peak_size.value()),
            "peak_slope_enabled": self.chk_peak_slope.isChecked(),
            "peak_slope_unit": ("rms" if self.combo_peak_slope_unit.currentIndex() == 0 else "db"),
            "peak_min_slope": float(self.spin_peak_slope.value()),
            "nms_enabled": self.chk_nms.isChecked(),
            "nms_unit": "time" if self.combo_nms_unit.currentIndex() == 0 else "points",
            "nms_time_sec": float(self.spin_nms_time.value()),
            "nms_points": int(self.spin_nms_points.value()),
            "spl_window_unit": ("time" if self.combo_spl_window_unit.currentIndex() == 0 else "points"),
            "spl_window_time_sec": float(self.spin_spl_window_time.value()),
            "spl_window_points": int(self.spin_spl_window_points.value()),
            "duration_enabled": self.chk_duration.isChecked(),
            "duration_min": float(self.spin_duration_min.value()),
            "duration_max": float(self.spin_duration_max.value()),
            # advanced mode
            "advanced_mode": bool(getattr(self, "advanced_visible", False)),
            "filter_order": int(self.spin_filter_order.value()),
            "convex_unit": (
                "audio"
                if self.combo_convex_unit.currentIndex() == 0
                else ("points" if self.combo_convex_unit.currentIndex() == 1 else "time")
            ),
            "convex_audio_ratio": float(self.spin_convex_audio_ratio.value()),
            "convex_points": int(self.spin_convex_points.value()),
            "convex_time_sec": float(self.spin_convex_time.value()),
            "duration_ref_unit": ("peak" if self.combo_duration_ref_unit.currentIndex() == 0 else "db"),
            "duration_ref_value": float(self.spin_duration_ref.value()),
            # test option
            "test_peak_op": self.combo_test_peak_op.currentText(),
            "test_peak_value": int(self.spin_test_peak_value.value()),
        }
        default_config.update(self.smoothing_widget.get_config())
        return default_config

    def _update_duration_ref_unit(self):
        # adjust the range and precision according to the selected unit
        is_peak_unit = self.combo_duration_ref_unit.currentIndex() == 0
        if is_peak_unit:
            self.spin_duration_ref.setRange(0.00, 1.00)
            self.spin_duration_ref.setDecimals(2)
            self.spin_duration_ref.setSingleStep(0.01)
        else:
            self.spin_duration_ref.setRange(-200.0, 500.0)
            self.spin_duration_ref.setDecimals(1)
            self.spin_duration_ref.setSingleStep(1.0)

    def _update_smooth_unit_enabled(self):
        is_time = self.combo_smooth_unit.currentIndex() == 0
        self.spin_smooth_time.setVisible(is_time)
        self.spin_smooth_points.setVisible(not is_time)

    def _update_nms_unit_enabled(self):
        is_time = self.combo_nms_unit.currentIndex() == 0
        self.spin_nms_time.setVisible(is_time)
        self.spin_nms_points.setVisible(not is_time)

    def _update_spl_window_unit_enabled(self):
        is_time = self.combo_spl_window_unit.currentIndex() == 0
        self.spin_spl_window_time.setVisible(is_time)
        self.spin_spl_window_points.setVisible(not is_time)

    def _update_convex_unit_enabled(self):
        idx = self.combo_convex_unit.currentIndex()
        is_audio = idx == 0
        is_points = idx == 1
        is_time = idx == 2
        self.spin_convex_audio_ratio.setVisible(is_audio)
        self.spin_convex_points.setVisible(is_points)
        self.spin_convex_time.setVisible(is_time)

    def _update_peak_units(self):
        is_db_for_size = self.combo_peak_size_unit.currentIndex() == 1
        if is_db_for_size:
            self.spin_peak_size.setRange(-200.0, 500.0)
            self.spin_peak_size.setDecimals(1)
            self.spin_peak_size.setSingleStep(1.0)
        else:
            # rmsV/Vmax -> 0~1 decimal
            self.spin_peak_size.setRange(0.0, 1.0)
            self.spin_peak_size.setDecimals(3)
            self.spin_peak_size.setSingleStep(0.001)
            v = float(self.spin_peak_size.value())
            if v < 0.0:
                self.spin_peak_size.setValue(0.0)
            elif v > 1.0:
                self.spin_peak_size.setValue(1.0)

        # peak slope
        is_db_for_slope = self.combo_peak_slope_unit.currentIndex() == 1
        if is_db_for_slope:
            self.spin_peak_slope.setRange(0.0, 10000.0)
            self.spin_peak_slope.setDecimals(1)
            self.spin_peak_slope.setSingleStep(1.0)
        else:
            self.spin_peak_slope.setRange(0.0, 1.0)
            self.spin_peak_slope.setDecimals(3)
            self.spin_peak_slope.setSingleStep(0.001)
            v2 = float(self.spin_peak_slope.value())
            if v2 < 0.0:
                self.spin_peak_slope.setValue(0.0)
            elif v2 > 1.0:
                self.spin_peak_slope.setValue(1.0)

    def on_click_default_btn(self):
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config("PD", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        self.accept()
        return config_data

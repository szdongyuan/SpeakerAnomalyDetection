from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QGroupBox, QHBoxLayout, QVBoxLayout, QPushButton, QButtonGroup
from PyQt5.QtWidgets import QLabel, QCheckBox, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QRadioButton

from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils


class PDConfigWindow(QDialog):
    def __init__(self, config_manager, model_type):
        super().__init__()
        self.config_manager = config_manager
        self.load_config = self.config_manager.load_config().get(model_type, {})

        self.init_ui()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
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
        self.btn_toggle_advanced = QPushButton("高级模式 >>>")
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

        self.setStyleSheet(
            ui_style_const.qgroupbox_style
            + ui_style_const.qcheckbox_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qlabel_style
            + ui_style_const.qspinbox_style
            + ui_style_const.qdoublespinbox_style
            + ui_style_const.qradiobutton_style
            + ui_style_const.qlineedit_style
            + ui_style_const.qcombobox_style
        )
        # adapt the size according to the visibility of the panel
        self.adjustSize()

    def create_preprocess_group(self):
        group_box = QGroupBox("预处理选项")
        vbox = QVBoxLayout()

        # filter (two rows: main parameters + order)
        row_filter_main = QHBoxLayout()
        self.chk_filter = QCheckBox("滤波")
        self.chk_filter.setChecked(self.load_config.get("filter_enabled", False))
        self.chk_filter.stateChanged.connect(self.get_default_config)
        row_filter_main.addWidget(self.chk_filter)
        row_filter_main.addStretch()
        row_filter_main.addWidget(QLabel("范围(Hz):"))
        self.edit_filter_ranges = QLineEdit()
        self.edit_filter_ranges.setPlaceholderText("0,300; 700,1000;")
        self.edit_filter_ranges.setText(self.load_config.get("filter_ranges", ""))
        self.edit_filter_ranges.textChanged.connect(lambda _: self.get_default_config())
        row_filter_main.addWidget(self.edit_filter_ranges)
        row_filter_main.addWidget(QLabel("类型:"))
        self.combo_filter_type = QComboBox()
        self.combo_filter_type.addItems(["带通", "带阻"])
        self.combo_filter_type.setCurrentIndex(
            0 if self.load_config.get("filter_type", "bandpass") == "bandpass" else 1
        )
        self.combo_filter_type.currentIndexChanged.connect(lambda _: self.get_default_config())
        row_filter_main.addWidget(self.combo_filter_type)

        row_filter_order = QHBoxLayout()
        row_filter_order.addStretch()
        row_filter_order.addWidget(QLabel("阶数"))
        self.spin_filter_order = QSpinBox()
        self.spin_filter_order.setRange(1, 20)
        self.spin_filter_order.setValue(int(self.load_config.get("filter_order", 4)))
        self.spin_filter_order.valueChanged.connect(lambda _: self.get_default_config())
        row_filter_order.addWidget(self.spin_filter_order)

        # smooth (two rows: main parameters + algorithm)
        row_smooth_main = QHBoxLayout()
        self.chk_smooth = QCheckBox("平滑")
        self.chk_smooth.setChecked(self.load_config.get("smooth_enabled", False))
        self.chk_smooth.stateChanged.connect(self.get_default_config)
        row_smooth_main.addWidget(self.chk_smooth)
        row_smooth_main.addStretch()
        row_smooth_main.addWidget(QLabel("单位:"))
        self.combo_smooth_unit = QComboBox()
        self.combo_smooth_unit.addItems(["时间(秒)", "格点数"])
        self.combo_smooth_unit.setCurrentIndex(0 if self.load_config.get("smooth_unit", "time") == "time" else 1)
        self.combo_smooth_unit.currentIndexChanged.connect(
            lambda _: (self._update_smooth_unit_enabled(), self.get_default_config())
        )
        row_smooth_main.addWidget(self.combo_smooth_unit)
        self.spin_smooth_time = QDoubleSpinBox()
        self.spin_smooth_time.setRange(0.00, 999.00)
        self.spin_smooth_time.setDecimals(4)
        self.spin_smooth_time.setSingleStep(0.01)
        self.spin_smooth_time.setValue(float(self.load_config.get("smooth_time_sec", 0.02)))
        self.spin_smooth_time.valueChanged.connect(lambda _: self.get_default_config())
        row_smooth_main.addWidget(self.spin_smooth_time)
        self.spin_smooth_points = QSpinBox()
        self.spin_smooth_points.setRange(1, 99999)
        self.spin_smooth_points.setValue(int(self.load_config.get("smooth_points", 0)))
        self.spin_smooth_points.valueChanged.connect(lambda _: self.get_default_config())
        row_smooth_main.addWidget(self.spin_smooth_points)

        row_smooth_algo = QHBoxLayout()
        row_smooth_algo.addStretch()
        row_smooth_algo.addWidget(QLabel("平滑算法:"))
        self.group_smooth_algo = QButtonGroup(self)
        self.rb_algo1 = QRadioButton("平均")
        self.rb_algo2 = QRadioButton("Golay")
        self.rb_algo3 = QRadioButton("Gaussian")
        row_smooth_algo.addWidget(self.rb_algo1)
        row_smooth_algo.addWidget(self.rb_algo2)
        row_smooth_algo.addWidget(self.rb_algo3)
        self.group_smooth_algo.addButton(self.rb_algo1, 1)
        self.group_smooth_algo.addButton(self.rb_algo2, 2)
        self.group_smooth_algo.addButton(self.rb_algo3, 3)
        algo_saved = int(self.load_config.get("smooth_algo", 1))
        if algo_saved == 2:
            self.rb_algo2.setChecked(True)
        elif algo_saved == 3:
            self.rb_algo3.setChecked(True)
        else:
            self.rb_algo1.setChecked(True)
        self.group_smooth_algo.buttonClicked.connect(lambda _: self.get_default_config())

        # SPL calculation window length (no check box, default enabled; support time/grid point number)
        row_splwin = QHBoxLayout()
        row_splwin.addWidget(QLabel("SPL计算窗长"))
        row_splwin.addStretch()
        row_splwin.addWidget(QLabel("单位:"))
        self.combo_spl_window_unit = QComboBox()
        self.combo_spl_window_unit.addItems(["时间(秒)", "格点数"])
        self.combo_spl_window_unit.setCurrentIndex(
            0 if self.load_config.get("spl_window_unit", "time") == "time" else 1
        )
        self.combo_spl_window_unit.currentIndexChanged.connect(
            lambda _: (self._update_spl_window_unit_enabled(), self.get_default_config())
        )
        row_splwin.addWidget(self.combo_spl_window_unit)
        self.spin_spl_window_time = QDoubleSpinBox()
        self.spin_spl_window_time.setRange(0.000, 999.000)
        self.spin_spl_window_time.setDecimals(4)
        self.spin_spl_window_time.setSingleStep(0.001)
        self.spin_spl_window_time.setValue(float(self.load_config.get("spl_window_time_sec", 0.050)))
        self.spin_spl_window_time.valueChanged.connect(lambda _: self.get_default_config())
        row_splwin.addWidget(self.spin_spl_window_time)
        self.spin_spl_window_points = QSpinBox()
        self.spin_spl_window_points.setRange(1, 99999)
        self.spin_spl_window_points.setValue(int(self.load_config.get("spl_window_points", 0)))
        self.spin_spl_window_points.valueChanged.connect(lambda _: self.get_default_config())
        row_splwin.addWidget(self.spin_spl_window_points)

        vbox.addLayout(row_filter_main)
        vbox.addLayout(row_filter_order)
        # place the SPL calculation window length between the filter and smooth
        vbox.addLayout(row_splwin)
        vbox.addLayout(row_smooth_main)
        vbox.addLayout(row_smooth_algo)
        vbox.setSpacing(8)
        vbox.setContentsMargins(10, 12, 10, 12)
        group_box.setLayout(vbox)
        # initialize the display state
        self._update_smooth_unit_enabled()
        self._update_spl_window_unit_enabled()
        return group_box

    def create_detect_group(self):
        group_box = QGroupBox("峰值提取参数")
        vbox = QVBoxLayout()

        # peak count
        row_count = QHBoxLayout()
        self.chk_peak_count = QCheckBox("峰值个数")
        self.chk_peak_count.setChecked(self.load_config.get("peak_count_enabled", True))
        self.chk_peak_count.stateChanged.connect(self.get_default_config)
        row_count.addWidget(self.chk_peak_count)
        row_count.addStretch()
        row_count.addWidget(QLabel("最大峰数目:"))
        self.spin_peak_count = QSpinBox()
        self.spin_peak_count.setRange(1, 9999)
        self.spin_peak_count.setValue(int(self.load_config.get("peak_count", 3)))
        self.spin_peak_count.valueChanged.connect(lambda _: self.get_default_config())
        row_count.addWidget(self.spin_peak_count)
        # row_count.addWidget(QLabel("个"))

        # peak size
        row_size = QHBoxLayout()
        self.chk_peak_size = QCheckBox("峰值大小")
        self.chk_peak_size.setChecked(self.load_config.get("peak_size_enabled", True))
        self.chk_peak_size.stateChanged.connect(self.get_default_config)
        row_size.addWidget(self.chk_peak_size)
        row_size.addStretch()
        row_size.addWidget(QLabel("单位:"))
        self.combo_peak_size_unit = QComboBox()
        self.combo_peak_size_unit.addItems(["rmsV", "dBL"])
        peak_size_unit_saved = self.load_config.get("peak_size_unit", "db")
        self.combo_peak_size_unit.setCurrentIndex(0 if peak_size_unit_saved == "rms" else 1)
        self.combo_peak_size_unit.currentIndexChanged.connect(
            lambda _: (self._update_peak_units(), self.get_default_config())
        )
        row_size.addWidget(self.combo_peak_size_unit)
        self.spin_peak_size = QDoubleSpinBox()
        self.spin_peak_size.setRange(-100.0, 200.0)
        self.spin_peak_size.setDecimals(2)
        self.spin_peak_size.setSingleStep(1.0)
        self.spin_peak_size.setValue(float(self.load_config.get("peak_min_value", 100.0)))
        self.spin_peak_size.valueChanged.connect(lambda _: self.get_default_config())
        row_size.addWidget(self.spin_peak_size)

        # peak slope
        row_slope = QHBoxLayout()
        self.chk_peak_slope = QCheckBox("峰凸起度")
        self.chk_peak_slope.setChecked(self.load_config.get("peak_slope_enabled", False))
        self.chk_peak_slope.stateChanged.connect(self.get_default_config)
        row_slope.addWidget(self.chk_peak_slope)
        row_slope.addStretch()
        row_slope.addWidget(QLabel("单位:"))
        self.combo_peak_slope_unit = QComboBox()
        self.combo_peak_slope_unit.addItems(["rmsV", "dBL"])
        peak_slope_unit_saved = self.load_config.get("peak_slope_unit", "db")
        self.combo_peak_slope_unit.setCurrentIndex(0 if peak_slope_unit_saved == "rms" else 1)
        self.combo_peak_slope_unit.currentIndexChanged.connect(
            lambda _: (self._update_peak_units(), self.get_default_config())
        )
        row_slope.addWidget(self.combo_peak_slope_unit)
        self.spin_peak_slope = QDoubleSpinBox()
        self.spin_peak_slope.setRange(0.0, 200.0)
        self.spin_peak_slope.setDecimals(3)
        self.spin_peak_slope.setSingleStep(1.0)
        self.spin_peak_slope.setValue(float(self.load_config.get("peak_min_slope", 100.0)))
        self.spin_peak_slope.valueChanged.connect(lambda _: self.get_default_config())
        row_slope.addWidget(self.spin_peak_slope)

        # minimum peak distance (support time/grid point number)
        row_nms = QHBoxLayout()
        self.chk_nms = QCheckBox("最小峰间距")
        self.chk_nms.setChecked(self.load_config.get("nms_enabled", False))
        self.chk_nms.stateChanged.connect(self.get_default_config)
        row_nms.addWidget(self.chk_nms)
        row_nms.addStretch()
        row_nms.addWidget(QLabel("单位:"))
        self.combo_nms_unit = QComboBox()
        self.combo_nms_unit.addItems(["时间(秒)", "格点数"])
        self.combo_nms_unit.setCurrentIndex(0 if self.load_config.get("nms_unit", "time") == "time" else 1)
        self.combo_nms_unit.currentIndexChanged.connect(
            lambda _: (self._update_nms_unit_enabled(), self.get_default_config())
        )
        row_nms.addWidget(self.combo_nms_unit)
        self.spin_nms_time = QDoubleSpinBox()
        self.spin_nms_time.setRange(0.00, 100.00)
        self.spin_nms_time.setDecimals(3)
        self.spin_nms_time.setSingleStep(0.01)
        self.spin_nms_time.setValue(float(self.load_config.get("nms_time_sec", 0.50)))
        self.spin_nms_time.valueChanged.connect(lambda _: self.get_default_config())
        row_nms.addWidget(self.spin_nms_time)
        self.spin_nms_points = QSpinBox()
        self.spin_nms_points.setRange(1, 99999)
        self.spin_nms_points.setValue(int(self.load_config.get("nms_points", 0)))
        self.spin_nms_points.valueChanged.connect(lambda _: self.get_default_config())
        row_nms.addWidget(self.spin_nms_points)

        # duration
        row_duration = QHBoxLayout()
        self.chk_duration = QCheckBox("峰持续时间")
        self.chk_duration.setChecked(self.load_config.get("duration_enabled", False))
        self.chk_duration.stateChanged.connect(self.get_default_config)
        row_duration.addWidget(self.chk_duration)
        row_duration.addStretch()
        row_duration.addWidget(QLabel("最短"))
        self.spin_duration_min = QDoubleSpinBox()
        self.spin_duration_min.setRange(0.0, 1000.0)
        self.spin_duration_min.setDecimals(3)
        self.spin_duration_min.setSingleStep(0.001)
        self.spin_duration_min.setValue(float(self.load_config.get("duration_min", 0.0)))
        self.spin_duration_min.valueChanged.connect(lambda _: self.get_default_config())
        row_duration.addWidget(self.spin_duration_min)
        row_duration.addWidget(QLabel("最长"))
        self.spin_duration_max = QDoubleSpinBox()
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
        group_box = QGroupBox("测试选项")
        vbox = QVBoxLayout()

        row_peak_condition = QHBoxLayout()
        row_peak_condition.addWidget(QLabel("峰值点数目"))
        row_peak_condition.addStretch()
        self.combo_test_peak_op = QComboBox()
        self.combo_test_peak_op.addItems([">", "<", "=", "≥", "≤"])
        saved_op = self.load_config.get("test_peak_op", "≥")
        try:
            idx = [">", "<", "=", "≥", "≤"].index(saved_op)
        except ValueError:
            idx = 3
        self.combo_test_peak_op.setCurrentIndex(idx)
        self.combo_test_peak_op.currentIndexChanged.connect(lambda _: self.get_default_config())
        row_peak_condition.addWidget(self.combo_test_peak_op)
        self.spin_test_peak_value = QSpinBox()
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
        adv_group = QGroupBox("高级选项")
        adv_layout = QVBoxLayout()
        adv_layout.addWidget(self.create_preprocess_group())

        row_convex_len = QHBoxLayout()
        row_convex_len.addWidget(QLabel("峰凸起度计算窗口"))
        row_convex_len.addStretch(1)
        row_convex_len.addWidget(QLabel("单位:"))
        self.combo_convex_unit = QComboBox()
        self.combo_convex_unit.addItems(["音频长度", "格点数", "时长(秒)"])
        self.combo_convex_unit.setCurrentIndex(
            {"audio": 0, "points": 1, "time": 2}.get(self.load_config.get("convex_unit", "audio"), 0)
        )
        self.combo_convex_unit.currentIndexChanged.connect(
            lambda _: (self._update_convex_unit_enabled(), self.get_default_config())
        )
        row_convex_len.addWidget(self.combo_convex_unit)
        self.spin_convex_audio_ratio = QDoubleSpinBox()
        self.spin_convex_audio_ratio.setRange(0.001, 1.000)
        self.spin_convex_audio_ratio.setDecimals(4)
        self.spin_convex_audio_ratio.setSingleStep(0.01)
        self.spin_convex_audio_ratio.setValue(float(self.load_config.get("convex_audio_ratio", 1.0)))
        self.spin_convex_audio_ratio.valueChanged.connect(lambda _: self.get_default_config())
        row_convex_len.addWidget(self.spin_convex_audio_ratio)
        self.spin_convex_points = QSpinBox()
        self.spin_convex_points.setRange(1, 100000000)
        self.spin_convex_points.setValue(int(self.load_config.get("convex_points", 1024)))
        self.spin_convex_points.valueChanged.connect(lambda _: self.get_default_config())
        row_convex_len.addWidget(self.spin_convex_points)
        self.spin_convex_time = QDoubleSpinBox()
        self.spin_convex_time.setRange(0.000, 999.000)
        self.spin_convex_time.setDecimals(3)
        self.spin_convex_time.setSingleStep(0.1)
        self.spin_convex_time.setValue(float(self.load_config.get("convex_time_sec", 0.0)))
        self.spin_convex_time.valueChanged.connect(lambda _: self.get_default_config())
        row_convex_len.addWidget(self.spin_convex_time)

        # 峰持续时间参考点（始终启用）
        row_dmode = QHBoxLayout()
        row_dmode.addWidget(QLabel("峰持续时间参考点"))
        row_dmode.addStretch(1)
        row_dmode.addWidget(QLabel("单位:"))

        ref_unit_saved = self.load_config.get("duration_ref_unit", "peak")
        ref_value_saved = float(self.load_config.get("duration_ref_value", 0.50 if ref_unit_saved == "peak" else 100.0))

        self.combo_duration_ref_unit = QComboBox()
        self.combo_duration_ref_unit.addItems(["Vpeak", "dBL"])
        self.combo_duration_ref_unit.setCurrentIndex(0 if ref_unit_saved == "peak" else 1)
        self.combo_duration_ref_unit.currentIndexChanged.connect(
            lambda _: (self._update_duration_ref_unit(), self.get_default_config())
        )
        row_dmode.addWidget(self.combo_duration_ref_unit)

        self.spin_duration_ref = QDoubleSpinBox()
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
        btn_layout = QHBoxLayout()
        default_btn = QPushButton(" 设为默认 ")
        ok_btn = QPushButton(" 确  认 ")
        default_btn.clicked.connect(self.on_click_default_btn)
        ok_btn.clicked.connect(self.on_click_ok_btn)
        btn_layout.addWidget(default_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        return btn_layout

    def get_default_config(self):
        default_config = {
            # preprocess(advanced)
            "filter_enabled": self.chk_filter.isChecked(),
            "filter_ranges": self.edit_filter_ranges.text().strip(),
            "filter_type": "bandpass" if self.combo_filter_type.currentIndex() == 0 else "bandstop",
            "smooth_enabled": self.chk_smooth.isChecked(),
            "smooth_unit": "time" if self.combo_smooth_unit.currentIndex() == 0 else "points",
            "smooth_time_sec": float(self.spin_smooth_time.value()),
            "smooth_points": int(self.spin_smooth_points.value()),
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
            "smooth_algo": int(self.group_smooth_algo.checkedId() or 1),
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

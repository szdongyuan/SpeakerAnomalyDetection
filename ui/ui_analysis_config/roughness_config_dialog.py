from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QWidget

from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import CheckBox, DoubleSpinBox, GroupBox, PushButton
from ui.ui_src import ui_resources


def default_roughness_config():
    return {
        "enabled": True,
        "display": {
            "summary_metrics": [
                "r_mean_asper",
            ],
            "curves": ["roughness_time"],
        },
        "save": {
            "summary": False,
            "curve": False,
            "specific_roughness": False,
        },
        "advanced": {
            "model": "daniel_weber_mosqito_dw",
            "overlap": 0.5,
        },
        "limit_checked": False,
        "limit_metric": "curve_y",
        "curve_limit_unit": "asper",
        "curve_upper_enabled": False,
        "curve_upper_value": 0.0,
        "curve_lower_enabled": False,
        "curve_lower_value": 0.0,
    }


def _with_roughness_defaults(load_config):
    cfg = dict(load_config or {})
    defaults = default_roughness_config()
    for section in ("display", "save", "advanced"):
        current = dict(cfg.get(section, {}) or {})
        for key, value in defaults[section].items():
            current.setdefault(key, value)
        cfg[section] = current
    cfg.setdefault("enabled", defaults["enabled"])
    for key, value in defaults.items():
        if key not in ("display", "save", "advanced", "enabled"):
            cfg.setdefault(key, value)
    return cfg


class RoughnessConfigPanel(QWidget):
    """Reusable editor for the ROUGH item config."""

    def __init__(self, load_config=None, title_prefix=""):
        super().__init__()
        self.load_config = _with_roughness_defaults(load_config)
        self.title_prefix = title_prefix
        self.init_ui()

    def _group_title(self, title):
        return f"{self.title_prefix}{title}" if self.title_prefix else title

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._create_output_group())
        layout.addWidget(self._create_limit_group())
        layout.addStretch()
        self.setLayout(layout)

    def _create_output_group(self):
        group = GroupBox(self._group_title("显示设置"))
        group.setMinimumHeight(150)
        layout = QHBoxLayout()
        layout.setSpacing(18)
        layout.setContentsMargins(12, 22, 12, 14)

        display_cfg = self.load_config.get("display", {}) or {}
        summary_metrics = display_cfg.get("summary_metrics", []) or []
        curves = display_cfg.get("curves", []) or []

        metric_group = GroupBox("指标")
        metric_group.setMinimumSize(330, 116)
        metric_layout = QVBoxLayout()
        metric_layout.setSpacing(8)
        metric_layout.setContentsMargins(12, 20, 12, 12)

        self.show_rmean_box = CheckBox("平均粗糙度 Rmean（asper）")
        self.show_rmean_box.setToolTip("Rmean：所有 200 ms 分帧粗糙度的平均值。")
        self.show_rmean_box.setChecked("r_mean_asper" in summary_metrics)

        metric_layout.addWidget(self.show_rmean_box)
        metric_group.setLayout(metric_layout)

        graph_group = GroupBox("图形")
        graph_group.setMinimumSize(330, 116)
        graph_layout = QVBoxLayout()
        graph_layout.setSpacing(10)
        graph_layout.setContentsMargins(12, 20, 12, 12)

        self.show_curve_box = CheckBox("粗糙度曲线（R-t）")
        self.show_curve_box.setToolTip("显示 Daniel-Weber roughness 随 200 ms 分帧变化的曲线。")
        self.show_curve_box.setChecked("roughness_time" in curves)
        graph_layout.addWidget(self.show_curve_box)
        graph_group.setLayout(graph_layout)

        layout.addWidget(metric_group)
        layout.addWidget(graph_group, 1)
        group.setLayout(layout)
        return group

    def _create_limit_group(self):
        group = GroupBox(self._group_title("判定阈值"))
        group.setMinimumHeight(155)
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(12, 24, 12, 14)

        self.limit_checked_box = CheckBox("启用粗糙度曲线 OK/NG 判定")
        self.limit_checked_box.setToolTip("按粗糙度曲线 R(t) 的纵轴值逐点判断，上下限单位为 asper。")
        self.limit_checked_box.setChecked(bool(self.load_config.get("limit_checked", False)))
        layout.addWidget(self.limit_checked_box)

        upper_layout = QHBoxLayout()
        self.upper_enabled_box = CheckBox("上限")
        self.upper_enabled_box.setChecked(bool(self.load_config.get("curve_upper_enabled", False)))
        self.upper_spin = DoubleSpinBox()
        self.upper_spin.setRange(0.0, 10000.0)
        self.upper_spin.setDecimals(4)
        self.upper_spin.setValue(float(self.load_config.get("curve_upper_value", 0.0)))
        self.upper_spin.setSuffix(" asper")
        self.upper_spin.setMinimumWidth(160)
        self.upper_spin.setMaximumWidth(220)
        upper_layout.addWidget(self.upper_enabled_box)
        upper_layout.addWidget(self.upper_spin)
        upper_layout.addStretch(1)
        layout.addLayout(upper_layout)

        lower_layout = QHBoxLayout()
        self.lower_enabled_box = CheckBox("下限")
        self.lower_enabled_box.setChecked(bool(self.load_config.get("curve_lower_enabled", False)))
        self.lower_spin = DoubleSpinBox()
        self.lower_spin.setRange(0.0, 10000.0)
        self.lower_spin.setDecimals(4)
        self.lower_spin.setValue(float(self.load_config.get("curve_lower_value", 0.0)))
        self.lower_spin.setSuffix(" asper")
        self.lower_spin.setMinimumWidth(160)
        self.lower_spin.setMaximumWidth(220)
        lower_layout.addWidget(self.lower_enabled_box)
        lower_layout.addWidget(self.lower_spin)
        lower_layout.addStretch(1)
        layout.addLayout(lower_layout)

        group.setLayout(layout)
        return group

    def get_default_config(self):
        summary_metrics = []
        if self.show_rmean_box.isChecked():
            summary_metrics.append("r_mean_asper")

        curves = ["roughness_time"] if self.show_curve_box.isChecked() else []

        return {
            "enabled": True,
            "display": {
                "summary_metrics": summary_metrics,
                "curves": curves,
            },
            "save": {
                "summary": False,
                "curve": False,
                "specific_roughness": False,
            },
            "advanced": {
                "model": "daniel_weber_mosqito_dw",
                "overlap": 0.5,
            },
            "limit_checked": self.limit_checked_box.isChecked(),
            "limit_metric": "curve_y",
            "curve_limit_unit": "asper",
            "curve_upper_enabled": self.upper_enabled_box.isChecked(),
            "curve_upper_value": self.upper_spin.value(),
            "curve_lower_enabled": self.lower_enabled_box.isChecked(),
            "curve_lower_value": self.lower_spin.value(),
        }


class RoughnessConfigWindow(QDialog):
    """Configuration dialog for standalone Daniel-Weber roughness."""

    def __init__(self, config_manager, model_type):
        super().__init__()
        self.config_manager = config_manager
        self.model_type = model_type
        self.load_config = self.config_manager.load_config().get(model_type, {})
        self.panel = RoughnessConfigPanel(self.load_config)
        self.init_ui()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        self.setWindowTitle("粗糙度分析配置")
        self.setMinimumSize(500, 430)

        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.addWidget(self.panel)
        layout.addLayout(self._create_btn_layout())
        self.setLayout(layout)

    def _create_btn_layout(self):
        btn_layout = QHBoxLayout()
        default_btn = PushButton(" 设为默认 ")
        default_btn.clicked.connect(self.on_default_btn_clicked)
        ok_btn = PushButton(" 确 认 ")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        btn_layout.addWidget(default_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        return btn_layout

    def get_default_config(self):
        return self.panel.get_default_config()

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config("ROUGH", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        self.accept()
        return self.get_default_config()

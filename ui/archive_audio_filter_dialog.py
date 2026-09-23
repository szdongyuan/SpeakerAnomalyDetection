"""Project-aware filters for archived audio recordings."""

from collections import Counter
import copy
import re

from PyQt5.QtCore import QDate, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QSizePolicy, QVBoxLayout,
)

from base.audio_record_filter import (
    METADATA_FILTER_FIELDS, UNKNOWN,
    matches_filter_value, natural_sort_key,
)
from consts.running_consts import DEFAULT_DIR
from ui.config_dialog_base import ConfigDialogBase
from ui.dialog_enter_policy import install_dialog_enter_policy


_MIN_SAMPLE_RATE = 4000
_MAX_SAMPLE_RATE = 192000

_SELECT_FIELDS = (
    ("select_product_model", "产品型号", "全部型号"),
    ("select_sample_number", "样本编号", "全部样本"),
    ("select_test_round", "测试轮次", "全部轮次"),
    ("select_port", "端口", "全部端口"),
    ("select_condition", "档位", "全部档位"),
)


class ArchiveAudioFilterDialog(ConfigDialogBase):
    def __init__(self, rows, metadata_by_id, filter_config=None, parent=None):
        super().__init__(parent)
        self.rows = rows
        self.metadata_by_id = metadata_by_id
        self.filter_config = copy.deepcopy(filter_config or {})
        self.combos = {}
        self.setWindowTitle("筛选音频")
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.resize(640, 510)
        self._build_ui()
        self._restore_filters()
        self.apply_config_dialog_theme()
        install_dialog_enter_policy(self, self.apply_button)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(18)
        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        project = self._new_combo("select_project")
        grid.addWidget(QLabel("所属项目"), 0, 0, 1, 2)
        grid.addWidget(project, 1, 0, 1, 2)
        for index, (key, label, _all_text) in enumerate(_SELECT_FIELDS):
            row, column = 2 + (index // 2) * 2, index % 2
            grid.addWidget(QLabel(label), row, column)
            grid.addWidget(self._new_combo(key), row + 1, column)
        layout.addLayout(grid)

        labels_row, self.label_boxes = self._checkbox_row(
            "音频标签", [("OK", "OK"), ("NG", "NG"), ("未标记", "not_labeled")],
        )
        layout.addLayout(labels_row)

        dates = QHBoxLayout()
        dates.setSpacing(8)
        dates.addWidget(QLabel("录音日期"))
        self.date_filter_combobox = QComboBox()
        self.date_filter_combobox.setAccessibleName("录音日期")
        self.date_filter_combobox.setEditable(True)
        self.date_filter_combobox.addItem("ALL")
        self.date_filter_combobox.addItems(sorted({row[4] for row in self.rows if row[4]}))
        self.date_filter_combobox.setMinimumWidth(180)
        dates.addWidget(self.date_filter_combobox, 1)
        dates.addStretch(1)
        layout.addLayout(dates)

        rates_row = QHBoxLayout()
        rates_row.setSpacing(8)
        rates_row.addWidget(QLabel("采样率 (Hz)"))
        self.sample_rate_combobox = QComboBox()
        self.sample_rate_combobox.setAccessibleName("采样率 (Hz)")
        self.sample_rate_combobox.setEditable(True)
        self.sample_rate_combobox.setInsertPolicy(QComboBox.NoInsert)
        self.sample_rate_combobox.addItem("ALL")
        rates = {44100, 48000} | {row[3] for row in self.rows if row[3]}
        self.sample_rate_combobox.addItems([
            str(rate) for rate in sorted(rates)
            if _MIN_SAMPLE_RATE <= rate <= _MAX_SAMPLE_RATE
        ])
        self.sample_rate_combobox.setMinimumWidth(180)
        self.sample_rate_combobox.setToolTip(
            f"输入 {_MIN_SAMPLE_RATE}–{_MAX_SAMPLE_RATE} Hz 的整数采样率（含边界），"
            "多个值用逗号分隔；ALL 表示不限采样率。"
        )
        rates_row.addWidget(self.sample_rate_combobox, 1)
        rates_row.addStretch(1)
        layout.addLayout(rates_row)
        layout.addStretch()
        buttons = QHBoxLayout()
        self.reset_button = QPushButton("重置条件")
        self.cancel_button = QPushButton("取消")
        self.apply_button = QPushButton("应用筛选")
        self.apply_button.setDefault(True)
        self.reset_button.clicked.connect(self.reset_filters)
        self.cancel_button.clicked.connect(self.reject)
        self.apply_button.clicked.connect(self.apply_filters)
        buttons.addWidget(self.reset_button)
        buttons.addStretch()
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.apply_button)
        layout.addLayout(buttons)
        project.currentIndexChanged.connect(self._project_changed)

    def _new_combo(self, key):
        combo = QComboBox()
        combo.setObjectName(key)
        combo.setMinimumContentsLength(14)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.combos[key] = combo
        return combo

    @staticmethod
    def _checkbox_row(label, options):
        row = QHBoxLayout()
        row.setSpacing(16)
        row.addWidget(QLabel(label))
        boxes = {}
        for text, value in options:
            box = QCheckBox(text)
            box.setChecked(True)
            boxes[value] = box
            row.addWidget(box)
        row.addStretch()
        return row, boxes

    @staticmethod
    def _set_options(combo, all_text, options, selected=None):
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(all_text, None)
        selected_index = 0
        for text, value, tooltip in options:
            combo.addItem(text, value)
            index = combo.count() - 1
            if tooltip:
                combo.setItemData(index, tooltip, Qt.ToolTipRole)
            if value == selected:
                selected_index = index
        combo.setCurrentIndex(selected_index)
        combo.blockSignals(False)

    def _restore_filters(self):
        projects = {
            info.project_key: info.project_name for info in self.metadata_by_id.values()
            if info.project_key is not None
        }
        names = Counter(projects.values())
        options = []
        for key, name in sorted(projects.items(), key=lambda item: (natural_sort_key(item[1]), item[0])):
            text = f"{name} — {key}" if names[name] > 1 else name
            options.append((text, key, key))
        if any(info.project_key is None for info in self.metadata_by_id.values()):
            options.append(("未识别", UNKNOWN, "无法从保存路径和文件名识别项目"))
        self._set_options(self.combos["select_project"], "全部项目", options,
                          self.filter_config.get("select_project"))
        self._project_changed()
        for key, _label, _all_text in _SELECT_FIELDS:
            combo = self.combos[key]
            selected = self.filter_config.get(key)
            for index in range(combo.count()):
                if combo.itemData(index) == selected:
                    combo.setCurrentIndex(index)
                    break
        chosen = self.filter_config.get("select_labels", self.label_boxes)
        for value, box in self.label_boxes.items():
            box.setChecked(value in chosen)
        selected_rates = self.filter_config.get("select_sample_rate")
        self.sample_rate_combobox.setCurrentText(
            "ALL" if selected_rates is None else ", ".join(map(str, selected_rates))
        )
        self.date_filter_combobox.setCurrentText(self.filter_config.get("select_record_date", "ALL"))

    def _project_changed(self):
        selected_project = self.combos["select_project"].currentData()
        self.combos["select_project"].setToolTip(
            selected_project if isinstance(selected_project, str) else ""
        )
        rows = [row for row in self.rows if matches_filter_value(
            self.metadata_by_id[row[0]].project_key, selected_project,
        )]
        for key, _label, all_text in _SELECT_FIELDS:
            values = {
                row[2] if key == "select_product_model" else
                getattr(self.metadata_by_id[row[0]], METADATA_FILTER_FIELDS[key])
                for row in rows
            }
            options = [
                (f"第 {value} 轮" if key == "select_test_round" else str(value), value, "")
                for value in sorted((v for v in values if v is not None), key=natural_sort_key)
            ]
            if None in values:
                options.append(("未识别", UNKNOWN, "该字段无法可靠解析"))
            combo = self.combos[key]
            self._set_options(combo, all_text, options, combo.currentData())

    def reset_filters(self):
        self.combos["select_project"].setCurrentIndex(0)
        for combo in self.combos.values():
            combo.setCurrentIndex(0)
        for box in self.label_boxes.values():
            box.setChecked(True)
        self.date_filter_combobox.setCurrentText("ALL")
        self.sample_rate_combobox.setCurrentText("ALL")

    def apply_filters(self):
        filters = {
            key: combo.currentData() for key, combo in self.combos.items()
            if combo.currentData() is not None
        }
        labels = [value for value, box in self.label_boxes.items() if box.isChecked()]
        if not labels:
            QMessageBox.warning(self, "提示", "请至少选择一个标签。")
            return
        if len(labels) != len(self.label_boxes):
            filters["select_labels"] = labels
        selected_rates = self.sample_rate_combobox.currentText().strip()
        if selected_rates.upper() != "ALL":
            parts = [part.strip() for part in selected_rates.replace("，", ",").split(",")]
            try:
                if any(re.fullmatch(r"[0-9]+", part) is None for part in parts):
                    raise ValueError
                rates = list(dict.fromkeys(int(part) for part in parts))
                if any(not _MIN_SAMPLE_RATE <= rate <= _MAX_SAMPLE_RATE for rate in rates):
                    raise ValueError
            except ValueError:
                QMessageBox.warning(
                    self, "提示",
                    f"请输入 {_MIN_SAMPLE_RATE}–{_MAX_SAMPLE_RATE} Hz 范围内的正整数采样率（含边界），"
                    "多个值用逗号分隔，或选择 ALL。",
                )
                return
            filters["select_sample_rate"] = rates
        selected_date = self.date_filter_combobox.currentText().strip()
        if selected_date.upper() != "ALL":
            parsed_date = QDate.fromString(selected_date, "yyyy-MM-dd")
            if not parsed_date.isValid() or parsed_date.toString("yyyy-MM-dd") != selected_date:
                QMessageBox.warning(self, "提示", "日期格式错误，请输入有效日期，例如 2026-09-14，或选择 ALL。")
                return
            filters["select_record_date"] = selected_date
        self.filter_config = filters
        self.accept()

    def exec(self):
        if super().exec() == self.Accepted:
            return (1 if self.filter_config else 2), self.filter_config
        return 0, {}

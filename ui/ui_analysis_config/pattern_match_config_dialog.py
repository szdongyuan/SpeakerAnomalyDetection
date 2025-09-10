import os
import numpy as np

from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QIcon, QStandardItem, QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import QDialog, QGroupBox, QHBoxLayout, QVBoxLayout, QPushButton, QWidget, QFormLayout
from PyQt5.QtWidgets import QLabel, QCheckBox, QComboBox, QLineEdit, QSplitter, QFrame, QRadioButton, QMessageBox
from PyQt5.QtWidgets import QFileDialog

from base.file_ops import FileOps
from base.load_audio import load_audio_simple
from base.load_config import LoadUiConfig
from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.audio_clip_extraction_dialog import AudioClipExtractionDialog
from ui.custom_ui_widget.custom_table_widget import DataView
from ui.generic_feature_params_dialog import GenericFeatureParamsDialog
from ui.custom_ui_widget.popuputils import PopupUtils


class PatternMatchConfigWindow(QDialog):
    def __init__(self, config_manager, model_type):
        super().__init__()
        self.config_manager = config_manager
        _, self.feature_registry = self.load_features_param_config()
        self.load_config = self.config_manager.load_config().get(model_type, {})
        self.pattern_list = []
        # self.audio_file_path = None
        # self.pattern_save_path = None
        # self.audio_data = None
        self.sample_rate = self.load_config["sample_rate"]
        # self.selected_region_time = (None, None)
        self.config_data = None

        self.feature_params = {
            key: {p_name: p_def["default"] for p_name, p_def in info["params"].items()}
            for key, info in self.feature_registry.items()
            if info.get("params")
        }

        self.init_ui()
        self.on_strategy_radio_changed()
        self.on_filter_toggled()
        self.on_feature_type_changed()
        self.populate_ui_from_config()

    def init_ui(self):
        self.setWindowTitle("模式匹配参数配置")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumSize(800, 750)
        self.resize(800, 750)
        self.main_layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Vertical)

        pattern_group_box = self.create_pattern_group_box()
        splitter.addWidget(pattern_group_box)

        options_container = QWidget()
        options_layout = self.create_options_layout()
        options_container.setLayout(options_layout)
        splitter.addWidget(options_container)

        splitter.setSizes([450, 300])
        splitter.setCollapsible(0, False)
        self.main_layout.addWidget(splitter)

        btn_layout = self.create_btn_layout()
        self.main_layout.addLayout(btn_layout)

        self.setLayout(self.main_layout)
        self.setStyleSheet(
            ui_style_const.qcheckbox_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qlabel_style
            + ui_style_const.qlineedit_style
            + ui_style_const.qgroupbox_style
            + ui_style_const.qcombobox_style
            + ui_style_const.qdialog_style
            + ui_style_const.qradiobutton_style
            + ui_style_const.qtextedit_style
            + ui_style_const.qtableview_style
        )

    def create_pattern_group_box(self):
        self.data_view = DataView(len(self.pattern_list), 2, [])
        self.data_view.set_h_header(["模板文件", "模板时长 (s)"])
        extract_btn = QPushButton("提取模板")
        extract_btn.clicked.connect(self.on_click_extract_btn)
        add_btn = QPushButton("添加模板")
        add_btn.clicked.connect(self.on_click_add_btn)
        remove_btn = QPushButton("删除模板")
        remove_btn.clicked.connect(self.on_click_remove_btn)
        self.n_chosen_pattern_label = QLabel("已加载模板： 0")

        btn_layout = QVBoxLayout()
        btn_layout.addWidget(extract_btn)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addWidget(self.n_chosen_pattern_label, alignment=Qt.AlignBottom)

        layout = QHBoxLayout()
        layout.addWidget(self.data_view)
        layout.addLayout(btn_layout)
        group = QGroupBox("模板选择")
        group.setLayout(layout)
        return group

    def create_options_layout(self):
        layout = QHBoxLayout()
        processing_feature_group = self.create_processing_and_feature_group()
        strategy_group = self.create_strategy_group()
        layout.addWidget(processing_feature_group)
        layout.addWidget(strategy_group)
        layout.setStretch(0, 1)
        layout.setStretch(1, 1)
        return layout

    def create_processing_and_feature_group(self):
        group = QGroupBox("特征与预处理")
        layout = QVBoxLayout()

        feature_label = QLabel("<b>特征类型</b>")
        layout.addWidget(feature_label)

        combo_layout = QHBoxLayout()
        self.feature_combo = QComboBox()
        for key, info in self.feature_registry.items():
            self.feature_combo.addItem(info["display_name"], userData=key)
        self.feature_combo.currentIndexChanged.connect(self.on_feature_type_changed)
        combo_layout.addWidget(self.feature_combo)

        self.feature_params_btn = QPushButton("特征参数")
        self.feature_params_btn.clicked.connect(self.on_click_feature_params)
        combo_layout.addWidget(self.feature_params_btn)
        layout.addLayout(combo_layout)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        filter_label = QLabel("<b>带阻滤波</b>")
        layout.addWidget(filter_label)
        self.filter_checkbox = QCheckBox("启用")
        self.filter_checkbox.toggled.connect(self.on_filter_toggled)
        layout.addWidget(self.filter_checkbox)

        filter_range_layout = QFormLayout()
        self.low_freq_edit = QLineEdit("0")
        self.low_freq_edit.setValidator(QIntValidator(0, 20000, self))
        self.high_freq_edit = QLineEdit("5000")
        self.high_freq_edit.setValidator(QIntValidator(0, 20000, self))
        low_freq_label = QLabel("最低频率 (Hz):")
        high_freq_label = QLabel("最高频率 (Hz):")
        filter_range_layout.addRow(low_freq_label, self.low_freq_edit)
        filter_range_layout.addRow(high_freq_label, self.high_freq_edit)

        layout.addLayout(filter_range_layout)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def create_strategy_group(self):
        group = QGroupBox("匹配策略")
        main_layout = QVBoxLayout()

        metric_layout = QHBoxLayout()
        metric_label = QLabel("<b>相似度度量:</b>")
        self.similarity_metric_combo = QComboBox()
        self.similarity_metric_combo.addItem("欧氏距离 (Euclidean)", "euclidean")
        self.similarity_metric_combo.addItem("余弦相似度 (Cosine)", "cosine")
        metric_layout.addWidget(metric_label)
        metric_layout.addWidget(self.similarity_metric_combo)
        main_layout.addLayout(metric_layout)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        return_label = QLabel("<b>匹配点返回策略:</b>")
        main_layout.addWidget(return_label)
        fixed_threshold_layout = QHBoxLayout()
        self.fixed_threshold_radio = QRadioButton("固定阈值:")
        self.fixed_threshold_radio.setChecked(True)
        self.fixed_threshold_radio.toggled.connect(self.on_strategy_radio_changed)
        self.threshold_edit = QLineEdit("0.9")
        self.threshold_edit.setValidator(QDoubleValidator(0.0, 100, 5, self))
        fixed_threshold_layout.addWidget(self.fixed_threshold_radio)
        fixed_threshold_layout.addWidget(self.threshold_edit)
        self.adaptive_threshold_radio = QRadioButton("自适应阈值")
        self.adaptive_threshold_radio.toggled.connect(self.on_strategy_radio_changed)
        main_layout.addLayout(fixed_threshold_layout)
        main_layout.addWidget(self.adaptive_threshold_radio)
        main_layout.addStretch()
        group.setLayout(main_layout)
        return group

    def create_btn_layout(self):
        layout = QHBoxLayout()
        ok_btn = QPushButton(" 确  认 ")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        default_btn = QPushButton("设为默认")
        default_btn.clicked.connect(self.on_click_default_btn)
        layout.addWidget(default_btn)
        layout.addStretch()
        layout.addWidget(ok_btn)
        return layout

    def on_click_extract_btn(self):
        dlg = AudioClipExtractionDialog(save_clip=True, dialog_title="选择模板片段")
        _, clip_path, clip_len = dlg.on_exec()
        if clip_path is not None:
            self.pattern_list.append({"clip_path": clip_path, "clip_len": clip_len})
            self.refresh_data_view()

    def on_click_add_btn(self):
        file_names, _ = QFileDialog.getOpenFileNames(
            self, "选择音频文件", DEFAULT_DIR + "audio_data/pattern/", "音频文件 (*.wav)"
        )
        for file_name in file_names:
            relative_path = FileOps.get_relative_path(file_name, DEFAULT_DIR)
            pattern_data, _ = load_audio_simple(file_name, self.sample_rate)
            self.pattern_list.append({"clip_path": relative_path, "clip_len": len(pattern_data)})
        self.refresh_data_view()

    def on_click_remove_btn(self):
        row_idx = self.data_view.currentIndex().row()
        self.pattern_list.pop(row_idx)
        self.refresh_data_view()

    @staticmethod
    def load_features_param_config():
        default_config_file = os.path.join(DEFAULT_DIR, "ui", "ui_config", "features_param.json")
        code, data = LoadUiConfig.load_data_from_json(default_config_file)
        if code == 0:
            return True, data
        else:
            return False, {}

    def on_click_feature_params(self):
        feature_key = self.feature_combo.currentData()
        if not feature_key or not self.feature_registry[feature_key].get("params"):
            QMessageBox.information(self, "提示", "当前特征类型没有可配置的参数。")
            return

        param_definitions = self.feature_registry[feature_key]["params"]
        current_values = self.feature_params.get(feature_key, {})
        dialog = GenericFeatureParamsDialog(param_definitions, current_values)
        if dialog.exec_() == QDialog.Accepted:
            self.feature_params[feature_key] = dialog.get_params()

    def on_feature_type_changed(self):
        feature_key = self.feature_combo.currentData()
        has_params = feature_key and self.feature_registry[feature_key].get("params")
        self.feature_params_btn.setEnabled(bool(has_params))

    def on_strategy_radio_changed(self):
        is_fixed_checked = self.fixed_threshold_radio.isChecked()
        self.threshold_edit.setEnabled(is_fixed_checked)

    def on_filter_toggled(self):
        is_checked = self.filter_checkbox.isChecked()
        self.low_freq_edit.setEnabled(is_checked)
        self.high_freq_edit.setEnabled(is_checked)

    def populate_ui_from_config(self):
        if not self.load_config:
            return

        self.pattern_list = self.load_config.get("pattern_list", [])
        self.refresh_data_view()

        feature_type = self.load_config.get("feature_type")
        if feature_type:
            index = self.feature_combo.findData(feature_type)
            if index >= 0:
                self.feature_combo.setCurrentIndex(index)

        feature_params = self.load_config.get("feature_params")
        if feature_params and feature_type in self.feature_params:
            self.feature_params[feature_type] = feature_params

        if self.load_config.get("apply_filter"):
            self.filter_checkbox.setChecked(True)
            filter_range = self.load_config.get("filter_range_hz", [0, 5000])
            self.low_freq_edit.setText(str(filter_range[0]))
            self.high_freq_edit.setText(str(filter_range[1]))

        metric = self.load_config.get("similarity_metric")
        if metric:
            index = self.similarity_metric_combo.findData(metric)
            if index >= 0:
                self.similarity_metric_combo.setCurrentIndex(index)

        strategy = self.load_config.get("threshold_strategy")
        if strategy == "adaptive_threshold":
            self.adaptive_threshold_radio.setChecked(True)
        else:
            self.fixed_threshold_radio.setChecked(True)
            threshold_value = self.load_config.get("threshold_value", 0.9)
            self.threshold_edit.setText(str(threshold_value))

    def refresh_data_view(self):
        self.data_view.model().setRowCount(0)
        for idx, pattern in enumerate(self.pattern_list):
            self.data_view.model().setItem(idx, 0, QStandardItem(pattern["clip_path"]))
            pattern_len = np.round(pattern["clip_len"] / self.sample_rate, 3)
            self.data_view.model().setItem(idx, 1, QStandardItem(str(pattern_len)))
        self.n_chosen_pattern_label.setText("已加载模板： %s" % len(self.pattern_list))
        self.data_view.horizontalHeader().setSectionResizeMode(3)
        width = self.data_view.columnWidth(0)
        self.data_view.horizontalHeader().setSectionResizeMode(0)
        self.data_view.setColumnWidth(0, width + 60)

    def get_config(self):
        feature_key = self.feature_combo.currentData()
        # start_time, end_time = self.selected_region_time
        # start_frame = int(start_time * self.sample_rate) if start_time is not None else 0
        # end_frame = int(end_time * self.sample_rate) if end_time is not None else 0
        config = {
            # "audio_file_path": self.audio_file_path,
            # "pattern_save_path": self.pattern_save_path,
            "pattern_list": self.pattern_list,
            "sample_rate": self.sample_rate,
            # "pattern_region_time": (start_frame, end_frame),
            # "pattern_duration_sec": end_frame - start_frame,
            "feature_type": feature_key,
            "feature_params": self.feature_params.get(feature_key, {}),
            "apply_filter": self.filter_checkbox.isChecked(),
            "filter_range_hz": (None, None),
            "algorithm": "dtw",
            "similarity_metric": self.similarity_metric_combo.currentData(),
            "threshold_strategy": "fixed_threshold" if self.fixed_threshold_radio.isChecked() else "adaptive_threshold",
            "threshold_value": None,
        }

        if self.filter_checkbox.isChecked():
            config["filter_range_hz"] = (int(self.low_freq_edit.text()), int(self.high_freq_edit.text()))

        if self.fixed_threshold_radio.isChecked():
            config["threshold_value"] = float(self.threshold_edit.text())
        return config

    def on_click_default_btn(self):
        config_data = self.get_config()
        if not self.validate_config(config_data):
            return
        # config_data["audio_file_path"] = FileOps.get_relative_path(self.audio_file_path, DEFAULT_DIR)
        # config_data["pattern_save_path"] = FileOps.get_relative_path(self.pattern_save_path, DEFAULT_DIR)
        save_flag = self.config_manager.save_default_config("PM", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config = self.get_config()
        if not self.validate_config(config):
            return None
        # try:
        #     start_time, end_time = self.selected_region_time
        #     start_sample = int(start_time * self.sample_rate)
        #     end_sample = int(end_time * self.sample_rate)
        #     pattern_data = self.audio_data[start_sample:end_sample]
        #     save_audio_simple(config["pattern_save_path"], pattern_data, self.sample_rate)
        #     config["audio_file_path"] = FileOps.get_relative_path(self.audio_file_path, DEFAULT_DIR)
        #     config["pattern_save_path"] = FileOps.get_relative_path(self.pattern_save_path, DEFAULT_DIR)
        # except Exception as e:
        #     QMessageBox.critical(self, "错误", f"保存模板文件失败:\n{e}")
        #     return

        self.config_data = config
        self.accept()
        return self.config_data

    def validate_config(self, config):
        if config["apply_filter"]:
            low, high = config["filter_range_hz"]
            if low is None or high is None or low >= high:
                QMessageBox.warning(self, "提示", "输入的频率范围无效，最低频率必须小于最高频率。")
                return False
        return True

    def eventFilter(self, watched, event):
        if watched is self.plot_widget.viewport():
            if event.type() in [QEvent.MouseMove, QEvent.HoverMove]:
                scene_pos = self.plot_widget.mapToScene(event.pos())
                viewbox_rect = self.plot_widget.getPlotItem().getViewBox().sceneBoundingRect()
                if viewbox_rect.contains(scene_pos):
                    self.plot_widget.viewport().setCursor(Qt.CrossCursor)
                else:
                    self.plot_widget.viewport().setCursor(Qt.ArrowCursor)
            elif event.type() == QEvent.Leave:
                self.plot_widget.viewport().setCursor(Qt.ArrowCursor)
        return super().eventFilter(watched, event)

import json
from datetime import datetime

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFrame, QWidget, QStackedWidget

from base.save_data import ensure_test_result_file
from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR


class SequenceCountBoard(QWidget):

    def __init__(self, analysis_config, parent=None):
        super(SequenceCountBoard, self).__init__(parent)

        self.analysis_config = analysis_config
        self.mode = str()

        self.test_btn = QPushButton("测试")
        self.mark_btn = QPushButton("标记")
        self.total_line_edit = QLineEdit()
        self.ok_line_edit = QLineEdit()
        self.ng_line_edit = QLineEdit()
        self.yield_line_edit = QLineEdit()
        self.model_line_edit = QLineEdit()
        self.datatime_line_edit = QLineEdit()
        self.mark_total_edit = QLineEdit("0")
        self.mark_ok_edit = QLineEdit("0")
        self.mark_ng_edit = QLineEdit("0")
        self.review_session_edit = QLineEdit("无")
        self.ok_btn = QPushButton(" OK ")
        self.ng_btn = QPushButton(" NG ")
        self.reset_btn = QPushButton("重置统计")
        self.review_session_widget = None

        self.set_lineedit()
        self.set_btn()

        self.init_ui()

        self.on_mark_btn_clicked()
        self.on_test_btn_clicked()

    def init_ui(self):
        mode_btn_layout = self.create_mode_btn_layout()

        separator_line = QFrame()
        separator_line.setFrameShape(QFrame.HLine)
        separator_line.setFrameShadow(QFrame.Sunken)
        separator_line.setStyleSheet("color: gray;")
        separator_line.setFixedHeight(2)

        test_widget = self.set_test_widget()
        mark_widget = self.set_mark_widget()

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(test_widget)
        self.stacked_widget.addWidget(mark_widget)

        layout = QVBoxLayout()
        layout.addLayout(mode_btn_layout)
        layout.addWidget(separator_line)
        layout.addWidget(self.stacked_widget)
        layout.addStretch()

        self.setLayout(layout)

        self.set_test_text()
        self.set_mark_text()

    def create_horizontal_layout(self, label_str, item):
        label = QLabel(label_str)

        layout = QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(item)

        return layout

    def set_lineedit(self):
        self.total_line_edit.setAlignment(Qt.AlignCenter)
        self.ok_line_edit.setAlignment(Qt.AlignCenter)
        self.ng_line_edit.setAlignment(Qt.AlignCenter)
        self.yield_line_edit.setAlignment(Qt.AlignCenter)
        self.model_line_edit.setAlignment(Qt.AlignCenter)
        self.datatime_line_edit.setAlignment(Qt.AlignCenter)
        self.mark_total_edit.setAlignment(Qt.AlignCenter)
        self.mark_ok_edit.setAlignment(Qt.AlignCenter)
        self.mark_ng_edit.setAlignment(Qt.AlignCenter)
        self.review_session_edit.setAlignment(Qt.AlignCenter)

        self.total_line_edit.setDisabled(True)
        self.ok_line_edit.setDisabled(True)
        self.ng_line_edit.setDisabled(True)
        self.yield_line_edit.setDisabled(True)
        self.model_line_edit.setDisabled(True)
        self.datatime_line_edit.setDisabled(True)
        self.mark_total_edit.setDisabled(True)
        self.mark_ok_edit.setDisabled(True)
        self.mark_ng_edit.setDisabled(True)
        self.review_session_edit.setDisabled(True)

        self.total_line_edit.setFixedSize(130, 35)
        self.ok_line_edit.setFixedSize(130, 35)
        self.ng_line_edit.setFixedSize(130, 35)
        self.yield_line_edit.setFixedSize(130, 35)
        self.model_line_edit.setFixedSize(130, 35)
        self.datatime_line_edit.setFixedSize(130, 35)
        self.mark_total_edit.setFixedSize(130, 35)
        self.mark_ok_edit.setFixedSize(130, 35)
        self.mark_ng_edit.setFixedSize(130, 35)
        self.review_session_edit.setFixedSize(170, 35)

    def set_btn(self):
        self.reset_btn.setStyleSheet(ui_style_const.qpushbutton_style)
        self.ok_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/green_circle.png"))
        self.ok_btn.setStyleSheet(ui_style_const.sequence_qpushbutton_style)
        self.ok_btn.setFixedSize(180, 80)
        self.ok_btn.setIconSize(QSize(24, 24))
        self.ng_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/red_circle.png"))
        self.ng_btn.setStyleSheet(ui_style_const.sequence_qpushbutton_style)
        self.ng_btn.setFixedSize(180, 80)
        self.ng_btn.setIconSize(QSize(24, 24))

    def create_mode_btn_layout(self):
        mode_label = QLabel("模式：")
        self.test_btn.setFixedSize(100, 35)
        self.mark_btn.setFixedSize(100, 35)
        self.test_btn.clicked.connect(self.on_test_btn_clicked)
        self.mark_btn.clicked.connect(self.on_mark_btn_clicked)

        model_button_layout = QHBoxLayout()
        model_button_layout.addWidget(mode_label)
        model_button_layout.addSpacing(70)
        model_button_layout.addWidget(self.test_btn)
        model_button_layout.addWidget(self.mark_btn)
        model_button_layout.setSpacing(0)

        return model_button_layout

    def set_test_widget(self):
        total_layout = self.create_horizontal_layout("总    数：", self.total_line_edit)
        ok_layout = self.create_horizontal_layout("OK    数：", self.ok_line_edit)
        ng_layout = self.create_horizontal_layout("NG    数：", self.ng_line_edit)
        yield_layout = self.create_horizontal_layout("合 格 率：", self.yield_line_edit)
        model_layout = self.create_horizontal_layout("当前模型：", self.model_line_edit)
        datatime_layout = self.create_horizontal_layout("录制日期：", self.datatime_line_edit)
        reset_btn_layout = QHBoxLayout()
        reset_btn_layout.addStretch()
        reset_btn_layout.addWidget(self.reset_btn)
        reset_btn_layout.addStretch()

        test_layout = QVBoxLayout()
        test_layout.addLayout(total_layout, stretch=1)
        test_layout.addLayout(ok_layout, stretch=1)
        test_layout.addLayout(ng_layout, stretch=1)
        test_layout.addLayout(yield_layout, stretch=1)
        test_layout.addLayout(model_layout, stretch=1)
        test_layout.addLayout(datatime_layout, stretch=1)
        test_layout.addLayout(reset_btn_layout, stretch=1)

        test_widget = QWidget()
        test_widget.setLayout(test_layout)

        return test_widget

    def set_mark_widget(self):
        review_label = QLabel("当前审核：")
        review_layout = QHBoxLayout()
        review_layout.addWidget(review_label)
        review_layout.addWidget(self.review_session_edit)
        self.review_session_widget = QWidget()
        self.review_session_widget.setLayout(review_layout)
        self.review_session_widget.hide()
        total_layout = self.create_horizontal_layout("总    数：", self.mark_total_edit)
        ok_layout = self.create_horizontal_layout("OK    数：", self.mark_ok_edit)
        ng_layout = self.create_horizontal_layout("NG    数：", self.mark_ng_edit)
        ok_btn_layout = QHBoxLayout()
        ok_btn_layout.addStretch()
        ok_btn_layout.addWidget(self.ok_btn)
        ok_btn_layout.addStretch()
        ng_btn_layout = QHBoxLayout()
        ng_btn_layout.addStretch()
        ng_btn_layout.addWidget(self.ng_btn)
        ng_btn_layout.addStretch()

        mark_layout = QVBoxLayout()
        mark_layout.addWidget(self.review_session_widget, stretch=1)
        mark_layout.addLayout(total_layout, stretch=1)
        mark_layout.addLayout(ok_layout, stretch=1)
        mark_layout.addLayout(ng_layout, stretch=1)
        mark_layout.addSpacing(10)
        mark_layout.addLayout(ok_btn_layout, stretch=1)
        mark_layout.addLayout(ng_btn_layout, stretch=1)

        mark_widget = QWidget()
        mark_widget.setLayout(mark_layout)

        return mark_widget
    
    def on_test_btn_clicked(self):
        default_ai_model = self.analysis_config.get("default_ai")
        if default_ai_model:
            analyse_model_name = self.analysis_config.get(default_ai_model, {}).get("analyse_model_name", None)
            self.model_line_edit.setText(analyse_model_name)
            self.model_line_edit.setCursorPosition(0)
            self.force_test_mode()
        else:
            self.on_mark_btn_clicked()
            return

    def on_mark_btn_clicked(self):
        self.stacked_widget.setCurrentIndex(1)
        self.mode = "mark"
        self.test_btn.setStyleSheet("background-color: #E0E0E0; color: #666666; border: none;")
        self.mark_btn.setStyleSheet("background-color: #007BFF; color: white; border: none;")
        self.mark_btn.setEnabled(False)
        self.test_btn.setEnabled(True)

    def set_test_text(self):
        current_time = datetime.now().strftime("%Y-%m-%d")
        ensure_test_result_file(self.analysis_config)
        test_result_path = DEFAULT_DIR + f"log/test_result_log/{current_time}.dat"
        with open(test_result_path, "r") as f:
            lines = f.readlines()
            total = lines[0].split(":")[1].strip()
            ok = lines[1].split(":")[1].strip()
            ng = lines[2].split(":")[1].strip()
            ok_percent = lines[3].split(":")[1].strip()
            current_model = lines[4].split(":")[1].strip()
            datatime = lines[5].split(":")[1].strip()
            self.total_line_edit.setText(total)
            self.ok_line_edit.setText(ok)
            self.ng_line_edit.setText(ng)
            self.yield_line_edit.setText(ok_percent)
            self.model_line_edit.setText(current_model)
            self.model_line_edit.setCursorPosition(0)
            self.datatime_line_edit.setText(datatime)

    def set_mark_text(self):
        mark_result_path = DEFAULT_DIR + "ui/ui_config/mark_result.json"
        with open(mark_result_path, "r") as f:
            data = json.load(f)
            self.mark_total_edit.setText(str(data["total"]))
            self.mark_ok_edit.setText(str(data["ok"]))
            self.mark_ng_edit.setText(str(data["ng"]))

    def set_test_result_file(self, params, analyse_model_name):
        current_time = datetime.now().strftime("%Y-%m-%d")
        ensure_test_result_file(self.analysis_config)
        test_result_path = DEFAULT_DIR + f"log/test_result_log/{current_time}.dat"
        lines = list()
        total = int(self.total_line_edit.text())
        ok = int(self.ok_line_edit.text())
        ng = int(self.ng_line_edit.text())

        if params == "OK":
            ok += 1
        elif params == "NG":
            ng += 1
        total += 1
        lines.append(f"total: {total}\n")
        lines.append(f"ok: {ok}\n")
        lines.append(f"ng: {ng}\n")
        ok_percent = round(ok / total * 100, 2) if total > 0 else 0
        lines.append(f"ok_percent: {ok_percent}%\n")
        if analyse_model_name:
            lines.append(f"current_model: {analyse_model_name}\n")
        lines.append(f"current_time: {current_time}\n")
        with open(test_result_path, "w") as f:
            f.writelines(lines)

    def set_mark_result_file(self, params):
        mark_result_path = DEFAULT_DIR + "ui/ui_config/mark_result.json"
        datatime = datetime.now().strftime("%Y-%m-%d")
        total = int(self.mark_total_edit.text())
        ok = int(self.mark_ok_edit.text())
        ng = int(self.mark_ng_edit.text())
        data = dict()
        total = total + 1
        if params == "OK":
            ok = ok + 1
        elif params == "NG":
            ng = ng + 1
        data["total"] = total
        data["ng"] = ng
        data["ok"] = ok
        data["datatime"] = datatime
        with open(mark_result_path, "w") as f:
            json.dump(data, f, indent=4)

    def set_review_session_text(self, text):
        self.review_session_edit.setText(text)
        self.review_session_edit.setCursorPosition(0)
        self.review_session_edit.setToolTip(text)

    def set_review_session_visible(self, visible):
        if self.review_session_widget is None:
            return
        self.review_session_widget.setVisible(visible)

    def set_mark_mode_enabled(self, enabled):
        self.mark_btn.setEnabled(enabled)
        if enabled:
            if self.mode != "mark":
                self.test_btn.setEnabled(True)
        else:
            self.mark_btn.setStyleSheet("background-color: #E0E0E0; color: #666666; border: none;")
            if self.mode == "mark":
                self.force_test_mode()
            self.mark_btn.setEnabled(False)

    def force_test_mode(self):
        default_ai_model = self.analysis_config.get("default_ai")
        if default_ai_model:
            analyse_model_name = self.analysis_config.get(default_ai_model, {}).get("analyse_model_name", None)
            self.model_line_edit.setText(analyse_model_name)
            self.model_line_edit.setCursorPosition(0)
        self.test_btn.setStyleSheet("background-color: #007BFF; color: white; border: none;")
        self.mark_btn.setStyleSheet("background-color: #E0E0E0; color: #666666; border: none;")
        self.stacked_widget.setCurrentIndex(0)
        self.mode = "test"
        self.test_btn.setEnabled(False)
        self.mark_btn.setEnabled(True)


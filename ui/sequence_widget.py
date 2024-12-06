import sys
import time
from binascii import a2b_qp

from PyQt5.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QGridLayout, QApplication, QSpacerItem, \
    QSizePolicy, QHBoxLayout, QVBoxLayout

from consts import ui_style_const


class SequenceWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.collect_or_analyse_layout = QHBoxLayout()
        self.collect_layout = CollectWindow()
        self.analyse_layout = AnalyseWindow()
        self.sequence_layout = QVBoxLayout()
        self.serial_num = 1
        self.player_icon_flag = False
        self.label_s_or_n_count = QLabel("1")
        self.collect_btn = QPushButton(" 采  集 ")
        self.analyse_btn = QPushButton(" 分  析 ")
        self.player_btn = QPushButton()
        self.widget_flag = True
        self.init_ui()

    def init_ui(self):

        button_layout = self.create_title_btn_layout()
        layout_data = self.create_data_layout()

        self.sequence_layout.addLayout(button_layout)
        self.sequence_layout.addLayout(layout_data)
        self.sequence_layout.addWidget(self.collect_layout)
        self.sequence_layout.setAlignment(Qt.AlignCenter)

        self.collect_layout.next_btn.clicked.connect(self.swap_analyse_widget)

        self.setLayout(self.sequence_layout)

        self.setStyleSheet(ui_style_const.qlabel_stytle +
                           ui_style_const.qpushbutton_stytle +
                           ui_style_const.qlineedit_stytle)

    def create_title_btn_layout(self):
        button_layout = QHBoxLayout()
        h_spacer_btn_center = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Minimum)
        h_spacer_btn_left = QSpacerItem(30, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)
        h_spacer_btn_right = QSpacerItem(30, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.collect_btn.setStyleSheet("background-color: #a9d18e; color: white;")
        self.analyse_btn.setStyleSheet("background-color: #4472c4; color: white;")
        self.collect_btn.clicked.connect(self.swap_collect_widget)
        self.analyse_btn.clicked.connect(self.swap_analyse_widget)
        button_layout.addItem(h_spacer_btn_left)
        button_layout.addWidget(self.collect_btn)
        button_layout.addItem(h_spacer_btn_center)
        button_layout.addWidget(self.analyse_btn)
        button_layout.addItem(h_spacer_btn_right)

        return button_layout

    def create_data_layout(self):
        label_type = QLabel(" 型 号 ")
        label_type.setStyleSheet("background-color: #4472c4; color: white;border: 1px solid rgb(173, 173, 173);")
        label_type.setFixedHeight(25)
        lineedit_type = QLineEdit("S004-1")
        lineedit_type.setAlignment(Qt.AlignCenter)
        label_s_or_n = QLabel("  S/N  ")
        label_s_or_n.setStyleSheet("background-color: #4472c4; color: white;border: 1px solid rgb(173, 173, 173);")
        label_s_or_n.setFixedHeight(25)
        self.label_s_or_n_count.setAlignment(Qt.AlignCenter)
        self.label_s_or_n_count.setStyleSheet("background-color: white; border: None")
        self.label_s_or_n_count.setMinimumSize(100, 12)
        self.label_s_or_n_count.setMaximumSize(100, 21)
        self.player_btn.setFixedSize(40, 40)
        self.player_btn.setStyleSheet("border-radius: 20px;border: 1px solid rgb(173, 173, 173);")
        self.player_btn.setIcon(QIcon("./ui_pic/sequence_pic/bofang.png"))
        self.player_btn.setIconSize(QSize(40, 40))
        self.player_btn.clicked.connect(self.clicked_player_btn)

        h_spacer_1 = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Expanding)
        h_spacer_2 = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Expanding)
        h_spacer_3 = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Expanding)
        h_spacer_4 = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Expanding)

        data_layout = QHBoxLayout()
        data_layout.addWidget(label_type)
        data_layout.addItem(h_spacer_1)
        data_layout.addWidget(lineedit_type)
        data_layout.addItem(h_spacer_2)
        data_layout.addWidget(label_s_or_n)
        data_layout.addItem(h_spacer_3)
        data_layout.addWidget(self.label_s_or_n_count)
        data_layout.addItem(h_spacer_4)
        data_layout.addWidget(self.player_btn)
        return data_layout

    def create_collect_or_analyse_layout(self):
        if self.widget_flag:
            if self.collect_layout not in globals():
                self.collect_layout = CollectWindow()
            self.sequence_layout.replaceWidget(self.analyse_layout, self.collect_layout)
            self.analyse_layout.deleteLater()
            self.sequence_layout.addWidget(self.collect_layout)
            self.analyse_btn.setStyleSheet("background-color: #4472c4; color: white;")
            self.collect_btn.setStyleSheet("background-color: #a9d18e; color: white;")
        else:
            if self.analyse_layout not in globals():
                self.analyse_layout = AnalyseWindow()
            self.sequence_layout.replaceWidget(self.collect_layout, self.analyse_layout)
            self.collect_layout.deleteLater()
            self.sequence_layout.addWidget(self.analyse_layout)
            self.analyse_btn.setStyleSheet("background-color: #a9d18e; color: white;")
            self.collect_btn.setStyleSheet("background-color: #4472c4; color: white;")
        self.collect_layout.next_btn.clicked.connect(self.swap_analyse_widget)
        self.analyse_layout.ok_btn.clicked.connect(self.clicked_ok_or_ng)
        self.analyse_layout.ng_btn.clicked.connect(self.clicked_ok_or_ng)
        # self.analyse_layout.analyse_btn.clicked(self.clicked_analyse_btn)

    def swap_analyse_widget(self):
        if not self.widget_flag:
            return
        self.widget_flag = False
        self.create_collect_or_analyse_layout()

    def swap_collect_widget(self):
        if self.widget_flag:
            return
        self.widget_flag = True
        self.create_collect_or_analyse_layout()

    def clicked_ok_or_ng(self):
        self.serial_num +=1
        self.label_s_or_n_count.setNum(self.serial_num)
        self.swap_collect_widget()
        self.player_icon_flag = False
        self.update_player_icon()

    def clicked_player_btn(self):
        self.player_icon_flag = True
        self.player_btn.setDisabled(True)
        self.worker = WorkerThread()
        self.worker.update_signal.connect(self.update_player_icon)
        self.worker.start()


    def update_player_icon(self):
        if self.player_icon_flag:
            self.player_btn.setIcon(QIcon("./ui_pic/sequence_pic/chongbo.png"))
            self.player_btn.setIconSize(QSize(25, 25))
        else:
            self.player_btn.setIcon(QIcon("./ui_pic/sequence_pic/bofang.png"))
            self.player_btn.setIconSize(QSize(40, 40))
        self.player_btn.setDisabled(False)


class CollectWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.next_btn = QPushButton("下\n一\n步")
        self.init_ui()

    def init_ui(self):
        self.next_btn.setStyleSheet("background-color: #4472c4; color: white;")

        layout = QHBoxLayout()
        line_widget = QWidget()
        line_widget.setMinimumSize(200, 200)
        line_widget.setStyleSheet("border: 1px solid rgb(173, 173, 173);")

        # Todo: add line chart to widget

        self.next_btn.setMaximumWidth(30)
        self.next_btn.setMinimumHeight(100)
        layout.addWidget(line_widget)
        layout.addWidget(self.next_btn)

        self.setLayout(layout)


class AnalyseWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.analyse_btn = QPushButton(" 分 析 ")
        self.ai_analyse_score_lineedit = QLineEdit()
        self.ok_btn = QPushButton(" OK ")
        self.ng_btn = QPushButton(" NG ")
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        self.setStyleSheet("border: 1px solid rgb(173, 173, 173);")
        signal_analyse_widget = QWidget()
        signal_analyse_widget.setFixedSize(210, 200)
        ai_analyse_widget = QWidget()
        ai_analyse_layout = self.create_ai_analyse_widget()
        # ai_analyse_widget.setFixedSize(210, 200)
        ai_analyse_widget.setLayout(ai_analyse_layout)

        btn_layout = QVBoxLayout()
        self.ok_btn.setIcon(QIcon("./ui_pic/sequence_pic/lvseyuan.png"))
        self.ok_btn.setStyleSheet("background-color: #4472c4; color: white;")
        self.ng_btn.setIcon(QIcon("./ui_pic/sequence_pic/hongseyuan.png"))
        self.ng_btn.setStyleSheet("background-color: #4472c4; color: white;")
        self.ok_btn.setMaximumWidth(100)
        self.ng_btn.setMaximumWidth(100)
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.ng_btn)

        layout.addWidget(signal_analyse_widget)
        layout.addWidget(ai_analyse_widget)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def create_ai_analyse_widget(self):
        ai_analyse_layout = QVBoxLayout()

        ai_title_layout = QHBoxLayout()
        title_label = QLabel("AI分析")
        title_label.setStyleSheet("border: None")
        h_title_space = QSpacerItem(30, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)
        ai_title_layout.addWidget(title_label)
        ai_title_layout.addItem(h_title_space)

        model_layout = QHBoxLayout()
        model_label = QLabel("模型")
        model_label.setStyleSheet("background-color: #4472c4; color: white;")
        model_lineedit = QLineEdit("/model/001.keras")
        model_lineedit.setAlignment(Qt.AlignCenter)
        model_layout.addWidget(model_label)
        model_layout.addWidget(model_lineedit)

        analyse_btn_layout =QHBoxLayout()
        self.analyse_btn.setStyleSheet("background-color: #4472c4; color: white;")
        h_analyse_btn_space_left = QSpacerItem(30, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)
        h_analyse_btn_space_right = QSpacerItem(30, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)
        analyse_btn_layout.addItem(h_analyse_btn_space_left)
        analyse_btn_layout.addWidget(self.analyse_btn)
        analyse_btn_layout.addItem(h_analyse_btn_space_right)

        analyse_score_layout = QHBoxLayout()
        self.ai_analyse_score_lineedit.setPlaceholderText("AI 评分")
        self.ai_analyse_score_lineedit.setAlignment(Qt.AlignCenter)
        self.ai_analyse_score_lineedit.setDisabled(True)
        analyse_score_layout.addWidget(self.ai_analyse_score_lineedit)
        analyse_score_layout.setContentsMargins(20, 0, 20, 0)

        v_ai_analyse_center_space = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Expanding)
        v_ai_analyse_bottom_space = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Expanding)

        ai_analyse_layout.addLayout(ai_title_layout)
        ai_analyse_layout.addLayout(model_layout)
        ai_analyse_layout.addItem(v_ai_analyse_center_space)
        ai_analyse_layout.addLayout(analyse_btn_layout)
        ai_analyse_layout.addLayout(analyse_score_layout)
        ai_analyse_layout.addItem(v_ai_analyse_bottom_space)

        return ai_analyse_layout


class WorkerThread(QThread):
    update_signal = pyqtSignal()  # 用来发送更新信号

    def run(self):
        # 模拟耗时操作
        time.sleep(5)
        self.update_signal.emit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SequenceWindow()
    # window = CollectWindow()
    window.show()
    app.exec()

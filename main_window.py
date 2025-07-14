import sys

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor
from PyQt5.QtWidgets import QAction, QApplication, QLabel, QMainWindow, QStatusBar, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QHBoxLayout, QSpacerItem, QSizePolicy, QPushButton, QMenuBar, QMessageBox

from base.log_manager import LogManager
from base.db_manager import DataSave
from consts import ui_style_const
from consts.model_consts import DATABASE_PATH
from consts.running_consts import DEFAULT_DIR
from ui.ai_window import AiWindow
from ui.audio_data_manage_dialog import audioDataManageDialog
from ui.calibration_window import CalibrationWindow
from ui.hardware_window import HardwareWindow, get_default_device
from ui.login_window import AddAccountWindow, ChangePwdWindow, LoginWindow
from ui.sequence_widget import SequenceWindow
from ui.stimulus_window import StimulusWindow
from ui.analysis_model_select_dialog import AnalysisModelSelect


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        # set up statusbar object data
        self.user_name = None
        self.access_lvl = None
        self.refresh_stimulus_flag = None
        self.mic = get_default_device("mic")
        self.speaker = get_default_device("speaker")

        # set mouse drog date
        self.resize_direction = None
        self.last_pos = QPoint()
        self.wid = None
        self.heigh = None
        self.window_x = None
        self.window_y = None

        # reset mouse event
        self.mousePressEvent = self.mousepressevent
        self.mouseReleaseEvent = self.mousereleaseevent
        self.mouseMoveEvent = self.mousemoveevent

        # set the menubar action
        self.function_action_stimulus = QAction("激励信号", self)
        self.function_action_test_sequence = QAction("分析队列", self)
        self.function_action_ai_training = QAction("训练AI模型", self)
        self.function_audio_manager = QAction("音频数据管理", self)
        self.function_action_exit = QAction("退出", self)
        self.hardware_action_selection = QAction("硬件选择", self)
        self.hardware_action_calibration = QAction("校准", self)
        self.user_action_switch_account = QAction("切换用户", self)
        self.user_action_add_account = QAction("添加用户", self)
        self.user_action_change_pwd = QAction("修改密码", self)
        # set the operator and engineer and admin power
        self.widget_list_operator = [self.user_action_change_pwd]
        self.widget_list_engineer = self.widget_list_operator + [
            self.function_action_stimulus,
            self.function_action_test_sequence,
            self.function_action_ai_training,
            self.hardware_action_selection,
            self.hardware_action_calibration,
        ]
        self.widget_list_admin = self.widget_list_engineer + [self.user_action_add_account]

        self.init_ui()

    def init_ui(self):
        # initialize the window layout
        self.set_title()
        self.init_menu()
        self.init_sequence_widget()
        self.sequence_window.close()
        self.on_access_lvl_changed()
        self.show_statusbar_layout()
        self.showMaximized()
        self.on_login_window_init()

    def set_title(self):
        # hide the window title bar and reset the window title bar
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        title_layout = QHBoxLayout()
        title_btn_layout = self.set_title_btn()
        icon_label = QLabel()
        icon_label.setStyleSheet("background-color: transparent")
        title_icon = QPixmap(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico")
        icon_label.setPixmap(title_icon)
        icon_label.setFixedSize(25, 25)
        icon_label.setScaledContents(True)
        current_version = self.get_current_version()
        title_label = QLabel(f"谛听异音检测 -{current_version} beta")
        h_spacer = QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        title_layout.addWidget(icon_label)
        title_layout.addWidget(title_label)
        title_layout.addItem(h_spacer)
        title_layout.addLayout(title_btn_layout)
        self.setMinimumSize(1030, 760)
        title_layout.setContentsMargins(10, 3, 15, 0)
        self.setStyleSheet(
            ui_style_const.qlabel_stytle + ui_style_const.qpushbutton_stytle + ui_style_const.qmainwindow_stytle
        )
        self.get_current_version()

        return title_layout

    @staticmethod
    def get_current_version():
        with DataSave(DATABASE_PATH) as db:
            current_version = db.query_matching_data([("current_version",)], "system_info_table", ["name"], ["value"])
            return current_version[0][0]

    def set_title_btn(self):
        # create three button, include minimize, switch size and close
        self.min_btn = QPushButton()
        self.min_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/main_window_pic/minsize.svg"))
        self.min_btn.setStyleSheet("border: None; background-color: transparent")
        self.min_btn.clicked.connect(self.showMinimized)
        self.max_flag = True
        self.max_btn = QPushButton()
        self.max_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/main_window_pic/normalsize.svg"))
        self.max_btn.clicked.connect(self.show_window_size)
        self.max_btn.setStyleSheet("border: None; background-color: transparent")
        self.close_btn = QPushButton()
        self.close_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/main_window_pic/close.svg"))
        self.close_btn.setStyleSheet("border: None; background-color: transparent")
        self.close_btn.clicked.connect(self.close)

        self.max_btn.setMouseTracking(True)
        self.min_btn.setMouseTracking(True)
        self.close_btn.setMouseTracking(True)

        title_btn_layout = QHBoxLayout()
        title_btn_layout.addWidget(self.min_btn)
        title_btn_layout.addWidget(self.max_btn)
        title_btn_layout.addWidget(self.close_btn)
        title_btn_layout.setSpacing(20)

        return title_btn_layout

    def show_window_size(self):
        # change the window size, and update the mouse tracking
        if self.max_flag:
            self.max_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/main_window_pic/maxsize.svg"))
            self.showNormal()
            self.sequence_window.setMouseTracking(True)
            self.setMouseTracking(True)
            self.statusBar().setMouseTracking(True)
            self.max_flag = False
        else:
            self.max_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/main_window_pic/normalsize.svg"))
            self.showMaximized()
            self.max_flag = True
            self.sequence_window.setMouseTracking(False)
            self.setMouseTracking(False)
            self.statusBar().setMouseTracking(False)

    def init_sequence_widget(self):
        # create sequence widget, and set main window layout
        main_window = QWidget()
        layout = QVBoxLayout()
        self.sequence_window = SequenceWindow()
        menu_bar = self.init_menu()
        title_layout = self.set_title()
        layout.addLayout(title_layout)
        layout.addSpacing(5)
        layout.addWidget(menu_bar)
        layout.addSpacing(1)
        layout.addWidget(self.sequence_window)
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        main_window.setLayout(layout)
        main_window.setMouseTracking(True)  # local variable start mouse tracking
        self.setCentralWidget(main_window)
        # transmit the mic and speaker to sequence widget
        self.sequence_window.mic = self.mic
        self.sequence_window.speaker = self.speaker

    def init_menu(self):
        # create menu bar, and link the menu bar to action
        menu_bar = QMenuBar()
        menu_bar.setStyleSheet(ui_style_const.main_window_menubar_stytle)
        function_menu = menu_bar.addMenu("功能")
        hardware_menu = menu_bar.addMenu("硬件")
        user_menu = menu_bar.addMenu("用户")
        help_menu = menu_bar.addMenu("帮助")

        function_menu.addAction(self.function_action_stimulus)
        self.function_action_stimulus.triggered.disconnect()
        self.function_action_stimulus.triggered.connect(self.on_stimulus_window_init)
        function_menu.addAction(self.function_action_test_sequence)
        self.function_action_test_sequence.triggered.disconnect()
        self.function_action_test_sequence.triggered.connect(self.analysis_model_select)
        function_menu.addSeparator()
        function_menu.addAction(self.function_action_ai_training)
        self.function_action_ai_training.triggered.disconnect()
        self.function_action_ai_training.triggered.connect(self.on_ai_window_init)
        function_menu.addAction(self.function_audio_manager)
        self.function_audio_manager.triggered.disconnect()
        self.function_audio_manager.triggered.connect(self.on_audio_manager_init)
        function_menu.addSeparator()

        function_menu.addAction(self.function_action_exit)
        self.function_action_exit.triggered.disconnect()
        self.function_action_exit.triggered.connect(self.on_window_close)
        hardware_menu.addAction(self.hardware_action_selection)
        self.hardware_action_selection.triggered.disconnect()
        self.hardware_action_selection.triggered.connect(self.on_hardware_window_init)
        hardware_menu.addAction(self.hardware_action_calibration)
        self.hardware_action_calibration.triggered.disconnect()
        self.hardware_action_calibration.triggered.connect(self.on_calibration_window_init)

        user_menu.addAction(self.user_action_switch_account)
        self.user_action_switch_account.triggered.disconnect()
        self.user_action_switch_account.triggered.connect(self.on_login_window_init)
        user_menu.addAction(self.user_action_add_account)
        self.user_action_add_account.triggered.disconnect()
        self.user_action_add_account.triggered.connect(self.on_add_account_window_init)
        user_menu.addAction(self.user_action_change_pwd)
        self.user_action_change_pwd.triggered.disconnect()
        self.user_action_change_pwd.triggered.connect(self.on_change_pwd_window_init)

        return menu_bar

    def on_stimulus_window_init(self):
        # set the speaker test audio
        dlg = StimulusWindow()
        if dlg.is_close_window:
            QMessageBox.warning(self, "警告", "请先设置校准参数")
            return
        dlg.speaker = self.speaker
        self.refresh_stimulus_flag = dlg.on_exec()
        self.sequence_window.refresh_stimulus_flag = self.refresh_stimulus_flag

    def analysis_model_select(self):
        # Test items for configuring speakers
        analysis_model_select_dialog = AnalysisModelSelect()
        analysis_model_select_dialog.exec()

    def show_statusbar_layout(self):
        # create status bar, show the user data and device data, and close drag status bar modify window size
        self.user_label = QLabel()
        self.user_label.setAlignment(Qt.AlignLeft)
        self.user_label.setStyleSheet(ui_style_const.qlabel_stytle)
        self.user_label.setText(
            "当前用户：{name}  用户等级：{level}".format(name=self.user_name, level=self.access_lvl)
        )
        self.device_label = QLabel()
        self.device_label.setStyleSheet(ui_style_const.qlabel_stytle)
        device_txt = "麦克风：{mic}  扬声器：{speaker}".format(mic=self.mic["name"], speaker=self.speaker["name"])
        self.device_label.setText(device_txt)

        statusbar = QStatusBar()
        statusbar.setSizeGripEnabled(False)
        statusbar.addWidget(self.user_label)
        statusbar.addPermanentWidget(self.device_label)
        self.setStatusBar(statusbar)

    def update_statusbar(self):
        # update the status bar data
        device_txt = "麦克风：{mic}  扬声器：{speaker}".format(mic=self.mic["name"], speaker=self.speaker["name"])
        self.device_label.setText(device_txt)
        self.user_label.setText(
            "当前用户：{name}  用户等级：{level}".format(name=self.user_name, level=self.access_lvl)
        )

    @staticmethod
    def on_audio_manager_init():
        dlg = audioDataManageDialog(LogManager.set_log_handler("core"))
        dlg.exec()

    @staticmethod
    def on_ai_window_init():
        dlg = AiWindow(LogManager.set_log_handler("train"))
        dlg.exec()

    def on_access_lvl_changed(self):
        # Set the executable function according to the user level
        widget_dict = {
            "Operator": self.widget_list_operator,
            "Engineer": self.widget_list_engineer,
            "Admin": self.widget_list_admin,
        }
        for widget in self.widget_list_admin:
            widget.setDisabled(True)
        for widget in widget_dict.get(self.access_lvl, []):
            widget.setEnabled(True)

    def on_login_window_init(self):
        # check the user info, if the user info is correct, then show the main window
        dlg = LoginWindow()
        access_lvl, user_name = dlg.on_exec()
        if access_lvl is not None:
            self.access_lvl, self.user_name = access_lvl, user_name
            self.sequence_window.show()
            self.update_statusbar()
        self.on_access_lvl_changed()

    @staticmethod
    def on_add_account_window_init():
        # add a new user
        dlg = AddAccountWindow(LogManager.set_log_handler("core"))
        dlg.exec()

    def on_change_pwd_window_init(self):
        # change the password
        dlg = ChangePwdWindow(self.user_name, LogManager.set_log_handler("core"))
        dlg.exec()

    def on_hardware_window_init(self):
        # select the mic and speaker
        dlg = HardwareWindow()
        self.speaker, self.mic = dlg.on_exec()
        self.update_statusbar()
        self.sequence_window.mic = self.mic
        self.sequence_window.speaker = self.speaker

    def on_calibration_window_init(self):
        # calibration the mic and speaker
        dlg = CalibrationWindow()
        dlg.speaker = self.speaker
        dlg.exec()

    def on_window_close(self):
        # close the window
        self.close()

    def closeEvent(self, event):
        if hasattr(SequenceWindow, "tcp_server") and SequenceWindow.tcp_server:
            SequenceWindow.tcp_server.stop()
            SequenceWindow.tcp_server = None
        event.accept()

    def mousepressevent(self, event):
        # If the mouse is pressed, recoed mouse move data, start the window resizing
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            self.wid = self.width()
            self.heigh = self.height()
            self.window_x = self.x()
            self.window_y = self.y()
            self.last_pos = event.globalPos()
            event.accept()

    def mousereleaseevent(self, event):
        # If the mouse is released, stop the window resizing and clear the resize direction
        if event.button() == Qt.LeftButton:
            self.resize_direction = None
            event.accept()

    def geometry_window(
        self, window_x, window_y, pos: QPoint, is_update_width: bool, is_update_height: bool, resize_direction: str
    ):
        if resize_direction == "right":
            self.setGeometry(window_x, window_y, self.wid + pos.x(), self.heigh)
        elif resize_direction == "left" and is_update_width:
            self.setGeometry(window_x + pos.x(), window_y, self.wid - pos.x(), self.heigh)
        elif resize_direction == "bottom":
            self.setGeometry(window_x, window_y, self.wid, self.heigh + pos.y())
        elif resize_direction == "top" and is_update_height:
            self.setGeometry(window_x, window_y + pos.y(), self.wid, self.heigh - pos.y())
        elif resize_direction == "right_top" and is_update_height:
            self.setGeometry(window_x, window_y + pos.y(), self.wid + pos.x(), self.heigh - pos.y())
        elif resize_direction == "right_bottom":
            self.setGeometry(window_x, window_y, self.wid + pos.x(), self.heigh + pos.y())
        elif resize_direction == "left_top":
            if is_update_width and is_update_height:
                self.setGeometry(window_x + pos.x(), window_y + pos.y(), self.wid - pos.x(), self.heigh - pos.y())
            elif is_update_width and not is_update_height:
                self.setGeometry(window_x + pos.x(), window_y, self.wid - pos.x(), self.heigh)
            elif not is_update_width and is_update_height:
                self.setGeometry(window_x, window_y + pos.y(), self.wid, self.heigh - pos.y())
        elif resize_direction == "left_bottom" and is_update_width:
            self.setGeometry(window_x + pos.x(), window_y, self.wid - pos.x(), self.heigh + pos.y())

    def updata_cursor_base_direction(self, left: str, top: str, right: str, bottom: str):
        if right and top:
            self.resize_direction = "right_top"
            self.setCursor(Qt.SizeBDiagCursor)
        elif right and bottom:
            self.resize_direction = "right_bottom"
            self.setCursor(Qt.SizeFDiagCursor)
        elif left and top:
            self.resize_direction = "left_top"
            self.setCursor(Qt.SizeFDiagCursor)
        elif left and bottom:
            self.resize_direction = "left_bottom"
            self.setCursor(Qt.SizeBDiagCursor)
        elif right:
            self.resize_direction = "right"
            self.setCursor(Qt.SizeHorCursor)
        elif bottom:
            self.resize_direction = "bottom"
            self.setCursor(Qt.SizeVerCursor)
        elif left:
            self.resize_direction = "left"
            self.setCursor(Qt.SizeHorCursor)
        elif top:
            self.resize_direction = "top"
            self.setCursor(Qt.SizeVerCursor)
        else:
            self.resize_direction = None
            self.setCursor(Qt.ArrowCursor)

    def mousemoveevent(self, event):
        # record the window size, the window position, and the mouse position
        width = self.width()
        height = self.height()
        right = event.pos().x() >= width - 3 and event.pos().x() <= width
        left = event.pos().x() <= 3 and event.pos().x() >= 0
        top = event.pos().y() <= 3 and event.pos().y() >= 0
        bottom = event.pos().y() >= height - 3 and event.pos().y() <= height

        if event.buttons() & Qt.LeftButton:
            # Move the window
            if not self.max_flag and not self.resize_direction:
                self.move(event.globalPos() - self.drag_position)
            # clicked the window border, resize the window according to the direction
            pos = event.globalPos() - self.last_pos
            is_update_width = width - pos.x() > 1030
            is_update_height = height - pos.y() > 760
            self.geometry_window(
                self.window_x, self.window_y, pos, is_update_width, is_update_height, self.resize_direction
            )
        else:
            # Determine whether you can drag the window size, If so, record the drag direction and set the cursor stutle
            self.updata_cursor_base_direction(left, top, right, bottom)
        event.accept()

    def paintEvent(self, event):
        # Set the window Background-color
        painter = QPainter(self)
        width = self.width()
        height = self.height()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(208, 206, 202))
        painter.drawRect(1, 1, width - 2, 31)
        painter.setBrush(QColor(208, 206, 202, 124))
        painter.drawRect(1, 31, width - 2, 41)
        painter.drawRect(1, height - 24, width - 2, 23)
        painter.end()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

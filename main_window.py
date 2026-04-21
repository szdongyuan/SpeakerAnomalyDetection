import sys

from PyQt5.QtCore import Qt, QPoint, QTimer, QUrl
from PyQt5.QtGui import QIcon, QPixmap, QDesktopServices
from PyQt5.QtWidgets import QApplication, QMainWindow, QStatusBar, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QHBoxLayout, QSpacerItem, QSizePolicy, QDialog

from base.log_manager import LogManager
from base.db_manager import DataSave
from base.sound_device_manager import SoundDeviceManager
from consts.model_consts import DATABASE_PATH
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.widgets import PushButton, MenuBar, Label, Action, MessageBox
from ui.ai_window import AiWindow
from ui.archive_audio_data_dialog import ArchiveAudioDataDialog
from ui.calibration_window import CalibrationWindow
from ui.hardware_window import HardwareWindow
from ui.login_window import AddAccountWindow, ChangePwdWindow, LoginWindow
from ui.operation_sequence import AnalysisModelSelect
from ui.sequence.sequence_widget import SequenceWindow
from ui.custom_ui_widget.traypopuppanel import TrayPopupButton
from ui.ui_src import ui_resources


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        # set up statusbar object data
        self.user_name = None
        self.access_lvl = None
        self.refresh_stimulus_flag = None
        startup_devices = SoundDeviceManager().get_startup_devices()
        self.mic = startup_devices.get("mic")
        self.speaker = startup_devices.get("speaker")
        self.startup_device_fallback_targets = startup_devices.get("fallback_targets", [])
        self.startup_device_notice_message = startup_devices.get("startup_notice_message")

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
        self.function_action_test_sequence = Action("测试队列", self)
        self.function_action_ai_training = Action("训练AI模型", self)
        self.function_audio_manager = Action("音频数据管理", self)
        self.function_action_exit = Action("退出", self)
        self.hardware_action_selection = Action("硬件选择", self)
        self.hardware_action_calibration = Action("校准", self)
        self.user_action_switch_account = Action("切换用户", self)
        self.user_action_add_account = Action("添加用户", self)
        self.user_action_change_pwd = Action("修改密码", self)
        # set the operator and engineer and admin power
        self.widget_list_operator = [self.user_action_change_pwd]
        self.widget_list_engineer = self.widget_list_operator + [
            self.function_action_test_sequence,
            self.function_action_ai_training,
            self.hardware_action_selection,
            self.hardware_action_calibration,
        ]
        self.widget_list_admin = self.widget_list_engineer + [self.user_action_add_account]

        self.tray_popup_button: TrayPopupButton = None

        self.init_ui()

    def init_ui(self):
        # initialize the window layout
        self.init_sequence_widget()
        self.sequence_window.close()
        self.on_access_lvl_changed()
        self.show_statusbar_layout()
        self.showMaximized()
        self.on_login_window_init()
        if self.startup_device_notice_message or self.startup_device_fallback_targets:
            QTimer.singleShot(0, self.show_startup_device_warning)

    def set_title(self):
        # hide the window title bar and reset the window title bar
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        title_layout = QHBoxLayout()
        title_btn_layout = self.set_title_btn()
        icon_label = Label()
        icon_label.setObjectName("icon_label")
        title_icon = QPixmap(":/ui/icon/ting.ico")
        icon_label.setPixmap(title_icon)
        icon_label.setFixedSize(25, 25)
        icon_label.setScaledContents(True)
        current_version = self.get_current_version()
        title_label = Label(f"谛听异音检测 -{current_version} beta")
        h_spacer = QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        title_layout.addWidget(icon_label)
        title_layout.addWidget(title_label)
        title_layout.addItem(h_spacer)
        title_layout.addLayout(title_btn_layout)
        self.setMinimumSize(1030, 760)
        title_layout.setContentsMargins(10, 3, 15, 0)

        title_widget = QWidget()
        title_widget.setObjectName("TitleWidget")
        title_widget.setLayout(title_layout)
        return title_widget

    @staticmethod
    def get_current_version():
        with DataSave(DATABASE_PATH) as db:
            current_version = db.query_matching_data([("current_version",)], "system_info_table", ["name"], ["value"])
            return current_version[0][0]

    def set_title_btn(self):
        # create three button, include minimize, switch size and close
        self.min_btn = PushButton()
        self.min_btn.setIcon(QIcon(":/ui/icon/minus.png"))
        self.min_btn.clicked.connect(self.showMinimized)
        self.max_flag = True
        self.max_btn = PushButton()
        self.max_btn.setIcon(QIcon(":/ui/icon/restore.png"))
        self.max_btn.clicked.connect(self.show_window_size)
        self.close_btn = PushButton()
        self.close_btn.setIcon(QIcon(":/ui/icon/fork.png"))
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
            self.max_btn.setIcon(QIcon(":/ui/icon/maximize.png"))
            self.showNormal()
            self.sequence_window.setMouseTracking(True)
            self.setMouseTracking(True)
            self.statusBar().setMouseTracking(True)
            self.max_flag = False
        else:
            self.max_btn.setIcon(QIcon(":/ui/icon/restore.png"))
            self.showMaximized()
            self.max_flag = True
            self.sequence_window.setMouseTracking(False)
            self.setMouseTracking(False)
            self.statusBar().setMouseTracking(False)

    def init_sequence_widget(self):
        # create sequence widget, and set main window layout
        main_window = QWidget()
        main_window.setObjectName("MainWindow")
        layout = QVBoxLayout()
        self.sequence_window = SequenceWindow()
        menu_bar = self.init_menu()
        title_widget = self.set_title()
        layout.addWidget(title_widget)
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
        menu_bar = MenuBar()
        function_menu = menu_bar.addMenu("功能")
        hardware_menu = menu_bar.addMenu("硬件")
        user_menu = menu_bar.addMenu("用户")
        help_action = menu_bar.addAction("帮助")
        help_action.triggered.connect(self.on_help_open_website)

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

    @staticmethod
    def on_help_open_website():
        # Open company website in system default browser
        QDesktopServices.openUrl(QUrl("https://suzhoudongyuan.com/"))

    def analysis_model_select(self):
        # Test items for configuring speakers
        analysis_model_select_dialog = AnalysisModelSelect(
            self.sequence_window.using_config_path, mic=self.mic, speaker=self.speaker
        )
        analysis_model_select_dialog.exec()
        # Refresh active sequence config without forcing mode switch
        self.sequence_window.on_sequence_config_updated()

    def show_statusbar_layout(self):
        # create status bar, show the user data and device data, and close drag status bar modify window size
        self.user_label = Label()
        self.user_label.setAlignment(Qt.AlignLeft)
        self.tray_popup_button = TrayPopupButton()
        self.update_statusbar()

        statusbar = QStatusBar()
        statusbar.setObjectName("StatusBar")
        statusbar.setSizeGripEnabled(False)
        statusbar.addWidget(self.user_label)
        statusbar.addPermanentWidget(self.tray_popup_button)
        self.setStatusBar(statusbar)

    def update_statusbar(self):
        # update the status bar data
        mic_name = self.mic["name"] if self.mic else "无可用输入设备"
        speaker_name = self.speaker["name"] if self.speaker else "无可用输出设备"
        self.tray_popup_button.set_in_device(mic_name)
        self.tray_popup_button.set_out_device(speaker_name)
        self.user_label.setText(
            "当前用户：{name}  用户等级：{level}".format(name=self.user_name, level=self.access_lvl)
        )

    @staticmethod
    def on_audio_manager_init():
        dlg = ArchiveAudioDataDialog(LogManager.set_log_handler("core"))
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
        # Prevent hardware changes during playback/recording
        if self.sequence_window.player_status_flag:
            MessageBox.warning(self, "提示", "播放或录音进行中，请等待完成后再修改硬件设置")
            return

        dlg = HardwareWindow(self.speaker, self.mic)
        self.speaker, self.mic = dlg.on_exec()
        self.update_statusbar()
        self.sequence_window.mic = self.mic
        self.sequence_window.speaker = self.speaker

    def show_startup_device_warning(self):
        if self.startup_device_notice_message:
            MessageBox.warning(self, "提示", self.startup_device_notice_message)
            return

        MessageBox.warning(self, "提示", "已保存的音频设备不存在或配置无效，正在使用系统默认麦克风和扬声器。")

    def on_calibration_window_init(self):
        # calibration the mic and speaker
        dlg = CalibrationWindow()
        dlg.speaker = self.speaker
        dlg.exec()
        if dlg.input_calibration_flag:
            self.sequence_window.update_v2pa_factor()

    def on_window_close(self):
        # close the window
        self.close()

    def _close_all_subwindows(self):
        """
        Close all other top-level windows/dialogs besides the main window itself.

        This is a best-effort cleanup to avoid leaving orphan dialogs/tool windows
        alive when the main window exits.
        """
        try:
            top_levels = QApplication.topLevelWidgets()
        except Exception:
            top_levels = []

        for w in top_levels:
            if w is None or w is self:
                continue
            # Only close visible windows to avoid touching already-closed widgets
            # that may still be referenced by Qt/Python.
            try:
                if hasattr(w, "isVisible") and w.isVisible():
                    w.close()
            except Exception:
                # Best-effort: ignore any close errors
                pass

    def closeEvent(self, event):
        if hasattr(SequenceWindow, "tcp_server") and SequenceWindow.tcp_server:
            SequenceWindow.tcp_server.stop()
            SequenceWindow.tcp_server = None

        # Close any other sub windows/dialogs that may still be open.
        self._close_all_subwindows()

        # Best-effort: rebuild daily Excel from CSV spool before exit (fast_mode).
        # Retry loop if there are failures (e.g., Excel file is open)
        if hasattr(self, "sequence_window") and self.sequence_window is not None:
            while True:
                # Show "saving" dialog
                saving_dialog = QDialog(self)
                saving_dialog.setWindowTitle("正在保存")
                saving_dialog.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
                saving_dialog.setFixedSize(250, 80)
                layout = QVBoxLayout(saving_dialog)
                label = Label("正在保存数据，请稍候...")
                label.setAlignment(Qt.AlignCenter)
                layout.addWidget(label)
                saving_dialog.show()
                QApplication.processEvents()

                try:
                    failures = self.sequence_window.flush_excel_spool_build(on_close=False)
                except Exception as e:
                    failures = [("unknown", str(e))]

                saving_dialog.close()

                if not failures:
                    break

                msg_box = MessageBox(self)
                msg_box.setIcon(MessageBox.Warning)
                msg_box.setWindowTitle("Excel同步失败")
                msg_box.setText("无法将数据同步到Excel文件，可能是文件被占用或权限不足。\n请关闭相关Excel文件后重试。")
                retry_btn = msg_box.addButton("重试", MessageBox.AcceptRole)
                msg_box.addButton("忽略", MessageBox.RejectRole)
                msg_box.setDefaultButton(retry_btn)
                msg_box.exec_()

                if msg_box.clickedButton() == retry_btn:
                    continue
                else:
                    break

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

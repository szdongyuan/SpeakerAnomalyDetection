import sys

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor
from PyQt5.QtWidgets import QAction, QApplication, QLabel, QMainWindow, QStatusBar, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QHBoxLayout, QSpacerItem, QSizePolicy, QPushButton, QMenuBar, QMessageBox, QDialog

from base.log_manager import LogManager
from base.db_manager import DataSave
from base.hardware_selection import restore_or_default, save_if_changed
from base.sound_device_manager import SoundDeviceManager
from consts import ui_style_const
from consts.model_consts import DATABASE_PATH
from consts.running_consts import DEFAULT_DIR
from ui.archive_audio_data_dialog import ArchiveAudioDataDialog
from ui.calibration_window import CalibrationWindow
from ui.hardware_window import open_hardware_selection_window
from ui.login_window import AddAccountWindow, ChangePwdWindow, LoginWindow
from ui.operation_sequence import AnalysisModelSelect
from ui.product_test_project_config_dialog import ProductTestProjectConfigDialog
from ui.sequence.sequence_widget import SequenceWindow


class MainWindow(QMainWindow):

    def __init__(self, *, recording_bridge=None):
        super().__init__()
        if recording_bridge is None:
            from base.recording_service import RecordingService
            from ui.recording_service_bridge import RecordingServiceBridge
            recording_bridge = RecordingServiceBridge(RecordingService(), self)
        self.recording_bridge = recording_bridge
        QApplication.instance().aboutToQuit.connect(self.recording_bridge.shutdown)
        # set up statusbar object data
        self.user_name = None
        self.access_lvl = None
        self.refresh_stimulus_flag = None
        # Restore the operator's last hardware choice from
        # ``configs/hardware_selection.json``. Each side falls back
        # independently when the saved device cannot be matched against
        # current hardware: missing mic -> OS default + In1 only
        # (PaError-9998 safety); missing speaker -> OS default + all
        # channels. Any I/O failure degrades to the same all-defaults
        # path, so a corrupt file can never block startup.
        (
            self.mic,
            self.speaker,
            self.mic_channels,
            self.speaker_channels,
        ) = restore_or_default()

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
        self.function_action_product_test_program = QAction("产品测试程序配置", self)
        self.function_action_test_sequence = QAction("测试队列", self)
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
            self.function_action_product_test_program,
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
        self.sequence_window.refresh_channel_windows()

    def set_title(self):
        # hide the window title bar and reset the window title bar
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        title_bar = QWidget()
        title_bar.setObjectName("mainWindowTitleRow")
        title_bar.setFixedHeight(31)
        title_bar.setStyleSheet(ui_style_const.main_window_title_row_style)
        title_layout = QHBoxLayout(title_bar)
        title_btn_layout = self.set_title_btn()
        icon_label = QLabel()
        icon_label.setStyleSheet("background-color: transparent")
        title_icon = QPixmap(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico")
        icon_label.setPixmap(title_icon)
        icon_label.setFixedSize(25, 25)
        icon_label.setScaledContents(True)
        current_version = self.get_current_version()
        title_label = QLabel(f"希听异音检测 -{current_version} beta")
        title_label.setStyleSheet(ui_style_const.main_window_title_label_style)
        h_spacer = QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        title_layout.addWidget(icon_label)
        title_layout.addWidget(title_label)
        title_layout.addItem(h_spacer)
        title_layout.addLayout(title_btn_layout)
        self.setMinimumSize(1030, 760)
        title_layout.setContentsMargins(10, 3, 15, 0)
        self.setStyleSheet(
            ui_style_const.main_window_base_style
        )
        self.get_current_version()

        return title_bar

    @staticmethod
    def get_current_version():
        with DataSave(DATABASE_PATH) as db:
            current_version = db.query_matching_data([("current_version",)], "system_info_table", ["name"], ["value"])
            return current_version[0][0]

    def set_title_btn(self):
        # create three button, include minimize, switch size and close
        self.min_btn = QPushButton()
        self.min_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/main_window_pic/minsize.svg"))
        self.min_btn.setStyleSheet(ui_style_const.main_window_title_button_style)
        self.min_btn.clicked.connect(self.showMinimized)
        self.max_flag = True
        self.max_btn = QPushButton()
        self.max_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/main_window_pic/normalsize.svg"))
        self.max_btn.clicked.connect(self.show_window_size)
        self.max_btn.setStyleSheet(ui_style_const.main_window_title_button_style)
        self.close_btn = QPushButton()
        self.close_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/main_window_pic/close.svg"))
        self.close_btn.setStyleSheet(ui_style_const.main_window_close_button_style)
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
        self.sequence_window = SequenceWindow(recording_bridge=self.recording_bridge)
        menu_bar = self.init_menu()
        title_bar = self.set_title()
        menu_row = self._create_menu_row(menu_bar)
        layout.addWidget(title_bar)
        layout.addWidget(menu_row)
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
        self.sequence_window.mic_channels = self.mic_channels
        self.sequence_window.speaker_channels = self.speaker_channels
        self.sequence_window.update_v2pa_factor()

    def _expand_sequence_workspace(self):
        """Let the logged-in workspace consume the remaining window height."""
        main_layout = self.centralWidget().layout()
        sequence_index = main_layout.indexOf(self.sequence_window)
        main_layout.setAlignment(Qt.Alignment())
        main_layout.setStretch(sequence_index, 1)


    @staticmethod
    def _create_menu_row(menu_bar):
        menu_row = QWidget()
        menu_row.setObjectName("mainWindowMenuRow")
        menu_row.setMinimumHeight(29)
        menu_row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        menu_row.setStyleSheet(ui_style_const.main_window_menu_row_style)

        menu_bar.setObjectName("mainWindowMenuBar")
        menu_bar.setNativeMenuBar(False)
        menu_bar.setMinimumHeight(27)
        menu_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        menu_bar.setContentsMargins(0, 0, 0, 0)

        menu_layout = QHBoxLayout(menu_row)
        menu_layout.setContentsMargins(0, 0, 0, 0)
        menu_layout.setSpacing(0)
        menu_layout.addWidget(menu_bar)
        return menu_row

    def init_menu(self):
        # create menu bar, and link the menu bar to action
        menu_bar = QMenuBar()
        menu_bar.setObjectName("mainWindowMenuBar")
        menu_bar.setNativeMenuBar(False)
        menu_bar.setStyleSheet(ui_style_const.main_window_menubar_style)
        function_menu = menu_bar.addMenu("功能")
        hardware_menu = menu_bar.addMenu("硬件")
        user_menu = menu_bar.addMenu("用户")
        help_menu = menu_bar.addMenu("帮助")

        function_menu.addAction(self.function_action_product_test_program)
        self.function_action_product_test_program.triggered.disconnect()
        self.function_action_product_test_program.triggered.connect(
            self.on_product_test_program_config
        )
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

    def analysis_model_select(self):
        # Test items for configuring speakers
        if getattr(
            self.sequence_window,
            "_analysis_round_config_locked",
            False,
        ):
            QMessageBox.information(
                self,
                "配置已锁定",
                "当前轮次尚未完成，暂时不能修改测试队列配置。",
            )
            return
        self._open_analysis_model_select(self.sequence_window.using_config_path)

    def _open_analysis_model_select(self, using_config_path):
        analysis_model_select_dialog = AnalysisModelSelect(
            using_config_path,
            mic=self.mic,
            speaker=self.speaker,
            mic_channels=self.mic_channels,
            speaker_channels=self.speaker_channels,
        )
        analysis_model_select_dialog.exec()
        # Refresh active sequence config without forcing mode switch
        self.sequence_window.on_sequence_config_updated()

    def on_product_test_program_config(self):
        if getattr(
            self.sequence_window,
            "_analysis_round_config_locked",
            False,
        ):
            QMessageBox.information(
                self,
                "配置已锁定",
                "当前轮次尚未完成，暂时不能修改产品测试配置。",
            )
            return
        self.sequence_window._product_test_program_config_dialog_open = True
        try:
            dialog = ProductTestProjectConfigDialog(
                None,
                self._open_analysis_model_select,
                self,
            )
            dialog.programs_changed.connect(
                self.sequence_window.on_product_test_program_updated
            )
            dialog.exec()
        finally:
            self.sequence_window._product_test_program_config_dialog_open = False

    def show_statusbar_layout(self):
        # create status bar, show the user data and device data, and close drag status bar modify window size
        self.user_label = QLabel()
        self.user_label.setAlignment(Qt.AlignLeft)
        self.user_label.setStyleSheet(ui_style_const.main_window_status_label_style)
        self.device_label = QLabel()
        self.device_label.setStyleSheet(ui_style_const.main_window_status_label_style)
        self.update_statusbar()

        statusbar = QStatusBar()
        statusbar.setSizeGripEnabled(False)
        statusbar.setStyleSheet(ui_style_const.main_window_statusbar_style)
        statusbar.addWidget(self.user_label)
        statusbar.addPermanentWidget(self.device_label)
        self.setStatusBar(statusbar)

    def update_statusbar(self):
        # update the status bar data
        mic_name = self.mic["name"] if self.mic else "无可用输入设备"
        speaker_name = self.speaker["name"] if self.speaker else "无可用输出设备"
        device_txt = "麦克风：{mic}  扬声器：{speaker}".format(mic=mic_name, speaker=speaker_name)
        self.device_label.setText(device_txt)
        self.user_label.setText(
            "当前用户：{name}  用户等级：{level}".format(name=self.user_name, level=self.access_lvl)
        )

    @staticmethod
    def on_audio_manager_init():
        dlg = ArchiveAudioDataDialog(LogManager.set_log_handler("core"))
        dlg.exec()

    @staticmethod
    def on_ai_window_init():
        from ui.ai_window import AiWindow

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
            self._expand_sequence_workspace()
            self.sequence_window.show()
            if hasattr(self.sequence_window, "init_serial_trigger_runtime"):
                self.sequence_window.init_serial_trigger_runtime()
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
            QMessageBox.warning(self, "提示", "播放或录音进行中，请等待完成后再修改硬件设置")
            return
        # 将当前驱动/设备/通道作为初始值回填到硬件选择窗口
        driver_name = None
        try:
            if self.speaker and self.speaker.get("hostapi") is not None:
                driver_name = SoundDeviceManager.get_api_info(int(self.speaker.get("hostapi"))).get("name")
            elif self.mic and self.mic.get("hostapi") is not None:
                driver_name = SoundDeviceManager.get_api_info(int(self.mic.get("hostapi"))).get("name")
        except Exception:
            driver_name = None

        (
            accepted,
            self.speaker,
            self.speaker_channels,
            self.mic,
            self.mic_channels,
        ) = open_hardware_selection_window(
            driver=driver_name,
            speaker_device=self.speaker,
            speaker_channels=self.speaker_channels,
            mic_device=self.mic,
            mic_channels=self.mic_channels,
        )
        # Only persist on explicit OK. Cancel must be a strict no-op for
        # disk state, even when the JSON is currently missing/corrupt and
        # the in-memory state differs from what is on disk.
        if accepted:
            save_if_changed(
                self.mic, self.speaker, self.mic_channels, self.speaker_channels
            )
        self.update_statusbar()
        self.sequence_window.mic = self.mic
        self.sequence_window.speaker = self.speaker
        self.sequence_window.mic_channels = self.mic_channels
        self.sequence_window.speaker_channels = self.speaker_channels
        self.sequence_window.update_v2pa_factor()
        self.sequence_window.refresh_channel_windows()

    def on_calibration_window_init(self):
        # calibration the mic and speaker
        dlg = CalibrationWindow(
            input_device=self.mic,
            input_channels=self.mic_channels,
            recording_bridge=getattr(self, "recording_bridge", None),
        )
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

    def _shutdown_product_pdf_exporter_before_exit(self):
        sequence_window = getattr(self, "sequence_window", None)
        shutdown_product_pdf = getattr(
            sequence_window,
            "_shutdown_product_pdf_exporter",
            None,
        )
        if callable(shutdown_product_pdf):
            shutdown_product_pdf()

    def closeEvent(self, event):
        sequence = getattr(self, "sequence_window", None)
        has_pending_analysis = getattr(
            sequence,
            "_analysis_has_pending_tasks",
            None,
        )
        if callable(has_pending_analysis) and has_pending_analysis():
            event.ignore()
            QMessageBox.information(
                self,
                "分析任务未完成",
                "还有分析任务未完成，请等待分析结束后再退出。",
            )
            return
        bridge = getattr(self, "recording_bridge", None)
        if (bridge is not None and not bridge.service.closed.is_set()
                and not getattr(self, "_recording_shutdown_reported", False)):
            event.ignore()
            if not getattr(self, "_recording_close_requested", False):
                self._recording_close_requested = True
                sequence = getattr(self, "sequence_window", None)
                cancel = getattr(sequence, "_cancel_process_recording", None)
                if callable(cancel):
                    cancel()
                self.setEnabled(False)
                bridge.shutdown(self._finish_recording_shutdown)
            return
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
                label = QLabel("正在保存数据，请稍候...")
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

                msg_box = QMessageBox(self)
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setWindowTitle("Excel同步失败")
                msg_box.setText("无法将数据同步到Excel文件，可能是文件被占用或权限不足。\n请关闭相关Excel文件后重试。")
                retry_btn = msg_box.addButton("重试", QMessageBox.AcceptRole)
                msg_box.addButton("忽略", QMessageBox.RejectRole)
                msg_box.setDefaultButton(retry_btn)
                msg_box.exec_()

                if msg_box.clickedButton() == retry_btn:
                    continue
                else:
                    break

        self._shutdown_product_pdf_exporter_before_exit()
        event.accept()

    def _finish_recording_shutdown(self):
        self._recording_shutdown_reported = True
        if not self.recording_bridge.service.closed.is_set():
            # The bounded service callback confirms worker death, not every
            # parent reader/path release. Report honestly and permit app exit;
            # no pending audio is moved/deleted and no lease is fabricated.
            diagnostics = "\n".join(self.recording_bridge.service.diagnostics[-5:])
            QMessageBox.warning(self, "录音资源清理未完成",
                "录音进程已停止，部分文件资源尚未释放。退出后请检查这些文件；本次不再移动或删除它们。\n" + diagnostics)
        self.close()

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
        painter.setBrush(QColor(ui_style_const.COLOR_PAGE_BG))
        painter.drawRect(1, 1, width - 2, height - 2)
        painter.setBrush(QColor(ui_style_const.COLOR_TITLE_BAR_BG))
        painter.drawRect(1, 1, width - 2, 31)
        painter.setBrush(QColor(ui_style_const.COLOR_MENU_BAR_BG))
        painter.drawRect(1, 31, width - 2, 29)
        painter.setBrush(QColor(ui_style_const.COLOR_MENU_BAR_BG))
        painter.drawRect(1, height - 24, width - 2, 23)
        painter.end()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

import re
import weakref

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QVBoxLayout, QMessageBox

from base.load_config import LoadUiConfig
from base.save_data import save_recorded_data_to_json
from base.utils.custom_signals import sign
from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR


class SequenceWidgetUiOpsMixin:

    def showEvent(self, event):
        """
        MainWindow shows SequenceWindow only after login success.
        Defer the missing-config prompt until the first showEvent.
        """
        super().showEvent(event)
        self._missing_config_prompt_enabled = True
        # try:
        #     self.refresh_channel_windows()
        # except Exception:
        #     pass
        if not self.sequence_config and not self._missing_config_prompted:
            QMessageBox.warning(
                self,
                "提示",
                "当前未找到可用配置文件。\n"
                "请在上方【使用配置】下拉框中选择配置；\n"
                "如无可选项，请到【功能-测试队列】中保存或导入配置。",
            )
            self._missing_config_prompted = True

    def refresh_channel_windows(self):
        """
        Refresh plot subwindows based on current mic_channels selection.

        MainWindow assigns mic_channels after SequenceWindow construction, so this should be
        called at least once after the window is shown, and again after hardware selection changes.
        """
        channels = []
        try:
            channels = list(getattr(self, "mic_channels", []) or [])
        except Exception:
            channels = []
        if not channels:
            channels = [0]

        self._active_input_channels = [int(x) for x in channels]

        if self.channel_workspace is not None:
            self.channel_workspace.set_channels(self._active_input_channels)
        try:
            self.default_logger.info(f"Plot workspace channels: {self._active_input_channels}")
        except Exception:
            pass

    def init_ui(self):
        """
        Initializes the user interface of the SequenceWindow.

        This method sets up the window icon, minimum height, and creates the main layout
        by adding toolbar and waveform layouts. It also connects button click events to
        their respective handlers and applies style sheets to the widgets.
        """
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumHeight(700)
        waveform_layout = self.create_waveform_layout()

        sequence_layout = QVBoxLayout()
        sequence_layout.addWidget(self.toolsbar)
        sequence_layout.addLayout(waveform_layout)
        sequence_layout.setAlignment(Qt.AlignCenter)
        sequence_layout.setContentsMargins(1, 0, 1, 0)

        self.add_file_to_using_file_combobox()

        self.setLayout(sequence_layout)

        # When test-queue config is confirmed, refresh combobox + reload config first.
        # (No global signal dependency; MainWindow calls on_sequence_config_updated after dialog closes.)
        # Streaming audio chunk signal for real-time waveform updates
        sign.stream_audio_chunk_signal.connect(self.on_audio_chunk_received, Qt.AutoConnection)
        # Register this instance as current target for TCP callbacks
        self.__class__._active_instance_ref = weakref.ref(self)
        self.setStyleSheet(
            ui_style_const.qcombobox_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qlineedit_style
            + ui_style_const.qframe_style
            + ui_style_const.qlabel_style
            + ui_style_const.qcheckbox_style
        )

    def set_member_connect(self):
        self.player_btn.clicked.connect(lambda: self.on_clicked_player_btn())
        self.replayer_btn.clicked.connect(lambda: self.judge_play_and_record(is_replay=True))
        self.data_btn.clicked.connect(lambda: self.run(show_windows=True))
        self.lineedit_type.editingFinished.connect(lambda: self.lineedit_type_lose_focus(self.lineedit_type))
        self.lineedit_count.editingFinished.connect(lambda: self.lineedit_count_lose_focus(self.lineedit_count))
        self.lineedit_count.returnPressed.connect(lambda: self.validate_count(self.lineedit_count, True))

        # 扫码键盘楔入模式：信号交给 BarcodeRouter 处理
        self.lineedit_s_or_n.returnPressed.connect(self._barcode_router.on_barcode_return_pressed)
        self.lineedit_s_or_n.textChanged.connect(self._barcode_router.on_barcode_text_changed)
        if getattr(self, "left_panel", None) is not None:
            self.lineedit_s_or_n.textChanged.connect(self.left_panel.set_current_barcode)

        self.barcode_scanner_box.clicked.connect(self.clicked_scanner)
        self.tcp_btn.clicked.connect(self.on_tcp_btn_clicked)
        self.serial_trigger_btn.clicked.connect(self.on_serial_trigger_btn_clicked)
        self.count_board.ok_btn.clicked.connect(self.clicked_ok_or_ng)
        self.count_board.ng_btn.clicked.connect(self.clicked_ok_or_ng)
        # “重置统计”按钮：重置测试计数 + 恢复重播/分析按钮状态
        self.count_board.reset_btn.clicked.connect(self.on_reset_statistics_clicked)
        self.count_board.mark_btn.clicked.connect(self.on_mark_btn_clicked)
        self.using_file_combobox.currentTextChanged.connect(self.on_using_file_combobox_changed)

    def on_mark_btn_clicked(self):
        self.data_struct.store_wave_data = None
        self.data_struct.store_wave_data_multi = None
        self._clear_plot_area()
        self._close_analysis_windows()
        try:
            self._reset_barcode_commit_dedup()
        except Exception:
            pass
        self.player_btn.setDisabled(False)
        self.replayer_btn.setDisabled(True)
        self.data_btn.setDisabled(True)

    def init_lineedit_text(self):
        last_recorded_info = LoadUiConfig().load_last_recorded_info(self.default_logger)
        if last_recorded_info:
            product_model = last_recorded_info.get("product_model", "S004-1")
        else:
            product_model = "S004-1"
        self.lineedit_type.setText(product_model)
        # 型号/计数：默认只读（单击进入编辑态），避免扫码枪后缀(Tab/Enter)导致焦点跳转/误触发
        try:
            self.lineedit_type.setReadOnly(True)
        except Exception:
            pass

        result, _ = LoadUiConfig.load_recorded_num_from_json(self.default_logger)
        if result is None:
            self.current_recorded_count = 1
        else:
            self.current_recorded_count = result

        self.lineedit_count.setText(str(self.current_recorded_count))
        try:
            self.lineedit_count.setReadOnly(True)
        except Exception:
            pass

        if getattr(self, "left_panel", None) is not None:
            self.left_panel.set_current_barcode(self.lineedit_s_or_n.text())

    @property
    def player_btn(self):
        return self.toolsbar.player_btn

    @property
    def replayer_btn(self):
        return self.toolsbar.replayer_btn

    @property
    def data_btn(self):
        return self.toolsbar.data_btn

    @property
    def using_file_combobox(self):
        return self.toolsbar.using_file_combobox

    @property
    def lineedit_type(self):
        return self.toolsbar.lineedit_type

    @property
    def lineedit_count(self):
        return self.toolsbar.lineedit_count

    @property
    def lineedit_s_or_n(self):
        return self.toolsbar.lineedit_s_or_n

    @property
    def barcode_scanner_box(self):
        return self.toolsbar.barcode_scanner_box

    @property
    def tcp_btn(self):
        return self.toolsbar.tcp_btn

    @property
    def serial_trigger_btn(self):
        return self.toolsbar.serial_trigger_btn

    @property
    def serial_trigger_status_label(self):
        return self.toolsbar.serial_trigger_status_label

    @property
    def serial_trigger_code_label(self):
        return self.toolsbar.serial_trigger_code_label

    def lineedit_count_lose_focus(self, lineedit):
        self.current_recorded_count = int(lineedit.text())
        save_recorded_data_to_json(
            self.lineedit_type.text(),
            self.lineedit_count.text(),
            self.lineedit_s_or_n.text(),
            self.barcode_scanner_box.isChecked(),
        )
        # 退出编辑态：回到只读
        try:
            lineedit.setReadOnly(True)
        except Exception:
            pass
        lineedit.clearFocus()
        if lineedit.text() == "":
            result_count, _ = LoadUiConfig.load_recorded_num_from_json(self.default_logger)
            lineedit.setText(str(result_count))

    def lineedit_type_lose_focus(self, lineedit):
        save_recorded_data_to_json(
            self.lineedit_type.text(),
            self.lineedit_count.text(),
            self.lineedit_s_or_n.text(),
            self.barcode_scanner_box.isChecked(),
        )
        # 退出编辑态：回到只读
        try:
            lineedit.setReadOnly(True)
        except Exception:
            pass
        lineedit.clearFocus()
        if lineedit.text() == "":
            last_recorded_info = LoadUiConfig().load_last_recorded_info(self.default_logger)
            lineedit.setText(str(last_recorded_info.get("product_model", "S004-1")))

    def validate_count(self, lineedit, is_s_or_n: bool):
        """
            Validates the count input from the user.

            This method checks if the user input in the lineedit is a valid number. If the input is not a number,
            it restores the previously recorded number. If the input is valid, it updates the recorded number and saves
        it to a file.

            Parameters:
            lineedit (QLineEdit): The QLineEdit object containing the user's count input.
        """
        s_or_n_count = lineedit.text()
        result_count, result_scanner_barcode = LoadUiConfig.load_recorded_num_from_json(self.default_logger)
        reg = None
        if is_s_or_n:
            reg = r"^[0-9]*$"
        else:
            reg = r"^[0-9a-zA-Z]*$"

        if not re.match(reg, s_or_n_count):
            if is_s_or_n:
                lineedit.setText(str(result_count))
            else:
                lineedit.setText(str(result_scanner_barcode))
        elif s_or_n_count != "":
            if is_s_or_n:
                self.lineedit_s_or_n.setText("")
        if s_or_n_count == "":
            if is_s_or_n:
                lineedit.setText(str(result_count))
            else:
                lineedit.setText(str(result_scanner_barcode))

    def update_player_btn_is_playing(self):
        self.player_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/pause.png"))
        self.player_btn.setIconSize(QSize(35, 35))
        self.player_btn.setDisabled(True)

    def update_player_btn_is_paused(self):
        self.player_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/play.png"))
        self.player_btn.setIconSize(QSize(35, 35))
        can_start = bool(getattr(self, "sequence_config", None))
        can_start = can_start and not getattr(self, "player_status_flag", False)
        can_start = can_start and not getattr(self, "_record_workflow_busy", False)
        self.player_btn.setDisabled(not can_start)

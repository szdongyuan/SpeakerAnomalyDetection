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

    @staticmethod
    def _normalize_saved_sequence_mode(mode_value):
        mode = str(mode_value or "").strip().lower()
        return mode if mode in ("test", "mark") else ""

    def _current_condition_mode(self):
        mode = self._normalize_saved_sequence_mode(getattr(getattr(self, "count_board", None), "mode", ""))
        return mode or "test"

    def _apply_condition_mode_to_waveforms(self, mode=None):
        channel_workspace = getattr(self, "channel_workspace", None)
        if channel_workspace is None or not hasattr(channel_workspace, "set_mode"):
            return
        channel_workspace.set_mode(self._normalize_saved_sequence_mode(mode) or self._current_condition_mode())

    def _sync_condition_mode_combobox_from_count_board(self):
        combo = getattr(getattr(self, "toolsbar", None), "condition_mode_combobox", None)
        if combo is None:
            return
        text = "标记" if self._current_condition_mode() == "mark" else "测试"
        was_blocked = combo.blockSignals(True)
        try:
            combo.setCurrentText(text)
        finally:
            combo.blockSignals(was_blocked)
        self._apply_condition_mode_to_waveforms()

    def _persist_sequence_page_state(self, sequence_mode=None):
        """
        Persist product_model / scanner_barcode (and optionally sequence_mode).

        We intentionally do NOT write ``scanner_barcode_check`` here; that field is
        owned exclusively by the S/N checkbox click handler so unrelated callers
        (mode switch, type/edit lose-focus) can never overwrite it with a stale
        UI value.
        """
        normalized_mode = self._normalize_saved_sequence_mode(sequence_mode)
        save_recorded_data_to_json(
            product_model=self.lineedit_type.text(),
            scanner_barcode=self.lineedit_s_or_n.text(),
            sequence_mode=normalized_mode or None,
        )

    def _restore_last_sequence_mode(self):
        if self.count_board is None:
            return
        last_recorded_info = LoadUiConfig.load_last_recorded_info(self.default_logger)
        if not isinstance(last_recorded_info, dict):
            return
        saved_mode = self._normalize_saved_sequence_mode(last_recorded_info.get("sequence_mode"))
        if saved_mode == "mark":
            self.count_board.on_mark_btn_clicked()
            return
        if saved_mode != "test":
            return
        mode_state = self.count_board.get_mode_state() if hasattr(self.count_board, "get_mode_state") else {}
        can_enter_test_mode = bool((mode_state or {}).get("test_available", True))
        if can_enter_test_mode:
            self.count_board.on_test_btn_clicked()
        else:
            self.count_board.on_mark_btn_clicked()

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
        Refresh the fixed directional waveform windows.

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
            configure_waveform_workspace = getattr(self, "_configure_direction_waveform_workspace", None)
            if callable(configure_waveform_workspace):
                configure_waveform_workspace()
        try:
            self.default_logger.info(f"Directional waveform workspace ready, input channels: {self._active_input_channels}")
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
        if hasattr(self.using_file_combobox, "before_show_popup"):
            self.using_file_combobox.before_show_popup = self.update_using_file_combobox
        self.using_file_combobox.currentTextChanged.connect(self.on_using_file_combobox_changed)
        self.condition_mode_combobox.currentTextChanged.connect(self.on_condition_mode_combobox_changed)
        self._sync_condition_mode_combobox_from_count_board()

    def on_condition_mode_combobox_changed(self, text):
        target_mode = "mark" if str(text or "").strip() == "标记" else "test"
        if self.count_board is None:
            self._apply_condition_mode_to_waveforms(target_mode)
            return

        current_mode = self._current_condition_mode()
        if target_mode != current_mode and not self._confirm_mode_switch_if_round_incomplete():
            self._sync_condition_mode_combobox_from_count_board()
            return

        if target_mode == "mark":
            self.count_board.on_mark_btn_clicked()
        else:
            self.count_board.on_test_btn_clicked()
        self._sync_condition_mode_combobox_from_count_board()

    def _confirm_mode_switch_if_round_incomplete(self):
        has_incomplete_round = getattr(self, "_has_incomplete_manual_product_condition_round", None)
        if not callable(has_incomplete_round) or not has_incomplete_round():
            return True
        reply = QMessageBox.question(
            self,
            "切换模式",
            "本轮还未结束，是否切换模式？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def on_mark_btn_clicked(self):
        self.data_struct.store_wave_data = None
        self.data_struct.store_wave_data_multi = None
        clear_all_direction_waveforms = getattr(self, "clear_all_direction_waveforms", None)
        if callable(clear_all_direction_waveforms):
            clear_all_direction_waveforms()
        else:
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

        # Manual count is removed from workflow; keep the hidden widget inert.
        self.lineedit_count.setText("")
        self.lineedit_count.setReadOnly(True)

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
    def condition_mode_combobox(self):
        return self.toolsbar.condition_mode_combobox

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
    def serial_trigger_code_label(self):
        return self.toolsbar.serial_trigger_code_label

    def lineedit_count_lose_focus(self, lineedit):
        lineedit.setText("")
        self._persist_sequence_page_state()
        # 退出编辑态：回到只读
        try:
            lineedit.setReadOnly(True)
        except Exception:
            pass
        lineedit.clearFocus()
        if lineedit.text() == "":
            lineedit.setText("")

    def lineedit_type_lose_focus(self, lineedit):
        self._persist_sequence_page_state()
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

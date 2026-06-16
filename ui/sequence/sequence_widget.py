import json
import os
import re
import threading
import weakref
from datetime import datetime
import time

import librosa
import numpy as np
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QEvent, QSize, Qt, QTimer, QSignalBlocker, Q_ARG, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QLineEdit,
    QDialog,
    QLabel,
)
from base.data_struct.data_deal_struct import DataDealStruct
from base.excel_result_exporter import (
    ExportResult,
    build_excel_from_csv_spool,
    export_analysis_to_csv_spool,
    export_analysis_to_excel,
    resolve_excel_output_path,
    resolve_excel_spool_dir,
)
from base.file_ops import FileOps
from base.load_audio import load_audio_simple
from base.mes_result_exporter import _validate_mes_runtime_config, select_mes_export_config, write_mes_result
from base.shortcut_trigger_manager import ShortcutTriggerManager
from base.unified_hid_device_manager import UnifiedHardwareManager
from PyQt5.QtCore import QMetaObject
from base.utils.custom_signals import sign
from base.load_config import LoadUiConfig
from ui.sequence.barcode_router import BarcodeRouter
from base.log_manager import LogManager
from base.play_and_record import (
    get_recorded_info,
    stream_play_and_record,
    stream_record_without_play,
)
from base.recording_management import RecordingManager
from base.save_data import ensure_test_result_file, save_recorded_data_to_json, save_audio_simple
from base.soundcard_audio_processor import SoundcardAudioProcessor
from base.soundcard_calibration_manager import resolve_analysis_v2pa_factor_for_channel
from base.tcp_service import TcpServer, check_tcp_msg_format
from base.temp_tcp_client import TempTcpClient
from base.streaming_file_writer import StreamingWavWriter
from base.pre_processing.alignment_processing import AlignmentProcessing
from base.pre_processing.split_repeat_signal import SplitRepeatSignal
from base.acquisition_recording_defaults import normalize_play_record_detail, normalize_record_only_detail
from consts import ui_style_const, error_code
from consts.action_code import RequestTypeEnum
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.widgets import MessageBox
from ui.operation_sequence import AnalysisModelSelect
from ui.sequence.channel_plot_workspace import ChannelPlotWorkspace
from ui.sequence.sequence_tools_bar import SequenceToolsBar
from ui.sequence.sn_regex_manage_dialog import SnRegexManageDialog
from ui.signal_analysis_window import AnalysisResultSummaryWindow, get_class_mapping
from ui.sequence.sequencement_count_board import SequenceCountBoard
from ui.tcp_config_dialog import TcpConfigDialog
from ui.ui_src import ui_resources


class SequenceWindow(QWidget):
    tcp_server = None
    _active_instance_ref = None

    def __init__(self):
        """Initializes the class instance, setting up the user interface and necessary parameters."""
        super().__init__()
        self.data_struct = DataDealStruct()
        self.recorded_path = None
        self.count_board = None
        self.toolsbar = SequenceToolsBar()

        self.v2pa_factor = None
        self.analysis_types_requiring_v2pa = {"SPL", "SPLF", "HD", "RB", "PRB", "LP", "PD", "ED", "FBA"}
        self.using_config_path, self.registry = self.get_sequence_config_from_registry()
        self.sequence_config = list()
        self.analysis_config = dict()
        self.get_sequence_config_from_json()
        self.init_data_struct_stimulus_config()
        self.init_fft_and_stft_flag()
        self.signal_info = {}
        self.analysis_window = []
        self._analysis_result_summary_window = None
        self._excel_export_cache = None
        self._excel_exported_record_id = None
        self._excel_spool_build_delay_ms = 30_000
        self._excel_spool_build_timer = QTimer(self)
        self._excel_spool_build_timer.setSingleShot(True)
        self._excel_spool_build_timer.timeout.connect(self._on_excel_spool_build_timeout)
        self._excel_spool_build_in_progress = False
        self._excel_spool_targets_map = {}
        self._excel_spool_build_pending_map = {}
        self._excel_spool_build_pending_cfgs = []
        self._excel_spool_build_lock = threading.Lock()
        self._excel_spool_build_thread = None

        self.init_result_files()
        self.count_board = SequenceCountBoard(self.analysis_config)
        self._refresh_test_mode_availability()
        self.player_status_flag = False
        # True while a record run is still processing (record -> analysis -> save -> count updates).
        # Used to prevent starting a new recording before the full workflow completes.
        self._record_workflow_busy = False
        self._barcode_scanner_box_enabled_before_recording = None
        self.audio_devices_available = True
        self.audio_devices_unavailable_message = ""
        self.recorded_signal_info = {}
        self.ip_format = True
        self.port_format = True
        self.clicked_player_flag = False
        self.tcp_flag = False
        self.tcp_ip, self.tcp_port = LoadUiConfig.get_tcp_config()
        self.mode = None
        self.current_recorded_count = None
        self.last_play_count = None  # Cache last play count for replay

        self.default_logger = LogManager.set_log_handler("core")
        self._missing_config_prompted = False
        # Only show missing-config prompt after the window is shown (i.e. after login success).
        self._missing_config_prompt_enabled = False
        self._external_trigger_mode_warning_box = None

        self._barcode_debounce_ms = 50
        self._barcode_fast_input_max_seconds = 0.4  # 首字符到末字符的总耗时，小于该值更像扫码（扫码枪通常 < 0.2秒）
        self._barcode_min_length_for_auto_commit = 7  # 条码最小长度，太短容易误触发（可按实际条码规则调整）

        # 扫码逻辑委托到 BarcodeRouter（需先创建，再连接定时器/信号）
        self._barcode_router = BarcodeRouter(self)

        self._barcode_debounce_timer = QTimer(self)
        self._barcode_debounce_timer.setSingleShot(True)
        self._barcode_debounce_timer.setInterval(self._barcode_debounce_ms)
        # 扫码逻辑委托到 BarcodeRouter，SequenceWidget 只保留业务提交入口 _commit_barcode
        self._barcode_debounce_timer.timeout.connect(self._barcode_router.on_barcode_debounce_timeout)
        self._barcode_first_char_ts = None
        self._barcode_last_char_ts = None
        # 当焦点不在 S/N 输入框时，用事件过滤器捕获扫码枪按键序列（避免"必须点到输入框才生效"）
        self._barcode_capture_buffer = ""
        self._barcode_capture_first_ts = None
        self._barcode_capture_last_ts = None
        self._barcode_capture_target_lineedit = None
        self._barcode_capture_target_text = None
        self._barcode_capture_target_cursor_pos = None
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self._awaiting_ok_ng = False
        self._sn_clear_on_next_scan = False
        self._sn_textchange_manual_guard = False

        # 条码提交去重机制：防止 HID 模式和键盘楔入模式同时触发导致重复提交
        self._last_committed_barcode = None
        self._last_committed_barcode_time = 0.0
        self._barcode_commit_dedup_window_sec = 0.8  # 800ms 去重窗口

        # HID 模式激活标志：当 HID 模式成功接收条码后，暂时忽略键盘楔入模式的输入
        self._hid_mode_active_until = 0.0  # 时间戳，在此之前忽略键盘输入

        # Streaming state variables
        self.streaming_buffer_multi = []
        self.streaming_plot_item = None  # Persistent plot item for zoom preservation
        self.streaming_wav_writer = None  # WAV file writer for incremental saving
        self.streaming_processor = None  # StreamingAudioProcessor instance
        self.streaming_stimulus_data = None  # Stimulus data for alignment (play+record mode)
        self.streaming_mode = None  # "play_record" or "record_only"
        self._active_input_channels = [0]
        self.channel_workspace = None
        self._streaming_first_chunk_logged = False

        # Analysis window geometry persistence (per analysis item key)
        self._analysis_window_key_by_obj = weakref.WeakKeyDictionary()
        self._analysis_window_geometry_path = os.path.join(
            DEFAULT_DIR, "ui", "ui_config", "analysis_window_geometry.json"
        )
        self._analysis_window_geometry = self._load_analysis_window_geometry()
        self._analysis_window_geometry_flush_timer = QTimer(self)
        self._analysis_window_geometry_flush_timer.setSingleShot(True)
        self._analysis_window_geometry_flush_timer.timeout.connect(self._flush_analysis_window_geometry)
        self._analysis_window_geometry_dirty = False

        self.hw_manager = UnifiedHardwareManager()

        # 全局快捷键触发录音（独立于扫码枪/光电开关，窗口初始化即启用）
        self.shortcut_mgr = ShortcutTriggerManager(self)
        self.shortcut_mgr.sig_triggered.connect(self.on_shortcut_triggered, Qt.QueuedConnection)
        self.shortcut_mgr.start()
        self._shortcut_processing = False  # 防止快捷键重入

        # Create QTimer in Qt main thread for queue polling
        self.streaming_poll_timer = QTimer(self)
        self.streaming_poll_timer.timeout.connect(self._poll_streaming_queue)

        self.set_member_connect()
        self.bind_hw_signals()
        self.init_lineedit_text()
        self.init_ui()

    def showEvent(self, event):
        """
        MainWindow shows SequenceWindow only after login success.
        Defer the missing-config prompt until the first showEvent.
        """
        super().showEvent(event)
        self._missing_config_prompt_enabled = True
        if not self.sequence_config and not self._missing_config_prompted:
            MessageBox.warning(
                self,
                "提示",
                "当前未找到可用配置文件。\n"
                "请在上方【使用配置】下拉框中选择配置；\n"
                "如无可选项，请到【功能-测试队列】中保存或导入配置。",
            )
            self._missing_config_prompted = True

    def init_ui(self):
        """
        Initializes the user interface of the SequenceWindow.

        This method sets up the window icon, minimum height, and creates the main layout
        by adding toolbar and waveform layouts. It also connects button click events to
        their respective handlers and applies style sheets to the widgets.
        """
        self.setObjectName("SequenceWindow")
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        self.setMinimumHeight(600)
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
        SequenceWindow._active_instance_ref = weakref.ref(self)

    @pyqtSlot(str, bool)
    def _tcp_run_test(self, label: str = "not_labeled", skip_sn_regex_validation: bool = False):
        """
        TCP 回调线程通过 QueuedConnection 投递到 Qt 主线程的入口。
        直接调用 start_this_play 可能发生跨线程 UI 操作风险，因此统一走这里。
        """
        if not self._ensure_external_trigger_mode_supported("TCP"):
            return
        self.start_this_play(label, skip_sn_regex_validation=skip_sn_regex_validation)

    def _get_mode_display_name(self, mode: str) -> str:
        mode_names = {
            "RECORD_ONLY": "仅录制",
            "PLAY_AND_RECORD": "播放录制",
            "IMPORT_AUDIO": "导入音频",
            "IMPORT_STIMULUS_AUDIO": "导入激励与音频",
        }
        return mode_names.get(mode, mode or "未配置")

    def _show_external_trigger_mode_warning(self, trigger_source: str, current_mode: str) -> None:
        text = (
            f"当前工作模式为 {current_mode}，不支持{trigger_source}启动工作流。\n"
            f"仅【仅录制】和【播放录制】模式支持该功能。"
        )
        msg_box = self._external_trigger_mode_warning_box
        if msg_box is not None and msg_box.isVisible():
            msg_box.raise_()
            msg_box.activateWindow()
            return

        msg_box = MessageBox(self)
        msg_box.setIcon(MessageBox.Warning)
        msg_box.setWindowTitle("提示")
        msg_box.setText(text)
        msg_box.setStandardButtons(MessageBox.Ok)
        msg_box.finished.connect(lambda *_: setattr(self, "_external_trigger_mode_warning_box", None))
        self._external_trigger_mode_warning_box = msg_box
        msg_box.open()

    def _clear_sn_for_external_trigger_rejection(self, trigger_source: str) -> None:
        if "扫码枪" not in str(trigger_source or ""):
            return
        try:
            with QSignalBlocker(self.lineedit_s_or_n):
                self.lineedit_s_or_n.clear()
        except Exception:
            self.lineedit_s_or_n.clear()

    def _ensure_external_trigger_mode_supported(self, trigger_source: str) -> bool:
        supported_modes = {"RECORD_ONLY", "PLAY_AND_RECORD"}
        if self.mode in supported_modes:
            return True

        current_mode = self._get_mode_display_name(self.mode)
        self.default_logger.warning(
            f"{trigger_source} 自动启动被阻止：当前工作模式 {current_mode} 不支持外部自动启动工作流"
        )
        self._clear_sn_for_external_trigger_rejection(trigger_source)
        self._show_external_trigger_mode_warning(trigger_source, current_mode)
        return False

    def on_sequence_config_updated(self, *_):
        """
        Called when the test-queue window confirms config changes.

        Refresh the combobox items from registry, then reload the active config so the
        main window immediately reflects newly saved/imported entries.
        """
        try:
            old_mode = self.mode
            self.update_using_file_combobox()
            self.get_sequence_config_from_json()
            new_mode = self.mode
            mode_changed = old_mode != new_mode

            if mode_changed:
                self._clear_plot_area()
                self.data_struct.clear_data()
                self.refresh_channel_windows()
            self.init_data_struct_stimulus_config()
            self.init_fft_and_stft_flag()

            if self.count_board:
                self.count_board.analysis_config = self.analysis_config
                self._refresh_test_mode_availability()
        except Exception as e:
            self.default_logger.warning(f"Failed to refresh sequence config after update: {e}")

    def _refresh_test_mode_availability(self):
        """
        Enable/disable test mode based on whether current config can output OK/NG.
        """
        try:
            can_output, reason = self._can_output_ok_ng()
            if self.count_board:
                # Keep UX consistent: disable test if not eligible
                self.count_board.set_test_available(bool(can_output), reason or "")
        except Exception:
            if self.count_board:
                self.count_board.set_test_available(False, "无法判定是否具备OK/NG输出能力")

    def update_v2pa_factor(self):
        self.v2pa_factor = None

    def _summarize_ok_ng(self):
        """
        Summarize DataDealStruct.analysis_result_dict into overall OK/NG.
        Rule: all items OK -> OK; otherwise NG.
        """
        result_dict = getattr(self.data_struct, "analysis_result_dict", None)
        if not isinstance(result_dict, dict) or len(result_dict) == 0:
            return False, "NG"
        passed = True
        for _, v in result_dict.items():
            try:
                ok = bool(v[0])
            except Exception:
                ok = False
            if not ok:
                passed = False
                break
        return passed, ("OK" if passed else "NG")

    def _get_tcp_callback_file_name(self):
        file_path = getattr(self, "recorded_path", "") or ""
        if not file_path and isinstance(getattr(self, "recorded_signal_info", None), dict):
            file_path = self.recorded_signal_info.get("file_path") or ""
        normalized_path = str(file_path).replace("\\", "/") if file_path else ""
        return os.path.basename(normalized_path) if normalized_path else ""

    def _get_tcp_callback_label(self):
        result_dict = getattr(getattr(self, "data_struct", None), "analysis_result_dict", None)
        if not isinstance(result_dict, dict) or not result_dict:
            return "not_labeled"
        try:
            _passed, label = self._summarize_ok_ng()
        except Exception as e:
            self.default_logger.warning(f"tcp_callback_summary_error: {e}")
            return "not_labeled"
        return label if label in ("OK", "NG") else "not_labeled"

    def _build_tcp_analysis_result_payload(self):
        now = datetime.now()
        return {
            "TimeStamp": f"{now.strftime('%Y-%m-%d %H:%M:%S')},{now.microsecond // 1000:03d}",
            "Label": self._get_tcp_callback_label(),
            "FileName": self._get_tcp_callback_file_name(),
        }

    def _send_tcp_analysis_result_callback(self):
        if not getattr(self, "tcp_flag", False):
            return
        try:
            tcp_server = getattr(SequenceWindow, "tcp_server", None)
            client_address = getattr(tcp_server, "client_address", None)
            if not client_address:
                self.default_logger.warning("tcp_callback_skip: no client address")
                return
            payload = self._build_tcp_analysis_result_payload()
            message = json.dumps(payload, ensure_ascii=False)
            TempTcpClient(client_address[0], client_address[1], message)
            self.default_logger.info(f"tcp_callback_sent: {message}")
        except Exception as e:
            self.default_logger.error(f"tcp_callback_send_error: {e}")

    def _can_output_ok_ng(self):
        """
        Decide whether current analysis_config is expected to produce OK/NG output.

        We rely on analysis_result_dict being written by a subset of analysis widgets:
        - AI always writes (label + deviation)
        - SPL/SPLF/FR/HD/RB/PRB write only when threshold/compare (limit or golden) is enabled.
        """
        cfg = self.analysis_config or {}
        seq = cfg.get("display_sequence") or []
        if not isinstance(seq, list) or len(seq) == 0:
            return False, "当前配置未选择任何分析项"

        candidates = []
        for key in seq:
            item_cfg = cfg.get(key)
            if not isinstance(item_cfg, dict):
                continue
            t = item_cfg.get("type")
            if t == "AI":
                candidates.append(key)
                continue
            if t in ("SPL", "SPLF", "FR", "HD", "RB", "PRB"):
                if item_cfg.get("limit_checked"):
                    candidates.append(key)

        if candidates:
            return True, ""
        return False, "当前配置未启用阈值对比，无法产出OK/NG"

    def set_member_connect(self):
        self.player_btn.clicked.connect(lambda: self.on_clicked_player_btn())
        self.replayer_btn.clicked.connect(lambda: self.judge_play_and_record(is_replay=True))
        self.data_btn.clicked.connect(self.run)
        self.lineedit_type.editingFinished.connect(lambda: self.lineedit_type_lose_focus(self.lineedit_type))
        self.lineedit_count.editingFinished.connect(lambda: self.lineedit_count_lose_focus(self.lineedit_count))
        self.lineedit_count.returnPressed.connect(lambda: self.validate_count(self.lineedit_count, True))

        # 扫码键盘楔入模式：信号交给 BarcodeRouter 处理
        self.lineedit_s_or_n.returnPressed.connect(self._barcode_router.on_barcode_return_pressed)
        self.lineedit_s_or_n.textChanged.connect(self._barcode_router.on_barcode_text_changed)

        self.sn_regex_manage_btn.clicked.connect(self.open_sn_regex_manage_dialog)
        self.barcode_scanner_box.clicked.connect(self.clicked_scanner)
        self.tcp_btn.clicked.connect(self.on_tcp_btn_clicked)
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

    def _set_sn_input_recording_read_only(self, is_read_only):
        """Keep S/N editing controls locked only for the active play/record lifecycle."""
        if not self.tcp_flag:
            self.lineedit_s_or_n.setReadOnly(is_read_only)

            if is_read_only:
                if self._barcode_scanner_box_enabled_before_recording is None:
                    self._barcode_scanner_box_enabled_before_recording = self.barcode_scanner_box.isEnabled()
                self.barcode_scanner_box.setEnabled(False)
            else:
                if self._barcode_scanner_box_enabled_before_recording is not None:
                    self.barcode_scanner_box.setEnabled(self._barcode_scanner_box_enabled_before_recording)
                self._barcode_scanner_box_enabled_before_recording = None

    @property
    def sn_regex_manage_btn(self):
        return self.toolsbar.sn_regex_manage_btn

    @property
    def barcode_scanner_box(self):
        return self.toolsbar.barcode_scanner_box

    @property
    def tcp_btn(self):
        return self.toolsbar.tcp_btn

    def init_data_struct_stimulus_config(self):
        if not self.sequence_config:
            return
        acq_config = self.sequence_config[0]["seq1"]["acq"]
        if acq_config["mode"] in ["PLAY_AND_RECORD", "IMPORT_STIMULUS_AUDIO"]:
            modified = AnalysisModelSelect.set_data_struct_stimulus_signal(
                self.data_struct,
                acq_config["detail"],
                using_config_path=self.using_config_path,
            )
            # If stimulus wav was missing and we regenerated it, write back config to avoid future failures.
            if modified:
                try:
                    ok = LoadUiConfig.save_sequence_config_to_json(self.sequence_config, self.using_config_path)
                    if not ok:
                        LogManager.set_log_handler("core").warning(
                            f"Failed to write back regenerated stimulus path to config: {self.using_config_path}"
                        )
                except Exception as e:
                    LogManager.set_log_handler("core").warning(
                        f"Failed to write back regenerated stimulus path to config: {self.using_config_path}. {e}"
                    )
        else:
            self.data_struct.sample_rate = acq_config["detail"]["sample_rate"]
            self.data_struct.stimulus_data = None
            self.data_struct.stimulus_info = None

    def create_waveform_layout(self):
        """
            Create waveform display layout

            This function is responsible for generating a horizontal layout to display the waveform and related button area.
            It first creates a horizontal layout object and a plot widget, then sets the background color and creates
        the button layout.
            Finally, it adds these components to the layout and sets the layout margins.

            Returns:
                QHBoxLayout: The configured wavefrom layout object.
        """
        layout = QHBoxLayout()

        self.channel_workspace = ChannelPlotWorkspace(self)

        layout.addWidget(self.count_board, stretch=1)
        layout.addSpacing(20)
        layout.addWidget(self.channel_workspace, stretch=8)
        layout.setContentsMargins(40, 20, 40, 20)
        return layout

    def init_fft_and_stft_flag(self):
        model_item_list = self.analysis_config.get("display_sequence", "")
        for item_name in model_item_list:
            self.data_struct.add_stft_or_fft_count(self.analysis_config[item_name]["type"])

    def init_result_files(self):
        current_time = datetime.now().strftime("%Y-%m-%d")
        # Ensure daily test result file exists (no model field).
        try:
            ensure_test_result_file(self.analysis_config or {})
        except Exception:
            test_result_path = DEFAULT_DIR + f"log/test_result_log/{current_time}.dat"
            if not os.path.exists(test_result_path):
                os.makedirs(os.path.dirname(test_result_path), exist_ok=True)
                with open(test_result_path, "w") as f:
                    f.write(f"total: 0\n" f"ok: 0\n" f"ng: 0\n" f"ok_percent: 0%\n" f"datatime: {current_time}\n")

        mark_result_path = DEFAULT_DIR + "ui/ui_config/mark_result.json"
        mark_result_template = {"total": 0, "ok": 0, "ng": 0, "not_labels": 0, "datatime": current_time}
        if not os.path.exists(mark_result_path):
            os.makedirs(os.path.dirname(mark_result_path), exist_ok=True)
            with open(mark_result_path, "w") as f:
                json.dump(mark_result_template, f, indent=4)
        else:
            self.init_mark_result_file(mark_result_path, mark_result_template)

    def init_mark_result_file(self, mark_result_path, mark_result_template):
        with open(mark_result_path, "r") as f:
            data = json.load(f)
        current_date = datetime.now().strftime("%Y-%m-%d")
        if data["datatime"] != current_date:
            with open(mark_result_path, "w") as f:
                json.dump(mark_result_template, f, indent=4)

    def bind_hw_signals(self):
        """绑定 hardware manager 的信号, 避免重复连接"""
        try:
            self.hw_manager.sig_barcode.disconnect()
            self.hw_manager.sig_trigger.disconnect()
        except TypeError:
            pass
        self.hw_manager.sig_barcode.connect(self.on_barcode_received)
        self.hw_manager.sig_trigger.connect(self.on_sensor_triggered)

    def clicked_scanner(self):
        """Checkbox 状态改变时的回调"""
        if self.barcode_scanner_box.isChecked():
            # 配置文件加载失败不再致命：扫码枪可进入自动识别模式（支持热插拔）。
            # 注意：光电开关仍依赖 hotkey 配置。
            if not self.hw_manager.ensure_config_loaded():
                self.default_logger.warning(
                    "无法加载扫码枪/光电开关配置，将进入扫码枪自动识别模式（光电开关可能不可用）。"
                )

            self.lineedit_s_or_n.setEnabled(True)
            # 键盘楔入模式依赖“输入框有焦点”，开启后把焦点给 S/N 输入框
            try:
                self.lineedit_s_or_n.setFocus()
                self.lineedit_s_or_n.selectAll()
            except Exception:
                pass
            if self.hw_manager.start():
                self.default_logger.info("硬件监听已启动")
            else:
                self.default_logger.warning("硬件初始化失败，已静默降级为普通键盘输入模式")
        else:
            self.lineedit_s_or_n.clear()
            self.lineedit_s_or_n.setDisabled(True)
            self.hw_manager.stop()
            self.default_logger.info("硬件监听已停止")

    def on_barcode_received(self, barcode):
        """处理扫码枪信号（HID 模式）"""
        # 设置 HID 模式激活标志，在接下来 1 秒内忽略键盘楔入模式的输入
        # 这样可以避免 HID 模式和键盘模式同时工作导致的重复
        self._hid_mode_active_until = time.monotonic() + 1.0
        # 清空键盘楔入模式的缓冲区，防止残留数据干扰
        self._barcode_capture_buffer = ""
        self._barcode_capture_first_ts = None
        self._barcode_capture_last_ts = None
        self._barcode_capture_target_lineedit = None
        self._barcode_capture_target_text = None
        self._barcode_capture_target_cursor_pos = None
        self._barcode_debounce_timer.stop()
        self._commit_barcode(barcode, source="hid")

    def _normalize_barcode(self, text: str) -> str:
        if text is None:
            return ""
        return str(text).strip()

    # Windows 文件名中不允许的特殊字符
    _INVALID_FILENAME_CHARS = set('\\/:*?"<>|')

    def _barcode_has_invalid_chars(self, barcode: str) -> tuple:
        """检查条形码是否包含无法用于文件名的特殊字符，返回 (是否有, 特殊字符列表)"""
        found = [ch for ch in barcode if ch in self._INVALID_FILENAME_CHARS]
        return (bool(found), found)

    def _clear_barcode_input_safely(self):
        try:
            with QSignalBlocker(self.lineedit_s_or_n):
                self.lineedit_s_or_n.clear()
        except Exception:
            self.lineedit_s_or_n.clear()

    def _reset_barcode_dedup_state(self):
        self._last_committed_barcode = None
        self._last_committed_barcode_time = 0.0

    def _reset_barcode_commit_state(self, clear_dedup: bool = False):
        self._barcode_debounce_timer.stop()
        self._barcode_first_char_ts = None
        self._barcode_last_char_ts = None
        self._barcode_capture_buffer = ""
        self._barcode_capture_first_ts = None
        self._barcode_capture_last_ts = None
        self._barcode_capture_target_lineedit = None
        self._barcode_capture_target_text = None
        self._barcode_capture_target_cursor_pos = None
        if clear_dedup:
            self._reset_barcode_dedup_state()

    def _load_selected_sn_regex_rule(self):
        rules_payload = LoadUiConfig.load_sn_regex_rules_from_json()
        return LoadUiConfig.get_selected_sn_regex_rule(rules_payload)

    def _validate_sn_regex_before_start(
        self,
        sn_text=None,
        value_label="实际 SN 内容",
        retry_hint=None,
        skip_sn_regex_validation: bool = False,
    ):
        if skip_sn_regex_validation:
            return True
        if not self.barcode_scanner_box.isChecked():
            return True

        if sn_text is None:
            try:
                sn_text = self.lineedit_s_or_n.text()
            except Exception:
                sn_text = ""
        if sn_text is None:
            sn_text = ""
        else:
            sn_text = str(sn_text)

        selected_rule = self._load_selected_sn_regex_rule()
        if re.fullmatch(selected_rule["pattern"], sn_text) is not None:
            return True

        if retry_hint is None:
            retry_hint = "请检查当前 SN 内容或切换正确规则后重试。"

        sn_display = sn_text if sn_text else "（空）"
        self.default_logger.warning(
            "SN regex validation failed. "
            f"rule={selected_rule['name']}, pattern={selected_rule['pattern']}, sn={sn_text}"
        )
        MessageBox.warning(
            self,
            "SN 正则校验失败",
            f"当前 SN 内容不符合已启用规则：\n\n"
            f"规则名称：{selected_rule['name']}\n"
            f"规则表达式：{selected_rule['pattern']}\n"
            f"{value_label}：{sn_display}\n\n"
            f"{retry_hint}",
        )
        return False

    def open_sn_regex_manage_dialog(self):
        SnRegexManageDialog().exec_()

    def _commit_barcode(self, barcode: str, source: str = "wedge"):
        """
        统一条码提交入口：
        - source="hid": 来自 pywinusb HID raw handler 解析
        - source="wedge": 来自"键盘楔入模式"（扫码枪当键盘输入）
        """
        barcode = self._normalize_barcode(barcode)
        if not barcode:
            return

        if self.tcp_flag:
            return

        # 只在开启扫码枪功能时自动处理，避免影响用户正常输入
        if not self.barcode_scanner_box.isChecked():
            return

        if not self._ensure_external_trigger_mode_supported("扫码枪"):
            self._reset_barcode_commit_state()
            return

        if getattr(self, "_record_workflow_busy", False) or getattr(self, "player_status_flag", False):
            self.default_logger.debug(f"录音/播放忙碌中，忽略扫码提交 (source={source}): {barcode}")
            self._reset_barcode_commit_state()
            return

        # 条码提交去重：防止 HID 模式和键盘楔入模式同时触发导致重复提交
        now = time.monotonic()
        if (
            self._last_committed_barcode == barcode
            and (now - self._last_committed_barcode_time) < self._barcode_commit_dedup_window_sec
        ):
            self.default_logger.debug(f"忽略重复条码提交 (去重窗口内, source={source}): {barcode}")
            self._reset_barcode_commit_state()
            return
        # 更新去重记录
        self._last_committed_barcode = barcode
        self._last_committed_barcode_time = now

        try:
            fw = QApplication.focusWidget()
            fw_name = fw.__class__.__name__ if fw is not None else None
        except Exception:
            fw = None
            fw_name = None

        # 检查条形码是否包含特殊字符（会导致文件名无效）
        has_invalid, invalid_chars = self._barcode_has_invalid_chars(barcode)
        if has_invalid:
            unique_chars = sorted(set(invalid_chars))
            chars_display = "  ".join(repr(ch) for ch in unique_chars)
            MessageBox.warning(
                self,
                "条形码包含特殊字符",
                f"扫描到的内容包含无法用于文件名的特殊字符：\n\n"
                f"    {chars_display}\n\n"
                f"条形码内容：{barcode}\n\n"
                f"请检查条形码内容，或使用不含特殊字符的条形码。\n"
                f"关闭此窗口后可重新扫码。",
            )
            self._clear_barcode_input_safely()
            self._sn_textchange_manual_guard = False
            self._reset_barcode_commit_state(clear_dedup=True)
            return  # 不开始录音

        if not self._validate_sn_regex_before_start(
            barcode,
            value_label="实际扫码内容",
            retry_hint="请检查扫码内容或切换正确规则后重新扫码。",
        ):
            self._clear_barcode_input_safely()
            self._sn_textchange_manual_guard = False
            self._reset_barcode_commit_state(clear_dedup=True)
            return

        # 写回输入框（避免触发 textChanged 的二次提交）
        self._sn_textchange_manual_guard = False
        try:
            with QSignalBlocker(self.lineedit_s_or_n):
                self.lineedit_s_or_n.setText(barcode)
        except Exception:
            self.lineedit_s_or_n.setText(barcode)

        # 重置键盘楔入模式的输入统计
        self._reset_barcode_commit_state()

        # 自动开始测试
        if not getattr(self, "_record_workflow_busy", False):
            self.default_logger.info(f"S/N 收到({source}): {barcode}，自动开始测试")
            # 产线场景：上一轮的分析弹窗如果还开着，自动关闭，避免必须人工点关闭
            self._close_analysis_windows()
            self.start_this_play("not_labeled")

        # 扫码完成后"可选"聚焦 S/N：
        # - 默认：把焦点切回 S/N，便于连续扫码
        # - 例外：如果操作员当前正在"型号/计数"里编辑，则不抢焦点（更符合最简方案：不干扰手动输入）
        try:
            if fw is not self.lineedit_type and fw is not self.lineedit_count:
                self.lineedit_s_or_n.setFocus()
        except Exception:
            pass

    def on_sensor_triggered(self):
        """处理光电开关触发信号"""
        if not self._ensure_external_trigger_mode_supported("扫码枪/光电"):
            return
        if not getattr(self, "_record_workflow_busy", False):
            self.default_logger.info("光电触发响应: 开始测试")
            self.clicked_player_flag = True
            self.start_this_play("not_labeled")
        else:
            self.default_logger.info("正在测试中，忽略光电触发")

    def on_shortcut_triggered(self):
        """处理快捷键触发信号（F2）"""
        # 防止重入：MessageBox 嵌套事件循环中可能再次处理队列中的 F2 信号，
        # 导致递归雪崩（多层 MessageBox 堆叠 → 栈溢出 → 崩溃）。
        if self._shortcut_processing:
            return
        if not getattr(self, "_record_workflow_busy", False):
            self._shortcut_processing = True
            try:
                self.default_logger.info(f"快捷键触发响应: 开始测试 ({ShortcutTriggerManager.HOTKEY})")
                self.clicked_player_flag = True
                self.start_this_play("not_labeled")
            finally:
                self._shortcut_processing = False
        else:
            self.default_logger.info("正在测试中，忽略快捷键触发")

    def closeEvent(self, event):
        """窗口关闭时释放硬件资源，并强制等待Excel同步完成"""
        # Skip sync dialog if window was never shown (startup close)
        if not self.isVisible():
            if hasattr(self, "hw_manager"):
                self.hw_manager.stop()
            super().closeEvent(event)
            return

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
                failures = self.flush_excel_spool_build(on_close=True)
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

        if hasattr(self, "shortcut_mgr"):
            self.shortcut_mgr.stop()
        if hasattr(self, "hw_manager"):
            self.hw_manager.stop()
        super().closeEvent(event)

    def flush_excel_spool_build(self, *, on_close: bool = False) -> list[tuple[str, str]]:
        """
        Best-effort: stop the idle-timer, wait for any ongoing background build, then rebuild
        the daily .xlsx from the CSV spool so the final Excel exists on exit.

        Returns:
            A list of (cfg_name, error_message) tuples for any failed builds.
            Empty list means all builds succeeded or there was nothing to build.
        """
        failures: list[tuple[str, str]] = []

        try:
            self._excel_spool_build_timer.stop()
        except Exception:
            pass

        try:
            t = getattr(self, "_excel_spool_build_thread", None)
            if t is not None and getattr(t, "is_alive", None) and t.is_alive():
                try:
                    self.default_logger.info("excel_spool_build_wait_on_exit: waiting for background build thread...")
                except Exception:
                    pass
                try:
                    t.join()
                except Exception:
                    pass
        except Exception:
            pass

        # Prefer building from tracked spool targets. This supports "add_model_dir" where one
        # Excel config may export to multiple model-specific folders during one app session.
        try:
            targets = {}
            try:
                targets.update(getattr(self, "_excel_spool_targets_map", {}) or {})
                targets.update(getattr(self, "_excel_spool_build_pending_map", {}) or {})
            except Exception:
                targets = {}

            if not targets:
                product_model = ""
                try:
                    product_model = self.lineedit_type.text() or ""
                except Exception:
                    product_model = ""

                for k, v in (self.analysis_config or {}).items():
                    if not isinstance(v, dict):
                        continue
                    if v.get("type") != "Excel" or not v.get("enabled", True):
                        continue
                    if not v.get("fast_mode", True):
                        continue

                    try:
                        file_path = resolve_excel_output_path(v, product_model=product_model)
                        spool_dir = resolve_excel_spool_dir(v, file_path=file_path)
                        key = (str(k), os.path.normcase(os.path.abspath(str(file_path))))
                        targets[key] = (k, v, str(file_path), str(spool_dir))
                    except Exception as e:
                        self.default_logger.error(f"excel_spool_build_path_error[{k}]: {e}")

            if not targets:
                return failures

            for cfg_name, excel_cfg, file_path, spool_dir in list(targets.values()):
                ret = build_excel_from_csv_spool(excel_cfg, file_path=file_path, spool_dir=spool_dir)
                if ret.ok:
                    self.default_logger.info(
                        f"excel_spool_build_on_exit_ok[{cfg_name}]: {ret.message}"
                        if not on_close
                        else f"excel_spool_build_on_close_ok[{cfg_name}]: {ret.message}"
                    )
                else:
                    self.default_logger.warning(
                        f"excel_spool_build_on_exit_fail[{cfg_name}]: {ret.message}"
                        if not on_close
                        else f"excel_spool_build_on_close_fail[{cfg_name}]: {ret.message}"
                    )
                    failures.append((cfg_name, ret.message))
        except Exception as e:
            try:
                tag = "excel_spool_build_on_close_error" if on_close else "excel_spool_build_on_exit_error"
                self.default_logger.error(f"{tag}: {e}")
                failures.append(("unknown", str(e)))
            except Exception:
                failures.append(("unknown", str(e)))

        return failures

    def reset_test_reord(self):
        """
        Reset today's test counters (total/ok/ng/ok_percent) and refresh UI texts.
        """
        current_time = datetime.now().strftime("%Y-%m-%d")
        ensure_test_result_file(self.analysis_config)
        test_result_path = DEFAULT_DIR + f"log/test_result_log/{current_time}.dat"
        lines = [
            "total: 0\n",
            "ok: 0\n",
            "ng: 0\n",
            "ok_percent: 0%\n",
            f"datatime: {current_time}\n",
        ]
        with open(test_result_path, "w") as f:
            f.writelines(lines)
        # Refresh displayed counters
        try:
            self.count_board.set_test_text()
        except Exception:
            pass
        try:
            self.count_board.set_mark_text()
        except Exception:
            pass

    def on_reset_statistics_clicked(self):
        """
        Handler for count-board “重置统计” button.

        Expected behavior (用户期望):
        - Reset test counters (统计面板显示归零)
        - Reset related runtime UI states (重播/分析按钮回到禁用)
        """
        try:
            self.reset_test_reord()
        except Exception as e:
            try:
                self.default_logger.error(f"reset_statistics_error: {e}")
            except Exception:
                pass

        # Reset replay/analyze buttons and related runtime flags
        try:
            self.last_play_count = None
        except Exception:
            pass
        try:
            self.player_status_flag = False
        except Exception:
            pass
        try:
            self.clicked_player_flag = False
        except Exception:
            pass
        try:
            self._awaiting_ok_ng = False
            self._sn_clear_on_next_scan = False
        except Exception:
            pass
        try:
            # Clear cached wave so “分析”不会对旧数据误操作
            if hasattr(self.data_struct, "store_wave_data"):
                self.data_struct.store_wave_data = None
            if hasattr(self.data_struct, "store_wave_data_multi"):
                self.data_struct.store_wave_data_multi = None
        except Exception:
            pass
        try:
            self.replayer_btn.setDisabled(True)
        except Exception:
            pass
        try:
            self.data_btn.setDisabled(True)
        except Exception:
            pass
        try:
            # Restore player UI to idle state
            self.update_player_btn_is_paused()
        except Exception:
            pass

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
        sn_locked_for_recording = getattr(self, "_record_workflow_busy", False) or getattr(
            self, "player_status_flag", False
        )
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
            if is_s_or_n and not sn_locked_for_recording:
                self.lineedit_s_or_n.setText("")
        if s_or_n_count == "":
            if is_s_or_n:
                lineedit.setText(str(result_count))
            else:
                lineedit.setText(str(result_scanner_barcode))

    def swap_tcp_status(self):
        if self.tcp_flag:
            self.barcode_scanner_box.setEnabled(False)
            self.lineedit_s_or_n.setReadOnly(True)
            self.tcp_ip, self.tcp_port = LoadUiConfig.get_tcp_config()
            if hasattr(self, "tcp_server") and SequenceWindow.tcp_server:
                SequenceWindow.tcp_server.stop()
                SequenceWindow.tcp_server = None
            SequenceWindow.tcp_server = TcpServer(host=self.tcp_ip, port=self.tcp_port, callback=self.deal_package)
            SequenceWindow.tcp_server.start()
        else:
            self.barcode_scanner_box.setEnabled(True)
            self.lineedit_s_or_n.setReadOnly(False)
            if hasattr(self, "tcp_server") and SequenceWindow.tcp_server:
                SequenceWindow.tcp_server.stop()
                SequenceWindow.tcp_server = None

    @staticmethod
    def deal_package(info):
        """
        info: {
                  "RequestType": "0-9999",
                  "RequestContent": {
                    "User": "Alice",
                    "Action": "ScanBarcode",
                    "label": "NG"
                  },
                  "IsSync": false,
                  "Timestamp": "2025-04-09T16:30:00"
              }
        """
        ok, data = check_tcp_msg_format(info)
        if not ok:
            return data
        request_type = int(data.get("RequestType"))
        request_content = data.get("RequestContent", {})
        is_sync = data.get("IsSync")
        timestamp = data.get("Timestamp")
        request_id = f"{request_type}@{timestamp}"
        if request_id == SequenceWindow.tcp_server.request_id:
            return "pass"
        else:
            SequenceWindow.tcp_server.request_id = request_id
        # allocating task
        if request_type == RequestTypeEnum.RUN_TEST.value:
            # 兼容多种客户端字段命名：
            # - 老客户端: Label
            # - 示例文档: label
            label = request_content.get("Label") or request_content.get("label") or "not_labeled"
            # Dispatch to current SequenceWindow instance in Qt main thread
            try:
                ref = getattr(SequenceWindow, "_active_instance_ref", None)
                inst = ref() if callable(ref) else None
            except Exception:
                inst = None
            if inst is None:
                try:
                    LogManager.set_log_handler("core").warning(
                        "TCP RUN_TEST received, but no active SequenceWindow instance is available."
                    )
                except Exception:
                    pass
            else:
                try:
                    # 必须用 QueuedConnection 投递到 inst 所在线程（Qt 主线程），避免跨线程 UI 调用
                    QMetaObject.invokeMethod(
                        inst,
                        "_tcp_run_test",
                        Qt.QueuedConnection,
                        Q_ARG(str, str(label)),
                        Q_ARG(bool, True),
                    )
                except Exception as e:
                    try:
                        LogManager.set_log_handler("core").error(f"TCP dispatch RUN_TEST failed: {e}")
                    except Exception:
                        pass
        return "ok"

    def on_tcp_btn_clicked(self):
        tcp_config_dialog = TcpConfigDialog(self.tcp_flag, self.tcp_ip, self.tcp_port)
        result = tcp_config_dialog.exec()
        if result:
            self.tcp_flag, self.tcp_ip, self.tcp_port = result
            LoadUiConfig.write_tcp_config(self.tcp_ip, self.tcp_port, self.default_logger)
            self.swap_tcp_status()

    def clicked_ok_or_ng(self, manual=True):
        """
        Handles the logic when the OK or NG button is clicked.

        This method performs several actions in response to a user clicking the OK or NG button:
        1. Saves the current recorded count to a text file.
        2. Updates the displayed recorded count in the UI.
        3. Inserts the recorded data into the database with a label based on which button was clicked (OK/NG).
        4. Resets the player status flag and updates the player icon accordingly.
        5. Clears the signal information and waveform graph.
        6. Disables the replay and data buttons to prevent further actions until the next recording.

        Parameters:
            self: The instance of the class containing this method.
            manual: If True (default), this is a manual user click; if False, this is an auto-triggered call.
                    Only manual calls will disable the replay and data buttons.
        """
        if (
            not hasattr(self.data_struct, "store_wave_data")
            or self.data_struct.store_wave_data is None
            or len(self.data_struct.store_wave_data) == 0
        ):
            MessageBox.warning(self, "警告", "请先录制声音！")
            return
        if self.sequence_config:
            if self.sequence_config[0]["seq1"]["acq"]["mode"] == "IMPORT_AUDIO":
                MessageBox.warning(self, "警告", "当前为导入音频模式，无需点击 OK/NG 按钮。")
                return

            if self.sequence_config[0]["seq1"]["acq"]["mode"] == "IMPORT_STIMULUS_AUDIO":
                MessageBox.warning(self, "警告", "当前为导入激励信号与音频模式，无需点击 OK/NG 按钮。")
                return
        self.update_audio_label_info()
        self._maybe_export_excel_results()
        self.update_recorded_signal_info_to_db()
        self._close_analysis_windows()

        self.mark_result()
        self.data_struct.store_wave_data = None
        self.data_struct.store_wave_data_multi = None
        self.replayer_btn.setEnabled(False)
        self.data_btn.setEnabled(False)
        self.player_status_flag = False
        self.signal_info.clear()
        self.lineedit_s_or_n.clear()
        self._clear_plot_area()
        self._awaiting_ok_ng = False
        self._sn_clear_on_next_scan = False

        # Only disable buttons when manually clicking OK/NG
        # Auto-triggered calls should keep buttons enabled for user verification
        if manual:
            self.replayer_btn.setDisabled(True)
            self.data_btn.setEnabled(False)

        self.clicked_scanner()
        self.update_player_btn_is_paused()

    def update_recorded_signal_info_to_db(self):
        if self.recorded_signal_info["labels"] == "not_labeled":
            return
        new_file_path = FileOps.move_wav_to_dir(self.recorded_path, self.recorded_signal_info["labels"])
        old_file_path = self.recorded_signal_info["file_path"]
        self.recorded_signal_info["file_path"] = new_file_path.replace(DEFAULT_DIR, "")
        save_code, msg = RecordingManager().update_audio_label(self.recorded_signal_info, old_file_path)
        if save_code == error_code.OK:
            self.default_logger.info("Recorded signal successfully updated.")
        else:
            self.default_logger.error("Failed to update recorded signal.")

    def update_audio_label_info(self):
        button = self.sender()
        if button == self.count_board.ok_btn:
            self.recorded_signal_info["labels"] = "OK"
        elif button == self.count_board.ng_btn:
            self.recorded_signal_info["labels"] = "NG"

    def mark_result(self):
        button = self.sender()
        if button == self.count_board.ok_btn:
            self.count_board.set_mark_result_file("OK")
            self.count_board.set_mark_text()
        elif button == self.count_board.ng_btn:
            self.count_board.set_mark_result_file("NG")
            self.count_board.set_mark_text()

    def _finalize_test_run(self, label: str):
        """
        Finalize a test-mode run by applying the summarized OK/NG label,
        exporting results, updating DB label, and resetting UI state.
        """
        if label not in ("OK", "NG"):
            return
        try:
            self.recorded_signal_info["labels"] = label
        except Exception:
            return

        self._maybe_export_excel_results()
        self.update_recorded_signal_info_to_db()
        self.player_status_flag = False
        self.signal_info.clear()
        self.lineedit_s_or_n.clear()
        self.replayer_btn.setDisabled(True)
        self.data_btn.setEnabled(False)
        self._awaiting_ok_ng = False
        self._sn_clear_on_next_scan = False
        self.clicked_scanner()
        self.update_player_btn_is_paused()

    def on_clicked_player_btn(self, label="not_labeled"):
        if not self.sequence_config:
            MessageBox.warning(
                self,
                "提示",
                "未找到可用配置。\n"
                "请先在上方【使用配置】下拉框中选择配置；\n"
                "如无可选项，请到【功能-测试队列】中保存或导入配置。",
            )
            return
        acq_mode = self.sequence_config[0]["seq1"]["acq"]["mode"]
        if acq_mode == "IMPORT_AUDIO":
            self.import_audio_and_analyze()
            return
        if acq_mode == "IMPORT_STIMULUS_AUDIO":
            self.import_audio_and_analyze()
            return
        self.clicked_player_flag = True
        self.start_this_play(label)

    def import_audio_and_analyze(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频文件",
            DEFAULT_DIR + "audio_data/stored_data",
            "WAV Files (*.wav)",
        )
        if not file_path:
            return
        # Ensure subsequent exports (CSV/Excel) use this imported file as the current record id,
        # instead of accidentally reusing a stale `recorded_path` from previous recordings.
        try:
            self.recorded_path = file_path
            self.recorded_signal_info = {"file_path": file_path, "barcode": None, "labels": "not_labeled"}
        except Exception:
            pass
        acq_detail = self.sequence_config[0]["seq1"]["acq"]["detail"]
        sample_rate = acq_detail.get("sample_rate", 44100)
        audio_multi, _ = librosa.load(file_path, sr=sample_rate, mono=False)
        audio_multi = np.asarray(audio_multi, dtype=np.float32)
        if audio_multi.ndim == 1:
            audio_multi = audio_multi.reshape(1, -1)
        audio_multi = audio_multi.T
        self.data_struct.store_wave_data_multi = audio_multi

        # not recommended for metadata analysis, as it may cause inaccurate data
        self.data_struct.store_wave_data = audio_multi.mean(axis=1).astype(np.float32, copy=False)

        self.data_struct.sample_rate = sample_rate
        audio_y, _ = librosa.load(file_path, sr=None)
        self.data_struct.audio_lenth = len(audio_y)
        self._clear_plot_area()
        self.plot_waveform_to_workspace(self.data_struct.store_wave_data_multi, self.data_struct.sample_rate)
        self.data_btn.setEnabled(True)
        if self.analysis_config.get("auto_analysis"):
            self.run()

    def start_this_play(self, label="not_labeled", skip_sn_regex_validation: bool = False):
        if getattr(self, "_record_workflow_busy", False):
            return
        if getattr(self, "player_status_flag", False):
            return
        if not self.tcp_flag:
            if not self._validate_sn_regex_before_start(skip_sn_regex_validation=skip_sn_regex_validation):
                return
        if self.checked_work_status_message():
            return

        if self.clicked_player_flag is False:
            if self.tcp_flag and SequenceWindow.tcp_server.client_address is None:
                MessageBox.warning(self, "提示", "TCP链接异常")
                return

        if self.analysis_window:
            self.analysis_window = []
        if self._analysis_result_summary_window:
            self._analysis_result_summary_window = None
        # Increment count BEFORE recording (so display count = file count)
        self.current_recorded_count += 1
        self.lineedit_count.setText(str(self.current_recorded_count))

        # Cache this count for replay
        self.last_play_count = self.current_recorded_count

        save_recorded_data_to_json(
            self.lineedit_type.text(),
            self.lineedit_count.text(),
            self.lineedit_s_or_n.text(),
            self.barcode_scanner_box.isChecked(),
        )

        # Record with new count
        self.judge_play_and_record(label, is_replay=False)

        if self.clicked_player_flag is True:
            self.clicked_player_flag = False

    def set_audio_devices_available(self, available: bool, message: str = ""):
        self.audio_devices_available = bool(available)
        self.audio_devices_unavailable_message = message or ""
        self.update_player_btn_is_paused()

    def checked_work_status_message(self):
        if not self.sequence_config:
            MessageBox.warning(
                self,
                "提示",
                "未找到可用配置。\n"
                "请先在上方【使用配置】下拉框中选择配置；\n"
                "如无可选项，请到【功能-测试队列】中保存或导入配置。",
            )
            return True

        if not getattr(self, "audio_devices_available", True):
            MessageBox.warning(
                self,
                "提示",
                getattr(self, "audio_devices_unavailable_message", "")
                or "音频设备不可用，请检查设备连接或在【硬件-硬件选择】中重新选择设备。",
            )
            return True

        if not self.mic:
            MessageBox.warning(self, "提示", "未找到麦克风，请在硬件中设置")
            return True
        if not self.speaker:
            MessageBox.warning(self, "提示", "未找到扬声器，请在硬件中设置")
            return True

        return False

    def reset_work_pram(self, label, count=None):
        self.data_struct.clear_data()
        self._excel_export_cache = None
        self._excel_exported_record_id = None

        # Use provided count if available (for replay), otherwise use lineedit value
        count_str = str(count) if count is not None else self.lineedit_count.text()

        self.recorded_path, self.recorded_signal_info = get_recorded_info(
            self.lineedit_type.text(), count_str, self.lineedit_s_or_n.text(), label
        )
        acq_detail = self.sequence_config[0]["seq1"]["acq"]["detail"]
        acq_mode = self.sequence_config[0]["seq1"]["acq"]["mode"]
        normalized_detail = None
        if acq_mode == "RECORD_ONLY":
            normalized_detail = normalize_record_only_detail(acq_detail)
            acq_detail = normalized_detail
        elif acq_mode == "PLAY_AND_RECORD":
            normalized_detail = normalize_play_record_detail(acq_detail)
            acq_detail = normalized_detail
        if acq_mode in ["RECORD_ONLY", "IMPORT_AUDIO"]:
            self.data_struct.sample_rate = int(acq_detail.get("sample_rate", self.data_struct.sample_rate or 44100))
        total_time = float(acq_detail.get("total_time", 5.0))
        monitor_playback = acq_detail.get("monitor_playback", False)
        monitor_gain_db = float(acq_detail.get("monitor_gain_db", 0.0))
        acq_mode = self.sequence_config[0]["seq1"]["acq"]["mode"]
        sample_rate = self.data_struct.sample_rate
        delay_ms = None
        if normalized_detail is not None:
            delay_ms = normalized_detail["recording_start_delay_ms"]
        stimulus_dict, recorded_dict = LoadUiConfig.get_rec_and_play_dict_base_sequence_dict(
            self.data_struct, total_time, recording_start_delay_ms=delay_ms
        )

        # Add device information for streaming mode
        recorded_dict["device"] = self.mic
        recorded_dict["input_device"] = self.mic

        input_channels = list(getattr(self, "mic_channels", []) or [0])

        recorded_dict["input_channels"] = input_channels
        recorded_dict["channels"] = max(1, len(input_channels))

        if acq_mode == "PLAY_AND_RECORD":
            recorded_dict["output_device"] = self.speaker
        elif acq_mode == "RECORD_ONLY" and monitor_playback:
            try:
                monitor_input_channel = int(acq_detail.get("monitor_input_channel", input_channels[0]))
            except (TypeError, ValueError):
                monitor_input_channel = input_channels[0]
            if monitor_input_channel not in input_channels:
                monitor_input_channel = input_channels[0]
            recorded_dict["monitor_playback"] = True
            recorded_dict["monitor_input_channel"] = monitor_input_channel
            recorded_dict["monitor_gain_db"] = monitor_gain_db
            recorded_dict["output_device"] = self.speaker
        else:
            recorded_dict["output_device"] = None

        self._active_input_channels = [int(x) for x in input_channels]

        in_dev = recorded_dict.get("input_device")
        out_dev = recorded_dict.get("output_device")
        if in_dev and out_dev:
            if in_dev.get("hostapi") != out_dev.get("hostapi"):
                MessageBox.warning(
                    self,
                    "设备组合不支持",
                    "播放+录制需要选择同一驱动类型（Host API）的输入/输出设备。\n"
                    f"当前输入: {in_dev.get('name')} (hostapi={in_dev.get('hostapi')})\n"
                    f"当前输出: {out_dev.get('name')} (hostapi={out_dev.get('hostapi')})",
                )
                return None, None, None

        return stimulus_dict, recorded_dict, sample_rate

    def _should_use_streaming_recording(self):
        try:
            detail = self.sequence_config[0]["seq1"]["acq"].get("detail", {})
        except Exception:
            detail = {}
        return bool(detail.get("use_streaming_recording", False))

    def _start_streaming_recording(self, stimulus_dict, recorded_dict, sample_rate):
        nch = max(1, len(getattr(self, "_active_input_channels", []) or [0]))
        if self.sequence_config[0]["seq1"]["acq"]["mode"] in ["PLAY_AND_RECORD"]:
            temp_path = self.recorded_path.replace(".wav", "_temp.wav")
            self.streaming_wav_writer = StreamingWavWriter(temp_path, sample_rate)
            self.streaming_temp_path = temp_path

            self.streaming_processor, self.streaming_stimulus_data, _ = stream_play_and_record(
                stimulus_dict, recorded_dict, self.recorded_path, self.recorded_signal_info
            )
            self.streaming_mode = "play_record"

        else:
            self.streaming_wav_writer = StreamingWavWriter(self.recorded_path, sample_rate, channels=nch)

            self.streaming_processor, _ = stream_record_without_play(
                recorded_dict, self.recorded_path, self.recorded_signal_info
            )
            self.streaming_mode = "record_only"
            self.streaming_stimulus_data = None

        self.streaming_poll_timer.start(50)

    def _normalize_blocking_recorded_data(self, recorded_data):
        recorded_array = np.asarray(recorded_data, dtype=np.float32)
        if recorded_array.size == 0:
            raise ValueError("empty recorded data")

        if recorded_array.ndim == 1:
            mono = recorded_array.reshape(-1)
            return mono, mono.reshape(-1, 1)

        if recorded_array.ndim != 2:
            mono = recorded_array.reshape(-1)
            return mono, mono.reshape(-1, 1)

        selected_channel_count = len(getattr(self, "_active_input_channels", []) or [])
        if (
            selected_channel_count > 0
            and recorded_array.shape[0] == selected_channel_count
            and recorded_array.shape[1] != selected_channel_count
        ):
            recorded_array = recorded_array.T

        multi = np.asarray(recorded_array, dtype=np.float32)
        if multi.shape[1] == 1:
            mono = multi[:, 0]
        else:
            mono = multi.mean(axis=1).astype(np.float32, copy=False)
        return mono.reshape(-1), multi

    def _finish_recording_success(self, sample_rate):
        self.player_status_flag = False
        self._record_workflow_busy = False
        self._set_sn_input_recording_read_only(False)

        self.data_btn.setEnabled(True)
        self.replayer_btn.setEnabled(True)

        self._awaiting_ok_ng = True
        self._sn_clear_on_next_scan = True
        if self.barcode_scanner_box.isChecked():
            try:
                self.lineedit_s_or_n.setFocus()
                self.lineedit_s_or_n.selectAll()
            except Exception:
                pass

        if self.analysis_config.get("auto_analysis", False):
            self.run()

        self.update_player_btn_is_paused()

    def _finish_recording_failure(self, error):
        self.default_logger.error(f"blocking_recording_error: {error}")
        self.player_status_flag = False
        self._record_workflow_busy = False
        self._set_sn_input_recording_read_only(False)
        self.data_btn.setEnabled(True)
        self.replayer_btn.setEnabled(True)
        self._awaiting_ok_ng = False
        self._sn_clear_on_next_scan = False
        self.update_player_btn_is_paused()
        MessageBox.warning(self, "提示", f"录音失败: {error}")

    def _start_blocking_recording(self, stimulus_dict, recorded_dict, sample_rate):
        try:
            mode = self.sequence_config[0]["seq1"]["acq"]["mode"]
            if mode == "PLAY_AND_RECORD":
                record_code, recorded_data = SoundcardAudioProcessor().sd_play_rec(
                    recorded_dict, stimulus_dict, self.recorded_path
                )
                if record_code != error_code.OK or recorded_data is None:
                    raise RuntimeError(recorded_data if recorded_data is not None else record_code)

                mono_data, multi_data = self._normalize_blocking_recorded_data(recorded_data)
                self.data_struct.store_wave_data = mono_data
                self.data_struct.store_wave_data_multi = multi_data
                self.recorded_signal_info["sample_rate"] = sample_rate

                save_code, save_msg = RecordingManager().save_signal_info_to_db(
                    self.recorded_signal_info, self.data_struct.stimulus_info
                )
                if save_code == error_code.OK:
                    self.default_logger.info(f"Database save successful: {save_msg}")
                else:
                    self.default_logger.error(f"Database save failed: {save_msg}")

                repeat_times = self.data_struct.stimulus_info.get("repeat_times", 1)
                if repeat_times > 1:
                    kwargs = {"repeat_times": repeat_times}
                    self.data_struct.split_repeat_data = SplitRepeatSignal().split_repeat_signal(
                        self.data_struct.store_wave_data, sample_rate, **kwargs
                    )
                self.plot_waveform_to_workspace(self.data_struct.store_wave_data_multi, sample_rate)

            elif mode == "RECORD_ONLY":
                record_code, recorded_data = SoundcardAudioProcessor.sd_rec(recorded_dict)
                if record_code != error_code.OK or recorded_data is None:
                    raise RuntimeError(recorded_data if recorded_data is not None else record_code)

                mono_data, multi_data = self._normalize_blocking_recorded_data(recorded_data)
                self.data_struct.store_wave_data = mono_data
                self.data_struct.store_wave_data_multi = multi_data
                self.recorded_signal_info["sample_rate"] = sample_rate

                if multi_data.shape[1] > 1:
                    save_audio_simple(self.recorded_path, multi_data, sample_rate)
                else:
                    save_audio_simple(self.recorded_path, mono_data, sample_rate)

                save_code, save_msg = RecordingManager().save_signal_info_to_db(self.recorded_signal_info, None)
                if save_code == error_code.OK:
                    self.default_logger.info(f"Database save successful: {save_msg}")
                else:
                    self.default_logger.error(f"Database save failed: {save_msg}")
                self.plot_waveform_to_workspace(multi_data, sample_rate)

            else:
                raise RuntimeError(f"unsupported blocking recording mode: {mode}")

            self._finish_recording_success(sample_rate)
            self.default_logger.info("Blocking recording completed successfully")

        except Exception as e:
            self._finish_recording_failure(e)

    def judge_play_and_record(self, label="not_labeled", is_replay=False):
        if getattr(self, "_record_workflow_busy", False):
            return
        if not self.tcp_flag:
            if is_replay and not self._validate_sn_regex_before_start():
                return
        if self.checked_work_status_message():
            return
        if is_replay and self.last_play_count is None:
            MessageBox.warning(self, "提示", "请先进行录音")
            return

        if self.analysis_window:
            self.analysis_window = []
        if self._analysis_result_summary_window:
            self._analysis_result_summary_window = None

        self._record_workflow_busy = True
        self._set_sn_input_recording_read_only(True)

        self._clear_plot_area()
        # CRITICAL: Clean up any existing streaming resources before starting new recording
        # This prevents device conflicts and freezing when replay is clicked multiple times
        self._cleanup_streaming_resources()

        self.update_player_btn_is_playing()

        # Clear plot and reset streaming state for NEW recording
        if self.player_status_flag:
            self._clear_plot_area()

        self.streaming_buffer_multi = []
        self.streaming_plot_item = None  # Reset plot item for new recording

        self.player_status_flag = True

        # Disable replay and data buttons during recording/playback
        # They will be re-enabled in _on_streaming_complete()
        self.replayer_btn.setDisabled(True)
        self.data_btn.setDisabled(True)

        QApplication.processEvents()

        # For replay: use cached count to overwrite the same file
        # For play: use current lineedit count (already incremented in start_this_play)
        try:
            if is_replay:
                stimulus_dict, recorded_dict, sample_rate = self.reset_work_pram(label, count=self.last_play_count)
            else:
                stimulus_dict, recorded_dict, sample_rate = self.reset_work_pram(label)
        except Exception as e:
            self.default_logger.error(f"reset_work_pram_error: {e}")
            if not is_replay:
                # rollback the increment done in start_this_play()
                self.current_recorded_count -= 1
                self.lineedit_count.setText(str(self.current_recorded_count))
            self.player_status_flag = False
            self._record_workflow_busy = False
            self._set_sn_input_recording_read_only(False)
            self.update_player_btn_is_paused()
            MessageBox.warning(self, "提示", f"初始化录音失败: {e}")
            return

        if stimulus_dict is None:
            if not is_replay:
                # rollback the increment done in start_this_play()
                self.current_recorded_count -= 1
                self.lineedit_count.setText(str(self.current_recorded_count))
            self.player_status_flag = False
            self._record_workflow_busy = False
            self._set_sn_input_recording_read_only(False)
            self.update_player_btn_is_paused()
            return

        if self._should_use_streaming_recording():
            try:
                self._start_streaming_recording(stimulus_dict, recorded_dict, sample_rate)
            except Exception as e:
                self.default_logger.error(f"start_streaming_error: {e}")
                self._cleanup_streaming_resources()
                self.player_status_flag = False
                self._record_workflow_busy = False
                self._set_sn_input_recording_read_only(False)
                self.update_player_btn_is_paused()
                MessageBox.warning(self, "提示", f"启动录音失败: {e}")
                return
            # Return immediately - completion will be handled by _on_streaming_complete()
            # Note: Don't enable buttons yet, that happens in _on_streaming_complete()
            return

        self._start_blocking_recording(stimulus_dict, recorded_dict, sample_rate)
        return

    def run(self):
        """
        Executes the analysis tasks and displays the analysis windows.

        This method initializes the analysis windows based on the configuration and creates corresponding
        analysis instances according to the analysis types specified in the configuration. It then performs
        the respective calculations for each instance and displays the windows. The window positions are
        adjusted based on the screen size to ensure they do not overlap.
        """
        # Only reflect THIS run(): clear previous summary results first
        self.data_struct.analysis_result_dict.clear()
        if self.sequence_config[0]["seq1"]["acq"]["mode"] == "IMPORT_STIMULUS_AUDIO":
            stimulus_length = round(
                self.data_struct.stimulus_info.get("total_time") * self.data_struct.stimulus_info.get("sample_rate")
            )
            if self.data_struct.audio_lenth != stimulus_length:
                MessageBox.warning(
                    self,
                    "音频长度校验失败",
                    f"导入音频长度({self.data_struct.audio_lenth})\n"
                    f"与激励信号长度({stimulus_length})不一致！无法分析！",
                )
                return
        if self.analysis_window:
            self.analysis_window = []
        if self._analysis_result_summary_window:
            self._analysis_result_summary_window = None

        width = int((self.screen().size().width() - 400) / 3)
        height = int((self.screen().size().height() - 400) / 3)
        window_width = ui_style_const.scale_size_px(600)
        window_height = ui_style_const.scale_size_px(500)
        if self.analysis_config:
            item_sort_list = self.analysis_config.get("display_sequence", [])
            for key in item_sort_list:
                key_config = self.analysis_config.get(key)
                if not isinstance(key_config, dict):
                    continue
                item_type = key_config.get("type")
                self.instance_analysis_class(key, item_type, key_config)
            for instance in self.analysis_window:
                # Bind this instance to its analysis item key (used for geometry restore/persist)
                instance_key = getattr(instance, "_sequence_analysis_key", None)
                mismatch_info = getattr(instance, "_channel_mismatch_info", None)
                if getattr(instance, "_channel_mismatch", False):
                    self._show_channel_mismatch_warning(instance_key or "分析项", mismatch_info=mismatch_info)
                    continue
                if hasattr(instance, "calculate_spl"):
                    result = instance.calculate_spl()
                    if not result:
                        continue
                    instance.show()
                elif hasattr(instance, "calculate_fr"):
                    result = instance.calculate_fr()
                    if not result:
                        continue
                    instance.show()
                elif hasattr(instance, "calculate_thd"):
                    instance.calculate_thd()
                    instance.show()
                elif hasattr(instance, "calculate_ai_scores"):
                    instance.calculate_ai_scores(
                        self.count_board.mode, self.analysis_config, self.sequence_config[0]["seq1"]["acq"]["mode"]
                    )
                    instance.show()
                elif hasattr(instance, "calculate_spec"):
                    instance.calculate_spec()
                    instance.show()
                elif hasattr(instance, "calculate_reference_spectrum"):
                    result = instance.calculate_reference_spectrum()
                    if not result:
                        continue
                    instance.show()
                elif hasattr(instance, "calculate_peak_detection"):
                    instance.calculate_peak_detection()
                    instance.show()
                elif hasattr(instance, "calculate_loose_particle"):
                    instance.calculate_loose_particle()
                    instance.show()
                elif hasattr(instance, "calculate_pattern_match"):
                    instance.calculate_pattern_match()
                    instance.show()
                elif hasattr(instance, "calculate_pipeline_pd_pm"):
                    instance.calculate_pipeline_pd_pm()
                    instance.show()
                elif hasattr(instance, "calculate_fba"):
                    result = instance.calculate_fba()
                    if not result:
                        continue
                    instance.show()

                # Restore last geometry if available; otherwise fallback to default cascade
                default_geo = {"x": width, "y": height, "w": window_width, "h": window_height}
                geo = self._get_analysis_window_geometry(instance_key) if instance_key else None
                if geo is None:
                    geo = default_geo
                    # Persist the default once so next run restores from the same place
                    if instance_key:
                        self._set_analysis_window_geometry(instance_key, geo)
                instance.setGeometry(int(geo["x"]), int(geo["y"]), int(geo["w"]), int(geo["h"]))
                instance.setMinimumSize(QSize(200, 155))

                # Install event filter to capture move/resize and persist geometry (no close listener)
                if instance_key:
                    self._analysis_window_key_by_obj[instance] = instance_key
                    instance.installEventFilter(self)

                width += 20
                height += 20

            self._handle_post_analysis_exports()
            if self.count_board.mode == "test":
                # Test mode: decide label from analysis_result_dict summary and auto-finalize.
                can_output, _reason = self._can_output_ok_ng()
                if not can_output:
                    MessageBox.warning(self, "提示", "当前配置无法产出 OK/NG 汇总结果，无法执行测试模式自动判定。")
                else:
                    _passed, label = self._summarize_ok_ng()
                    try:
                        self.count_board.set_test_result_file(label)
                        self.count_board.set_test_text()
                    except Exception:
                        pass
                    self._finalize_test_run(label)

        # Show summary window at the end (also in test mode), only if dict is not empty
        self._maybe_show_analysis_result_summary(width, height)
        self._send_tcp_analysis_result_callback()

    def _handle_post_analysis_exports(self):
        # Cache first so MES and Excel both read the same completed-analysis state.
        self._capture_excel_export_cache()
        try:
            self._maybe_write_mes_result()
        except Exception as e:
            self.default_logger.warning(f"mes_export_error_after_analysis: {e}")
        try:
            self._maybe_export_excel_results()
        except Exception as e:
            self.default_logger.warning(f"excel_export_error_after_mes: {e}")

    @staticmethod
    def _validate_mes_summary_input(result_dict):
        if not isinstance(result_dict, dict):
            return False, "analysis_result_dict is not a dict"
        if not result_dict:
            return False, "analysis_result_dict is empty"

        for item_name, item_result in result_dict.items():
            if not isinstance(item_result, (tuple, list)):
                return False, f"{item_name!r} is not a tuple/list"
            if len(item_result) < 1:
                return False, f"{item_name!r} has no status flag"
            if not isinstance(item_result[0], (bool, np.bool_)):
                return False, f"{item_name!r} has non-boolean status flag"

        return True, ""

    def _maybe_write_mes_result(self):
        result_dict = getattr(self.data_struct, "analysis_result_dict", None)
        if not isinstance(result_dict, dict) or not result_dict:
            return

        can_output, _reason = self._can_output_ok_ng()
        if not can_output:
            return

        cfg_name, mes_cfg = select_mes_export_config(self.analysis_config)
        if not mes_cfg:
            return

        try:
            sn = (self.lineedit_s_or_n.text() or "").strip()
        except Exception:
            sn = ""
        if not sn:
            return
        if any(ch in sn for ch in (",", "\r", "\n")):
            self.default_logger.warning(f"mes_write_skip_bad_sn[{cfg_name}]: {sn!r}")
            return

        cfg_ok, cfg_message = _validate_mes_runtime_config(mes_cfg)
        if not cfg_ok:
            self.default_logger.warning(f"mes_write_skip_bad_config[{cfg_name}]: {cfg_message}")
            return

        summary_input_ok, summary_input_message = SequenceWindow._validate_mes_summary_input(result_dict)
        if not summary_input_ok:
            self.default_logger.warning(f"mes_write_skip_invalid_summary_input[{cfg_name}]: {summary_input_message}")
            return

        try:
            summary = self._summarize_ok_ng()
        except Exception as e:
            self.default_logger.warning(f"mes_write_skip_summary_error[{cfg_name}]: {e}")
            return
        if not isinstance(summary, tuple) or len(summary) != 2:
            self.default_logger.warning(f"mes_write_skip_bad_summary[{cfg_name}]: {summary!r}")
            return

        _passed, label = summary
        if label not in ("OK", "NG"):
            self.default_logger.warning(f"mes_write_skip_bad_label[{cfg_name}]: {label!r}")
            return
        ret = write_mes_result(mes_cfg, sn=sn, label=label, logger=self.default_logger)
        if not ret.ok:
            self.default_logger.warning(f"mes_write_fail[{cfg_name}]: {ret.message}")

    def _maybe_show_analysis_result_summary(self, width: int, height: int):
        result_dict = getattr(self.data_struct, "analysis_result_dict", None)
        if not isinstance(result_dict, dict) or len(result_dict) == 0:
            return

        # Create or reuse summary window
        if self._analysis_result_summary_window is None:
            self._analysis_result_summary_window = AnalysisResultSummaryWindow(result_dict)
        else:
            try:
                self._analysis_result_summary_window.set_results(result_dict)
            except Exception:
                # fallback: recreate if something went wrong
                self._analysis_result_summary_window = AnalysisResultSummaryWindow(result_dict)

        summary = self._analysis_result_summary_window
        summary_key = "__analysis_result_summary__"
        setattr(summary, "_sequence_analysis_key", summary_key)

        # Restore geometry if available; otherwise cascade default near the other windows
        default_geo = {"x": width, "y": height, "w": 520, "h": 360}
        geo = self._get_analysis_window_geometry(summary_key)
        if geo is None:
            geo = default_geo
            self._set_analysis_window_geometry(summary_key, geo)
        summary.setGeometry(int(geo["x"]), int(geo["y"]), int(geo["w"]), int(geo["h"]))
        summary.setMinimumSize(QSize(360, 220))

        # Ensure eventFilter is installed once for persistence
        if summary_key:
            if summary not in self._analysis_window_key_by_obj:
                self._analysis_window_key_by_obj[summary] = summary_key
                summary.installEventFilter(self)

        summary.show()
        summary.raise_()
        try:
            summary.activateWindow()
        except Exception:
            pass

    def _capture_excel_export_cache(self):
        """
        Cache current analysis results for later Excel export.

        Export may be triggered immediately after analysis (and also on OK/NG click / test finalization),
        with per-record dedupe to avoid duplicate writes when users rerun analysis.
        """
        try:
            record_id = None
            if isinstance(self.recorded_signal_info, dict):
                record_id = self.recorded_signal_info.get("file_path")
            if not record_id:
                record_id = self.recorded_path

            now_dt = datetime.now()
            sn = ""
            product_model = ""
            # Second column includes accurate time (to seconds): YYYY/M/D HH:MM:SS
            time_part = now_dt.strftime("%H:%M:%S")
            date_text = f"{now_dt.year}/{now_dt.month}/{now_dt.day} {time_part}"
            if isinstance(self.recorded_signal_info, dict):
                sn = self.recorded_signal_info.get("barcode") or ""
            try:
                product_model = self.lineedit_type.text() or ""
            except Exception:
                product_model = ""

            analysis_items_data = {}
            for inst in self.analysis_window or []:
                key = getattr(inst, "_sequence_analysis_key", None)
                if not key:
                    continue
                cfg = self.analysis_config.get(key)
                if not isinstance(cfg, dict):
                    continue
                t = cfg.get("type")
                if not t or t == "Excel":
                    continue
                item = {"type": t, "result": getattr(inst, "result", None)}
                detail = getattr(inst, "export_detail", None)
                if isinstance(detail, dict):
                    item.update(detail)
                analysis_items_data[key] = item

            self._excel_export_cache = {
                "record_id": record_id,
                "sn": sn,
                "product_model": product_model,
                "date_text": date_text,
                "analysis_items_data": analysis_items_data,
                "analysis_result_dict": dict(getattr(self.data_struct, "analysis_result_dict", {}) or {}),
            }
        except Exception as e:
            self.default_logger.error(f"capture_excel_export_cache_error: {e}")
            self._excel_export_cache = None

    def _schedule_excel_spool_build(self, excel_cfg_list):
        """
        Debounced Excel builder for CSV-spool mode.

        - Per record: only append to CSV (fast)
        - On idle: rebuild the daily .xlsx from CSV (write_only, faster than incremental save)
        """
        try:
            for cfg_name, excel_cfg, file_path, spool_dir in list(excel_cfg_list or []):
                try:
                    fp = str(file_path)
                    sd = str(spool_dir)
                    key = (str(cfg_name), os.path.normcase(os.path.abspath(fp)))
                    item = (cfg_name, excel_cfg, fp, sd)
                    self._excel_spool_targets_map[key] = item
                    self._excel_spool_build_pending_map[key] = item
                except Exception as e:
                    self.default_logger.error(f"excel_spool_schedule_path_error[{cfg_name}]: {e}")

            self._excel_spool_build_pending_cfgs = list(self._excel_spool_build_pending_map.values())
            if self._excel_spool_build_pending_cfgs:
                self._excel_spool_build_timer.start(self._excel_spool_build_delay_ms)
        except Exception as e:
            self.default_logger.error(f"excel_spool_schedule_error: {e}")

    def _on_excel_spool_build_timeout(self):
        """
        Called after a quiet period to rebuild .xlsx from CSV spool in a background thread.
        """
        try:
            # Avoid running during active play/record; keep cycle time stable.
            if getattr(self, "player_status_flag", False):
                self._excel_spool_build_timer.start(self._excel_spool_build_delay_ms)
                return

            with self._excel_spool_build_lock:
                if self._excel_spool_build_in_progress:
                    self._excel_spool_build_timer.start(self._excel_spool_build_delay_ms)
                    return
                pending_map = getattr(self, "_excel_spool_build_pending_map", None)
                if not isinstance(pending_map, dict) or not pending_map:
                    return
                pending = list(pending_map.values())
                pending_map.clear()
                self._excel_spool_build_pending_cfgs = []
                self._excel_spool_build_in_progress = True

            def _worker(cfgs):
                try:
                    for cfg_name, excel_cfg, file_path, spool_dir in cfgs:
                        ret = build_excel_from_csv_spool(excel_cfg, file_path=file_path, spool_dir=spool_dir)
                        if ret.ok:
                            self.default_logger.info(f"excel_spool_build_ok[{cfg_name}]: {ret.message}")
                        else:
                            self.default_logger.warning(f"excel_spool_build_fail[{cfg_name}]: {ret.message}")
                except Exception as e:
                    self.default_logger.error(f"excel_spool_build_error: {e}")
                finally:
                    with self._excel_spool_build_lock:
                        self._excel_spool_build_in_progress = False

            t = threading.Thread(target=_worker, args=(pending,), daemon=False)
            self._excel_spool_build_thread = t
            t.start()
        except Exception as e:
            self.default_logger.error(f"excel_spool_build_timeout_error: {e}")
            with self._excel_spool_build_lock:
                self._excel_spool_build_in_progress = False

    def _maybe_export_excel_results(self):
        """
        Export selected analysis items to Excel, if the global Excel analysis item exists in config.
        Shows retry dialog on failure allowing user to close open files and retry.
        """
        # Find Excel exporter config(s)
        excel_cfg_list = []
        for k, v in (self.analysis_config or {}).items():
            if not isinstance(v, dict):
                continue
            if v.get("type") == "Excel":
                excel_cfg_list.append((k, v))
        if not excel_cfg_list:
            return

        record_id = None
        if isinstance(self.recorded_signal_info, dict):
            record_id = self.recorded_signal_info.get("file_path")
        if not record_id:
            record_id = self.recorded_path
        if not record_id:
            return

        if self._excel_exported_record_id == record_id:
            return

        cache = self._excel_export_cache
        if not isinstance(cache, dict) or cache.get("record_id") != record_id:
            self.default_logger.warning("excel_export_skip: no matching cached analysis results for current record")
            return

        sn = cache.get("sn") or ""
        product_model = cache.get("product_model") or ""
        now_dt = datetime.now()
        date_text = f"{now_dt.year}/{now_dt.month}/{now_dt.day} {now_dt.strftime('%H:%M:%S')}"
        analysis_items_data = cache.get("analysis_items_data") or {}
        analysis_result_dict = cache.get("analysis_result_dict") or {}

        # Retry loop for export operations
        while True:
            all_ok = True
            spool_cfgs = []
            failed_exports = []

            for cfg_name, excel_cfg in excel_cfg_list:
                use_spool = bool(excel_cfg.get("fast_mode", True))
                if use_spool:
                    try:
                        file_path = resolve_excel_output_path(excel_cfg, product_model=product_model)
                        spool_dir = resolve_excel_spool_dir(excel_cfg, file_path=file_path)
                        spool_cfgs.append((cfg_name, excel_cfg, file_path, spool_dir))
                        ret = export_analysis_to_csv_spool(
                            excel_cfg,
                            sn=sn,
                            date_text=date_text,
                            analysis_items_data=analysis_items_data,
                            analysis_config=self.analysis_config,
                            analysis_result_dict=analysis_result_dict,
                            product_model=product_model,
                            file_path=file_path,
                            spool_dir=spool_dir,
                        )
                    except Exception as e:
                        ret = ExportResult(ok=False, message=f"导出异常: {e}")
                else:
                    try:
                        file_path = resolve_excel_output_path(excel_cfg, product_model=product_model)
                        ret = export_analysis_to_excel(
                            excel_cfg,
                            sn=sn,
                            date_text=date_text,
                            analysis_items_data=analysis_items_data,
                            analysis_config=self.analysis_config,
                            analysis_result_dict=analysis_result_dict,
                            product_model=product_model,
                            file_path=file_path,
                        )
                    except Exception as e:
                        ret = ExportResult(ok=False, message=f"导出异常: {e}")
                if ret.ok:
                    if use_spool:
                        self.default_logger.info(f"excel_spool_ok[{cfg_name}]: {ret.message}")
                    else:
                        self.default_logger.info(f"excel_export_ok[{cfg_name}]: {ret.message}")
                else:
                    all_ok = False
                    if use_spool:
                        self.default_logger.error(f"excel_spool_fail[{cfg_name}]: {ret.message}")
                    else:
                        self.default_logger.error(f"excel_export_fail[{cfg_name}]: {ret.message}")
                    failed_exports.append((cfg_name, ret.message))

            if all_ok:
                self._excel_exported_record_id = record_id
                if spool_cfgs:
                    self._schedule_excel_spool_build(spool_cfgs)
                break

            # If there are failures, show retry dialog
            msg_box = MessageBox(self)
            msg_box.setIcon(MessageBox.Warning)
            msg_box.setWindowTitle("数据保存失败")
            msg_box.setText(
                "无法保存数据到文件，可能是文件被占用、保存目录不可达或权限不足。\n请检查保存目录或关闭相关文件后重试。"
            )
            try:
                if failed_exports:
                    detail_lines = [f"{name}: {msg}" for name, msg in failed_exports[:5]]
                    if len(failed_exports) > 5:
                        detail_lines.append("...")
                    msg_box.setInformativeText("\n".join(detail_lines))
            except Exception:
                pass
            retry_btn = msg_box.addButton("重试", MessageBox.AcceptRole)
            msg_box.addButton("忽略", MessageBox.RejectRole)
            msg_box.setDefaultButton(retry_btn)
            msg_box.exec_()

            if msg_box.clickedButton() == retry_btn:
                continue
            else:
                break

    def _show_channel_mismatch_warning(self, analysis_name: str, err: Exception = None, mismatch_info: dict = None):
        configured_channel_text = "未知"
        active_channels_text = "未知"
        if isinstance(mismatch_info, dict):
            raw_channel = mismatch_info.get("raw_channel")
            active_channels = mismatch_info.get("active_input_channels", [])
            try:
                configured_channel_text = f"In{int(raw_channel) + 1}"
            except Exception:
                configured_channel_text = str(raw_channel)
            try:
                active_channels_text = ", ".join([f"In{int(ch) + 1}" for ch in active_channels]) or "无"
            except Exception:
                active_channels_text = str(active_channels)

        detail_text = ""
        if err is not None:
            detail_text = f"\n\n详细信息: {err}"

        MessageBox.warning(
            self,
            "通道配置不匹配",
            f"{analysis_name} 配置通道与本次录制通道不一致。\n"
            f"当前配置通道: {configured_channel_text}\n"
            f"本次录制通道: {active_channels_text}\n"
            f"请在分析参数中重新选择通道后再分析。"
            f"{detail_text}",
        )

    def instance_analysis_class(self, key, type, params):
        """
        Instantiates and configures an analysis class based on the given type and parameters,
        and adds it to the analysis window list.
        """
        class_mapping = get_class_mapping()
        requires_v2pa = type in self.analysis_types_requiring_v2pa
        if type in class_mapping.keys():
            cls_map = class_mapping.get(type)
            if cls_map:
                if self.mode == "RECORD_ONLY":
                    mapped_channel = 0
                    raw_channel = 0
                    if isinstance(params, dict):
                        try:
                            raw_channel = int(params.get("analysis_channel", 0) or 0)
                        except (TypeError, ValueError):
                            raw_channel = 0
                    if raw_channel < 0:
                        raw_channel = 0
                    channel_mismatch = False
                    active_input_channels = [int(ch) for ch in (getattr(self, "_active_input_channels", None) or [0])]
                    if raw_channel in active_input_channels:
                        mapped_channel = int(active_input_channels.index(raw_channel))
                    else:
                        channel_mismatch = True
                    display_key = f"{key}--通道{raw_channel + 1}"
                    class_instance = cls_map(display_key)
                    class_instance.data_struct = self.data_struct
                    # Bind analysis key for geometry restore/persist
                    setattr(class_instance, "_sequence_analysis_key", key)
                    setattr(class_instance, "_channel_mismatch", channel_mismatch)
                    setattr(
                        class_instance,
                        "_channel_mismatch_info",
                        {
                            "raw_channel": raw_channel,
                            "active_input_channels": list(active_input_channels),
                        },
                    )
                    if requires_v2pa and not channel_mismatch:
                        try:
                            class_instance.v2pa_factor = resolve_analysis_v2pa_factor_for_channel(
                                raw_channel,
                                warn_callback=lambda msg: MessageBox.warning(self, "提示", msg),
                            )
                        except ValueError as err:
                            MessageBox.warning(self, "提示", str(err))
                            return
                    runtime_params = dict(params) if isinstance(params, dict) else {}
                    runtime_params["analysis_channel"] = mapped_channel
                    # Inject sequence-level golden baseline path into per-item params
                    if isinstance(getattr(self, "analysis_config", None), dict):
                        golden_path = self.analysis_config.get("golden_sample_result_path")
                        if golden_path:
                            runtime_params["golden_sample_result_path"] = golden_path
                    class_instance.analysis_config = runtime_params
                    self.analysis_window.append(class_instance)
                    return
                class_instance = cls_map(key)
                # Bind analysis key for geometry restore/persist
                setattr(class_instance, "_sequence_analysis_key", key)
                raw_channel = 0
                if isinstance(params, dict):
                    try:
                        raw_channel = int(params.get("analysis_channel", 0) or 0)
                    except (TypeError, ValueError):
                        raw_channel = 0
                if raw_channel < 0:
                    raw_channel = 0
                if requires_v2pa:
                    try:
                        class_instance.v2pa_factor = resolve_analysis_v2pa_factor_for_channel(
                            raw_channel,
                            warn_callback=lambda msg: MessageBox.warning(self, "提示", msg),
                        )
                    except ValueError as err:
                        MessageBox.warning(self, "提示", str(err))
                        return
                # Inject sequence-level golden baseline path into per-item params
                if isinstance(params, dict) and isinstance(getattr(self, "analysis_config", None), dict):
                    golden_path = self.analysis_config.get("golden_sample_result_path")
                    if golden_path:
                        params["golden_sample_result_path"] = golden_path
                class_instance.analysis_config = params
                self.analysis_window.append(class_instance)

    def eventFilter(self, obj, event):
        """
        Persist analysis window geometry on move/resize (no close handling).
        """
        try:
            if obj in self._analysis_window_key_by_obj:
                et = event.type()
                if et in (QEvent.Move, QEvent.Resize):
                    key = self._analysis_window_key_by_obj.get(obj)
                    if key:
                        rect = obj.geometry()
                        geo = {"x": rect.x(), "y": rect.y(), "w": rect.width(), "h": rect.height()}
                        self._set_analysis_window_geometry(key, geo)
        except Exception as e:
            # Never break Qt event loop
            self.default_logger.error(f"eventFilter geometry persist error: {e}")

        # 键盘事件捕获（扫码枪键盘楔入模式）
        try:
            # 型号/计数：单击进入编辑态（默认只读）
            # 设计：只读时单击解锁；编辑时单击不反向上锁（否则用户无法用鼠标定位光标）。
            # 回到只读：依赖失去焦点（lineedit_*_lose_focus 已处理）。
            if event.type() == QEvent.MouseButtonPress:
                if obj is self.lineedit_type or obj is self.lineedit_count:
                    try:
                        if getattr(self, "_record_workflow_busy", False) or getattr(self, "player_status_flag", False):
                            return True
                        if isinstance(obj, QLineEdit) and obj.isReadOnly():
                            obj.setReadOnly(False)
                            obj.setFocus()
                            obj.selectAll()
                            return True
                    except Exception:
                        pass

            if event.type() == QEvent.KeyPress and self.barcode_scanner_box.isChecked():
                now = time.monotonic()
                fw = QApplication.focusWidget()

                # 如果 HID 模式刚刚接收到条码，忽略所有键盘输入（避免 HID 和键盘模式同时工作的重复问题）
                ch = event.text()
                # "最简焦点方案"下，型号/计数输入框永远不拦截（保证手动输入不受影响）
                if (
                    ch
                    and ch.isprintable()
                    and now < self._hid_mode_active_until
                    and fw is not self.lineedit_type
                    and fw is not self.lineedit_count
                ):
                    return True  # 吞掉事件

                # 焦点在 S/N 输入框
                if fw is self.lineedit_s_or_n:
                    # HID 模式激活窗口内，吞掉键盘输入（避免 HID + 键盘模式重复导致 S/N 内容翻倍）
                    if ch and ch.isprintable() and now < self._hid_mode_active_until:
                        return True  # 吞掉事件

                    if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_Z:
                        return True

                    key = event.key()
                    manual_edit_keys = {
                        Qt.Key_Backspace,
                        Qt.Key_Delete,
                        Qt.Key_Left,
                        Qt.Key_Right,
                        Qt.Key_Up,
                        Qt.Key_Down,
                        Qt.Key_Home,
                        Qt.Key_End,
                        Qt.Key_PageUp,
                        Qt.Key_PageDown,
                    }
                    if key in manual_edit_keys:
                        self._sn_textchange_manual_guard = True
                        self._reset_barcode_commit_state()
                        return super().eventFilter(obj, event)

                    if ch and ch.isprintable() and not ch.isspace():
                        try:
                            current_text = self.lineedit_s_or_n.text()
                            selected_text = self.lineedit_s_or_n.selectedText()
                        except Exception:
                            current_text = ""
                            selected_text = ""
                        if not current_text or (selected_text and len(selected_text) == len(current_text)):
                            self._sn_textchange_manual_guard = False

                    # 在"待确认"状态下，下一次扫码先清空旧内容，避免拼接
                    if (
                        self._sn_clear_on_next_scan
                        and not self.player_status_flag
                        and ch
                        and ch.isprintable()
                        and not ch.isspace()
                    ):
                        try:
                            with QSignalBlocker(self.lineedit_s_or_n):
                                self.lineedit_s_or_n.clear()
                            self._barcode_first_char_ts = None
                            self._barcode_last_char_ts = None
                            self._barcode_debounce_timer.stop()
                            self._sn_textchange_manual_guard = False
                            self._sn_clear_on_next_scan = False
                        except Exception:
                            pass
                    return super().eventFilter(obj, event)

                # 其余键盘事件交给 BarcodeRouter 做路由/吞键/缓冲
                try:
                    handled = self._barcode_router.handle_keypress(obj, event)
                    if handled is True:
                        return True
                except Exception:
                    # router 逻辑异常时不影响主流程
                    pass

        except Exception:
            # 出异常时不影响主流程
            return super().eventFilter(obj, event)

        return super().eventFilter(obj, event)

    def _load_analysis_window_geometry(self):
        """
        Load persisted analysis window geometries.
        """
        try:
            if not os.path.exists(self._analysis_window_geometry_path):
                os.makedirs(os.path.dirname(self._analysis_window_geometry_path), exist_ok=True)
                with open(self._analysis_window_geometry_path, "w", encoding="utf-8") as f:
                    json.dump({}, f, indent=2, ensure_ascii=False)
                return {}
            with open(self._analysis_window_geometry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            self.default_logger.warning(f"Failed to load analysis window geometry: {e}")
            return {}

    def _flush_analysis_window_geometry(self):
        """
        Flush geometry cache to disk (atomic write).
        """
        if not self._analysis_window_geometry_dirty:
            return
        try:
            os.makedirs(os.path.dirname(self._analysis_window_geometry_path), exist_ok=True)
            tmp_path = self._analysis_window_geometry_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._analysis_window_geometry, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self._analysis_window_geometry_path)
            self._analysis_window_geometry_dirty = False
        except Exception as e:
            self.default_logger.warning(f"Failed to save analysis window geometry: {e}")

    def _set_analysis_window_geometry(self, key: str, geo: dict):
        """
        Update in-memory geometry and schedule a debounced flush.
        """
        if not key or not isinstance(geo, dict):
            return
        norm = self._normalize_geometry(geo)
        if norm is None:
            return
        self._analysis_window_geometry[key] = norm
        self._analysis_window_geometry_dirty = True
        # Debounce writes to avoid heavy IO while dragging
        if not self._analysis_window_geometry_flush_timer.isActive():
            self._analysis_window_geometry_flush_timer.start(200)

    def _get_analysis_window_geometry(self, key: str):
        """
        Get persisted geometry if valid and on-screen; otherwise None.
        """
        if not key:
            return None
        geo = self._analysis_window_geometry.get(key)
        geo = self._normalize_geometry(geo) if isinstance(geo, dict) else None
        if geo is None:
            return None
        if not self._is_geometry_on_any_screen(geo):
            return None
        return geo

    @staticmethod
    def _normalize_geometry(geo: dict):
        """
        Ensure geometry has x/y/w/h ints with sane bounds.
        """
        if not isinstance(geo, dict):
            return None
        try:
            x = int(geo.get("x"))
            y = int(geo.get("y"))
            w = int(geo.get("w"))
            h = int(geo.get("h"))
            # basic sanity
            if w < 200 or h < 150:
                return None
            return {"x": x, "y": y, "w": w, "h": h}
        except Exception:
            return None

    @staticmethod
    def _is_geometry_on_any_screen(geo: dict) -> bool:
        """
        Check whether the saved top-left point is within any screen's available geometry.
        """
        try:
            x, y = int(geo["x"]), int(geo["y"])
            for screen in QApplication.screens():
                ag = screen.availableGeometry()
                if ag.contains(x, y):
                    return True
            return True if QApplication.primaryScreen() is None else False
        except Exception:
            return False

    def del_geometry_config(self):
        if (
            hasattr(self, "_analysis_window_geometry_flush_timer")
            and self._analysis_window_geometry_flush_timer.isActive()
        ):
            self._analysis_window_geometry_flush_timer.stop()

        self._analysis_window_geometry = {}
        self._analysis_window_geometry_dirty = False

        file_path = self._analysis_window_geometry_path
        if os.path.exists(file_path):
            os.remove(file_path)

    def get_sequence_config_from_registry(self):
        """
        Retrieves the sequence configuration from the registry.
        """
        registry = LoadUiConfig._load_sequence_config_registry()
        # IMPORTANT: Do not auto-add "默认配置" into registry.
        # The combobox should either never show it, or always show it if it already exists in registry.
        user_keys = [k for k in (registry or {}).keys() if k not in ("using_config_path", "默认配置")]
        using_config_path = registry.get("using_config_path", None)
        default_path = registry.get("默认配置")
        # Fallback when using_config_path missing or points to a non-existent file.
        if (not using_config_path) or (isinstance(using_config_path, str) and not os.path.exists(using_config_path)):
            fallback_path = None
            # Prefer user saved/imported entries (if any), otherwise fallback to built-in default.
            if user_keys:
                for k in sorted(user_keys):
                    p = registry.get(k)
                    if isinstance(p, str) and os.path.exists(p):
                        fallback_path = p
                        break
            if not fallback_path and isinstance(default_path, str) and os.path.exists(default_path):
                fallback_path = default_path

            using_config_path = fallback_path
            if using_config_path:
                LoadUiConfig.update_using_config_path(using_config_path)
        return using_config_path, registry

    def update_using_file_combobox(self):
        """
        Updates the using file combobox.
        """
        self.using_config_path, self.registry = self.get_sequence_config_from_registry()
        # Updating items will trigger currentTextChanged multiple times (clear/add/setIndex).
        # Block signals to avoid re-entrant loads and transient empty-text callbacks.
        self.using_file_combobox.blockSignals(True)
        try:
            self.using_file_combobox.clear()
            self.add_file_to_using_file_combobox()
        finally:
            self.using_file_combobox.blockSignals(False)

    def add_file_to_using_file_combobox(self):
        """
        Adds file to the using file combobox.
        """
        selected_key = None
        using_path = self.registry.get("using_config_path")

        keys = [k for k in (self.registry or {}).keys() if k != "using_config_path"]
        # Do NOT filter "默认配置" by business logic here:
        # - If registry contains "默认配置", it must appear in combobox (always pinned on top).
        # - If registry does not contain it, it won't appear.
        ordered_keys = []
        if "默认配置" in keys:
            ordered_keys.append("默认配置")
            keys.remove("默认配置")
        ordered_keys.extend(sorted(keys))

        # Only show entries whose file path exists. If none exist, show "无配置".
        visible_count = 0
        for key in ordered_keys:
            value = self.registry.get(key)
            if not isinstance(value, str) or (value and not os.path.exists(value)):
                continue
            self.using_file_combobox.addItem(key, value)
            visible_count += 1
            if using_path and value == using_path:
                selected_key = key

        if visible_count == 0:
            self.using_file_combobox.addItem("无配置", None)
            selected_key = "无配置"

        # Select the item matching the current using_config_path (if any)
        if selected_key:
            idx = self.using_file_combobox.findText(selected_key)
            if idx >= 0:
                self.using_file_combobox.setCurrentIndex(idx)

    def on_using_file_combobox_changed(self, text):
        """
        Handles the change of the using file combobox.
        """
        # Prefer the item's userData (full file path). During combobox refresh,
        # `text` may temporarily be empty which would otherwise resolve to None.
        if self.player_status_flag:
            self.restore_previous_configuration()
            MessageBox.warning(self, "警告", "正在录音，请稍后...")
            return

        path = None
        try:
            path = self.using_file_combobox.currentData()
        except Exception:
            path = None
        if not path:
            path = self.registry.get(text)

        self.using_config_path = path
        LoadUiConfig.update_using_config_path(self.using_config_path)
        self.get_sequence_config_from_json()
        self.refresh_channel_windows()
        self.init_data_struct_stimulus_config()
        self.update_player_btn_is_paused()
        self.replayer_btn.setDisabled(True)
        self.data_btn.setDisabled(True)
        self.data_struct.store_wave_data = None
        self.data_struct.store_wave_data_multi = None

        # 1. 强制清除下拉框焦点
        self.using_file_combobox.clearFocus()
        # 2. 尝试聚焦 S/N 框 (提升体验)
        if self.lineedit_s_or_n.isEnabled():
            try:
                self.lineedit_s_or_n.setFocus()
                self.lineedit_s_or_n.selectAll()  # 全选，方便覆盖旧条码
            except Exception:
                pass
        else:
            self.setFocus()  # 给主窗口，依靠 BarcodeRouter 后台捕获

    def restore_previous_configuration(self):
        """恢复到之前的配置选项"""
        index = self.using_file_combobox.findData(self.using_config_path)
        if index >= 0:
            self.using_file_combobox.blockSignals(True)
            self.using_file_combobox.setCurrentIndex(index)
            self.using_file_combobox.blockSignals(False)
            self.default_logger.warning("已恢复到之前的配置选项")

    def get_sequence_config_from_json(self):
        """
        Retrieves the sequence configuration from a JSON file.

        This method attempts to load the sequence configuration from a JSON file by calling the `load_sequence_from_json()` method.
        If the loading is successful and the result is valid, it returns the configuration; otherwise, it returns an empty dictionary.

        Returns:
            dict: The sequence configuration if loading is successful and the result is valid; otherwise, an empty dictionary.
        """
        # Avoid noisy stdout prints; use logger if needed for debugging.
        # self.default_logger.debug(f"Loading sequence config: {self.using_config_path}")
        load_code, result = LoadUiConfig().load_sequence_config_from_json(self.using_config_path)
        if load_code == error_code.OK and result:
            self.sequence_config = result
            seq = self.sequence_config[0]["seq1"]
            self.analysis_config = seq.get("analysis_list", {})
            mode = seq["acq"]["mode"]
            self.mode = mode
            if mode == "IMPORT_AUDIO":
                self.replayer_btn.setDisabled(True)
            if self.count_board:
                self.count_board.analysis_config = seq.get("analysis_list", {})
                self._refresh_test_mode_availability()
            self._set_sequence_config_available_state(True)
            self._missing_config_prompted = False
        else:
            self.sequence_config = []
            self.analysis_config = dict()
            self.mode = None
            self._set_sequence_config_available_state(False)
            # Only show prompt after login success (window shown).
            if getattr(self, "_missing_config_prompt_enabled", False) and not getattr(
                self, "_missing_config_prompted", False
            ):
                MessageBox.warning(
                    self,
                    "提示",
                    "当前未找到可用配置文件。\n"
                    "请在上方【使用配置】下拉框中选择配置；\n"
                    "如无可选项，请到【功能-测试队列】中保存或导入配置。",
                )
                self._missing_config_prompted = True

    def _set_sequence_config_available_state(self, available: bool):
        """
        Enable/disable key actions based on whether a valid sequence config is loaded.
        """
        try:
            if available:
                self.update_player_btn_is_paused()
                # replay/data depend on runtime state; keep conservative defaults here
            else:
                self.player_btn.setDisabled(True)
                self.replayer_btn.setDisabled(True)
                self.data_btn.setDisabled(True)
        except Exception:
            # During early init some widgets may not be ready; ignore safely.
            pass

    def on_audio_chunk_received(self, chunk):
        if self.streaming_mode == "play_record":
            self.on_audio_chunk_received_playrec(chunk)
        elif self.streaming_mode == "record_only":
            self.on_audio_chunk_received_rec(chunk)

    def on_audio_chunk_received_playrec(self, payload):
        """
        Handle streaming audio chunk for real-time waveform display.

        Updates plot incrementally while preserving zoom/pan state.
        Also writes chunk to file via connected wav_writer.

        Args:
            chunk (np.ndarray | dict): Audio chunk received from streaming processor
        """
        if isinstance(payload, dict):
            mono_source = payload.get("mono")
            multi_source = payload.get("multi")
            if multi_source is not None:
                multi_chunk = np.asarray(multi_source, dtype=np.float32)
                if multi_chunk.ndim == 0:
                    multi_chunk = multi_chunk.reshape(1, 1)
                elif multi_chunk.ndim == 1:
                    multi_chunk = multi_chunk.reshape(-1, 1)

                wav_chunk = multi_chunk
                if mono_source is None:
                    if multi_chunk.shape[1] == 1:
                        plot_chunk = multi_chunk[:, 0]
                    else:
                        plot_chunk = multi_chunk.mean(axis=1).astype(np.float32, copy=False)
                else:
                    plot_chunk = np.asarray(mono_source, dtype=np.float32)
            elif mono_source is not None:
                plot_chunk = np.asarray(mono_source, dtype=np.float32)
            else:
                plot_chunk = np.array([], dtype=np.float32)
        else:
            plot_chunk = np.asarray(payload, dtype=np.float32)

        if plot_chunk.ndim > 1:
            plot_chunk = plot_chunk[:, 0]

        # Append mono display chunk to streaming buffer
        mono = plot_chunk.reshape(-1)
        self.streaming_buffer_multi.append(mono)

        # Calculate accumulated audio data
        accumulated_audio = np.concatenate(self.streaming_buffer_multi)

        # Calculate time axis for accumulated data
        # Use np.arange to ensure fixed time stamps for each sample point
        # This prevents time drift caused by linspace's varying interval (N/(N-1)/fs instead of 1/fs)
        sample_rate = self.data_struct.sample_rate
        time_axis = np.arange(len(accumulated_audio)) / sample_rate

        # Update plot - preserve zoom/pan by updating existing PlotDataItem
        wins = self.channel_workspace.all_subwindows()
        wins[0].set_data(time_axis, accumulated_audio)
        if self.streaming_plot_item is None:
            # First chunk - create new plot item
            self.streaming_plot_item = wins[0]

        # Write chunk to file (if wav_writer is connected)
        if self.streaming_wav_writer:
            try:
                self.streaming_wav_writer.write_chunk(mono)
            except Exception as e:
                self.default_logger.error(f"Error writing audio chunk to file: {e}")

    def on_audio_chunk_received_rec(self, payload):
        """
        Handle streaming audio chunk for real-time waveform display.
        Args:
            payload (dict): {"mono": 1D ndarray, "multi": 2D ndarray}
        """
        if not isinstance(payload, dict):
            return
        multi = payload.get("multi")
        if multi is None:
            return
        multi_arr = np.asarray(multi, dtype=np.float32)
        if multi_arr.ndim == 1:
            multi_arr = multi_arr.reshape(-1, 1)
        if multi_arr.ndim != 2 or multi_arr.shape[0] <= 0:
            return

        if not getattr(self, "_streaming_first_chunk_logged", False):
            try:
                self.default_logger.info(f"First streaming chunk received: multi shape={multi_arr.shape}")
            except Exception:
                pass
            self._streaming_first_chunk_logged = True

        self.streaming_buffer_multi.append(multi_arr)
        if len(self.streaming_buffer_multi) == 1:
            accumulated = self.streaming_buffer_multi[0]
        else:
            accumulated = np.concatenate(self.streaming_buffer_multi, axis=0)

        sample_rate = float(self.data_struct.sample_rate or 1.0)
        time_axis = np.arange(accumulated.shape[0]) / sample_rate

        if self.channel_workspace is not None:
            wins = self.channel_workspace.all_subwindows()
        else:
            wins = []

        ch_n = int(accumulated.shape[1])
        for i, w in enumerate(wins):
            if i < ch_n:
                w.set_data(time_axis, accumulated[:, i])
            else:
                w.clear_plot()

        if self.streaming_wav_writer:
            try:
                self.streaming_wav_writer.write_chunk(multi_arr)
            except Exception as e:
                self.default_logger.error(f"Error writing audio chunk to file: {e}")

    def _poll_streaming_queue(self):
        """
        Poll streaming queue and check for completion.

        Called by QTimer every 50ms from Qt main thread.
        Processes audio chunks from queue and checks if recording is finished.
        """
        if self.streaming_processor is None:
            return

        # Process all available chunks from queue (non-blocking)
        self.streaming_processor.process_queue()

        # Check if recording is complete
        if not self.streaming_processor.is_recording:
            self.streaming_poll_timer.stop()
            self._on_streaming_complete()

    def _on_streaming_complete(self):
        """
        Handle streaming completion: alignment, file save, and analysis.

        Called when streaming recording is finished. Performs:
        - Get recorded data from processor
        - Alignment (for play+record mode)
        - Finalize WAV file
        - Store data for analysis
        - Save to database
        - Enable buttons and optionally run analysis
        """
        try:
            # Get the complete recorded data
            recorded_data = self.streaming_processor.get_recorded_data()
            recorded_data_multi = None
            if self.streaming_mode == "record_only":
                recorded_data_multi = self.streaming_processor.get_recorded_data_multi()
            sample_rate = self.data_struct.sample_rate

            # VERIFICATION: Check if we captured the expected number of samples
            expected_samples = self.streaming_processor.target_samples
            actual_samples = len(recorded_data)
            if actual_samples != expected_samples:
                self.default_logger.warning(
                    f"Sample count mismatch! Expected: {expected_samples}, Got: {actual_samples}, "
                    f"Missing: {expected_samples - actual_samples} samples ({(expected_samples - actual_samples) / sample_rate * 1000:.1f}ms)"
                )
            else:
                self.default_logger.info(f"Recording complete: {actual_samples} samples captured (matches target)")

            # Perform alignment if in play+record mode
            if self.streaming_mode == "play_record" and self.streaming_stimulus_data is not None:
                # Finalize temp file first
                if self.streaming_wav_writer:
                    self.streaming_wav_writer.finalize()
                    self.streaming_wav_writer = None

                aligned_data = AlignmentProcessing.align_play_and_rec_data_using_gccphat(
                    self.streaming_stimulus_data, recorded_data
                )

                # Update plot with aligned data (refresh display)
                final_time_axis = np.linspace(0, len(aligned_data) / sample_rate, len(aligned_data))
                if self.streaming_plot_item is not None:
                    self.streaming_plot_item.set_data(final_time_axis, aligned_data)

                # Store aligned data
                self.data_struct.store_wave_data = aligned_data
                self.data_struct.store_wave_data_multi = np.asarray(aligned_data, dtype=np.float32).reshape(-1, 1)

                # Save aligned data to final file
                save_audio_simple(self.recorded_path, aligned_data, sample_rate)

                # Save to database
                self.recorded_signal_info["sample_rate"] = sample_rate
                save_code, save_msg = RecordingManager().save_signal_info_to_db(
                    self.recorded_signal_info, self.data_struct.stimulus_info
                )
                if save_code == error_code.OK:
                    self.default_logger.info(f"Database save successful: {save_msg}")
                else:
                    self.default_logger.error(f"Database save failed: {save_msg}")

                # Delete temp file AFTER successful save (for data safety)
                try:
                    if hasattr(self, "streaming_temp_path") and os.path.exists(self.streaming_temp_path):
                        os.remove(self.streaming_temp_path)
                        self.default_logger.info(f"Deleted temp file: {self.streaming_temp_path}")
                except Exception as e:
                    self.default_logger.warning(f"Failed to delete temp file: {e}")

            else:
                # Record-only mode - no alignment needed
                if recorded_data_multi is not None:
                    recorded_array = np.asarray(recorded_data_multi, dtype=np.float32)
                else:
                    recorded_array = np.asarray(recorded_data, dtype=np.float32)
                if recorded_array.ndim == 2:
                    actual_input_channels = list(getattr(self.streaming_processor, "_rec_in_sel", []) or [])
                    if not actual_input_channels:
                        actual_input_channels = list(getattr(self, "_active_input_channels", []) or [])
                    if actual_input_channels:
                        self._active_input_channels = [int(ch) for ch in actual_input_channels[: recorded_array.shape[1]]]
                    if len(getattr(self, "_active_input_channels", []) or []) != recorded_array.shape[1]:
                        self._active_input_channels = list(range(recorded_array.shape[1]))
                    self.data_struct.store_wave_data_multi = recorded_array
                    if recorded_array.shape[1] == 1:
                        mono_recorded_data = recorded_array[:, 0]
                    elif recorded_array.shape[1] > 1:
                        mono_recorded_data = recorded_array.mean(axis=1).astype(np.float32, copy=False)
                    else:
                        mono_recorded_data = np.array([], dtype=np.float32)
                else:
                    mono_recorded_data = recorded_array.reshape(-1)
                    actual_input_channels = list(getattr(self.streaming_processor, "_rec_in_sel", []) or [])
                    self._active_input_channels = [int(actual_input_channels[0])] if actual_input_channels else [0]
                    self.data_struct.store_wave_data_multi = mono_recorded_data.reshape(-1, 1)
                self.data_struct.store_wave_data = mono_recorded_data

                # Finalize WAV file (for record-only, this is the final file)
                if self.streaming_wav_writer:
                    self.streaming_wav_writer.finalize()
                    self.streaming_wav_writer = None

                # Save to database
                self.recorded_signal_info["sample_rate"] = sample_rate
                save_code, save_msg = RecordingManager().save_signal_info_to_db(self.recorded_signal_info, None)
                if save_code == error_code.OK:
                    self.default_logger.info(f"Database save successful: {save_msg}")
                else:
                    self.default_logger.error(f"Database save failed: {save_msg}")

            # Handle repeat signal splitting if needed
            if self.streaming_mode == "play_record":
                repeat_times = self.data_struct.stimulus_info.get("repeat_times", 1)
                if repeat_times > 1:
                    kwargs = {"repeat_times": repeat_times}
                    self.data_struct.split_repeat_data = SplitRepeatSignal().split_repeat_signal(
                        self.data_struct.store_wave_data, sample_rate, **kwargs
                    )

            # Clean up streaming state
            self.streaming_processor = None
            self.streaming_stimulus_data = None
            self.streaming_mode = None
            # Recording has fully ended; release S/N input and busy gating before any follow-up analysis/UI work.
            self.player_status_flag = False  # Recording complete, allow hardware access
            self._record_workflow_busy = False
            self._set_sn_input_recording_read_only(False)

            # Enable buttons for replay and data analysis
            self.data_btn.setEnabled(True)
            self.replayer_btn.setEnabled(True)

            self._awaiting_ok_ng = True
            self._sn_clear_on_next_scan = True
            # 更稳的体验：录音结束后让下一次扫码直接覆盖旧 S/N（避免拼接）
            if self.barcode_scanner_box.isChecked():
                try:
                    self.lineedit_s_or_n.setFocus()
                    self.lineedit_s_or_n.selectAll()
                except Exception:
                    pass

            # Run auto-analysis if enabled
            if self.analysis_config.get("auto_analysis", False):
                self.run()

            # Update player button state
            self.update_player_btn_is_paused()

            self.default_logger.info("Streaming recording completed successfully")

        except Exception as e:
            self.default_logger.error(f"Error in streaming completion: {e}")
            # Clean up on error
            if self.streaming_wav_writer:
                self.streaming_wav_writer.finalize()
                self.streaming_wav_writer = None
            self.streaming_processor = None
            self.streaming_stimulus_data = None
            self.streaming_mode = None
            self.player_status_flag = False  # Clear flag even on error to prevent permanent blocking
            # Still enable buttons even on error
            self.data_btn.setEnabled(True)
            self.replayer_btn.setEnabled(True)
            self._awaiting_ok_ng = False
            self._sn_clear_on_next_scan = False
            self._record_workflow_busy = False
            self._set_sn_input_recording_read_only(False)
            self.update_player_btn_is_paused()

    def _cleanup_streaming_resources(self):
        """
        Clean up all streaming resources (timer, processor, wav_writer).

        Called before starting a new recording to prevent resource conflicts.
        Safe to call multiple times (idempotent).
        """
        # 1. Stop timer first (prevent callbacks during cleanup)
        try:
            if self.streaming_poll_timer.isActive():
                self.streaming_poll_timer.stop()
                self.default_logger.debug("Stopped streaming poll timer")
        except Exception as e:
            self.default_logger.error(f"Error stopping timer: {e}")

        # 2. Stop processor (stops audio streams - CRITICAL for preventing device conflicts)
        try:
            if self.streaming_processor is not None:
                self.streaming_processor.stop_streaming()
                self.streaming_processor = None
                self.default_logger.debug("Stopped streaming processor")
        except Exception as e:
            self.default_logger.error(f"Error stopping processor: {e}")

        # 3. Finalize wav writer
        try:
            if self.streaming_wav_writer is not None:
                self.streaming_wav_writer.finalize()
                self.streaming_wav_writer = None
                self.default_logger.debug("Finalized wav writer")
        except Exception as e:
            self.default_logger.error(f"Error finalizing wav writer: {e}")

        # 4. Clean up other state
        self.streaming_stimulus_data = None
        self.streaming_mode = None

    def update_player_btn_is_playing(self):
        self.player_btn.setIcon(QIcon(":/ui/icon/pause.png"))
        self.player_btn.setIconSize(QSize(35, 35))
        self.player_btn.setDisabled(True)

    def update_player_btn_is_paused(self):
        self.player_btn.setIcon(QIcon(":/ui/icon/play.png"))
        self.player_btn.setIconSize(QSize(35, 35))
        can_start = bool(getattr(self, "sequence_config", None))
        can_start = can_start and not getattr(self, "player_status_flag", False)
        can_start = can_start and not getattr(self, "_record_workflow_busy", False)
        can_start = can_start and getattr(self, "audio_devices_available", True)
        self.player_btn.setDisabled(not can_start)

    def refresh_channel_windows(self) -> None:
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

        if (not self.sequence_config) or self.mode in (
            "PLAY_AND_RECORD",
            "IMPORT_STIMULUS_AUDIO",
            "IMPORT_AUDIO",
        ):
            self.channel_workspace.set_single_canvas_mode(channel_index=0)
            self.default_logger.info(f"Single-canvas mode activated for mode {self.mode}")
        else:
            if self.channel_workspace is not None:
                self.channel_workspace.set_channels(self._active_input_channels)
            self.default_logger.info(f"Plot workspace channels: {self._active_input_channels}")

    def _clear_plot_area(self) -> None:
        if self.channel_workspace is not None:
            self.channel_workspace.clear_plots()

    def plot_waveform_to_workspace(self, recorded_signal, sample_rate: float) -> None:
        """
        Plot waveform data to the channel subwindows.

        - If recorded_signal is 2D: shape (frames, channels), each channel plots to its own subwindow.
        - If recorded_signal is 1D: plot the same waveform to all subwindows (best-effort fallback).
        """
        if self.channel_workspace is None:
            return

        wins = self.channel_workspace.all_subwindows()
        if not wins:
            return

        if recorded_signal is None:
            self._clear_plot_area()
            return

        y = np.asarray(recorded_signal)
        if y.ndim == 1:
            frames = int(y.shape[0])
            if frames <= 0:
                self._clear_plot_area()
                return
            t = np.arange(frames) / float(sample_rate or 1.0)
            for w in wins:
                w.set_data(t, y)
            return

        if y.ndim == 2:
            frames = int(y.shape[0])
            if frames <= 0:
                self._clear_plot_area()
                return
            t = np.arange(frames) / float(sample_rate or 1.0)
            ch_n = int(y.shape[1])
            for i, w in enumerate(wins):
                if i < ch_n:
                    w.set_data(t, y[:, i])
                else:
                    w.clear_plot()
            return

        # Unexpected shape -> clear for safety
        self._clear_plot_area()

    def _close_analysis_windows(self):
        """
        关闭上一轮弹出的分析窗口/汇总窗口。
        目的：产线连续扫码时无需手工关闭弹窗。
        """
        # 关闭各分析窗口（self.analysis_window 里缓存的是窗口对象）
        try:
            if hasattr(self, "analysis_window") and self.analysis_window:
                for w in list(self.analysis_window):
                    try:
                        if w is not None:
                            w.close()
                    except Exception:
                        pass
                self.analysis_window = []
        except Exception:
            pass

        # 关闭汇总窗口
        try:
            if getattr(self, "_analysis_result_summary_window", None) is not None:
                try:
                    self._analysis_result_summary_window.close()
                except Exception:
                    pass
                self._analysis_result_summary_window = None
        except Exception:
            pass

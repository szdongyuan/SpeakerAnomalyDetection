import os
import threading
import weakref

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QWidget

from base.data_struct.data_deal_struct import DataDealStruct
from base.load_config import LoadUiConfig
from base.log_manager import LogManager
from base.soundcard_calibration_manager import get_mic_v2pa_factor
from base.unified_hid_device_manager import UnifiedHardwareManager
from consts.running_consts import DEFAULT_DIR
from ui.sequence.barcode_router import BarcodeRouter
from ui.sequence.motor_left_panel import MotorDetectionLeftPanel
from ui.sequence.sequence_tools_bar import SequenceToolsBar
from ui.sequence.sequencement_count_board import SequenceCountBoard

from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin
from ui.sequence.sequence_widget_test_metadata_ops import SequenceWidgetTestMetadataOpsMixin
from ui.sequence.sequence_widget_barcode_ops import SequenceWidgetBarcodeOpsMixin
from ui.sequence.sequence_widget_tcp_ops import SequenceWidgetTcpOpsMixin
from ui.sequence.sequence_widget_serial_trigger_ops import SequenceWidgetSerialTriggerOpsMixin
from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin
from ui.sequence.sequence_widget_config_ops import SequenceWidgetConfigOpsMixin
from ui.sequence.sequence_widget_product_pdf_ops import SequenceWidgetProductPdfOpsMixin
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
from ui.sequence.multichannel_waveform_session import MultichannelWaveformSession


class SequenceWindow(
    SequenceWidgetUiOpsMixin,
    SequenceWidgetTestMetadataOpsMixin,
    SequenceWidgetBarcodeOpsMixin,
    SequenceWidgetTcpOpsMixin,
    SequenceWidgetSerialTriggerOpsMixin,
    SequenceWidgetAnalysisOpsMixin,
    SequenceWidgetConfigOpsMixin,
    SequenceWidgetProductPdfOpsMixin,
    SequenceWidgetStreamingOpsMixin,
    QWidget,
):
    tcp_server = None
    _active_instance_ref = None

    def __init__(self, *, recording_bridge=None):
        """Initializes the class instance, setting up the user interface and necessary parameters."""
        super().__init__()
        self.recording_bridge = recording_bridge
        self._owns_recording_bridge = False
        self.data_struct = DataDealStruct()
        self.recorded_path = None
        self.count_board = None
        self.toolsbar = SequenceToolsBar()

        self.mic = None
        self.mic_channels = []
        self.speaker = None
        self.speaker_channels = []
        self.v2pa_factor = get_mic_v2pa_factor(
            self.mic,
            self.mic_channels,
        )
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
        self._excel_spool_build_pending_cfgs = []
        self._excel_spool_build_lock = threading.Lock()
        self._excel_spool_build_thread = None

        self.init_result_files()
        self.count_board = SequenceCountBoard(self.analysis_config)
        self.product_test_condition_configs = self.load_active_product_test_condition_configs()
        self.product_test_close_trigger_state = (
            self.load_active_product_test_close_trigger_state()
        )
        self.product_test_pdf_report_config = self.load_active_product_test_pdf_report_config()
        self._product_pdf_report_states = {}
        self._product_pdf_report_paths = {}
        self.default_logger = LogManager.set_log_handler("core")
        self.left_panel = MotorDetectionLeftPanel(
            self.count_board,
            condition_configs=self.product_test_condition_configs,
        )
        self._init_test_round_metadata()
        self._refresh_test_mode_availability()
        self.player_status_flag = False
        # True while a record run is still processing (record -> analysis -> save -> count updates).
        # Used to prevent starting a new recording before the full workflow completes.
        self._record_workflow_busy = False
        self.recorded_signal_info = {}
        self.ip_format = True
        self.port_format = True
        self.clicked_player_flag = False
        self.tcp_flag = False
        self.tcp_ip, self.tcp_port = LoadUiConfig.get_tcp_config()
        self.mode = None
        self.last_play_count = None  # Cache last run token for replay overwrite
        self._current_cycle_recorded_count = None
        self._current_run_recording_token = ""
        self._manual_product_condition_index = 0
        self._manual_product_condition_group_id = ""
        self._manual_product_condition_results = {}
        self._manual_product_condition_completed_keys = set()
        self._manual_product_condition_counted_group_labels = {}
        self._active_product_condition_key = ""
        self._active_product_condition_config = None

        self._missing_config_prompted = False
        # Only show missing-config prompt after the window is shown (i.e. after login success).
        self._missing_config_prompt_enabled = False

        self._barcode_debounce_ms = 120
        self._barcode_fast_input_max_seconds = 0.4  # 首字符到末字符的总耗时，小于该值更像扫码（扫码枪通常 < 0.2秒）
        self._barcode_min_length_for_auto_commit = 4  # 条码最小长度，太短容易误触发（可按实际条码规则调整）

        # 扫码逻辑委托到 BarcodeRouter（需先创建，再连接定时器/信号）
        self._barcode_router = BarcodeRouter(self)

        self._barcode_debounce_timer = QTimer(self)
        self._barcode_debounce_timer.setSingleShot(True)
        self._barcode_debounce_timer.setInterval(self._barcode_debounce_ms)
        # 扫码逻辑委托到 BarcodeRouter，SequenceWidget 只保留业务提交入口 _commit_barcode
        self._barcode_debounce_timer.timeout.connect(self._barcode_router.on_barcode_debounce_timeout)
        self._serial_trigger_delay_timer = QTimer(self)
        self._serial_trigger_delay_timer.setSingleShot(True)
        self._serial_trigger_delay_timer.timeout.connect(self._on_serial_trigger_delay_timeout)
        self._current_trigger_direction = ""
        self._active_recording_direction = ""
        self._manual_direction_fallback_next_direction = "forward"
        self._ai_cycle_started_at = ""
        self._current_cycle_first_direction = ""
        self._ai_cycle_direction_results = {"forward": None, "reverse": None}
        self._mark_cycle_direction_labels = {"forward": "not_labeled", "reverse": "not_labeled"}
        self._mark_cycle_summary_label = ""
        self._direction_waveform_cache = {"forward": None, "reverse": None}
        self._condition_record_cache = {}
        self._waveform_display_override_direction = ""
        self._pending_serial_trigger_direction = ""
        self._queued_directional_trigger = ""
        self._serial_product_condition_executing = False
        self._serial_product_session_started = False
        self._serial_product_latched_frame = ""
        self._serial_product_waiting_for_close = False
        self._product_test_program_config_dialog_open = False
        self._serial_product_error_dialog_open = False
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
        self._sn_locked_for_product_round = False

        # 条码提交去重机制：防止 HID 模式和键盘楔入模式同时触发导致重复提交
        self._last_committed_barcode = None
        self._last_committed_barcode_time = 0.0
        self._barcode_commit_dedup_window_sec = 0.8  # 800ms 去重窗口

        # HID 模式激活标志：当 HID 模式成功接收条码后，暂时忽略键盘楔入模式的输入
        self._hid_mode_active_until = 0.0  # 时间戳，在此之前忽略键盘输入

        # Streaming state variables
        self._streaming_waveform_session = MultichannelWaveformSession(
            max_points=self._WAVEFORM_DISPLAY_MAX_POINTS,
        )
        self._streaming_waveform_generation = 0
        self._streaming_waveform_refresh_scheduled = False
        self._streaming_waveform_pending = False
        self._streaming_waveform_live_enabled = False
        self._streaming_waveform_failure_logged = False
        self._streaming_chunk_contract_failed = False
        self.streaming_wav_writer = None  # WAV file writer for incremental saving
        self.streaming_processor = None  # StreamingAudioProcessor instance
        self._streaming_completion_processor = None
        self.streaming_stimulus_data = None  # Stimulus data for alignment (play+record mode)
        self.streaming_mode = None  # "play_record" or "record_only"
        self._configured_input_channels = None
        self._recording_input_channels = None
        self._pending_configured_input_channels = None
        self._channel_selection_error = ""
        self._active_input_channels = [0]
        self._waveform_presentation_owner = "hardware"
        self.channel_workspace = None
        self.recent_session_panel = None
        self.recent_test_sessions = []
        self.recent_test_session_by_id = {}
        self._recent_session_seq = 0
        self._recent_session_max_items = 20
        self._current_recent_session_id = None
        self._pending_recent_session_append = False
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

        # Startup statistics are daily; keep same-day counters and only roll over on a new date.
        self.reset_statistics_on_startup()

        # Restore persisted mode before UI callbacks are registered to avoid startup side effects.
        self._restore_last_sequence_mode()

        self.set_member_connect()
        self.bind_hw_signals()
        self.init_lineedit_text()
        self.init_ui()
        self._serial_trigger_runtime_initialized = False
        self.restore_scanner_checkbox_state()

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
from ui.sequence.sequence_widget_barcode_ops import SequenceWidgetBarcodeOpsMixin
from ui.sequence.sequence_widget_tcp_ops import SequenceWidgetTcpOpsMixin
from ui.sequence.sequence_widget_serial_trigger_ops import SequenceWidgetSerialTriggerOpsMixin
from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin
from ui.sequence.sequence_widget_config_ops import SequenceWidgetConfigOpsMixin
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin


class SequenceWindow(
    SequenceWidgetUiOpsMixin,
    SequenceWidgetBarcodeOpsMixin,
    SequenceWidgetTcpOpsMixin,
    SequenceWidgetSerialTriggerOpsMixin,
    SequenceWidgetAnalysisOpsMixin,
    SequenceWidgetConfigOpsMixin,
    SequenceWidgetStreamingOpsMixin,
    QWidget,
):
    tcp_server = None
    _active_instance_ref = None

    def __init__(self):
        """Initializes the class instance, setting up the user interface and necessary parameters."""
        super().__init__()
        self.data_struct = DataDealStruct()
        self.recorded_path = None
        self.count_board = None
        self.toolsbar = SequenceToolsBar()

        self.v2pa_factor = get_mic_v2pa_factor()
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
        self.left_panel = MotorDetectionLeftPanel(self.count_board)
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
        self.current_recorded_count = None
        self.last_play_count = None  # Cache last play count for replay

        self.default_logger = LogManager.set_log_handler("core")
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
        self._manual_direction_fallback_next_direction = "forward"
        self._ai_cycle_started_at = ""
        self._ai_cycle_direction_results = {"forward": None, "reverse": None}
        self._pending_serial_trigger_direction = ""
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

        # 条码提交去重机制：防止 HID 模式和键盘楔入模式同时触发导致重复提交
        self._last_committed_barcode = None
        self._last_committed_barcode_time = 0.0
        self._barcode_commit_dedup_window_sec = 0.8  # 800ms 去重窗口

        # HID 模式激活标志：当 HID 模式成功接收条码后，暂时忽略键盘楔入模式的输入
        self._hid_mode_active_until = 0.0  # 时间戳，在此之前忽略键盘输入

        # Streaming state variables
        self.streaming_buffer_multi = []  # list[np.ndarray] chunks, shape (frames, channels)
        self.streaming_wav_writer = None  # WAV file writer for incremental saving
        self.streaming_processor = None  # StreamingAudioProcessor instance
        self.streaming_stimulus_data = None  # Stimulus data for alignment (play+record mode)
        self.streaming_mode = None  # "play_record" or "record_only"
        self._active_input_channels = [0]
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
        # Create QTimer in Qt main thread for queue polling
        self.streaming_poll_timer = QTimer(self)
        self.streaming_poll_timer.timeout.connect(self._poll_streaming_queue)

        self.set_member_connect()
        self.bind_hw_signals()
        self.init_lineedit_text()
        self.init_ui()
        self._serial_trigger_runtime_initialized = False

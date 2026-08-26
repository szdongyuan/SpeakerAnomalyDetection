import os
import re
from uuid import uuid4
from weakref import WeakMethod, ref

import numpy as np
from PyQt5 import sip
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import (
    QObject,
    QEvent,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QSplitter,
)
from base.data_struct.data_deal_struct import DataDealStruct
from base.shortcut_trigger_manager import ShortcutTriggerManager
from base.unified_hid_device_manager import UnifiedHardwareManager
from base.load_config import LoadUiConfig
from ui.sequence.barcode_router import BarcodeRouter
from base.log_manager import LogManager
from base.streaming_file_writer import StreamingWavWriter
from base.streaming_audio_processor import StreamingAudioProcessor
from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.widgets import MessageBox
from ui.operation_sequence import AnalysisModelSelect
from ui.sequence.channel_plot_workspace import ChannelPlotWorkspace
from ui.sequence.sequence_configuration_controller import SequenceConfigurationController
from ui.sequence.sequence_configuration_analysis_flags import (
    DataStructAnalysisFlagProjectionPort,
    SequenceAnalysisFlagProjectionService,
)
from ui.sequence.sequence_configuration_model import SequenceConfigurationModel
from ui.sequence.sequence_configuration_view import SequenceConfigurationView
from ui.sequence.sequence_tools_bar import SequenceToolsBar
from ui.sequence.sequence_event_bus import SequenceEventBus
from ui.sequence.sequence_analysis_controller import (
    SequenceAnalysisController,
    SequenceAnalysisTransportController,
)
from ui.sequence.sequence_analysis_transport_service import (
    SequenceAnalysisTransportService,
)
from ui.sequence.sequence_analysis_model import SequenceAnalysisModel
from ui.sequence.sequence_analysis_view import SequenceAnalysisView
from ui.sequence.sequence_export_controller import SequenceExportController
from ui.sequence.sequence_export_model import (
    SequenceExportModel,
)
from ui.sequence.sequence_export_service import SequenceExportService
from ui.sequence.sequence_export_view import SequenceExportView
from ui.sequence.sequence_recording_controller import (
    BlockingRecordingAdapter,
    SequenceRecordingController,
)
from ui.sequence.sequence_recording_import_service import (
    SequenceImportedAudioService,
)
from ui.sequence.sequence_recording_model import (
    RecordingModel,
    RecordingStreamingStimulusPort,
)
from ui.sequence.sequence_recording_service import (
    RecordingAdmissionInputs,
    RecordingAdmissionService,
    RecordingCancellationService,
    RecordingCountBoardPersistence,
    RecordingLabelContext,
    RecordingLabelService,
    RecordingManualLabelRequestService,
    RecordingMarkActionService,
    RecordingPersistenceService,
    RecordingReadinessRuntimeCapabilities,
    RecordingStatisticsService,
    SequenceRecordingReadinessService,
)
from ui.sequence.sequence_recording_view import (
    RecordingCountProjection,
    SequenceRecordingAnalysisWindowsPort,
    SequenceRecordingImportView,
    SequenceRecordingLabelProjection,
    SequenceRecordingMarkActionProjection,
    SequenceRecordingStatisticsProjection,
    SequenceRecordingView,
)
from ui.sequence.sequence_recording_worker import SequenceStreamingRecordingService
from ui.sequence.sequence_resource_lifecycle_controller import (
    _CANONICAL_TCP_MIRROR_STATE,
    SequenceResourceLifecycleController,
    SequenceResourceLifecycleModel,
    SequenceResourceLifecycleView,
)
from ui.sequence.sequence_messages import (
    ImportAudioRequested,
    ShutdownReady,
)
from ui.sequence.sequence_trigger_controller import SequenceTriggerController
from ui.sequence.sequence_trigger_resource_lifecycle_port import (
    SequenceTriggerResourceLifecyclePort,
)
from ui.sequence.sequence_trigger_model import SequenceTriggerModel
from ui.sequence.sequence_trigger_view import SequenceTriggerView
from ui.sequence.sequence_workflow_controller import (
    SequenceShutdownCoordinator,
    SequenceWorkflowController,
)
from ui.sequence.sequence_workflow_model import SequenceWorkflowModel
from ui.sequence.sequence_workflow_policy import (
    SequenceAutomaticAnalysisPolicyService,
)
from ui.sequence.sequence_workflow_view import SequenceWorkflowView
from ui.sequence.sequencement_count_board import SequenceCountBoard


_RECORDING_BUTTON_STATE_NOT_CAPTURED = object()


class _SequenceWindowMeta(type(QWidget)):
    def __getattribute__(cls, attribute):
        if attribute == "tcp_server":
            return _CANONICAL_TCP_MIRROR_STATE.read()
        return super().__getattribute__(attribute)

    def __setattr__(cls, attribute, value):
        if attribute == "tcp_server":
            _CANONICAL_TCP_MIRROR_STATE.write(value)
            return
        super().__setattr__(attribute, value)

    def __delattr__(cls, attribute):
        if attribute == "tcp_server":
            _CANONICAL_TCP_MIRROR_STATE.write(None)
            return
        super().__delattr__(attribute)


class SequenceWindow(QWidget, metaclass=_SequenceWindowMeta):
    shutdown_ready = pyqtSignal(object)
    shutdown_aborted = pyqtSignal(object)

    @property
    def shortcut_mgr(self):
        return self.resource_lifecycle_controller.shortcut_mgr

    @shortcut_mgr.setter
    def shortcut_mgr(self, manager) -> None:
        self.resource_lifecycle_controller.shortcut_mgr = manager

    @property
    def hw_manager(self):
        return self.resource_lifecycle_controller.hw_manager

    @hw_manager.setter
    def hw_manager(self, manager) -> None:
        self.resource_lifecycle_controller.hw_manager = manager

    @property
    def trigger_controller(self):
        return getattr(self, "_trigger_domain_controller", None)

    @trigger_controller.setter
    def trigger_controller(self, controller) -> None:
        if getattr(self, "_trigger_domain_controller", None) is controller:
            return
        self._trigger_domain_controller = controller
        self.resource_lifecycle_controller.tcp_resource_port = (
            None
            if controller is None
            else SequenceTriggerResourceLifecyclePort(controller)
        )

    @property
    def recorded_path(self):
        return self.recording_model.recorded_path

    @recorded_path.setter
    def recorded_path(self, value) -> None:
        self.recording_model.recorded_path = value

    @property
    def recorded_signal_info(self):
        return self.recording_model.recorded_signal_info

    @recorded_signal_info.setter
    def recorded_signal_info(self, value) -> None:
        self.recording_model.recorded_signal_info = value

    @property
    def current_recorded_count(self):
        return self.recording_model.current_recorded_count

    @current_recorded_count.setter
    def current_recorded_count(self, value) -> None:
        self.recording_model.current_recorded_count = value

    @property
    def last_play_count(self):
        return self.recording_model.last_play_count

    @last_play_count.setter
    def last_play_count(self, value) -> None:
        self.recording_model.last_play_count = value

    def __init__(self):
        """Initializes the class instance, setting up the user interface and necessary parameters."""
        super().__init__()
        self.recording_model = RecordingModel()
        self.recording_streaming_stimulus_port = RecordingStreamingStimulusPort(
            self.recording_model
        )
        self._initialize_recording_view_state()
        self.workflow_model = SequenceWorkflowModel()
        self.workflow_automatic_analysis_policy = (
            SequenceAutomaticAnalysisPolicyService()
        )
        self.workflow_view = SequenceWorkflowView(
            self.workflow_model,
            refresh_player_button=self.update_player_btn_is_paused,
            synchronize_shutdown=self._synchronize_workflow_shutdown,
            parent=self,
        )
        self.configuration_model = SequenceConfigurationModel(
            data_struct=DataDealStruct(),
            workflow_model=self.workflow_model,
            streaming_stimulus_port=self.recording_streaming_stimulus_port,
        )
        self.analysis_flag_projection_service = (
            SequenceAnalysisFlagProjectionService(
                DataStructAnalysisFlagProjectionPort(
                    self.configuration_model.data_struct
                )
            )
        )
        self.sequence_event_bus = SequenceEventBus(self)
        self.workflow_bus = self.sequence_event_bus
        self.resource_lifecycle_model = SequenceResourceLifecycleModel()
        self.resource_lifecycle_view = SequenceResourceLifecycleView(self)
        self.resource_lifecycle_controller = (
            SequenceResourceLifecycleController(
                self.resource_lifecycle_view,
                self.resource_lifecycle_model,
                lifecycle_bus=self.sequence_event_bus,
                parent=self,
            )
        )
        SequenceWindow.tcp_server = None
        self.recorded_path = None
        self.count_board = None
        self.toolsbar = SequenceToolsBar()
        self.default_logger = LogManager.set_log_handler("core")
        self.configuration_view = SequenceConfigurationView(
            parent=self,
            combobox=self.using_file_combobox,
            player_button=self.player_btn,
            replay_button=self.replayer_btn,
            data_button=self.data_btn,
            serial_input=self.lineedit_s_or_n,
        )
        self.configuration_controller = SequenceConfigurationController(
            self.configuration_model,
            self.configuration_view,
            configuration_publisher=self.sequence_event_bus.events.configuration_changed.emit,
            availability_changed=self.update_player_btn_is_paused,
            refresh_channels=self.refresh_channel_windows,
            clear_plot=self._clear_plot_area,
            clear_import_identity=self._clear_import_recording_identity,
            plot_state_capturer=self._capture_plot_projection_state,
            plot_state_restorer=self._restore_plot_projection_state,
            import_identity_state_capturer=(
                self._capture_import_recording_identity
            ),
            import_identity_state_restorer=(
                self._restore_import_recording_identity
            ),
            analysis_flag_projection_service=(
                self.analysis_flag_projection_service
            ),
            analysis_config_changed=self._set_count_board_analysis_config,
            refresh_test_mode_availability=self._refresh_test_mode_availability,
            logger=self.default_logger,
            parent=self,
        )
        self.analysis_model = SequenceAnalysisModel()
        self.analysis_view = SequenceAnalysisView(
            self.analysis_model,
            parent=self,
            logger=self.default_logger,
            geometry_path=os.path.join(
                DEFAULT_DIR, "ui", "ui_config", "analysis_window_geometry.json"
            ),
            warning_presenter=(
                lambda title, text: MessageBox.warning(self, title, text)
            ),
        )
        self.recording_import_view = SequenceRecordingImportView(
            parent=self,
            warning_presenter=(
                lambda title, text: MessageBox.warning(self, title, text)
            ),
            clear_import_plot=self._clear_plot_area,
            plot_imported_audio=self.plot_waveform_to_workspace,
            import_data_enabled_setter=self.data_btn.setEnabled,
            import_projection_capturer=(
                self._capture_import_recording_projection
            ),
            import_projection_restorer=(
                self._restore_import_recording_projection
            ),
            import_plot_projection_restorer=(
                self._restore_plot_projection_state
            ),
        )
        self.recording_import_service = SequenceImportedAudioService(
            logger=self.default_logger,
            reference_logger=LogManager.set_log_handler("core"),
        )
        self.analysis_transport_service = SequenceAnalysisTransportService(
            tcp_enabled_provider=lambda: bool(getattr(self, "tcp_flag", False)),
            tcp_server_provider=self._get_tcp_mirror_identity,
            logger=self.default_logger,
        )
        self.analysis_controller = SequenceAnalysisController(
            self.analysis_model,
            self.analysis_view,
            bus=self.sequence_event_bus,
            runtime=self,
            workflow_identity_provider=self._analysis_workflow_identity,
            transport_service=self.analysis_transport_service,
            logger=self.default_logger,
            parent=self,
        )
        self.recording_admission_service = RecordingAdmissionService(
            raw_inputs=self._recording_admission_inputs,
        )
        self.recording_persistence_service = RecordingPersistenceService()
        self.export_model = SequenceExportModel()
        self.export_service = SequenceExportService(
            logger=self.default_logger,
        )
        self.export_view = SequenceExportView(
            parent=self,
            logger=self.default_logger,
        )
        self.export_controller = SequenceExportController(
            self.export_model,
            self.export_view,
            bus=self.sequence_event_bus,
            service=self.export_service,
            logger=self.default_logger,
            parent=self,
        )
        self.recording_count_projection = RecordingCountProjection(
            self.recording_model,
            self.lineedit_count,
        )
        self.recording_label_projection = SequenceRecordingLabelProjection(self)
        self.recording_statistics_projection = (
            SequenceRecordingStatisticsProjection(self)
        )
        self.recording_mark_action_projection = (
            SequenceRecordingMarkActionProjection(
                self,
                clear_plot=self._clear_plot_area,
                analysis_windows_port=SequenceRecordingAnalysisWindowsPort(
                    self.analysis_view
                ),
                capture_plot=self._capture_plot_projection_state,
                restore_plot=self._restore_plot_projection_state,
            )
        )
        self.recording_mark_action_service = RecordingMarkActionService(
            self.recording_model,
            self.recording_mark_action_projection,
            workflow_generation_provider=(
                lambda: self.workflow_model.workflow_generation
            ),
            logger=self.default_logger,
        )
        self.recording_count_board_persistence = RecordingCountBoardPersistence(
            statistics_model=self.recording_model,
        )
        self.recording_statistics_service = RecordingStatisticsService(
            self.recording_model,
            self.recording_count_board_persistence,
            self.recording_statistics_projection,
            logger=self.default_logger,
        )
        self.recording_view = SequenceRecordingView(
            set_recording_locked=self._set_sn_input_recording_read_only,
            set_started=self._recording_view_started,
            set_finished=self._recording_view_finished,
            present_error=self._present_recording_error,
            present_readiness_warning=(
                lambda title, text: MessageBox.warning(self, title, text)
            ),
            plot_recording=self.plot_waveform_to_workspace,
            commit_identity=self.recording_model.commit_identity,
            commit_label_projection=self.recording_label_projection,
            logger=self.default_logger,
            parent=self,
        )
        self.recording_manual_label_request_service = (
            RecordingManualLabelRequestService(
                data_provider=lambda: getattr(
                    self.data_struct, "store_wave_data", None
                ),
                sequence_config_provider=lambda: self.sequence_config,
                retained_record_id_provider=(
                    lambda: self.workflow_model.retained_record_id
                ),
                recorded_signal_info_provider=(
                    lambda: self.recording_model.recorded_signal_info
                ),
                recorded_path_provider=(
                    lambda: self.recording_model.recorded_path
                ),
                ok_button=lambda: self.count_board.ok_btn,
                ng_button=lambda: self.count_board.ng_btn,
                publish=(
                    self.sequence_event_bus.commands.manual_label_requested.emit
                ),
                present_warning=(
                    lambda title, text: MessageBox.warning(self, title, text)
                ),
            )
        )
        self.recording_readiness_service = SequenceRecordingReadinessService(
            runtime_capabilities_provider=lambda: RecordingReadinessRuntimeCapabilities(
                audio_devices_available=(
                    self.configuration_model.audio_devices_available
                ),
                audio_devices_unavailable_message=(
                    self.configuration_model.audio_devices_unavailable_message
                ),
            ),
            view=self.recording_view,
            logger=self.default_logger,
        )
        self.workflow_controller = SequenceWorkflowController(
            self.workflow_model,
            self.sequence_event_bus,
            configuration_snapshot_provider=self._workflow_configuration_snapshot,
            start_readiness=self.recording_readiness_service,
            replay_readiness=self.recording_admission_service.replay_readiness,
            session_snapshot_factory=self.recording_admission_service.session_snapshot,
            recording_snapshot_lookup=self._workflow_analysis_snapshot_lookup,
            retain_recording_snapshot=(
                self.recording_model.retain_recording_snapshot
            ),
            clear_retained_recording_snapshot=(
                self.recording_model.clear_retained_recording_snapshot
            ),
            automatic_analysis_policy=self.workflow_automatic_analysis_policy,
            export_decision_requires_terminal=True,
            parent=self,
        )
        self.analysis_transport_controller = SequenceAnalysisTransportController(
            bus=self.sequence_event_bus,
            authorization_claimer=(
                self.workflow_model.claim_analysis_transport
            ),
            claim_releaser=self.workflow_model.release_analysis_transport_claim,
            claim_committer=self.workflow_model.commit_analysis_transport_claim,
            claim_abandoner=self.workflow_model.abandon_analysis_transport_claim,
            service=self.analysis_transport_service,
            logger=self.default_logger,
            parent=self,
        )
        self.recording_label_service = RecordingLabelService(
            context_provider=self._recording_label_context,
            count_board_persistence=self.recording_count_board_persistence,
        )
        self.blocking_recording_adapter = BlockingRecordingAdapter(
            data_struct=self.data_struct,
            save_database=self.recording_persistence_service.save_recording_database,
            commit_count=self.recording_count_projection,
            persist_count=self.recording_persistence_service.persist_count,
            logger=self.default_logger,
        )
        self.streaming_recording_service = SequenceStreamingRecordingService(
            view=self.recording_view,
            processor_factory=StreamingAudioProcessor,
            writer_factory=StreamingWavWriter,
            logger=self.default_logger,
            parent=self,
        )
        self.recording_cancellation_service = RecordingCancellationService(
            self.blocking_recording_adapter
        )
        self.recording_controller = SequenceRecordingController(
            self.recording_model,
            self.sequence_event_bus,
            view=self.recording_view,
            label_service=self.recording_label_service,
            mark_action_service=self.recording_mark_action_service,
            manual_label_request_service=(
                self.recording_manual_label_request_service
            ),
            import_view=self.recording_import_view,
            import_runtime=self,
            import_service=self.recording_import_service,
            import_workflow_identity_provider=self._workflow_import_identity,
            prepare_session=self.blocking_recording_adapter.prepare,
            blocking_acquirer=self.blocking_recording_adapter.acquire,
            transaction_factory=self.blocking_recording_adapter.transaction,
            use_streaming=lambda prepared: bool(
                prepared.acquisition_context.get("use_streaming", False)
            ),
            streaming_adapter=self.streaming_recording_service.start,
            request_blocking_cancel=self.blocking_recording_adapter.request_cancel,
            close_streaming_admission=self.streaming_recording_service.close_admission,
            quiesce_streaming=self.streaming_recording_service.quiesce,
            cancel_adapter=self.recording_cancellation_service.cancel,
            workflow_generation_provider=lambda: self.workflow_model.workflow_generation,
            logger=self.default_logger,
            parent=self,
        )
        self.sequence_event_bus.events.configuration_changed.connect(
            self.configuration_controller.handle_configuration_changed,
            Qt.QueuedConnection,
        )
        self.workflow_view.connect_state_changed(
            self.sequence_event_bus.events.workflow_state_changed,
            Qt.QueuedConnection,
        )
        self.sequence_event_bus.events.workflow_command_rejected.connect(
            self.recording_admission_service.discard_rejected,
        )
        self._wire_workflow_continuation_ports()
        self._wire_analysis_workflow_channels()
        self.sequence_event_bus.commands.load_imported_audio_requested.connect(
            self.recording_controller.handle_load_imported_audio_requested,
            Qt.QueuedConnection,
        )
        self.sequence_event_bus.commands.cancel_imported_audio_requested.connect(
            self.recording_controller.handle_cancel_imported_audio_requested,
            Qt.QueuedConnection,
        )

        self.v2pa_factor = None
        self.get_sequence_config_from_registry()
        self.sequence_config = list()
        self.analysis_config = dict()
        self.get_sequence_config_from_json()
        self.init_data_struct_stimulus_config()
        self.signal_info = {}
        self.analysis_window = []
        self._analysis_result_summary_window = None

        self.count_board = SequenceCountBoard(self.analysis_config)
        self.init_result_files()
        self._refresh_test_mode_availability()
        self._barcode_scanner_box_enabled_before_recording = None
        self.audio_devices_available = True
        self.audio_devices_unavailable_message = ""
        self.recorded_signal_info = {}
        tcp_ip, tcp_port = LoadUiConfig.get_tcp_config()
        self.current_recorded_count = None
        self.last_play_count = None  # Cache last play count for replay
        self.hw_manager = UnifiedHardwareManager()
        self.shortcut_mgr = ShortcutTriggerManager(self)
        self.trigger_model = SequenceTriggerModel(tcp_host=tcp_ip, tcp_port=tcp_port)
        self.trigger_view = SequenceTriggerView(
            parent=self,
            serial_input=self.lineedit_s_or_n,
            scanner_checkbox=self.barcode_scanner_box,
            product_input=self.lineedit_type,
            count_input=self.lineedit_count,
            prepare_for_continuous_scan=self._close_analysis_windows,
        )
        self.trigger_controller = SequenceTriggerController(
            self.trigger_model,
            self.trigger_view,
            start_publisher=self.sequence_event_bus.commands.start_test_requested.emit,
            barcode_publisher=self.sequence_event_bus.commands.barcode_committed.emit,
            event_bus=self.sequence_event_bus,
            configuration_generation_provider=(
                lambda: self.configuration_model.configuration_generation
            ),
            workflow_active_provider=self.workflow_model.is_workflow_active,
            external_mode_available_provider=(
                self.configuration_controller.is_mode_available_for_external_trigger
            ),
            acquisition_mode_provider=lambda: self.mode,
            logger=self.default_logger,
            hardware_manager=self.hw_manager,
            shortcut_manager=self.shortcut_mgr,
            tcp_config_writer=(
                lambda host, port: LoadUiConfig.write_tcp_config(
                    host, port, self.default_logger
                )
            ),
            tcp_mirror_getter=self._get_tcp_mirror_identity,
            tcp_mirror_setter=self._set_tcp_mirror_identity,
            parent=self,
        )
        self._barcode_router = BarcodeRouter(self.trigger_controller, self)
        self._initialize_shutdown_ready_release_port()
        self.shutdown_coordinator = SequenceShutdownCoordinator(
            self.workflow_model,
            self.sequence_event_bus,
            view=self.export_view,
            cleanup_resources=(
                self.resource_lifecycle_controller.complete_application_shutdown_before_ready
            ),
            shutdown_ready=self.workflow_controller.handle_shutdown_ready,
            finalize_after_ready_ack=(
                self.resource_lifecycle_controller.complete_application_shutdown_after_ready_ack
            ),
            release_shutdown_close=self._release_staged_shutdown_close,
            shutdown_aborted=self.shutdown_aborted.emit,
            logger=self.default_logger,
            parent=self,
        )
        self._register_resource_lifecycle_recipients()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self._sn_clear_on_next_scan = False

        self.streaming_stimulus_data = None  # Stimulus data for alignment (play+record mode)
        self._active_input_channels = [0]
        self.channel_workspace = None

        self.trigger_controller.bind_shortcut_signal()
        self.shortcut_mgr.start()

        self.set_member_connect()
        self.bind_hw_signals()
        self.init_lineedit_text()
        self.init_ui()

    def _register_resource_lifecycle_recipients(self) -> None:
        register = self.sequence_event_bus.register_resource_lifecycle_recipient
        for name, controller in (
            ("trigger", self.trigger_controller),
            ("analysis-transport", self.analysis_transport_controller),
            ("analysis", self.analysis_controller),
            ("workflow", self.workflow_controller),
            ("recording", self.recording_controller),
            ("export", self.export_controller),
        ):
            token = register(
                "disconnect-domains",
                name,
                controller.disconnect,
                owner=controller,
            )
            if not self.resource_lifecycle_model.retain_resource_lifecycle_registration(
                "disconnect-domains", name, token, controller
            ):
                raise RuntimeError(
                    f"resource lifecycle registration was not retained: {name}"
                )
    def showEvent(self, event):
        """
        MainWindow shows SequenceWindow only after login success.
        Defer the missing-config prompt until the first showEvent.
        """
        self.resource_lifecycle_controller.resume_child_resources()
        super().showEvent(event)
        self.configuration_view.present_missing_configuration_prompt(
            self.sequence_config,
            eligible=True,
        )

    def init_ui(self):
        """
        Initializes the user interface of the SequenceWindow.

        This method sets up the window icon, minimum height, and creates the main layout
        by adding toolbar and waveform layouts. It also connects button click events to
        their respective handlers and applies style sheets to the widgets.
        """
        self.setObjectName("SequenceWindow")
        self.setMinimumHeight(600)
        waveform_layout = self.create_waveform_layout()

        sequence_layout = QVBoxLayout()
        sequence_layout.addWidget(self.toolsbar)
        sequence_layout.addLayout(waveform_layout, stretch=1)
        sequence_layout.setAlignment(Qt.AlignTop)
        sequence_layout.setSpacing(0)
        sequence_layout.setContentsMargins(1, 0, 1, 0)

        self.add_file_to_using_file_combobox()

        self.setLayout(sequence_layout)

        # When test-queue config is confirmed, refresh combobox + reload config first.
        # (No global signal dependency; MainWindow calls on_sequence_config_updated after dialog closes.)
    def _tcp_run_test(self, label: str = "not_labeled", skip_sn_regex_validation: bool = False):
        self.trigger_controller.handle_tcp_run_test(
            label, skip_sn_regex_validation=skip_sn_regex_validation
        )
        return None

    def _get_mode_display_name(self, mode: str) -> str:
        return self.trigger_view.mode_display_name(mode)

    def _show_external_trigger_mode_warning(self, trigger_source: str, current_mode: str) -> None:
        self.trigger_view.show_mode_rejection(trigger_source, current_mode)
        return None

    def _clear_sn_for_external_trigger_rejection(self, trigger_source: str) -> None:
        self.trigger_controller.clear_serial_for_external_rejection(trigger_source)
        return None

    def _ensure_external_trigger_mode_supported(self, trigger_source: str) -> bool:
        return self.trigger_controller.ensure_external_mode_supported(trigger_source)

    def _has_imported_recording_runtime_state(self):
        return self.configuration_controller.has_imported_recording_runtime_state()

    def _has_import_stimulus_runtime_reference(self):
        return self.configuration_controller.has_import_stimulus_runtime_reference()

    def _validate_import_stimulus_analysis_readiness(self):
        return self.configuration_controller.validate_import_stimulus_analysis_readiness()

    def _refresh_import_stimulus_analysis_reference(self, acq_detail):
        return self.configuration_controller.refresh_import_stimulus_analysis_reference(
            acq_detail
        )

    def on_sequence_config_updated(self, *_):
        self.configuration_controller.on_sequence_config_updated(*_)
        return None

    def _set_count_board_analysis_config(self, analysis_config):
        if self.count_board is not None:
            self.count_board.analysis_config = analysis_config

    def _clear_import_recording_identity(self):
        self.recorded_path = None
        self.recorded_signal_info = None

    def _capture_import_recording_identity(self):
        return (
            getattr(self, "recorded_path", None),
            getattr(self, "recorded_signal_info", None),
        )

    def _restore_import_recording_identity(self, state):
        self.recorded_path = state[0]
        self.recorded_signal_info = state[1]

    def _capture_plot_projection_state(self):
        recorded_signal = getattr(self.data_struct, "store_wave_data_multi", None)
        if recorded_signal is None:
            recorded_signal = getattr(self.data_struct, "store_wave_data", None)
        return (
            recorded_signal,
            getattr(self.data_struct, "sample_rate", None),
        )

    def _restore_plot_projection_state(self, state):
        recorded_signal, sample_rate = state
        self._clear_plot_area()
        if recorded_signal is not None:
            self.plot_waveform_to_workspace(recorded_signal, sample_rate)

    def _capture_import_recording_projection(self):
        return (
            self._capture_plot_projection_state(),
            self.data_btn.isEnabled(),
        )

    def _restore_import_recording_projection(self, state):
        plot_state, data_enabled = state
        self._restore_plot_projection_state(plot_state)
        self.data_btn.setEnabled(data_enabled)

    def _refresh_test_mode_availability(self):
        """
        Enable/disable test mode based on whether current config can output OK/NG.
        """
        can_output, reason = self.configuration_controller.can_output_ok_ng()
        if self.count_board:
            self.count_board.set_test_available(bool(can_output), reason or "")

    def update_v2pa_factor(self):
        return self.analysis_controller.update_v2pa_factor()

    def _summarize_ok_ng(self):
        return self.analysis_controller.summarize_ok_ng(
            getattr(self.data_struct, "analysis_result_dict", None)
        )

    def set_member_connect(self):
        self.player_btn.clicked.connect(lambda: self.on_clicked_player_btn())
        self.replayer_btn.clicked.connect(self.on_clicked_replayer_btn)
        self.data_btn.clicked.connect(self.analysis_controller.request_manual_analysis)
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
        self.count_board.reset_btn.clicked.connect(
            self.recording_statistics_service.reset_statistics
        )
        self.count_board.bind_mark_action(self.on_mark_btn_clicked)
        self.using_file_combobox.currentTextChanged.connect(self.on_using_file_combobox_changed)

    def on_mark_btn_clicked(self):
        return self.recording_controller.request_mark_action()

    def init_lineedit_text(self):
        last_recorded_info = LoadUiConfig().load_last_recorded_info(self.default_logger)
        if last_recorded_info:
            product_model = last_recorded_info.get("product_model", "S004-1")
        else:
            product_model = "S004-1"
        self.lineedit_type.setText(product_model)
        # 型号/计数：默认只读（单击进入编辑态），避免扫码枪后缀(Tab/Enter)导致焦点跳转/误触发
        self.lineedit_type.setReadOnly(True)

        result, _ = LoadUiConfig.load_recorded_num_from_json(self.default_logger)
        if result is None:
            self.current_recorded_count = 1
        else:
            self.current_recorded_count = result

        self.lineedit_count.setText(str(self.current_recorded_count))
        self.lineedit_count.setReadOnly(True)

    @property
    def player_btn(self):
        return self.toolsbar.player_btn

    @property
    def data_struct(self):
        return self.configuration_model.data_struct

    @property
    def mic(self):
        return self.configuration_model.mic

    @mic.setter
    def mic(self, value):
        self.configuration_model.mic = value

    @property
    def speaker(self):
        return self.configuration_model.speaker

    @speaker.setter
    def speaker(self, value):
        self.configuration_model.speaker = value

    @property
    def mic_channels(self):
        return self.configuration_model.mic_channels

    @mic_channels.setter
    def mic_channels(self, value):
        self.configuration_model.mic_channels = value

    @property
    def sequence_config(self):
        return self.configuration_model.sequence_config

    @sequence_config.setter
    def sequence_config(self, value):
        self.configuration_model.sequence_config = value

    @property
    def analysis_config(self):
        return self.configuration_model.analysis_config

    @analysis_config.setter
    def analysis_config(self, value):
        self.configuration_model.analysis_config = value

    @property
    def analysis_window(self):
        return self.analysis_model.analysis_instances

    @analysis_window.setter
    def analysis_window(self, value):
        self.analysis_model.analysis_instances = list(value or [])

    @property
    def _analysis_result_summary_window(self):
        return self.analysis_view.summary_window

    @_analysis_result_summary_window.setter
    def _analysis_result_summary_window(self, value):
        self.analysis_view.summary_window = value

    @property
    def using_config_path(self):
        return self.configuration_model.using_config_path

    @using_config_path.setter
    def using_config_path(self, value):
        self.configuration_model.using_config_path = value

    @property
    def registry(self):
        return self.configuration_model.registry

    @registry.setter
    def registry(self, value):
        self.configuration_model.replace_registry(
            value,
            using_config_path=self.configuration_model.using_config_path,
            entries=self.configuration_model.registry_entries,
        )

    @property
    def mode(self):
        return self.configuration_model.acquisition_mode

    @mode.setter
    def mode(self, value):
        self.configuration_model.acquisition_mode = value

    @property
    def tcp_flag(self):
        return bool(self.trigger_model.tcp_enabled)

    @tcp_flag.setter
    def tcp_flag(self, value):
        self.trigger_model.tcp_enabled = bool(value)

    @property
    def tcp_ip(self):
        return self.trigger_model.tcp_host

    @tcp_ip.setter
    def tcp_ip(self, value):
        self.trigger_model.tcp_host = value

    @property
    def tcp_port(self):
        return self.trigger_model.tcp_port

    @tcp_port.setter
    def tcp_port(self, value):
        self.trigger_model.tcp_port = value

    @property
    def audio_devices_available(self):
        return self.configuration_model.audio_devices_available

    @audio_devices_available.setter
    def audio_devices_available(self, value):
        self.configuration_model.audio_devices_available = bool(value)

    @property
    def audio_devices_unavailable_message(self):
        return self.configuration_model.audio_devices_unavailable_message

    @audio_devices_unavailable_message.setter
    def audio_devices_unavailable_message(self, value):
        self.configuration_model.audio_devices_unavailable_message = value or ""

    @property
    def streaming_stimulus_data(self):
        return self.recording_model.streaming_stimulus_data

    @streaming_stimulus_data.setter
    def streaming_stimulus_data(self, value):
        self.recording_model.streaming_stimulus_data = value

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
        return self.configuration_controller.init_data_struct_stimulus_config()

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

        self.waveform_splitter = QSplitter(Qt.Horizontal)
        self.waveform_splitter.setObjectName("SequenceWaveformSplitter")
        self.waveform_splitter.setChildrenCollapsible(False)
        splitter_side_padding = ui_style_const.scale_size_px(3)
        self.waveform_splitter.setHandleWidth(1)
        self.waveform_splitter.setStyleSheet(
            f"""
            QSplitter#SequenceWaveformSplitter::handle {{
                background-color: #b8b8b8;
                border: 0;
                margin-left: {splitter_side_padding}px;
                margin-right: {splitter_side_padding}px;
            }}
            """
        )
        self.waveform_splitter.addWidget(self.count_board)
        self.waveform_splitter.addWidget(self.channel_workspace)
        self.waveform_splitter.setStretchFactor(0, 1)
        self.waveform_splitter.setStretchFactor(1, 8)

        self._init_count_board_splitter_state()

        layout.addWidget(self.waveform_splitter)
        layout.setContentsMargins(40, 20, 40, 20)
        return layout

    def _init_count_board_splitter_state(self) -> None:
        collapsed_hint = self.count_board.collapsed_width_hint()
        self._count_board_collapsed_width = max(collapsed_hint, ui_style_const.scale_size_px(36))
        self._count_board_collapse_threshold = self._count_board_collapsed_width + ui_style_const.scale_size_px(16)
        self._count_board_expand_threshold = self._count_board_collapse_threshold + ui_style_const.scale_size_px(24)
        self._last_count_board_expanded_width = self.count_board.expanded_width_hint()
        self._count_board_max_width = self._last_count_board_expanded_width + ui_style_const.scale_size_px(120)
        self._applying_count_board_splitter_state = False
        self._count_board_splitter_reconcile_pending = False

        self.count_board.set_compact_resize_enabled(True)
        self.count_board.setMaximumWidth(self._count_board_max_width)
        self.waveform_splitter.installEventFilter(self)
        self.count_board.installEventFilter(self)
        self.count_board.collapsed_changed.connect(self._on_count_board_collapsed_changed)
        self.waveform_splitter.splitterMoved.connect(self._on_waveform_splitter_moved)
        QTimer.singleShot(0, self._apply_initial_waveform_splitter_sizes)

    def _apply_initial_waveform_splitter_sizes(self) -> None:
        if not hasattr(self, "waveform_splitter"):
            return
        available_width = self._count_board_splitter_available_width()
        if available_width <= 0:
            return

        if not self._can_expand_count_board(available_width):
            was_collapsed = self.count_board.is_collapsed()
            self.count_board.set_collapsed(True)
            if was_collapsed:
                self._apply_count_board_splitter_sizes(True)
            return

        expanded_width = self._bounded_count_board_width(self.count_board.expanded_width_hint())
        left_width = min(expanded_width, max(0, available_width - 1))
        self.waveform_splitter.setSizes([left_width, max(1, available_width - left_width)])

    def _schedule_count_board_splitter_reconcile(self) -> None:
        if self._applying_count_board_splitter_state:
            return
        if getattr(self, "_count_board_splitter_reconcile_pending", False):
            return
        self._count_board_splitter_reconcile_pending = True
        QTimer.singleShot(0, self._reconcile_count_board_splitter_state)

    def _reconcile_count_board_splitter_state(self) -> None:
        self._count_board_splitter_reconcile_pending = False
        if self._applying_count_board_splitter_state:
            return
        if not hasattr(self, "waveform_splitter"):
            return

        sizes = self.waveform_splitter.sizes()
        left_width = sizes[0] if sizes else 0
        available_width = self._count_board_splitter_available_width()
        minimum_expanded_width = self.count_board.expanded_width_hint()

        if not self.count_board.is_collapsed() and (
            left_width < minimum_expanded_width or not self._can_expand_count_board(available_width)
        ):
            self.count_board.set_collapsed(True)

    def _on_count_board_collapsed_changed(self, collapsed: bool) -> None:
        if self._applying_count_board_splitter_state:
            return
        self._apply_count_board_splitter_sizes(bool(collapsed))

    def _on_waveform_splitter_moved(self, pos: int, index: int) -> None:
        if self._applying_count_board_splitter_state:
            return

        sizes = self.waveform_splitter.sizes()
        left_width = sizes[0] if sizes else int(pos)
        minimum_expanded_width = self.count_board.expanded_width_hint()

        if left_width < minimum_expanded_width:
            was_collapsed = self.count_board.is_collapsed()
            self.count_board.set_collapsed(True)
            if was_collapsed:
                self._apply_count_board_splitter_sizes(True)
            return

        self._last_count_board_expanded_width = self._bounded_count_board_width(left_width)
        self.count_board.set_collapsed(False)
        if left_width > self._count_board_max_width:
            self._apply_count_board_splitter_sizes(False)

    def _apply_count_board_splitter_sizes(self, collapsed: bool) -> None:
        total_width = self._count_board_splitter_available_width()
        if total_width <= 0:
            total_width = self.count_board.expanded_width_hint() * 4

        if collapsed:
            left_width = self._count_board_collapsed_width
        else:
            if not self._can_expand_count_board(total_width):
                self._applying_count_board_splitter_state = True
                try:
                    self.count_board.set_collapsed(True)
                    left_width = min(self._count_board_collapsed_width, max(0, total_width - 1))
                    self.waveform_splitter.setSizes([left_width, max(1, total_width - left_width)])
                finally:
                    self._applying_count_board_splitter_state = False
                return

            left_width = max(self._last_count_board_expanded_width, self.count_board.expanded_width_hint())
            left_width = self._bounded_count_board_width(left_width)
            left_width = min(left_width, max(self._count_board_expand_threshold + 1, total_width - 1))

        right_width = max(1, total_width - left_width)
        self._applying_count_board_splitter_state = True
        try:
            self.waveform_splitter.setSizes([left_width, right_width])
        finally:
            self._applying_count_board_splitter_state = False

    def _bounded_count_board_width(self, width: int) -> int:
        return min(max(int(width), self._count_board_collapsed_width), self._count_board_max_width)

    def _count_board_splitter_available_width(self) -> int:
        sizes = self.waveform_splitter.sizes()
        total_width = sum(sizes) if sizes else 0
        if total_width > 0:
            return total_width
        return max(0, self.waveform_splitter.width() - self.waveform_splitter.handleWidth())

    def _can_expand_count_board(self, available_width=None) -> bool:
        if available_width is None:
            available_width = self._count_board_splitter_available_width()
        if available_width <= 0:
            return True
        return available_width >= self.count_board.expanded_width_hint() + 1

    def init_result_files(self):
        return self.recording_statistics_service.initialize_statistics()

    def bind_hw_signals(self):
        """绑定 hardware manager 的信号, 避免重复连接"""
        self.trigger_controller.bind_hardware_signals()
        return None

    def clicked_scanner(self):
        """Checkbox 状态改变时的回调"""
        self.trigger_controller.set_scanner_checked(
            self.barcode_scanner_box.isChecked()
        )
        return None

    def _normalize_barcode(self, text: str) -> str:
        return self.trigger_controller.normalize_barcode(text)

    # Windows 文件名中不允许的特殊字符
    _INVALID_FILENAME_CHARS = set('\\/:*?"<>|')

    def _barcode_has_invalid_chars(self, barcode: str) -> tuple:
        """检查条形码是否包含无法用于文件名的特殊字符，返回 (是否有, 特殊字符列表)"""
        found = self.trigger_controller.barcode_invalid_characters(barcode)
        return (bool(found), list(found))

    def _reset_barcode_commit_state(self, clear_dedup: bool = False):
        self.trigger_controller.reset_barcode_state(clear_dedup=clear_dedup)
        return None

    def _load_selected_sn_regex_rule(self):
        return self.trigger_controller.load_selected_sn_regex_rule()

    def _validate_sn_regex_before_start(
        self,
        sn_text=None,
        value_label="实际 SN 内容",
        retry_hint=None,
        skip_sn_regex_validation: bool = False,
    ):
        if retry_hint is None:
            retry_hint = "请检查当前 SN 内容或切换正确规则后重试。"
        return self.trigger_controller.validate_sn_regex(
            sn_text,
            value_label=value_label,
            retry_hint=retry_hint,
            skip_sn_regex_validation=skip_sn_regex_validation,
        )

    def open_sn_regex_manage_dialog(self):
        if self.trigger_controller.is_active:
            self.trigger_view.open_regex_dialog()
        return None

    def _commit_barcode(self, barcode: str, source: str = "wedge"):
        self.trigger_controller.commit_barcode(barcode, source=source)
        return None

    def on_sensor_triggered(self):
        """处理光电开关触发信号"""
        self.trigger_controller.handle_optical_trigger()
        return None

    def on_shortcut_triggered(self):
        """处理快捷键触发信号（F2）"""
        self.trigger_controller.handle_shortcut_trigger()
        return None

    def _get_tcp_mirror_identity(self):
        return self.resource_lifecycle_controller.read_tcp_mirror_identity()

    def _set_tcp_mirror_identity(self, server) -> bool:
        return self.resource_lifecycle_controller.write_tcp_mirror_identity(
            server
        )


    def register_shutdown_ready_recipient(self, recipient, *, owner=None) -> bool:
        if (
            not callable(recipient)
            or not isinstance(owner, QObject)
            or sip.isdeleted(owner)
        ):
            return False
        register = getattr(
            self.sequence_event_bus,
            "register_workflow_continuation_recipient",
            None,
        )
        if not callable(register):
            return False
        try:
            recipient_ref = WeakMethod(recipient)
        except TypeError:
            try:
                recipient_ref = ref(recipient)
            except TypeError:
                return False
        register(
            "shutdown-ready",
            "main-window",
            self._stage_shutdown_ready,
            owner=self,
        )
        self._shutdown_ready_recipient_ref = recipient_ref
        self._shutdown_ready_recipient_owner_ref = ref(owner)
        return True

    def _initialize_shutdown_ready_release_port(self) -> None:
        self._shutdown_ready_recipient_ref = None
        self._shutdown_ready_recipient_owner_ref = None
        self._shutdown_ready_staged_event = None
        self._shutdown_close_release_pending_generation = None
        self._shutdown_close_release_attempt = 0
        self._shutdown_close_release_attempt_generation = None
        self._shutdown_close_release_attempt_token = None
        self._shutdown_close_release_retry_queued_generation = None
        self._shutdown_close_release_timer = QTimer(self)
        self._shutdown_close_release_timer.setSingleShot(True)
        self._shutdown_close_release_timer.timeout.connect(
            self._attempt_staged_shutdown_close_release
        )

    def _stage_shutdown_ready(self, event) -> bool:
        if (
            sip.isdeleted(self)
            or type(event) is not ShutdownReady
            or event.shutdown_generation
            != self.workflow_model.shutdown_generation
        ):
            return False
        current = self._shutdown_ready_staged_event
        if current is not None and current != event:
            return False
        self._shutdown_ready_staged_event = event
        return True

    def _release_staged_shutdown_close(self, shutdown_generation: int) -> bool:
        if sip.isdeleted(self) or type(shutdown_generation) is not int:
            return False
        event = self._shutdown_ready_staged_event
        if (
            type(event) is not ShutdownReady
            or event.shutdown_generation != shutdown_generation
        ):
            return False
        self._shutdown_close_release_pending_generation = shutdown_generation
        self._shutdown_close_release_attempt = 0
        self._attempt_staged_shutdown_close_release()
        return True

    def _attempt_staged_shutdown_close_release(self) -> bool:
        if sip.isdeleted(self):
            return False
        generation = self._shutdown_close_release_pending_generation
        event = self._shutdown_ready_staged_event
        if (
            generation is None
            or type(event) is not ShutdownReady
            or event.shutdown_generation != generation
        ):
            return False
        if self._shutdown_close_release_attempt_token is not None:
            if generation == self._shutdown_close_release_attempt_generation:
                self._shutdown_close_release_retry_queued_generation = generation
                timer = self._shutdown_close_release_timer
                if not sip.isdeleted(timer) and not timer.isActive():
                    timer.start(0)
                return False
            return False
        attempt_token = object()
        self._shutdown_close_release_attempt_generation = generation
        self._shutdown_close_release_attempt_token = attempt_token
        self._shutdown_close_release_retry_queued_generation = None
        owner_ref = self._shutdown_ready_recipient_owner_ref
        recipient_ref = self._shutdown_ready_recipient_ref
        owner = None if owner_ref is None else owner_ref()
        recipient = None if recipient_ref is None else recipient_ref()
        if owner is None or recipient is None or sip.isdeleted(owner):
            self._shutdown_close_release_attempt_generation = None
            self._shutdown_close_release_attempt_token = None
            self._shutdown_close_release_pending_generation = None
            self._shutdown_ready_staged_event = None
            return True
        try:
            released = recipient(event) is True
        except BaseException:
            released = False
        if sip.isdeleted(self):
            return True
        if self._shutdown_close_release_attempt_token is not attempt_token:
            return False
        self._shutdown_close_release_attempt_generation = None
        self._shutdown_close_release_attempt_token = None
        if released:
            self._shutdown_close_release_timer.stop()
            self._shutdown_close_release_retry_queued_generation = None
            self._shutdown_close_release_pending_generation = None
            self._shutdown_ready_staged_event = None
            return True
        timer = self._shutdown_close_release_timer
        if sip.isdeleted(timer):
            return False
        if (
            self._shutdown_close_release_attempt < 5
            and not timer.isActive()
        ):
            delay = min(25 * (2**self._shutdown_close_release_attempt), 400)
            self._shutdown_close_release_attempt += 1
            timer.start(delay)
        return False

    def _restart_staged_shutdown_close_release(
        self, shutdown_generation: int
    ) -> bool:
        if (
            sip.isdeleted(self)
            or type(shutdown_generation) is not int
            or self._shutdown_close_release_pending_generation
            != shutdown_generation
            or type(self._shutdown_ready_staged_event) is not ShutdownReady
            or self._shutdown_ready_staged_event.shutdown_generation
            != shutdown_generation
        ):
            return False
        timer = self._shutdown_close_release_timer
        if sip.isdeleted(timer):
            return False
        self._shutdown_close_release_attempt = 0
        self._shutdown_close_release_retry_queued_generation = shutdown_generation
        timer.stop()
        timer.start(0)
        return True

    def request_application_shutdown(self, shutdown_generation: int) -> bool:
        coordinator = getattr(self, "shutdown_coordinator", None)
        if coordinator is None:
            return False
        return coordinator.request_shutdown(
            shutdown_generation,
            self.is_workflow_active(),
        )

    def raise_shutdown_progress(self, shutdown_generation: int) -> bool:
        if sip.isdeleted(self):
            return False
        pending_generation = getattr(
            self, "_shutdown_close_release_pending_generation", None
        )
        if pending_generation is not None:
            return self._restart_staged_shutdown_close_release(
                shutdown_generation
            )
        coordinator = getattr(self, "shutdown_coordinator", None)
        if coordinator is None or not getattr(coordinator, "_active", False):
            return False
        return coordinator.raise_progress(shutdown_generation)

    def _lightweight_child_cleanup(self) -> bool:
        return self.resource_lifecycle_controller.lightweight_child_cleanup()

    def closeEvent(self, event):
        """Translate child close without running the application flush here."""
        if not self.isVisible():
            self._lightweight_child_cleanup()
            super().closeEvent(event)
            return
        owner = self.window()
        delegate = (
            None
            if owner is self
            else getattr(owner, "request_application_shutdown_from_child", None)
        )
        if callable(delegate):
            event.ignore()
            delegate()
            return
        self._lightweight_child_cleanup()
        super().closeEvent(event)

    def flush_excel_spool_build(self, *, on_close: bool = False) -> list[tuple[str, str]]:
        """Run the documented blocking compatibility adapter directly."""
        service = self.export_service
        model = self.export_model
        product_model = ""
        lineedit = getattr(self, "lineedit_type", None)
        if lineedit is not None:
            try:
                product_model = lineedit.text() or ""
            except (AttributeError, RuntimeError, TypeError):
                product_model = ""
        return service.flush_spool_targets(
            model.tracked_spool_targets(),
            analysis_config=getattr(self, "analysis_config", {}),
            product_model=product_model,
            on_close=on_close,
        )

    def reset_test_reord(self):
        return self.recording_statistics_service.reset_statistics()

    def on_reset_statistics_clicked(self):
        return self.recording_statistics_service.reset_statistics()

    def lineedit_count_lose_focus(self, lineedit):
        self.current_recorded_count = int(lineedit.text())
        self.recording_persistence_service.persist_view_count(
            self.lineedit_type.text(),
            self.lineedit_count.text(),
            self.lineedit_s_or_n.text(),
            self.barcode_scanner_box.isChecked(),
        )
        # 退出编辑态：回到只读
        lineedit.setReadOnly(True)
        lineedit.clearFocus()
        if lineedit.text() == "":
            result_count, _ = LoadUiConfig.load_recorded_num_from_json(self.default_logger)
            lineedit.setText(str(result_count))

    def lineedit_type_lose_focus(self, lineedit):
        self.recording_persistence_service.persist_view_count(
            self.lineedit_type.text(),
            self.lineedit_count.text(),
            self.lineedit_s_or_n.text(),
            self.barcode_scanner_box.isChecked(),
        )
        # 退出编辑态：回到只读
        lineedit.setReadOnly(True)
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

    def on_tcp_btn_clicked(self):
        self.trigger_controller.open_tcp_configuration()
        return None

    def clicked_ok_or_ng(self, manual=True):
        return self.recording_controller.request_manual_label(
            self.sender(), manual=manual
        )

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
        if acq_mode in {"IMPORT_AUDIO", "IMPORT_STIMULUS_AUDIO"}:
            self.sequence_event_bus.commands.import_audio_requested.emit(
                ImportAudioRequested(
                    f"import-audio-{uuid4().hex}", acq_mode, None
                )
            )
            return
        self.trigger_controller.request_start(label=label, source="manual")

    def import_audio_and_analyze(self, admitted=None):
        return self.recording_controller.handle_load_imported_audio_requested(admitted)













    def start_this_play(self, label="not_labeled", skip_sn_regex_validation: bool = False):
        """Compatibility command adapter for manual recording starts."""
        return self.trigger_controller.request_start(
            label=label,
            source="manual",
            skip_sn_regex_validation=skip_sn_regex_validation,
        )

    def set_audio_devices_available(self, available: bool, message: str = ""):
        return self.configuration_controller.set_audio_devices_available(
            available, message
        )

    def reset_work_pram(self, command, count=None):
        """Deprecated adapter to the formal recording preparation boundary."""
        return self.blocking_recording_adapter.prepare(command)

    def _start_streaming_recording(self, prepared, terminal, sample_rate=None):
        """Deprecated adapter to the formal streaming recording service."""
        return self.streaming_recording_service.start(prepared, terminal)

    def _start_blocking_recording(self, prepared, *_args, **_kwargs):
        """Deprecated adapter to the formal blocking acquisition boundary."""
        return self.blocking_recording_adapter.acquire(prepared)

    def judge_play_and_record(self, label="not_labeled", is_replay=False):
        """Deprecated start adapter; workflow admission remains controller-owned."""
        source = "replay" if is_replay else "manual"
        return self.trigger_controller.request_start(label=label, source=source)

    def run(self):
        """Compatibility facade for synchronous analysis execution."""
        return self.analysis_controller.run()

    def _handle_post_analysis_exports(self, handoff=None):
        """Deprecated adapter to the formal Export Controller command boundary."""
        return self._maybe_export_excel_results(handoff)

    def _maybe_show_analysis_result_summary(self, width: int, height: int):
        return self.analysis_view.show_summary(
            getattr(self.data_struct, "analysis_result_dict", {}), width, height
        )

    def _maybe_export_excel_results(self, command=None):
        """Deprecated adapter accepting only a formal ExportRequested command."""
        return self.export_controller.handle_export_requested(command)

    def instance_analysis_class(self, key, type, params):
        """Compatibility facade for Analysis Controller instance creation."""
        return self.analysis_controller.instance_analysis_class(key, type, params)

    def eventFilter(self, obj, event):
        """
        Persist analysis window geometry on move/resize (no close handling).
        """
        is_count_board_splitter_resize = (
            obj is getattr(self, "waveform_splitter", None) or obj is getattr(self, "count_board", None)
        ) and event.type() == QEvent.Resize
        if is_count_board_splitter_resize:
            self._schedule_count_board_splitter_reconcile()

        if event.type() in (QEvent.Move, QEvent.Resize):
            analysis_view = getattr(self, "analysis_view", None)
            if analysis_view is not None:
                analysis_view.geometry_event(obj)

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
                character = event.text()
                if (
                    QApplication.focusWidget() is self.lineedit_s_or_n
                    and self._sn_clear_on_next_scan
                    and not self.player_status_flag
                    and character
                    and character.isprintable()
                    and not character.isspace()
                ):
                    self.trigger_view.clear_serial_text()
                    self.trigger_controller.reset_barcode_state()
                    self.trigger_model.sn_textchange_manual_guard = False
                    self._sn_clear_on_next_scan = False
                handled = self._barcode_router.handle_keypress(obj, event)
                if handled is True:
                    return True

        except Exception:
            # 出异常时不影响主流程
            return super().eventFilter(obj, event)

        return super().eventFilter(obj, event)

    def _set_analysis_window_geometry(self, key: str, geo: dict):
        return self.analysis_view.set_geometry(key, geo)

    def _get_analysis_window_geometry(self, key: str):
        return self.analysis_view.get_geometry(key)

    def get_sequence_config_from_registry(self):
        return self.configuration_controller.get_sequence_config_from_registry()

    def update_using_file_combobox(self):
        return self.configuration_controller.update_using_file_combobox()

    def add_file_to_using_file_combobox(self):
        return self.configuration_controller.add_file_to_using_file_combobox()

    def on_using_file_combobox_changed(self, text):
        return self.configuration_controller.on_using_file_combobox_changed(text)

    def restore_previous_configuration(self):
        return self.configuration_controller.restore_previous_configuration()

    def get_sequence_config_from_json(self):
        return self.configuration_controller.get_sequence_config_from_json()

    def on_audio_chunk_received_playrec(self, payload):
        """Deprecated waveform adapter to the Recording View."""
        return self.recording_view.queue_recording_batch(payload)

    def on_audio_chunk_received_rec(self, payload):
        """Deprecated waveform adapter to the Recording View."""
        return self.recording_view.queue_recording_batch(payload)

    def update_player_btn_is_playing(self):
        self.player_btn.setIcon(QIcon(":/ui/icon/pause.png"))
        self.player_btn.setIconSize(QSize(35, 35))
        self.player_btn.setDisabled(True)

    def _initialize_recording_view_state(self):
        self._recording_button_state_before_recording = (
            _RECORDING_BUTTON_STATE_NOT_CAPTURED
        )

    def _recording_view_started(self):
        """Project formal Recording Controller start state into widgets."""
        self._recording_button_state_before_recording = (
            self.data_btn.isEnabled(),
            self.replayer_btn.isEnabled(),
        )
        self._clear_plot_area()
        self.update_player_btn_is_playing()
        self.replayer_btn.setDisabled(True)
        self.data_btn.setDisabled(True)

    def _recording_view_finished(self, successful):
        self._set_sn_input_recording_read_only(False)
        previous_buttons = getattr(
            self,
            "_recording_button_state_before_recording",
            _RECORDING_BUTTON_STATE_NOT_CAPTURED,
        )
        self._recording_button_state_before_recording = (
            _RECORDING_BUTTON_STATE_NOT_CAPTURED
        )
        if successful:
            self.data_btn.setEnabled(True)
            self.replayer_btn.setEnabled(True)
        elif isinstance(previous_buttons, tuple) and len(previous_buttons) == 2:
            self.data_btn.setEnabled(bool(previous_buttons[0]))
            self.replayer_btn.setEnabled(bool(previous_buttons[1]))
        self._sn_clear_on_next_scan = bool(successful)
        if successful and self.barcode_scanner_box.isChecked():
            try:
                self.lineedit_s_or_n.setFocus()
                self.lineedit_s_or_n.selectAll()
            except (RuntimeError, TypeError):
                pass
        self.update_player_btn_is_paused()

    def _present_recording_error(self, reason):
        MessageBox.warning(self, "提示", f"录音失败: {reason}")

    def _close_streaming_recording_admission(self, _prepared):
        """Protected migration alias for producer-gate closure."""
        return self.streaming_recording_service.close_admission(_prepared)

    def _quiesce_streaming_recording(self, _prepared, _reason, handle):
        """Protected migration alias for blocking producer quiescence."""
        return self.streaming_recording_service.quiesce(
            _prepared, _reason, handle
        )

    def _workflow_configuration_snapshot(self):
        """Freeze current facade configuration only when a workflow is admitted."""
        return self.configuration_model.current_snapshot()

    def _analysis_workflow_identity(self):
        model = self.workflow_model
        return {
            "import_id": model.active_import_id,
            "analysis_id": model.active_analysis_id,
            "source_id": model.analysis_source_id,
            "workflow_generation": model.workflow_generation,
            "phase": model.phase.name,
            "cancelling_domain": model.cancelling_domain,
        }

    def _workflow_import_identity(self):
        model = self.workflow_model
        return {
            "import_id": model.active_import_id,
            "workflow_generation": model.workflow_generation,
            "phase": model.phase.name,
        }

    def _wire_analysis_workflow_channels(self):
        """Wire formal Workflow commands to the Analysis owner."""
        commands = self.sequence_event_bus.commands
        commands.analysis_requested.connect(
            self.analysis_controller.handle_analysis_requested,
            Qt.QueuedConnection,
        )
        # Analysis execution is synchronous on this QObject's main thread. A
        # cancellation already admitted by Workflow must update its local gate
        # in the emitting stack, before completion can win the terminal race.
        commands.cancel_analysis_requested.connect(
            self.analysis_controller.handle_cancel_analysis_requested,
            Qt.DirectConnection,
        )

    def _wire_workflow_continuation_ports(self):
        """Register this facade's direct, instance-owned continuation ports."""
        identity = id(self)
        self._workflow_state_continuation_recipient_name = (
            f"sequence-window:{identity}:workflow-state"
        )
        self.sequence_event_bus.register_workflow_continuation_recipient(
            "workflow-state",
            self._workflow_state_continuation_recipient_name,
            self._project_workflow_state,
            owner=self,
        )

    def _workflow_analysis_snapshot_lookup(self, record_id):
        return self.recording_model.retained_analysis_inputs(record_id)

    def _recording_admission_inputs(self):
        """Expose raw View/runtime values to the Recording admission owner."""
        return RecordingAdmissionInputs(
            configuration_generation=self.configuration_model.configuration_generation,
            product_model=self.lineedit_type.text(),
            serial_number=self.lineedit_s_or_n.text(),
            scanner_enabled=self.barcode_scanner_box.isChecked(),
            current_recorded_count=self.current_recorded_count,
            last_play_count=self.last_play_count,
            recorded_path=self.recorded_path,
            recorded_signal_info=self.recorded_signal_info,
            stimulus_data=getattr(self.data_struct, "stimulus_data", None),
            stimulus_info=getattr(self.data_struct, "stimulus_info", None),
            alignment_sample_count=getattr(
                self.data_struct, "alignment_sample_count", None
            ),
        )

    def _recording_label_context(self):
        """Expose the mutable Recording projection selected by Workflow."""
        return RecordingLabelContext(
            recorded_path=self.recorded_path,
            recorded_signal_info=self.recorded_signal_info,
        )

    def on_clicked_replayer_btn(self):
        """Publish the Recording owner's immutable replay command."""
        command = self.recording_admission_service.create_replay_request()
        if command is None:
            MessageBox.warning(self, "提示", "请先进行录音")
            return False
        self.sequence_event_bus.commands.replay_requested.emit(command)
        return True

    @property
    def player_status_flag(self):
        """Read-only projection of recording-active workflow phases."""
        return bool(self.workflow_model.player_status_flag)

    @property
    def _record_workflow_busy(self):
        """Read-only alias matching ``player_status_flag`` exactly."""
        return bool(self.workflow_model.record_workflow_busy)

    @property
    def _awaiting_ok_ng(self):
        """Read-only compatibility projection of Workflow awaiting-label state."""
        return bool(self.workflow_model.awaiting_label)

    def _project_workflow_state(self, *_):
        """Compatibility delegate to the formal Workflow View."""
        return self.workflow_view.project_state_changed(*_)

    def _synchronize_workflow_shutdown(self):
        """Synchronize shutdown when its coordinator is available."""
        coordinator = getattr(self, "shutdown_coordinator", None)
        if coordinator is None:
            return False
        return coordinator.synchronize()

    def is_workflow_active(self):
        """Return the canonical workflow activity diagnostic."""
        model = getattr(self, "workflow_model", None)
        if model is not None:
            return model.is_workflow_active()
        return False

    def update_player_btn_is_paused(self):
        self.player_btn.setIcon(QIcon(":/ui/icon/play.png"))
        self.player_btn.setIconSize(QSize(35, 35))
        can_start = bool(getattr(self, "sequence_config", None))
        can_start = can_start and not getattr(self, "player_status_flag", False)
        can_start = can_start and not getattr(self, "_record_workflow_busy", False)
        can_start = can_start and getattr(self, "audio_devices_available", True)
        self.player_btn.setDisabled(not can_start)

    def refresh_channel_windows(self) -> None:
        """Compatibility delegate for the ChannelPlotWorkspace View shell."""
        return self._refresh_channel_workspace()

    def _refresh_channel_workspace(self) -> None:
        """
        Refresh plot subwindows based on current mic_channels selection.

        MainWindow assigns mic_channels after SequenceWindow construction, so this should be
        called at least once after the window is shown, and again after hardware selection changes.
        """
        channels = list(self.mic_channels or [])
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
        return self.analysis_view.close_windows()

from time import monotonic
from PyQt5.QtTest import QTest
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QMainWindow

from main_window import MainWindow
from unit_test.ui.test_workbench_recording_startup import startup

@pytest.mark.parametrize('launcher', [False, True])
@pytest.mark.parametrize('injected', [False, True])
def test_real_entrypoint_does_not_initialize_legacy_runtime(startup, monkeypatch, launcher, injected, ui_qapp):
    import builtins
    import importlib
    from base.unified_hid_device_manager import UnifiedHardwareManager
    from base.analysis import service as current_analysis
    from ui.sequence.workbench_input_ops import WorkbenchInputOpsMixin
    factory = Mock(return_value=startup.analysis_service)
    monkeypatch.setattr(current_analysis, 'AnalysisService', factory)
    forbidden = Mock(side_effect=AssertionError('legacy runtime initialized'))
    retired = {
        'ui.sequence.sequence_widget_config_ops',
        'ui.sequence.sequence_widget_ui_ops',
        'ui.sequence.sequence_widget_barcode_ops',
        'ui.sequence.sequence_widget_streaming_ops',
        'ui.sequence.sequence_widget_test_metadata_ops',
        'ui.sequence.sequence_widget_raw_csv_ops',
        'ui.sequence.sequence_widget_analysis_ops',
        'ui.sequence.sequence_widget_analysis_process_ops',
        'ui.sequence.sequence_widget_recording_process_ops',
        'ui.sequence.sequence_widget_recording_retry_ops',
        'ui.sequence.sequence_widget_progress_ops',
        'ui.sequence.sequence_widget_round_reset_ops',
        'ui.sequence.sequence_widget_serial_trigger_ops',
        'ui.sequence.product_condition_state',
        'ui.sequence.product_condition_result_ops',
        'ui.sequence.sequencement_count_board',
        'ui.sequence.recording_process_context',
        'ui.sequence.analysis_task_builder',
        'ui.sequence.analysis_report_snapshot',
        'ui.sequence.workbench_toolbar_preferences',
        'ui.sequence.toolbar_serial_button',
        'ui.sequence.motor_mode_switch_panel',
        'ui.sequence.streaming_waveform_accumulator',
        'ui.analysis_multichannel_result_window',
        'ui.signal_analysis_window',
        'ui.reference_spectrum_analysis_window',
        'base.analysis_service',
        'base.analysis_worker',
        'base.segmented_analysis_worker',
        'base.analysis_process_protocol',
        'base.analysis_result_summary',
        'base.product_test_progress',
        'base.product_test_runtime_policy',
        'base.wav_channel_mapping',
        'ui.sequence.analysis_channel_preflight',
        'base.test_round_data',
    }
    input_initializations = []
    initialize_input = WorkbenchInputOpsMixin.init_workbench_input

    def initialize_checked(view, can_commit):
        assert view.default_logger is not None
        input_initializations.append(view)
        initialize_input(view, can_commit)

    monkeypatch.setattr(WorkbenchInputOpsMixin, 'init_workbench_input', initialize_checked)
    original_import = builtins.__import__
    original_import_module = importlib.import_module

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        candidates = {name, *(f'{name}.{item}' for item in fromlist or ())}
        assert not candidates & retired, f'legacy import: {candidates & retired}'
        return original_import(name, globals, locals, fromlist, level)

    def guarded_import_module(name, package=None):
        absolute = importlib.util.resolve_name(name, package) if name.startswith('.') else name
        assert absolute not in retired, f'legacy dynamic import: {absolute}'
        return original_import_module(name, package)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.setattr(importlib, 'import_module', guarded_import_module)
    monkeypatch.setattr(UnifiedHardwareManager, 'start_scanner_and_sensor_listeners', forbidden)
    monkeypatch.setattr(UnifiedHardwareManager, 'start_serial_discrete_input_listener', forbidden)
    options = {} if injected else {'analysis_service': None}
    window = startup.launch(lambda _: ('Engineer', 'test'), launcher=launcher, **options)
    view = window.sequence_window
    assert input_initializations == [view]
    for name in ('workbench_mode', 'count_board', 'sequence_config', 'using_config_path',
                 '_analysis_process_timer', '_serial_trigger_delay_timer', 'data_struct'):
        assert not hasattr(view, name), name
    for name in ('data_btn', 'replayer_btn', 'serial_trigger_btn', 'current_round_spinbox',
                 'reset_round_button', 'condition_mode_combobox', 'lineedit_count'):
        assert not hasattr(view.toolsbar, name), name
    assert len(view.findChildren(QTimer, options=Qt.FindDirectChildrenOnly)) == 1
    assert not view._workbench_input_closed  # initialization hides the view, without releasing inputs
    assert window.analysis_service is startup.analysis_service
    ui_qapp.processEvents()
    assert len(startup.analysis_service.sync_calls) == 1
    assert factory.call_count == (0 if injected else 1)
    view.close()
    assert not view._workbench_input_closed
    window.close()
    deadline = monotonic() + 5
    while not getattr(window, "_exit_cleanup_complete", False) and monotonic() < deadline:
        ui_qapp.processEvents()
        QTest.qWait(10)
    assert window._exit_cleanup_complete
    assert view._workbench_input_closed
    assert startup.bridge.shutdown_calls == 1
    assert startup.analysis_service.shutdown_calls == 1
    forbidden.assert_not_called()

@pytest.mark.parametrize('state, configuration, hardware, calibration, scanner', [
    ('idle', False, True, True, True),
    ('analysis', True, True, True, True),
    ('deletion failure', True, True, True, True),
    ('archive', True, True, True, True),
    ('capture', True, False, False, False),
    ('service', True, True, False, False),
    ('editor', True, False, False, False),
    ('closing', True, False, False, False),
    ('prewarm', True, False, False, False),
])
def test_admission_uses_each_current_owner(ui_qapp, state, configuration, hardware, calibration, scanner):
    window = MainWindow.__new__(MainWindow)
    QMainWindow.__init__(window)
    window._closing = state == 'closing'
    window._configuration_editor_open = state == 'editor'
    window._configuration_deletion_error = 'deletion failed' if state == 'deletion failure' else ''
    window.recording_bridge = SimpleNamespace(
        hardware_busy=state == 'capture',
        service=SimpleNamespace(busy=state == 'service', can_start_recording=state != 'service'))
    window.workbench_controller = SimpleNamespace(recording_busy=state == 'archive')
    window.analysis_service = SimpleNamespace(pending_task_count=int(state == 'analysis'))
    window.ve_prewarm_lifetime = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(state='pending' if state == 'prewarm' else 'ready'),
        admission_for=lambda _: 'pending' if state == 'prewarm' else 'allowed')
    window._exit_cleanup_complete = True
    assert window._workbench_configuration_busy() is configuration
    assert window._hardware_selection_admission_available() is hardware
    assert window._calibration_admission_available() is calibration
    assert window._workbench_scanner_admission_available() is scanner
    window._exit_cleanup_complete = True
    window.close()

def test_open_editor_blocks_current_recording_admission(startup):
    window = startup.launch(lambda _: ('Engineer', 'test'))
    startup.bridge.change_worker(1)
    startup.bridge.answer_probe()
    snapshot = window.workbench_recording_device.snapshot
    assert window.workbench_controller._recording.hardware_available(snapshot)
    window._configuration_editor_open = True
    assert not window.workbench_controller._recording.hardware_available(snapshot)

def test_current_csv_drain_keeps_inputs_until_shared_service_releases(startup, ui_qapp, tmp_path):
    import multiprocessing
    from base.raw_audio_csv_service import RawAudioCsvService
    from ui.raw_audio_csv_service_bridge import RawAudioCsvServiceBridge
    from unit_test.base.raw_audio_csv_fakes import blocked_worker
    from unit_test.base.test_raw_audio_csv_worker import make_command

    gate = multiprocessing.get_context('spawn').Event()
    service = RawAudioCsvService(worker_target=blocked_worker, worker_args=(gate,))
    bridge = RawAudioCsvServiceBridge(service, ui_qapp)
    window = startup.launch(lambda _: ('Engineer', 'test'), raw_audio_csv_bridge=bridge)
    request = make_command(tmp_path, 'current-close').request
    reservation = service.reserve(request.recording_id).reservation
    assert service.commit(reservation, request) == 'accepted'
    try:
        window.close()
        ui_qapp.processEvents()
        assert window._closing
        assert not getattr(window, '_exit_cleanup_complete', False)
        assert not window.sequence_window._workbench_input_closed
        assert not window._workbench_scanner_admission_available()
        assert service.snapshot().outstanding == 1
        gate.set()
        deadline = monotonic() + 15
        while not getattr(window, '_exit_cleanup_complete', False) and monotonic() < deadline:
            ui_qapp.processEvents()
            QTest.qWait(10)
        assert window._exit_cleanup_complete
        assert service.closed.is_set()
        assert window.sequence_window._workbench_input_closed
        assert startup.bridge.shutdown_calls == 1
    finally:
        gate.set()
        service.begin_shutdown()
        assert service.closed.wait(15)
        bridge.close_delivery()

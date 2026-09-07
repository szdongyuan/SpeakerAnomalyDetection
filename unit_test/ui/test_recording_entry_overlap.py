from types import SimpleNamespace
from unittest import mock

import pytest


class _Service:
    def __init__(self, can_start):
        self.can_start_recording = can_start
        self.busy = True


def _analysis_snapshot(*types):
    keys = [f"item-{index}" for index, _value in enumerate(types)]
    analysis = {"display_sequence": keys}
    analysis.update({key: {"type": value} for key, value in zip(keys, types)})
    return {"analysis_config": analysis}


def _admission_host(can_start, *, new_types=("SPL",), old_types=("SPL",),
                    ve_prewarm_lifetime=None, mic=None, mic_channels=None):
    from ui.sequence.sequence_widget_recording_process_ops import (
        SequenceWidgetRecordingProcessOpsMixin,
    )

    old_context = SimpleNamespace(
        processor=object(), final=True, failed=False, cancelled=False,
        recent_session_config_snapshot=_analysis_snapshot(*old_types),
        enabled_analysis_identifiers=tuple(old_types),
    )
    host = SimpleNamespace(
        recording_bridge=SimpleNamespace(service=_Service(can_start)),
        _recording_process_contexts={"A": old_context},
        _record_workflow_busy=True,
        player_status_flag=True,
        analysis_config=_analysis_snapshot(*new_types)["analysis_config"],
        ve_prewarm_lifetime=ve_prewarm_lifetime,
        mic=mic,
        mic_channels=[] if mic_channels is None else mic_channels,
    )
    host.streaming_processor = old_context.processor
    host._can_start_recording_workflow = (
        SequenceWidgetRecordingProcessOpsMixin._can_start_recording_workflow.__get__(host)
    )
    host._can_start_calibration_workflow = (
        SequenceWidgetRecordingProcessOpsMixin._can_start_calibration_workflow.__get__(host)
    )
    return host


def _prewarm_lifetime(state, *, device=None, channels=(7, 1), failed_device=None):
    from base.ve3668n_input import ve_acquisition_signature
    from base.ve3668n_prewarm_lifetime import VePrewarmLifetime
    from unit_test.base.ve3668n_fakes import device_info

    device = device or device_info()
    signature = ve_acquisition_signature(
        device, channels, device["input_config"]["sample_rate"])
    lifetime = VePrewarmLifetime()
    if state != "available":
        assert lifetime.claim("selection", signature)
    if state == "succeeded":
        assert lifetime.mark_succeeded("selection", signature)
    elif state == "skipped_busy":
        assert lifetime.mark_skipped_busy("selection", signature, "hardware busy")
    elif state in ("failed", "ownership_uncertain"):
        failed_device = failed_device or device
        failed_signature = ve_acquisition_signature(
            failed_device, channels, failed_device["input_config"]["sample_rate"])
        if failed_signature != signature:
            lifetime = VePrewarmLifetime()
            assert lifetime.claim("selection", failed_signature)
            signature = failed_signature
        fault = SimpleNamespace(
            stage="start_task", code=-12001, detail="start failed",
            diagnostics=("native start failed",))
        assert lifetime.mark_failed(
            "selection", signature, fault,
            ownership_safe=state != "ownership_uncertain")
    return lifetime, device, list(channels)


@pytest.mark.parametrize("state,expected", [
    ("available", True),
    ("pending", False),
    ("succeeded", True),
    ("skipped_busy", True),
    ("failed", False),
    ("ownership_uncertain", False),
])
def test_shared_admission_combines_process_prewarm_outcome(state, expected):
    lifetime, device, channels = _prewarm_lifetime(state)
    host = _admission_host(
        True, ve_prewarm_lifetime=lifetime, mic=device, mic_channels=channels)

    assert host._can_start_recording_workflow() is expected
    assert host._can_start_calibration_workflow() is expected


def test_shared_admission_scopes_safe_failure_to_exact_ve_signature():
    from copy import deepcopy
    from unit_test.base.ve3668n_fakes import device_info

    failed_device = device_info(machine_id="failed-device")
    selected_device = deepcopy(failed_device)
    selected_device["machine_id"] = "replacement-device"
    lifetime, _, channels = _prewarm_lifetime(
        "failed", device=selected_device, failed_device=failed_device)

    replacement = _admission_host(
        True, ve_prewarm_lifetime=lifetime,
        mic=selected_device, mic_channels=channels)
    ordinary = _admission_host(
        True, ve_prewarm_lifetime=lifetime,
        mic={"backend": "soundcard"}, mic_channels=[0])

    assert replacement._can_start_recording_workflow()
    assert replacement._can_start_calibration_workflow()
    assert ordinary._can_start_recording_workflow()
    assert ordinary._can_start_calibration_workflow()


def test_prewarm_terminal_refreshes_recording_and_hardware_controls():
    from main_window import MainWindow

    refresh_recording = mock.Mock()
    refresh_hardware = mock.Mock()
    host = SimpleNamespace(
        sequence_window=SimpleNamespace(
            update_player_btn_is_paused=refresh_recording),
        _update_hardware_busy_state=refresh_hardware,
    )

    MainWindow._refresh_ve_admission_controls(host)

    refresh_recording.assert_called_once_with()
    refresh_hardware.assert_called_once_with()


@pytest.mark.parametrize("state,expected", [
    ("available", True),
    ("pending", False),
    ("succeeded", True),
    ("skipped_busy", True),
    ("failed", True),
    ("ownership_uncertain", True),
])
def test_main_hardware_selection_uses_live_ownership_after_historical_failure(
    state, expected,
):
    from main_window import MainWindow

    lifetime, _device, _channels = _prewarm_lifetime(state)
    host = SimpleNamespace(
        ve_prewarm_lifetime=lifetime,
        _hardware_busy=lambda: False,
    )

    assert MainWindow._hardware_selection_admission_available(host) is expected


def test_confirmed_worker_death_reopens_nonmatching_hardware_without_new_prewarm():
    from copy import deepcopy
    from main_window import MainWindow
    from unit_test.base.ve3668n_fakes import device_info

    failed_device = device_info(machine_id="failed-device")
    replacement_device = deepcopy(failed_device)
    replacement_device["machine_id"] = "replacement-device"
    lifetime, _, channels = _prewarm_lifetime(
        "ownership_uncertain", device=replacement_device,
        failed_device=failed_device)

    ordinary = _admission_host(
        False, ve_prewarm_lifetime=lifetime,
        mic={"backend": "soundcard"}, mic_channels=[0])
    replacement = _admission_host(
        False, ve_prewarm_lifetime=lifetime,
        mic=replacement_device, mic_channels=channels)
    exact = _admission_host(
        False, ve_prewarm_lifetime=lifetime,
        mic=failed_device, mic_channels=channels)

    # Live service ownership is still unsafe: every capture stays closed.
    assert not ordinary._can_start_recording_workflow()
    assert not replacement._can_start_calibration_workflow()

    # The service later confirms worker death and reopens its capture slot.
    for host in (ordinary, replacement, exact):
        host.recording_bridge.service.can_start_recording = True
    assert ordinary._can_start_recording_workflow()
    assert ordinary._can_start_calibration_workflow()
    assert replacement._can_start_recording_workflow()
    assert replacement._can_start_calibration_workflow()
    assert not exact._can_start_recording_workflow()
    assert not exact._can_start_calibration_workflow()

    live = SimpleNamespace(hardware_busy=True)
    main = SimpleNamespace(
        ve_prewarm_lifetime=lifetime,
        recording_bridge=live,
        sequence_window=SimpleNamespace(
            player_status_flag=False,
            _can_start_calibration_workflow=lambda: True),
        mic={"backend": "soundcard"},
        mic_channels=[0],
    )
    main._hardware_busy = MainWindow._hardware_busy.__get__(main)
    main._ve_signature = lambda mic, selected: MainWindow._ve_signature(
        mic, selected)
    assert not MainWindow._hardware_selection_admission_available(main)
    assert not MainWindow._calibration_admission_available(main)
    live.hardware_busy = False
    assert MainWindow._hardware_selection_admission_available(main)
    assert MainWindow._calibration_admission_available(main)
    main.mic, main.mic_channels = replacement_device, channels
    assert MainWindow._calibration_admission_available(main)
    main.mic = failed_device
    assert not MainWindow._calibration_admission_available(main)

    # Consuming the sole opportunity is permanent even though ownership healed.
    failed_signature = lifetime.snapshot().failed_signature
    assert not lifetime.claim("second-selection", failed_signature)


def test_shared_admission_allows_old_result_finalization_but_obeys_service_slot():
    assert _admission_host(True)._can_start_recording_workflow()
    assert not _admission_host(False)._can_start_recording_workflow()


@pytest.mark.parametrize("analysis_types", [
    ("SPL",), ("FBA",), ("SPEC",), ("SPL", "FBA", "SPEC"), (),
])
def test_shared_admission_allows_only_approved_new_and_old_analysis(analysis_types):
    assert _admission_host(True, new_types=analysis_types,
                           old_types=("SPL",))._can_start_recording_workflow()
    assert _admission_host(True, new_types=("SPEC",),
                           old_types=analysis_types)._can_start_recording_workflow()


@pytest.mark.parametrize("analysis_type", [
    "ED", "PD", "PM", "FFT", "CQT", "future-analysis",
])
@pytest.mark.parametrize("location", ["new", "old"])
def test_shared_admission_retains_broad_busy_for_ineligible_analysis(
    analysis_type, location,
):
    new_types = (analysis_type,) if location == "new" else ("SPL",)
    old_types = (analysis_type,) if location == "old" else ("SPEC",)
    host = _admission_host(True, new_types=new_types, old_types=old_types)

    assert not host._can_start_recording_workflow()


def test_shared_admission_reports_capacity_backpressure_normally():
    host = _admission_host(False, new_types=("FBA",), old_types=("SPEC",))

    assert not host._can_start_recording_workflow()


def test_shared_admission_preserves_nonrecording_playback_and_close_blockers():
    playback = _admission_host(True)
    playback.streaming_processor = object()
    assert not playback._can_start_recording_workflow()

    playback_without_processor = _admission_host(True)
    playback_without_processor.streaming_processor = None
    assert not playback_without_processor._can_start_recording_workflow()

    closing = _admission_host(True)
    closing._recording_closed = True
    assert not closing._can_start_recording_workflow()


def test_shared_admission_preserves_explicit_modal_preflight_cleanup_and_shutdown_blockers():
    blocker_names = (
        "_test_metadata_validation_open",
        "_product_test_program_config_dialog_open",
        "_serial_product_error_dialog_open",
        "_recording_cleanup_in_progress",
        "_streaming_cleanup_in_progress",
        "_closing",
        "_shutdown_started",
    )
    for blocker_name in blocker_names:
        host = _admission_host(True)
        setattr(host, blocker_name, True)
        assert not host._can_start_recording_workflow(), blocker_name


def test_shared_admission_checks_both_production_playback_controllers():
    for controller_name in ("_condition_playback_controller", "recent_session_panel"):
        for playing in (False, True):
            host = _admission_host(True)
            controller = SimpleNamespace(is_audio_playing=mock.Mock(return_value=playing))
            if controller_name == "recent_session_panel":
                host.recent_session_panel = SimpleNamespace(playback_controller=controller)
            else:
                host._condition_playback_controller = controller

            assert host._can_start_recording_workflow() is (not playing)
            controller.is_audio_playing.assert_called_once_with()


def test_shared_admission_does_not_swallow_playback_controller_errors():
    host = _admission_host(True)
    host._condition_playback_controller = SimpleNamespace(
        is_audio_playing=mock.Mock(side_effect=RuntimeError("playback state unavailable")))

    import pytest
    with pytest.raises(RuntimeError, match="playback state unavailable"):
        host._can_start_recording_workflow()


def test_sensor_and_delayed_directional_entries_use_shared_admission():
    from ui.sequence.sequence_widget_barcode_ops import SequenceWidgetBarcodeOpsMixin

    for allowed in (True, False):
        host = _admission_host(allowed)
        host.default_logger = SimpleNamespace(info=lambda *_: None)
        host.start_this_play = mock.Mock()
        host._start_directional_workflow = mock.Mock()
        host._pending_serial_trigger_direction = "forward"

        SequenceWidgetBarcodeOpsMixin.on_sensor_triggered(host)
        SequenceWidgetBarcodeOpsMixin._on_serial_trigger_delay_timeout(host)

        assert host.start_this_play.called is allowed
        assert host._start_directional_workflow.called is allowed


def test_serial_product_entry_checks_shared_admission_before_preparing():
    from ui.sequence.sequence_widget_serial_trigger_ops import (
        SequenceWidgetSerialTriggerOpsMixin,
    )

    for allowed in (True, False):
        host = _admission_host(allowed)
        host.default_logger = SimpleNamespace(info=lambda *_: None, warning=lambda *_: None, error=lambda *_: None)
        host._prepare_next_manual_product_condition_recording = mock.Mock(return_value=True)
        host._is_import_audio_mode = lambda: False
        host.checked_work_status_message = lambda: False
        host.start_this_play = mock.Mock()
        host._get_active_product_condition_key = lambda: "condition"
        host._serial_product_condition_executing = False
        host.player_status_flag = False

        result = SequenceWidgetSerialTriggerOpsMixin._start_serial_product_condition(host, "AA")

        assert result is allowed
        assert host._prepare_next_manual_product_condition_recording.called is allowed
        assert host.start_this_play.called is allowed


def _actual_product_entry_host(can_start, *, trigger_state=""):
    from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin
    from ui.sequence.sequence_widget_serial_trigger_ops import SequenceWidgetSerialTriggerOpsMixin

    host = _admission_host(can_start)
    condition = {"condition_name": "running", "trigger_state": trigger_state}
    host.product_test_condition_configs = [condition]
    host._product_condition_sequence = lambda: [condition]
    host._load_sequence_config_for_product_condition = lambda _condition: (True, "")
    host._validate_active_product_program_acquisition_modes = lambda: (True, "")
    host._begin_test_round_metadata = lambda: True
    host._generate_recording_token = lambda: "group"
    host._manual_product_condition_index = 0
    host._manual_product_condition_group_id = ""
    host._manual_product_condition_completed_keys = set()
    host._manual_product_condition_results = {}
    host._set_product_condition_round_pending = lambda: None
    host._set_active_product_condition_stage = lambda *_: None
    host._is_import_audio_mode = lambda: False
    host.checked_work_status_message = lambda: False
    host._get_active_product_condition_key = lambda: "running"
    host._product_condition_runtime_key = lambda *_: "running"
    host._serial_product_conditions = lambda: [(condition, trigger_state)]
    host._serial_product_close_frame = lambda: ""
    host._serial_product_latched_frame = ""
    host._serial_product_condition_executing = False
    host._serial_product_session_started = False
    host._serial_product_error_dialog_open = False
    host._product_test_program_config_dialog_open = False
    host._prepare_next_manual_product_condition_recording = (
        SequenceWidgetAnalysisOpsMixin._prepare_next_manual_product_condition_recording.__get__(host)
    )
    host.start_this_play = mock.Mock(
        side_effect=lambda *_: (
            setattr(host, "_record_workflow_busy", True),
            setattr(host, "player_status_flag", True),
        )
    )
    host.default_logger = SimpleNamespace(
        info=lambda *_: None, warning=lambda *_: None, error=lambda *_: None)
    host._start_serial_product_condition = (
        SequenceWidgetSerialTriggerOpsMixin._start_serial_product_condition.__get__(host)
    )
    return host


def test_actual_product_button_preparation_uses_capture_admission_not_broad_busy():
    from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin

    for allowed in (True, False):
        host = _actual_product_entry_host(allowed)
        SequenceWidgetAnalysisOpsMixin.on_clicked_player_btn(host)
        assert host.start_this_play.called is allowed


def test_actual_full_serial_frame_uses_capture_admission_not_broad_busy():
    from ui.sequence.sequence_widget_serial_trigger_ops import (
        SequenceWidgetSerialTriggerOpsMixin,
    )

    for allowed in (True, False):
        host = _actual_product_entry_host(allowed, trigger_state="AA BB")
        SequenceWidgetSerialTriggerOpsMixin.on_serial_full_frame_received(
            host, {"raw_hex": "AA BB", "product_full_frame": True})
        assert host.start_this_play.called is allowed


def test_production_judge_entry_reaches_start_when_old_result_is_busy():
    from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin

    host = _admission_host(True)
    host.checked_work_status_message = lambda: False
    host._begin_test_round_metadata = lambda: True
    host.analysis_window = []
    host._analysis_result_summary_window = None
    host._clear_plot_area = mock.Mock()
    host._cleanup_streaming_resources = mock.Mock()
    host.update_player_btn_is_playing = mock.Mock()
    host.replayer_btn = SimpleNamespace(setDisabled=mock.Mock())
    host.data_btn = SimpleNamespace(setDisabled=mock.Mock())
    host.reset_work_pram = mock.Mock(return_value=({"device": {}, "num_frames": 1}, 48000))
    host._capture_recording_wav_calibration_metadata = mock.Mock()
    host._start_process_recording = mock.Mock()
    host.default_logger = SimpleNamespace(error=lambda *_: None)
    host._recording_workflow_token = object()
    host._recording_wav_calibration_metadata = None
    host._clear_plot_area = mock.Mock()
    host.player_status_flag = True

    SequenceWidgetAnalysisOpsMixin.judge_play_and_record(host)

    host._start_process_recording.assert_called_once()


def test_button_state_uses_shared_capture_admission_even_when_broad_busy_is_true():
    from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin

    for allowed in (True, False):
        host = _admission_host(allowed)
        button = SimpleNamespace(setIcon=lambda *_: None, setIconSize=lambda *_: None,
                                 setDisabled=mock.Mock(), setToolTip=lambda *_: None)
        host.player_btn = button
        host.sequence_config = [{}]
        SequenceWidgetUiOpsMixin.update_player_btn_is_paused(host)
        assert button.setDisabled.call_args.args == (not allowed,)


def test_barcode_commit_and_immediate_directional_entry_use_shared_admission(ui_qapp):
    from ui.sequence.sequence_widget_barcode_ops import SequenceWidgetBarcodeOpsMixin

    for allowed in (True, False):
        host = _admission_host(allowed)
        host.default_logger = SimpleNamespace(info=lambda *_: None)
        host.barcode_scanner_box = SimpleNamespace(isChecked=lambda: True)
        host._is_sn_locked_for_product_round = lambda: False
        host._is_barcode_commit_temporarily_suppressed = lambda: False
        host._should_lock_sn_for_cycle = lambda: False
        host._normalize_barcode = lambda text: str(text).strip()
        host._barcode_has_invalid_chars = lambda _barcode: (False, [])
        host._normalize_trigger_direction = lambda direction: direction
        host._resolve_serial_trigger_delay_ms = lambda _config: 0
        host._serial_trigger_config = {}
        host._start_directional_workflow = mock.Mock()
        host._last_committed_barcode = None
        host._last_committed_barcode_time = 0.0
        host._barcode_commit_dedup_window_sec = 1.0
        host.lineedit_s_or_n = SimpleNamespace(
            setText=mock.Mock(), setFocus=lambda: None, selectAll=lambda: None)
        host.lineedit_type = object()
        host.lineedit_count = object()

        SequenceWidgetBarcodeOpsMixin._commit_barcode(host, "SN-1")
        SequenceWidgetBarcodeOpsMixin.on_directional_triggered(host, "forward")

        assert host.lineedit_s_or_n.setText.called is allowed
        assert host._start_directional_workflow.called is allowed

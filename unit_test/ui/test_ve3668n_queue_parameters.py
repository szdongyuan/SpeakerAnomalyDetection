"""Active queues own the frozen VK parameters at every runtime boundary."""
from copy import deepcopy
import json
from types import MethodType, SimpleNamespace
from unittest import mock

import pytest
import soundfile as sf

from base.ve3668n_input import ve_acquisition_signature
from base.ve3668n_prewarm_lifetime import VePrewarmLifetime
from base.wav_calibration_metadata import inspect_wav_calibration_metadata
from main_window import MainWindow
from unit_test.base.ve3668n_fakes import device_info, input_config
from unit_test.ui.test_ve3668n_recording import (
    RecordingHost, host_factory, capture_audio, finish_ve_capture, save_calibration,
)
from ui.sequence.sequence_widget_serial_trigger_ops import SequenceWidgetSerialTriggerOpsMixin
from unit_test.ui.test_ve3668n_prewarm_trigger import _WindowHarness, _completion, _discovery
from unit_test.ui.test_ve3668n_calibration import setup, seed_old, offered, accept, release


class ConditionRecordingHost(RecordingHost, SequenceWidgetSerialTriggerOpsMixin):
    pass


@pytest.fixture
def condition_host(host_factory, tmp_path):
    from PyQt5.QtWidgets import QPushButton
    from ui.sequence.sequence_widget_config_ops import SequenceWidgetConfigOpsMixin
    from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin

    host = host_factory(host_type=ConditionRecordingHost)
    target = deepcopy(host.sequence_config)
    target[0]["seq1"]["acq"]["detail"].update(sample_rate=96000, ve_range_index=5)
    path = tmp_path / "target-queue.json"
    path.write_text(json.dumps(target), encoding="utf-8")
    host.target_queue_path = path
    host.toolsbar = SimpleNamespace(player_btn=QPushButton())
    host.toolsbar.player_btn.clicked.connect(host.on_clicked_player_btn)
    host.product_test_condition_configs = [
        {"key": "old", "test_queue": "old-queue", "condition_name": "old"},
        {"key": "target", "test_queue": str(path), "condition_name": "target"},
    ]
    host._manual_product_condition_index = 1
    host._get_active_product_condition_key = MethodType(
        SequenceWidgetAnalysisOpsMixin._get_active_product_condition_key, host)
    host.checked_work_status_message = lambda: False  # Business metadata is outside queue admission.
    host._is_sequence_config_path = MethodType(SequenceWidgetConfigOpsMixin._is_sequence_config_path, host)
    host._refresh_test_mode_availability = mock.Mock()
    host._set_sequence_config_available_state = mock.Mock()
    host.init_fft_and_stft_flag = mock.Mock()
    host._load_sequence_config_for_product_condition = mock.Mock(side_effect=lambda condition:
        SequenceWidgetConfigOpsMixin._apply_sequence_config_from_path(host, condition["test_queue"]))
    host.ve_prewarm_lifetime = VePrewarmLifetime()
    yield host
    host.toolsbar.player_btn.close()


def set_previous_queue_failure(host, failure):
    detail = host.sequence_config[0]["seq1"]["acq"]["detail"]
    if failure == "invalid":
        detail["sample_rate"] = None
    elif failure == "profile_io":
        host.ve_profile_store.load = mock.Mock(side_effect=OSError("old profile denied"))
    else:
        detail.update(sample_rate=48000, ve_range_index=1)
        device = device_info(input_config=input_config(48000, range_min=-5, range_max=5))
        signature = ve_acquisition_signature(device, host.mic_channels, 48000)
        host.ve_prewarm_lifetime.claim("old", signature)
        host.ve_prewarm_lifetime.mark_failed("old", signature,
            SimpleNamespace(stage="start", detail="old configuration rejected"), ownership_safe=True)


def enter_condition(host, entry):
    if entry == "manual":
        from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin
        SequenceWidgetUiOpsMixin.update_player_btn_is_paused(host)
        assert host.player_btn.isEnabled()
        host.player_btn.click()
    else:
        host.product_test_condition_configs[0]["trigger_state"] = "01 02"
        host.product_test_condition_configs[1]["trigger_state"] = "03 04"
        host._manual_product_condition_index = 0
        host.on_serial_full_frame_received({"raw_hex": "03 04"})


def test_device_resolver_copies_identity_and_replaces_stale_profile():
    from base.ve3668n_recording_config import resolve_ve_recording_device
    device = device_info(input_config=None)
    before = deepcopy(device)
    resolved = resolve_ve_recording_device(device, {"sample_rate": 96000, "ve_range_index": 5})
    assert resolved["input_config"] == input_config(96000, range_min=-.1, range_max=.1)
    assert device == before
    with pytest.raises(ValueError, match="device"):
        resolve_ve_recording_device(None, {})


def test_queue_switches_freeze_frames_trim_preview_metadata_and_calibration(host_factory):
    host = host_factory(live=True)
    save_calibration(host, physical=7, factor=12)
    profile_bytes = host.ve_profile_store.path.read_bytes()
    calibration_bytes = host.ve_calibration_store.path.read_bytes()
    base = deepcopy(host.sequence_config)
    queue_a, queue_b = deepcopy(base), deepcopy(base)
    queue_a[0]["seq1"]["acq"]["detail"].update(sample_rate=48000, ve_range_index=1)
    queue_b[0]["seq1"]["acq"]["detail"].update(sample_rate=96000, ve_range_index=5)
    signatures = []
    for queue, rate, limit in ((queue_a, 48000, 5), (queue_b, 96000, .1), (queue_a, 48000, 5)):
        host.sequence_config = deepcopy(queue)
        host.judge_play_and_record()
        session = host._recording_process_session
        request = session.request
        assert request.sample_rate == rate
        assert request.target_samples == int(.05 * rate)
        assert request.trim_samples == int(.01 * rate)
        assert request.device["input_config"].to_dict() == input_config(rate, range_min=-limit, range_max=limit)
        signatures.append(ve_acquisition_signature(request.device, request.channels, rate))
        host.sequence_config[0]["seq1"]["acq"]["detail"].update(sample_rate=8000, ve_range_index=6)
        capture, audio = capture_audio(request)
        preview = capture.snapshot(generation=session.generation, sequence=1)
        assert preview.waveforms[0].time[-1] == 0
        assert preview.waveforms[0].time[0] == pytest.approx(-(int(.04 * rate) - 1) / rate)
        assert sf.info(request.path).samplerate == rate
        assert sf.info(request.path).frames == int(.04 * rate)
        metadata = inspect_wav_calibration_metadata(request.path).metadata
        assert metadata["acquisition"]["sample_rate"] == rate
        assert metadata["acquisition"]["range_max"] == limit
        assert metadata["recorded_channels"][0]["v2pa_factor"] == 12
        finish_ve_capture(host, session, audio)
        assert host.data_struct.sample_rate == rate
    assert signatures[0] == signatures[2] != signatures[1]
    assert host.ve_profile_store.path.read_bytes() == profile_bytes
    assert host.ve_calibration_store.path.read_bytes() == calibration_bytes


@pytest.mark.parametrize("key,bad", [("sample_rate", True), ("sample_rate", 48000.5),
                                     ("sample_rate", None), ("ve_range_index", 7)])
def test_invalid_explicit_queue_parameters_never_fall_back(host_factory, key, bad):
    host = host_factory()
    detail = host.sequence_config[0]["seq1"]["acq"]["detail"]
    detail.update(sample_rate=48000, ve_range_index=1)
    detail[key] = bad
    host.ve_profile_store.load = mock.Mock(side_effect=AssertionError("unneeded profile read"))
    with pytest.raises(ValueError, match=key):
        host.reset_work_pram("not_labeled")
    host.ve_profile_store.load.assert_not_called()


def test_explicit_queue_ignores_profile_io_but_missing_rate_propagates_it(host_factory):
    host = host_factory()
    detail = host.sequence_config[0]["seq1"]["acq"]["detail"]
    detail.update(sample_rate=48000, ve_range_index=1)
    host.ve_profile_store.load = mock.Mock(side_effect=OSError("profile denied"))
    assert host.reset_work_pram("not_labeled")[1] == 48000
    host.ve_profile_store.load.assert_not_called()
    del detail["sample_rate"]
    with pytest.raises(OSError, match="profile denied"):
        host.reset_work_pram("not_labeled")


def test_prewarm_and_both_admissions_use_current_queue(host_factory):
    host = host_factory()
    detail = host.sequence_config[0]["seq1"]["acq"]["detail"]
    detail.update(sample_rate=96000, ve_range_index=5)
    window = _WindowHarness()
    window.sequence_window = host
    window.mic, window.mic_channels = host.mic, host.mic_channels
    host.ve_prewarm_lifetime = window.ve_prewarm_lifetime
    assert window._try_start_ve_prewarm(host.mic, host.mic_channels, "queue") == "accepted"
    request = window.recording_bridge.calls[0]
    assert request.sample_rate == 96000 and request.device["input_config"]["range_max"] == .1
    lifetime = window.ve_prewarm_lifetime
    state = lifetime.snapshot()
    lifetime.mark_failed(state.token, request.signature,
                         SimpleNamespace(stage="start", detail="rejected"), ownership_safe=True)
    assert not host._ve_prewarm_admission_available()
    assert not MainWindow._calibration_admission_available(window)
    detail.update(sample_rate=48000, ve_range_index=1)
    assert host._ve_prewarm_admission_available()
    assert MainWindow._calibration_admission_available(window)


def test_queue_calibration_uses_frozen_config_and_retains_coefficients(setup):
    detail = {"sample_rate": 96000, "ve_range_index": 5}
    widget = setup.make(ve_queue_config_provider=lambda: detail)
    seed_old(setup, widget)
    before = setup.profiles.path.read_bytes()
    assert widget.saved_v2pa_factors[7] == 73
    session, audio = offered(widget)
    assert session.request.sample_rate == 96000
    assert session.request.device["input_config"]["range_max"] == .1
    assert session.request.target_samples == 960000
    detail.update(sample_rate=48000, ve_range_index=1)
    widget._on_calibration_result_ready(session, audio)
    setup.bridge.service.accept_result.assert_called_once()
    accept(widget, session, audio)
    release(widget, session)
    assert setup.calibrations.get_record(session.request.device, 7)["calibration_sample_rate"] == 96000
    assert setup.profiles.path.read_bytes() == before


@pytest.mark.parametrize("failure", ["invalid", "profile_io"])
def test_admission_configuration_failures_are_diagnostic_and_repairable(host_factory, failure):
    host = host_factory()
    host.ve_prewarm_lifetime = VePrewarmLifetime()
    window = _WindowHarness(lifetime=host.ve_prewarm_lifetime)
    window.sequence_window = host
    window.mic, window.mic_channels = host.mic, host.mic_channels
    detail = host.sequence_config[0]["seq1"]["acq"]["detail"]
    if failure == "invalid":
        detail["sample_rate"] = True
        diagnostic = "sample_rate"
    else:
        host.ve_profile_store.load = mock.Mock(side_effect=OSError("profile denied"))
        diagnostic = "profile denied"
    assert not host._ve_prewarm_admission_available()
    assert diagnostic in host._ve_recording_config_error
    assert not MainWindow._calibration_admission_available(window)
    assert diagnostic in window._status.messages[-1]
    assert window._try_start_ve_prewarm(host.mic, host.mic_channels, "startup") == "invalid"
    assert not window.recording_bridge.calls
    assert host.ve_prewarm_lifetime.snapshot().state == "available"
    detail.update(sample_rate=48000, ve_range_index=1)
    assert host._ve_prewarm_admission_available()
    assert host._ve_recording_config_error is None
    assert MainWindow._calibration_admission_available(window)


def test_missing_queue_rate_uses_profile_and_missing_profile_defaults_without_writes(host_factory):
    host = host_factory(96000)
    assert host.reset_work_pram("not_labeled")[1] == 96000
    host.ve_profile_store.path.unlink()
    recorded, rate = host.reset_work_pram("not_labeled")
    assert rate == 51200
    assert recorded["device"]["input_config"]["range_max"] == 10
    assert not host.ve_profile_store.path.exists()


def test_queue_files_reload_independently_and_shared_references_see_same_edit(host_factory, tmp_path):
    from ui.sequence.sequence_widget_config_ops import SequenceWidgetConfigOpsMixin

    host = host_factory()
    paths = [tmp_path / "queue-a.json", tmp_path / "queue-b.json"]
    for path, rate, index in ((paths[0], 48000, 1), (paths[1], 96000, 5)):
        queue = deepcopy(host.sequence_config)
        queue[0]["seq1"]["acq"]["detail"].update(sample_rate=rate, ve_range_index=index)
        path.write_text(json.dumps(queue), encoding="utf-8")
    before = [path.read_bytes() for path in paths]
    # Reconstruct a fresh workflow host as on restart, and use the production
    # queue loader used by manual and serial condition switching.
    for host in (host, host_factory()):
        host._is_sequence_config_path = MethodType(SequenceWidgetConfigOpsMixin._is_sequence_config_path, host)
        host.count_board = None
        host._set_sequence_config_available_state = mock.Mock()
        host.init_fft_and_stft_flag = mock.Mock()
        for path, rate, limit in ((paths[0], 48000, 5), (paths[1], 96000, .1), (paths[0], 48000, 5)):
            assert SequenceWidgetConfigOpsMixin._apply_sequence_config_from_path(host, str(path))[0]
            recorded, actual_rate = host.reset_work_pram("not_labeled")
            assert actual_rate == rate
            assert recorded["device"]["input_config"]["range_max"] == limit
    assert [path.read_bytes() for path in paths] == before
    changed = json.loads(paths[0].read_text(encoding="utf-8"))
    changed[0]["seq1"]["acq"]["detail"].update(sample_rate=32000, ve_range_index=4)
    paths[0].write_text(json.dumps(changed), encoding="utf-8")
    # Both references load the same changed queue; no condition overrides exist.
    for _condition in ("A", "A alias"):
        assert SequenceWidgetConfigOpsMixin._apply_sequence_config_from_path(host, str(paths[0]))[0]
        recorded, actual_rate = host.reset_work_pram("not_labeled")
        assert actual_rate == 32000
        assert recorded["device"]["input_config"]["range_max"] == .5
    assert paths[1].read_bytes() == before[1]
    del changed[0]["seq1"]["acq"]["detail"]["sample_rate"]
    paths[0].write_text(json.dumps(changed), encoding="utf-8")
    assert SequenceWidgetConfigOpsMixin._apply_sequence_config_from_path(host, str(paths[0]))[0]
    assert host.reset_work_pram("not_labeled")[1] == 51200


def test_record_button_shows_config_error_then_clears_it_after_repair(host_factory):
    from PyQt5.QtWidgets import QPushButton
    from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin

    host = host_factory()
    host.toolsbar = SimpleNamespace(player_btn=QPushButton())
    host.ve_prewarm_lifetime = VePrewarmLifetime()
    detail = host.sequence_config[0]["seq1"]["acq"]["detail"]
    detail["sample_rate"] = True
    try:
        SequenceWidgetUiOpsMixin.update_player_btn_is_paused(host)
        assert not host.player_btn.isEnabled()
        assert "sample_rate" in host.player_btn.toolTip()
        detail["sample_rate"] = 48000
        SequenceWidgetUiOpsMixin.update_player_btn_is_paused(host)
        assert host.player_btn.isEnabled()
        assert "sample_rate" not in host.player_btn.toolTip()
    finally:
        host.player_btn.close()


def test_late_prewarm_completion_reports_invalid_current_queue_without_losing_owner(host_factory):
    host = host_factory()
    detail = host.sequence_config[0]["seq1"]["acq"]["detail"]
    detail["sample_rate"] = 48000
    window = _WindowHarness()
    window.sequence_window = host
    window.mic, window.mic_channels = host.mic, host.mic_channels
    assert window._try_start_ve_prewarm(host.mic, host.mic_channels, "startup") == "accepted"
    request = window.recording_bridge.calls[0]
    detail["sample_rate"] = None
    window.recording_bridge.callback(_completion(request))
    assert window.ve_prewarm_lifetime.snapshot().state == "succeeded"
    assert "sample_rate" in window._status.messages[-1]


@pytest.mark.parametrize("rate", [96000, None, 7999])
def test_startup_discovery_uses_explicit_queue_without_reading_broken_profile(host_factory, rate):
    host = host_factory()
    host.sequence_config[0]["seq1"]["acq"]["detail"].update(sample_rate=rate, ve_range_index=5)
    host.ve_profile_store.load = mock.Mock(side_effect=OSError("unneeded broken profile"))
    window = _WindowHarness()
    window.sequence_window = host
    window.ve_profile_store, window.ve_calibration_store = host.ve_profile_store, host.ve_calibration_store
    window.mic = device_info(available=False, input_config=None)
    window.mic_channels = host.mic_channels
    window._on_ve_discovery_result(_discovery(device_info()))
    if rate == 96000:
        assert window.mic["available"]
        assert window.recording_bridge.calls[0].sample_rate == 96000
        assert window.recording_bridge.calls[0].device["input_config"]["range_max"] == .1
    else:
        assert window.mic["available"]
        assert "sample_rate" in window._status.messages[-1]
        assert not window.recording_bridge.calls
    host.ve_profile_store.load.assert_not_called()


@pytest.mark.parametrize("failure", ["invalid_queue", "profile_io", "offline"])
def test_discovery_configuration_failure_recovers_on_valid_queue_without_rediscovery(host_factory, failure):
    host = host_factory()
    detail = host.sequence_config[0]["seq1"]["acq"]["detail"]
    if failure == "invalid_queue":
        detail["sample_rate"] = None
    else:
        host.ve_profile_store.load = mock.Mock(side_effect=OSError("profile denied"))
    window = _WindowHarness()
    window.sequence_window = host
    host.ve_prewarm_lifetime = window.ve_prewarm_lifetime
    window.ve_profile_store, window.ve_calibration_store = host.ve_profile_store, host.ve_calibration_store
    window.mic = device_info(available=False, input_config=None)
    window.mic_channels = host.mic_channels
    window._on_ve_discovery_result(_discovery(*(() if failure == "offline" else (device_info(),))))
    assert not window.recording_bridge.calls
    assert not host._ve_prewarm_admission_available()
    if failure != "offline":
        assert window.mic["available"]
        assert "sample_rate" in window._status.messages[-1] or "profile denied" in window._status.messages[-1]
    detail.update(sample_rate=96000, ve_range_index=5)
    result = window._try_start_ve_prewarm(window.mic, window.mic_channels, "repaired queue")
    if failure == "offline":
        assert result == "invalid" and not window.recording_bridge.calls
    else:
        assert result == "accepted"
        request = window.recording_bridge.calls[0]
        assert request.sample_rate == 96000 and request.device["input_config"]["range_max"] == .1


@pytest.mark.parametrize("entry", ["manual", "serial"])
@pytest.mark.parametrize("previous", ["invalid", "profile_io", "failed_prewarm"])
def test_condition_entry_loads_target_before_configuration_admission(condition_host, entry, previous):
    host = condition_host
    set_previous_queue_failure(host, previous)
    assert not host._can_start_recording_workflow()
    enter_condition(host, entry)
    host._load_sequence_config_for_product_condition.assert_called_once()
    request = host.recording_bridge.requests[0]
    assert request.sample_rate == 96000 and request.device["input_config"]["range_max"] == .1


@pytest.mark.parametrize("entry", ["manual", "serial"])
def test_invalid_target_is_rejected_after_load_before_round_mutation(condition_host, entry):
    host = condition_host
    queue = json.loads(host.target_queue_path.read_text(encoding="utf-8"))
    queue[0]["seq1"]["acq"]["detail"]["sample_rate"] = None
    host.target_queue_path.write_text(json.dumps(queue), encoding="utf-8")
    enter_condition(host, entry)
    host._load_sequence_config_for_product_condition.assert_called_once()
    assert not host.recording_bridge.requests
    assert not getattr(host, "_manual_product_condition_group_id", "")
    assert not host._get_active_product_condition_key()
    assert "sample_rate" in host._ve_recording_config_error
    from ui.sequence import sequence_widget_analysis_ops as analysis
    assert "sample_rate" in analysis.QMessageBox.warning.call_args.args[-1]


@pytest.mark.parametrize("entry", ["manual", "serial", "direct"])
@pytest.mark.parametrize("busy", ["service", "pending_prewarm", "recording"])
def test_busy_condition_entry_cannot_load_or_mutate_target(condition_host, entry, busy):
    host = condition_host
    set_previous_queue_failure(host, "invalid")
    if busy == "service":
        host.recording_bridge.service.busy = True
    elif busy == "recording":
        host._record_workflow_busy = True
    else:
        signature = ve_acquisition_signature(host.mic, host.mic_channels, 51200)
        host.ve_prewarm_lifetime.claim("pending", signature)
    host._cancel_pending_serial_trigger_delay = mock.Mock()
    before = deepcopy(host.sequence_config)
    if entry == "manual":
        host.on_clicked_player_btn()
    elif entry == "serial":
        host.product_test_condition_configs[0]["trigger_state"] = "01 02"
        host.product_test_condition_configs[1]["trigger_state"] = "03 04"
        host.on_serial_full_frame_received({"raw_hex": "03 04"})
    else:
        host.start_this_play()
    assert host.sequence_config == before
    assert host._manual_product_condition_index == 1
    host._load_sequence_config_for_product_condition.assert_not_called()
    host._cancel_pending_serial_trigger_delay.assert_not_called()
    assert not host.recording_bridge.requests
    assert not getattr(host, "_manual_product_condition_group_id", "")

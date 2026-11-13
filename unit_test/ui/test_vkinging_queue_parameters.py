"""Active queues own the frozen VK parameters at every runtime boundary."""
from copy import deepcopy
from unittest import mock

import pytest

from base.vkinging_input import ve_acquisition_signature
from main_window import MainWindow
from unit_test.base.vkinging_fakes import device_info, input_config
from ui.vkinging_presentation import ve_failure_text
from ui.sequence.workbench_editor_context import WorkbenchEditorContext, resolve_editor_ve_device
from unit_test.ui.vkinging_editor_fixtures import editor_factory as host_factory
from unit_test.ui.test_vkinging_prewarm_trigger import _WindowHarness, _discovery
from unit_test.ui.test_vkinging_calibration import setup, seed_old, offered, accept, release


@pytest.mark.parametrize("exception_type", [ValueError, OSError])
@pytest.mark.parametrize("detail_has_identity", [False, True])
def test_profile_error_is_private_and_blocks_dialog_until_repaired(
        ui_qapp, monkeypatch, caplog, exception_type, detail_has_identity):
    from ui import acquisition_config_window as config_ui

    device = device_info(machine_id="private-config-machine-id")
    before = deepcopy(device)
    detail = ("profile denied private-config-machine-id other-config-machine-id"
              if detail_has_identity else "profile denied")
    provider = mock.Mock(side_effect=exception_type(detail))
    warnings = mock.Mock()
    monkeypatch.setattr(config_ui.QMessageBox, "warning", warnings)
    dialog = config_ui.RecordConfigWindow({}, mic=device, ve_profile_provider=provider)
    try:
        assert dialog._sample_rate_load_error == detail
        assert dialog.samplerate_combo.toolTip() == ve_failure_text("configuration")
        assert device["machine_id"] in caplog.text and detail in caplog.text
        dialog.on_click_ok_btn()
        assert warnings.call_args.args[2] == ve_failure_text("configuration")
        assert dialog.final_data is None
        dialog.samplerate_combo.setEditText("96000")
        assert dialog.samplerate_combo.toolTip() == ""
        dialog.on_click_ok_btn()
        assert dialog.final_data["sample_rate"] == 96000
        assert dialog.result() == dialog.Accepted
        assert warnings.call_count == 1
        assert device == before
    finally:
        dialog.close()


def test_device_resolver_copies_identity_and_replaces_stale_profile():
    from base.vkinging_recording_config import resolve_ve_recording_device
    device = device_info(input_config=None)
    before = deepcopy(device)
    resolved = resolve_ve_recording_device(device, {"sample_rate": 96000, "ve_range_index": 5})
    assert resolved["input_config"] == input_config(96000, range_min=-.1, range_max=.1)
    assert device == before
    with pytest.raises(ValueError, match="device"):
        resolve_ve_recording_device(None, {})


@pytest.mark.parametrize("key,bad", [("sample_rate", True), ("sample_rate", 48000.5),
                                     ("sample_rate", None), ("ve_range_index", 7)])
def test_invalid_explicit_queue_parameters_never_fall_back(host_factory, key, bad):
    host = host_factory()
    detail = host.queue_data[0]["seq1"]["acq"]["detail"]
    detail.update(sample_rate=48000, ve_range_index=1)
    detail[key] = bad
    host.ve_profile_store.load = mock.Mock(side_effect=AssertionError("unneeded profile read"))
    with pytest.raises(ValueError, match=key):
        resolve_editor_ve_device(host.queue_data, host.mic, host.ve_profile_store, host.ve_calibration_store)
    host.ve_profile_store.load.assert_not_called()


def test_explicit_queue_ignores_profile_io_but_missing_rate_propagates_it(host_factory):
    host = host_factory()
    detail = host.queue_data[0]["seq1"]["acq"]["detail"]
    detail.update(sample_rate=48000, ve_range_index=1)
    host.ve_profile_store.load = mock.Mock(side_effect=OSError("profile denied"))
    assert resolve_editor_ve_device(host.queue_data, host.mic, host.ve_profile_store, host.ve_calibration_store)["input_config"]["sample_rate"] == 48000
    host.ve_profile_store.load.assert_not_called()
    del detail["sample_rate"]
    with pytest.raises(OSError, match="profile denied"):
        resolve_editor_ve_device(host.queue_data, host.mic, host.ve_profile_store, host.ve_calibration_store)


def test_independent_calibration_uses_frozen_text_and_retains_coefficients(setup):
    widget = setup.make()
    seed_old(setup, widget)
    before = setup.profiles.path.read_bytes()
    assert widget.saved_v2pa_factors[7] == 73
    session, audio = offered(widget)
    assert session.request.sample_rate == 48000
    assert session.request.device["input_config"]["range_max"] == 2.5
    assert session.request.target_samples == 480000
    setup.write_settings(96000, .1)
    widget._on_calibration_result_ready(session, audio)
    setup.bridge.service.accept_result.assert_called_once()
    accept(widget, session, audio)
    release(widget, session)
    assert setup.calibrations.get_record(session.request.device, 7)["calibration_sample_rate"] == 48000
    assert setup.profiles.path.read_bytes() == before


def test_missing_queue_rate_uses_profile_and_missing_profile_defaults_without_writes(host_factory):
    host = host_factory(96000)
    assert resolve_editor_ve_device(host.queue_data, host.mic, host.ve_profile_store, host.ve_calibration_store)["input_config"]["sample_rate"] == 96000
    host.ve_profile_store.path.unlink()
    resolved = resolve_editor_ve_device(host.queue_data, host.mic, host.ve_profile_store, host.ve_calibration_store)
    assert resolved["input_config"]["sample_rate"] == 51200
    assert resolved["input_config"]["range_max"] == 10
    assert not host.ve_profile_store.path.exists()


@pytest.mark.parametrize("rate", [4000, 4001, 7999, 96000, None, 3999])
def test_startup_discovery_uses_explicit_queue_without_reading_broken_profile(host_factory, rate, caplog):
    host = host_factory()
    host.queue_data[0]["seq1"]["acq"]["detail"].update(sample_rate=rate, ve_range_index=5)
    host.ve_profile_store.load = mock.Mock(side_effect=OSError("unneeded broken profile"))
    window = _WindowHarness()
    window.sequence_window = host
    window.editor_context = WorkbenchEditorContext(None, host.queue_data, host.default_logger)
    window.ve_profile_store, window.ve_calibration_store = host.ve_profile_store, host.ve_calibration_store
    window.mic = device_info(available=False, input_config=None)
    window.mic_channels = host.mic_channels
    window._on_ve_discovery_result(_discovery(device_info()))
    if rate in (4000, 4001, 7999, 96000):
        assert window.mic["available"]
        assert window.recording_bridge.calls[0].sample_rate == rate
        assert window.recording_bridge.calls[0].device["input_config"]["range_max"] == .1
    else:
        assert window.mic["available"]
        assert window._status.messages[-1] == ve_failure_text("configuration")
        assert "sample_rate" in caplog.text and host.mic["machine_id"] in caplog.text
        assert not window.recording_bridge.calls
    host.ve_profile_store.load.assert_not_called()


def test_calibration_admission_never_resolves_queue_or_profile(host_factory):
    host = host_factory()
    window = _WindowHarness()
    window.mic, window.mic_channels = host.mic, host.mic_channels
    window._ve_signature = mock.Mock(side_effect=AssertionError("queue signature read"))
    assert MainWindow._calibration_admission_available(window)
    window._ve_signature.assert_not_called()
    signature = ve_acquisition_signature(host.mic, host.mic_channels,
                                        host.mic["input_config"]["sample_rate"])
    window.ve_prewarm_lifetime.claim("pending", signature)
    assert not MainWindow._calibration_admission_available(window)

import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

import ui.stimulus_window as module
from base.stimulus_resolver import _generate_stimulus_data
from base.soundcard_calibration_manager import SoundcardCalibrationManager
from ui.stimulus_window import StimulusWindow
from unit_test.ui.test_speaker_calibration_db_consumers import _registered_speaker_db


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def config(method="chirp", amplitude=0.5, voltage=1.0):
    info = dict(stimulus_method=method, stimulus_type="linear", start_freq=100,
                stop_freq=400, total_time=1.0, repeat_times=1, num_steps=3,
                sample_rate=48000, voltage=voltage, voltage_type="RMS",
                use_custom_stimulus=True)
    if amplitude is not None:
        info["amplitude"] = amplitude
    if method == "frequency_stepped":
        info.update(stimulus_type="custom_linear", frequency_mode="custom_linear",
                    min_duration=0.02, min_cycles=4.0)
    elif method == "noise":
        info["stimulus_type"] = "white_noise"
    return dict(stimulus_info=info, stimulus_signal_path="")


@pytest.fixture
def environment(qapp, tmp_path, monkeypatch):
    repo, speaker = _registered_speaker_db(tmp_path, monkeypatch)
    repo.update_output_amplitude_calibration(speaker["hardware_id"], [0.25, 0.0], max_voltage=2.0)
    speaker.update(index=7, samplerate=48000)
    calls, warnings, windows = [], [], []
    monkeypatch.setattr(module.SoundcardAudioProcessor, "sd_play", lambda self, params: calls.append(params) or (0, "ok"))
    monkeypatch.setattr(module.MessageBox, "warning", lambda *args: warnings.append(args[-1]))
    monkeypatch.setattr(module, "load_audio_simple", lambda *args: (np.zeros(32), None))
    monkeypatch.setattr(module, "save_audio_simple", lambda *args: None)
    monkeypatch.setattr(module.model_consts, "STORED_STIMULUS_PATH", str(tmp_path / "stored"))
    monkeypatch.setattr(module, "DEFAULT_DIR", str(tmp_path) + "/")
    monkeypatch.setattr(StimulusWindow, "set_ai_popup", lambda self: None)

    def create(payload=None, **kwargs):
        window = StimulusWindow(stimulus_config_data=payload or config(), speaker=speaker, **kwargs)
        windows.append(window)
        return window

    yield create, calls, warnings
    for window in windows:
        window.close()


@pytest.mark.parametrize("amplitude", [0.5, None])
def test_initial_preview_prepares_latest_amplitude_without_editing(environment, monkeypatch, amplitude):
    create, calls, _ = environment
    window = create(config(amplitude=amplitude))
    monkeypatch.setattr(SoundcardCalibrationManager, "load_output_amplitude_calibration",
                        lambda *args: pytest.fail("Preview must reuse the modal window preparation"))
    window.play_btn_clicked()
    window.play_btn_clicked()
    assert [call["amplitude"] for call in calls] == [0.25, 0.25]
    assert window.stimulus_info["voltage"] == 1.0


@pytest.mark.parametrize("method", ["chirp", "step", "noise", "frequency_stepped"])
@pytest.mark.parametrize("amplitude", [0.5, None])
def test_default_loader_and_import_support_missing_or_old_amplitude(environment, tmp_path, monkeypatch, method, amplitude):
    create, calls, _ = environment
    payload = config(method, amplitude)
    path = tmp_path / "ui" / "ui_config" / "default_stimulus.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(module, "DEFAULT_DIR", str(tmp_path) + "/")
    window = create()
    window.default_config_btn_clicked()
    window.play_btn_clicked()
    assert calls[-1]["amplitude"] == 0.25
    monkeypatch.setattr(module, "LoadStimulusDialog", lambda *args: SimpleNamespace(exec=lambda: deepcopy(payload["stimulus_info"])))
    window.load_config_btn_clicked()
    window.play_btn_clicked()
    assert calls[-1]["amplitude"] == 0.25
    window.ok_btn_clicked()
    assert window.final_save_data["stimulus_info"]["amplitude"] == 0.25
    assert window.final_save_data["stimulus_info"]["voltage"] == 1.0


@pytest.mark.parametrize("method", ["chirp", "step", "noise", "frequency_stepped"])
def test_waveform_generation_does_not_apply_output_gain(environment, method):
    create, _, _ = environment
    signals = []
    for amplitude in [0.5, None]:
        np.random.seed(42)
        window = create(config(method, amplitude))
        signals.append(window.stimulus_data.copy())
        if method == "frequency_stepped":
            assert np.max(np.abs(window.stimulus_data)) > 0.9
    np.testing.assert_array_equal(*signals)
    signals = []
    for amplitude in [0.5, None]:
        np.random.seed(42)
        data, _, info, _ = _generate_stimulus_data(config(method, amplitude), 48000)
        signals.append(data)
        assert info.get("amplitude") == amplitude
    np.testing.assert_array_equal(*signals)


def test_voltage_and_type_edits_refresh_preparation_without_changing_raw_waveform(environment, monkeypatch):
    create, calls, _ = environment
    window = create(config("frequency_stepped", None))
    original_data = window.stimulus_data.copy()
    resolutions = []
    resolve = module.resolve_output_amplitude

    def track(speaker, voltage, voltage_type):
        resolutions.append((voltage, voltage_type))
        return resolve(speaker, voltage, voltage_type)

    monkeypatch.setattr(module, "resolve_output_amplitude", track)
    window.voltage_spin_box.setValue(2.0)
    window.voltage_combo_box.setCurrentText("Peak")
    window.play_btn_clicked()
    window.ok_btn_clicked()
    assert resolutions == [(2.0, "RMS"), (2.0, "Peak")]
    assert calls[-1]["amplitude"] == 0.5
    np.testing.assert_array_equal(calls[-1]["data"], original_data)
    assert window.final_save_data["stimulus_info"]["amplitude"] == 0.5
    assert window.final_save_data["stimulus_info"]["voltage_type"] == "Peak"


@pytest.mark.parametrize("entry", ["default", "import"])
@pytest.mark.parametrize("voltage", [3.0, 2.004])
def test_loading_above_max_into_open_window_keeps_target_visible(environment, tmp_path, monkeypatch, entry, voltage):
    create, calls, warnings = environment
    window = create()
    payload = config("frequency_stepped", None, voltage)
    if entry == "default":
        path = tmp_path / "ui" / "ui_config" / "default_stimulus.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        window.default_config_btn_clicked()
    else:
        monkeypatch.setattr(module, "LoadStimulusDialog", lambda *args: SimpleNamespace(exec=lambda: payload["stimulus_info"]))
        window.load_config_btn_clicked()
    assert window.voltage_spin_box.value() == voltage
    assert window.stimulus_info["voltage"] == voltage
    window.play_btn_clicked()
    assert not calls
    assert f"{voltage:g} V" in warnings[-1]
    assert window.save_stimulus_to_json()["stimulus_info"]["voltage"] == voltage


@pytest.mark.parametrize("voltage", [3.0, 2.004])
def test_above_max_retains_target_rejects_preview_and_recovers(environment, voltage):
    create, calls, warnings = environment
    window = create(config(voltage=voltage))
    assert window.voltage_spin_box.value() == voltage
    assert window.stimulus_info["voltage"] == voltage
    assert window.stimulus_info.get("amplitude") is None
    window.play_btn_clicked()
    assert calls == []
    assert f"{voltage:g} V" in warnings[-1] and "2 V" in warnings[-1]
    assert "重新校准" in warnings[-1]
    saved = window.save_stimulus_to_json()["stimulus_info"]
    assert saved["voltage"] == voltage and saved["voltage_type"] == "RMS"
    assert saved.get("amplitude") is None
    window.ok_btn_clicked()
    assert window.final_save_data["stimulus_info"]["voltage"] == voltage
    window.voltage_spin_box.setValue(2.0)
    window.play_btn_clicked()
    assert calls[-1]["amplitude"] == 0.5


def test_restoring_legacy_external_branch_preserves_fractional_voltage(environment, tmp_path):
    create, calls, warnings = environment
    window = create()
    snapshot = window._stimulus_state_snapshot()
    snapshot["stimulus_info"].update(voltage=2.004, use_custom_stimulus=False)
    snapshot["load_wav_path"] = snapshot["load_stimulus_signal_path"] = str(tmp_path / "external.wav")
    window._pre_step_sc_legacy_branch_snapshot = snapshot
    assert window._restore_pre_step_sc_legacy_external_branch()
    assert window.voltage_spin_box.value() == 2.004
    window.play_btn_clicked()
    assert not calls
    assert "2.004 V" in warnings[-1]
    assert window.save_stimulus_to_json()["stimulus_info"]["voltage"] == 2.004


@pytest.mark.parametrize("voltage", [1.0, 3.0])
def test_offline_reference_actual_preview_uses_device_preparation(environment, voltage):
    create, calls, warnings = environment
    window = create(config("frequency_stepped", None, voltage), offline_reference_authoring=True)
    window.play_btn_clicked()
    if voltage == 1.0:
        assert calls[-1]["amplitude"] == 0.25
    else:
        assert not calls
        assert "3 V" in warnings[-1] and "2 V" in warnings[-1]


def test_pure_offline_generation_and_save_need_no_device(qapp, monkeypatch):
    monkeypatch.setattr(module, "save_audio_simple", lambda *args: None)
    window = StimulusWindow(config("frequency_stepped", None), speaker=None, offline_reference_authoring=True)
    assert np.max(np.abs(window.stimulus_data)) > 0.9
    assert window.save_stimulus_to_json()["stimulus_info"]["voltage"] == 1.0
    window.close()
    assert "amplitude" not in StimulusWindow._fallback_stimulus_info()

import json

from base.hardware_management import HardwareManagementRepository
from consts import model_consts
from ui.load_stimulus_dialog import LoadStimulusDialog
from ui.stimulus_window import StimulusWindow
from unit_test.base.test_hardware_management import create_system_db, runtime_device


def _registered_speaker_db(tmp_path, monkeypatch):
    db_path = tmp_path / "system_data.db"
    create_system_db(db_path)
    monkeypatch.setattr(model_consts, "SYSTEM_DATABASE_PATH", str(db_path))
    repo = HardwareManagementRepository(str(db_path))
    asset = repo.register_asset(runtime_device(inputs=0, outputs=1), "Windows WASAPI", "Speaker", 48000)
    return repo, asset


def _stimulus_window_for_speaker(hardware_id):
    window = StimulusWindow.__new__(StimulusWindow)
    window.speaker = {"hardware_id": hardware_id}
    window.is_close_window = False
    return window


def _select_speaker(monkeypatch, hardware_id):
    payload = {"speaker": {"hardware_id": hardware_id}, "mic": {"hardware_id": "mic-1"}, "mic_channels": [0]}
    monkeypatch.setattr(
        "base.soundcard_calibration_manager.SoundDeviceManager.load_selected_devices",
        staticmethod(lambda: payload),
    )


def test_stimulus_prediction_reads_database_and_ignores_legacy_file(tmp_path, monkeypatch):
    repo, asset = _registered_speaker_db(tmp_path, monkeypatch)
    repo.update_output_amplitude_calibration(asset["hardware_id"], [0.5, 0.0], max_voltage=1.0)
    _select_speaker(monkeypatch, asset["hardware_id"])
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    monkeypatch.setattr(model_consts, "JSON_DIR_PATH", str(legacy_dir))
    (legacy_dir / "calibration_coefficients.json").write_text(
        json.dumps({"calibration_coefficients": [99.0, 0.0], "max_voltage": 99.0}),
        encoding="utf-8",
    )

    window = _stimulus_window_for_speaker(asset["hardware_id"])

    assert window.get_predict_amplitude(0.5) == 0.25


def test_stimulus_prediction_uses_window_speaker_over_global_selected_speaker(tmp_path, monkeypatch):
    repo, window_asset = _registered_speaker_db(tmp_path, monkeypatch)
    global_asset = repo.register_asset(
        runtime_device(name="Global Speaker", inputs=0, outputs=1, index=8),
        "Windows WASAPI",
        "Global Speaker",
        48000,
    )
    repo.update_output_amplitude_calibration(window_asset["hardware_id"], [0.5, 0.0], max_voltage=1.0)
    repo.update_output_amplitude_calibration(global_asset["hardware_id"], [9.0, 0.0], max_voltage=1.0)
    _select_speaker(monkeypatch, global_asset["hardware_id"])
    window = _stimulus_window_for_speaker(window_asset["hardware_id"])

    assert window.get_predict_amplitude(0.5) == 0.25


def test_stimulus_prediction_returns_zero_when_missing_or_unregistered(tmp_path, monkeypatch):
    repo, asset = _registered_speaker_db(tmp_path, monkeypatch)
    _select_speaker(monkeypatch, asset["hardware_id"])
    assert _stimulus_window_for_speaker(asset["hardware_id"]).get_predict_amplitude(0.5) == 0.0

    _select_speaker(monkeypatch, "")
    assert _stimulus_window_for_speaker("").get_predict_amplitude(0.5) == 0.0


def test_stimulus_max_input_voltage_reads_database(tmp_path, monkeypatch):
    repo, asset = _registered_speaker_db(tmp_path, monkeypatch)
    repo.update_output_amplitude_calibration(asset["hardware_id"], [0.5, 0.0], max_voltage=1.25)
    _select_speaker(monkeypatch, asset["hardware_id"])
    window = _stimulus_window_for_speaker(asset["hardware_id"])

    assert window.get_max_input_voltage() == 1.25
    assert window.is_close_window is False


def test_stimulus_max_input_voltage_blocks_without_registered_speaker(tmp_path, monkeypatch):
    _registered_speaker_db(tmp_path, monkeypatch)
    _select_speaker(monkeypatch, "")
    window = _stimulus_window_for_speaker("")

    assert window.get_max_input_voltage() == 0.0
    assert window.is_close_window is True


def test_load_stimulus_dialog_max_voltage_reads_database(tmp_path, monkeypatch):
    repo, asset = _registered_speaker_db(tmp_path, monkeypatch)
    repo.update_output_amplitude_calibration(asset["hardware_id"], [0.5, 0.0], max_voltage=1.25)
    _select_speaker(monkeypatch, asset["hardware_id"])
    dialog = LoadStimulusDialog.__new__(LoadStimulusDialog)

    assert dialog._get_max_input_voltage() == 1.25


def test_load_stimulus_dialog_max_voltage_returns_none_without_registered_speaker(tmp_path, monkeypatch):
    _registered_speaker_db(tmp_path, monkeypatch)
    _select_speaker(monkeypatch, "")
    dialog = LoadStimulusDialog.__new__(LoadStimulusDialog)

    assert dialog._get_max_input_voltage() is None

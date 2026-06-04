import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

import ui.load_stimulus_dialog as load_dialog_module
from consts import error_code
from ui.load_stimulus_dialog import LoadStimulusDialog


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _valid_step_sc_row():
    payload = {
        "stimulus_method": "frequency_stepped",
        "stimulus_type": "custom_linear",
        "frequency_mode": "custom_linear",
        "repeat_times": 2,
        "start_freq": 1000,
        "stop_freq": 2000,
        "sample_rate": 48000,
        "total_time": 0.1,
        "num_steps": 2,
        "voltage_type": "RMS",
        "voltage": 1.0,
        "frequencies": [1000.0, 2000.0],
        "min_duration": 0.01,
        "min_cycles": 4.0,
    }
    return {
        "stimulus_id": "step-id",
        "stimulus_method": "frequency_stepped",
        "stimulus_type": "custom_linear",
        "repeat_times": 2,
        "start_freq": 1000,
        "stop_freq": 2000,
        "sample_rate": 48000,
        "total_time": 0.1,
        "num_steps": 2,
        "voltage_type": "RMS",
        "voltage": 1.0,
        "is_default": 1,
        "stimulus_name": "valid_step_sc",
        "stimulus_metadata_json": "{}",
        "step_sc_row_state": "valid",
        "stimulus_payload": {**payload, "stimulus_id": "step-id", "stimulus_name": "valid_step_sc", "is_default": 1},
    }


def test_load_dialog_returns_valid_frequency_stepped_payload_and_displays_row(qapp, monkeypatch):
    row = _valid_step_sc_row()
    monkeypatch.setattr(
        load_dialog_module.StimulusSignalManagement,
        "query_all_stimulus_info",
        staticmethod(lambda: (error_code.OK, [row])),
    )

    dialog = LoadStimulusDialog(_Logger())
    dialog.model().item(0, 0).setCheckState(Qt.Checked)

    selected = dialog.get_selected_stimulus_config()

    assert dialog.loaded_stimulus[0]["step_sc_row_state"] == "valid"
    assert selected["stimulus_method"] == "frequency_stepped"
    assert selected["frequencies"] == [1000.0, 2000.0]
    assert selected["min_duration"] == pytest.approx(0.01)
    assert "stimulus_id" not in selected
    assert "stimulus_name" not in selected
    assert "is_default" not in selected


def test_load_dialog_preserves_dict_row_zero_voltage_for_legacy_scalar_stimulus(qapp, monkeypatch):
    row = {
        "stimulus_id": "chirp-id",
        "stimulus_method": "chirp",
        "stimulus_type": "linear",
        "repeat_times": 1,
        "start_freq": 100,
        "stop_freq": 1000,
        "sample_rate": 48000,
        "total_time": 1.0,
        "num_steps": 1,
        "voltage_type": "RMS",
        "voltage": 0.0,
        "is_default": 0,
        "stimulus_name": "zero_voltage_chirp",
        "stimulus_metadata_json": None,
    }
    monkeypatch.setattr(
        load_dialog_module.StimulusSignalManagement,
        "query_all_stimulus_info",
        staticmethod(lambda: (error_code.OK, [row])),
    )

    dialog = LoadStimulusDialog(_Logger())
    dialog.model().item(0, 0).setCheckState(Qt.Checked)

    selected = dialog.get_selected_stimulus_config()

    assert dialog.loaded_stimulus[0]["voltage"] == pytest.approx(0.0)
    assert dialog.model().data(dialog.model().index(0, 11)) == "0.00"
    assert selected["voltage"] == pytest.approx(0.0)


def test_load_dialog_keeps_invalid_frequency_stepped_visible_but_refuses_selection(qapp, monkeypatch):
    warnings = []
    invalid_row = {
        "stimulus_id": "bad-id",
        "stimulus_method": "frequency_stepped",
        "stimulus_type": "custom_linear",
        "repeat_times": 1,
        "start_freq": 1000,
        "stop_freq": 1000,
        "sample_rate": 48000,
        "total_time": 1,
        "num_steps": 1,
        "voltage_type": "RMS",
        "voltage": 1.0,
        "is_default": 0,
        "stimulus_name": "invalid_step_sc",
        "stimulus_metadata_json": "{bad-json",
        "step_sc_row_state": "invalid_metadata",
    }
    monkeypatch.setattr(
        load_dialog_module.StimulusSignalManagement,
        "query_all_stimulus_info",
        staticmethod(lambda: (error_code.OK, [invalid_row])),
    )
    monkeypatch.setattr(
        load_dialog_module.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )

    dialog = LoadStimulusDialog(_Logger())
    assert dialog.model().rowCount() == 1
    assert dialog.loaded_stimulus[0]["step_sc_row_state"] == "invalid_metadata"
    dialog.model().item(0, 0).setCheckState(Qt.Checked)

    assert dialog.get_selected_stimulus_config() is None
    assert warnings


def test_load_dialog_blocks_scalar_edits_for_frequency_stepped_but_keeps_name_edit(qapp, monkeypatch):
    warnings = []
    row = _valid_step_sc_row()
    updates = []

    class FakeManagement:
        @staticmethod
        def query_all_stimulus_info():
            return error_code.OK, [row]

        @staticmethod
        def update_stimulus_info_to_db(update_info):
            updates.append(("rename", update_info))
            return error_code.OK, "renamed"

        @staticmethod
        def update_stimulus_params_to_db(stimulus_id, update_params):
            updates.append(("params", stimulus_id, update_params))
            return error_code.OK, "updated"

    monkeypatch.setattr(load_dialog_module, "StimulusSignalManagement", FakeManagement)
    monkeypatch.setattr(
        load_dialog_module.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )

    dialog = LoadStimulusDialog(_Logger())
    start_index = dialog.model().index(0, 5)
    dialog.model().setData(start_index, "1200")
    dialog.on_data_changed(start_index)

    assert warnings
    assert all(update[0] != "params" for update in updates)
    assert dialog.loaded_stimulus[0]["start_freq"] == 1000

    name_index = dialog.model().index(0, 1)
    dialog.model().setData(name_index, "renamed_step_sc")
    dialog.on_data_changed(name_index)

    assert updates[-1] == ("rename", {"stimulus_id": "step-id", "new_name": "renamed_step_sc"})
    assert dialog.loaded_stimulus[0]["stimulus_name"] == "renamed_step_sc"

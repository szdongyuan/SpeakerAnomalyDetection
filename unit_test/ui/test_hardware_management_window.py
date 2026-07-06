import os
import sqlite3
import sys
import importlib
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QComboBox, QDialog, QSpinBox, QStyledItemDelegate


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


class FakeRepository:
    def __init__(
        self,
        assets=None,
        tables_exist=True,
        tables_error=None,
        list_error=None,
        update_error=None,
        register_error=None,
        delete_error=None,
        delete_result=True,
        get_asset_error=None,
        get_asset_missing=False,
    ):
        self._assets = list(assets or [])
        self._tables_exist = tables_exist
        self.tables_error = tables_error
        self.list_error = list_error
        self.update_error = update_error
        self.register_error = register_error
        self.delete_error = delete_error
        self.delete_result = delete_result
        self.get_asset_error = get_asset_error
        self.get_asset_missing = get_asset_missing
        self.updated_fields = []
        self.registered = []
        self.deleted = []
        self.list_count = 0
        self.get_asset_count = 0

    def tables_exist(self):
        if self.tables_error:
            raise self.tables_error
        return self._tables_exist

    def list_assets(self):
        self.list_count += 1
        if self.list_error:
            raise self.list_error
        return [dict(asset) for asset in self._assets]

    def update_asset_fields(self, hardware_id, fields):
        self.updated_fields.append((hardware_id, dict(fields)))
        if self.update_error:
            raise self.update_error
        for asset in self._assets:
            if asset["hardware_id"] == hardware_id:
                asset.update(fields)
                asset["updated_at"] = "2026-06-23 10:20:30"
                return True
        return False

    def get_asset(self, hardware_id):
        self.get_asset_count += 1
        if self.get_asset_error:
            raise self.get_asset_error
        if self.get_asset_missing:
            return None
        for asset in self._assets:
            if asset["hardware_id"] == hardware_id:
                return dict(asset)
        return None

    def register_asset(self, runtime_device, hostapi_name, display_name, samplerate, bit_depth, latency_ms):
        self.registered.append(
            (runtime_device, hostapi_name, display_name, samplerate, bit_depth, latency_ms)
        )
        if self.register_error:
            raise self.register_error
        return {"hardware_id": "new-id"}

    def delete_asset(self, hardware_id):
        self.deleted.append(hardware_id)
        if self.delete_error:
            raise self.delete_error
        if not self.delete_result:
            return False
        self._assets = [asset for asset in self._assets if asset["hardware_id"] != hardware_id]
        return True


class FakeSoundDeviceManager:
    @staticmethod
    def get_api_info(api_index=None):
        apis = [{"name": "API-A"}, {"name": "API-B"}]
        if api_index is None:
            return apis
        return apis[api_index]

    @staticmethod
    def get_device_info():
        return {
            "API-A": {
                "input": [
                    {
                        "index": 1,
                        "name": "Mic 441",
                        "hostapi": 0,
                        "default_samplerate": 44100.0,
                        "max_input_channels": 2,
                        "max_output_channels": 0,
                    }
                ],
                "output": [
                    {
                        "index": 2,
                        "name": "Speaker 96",
                        "hostapi": 0,
                        "default_samplerate": 96000.0,
                        "max_input_channels": 0,
                        "max_output_channels": 2,
                    }
                ],
            },
            "API-B": {
                "input": [
                    {
                        "index": 3,
                        "name": "Interface 48",
                        "hostapi": 1,
                        "default_samplerate": 48000.0,
                        "max_input_channels": 4,
                        "max_output_channels": 4,
                    }
                ],
                "output": [],
            },
        }


class FakeSoundDeviceManagerWithOther(FakeSoundDeviceManager):
    @staticmethod
    def get_api_info(api_index=None):
        apis = [{"name": "API-A", "devices": [1, 2, 4]}, {"name": "API-B", "devices": [3]}]
        if api_index is None:
            return apis
        return apis[api_index]


class FakeSoundDeviceManagerEnumerationFailure:
    @staticmethod
    def get_api_info(api_index=None):
        raise RuntimeError("enumeration failed")

    @staticmethod
    def get_device_info():
        raise RuntimeError("device enumeration failed")


class FakeSoundDeviceQuery:
    @staticmethod
    def query_devices():
        return [
            {
                "index": 0,
                "name": "Unused",
                "hostapi": 0,
                "default_samplerate": 48000.0,
                "max_input_channels": 1,
                "max_output_channels": 0,
            },
            {
                "index": 1,
                "name": "Mic 441",
                "hostapi": 0,
                "default_samplerate": 44100.0,
                "max_input_channels": 2,
                "max_output_channels": 0,
            },
            {
                "index": 2,
                "name": "Speaker 96",
                "hostapi": 0,
                "default_samplerate": 96000.0,
                "max_input_channels": 0,
                "max_output_channels": 2,
            },
            {
                "index": 3,
                "name": "Interface 48",
                "hostapi": 1,
                "default_samplerate": 48000.0,
                "max_input_channels": 4,
                "max_output_channels": 4,
            },
            {
                "index": 4,
                "name": "Control Surface",
                "hostapi": 0,
                "default_samplerate": 32000.0,
                "max_input_channels": 0,
                "max_output_channels": 0,
            },
        ]


def sample_asset():
    return {
        "hardware_id": "hw-1",
        "hardware_type": "microphone",
        "display_name": "Old name",
        "device_name": "Runtime mic",
        "hostapi_name": "API-A",
        "samplerate": 44100,
        "bit_depth": 32,
        "latency_ms": 100,
        "max_input_channels": 2,
        "max_output_channels": 0,
        "updated_at": "2026-06-23 09:00:00",
    }


def test_open_hardware_management_window_warns_when_tables_are_missing(qapp, monkeypatch):
    import ui.hardware_management_window as module

    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))

    result = module.open_hardware_management_window(parent=None, repository=FakeRepository(tables_exist=False))

    assert result is None
    assert warnings
    assert warnings[-1][2] == "硬件管理表不存在，请使用最新版数据库"


def test_hardware_management_table_model_owns_column_configuration(qapp):
    import ui.hardware_management_window as module

    assert not hasattr(module, "READ_ONLY_COLUMNS")
    assert not hasattr(module, "EDITABLE_COLUMNS")
    assert not hasattr(module, "COLUMNS")
    assert not hasattr(module, "HEADER_LABELS")

    model = module.HardwareManagementTableModel(FakeRepository([sample_asset()]))

    assert model.columns is module.HardwareManagementTableModel.COLUMNS
    assert model.columns[:4] == ("display_name", "hostapi_name", "device_name", "hardware_type")
    assert module.HardwareManagementTableModel.HEADER_LABELS["hardware_id"] == "硬件ID"


def test_hardware_management_window_no_longer_exports_removed_exception_groups():
    import ui.hardware_management_window as module

    for name in ("REPOSITORY_FAILURE_EXCEPTIONS", "PORT_AUDIO_ERROR", "DEVICE_ENUMERATION_EXCEPTIONS"):
        assert not hasattr(module, name)


def test_hardware_management_window_uses_audio_constant_bit_depths():
    import ui.hardware_management_window as module
    from consts.audio_consts import VALID_BIT_DEPTHS

    assert module.VALID_BIT_DEPTHS is VALID_BIT_DEPTHS


def test_open_hardware_management_window_warns_when_table_probe_fails(qapp, monkeypatch):
    import ui.hardware_management_window as module

    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))

    result = module.open_hardware_management_window(
        parent=None,
        repository=FakeRepository(tables_error=sqlite3.Error("probe failed")),
    )

    assert result is None
    assert warnings
    assert warnings[-1][2] == "probe failed"


def test_open_hardware_management_window_warns_when_initial_read_fails(qapp, monkeypatch):
    import ui.hardware_management_window as module

    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))

    result = module.open_hardware_management_window(
        parent=None,
        repository=FakeRepository(tables_exist=True, list_error=sqlite3.Error("read failed")),
    )

    assert result is None
    assert warnings
    assert warnings[-1][2] == "read failed"


def test_open_hardware_management_window_wires_parent_update_callback(qapp, monkeypatch):
    import ui.hardware_management_window as module

    executed = []
    monkeypatch.setattr(module.QDialog, "exec_", lambda self: executed.append(self))
    parent = module.QDialog()
    parent.mic = {"hardware_id": "hw-1"}
    parent.speaker = {"hardware_id": "speaker-1"}
    parent.on_registered_audio_hardware_updated = lambda _hardware_id, _asset: None

    window = module.open_hardware_management_window(
        parent=parent,
        repository=FakeRepository([sample_asset()]),
    )

    assert executed == [window]
    assert window.on_selected_devices_updated is parent.on_registered_audio_hardware_updated


def test_hardware_management_window_constructor_warns_when_initial_read_fails(qapp, monkeypatch):
    import ui.hardware_management_window as module

    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))

    window = module.HardwareManagementWindow(repository=FakeRepository(list_error=module.HardwareManagementError("read failed")))

    assert window.model.rowCount() == 0
    assert warnings
    assert warnings[-1][2] == "read failed"


def test_registration_dialog_uses_api_first_device_defaults(qapp, monkeypatch):
    import ui.hardware_management_window as module

    monkeypatch.setattr(module, "SoundDeviceManager", FakeSoundDeviceManager)
    monkeypatch.setattr(module, "sd", FakeSoundDeviceQuery)
    dialog = module.HardwareRegistrationDialog(repository=FakeRepository())
    form = dialog.layout().itemAt(0).layout()
    labels = [form.itemAt(row, form.LabelRole).widget().text() for row in range(form.rowCount())]

    assert [dialog.api_combo.itemText(i) for i in range(dialog.api_combo.count())] == ["API-A", "API-B"]
    assert labels.index("显示名称") < labels.index("设备")
    assert "设备名称" in dialog.display_name_edit.placeholderText()
    assert dialog.device_combo.currentData() is None
    assert dialog.samplerate_combo.currentText() == ""
    assert [dialog.samplerate_combo.itemText(i) for i in range(dialog.samplerate_combo.count())] == [""]
    assert dialog.bit_depth_combo.currentText() == "32"
    assert dialog.latency_spin.value() == 100
    assert dialog.latency_spin.maximum() == 1000

    dialog.device_combo.setCurrentIndex(1)
    assert dialog.device_combo.currentData()["name"] == "Mic 441"
    assert [dialog.samplerate_combo.itemText(i) for i in range(dialog.samplerate_combo.count())] == ["44100", "48000"]
    assert dialog.samplerate_combo.currentText() == "44100"
    assert dialog.display_name_edit.text() == ""

    dialog.device_combo.setCurrentIndex(2)
    assert dialog.device_combo.currentData()["name"] == "Speaker 96"
    assert dialog.samplerate_combo.currentText() == "48000"
    assert dialog.display_name_edit.text() == ""

    dialog.display_name_edit.setText("Custom interface")
    dialog.api_combo.setCurrentText("API-B")
    dialog.device_combo.setCurrentIndex(1)
    assert dialog.device_combo.currentData()["name"] == "Interface 48"
    assert dialog.samplerate_combo.currentText() == "48000"
    assert dialog.display_name_edit.text() == "Custom interface"


def test_registration_dialog_refresh_reenumerates_post_startup_devices(qapp, monkeypatch):
    import ui.hardware_management_window as module

    refresh_calls = []
    enumeration_generation = {"value": 0}

    class RefreshingSoundDeviceManager(FakeSoundDeviceManagerWithOther):
        @staticmethod
        def refresh_available_device():
            refresh_calls.append(True)
            enumeration_generation["value"] = 1

        @staticmethod
        def get_api_info(api_index=None):
            if enumeration_generation["value"] == 0:
                apis = [{"name": "API-A", "devices": [1, 2]}]
            else:
                apis = [{"name": "API-A", "devices": [1, 2, 4]}]
            if api_index is None:
                return apis
            return apis[api_index]

    monkeypatch.setattr(module, "SoundDeviceManager", RefreshingSoundDeviceManager)
    monkeypatch.setattr(module, "sd", FakeSoundDeviceQuery)

    dialog = module.HardwareRegistrationDialog(repository=FakeRepository())

    assert dialog.refresh_btn.text() == "刷新"
    assert "Control Surface" not in [
        dialog.device_combo.itemText(index)
        for index in range(dialog.device_combo.count())
    ]

    dialog.refresh_devices()

    assert refresh_calls == [True]
    assert "Control Surface" in [
        dialog.device_combo.itemText(index)
        for index in range(dialog.device_combo.count())
    ]


def test_registration_dialog_refresh_failure_warns_and_preserves_current_combos(qapp, monkeypatch):
    import ui.hardware_management_window as module

    class FailingRefreshSoundDeviceManager(FakeSoundDeviceManager):
        @staticmethod
        def refresh_available_device():
            raise RuntimeError("refresh failed")

    monkeypatch.setattr(module, "SoundDeviceManager", FailingRefreshSoundDeviceManager)
    monkeypatch.setattr(module, "sd", FakeSoundDeviceQuery)
    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))

    dialog = module.HardwareRegistrationDialog(repository=FakeRepository())
    before_apis = [dialog.api_combo.itemText(i) for i in range(dialog.api_combo.count())]
    before_devices = [dialog.device_combo.itemText(i) for i in range(dialog.device_combo.count())]

    dialog.refresh_devices()

    assert warnings[-1][2] == "refresh failed"
    assert [dialog.api_combo.itemText(i) for i in range(dialog.api_combo.count())] == before_apis
    assert [dialog.device_combo.itemText(i) for i in range(dialog.device_combo.count())] == before_devices


def test_registration_dialog_refresh_preserves_display_name_restores_device_and_samplerate(qapp, monkeypatch):
    import ui.hardware_management_window as module

    refresh_calls = []

    class RefreshingSoundDeviceManager(FakeSoundDeviceManagerWithOther):
        @staticmethod
        def refresh_available_device():
            refresh_calls.append(True)

    monkeypatch.setattr(module, "SoundDeviceManager", RefreshingSoundDeviceManager)
    monkeypatch.setattr(module, "sd", FakeSoundDeviceQuery)

    dialog = module.HardwareRegistrationDialog(repository=FakeRepository())
    dialog.display_name_edit.setText("User typed name")

    dialog.refresh_devices()
    assert dialog.device_combo.currentData() is None
    assert dialog.samplerate_combo.currentText() == ""
    assert dialog.display_name_edit.text() == "User typed name"

    dialog.device_combo.setCurrentText("Mic 441")
    dialog.refresh_devices()
    assert dialog.api_combo.currentText() == "API-A"
    assert dialog.device_combo.currentData()["name"] == "Mic 441"
    assert dialog.samplerate_combo.currentText() == "44100"
    assert dialog.display_name_edit.text() == "User typed name"

    dialog.api_combo.setCurrentText("API-B")
    dialog.device_combo.setCurrentText("Interface 48")
    dialog.refresh_devices()
    assert dialog.api_combo.currentText() == "API-B"
    assert dialog.device_combo.currentData()["name"] == "Interface 48"
    assert dialog.samplerate_combo.currentText() == "48000"

    dialog.api_combo.setCurrentText("API-A")
    dialog.device_combo.setCurrentText("Speaker 96")
    dialog.refresh_devices()
    assert dialog.device_combo.currentData()["name"] == "Speaker 96"
    assert dialog.samplerate_combo.currentText() == "48000"
    assert refresh_calls == [True, True, True, True]


def test_registration_dialog_refresh_restores_unique_same_name_device_when_index_changes(qapp, monkeypatch):
    import ui.hardware_management_window as module

    generation = {"value": 0}

    class ReindexedSoundDeviceManager:
        @staticmethod
        def refresh_available_device():
            generation["value"] = 1

        @staticmethod
        def get_api_info(api_index=None):
            device_indexes = [2] if generation["value"] == 0 else [4]
            apis = [{"name": "API-A", "devices": device_indexes}]
            if api_index is None:
                return apis
            return apis[api_index]

        @staticmethod
        def get_device_info():
            return {}

    class ReindexedSoundDeviceQuery:
        @staticmethod
        def query_devices():
            return [
                {},
                {},
                {
                    "index": 2,
                    "name": "Speaker 96",
                    "hostapi": 0,
                    "default_samplerate": 96000.0,
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                },
                {},
                {
                    "index": 4,
                    "name": "Speaker 96",
                    "hostapi": 0,
                    "default_samplerate": 44100.0,
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                },
            ]

    monkeypatch.setattr(module, "SoundDeviceManager", ReindexedSoundDeviceManager)
    monkeypatch.setattr(module, "sd", ReindexedSoundDeviceQuery)

    dialog = module.HardwareRegistrationDialog(repository=FakeRepository())
    dialog.device_combo.setCurrentText("Speaker 96")
    assert dialog.device_combo.currentData()["index"] == 2
    assert dialog.samplerate_combo.currentText() == "48000"

    dialog.refresh_devices()

    assert dialog.api_combo.currentText() == "API-A"
    assert dialog.device_combo.currentData()["index"] == 4
    assert dialog.device_combo.currentData()["name"] == "Speaker 96"
    assert dialog.samplerate_combo.currentText() == "44100"


def test_registration_dialog_refresh_does_not_restore_device_under_different_api(qapp, monkeypatch):
    import ui.hardware_management_window as module

    generation = {"value": 0}

    class ApiDisappearsSoundDeviceManager:
        @staticmethod
        def refresh_available_device():
            generation["value"] = 1

        @staticmethod
        def get_api_info(api_index=None):
            if generation["value"] == 0:
                apis = [{"name": "API-A", "devices": [1]}]
            else:
                apis = [{"name": "API-C", "devices": [1]}]
            if api_index is None:
                return apis
            return apis[api_index]

        @staticmethod
        def get_device_info():
            return {}

    monkeypatch.setattr(module, "SoundDeviceManager", ApiDisappearsSoundDeviceManager)
    monkeypatch.setattr(module, "sd", FakeSoundDeviceQuery)

    dialog = module.HardwareRegistrationDialog(repository=FakeRepository())
    dialog.device_combo.setCurrentText("Mic 441")

    dialog.refresh_devices()

    assert dialog.api_combo.currentText() == "API-C"
    assert dialog.device_combo.currentData() is None
    assert dialog.device_combo.currentText() == ""
    assert dialog.samplerate_combo.currentText() == ""


def test_registration_dialog_successful_refresh_clears_discovery_error(qapp, monkeypatch):
    import ui.hardware_management_window as module

    generation = {"value": 0}

    class RecoveringSoundDeviceManager(FakeSoundDeviceManager):
        @staticmethod
        def refresh_available_device():
            generation["value"] = 1

        @staticmethod
        def get_api_info(api_index=None):
            if generation["value"] == 0:
                raise RuntimeError("enumeration failed")
            return FakeSoundDeviceManager.get_api_info(api_index)

    monkeypatch.setattr(module, "SoundDeviceManager", RecoveringSoundDeviceManager)
    monkeypatch.setattr(module, "sd", FakeSoundDeviceQuery)
    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))

    repository = FakeRepository()
    dialog = module.HardwareRegistrationDialog(repository=repository)
    assert dialog.api_combo.count() == 0
    assert warnings[-1][2] == "enumeration failed"

    dialog.refresh_devices()
    dialog.device_combo.setCurrentText("Mic 441")
    dialog.accept_registration()

    assert dialog.result() == QDialog.Accepted
    assert repository.registered[-1][1:] == ("API-A", "Mic 441", 44100, 32, 100)


def test_hardware_management_window_uses_sound_device_manager_sd(monkeypatch):
    import base.sound_device_manager as sound_device_manager

    original_module = sys.modules.pop("ui.hardware_management_window", None)
    ui_package = sys.modules.get("ui")
    original_package_attr = getattr(ui_package, "hardware_management_window", None) if ui_package else None
    sentinel_sd = object()
    monkeypatch.setattr(sound_device_manager, "sd", sentinel_sd)

    try:
        module = importlib.import_module("ui.hardware_management_window")

        assert module.sd is sentinel_sd
    finally:
        sys.modules.pop("ui.hardware_management_window", None)
        if original_module is not None:
            sys.modules["ui.hardware_management_window"] = original_module
            if ui_package is not None:
                setattr(ui_package, "hardware_management_window", original_module)
        elif ui_package is not None and original_package_attr is not None:
            setattr(ui_package, "hardware_management_window", original_package_attr)
        elif ui_package is not None and hasattr(ui_package, "hardware_management_window"):
            delattr(ui_package, "hardware_management_window")


def test_registration_dialog_registers_selected_device_and_stays_open_on_failure(qapp, monkeypatch):
    import ui.hardware_management_window as module

    monkeypatch.setattr(module, "SoundDeviceManager", FakeSoundDeviceManager)
    monkeypatch.setattr(module, "sd", FakeSoundDeviceQuery)
    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))

    repository = FakeRepository(register_error=module.HardwareManagementError("bad input"))
    dialog = module.HardwareRegistrationDialog(repository=repository)
    dialog.device_combo.setCurrentIndex(1)
    assert dialog.display_name_edit.text() == ""
    dialog.accept_registration()

    assert dialog.result() == QDialog.Rejected
    assert repository.registered[-1][1:] == ("API-A", "Mic 441", 44100, 32, 100)
    assert warnings

    repository.register_error = None
    dialog.display_name_edit.setText("Custom interface")
    dialog.accept_registration()
    assert dialog.result() == QDialog.Accepted
    assert repository.registered[-1][1:] == ("API-A", "Custom interface", 44100, 32, 100)


def test_registration_dialog_enumerates_and_registers_other_devices(qapp, monkeypatch):
    import ui.hardware_management_window as module

    monkeypatch.setattr(module, "SoundDeviceManager", FakeSoundDeviceManagerWithOther)
    monkeypatch.setattr(module, "sd", FakeSoundDeviceQuery)

    repository = FakeRepository()
    dialog = module.HardwareRegistrationDialog(repository=repository)

    assert [dialog.device_combo.itemText(i) for i in range(dialog.device_combo.count())] == [
        "",
        "Mic 441",
        "Speaker 96",
        "Control Surface",
    ]

    dialog.device_combo.setCurrentText("Control Surface")
    dialog.accept_registration()

    assert dialog.result() == QDialog.Accepted
    runtime_device, hostapi_name, display_name, samplerate, bit_depth, latency_ms = repository.registered[-1]
    assert runtime_device["name"] == "Control Surface"
    assert runtime_device["max_input_channels"] == 0
    assert runtime_device["max_output_channels"] == 0
    assert hostapi_name == "API-A"
    assert display_name == "Control Surface"
    assert samplerate == 48000
    assert bit_depth == 32
    assert latency_ms == 100


def test_registration_dialog_whitespace_display_name_falls_back_to_device_name(qapp, monkeypatch):
    import ui.hardware_management_window as module

    monkeypatch.setattr(module, "SoundDeviceManager", FakeSoundDeviceManager)
    monkeypatch.setattr(module, "sd", FakeSoundDeviceQuery)

    repository = FakeRepository()
    dialog = module.HardwareRegistrationDialog(repository=repository)
    dialog.device_combo.setCurrentIndex(2)
    dialog.display_name_edit.setText("   ")

    dialog.accept_registration()

    assert dialog.result() == QDialog.Accepted
    assert repository.registered[-1][1:] == ("API-A", "Speaker 96", 48000, 32, 100)


def test_registration_dialog_surfaces_device_enumeration_failure(qapp, monkeypatch):
    import ui.hardware_management_window as module

    monkeypatch.setattr(module, "SoundDeviceManager", FakeSoundDeviceManagerEnumerationFailure)
    monkeypatch.setattr(module, "sd", FakeSoundDeviceQuery)
    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))

    dialog = module.HardwareRegistrationDialog(repository=FakeRepository())

    assert dialog.api_combo.count() == 0
    assert warnings
    assert warnings[-1][2] == "enumeration failed"

    dialog.accept_registration()

    assert warnings[-1][2] == "enumeration failed"


def test_registration_dialog_catches_database_failure_and_stays_open(qapp, monkeypatch):
    import ui.hardware_management_window as module

    monkeypatch.setattr(module, "SoundDeviceManager", FakeSoundDeviceManager)
    monkeypatch.setattr(module, "sd", FakeSoundDeviceQuery)
    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))

    repository = FakeRepository(register_error=sqlite3.Error("database locked"))
    dialog = module.HardwareRegistrationDialog(repository=repository)
    dialog.device_combo.setCurrentIndex(1)
    assert dialog.display_name_edit.text() == ""

    dialog.accept_registration()

    assert dialog.result() == QDialog.Rejected
    assert repository.registered[-1][1:] == ("API-A", "Mic 441", 44100, 32, 100)
    assert warnings


def test_table_model_marks_only_mutable_columns_editable(qapp):
    import ui.hardware_management_window as module

    model = module.HardwareManagementTableModel(FakeRepository([sample_asset()]))
    editable = {"display_name", "samplerate", "bit_depth", "latency_ms"}

    for column, field in enumerate(model.columns):
        flags = model.flags(model.index(0, column))
        assert bool(flags & Qt.ItemIsEditable) is (field in editable)


def test_hardware_management_table_hides_id_and_orders_identity_columns(qapp):
    import ui.hardware_management_window as module

    window = module.HardwareManagementWindow(repository=FakeRepository([sample_asset()]))

    headers = [
        window.model.headerData(column, Qt.Horizontal, Qt.DisplayRole)
        for column in range(window.model.columnCount())
    ]

    assert "硬件ID" not in headers
    assert headers[:4] == ["显示名称", "驱动", "设备名称", "类型"]
    assert window.model.columns[:4] == ("display_name", "hostapi_name", "device_name", "hardware_type")


def test_hardware_management_table_uses_single_cell_selection(qapp):
    import ui.hardware_management_window as module

    window = module.HardwareManagementWindow(repository=FakeRepository([sample_asset()]))

    assert window.table.selectionBehavior() == window.table.SelectItems
    assert window.table.selectionMode() == window.table.SingleSelection

    index = window.model.index(0, window.model.column_index("device_name"))
    window.table.setCurrentIndex(index)
    window.table.selectionModel().select(index, window.table.selectionModel().ClearAndSelect)
    qapp.processEvents()

    assert window.table.selectionModel().selectedIndexes() == [index]
    assert window.table.selectionModel().selectedRows() == []


def test_display_name_column_is_single_check_selection_source(qapp):
    import ui.hardware_management_window as module

    second_asset = sample_asset()
    second_asset.update(
        {
            "hardware_id": "hw-2",
            "display_name": "Second name",
            "device_name": "Runtime mic 2",
        }
    )
    model = module.HardwareManagementTableModel(FakeRepository([sample_asset(), second_asset]))
    first_index = model.index(0, model.column_index("display_name"))
    second_index = model.index(1, model.column_index("display_name"))

    assert model.flags(first_index) & Qt.ItemIsUserCheckable
    assert model.data(first_index, Qt.CheckStateRole) == Qt.Unchecked

    assert model.setData(first_index, Qt.Checked, Qt.CheckStateRole)
    assert model.data(first_index, Qt.CheckStateRole) == Qt.Checked
    assert model.checked_hardware_id() == "hw-1"

    assert model.setData(second_index, Qt.Checked, Qt.CheckStateRole)
    assert model.data(first_index, Qt.CheckStateRole) == Qt.Unchecked
    assert model.data(second_index, Qt.CheckStateRole) == Qt.Checked
    assert model.checked_hardware_id() == "hw-2"

    assert model.setData(second_index, Qt.Unchecked, Qt.CheckStateRole)
    assert model.data(second_index, Qt.CheckStateRole) == Qt.Unchecked
    assert model.checked_hardware_id() is None


def test_display_name_checkbox_does_not_prevent_inline_edit(qapp):
    import ui.hardware_management_window as module

    repository = FakeRepository([sample_asset()])
    model = module.HardwareManagementTableModel(repository)
    index = model.index(0, model.column_index("display_name"))

    assert model.flags(index) & Qt.ItemIsUserCheckable
    assert model.flags(index) & Qt.ItemIsEditable

    assert model.setData(index, Qt.Checked, Qt.CheckStateRole)
    assert model.setData(index, "New name", Qt.EditRole)

    assert repository.updated_fields[-1] == ("hw-1", {"display_name": "New name"})
    assert model.data(index, Qt.DisplayRole) == "New name"
    assert model.data(index, Qt.CheckStateRole) == Qt.Checked


def test_editable_numeric_columns_are_model_rendered_display_cells(qapp):
    import ui.hardware_management_window as module

    asset = sample_asset()
    window = module.HardwareManagementWindow(repository=FakeRepository([asset]))

    for field in ("samplerate", "bit_depth", "latency_ms"):
        index = window.model.index(0, window.model.column_index(field))
        assert window.model.data(index, Qt.DisplayRole) == asset[field]
        assert window.model.data(index, Qt.CheckStateRole) is None
        assert window.table.indexWidget(index) is None


def test_hardware_management_actions_are_below_table(qapp):
    import ui.hardware_management_window as module

    window = module.HardwareManagementWindow(repository=FakeRepository([sample_asset()]))

    layout = window.layout()
    assert layout.itemAt(0).widget() is window.table
    action_item = layout.itemAt(1)
    assert action_item.layout() is not None
    button_texts = [
        action_item.layout().itemAt(index).widget().text()
        for index in range(action_item.layout().count())
        if action_item.layout().itemAt(index).widget() is not None
    ]
    assert button_texts == ["注册", "删除"]


def test_table_delegates_restrict_sample_rate_bit_depth_and_latency(qapp):
    import ui.hardware_management_window as module

    window = module.HardwareManagementWindow(repository=FakeRepository([sample_asset()]))

    samplerate_delegate = window.table.itemDelegateForColumn(window.model.column_index("samplerate"))
    bit_depth_delegate = window.table.itemDelegateForColumn(window.model.column_index("bit_depth"))
    latency_delegate = window.table.itemDelegateForColumn(window.model.column_index("latency_ms"))

    assert isinstance(samplerate_delegate, QStyledItemDelegate)
    assert isinstance(bit_depth_delegate, QStyledItemDelegate)
    assert isinstance(latency_delegate, QStyledItemDelegate)

    samplerate_editor = samplerate_delegate.createEditor(
        window.table, None, window.model.index(0, window.model.column_index("samplerate"))
    )
    bit_depth_editor = bit_depth_delegate.createEditor(
        window.table, None, window.model.index(0, window.model.column_index("bit_depth"))
    )
    latency_editor = latency_delegate.createEditor(
        window.table, None, window.model.index(0, window.model.column_index("latency_ms"))
    )

    assert isinstance(samplerate_editor, QComboBox)
    assert [samplerate_editor.itemText(i) for i in range(samplerate_editor.count())] == ["44100", "48000"]
    assert isinstance(bit_depth_editor, QComboBox)
    assert [bit_depth_editor.itemData(i) for i in range(bit_depth_editor.count())] == [8, 16, 24, 32]
    assert [bit_depth_editor.itemText(i) for i in range(bit_depth_editor.count())] == [
        "8bit",
        "16bit",
        "24bit",
        "32bit",
    ]
    assert isinstance(latency_editor, QSpinBox)
    assert latency_editor.minimum() == 0
    assert latency_editor.maximum() == 1000


def test_combo_delegate_opening_existing_non_first_value_does_not_commit(qapp):
    import ui.hardware_management_window as module

    asset = sample_asset()
    asset["samplerate"] = 48000
    repository = FakeRepository([asset])
    window = module.HardwareManagementWindow(repository=repository)
    index = window.model.index(0, window.model.column_index("samplerate"))
    delegate = window.table.itemDelegateForColumn(window.model.column_index("samplerate"))
    editor = delegate.createEditor(window.table, None, index)
    delegate.commitData.connect(lambda changed_editor: delegate.setModelData(changed_editor, window.model, index))

    delegate.setEditorData(editor, index)
    qapp.processEvents()

    assert repository.updated_fields == []
    assert window.model.data(index, Qt.DisplayRole) == 48000


def test_valid_table_edit_commits_and_refreshes_row(qapp):
    import ui.hardware_management_window as module

    repository = FakeRepository([sample_asset()])
    model = module.HardwareManagementTableModel(repository)
    index = model.index(0, model.column_index("display_name"))

    assert model.setData(index, "New name", Qt.EditRole)

    assert repository.updated_fields == [("hw-1", {"display_name": "New name"})]
    assert model.data(index, Qt.DisplayRole) == "New name"
    assert model.data(model.index(0, model.column_index("updated_at")), Qt.DisplayRole) == "2026-06-23 10:20:30"


def test_successful_inline_edit_emits_refreshed_asset(qapp):
    import ui.hardware_management_window as module

    repository = FakeRepository([sample_asset()])
    model = module.HardwareManagementTableModel(repository)
    events = []
    model.asset_updated.connect(lambda hardware_id, field, asset: events.append((hardware_id, field, asset)))
    index = model.index(0, model.column_index("samplerate"))

    assert model.setData(index, 48000, Qt.EditRole)

    assert len(events) == 1
    assert events[0][0] == "hw-1"
    assert events[0][1] == "samplerate"
    assert events[0][2]["samplerate"] == 48000
    assert events[0][2]["updated_at"] == "2026-06-23 10:20:30"


def test_inline_edit_write_failure_does_not_emit_asset_updated(qapp, monkeypatch):
    import ui.hardware_management_window as module

    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: None))
    repository = FakeRepository([sample_asset()], update_error=module.HardwareManagementError("write failed"))
    model = module.HardwareManagementTableModel(repository)
    events = []
    model.asset_updated.connect(lambda *args: events.append(args))

    assert not model.setData(model.index(0, model.column_index("samplerate")), 48000, Qt.EditRole)
    assert events == []


def test_check_state_row_selection_does_not_emit_asset_updated(qapp):
    import ui.hardware_management_window as module

    model = module.HardwareManagementTableModel(FakeRepository([sample_asset()]))
    events = []
    model.asset_updated.connect(lambda *args: events.append(args))

    assert model.setData(
        model.index(0, model.column_index("display_name")),
        Qt.Checked,
        Qt.CheckStateRole,
    )

    assert events == []


def test_successful_inline_edit_preserves_checked_hardware_and_refreshes_row(qapp):
    import ui.hardware_management_window as module

    repository = FakeRepository([sample_asset()])
    window = module.HardwareManagementWindow(repository=repository)
    index = window.model.index(0, window.model.column_index("display_name"))

    assert window.model.setData(index, Qt.Checked, Qt.CheckStateRole)
    assert window.model.checked_hardware_id() == "hw-1"

    assert window.model.setData(index, "New name", Qt.EditRole)
    qapp.processEvents()

    assert window.model.checked_hardware_id() == "hw-1"
    assert window.model.data(index, Qt.CheckStateRole) == Qt.Checked
    assert window.model.data(index, Qt.DisplayRole) == "New name"
    assert (
        window.model.data(window.model.index(0, window.model.column_index("updated_at")), Qt.DisplayRole)
        == "2026-06-23 10:20:30"
    )


def test_invalid_table_edit_warns_and_restores_previous_value(qapp, monkeypatch):
    import ui.hardware_management_window as module

    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    repository = FakeRepository([sample_asset()])
    model = module.HardwareManagementTableModel(repository)
    index = model.index(0, model.column_index("display_name"))

    assert not model.setData(index, "  ", Qt.EditRole)

    assert repository.updated_fields == []
    assert model.data(index, Qt.DisplayRole) == "Old name"
    assert warnings


def test_repository_write_failure_warns_and_reloads_table(qapp, monkeypatch):
    import ui.hardware_management_window as module

    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    repository = FakeRepository([sample_asset()], update_error=module.HardwareManagementError("write failed"))
    model = module.HardwareManagementTableModel(repository)
    index = model.index(0, model.column_index("samplerate"))

    assert not model.setData(index, 48000, Qt.EditRole)

    assert repository.updated_fields == [("hw-1", {"samplerate": 48000})]
    assert model.data(index, Qt.DisplayRole) == 44100
    assert repository.list_count >= 2
    assert warnings


def test_repository_database_write_failure_warns_and_reloads_table(qapp, monkeypatch):
    import ui.hardware_management_window as module

    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    repository = FakeRepository([sample_asset()], update_error=sqlite3.Error("database locked"))
    model = module.HardwareManagementTableModel(repository)
    index = model.index(0, model.column_index("samplerate"))

    assert not model.setData(index, 48000, Qt.EditRole)

    assert repository.updated_fields == [("hw-1", {"samplerate": 48000})]
    assert model.data(index, Qt.DisplayRole) == 44100
    assert repository.list_count >= 2
    assert warnings


def test_repository_write_failure_then_reload_failure_warns_and_preserves_row(qapp, monkeypatch):
    import ui.hardware_management_window as module

    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    repository = FakeRepository([sample_asset()], update_error=module.HardwareManagementError("write failed"))
    model = module.HardwareManagementTableModel(repository)
    repository.list_error = module.HardwareManagementError("reload failed")
    index = model.index(0, model.column_index("samplerate"))

    assert not model.setData(index, 48000, Qt.EditRole)

    assert repository.updated_fields == [("hw-1", {"samplerate": 48000})]
    assert model.data(index, Qt.DisplayRole) == 44100
    assert model.data(model.index(0, model.column_index("display_name")), Qt.DisplayRole) == "Old name"
    assert [warning[2] for warning in warnings] == ["write failed", "reload failed"]


def test_repository_success_then_get_asset_failure_falls_back_to_full_reload(qapp, monkeypatch):
    import ui.hardware_management_window as module

    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    repository = FakeRepository([sample_asset()], get_asset_error=module.HardwareManagementError("row reload failed"))
    model = module.HardwareManagementTableModel(repository)
    index = model.index(0, model.column_index("samplerate"))

    assert model.setData(index, 48000, Qt.EditRole)

    assert repository.updated_fields == [("hw-1", {"samplerate": 48000})]
    assert repository.get_asset_count == 1
    assert repository.list_count >= 2
    assert model.data(index, Qt.DisplayRole) == 48000
    assert model.data(model.index(0, model.column_index("updated_at")), Qt.DisplayRole) == "2026-06-23 10:20:30"
    assert warnings == []


def test_get_asset_failure_fallback_reload_preserves_checked_hardware_id(qapp, monkeypatch):
    import ui.hardware_management_window as module

    second_asset = sample_asset()
    second_asset.update(
        {
            "hardware_id": "hw-2",
            "display_name": "Second name",
            "device_name": "Runtime mic 2",
        }
    )
    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    repository = FakeRepository(
        [sample_asset(), second_asset],
        get_asset_error=module.HardwareManagementError("row reload failed"),
    )
    window = module.HardwareManagementWindow(repository=repository)
    index = window.model.index(1, window.model.column_index("display_name"))

    assert window.model.setData(index, Qt.Checked, Qt.CheckStateRole)
    assert window.model.checked_hardware_id() == "hw-2"

    assert window.model.setData(index, "Second edited", Qt.EditRole)
    qapp.processEvents()

    assert window.model.checked_hardware_id() == "hw-2"
    assert window.model.data(index, Qt.CheckStateRole) == Qt.Checked
    assert window.model.data(index, Qt.DisplayRole) == "Second edited"
    assert repository.updated_fields == [("hw-2", {"display_name": "Second edited"})]
    assert repository.get_asset_count == 1
    assert repository.list_count >= 2
    assert warnings == []


def test_get_asset_failure_fallback_reload_does_not_create_checked_hardware(qapp, monkeypatch):
    import ui.hardware_management_window as module

    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    repository = FakeRepository(
        [sample_asset()],
        get_asset_error=module.HardwareManagementError("row reload failed"),
    )
    window = module.HardwareManagementWindow(repository=repository)
    index = window.model.index(0, window.model.column_index("display_name"))

    assert window.model.checked_hardware_id() is None

    assert window.model.setData(index, "Edited name", Qt.EditRole)
    qapp.processEvents()

    reloaded_index = window.model.index(0, window.model.column_index("display_name"))
    assert window.model.checked_hardware_id() is None
    assert window.model.data(reloaded_index, Qt.CheckStateRole) == Qt.Unchecked
    assert window.model.data(reloaded_index, Qt.DisplayRole) == "Edited name"
    assert repository.updated_fields == [("hw-1", {"display_name": "Edited name"})]
    assert repository.get_asset_count == 1
    assert repository.list_count >= 2
    assert warnings == []


def test_get_asset_failure_fallback_reload_does_not_move_checked_hardware(qapp, monkeypatch):
    import ui.hardware_management_window as module

    first_asset = sample_asset()
    second_asset = sample_asset()
    second_asset.update(
        {
            "hardware_id": "hw-2",
            "display_name": "Second name",
            "device_name": "Runtime mic 2",
        }
    )
    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    repository = FakeRepository(
        [first_asset, second_asset],
        get_asset_error=module.HardwareManagementError("row reload failed"),
    )
    window = module.HardwareManagementWindow(repository=repository)
    first_index = window.model.index(0, window.model.column_index("display_name"))
    second_index = window.model.index(1, window.model.column_index("display_name"))

    assert window.model.setData(first_index, Qt.Checked, Qt.CheckStateRole)
    assert window.model.checked_hardware_id() == "hw-1"

    assert window.model.setData(second_index, "Second edited", Qt.EditRole)
    qapp.processEvents()

    reloaded_first_index = window.model.index(0, window.model.column_index("display_name"))
    reloaded_second_index = window.model.index(1, window.model.column_index("display_name"))
    assert window.model.checked_hardware_id() == "hw-1"
    assert window.model.data(reloaded_first_index, Qt.CheckStateRole) == Qt.Checked
    assert window.model.data(reloaded_second_index, Qt.CheckStateRole) == Qt.Unchecked
    assert window.model.data(reloaded_second_index, Qt.DisplayRole) == "Second edited"
    assert repository.updated_fields == [("hw-2", {"display_name": "Second edited"})]
    assert repository.get_asset_count == 1
    assert repository.list_count >= 2
    assert warnings == []


def test_get_asset_failure_fallback_reload_uses_authoritative_db_value(qapp, monkeypatch):
    import ui.hardware_management_window as module

    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    authoritative_asset = sample_asset()
    repository = FakeRepository(
        [authoritative_asset],
        get_asset_error=module.HardwareManagementError("row reload failed"),
    )

    def update_with_different_authoritative_value(hardware_id, fields):
        repository.updated_fields.append((hardware_id, dict(fields)))
        for asset in repository._assets:
            if asset["hardware_id"] == hardware_id:
                asset["display_name"] = "Database canonical name"
                asset["updated_at"] = "2026-06-23 10:20:30"
                return True
        return False

    repository.update_asset_fields = update_with_different_authoritative_value
    window = module.HardwareManagementWindow(repository=repository)
    index = window.model.index(0, window.model.column_index("display_name"))

    assert window.model.setData(index, Qt.Checked, Qt.CheckStateRole)

    assert window.model.setData(index, "Optimistic local name", Qt.EditRole)
    qapp.processEvents()

    reloaded_index = window.model.index(0, window.model.column_index("display_name"))
    assert window.model.checked_hardware_id() == "hw-1"
    assert window.model.data(reloaded_index, Qt.CheckStateRole) == Qt.Checked
    assert window.model.data(reloaded_index, Qt.DisplayRole) == "Database canonical name"
    assert window.table.model().data(reloaded_index, Qt.DisplayRole) == "Database canonical name"
    assert repository.updated_fields == [("hw-1", {"display_name": "Optimistic local name"})]
    assert repository.get_asset_count == 1
    assert repository.list_count >= 2
    assert warnings == []


def test_repository_success_then_get_asset_none_falls_back_to_full_reload(qapp, monkeypatch):
    import ui.hardware_management_window as module

    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    repository = FakeRepository([sample_asset()], get_asset_missing=True)
    model = module.HardwareManagementTableModel(repository)
    events = []
    model.asset_updated.connect(lambda hardware_id, field, asset: events.append((hardware_id, field, asset)))
    index = model.index(0, model.column_index("samplerate"))

    assert model.setData(index, 48000, Qt.EditRole)

    assert repository.updated_fields == [("hw-1", {"samplerate": 48000})]
    assert repository.get_asset_count == 1
    assert repository.list_count >= 2
    assert model.rowCount() == 1
    assert model.data(index, Qt.DisplayRole) == 48000
    assert model.assets[0]["hardware_id"] == "hw-1"
    assert events == [("hw-1", "samplerate", dict(model.assets[0]))]
    assert events[0][2]["samplerate"] == 48000
    assert events[0][2]["updated_at"] == "2026-06-23 10:20:30"
    assert warnings == []


def test_repository_success_then_authoritative_reads_fail_keeps_committed_local_value(qapp, monkeypatch):
    import ui.hardware_management_window as module

    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    repository = FakeRepository([sample_asset()], get_asset_error=module.HardwareManagementError("row reload failed"))
    window = module.HardwareManagementWindow(repository=repository)
    repository.list_error = module.HardwareManagementError("full reload failed")
    events = []
    window.model.asset_updated.connect(lambda hardware_id, field, asset: events.append((hardware_id, field, asset)))
    index = window.model.index(0, window.model.column_index("samplerate"))

    assert window.model.setData(
        window.model.index(0, window.model.column_index("display_name")),
        Qt.Checked,
        Qt.CheckStateRole,
    )

    assert window.model.setData(index, 48000, Qt.EditRole)
    qapp.processEvents()

    assert repository.updated_fields == [("hw-1", {"samplerate": 48000})]
    assert repository.get_asset_count == 1
    assert window.model.rowCount() == 1
    assert window.model.data(index, Qt.DisplayRole) == 48000
    assert window.model.assets[0]["hardware_id"] == "hw-1"
    assert window.model.checked_hardware_id() == "hw-1"
    assert events == [("hw-1", "samplerate", dict(window.model.assets[0]))]
    assert events[0][2]["samplerate"] == 48000
    assert events[0][2]["updated_at"] == "2026-06-23 09:00:00"
    assert [warning[2] for warning in warnings] == ["row reload failed", "full reload failed"]


def test_repository_success_refreshes_row_without_full_reload(qapp, monkeypatch):
    import ui.hardware_management_window as module

    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    repository = FakeRepository([sample_asset()])
    model = module.HardwareManagementTableModel(repository)
    repository.list_error = module.HardwareManagementError("reload failed")
    list_count_after_initial_load = repository.list_count
    index = model.index(0, model.column_index("samplerate"))

    assert model.setData(index, 48000, Qt.EditRole)

    assert repository.updated_fields == [("hw-1", {"samplerate": 48000})]
    assert repository.get_asset_count == 1
    assert repository.list_count == list_count_after_initial_load
    assert model.data(index, Qt.DisplayRole) == 48000
    assert model.data(model.index(0, model.column_index("updated_at")), Qt.DisplayRole) == "2026-06-23 10:20:30"
    assert warnings == []


def test_repository_unexpected_runtime_error_is_not_swallowed(qapp, monkeypatch):
    import ui.hardware_management_window as module

    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    repository = FakeRepository([sample_asset()], update_error=RuntimeError("programming bug"))
    model = module.HardwareManagementTableModel(repository)
    index = model.index(0, model.column_index("samplerate"))

    with pytest.raises(RuntimeError, match="programming bug"):
        model.setData(index, 48000, Qt.EditRole)

    assert warnings == []


def test_refresh_read_failure_warns_and_preserves_current_rows(qapp, monkeypatch):
    import ui.hardware_management_window as module

    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    repository = FakeRepository([sample_asset()])
    window = module.HardwareManagementWindow(repository=repository)
    repository.list_error = module.HardwareManagementError("refresh failed")

    window.refresh()

    assert window.model.rowCount() == 1
    assert window.model.data(window.model.index(0, window.model.column_index("display_name")), Qt.DisplayRole) == "Old name"
    assert warnings
    assert warnings[-1][2] == "refresh failed"


def test_delete_ignores_selected_cell_without_checked_hardware(qapp, monkeypatch):
    import ui.hardware_management_window as module

    warnings = []
    questions = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    monkeypatch.setattr(module.MessageBox, "question", staticmethod(lambda *args, **kwargs: questions.append(args)))
    repository = FakeRepository([sample_asset()])
    window = module.HardwareManagementWindow(repository=repository)
    index = window.model.index(0, window.model.column_index("device_name"))
    window.table.setCurrentIndex(index)
    window.table.selectionModel().select(index, window.table.selectionModel().ClearAndSelect)
    qapp.processEvents()

    window.delete_selected_asset()

    assert repository.deleted == []
    assert questions == []
    assert warnings
    assert warnings[-1][2] == "请选择要删除的硬件"


def test_delete_uses_checked_hardware_not_selected_cell(qapp, monkeypatch):
    import ui.hardware_management_window as module

    second_asset = sample_asset()
    second_asset.update(
        {
            "hardware_id": "hw-2",
            "display_name": "Second name",
            "device_name": "Runtime mic 2",
        }
    )
    monkeypatch.setattr(module.MessageBox, "question", staticmethod(lambda *args, **kwargs: module.MessageBox.Yes))
    repository = FakeRepository([sample_asset(), second_asset])
    window = module.HardwareManagementWindow(repository=repository)
    checked_index = window.model.index(0, window.model.column_index("display_name"))
    assert window.model.setData(checked_index, Qt.Checked, Qt.CheckStateRole)
    window.table.clearSelection()
    selected_index = window.model.index(1, window.model.column_index("device_name"))
    window.table.setCurrentIndex(selected_index)
    window.table.selectionModel().select(selected_index, window.table.selectionModel().ClearAndSelect)
    qapp.processEvents()

    window.delete_selected_asset()

    assert repository.deleted == ["hw-1"]


def test_delete_clears_saved_selection_and_notifies_parent_when_deleted_hardware_was_selected(qapp, monkeypatch):
    import ui.hardware_management_window as module

    cleared = []
    invalidated = []
    monkeypatch.setattr(module.MessageBox, "question", staticmethod(lambda *args, **kwargs: module.MessageBox.Yes))
    monkeypatch.setattr(
        module.SoundDeviceManager,
        "clear_selected_devices_for_deleted_hardware",
        staticmethod(lambda hardware_id: cleared.append(hardware_id) or True),
    )
    repository = FakeRepository([sample_asset()])
    window = module.HardwareManagementWindow(
        repository=repository,
        on_selected_devices_invalidated=lambda hardware_id: invalidated.append(hardware_id),
    )
    assert window.model.setData(
        window.model.index(0, window.model.column_index("display_name")),
        Qt.Checked,
        Qt.CheckStateRole,
    )

    window.delete_selected_asset()

    assert repository.deleted == ["hw-1"]
    assert cleared == ["hw-1"]
    assert invalidated == ["hw-1"]


def test_delete_does_not_notify_parent_when_deleted_hardware_was_not_selected(qapp, monkeypatch):
    import ui.hardware_management_window as module

    invalidated = []
    monkeypatch.setattr(module.MessageBox, "question", staticmethod(lambda *args, **kwargs: module.MessageBox.Yes))
    monkeypatch.setattr(
        module.SoundDeviceManager,
        "clear_selected_devices_for_deleted_hardware",
        staticmethod(lambda _hardware_id: False),
    )
    repository = FakeRepository([sample_asset()])
    window = module.HardwareManagementWindow(
        repository=repository,
        on_selected_devices_invalidated=lambda hardware_id: invalidated.append(hardware_id),
    )
    assert window.model.setData(
        window.model.index(0, window.model.column_index("display_name")),
        Qt.Checked,
        Qt.CheckStateRole,
    )

    window.delete_selected_asset()

    assert invalidated == []


def test_selected_hardware_edit_notifies_parent_when_current_selection_matches(qapp):
    import ui.hardware_management_window as module

    repository = FakeRepository([sample_asset()])
    updates = []
    window = module.HardwareManagementWindow(
        repository=repository,
        selected_devices_provider=lambda: ({"hardware_id": "hw-1"}, {"hardware_id": "speaker-1"}),
        on_selected_devices_updated=lambda hardware_id, asset: updates.append((hardware_id, asset)),
    )
    index = window.model.index(0, window.model.column_index("samplerate"))

    assert window.model.setData(index, 48000, Qt.EditRole)
    qapp.processEvents()

    assert len(updates) == 1
    assert updates[0][0] == "hw-1"
    assert updates[0][1]["samplerate"] == 48000


def test_selected_hardware_edit_notifies_parent_when_committed_fallback_preserves_local_asset(qapp, monkeypatch):
    import ui.hardware_management_window as module

    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    repository = FakeRepository([sample_asset()], get_asset_error=module.HardwareManagementError("row reload failed"))
    updates = []
    window = module.HardwareManagementWindow(
        repository=repository,
        selected_devices_provider=lambda: ({"hardware_id": "hw-1"}, {"hardware_id": "speaker-1"}),
        on_selected_devices_updated=lambda hardware_id, asset: updates.append((hardware_id, asset)),
    )
    repository.list_error = module.HardwareManagementError("full reload failed")
    index = window.model.index(0, window.model.column_index("samplerate"))

    assert window.model.setData(index, 48000, Qt.EditRole)
    qapp.processEvents()

    assert updates == [("hw-1", dict(window.model.assets[0]))]
    assert updates[0][1]["samplerate"] == 48000
    assert updates[0][1]["updated_at"] == "2026-06-23 09:00:00"
    assert [warning[2] for warning in warnings] == ["row reload failed", "full reload failed"]


def test_unselected_hardware_edit_does_not_notify_parent(qapp):
    import ui.hardware_management_window as module

    repository = FakeRepository([sample_asset()])
    updates = []
    window = module.HardwareManagementWindow(
        repository=repository,
        selected_devices_provider=lambda: ({"hardware_id": "other-mic"}, {"hardware_id": "speaker-1"}),
        on_selected_devices_updated=lambda hardware_id, asset: updates.append((hardware_id, asset)),
    )
    index = window.model.index(0, window.model.column_index("samplerate"))

    assert window.model.setData(index, 48000, Qt.EditRole)
    qapp.processEvents()

    assert updates == []
    assert window.model.data(index, Qt.DisplayRole) == 48000


def test_delete_notifies_parent_when_deleted_hardware_matches_in_memory_selection_only(qapp, monkeypatch):
    import ui.hardware_management_window as module

    invalidated = []
    warnings = []
    clear_result = types.SimpleNamespace(
        status="no_match",
        matched=False,
        cleared=False,
        clear_failed=False,
        error="",
    )
    monkeypatch.setattr(module.MessageBox, "question", staticmethod(lambda *args, **kwargs: module.MessageBox.Yes))
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    monkeypatch.setattr(
        module.SoundDeviceManager,
        "clear_selected_devices_for_deleted_hardware",
        staticmethod(lambda _hardware_id: clear_result),
    )
    repository = FakeRepository([sample_asset()])
    window = module.HardwareManagementWindow(
        repository=repository,
        on_selected_devices_invalidated=lambda hardware_id: invalidated.append(hardware_id),
        selected_devices_provider=lambda: (
            {"hardware_id": "hw-1", "name": "Selected Mic"},
            {"hardware_id": "hw-2", "name": "Selected Speaker"},
        ),
    )
    assert window.model.setData(
        window.model.index(0, window.model.column_index("display_name")),
        Qt.Checked,
        Qt.CheckStateRole,
    )

    window.delete_selected_asset()

    assert repository.deleted == ["hw-1"]
    assert invalidated == ["hw-1"]
    assert warnings == []


def test_delete_blocks_when_audio_workflow_active(qapp, monkeypatch):
    import ui.hardware_management_window as module

    invalidated = []
    cleared = []
    warnings = []
    questions = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    monkeypatch.setattr(module.MessageBox, "question", staticmethod(lambda *args, **kwargs: questions.append(args)))
    monkeypatch.setattr(
        module.SoundDeviceManager,
        "clear_selected_devices_for_deleted_hardware",
        staticmethod(lambda hardware_id: cleared.append(hardware_id) or True),
    )
    repository = FakeRepository([sample_asset()])
    window = module.HardwareManagementWindow(
        repository=repository,
        on_selected_devices_invalidated=lambda hardware_id: invalidated.append(hardware_id),
        selected_devices_provider=lambda: ({"hardware_id": "hw-1"}, None),
        audio_workflow_active_provider=lambda: True,
    )
    assert window.model.setData(
        window.model.index(0, window.model.column_index("display_name")),
        Qt.Checked,
        Qt.CheckStateRole,
    )

    window.delete_selected_asset()

    assert repository.deleted == []
    assert cleared == []
    assert invalidated == []
    assert questions == []
    assert warnings
    assert "播放或录音进行中" in warnings[-1][2]


def test_delete_warns_and_invalidates_when_matching_saved_selection_clear_fails(qapp, monkeypatch):
    import ui.hardware_management_window as module

    invalidated = []
    warnings = []
    clear_result = types.SimpleNamespace(
        status="clear_failed",
        matched=True,
        cleared=False,
        clear_failed=True,
        error="permission denied",
    )
    monkeypatch.setattr(module.MessageBox, "question", staticmethod(lambda *args, **kwargs: module.MessageBox.Yes))
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    monkeypatch.setattr(
        module.SoundDeviceManager,
        "clear_selected_devices_for_deleted_hardware",
        staticmethod(lambda _hardware_id: clear_result),
    )
    repository = FakeRepository([sample_asset()])
    window = module.HardwareManagementWindow(
        repository=repository,
        on_selected_devices_invalidated=lambda hardware_id: invalidated.append(hardware_id),
        selected_devices_provider=lambda: (None, None),
    )
    assert window.model.setData(
        window.model.index(0, window.model.column_index("display_name")),
        Qt.Checked,
        Qt.CheckStateRole,
    )

    window.delete_selected_asset()

    assert repository.deleted == ["hw-1"]
    assert invalidated == ["hw-1"]
    assert warnings
    assert "已删除硬件" in warnings[-1][2]
    assert "permission denied" in warnings[-1][2]


def test_delete_database_failure_warns_and_reloads_table(qapp, monkeypatch):
    import ui.hardware_management_window as module

    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    monkeypatch.setattr(module.MessageBox, "question", staticmethod(lambda *args, **kwargs: module.MessageBox.Yes))
    repository = FakeRepository([sample_asset()], delete_error=sqlite3.Error("database locked"))
    window = module.HardwareManagementWindow(repository=repository)
    assert window.model.setData(
        window.model.index(0, window.model.column_index("display_name")),
        Qt.Checked,
        Qt.CheckStateRole,
    )

    window.delete_selected_asset()

    assert repository.deleted == ["hw-1"]
    assert window.model.rowCount() == 1
    assert repository.list_count >= 2
    assert warnings


def test_delete_false_warns_refreshes_and_keeps_row(qapp, monkeypatch):
    import ui.hardware_management_window as module

    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    monkeypatch.setattr(module.MessageBox, "question", staticmethod(lambda *args, **kwargs: module.MessageBox.Yes))
    repository = FakeRepository([sample_asset()], delete_result=False)
    window = module.HardwareManagementWindow(repository=repository)
    assert window.model.setData(
        window.model.index(0, window.model.column_index("display_name")),
        Qt.Checked,
        Qt.CheckStateRole,
    )

    window.delete_selected_asset()

    assert repository.deleted == ["hw-1"]
    assert window.model.rowCount() == 1
    assert repository.list_count >= 2
    assert warnings


def test_delete_success_then_refresh_failure_warns_and_removes_deleted_row(qapp, monkeypatch):
    import ui.hardware_management_window as module

    warnings = []
    monkeypatch.setattr(module.MessageBox, "warning", staticmethod(lambda *args, **kwargs: warnings.append(args)))
    monkeypatch.setattr(module.MessageBox, "question", staticmethod(lambda *args, **kwargs: module.MessageBox.Yes))
    repository = FakeRepository([sample_asset()])
    window = module.HardwareManagementWindow(repository=repository)
    repository.list_error = module.HardwareManagementError("refresh failed")
    assert window.model.setData(
        window.model.index(0, window.model.column_index("display_name")),
        Qt.Checked,
        Qt.CheckStateRole,
    )

    window.delete_selected_asset()

    assert repository.deleted == ["hw-1"]
    assert window.model.rowCount() == 0
    assert warnings
    assert warnings[-1][2] == "refresh failed"

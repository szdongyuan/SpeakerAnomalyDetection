from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from base import recording_management
from base.recording_management import RecordingManager
from consts import error_code
from ui.sequence import sequence_widget_serial_trigger_ops as serial_ops_module
from ui.sequence.sequence_widget_serial_trigger_ops import SequenceWidgetSerialTriggerOpsMixin


class _Logger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)


class _RecentPanel:
    def __init__(self):
        self.removed = []

    def remove_session(self, session_id):
        self.removed.append(session_id)


def test_discard_recent_session_group_removes_only_target_round_and_recordings(monkeypatch):
    deleted_paths = []

    class _RecordingManager:
        def delete_audio(self, file_path):
            deleted_paths.append(file_path)
            return error_code.OK, "deleted"

    monkeypatch.setattr(serial_ops_module, "RecordingManager", _RecordingManager)
    host = SimpleNamespace(
        recent_test_sessions=["r1-1", "r1-2", "r2-1"],
        recent_test_session_by_id={
            "r1-1": {"group_id": "round-1", "recorded_path": "one.wav"},
            "r1-2": {
                "group_id": "round-1",
                "recorded_path": "",
                "recorded_signal_info": {"file_path": "two.wav"},
            },
            "r1-map-only": {
                "group_id": "round-1",
                "recorded_path": "map-only.wav",
            },
            "r2-1": {"group_id": "round-2", "recorded_path": "other.wav"},
        },
        _current_recent_session_id="r1-2",
        _pending_recent_session_append=True,
        recent_session_panel=_RecentPanel(),
        default_logger=_Logger(),
    )

    removed_count = SequenceWidgetSerialTriggerOpsMixin._delete_serial_product_round_records(
        host,
        "round-1",
    )

    assert removed_count == 3
    assert deleted_paths == ["one.wav", "two.wav", "map-only.wav"]
    assert host.recent_test_sessions == ["r2-1"]
    assert set(host.recent_test_session_by_id) == {"r2-1"}
    assert host.recent_session_panel.removed == ["r1-1", "r1-2", "r1-map-only"]
    assert host._current_recent_session_id is None
    assert host._pending_recent_session_append is False


def test_delete_audio_uses_the_normalized_database_path(tmp_path, monkeypatch):
    application_root = tmp_path / "application"
    audio_path = application_root / "audio_data" / "round.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"wav")
    database = Mock()
    data_save = MagicMock()
    data_save.return_value.__enter__.return_value = database
    monkeypatch.setattr(recording_management.running_consts, "DEFAULT_DIR", str(application_root))
    monkeypatch.setattr(recording_management, "DataSave", data_save)

    code, _message = RecordingManager().delete_audio(str(audio_path))

    assert code == error_code.OK
    database.delete_with_condition.assert_called_once_with(
        "audio_data_table",
        {"file_path": "audio_data/round.wav"},
    )
    assert not audio_path.exists()


def test_delete_audio_removes_database_record_when_wav_is_already_missing(
    tmp_path,
    monkeypatch,
):
    application_root = tmp_path / "application"
    missing_audio_path = application_root / "audio_data" / "missing.wav"
    database = Mock()
    data_save = MagicMock()
    data_save.return_value.__enter__.return_value = database
    monkeypatch.setattr(recording_management.running_consts, "DEFAULT_DIR", str(application_root))
    monkeypatch.setattr(recording_management, "DataSave", data_save)

    code, _message = RecordingManager().delete_audio(str(missing_audio_path))

    assert code == error_code.OK
    database.delete_with_condition.assert_called_once_with(
        "audio_data_table",
        {"file_path": "audio_data/missing.wav"},
    )

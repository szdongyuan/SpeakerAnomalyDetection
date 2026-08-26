from __future__ import annotations

import inspect


def test_import_transaction_is_owned_by_recording_mvc():
    from ui.sequence.sequence_analysis_controller import SequenceAnalysisController
    from ui.sequence.sequence_analysis_model import SequenceAnalysisModel
    from ui.sequence.sequence_analysis_view import SequenceAnalysisView
    from ui.sequence.sequence_recording_controller import SequenceRecordingController
    from ui.sequence.sequence_recording_import_service import (
        SequenceImportedAudioService,
    )
    from ui.sequence.sequence_recording_model import RecordingModel
    from ui.sequence.sequence_recording_view import SequenceRecordingImportView

    assert callable(SequenceImportedAudioService)
    assert callable(SequenceRecordingImportView)
    assert "def disconnect(" in inspect.getsource(SequenceAnalysisController)
    assert hasattr(SequenceRecordingController, "handle_load_imported_audio_requested")
    assert hasattr(SequenceRecordingController, "handle_cancel_imported_audio_requested")
    assert hasattr(RecordingModel, "begin_import")
    assert hasattr(RecordingModel, "retire_import")

    for owner in (
        SequenceAnalysisController,
        SequenceAnalysisModel,
        SequenceAnalysisView,
    ):
        source = inspect.getsource(owner)
        assert "handle_load_imported_audio_requested" not in source
        assert "handle_cancel_imported_audio_requested" not in source
        assert "active_import_id" not in source
        assert "import_runtime_consistent" not in source


def test_sequence_window_routes_import_commands_to_recording_controller():
    from ui.sequence.sequence_widget import SequenceWindow

    source = inspect.getsource(SequenceWindow.__init__)
    assert (
        "self.recording_controller.handle_load_imported_audio_requested" in source
    )
    assert (
        "self.recording_controller.handle_cancel_imported_audio_requested" in source
    )
    assert "self.analysis_controller.handle_load_imported_audio_requested" not in source
    assert "self.analysis_controller.handle_cancel_imported_audio_requested" not in source

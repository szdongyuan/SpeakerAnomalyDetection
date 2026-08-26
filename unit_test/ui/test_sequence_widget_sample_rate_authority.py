import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base import play_and_record
from ui.sequence import sequence_widget
from ui.sequence import sequence_recording_import_service as import_service_module
from ui.sequence.sequence_recording_import_owner import (
    SequenceRecordingImportController,
)
from ui.sequence.sequence_recording_import_service import SequenceImportedAudioService
from ui.sequence.sequence_recording_model import RecordingModel
from ui.sequence.sequence_messages import ConfigurationSnapshot, LoadImportedAudioRequested
from ui.sequence.sequence_recording_service import (
    RecordingReadinessRuntimeCapabilities,
    SequenceRecordingReadinessService,
)
from ui.sequence.sequence_recording_view import SequenceRecordingView


_DEFAULT_DEVICE = object()
EXPECTED_MISSING_SPEAKER_PROMPT = "未选择扬声器，请在【硬件-硬件选择】中选择扬声器。"


def _window(mode, detail, mic=_DEFAULT_DEVICE, speaker=_DEFAULT_DEVICE):
    obj = SimpleNamespace()
    obj.sequence_config = [{"seq1": {"acq": {"mode": mode, "detail": detail}}}]
    obj.mic = {"samplerate": 48000, "name": "mic", "hostapi": 1, "index": 1} if mic is _DEFAULT_DEVICE else mic
    obj.speaker = (
        {"samplerate": 48000, "name": "speaker", "hostapi": 1, "index": 2}
        if speaker is _DEFAULT_DEVICE
        else speaker
    )
    obj.mic_channels = [0]
    obj.data_struct = SimpleNamespace(
        sample_rate=None,
        stimulus_data=np.zeros(10),
        stimulus_info={"total_time": 0.01, "sample_rate": 44100},
        clear_data=lambda: None,
    )
    obj.lineedit_type = SimpleNamespace(text=lambda: "model")
    obj.lineedit_count = SimpleNamespace(text=lambda: "1")
    obj.lineedit_s_or_n = SimpleNamespace(text=lambda: "")
    obj._active_input_channels = []
    obj._build_current_wav_calibration_metadata = lambda: {"recorded_channels": []}
    obj.using_config_path = "sequence.json"
    return obj


class _Signal:
    def emit(self, _value):
        return None


def _execute_import(window, selected_path="input.wav"):
    snapshot = ConfigurationSnapshot(
        window.sequence_config,
        window.analysis_config,
        using_config_path=window.using_config_path,
    )
    command = LoadImportedAudioRequested(
        "compat-command",
        "compat-import",
        window.sequence_config[0]["seq1"]["acq"]["mode"],
        selected_path,
        snapshot,
    )
    view = SimpleNamespace(
        choose_import_audio_path=lambda selected: selected,
        present_import_warning=lambda title, message: sequence_widget.MessageBox.warning(
            window, title, message
        ),
        clear_import_projection=window._clear_plot_area,
        show_imported_audio=window.plot_waveform_to_workspace,
        set_import_data_enabled=window.data_btn.setEnabled,
    )
    bus = SimpleNamespace(
        events=SimpleNamespace(
            imported_audio_ready=_Signal(), imported_audio_failed=_Signal()
        )
    )
    service = SequenceImportedAudioService(
        audio_loader=import_service_module.load_audio_preserve_rate,
        metadata_reader=import_service_module.read_wav_calibration_metadata,
        reference_builder=import_service_module.set_data_struct_analysis_reference_signal,
    )
    controller = SequenceRecordingImportController(
        RecordingModel(), view, bus=bus, runtime=window, import_service=service
    )
    return controller.handle_load_imported_audio_requested(command)


def _recording_readiness(window, warnings, **service_overrides):
    service = SequenceRecordingReadinessService(
        runtime_capabilities_provider=lambda: RecordingReadinessRuntimeCapabilities(
            audio_devices_available=window.audio_devices_available,
            audio_devices_unavailable_message=getattr(
                window, "audio_devices_unavailable_message", ""
            ),
        ),
        view=SequenceRecordingView(
            present_readiness_warning=lambda title, text: warnings.append(
                (window, title, text)
            )
        ),
        **service_overrides,
    )
    configuration = ConfigurationSnapshot(
        sequence_config=window.sequence_config,
        analysis_config={},
        mic=window.mic,
        speaker=window.speaker,
        mic_channels=tuple(window.mic_channels),
    )
    command = SimpleNamespace(
        command_id="sample-rate-authority",
        configuration_generation=0,
    )
    return service(command, configuration)


def test_record_only_uses_mic_samplerate_not_detail_sample_rate():
    from ui.sequence.sequence_recording_controller import BlockingRecordingAdapter

    resolved = BlockingRecordingAdapter._resolved_sample_rate(
        "RECORD_ONLY",
        {"sample_rate": 44_100, "monitor_playback": False},
        {"samplerate": 48_000},
        None,
    )
    assert resolved == 48_000


def test_play_and_record_blocks_mismatched_samplerates():
    from ui.sequence.sequence_recording_controller import BlockingRecordingAdapter

    with pytest.raises(RuntimeError, match="采样率"):
        BlockingRecordingAdapter._resolved_sample_rate(
            "PLAY_AND_RECORD", {}, {"samplerate": 44_100}, {"samplerate": 48_000}
        )


def test_play_and_record_uses_device_samplerate_not_stimulus_config_rate():
    from ui.sequence.sequence_recording_controller import BlockingRecordingAdapter

    resolved = BlockingRecordingAdapter._resolved_sample_rate(
        "PLAY_AND_RECORD",
        {"sample_rate": 44_100},
        {"samplerate": 48_000},
        {"samplerate": 48_000},
    )
    assert resolved == 48_000


def test_record_only_monitor_blocks_mismatched_samplerates():
    from ui.sequence.sequence_recording_controller import BlockingRecordingAdapter

    with pytest.raises(RuntimeError, match="采样率"):
        BlockingRecordingAdapter._resolved_sample_rate(
            "RECORD_ONLY",
            {"monitor_playback": True},
            {"samplerate": 44_100},
            {"samplerate": 48_000},
        )


def test_record_only_without_monitor_does_not_require_speaker_samplerate():
    from ui.sequence.sequence_recording_controller import BlockingRecordingAdapter

    resolved = BlockingRecordingAdapter._resolved_sample_rate(
        "RECORD_ONLY",
        {"monitor_playback": False},
        {"samplerate": 48_000},
        None,
    )
    assert resolved == 48_000


def test_preflight_record_only_without_monitor_playback_does_not_require_speaker():
    win = _window(
        "RECORD_ONLY",
        {"sample_rate": 44100, "total_time": 1.0, "monitor_playback": False},
        mic={"samplerate": 48000, "hostapi": 1, "name": "mic"},
        speaker=None,
    )
    win.audio_devices_available = True
    warnings = []
    assert _recording_readiness(win, warnings) == (True, "")
    assert warnings == []


def test_preflight_record_only_string_false_monitor_does_not_require_speaker():
    win = _window(
        "RECORD_ONLY",
        {"sample_rate": 44100, "total_time": 1.0, "monitor_playback": "False"},
        mic={"samplerate": 48000, "hostapi": 1, "name": "mic"},
        speaker=None,
    )
    win.audio_devices_available = True
    warnings = []
    assert _recording_readiness(win, warnings) == (True, "")
    assert warnings == []


def test_preflight_record_only_monitor_playback_missing_speaker_uses_exact_prompt():
    win = _window(
        "RECORD_ONLY",
        {"sample_rate": 44100, "total_time": 1.0, "monitor_playback": True},
        mic={"samplerate": 48000, "hostapi": 1, "name": "mic"},
        speaker=None,
    )
    win.audio_devices_available = True
    warnings = []
    assert _recording_readiness(
        win,
        warnings,
        input_sample_rate_resolver=lambda *_args: pytest.fail(
            "sample-rate resolution must not run"
        ),
        duplex_sample_rate_resolver=lambda *_args: pytest.fail(
            "sample-rate resolution must not run"
        ),
    )[0] is False
    assert [args[1:] for args in warnings] == [("提示", EXPECTED_MISSING_SPEAKER_PROMPT)]


def test_preflight_play_and_record_missing_speaker_uses_exact_prompt():
    win = _window(
        "PLAY_AND_RECORD",
        {"stimulus_info": {"sample_rate": 44100, "total_time": 0.01}},
        mic={"samplerate": 48000, "hostapi": 1, "name": "mic"},
        speaker=None,
    )
    win.audio_devices_available = True
    warnings = []
    assert _recording_readiness(
        win,
        warnings,
        input_sample_rate_resolver=lambda *_args: pytest.fail(
            "sample-rate resolution must not run"
        ),
        duplex_sample_rate_resolver=lambda *_args: pytest.fail(
            "sample-rate resolution must not run"
        ),
    )[0] is False
    assert [args[1:] for args in warnings] == [("提示", EXPECTED_MISSING_SPEAKER_PROMPT)]


def test_record_only_invalid_mic_samplerate_warns_without_exception():
    from ui.sequence.sequence_recording_controller import BlockingRecordingAdapter

    with pytest.raises(RuntimeError):
        BlockingRecordingAdapter._resolved_sample_rate(
            "RECORD_ONLY",
            {"monitor_playback": False},
            {"samplerate": None},
            None,
        )


def test_start_streaming_recording_passes_real_reset_rate_to_record_writer_and_processor():
    calls = []
    prepared = SimpleNamespace(snapshot=SimpleNamespace(sample_rate=48_000))
    terminal = object()
    facade = SimpleNamespace(
        streaming_recording_service=SimpleNamespace(
            start=lambda received, port: calls.append((received, port)) or True
        )
    )

    assert sequence_widget.SequenceWindow._start_streaming_recording(
        facade, prepared, terminal, 48_000
    ) is True
    assert calls == [(prepared, terminal)]
    assert calls[0][0].snapshot.sample_rate == 48_000


def test_start_streaming_recording_passes_resolved_rate_to_playrec_writer_and_processor():
    calls = []
    prepared = SimpleNamespace(snapshot=SimpleNamespace(sample_rate=48_000))
    terminal = object()
    facade = SimpleNamespace(
        streaming_recording_service=SimpleNamespace(
            start=lambda received, port: calls.append((received, port)) or True
        )
    )

    assert sequence_widget.SequenceWindow._start_streaming_recording(
        facade, prepared, terminal, 48_000
    ) is True
    assert calls == [(prepared, terminal)]
    assert calls[0][0].snapshot.sample_rate == 48_000


def test_import_audio_preserves_file_native_samplerate(monkeypatch):
    win = _window("IMPORT_AUDIO", {"sample_rate": 44100})
    win.recorded_path = None
    win.recorded_signal_info = {}
    win.plot_waveform_to_workspace = lambda *args: None
    win._clear_plot_area = lambda: None
    win.data_btn = SimpleNamespace(setEnabled=lambda enabled: None)
    win.analysis_config = {"auto_analysis": False}
    load_calls = []
    metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True}
        ]
    }
    monkeypatch.setattr(import_service_module, "read_wav_calibration_metadata", lambda path, logger=None: metadata)
    monkeypatch.setattr(
        import_service_module,
        "load_audio_preserve_rate",
        lambda path, mono=False: load_calls.append((path, mono))
        or (np.zeros((2, 320), dtype=np.float32), 32000),
    )

    _execute_import(win)

    assert win.data_struct.sample_rate == 32000
    assert load_calls[0] == ("input.wav", False)


def test_import_stimulus_audio_generates_temporary_reference_at_recording_rate(monkeypatch):
    win = _window("IMPORT_STIMULUS_AUDIO", {"stimulus_info": {"sample_rate": 44100, "total_time": 0.01}})
    win.recorded_path = None
    win.recorded_signal_info = {}
    win.plot_waveform_to_workspace = lambda *args: None
    win._clear_plot_area = lambda: None
    win.data_btn = SimpleNamespace(setEnabled=lambda enabled: None)
    win.analysis_config = {"auto_analysis": False}
    calls = []
    metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "v2pa_factor": 2.5, "standard_spl": 94.0, "calibrated": True}
        ]
    }
    monkeypatch.setattr(import_service_module, "read_wav_calibration_metadata", lambda path, logger=None: metadata)
    monkeypatch.setattr(
        import_service_module,
        "load_audio_preserve_rate",
        lambda path, mono=False: (np.zeros((2, 320), dtype=np.float32), 32000),
    )
    monkeypatch.setattr(
        import_service_module,
        "set_data_struct_analysis_reference_signal",
        lambda data_struct, detail, using_config_path=None, *, runtime_sample_rate, **kwargs: calls.append(
            (detail, runtime_sample_rate)
        )
        or True,
    )

    _execute_import(win)

    assert win.data_struct.sample_rate == 32000
    assert calls == [(win.sequence_config[0]["seq1"]["acq"]["detail"], 32000)]

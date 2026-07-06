from types import SimpleNamespace

from consts import error_code
from ui.archive_audio_data_dialog import ArchiveAudioDataDialog


def _dialog(speaker):
    calls = []
    return SimpleNamespace(
        speaker=speaker,
        playback_controller=SimpleNamespace(
            start_audio_playback=lambda path, device=None, output_sample_rate=None: calls.append(
                (path, device, output_sample_rate)
            )
            or (error_code.OK, "ok"),
        ),
        _playback_calls=calls,
    )


def test_archive_playback_passes_selected_speaker_samplerate():
    dialog = _dialog({"index": 7, "samplerate": 48000, "name": "speaker"})

    code, _ = ArchiveAudioDataDialog._start_resolved_audio_playback(dialog, "audio.wav")

    assert code == error_code.OK
    assert dialog._playback_calls == [("audio.wav", 7, 48000)]


def test_archive_playback_fails_without_valid_speaker_samplerate():
    dialog = _dialog({"index": 7, "default_samplerate": 44100, "name": "speaker"})

    code, msg = ArchiveAudioDataDialog._start_resolved_audio_playback(dialog, "audio.wav")

    assert code != error_code.OK
    assert "采样率" in msg or "sample rate" in msg.lower()
    assert dialog._playback_calls == []


def test_archive_playback_rejects_non_integral_speaker_samplerate():
    dialog = _dialog({"index": 7, "samplerate": 48000.5, "name": "speaker"})

    code, msg = ArchiveAudioDataDialog._start_resolved_audio_playback(dialog, "audio.wav")

    assert code != error_code.OK
    assert "采样率" in msg or "sample rate" in msg.lower()
    assert dialog._playback_calls == []

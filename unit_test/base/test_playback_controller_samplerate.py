import numpy as np
import pytest

from base.playback_controller import PlaybackController
from consts import error_code


def _inactive_stream():
    return type(
        "S",
        (),
        {"active": False, "closed": False, "close": lambda self, ignore_errors=True: None},
    )()


def test_playback_controller_uses_requested_output_samplerate(monkeypatch):
    controller = PlaybackController()
    monkeypatch.setattr(
        controller,
        "_load_playback_audio",
        lambda path: (np.zeros((10, 1), dtype=np.float32), 44100),
    )
    calls = {}
    monkeypatch.setattr("base.playback_controller.os.path.isfile", lambda path: True)
    monkeypatch.setattr(
        "base.playback_controller.sd.play",
        lambda data, samplerate, device=None, blocking=False: calls.update(
            samplerate=samplerate,
            device=device,
            frames=len(data),
        ),
    )
    monkeypatch.setattr("base.playback_controller.sd.get_stream", _inactive_stream)

    code, _ = controller.start_audio_playback("file.wav", device=3, output_sample_rate=48000)

    assert code == error_code.OK
    assert calls["samplerate"] == 48000
    assert calls["device"] == 3


def test_playback_controller_resamples_when_file_rate_differs(monkeypatch):
    controller = PlaybackController()
    source = np.zeros((10, 1), dtype=np.float32)
    resampled = np.zeros((20, 1), dtype=np.float32)
    monkeypatch.setattr(controller, "_load_playback_audio", lambda path: (source, 24000))
    monkeypatch.setattr(controller, "_resample_audio", lambda data, from_rate, to_rate: resampled)
    calls = {}
    monkeypatch.setattr("base.playback_controller.os.path.isfile", lambda path: True)
    monkeypatch.setattr(
        "base.playback_controller.sd.play",
        lambda data, samplerate, device=None, blocking=False: calls.update(
            data=data,
            samplerate=samplerate,
        ),
    )
    monkeypatch.setattr("base.playback_controller.sd.get_stream", _inactive_stream)

    code, _ = controller.start_audio_playback("file.wav", output_sample_rate=48000)

    assert code == error_code.OK
    assert calls["samplerate"] == 48000
    assert calls["data"] is resampled


def test_playback_controller_fails_without_output_samplerate_by_default(monkeypatch):
    controller = PlaybackController()
    monkeypatch.setattr(
        controller,
        "_load_playback_audio",
        lambda path: (np.zeros((10, 1), dtype=np.float32), 44100),
    )
    monkeypatch.setattr("base.playback_controller.os.path.isfile", lambda path: True)

    code, msg = controller.start_audio_playback("file.wav")

    assert code != error_code.OK
    assert "采样率" in msg or "sample rate" in msg.lower()


def test_playback_controller_rejects_non_integral_output_samplerate(monkeypatch):
    controller = PlaybackController()
    monkeypatch.setattr(
        controller,
        "_load_playback_audio",
        lambda path: (np.zeros((10, 1), dtype=np.float32), 44100),
    )
    monkeypatch.setattr("base.playback_controller.os.path.isfile", lambda path: True)
    calls = []
    monkeypatch.setattr("base.playback_controller.sd.play", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr("base.playback_controller.sd.get_stream", _inactive_stream)

    code, msg = controller.start_audio_playback("file.wav", output_sample_rate=48000.5)

    assert code != error_code.OK
    assert "采样率" in msg or "sample rate" in msg.lower()
    assert calls == []


@pytest.mark.parametrize("sample_rate", [float("inf"), float("nan")])
def test_playback_controller_rejects_non_finite_output_samplerate(monkeypatch, sample_rate):
    controller = PlaybackController()
    monkeypatch.setattr(
        controller,
        "_load_playback_audio",
        lambda path: (np.zeros((10, 1), dtype=np.float32), 44100),
    )
    monkeypatch.setattr("base.playback_controller.os.path.isfile", lambda path: True)
    calls = []
    monkeypatch.setattr("base.playback_controller.sd.play", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr("base.playback_controller.sd.get_stream", _inactive_stream)

    code, msg = controller.start_audio_playback("file.wav", output_sample_rate=sample_rate)

    assert code != error_code.OK
    assert "采样率" in msg or "sample rate" in msg.lower()
    assert calls == []

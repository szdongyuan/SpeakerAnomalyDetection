import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import Mock

from base.playback_controller import PlaybackController
from base import playback_controller as module
from base.sound_device_manager import SoundDeviceManager
from consts import error_code


def test_prepare_playback_audio_downmixes_multichannel_to_stereo():
    audio = np.array(
        [
            [1.0, 3.0, 5.0, 7.0],
            [2.0, 4.0, 6.0, 8.0],
        ],
        dtype=np.float32,
    )

    playback_audio = PlaybackController._prepare_playback_audio(audio, output_max_channels=2)

    assert playback_audio.shape == (2, 2)
    np.testing.assert_allclose(playback_audio[:, 0], [4.0, 5.0])
    np.testing.assert_allclose(playback_audio[:, 1], [4.0, 5.0])


def test_prepare_playback_audio_downmixes_to_mono_for_single_channel_output():
    audio = np.array(
        [
            [1.0, 3.0],
            [2.0, 4.0],
        ],
        dtype=np.float32,
    )

    playback_audio = PlaybackController._prepare_playback_audio(audio, output_max_channels=1)

    assert playback_audio.shape == (2, 1)
    np.testing.assert_allclose(playback_audio[:, 0], [2.0, 3.0])


def test_prepare_playback_audio_keeps_stereo_when_output_supports_it():
    audio = np.array(
        [
            [1.0, 3.0],
            [2.0, 4.0],
        ],
        dtype=np.float32,
    )

    playback_audio = PlaybackController._prepare_playback_audio(audio, output_max_channels=2)

    assert playback_audio.shape == (2, 2)
    np.testing.assert_allclose(playback_audio, audio)


@pytest.fixture
def playback(tmp_path, monkeypatch):
    path = tmp_path / "audio.wav"
    path.touch()
    controller = PlaybackController()
    monkeypatch.setattr(controller, "_load_playback_audio", lambda _: (np.ones((4, 4)), 8000))
    monkeypatch.setattr(module.threading, "Thread", Mock())
    stream = SimpleNamespace(active=True, closed=False, close=Mock(), stop=Mock())
    unrelated_recorder = SimpleNamespace(close=Mock(), stop=Mock())
    fake = SimpleNamespace(play=Mock(), get_stream=Mock(return_value=stream), stop=Mock(),
                           _terminate=Mock(), _initialize=Mock(), query_devices=Mock())
    monkeypatch.setattr(module, "sd", fake)
    defaults = Mock(side_effect=[(error_code.OK, {"index": 2, "max_output_channels": 2}),
                                (error_code.OK, {"index": 3, "max_output_channels": 1})])
    monkeypatch.setattr(SoundDeviceManager, "get_default_device", defaults)
    return controller, str(path), fake, defaults, unrelated_recorder


def test_file_playback_freezes_default_for_capability_and_mono_retry(playback):
    controller, path, fake, defaults, _ = playback
    fake.play.side_effect = [RuntimeError("stereo rejected"), None, None]
    assert controller.start_audio_playback(path, device=1)[0] == error_code.OK
    assert [(call.kwargs["device"], call.args[0].shape[1]) for call in fake.play.call_args_list] == [(2, 2), (2, 1)]
    assert defaults.call_count == 1
    controller.stop_audio_playback()
    assert controller.start_audio_playback(path, device=1)[0] == error_code.OK
    assert fake.play.call_args.kwargs["device"] == 3
    assert fake.play.call_args.args[0].shape[1] == 1
    assert defaults.call_count == 2
    fake.query_devices.assert_not_called()
    fake._terminate.assert_not_called()
    fake._initialize.assert_not_called()


@pytest.mark.parametrize("failure", ["device removed", "stream open rejected", "channels rejected"])
def test_file_playback_failure_resets_owned_state_without_fallback_or_global_stop(playback, failure):
    controller, path, fake, defaults, recorder = playback
    # The backend's last convenience stream belongs to a recorder, not us.
    fake.get_stream.return_value = recorder
    fake.play.side_effect = RuntimeError(failure)
    code, message = controller.start_audio_playback(path, device=1)
    assert code == error_code.INVALID_PLAY and failure in message
    assert [call.kwargs["device"] for call in fake.play.call_args_list] == [2, 2]
    assert defaults.call_count == 1
    assert not controller.is_audio_playing()
    assert controller.get_current_playing_file() is None
    assert controller._playback_stream is None
    fake.get_stream.assert_not_called()
    fake.stop.assert_not_called()
    fake._terminate.assert_not_called()
    recorder.close.assert_not_called()
    recorder.stop.assert_not_called()


def test_file_playback_no_output_fails_before_open(playback):
    controller, path, fake, defaults, _ = playback
    defaults.side_effect = [(error_code.MISSING_HARDWARE_DEVICE, None)]
    code, message = controller.start_audio_playback(path, device=1)
    assert code == error_code.INVALID_PLAY and "default output" in message.lower()
    fake.play.assert_not_called()
    assert not controller.is_audio_playing()

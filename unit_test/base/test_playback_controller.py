import mock
import numpy as np

from consts import error_code


class FakeStream(object):
    def __init__(self, active=True, closed=False):
        self.active = active
        self.closed = closed
        self.stop_called = False
        self.close_called = False

    def stop(self, ignore_errors=True):
        self.stop_called = True
        self.active = False

    def close(self, ignore_errors=True):
        self.close_called = True
        self.closed = True


class TestPlaybackController(object):
    @mock.patch("base.playback_controller.os.path.isfile")
    def test_start_audio_playback_returns_invalid_path_when_file_missing(self, mock_isfile):
        from base.playback_controller import PlaybackController

        mock_isfile.return_value = False

        result = PlaybackController().start_audio_playback("missing.wav")

        assert result[0] == error_code.INVALID_PATH

    @mock.patch("base.playback_controller.threading.Thread")
    @mock.patch("base.playback_controller.sd.get_stream")
    @mock.patch("base.playback_controller.sd.play")
    @mock.patch("base.playback_controller.sd.query_devices")
    @mock.patch("base.playback_controller.os.path.isfile")
    def test_start_audio_playback_sets_playback_state(
        self,
        mock_isfile,
        mock_query_devices,
        mock_play,
        mock_get_stream,
        mock_thread,
    ):
        from base.playback_controller import PlaybackController

        mock_isfile.return_value = True
        mock_query_devices.return_value = {"max_output_channels": 2}
        fake_stream = FakeStream(active=True, closed=False)
        mock_get_stream.return_value = fake_stream
        mock_thread.return_value = mock.Mock()
        controller = PlaybackController()
        controller._load_playback_audio = mock.Mock(return_value=(np.ones((4, 1), dtype=np.float32), 44100))

        result = controller.start_audio_playback("exists.wav")

        assert result == (error_code.OK, "Audio playback started.")
        assert controller.is_audio_playing() is True
        assert controller.get_current_playing_file().endswith("exists.wav")
        mock_play.assert_called_once()

    @mock.patch("base.playback_controller.threading.Thread")
    @mock.patch("base.playback_controller.sd.get_stream")
    @mock.patch("base.playback_controller.sd.play")
    @mock.patch("base.playback_controller.sd.query_devices")
    @mock.patch("base.playback_controller.os.path.isfile")
    def test_start_audio_playback_downmixes_multichannel_audio_for_stereo_output(
        self,
        mock_isfile,
        mock_query_devices,
        mock_play,
        mock_get_stream,
        mock_thread,
    ):
        from base.playback_controller import PlaybackController

        mock_isfile.return_value = True
        mock_query_devices.return_value = {"max_output_channels": 2}
        mock_get_stream.return_value = FakeStream(active=True, closed=False)
        mock_thread.return_value = mock.Mock()
        controller = PlaybackController()
        controller._load_playback_audio = mock.Mock(
            return_value=(np.arange(24, dtype=np.float32).reshape(6, 4), 44100)
        )

        result = controller.start_audio_playback("exists.wav")

        assert result == (error_code.OK, "Audio playback started.")
        play_data = mock_play.call_args[0][0]
        assert play_data.shape == (6, 2)

    @mock.patch("base.playback_controller.threading.Thread")
    @mock.patch("base.playback_controller.sd.get_stream")
    @mock.patch("base.playback_controller.sd.play")
    @mock.patch("base.playback_controller.sd.query_devices")
    @mock.patch("base.playback_controller.os.path.isfile")
    def test_start_audio_playback_downmixes_to_mono_for_single_channel_output(
        self,
        mock_isfile,
        mock_query_devices,
        mock_play,
        mock_get_stream,
        mock_thread,
    ):
        from base.playback_controller import PlaybackController

        mock_isfile.return_value = True
        mock_query_devices.return_value = {"max_output_channels": 1}
        mock_get_stream.return_value = FakeStream(active=True, closed=False)
        mock_thread.return_value = mock.Mock()
        controller = PlaybackController()
        controller._load_playback_audio = mock.Mock(
            return_value=(np.arange(24, dtype=np.float32).reshape(6, 4), 44100)
        )

        result = controller.start_audio_playback("exists.wav")

        assert result == (error_code.OK, "Audio playback started.")
        play_data = mock_play.call_args[0][0]
        assert play_data.shape == (6, 1)

    @mock.patch("base.playback_controller.os.path.isfile")
    def test_stop_audio_playback_stops_stream_and_clears_state(self, mock_isfile):
        from base.playback_controller import PlaybackController

        mock_isfile.return_value = True
        controller = PlaybackController()
        controller._load_playback_audio = mock.Mock(return_value=(np.ones((4, 1), dtype=np.float32), 44100))
        controller._playback_is_running = True
        controller._playback_current_file = "exists.wav"
        controller._playback_stream = FakeStream(active=True, closed=False)

        result = controller.stop_audio_playback()

        assert result == (error_code.OK, "Audio playback stopped.")
        assert controller.is_audio_playing() is False
        assert controller.get_current_playing_file() is None

    def test_is_audio_playing_resets_state_when_stream_inactive(self):
        from base.playback_controller import PlaybackController

        controller = PlaybackController()
        controller._playback_is_running = True
        controller._playback_current_file = "exists.wav"
        controller._playback_stream = FakeStream(active=False, closed=False)

        result = controller.is_audio_playing()

        assert result is False
        assert controller.get_current_playing_file() is None

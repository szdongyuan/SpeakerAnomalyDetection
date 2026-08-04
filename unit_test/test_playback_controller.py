import numpy as np

from base.playback_controller import PlaybackController


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

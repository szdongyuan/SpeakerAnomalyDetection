from unittest import mock
import numpy as np
import pytest

from base.soundcard_audio_processor import SoundcardAudioProcessor
from consts import error_code


class TestSoundcardAudioProcessor(object):

    test_path = "base.soundcard_audio_processor.SoundcardAudioProcessor"

    @pytest.mark.parametrize("corr_ret, stimulus_signal, recorded_signal, result_set", [
        (1, [1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1], -4),
        (0, [1, 1, 1], [1, 1, 1], -2),
        (0, [], [], 1),
    ])
    @mock.patch("scipy.signal.correlate")
    def test_calculate_alignment(self, mock_corr, corr_ret, stimulus_signal, recorded_signal, result_set):
        mock_corr.return_value = corr_ret
        result = SoundcardAudioProcessor().calculate_alignment(stimulus_signal, recorded_signal)
        assert result == result_set

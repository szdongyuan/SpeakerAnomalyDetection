from unittest import mock
import numpy as np
import pytest

from matplotlib import pyplot as plt

from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from unit_test.compare_methods import compare_dicts

test_path = "base.pre_processing.audio_thd_frequency_response_analysis"


class TestAudioThdFrequencyResponseAnalysis:
    @pytest.mark.parametrize("input_ret, result_ret", [
        ({"fftfreq_ret": np.array([1, 2, 3]),
          "fft_ret": np.array([2, 3, 4]),
          "smooth_curve_ret": np.array([1, 1, 1]),
          "reference_signal": np.array(range(1024)),
          "recorded_signal": np.array(range(1024)),
          "sr": 44100,
          "kwargs": {"smooth": True, "delay_frames": 123}},
         (np.array([1, 1, 1]), np.array([1, 2, 3]))),

        ({"fftfreq_ret": np.array([1, 2, 3]),
          "fft_ret": np.array([2, 3, 4]),
          "smooth_curve_ret": np.array([1, 1, 1]),
          "reference_signal": np.array(range(1024)),
          "recorded_signal": np.array(range(1024)),
          "sr": 44100,
          "kwargs": {"smooth": False, "delay_frames": 123}},
         (np.array([0, 0, 0]), np.array([1, 2, 3]))),
    ])
    @mock.patch(test_path + ".AudioThdFrequencyResponseAnalysis.smooth_curve")
    @mock.patch(test_path + ".np.fft.fft")
    @mock.patch(test_path + ".np.fft.fftfreq")
    def test_calculate_fr(self, mock_fftfreq, mock_fft, mock_smooth_curve, input_ret, result_ret):
        mock_fftfreq.return_value = input_ret["fftfreq_ret"]
        mock_fft.return_value = input_ret["fft_ret"]
        mock_smooth_curve.return_value = input_ret["smooth_curve_ret"]
        atf = AudioThdFrequencyResponseAnalysis()
        result = atf.calculate_fr(input_ret["reference_signal"],
                                  input_ret["recorded_signal"],
                                  input_ret["sr"],
                                  **input_ret["kwargs"])
        assert np.array_equal(result, result_ret)

    @pytest.mark.parametrize("oct_width_ret, result_ret", [
        (1 / 3, np.array([10, 20, 30, 40, 50])),
        (1 / 6, np.array([10, 20, 30, 40, 50])),
        (1, np.array([10, 20, 35, 40, 45])),
    ])
    def test_smooth_curve(self, oct_width_ret, result_ret):
        frequencies_ret = np.array([100, 200, 300, 400, 500])
        data_ret = np.array([10, 20, 30, 40, 50])
        atf = AudioThdFrequencyResponseAnalysis()
        result = atf.smooth_curve(frequencies_ret, data_ret, oct_width_ret)
        assert np.array_equal(result, result_ret)
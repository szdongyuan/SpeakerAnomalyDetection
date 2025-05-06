import mock
import numpy as np
import pytest

from matplotlib import pyplot as plt

from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from unit_test.compare_methods import compare_dicts


class TestAudioThdFrequencyResponseAnalysis(object):

    test_path = "base.pre_processing.audio_thd_frequency_response_analysis"

    @pytest.mark.parametrize("input_ret, result_ret", [
        # test_case1:
        ({"subplots_ret": [(mock.Mock(), mock.Mock()), (mock.Mock(), mock.Mock()), (mock.Mock(), mock.Mock())],
          "reference_signal": np.array([1, 1]),
          "recorded_signal": np.array([1, 3]),
          "kwargs": {"thd": True, "frequency_response": True},
          "calculate_thd_ret": ([1, 2], np.array([3, 4]), [5, 6]),
          "calculate_fr_ret": (np.array([7, 8]), np.array([9, 10])),
          "plot_thd_ret": mock.Mock(),
          "plot_harmonic_ret": mock.Mock(),
          "plot_frequency_response_ret": mock.Mock()},
         ({"subplots_callcount": 3,
           "calculate_thd_callcount": 2,
           "calculate_fr_callcount": 2,
           "plot_thd_callcount": 2,
           "plot_harmonic_callcount": 2,
           "plot_frequency_response_callcount": 2,
           "plt": plt})
         ),
        # test_case2:
        ({"subplots_ret": [(mock.Mock(), mock.Mock()), (mock.Mock(), mock.Mock()), (mock.Mock(), mock.Mock())],
          "reference_signal": np.array([1, 1]),
          "recorded_signal": np.array([1, 3]),
          "kwargs": {"thd": True, "frequency_response": False},
          "calculate_thd_ret": ([1, 2], np.array([3, 4]), [5, 6]),
          "calculate_fr_ret": [],
          "plot_thd_ret": mock.Mock(),
          "plot_harmonic_ret": mock.Mock(),
          "plot_frequency_response_ret": []},
         ({"subplots_callcount": 2,
           "calculate_thd_callcount": 2,
           "calculate_fr_callcount": 0,
           "plot_thd_callcount": 2,
           "plot_harmonic_callcount": 2,
           "plot_frequency_response_callcount": 0,
           "plt": plt})
         ),
        # test_case3:
        ({"subplots_ret": [(mock.Mock(), mock.Mock()), (mock.Mock(), mock.Mock()), (mock.Mock(), mock.Mock())],
          "reference_signal": np.array([1, 1]),
          "recorded_signal": np.array([1, 3]),
          "kwargs": {"thd": False, "frequency_response": True},
          "calculate_thd_ret": [],
          "calculate_fr_ret": (np.array([7, 8]), np.array([9, 10])),
          "plot_thd_ret": [],
          "plot_harmonic_ret": [],
          "plot_frequency_response_ret": mock.Mock()},
         ({"subplots_callcount": 1,
           "calculate_thd_callcount": 0,
           "calculate_fr_callcount": 2,
           "plot_thd_callcount": 0,
           "plot_harmonic_callcount": 0,
           "plot_frequency_response_callcount": 2,
           "plt": plt})
         ),
        # test_case4:
        ({"subplots_ret": [],
          "reference_signal": np.array([1, 1]),
          "recorded_signal": np.array([1, 3]),
          "kwargs": {"thd": False, "frequency_response": False},
          "calculate_thd_ret": [],
          "calculate_fr_ret": [],
          "plot_thd_ret": [],
          "plot_harmonic_ret": [],
          "plot_frequency_response_ret": []},
         ({"subplots_callcount": 0,
           "calculate_thd_callcount": 0,
           "calculate_fr_callcount": 0,
           "plot_thd_callcount": 0,
           "plot_harmonic_callcount": 0,
           "plot_frequency_response_callcount": 0,
           "plt": plt})
         ),
    ])
    @mock.patch(test_path + ".plot_frequency_response")
    @mock.patch(test_path + ".plot_harmonic")
    @mock.patch(test_path + ".plot_thd")
    @mock.patch(test_path + ".AudioThdFrequencyResponseAnalysis.calculate_fr")
    @mock.patch(test_path + ".AudioThdFrequencyResponseAnalysis.calculate_thd")
    @mock.patch(test_path + ".plt.subplots")
    def test_process_calculate(self, mock_subplots, mock_calculate_thd, mock_calculate_fr, mock_plot_thd,
                               mock_plot_harmonic, mock_plot_frequency_response, input_ret, result_ret):
        mock_subplots.side_effect = input_ret["subplots_ret"]
        mock_calculate_thd.return_value = input_ret["calculate_thd_ret"]
        mock_calculate_fr.return_value = input_ret["calculate_fr_ret"]
        mock_plot_thd.return_value = input_ret["plot_thd_ret"]
        mock_plot_harmonic.return_value = input_ret["plot_harmonic_ret"]
        mock_plot_frequency_response.return_value = input_ret["plot_frequency_response_ret"]
        sr = list(range(len(input_ret["recorded_signal"])))
        atf = AudioThdFrequencyResponseAnalysis()
        result = atf.process_calculate(input_ret["reference_signal"],
                                       input_ret["recorded_signal"],
                                       sr,
                                       **input_ret["kwargs"])
        assert mock_subplots.call_count == result_ret["subplots_callcount"]
        assert mock_calculate_thd.call_count == result_ret["calculate_thd_callcount"]
        assert mock_calculate_fr.call_count == result_ret["calculate_fr_callcount"]
        assert mock_plot_thd.call_count == result_ret["plot_thd_callcount"]
        assert mock_plot_harmonic.call_count == result_ret["plot_harmonic_callcount"]
        assert mock_plot_frequency_response.call_count == result_ret["plot_frequency_response_callcount"]
        assert result == result_ret["plt"]

    @pytest.mark.parametrize("get_harmonic_ret, reference_signal, recorded_signal, sr", [
        (({10: {"bf_v": 20, "yf": 0.123, "harmonic": [1, 0.2, 0.3]},
           20: {"bf_v": 30, "yf": 0.345, "harmonic": [0.5, 0.8, 0.3]}},
          [10, 20]),
         np.array([1, 2]),
         np.array([3, 4]),
         44100),
    ])
    @mock.patch(
        "base.pre_processing.audio_thd_frequency_response_analysis.AudioThdFrequencyResponseAnalysis.get_harmonic")
    def test_calculate_thd(self, mock_get_harmonic, get_harmonic_ret, reference_signal, recorded_signal, sr):
        mock_get_harmonic.return_value = get_harmonic_ret
        atf = AudioThdFrequencyResponseAnalysis()
        plot_x, plot_h, plot_thd = atf.calculate_thd(reference_signal, recorded_signal, sr)
        assert plot_x == [10, 20]
        assert np.array_equal(plot_h, np.array([[1, 0.5], [0.2, 0.8], [0.3, 0.3], [0, 0], [0, 0], [0, 0]]))
        assert plot_thd == [0.3391817326856071, 0.8630747123996122]

    @pytest.mark.parametrize("input_ret, result_ret", [
        ({"fftfreq_ret": np.array(range(1024)),
          "fft_ret": np.array(range(1024)),
          "reference_signal": np.array(range(1024)),
          "recorded_signal": np.array(range(1024)),
          "sr": 5100,
          "kwargs": {"gap_len": 10, "delay_frames": 512}},
         ({254: {"bf_v": 254, "yf": np.array(range(0, 255, 1)), "harmonic": [254]}},
          [254, 254]))
    ])
    @mock.patch(test_path + ".np.fft.fft")
    @mock.patch(test_path + ".np.fft.fftfreq")
    def test_get_harmonic(self, mock_fftfreq, mock_fft, input_ret, result_ret):
        mock_fftfreq.return_value = input_ret["fftfreq_ret"]
        mock_fft.return_value = input_ret["fft_ret"]
        atf = AudioThdFrequencyResponseAnalysis()
        result = atf.get_harmonic(input_ret["reference_signal"],
                                  input_ret["recorded_signal"],
                                  input_ret["sr"],
                                  **input_ret["kwargs"])
        # assert result[1] == result_ret[1]
        assert compare_dicts(result[0], result_ret[0])

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
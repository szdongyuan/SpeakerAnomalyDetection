import mock
import numpy as np
import pytest

from base.soundcard_control import SoundcardControl
from consts import error_code
from unit_test.compare_methods import assert_equal


class TestSoundcardControl(object):

    test_path = "base.soundcard_control.SoundcardControl"
    @pytest.mark.parametrize("add_set, predict_set, amplitude_voltage_data, target_voltage, result_set", [
        ([], [], [], 3, (error_code.INVALID_DATA_LOADING, "Input data is empty.")),
        (mock.Mock(), 0.1503, [(0.1, 2), (0.2, 3.99), (0.3, 5.97)], 3, 0.1503),
    ])
    @mock.patch(test_path + ".predict_amplitude")
    @mock.patch(test_path + ".add_data")
    def test_calibrate_amplitude(self, mock_add, mock_predict, add_set, predict_set, amplitude_voltage_data,
                                 target_voltage, result_set):
        mock_add.return_value = add_set
        mock_predict.return_value = predict_set
        result = SoundcardControl().calibrate_amplitude(amplitude_voltage_data, target_voltage)
        assert result == result_set

    @pytest.mark.parametrize("amplitude, voltage, result_set", [
        ([], 3, (error_code.INVALID_DATA_LOADING, "Input data cannot be None.")),
        (0.1, [], (error_code.INVALID_DATA_LOADING, "Input data cannot be None.")),
        ([0.1], 3, (error_code.INVALID_TYPE_DATA, "Input data must be numeric.")),
        (0.1, 3, (error_code.OK, "Successfully add data.")),
    ])
    def test_add_data(self, amplitude, voltage, result_set):
        result = SoundcardControl().add_data(amplitude, voltage)
        assert result == result_set

    @pytest.mark.parametrize("amplitudes_set, voltages_set, result_set", [
        ([], [1, 2], (error_code.INVALID_DATA_LOADING, "Amplitudes and voltages must not be empty.")),
        ([0.1, 0.2], [], (error_code.INVALID_DATA_LOADING, "Amplitudes and voltages must not be empty.")),
        ([0.1], [1, 2], (error_code.INVALID_DATA_LOADING, "Amplitudes and voltages must have the same length.")),
        ([0.1, 0.2, 0.3], [2, 3.99, 5.97], (error_code.OK, np.poly1d([0.05037773, -0.00083921]))),
    ])
    def test_fit(self, amplitudes_set, voltages_set, result_set):
        sc = SoundcardControl()
        setattr(sc, "amplitudes", amplitudes_set)
        setattr(sc, "voltages", voltages_set)
        result = sc.fit()
        if isinstance(result[1], str):
            assert result == result_set
        else:
            assert result[0] == result_set[0]
            assert_equal(result[1].coefficients, result_set[1].coefficients)

    @pytest.mark.parametrize("fit_function, target_voltage, result_ret", [
        (np.poly1d([0.05038, -0.0008392]), 3, 0.1503),
        (np.poly1d([0.05038, -0.0008392]), [1, 2], [0.0495, 0.0999]),
        (np.poly1d([0.05038, -0.0008392]), [], []),
    ])
    def test_predict_amplitude(self, fit_function, target_voltage, result_ret):
        result = SoundcardControl().predict_amplitude(fit_function, target_voltage)
        assert_equal(result, result_ret)

    @pytest.mark.parametrize("input_ret, result_ret", [
        ({"directory_ret": mock.Mock(),
          "mic_ret": [],
          "speaker_ret": [],
          "start_process_set": [],
          "join_process_set": [],
          "process_call_count": 0,
          "record_dict": [],
          "stimulus_dict": {"data": [1, 2, 3], "sr": 44100}},
         (error_code.INVALID_DATA_LOADING, "The record_dict is empty or invalid.")
         ),
        ({"directory_ret": mock.Mock(),
          "mic_ret": [],
          "speaker_ret": [],
          "start_process_set": [],
          "join_process_set": [],
          "process_call_count": 0,
          "record_dict": {"sr": 44100, "channels": 1},
          "stimulus_dict": [1, 2]},
         (error_code.INVALID_DATA_LOADING, "The stimulus_dict is empty or invalid.")
         ),
        ({"directory_ret": mock.Mock(),
          "mic_ret": mock.Mock(),
          "speaker_ret": mock.Mock(),
          "start_process_set": [(error_code.OK, "Successful start process."),
                                (error_code.OK, "Successful start process.")],
          "join_process_set": [(error_code.OK, "Waiting process successful."),
                               (error_code.OK, "Waiting process successful.")],
          "process_call_count": 2,
          "record_dict": {"sr": 44100, "channels": 1},
          "stimulus_dict": {"data": [1, 2, 3], "sr": 44100}},
         (error_code.OK, "All processes have finished.")
         ),
    ])
    @mock.patch(test_path + ".join_process")
    @mock.patch(test_path + ".start_process")
    @mock.patch(test_path + ".speaker_worker")
    @mock.patch(test_path + ".mic_worker")
    @mock.patch(test_path + ".ensure_directory_exists")
    def test_initialize_audio_processes(self, mock_directory, mock_mic, mock_speaker, mock_start_process,
                                        mock_join_process, input_ret, result_ret):
        mock_directory.return_value = input_ret["directory_ret"]
        mock_mic.return_value = input_ret["mic_ret"]
        mock_speaker.return_value = input_ret["speaker_ret"]
        mock_start_process.side_effect = input_ret["start_process_set"]
        mock_join_process.side_effect = input_ret["join_process_set"]
        result = SoundcardControl().initialize_audio_processes(input_ret["record_dict"], input_ret["stimulus_dict"])
        assert result == result_ret
        assert mock_start_process.call_count == input_ret["process_call_count"]
        assert mock_join_process.call_count == input_ret["process_call_count"]

    @pytest.mark.parametrize("speaker_set, play_set, stimulus_params, result_set", [
        (mock.Mock(), mock.Mock(), {"data": [1, 2, 3], "sr": 44100}, None),
        (Exception('xxx'), mock.Mock(), {"data": [1, 2, 3], "sr": 44100},
         (error_code.INVALID_PLAY, "Failed to play audio.'Exception' object has no attribute 'play'")),
    ])
    @mock.patch("soundcard.default_speaker")
    def test_speaker_worker(self, mock_speaker, speaker_set, play_set, stimulus_params, result_set):
        mock_speaker.return_value = speaker_set
        mock_speaker.play = play_set
        result = SoundcardControl().speaker_worker(stimulus_params)
        assert result == result_set

    @pytest.mark.parametrize("input_ret, result_ret", [
            ({"default_mic_ret": mock.Mock(),
              "record_ret": np.array([[1], [0], [0], [1], [0]]),
              "alignment_ret": 1,
              "wavfile_ret": mock.Mock(),
              "record_params": {"num_frames": 100, "sr": 44100, "channels": 1, "prolong_frames": 2},
              "stimulus_params": {"data": np.array([1, 2, 3]), "sr": 44100}},
             (error_code.OK, "Recording and stimulus saved.")
             ),
            ({"default_mic_ret": mock.Mock(),
              "record_ret": np.array([[1], [0], [0], [1], [0]]),
              "alignment_ret": 3,
              "wavfile_ret": mock.Mock(),
              "record_params": {"num_frames": 100, "sr": 44100, "channels": 1, "prolong_frames": 2},
              "stimulus_params": {"data": np.array([1, 2, 3]), "sr": 44100}},
             (error_code.INVALID_RECORD, "Failed to record full signal, please increase prolong time.")
             ),
        ])
    @mock.patch("scipy.io.wavfile.write")
    @mock.patch(test_path + ".calculate_alignment")
    @mock.patch("soundcard.default_microphone")
    def test_mic_worker(self, mock_default_mic, mock_alignment, mock_wavfile, input_ret, result_ret):
        mock_default_mic.return_value = input_ret["default_mic_ret"]
        mock_default_mic.return_value.record.return_value = input_ret["record_ret"]
        mock_alignment.return_value = input_ret["alignment_ret"]
        mock_wavfile.return_value = input_ret["wavfile_ret"]
        result = SoundcardControl().mic_worker(input_ret["record_params"], input_ret["stimulus_params"])
        assert result == result_ret

    @pytest.mark.parametrize("corr_ret, stimulus_signal, recorded_signal, result_set", [
        (1, [1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1], -4),
        (0, [1, 1, 1], [1, 1, 1], -2),
        (0, [], [], 1),
    ])
    @mock.patch("scipy.signal.correlate")
    def test_calculate_alignment(self, mock_corr, corr_ret, stimulus_signal, recorded_signal, result_set):
        mock_corr.return_value = corr_ret
        result = SoundcardControl().calculate_alignment(stimulus_signal, recorded_signal)
        assert result == result_set

    @pytest.mark.parametrize("process_set, start_set, result_set", [
        (mock.Mock(), mock.Mock(), (error_code.OK, "Successful start process.")),
        (mock.Mock(), Exception(),
         (error_code.INVALID_PROCESS, "Error starting process. 'Exception' object is not callable")),

    ])
    def test_start_process(self, process_set, start_set, result_set):
        process_set.start = start_set
        result = SoundcardControl().start_process(process_set)
        assert result == result_set

    @pytest.mark.parametrize("process_set, join_set, result_set", [
        (mock.Mock(), mock.Mock(), (error_code.OK, "Waiting process successful.")),
        (mock.Mock(), Exception(),
         (error_code.INVALID_PROCESS, "Waiting process error.'Exception' object is not callable")),
    ])
    def test_join_process(self, process_set, join_set, result_set):
        process_set.join = join_set
        result = SoundcardControl().join_process(process_set)
        assert result == result_set

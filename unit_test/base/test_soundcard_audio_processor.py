import mock
import numpy as np
import pytest

from base.soundcard_audio_processor import SoundcardAudioProcessor
from consts import error_code


class TestSoundcardAudioProcessor(object):

    test_path = "base.soundcard_audio_processor.SoundcardAudioProcessor"
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
        result = SoundcardAudioProcessor().initialize_audio_processes(input_ret["record_dict"], input_ret["stimulus_dict"])
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
        result = SoundcardAudioProcessor().speaker_worker(stimulus_params)
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
        result = SoundcardAudioProcessor().mic_worker(input_ret["record_params"], input_ret["stimulus_params"])
        assert result == result_ret

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

    @pytest.mark.parametrize("process_set, start_set, result_set", [
        (mock.Mock(), mock.Mock(), (error_code.OK, "Successful start process.")),
        (mock.Mock(), Exception(),
         (error_code.INVALID_PROCESS, "Error starting process. 'Exception' object is not callable")),

    ])
    def test_start_process(self, process_set, start_set, result_set):
        process_set.start = start_set
        result = SoundcardAudioProcessor().start_process(process_set)
        assert result == result_set

    @pytest.mark.parametrize("process_set, join_set, result_set", [
        (mock.Mock(), mock.Mock(), (error_code.OK, "Waiting process successful.")),
        (mock.Mock(), Exception(),
         (error_code.INVALID_PROCESS, "Waiting process error.'Exception' object is not callable")),
    ])
    def test_join_process(self, process_set, join_set, result_set):
        process_set.join = join_set
        result = SoundcardAudioProcessor().join_process(process_set)
        assert result == result_set

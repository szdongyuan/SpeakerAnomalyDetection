import multiprocessing
import numpy as np
import os
import soundcard as sc
from scipy import signal
from scipy.io import wavfile

from base.log_manager import LogManager
from consts import error_code


class SoundcardAudioProcessor(object):

    def __init__(self):
        self.logger = LogManager("soundcard_core")

    def initialize_audio_processes(self, record_dict: dict, stimulus_dict: dict, stimulus_path: str = "stimulus.wav",
                                   recording_path: str = "recording.wav"):
        """
            Initialize audio processes, including mic and speaker.
            Args:
                record_dict: dict
                    A dictionary of recording parameters.
                stimulus_dict: dict
                    A dictionary of stimulus parameters.
                stimulus_path: str
                    The save path of stimulus signal.
                recording_path: str
                    The save path of recording signal.
            Returns:
                A tuple containing the status code and message.
        """
        if not isinstance(record_dict, dict) or not record_dict:
            self.logger.warning("The record_dict is empty or invalid.")
            return error_code.INVALID_DATA_LOADING, "The record_dict is empty or invalid."
        if not isinstance(stimulus_dict, dict) or not stimulus_dict:
            self.logger.warning("The stimulus_dict is empty or invalid.")
            return error_code.INVALID_DATA_LOADING, "The stimulus_dict is empty or invalid."
        self.ensure_directory_exists(stimulus_path)
        self.ensure_directory_exists(recording_path)
        processes = [
            multiprocessing.Process(target=self.mic_worker, args=(record_dict, stimulus_dict, stimulus_path, recording_path)),
            multiprocessing.Process(target=self.speaker_worker, args=(stimulus_dict,))
        ]
        for process in processes:
            start_process_code, msg = self.start_process(process)
            if start_process_code != error_code.OK:
                return start_process_code, msg
        for process in processes:
            join_process_code, msg = self.join_process(process)
            if join_process_code != error_code.OK:
                return join_process_code, msg
        self.logger.info("All processes have finished.")
        return error_code.OK, "All processes have finished."

    @staticmethod
    def speaker_worker(stimulus_params: dict):
        """
            Play the stimulus audio.
            Args:
                stimulus_params: dict
                    A dictionary containing audio data and sampling rate.
            Returns:
                A tuple containing the status code and message.
        """
        try:
            data = stimulus_params.get("data") * stimulus_params.get("amplitude")
            sr = stimulus_params.get("sr")
            default_speaker = sc.default_speaker()
            default_speaker.play(data, samplerate=sr)
            return error_code.OK, "play successfully"
        except Exception as e:
            err_msg = "Failed to play audio.%s" % (str(e)[:50])
            return error_code.INVALID_PLAY, err_msg

    def mic_worker(self, record_params: dict, stimulus_params: dict, stimulus_path: str, recording_path: str):
        """
            Record audio and align it with the stimulus audio, save audio data as wav.
            Args:
                record_params: dict
                    A dictionary of recording parameters.
                stimulus_params: dict
                    A dictionary of stimulus parameters.
                stimulus_path: str
                    The save path of stimulus signal.
                recording_path: str
                    The save path of recording signal.
            Returns:
                A tuple containing the status code and message.
        """
        try:
            num_frames = record_params.get("num_frames")
            sr = record_params.get("sr")
            channels = record_params.get("channels", 1)
            default_mic = sc.default_microphone()
            recorded_data = default_mic.record(numframes=num_frames, samplerate=sr, channels=channels).T[0]
            stimulus_data = np.array(stimulus_params.get("data") * stimulus_params.get("amplitude"))
            align_frames = self.calculate_alignment(stimulus_data, recorded_data)
            if align_frames < record_params.get("prolong_frames"):
                wavfile.write(stimulus_path, sr, stimulus_data.astype("float32"))
                aligned_data = recorded_data[align_frames: align_frames + len(stimulus_data)]
                wavfile.write(recording_path, sr, aligned_data.astype("float32"))
                self.logger.info("Recording and stimulus saved.")
                return error_code.OK, "Recording and stimulus saved."
            else:
                self.logger.error("Failed to record full signal, please increase prolong time.")
                return error_code.INVALID_RECORD, "Failed to record full signal, please increase prolong time."
        except Exception as e:
            err_msg = "Failed to recorded audio.%s" % (str(e)[:50])
            self.logger.error(err_msg)
            return error_code.INVALID_RECORD, err_msg

    @staticmethod
    def ensure_directory_exists(save_path: str):
        """
            Ensure that the directory where the save path resides exists.
            Args:
                save_path: str
                    The save path of audio signals.
            Returns:
        """
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def calculate_alignment(stimulus_signal, recorded_signal):
        """
            Args:
                stimulus_signal: np.ndarray
                    The stimulus audio signal.
                recorded_signal: np.ndarray
                    The recorded audio signal.
            Returns:
                align_frames: int
                    The index of the alignment frames.
        """
        corr = signal.correlate(recorded_signal, stimulus_signal)
        align_frames = np.argmax(np.abs(corr)) - len(stimulus_signal) + 1
        return align_frames

    def start_process(self, process):
        """
             Start the given process.
            Args:
                process: multiprocessing.Process
                    The process to start.
            Returns:
                A tuple containing the status code and message.
        """
        try:
            process.start()
            self.logger.info(f"Starting process {process.name}")
            return error_code.OK, "Successful start process."
        except Exception as e:
            err_msg = "Error starting process. %s" % (str(e)[:40])
            self.logger.error(err_msg)
            return error_code.INVALID_PROCESS, err_msg

    def join_process(self, process):
        """
            Wait for the given process to finish.
            Args:
                process: multiprocessing.Process
                    The process to wait for.
            Returns:
                A tuple containing the status code and message.
        """
        try:
            process.join()
            self.logger.info(f"Waiting process {process.name}.")
            return error_code.OK, "Waiting process successful."
        except Exception as e:
            err_msg = "Waiting process error.%s" % (str(e)[:40])
            self.logger.error(err_msg)
            return error_code.INVALID_PROCESS, err_msg

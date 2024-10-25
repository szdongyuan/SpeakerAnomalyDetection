import multiprocessing
import os

import numpy as np
import time
import soundcard as sc
from scipy import signal
from scipy.io import wavfile

from base.log_manager import LogManager
from base.pre_processing.swept_sine_chirps import SineChirps
from consts import error_code


class SoundcardControl(object):

    def __init__(self):
        self.amplitudes = []
        self.voltages = []
        self.logger = LogManager("soundcard_core")

    def calibrate_amplitude(self, amplitude_voltage_data, target_voltage):
        """
            Args:
                amplitude_voltage_data: list
                    contains a list of amplitudes and measured voltages.
                    example: [(0.1, 2), (0.2, 3.99), (0.3, 5.97)]
                target_voltage: int or float or list
            Returns:
                predict_amplitude: int or float or list
                    The amplitude corresponding to the target voltage.
        """
        if not amplitude_voltage_data:
            return error_code.INVALID_DATA_LOADING, "Input data is empty."
        for amplitude, voltage in amplitude_voltage_data:
            self.add_data(amplitude, voltage)
        fit_code, fit_function = self.fit()
        predict_amplitude = None
        if fit_code == error_code.OK:
            predict_amplitude = self.predict_amplitude(fit_function, target_voltage)
        return predict_amplitude

    def add_data(self, amplitude, voltage):
        """
            Add amplitude and voltage data.
            Args:
                amplitude: int or float
                    The input amplitude value.
                voltage: int or float
                    The input voltage value.
            Returns:
                 A tuple containing the status code and message.
        """
        if not amplitude or not voltage:
            return error_code.INVALID_DATA_LOADING, "Input data cannot be None."
        if not isinstance(amplitude, (int, float)) or not isinstance(voltage, (int, float)):
            return error_code.INVALID_TYPE_DATA, "Input data must be numeric."
        self.amplitudes.append(amplitude)
        self.voltages.append(voltage)
        return error_code.OK, "Successfully add data."

    def fit(self):
        """
            Fit amplitude and voltage data to obtain a linear relationship.
            Returns:
                 A tuple containing the status code and the fitting function.
        """
        if not self.amplitudes or not self.voltages:
            return error_code.INVALID_DATA_LOADING, "Amplitudes and voltages must not be empty."
        if len(self.amplitudes) != len(self.voltages):
            return error_code.INVALID_DATA_LOADING, "Amplitudes and voltages must have the same length."
        coefficients = np.polyfit(self.voltages, self.amplitudes, 1)
        return error_code.OK, np.poly1d(coefficients)

    @staticmethod
    def predict_amplitude(fit_function, target_voltage):
        """
            Predict the corresponding amplitude based on the fitting function and target voltage.
            Args:
                fit_function: callable
                    The function obtained from fit.
                target_voltage: int or float
                    The target voltage value.
            Returns:
                predict_amplitude: float
                    The predicted amplitude(four decimal places).
        """
        predict_amplitude = fit_function(target_voltage)
        return np.round(predict_amplitude, 4)

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
            data = stimulus_params.get("data")
            sr = stimulus_params.get("sr")
            default_speaker = sc.default_speaker()
            default_speaker.play(data, samplerate=sr)
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
            stimulus_data = np.array(stimulus_params.get("data"))
            align_frames = self.calculate_alignment(stimulus_data, recorded_data)
            if align_frames < record_params.get("prolong_frames"):
                print(stimulus_path)
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

    @staticmethod
    def amplitude_fitting(measured_value):
        """
            Calculate the fitted amplitude based on the measured value.
            Args:
                measured_value: float or array-like
                    The measured value for which the amplitude is to be fitted.

                    coefficients = np.polyfit(measured_value_list, amplitude_value_list, 1)
                    measured_value_list = [2, 3.99, 5.97, 7.96, 9.94, 11.93, 13.91, 15.89, 17.87, 19.56]
                    amplitude_value_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
                Returns:
                    amplitude_fit: float or array-like
                        The fitted amplitude(four decimal places).
        """
        amplitude_fit = np.polyval([0.05081619, -0.0039981], measured_value)
        return np.round(amplitude_fit, 4)

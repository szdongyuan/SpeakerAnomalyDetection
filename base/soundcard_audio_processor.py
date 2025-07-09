import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from base.log_manager import LogManager
from consts import error_code


class SoundcardAudioProcessor(object):

    def __init__(self):
        self.logger = LogManager.set_log_handler("soundcard_core")

    def sd_play_rec(self, record_dict, stimulus_dict, recording_path):
        data = stimulus_dict.get("data") * stimulus_dict.get("amplitude")
        prepare_frames = record_dict.get("prepare_frames", 1000)
        prolong_frames = record_dict.get("prolong_frames", 10000)
        prolong_data = [0] * prepare_frames + list(data) + [0] * prolong_frames
        sr = stimulus_dict.get("sr")
        rec_data = sd.playrec(prolong_data, samplerate=sr, channels=1, blocking=True).T[0]
        align_frames = self.calculate_alignment(prolong_data, rec_data)
        aligned_data = rec_data[align_frames + prepare_frames: align_frames + prepare_frames + len(data)]
        wavfile.write(recording_path, sr, aligned_data.astype("float32"))
        return error_code.OK, aligned_data

    @staticmethod
    def sd_play(stimulus_params):
        try:
            data = stimulus_params.get("data") * stimulus_params.get("amplitude")
            print(stimulus_params.get("amplitude"))
            print(data)
            sr = stimulus_params.get("sr")
            blocking = stimulus_params.get("blocking", True)
            sd.play(data, samplerate=sr, blocking=blocking)
            return error_code.OK, "play successfully"
        except Exception as e:
            err_msg = "Failed to play audio. [%s]" % (str(e)[:50])
            return error_code.INVALID_PLAY, err_msg

    @staticmethod
    def sd_rec(recorded_dict):
        num_frames = recorded_dict.get("num_frames", 441000)
        sample_rate = recorded_dict.get("sample_rate", 44100)
        channels = recorded_dict.get("channels", 1)
        blocking = recorded_dict.get("blocking", True)
        prolong_frames = recorded_dict.get("prolong_frames", 0)
        recorded_data = sd.rec(frames=num_frames, samplerate=sample_rate, channels=channels, blocking=blocking).T[0]
        if prolong_frames > 0:
            recorded_data = recorded_data[prolong_frames:]

        return error_code.OK, recorded_data

    @staticmethod
    def gcc_phat(recorded_signal, stimulus_signal):
        """计算GCC-PHAT互相关函数并返回延迟。"""
        n = len(recorded_signal) + len(stimulus_signal)
        SIG = np.fft.rfft(recorded_signal, n=n)
        REF = np.fft.rfft(stimulus_signal, n=n)
        R = SIG * np.conj(REF)
        R_normal = R / (np.abs(R) + 1e-10)
        corr_func = np.fft.irfft(R_normal)
        max_shift = n // 2
        corr_func_shifted = np.fft.fftshift(corr_func)
        delay_samples = np.argmax(np.abs(corr_func_shifted)) - max_shift
        return delay_samples, corr_func_shifted, max_shift

    @staticmethod
    def calculate_alignment(stimulus_signal, recorded_signal):
        """
        使用GCC-PHAT对齐两个信号。

        Args:
            stimulus_signal (np.ndarray): 激励信号。
            recorded_signal (np.ndarray): 录音信号。
        Returns:
            int: 计算出的对齐帧数（延迟）。
        """
        align_frames, corr_func, max_shift = SoundcardAudioProcessor.gcc_phat(recorded_signal, stimulus_signal)
        return align_frames

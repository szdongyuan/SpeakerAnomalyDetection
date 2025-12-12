import numpy as np

from base.log_manager import LogManager
from base.save_data import save_audio_simple
from base.sound_device_manager import sd
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
        align_frames = self.calculate_alignment(data, rec_data)
        aligned_data = rec_data[align_frames: align_frames + len(data)]
        save_audio_simple(recording_path, aligned_data, sr)
        return error_code.OK, aligned_data

    @staticmethod
    def sd_play(stimulus_params):
        try:
            data = stimulus_params.get("data") * stimulus_params.get("amplitude")
            print(stimulus_params.get("amplitude"))
            print(data)
            sr = stimulus_params.get("sr")
            blocking = stimulus_params.get("blocking", True)
            device = stimulus_params.get("device", None)
            sd.play(data, samplerate=sr, device=device, blocking=blocking)
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
    def gcc_phat(stimulus_signal, recorded_signal):
        """计算GCC-PHAT互相关函数并返回延迟。"""
        n = len(recorded_signal) + len(stimulus_signal)
        n_11 = len(stimulus_signal) // 11
        SIG = np.fft.rfft(recorded_signal, n=n)
        REF = np.fft.rfft(stimulus_signal, n=n)
        R = SIG * np.conj(REF)
        max_shift = n // 2
        corr_func_r = np.fft.irfft(R)
        corr_func_shifted_r = np.fft.fftshift(corr_func_r)
        new_delay_samples_r = 0
        tmp_max = 0
        for i in range(n // 3, n - len(stimulus_signal) // 12, len(stimulus_signal) // 12):
            max_min_diff = max(corr_func_shifted_r[i: i + n_11]) - min(corr_func_shifted_r[i: i + n_11])
            if max_min_diff >= tmp_max:
                tmp_max = max_min_diff
                new_delay_samples_r = i + np.argmax(np.abs(corr_func_shifted_r[i: i + n_11]))
        new_delay_samples_r -= max_shift
        return new_delay_samples_r, corr_func_shifted_r, max_shift

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
        align_frames, corr_func, max_shift = SoundcardAudioProcessor.gcc_phat(stimulus_signal, recorded_signal)
        return align_frames

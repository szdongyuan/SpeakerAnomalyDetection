import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from base.utils.plot_audio_features import PlotManager


class AudioThdFrequencyResponseAnalysis(object):

    def process_calculate(self, reference_signal: np.ndarray, recorded_signal: np.ndarray, sr, **kwargs):
        results = {
            "thd_fig": None,
            "harmonic_fig": None,
            "frequency_response_fig": None
        }
        if kwargs.get("thd", True):
            results["thd_fig"], ax_thd = plt.subplots(figsize=(18, 10))
            results["harmonic_fig"], ax_harmonic = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
        if kwargs.get("frequency_response", True):
            results["frequency_response_fig"], ax_fr = plt.subplots(figsize=(13, 6))
        for i in range(len(recorded_signal)):
            pm = PlotManager()
            if kwargs.get("thd", True):
                thd_kwargs = kwargs.get("thd_kwargs", {})
                x, h, thd = self.calculate_thd(reference_signal, recorded_signal[i], sr[i], **thd_kwargs)
                pm.plot_thd(ax_thd, x, thd)
                pm.plot_harmonic(ax_harmonic, x, h)
            if kwargs.get("frequency_response", True):
                frequency_response_kwargs = kwargs.get("frequency_response_kwargs", {})
                fr, frequency_list = self.calculate_fr(reference_signal, recorded_signal[i], sr[i],
                                                       **frequency_response_kwargs)
                pm.plot_frequency_response(ax_fr, frequency_list, fr)
        return results

    def calculate_thd(self, freq_dict, base_freq_list, recorded_signal, sr, **kwargs):
        plot_x, plot_h, plot_thd = [], [], []
        gap_len = kwargs.get("gap_len", 10)
        delay_frames = kwargs.get("delay_frames", 0)
        harmonics_list = kwargs.get("harmonics", list(range(1, 6)))
        freq = self.get_harmonic(recorded_signal, freq_dict, sr, harmonics_list, gap_len, delay_frames)
        n_harmonics = len(harmonics_list)
        for i in range(int(min(base_freq_list)), gap_len * (len(freq) + 1), gap_len):
            plot_x.append(i)
            harmonic = freq[i]["harmonic"]
            f = freq[i]["harmonic_base"]
            plot_h.append(harmonic + [0] * (n_harmonics - len(harmonic)))
            h = harmonic
            td = (sum([i ** 2 for i in h])) ** 0.5
            plot_thd.append((td / (f ** 2 + td ** 2) ** 0.5) * 100)
        plot_h = np.array(plot_h).T
        return plot_x, plot_h, plot_thd

    @staticmethod
    def calculate_spectrum(reference_signal, sr, gap_len=10, delay_frames=0):
        win_len = sr // gap_len
        xf = np.fft.fftfreq(win_len, 1 / sr)
        base_freq_list = []
        freq_dict = {}
        for i in range(0, len(reference_signal) - win_len - delay_frames, 3):
            input_fft = np.abs(np.fft.fft(reference_signal[i: i + win_len])[: win_len // 2])
            argmax = np.argmax(input_fft)
            base_freq = xf[argmax]
            base_freq_list.append(base_freq)
            if freq_dict.get(base_freq, {'bf_v': 0}).get("bf_v") < np.max(input_fft):
                freq_dict[base_freq] = {"bf_v": np.max(input_fft), "i": i, "argmax": argmax}
        return freq_dict, base_freq_list

    @staticmethod
    def get_harmonic(recorded_signal, freq_dict, sr, harmonics, gap_len=10, delay_frames=0):
        win_len = sr // gap_len
        for base_freq in freq_dict:
            i_with_delay = freq_dict[base_freq]["i"] + delay_frames
            argmax = freq_dict[base_freq]["argmax"]
            data_fft = np.abs(np.fft.fft(recorded_signal[i_with_delay: i_with_delay + win_len])[: win_len // 2])
            harmonics_base = data_fft[argmax]
            harmonic_list = []
            for j in harmonics:
                if argmax * (j + 1) < win_len // 2:
                    harmonic_list.append(data_fft[argmax * (j + 1)])
            freq_dict[base_freq]["harmonic"] = harmonic_list
            freq_dict[base_freq]["harmonic_base"] = harmonics_base
        return freq_dict

    @staticmethod
    def calculate_fr(reference_signal, recorded_signal, sr):
        num = sr // 10
        frequency_list, pxx = signal.welch(reference_signal, sr, nfft=num)
        _, pxy = signal.csd(recorded_signal, reference_signal, sr, nfft=num)
        h = np.abs(pxy / pxx)
        fr = 20 * np.log10(h)
        idx_1 = np.argmax(np.abs(np.fft.fft(reference_signal[:1024], num)[:num // 2]))
        mid_slice = len(reference_signal) // 2
        idx_2 = np.argmax(np.abs(np.fft.fft(reference_signal[mid_slice - 511: mid_slice + 513], num)[:num // 2]))
        idx_3 = np.argmax(np.abs(np.fft.fft(reference_signal[-1024:], num)[:num // 2]))
        start_idx, stop_idx = min([idx_1, idx_2, idx_3]), max([idx_1, idx_2, idx_3])
        return fr[start_idx:stop_idx], frequency_list[start_idx:stop_idx]

    @staticmethod
    def spl_calculation(recorded_signal, reference_pressure=20e-6):
        spl = 20 * np.log10(np.abs(recorded_signal) / reference_pressure)
        spl_smooth = np.convolve(spl, np.ones(1102) / 1102, mode='same')
        return spl_smooth

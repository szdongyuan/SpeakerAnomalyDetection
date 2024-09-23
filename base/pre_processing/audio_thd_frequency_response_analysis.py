import matplotlib.pyplot as plt
import numpy as np

from base.utils.plot_audio_features import plot_thd, plot_frequency_response, plot_harmonic


class AudioThdFrequencyResponseAnalysis(object):

    def process_calculate(self, reference_signal: np.ndarray, recorded_signal: np.ndarray, sr, **kwargs):
        if kwargs.get("thd", True):
            fig, ax1 = plt.subplots()
            fig, ax2 = plt.subplots(nrows=2, ncols=3)
        if kwargs.get("frequency_response", True):
            fig, ax3 = plt.subplots()
        for i in range(len(recorded_signal)):
            if kwargs.get("thd", True):
                thd_kwargs = kwargs.get("thd_kwargs", {})
                x, h, thd = self.calculate_thd(reference_signal, recorded_signal[i], sr[i], **thd_kwargs)
                plot_thd(ax1, x, thd)
                plot_harmonic(ax2, x, h)
            if kwargs.get("frequency_response", True):
                frequency_response_kwargs = kwargs.get("frequency_response_kwargs", {})
                fr, frequency_list = self.calculate_fr(reference_signal, recorded_signal[i], sr[i], **frequency_response_kwargs)
                plot_frequency_response(ax3, frequency_list, fr)
        return plt

    def calculate_thd(self, reference_signal, recorded_signal, sr, **kwargs):
        plot_x, plot_h, plot_thd = [], [], []
        freq_dict, base_freq_list = self.get_harmonic(reference_signal, recorded_signal, sr, **kwargs)
        gap_len = kwargs.get("gap_len", 10)
        for i in range(int(base_freq_list[0]), gap_len * (len(freq_dict) + 1), gap_len):
            plot_x.append(i)
            harmonic = freq_dict[i]["harmonic"]
            plot_h.append(harmonic + [0] * (6 - len(harmonic)))
            f = harmonic[0]
            h = harmonic[1:]
            td = (sum([i ** 2 for i in h])) ** 0.5
            plot_thd.append(td / (f ** 2 + td ** 2) ** 0.5)
        plot_h = np.array(plot_h).T
        return plot_x, plot_h, plot_thd

    @staticmethod
    def get_harmonic(reference_signal, recorded_signal, sr, **kwargs):
        gap_len = kwargs.get("gap_len", 10)
        delay_frames = kwargs.get("delay_frames", 9008)
        win_len = sr // gap_len
        xf = np.fft.fftfreq(win_len, 1 / sr)
        freq_dict = {}
        base_freq_list = []
        for i in range(len(reference_signal) - win_len - delay_frames):
            input_fft = np.abs(np.fft.fft(reference_signal[i: i + win_len])[: win_len // 2])
            argmax = np.argmax(input_fft)
            base_freq = xf[argmax]
            base_freq_list.append(base_freq)
            if freq_dict.get(base_freq, {'bf_v': 0}).get("bf_v") < np.max(input_fft):
                data_fft = np.abs(
                    np.fft.fft(recorded_signal[i + delay_frames: i + delay_frames + win_len])[: win_len // 2])
                harmonic_list = []
                for j in range(6):
                    if argmax * (j + 1) < win_len // 2:
                        harmonic_list.append(data_fft[argmax * (j + 1)])
                freq_dict[base_freq] = {"bf_v": np.max(input_fft), "yf": data_fft, "harmonic": harmonic_list}
        return freq_dict, base_freq_list

    def calculate_fr(self, reference_signal, recorded_signal, sr, **kwargs):
        smooth = kwargs.get("smooth", True)
        delay_frames = kwargs.get("delay_frames", 9008)
        recorded_signal = recorded_signal[delay_frames:]
        ch_len = len(reference_signal)
        frequency_list = np.fft.fftfreq(ch_len, 1 / sr)[: ch_len // 2]
        yf = np.abs(np.fft.fft(recorded_signal))[: ch_len // 2]
        xf = np.abs(np.fft.fft(reference_signal))[: ch_len // 2]
        fr = 20 * np.log10(yf / xf)
        if smooth:
            fr = self.smooth_curve(frequency_list, fr)
        print(fr, frequency_list)
        return fr, frequency_list

    @staticmethod
    def smooth_curve(frequencies, data, oct_width=1 / 3):
        frequencies = np.where(frequencies == 0, 1e-10, frequencies)
        log_freqs = np.log2(frequencies)
        half_window = oct_width / 2
        smoothed_data = np.zeros_like(data)
        for i in range(len(frequencies)):
            current_log_freq = log_freqs[i]
            lower_bound = current_log_freq - half_window
            upper_bound = current_log_freq + half_window
            mask = (log_freqs >= lower_bound) & (log_freqs <= upper_bound)
            smoothed_data[i] = np.mean(data[mask])
        return smoothed_data

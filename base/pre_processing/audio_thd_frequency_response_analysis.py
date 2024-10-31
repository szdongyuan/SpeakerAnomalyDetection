import matplotlib.pyplot as plt
import numpy as np

from base.utils.plot_audio_features_manager import PlotManager


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
                fr, frequency_list = self.calculate_fr(reference_signal, recorded_signal[i], sr[i], **frequency_response_kwargs)
                pm.plot_frequency_response(ax_fr, frequency_list, fr)
        return results

    def calculate_thd(self, reference_signal, recorded_signal, sr, **kwargs):
        plot_x, plot_h, plot_thd = [], [], []
        freq_dict, base_freq_list = self.get_harmonic(reference_signal, recorded_signal, sr, **kwargs)
        gap_len = kwargs.get("gap_len", 10)
        n_harmonics = len(kwargs.get("harmonics", list(range(6))))
        for i in range(int(base_freq_list[0]), gap_len * (len(freq_dict) + 1), gap_len):
            plot_x.append(i)
            harmonic = freq_dict[i]["harmonic"]
            plot_h.append(harmonic + [0] * (n_harmonics - len(harmonic)))
            f = harmonic[0]
            h = harmonic[1:]
            td = (sum([i ** 2 for i in h])) ** 0.5
            plot_thd.append(td / (f ** 2 + td ** 2) ** 0.5)
        plot_h = np.array(plot_h).T
        return plot_x, plot_h, plot_thd

    @staticmethod
    def get_harmonic(reference_signal, recorded_signal, sr, **kwargs):
        gap_len = kwargs.get("gap_len", 10)
        delay_frames = kwargs.get("delay_frames", 0)
        harmonics = kwargs.get("harmonics", list(range(6)))

        win_len = sr // gap_len
        xf = np.fft.fftfreq(win_len, 1 / sr)
        freq_dict = {}
        base_freq_list = []

        for i in range(0, len(reference_signal) - win_len - delay_frames, 3):
            input_fft = np.abs(np.fft.fft(reference_signal[i: i + win_len])[: win_len // 2])
            argmax = np.argmax(input_fft)
            base_freq = xf[argmax]
            base_freq_list.append(base_freq)
            if freq_dict.get(base_freq, {'bf_v': 0}).get("bf_v") < np.max(input_fft):
                freq_dict[base_freq] = {"bf_v": np.max(input_fft), "i": i, "argmax": argmax}

        for base_freq in freq_dict:
            i_with_delay = freq_dict[base_freq]["i"] + delay_frames
            argmax = freq_dict[base_freq]["argmax"]
            data_fft = np.abs(np.fft.fft(recorded_signal[i_with_delay: i_with_delay + win_len])[: win_len // 2])
            harmonic_list = []
            for j in harmonics:
                if argmax * (j + 1) < win_len // 2:
                    harmonic_list.append(data_fft[argmax * (j + 1)])
            freq_dict[base_freq]["harmonic"] = harmonic_list
        return freq_dict, base_freq_list

    def calculate_fr(self, reference_signal, recorded_signal, sr, **kwargs):
        smooth = kwargs.get("smooth", True)
        delay_frames = kwargs.get("delay_frames", 0)
        recorded_signal = recorded_signal[delay_frames:]
        ch_len = len(reference_signal)
        frequency_list = np.fft.fftfreq(ch_len, 1 / sr)[: ch_len // 2]
        yf = np.abs(np.fft.fft(recorded_signal))[: ch_len // 2]
        xf = np.abs(np.fft.fft(reference_signal))[: ch_len // 2]
        fr = 20 * np.log10(yf / xf)
        if smooth:
            fr = self.smooth_curve(frequency_list, fr)
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

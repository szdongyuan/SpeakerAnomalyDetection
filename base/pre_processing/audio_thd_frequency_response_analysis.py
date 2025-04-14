import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter
from scipy.signal import savgol_filter

from base.utils.plot_audio_features import PlotManager


class AudioThdFrequencyResponseAnalysis(object):

    def process_calculate(self, reference_signal: np.ndarray, recorded_signal, sr, **kwargs):
        """
            Calculate and plot THD, harmonic, and frequency response figures, and return the result images.

            Args:
                - reference_signal: ndarray
                    The input reference signal.
                - recorded_signal: list
                    A list of recorded signals
                - sr: list
                    A list consisting of the sample rate of the signal
                - kwargs : dict
                    Additional optional parameters

            Returns:
                - results: dict
                    Each key corresponds to a graph (matplotlib.figure.Figure object)

        """
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
                freq_dict, base_freq_list = self.calculate_spectrum(reference_signal, sr[i])
                x, h, thd = self.calculate_thd(freq_dict, base_freq_list, recorded_signal[i], sr[i], **thd_kwargs)
                pm.plot_thd(ax_thd, x, thd)
                pm.plot_harmonic(ax_harmonic, x, h)

                ## 1234567890
            if kwargs.get("frequency_response", True):
                fr, frequency_list = self.calculate_fr(reference_signal, recorded_signal[i], sr[i])
                pm.plot_frequency_response(ax_fr, frequency_list, fr)
                ## 1234567890
                # 9876543210
                ### basdsadasds
                ## basdsadasds
        return results

    def calculate_thd(self, freq_dict, base_freq_list, recorded_signal, sr, **kwargs):
        """
            Calculate the Total Harmonic Distortion (THD).

            Args:
                - freq_dict: dict
                    The input reference signal.
                - base_freq_list: list

                - recorded_signal: ndarray
                    The input recorded signal
                - sr: int
                    The sample rate of the signals.
                - kwargs : optional
                    - gap_len : int, default 10
                        The length of each gap between frequency points.
                    - delay_frames : int, default 0
                        The number of frames to delay.
                    - harmonics : list, default [1, 2, 3, 4, 5]
                        The list of harmonics to be calculated.
            Returns:
                - plot_x: list
                    the base frequency at different time points.
                - plot_h : ndarray
                    The harmonic amplitudes at each time point.
                - plot_thd : list
                    The Total Harmonic Distortion (THD) at each time point.
        """
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
            h = harmonic
            td = (sum([i ** 2 for i in h])) ** 0.5
            plot_h.append([f] + harmonic + [0] * (n_harmonics - len(harmonic)))
            plot_thd.append((td / (f ** 2 + td ** 2) ** 0.5) * 100)
        plot_h = np.array(plot_h).T
        return plot_x, plot_h, plot_thd

    @staticmethod
    def calculate_spectrum(reference_signal, sr, gap_len=10, delay_frames=0):
        """
            Calculate the spectrum of the reference signal, returning the base frequency
            and its maximum amplitude for each time window.

            Args:
                - reference_signal : ndarray
                    The input reference signal.
                - sr: int
                    The sample rate of the signals.
                - gap_len : int, optional (default is 10)
                    The length of each time window used to calculate.
                - delay_frames : int, default 0
                    The number of frames to delay.

            Returns:
                - freq_dict: dict
                    A dictionary with the base frequency as the key, storing the maximum amplitude,
                    position, and index information.
                - base_freq_list: list
                    A list containing the base frequency for each time window.
        """
        win_len = sr // gap_len
        hop_length = 3
        noverlap = win_len - hop_length
        if noverlap < 0:
             noverlap = 0 

        f, t, Zxx = signal.stft(reference_signal, fs=sr, nperseg=win_len, noverlap=noverlap,
                                nfft=win_len, return_onesided=True, boundary=None, padded=False, window='boxcar')

        abs_Zxx = np.abs(Zxx)
        argmax_indices = np.argmax(abs_Zxx, axis=0)
        all_base_freqs = f[argmax_indices] 
        max_amplitudes = np.max(abs_Zxx, axis=0) 

        freq_dict = {}
        num_windows = abs_Zxx.shape[1]

        for j in range(num_windows):
            base_freq = all_base_freqs[j]
            amplitude = max_amplitudes[j]
            i = j * hop_length
            argmax = argmax_indices[j]

            # Update dict if this window has a higher amplitude for this base_freq
            if amplitude > freq_dict.get(base_freq, {'bf_v': -1.0}).get("bf_v"):
                 freq_dict[base_freq] = {"bf_v": amplitude, "i": i, "argmax": argmax}

        base_freq_list = list(all_base_freqs)

        return freq_dict, base_freq_list

    @staticmethod
    def get_harmonic(recorded_signal, freq_dict, sr, harmonics, gap_len=10, delay_frames=0):
        """
            Extract harmonic information from the recorded signal and update the frequency dictionary.

            Args:
                - recorded_signal : ndarray
                    The input recorded signal.
                - freq_dict: dict
                    A dictionary with the base frequency as the key, storing the maximum amplitude,
                    position, and index information.
                - sr: int
                    The sample rate of the signals.
                - harmonics: list
                    A list specifying the harmonics to extract, e.g., [1, 2, 3, 4, 5] for 1st to 5th harmonics.
                - gap_len : int, optional (default is 10)
                    The length of each time window used to calculate.
                - delay_frames : int, default 0
                    The number of frames to delay.

            Returns:
                - freq_dict: dict
                    The updated frequency dictionary with harmonic amplitude lists and the base frequency amplitude.

        """
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
    def calculate_fr(reference_signal, recorded_signal, sr, is_smooth=True):
        """
            Calculate the frequency response (FR).

            Args:
                - reference_signal : ndarray
                    The input reference signal.
                - recorded_signal : ndarray
                    The input recorded signal
                - sr: int
                    The sample rate of the signals.

            Returns:
                - fr: ndarray
                    The calculated frequency response in decibels (dB), clipped to the valid frequency range.
                - frequency_list : ndarray
                    The corresponding frequency list, aligned with the response (`fr`).
        """
        num = sr
        hop_length = num / 2
        window = np.hanning(num)
        frequency_list, pxy = signal.csd(reference_signal, recorded_signal, window=window, nfft=num, noverlap=hop_length, fs=sr)
        _, pxx = signal.welch(reference_signal, fs=sr, nperseg=num, noverlap=hop_length, window=window)
        h = np.abs(pxy / pxx)
        fr = 10 * np.log10(h)
        slice_length = 1024
        idx_1 = np.argmax(np.abs(np.fft.fft(reference_signal[:slice_length], num)[:num // 2]))
        mid_slice = len(reference_signal) // 2
        idx_2 = np.argmax(np.abs(np.fft.fft(reference_signal[mid_slice - (slice_length // 2) - 1:
                                                             mid_slice + (slice_length // 2) + 1], num)[:num // 2]))
        idx_3 = np.argmax(np.abs(np.fft.fft(reference_signal[-slice_length:], num)[:num // 2]))
        start_idx, stop_idx = min([idx_1, idx_2, idx_3]), max([idx_1, idx_2, idx_3])
        if is_smooth:
            fr = savgol_filter(fr, window_length=60, polyorder=3)
        return fr[start_idx:stop_idx], frequency_list[start_idx:stop_idx]

    @staticmethod
    def spl_calculation(recorded_signal, reference_pressure=20e-6, window_size=1201, is_smooth=True):
        """
            Calculate the Sound Pressure Level (SPL) of the recorded signal.

            Args:
                - recorded_signal : ndarray
                    The input recorded signal
                - reference_pressure : float
                    The reference sound pressure, defaulting to 20 μPa (20e-6 Pa),
                    used as the baseline for SPL calculation.
                - window_size: int
                    The sliding window length

            Returns:
                - spl_smooth : ndarray
                    The computed SPL (in dB) after smoothing.
        """
        amplitude_list = maximum_filter(np.abs(recorded_signal), size=window_size)
        spl = 20 * np.log10(np.array(amplitude_list) / reference_pressure)
        if is_smooth:
            spl_smooth = np.convolve(spl, np.ones(1102) / 1102, mode='same')
            return spl_smooth
        else:
            return spl


import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter
from scipy.signal import savgol_filter, medfilt
import librosa

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
            if kwargs.get("frequency_response", True):
                fr, frequency_list = self.calculate_fr(reference_signal, recorded_signal[i], sr[i])
                pm.plot_frequency_response(ax_fr, frequency_list, fr)
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
    def calculate_spectrum(reference_signal, sr, gap_len=10, hop_length = 3, window = 'boxcar'):
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
        noverlap = win_len - hop_length
        if noverlap < 0:
             noverlap = 0 

        f, t, xf = signal.stft(reference_signal, fs=sr, nperseg=win_len, noverlap=noverlap,
                                nfft=win_len, return_onesided=True, boundary=None, padded=False, window=window)

        abs_xf = np.abs(xf)
        argmax_indices = np.argmax(abs_xf, axis=0)
        all_base_freqs = f[argmax_indices] 
        max_amplitudes = np.max(abs_xf, axis=0) 

        freq_dict = {}
        num_windows = abs_xf.shape[1]

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
    def calculate_fundamental_freq(reference_signal, sr, **kwargs):
        """
            Calculate the fundamental frequency of the reference signal. more accurate than the stft method(calculate_spectrum() method).

            Args:
                - reference_signal : ndarray
                    The input reference signal.
                - sr: int
                    The sample rate of the signals.
                - kwargs : optional
                    - method : string,'yin','pyin','stft','cqt','database'. default 'yin'. 
                        The method to calculate the fundamental frequency.
                        if use 'database', it means get the stimulus from the database to calculate the fundamental frequency.
                    - window : string, default 'hann'
                        The window to use for the stft.
                    - database_path : if method is 'database', default '../../data_base/audio_data.db'
                    - f0_min : int, default 50
                        The minimum frequency to consider for the fundamental frequency.
                    - f0_max : int, default 10000
                        The maximum frequency to consider for the fundamental frequency.
                    - frame_size : int, default 1024, if use stft, it means the window size.
                    - hop_length : int, default 128
                    - need_sort : bool, default False
                        Whether to sort the fundamental frequency by value.
                    - unique : bool, default False
                        Whether to remove duplicate fundamental frequencies, preserving the order of first appearance if need_sort is False.

        """
        frame_size = kwargs.get("frame_size", 1024)
        hop_length = kwargs.get("hop_length", 128)
        f0_min = kwargs.get("f0_min", 50)
        f0_max = kwargs.get("f0_max", 10000)

        if kwargs.get("method", "yin") == "yin":
            f0 = librosa.yin(reference_signal, sr=sr, fmin=f0_min, fmax=f0_max, hop_length=hop_length)
        elif kwargs.get("method", "yin") == "pyin":
            f0, _, _ = librosa.pyin(reference_signal, sr=sr, fmin=f0_min, fmax=f0_max, hop_length=hop_length)
        elif kwargs.get("method", "yin") == "stft":
            # librosa 0.9.0+ 返回的是 Zxx - 单个复数数组
            Zxx = librosa.stft(reference_signal, n_fft=frame_size, hop_length=hop_length, 
                              win_length=frame_size, window=kwargs.get("window", 'hann'))
            f_stft = librosa.fft_frequencies(sr=sr, n_fft=frame_size)
            amp = np.abs(Zxx)
            max_amp_indices = np.argmax(amp, axis=0)
            f0 = f_stft[max_amp_indices]
        elif kwargs.get("method", "yin") == "cqt": ## 后续添加
            return None, None
        elif kwargs.get("method", "yin") == "database":
            return None, None

        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

        if kwargs.get("unique", False):
            _, unique_indices = np.unique(f0, return_index=True)
            sorted_unique_indices = np.sort(unique_indices) 
            f0 = f0[sorted_unique_indices]
            times = times[sorted_unique_indices]

        if kwargs.get("need_sort", False):
            sort_indices = np.argsort(f0)
            f0 = f0[sort_indices]
            times = times[sort_indices]

        return f0, times

    @staticmethod
    def compute_cqt(y, sr=44100, hop_length=128, n_fft=1024, fmin=None, fmax=None, bins_per_octave=None, n_bins=None):
        """
        Compute the Constant-Q Transform (CQT) of an audio signal.
    
        Parameters
        ----------
        y : np.ndarray
            Audio time series
        sr : number > 0
            Sampling rate of y
        hop_length : int > 0
            Number of samples between frames
        n_fft : int > 0
            window size, used to determine frequency resolution, 为了可以像stft那样使用
        fmin : float > 0
            Minimum frequency
        fmax : float > 0
            Maximum frequency. If None, defaults to sr/3
        bins_per_octave : int > 0 or None
            Number of bins per octave. If None, calculated based on n_fft.
        n_bins : int > 0 or None
            Total number of CQT bins. If None, calculated based on fmin, fmax and bins_per_octave.
        
        Returns:
            - C : np.ndarray
                Constant-Q transform of y, a complex number
            - freqs : np.ndarray
                Frequencies corresponding to each bin of CQT
            - times : np.ndarray
                Time points corresponding to each frame of CQT
        """
    
        if fmin is None:
            fmin = librosa.note_to_hz('C1')  # 32.7 Hz
    
        if fmax is None:
            fmax = librosa.note_to_hz('C9')
    
        if bins_per_octave is None:
            # 表示每个八度内有多少频率点，对应的频率对数增加, 增加n_fft会增加频率分辨率，和预期相符
            bins_per_octave = int(12 * np.log2(n_fft/1024) + 24)  
            bins_per_octave = max(12, bins_per_octave)  
    
    
        if n_bins is None:
            n_octaves =  np.log2(fmax / fmin)    # 八度，表示频率区间跨越了多少频率翻倍的区间
            n_bins = int(np.ceil(n_octaves * bins_per_octave))   # n_bins ≈ bins_per_octave * log2(fmax/fmin)，最终的频域采样点数
    
        # fmax parameter is not supported in librosa.cqt
        C = librosa.cqt(
            y=y,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave
        )
    
        freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
        times = librosa.times_like(C, sr=sr, hop_length=hop_length)
        return C, freqs, times

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

    @staticmethod    
    def calculate_loose_particle_spl(recorded_signal, kernel_size):
        """
            Calculate the sound pressure level of loose particles.

            This method processes the recorded signal using Fourier Transform and median filtering to compute its sound pressure level.

            Args:
                -recorded_signal (np.array): The recorded audio signal.
                -kernel_size (int): The size of the median filter kernel, must be an odd number.

            Returns:
                -filtered_spl:np.array 
                    The sound pressure level array after median filtering.
        """
        fft_result = np.fft.fft(recorded_signal)
        window = np.hanning(len(fft_result))
        window_signal = window * fft_result
        length = len(window_signal)
        positive_fft = window_signal[0:int(length/2)]
        analytic_signal = np.fft.ifft(positive_fft)
        amplitude = np.abs(analytic_signal)
        reference_pressure = 20e-6
        signal_spl = 20 * np.log10(amplitude / reference_pressure)
        filtered_spl = medfilt(signal_spl, kernel_size)

        return filtered_spl


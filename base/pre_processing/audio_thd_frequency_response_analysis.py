from typing import Optional
import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import hilbert
from scipy.ndimage import maximum_filter, uniform_filter1d
from scipy.signal import savgol_filter, medfilt, bessel, filtfilt
import librosa

from base.utils.plot_audio_features import PlotManager
from base.pre_processing.harmonic_index_builder import HarmonicIndexBuilder
from base.pre_processing.step_signal_hd import StepSignalHD
from base.pre_processing.chirp_signal_hd import ChirpSignalHD
from base.pre_processing.perceptual_step_signal_hd import PerceptualStepSignalHD
from base.pre_processing.perceptual_chirp_signal_hd import PerceptualChirpSignalHD


class AudioThdFrequencyResponseAnalysis(object):

    def process_calculate(self, reference_signal: np.ndarray, recorded_signal, sr, **kwargs):
        """
            Calculate and plot THD, harmonic, and frequency response figures, and return the result images.

            Uses three-phase architecture. Requires thd_kwargs['stimulus_metadata'].

            Args:
                - reference_signal: ndarray
                    The input reference signal (not used in three-phase architecture, kept for API compatibility).
                - recorded_signal: list
                    A list of recorded signals
                - sr: list
                    A list consisting of the sample rate of the signal
                - kwargs : dict
                    Required: thd_kwargs with 'stimulus_metadata' key

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

                # Three-phase architecture (required)
                if 'stimulus_metadata' not in thd_kwargs:
                    raise ValueError(
                        "thd_kwargs must contain 'stimulus_metadata'. "
                        "Legacy methods have been removed. "
                        "See docs/hd_refactoring_guide.md for migration instructions."
                    )

                x, h, thd = self._calculate_thd_three_phase(
                    recorded_signal[i], sr[i], thd_kwargs
                )

                pm.plot_thd(ax_thd, x, thd)
                pm.plot_harmonic(ax_harmonic, x, h)
            if kwargs.get("frequency_response", True):
                fr, frequency_list = self.calculate_fr(reference_signal, recorded_signal[i], sr[i])
                pm.plot_frequency_response(ax_fr, frequency_list, fr)
        return results

    def _calculate_thd_three_phase(self, recorded_signal, sr, thd_kwargs):
        """
        NEW METHOD: Calculate THD using three-phase architecture.

        Step signals use STFT exclusively. Chirp signals use STFT exclusively.

        Returns: (x, h, thd) for plotting (backward compatible with existing plots)
        """
        stimulus_metadata = thd_kwargs['stimulus_metadata']
        harmonic_orders = thd_kwargs.get('harmonic_orders', [2, 3, 4, 5])

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1A: Build Overall Index Matrix
        # ═══════════════════════════════════════════════════════════════════
        builder = HarmonicIndexBuilder()

        if stimulus_metadata['stimulus_method'] == 'steps':
            # STFT window type for step signals
            stft_window_type = thd_kwargs.get('stft_window_type', 'hann')

            # Calculate STFT parameters (full step duration - no trimming)
            single_rep_duration = stimulus_metadata['total_time'] / stimulus_metadata['repeat_times']
            step_duration = single_rep_duration / stimulus_metadata['num_steps']
            step_samples = int(step_duration * sr)
            n_fft = step_samples  # STFT window size = step duration

            index_matrix, fund_freqs, fft_freqs = builder.build_step_signal_index_matrix(
                stimulus_metadata, sr=sr, n_fft=n_fft, max_harmonic_order=35
            )
        elif stimulus_metadata['stimulus_method'] == 'chirps':
            stft_window_size = thd_kwargs.get('stft_window_size', 2048)
            stft_hop_size = thd_kwargs.get('stft_hop_size', 1024)

            index_matrix, fund_freqs, time_array, fft_freqs = builder.build_chirp_signal_index_matrix(
                stimulus_metadata, sr=sr, n_fft=stft_window_size,
                hop_length=stft_hop_size, max_harmonic_order=35
            )
        else:
            raise ValueError(f"Unsupported stimulus_method: {stimulus_metadata['stimulus_method']}")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1B: Select User Configuration
        # ═══════════════════════════════════════════════════════════════════
        mask_matrix = builder.create_mask_from_indices(
            index_matrix, harmonic_orders, len(fft_freqs)
        )
        fundamental_bins = index_matrix[:, 1]

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 2: Calculate THD
        # ═══════════════════════════════════════════════════════════════════
        if stimulus_metadata['stimulus_method'] == 'steps':
            analyzer = StepSignalHD(sample_rate=sr)
            result = analyzer.compute_distortion(
                recorded_signal, stimulus_metadata, harmonic_orders,
                harmonic_mask=(mask_matrix, fund_freqs, fundamental_bins),
                stft_window_type=stft_window_type
            )

            # Format for plotting (backward compatible)
            x = result['frequencies']
            thd = result['thd']
            # h needs to be (6, n_steps) for plotting - 6 harmonics expected by plot_harmonic
            # Row 0: fundamental, Rows 1-5: harmonics 1-5
            h = np.zeros((6, len(x)))
            h[0, :] = x  # First row is fundamental frequencies (used as placeholder)

            # Extract harmonic amplitudes from STFT spectrum using index matrix
            spectrum = result['spectrum_matrix']  # Shape: (n_bins+1, n_steps) with dummy bin
            for step_idx in range(len(x)):
                # Extract harmonics 1-5 using index matrix
                for harmonic_order in range(1, 6):
                    bin_idx = index_matrix[step_idx, harmonic_order]
                    if bin_idx > 0:  # Not sentinel/dummy bin
                        h[harmonic_order, step_idx] = spectrum[bin_idx, step_idx]


        elif stimulus_metadata['stimulus_method'] == 'chirps':
            analyzer = ChirpSignalHD(sample_rate=sr)
            stft_window_size = thd_kwargs.get('stft_window_size', 2048)
            stft_hop_size = thd_kwargs.get('stft_hop_size', 1024)

            if 'time_array' not in locals():
                # Rebuild time_array if needed
                single_rep_duration = stimulus_metadata['total_time'] / stimulus_metadata['repeat_times']
                num_samples = int(single_rep_duration * sr)
                num_frames = 1 + (num_samples - stft_window_size) // stft_hop_size
                time_array = (np.arange(num_frames) * stft_hop_size + stft_window_size / 2) / sr

            result = analyzer.compute_distortion(
                recorded_signal, stimulus_metadata, harmonic_orders,
                harmonic_mask=(mask_matrix, None, fund_freqs, time_array, fundamental_bins),
                stft_window_size=stft_window_size,
                stft_hop_size=stft_hop_size
            )

            x = result['frequencies']
            thd = result['thd']
            # h needs to be (6, n_frames) for plotting
            h = np.zeros((6, len(x)))
            h[0, :] = x

        return x, h, thd

    def _calculate_perceptual_thd_three_phase(
        self,
        recorded_signal: np.ndarray,
        sample_rate: int,
        thd_kwargs: dict,
        spl_calibration_db: float = 0.0,
        noise_spectrum: np.ndarray = None
    ) -> tuple:
        """
        Calculate perceptual loudness (phons) using three-phase architecture with psychoacoustic models.

        Similar to _calculate_thd_three_phase but returns perceived loudness instead of THD percentage.

        Args:
            recorded_signal: Recorded audio signal
            sample_rate: Sample rate
            thd_kwargs: {
                'stimulus_metadata': dict with stimulus configuration,
                'harmonic_orders': list of harmonic orders (e.g., [10, 11, 12])
            }
            spl_calibration_db: Microphone calibration deviation in dB (default 0.0).
                This is the deviation/offset value (e.g., from mic_calibration.txt).
                Applied in amplitude domain before log transform:
                calibrated_amp = amp * 10^(calibration_db/20)
            noise_spectrum: Optional (n_fft//2 + 1,) background noise magnitude spectrum

        Returns:
            (freq_value, harmonic, perceptual_loudness):
                - freq_value: Fundamental frequencies
                - harmonic: Harmonic orders array
                - perceptual_loudness: Perceived loudness in phons
        """
        stimulus_metadata = thd_kwargs['stimulus_metadata']
        harmonic_orders = thd_kwargs.get('harmonic_orders', [])

        # Phase 1A: Build overall index matrix
        builder = HarmonicIndexBuilder()

        stimulus_method = stimulus_metadata['stimulus_method']

        if stimulus_method == 'steps':
            # Calculate STFT parameters (full step duration - no trimming)
            single_rep_duration = stimulus_metadata['total_time'] / stimulus_metadata['repeat_times']
            step_duration = single_rep_duration / stimulus_metadata['num_steps']
            step_samples = int(step_duration * sample_rate)
            n_fft = step_samples  # STFT window size = step duration

            index_matrix, fund_freqs, fft_freqs = builder.build_step_signal_index_matrix(
                stimulus_metadata, sr=sample_rate, n_fft=n_fft, max_harmonic_order=35
            )
        elif stimulus_method == 'chirps':
            stft_window_size = thd_kwargs.get('stft_window_size', 2048)
            stft_hop_size = thd_kwargs.get('stft_hop_size', 1024)

            index_matrix, fund_freqs, time_array, fft_freqs = builder.build_chirp_signal_index_matrix(
                stimulus_metadata, sr=sample_rate, n_fft=stft_window_size,
                hop_length=stft_hop_size, max_harmonic_order=35
            )
        else:
            raise ValueError(f"Unsupported stimulus_method: {stimulus_method}")

        # Phase 1B: Create mask from selected harmonics
        mask_matrix = builder.create_mask_from_indices(
            index_matrix, harmonic_orders, len(fft_freqs)
        )
        fundamental_bins = index_matrix[:, 1]

        # Build masking_mask_matrix for cumulative masking
        masking_mask_matrix = None
        masking_config = thd_kwargs.get('masking_config')
        if masking_config and masking_config.get('enable_cumulative'):
            # For cumulative masking, include all lower-order harmonics up to max analyzed harmonic
            max_harmonic = max(harmonic_orders)
            masking_orders = list(range(1, max_harmonic))  # Fundamental to (max - 1)

            if masking_orders:  # Only create if there are masking harmonics
                masking_mask_matrix = builder.create_mask_from_indices(
                    index_matrix, masking_orders, len(fft_freqs)
                )

        # Phase 2: Compute perceptual loudness using perceptual analyzers
        if stimulus_method == 'steps':
            analyzer = PerceptualStepSignalHD(sample_rate)
            result = analyzer.compute_distortion(
                recorded_signal, stimulus_metadata, harmonic_orders,
                harmonic_mask=(mask_matrix, masking_mask_matrix, fund_freqs, fundamental_bins),
                spl_calibration_db=spl_calibration_db,
                masking_config=masking_config,
                noise_spectrum=noise_spectrum
            )
        else:  # chirps
            analyzer = PerceptualChirpSignalHD(sample_rate)
            result = analyzer.compute_distortion(
                recorded_signal, stimulus_metadata, harmonic_orders,
                harmonic_mask=(mask_matrix, masking_mask_matrix, fund_freqs, time_array, fundamental_bins),
                spl_calibration_db=spl_calibration_db,
                masking_config=masking_config,
                noise_spectrum=noise_spectrum
            )

        # Extract results
        freq_value = result['frequencies']
        perceptual_loudness = result['perceptual_loudness']
        harmonic = np.array(harmonic_orders)

        return freq_value, harmonic, perceptual_loudness

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
    def spl_calculation(recorded_signal,
                        reference_pressure: float = 20e-6, 
                        window_size: int = 1201, 
                        method: str = "rms", 
                        padding_mode: str = "zero", 
                        padding_cval: Optional[float] = 0.0,
                        deviation: Optional[float] = None):
        """
            Compute SPL (dB) with a sliding window.

            - method: "rms" | "envelope" | "hilbert" (default "rms")
            - padding_mode: "reflect" | "nearest" | "mirror" | "wrap" | "constant"
              Accepts alias "zero" → "constant"
            - padding_cval: used only when padding_mode == "constant"
        """
        method = str(method).strip().lower()
        padding_mode_in = str(padding_mode).strip().lower()

        alias_to_mode = {
            "zero": "constant",
            "zeros": "constant",
            "const": "constant",
            "constant": "constant",
        }
        valid_modes = {"reflect", "nearest", "mirror", "wrap", "constant"}
        mode = alias_to_mode.get(padding_mode_in, padding_mode_in)
        if mode not in valid_modes:
            mode = "constant"

        valid_methods = {"rms", "envelope", "hilbert"}
        if method not in valid_methods:
            method = "rms"

        if mode == "constant":
            cval = 0.0 if padding_cval is None else float(padding_cval)
        else:
            cval = 0.0

        signal_float = np.asarray(recorded_signal, dtype=float)
        if method == "envelope":
            amplitude_list = maximum_filter(np.abs(signal_float), size=window_size, mode=mode, cval=cval)
        elif method == "hilbert":
            amplitude_envelope = np.abs(hilbert(signal_float))
            amplitude_list = uniform_filter1d(
                amplitude_envelope,
                size=window_size,
                axis=0,
                mode=mode,
                cval=cval
            )
        else: 
            amplitude_list = np.sqrt(
                uniform_filter1d(signal_float ** 2, size=window_size, axis=0, mode=mode, cval=cval)
            )

        # Avoid log of zero or negative due to numerical issues
        amplitude_list = np.maximum(np.asarray(amplitude_list, dtype=float), 1.0e-10)    
        spl = 20 * np.log10(amplitude_list / float(reference_pressure))
        if(deviation is not None):
            spl = spl + deviation
        return spl

    @staticmethod    
    def calculate_loose_particle_spl(recorded_signal, cutoff, sr, kernel_size):
        """
            Calculate the sound pressure level of loose particles.

            This method processes the recorded signal using Fourier Transform and median filtering to compute its sound pressure level.

            Args:
                -recorded_signal (np.array): The recorded audio signal.
                -kernel_size (int): The size of the median filter kernel, must be an odd number.

            Returns:
                -filtered_spl:np.array 
                    The sound pressure level array after median filtering.
                -rms_deviation: float
                    The root mean square deviation of the sound pressure level.
        """
        nyquist = 0.5 * sr
        normal_cutoff = cutoff / nyquist
        b, a = bessel(4, normal_cutoff, btype='high', analog=False)
        analytic_signal = filtfilt(b, a, recorded_signal)
        amplitude = np.abs(analytic_signal)
        reference_pressure = 20e-6
        signal_spl = 20 * np.log10(amplitude / reference_pressure)
        filtered_spl = medfilt(signal_spl, kernel_size)
        sum_squares = float()
        for i in range(len(filtered_spl)): 
            sum_squares += filtered_spl[i] ** 2
        rms_deviation = np.sqrt(sum_squares / len(filtered_spl)) * (np.sqrt(2) / 2)

        return filtered_spl, rms_deviation


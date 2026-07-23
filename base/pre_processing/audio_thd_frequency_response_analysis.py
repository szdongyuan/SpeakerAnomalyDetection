from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import hilbert
from scipy.ndimage import maximum_filter, uniform_filter1d
from scipy.signal import savgol_filter, medfilt, bessel, filtfilt
import librosa

from base.utils.plot_audio_features import PlotManager
from base.core_algorithm.harmonic_distortion.harmonic_index_builder import HarmonicIndexBuilder
from base.core_algorithm.harmonic_distortion.step_signal_hd import StepSignalHD
from base.core_algorithm.harmonic_distortion.synchronous_harmonic_detector import (
    SynchronousHarmonicDetector,
    validate_selected_harmonic_orders,
)
from base.core_algorithm.harmonic_distortion.chirp_signal_hd import ChirpSignalHD
from base.core_algorithm.harmonic_distortion.perceptual_step_signal_hd import PerceptualStepSignalHD
from base.core_algorithm.harmonic_distortion.perceptual_chirp_signal_hd import PerceptualChirpSignalHD
from base.stimulus_signal.frequency_stepped import resolve_frequency_stepped_schedule
from consts.frequency_stepped_consts import (
    FREQUENCY_STEPPED_DEFAULT_MAX_HARMONIC_ORDER,
    FREQUENCY_STEPPED_METHOD,
    FREQUENCY_STEPPED_MIN_PERCEPTUAL_HARMONIC_ORDER,
)
from consts.harmonic_detection_consts import (
    HARMONIC_DETECTION_METHOD_FOURIER,
    HARMONIC_DETECTION_METHOD_KEY,
    HARMONIC_DETECTION_METHOD_SYNCHRONOUS,
    normalize_harmonic_detection_method,
)


def _cached_segment_bounds(segment_payload):
    try:
        start = int(segment_payload["start_sample"])
        end = int(segment_payload["end_sample"])
        sample_count = int(segment_payload["sample_count"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("frequency_stepped segment metadata is invalid") from None
    if start < 0 or end < start or sample_count != end - start:
        raise ValueError("frequency_stepped segment metadata is invalid")
    return start, end


def _validate_cached_frequency_stepped_segments(stimulus_metadata, signal_length, sample_rate):
    segments = stimulus_metadata.get("segments")
    if segments is None:
        return
    schedule_sample_rate = stimulus_metadata.get("schedule_sample_rate")
    if schedule_sample_rate is not None:
        if int(schedule_sample_rate) != int(sample_rate):
            return
    if not isinstance(segments, (list, tuple)):
        raise ValueError("frequency_stepped segments must be a list")

    bounds = [_cached_segment_bounds(segment) for segment in segments]
    previous_end = -1
    for index, (start, end) in enumerate(bounds):
        if start < previous_end:
            raise ValueError("frequency_stepped segments overlap")
        if end > signal_length:
            has_later_recorded_segment = any(later_start < signal_length for later_start, _ in bounds[index + 1 :])
            if has_later_recorded_segment:
                raise ValueError("frequency_stepped segment is non-trailing out of range")
            return
        previous_end = end


def _frequency_stepped_analysis_segments(recorded_signal, stimulus_metadata, sample_rate):
    recorded = np.asarray(recorded_signal, dtype=float)
    if recorded.ndim != 1:
        recorded = np.ravel(recorded)

    _validate_cached_frequency_stepped_segments(stimulus_metadata, int(recorded.size), sample_rate)
    schedule = resolve_frequency_stepped_schedule(stimulus_metadata, sample_rate)

    segments = []
    all_segments = list(schedule.segments)
    for index, segment in enumerate(all_segments):
        start = int(segment.start_sample)
        end = int(segment.end_sample)
        if start < 0 or end < start:
            raise ValueError("frequency_stepped resolved segment metadata is invalid")
        if end > recorded.size:
            has_later_recorded_segment = any(
                int(later.start_sample) < recorded.size for later in all_segments[index + 1 :]
            )
            if has_later_recorded_segment:
                raise ValueError("frequency_stepped segment is non-trailing out of range")
            break
        if start >= recorded.size:
            break
        segment_signal = recorded[start:end]
        if segment_signal.size != end - start:
            raise ValueError("frequency_stepped segment slicing failed")
        segments.append((segment, segment_signal))
    return schedule, segments


def _frequency_stepped_index_row(frequency_hz, sample_rate, n_fft, max_harmonic_order):
    fft_freqs = np.fft.rfftfreq(int(n_fft), d=1.0 / float(sample_rate))
    nyquist = float(sample_rate) / 2.0
    index_row = np.zeros(int(max_harmonic_order) + 1, dtype=np.int32)
    for harmonic_order in range(1, int(max_harmonic_order) + 1):
        harmonic_freq = float(frequency_hz) * harmonic_order
        if harmonic_freq < nyquist:
            index_row[harmonic_order] = int(np.argmin(np.abs(fft_freqs - harmonic_freq))) + 1
    return index_row, np.insert(fft_freqs, 0, 0.0)


def _frequency_stepped_stft_magnitude(segment_signal, sample_rate, window_type):
    segment = np.asarray(segment_signal, dtype=float)
    n_fft = int(segment.size)
    _, _, stft_matrix = signal.stft(
        segment,
        fs=int(sample_rate),
        window=window_type,
        nperseg=n_fft,
        noverlap=0,
        nfft=n_fft,
        return_onesided=True,
        boundary=None,
        padded=False,
    )
    return np.abs(stft_matrix)


def _frequency_stepped_spectrum_column(segment_signal, sample_rate, stft_window_type):
    spectrum = _frequency_stepped_stft_magnitude(segment_signal, sample_rate, stft_window_type)
    return np.insert(spectrum, 0, 0.0, axis=0)


class AudioThdFrequencyResponseAnalysis(object):

    @staticmethod
    def _resample_to_min_sr_for_loudness(
        recorded_signal: np.ndarray,
        original_sr: int,
        target_sr: int,
        expected_len_target: Optional[int],
    ) -> tuple[np.ndarray, int]:
        """
        Resample the time-domain signal to meet loudness model requirements.

        We resample when original_sr < target_sr (e.g., 44.1 kHz -> 48 kHz) so that
        ISO 532-1 fine-band loudness has valid frequency coverage up to 24 kHz.

        Returns:
            (resampled_signal, target_sr)
        """

        def _maybe_adjust_length(y: np.ndarray, expected: Optional[int]) -> np.ndarray:
            if expected is None:
                return y
            expected_n = int(expected)
            if expected_n <= 0:
                return y
            # Only adjust when the mismatch is small (rounding / resampling artifacts).
            # If the mismatch is large (e.g., extra pre/post-roll), do not trim.
            tol = max(128, int(0.001 * expected_n))
            if abs(int(y.size) - expected_n) > tol:
                return y
            if y.size < expected_n:
                return np.pad(y, (0, expected_n - y.size), mode="constant", constant_values=0.0)
            if y.size > expected_n:
                return y[:expected_n]
            return y

        if int(original_sr) >= int(target_sr):
            y = np.asarray(recorded_signal, dtype=np.float32)
            return _maybe_adjust_length(y, expected_len_target), int(original_sr)

        try:
            from scipy.signal import resample_poly
        except Exception as e:
            raise ImportError(
                "PRB loudness requires resampling to >= 48 kHz when input sample rate < 48 kHz, "
                "but scipy is not available to perform resampling."
            ) from e

        y = np.asarray(recorded_signal, dtype=np.float32)
        y = resample_poly(y, up=int(target_sr), down=int(original_sr)).astype(np.float32, copy=False)
        y = _maybe_adjust_length(y, expected_len_target)
        return y, int(target_sr)

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
        results = {"thd_fig": None, "harmonic_fig": None, "frequency_response_fig": None}
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
                if "stimulus_metadata" not in thd_kwargs:
                    raise ValueError(
                        "thd_kwargs must contain 'stimulus_metadata'. "
                        "Legacy methods have been removed. "
                        "See docs/hd_refactoring_guide.md for migration instructions."
                    )

                x, h, thd = self.calculate_thd_three_phase(recorded_signal[i], sr[i], thd_kwargs)

                # Apply 1/6 octave smoothing for chirp signals only
                stimulus_metadata = thd_kwargs["stimulus_metadata"]
                octave_smoothing = 6 if stimulus_metadata["stimulus_method"] == "chirps" else None

                pm.plot_thd(ax_thd, x, thd, octave_smoothing=octave_smoothing)
                pm.plot_harmonic(ax_harmonic, x, h)
            if kwargs.get("frequency_response", True):
                fr, frequency_list = self.calculate_fr(reference_signal, recorded_signal[i], sr[i])
                pm.plot_frequency_response(ax_fr, frequency_list, fr)
        return results

    def calculate_thd_three_phase(self, recorded_signal, sr, thd_kwargs):
        """
        NEW METHOD: Calculate THD using three-phase architecture.

        Ordinary step signals use synchronous harmonic fitting. Chirp signals use STFT.

        Returns: (x, h, thd) for plotting (backward compatible with existing plots)
        """
        stimulus_metadata = thd_kwargs["stimulus_metadata"]
        harmonic_orders = thd_kwargs.get("harmonic_orders", [2, 3, 4, 5])
        stimulus_method = stimulus_metadata["stimulus_method"]
        detection_method = normalize_harmonic_detection_method(
            thd_kwargs.get(HARMONIC_DETECTION_METHOD_KEY),
            strict=True,
        )

        if stimulus_method == FREQUENCY_STEPPED_METHOD:
            return self._calculate_frequency_stepped_thd(
                recorded_signal=recorded_signal,
                sample_rate=sr,
                stimulus_metadata=stimulus_metadata,
                harmonic_orders=harmonic_orders,
                stft_window_type=thd_kwargs.get("stft_window_type", "hann"),
                harmonic_detection_method=detection_method,
            )

        if stimulus_method == "steps":
            analyzer = StepSignalHD(sample_rate=sr)
            if detection_method == HARMONIC_DETECTION_METHOD_SYNCHRONOUS:
                result = analyzer.compute_distortion(
                    recorded_signal,
                    stimulus_metadata,
                    harmonic_orders,
                    harmonic_mask=None,
                    stft_window_type=thd_kwargs.get("stft_window_type", "hann"),
                    harmonic_detection_method=detection_method,
                )
                x = result["frequencies"]
                return x, result["harmonic_amplitudes"], result["thd"]

            if detection_method == HARMONIC_DETECTION_METHOD_FOURIER:
                validated_harmonic_orders = list(
                    validate_selected_harmonic_orders(
                        harmonic_orders,
                        FREQUENCY_STEPPED_DEFAULT_MAX_HARMONIC_ORDER,
                    )
                )
                single_rep_duration = stimulus_metadata["total_time"] / stimulus_metadata["repeat_times"]
                step_duration = single_rep_duration / stimulus_metadata["num_steps"]
                step_samples = int(step_duration * sr)
                builder = HarmonicIndexBuilder()
                index_matrix, fund_freqs, fft_freqs = builder.build_step_signal_index_matrix(
                    stimulus_metadata,
                    sr=sr,
                    n_fft=step_samples,
                    max_harmonic_order=FREQUENCY_STEPPED_DEFAULT_MAX_HARMONIC_ORDER,
                )
                mask_matrix = builder.create_mask_from_indices(index_matrix, validated_harmonic_orders, len(fft_freqs))
                fundamental_bins = index_matrix[:, 1]
                result = analyzer.compute_distortion(
                    recorded_signal,
                    stimulus_metadata,
                    validated_harmonic_orders,
                    harmonic_mask=(mask_matrix, fund_freqs, fundamental_bins),
                    stft_window_type=thd_kwargs.get("stft_window_type", "hann"),
                    harmonic_detection_method=detection_method,
                )
                x = result["frequencies"]
                h = np.zeros((6, len(x)), dtype=float)
                h[0, :] = x
                spectrum = result["spectrum_matrix"]
                for step_idx in range(len(x)):
                    for harmonic_order in range(1, 6):
                        if harmonic_order < index_matrix.shape[1]:
                            bin_idx = int(index_matrix[step_idx, harmonic_order])
                            if bin_idx > 0:
                                h[harmonic_order, step_idx] = float(spectrum[bin_idx, step_idx])
                return x, h, result["thd"]

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1A: Build Overall Index Matrix
        # ═══════════════════════════════════════════════════════════════════
        builder = HarmonicIndexBuilder()

        if stimulus_metadata["stimulus_method"] == "chirps":
            stft_window_size = thd_kwargs.get("stft_window_size", 2048)
            stft_hop_size = thd_kwargs.get("stft_hop_size", 1024)

            index_matrix, fund_freqs, time_array, fft_freqs = builder.build_chirp_signal_index_matrix(
                stimulus_metadata,
                sr=sr,
                n_fft=stft_window_size,
                hop_length=stft_hop_size,
                max_harmonic_order=FREQUENCY_STEPPED_DEFAULT_MAX_HARMONIC_ORDER,
            )
        else:
            raise ValueError(f"Unsupported stimulus_method: {stimulus_metadata['stimulus_method']}")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1B: Select User Configuration
        # ═══════════════════════════════════════════════════════════════════
        mask_matrix = builder.create_mask_from_indices(index_matrix, harmonic_orders, len(fft_freqs))
        fundamental_bins = index_matrix[:, 1]

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 2: Calculate THD
        # ═══════════════════════════════════════════════════════════════════
        if stimulus_metadata["stimulus_method"] == "chirps":
            analyzer = ChirpSignalHD(sample_rate=sr)
            stft_window_size = thd_kwargs.get("stft_window_size", 2048)
            stft_hop_size = thd_kwargs.get("stft_hop_size", 1024)

            if "time_array" not in locals():
                # Rebuild time_array if needed
                single_rep_duration = stimulus_metadata["total_time"] / stimulus_metadata["repeat_times"]
                num_samples = int(single_rep_duration * sr)
                num_frames = 1 + (num_samples - stft_window_size) // stft_hop_size
                time_array = (np.arange(num_frames) * stft_hop_size + stft_window_size / 2) / sr

            result = analyzer.compute_distortion(
                recorded_signal,
                stimulus_metadata,
                harmonic_orders,
                harmonic_mask=(mask_matrix, None, fund_freqs, time_array, fundamental_bins),
                stft_window_size=stft_window_size,
                stft_hop_size=stft_hop_size,
            )

            x = result["frequencies"]
            thd = result["thd"]
            # h needs to be (6, n_frames) for plotting
            h = np.zeros((6, len(x)))
            h[0, :] = x

        return x, h, thd

    def _calculate_frequency_stepped_thd(
        self,
        *,
        recorded_signal,
        sample_rate,
        stimulus_metadata,
        harmonic_orders,
        stft_window_type="hann",
        harmonic_detection_method=HARMONIC_DETECTION_METHOD_SYNCHRONOUS,
    ):
        schedule, segment_pairs = _frequency_stepped_analysis_segments(
            recorded_signal,
            stimulus_metadata,
            int(sample_rate),
        )
        if not segment_pairs:
            return np.array([]), np.zeros((6, 0), dtype=float), np.array([])

        method = normalize_harmonic_detection_method(harmonic_detection_method, strict=True)
        if method == HARMONIC_DETECTION_METHOD_FOURIER:
            return self._calculate_frequency_stepped_fourier_thd(
                segment_pairs=segment_pairs,
                sample_rate=int(sample_rate),
                harmonic_orders=harmonic_orders,
                stft_window_type=stft_window_type,
            )

        detector = SynchronousHarmonicDetector()

        frequencies = []
        harmonic_columns = []
        thd_values = []
        for segment, segment_signal in segment_pairs:
            amplitudes, distortion = detector.analyze(
                segment_signal,
                f0=float(segment.frequency_hz),
                sample_rate=int(sample_rate),
                harmonic_orders=harmonic_orders,
                stft_window_type=stft_window_type,
            )

            frequencies.append(float(segment.frequency_hz))
            harmonic_column = np.zeros(6, dtype=float)
            harmonic_column[0] = float(segment.frequency_hz)
            for order in range(1, 6):
                harmonic_column[order] = float(amplitudes.get(order, 0.0))
            harmonic_columns.append(harmonic_column)
            thd_values.append(float(distortion))

        x = np.asarray(frequencies, dtype=float)
        h = np.column_stack(harmonic_columns)
        thd = np.asarray(thd_values, dtype=float)
        sort_idx = np.argsort(x, kind="stable")
        return x[sort_idx], h[:, sort_idx], thd[sort_idx]

    def _calculate_frequency_stepped_fourier_thd(
        self,
        *,
        segment_pairs,
        sample_rate,
        harmonic_orders,
        stft_window_type="hann",
    ):
        validated_harmonic_orders = list(
            validate_selected_harmonic_orders(
                harmonic_orders,
                FREQUENCY_STEPPED_DEFAULT_MAX_HARMONIC_ORDER,
            )
        )
        max_harmonic_order = max([FREQUENCY_STEPPED_DEFAULT_MAX_HARMONIC_ORDER, *validated_harmonic_orders])
        builder = HarmonicIndexBuilder()
        analyzer = StepSignalHD(sample_rate=int(sample_rate))

        frequencies = []
        harmonic_columns = []
        thd_values = []
        for segment, segment_signal in segment_pairs:
            n_fft = int(segment_signal.size)
            index_row, fft_freqs = _frequency_stepped_index_row(
                segment.frequency_hz,
                int(sample_rate),
                n_fft,
                max_harmonic_order,
            )
            index_matrix = index_row.reshape(1, -1)
            spectrum = _frequency_stepped_spectrum_column(segment_signal, int(sample_rate), stft_window_type)
            mask_matrix = builder.create_mask_from_indices(
                index_matrix,
                validated_harmonic_orders,
                len(fft_freqs),
            )
            fundamental_bins = index_matrix[:, 1]
            thd = analyzer.compute_thd_batch(spectrum, mask_matrix, fundamental_bins)

            harmonic_column = np.zeros(6, dtype=float)
            harmonic_column[0] = float(segment.frequency_hz)
            for order in range(1, 6):
                if order < index_matrix.shape[1]:
                    bin_idx = int(index_matrix[0, order])
                    if bin_idx > 0:
                        harmonic_column[order] = float(spectrum[bin_idx, 0])

            frequencies.append(float(segment.frequency_hz))
            harmonic_columns.append(harmonic_column)
            thd_values.append(float(np.asarray(thd).reshape(-1)[0]))

        x = np.asarray(frequencies, dtype=float)
        h = np.column_stack(harmonic_columns)
        thd = np.asarray(thd_values, dtype=float)
        sort_idx = np.argsort(x, kind="stable")
        return x[sort_idx], h[:, sort_idx], thd[sort_idx]

    def calculate_perceptual_thd_three_phase(
        self, recorded_signal: np.ndarray, sample_rate: int, thd_kwargs: dict, v2pa_factor: float = 1.0
    ) -> tuple:
        """
        Calculate perceptual loudness (phons) using three-phase architecture with psychoacoustic models.

        Similar to calculate_thd_three_phase but returns perceived loudness instead of THD percentage.

        Args:
            recorded_signal: Recorded audio signal
            sample_rate: Sample rate
            thd_kwargs: {
                'stimulus_metadata': dict with stimulus configuration,
                'harmonic_orders': list of harmonic orders (e.g., [10, 11, 12])
            }
            v2pa_factor: Microphone calibration multiplier (V -> Pa), default 1.0.
                Applied in amplitude domain before log transform:
                calibrated_amp = amp * v2pa_factor

        Returns:
            (freq_value, harmonic, perceptual_loudness):
                - freq_value: Fundamental frequencies
                - harmonic: Harmonic orders array
                - perceptual_loudness: Perceived loudness in phons
        """
        stimulus_metadata = thd_kwargs["stimulus_metadata"]
        harmonic_orders = thd_kwargs.get("harmonic_orders", [])

        # Loudness (mosqito / ISO 532-1) requires analysis at >= 48 kHz so that
        # the fine-band spectrum covers up to 24 kHz. For lower-rate recordings
        # (e.g., 44.1 kHz), resample to 48 kHz as the standard analysis path.
        analysis_sr = int(sample_rate)
        if analysis_sr < 48000:
            expected_len_target = None
            total_time = stimulus_metadata.get("total_time")
            if total_time is not None:
                expected_len_target = int(round(float(total_time) * 48000))
            recorded_signal, analysis_sr = self._resample_to_min_sr_for_loudness(
                recorded_signal=recorded_signal,
                original_sr=analysis_sr,
                target_sr=48000,
                expected_len_target=expected_len_target,
            )

        # Phase 1A: Build overall index matrix
        builder = HarmonicIndexBuilder()

        stimulus_method = stimulus_metadata["stimulus_method"]

        if stimulus_method == FREQUENCY_STEPPED_METHOD:
            return self._calculate_frequency_stepped_perceptual_thd(
                recorded_signal=recorded_signal,
                sample_rate=analysis_sr,
                stimulus_metadata=stimulus_metadata,
                harmonic_orders=harmonic_orders,
                masking_config=thd_kwargs.get("masking_config"),
                v2pa_factor=v2pa_factor,
                stft_window_type=thd_kwargs.get("stft_window_type", "hann"),
            )

        if stimulus_method == "steps":
            # Calculate STFT parameters (full step duration - no trimming)
            single_rep_duration = stimulus_metadata["total_time"] / stimulus_metadata["repeat_times"]
            step_duration = single_rep_duration / stimulus_metadata["num_steps"]
            step_samples = int(step_duration * analysis_sr)
            n_fft = step_samples  # STFT window size = step duration

            index_matrix, fund_freqs, fft_freqs = builder.build_step_signal_index_matrix(
                stimulus_metadata,
                sr=analysis_sr,
                n_fft=n_fft,
                max_harmonic_order=FREQUENCY_STEPPED_DEFAULT_MAX_HARMONIC_ORDER,
            )
        elif stimulus_method == "chirps":
            stft_window_size = thd_kwargs.get("stft_window_size", 2048)
            stft_hop_size = thd_kwargs.get("stft_hop_size", 1024)

            index_matrix, fund_freqs, time_array, fft_freqs = builder.build_chirp_signal_index_matrix(
                stimulus_metadata,
                sr=analysis_sr,
                n_fft=stft_window_size,
                hop_length=stft_hop_size,
                max_harmonic_order=FREQUENCY_STEPPED_DEFAULT_MAX_HARMONIC_ORDER,
            )
        else:
            raise ValueError(f"Unsupported stimulus_method: {stimulus_method}")

        # Phase 1B: Create mask from selected harmonics
        mask_matrix = builder.create_mask_from_indices(index_matrix, harmonic_orders, len(fft_freqs))
        fundamental_bins = index_matrix[:, 1]

        # Build masking_mask_matrix for cumulative masking
        masking_mask_matrix = None
        masking_config = thd_kwargs.get("masking_config")
        if masking_config and masking_config.get("enable_cumulative"):
            # For cumulative masking, include all lower-order harmonics up to max analyzed harmonic
            max_harmonic = max(harmonic_orders)
            masking_orders = list(range(1, max_harmonic))  # Fundamental to (max - 1)

            if masking_orders:  # Only create if there are masking harmonics
                masking_mask_matrix = builder.create_mask_from_indices(index_matrix, masking_orders, len(fft_freqs))

        # Phase 2: Compute perceptual loudness using perceptual analyzers
        if stimulus_method == "steps":
            analyzer = PerceptualStepSignalHD(analysis_sr)
            result = analyzer.compute_distortion(
                recorded_signal,
                stimulus_metadata,
                harmonic_orders,
                harmonic_mask=(mask_matrix, masking_mask_matrix, fund_freqs, fundamental_bins),
                v2pa_factor=v2pa_factor,
                masking_config=masking_config,
            )
        else:  # chirps
            analyzer = PerceptualChirpSignalHD(analysis_sr)
            result = analyzer.compute_distortion(
                recorded_signal,
                stimulus_metadata,
                harmonic_orders,
                harmonic_mask=(mask_matrix, masking_mask_matrix, fund_freqs, time_array, fundamental_bins),
                v2pa_factor=v2pa_factor,
                masking_config=masking_config,
            )

        # Extract results
        freq_value = result["frequencies"]
        perceptual_loudness = result["perceptual_loudness"]
        harmonic = np.array(harmonic_orders)

        return freq_value, harmonic, perceptual_loudness

    def _calculate_frequency_stepped_perceptual_thd(
        self,
        *,
        recorded_signal,
        sample_rate,
        stimulus_metadata,
        harmonic_orders,
        masking_config,
        v2pa_factor,
        stft_window_type="hann",
    ):
        schedule, segment_pairs = _frequency_stepped_analysis_segments(
            recorded_signal,
            stimulus_metadata,
            int(sample_rate),
        )
        if not segment_pairs:
            return np.array([]), np.asarray(harmonic_orders), np.array([])

        max_harmonic_order = max(
            [
                FREQUENCY_STEPPED_DEFAULT_MAX_HARMONIC_ORDER,
                *[int(order) for order in harmonic_orders],
                FREQUENCY_STEPPED_MIN_PERCEPTUAL_HARMONIC_ORDER,
            ]
        )
        builder = HarmonicIndexBuilder()
        analyzer = PerceptualStepSignalHD(int(sample_rate))

        frequencies = []
        loudness_values = []
        for segment, segment_signal in segment_pairs:
            n_fft = int(segment_signal.size)
            index_row, fft_freqs = _frequency_stepped_index_row(
                segment.frequency_hz,
                int(sample_rate),
                n_fft,
                max_harmonic_order,
            )
            index_matrix = index_row.reshape(1, -1)
            spectrum = _frequency_stepped_spectrum_column(segment_signal, int(sample_rate), stft_window_type)
            mask_matrix = builder.create_mask_from_indices(index_matrix, harmonic_orders, len(fft_freqs))
            fundamental_bins = index_matrix[:, 1]
            loudness = analyzer.compute_perceptual_thd_batch(
                spectrum,
                mask_matrix,
                fundamental_bins,
                np.asarray([float(segment.frequency_hz)], dtype=float),
                masking_mask_matrix=None,
                masking_config=masking_config,
                v2pa_factor=v2pa_factor,
                n_fft=n_fft,
            )
            frequencies.append(float(segment.frequency_hz))
            loudness_values.append(float(np.asarray(loudness).reshape(-1)[0]))

        freq_value = np.asarray(frequencies, dtype=float)
        perceptual_loudness = np.asarray(loudness_values, dtype=float)
        sort_idx = np.argsort(freq_value, kind="stable")
        return freq_value[sort_idx], np.asarray(harmonic_orders), perceptual_loudness[sort_idx]

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
            Zxx = librosa.stft(
                reference_signal,
                n_fft=frame_size,
                hop_length=hop_length,
                win_length=frame_size,
                window=kwargs.get("window", "hann"),
            )
            f_stft = librosa.fft_frequencies(sr=sr, n_fft=frame_size)
            amp = np.abs(Zxx)
            max_amp_indices = np.argmax(amp, axis=0)
            f0 = f_stft[max_amp_indices]
        elif kwargs.get("method", "yin") == "cqt":  ## 后续添加
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
            fmin = librosa.note_to_hz("C1")  # 32.7 Hz

        if fmax is None:
            fmax = librosa.note_to_hz("C9")

        if bins_per_octave is None:
            # 表示每个八度内有多少频率点，对应的频率对数增加, 增加n_fft会增加频率分辨率，和预期相符
            bins_per_octave = int(12 * np.log2(n_fft / 1024) + 24)
            bins_per_octave = max(12, bins_per_octave)

        if n_bins is None:
            n_octaves = np.log2(fmax / fmin)  # 八度，表示频率区间跨越了多少频率翻倍的区间
            n_bins = int(
                np.ceil(n_octaves * bins_per_octave)
            )  # n_bins ≈ bins_per_octave * log2(fmax/fmin)，最终的频域采样点数

        # fmax parameter is not supported in librosa.cqt
        C = librosa.cqt(y=y, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)

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
        frequency_list, pxy = signal.csd(
            reference_signal, recorded_signal, window=window, nfft=num, noverlap=hop_length, fs=sr
        )
        _, pxx = signal.welch(reference_signal, fs=sr, nperseg=num, noverlap=hop_length, window=window)
        h = np.abs(pxy / pxx)
        fr = 10 * np.log10(h)
        slice_length = 1024
        idx_1 = np.argmax(np.abs(np.fft.fft(reference_signal[:slice_length], num)[: num // 2]))
        mid_slice = len(reference_signal) // 2
        idx_2 = np.argmax(
            np.abs(
                np.fft.fft(
                    reference_signal[mid_slice - (slice_length // 2) - 1 : mid_slice + (slice_length // 2) + 1], num
                )[: num // 2]
            )
        )
        idx_3 = np.argmax(np.abs(np.fft.fft(reference_signal[-slice_length:], num)[: num // 2]))
        start_idx, stop_idx = min([idx_1, idx_2, idx_3]), max([idx_1, idx_2, idx_3])
        if is_smooth:
            fr = savgol_filter(fr, window_length=60, polyorder=3)
        return fr[start_idx:stop_idx], frequency_list[start_idx:stop_idx]

    @staticmethod
    def spl_calculation(
        recorded_signal,
        reference_pressure: float = 20e-6,
        window_size: int = 1201,
        method: str = "rms",
        padding_mode: str = "zero",
        padding_cval: Optional[float] = 0.0,
        v2pa_factor: Optional[float] = None,
        trim_edges: bool = False,
    ):
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
            v_amp_list = maximum_filter(np.abs(signal_float), size=window_size, mode=mode, cval=cval)
        elif method == "hilbert":
            amplitude_envelope = np.abs(hilbert(signal_float))
            v_amp_list = uniform_filter1d(amplitude_envelope, size=window_size, axis=0, mode=mode, cval=cval)
        else:
            v_amp_list = np.sqrt(uniform_filter1d(signal_float**2, size=window_size, axis=0, mode=mode, cval=cval))

        if not v2pa_factor:
            v2pa_factor = 1
        pa_amp_list = np.asarray(v_amp_list, dtype=float) * v2pa_factor
        pa_amp_list = np.maximum(pa_amp_list, 1.0e-10)  # Avoid log of zero or negative due to numerical issues
        spl = 20 * np.log10(pa_amp_list / float(reference_pressure))

        if not trim_edges:
            return spl

        n = int(signal_float.size)
        w = int(window_size)
        if w <= 1 or w > n:
            return spl
        out_len = n - w + 1
        start = w // 2
        end = start + out_len
        return spl[start:end]

    @staticmethod
    def calculate_loose_particle_spl(recorded_signal, cutoff, sr, kernel_size, v2pa_factor):
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
        b, a = bessel(4, normal_cutoff, btype="high", analog=False)
        analytic_signal = filtfilt(b, a, recorded_signal)
        signal_spl = AudioThdFrequencyResponseAnalysis.spl_calculation(analytic_signal, v2pa_factor=float(v2pa_factor))
        filtered_spl = medfilt(signal_spl, kernel_size)
        sum_squares = float()
        for i in range(len(filtered_spl)):
            sum_squares += filtered_spl[i] ** 2
        rms_deviation = np.sqrt(sum_squares / len(filtered_spl)) * (np.sqrt(2) / 2)

        return filtered_spl, rms_deviation

"""
StepSignalHD - Phase 2 analyzer for step signals

Computes THD for step signals using pre-built masks from Phase 1B.
"""
import numpy as np
from typing import Dict, Tuple
from scipy import signal as scipy_signal
from base.core_algorithm.harmonic_distortion.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer
from base.core_algorithm.harmonic_distortion.harmonic_index_builder import HarmonicIndexBuilder
from base.core_algorithm.harmonic_distortion.synchronous_harmonic_detector import SynchronousHarmonicDetector
from consts.harmonic_detection_consts import (
    HARMONIC_DETECTION_METHOD_FOURIER,
    HARMONIC_DETECTION_METHOD_SYNCHRONOUS,
    normalize_harmonic_detection_method,
)


class StepSignalHD(HarmonicDistortionAnalyzer):
    """THD analyzer for step signals."""

    def compute_distortion(
        self,
        recorded_signal: np.ndarray,
        stimulus_metadata: Dict,
        harmonic_orders: list,
        harmonic_mask: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
        stft_window_type: str = 'hann',
        **kwargs
    ) -> Dict:
        """
        Compute THD for ordinary step signals using synchronous harmonic fitting.

        Args:
            recorded_signal: Recorded audio
            stimulus_metadata: Config with num_steps, repeat_times, total_time
            harmonic_orders: Selected harmonics
            harmonic_mask: Optional (mask_matrix, fundamental_freqs, fundamental_bins) from Phase 1B
            stft_window_type: Window function for weighted fitting (default 'hann')

        Returns:
            {
                'frequencies': fundamental_freqs,
                'thd': thd_values,
                'num_repetitions': repeat_times,
                'harmonic_amplitudes': averaged fitted amplitudes, rows 0-5
            }
        """
        method = normalize_harmonic_detection_method(
            kwargs.get("harmonic_detection_method", HARMONIC_DETECTION_METHOD_SYNCHRONOUS),
            strict=True,
        )
        if method == HARMONIC_DETECTION_METHOD_FOURIER:
            return self._compute_distortion_fourier(
                recorded_signal,
                stimulus_metadata,
                harmonic_orders,
                harmonic_mask=harmonic_mask,
                stft_window_type=stft_window_type,
            )
        return self._compute_distortion_synchronous(
            recorded_signal,
            stimulus_metadata,
            harmonic_orders,
            harmonic_mask=harmonic_mask,
            stft_window_type=stft_window_type,
        )

    def _compute_distortion_synchronous(
        self,
        recorded_signal: np.ndarray,
        stimulus_metadata: Dict,
        harmonic_orders: list,
        harmonic_mask: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
        stft_window_type: str = 'hann',
    ) -> Dict:
        if harmonic_mask is not None:
            _, fundamental_freqs, _ = harmonic_mask
        else:
            fundamental_freqs = self._step_frequencies(stimulus_metadata)

        num_steps = int(stimulus_metadata['num_steps'])
        repeat_times = int(stimulus_metadata['repeat_times'])
        total_time = float(stimulus_metadata['total_time'])
        single_rep_duration = total_time / repeat_times
        step_duration = single_rep_duration / num_steps
        step_samples = int(step_duration * self.sample_rate)

        # Split into repetitions
        repetitions = self._split_repetitions(recorded_signal, repeat_times)

        detector = SynchronousHarmonicDetector()
        thd_per_rep = []
        amplitude_per_rep = []
        fundamental_freqs = np.asarray(fundamental_freqs, dtype=float)
        for repetition_signal in repetitions:
            rep_thd = []
            rep_amplitudes = np.zeros((6, num_steps), dtype=float)
            rep_amplitudes[0, :] = fundamental_freqs[:num_steps]

            for step_idx, f0 in enumerate(fundamental_freqs[:num_steps]):
                start = step_idx * step_samples
                end = start + step_samples
                segment = repetition_signal[start:end]
                amplitudes, distortion = detector.analyze(
                    segment,
                    f0=float(f0),
                    sample_rate=self.sample_rate,
                    harmonic_orders=harmonic_orders,
                    stft_window_type=stft_window_type,
                )
                rep_thd.append(distortion)
                for order in range(1, 6):
                    rep_amplitudes[order, step_idx] = float(amplitudes.get(order, 0.0))

            thd_per_rep.append(np.asarray(rep_thd, dtype=float))
            amplitude_per_rep.append(rep_amplitudes)

        # Average across repetitions
        averaged_thd = np.mean(thd_per_rep, axis=0)
        averaged_amplitudes = np.mean(amplitude_per_rep, axis=0)
        num_frames = len(averaged_thd)

        return {
            'frequencies': fundamental_freqs[:num_frames],
            'thd': averaged_thd,
            'num_repetitions': repeat_times,
            'harmonic_amplitudes': averaged_amplitudes[:, :num_frames],
        }

    def _compute_distortion_fourier(
        self,
        recorded_signal: np.ndarray,
        stimulus_metadata: Dict,
        harmonic_orders: list,
        harmonic_mask: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
        stft_window_type: str = "hann",
    ) -> Dict:
        if harmonic_mask is None:
            raise ValueError("Fourier step HD/RB analysis requires a harmonic mask")
        mask_matrix, fundamental_freqs, fundamental_bins = harmonic_mask

        num_steps = int(stimulus_metadata["num_steps"])
        repeat_times = int(stimulus_metadata["repeat_times"])
        total_time = float(stimulus_metadata["total_time"])
        repetitions = self._split_repetitions(recorded_signal, repeat_times)

        thd_per_rep = []
        spectrum_per_rep = []
        for repetition_signal in repetitions:
            single_rep_duration = total_time / repeat_times
            step_duration = single_rep_duration / num_steps
            step_samples = int(step_duration * self.sample_rate)
            spectrum_matrix = self._compute_stft(repetition_signal, step_samples, step_samples, stft_window_type)
            spectrum_with_dummy = np.insert(spectrum_matrix, 0, 0.0, axis=0)

            num_frames = min(spectrum_with_dummy.shape[1], mask_matrix.shape[1])
            spectrum_trimmed = spectrum_with_dummy[:, :num_frames]
            mask_trimmed = mask_matrix[:, :num_frames]
            fund_bins_trimmed = fundamental_bins[:num_frames]

            thd_per_rep.append(self.compute_thd_batch(spectrum_trimmed, mask_trimmed, fund_bins_trimmed))
            spectrum_per_rep.append(spectrum_trimmed)

        averaged_thd = np.mean(thd_per_rep, axis=0)
        averaged_spectrum = np.mean(spectrum_per_rep, axis=0)
        num_frames = len(averaged_thd)
        return {
            "frequencies": np.asarray(fundamental_freqs, dtype=float)[:num_frames],
            "thd": averaged_thd,
            "num_repetitions": repeat_times,
            "spectrum_matrix": averaged_spectrum,
        }

    def _split_repetitions(self, signal: np.ndarray, repeat_times: int) -> list:
        """Split signal into repetitions."""
        if repeat_times == 1:
            return [signal]

        rep_length = len(signal) // repeat_times
        return [signal[i*rep_length:(i+1)*rep_length] for i in range(repeat_times)]

    @staticmethod
    def _step_frequencies(stimulus_metadata: Dict) -> np.ndarray:
        num_steps = int(stimulus_metadata["num_steps"])
        start_freq = float(stimulus_metadata.get("start_freq", 100))
        stop_freq = float(stimulus_metadata.get("stop_freq", 1000))
        stimulus_type = stimulus_metadata.get("stimulus_type", "linear")
        if stimulus_type == "linear":
            return np.linspace(start_freq, stop_freq, num_steps)
        if stimulus_type == "log":
            return np.logspace(np.log10(start_freq), np.log10(stop_freq), num_steps)
        raise Exception("Invalid step type.")

    def _compute_stft(
        self,
        signal: np.ndarray,
        window_size: int,
        hop_size: int,
        window_type: str
    ) -> np.ndarray:
        """
        Compute STFT magnitude for step signal.

        For step signals with STFT, window_size = step_duration and
        hop_size = step_duration (no overlap), resulting in exactly
        one STFT frame per step.

        Args:
            signal: Input signal (one repetition)
            window_size: STFT window size (equals step duration in samples)
            hop_size: STFT hop size (equals step duration - no overlap)
            window_type: Window function type (e.g., 'hann')

        Returns:
            stft_magnitude: (n_bins, n_steps) magnitude spectrum
        """
        freqs, times, Zxx = scipy_signal.stft(
            signal,
            fs=self.sample_rate,
            window=window_type,
            nperseg=window_size,
            noverlap=window_size - hop_size,
            nfft=window_size,
            return_onesided=True,
            boundary=None,
            padded=False
        )

        return np.abs(Zxx)

    def _create_harmonic_mask(self, stimulus_metadata, harmonic_orders, masking_config=None):
        """
        Create binary mask matrices for harmonic extraction.

        Args:
            stimulus_metadata: Stimulus configuration
            harmonic_orders: List of harmonic orders to analyze (e.g., [10, 11, 12])
            masking_config: Optional masking configuration dict with keys:
                - 'masking_range': (start, end) harmonic orders for masking
                - 'enable_cumulative': bool to enable cumulative masking

        Returns:
            tuple: (mask_matrix, masking_mask_matrix, fundamental_freqs, fundamental_bins)
                - mask_matrix: Binary mask for selected harmonics
                - masking_mask_matrix: Binary mask for masking harmonics (or None)
                - fundamental_freqs: Fundamental frequencies
                - fundamental_bins: Fundamental bin indices
        """
        num_steps = stimulus_metadata['num_steps']
        total_time = stimulus_metadata['total_time']
        repeat_times = stimulus_metadata.get('repeat_times', 1)
        if not isinstance(repeat_times, int) or repeat_times <= 0:
            raise ValueError(f"repeat_times must be a positive integer, got {repeat_times}")

        # Calculate STFT parameters
        # NOTE: total_time is expected to include all repetitions (see docs/hd_refactoring_guide.md),
        # so STFT sizing should be derived from a single repetition.
        single_rep_duration = total_time / repeat_times
        step_duration = single_rep_duration / num_steps
        step_samples = int(step_duration * self.sample_rate)
        if step_samples < 2:
            raise ValueError(
                f"Step duration too short for STFT: step_samples={step_samples}. "
                f"Check total_time={total_time}, repeat_times={repeat_times}, num_steps={num_steps}."
            )
        n_fft = step_samples

        # Build index matrix with all harmonics
        builder = HarmonicIndexBuilder()

        # Determine max harmonic order needed
        max_analysis_order = max(harmonic_orders) if harmonic_orders else 12
        max_masking_order = 9  # Default
        if masking_config and masking_config.get('enable_cumulative', False):
            # Dynamic masking range: if not provided, infer from harmonic_orders
            if 'masking_range' in masking_config:
                masking_range = masking_config['masking_range']
                max_masking_order = masking_range[1]
            else:
                # Auto-compute: mask using harmonics 1 to (max_analyzed - 1)
                max_masking_order = max_analysis_order - 1

        max_harmonic_order = max(max_analysis_order, max_masking_order)

        # Need to add required metadata fields
        full_metadata = {
            'num_steps': num_steps,
            'start_freq': stimulus_metadata.get('start_freq', 100),
            'stop_freq': stimulus_metadata.get('stop_freq', 1000),
            'stimulus_type': stimulus_metadata.get('stimulus_type', 'linear'),
            'total_time': total_time
        }

        index_matrix, fundamental_freqs, fft_freqs_with_dummy = builder.build_step_signal_index_matrix(
            full_metadata, self.sample_rate, n_fft, max_harmonic_order
        )

        n_bins_with_dummy = len(fft_freqs_with_dummy)
        n_frames = num_steps

        # Create analysis mask for selected harmonics
        mask_matrix = builder.create_mask_from_indices(
            index_matrix, harmonic_orders, n_bins_with_dummy
        )

        # Extract fundamental bins (keep +1 offset for mask_matrix alignment)
        fundamental_bins = index_matrix[:, 1]

        # Create masking mask if cumulative masking enabled
        if masking_config and masking_config.get('enable_cumulative', False):
            # Dynamic masking range: if not provided, infer from harmonic_orders
            # Masking range should be (1, max_harmonic_order - 1)
            if 'masking_range' in masking_config:
                masking_range = masking_config['masking_range']
            else:
                # Auto-compute: mask using harmonics 1 to (max_analyzed - 1)
                masking_range = (1, max_analysis_order - 1)

            masking_orders = list(range(masking_range[0], masking_range[1] + 1))

            # Create masking mask - Note: masking_orders are actual harmonic numbers (1-9)
            # which map to columns 1-9 in the index_matrix, but we want harmonics 1-9,
            # not the fundamental. Column N in index_matrix = Nth harmonic.
            # For harmonics 1-9: columns 1-9, but column 1 is the fundamental.
            # Actually, looking at the index matrix structure:
            # - Column 0: sentinel
            # - Column 1: fundamental (0th harmonic conceptually, or 1st harmonic in some notations)
            # - Column 2: 2nd harmonic
            # - Column N: Nth harmonic
            #
            # The masking_range (1, 9) means we want the 1st through 9th harmonics.
            # In index_matrix terms:
            # - 1st harmonic = fundamental = column 1
            # - 2nd harmonic = column 2
            # - ...
            # - 9th harmonic = column 9
            masking_mask_matrix = np.zeros((n_bins_with_dummy, n_frames), dtype=np.float32)

            for frame_idx in range(n_frames):
                for order in masking_orders:
                    # Get the bin index for this harmonic order from index_matrix
                    if order < index_matrix.shape[1]:
                        bin_idx = index_matrix[frame_idx, order]
                        if bin_idx > 0:  # Valid bin (not sentinel/Nyquist)
                            masking_mask_matrix[bin_idx, frame_idx] = 1.0
        else:
            masking_mask_matrix = None

        return (mask_matrix, masking_mask_matrix, fundamental_freqs, fundamental_bins)

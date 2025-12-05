"""
StepSignalHD - Phase 2 analyzer for step signals

Computes THD for step signals using pre-built masks from Phase 1B.
"""
import numpy as np
from typing import Dict, Tuple
from scipy import signal as scipy_signal
from base.pre_processing.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer


class StepSignalHD(HarmonicDistortionAnalyzer):
    """THD analyzer for step signals."""

    def compute_distortion(
        self,
        recorded_signal: np.ndarray,
        stimulus_metadata: Dict,
        harmonic_orders: list,
        harmonic_mask: Tuple[np.ndarray, np.ndarray, np.ndarray],
        stft_window_type: str = 'hann',
        **kwargs
    ) -> Dict:
        """
        Compute THD for step signals using pre-built mask and STFT.

        Args:
            recorded_signal: Recorded audio
            stimulus_metadata: Config with num_steps, repeat_times, total_time
            harmonic_orders: Selected harmonics (for reference only)
            harmonic_mask: (mask_matrix, fundamental_freqs, fundamental_bins) from Phase 1B
            stft_window_type: Window function for STFT (default 'hann')

        Returns:
            {
                'frequencies': fundamental_freqs,
                'thd': thd_values,
                'num_repetitions': repeat_times,
                'spectrum_matrix': averaged_spectrum (with dummy bin at index 0)
            }
        """
        mask_matrix, fundamental_freqs, fundamental_bins = harmonic_mask

        num_steps = stimulus_metadata['num_steps']
        repeat_times = stimulus_metadata['repeat_times']
        total_time = stimulus_metadata['total_time']

        # Split into repetitions
        repetitions = self._split_repetitions(recorded_signal, repeat_times)

        thd_per_rep = []
        spectrum_per_rep = []
        for repetition_signal in repetitions:
            # Calculate STFT parameters
            single_rep_duration = total_time / repeat_times
            step_duration = single_rep_duration / num_steps
            step_samples = int(step_duration * self.sample_rate)
            stft_window_size = step_samples
            stft_hop_size = step_samples  # No overlap

            # Compute STFT (results in exactly num_steps frames)
            spectrum_matrix = self._compute_stft(
                repetition_signal, stft_window_size, stft_hop_size, stft_window_type
            )

            # Add dummy bin
            spectrum_with_dummy = np.insert(spectrum_matrix, 0, 0.0, axis=0)

            # Validate and align frame counts (defensive programming)
            num_frames = min(spectrum_with_dummy.shape[1], mask_matrix.shape[1])
            spectrum_trimmed = spectrum_with_dummy[:, :num_frames]
            mask_trimmed = mask_matrix[:, :num_frames]
            fund_bins_trimmed = fundamental_bins[:num_frames]

            # Compute THD using pre-built mask
            thd = self.compute_thd_batch(spectrum_trimmed, mask_trimmed, fund_bins_trimmed)
            thd_per_rep.append(thd)
            spectrum_per_rep.append(spectrum_trimmed)

        # Average across repetitions
        averaged_thd = np.mean(thd_per_rep, axis=0)
        averaged_spectrum = np.mean(spectrum_per_rep, axis=0)

        return {
            'frequencies': fundamental_freqs,
            'thd': averaged_thd,
            'num_repetitions': repeat_times,
            'spectrum_matrix': averaged_spectrum
        }

    def _split_repetitions(self, signal: np.ndarray, repeat_times: int) -> list:
        """Split signal into repetitions."""
        if repeat_times == 1:
            return [signal]

        rep_length = len(signal) // repeat_times
        return [signal[i*rep_length:(i+1)*rep_length] for i in range(repeat_times)]

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
        from base.pre_processing.harmonic_index_builder import HarmonicIndexBuilder

        num_steps = stimulus_metadata['num_steps']
        total_time = stimulus_metadata['total_time']

        # Calculate STFT parameters
        step_duration = total_time / num_steps
        step_samples = int(step_duration * self.sample_rate)
        n_fft = step_samples

        # Build index matrix with all harmonics
        builder = HarmonicIndexBuilder()

        # Determine max harmonic order needed
        max_analysis_order = max(harmonic_orders) if harmonic_orders else 12
        max_masking_order = 9  # Default
        if masking_config and masking_config.get('enable_cumulative', False):
            masking_range = masking_config['masking_range']
            max_masking_order = masking_range[1]

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
            masking_range = masking_config['masking_range']
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

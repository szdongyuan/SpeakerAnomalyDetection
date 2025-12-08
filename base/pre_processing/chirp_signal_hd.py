# base/pre_processing/chirp_signal_hd.py
"""
ChirpSignalHD - Phase 2 analyzer for chirp signals

Computes THD for chirp signals using pre-built masks from Phase 1B.
"""
import numpy as np
from typing import Dict, Tuple
from scipy import signal as scipy_signal
from base.pre_processing.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer


class ChirpSignalHD(HarmonicDistortionAnalyzer):
    """THD analyzer for chirp signals."""

    def compute_distortion(
        self,
        recorded_signal: np.ndarray,
        stimulus_metadata: Dict,
        harmonic_orders: list,
        harmonic_mask: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        stft_window_size: int = 2048,
        stft_hop_size: int = 1024,
        stft_window_type: str = 'hann',
        **kwargs
    ) -> Dict:
        """
        Compute THD for chirp signals using pre-built mask.

        Args:
            recorded_signal: Recorded audio
            stimulus_metadata: Config with repeat_times, total_time
            harmonic_orders: Selected harmonics (for reference)
            harmonic_mask: (mask_matrix, masking_mask_matrix, fund_freqs, time_array, fund_bins) from Phase 1B
            stft_window_size: STFT window size
            stft_hop_size: STFT hop size
            stft_window_type: Window function type

        Returns:
            {
                'frequencies': fundamental_freqs,
                'thd': thd_values,
                'times': time_array,
                'num_repetitions': repeat_times
            }
        """
        mask_matrix, masking_mask_matrix, fundamental_freqs, time_array, fundamental_bins = harmonic_mask
        repeat_times = stimulus_metadata['repeat_times']

        # Split into repetitions
        repetitions = self._split_repetitions(recorded_signal, repeat_times)

        thd_per_rep = []
        for repetition_signal in repetitions:
            # Compute STFT
            stft_magnitude = self._compute_stft(
                repetition_signal, stft_window_size, stft_hop_size, stft_window_type
            )

            # Add dummy bin
            stft_with_dummy = np.insert(stft_magnitude, 0, 0.0, axis=0)

            # Align frame counts (handle boundary effects)
            num_frames = min(stft_with_dummy.shape[1], mask_matrix.shape[1])
            stft_trimmed = stft_with_dummy[:, :num_frames]
            mask_trimmed = mask_matrix[:, :num_frames]
            fund_bins_trimmed = fundamental_bins[:num_frames]

            # Compute THD using pre-built mask
            thd = self.compute_thd_batch(stft_trimmed, mask_trimmed, fund_bins_trimmed)
            thd_per_rep.append(thd)

        # Average across repetitions
        averaged_thd = np.mean(thd_per_rep, axis=0)

        # Trim time and frequency arrays to match
        num_frames = len(averaged_thd)

        return {
            'frequencies': fundamental_freqs[:num_frames],
            'thd': averaged_thd,
            'times': time_array[:num_frames],
            'num_repetitions': repeat_times
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
        """Compute STFT magnitude."""
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

    def _create_harmonic_mask(
        self,
        stimulus_metadata: Dict,
        harmonic_orders: list,
        stft_window_size: int,
        stft_hop_size: int,
        masking_config=None
    ):
        """
        Create binary mask matrices for harmonic extraction.

        Args:
            stimulus_metadata: Stimulus configuration
            harmonic_orders: List of harmonic orders to analyze (e.g., [10, 11, 12])
            stft_window_size: STFT window size
            stft_hop_size: STFT hop size
            masking_config: Optional masking configuration dict with keys:
                - 'masking_range': (start, end) harmonic orders for masking
                - 'enable_cumulative': bool to enable cumulative masking

        Returns:
            tuple: (mask_matrix, masking_mask_matrix, fundamental_freqs, time_array, fundamental_bins)
                - mask_matrix: Binary mask for selected harmonics
                - masking_mask_matrix: Binary mask for masking harmonics (or None)
                - fundamental_freqs: Fundamental frequencies
                - time_array: Time values for each frame
                - fundamental_bins: Fundamental bin indices
        """
        from base.pre_processing.harmonic_index_builder import HarmonicIndexBuilder

        builder = HarmonicIndexBuilder()

        # Determine max harmonic order needed
        max_analysis_order = max(harmonic_orders) if harmonic_orders else 12
        max_masking_order = 9  # Default
        if masking_config and masking_config.get('enable_cumulative', False):
            masking_range = masking_config['masking_range']
            max_masking_order = masking_range[1]

        max_harmonic_order = max(max_analysis_order, max_masking_order)

        # Build index matrix with all harmonics
        index_matrix, fundamental_freqs, time_array, fft_freqs_with_dummy = builder.build_chirp_signal_index_matrix(
            stimulus_metadata, self.sample_rate, stft_window_size, stft_hop_size, max_harmonic_order
        )

        n_bins_with_dummy = len(fft_freqs_with_dummy)
        n_frames = len(time_array)

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

        return (mask_matrix, masking_mask_matrix, fundamental_freqs, time_array, fundamental_bins)

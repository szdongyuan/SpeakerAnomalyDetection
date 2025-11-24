"""
HarmonicIndexBuilder - Phase 1A: Build Overall Index Matrix

Builds index matrices containing ALL harmonics (1-35) from stimulus configuration.
These matrices are reusable for any harmonic selection.
"""
import numpy as np
from typing import Dict, Tuple


class HarmonicIndexBuilder:
    """Builds harmonic index matrices for step and chirp signals."""

    def build_step_signal_index_matrix(
        self,
        stimulus_metadata: Dict,
        sr: int,
        n_fft: int,
        max_harmonic_order: int = 35
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build overall index matrix for step signals with ALL harmonics.

        Args:
            stimulus_metadata: Config dict with start_freq, stop_freq, num_steps, stimulus_type
            sr: Sample rate
            n_fft: FFT size (after trimming)
            max_harmonic_order: Maximum harmonic order to compute (default 35)

        Returns:
            - index_matrix: (num_steps, max_order+1) int32 array
              Column 0: sentinel (all zeros)
              Column 1: fundamental bins (+1 offset)
              Column N (N>=2): Nth harmonic bins (+1 offset)
            - fundamental_freqs: (num_steps,) fundamental frequencies
            - fft_freqs: FFT frequency bins with dummy bin prepended
        """
        num_steps = stimulus_metadata['num_steps']
        start_freq = stimulus_metadata['start_freq']
        stop_freq = stimulus_metadata['stop_freq']
        stimulus_type = stimulus_metadata['stimulus_type']

        # Generate fundamental frequencies
        if stimulus_type == 'linear':
            fundamental_freqs = np.linspace(start_freq, stop_freq, num_steps)
        elif stimulus_type == 'log':
            fundamental_freqs = np.logspace(
                np.log10(start_freq), np.log10(stop_freq), num_steps
            )
        else:
            raise ValueError(f"Unsupported stimulus_type: {stimulus_type}")

        # Create FFT frequency bins
        fft_freqs = np.fft.rfftfreq(n_fft, d=1.0/sr)
        n_bins = len(fft_freqs)
        nyquist = sr / 2.0

        # Initialize index matrix: (num_steps, max_order+1)
        # Column 0: sentinel (stays zero)
        # Column 1: fundamental bins
        # Column N (N>=2): Nth harmonic bins
        index_matrix = np.zeros((num_steps, max_harmonic_order + 1), dtype=np.int32)

        # Build index for each step
        for step_idx, fund_freq in enumerate(fundamental_freqs):
            # Find fundamental bin and store in Column 1 (+1 for dummy row offset)
            fund_bin = np.argmin(np.abs(fft_freqs - fund_freq))
            index_matrix[step_idx, 1] = fund_bin + 1

            # Find harmonic bins (2nd to max_harmonic_order)
            for harmonic_order in range(2, max_harmonic_order + 1):
                harmonic_freq = fund_freq * harmonic_order  # Direct formula

                if harmonic_freq < nyquist:
                    harm_bin = np.argmin(np.abs(fft_freqs - harmonic_freq))
                    index_matrix[step_idx, harmonic_order] = harm_bin + 1
                else:
                    # Exceeds Nyquist - stays 0 (sentinel/dummy bin)
                    index_matrix[step_idx, harmonic_order] = 0

        # Prepend dummy bin to fft_freqs for consistency
        fft_freqs_with_dummy = np.insert(fft_freqs, 0, 0.0)

        return index_matrix, fundamental_freqs, fft_freqs_with_dummy

    def build_chirp_signal_index_matrix(
        self,
        stimulus_metadata: Dict,
        sr: int,
        n_fft: int,
        hop_length: int,
        max_harmonic_order: int = 35
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build overall index matrix for chirp signals with ALL harmonics.

        Args:
            stimulus_metadata: Config dict with start_freq, stop_freq, total_time, repeat_times, stimulus_type
            sr: Sample rate
            n_fft: STFT window size
            hop_length: STFT hop size
            max_harmonic_order: Maximum harmonic order (default 35)

        Returns:
            - index_matrix: (num_frames, max_order+1) int32 array
            - fundamental_freqs: (num_frames,) instantaneous frequencies
            - time_array: (num_frames,) frame center times
            - fft_freqs: FFT frequency bins with dummy bin prepended
        """
        start_freq = stimulus_metadata['start_freq']
        stop_freq = stimulus_metadata['stop_freq']
        total_time = stimulus_metadata['total_time']
        repeat_times = stimulus_metadata['repeat_times']
        stimulus_type = stimulus_metadata['stimulus_type']

        # Calculate single repetition duration and number of frames
        single_rep_duration = total_time / repeat_times
        num_samples = int(single_rep_duration * sr)
        num_frames = 1 + (num_samples - n_fft) // hop_length

        # Create time array (frame center times)
        time_array = (np.arange(num_frames) * hop_length + n_fft / 2) / sr

        # Compute instantaneous frequency at each frame
        if stimulus_type == 'linear':
            fund_freqs = start_freq + (stop_freq - start_freq) * time_array / single_rep_duration
        elif stimulus_type == 'log':
            fund_freqs = start_freq * np.exp(
                np.log(stop_freq / start_freq) * time_array / single_rep_duration
            )
        elif stimulus_type == 'mirror_linear':
            half_duration = single_rep_duration / 2
            fund_freqs = np.where(
                time_array < half_duration,
                stop_freq - (stop_freq - start_freq) * time_array / half_duration,
                start_freq + (stop_freq - start_freq) * (time_array - half_duration) / half_duration
            )
        elif stimulus_type == 'mirror_log':
            half_duration = single_rep_duration / 2
            fund_freqs = np.where(
                time_array < half_duration,
                stop_freq * np.exp(-np.log(stop_freq / start_freq) * time_array / half_duration),
                start_freq * np.exp(np.log(stop_freq / start_freq) * (time_array - half_duration) / half_duration)
            )
        else:
            raise ValueError(f"Unsupported stimulus_type: {stimulus_type}")

        # Create FFT frequency bins
        fft_freqs = np.fft.rfftfreq(n_fft, d=1.0/sr)
        nyquist = sr / 2.0

        # Initialize index matrix
        index_matrix = np.zeros((num_frames, max_harmonic_order + 1), dtype=np.int32)

        # Build index for each frame
        for frame_idx, fund_freq in enumerate(fund_freqs):
            # Find fundamental bin (+1 offset)
            fund_bin = np.argmin(np.abs(fft_freqs - fund_freq))
            index_matrix[frame_idx, 1] = fund_bin + 1

            # Find harmonic bins
            for harmonic_order in range(2, max_harmonic_order + 1):
                harmonic_freq = fund_freq * harmonic_order

                if harmonic_freq < nyquist:
                    harm_bin = np.argmin(np.abs(fft_freqs - harmonic_freq))
                    index_matrix[frame_idx, harmonic_order] = harm_bin + 1
                else:
                    index_matrix[frame_idx, harmonic_order] = 0

        # Prepend dummy bin
        fft_freqs_with_dummy = np.insert(fft_freqs, 0, 0.0)

        return index_matrix, fund_freqs, time_array, fft_freqs_with_dummy

    def create_mask_from_indices(
        self,
        index_matrix: np.ndarray,
        harmonic_orders: list,
        n_bins_with_dummy: int
    ) -> np.ndarray:
        """
        Convert selected harmonic indices to binary mask matrix.

        Phase 1B: Extract user-selected harmonics from overall index matrix.

        Args:
            index_matrix: (num_steps_or_frames, max_order+1) from Phase 1A
            harmonic_orders: List of selected harmonics, e.g., [2, 3, 4, 5]
            n_bins_with_dummy: Total FFT bins including dummy row 0

        Returns:
            mask_matrix: (n_bins_with_dummy, num_steps_or_frames) binary mask
                         Row 0: dummy bin (zeros)
                         Rows 1+: 1.0 for selected harmonics, 0.0 elsewhere
        """
        num_steps_or_frames = index_matrix.shape[0]

        # Initialize mask (all zeros)
        mask_matrix = np.zeros((n_bins_with_dummy, num_steps_or_frames), dtype=np.float32)

        # Extract columns: fundamental (column 1) + selected harmonics
        columns_to_extract = [1] + harmonic_orders
        selected_indices = index_matrix[:, columns_to_extract]

        # Set mask values using advanced indexing
        for harmonic_idx in range(selected_indices.shape[1]):
            bin_indices = selected_indices[:, harmonic_idx]
            frame_indices = np.arange(num_steps_or_frames)
            mask_matrix[bin_indices, frame_indices] = 1.0

        return mask_matrix

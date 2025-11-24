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

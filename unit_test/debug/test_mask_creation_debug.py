"""
Deeper diagnostic: Check if mask_matrix is correctly created.
"""
import numpy as np
import unittest
from base.pre_processing.harmonic_index_builder import HarmonicIndexBuilder


class TestMaskCreation(unittest.TestCase):
    """Test if mask_matrix is being created correctly for PRB."""

    def test_mask_creation_for_step_signal(self):
        """Test that create_mask_from_indices produces non-zero mask."""

        sample_rate = 48000
        duration = 0.1

        stimulus_metadata = {
            'stimulus_method': 'steps',
            'stimulus_type': 'linear',
            'start_freq': 1000.0,
            'stop_freq': 1000.0,
            'num_steps': 1,
            'total_time': duration,
            'repeat_times': 1,
            'sample_rate': sample_rate
        }

        # Build index matrix
        builder = HarmonicIndexBuilder()

        step_samples = int(duration * sample_rate)
        n_fft = step_samples

        index_matrix, fund_freqs, fft_freqs = builder.build_step_signal_index_matrix(
            stimulus_metadata, sr=sample_rate, n_fft=n_fft, max_harmonic_order=35
        )

        print(f"\n=== Index Matrix ===")
        print(f"index_matrix shape: {index_matrix.shape}")
        print(f"fund_freqs: {fund_freqs}")
        print(f"fft_freqs shape: {len(fft_freqs)}")
        print(f"index_matrix[0] (step 1): {index_matrix[0, :10]}")  # First 10 harmonics

        # Create mask for harmonics 10-15
        harmonic_orders = [10, 11, 12, 13, 14, 15]
        mask_matrix = builder.create_mask_from_indices(
            index_matrix, harmonic_orders, len(fft_freqs)
        )

        print(f"\n=== Mask Matrix ===")
        print(f"mask_matrix shape: {mask_matrix.shape}")
        print(f"mask_matrix sum: {np.sum(mask_matrix)}")
        print(f"mask_matrix max: {np.max(mask_matrix)}")
        print(f"Non-zero bins in mask: {np.count_nonzero(mask_matrix)}")

        # Check which bins are marked in the mask
        mask_col = mask_matrix[:, 0]  # First frame
        nonzero_bins = np.where(mask_col > 0)[0]
        print(f"Non-zero bins: {nonzero_bins}")

        # Check if harmonic bins are in index_matrix
        for h in harmonic_orders:
            h_idx = h  # harmonic column index
            if h_idx < index_matrix.shape[1]:
                bin_idx = index_matrix[0, h_idx]
                print(f"Harmonic {h}: bin index = {bin_idx}, freq = {fft_freqs[int(bin_idx)]:.2f} Hz, in mask: {mask_col[int(bin_idx)] > 0}")

        # The mask should not be all zeros
        self.assertGreater(np.sum(mask_matrix), 0,
                          "Mask should have non-zero values for selected harmonics")


if __name__ == '__main__':
    unittest.main()

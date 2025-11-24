"""
HarmonicDistortionAnalyzer - Base class for Phase 2: THD Calculation

Computes THD using pre-built masks from Phase 1B.
"""
import numpy as np
from typing import Dict
from abc import ABC, abstractmethod


class HarmonicDistortionAnalyzer(ABC):
    """Base analyzer for THD calculation with pre-built masks."""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    @abstractmethod
    def compute_distortion(
        self,
        recorded_signal: np.ndarray,
        stimulus_metadata: Dict,
        harmonic_orders: list,
        harmonic_mask: tuple,
        **kwargs
    ) -> Dict:
        """
        Compute THD using pre-built mask. Must be implemented by subclasses.

        Args:
            recorded_signal: Recorded audio
            stimulus_metadata: Config dict
            harmonic_orders: Selected harmonics
            harmonic_mask: Pre-built mask data from Phase 1B
            **kwargs: Additional parameters

        Returns:
            Result dict with 'frequencies', 'thd', etc.
        """
        pass

    def compute_thd_batch(
        self,
        spectrum_matrix: np.ndarray,
        mask_matrix: np.ndarray,
        fundamental_bins: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized THD computation using pre-built mask.

        Formula: THD = sqrt(sum(H_i²)) / sqrt(F² + sum(H_i²)) × 100%

        Args:
            spectrum_matrix: (n_bins+1, n_steps_or_frames) magnitude spectrum with dummy bin
            mask_matrix: (n_bins+1, n_steps_or_frames) binary mask for selected harmonics
            fundamental_bins: (n_steps_or_frames,) indices of fundamental in spectrum

        Returns:
            thd_percentage: (n_steps_or_frames,) THD values in percent
        """
        n_cols = spectrum_matrix.shape[1]

        # Extract fundamental amplitudes (vectorized)
        row_indices = fundamental_bins.astype(int)
        col_indices = np.arange(n_cols)
        fundamental_amplitudes = spectrum_matrix[row_indices, col_indices]

        # Create harmonic-only mask (exclude fundamental)
        harmonic_mask = mask_matrix.copy()
        harmonic_mask[row_indices, col_indices] = 0.0

        # Compute harmonic power (vectorized)
        harmonic_amplitudes_squared = (spectrum_matrix ** 2) * harmonic_mask
        harmonic_power = np.sum(harmonic_amplitudes_squared, axis=0)

        # Compute THD (vectorized)
        fundamental_power = fundamental_amplitudes ** 2
        total_power = fundamental_power + harmonic_power
        safe_total_power = np.maximum(total_power, 1e-10)  # Avoid division by zero

        thd_ratio = np.sqrt(harmonic_power / safe_total_power)
        thd_percentage = thd_ratio * 100.0

        return thd_percentage

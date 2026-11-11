"""Minimal mono-Pa roughness and existing-loudness-to-sharpness example.

Run from the repository root: python -m scripts.sound_quality_example
"""

import numpy as np

from base.core_algorithm.sound_quality.loudness import LoudnessAnalyzer, LoudnessMethod
from base.core_algorithm.sound_quality.roughness import compute_roughness_daniel_weber
from base.core_algorithm.sound_quality.sharpness import compute_sharpness_din45692


def main():
    sample_rate = 48_000
    time_s = np.arange(int(0.6 * sample_rate), dtype=np.float64) / sample_rate
    # Synthetic mono pressure in Pa. Real recordings must first be calibrated.
    signal_pa = (1.0 + 0.5 * np.sin(2 * np.pi * 70 * time_s)) * np.sin(
        2 * np.pi * 1000 * time_s
    )
    roughness = compute_roughness_daniel_weber(signal_pa, sample_rate)

    method = LoudnessMethod.TIME_VARYING_ISO532_1
    loudness = LoudnessAnalyzer(sample_rate).compute(
        signal_pa, method=method, v2pa_factor=1.0,  # Already Pa; do not calibrate twice.
    )
    sharpness = compute_sharpness_din45692(
        loudness.specific_loudness, loudness.bark_axis, loudness.time_s,
        method=method,
    )
    print(f"R_mean = {roughness.summary.r_mean_asper:.12f} asper")
    print(f"R5 = {roughness.summary.r5_asper:.12f} asper")
    print(f"S_mean = {sharpness.summary.s_mean:.12f} acum")
    print(f"S_stationary = {sharpness.summary.s_stationary:.12f} acum")


if __name__ == "__main__":
    main()

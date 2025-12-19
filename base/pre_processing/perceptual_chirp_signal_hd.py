"""
Compatibility shim.

Harmonic distortion / PRB analysis algorithms have moved to
`base.core_algorithm.harmonic_distortion`.
"""

from base.core_algorithm.harmonic_distortion.perceptual_chirp_signal_hd import PerceptualChirpSignalHD

__all__ = ["PerceptualChirpSignalHD"]


"""
Compatibility shim.

Harmonic distortion / PRB analysis algorithms have moved to
`base.core_algorithm.harmonic_distortion`.
"""

from base.core_algorithm.harmonic_distortion.chirp_signal_hd import ChirpSignalHD

__all__ = ["ChirpSignalHD"]


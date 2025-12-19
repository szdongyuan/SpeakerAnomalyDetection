"""
Compatibility shim.

Harmonic distortion / PRB analysis algorithms have moved to
`base.core_algorithm.harmonic_distortion`.
"""

from base.core_algorithm.harmonic_distortion.perceptual_step_signal_hd import PerceptualStepSignalHD

__all__ = ["PerceptualStepSignalHD"]


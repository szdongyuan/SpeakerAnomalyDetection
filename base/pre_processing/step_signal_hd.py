"""
Compatibility shim.

Harmonic distortion / PRB analysis algorithms have moved to
`base.core_algorithm.harmonic_distortion`.
"""

from base.core_algorithm.harmonic_distortion.step_signal_hd import StepSignalHD

__all__ = ["StepSignalHD"]


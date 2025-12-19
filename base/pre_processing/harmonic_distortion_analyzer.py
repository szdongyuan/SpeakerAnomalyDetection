"""
Compatibility shim.

Harmonic distortion / PRB analysis algorithms have moved to
`base.core_algorithm.harmonic_distortion`.
"""

from base.core_algorithm.harmonic_distortion.harmonic_distortion_analyzer import HarmonicDistortionAnalyzer

__all__ = ["HarmonicDistortionAnalyzer"]


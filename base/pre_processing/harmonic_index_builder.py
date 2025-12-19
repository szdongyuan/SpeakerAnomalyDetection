"""
Compatibility shim.

Harmonic distortion / PRB analysis algorithms have moved to
`base.core_algorithm.harmonic_distortion`.
"""

from base.core_algorithm.harmonic_distortion.harmonic_index_builder import HarmonicIndexBuilder

__all__ = ["HarmonicIndexBuilder"]


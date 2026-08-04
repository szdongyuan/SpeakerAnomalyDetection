"""Project loudness API backed by the project-local implementation.

The previous MoSQITo-backed implementation is kept under
``base.core_algorithm.sound_quality.mosqito_reference`` for reference and
regression comparison. Product-facing LOUD/SHRP now imports this module, so it
uses the project engineering implementation by default.
"""

from .engineering_loudness import (
    ONE_THIRD_CENTERS_HZ,
    P_REF,
    TARGET_FS_HZ,
    LoudnessAnalyzer,
    LoudnessMethod,
    LoudnessResult,
    LoudnessSummary,
    sones_to_phons,
)

MIN_ISO532_SAMPLE_RATE_HZ = TARGET_FS_HZ

__all__ = [
    "P_REF",
    "TARGET_FS_HZ",
    "MIN_ISO532_SAMPLE_RATE_HZ",
    "ONE_THIRD_CENTERS_HZ",
    "LoudnessAnalyzer",
    "LoudnessMethod",
    "LoudnessResult",
    "LoudnessSummary",
    "sones_to_phons",
]

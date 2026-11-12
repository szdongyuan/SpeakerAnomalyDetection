"""DIN 45692 sharpness reference constants."""

import numpy as np

from .sound_quality_common_consts import BARK_AXIS_24_BARK_0_1_STEP


def _readonly_array(values, *, dtype=np.float64) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    arr.setflags(write=False)
    return arr


# Sharpness: DIN 45692 defaults and g(z) weighting --------------------------
DEFAULT_K_DIN45692 = 0.11  # Calibration constant k; yields S = 1 acum for 1 kHz narrowband noise at 60 dB
DEFAULT_N_FLOOR_SONE = 0.1  # Minimum total loudness (sone) to compute sharpness; avoids division by zero
DEFAULT_TIME_SKIP_S = 0.1  # Skip initial transient when computing sharpness statistics (s)
SHARPNESS_DZ_BARK = 0.1  # Bark integration step (matches BARK_AXIS_24_BARK_0_1_STEP)
SHARPNESS_WEIGHTING_BREAK_BARK = 15.8  # g(z) breakpoint: g = 1 for z ≤ 15.8 Bark
SHARPNESS_WEIGHTING_EXP_COEFFICIENT = 0.42  # g(z) exponential coefficient: 0.42 * (z − 15.8)
SHARPNESS_WEIGHTING_EXP_SCALE = 0.15  # g(z) exponential term scale factor
SHARPNESS_WEIGHTING_OFFSET = 0.85  # g(z) offset for high-frequency region: g(z) = 0.15*exp(…) + 0.85
# Pre-computed DIN 45692 g(z) weighting curve (240 points); 1.0 below 15.8 Bark, exponential above
SHARPNESS_DIN45692_G_WEIGHT = _readonly_array(
    np.where(
        BARK_AXIS_24_BARK_0_1_STEP <= SHARPNESS_WEIGHTING_BREAK_BARK,
        1.0,
        SHARPNESS_WEIGHTING_EXP_SCALE
        * np.exp(SHARPNESS_WEIGHTING_EXP_COEFFICIENT * (BARK_AXIS_24_BARK_0_1_STEP - SHARPNESS_WEIGHTING_BREAK_BARK))
        + SHARPNESS_WEIGHTING_OFFSET,
    )
)

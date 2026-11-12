"""Reference grid shared by loudness and sharpness."""

import numpy as np


# Bark scale axis (0.1–24.0 Bark, step 0.1, 240 points) for specific loudness & sharpness
BARK_AXIS_24_BARK_0_1_STEP = np.linspace(0.1, 24.0, 240, dtype=np.float64)
BARK_AXIS_24_BARK_0_1_STEP.setflags(write=False)

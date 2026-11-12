"""Daniel-Weber roughness reference constants and lookup tables."""

import numpy as np


def _readonly_array(values, *, dtype=np.float64) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    arr.setflags(write=False)
    return arr


# Roughness: Daniel-Weber / Aures constants and weighting tables -------------
DEFAULT_ROUGHNESS_OVERLAP = 0.5  # Default frame overlap ratio (0.5 → 50%)
ROUGHNESS_FRAME_DURATION_S = 0.2  # Analysis frame length 200 ms
ROUGHNESS_MODEL_NAME = "Daniel-Weber roughness (MoSQITo roughness_dw compatible)"
MOSQITO_ROUGHNESS_MODEL_NAME = "MoSQITo roughness_dw (Daniel-Weber)"
ROUGHNESS_BARK_BAND_COUNT = 47  # Number of half-Bark critical-band channels (0.5–23.5 Bark)
ROUGHNESS_BARK_AXIS = _readonly_array(np.arange(1, ROUGHNESS_BARK_BAND_COUNT + 1, dtype=np.float64) / 2.0)
ROUGHNESS_FFT_AMPLITUDE_CORRECTION = 1.42  # Amplitude correction after Blackman-window FFT
ROUGHNESS_CALIBRATION_FACTOR = 0.25  # Global calibration factor to convert sum(r_spec) → asper
ROUGHNESS_H_WEIGHTING_START_BIN = 2  # First FFT bin used for H-weighting interpolation
ROUGHNESS_LOWER_SLOPE_DB_PER_BARK = -27.0  # Fixed lower masking slope (dB/Bark)
ROUGHNESS_UPPER_SLOPE_BASE_DB_PER_BARK = -24.0  # Base upper masking slope (dB/Bark)
ROUGHNESS_UPPER_SLOPE_FREQ_FACTOR = 230.0  # Frequency-dependent term in upper slope: −230/f
ROUGHNESS_UPPER_SLOPE_LEVEL_FACTOR = 0.2  # Level-dependent term in upper slope: +0.2*L_dB
ROUGHNESS_HALF_BARK_STEP = 0.5  # Channel spacing in Bark (half-Bark resolution)

# H-weighting bandpass transfer functions (modulation-rate weighting).
# Each pair (_X = freq Hz, _Y = gain) defines one prototype curve;
# neighbouring bands are filled by copying the nearest prototype.
ROUGHNESS_H2_X = _readonly_array([0, 17, 23, 25, 32, 37, 48, 67, 90, 114, 171, 206, 247, 294, 358])
ROUGHNESS_H2_Y = _readonly_array([0, 0.8, 0.95, 0.975, 1, 0.975, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0])
ROUGHNESS_H5_X = _readonly_array([0, 32, 43, 56, 69, 92, 120, 142, 165, 231, 277, 331, 397, 502])
ROUGHNESS_H5_Y = _readonly_array([0, 0.8, 0.95, 1, 0.975, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0])
ROUGHNESS_H16_X = _readonly_array([0, 23.5, 34, 47, 56, 63, 79, 100, 115, 135, 159, 172, 194, 215, 244, 290, 348, 415, 500, 645])
ROUGHNESS_H16_Y = _readonly_array([0, 0.4, 0.6, 0.8, 0.9, 0.95, 1, 0.975, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0])
ROUGHNESS_H21_X = _readonly_array([0, 19, 44, 52.5, 58, 75, 101.5, 114.5, 132.5, 143.5, 165.5, 197.5, 241, 290, 348, 415, 500, 645])
ROUGHNESS_H21_Y = _readonly_array([0, 0.4, 0.8, 0.9, 0.95, 1, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0])
ROUGHNESS_H42_X = _readonly_array([0, 15, 41, 49, 53, 64, 71, 88, 94, 106, 115, 137, 180, 238, 290, 348, 415, 500, 645])
ROUGHNESS_H42_Y = _readonly_array([0, 0.4, 0.8, 0.9, 0.965, 0.99, 1, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0])
# (band_index, max_freq_Hz, x_points, y_points) – 5 prototype curves assigned to 47 bands
ROUGHNESS_H_WEIGHTING_CURVES = (
    (1, 358.0, ROUGHNESS_H2_X, ROUGHNESS_H2_Y),
    (4, 502.0, ROUGHNESS_H5_X, ROUGHNESS_H5_Y),
    (15, 502.0, ROUGHNESS_H16_X, ROUGHNESS_H16_Y),
    (20, 502.0, ROUGHNESS_H21_X, ROUGHNESS_H21_Y),
    (41, 502.0, ROUGHNESS_H42_X, ROUGHNESS_H42_Y),
)

# g(zi) cross-band weighting: Bark breakpoints (0–24) and corresponding weights.
# Peaks near 4.5–5 Bark (≈ 500 Hz), reflects human roughness sensitivity.
ROUGHNESS_GZI_BARK_POINTS = _readonly_array(np.arange(0, 25, 1, dtype=np.float64))
ROUGHNESS_GZI_VALUES = _readonly_array([0.15,
 0.26,
 0.38,
 0.47,
 0.54,
 0.65,
 0.76,
 0.83,
 0.9,
 0.98,
 0.98,
 0.9,
 0.8,
 0.7,
 0.62,
 0.54,
 0.49,
 0.43,
 0.39,
 0.35,
 0.3,
 0.3,
 0.3,
 0.3,
 0.3])
# Outer/middle ear transfer function (dB vs Bark); models ear canal resonance & attenuation
ROUGHNESS_EAR_FILTER_BARK_POINTS = _readonly_array([0, 10, 12, 13, 14, 15, 16, 16.5, 17, 18, 18.5, 19, 20, 21, 21.5, 22, 22.5, 23, 23.5, 24, 25, 26])
ROUGHNESS_EAR_FILTER_DB = _readonly_array([0,
 0,
 1.15,
 2.31,
 3.85,
 5.62,
 6.92,
 7.38,
 6.92,
 4.23,
 2.31,
 0,
 -1.43,
 -2.59,
 -3.57,
 -5.19,
 -7.41,
 -11.3,
 -20,
 -40,
 -130,
 -999])
# Threshold in quiet for roughness (dB vs Bark); bins below this level are deemed inaudible
ROUGHNESS_LTQ_BARK_POINTS = _readonly_array([0,
 0.01,
 0.17,
 0.8,
 1,
 1.5,
 2,
 3.3,
 4,
 5,
 6,
 8,
 10,
 12,
 13.3,
 15,
 16,
 17,
 18,
 19,
 20,
 21,
 22,
 23,
 24,
 24.5,
 25])
ROUGHNESS_LTQ_DB = _readonly_array([130,
 70,
 60,
 30,
 25,
 20,
 15,
 10,
 8.1,
 6.3,
 5,
 3.5,
 2.5,
 1.7,
 0,
 -2.5,
 -4,
 -3.7,
 -1.5,
 1.4,
 3.8,
 5,
 7.5,
 15,
 48,
 60,
 130])
# Zwicker frequency-to-Bark conversion table (50 points, 0 Hz – 20 kHz, 0.5 Bark step)
ZWICKER_BARK_FREQUENCY_TABLE_HZ = _readonly_array([0,
 50,
 100,
 150,
 200,
 250,
 300,
 350,
 400,
 450,
 510,
 570,
 630,
 700,
 770,
 840,
 920,
 1000,
 1080,
 1170,
 1270,
 1370,
 1480,
 1600,
 1720,
 1850,
 2000,
 2150,
 2320,
 2500,
 2700,
 2900,
 3150,
 3400,
 3700,
 4000,
 4400,
 4800,
 5300,
 5800,
 6400,
 7000,
 7700,
 8500,
 9500,
 10500,
 12000,
 13500,
 15500,
 20000])
# Half-Bark axis (0–24.5 Bark, step 0.5, 50 points); y-axis for the frequency table above
ZWICKER_HALF_BARK_AXIS = _readonly_array(np.arange(0, 25, 0.5, dtype=np.float64))

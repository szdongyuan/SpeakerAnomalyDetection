"""Psychoacoustic reference constants and lookup tables.

These values are algorithm reference data, not tunable product settings. Keep
business defaults in UI/config files and keep standard/validated lookup tables
here so loudness code can be reviewed without large
anonymous arrays inside processing functions.
"""

import numpy as np


def _readonly_array(values, *, dtype=np.float64) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    arr.setflags(write=False)
    return arr


# Shared acoustic constants -------------------------------------------------
P_REF = 20.0e-6  # Sound pressure reference 20 µPa (hearing threshold)
TARGET_FS_HZ = 48_000  # Standard internal sample rate (Hz)
# ISO 1/3-octave band center frequencies (25 Hz – 12.5 kHz, 28 bands)
ONE_THIRD_CENTERS_HZ = _readonly_array([25.0,
 31.5,
 40.0,
 50.0,
 63.0,
 80.0,
 100.0,
 125.0,
 160.0,
 200.0,
 250.0,
 315.0,
 400.0,
 500.0,
 630.0,
 800.0,
 1000.0,
 1250.0,
 1600.0,
 2000.0,
 2500.0,
 3150.0,
 4000.0,
 5000.0,
 6300.0,
 8000.0,
 10000.0,
 12500.0])
# Bark scale axis (0.1–24.0 Bark, step 0.1, 240 points) for specific loudness & sharpness
BARK_AXIS_24_BARK_0_1_STEP = _readonly_array(np.linspace(0.1, 24.0, 240, dtype=np.float64))


# Loudness: ISO 532-1 / Zwicker time-varying 1/3-octave filters & core tables
# 6th-order IIR reference coefficients (3 cascaded stages, each [b0,b1,b2,a0,a1,a2])
LOUDNESS_TIME_VARYING_FILTER_REF = _readonly_array([[1, 2, 1, 1, -2, 1], [1, 0, -1, 1, -2, 1], [1, -2, 1, 1, -2, 1]])
# Per-band filter coefficient deltas (28×3×6); added to REF to get actual coefficients
LOUDNESS_TIME_VARYING_FILTER_DELTA = _readonly_array([[[0, 0, 0, 0, -0.00067026, 0.000659453],
  [0, 0, 0, 0, -0.000375071, 0.000361926],
  [0, 0, 0, 0, -0.000306523, 0.000297634]],
 [[0, 0, 0, 0, -0.000847258, 0.000830131],
  [0, 0, 0, 0, -0.000476448, 0.000455616],
  [0, 0, 0, 0, -0.000388773, 0.000374685]],
 [[0, 0, 0, 0, -0.0010721, 0.00104496],
  [0, 0, 0, 0, -0.000606567, 0.000573553],
  [0, 0, 0, 0, -0.000494004, 0.000471677]],
 [[0, 0, 0, 0, -0.00135836, 0.00131535],
  [0, 0, 0, 0, -0.000774327, 0.000722007],
  [0, 0, 0, 0, -0.000629154, 0.000593771]],
 [[0, 0, 0, 0, -0.0017238, 0.00165564],
  [0, 0, 0, 0, -0.00099178, 0.000908866],
  [0, 0, 0, 0, -0.000803529, 0.000747455]],
 [[0, 0, 0, 0, -0.00219188, 0.00208388],
  [0, 0, 0, 0, -0.00127545, 0.00114406],
  [0, 0, 0, 0, -0.00102976, 0.0009409]],
 [[0, 0, 0, 0, -0.00279386, 0.00262274],
  [0, 0, 0, 0, -0.00164828, 0.00144006],
  [0, 0, 0, 0, -0.0013252, 0.00118438]],
 [[0, 0, 0, 0, -0.00357182, 0.00330071],
  [0, 0, 0, 0, -0.00214252, 0.00181258],
  [0, 0, 0, 0, -0.00171397, 0.00149082]],
 [[0, 0, 0, 0, -0.00458305, 0.00415355],
  [0, 0, 0, 0, -0.00280413, 0.00228135],
  [0, 0, 0, 0, -0.00223006, 0.00187646]],
 [[0, 0, 0, 0, -0.00590655, 0.00522622],
  [0, 0, 0, 0, -0.00369947, 0.00287118],
  [0, 0, 0, 0, -0.00292205, 0.00236178]],
 [[0, 0, 0, 0, -0.00765243, 0.00657493],
  [0, 0, 0, 0, -0.0049254, 0.00361318],
  [0, 0, 0, 0, -0.00386007, 0.0029724]],
 [[0, 0, 0, 0, -0.0100023, 0.0082961],
  [0, 0, 0, 0, -0.00663788, 0.00455999],
  [0, 0, 0, 0, -0.00515982, 0.00375306]],
 [[0, 0, 0, 0, -0.013123, 0.010422],
  [0, 0, 0, 0, -0.00902274, 0.00573132],
  [0, 0, 0, 0, -0.00694543, 0.00471734]],
 [[0, 0, 0, 0, -0.0173693, 0.0130947],
  [0, 0, 0, 0, -0.0124176, 0.00720526],
  [0, 0, 0, 0, -0.00946002, 0.00593145]],
 [[0, 0, 0, 0, -0.0231934, 0.0164308],
  [0, 0, 0, 0, -0.0173009, 0.00904761],
  [0, 0, 0, 0, -0.0130358, 0.00744926]],
 [[0, 0, 0, 0, -0.0313292, 0.020637],
  [0, 0, 0, 0, -0.0244342, 0.0113731],
  [0, 0, 0, 0, -0.0182108, 0.00936778]],
 [[0, 0, 0, 0, -0.0428261, 0.0259325],
  [0, 0, 0, 0, -0.0349619, 0.0143046],
  [0, 0, 0, 0, -0.0257855, 0.0117912]],
 [[0, 0, 0, 0, -0.0591733, 0.0325054],
  [0, 0, 0, 0, -0.0506072, 0.0179513],
  [0, 0, 0, 0, -0.0369401, 0.0148094]],
 [[0, 0, 0, 0, -0.0826348, 0.0405894],
  [0, 0, 0, 0, -0.0740348, 0.0224476],
  [0, 0, 0, 0, -0.0534977, 0.0185371]],
 [[0, 0, 0, 0, -0.117018, 0.0508116],
  [0, 0, 0, 0, -0.109516, 0.0281387],
  [0, 0, 0, 0, -0.0785097, 0.0232872]],
 [[0, 0, 0, 0, -0.167714, 0.0637872],
  [0, 0, 0, 0, -0.163378, 0.0353729],
  [0, 0, 0, 0, -0.116419, 0.0293723]],
 [[0, 0, 0, 0, -0.242528, 0.0798576],
  [0, 0, 0, 0, -0.245161, 0.044337],
  [0, 0, 0, 0, -0.173972, 0.0370015]],
 [[0, 0, 0, 0, -0.353142, 0.099633],
  [0, 0, 0, 0, -0.369163, 0.0553535],
  [0, 0, 0, 0, -0.261399, 0.0465428]],
 [[0, 0, 0, 0, -0.516316, 0.124177],
  [0, 0, 0, 0, -0.555473, 0.0689403],
  [0, 0, 0, 0, -0.393998, 0.0586715]],
 [[0, 0, 0, 0, -0.756635, 0.155023],
  [0, 0, 0, 0, -0.834281, 0.0858123],
  [0, 0, 0, 0, -0.594547, 0.074396]],
 [[0, 0, 0, 0, -1.10165, 0.191713],
  [0, 0, 0, 0, -1.23939, 0.105243],
  [0, 0, 0, 0, -0.891666, 0.0940354]],
 [[0, 0, 0, 0, -1.58477, 0.239049],
  [0, 0, 0, 0, -1.80505, 0.128794],
  [0, 0, 0, 0, -1.325, 0.121333]],
 [[0, 0, 0, 0, -2.5063, 0.142308],
  [0, 0, 0, 0, -2.19464, 0.27647],
  [0, 0, 0, 0, -1.90231, 0.147304]]])
# Passband gain for each 1/3-octave band filter (28 values)
LOUDNESS_TIME_VARYING_FILTER_GAIN = _readonly_array([4.30764e-11,
 8.5934e-11,
 1.71424e-10,
 3.41944e-10,
 6.82035e-10,
 1.36026e-09,
 2.71261e-09,
 5.4087e-09,
 1.07826e-08,
 2.1491e-08,
 4.28228e-08,
 8.54316e-08,
 1.70009e-07,
 3.38215e-07,
 6.7199e-07,
 1.33531e-06,
 2.65172e-06,
 5.25477e-06,
 1.0378e-05,
 2.0487e-05,
 4.05198e-05,
 7.97914e-05,
 0.000156511,
 0.000304954,
 0.000599157,
 0.00116544,
 0.00227488,
 0.00391006])

# Reference SPL breakpoints (dB) for indexing into the DLL correction table
LOUDNESS_RAP_TABLE = _readonly_array([45, 55, 65, 71, 80, 90, 100, 120])
# Level correction lookup (8×11); interpolated by RAP level and band segment
LOUDNESS_DLL_TABLE = _readonly_array([[-32, -24, -16, -10, -5, 0, -7, -3, 0, -2, 0],
 [-29, -22, -15, -10, -4, 0, -7, -2, 0, -2, 0],
 [-27, -19, -14, -9, -4, 0, -6, -2, 0, -2, 0],
 [-25, -17, -12, -9, -3, 0, -5, -2, 0, -2, 0],
 [-23, -16, -11, -7, -3, 0, -4, -1, 0, -1, 0],
 [-20, -14, -10, -6, -3, 0, -4, -1, 0, -1, 0],
 [-18, -12, -9, -6, -2, 0, -3, -1, 0, -1, 0],
 [-15, -10, -8, -4, -2, 0, -3, -1, 0, -1, 0]])
# Threshold in quiet (dB) per critical band (20 bands)
LOUDNESS_LTQ_TABLE = _readonly_array([30, 18, 12, 8, 7, 6, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
# Free-field frequency response correction (dB) per band
LOUDNESS_A0_TABLE = _readonly_array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, -1.6, -3.2, -5.4, -5.6, -4, -1.5, 2, 5, 12])
# Diffuse-field correction (dB) for free→diffuse conversion
LOUDNESS_DDF_TABLE = _readonly_array([0, 0, 0.5, 0.9, 1.2, 1.6, 2.3, 2.8, 3, 2, 0, -1.4, -2, -1.9, -1, 0.5, 3, 4, 4.3, 4])
# Critical-band correction (dB); compensates 1/3-octave vs critical bandwidth mismatch
LOUDNESS_DCB_TABLE = _readonly_array([-0.25,
 -0.6,
 -0.8,
 -0.8,
 -0.5,
 0,
 0.5,
 1.1,
 1.5,
 1.7,
 1.8,
 1.8,
 1.7,
 1.6,
 1.4,
 1.2,
 0.8,
 0.5,
 0,
 -0.5])
# Upper Bark boundary of each critical band (21 values) in the Zwicker model
LOUDNESS_ZUP_TABLE = _readonly_array([0.9,
 1.8,
 2.8,
 3.5,
 4.4,
 5.4,
 6.6,
 7.9,
 9.2,
 10.6,
 12.3,
 13.8,
 15.2,
 16.7,
 18.1,
 19.3,
 20.6,
 21.8,
 22.7,
 23.6,
 24.0])
# Normalized range-of-slopes for upper-slope level interpolation (18 values)
LOUDNESS_RNS_TABLE = _readonly_array([21.5,
 18.0,
 15.1,
 11.5,
 9.0,
 6.1,
 4.4,
 3.1,
 2.13,
 1.36,
 0.82,
 0.42,
 0.3,
 0.22,
 0.15,
 0.1,
 0.035,
 0.0])
# Upper spreading slope table (18×8); masking-pattern high-frequency slope (dB/Bark)
LOUDNESS_USL_TABLE = _readonly_array([[13, 8.2, 6.3, 5.5, 5.5, 5.5, 5.5, 5.5],
 [9, 7.5, 6, 5.1, 4.5, 4.5, 4.5, 4.5],
 [7.8, 6.7, 5.6, 4.9, 4.4, 3.9, 3.9, 3.9],
 [6.2, 5.4, 4.6, 4.0, 3.5, 3.2, 3.2, 3.2],
 [4.5, 3.8, 3.6, 3.2, 2.9, 2.7, 2.7, 2.7],
 [3.7, 3.0, 2.8, 2.35, 2.2, 2.2, 2.2, 2.2],
 [2.9, 2.3, 2.1, 1.9, 1.8, 1.7, 1.7, 1.7],
 [2.4, 1.7, 1.5, 1.35, 1.3, 1.3, 1.3, 1.3],
 [1.95, 1.45, 1.3, 1.15, 1.1, 1.1, 1.1, 1.1],
 [1.5, 1.2, 0.94, 0.86, 0.82, 0.82, 0.82, 0.82],
 [0.72, 0.67, 0.64, 0.63, 0.62, 0.62, 0.62, 0.62],
 [0.59, 0.53, 0.51, 0.5, 0.42, 0.42, 0.42, 0.42],
 [0.4, 0.33, 0.26, 0.24, 0.24, 0.22, 0.22, 0.22],
 [0.27, 0.21, 0.2, 0.18, 0.17, 0.17, 0.17, 0.17],
 [0.16, 0.15, 0.14, 0.12, 0.11, 0.11, 0.11, 0.11],
 [0.12, 0.11, 0.1, 0.08, 0.08, 0.08, 0.08, 0.08],
 [0.09, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.05],
 [0.06, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02]])
# Extended upper-slope table (18×21); last column of USL_TABLE repeated to fill 21 bands
LOUDNESS_USL_FULL_TABLE = _readonly_array(
    np.append(LOUDNESS_USL_TABLE, np.tile(LOUDNESS_USL_TABLE[:, 7], 13).reshape(13, 18).T, axis=1)
)
LOUDNESS_SPECIFIC_BARK_BIN_COUNT = 240  # Specific-loudness Bark resolution (240 bins = 0.1 Bark step)
LOUDNESS_COMPARISON_DECIMALS = 8  # Decimal precision for loudness value comparisons
LOUDNESS_STATIONARY_WINDOW_EDGE_TRIM_RATIO = 0.25  # Discard 25% at each edge of the analysis window
LOUDNESS_CORE_S = 0.25  # Zwicker core-loudness constant s
LOUDNESS_CORE_MP1_SCALE = 0.0635  # Multiplicative factor in core-loudness formula (sone/Bark)
LOUDNESS_CORE_LTQ_EXPONENT_SCALE = 0.025  # Exponent scale for threshold-in-quiet term
LOUDNESS_CORE_LEVEL_EXPONENT_SCALE = 0.1  # Exponent scale for excitation-level term
LOUDNESS_CORE_POWER = 0.25  # Power-law exponent in core loudness (Stevens' power law)
LOUDNESS_LOW_BAND_CORRECTION_BASE = 0.4  # Low-frequency band correction base
LOUDNESS_LOW_BAND_CORRECTION_SCALE = 0.32  # Low-frequency band correction scale factor
LOUDNESS_LOW_BAND_CORRECTION_POWER = 0.2  # Low-frequency band correction power exponent
LOUDNESS_TEMPORAL_FAST_WEIGHT = 0.47  # Weight for the fast temporal integrator
LOUDNESS_TEMPORAL_SLOW_WEIGHT = 0.53  # Weight for the slow temporal integrator
LOUDNESS_TEMPORAL_FAST_TAU_S = 3.5e-3  # Fast time constant τ_fast = 3.5 ms (cochlear attack)
LOUDNESS_TEMPORAL_SLOW_TAU_S = 70e-3  # Slow time constant τ_slow = 70 ms (auditory integration)
LOUDNESS_INTERNAL_TIME_RESOLUTION_S = 0.0005  # Internal time-varying loudness step (s)
LOUDNESS_NONLINEAR_DECAY_INTERPOLATION_STEPS = 12  # Interpolation steps for nonlinear temporal decay
LOUDNESS_DEFAULT_OUTPUT_TIME_RESOLUTION_S = 0.002  # Default exported loudness curve step (s)
LOUDNESS_DEFAULT_OUTPUT_TIME_RESOLUTION_MS = LOUDNESS_DEFAULT_OUTPUT_TIME_RESOLUTION_S * 1000.0
LOUDNESS_DEFAULT_STATIONARY_FRAME_DURATION_S = 0.1  # Default stationary-loudness frame length (s)
LOUDNESS_DEFAULT_STATIONARY_OVERLAP_PERCENT = 50.0  # Default frame overlap (%)
LOUDNESS_DEFAULT_STATIONARY_HOP_DURATION_S = (  # Derived hop duration from frame length & overlap
    LOUDNESS_DEFAULT_STATIONARY_FRAME_DURATION_S
    * (1.0 - LOUDNESS_DEFAULT_STATIONARY_OVERLAP_PERCENT / 100.0)
)
LOUDNESS_DEFAULT_STATIONARY_NPERSEG = int(  # Frame length in samples at TARGET_FS_HZ
    round(LOUDNESS_DEFAULT_STATIONARY_FRAME_DURATION_S * TARGET_FS_HZ)
)
LOUDNESS_DEFAULT_STATIONARY_NHOP = int(  # Hop length in samples at TARGET_FS_HZ
    round(LOUDNESS_DEFAULT_STATIONARY_HOP_DURATION_S * TARGET_FS_HZ)
)


# Skip the initial transient when computing time-varying summary stats (avoids Nmax spikes)
LOUDNESS_TV_SUMMARY_SKIP_S = 0.1  # Skip first 100 ms

# SSTS Specific Loudness Reference Lines (from Seat SSTS Revision 40, Table 46 / Figure 6)
# Format: list of (Bark, sone/Bark) breakpoints — linearly interpolated between points.
# Units: Loudness in sone GF (free-field).
SSTS_SPECIFIC_LOUDNESS_REF_LINES = {
    "ref1": [(0, 0.33), (4, 0.33), (6, 0.18), (7, 0.18), (16, 0.18), (18, 0.1375), (24, 0.01)],
    "ref2": [(0, 0.50), (4, 0.50), (6, 0.22), (7, 0.22), (16, 0.22), (18, 0.1675), (24, 0.01)],
    "ref3": [(0, 0.40), (4, 0.40), (6, 0.267), (7, 0.20), (16, 0.20), (18, 0.20), (24, 0.01)],
    "ref4": [(0, 0.60), (4, 0.60), (6, 0.18), (7, 0.18), (16, 0.18), (18, 0.1375), (24, 0.01)],
}


def validate_reference_tables() -> None:
    """Fail fast when a reference table is accidentally edited to a bad shape."""
    expected_shapes = {
        "ONE_THIRD_CENTERS_HZ": (28,),
        "LOUDNESS_TIME_VARYING_FILTER_REF": (3, 6),
        "LOUDNESS_TIME_VARYING_FILTER_DELTA": (28, 3, 6),
        "LOUDNESS_TIME_VARYING_FILTER_GAIN": (28,),
        "LOUDNESS_RAP_TABLE": (8,),
        "LOUDNESS_DLL_TABLE": (8, 11),
        "LOUDNESS_LTQ_TABLE": (20,),
        "LOUDNESS_A0_TABLE": (20,),
        "LOUDNESS_DDF_TABLE": (20,),
        "LOUDNESS_DCB_TABLE": (20,),
        "LOUDNESS_ZUP_TABLE": (21,),
        "LOUDNESS_RNS_TABLE": (18,),
        "LOUDNESS_USL_TABLE": (18, 8),
        "LOUDNESS_USL_FULL_TABLE": (18, 21),
        "BARK_AXIS_24_BARK_0_1_STEP": (240,),
    }
    for name, shape in expected_shapes.items():
        actual = globals()[name].shape
        if actual != shape:
            raise RuntimeError(f"{name} has shape {actual}, expected {shape}")


validate_reference_tables()

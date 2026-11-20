"""Common constants for acoustic analysis algorithms."""

REFERENCE_PRESSURE_PA = 20e-6

GOLDEN_SAMPLE_CHECKED_KEY = "golden_sample_checked"
GOLDEN_SAMPLE_RESULT_PATH_KEY = "golden_sample_result_path"
GOLDEN_SAMPLE_DISPLAY_MODE_KEY = "golden_sample_display_mode"

GOLDEN_SAMPLE_DISPLAY_DEVIATION = "deviation"
GOLDEN_SAMPLE_DISPLAY_ENVELOPE = "envelope"
DEFAULT_GOLDEN_SAMPLE_DISPLAY_MODE = GOLDEN_SAMPLE_DISPLAY_DEVIATION

LIMIT_VALUE_SEMANTICS_BOUNDS = "bounds"
LIMIT_VALUE_SEMANTICS_OFFSET = "offset"

# Product SHRP contract; the general-purpose numerical cores retain both modes.
SHARPNESS_METHOD = "time_varying_iso532_1"
SHARPNESS_OUTPUT_TIME_RESOLUTION_MS = 2.0

# Presentation names shared by configuration, exports and result consumers.
SOUND_QUALITY_ANALYSES = {
    "SHRP": {
        "label": "尖锐度", "unit": "acum", "default_metric": "s_mean_acum",
        "metrics": (("s_max_acum", "最大尖锐度"), ("s_mean_acum", "平均尖锐度"),
                    ("s_stationary_acum", "稳态尖锐度")),
    },
    "ROUGH": {
        "label": "粗糙度", "unit": "asper", "default_metric": "r_mean_asper",
        "metrics": (("r_mean_asper", "平均粗糙度"), ("r_max_asper", "最大粗糙度")),
    },
}

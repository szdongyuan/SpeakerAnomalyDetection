"""Stable VE3668N input schema and acquisition constants."""

VE_BACKEND = "vkinging"
VE_MODEL = "VE3668N"
VE_MAX_INPUT_CHANNELS = 8
VE_INPUT_SCHEMA_VERSION = 1
VE_SAMPLE_RATES = (44100, 48000, 51200)  # Common suggestions, not a whitelist.
VE_SAMPLE_RATE_MIN = 8000
VE_SAMPLE_RATE_MAX = 102400
VE_DEFAULT_SAMPLE_RATE = 51200
VE_INPUT_MODE = "IEPE"
VE_UNIT = "V"
VE_RANGE_MIN = -10.0
VE_RANGE_MAX = 10.0
VE_RANGE_INDEX_CONFIG_KEY = "ve_range_index"
VE_RANGE_LIMITS = (10.0, 5.0, 2.5, 1.0, 0.5, 0.1, 0.02)
VE_RANGE_LABELS = ("±10 V", "±5 V", "±2.5 V", "±1 V", "±500 mV", "±100 mV", "±20 mV")
VE_INPUT_CONFIG_FIELDS = (
    "sample_rate", "input_mode", "unit", "range_min", "range_max",
)
VE_DEVICE_SNAPSHOT_FIELDS = (
    "backend", "model", "machine_id", "name", "address", "physical_channels",
    "max_input_channels", "available", "input_config",
)

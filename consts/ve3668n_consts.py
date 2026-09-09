"""Stable VE3668N input schema and acquisition constants."""

VE_BACKEND = "vkinging"
VE_MODEL = "VE3668N"
VE_MAX_INPUT_CHANNELS = 8
VE_INPUT_SCHEMA_VERSION = 1
VE_SAMPLE_RATES = (44100, 48000, 51200)
VE_DEFAULT_SAMPLE_RATE = 51200
VE_INPUT_MODE = "IEPE"
VE_UNIT = "V"
VE_RANGE_MIN = -10.0
VE_RANGE_MAX = 10.0
VE_INPUT_CONFIG_FIELDS = (
    "sample_rate", "input_mode", "unit", "range_min", "range_max",
)
VE_DEVICE_SNAPSHOT_FIELDS = (
    "backend", "model", "machine_id", "name", "address", "physical_channels",
    "max_input_channels", "available", "input_config",
)

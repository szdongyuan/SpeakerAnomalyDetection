"""Stable vkinging input protocol and persisted queue values."""

VKINGING_HOSTAPI_NAME = "vkinging"
VKINGING_INPUT_MODE = "IEPE"
VKINGING_UNIT = "V"
VKINGING_RANGE_INDEX_CONFIG_KEY = "vkinging_range_index"
VKINGING_RANGE_LIMITS = (10.0, 5.0, 2.5, 1.0, 0.5, 0.1, 0.02)
VKINGING_SAMPLE_RATE_MIN = 4000
VKINGING_SAMPLE_RATE_MAX = 102400

# Stable RIFF provenance protocol, independent of the calibration JSON.
VKINGING_WAV_CHUNK_ID = b"vkin"
VKINGING_WAV_MARKER = b"vkinging;unit=V;schema=1\x00"
VKINGING_WAV_SCHEMA_VERSION = 1

VALID_SAMPLE_RATES = (44100, 48000)
VALID_BIT_DEPTHS = (32, 64)
DEFAULT_FLOAT_BIT_DEPTH = 32
FLOAT_BIT_DEPTH_TO_DTYPE = {
    32: "float32",
    64: "float64",
}
DEVICE_ENUMERATION_BASE_EXCEPTIONS = (ValueError, RuntimeError)
RIFF_HEADER_SIZE = 12
CHUNK_HEADER_SIZE = 8
MAX_RIFF_SIZE = 0xFFFFFFFF


def normalize_float_bit_depth(bit_depth=DEFAULT_FLOAT_BIT_DEPTH):
    if isinstance(bit_depth, bool):
        return DEFAULT_FLOAT_BIT_DEPTH
    try:
        normalized = int(bit_depth)
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_FLOAT_BIT_DEPTH
    if normalized not in FLOAT_BIT_DEPTH_TO_DTYPE:
        return DEFAULT_FLOAT_BIT_DEPTH
    return normalized


def bit_depth_to_dtype(bit_depth=DEFAULT_FLOAT_BIT_DEPTH):
    return FLOAT_BIT_DEPTH_TO_DTYPE[normalize_float_bit_depth(bit_depth)]

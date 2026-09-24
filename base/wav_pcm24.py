"""Deterministic conversion to the PCM24 sample grid."""

import numpy as np

from consts.wav_format_consts import PCM24_MAX_CODE, PCM24_MIN_CODE, PCM24_SCALE


def quantize_pcm24(samples):
    """Return clipped PCM24 samples as float32 without modifying the input."""
    values = np.asarray(samples, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("PCM24 samples must be finite float32 values")
    codes = np.floor(np.clip(values, -1.0, 1.0) * PCM24_SCALE)
    # Integer PCM has one zero code; canonicalize -0.0 for byte-identical readback.
    codes = np.clip(codes, PCM24_MIN_CODE, PCM24_MAX_CODE).astype(np.int32)
    return (codes / PCM24_SCALE).astype(np.float32, copy=False)


def pack_pcm24_le(quantized):
    """Encode output of quantize_pcm24 in frame/channel little-endian order."""
    codes = (np.asarray(quantized, dtype=np.float32) * PCM24_SCALE).astype(np.int32)
    codes = codes.ravel(order="C")
    packed = np.empty((codes.size, 3), dtype=np.uint8)
    packed[:, 0] = codes & 0xFF
    packed[:, 1] = (codes >> 8) & 0xFF
    packed[:, 2] = (codes >> 16) & 0xFF
    return packed.tobytes()

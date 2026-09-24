import hashlib
import io
import math
import wave

import numpy as np
import pytest
import soundfile as sf

from base.wav_pcm24 import pack_pcm24_le, quantize_pcm24


def test_clips_only_overrange_samples():
    source = np.array([0.5, 1, 2, -3, -1], dtype=np.float32)
    original = source.copy()
    actual = quantize_pcm24(source)
    expected = np.array([0.5, 8388607 / 8388608,
                         8388607 / 8388608, -1, -1], dtype=np.float32)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, original)
    assert actual.dtype == np.float32


def test_floor_quantization_at_positive_and_negative_boundaries():
    lsb = 1 / 8388608
    source = np.array([
        0, 0.5, -0.5, 0.25 * lsb, -0.25 * lsb,
        0.75 * lsb, -0.75 * lsb, 1.25 * lsb, -1.25 * lsb,
        np.nextafter(np.float32(lsb), np.float32(0)),
        np.nextafter(np.float32(-lsb), np.float32(-1)),
        np.nextafter(np.float32(1), np.float32(0)),
        np.nextafter(np.float32(-1), np.float32(0)),
        -2.5, 2.5, 8.25,
    ], dtype=np.float32)
    expected_codes = np.array([
        0, 4194304, -4194304, 0, -1, 0, -1, 1, -2,
        0, -2, 8388607, -8388608, -8388608, 8388607, 8388607,
    ], dtype=np.int32)
    np.testing.assert_array_equal(quantize_pcm24(source), expected_codes / 8388608)


def test_noncontiguous_multichannel_input_preserves_shape_order_and_source():
    source = np.array([[0.5, 9, 2], [-3, 8, -0.5], [0, 7, 1]], dtype=np.float32)
    original = source.copy()
    view = source[:, ::2]
    assert not view.flags.c_contiguous
    actual = quantize_pcm24(view)
    expected = np.array([[0.5, 8388607 / 8388608], [-1, -0.5],
                         [0, 8388607 / 8388608]], dtype=np.float32)
    assert actual.shape == view.shape
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, original)
    assert not np.shares_memory(actual, source)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, 1e100, -1e100])
def test_rejects_nonfinite_float32_input(invalid):
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="finite float32"):
            quantize_pcm24([0.5, invalid])


def test_float32_conversion_precedes_quantization():
    # This float64 input rounds up to exactly one PCM24 LSB in float32.
    source = np.array([(1 - 2 ** -26) / 8388608], dtype=np.float64)
    np.testing.assert_array_equal(quantize_pcm24(source), [1 / 8388608])


@pytest.mark.parametrize("backend", ["soundfile", "wave"])
def test_negative_zero_and_underflow_match_pcm24_readback_bytes_and_digest(backend):
    source = np.array([-0.0, -1e-100], dtype=np.float64)
    original_bytes = source.tobytes()
    quantized = quantize_pcm24(source)
    buffer = io.BytesIO()
    if backend == "soundfile":
        sf.write(buffer, quantized, 48000, format="WAV", subtype="PCM_24")
    else:
        with wave.open(buffer, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(3)
            output.setframerate(48000)
            output.writeframes(pack_pcm24_le(quantized))
    buffer.seek(0)
    readback, _ = sf.read(buffer, dtype="float32")
    readback_bytes = readback.astype("<f4").tobytes()
    quantized_bytes = quantized.astype("<f4").tobytes()
    assert readback_bytes == bytes(8)
    assert quantized_bytes == readback_bytes
    assert hashlib.sha256(quantized_bytes).digest() == hashlib.sha256(readback_bytes).digest()
    assert source.tobytes() == original_bytes


def test_quantization_is_idempotent_and_error_is_at_most_one_lsb():
    source = np.random.default_rng(123).uniform(-1, 1, (73, 3)).astype(np.float32)
    actual = quantize_pcm24(source)
    np.testing.assert_array_equal(quantize_pcm24(actual), actual)
    error = source.astype(np.float64) - actual
    assert np.all(error >= 0)
    assert np.all(error <= 1 / 8388608)


@pytest.mark.parametrize("shape", [(0,), (0, 1), (0, 3), (4, 1)])
def test_preserves_empty_and_single_channel_shapes(shape):
    source = np.zeros(shape, dtype=np.float32)
    actual = quantize_pcm24(source)
    assert actual.shape == shape
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, source)


def test_pack_signed_boundaries_and_frame_order_as_little_endian_bytes():
    samples = np.array([[-1, 0.5], [8388607 / 8388608, -0.5],
                        [-1 / 8388608, 1 / 8388608]], dtype=np.float32)
    expected = bytes.fromhex("00 00 80 00 00 40 FF FF 7F 00 00 C0 FF FF FF 01 00 00")
    assert pack_pcm24_le(samples) == expected


def test_pack_noncontiguous_input_uses_frame_order_without_mutating_source():
    source = np.array([[0, 9, -1], [0.5, 8, -0.5]], dtype=np.float32)
    original = source.copy()
    assert pack_pcm24_le(source[:, ::2]) == bytes.fromhex("00 00 00 00 00 80 00 00 40 00 00 C0")
    np.testing.assert_array_equal(source, original)


def test_pack_empty_samples():
    assert pack_pcm24_le(np.empty((0, 2), dtype=np.float32)) == b""


@pytest.mark.parametrize("channels", [1, 2, 3])
def test_wave_packed_and_soundfile_written_samples_decode_identically(channels):
    source = np.array([0.5, 2, -3, 0.25 / 8388608, -0.25 / 8388608, 0],
                      dtype=np.float32).reshape(-1, channels)
    expected = np.array([0.5, 8388607 / 8388608, -1, 0, -1 / 8388608, 0],
                        dtype=np.float32).reshape(-1, channels)
    quantized = quantize_pcm24(source)
    packed = pack_pcm24_le(quantized)
    assert len(packed) == source.size * 3
    # Decode signed integers independently from the codec's vector operations.
    codes = [int.from_bytes(packed[i:i + 3], "little", signed=True)
             for i in range(0, len(packed), 3)]
    assert codes == [4194304, 8388607, -8388608, 0, -1, 0]

    wave_buffer = io.BytesIO()
    with wave.open(wave_buffer, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(3)
        output.setframerate(48000)
        output.writeframes(packed)
    wave_buffer.seek(0)
    with wave.open(wave_buffer, "rb") as recording:
        assert recording.getnframes() == len(source)
        assert recording.getnchannels() == channels
        assert recording.getsampwidth() == 3
        assert recording.readframes(len(source)) == packed

    sf_buffer = io.BytesIO()
    sf.write(sf_buffer, quantized, 48000, format="WAV", subtype="PCM_24")
    for buffer in (wave_buffer, sf_buffer):
        buffer.seek(0)
        with sf.SoundFile(buffer) as recording:
            assert recording.subtype == "PCM_24"
            assert recording.samplerate == 48000
            np.testing.assert_array_equal(recording.read(dtype="float32", always_2d=True), expected)


def test_arbitrary_chunking_matches_one_shot_and_independent_integer_codes():
    rng = np.random.default_rng(456)
    source = rng.uniform(-3, 3, (103, 3)).astype(np.float32)
    expected_codes = [min(8388607, max(-8388608, math.floor(float(value) * 8388608)))
                      for value in source.flat]
    expected_bytes = b"".join(code.to_bytes(3, "little", signed=True) for code in expected_codes)
    boundaries = sorted(rng.choice(len(source) + 1, 17, replace=False))
    chunks = np.split(source, boundaries)
    quantized_chunks = [quantize_pcm24(chunk) for chunk in chunks]
    one_shot = quantize_pcm24(source)
    np.testing.assert_array_equal(np.concatenate(quantized_chunks), one_shot)
    np.testing.assert_array_equal(one_shot.ravel(), np.array(expected_codes) / 8388608)
    assert b"".join(pack_pcm24_le(chunk) for chunk in quantized_chunks) == expected_bytes
    assert pack_pcm24_le(one_shot) == expected_bytes

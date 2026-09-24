"""Acceptance through real capture, PCM24 metadata, reader, analysis and CSV."""
import csv
import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from base.analysis_worker import _load_wav_once
from base.raw_audio_csv_exporter import export_raw_audio_csv
from base.recording_process_protocol import RecordingResult
from base.wav_calibration_metadata import read_wav_calibration_metadata
from ui.sequence.analysis_task_builder import build_analysis_task_request
from unit_test.base.test_recording_capture import request
from unit_test.base.test_recording_finalization import capture_audio, read_result
from unit_test.base.test_ve3668n_capture import start_capture
from unit_test.base.ve3668n_fakes import CaptureSDK, wav_metadata


def riff_chunks(path):
    """Parse declared chunk boundaries, independently of the product scanner."""
    raw = path.read_bytes()
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    assert struct.unpack_from("<I", raw, 4)[0] == len(raw) - 8
    chunks = {}
    offset = 12
    while offset < len(raw):
        kind, size = struct.unpack_from("<4sI", raw, offset)
        start = offset + 8
        assert start + size + (size & 1) <= len(raw)
        chunks[kind] = raw[start:start + size]
        if size & 1:
            assert raw[start + size] == 0
        offset = start + size + (size & 1)
    assert offset == len(raw)
    return chunks


@pytest.mark.parametrize("backend,purpose", [
    ("soundcard_mono", "main"), ("soundcard_multi", "main"),
    ("soundcard_mono", "calibration"), ("ve", "main"), ("ve", "calibration"),
])
def test_capture_to_analysis_and_csv_keeps_pcm24_values(tmp_path, purpose, backend):
    # Expected integer codes are literal, never produced by the codec under test.
    values = np.array([0.25, -0.25, 0.5, 2.5, -3, 2**-25,
                       -2**-25, 0.001953125, 1], dtype=np.float32)
    codes = np.array([2097152, -2097152, 4194304, 8388607, -8388608,
                      0, -1, 16384, 8388607], dtype=np.int32)
    columns = 1 if backend == "soundcard_mono" or purpose == "calibration" else 2
    input_columns = np.column_stack((values, values[::-1]))[:, :columns]
    expected = np.column_stack((codes, codes[::-1]))[:, :columns]
    expected = (expected / 8388608).astype(np.float32)
    metadata = None
    if backend == "ve":
        metadata = wav_metadata(("measured", "measured"), sample_rate=51200)
        metadata["acquisition"]["machine_id"] = "test-machine-1"
        if purpose == "calibration":
            input_columns = np.resize(input_columns, (512000, 1))
            expected = np.resize(expected, (512000, 1))
        sdk = CaptureSDK(counts=[1] * len(values) if purpose == "main" else ())
        samples = iter(input_columns)
        def next_sample(*args, **kwargs):
            sdk.values = next(samples)
        if purpose == "main":
            sdk.hooks["read_task_data"] = next_sample
        else:
            offset = 0
            read_native = sdk.read_task_data
            def calibration_samples(*args, **kwargs):
                nonlocal offset
                buffer, count = read_native(*args, **kwargs)
                buffer[:count] = input_columns[offset:offset + count, 0].tolist()
                offset += count
                return buffer, count
            sdk.read_task_data = calibration_samples
        capture, _, _, _ = start_capture(tmp_path, sdk, request_options={
            "purpose": purpose, "channels": (7,) if columns == 1 else (7, 1),
            "calibration_metadata": metadata if purpose == "main" else None,
            "target_samples": len(expected), "trim_samples": 2 if purpose == "main" else 0,
        }, queue_seconds=11)
        try:
            descriptor = capture.wait(3)
            assert capture.join(3)
        finally:
            capture.cancel()
            capture.join(3)
        req = capture.request
    else:
        channels = (2,) if columns == 1 else (2, 0)
        metadata = {"recorded_channels": [
            {"wav_channel_index": i, "physical_input_channel": channel,
             "calibrated": True, "v2pa_factor": 2.5 + i, "standard_spl": 94.0}
            for i, channel in enumerate(channels)
        ]}
        req = request(tmp_path, purpose=purpose, channels=channels,
                      calibration_metadata=metadata)
        source = np.zeros((len(values), 3), dtype=np.float32)
        source[:, channels] = input_columns
        original = source.copy()
        _, descriptor = capture_audio(req, data=source)
        np.testing.assert_array_equal(source, original)
    assert isinstance(descriptor, RecordingResult), descriptor
    if purpose == "main":
        expected = expected[2:]
        assert descriptor.metadata_appended
        assert read_wav_calibration_metadata(req.path) == metadata
    else:
        # Calibration captures deliberately skip product metadata and startup trim.
        assert not descriptor.metadata_appended
        assert read_wav_calibration_metadata(req.path) is None
    assert descriptor.sample_digest == hashlib.sha256(expected.astype("<f4").tobytes()).hexdigest()
    result = read_result(descriptor, req, block_frames=65536)
    assert result.error is None and result.handles_released
    np.testing.assert_array_equal(result.audio.multi, expected)
    np.testing.assert_array_equal(result.audio.mono, expected.mean(axis=1, dtype=np.float32))
    assert result.audio.multi.dtype == np.float32
    if purpose == "main":
        assert not result.audio.multi.flags.writeable

    analysis_channels = req.channels if purpose == "main" else (0,)
    task = build_analysis_task_request(
        condition_key="pcm24", wav_path=req.path, source="手动查看",
        sequence_config=[{"seq1": {"acq": {"mode": "RECORD_ONLY"}}}],
        saved_active_input_channels=list(req.channels),
        analysis_config={"display_sequence": ["SPL"], "SPL": {
            "type": "SPL", "analysis_channels": list(analysis_channels),
            "limit_checked": False, "weighting": "Z"}},
    )
    analysis_audio, rate = _load_wav_once(task)
    np.testing.assert_array_equal(analysis_audio, result.audio.multi)
    assert rate == req.sample_rate
    assert [item.raw_channel for item in task.channel_mapping] == list(analysis_channels)
    if purpose == "main":
        factors = [10.0, 10.0] if backend == "ve" else [2.5 + i for i in range(columns)]
        assert {item.raw_channel: item.v2pa_factor for item in task.instances} == dict(zip(req.channels, factors))
    target = tmp_path / "raw.csv"
    export_raw_audio_csv(req.path, target, req.channels, block_frames=65536)
    with target.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["time_s", *(f"CH{channel + 1}" for channel in req.channels)]
    assert [row[1:] for row in rows[1:]] == [
        [format(float(value), ".9g") for value in frame] for frame in expected]
    np.testing.assert_array_equal(np.array([row[1:] for row in rows[1:]], dtype=np.float32), expected)
    info = sf.info(req.path)
    assert (info.subtype, info.frames, info.channels, info.samplerate) == (
        "PCM_24", len(expected), columns, req.sample_rate)
    chunks = riff_chunks(Path(req.path))
    assert len(chunks[b"data"]) == len(expected) * columns * 3
    tag, channels, sample_rate, byte_rate, align, bits = struct.unpack_from("<HHIIHH", chunks[b"fmt "])
    assert (tag, channels, sample_rate, byte_rate, align, bits) == (
        1, columns, req.sample_rate, req.sample_rate * columns * 3, columns * 3, 24)
    reference = tmp_path / "float-reference.wav"
    sf.write(reference, expected, req.sample_rate, subtype="FLOAT")
    assert len(chunks[b"data"]) * 4 == len(riff_chunks(reference)[b"data"]) * 3

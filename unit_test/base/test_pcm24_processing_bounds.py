"""Deterministic work bounds; these tests make no wall-clock speed promises."""
import threading

import numpy as np
import pytest
import soundfile as sf

from base import save_data
from base.recording_result_reader import ResultReader
from base.streaming_file_writer import StreamingWavWriter
from unit_test.base.test_recording_capture import request
from unit_test.base.test_recording_finalization import capture_audio


@pytest.mark.parametrize("frames", [65537, 196613])
def test_whole_array_save_quantizes_only_bounded_writer_chunks(tmp_path, monkeypatch, frames):
    import base.streaming_file_writer as writer_module
    original = writer_module.quantize_pcm24
    sizes = []
    def observe(block):
        sizes.append(len(block))
        return original(block)
    monkeypatch.setattr(writer_module, "quantize_pcm24", observe)
    source = np.full((frames, 2), 2.5, dtype=np.float32)
    save_data.save_audio_simple(str(tmp_path / "bounded.wav"), source, 48000)
    assert sum(sizes) == frames
    assert len(sizes) == (frames + 65535) // 65536
    assert max(sizes) <= 65536
    np.testing.assert_array_equal(source, np.full_like(source, 2.5))
    assert sf.info(tmp_path / "bounded.wav").subtype == "PCM_24"


def test_capture_and_metadata_do_not_decode_audio_then_reader_reads_once(tmp_path, monkeypatch):
    from base import wav_calibration_metadata as metadata_module
    original_read = sf.SoundFile.read
    def forbidden(*args, **kwargs):
        pytest.fail("capture or owned metadata finalization decoded/copied audio")
    metadata = {"recorded_channels": [{"wav_channel_index": 0,
        "physical_input_channel": 0, "calibrated": True,
        "v2pa_factor": 2.5, "standard_spl": 94.0}]}
    req = request(tmp_path, channels=(0,), trim_samples=0,
                  calibration_metadata=metadata)
    writes = []
    class Writer(StreamingWavWriter):
        def write_chunk(self, block):
            writes.append(len(block))
            return super().write_chunk(block)
    with monkeypatch.context() as capture_patch:
        capture_patch.setattr(sf.SoundFile, "read", forbidden)
        capture_patch.setattr(sf, "read", forbidden)
        capture_patch.setattr(metadata_module, "_copy_stream_exact", forbidden)
        _, descriptor = capture_audio(req, writer_factory=Writer)
    assert descriptor.metadata_appended
    assert writes == [2, 3, 4]
    reads, owners, outcomes = [], [], []
    class Source(sf.SoundFile):
        def read(self, frames, **kwargs):
            block = original_read(self, frames, **kwargs)
            reads.append(len(block))
            owners.append(threading.get_ident())
            return block
    reader = ResultReader(descriptor, outcomes.append, request=req, block_frames=2, opener=Source)
    reader.start()
    reader.thread.join(3)
    assert not reader.thread.is_alive()
    assert len(outcomes) == 1 and outcomes[0].error is None
    assert reads == [2, 2, 2, 2, 1]
    assert sum(reads) == req.target_samples
    assert set(owners) == {reader.thread.ident}
    assert reader.thread.ident != threading.get_ident()
    assert outcomes[0].audio.is_prepared_for(req)

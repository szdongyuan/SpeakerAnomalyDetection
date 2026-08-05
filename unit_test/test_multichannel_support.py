import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.io import wavfile

from base.save_data import save_audio_simple
from base.streaming_audio_processor import StreamingAudioProcessor
from base.soundcard_audio_processor import SoundcardAudioProcessor


class TestMultiChannelWavSaving(unittest.TestCase):
    def test_save_audio_simple_multichannel(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "x.wav")
            x = (np.random.randn(1000, 3)).astype(np.float32)
            save_audio_simple(path, x, sr=48000)
            sr, y = wavfile.read(path)
            self.assertEqual(sr, 48000)
            self.assertEqual(y.shape, (1000, 3))


class TestStreamingProcessorPayload(unittest.TestCase):
    def test_queue_trims_and_shapes(self):
        p = StreamingAudioProcessor()
        p.target_samples = 5
        p.samples_captured = 0
        p.is_recording = True

        payload, reached = p._queue_chunk_and_maybe_stop(np.ones((10, 2), dtype=np.float32))
        self.assertTrue(reached)
        self.assertIn("mono", payload)
        self.assertIn("multi", payload)
        self.assertEqual(payload["multi"].shape, (5, 2))
        self.assertEqual(payload["mono"].shape, (5,))


class TestSoundcardProcessorSelection(unittest.TestCase):
    def test_sd_rec_selects_and_mixes(self):
        recorded_dict = {
            "num_frames": 4,
            "sample_rate": 44100,
            "channels": 2,
            "input_channels": [0, 2],
            "device": {"index": 7},
            "blocking": True,
            "prolong_frames": 0,
        }

        fake = np.array(
            [
                [1.0, 10.0, 100.0],
                [2.0, 20.0, 200.0],
                [3.0, 30.0, 300.0],
                [4.0, 40.0, 400.0],
            ],
            dtype=np.float32,
        )

        with mock.patch("base.soundcard_audio_processor.sd") as sd_mock:
            sd_mock.rec.return_value = fake
            code, mono = SoundcardAudioProcessor.sd_rec(recorded_dict)

        self.assertEqual(code, 0)  # error_code.OK
        self.assertEqual(sd_mock.rec.call_args.kwargs["device"], 7)
        self.assertTrue("_recorded_multi" in recorded_dict)
        multi = recorded_dict["_recorded_multi"]
        self.assertEqual(multi.shape, (4, 2))
        np.testing.assert_allclose(multi[:, 0], fake[:, 0])
        np.testing.assert_allclose(multi[:, 1], fake[:, 2])
        np.testing.assert_allclose(mono, (fake[:, 0] + fake[:, 2]) / 2.0)

    def test_sd_rec_preserves_zero_device_index(self):
        recorded_dict = {
            "num_frames": 2,
            "sample_rate": 44100,
            "channels": 1,
            "device": 0,
            "blocking": True,
        }

        with mock.patch("base.soundcard_audio_processor.sd") as sd_mock:
            sd_mock.rec.return_value = np.ones((2, 1), dtype=np.float32)
            code, _ = SoundcardAudioProcessor.sd_rec(recorded_dict)

        self.assertEqual(code, 0)
        self.assertEqual(sd_mock.rec.call_args.kwargs["device"], 0)


if __name__ == "__main__":
    unittest.main()


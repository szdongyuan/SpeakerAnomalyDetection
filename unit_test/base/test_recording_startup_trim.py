import os
import tempfile
import unittest
from unittest.mock import patch

from base import recording_settings
from base.play_and_record import resolve_startup_trim_samples


class TestRecordingStartupTrim(unittest.TestCase):
    def tearDown(self):
        recording_settings.reset_cache()

    def test_missing_global_config_uses_100_ms_startup_trim(self):
        with tempfile.TemporaryDirectory() as folder:
            missing_path = os.path.join(folder, "recording_settings.json")
            with patch.object(recording_settings, "_GLOBAL_SETTINGS_PATH", missing_path):
                recording_settings.reset_cache()

                self.assertEqual(recording_settings.resolve_startup_trim_ms({}), 100.0)
                self.assertEqual(resolve_startup_trim_samples({}, 48_000), 4_800)

    def test_product_config_can_explicitly_disable_startup_trim(self):
        self.assertEqual(
            resolve_startup_trim_samples({"startup_trim_ms": 0}, 48_000),
            0,
        )


if __name__ == "__main__":
    unittest.main()

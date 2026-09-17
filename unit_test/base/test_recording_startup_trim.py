import unittest
from unittest.mock import patch

from base import recording_settings
from base.play_and_record import resolve_startup_trim_samples


class TestRecordingStartupTrim(unittest.TestCase):
    def test_delay_resolves_from_queue_or_code_without_global_read(self):
        cases = [
            ({}, 100),
            ({"startup_trim_ms": 0}, 0),
            ({"startup_trim_ms": 123}, 123),
            ({"startup_trim_ms": 2000}, 2000),
            ({"startup_trim_ms": -1}, 100),
            ({"startup_trim_ms": True}, 100),
            ({"startup_trim_ms": "invalid"}, 100),
        ]
        with patch.object(
            recording_settings, "get_global_settings",
            side_effect=AssertionError("delay must not load global settings"),
        ) as read_global:
            for detail, expected in cases:
                with self.subTest(detail=detail):
                    self.assertEqual(
                        recording_settings.resolve_startup_trim_ms(detail), expected
                    )
                    self.assertEqual(
                        resolve_startup_trim_samples(detail, 48_000), expected * 48
                    )
            read_global.assert_not_called()

    def test_product_config_can_explicitly_disable_startup_trim(self):
        self.assertEqual(
            resolve_startup_trim_samples({"startup_trim_ms": 0}, 48_000),
            0,
        )


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from base.channel_layout import (
    DEFAULT_CHANNEL_LAYOUT,
    load_channel_layout,
    save_channel_layout,
)


class TestChannelLayout(unittest.TestCase):
    def test_aliases_support_channels_beyond_the_legacy_five(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = str(Path(temp_dir) / "channels.json")
            self.assertTrue(save_channel_layout({"CH8": "后排", "CH128": "参考", "CH129": "无效"}, path))
            restored = load_channel_layout(path)
            self.assertEqual(restored["CH8"], "后排")
            self.assertEqual(restored["CH128"], "参考")
            self.assertNotIn("CH129", restored)

    def test_missing_file_uses_complete_default_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "channel_layout.json"

            layout = load_channel_layout(str(path))

        self.assertEqual(layout, DEFAULT_CHANNEL_LAYOUT)
        self.assertIsNot(layout, DEFAULT_CHANNEL_LAYOUT)

    def test_partial_or_invalid_values_fall_back_per_channel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "channel_layout.json"
            path.write_text(
                json.dumps(
                    {
                        "CH1": " 正前 ",
                        "CH2": "",
                        "CH3": None,
                        "CH4": "右侧",
                        "unused": "ignored",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            layout = load_channel_layout(str(path))

        self.assertEqual(
            layout,
            {
                "CH1": "正前",
                "CH2": "后",
                "CH3": "左",
                "CH4": "右侧",
                "CH5": "上",
            },
        )

    def test_save_writes_normalized_complete_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "channel_layout.json"

            saved = save_channel_layout(
                {
                    "CH1": "前方",
                    "CH5": " 顶部 ",
                },
                str(path),
            )

            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(saved)
        self.assertEqual(payload, {
            "CH1": "前方",
            "CH2": "后",
            "CH3": "左",
            "CH4": "右",
            "CH5": "顶部",
        })


if __name__ == "__main__":
    unittest.main()

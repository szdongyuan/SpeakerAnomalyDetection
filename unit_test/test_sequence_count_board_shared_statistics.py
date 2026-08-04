import json
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

from PyQt5.QtWidgets import QApplication

from ui.sequence.sequencement_count_board import SequenceCountBoard


class TestSequenceCountBoardSharedStatistics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _prepare_default_dir(
        self,
        folder,
        total=0,
        ok=0,
        ng=0,
        not_labels=0,
        mark_total=0,
        mark_ok=0,
        mark_ng=0,
    ):
        default_dir = folder + os.sep
        current_time = datetime.now().strftime("%Y-%m-%d")
        os.makedirs(os.path.join(folder, "log", "test_result_log"), exist_ok=True)
        os.makedirs(os.path.join(folder, "ui", "ui_config"), exist_ok=True)
        with open(os.path.join(folder, "log", "test_result_log", f"{current_time}.dat"), "w") as f:
            ok_percent = round(ok / total * 100, 2) if total > 0 else 0
            f.writelines(
                [
                    f"total: {total}\n",
                    f"ok: {ok}\n",
                    f"ng: {ng}\n",
                    f"not_labels: {not_labels}\n",
                    f"ok_percent: {ok_percent}%\n",
                    f"datatime: {current_time}\n",
                ]
            )
        with open(os.path.join(folder, "ui", "ui_config", "mark_result.json"), "w") as f:
            json.dump(
                {
                    "total": mark_total,
                    "ok": mark_ok,
                    "ng": mark_ng,
                    "not_labels": 0,
                    "datatime": current_time,
                },
                f,
            )
        return default_dir

    def test_mark_mode_displays_shared_test_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            default_dir = self._prepare_default_dir(
                folder,
                total=5,
                ok=1,
                ng=1,
                not_labels=3,
                mark_total=99,
                mark_ok=98,
                mark_ng=1,
            )

            with patch("ui.sequence.sequencement_count_board.DEFAULT_DIR", default_dir), patch(
                "base.save_data.DEFAULT_DIR",
                default_dir,
            ):
                board = SequenceCountBoard({})
                board.on_mark_btn_clicked()

                self.assertEqual(board.total_line_edit.text(), "5")
                self.assertEqual(board.ok_line_edit.text(), "1")
                self.assertEqual(board.ng_line_edit.text(), "1")
                self.assertEqual(board.not_labeled_line_edit.text(), "3")

    def test_mark_relabel_updates_shared_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            default_dir = self._prepare_default_dir(folder)

            with patch("ui.sequence.sequencement_count_board.DEFAULT_DIR", default_dir), patch(
                "base.save_data.DEFAULT_DIR",
                default_dir,
            ):
                board = SequenceCountBoard({})
                board.on_mark_btn_clicked()

                board.update_mark_result_file_on_relabel("not_labeled", "NG")
                self.assertEqual(board.total_line_edit.text(), "1")
                self.assertEqual(board.ok_line_edit.text(), "0")
                self.assertEqual(board.ng_line_edit.text(), "1")
                self.assertEqual(board.not_labeled_line_edit.text(), "0")

                board.update_mark_result_file_on_relabel("NG", "OK")
                self.assertEqual(board.total_line_edit.text(), "1")
                self.assertEqual(board.ok_line_edit.text(), "1")
                self.assertEqual(board.ng_line_edit.text(), "0")
                self.assertEqual(board.not_labeled_line_edit.text(), "0")

                board.update_mark_result_file_on_relabel("OK", "not_labeled")
                self.assertEqual(board.total_line_edit.text(), "1")
                self.assertEqual(board.ok_line_edit.text(), "0")
                self.assertEqual(board.ng_line_edit.text(), "0")
                self.assertEqual(board.not_labeled_line_edit.text(), "1")

                board.update_mark_result_file_on_relabel("not_labeled", "OK")
                self.assertEqual(board.total_line_edit.text(), "1")
                self.assertEqual(board.ok_line_edit.text(), "1")
                self.assertEqual(board.ng_line_edit.text(), "0")
                self.assertEqual(board.not_labeled_line_edit.text(), "0")

    def test_not_labeled_result_counts_as_completed_shared_summary_item(self):
        with tempfile.TemporaryDirectory() as folder:
            default_dir = self._prepare_default_dir(folder)

            with patch("ui.sequence.sequencement_count_board.DEFAULT_DIR", default_dir), patch(
                "base.save_data.DEFAULT_DIR",
                default_dir,
            ):
                board = SequenceCountBoard({})
                board.set_test_result_file("not_labeled")
                board.set_test_text()

                self.assertEqual(board.total_line_edit.text(), "1")
                self.assertEqual(board.ok_line_edit.text(), "0")
                self.assertEqual(board.ng_line_edit.text(), "0")
                self.assertEqual(board.not_labeled_line_edit.text(), "1")


if __name__ == "__main__":
    unittest.main()

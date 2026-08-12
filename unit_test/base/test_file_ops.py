import os
from unittest import mock
from zipfile import ZipFile

import pytest

from base.file_ops import FileOps
from consts import error_code


class TestFileOps(object):

    @pytest.mark.parametrize("rmtree_ret, mkdir_ret, ret", [
        (mock.Mock(), [mock.Mock(), mock.Mock(), mock.Mock()],
         (error_code.OK, "finish creating empty okng dir")),
        (mock.Mock(), [mock.Mock(), Exception("xxx"), mock.Mock()],
         (error_code.INVALID_PATH, "failed to create [audio_data/], xxx"))
    ])
    @mock.patch("base.file_ops.os.mkdir")
    @mock.patch("base.file_ops.shutil.rmtree")
    def test_create_empty_okng(self, mock_rmtree, mock_mkdir, rmtree_ret, mkdir_ret, ret):
        mock_rmtree.return_value = rmtree_ret
        mock_mkdir.side_effect = mkdir_ret
        result = FileOps().create_empty_okng("audio_data/")
        assert result == ret


def test_create_zip_with_files_keeps_local_categories_and_database_at_root(tmp_path):
    base_dir = tmp_path / "application"
    audio_path = base_dir / "audio_data" / "stored_data" / "OK" / "sample.wav"
    database_path = base_dir / "database" / "audio_data.db"
    audio_path.parent.mkdir(parents=True)
    database_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"RIFF")
    database_path.write_bytes(b"database")
    output_path = tmp_path / "export.zip"

    FileOps.create_zip_with_files(
        [str(audio_path), "database/audio_data.db"],
        str(output_path),
        base_dir=str(base_dir),
    )

    with ZipFile(output_path, "r") as archive:
        assert set(archive.namelist()) == {"OK/sample.wav", "audio_data.db"}


@pytest.mark.skipif(os.name != "nt", reason="Windows mount semantics")
def test_create_zip_with_files_supports_unc_audio_outside_local_base_dir(tmp_path):
    source_path = r"\\192.168.2.100\项目文件\S004-1\OK\sample.wav"
    output_path = tmp_path / "export.zip"

    with mock.patch("base.file_ops.os.path.exists", return_value=True), mock.patch(
        "base.file_ops.ZipFile"
    ) as zip_class:
        zip_file = zip_class.return_value.__enter__.return_value

        FileOps.create_zip_with_files(
            [source_path],
            str(output_path),
            base_dir=r"D:\application",
        )

    zip_file.write.assert_called_once_with(
        source_path,
        os.path.join("OK", "sample.wav"),
    )

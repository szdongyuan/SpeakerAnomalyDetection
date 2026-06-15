import os
import pytest
import tempfile
from zipfile import ZipFile
from unittest import mock

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


def test_get_zip_arcname_categorizes_audio_paths():
    base_dir = os.path.abspath("project")
    path = os.path.join(base_dir, "audio_data", "stored_sample", "line", "model", "20260508", "OK", "a.wav")

    result = FileOps.get_zip_arcname(path, base_dir=base_dir, categorize=True)

    assert result == os.path.join("OK", "a.wav")


def test_ensure_directory_exists_creates_missing_parent(tmp_path):
    target = tmp_path / "missing" / "nested" / "out.txt"

    FileOps.ensure_directory_exists(target)

    assert target.parent.is_dir()


def test_ensure_directory_exists_accepts_current_directory_path():
    FileOps.ensure_directory_exists("out.txt")


def test_get_zip_arcname_keeps_database_at_root():
    base_dir = os.path.abspath("project")
    path = os.path.join(base_dir, "database", "audio_data.db")

    result = FileOps.get_zip_arcname(path, base_dir=base_dir, categorize=True)

    assert result == "audio_data.db"


def test_get_zip_arcname_uses_root_when_category_disabled():
    base_dir = os.path.abspath("project")
    path = os.path.join(base_dir, "audio_data", "stored_sample", "line", "model", "20260508", "NG", "b.wav")

    result = FileOps.get_zip_arcname(path, base_dir=base_dir, categorize=False)

    assert result == "b.wav"


def test_write_audio_to_zip_and_delete_source_success():
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_dir:
        audio_dir = os.path.join(temp_dir, "OK")
        audio_path = os.path.join(audio_dir, "a.wav")
        zip_path = os.path.join(temp_dir, "out.zip")
        os.mkdir(audio_dir)
        with open(audio_path, "wb") as audio_file:
            audio_file.write(b"audio")
        audio_data = ("audio-id", audio_path, "model", 44100, "2026-05-08", "OK", None, None)

        with ZipFile(zip_path, "w") as zip_file:
            result = FileOps.write_audio_to_zip_and_delete_source(zip_file, audio_data, base_dir=temp_dir)

        with ZipFile(zip_path, "r") as zip_file:
            assert "OK/a.wav" in zip_file.namelist()
            assert zip_file.read("OK/a.wav") == b"audio"
        assert result["status"] == "packaged_deleted"
        assert result["audio_data"] == audio_data
        assert result["db_delete_id"] == "audio-id"
        assert result["error"] is None
        assert not os.path.exists(audio_path)


def test_write_audio_to_zip_and_delete_source_missing_file():
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_dir:
        zip_path = os.path.join(temp_dir, "out.zip")
        audio_data = (
            "missing-id",
            os.path.join(temp_dir, "missing.wav"),
            "model",
            44100,
            "2026-05-08",
            "OK",
            None,
            None,
        )

        with ZipFile(zip_path, "w") as zip_file:
            result = FileOps.write_audio_to_zip_and_delete_source(zip_file, audio_data, base_dir=temp_dir)

        assert result["status"] == "package_failed"
        assert result["audio_data"] == audio_data
        assert result["db_delete_id"] is None
        assert isinstance(result["error"], FileNotFoundError)


def test_write_audio_to_zip_and_delete_source_delete_failure():
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_dir:
        audio_dir = os.path.join(temp_dir, "NG")
        audio_path = os.path.join(audio_dir, "b.wav")
        zip_path = os.path.join(temp_dir, "out.zip")
        os.mkdir(audio_dir)
        with open(audio_path, "wb") as audio_file:
            audio_file.write(b"audio")
        audio_data = ("delete-fail-id", audio_path, "model", 44100, "2026-05-08", "NG", None, None)

        def fail_remove(path):
            raise PermissionError(path)

        with ZipFile(zip_path, "w") as zip_file:
            result = FileOps.write_audio_to_zip_and_delete_source(
                zip_file,
                audio_data,
                base_dir=temp_dir,
                remove_func=fail_remove,
            )

        with ZipFile(zip_path, "r") as zip_file:
            assert "NG/b.wav" in zip_file.namelist()
            assert zip_file.read("NG/b.wav") == b"audio"
        assert result["status"] == "source_delete_failed"
        assert result["audio_data"] == audio_data
        assert result["db_delete_id"] is None
        assert isinstance(result["error"], PermissionError)
        assert os.path.exists(audio_path)


def test_create_zip_with_files_records_missing_file_failure():
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_dir:
        ok_dir = os.path.join(temp_dir, "OK")
        os.mkdir(ok_dir)
        existing_path = os.path.join(ok_dir, "exists.wav")
        missing_path = os.path.join(ok_dir, "missing.wav")
        zip_path = os.path.join(temp_dir, "out.zip")
        failures = []
        with open(existing_path, "wb") as audio_file:
            audio_file.write(b"audio")

        FileOps.create_zip_with_files(
            [existing_path, missing_path],
            zip_path,
            base_dir=temp_dir,
            failure_callback=lambda path, error: failures.append((path, error)),
        )

        with ZipFile(zip_path, "r") as zip_file:
            assert "OK/exists.wav" in zip_file.namelist()
            assert zip_file.read("OK/exists.wav") == b"audio"
            assert "OK/missing.wav" not in zip_file.namelist()
        assert len(failures) == 1
        assert failures[0][0] == missing_path
        assert isinstance(failures[0][1], FileNotFoundError)


def test_create_zip_with_files_creates_missing_output_parent(tmp_path):
    source_path = tmp_path / "OK" / "exists.wav"
    zip_path = tmp_path / "missing" / "package" / "out.zip"
    source_path.parent.mkdir()
    source_path.write_bytes(b"audio")

    FileOps.create_zip_with_files([str(source_path)], str(zip_path), base_dir=str(tmp_path))

    assert zip_path.is_file()
    with ZipFile(zip_path, "r") as zip_file:
        assert "OK/exists.wav" in zip_file.namelist()

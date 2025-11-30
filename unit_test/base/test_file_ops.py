from unittest import mock
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

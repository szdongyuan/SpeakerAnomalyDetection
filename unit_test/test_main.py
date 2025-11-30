from unittest import mock
import numpy as np
import pytest

from consts import error_code
import main


class TestMain(object):

    @pytest.mark.parametrize("copy_ret, ret", [
        ([(error_code.OK, "xxx")], (error_code.OK, "xxx")),
        ([(error_code.UNKNOWN_ERROR, "xxx")], (error_code.UNKNOWN_ERROR, "xxx")),
        (Exception("xxx"), (error_code.INVALID_DATA_LOADING, "Failed to load data from the database. xxx"))
    ])
    @mock.patch("main.copy_from_restored_audio_database")
    def test_load_data_from_database(self, mock_copy, copy_ret, ret):
        mock_copy.side_effect = copy_ret
        result = main.load_data_from_database()
        assert result == ret

    @pytest.mark.parametrize("process_ret, raw_signals, ret", [
        ([1, 2, 3], [1, 1, 1], np.array([1, 2, 3])),
        ([1, 2, 3], [], np.array([]))
    ])
    @mock.patch("main.PreprocessingManager.process")
    def test_preprocess_raw_signals(self, mock_process, process_ret, raw_signals, ret):
        mock_process.side_effect = process_ret
        fs = list(range(len(raw_signals)))
        preprocess_config = {}
        result = main.preprocess_raw_signals(raw_signals, fs, preprocess_config)
        assert list(result) == list(ret)

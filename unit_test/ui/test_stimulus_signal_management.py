import mock
import pytest

from base.stimulus_signal_management import StimulusSignalManagement
from consts import error_code


class TestStimulusSignalManagement(object):
    @pytest.mark.parametrize("database_set, query_set, update_set, stimulus_id, is_default, result_set", [
        (mock.Mock(), [(error_code.INVALID_QUERY, (1, 2, 3))], [], '12', 1,
         (error_code.INVALID_QUERY, "Stimulus ID 12 not found or query failed."),
         ),
        (mock.Mock(), [(error_code.OK, ())], [], '12', 1,
         (error_code.INVALID_QUERY, "Stimulus ID 12 not found or query failed."),
         ),
        (mock.Mock(), [(error_code.OK, [(1,)])], [], '12', 1,
         (error_code.OK, "Stimulus default settings are updated successfully."),
         ),
        (mock.Mock(), [(error_code.OK, [(0,)])],
         [(error_code.INVALID_UPDATE, ""), ()], '12', 1,
         (error_code.INVALID_UPDATE, "Failed to update stimulus_id 12."),
         ),
        (mock.Mock(), [(error_code.OK, [(0,)])],
         [(error_code.OK, "xxx"), (error_code.INVALID_UPDATE, "xxx")], '12', 1,
         (error_code.INVALID_UPDATE, "Failed to reset other records' is_default."),
         ),
        (mock.Mock(), [(error_code.OK, [(0,)])],
         [(error_code.OK, ""), (error_code.OK, "")], '12', 1,
         (error_code.OK, "Stimulus default settings are updated successfully."),
         ),
        (mock.Mock(), Exception('xxx'),
         [(error_code.OK, ""), (error_code.OK, "")], '12', 1,
         (error_code.INVALID_UPDATE, "Failed to update the stimulus default settings. xxx"),
         ),
        (mock.Mock(), [(error_code.OK, [(0,)])],
         Exception('xxx'), '12', 1,
         (error_code.INVALID_UPDATE, "Failed to update the stimulus default settings. xxx"),
         ),
    ])
    @mock.patch("base.stimulus_signal_management.DataSave")
    def test_update_stimulus_default(self, mock_database, database_set, query_set, update_set, stimulus_id, is_default,
                                     result_set):
        mock_database.return_value.__enter__.return_value = database_set
        mock_database.return_value.__enter__.return_value.query.side_effect = query_set
        mock_database.return_value.__enter__.return_value.update_table_data.side_effect = update_set
        result = StimulusSignalManagement().update_stimulus_default(stimulus_id, is_default)
        assert result == result_set

    @pytest.mark.parametrize("database_set, query_set, result_ret", [
        (mock.Mock(),
         [(error_code.OK, ('366f6b6e-9da8-11ef-91bc-107c610bb999', 'chirp', 'log', 2, 60, 2000, 44100, 5, 1))],
         (error_code.OK, ('366f6b6e-9da8-11ef-91bc-107c610bb999', 'chirp', 'log', 2, 60, 2000, 44100, 5, 1)),
         ),
        (mock.Mock(), [(error_code.OK, ())],
         (error_code.INVALID_QUERY, "Failed to query the default stimulus signal settings.")),
        (mock.Mock(),
         [(error_code.INVALID_QUERY, (1, 2, 3))],
         (error_code.INVALID_QUERY, "Failed to query the default stimulus signal settings.")
         ),
        (mock.Mock(), Exception('xxx'),
         (error_code.INVALID_QUERY, "Failed to query the default stimulus signal. xxx")
         ),
    ])
    @mock.patch("base.stimulus_signal_management.DataSave")
    def test_query_default_stimulus_info(self, mock_database, database_set, query_set, result_ret):
        mock_database.return_value.__enter__.return_value = database_set
        mock_database.return_value.__enter__.return_value.query.side_effect = query_set
        result = StimulusSignalManagement().query_default_stimulus_info()
        assert result == result_ret

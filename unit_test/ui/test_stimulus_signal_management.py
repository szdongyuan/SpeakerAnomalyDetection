import json

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

    @mock.patch("base.stimulus_signal_management.DataSave")
    def test_query_all_stimulus_info_returns_dict_rows(self, mock_database):
        metadata = {
            "stimulus_method": "frequency_stepped",
            "frequency_mode": "custom_linear",
            "stimulus_type": "custom_linear",
            "repeat_times": 1,
            "sample_rate": 48000,
            "frequencies": [1000, 2000],
            "min_duration": 0.01,
            "min_cycles": 4,
        }
        rows = [
            ("12", "chirp", "log", 1, 10, 20, 44100, 3, 1, "RMS", "0", 0, "legacy_chirp", None),
            (
                "step-valid",
                "frequency_stepped",
                "custom_linear",
                1,
                1000,
                2000,
                48000,
                1,
                2,
                "RMS",
                1.0,
                1,
                "valid_step",
                json.dumps(metadata),
            ),
            (
                "step-invalid",
                "frequency_stepped",
                "custom_linear",
                1,
                1000,
                2000,
                48000,
                1,
                2,
                "RMS",
                1.0,
                0,
                "invalid_step",
                "{bad-json",
            ),
        ]
        mock_database.return_value.__enter__.return_value = mock.Mock()
        mock_database.return_value.__enter__.return_value.query.side_effect = [(error_code.OK, rows)]

        result = StimulusSignalManagement().query_all_stimulus_info()

        assert result[0] == error_code.OK
        loaded = {row["stimulus_name"]: row for row in result[1]}
        assert loaded["legacy_chirp"]["stimulus_method"] == "chirp"
        assert loaded["legacy_chirp"]["voltage"] == pytest.approx(0.0)
        assert loaded["legacy_chirp"]["stimulus_metadata_json"] is None
        assert "stimulus_payload" not in loaded["legacy_chirp"]
        assert loaded["valid_step"]["step_sc_row_state"] == "valid"
        assert loaded["valid_step"]["stimulus_payload"]["stimulus_id"] == "step-valid"
        assert loaded["valid_step"]["stimulus_payload"]["frequencies"] == pytest.approx([1000, 2000])
        assert loaded["invalid_step"]["step_sc_row_state"] == "invalid_metadata"
        assert "stimulus_payload" not in loaded["invalid_step"]

    @pytest.mark.parametrize("database_ret, query_ret, result_ret", [
        (mock.Mock(), [(error_code.OK, [])],
         (error_code.INVALID_QUERY, "Failed to query stimulus signal info or no stimulus signal info."),
         ),
        (mock.Mock(), [(error_code.INVALID_QUERY, [('12', 'chirp'), ('13', 'step')])],
         (error_code.INVALID_QUERY, "Failed to query stimulus signal info or no stimulus signal info."),
         ),
        (mock.Mock(), Exception('xxx'),
         (error_code.INVALID_QUERY, "Failed to query stimulus signal. xxx")
         )
    ])
    @mock.patch("base.stimulus_signal_management.DataSave")
    def test_query_all_stimulus_info(self, mock_database, database_ret, query_ret, result_ret):
        mock_database.return_value.__enter__.return_value = database_ret
        mock_database.return_value.__enter__.return_value.query.side_effect = query_ret
        result = StimulusSignalManagement().query_all_stimulus_info()
        assert result == result_ret

    @pytest.mark.parametrize("input_ret, result_ret", [
        ({"database_ret": mock.Mock(),
          "default_ret": 0,
          "query_match_ret": [('12',)],
          "get_id_ret": mock.Mock(),
          "insert_ret": mock.Mock(),
          "stimulus_info": {"stimulus_method": 'chirp',
                            "stimulus_type": 'log',
                            "start_freq": 80,
                            "stop_freq": 1000,
                            "total_time": 3.0,
                            "repeat_times": 1,
                            "num_steps": 1,
                            "sample_rate": 44100},
          }, (error_code.INVALID_NAME, "This stimulus signals name info already exists.")
         ),
        ({"database_ret": mock.Mock(),
          "default_ret": 0,
          "query_match_ret": [],
          "get_id_ret": [('12', 'chirp', 'log', 1, 80, 1000, 44100, 3, 1, 0)],
          "insert_ret": [(error_code.OK, 'Successfully insert.')],
          "stimulus_info": {"stimulus_method": 'chirp',
                            "stimulus_type": 'log',
                            "start_freq": 80,
                            "stop_freq": 1000,
                            "total_time": 3.0,
                            "repeat_times": 1,
                            "num_steps": 1,
                            "sample_rate": 44100
                            }
          }, (error_code.OK, "Successfully saved stimulus signals to the database.")
         ),
        ({"database_ret": mock.Mock(),
          "default_ret": 0,
          "query_match_ret": [],
          "get_id_ret": [('12', 'chirp', 'log', 1, 80, 1000, 44100, 3, 1, 0)],
          "insert_ret": [(error_code.INVALID_INSERT, 'Failed to insert.')],
          "stimulus_info": {"stimulus_method": 'chirp',
                            "stimulus_type": 'log',
                            "start_freq": 80,
                            "stop_freq": 1000,
                            "total_time": 3.0,
                            "repeat_times": 1,
                            "num_steps": 1,
                            "sample_rate": 44100
                            }
          }, (error_code.INVALID_INSERT, 'Failed to insert.')
         ),
        ({"database_ret": mock.Mock(),
          "default_ret": 0,
          "query_match_ret": [],
          "get_id_ret": Exception('xxx'),
          "insert_ret": [(error_code.OK, 'Successfully insert.')],
          "stimulus_info": {"stimulus_method": 'chirp',
                            "stimulus_type": 'log',
                            "start_freq": 80,
                            "stop_freq": 1000,
                            "total_time": 3.0,
                            "repeat_times": 1,
                            "num_steps": 1,
                            "sample_rate": 44100
                            }
          }, (error_code.INVALID_SAVE, "Failed to save stimulus signals to the database. xxx")
         ),
    ])
    @mock.patch("base.stimulus_signal_management.DataSave")
    def test_save_stimulus_info_to_db(self, mock_database, input_ret, result_ret):
        mock_database.return_value.__enter__.return_value = input_ret["database_ret"]
        mock_database.return_value.__enter__.return_value.set_default.return_value = input_ret["default_ret"]
        mock_database.return_value.__enter__.return_value.query_matching_data.return_value = input_ret["query_match_ret"]
        mock_database.return_value.__enter__.return_value.get_data_id.side_effect = input_ret["get_id_ret"]
        mock_database.return_value.__enter__.return_value.insert_data_into_db.side_effect = input_ret["insert_ret"]
        result = StimulusSignalManagement().save_stimulus_info_to_db(input_ret["stimulus_info"])
        assert result == result_ret

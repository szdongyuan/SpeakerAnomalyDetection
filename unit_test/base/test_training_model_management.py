import json
import mock
import pytest
import sqlite3

import base.db_manager as db_manager_module
import base.training_model_management as training_model_management_module
from base.db_manager import DataSave
from base.training_model_management import TrainingModelManagement
from consts import error_code, model_consts


class TestTrainingModelManagement(object):
    @pytest.fixture(autouse=True)
    def isolate_database_init(self):
        with mock.patch("base.training_model_management.ensure_audio_database_ready"):
            yield

    @mock.patch("base.training_model_management.DataSave")
    def test_db_access_uses_current_audio_database_path_after_database_path_changes(self, mock_database, monkeypatch):
        old_path = model_consts.AUDIO_DATABASE_PATH
        new_path = old_path + ".next"
        mock_database.return_value.__enter__.return_value.query.return_value = (
            error_code.OK,
            [("demo-model", "132300 x 1")],
        )

        manager = TrainingModelManagement()
        monkeypatch.setattr(model_consts, "AUDIO_DATABASE_PATH", new_path)

        result = manager.get_all_model_name_from_db()

        assert result == (error_code.OK, [("demo-model", "132300 x 1")])
        mock_database.assert_called_with(new_path)

    def test_default_db_access_initializes_rotated_audio_database_path(self, tmp_path, monkeypatch):
        old_path = tmp_path / "old" / "audio_data.db"
        new_path = tmp_path / "new" / "audio_data.db"
        monkeypatch.setattr(model_consts, "AUDIO_DATABASE_PATH", str(old_path))
        monkeypatch.setattr(training_model_management_module, "ensure_audio_database_ready",
                            db_manager_module.ensure_audio_database_ready)
        manager = TrainingModelManagement()

        monkeypatch.setattr(model_consts, "AUDIO_DATABASE_PATH", str(new_path))

        result = manager.get_all_model_name_from_db()

        assert result == (error_code.INVALID_QUERY, "Failed to query all mdoel name.")
        assert new_path.is_file()
        with sqlite3.connect(new_path) as connection:
            table = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
                ("training_model_table",),
            ).fetchone()
        assert table == ("training_model_table",)

    def test_explicit_db_path_override_remains_authoritative_after_database_path_changes(self, tmp_path, monkeypatch):
        override_path = tmp_path / "override" / "audio_data.db"
        canonical_path = tmp_path / "canonical" / "audio_data.db"
        database = DataSave(str(override_path))
        code, msg = database.create_audio_tables()
        database.close()
        assert code == error_code.OK, msg
        monkeypatch.setattr(model_consts, "AUDIO_DATABASE_PATH", str(tmp_path / "initial" / "audio_data.db"))
        monkeypatch.setattr(training_model_management_module, "ensure_audio_database_ready",
                            db_manager_module.ensure_audio_database_ready)
        manager = TrainingModelManagement()
        manager.db_path = str(override_path)

        monkeypatch.setattr(model_consts, "AUDIO_DATABASE_PATH", str(canonical_path))

        result = manager.get_all_model_name_from_db()

        assert result == (error_code.INVALID_QUERY, "Failed to query all mdoel name.")
        assert override_path.is_file()
        assert not canonical_path.exists()

    @pytest.mark.parametrize("input_ret, result_ret", [
        ({"database_ret": mock.Mock(),
          "model_info_set": [(error_code.OK, ("model_info",))],
          "signal_length": 132300,
          "model_path": "../../models/cnn_config_001.keras",
          "config_path": "../../config.yml",
          "ret_str": json.dumps({"result": ['accuracy: 0.7', 'cm_info']}),
          "model_description": "No description",
          },
         (error_code.OK, "Successfully saved the training model info to the database."),
         ),
        ({"database_ret": mock.Mock(),
          "model_info_set": [(error_code.OK, ("model_info",))],
          "signal_length": 132300,
          "model_path": "../../models/cnn_config_001.keras",
          "config_path": "../../config.yml",
          "ret_str": None,
          "model_description": None,
          },
         (error_code.OK, "Successfully saved the training model info to the database."),
         ),
        ({"database_ret": Exception(),
          "model_info_set": [(error_code.OK, ("model_info",))],
          "signal_length": 132300,
          "model_path": "../../models/cnn_config_001.keras",
          "config_path": "../../config.yml",
          "ret_str": None,
          "model_description": None,
          },
         (error_code.INVALID_INSERT, "Failed to save the training model info to the database. "
                                     "'Exception' object has no attribute 'insert_data_into_db'"),
         ),
    ])
    @mock.patch("base.training_model_management.TrainingModelManagement.get_training_model_info_to_db")
    @mock.patch("base.training_model_management.DataSave")
    def test_save_training_model_info_to_db(self, mock_database, mock_model_info, input_ret, result_ret):
        mock_database.return_value.__enter__.return_value = input_ret["database_ret"]
        mock_model_info.side_effect = input_ret["model_info_set"]
        tmm = TrainingModelManagement()
        result = tmm.save_training_model_info_to_db(input_ret["signal_length"], input_ret["model_path"],
                                                    input_ret["config_path"],
                                                    input_ret["ret_str"], input_ret["model_description"])
        assert result == result_ret

    @pytest.mark.parametrize("database_ret, delete_set, model_name, result_ret", [
        (mock.Mock(), [(error_code.INVALID_DELETE, "No data matched the condition. No data was deleted.")], "test1",
         (error_code.INVALID_DELETE, "No data matched the condition. No data was deleted.")),
        (mock.Mock(), [(mock.Mock(), mock.Mock())], "",
         (error_code.INVALID_TYPE_DATA, "The model name is empty or invalid.")),
        (mock.Mock(), [(mock.Mock(), mock.Mock())], 111,
         (error_code.INVALID_TYPE_DATA, "The model name is empty or invalid.")),
        (mock.Mock(), Exception('xxx'), "test1", (error_code.INVALID_DELETE, "The delete operation failed. xxx")),
        (mock.Mock(), [(error_code.OK, "Delete the data that meets the condition.")], "save_model_to_db_test33",
         (error_code.OK, "Delete the data that meets the condition.")),
    ])
    @mock.patch("base.training_model_management.DataSave")
    def test_delete_model_info_from_db(self, mock_database, database_ret, delete_set, model_name, result_ret):
        mock_database.return_value.__enter__.return_value = database_ret
        mock_database.return_value.__enter__.return_value.delete_with_condition.side_effect = delete_set
        tmm = TrainingModelManagement()
        result = tmm.delete_model_info_from_db(model_name)
        assert result == result_ret

    @pytest.mark.parametrize("input_ret, result_ret", [
        ({"database_ret": mock.Mock(),
          "id_ret": 'a111',
          "check_db_ret": [],
          "signal_length": 132300,
          "model_path": "../../models/cnn_config_001.keras",
          "config_path": "../../config.yml",
          "existing_paths": ["../../models/cnn_config_001.keras", "../../config.yml"],
          "model_output_shape": (None, 1),
          "ret_str": json.dumps({"result": ['accuracy: 0.7', 'cm_info']}),
          "model_description": "No description",
          },
         (error_code.OK,
          ('a111', 'cnn_config_001', '../../models/cnn_config_001.keras',
           '../../config.yml', '132300 x 1', 1, 0.7, '2024-10-12 10:53:41', 'No description'))
         ),
        ({"database_ret": mock.Mock(),
          "id_ret": 'a111',
          "check_db_ret": [],
          "signal_length": 132300,
          "model_path": "../../models/0000001.keras",
          "config_path": "../../config.yml",
          "existing_paths": ["../../config.yml"],
          "model_output_shape": (None, 1),
          "ret_str": {"result": ['accuracy: 0.7', 'cm_info']},
          "model_description": "No description",
          },
         (error_code.INVALID_PATH, "The model path does not exist."),
         ),
        ({"database_ret": mock.Mock(),
          "id_ret": 'a111',
          "check_db_ret": [],
          "signal_length": 132300,
          "model_path": "../../models/cnn_config_001.keras",
          "config_path": "../../0001.yml",
          "existing_paths": ["../../models/cnn_config_001.keras"],
          "model_output_shape": (None, 1),
          "ret_str": json.dumps({"result": ['accuracy: 0.7', 'cm_info']}),
          "model_description": "No description",
          },
         (error_code.INVALID_PATH, "The config path does not exist."),
         ),
        ({"database_ret": mock.Mock(),
          "id_ret": 'a111',
          "check_db_ret": [('5da8a1fb-886f-11ef-a13b-107c610bb999')],
          "signal_length": 132300,
          "model_path": model_consts.DEFAULT_DIR + "models/001.keras",
          "config_path": model_consts.DEFAULT_DIR + "config.yml",
          "existing_paths": [model_consts.DEFAULT_DIR + "models/001.keras", model_consts.DEFAULT_DIR + "config.yml"],
          "model_output_shape": (None, 1),
          "ret_str": json.dumps({"result": ['accuracy: 0.526', 'cm_info']}),
          "model_description": "No description",
          },
         (error_code.INVALID_INSERT, "The model info existed."),
         ),
    ])
    @mock.patch("base.training_model_management.datetime")
    @mock.patch("base.training_model_management.load_model")
    @mock.patch("base.training_model_management.os.path.exists")
    @mock.patch("base.training_model_management.uuid.uuid1")
    def test_get_training_model_info_to_db(self, mock_id, mock_exists, mock_load_model, mock_datetime, input_ret,
                                           result_ret):
        input_ret["database_ret"].query_matching_data.return_value = input_ret["check_db_ret"]
        mock_id.return_value = input_ret["id_ret"]
        mock_exists.side_effect = lambda path: path in input_ret["existing_paths"]
        mock_load_model.return_value.output_shape = input_ret["model_output_shape"]
        mock_datetime.now.return_value.strftime.return_value = "2024-10-12 10:53:41"
        tmm = TrainingModelManagement()
        result = tmm.get_training_model_info_to_db(input_ret["database_ret"], input_ret["signal_length"],
                                                   input_ret["model_path"],
                                                   input_ret["config_path"], input_ret["ret_str"],
                                                   input_ret["model_description"])
        assert result == result_ret

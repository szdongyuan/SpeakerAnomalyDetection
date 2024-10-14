import json
import mock
import pytest
from freezegun import freeze_time

from base.training_model_management import TrainingModelManagement
from consts import error_code, model_consts


class TestTrainingModelManagement(object):
    @pytest.mark.parametrize("input_ret, result_ret", [
        ({"database_ret": mock.Mock(),
          "model_info_set": [(mock.Mock(), mock.Mock())],
          "model_path": "../../models/cnn_config_001.keras",
          "config_path": "../../config.yml",
          "ret_str": json.dumps({"result": ['accuracy: 0.7', 'cm_info']}),
          "model_description": "No description",
          },
         (error_code.OK, "Successfully saved the training model info to the database."),
         ),
        ({"database_ret": mock.Mock(),
          "model_info_set": [(mock.Mock(), mock.Mock())],
          "model_path": "../../models/cnn_config_001.keras",
          "config_path": "../../config.yml",
          "ret_str": None,
          "model_description": None,
          },
         (error_code.OK, "Successfully saved the training model info to the database."),
         ),
        ({"database_ret": Exception(),
          "model_info_set": [(mock.Mock(), mock.Mock())],
          "model_path": "../../models/cnn_config_001.keras",
          "config_path": "../../config.yml",
          "ret_str": None,
          "model_description": None,
          },
         (error_code.INVALID_INSERT, "Failed to save the training model info to the database. "
                                     "'Exception' object has no attribute 'insert_audio_files_info'"),
         ),
    ])
    @mock.patch("base.training_model_management.TrainingModelManagement.get_training_model_info_to_db")
    @mock.patch("base.training_model_management.DataSave")
    def test_save_training_model_info_to_db(self, mock_database, mock_model_info, input_ret, result_ret):
        mock_database.return_value.__enter__.return_value = input_ret["database_ret"]
        mock_model_info.side_effect = input_ret["model_info_set"]
        tmm = TrainingModelManagement()
        result = tmm.save_training_model_info_to_db(input_ret["model_path"], input_ret["config_path"],
                                                    input_ret["ret_str"], input_ret["model_description"])
        assert result == result_ret

    @pytest.mark.parametrize("input_ret, result_ret", [
        ({"database_ret": mock.Mock(),
          "id_ret": 'a111',
          "check_db_ret": [],
          "model_path": "../../models/cnn_config_001.keras",
          "config_path": "../../config.yml",
          "ret_str": json.dumps({"result": ['accuracy: 0.7', 'cm_info']}),
          "model_description": "No description",
          },
         (('a111', 'cnn_config_001', '../../models/cnn_config_001.keras',
           '../../config.yml', '132300 x 1', 1, 0.7, '2024-10-12 10:53:41',
           'No description'),
          "The training model information has been obtained.")
         ),
        ({"database_ret": mock.Mock(),
          "id_ret": 'a111',
          "check_db_ret": [],
          "model_path": "../../models/0000001.keras",
          "config_path": "../../config.yml",
          "ret_str": {"result": ['accuracy: 0.7', 'cm_info']},
          "model_description": "No description",
          },
         (error_code.INVALID_PATH, "The model path does not exist."),
         ),
        ({"database_ret": mock.Mock(),
          "id_ret": 'a111',
          "check_db_ret": [],
          "model_path": "../../models/cnn_config_001.keras",
          "config_path": "../../0001.yml",
          "ret_str": json.dumps({"result": ['accuracy: 0.7', 'cm_info']}),
          "model_description": "No description",
          },
         (error_code.INVALID_PATH, "The config path does not exist."),
         ),
        ({"database_ret": mock.Mock(),
          "id_ret": 'a111',
          "check_db_ret": [('5da8a1fb-886f-11ef-a13b-107c610bb999')],
          "model_path": model_consts.DEFAULT_DIR + "models/001.keras",
          "config_path": model_consts.DEFAULT_DIR + "config.yml",
          "ret_str": json.dumps({"result": ['accuracy: 0.526', 'cm_info']}),
          "model_description": "No description",
          },
         (error_code.INVALID_INSERT, "The model info existed."),
         ),
    ])
    @freeze_time("2024-10-12 10:53:41")
    @mock.patch("base.training_model_management.uuid.uuid1")
    @mock.patch("base.training_model_management.DataSave")
    def test_get_training_model_info_to_db(self, mock_database, mock_id, input_ret, result_ret):
        mock_database.return_value.__enter__.return_value = input_ret["database_ret"]
        mock_database.return_value.__enter__.return_value.check_database_info_equal.return_value = input_ret[
            "check_db_ret"]
        mock_id.return_value = input_ret["id_ret"]
        tmm = TrainingModelManagement()
        result = tmm.get_training_model_info_to_db(input_ret["database_ret"], input_ret["model_path"],
                                                   input_ret["config_path"], input_ret["ret_str"],
                                                   input_ret["model_description"])
        assert result == result_ret

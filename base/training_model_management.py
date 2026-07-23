import json
import os
import uuid
from datetime import datetime

from keras.models import load_model

from base.db_manager import DataSave, ensure_audio_database_ready
from base.file_ops import FileOps
from consts import model_consts, error_code


class TrainingModelManagement(object):
    def __init__(self):
        self._db_path_override = None
        ensure_audio_database_ready()

    @property
    def db_path(self):
        if self._db_path_override:
            return self._db_path_override
        ensure_audio_database_ready()
        return model_consts.AUDIO_DATABASE_PATH

    @db_path.setter
    def db_path(self, value):
        self._db_path_override = value

    def save_training_model_info_to_db(self, signal_length, model_path, config_path, ret_str=None,
                                       model_description="No description"):
        try:
            if os.path.isabs(model_path):
                model_path = FileOps.get_relative_path(model_path, model_consts.DEFAULT_DIR)
            with DataSave(self.db_path) as database:
                code, training_model_info = self.get_training_model_info_to_db(database, signal_length, model_path,
                                                                               config_path, ret_str, model_description)
                if code == error_code.OK:
                    database.insert_data_into_db("training_model_table",
                                                 model_consts.DB_MODEL_COLUMNS, [training_model_info])
                    return error_code.OK, "Successfully saved the training model info to the database."
                else:
                    return code, training_model_info
        except Exception as e:
            err_msg = "Failed to save the training model info to the database. %s" % (str(e)[:70])
            return error_code.INVALID_INSERT, err_msg

    def register_new_model_info_to_db(self,
                                      model_name: str = None,
                                      config_path: str = model_consts.CONFIG_PATH,
                                      input_dim: str = None,
                                      output_dim: str = None,
                                      model_description="No description",
                                      model_type: str = "keras",
                                      model_path: str = None):
        if not (model_name and input_dim and output_dim and config_path and model_path):
            return error_code.INVALID_DATA_LOADING, "The model info is empty."

        try:
            with DataSave(self.db_path) as database:
                model_id = str(uuid.uuid1())
                update_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                model_info = [model_id, model_name, model_path, config_path, input_dim, output_dim, None, update_date,
                              model_description]
                code, msg = database.insert_data_into_db("training_model_table",
                                                         model_consts.DB_MODEL_COLUMNS,
                                                         [model_info])
                if code != error_code.OK:
                    return code, msg

                return error_code.OK, "Successfully saved the model info to the database."
        except Exception as e:
            err_msg = "Failed to save the model info to the database. %s" % (str(e)[:70])
            return error_code.INVALID_INSERT, err_msg

    def update_model_info_to_db(self, model_data: dict):
        try:
            with DataSave(self.db_path) as database:
                result = database.query_matching_data([(model_data.get("old_name"),)], "training_model_table",
                                                      ["model_name"],
                                                      ['model_id'])
                if result:
                    if model_data.get("model_name", False):
                        base_name = os.path.basename(model_data["model_path"])
                        model_path = "models/" + base_name.replace(model_data["old_name"], model_data["model_name"])
                        update_data = {"model_name": model_data["model_name"],
                                       "model_path": model_path}
                        condition_field = {"model_name": model_data["old_name"],
                                           "model_path": model_data["model_path"]}
                    else:
                        update_data = {"model_name": model_data["old_name"],
                                       "model_description": model_data["model_description"]}
                        condition_field = {"model_name": model_data["old_name"],
                                           "model_description": model_data["old_description"]}
                    database.update_table_data("training_model_table", update_data, condition_field)
                    return error_code.OK, "Successfully update the model info to the database."
                else:
                    return error_code.INVALID_UPDATE, "The model name does not exist."
        except Exception as e:
            err_msg = "Failed to update the model info to the database. %s" % (str(e)[:70])
            return error_code.INVALID_INSERT, err_msg

    def delete_model_info_from_db(self, model_name: str):
        if not model_name or not isinstance(model_name, str):
            return error_code.INVALID_TYPE_DATA, "The model name is empty or invalid."
        delete_condition = {"model_name": model_name}
        try:
            with DataSave(self.db_path) as database:
                delete_code, msg = database.delete_with_condition("training_model_table", delete_condition)
                return delete_code, msg
        except Exception as e:
            err_msg = "The delete operation failed. %s" % (str(e)[:40])
            return error_code.INVALID_DELETE, err_msg

    @staticmethod
    def get_training_model_info_to_db(database, signal_length, model_path, config_path, ret_str=None,
                                      model_description="No description"):
        if not os.path.exists(model_path):
            return error_code.INVALID_PATH, "The model path does not exist."
        if not os.path.exists(config_path):
            return error_code.INVALID_PATH, "The config path does not exist."
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        training_model = load_model(model_path)
        input_dim = f"{signal_length} x {1}"
        output_shape = training_model.output_shape
        if len(output_shape) >= 2:
            output_dim = output_shape[1]
        else:
            output_dim = "Unknown"
        accuracy = float(json.loads(ret_str)["result"][0].split(':')[1]) if ret_str else None
        temp_data = (model_name, model_path, config_path, input_dim, output_dim, accuracy)
        result = database.query_matching_data([temp_data], "training_model_table", model_consts.MODEL_COLUMNS,
                                              ['model_id'])
        if result:
            return error_code.INVALID_INSERT, "The model info existed."
        else:
            model_id = str(uuid.uuid1())
            update_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            training_model_data = (model_id,) + temp_data + (update_date, model_description)
            return error_code.OK, training_model_data

    def get_model_path_from_db(self, model_name):
        try:
            with DataSave(self.db_path) as database:
                query_code, query_result = database.query("training_model_table",
                                                          ["model_path", "config_path"],
                                                          {"model_name": model_name})

                if query_code == error_code.OK and query_result:
                    return error_code.OK, query_result
                else:
                    return error_code.INVALID_QUERY, "Failed to query the model's path."
        except Exception as e:
            err_msg = "Failed to query the model path. %s" % (str(e)[:40])
            return error_code.INVALID_QUERY, err_msg

    def get_all_model_name_from_db(self):
        try:
            with DataSave(self.db_path) as database:
                query_code, query_result = database.query("training_model_table", ["model_name", "input_dim"])
                if query_code == error_code.OK and query_result:
                    return error_code.OK, query_result
                else:
                    return error_code.INVALID_QUERY, "Failed to query all mdoel name."
        except Exception as e:
            err_msg = "Failed to query the all model name. %s" % (str(e)[:40])
            return error_code.INVALID_QUERY, err_msg

    def get_all_model_info_from_db(self):
        try:
            with DataSave(self.db_path) as database:
                query_code, query_result = database.query("training_model_table",
                                                          ["model_name", "input_dim", "output_dim", "accuracy",
                                                           "model_description", "config_path", "model_path"])
                if query_code == error_code.OK and query_result:
                    return error_code.OK, query_result
                else:
                    return error_code.INVALID_QUERY, "Failed to query all mdoel name."
        except Exception as e:
            err_msg = "Failed to query the all model name. %s" % (str(e)[:40])
            return error_code.INVALID_QUERY, err_msg

    def get_input_dim_info_by_name(self, model_name: str):
        if not model_name or not isinstance(model_name, str):
            return error_code.INVALID_QUERY, "The model name is empty or invalid."
        try:
            with DataSave(self.db_path) as database:
                query_code, query_result = database.query(
                    "training_model_table",
                    ["input_dim"],
                    {"model_name": model_name},
                )

                if query_code == error_code.OK and query_result:
                    input_dim = query_result[0][0]
                    return error_code.OK, input_dim
                else:
                    return error_code.INVALID_QUERY, "Failed to query input_dim by model name."
        except Exception as e:
            err_msg = "Failed to query input_dim by model name. %s" % (str(e)[:40])
            return error_code.INVALID_QUERY, err_msg

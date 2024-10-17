import json
import os
import uuid
from datetime import datetime

from keras.models import load_model

from base.db_manager import DataSave
from consts import model_consts, error_code


class TrainingModelManagement(object):
    def __init__(self):
        self.db_path = model_consts.DATABASE_PATH

    def save_training_model_info_to_db(self, model_path, config_path, ret_str=None, model_description="No description"):
        try:
            with DataSave(self.db_path) as database:
                training_model_data, _ = self.get_training_model_info_to_db(database, model_path, config_path, ret_str,
                                                                            model_description)
                database.insert_audio_files_info("training_model_table",
                                                 model_consts.DB_MODEL_COLUMNS, [training_model_data])
            return error_code.OK, "Successfully saved the training model info to the database."
        except Exception as e:
            err_msg = "Failed to save the training model info to the database. %s" % (str(e)[:40])
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
    def get_training_model_info_to_db(database, model_path, config_path, ret_str=None,
                                      model_description="No description"):
        if not os.path.exists(model_path):
            return error_code.INVALID_PATH, "The model path does not exist."
        if not os.path.exists(config_path):
            return error_code.INVALID_PATH, "The config path does not exist."
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        training_model = load_model(model_path)
        input_dim = f"{training_model.input_shape[1]} x {training_model.input_shape[2]}"
        output_dim = training_model.output_shape[1]
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
        return training_model_data, "The training model information has been obtained."

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
        """
            Saves the training model information into the database.

            This method retrieves the training model information based on the provided model path and configuration file
            , then inserts it into the 'training_model_table' of the database.

            Args:
                - model_path: str
                    The path to the saved model file.
                - config_path: str
                    The path to the config file related to the model.
                - ret_str: str, optional
                    Used to read accuracy from it (default is None).
                - model_description:  str, optional
                    A description of the model (default is "No description").

            Returns:
                - tuple: A tuple containing an error code and a message.

        """
        try:
            with DataSave(self.db_path) as database:
                code, training_model_info = self.get_training_model_info_to_db(database, model_path, config_path,
                                                                               ret_str,
                                                                               model_description)
                if code == error_code.OK:
                    database.insert_data_into_db("training_model_table",
                                                 model_consts.DB_MODEL_COLUMNS, [training_model_info])
                    return error_code.OK, "Successfully saved the training model info to the database."
                else:
                    return code, training_model_info
        except Exception as e:
            err_msg = "Failed to save the training model info to the database. %s" % (str(e)[:70])
            return error_code.INVALID_INSERT, err_msg

    def delete_model_info_from_db(self, model_name: str):
        """
            Deletes model information from the database based on the provided model name.

            Args:
                - model_name: str
                    The name of the model whose information is to be deleted.

            Returns:
                - tuple: A tuple containing an error code and a message.

        """
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
        """
            This function checks if the model and config paths exist, extracts model information,
            and inserts it into the database if the model doesn't already exist in database.

            Args:
                - database: Database object
                    The database instance used to query and insert model data.
                - model_path: str
                    The file path to the model file.
                - config_path: str
                    The file path to the model config file.
                - ret_str: str, optional
                    A JSON string containing the training result, used to extract accuracy.
                - model_description: str, optional
                    A description of the model. Default is "No description".

            Returns:
                - tuple: A tuple containing an error code and a message or training model data.

        """
        if not os.path.exists(model_path):
            return error_code.INVALID_PATH, "The model path does not exist."
        if not os.path.exists(config_path):
            return error_code.INVALID_PATH, "The config path does not exist."
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        training_model = load_model(model_path)
        input_shape = training_model.input_shape
        if len(input_shape) >= 3:
            input_dim = f"{input_shape[1]} x {input_shape[2]}"
        elif len(input_shape) == 2:
            input_dim = f"{input_shape[1]}"
        else:
            input_dim = "Unknown"
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
        """
            This function retrieves the model's path and config path from the database
            based on the model name.

            Args:
                - model_name: str
                    The name of the model for which to retrieve the paths.

            Returns:
                - tuple: A tuple containing an error code and a message.

        """
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
        """
            This function retrieves all model names and input dimensions from the database.

            Returns:
                - tuple: A tuple containing an error code and a message.

        """
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

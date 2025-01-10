import os
import shutil

import numpy as np

from base.db_manager import DataSave
from base.file_ops import FileOps
from base.load_config import load_config
from consts import error_code, model_consts


def split_train_test(ratio=0.9,
                     train_ok_path=model_consts.TRAIN_OK_PATH,
                     train_ng_path=model_consts.TRAIN_NG_PATH,
                     test_ok_path=model_consts.TEST_OK_PATH,
                     test_ng_path=model_consts.TEST_NG_PATH):
    """
        The dataset is divided into the training set and the testing set according to the given ratio.

        Args:
        - ratio: float
            The ratio between the testing set and the training set.
        - train_ok_path: string
            The directory path that contains the "OK" training file.
        - train_ng_path: string
            The directory path that contains the "NG" training file.
        - test_ok_path: string
            The directory path that contains the "OK" testing file.
        - test_ng_path: string
            The directory path that contains the "NG" testing file.
    """
    restore_split(train_ok_path, train_ng_path,
                  test_ok_path, test_ng_path)
    for file in os.listdir(train_ok_path):
        if np.random.random() > ratio:
            dir_file = train_ok_path + "/" + file
            shutil.move(dir_file, test_ok_path)
    for file in os.listdir(train_ng_path):
        if np.random.random() > ratio:
            dir_file = train_ng_path + "/" + file
            shutil.move(dir_file, test_ng_path)
    print("finish splitting")


def restore_split(train_ok_path=model_consts.TRAIN_OK_PATH,
                  train_ng_path=model_consts.TRAIN_NG_PATH,
                  test_ok_path=model_consts.TEST_OK_PATH,
                  test_ng_path=model_consts.TEST_NG_PATH):
    """
        Restore the test file back to the training set.

        Args:
        - train_ok_path: string
            The directory path that contains the "OK" training file.
        - train_ng_path: string
            The directory path that contains the "NG" training file.
        - test_ok_path: string
            The directory path that contains the "OK" testing file.
        - test_ng_path: string
            The directory path that contains the "NG" testing file.
    """
    for file in os.listdir(test_ok_path):
        dir_file = test_ok_path + "/" + file
        shutil.move(dir_file, train_ok_path)
    for file in os.listdir(test_ng_path):
        dir_file = test_ng_path + "/" + file
        shutil.move(dir_file, train_ng_path)
    print("finish restore")


def copy_from_restored_audio(source_dir_list,
                             dest_dir=model_consts.TRAIN_PATH,
                             over_write=True):
    """
        Copy audio files from the source directories to the destination directory.

        Args:
        - source_dir_list: list
            List of source directories from which to copy files.
        - dest_dir: string
            The destination directory path of the file to be copied.
        - over_write: bool
            Whether to overwrite existing files.
        Returns:
        - error_code.OK: int
            The code indicating a successful operation.
    """
    if over_write:
        ret_code, ret_msg = FileOps().create_empty_okng(dest_dir)
        if ret_code != error_code.OK:
            print(ret_msg)
            return ret_code

    n_file = 0
    for source_dir in source_dir_list:
        source_dir = model_consts.STORED_SAMPLE_PATH + "/" + source_dir
        for audio_file in os.listdir(source_dir + "/OK"):
            shutil.copy(source_dir + "/OK/" + audio_file, dest_dir + "/OK")
            n_file += 1
        for audio_file in os.listdir(source_dir + "/NG"):
            shutil.copy(source_dir + "/NG/" + audio_file, dest_dir + "/NG")
            n_file += 1
    print("finish copy from restored audio. [%s] files" % n_file)
    return error_code.OK


def copy_from_restored_audio_database(dest_train_dir=model_consts.TRAIN_PATH,
                                      dest_test_dir=model_consts.TEST_PATH,
                                      config_path=model_consts.CONFIG_PATH, over_write=True):
    """
        Based on the config file Settings, query the required audio files to the database
        and copy them to the specified training and testing directory.

        Args:
        - dest_train_dir: str, optional
            The path to the folder where the training data is stored.
            The default values are taken from model_consts.TRAIN_PATH.
        - dest_test_dir: str
            The path to the folder where the test data is stored.
            The default values are taken from model_consts.TEST_PATH.
        - over_write: bool
            Identifies whether to overwrite the target directory. Default to True
        Returns:
        - error_code.OK: int
            tuple: A tuple containing an error code and a message.
    """

    try:
        data_load_config = load_config(config_path, module_name="data_load")
        code, result = retrieve_data_from_db(data_load_config)
        if code == error_code.OK and result:
            query_data = result
        else:
            return error_code.MISSING_SELECT_DATA, "No data was queried."
        if over_write:
            for dest_dir in [dest_train_dir, dest_test_dir]:
                ret_code, ret_msg = FileOps().create_empty_okng(dest_dir)
                if ret_code != error_code.OK:
                    return ret_code, ret_msg
        recode_date_info = data_load_config.get("record_date")
        n_file = 0
        train_data_date = []
        if recode_date_info and 'train_data_date' in recode_date_info:
            train_data_date = [str(item) for item in recode_date_info.get('train_data_date')]
        for file in query_data:
            if recode_date_info:
                dest_dir = dest_train_dir if file[2] in train_data_date else dest_test_dir
            else:
                dest_dir = dest_train_dir
            status_dir = "OK" if file[-1] == "OK" else "NG"
            shutil.copy(f"{file[0]}", f"{dest_dir}/{status_dir}")
            n_file += 1
        return error_code.OK, f"finish copy from restored {n_file} audio files."
    except Exception as e:
        err_msg = "Failed to copy the audio file. %s" % (str(e))
        return error_code.INVALID_QUERY, err_msg


def retrieve_data_from_db(data_load_config):
    """
        Retrieve data from the database.

        Args:
        - data_load_config: dict
                The path to the folder where the training data is stored.
                The default values are taken from model_consts.TRAIN_PATH.

        Returns:
            tuple: A tuple containing an error code and a message.

    """
    with DataSave(model_consts.DATABASE_PATH) as database:
        code, query_clause_data = generate_query_conditions(data_load_config)
        if code == error_code.OK and query_clause_data:
            query_code, query_result = database.query(table_name="audio_data_table",
                                                      query_column=model_consts.SELECT_COLUMNS,
                                                      query_clause_data=query_clause_data,
                                                      FK_related=True)
            if query_code == error_code.OK and query_result:
                return error_code.OK, query_result
            else:
                return error_code.INVALID_QUERY, "No data was queried."
        else:
            return error_code.INVALID_QUERY, "Failed to retrieve query conditions."


def generate_query_conditions(data_load_config):
    """
        Generate query condition data based on the given data loading configuration.

        Args:
        - data_load_config: dict
            The path to the folder where the training data is stored.
            The default values are taken from model_consts.TRAIN_PATH.

        Returns:
            tuple: A tuple containing an error code and a message or .

    """
    try:
        condition_mapping = DataSave(model_consts.DATABASE_PATH).get_data_config(data_load_config)
        query_clause_data = {}
        record_date_mapping = condition_mapping.get("record_date")
        if record_date_mapping is not None:
            query_clause_data["record_date"] = []
            for key, data_date_list in record_date_mapping.items():
                data_date_list = [] if not data_date_list else data_date_list
                query_clause_data["record_date"] += data_date_list
        for key, value in condition_mapping.items():
            if key != "record_date" and value is not None:
                query_clause_data[key] = value
        return error_code.OK, query_clause_data
    except Exception as e:
        err_msg = f"Failed to query data: {str(e)}"
        return error_code.INVALID_QUERY, err_msg

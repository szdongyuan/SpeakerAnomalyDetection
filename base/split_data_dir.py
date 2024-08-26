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


def copy_from_restored_audio_database(dest_train_dir=model_consts.TRAIN_PATH, dest_test_dir=model_consts.TEST_PATH, over_write=True):
    data_load_config = load_config("data_load")
    query_data = DataSave(model_consts.DATABASE_PATH).query_conditions()
    for dest_dir in [dest_train_dir, dest_test_dir]:
        ret_code, ret_msg = FileOps().create_empty_okng(dest_dir)
        if ret_code != error_code.OK:
            print(ret_msg)
            return ret_code
    file_list = []
    n_file = 0
    stored_sample_path = model_consts.STORED_SAMPLE_PATH
    if not query_data:
        return error_code.MISSING_SELECT_DATA, "No data was queried."
    for data_item in query_data:
        file_list.append((data_item[0], data_item[2], data_item[6]))
    recode_date_info = data_load_config.get("recode_date")
    train_data_date = [str(item) for item in recode_date_info.get('train_data_date')]
    for file in file_list:
        dest_dir = dest_train_dir if file[1] in train_data_date else dest_test_dir
        status_dir = "OK" if file[2] == "OK" else "NG"
        shutil.copy(f"{stored_sample_path}/{file[0]}", f"{dest_dir}/{status_dir}")
        n_file += 1
    print("finish copy from restored audio. [%s] files" % n_file)
    return error_code.OK

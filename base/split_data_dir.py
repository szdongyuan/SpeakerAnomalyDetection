import os
import shutil

import numpy as np

from consts.model_consts import TRAIN_OK_PATH, TRAIN_NG_PATH, TEST_OK_PATH, TEST_NG_PATH


def split_train_test(ratio=0.9,
                     train_ok_path=TRAIN_OK_PATH,
                     train_ng_path=TRAIN_NG_PATH,
                     test_ok_path=TEST_OK_PATH,
                     test_ng_path=TEST_NG_PATH):
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


def restore_split(train_ok_path=TRAIN_OK_PATH,
                  train_ng_path=TRAIN_NG_PATH,
                  test_ok_path=TEST_OK_PATH,
                  test_ng_path=TEST_NG_PATH):
    for file in os.listdir(test_ok_path):
        dir_file = test_ok_path + "/" + file
        shutil.move(dir_file, train_ok_path)
    for file in os.listdir(test_ng_path):
        dir_file = test_ng_path + "/" + file
        shutil.move(dir_file, train_ng_path)
    print("finish restore")

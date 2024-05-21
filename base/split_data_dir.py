import os
import shutil

import numpy as np

import consts.model_consts as model_consts

def split_train_test(ratio=0.9,
                     train_ok_path=model_consts.TRAIN_OK_PATH,
                     train_ng_path=model_consts.TRAIN_NG_PATH,
                     test_ok_path=model_consts.TEST_OK_PATH,
                     test_ng_path=model_consts.TEST_NG_PATH):
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
    for file in os.listdir(test_ok_path):
        dir_file = test_ok_path + "/" + file
        shutil.move(dir_file, train_ok_path)
    for file in os.listdir(test_ng_path):
        dir_file = test_ng_path + "/" + file
        shutil.move(dir_file, train_ng_path)
    print("finish restore")


def copy_from_restored_audio(source_dir_list, dest_dir=model_consts.TRAIN_PATH):
    shutil.rmtree(dest_dir)
    os.mkdir(dest_dir)
    os.mkdir(dest_dir + "/OK")
    os.mkdir(dest_dir + "/NG")
    for source_dir in source_dir_list:
        source_dir = model_consts.STORED_SAMPLE_PATH + "/" + source_dir
        for audio_file in os.listdir(source_dir + "/OK"):
            shutil.copy(source_dir + "/OK/" + audio_file, dest_dir + "/OK")
        for audio_file in os.listdir(source_dir + "/NG"):
            shutil.copy(source_dir + "/NG/" + audio_file, dest_dir + "/NG")
    print("finish copy from restored audio")

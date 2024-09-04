import argparse
import json
import os
import time

import numpy as np

from base.display import DisplayManager
from base.load_audio import get_pre_labeled_audios, get_audio_files_and_labels
from base.load_config import load_config
from base.pre_processing.preprocessing_manager import PreprocessingManager
from base.split_data_dir import copy_from_restored_audio_database
from consts import error_code, model_consts
from consts.model_consts import MODEL_MAPPING

DEFAULT_DATA_PATH = "audio_data/train"
DEFAULT_TEST_DATA = "audio_data/test"
DEFAULT_MODEL_PATH = "models/"


def train(pre_labeled_dir,
          save_model_path=None,
          predict_dir=None):
    time_0 = time.time()

    data_load_config = load_config("data_load")
    ret_code, ret = get_pre_labeled_audios(pre_labeled_dir, **data_load_config)
    if ret_code != error_code.OK:
        return json.dumps({"ret_code": ret_code,
                           "ret_msg": ret,
                           "result": ret})
    signals, file_names, fs, labels = ret
    print("finish audio loading")

    preprocess_config = load_config("preprocess")
    x_train = preprocess_raw_signals(signals, fs, preprocess_config)
    y_train = labels
    print("finish data preparing, data shape %s" % str(x_train.shape))

    model = init_model_from_config()
    if save_model_path and os.path.isfile(save_model_path):
        print("model [%s] exists, keep training" % save_model_path)
        model.load_model(save_model_path)
    else:
        print("init new model [%s]..." % save_model_path)
    model.fit(x_train, y_train)
    ret_msg = "finish training. time spent [%s] s" % (time.time() - time_0)
    print(ret_msg)

    if save_model_path:
        model.save_model(save_model_path)
        ret_msg += ". model saved."
    if predict_dir:
        evaluate(predict_dir, model=model, verbose=1)
    return json.dumps({"ret_code": error_code.OK,
                       "ret_msg": ret_msg,
                       "result": ret_msg})


def evaluate(predict_dir, load_model_path=None, model=None, **kwargs):
    ret_code, ret = get_pre_labeled_audios(predict_dir)
    if ret_code != error_code.OK:
        return json.dumps({"ret_code": ret_code,
                           "ret_msg": ret,
                           "result": ret})
    signals, file_names, fs, labels = ret

    preprocess_config = load_config("preprocess")
    x_test = preprocess_raw_signals(signals, fs, preprocess_config)
    y_test = labels

    if load_model_path:
        model = init_model_from_config()
        model.load_model(load_model_path)
    if not model:
        return json.dumps({"ret_code": error_code.MISSING_MODEL,
                           "ret_msg": "missing model",
                           "result": "missing model"})

    y_pred, pred_score = model.predict(x_test)

    len_test = len(y_test)
    acc = np.sum(y_pred == y_test) / len_test
    acc_info = "accuracy: %s" % round(acc, 3)
    dm = DisplayManager()
    display_cm = dm.display_confusion_matrix(y_test, y_pred)
    cm_info = "Confusion Matrix: \n%s" % display_cm

    verbose = kwargs.get("verbose", 0)
    if verbose >= 1:
        print("number of test cases: %s" % len_test)
        print(acc_info)
        print(cm_info)
        false_prediction = [file_names[i] for i in range(len_test) if y_test[i] != y_pred[i]]
        print("false prediction %s" % false_prediction)
        if verbose == 2:
            dm.display_pred_score(file_names, labels, pred_score)
        elif verbose == 3:
            dm.display_pred_score(file_names, labels, pred_score, to_csv=True)

    model_detail = kwargs.get("model_detail", False)
    if model_detail:
        print(model.model.summary())

    ret_str = json.dumps({"ret_code": error_code.OK,
                          "ret_msg": "finish evaluating",
                          "result": [acc_info, cm_info]})
    return ret_str


def predict(predict_dir, load_model_path=None, model=None, **kwargs):
    ret_code, ret = get_audio_files_and_labels(predict_dir)
    if ret_code != error_code.OK:
        return json.dumps({"ret_code": ret_code,
                           "ret_msg": ret,
                           "result": [[ret]]})
    signals, file_names, fs, _ = ret
    file_len = len(file_names)

    preprocess_config = load_config("preprocess")
    x_test = preprocess_raw_signals(signals, fs, preprocess_config)
    if load_model_path:
        model = init_model_from_config()
        model.load_model(load_model_path)
    if not model:
        return json.dumps({"ret_code": error_code.MISSING_MODEL,
                           "ret_msg": "missing model",
                           "result": [["missing model"]]})

    y_pred, pred_score = model.predict(x_test)
    result = [[file_names[i], "OK" if y_pred[i] else "NG", str(pred_score[i])] for i in range(file_len)]
    ret_str = json.dumps({"ret_code": error_code.OK,
                          "ret_msg": "finish predicting",
                          "result": result})
    return ret_str


def load_data_from_database():
    try:
        return copy_from_restored_audio_database(dest_train_dir=model_consts.TRAIN_PATH,
                                                 dest_test_dir=model_consts.TEST_PATH)
    except Exception as e:
        err_msg = "Failed to load data from the database. %s" % (str(e))
        return error_code.INVALID_DATA_LOADING, err_msg


def init_model_from_config():
    """
        Initialize the model based on configuration.

        Returns:
            Instantiate a model class based on the configuration.
    """
    model_config = load_config("model")
    model_obj = MODEL_MAPPING.get(model_config.get("model_name"))
    model = model_obj(model_config)
    return model


def preprocess_raw_signals(raw_signals, fs, preprocess_config):
    """
        Preprocess the original audio signal data.

        Args:
        - raw_signals: list
            List of the original audio data.
        - fs: list
            List of sampling rates for the original audio data.
        - preprocess_config:
            Loaded data preprocessing configuration.

        Returns:
            An array containing preprocessed audio signal data.
    """
    processed_data = []
    pm = PreprocessingManager()
    for i in range(len(raw_signals)):
        processed_data.append(pm.process(raw_signals[i], fs[i], **preprocess_config))
    return np.array(processed_data)


parser = argparse.ArgumentParser(description='speaker anomaly detection')
subparsers = parser.add_subparsers(help="sub-command help")
parser.set_defaults(func="None")

parser_train = subparsers.add_parser("train", help="train model")
parser_train.add_argument("-d", "--data",
                          required=True, help="training dataset path")
parser_train.add_argument("-m", "--model",
                          required=True, help="model save path")
parser_train.add_argument("-t", "--test",
                          help="validate dataset path")
parser_train.set_defaults(func="train")

parser_evaluate = subparsers.add_parser("evaluate", help="evaluate model")
parser_evaluate.add_argument("-t", "--test",
                             required=True, help="evaluate dataset path")
parser_evaluate.add_argument("-m", "--model",
                             required=True, help="saved model path")
parser_evaluate.add_argument("-v", "--verbose",
                             help="show detailed evaluate info, 0 ~ 3")
parser_evaluate.set_defaults(func="evaluate")

parser_predict = subparsers.add_parser("predict", help="predict samples")
parser_predict.add_argument("-t", "--test",
                            required=True, help="predict sample dir or file")
parser_predict.add_argument("-m", "--model",
                            required=True, help="saved model path")
parser_predict.add_argument("-v", "--verbose",
                            help="show detailed evaluate info, 0 ~ 3")
parser_predict.set_defaults(func="predict")

args = parser.parse_args()

if __name__ == "__main__":
    if args.func == "train":
        train(args.data, save_model_path=args.model, predict_dir=args.test)
    elif args.func == "evaluate":
        verbose = int(args.verbose) if args.verbose else None
        evaluate(args.test, load_model_path=args.model, verbose=verbose)
    elif args.func == "predict":
        predict(args.test, load_model_path=args.model)
    else:
        print("[%s] not support" % args.func)

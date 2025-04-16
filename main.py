# import argparse
import json
import os
import time

import numpy as np

from base.display import DisplayManager
from base.load_audio import get_pre_labeled_audios, get_audio_files_and_labels
from base.load_config import load_config
from base.log_manager import LogManager
from base.pre_processing.preprocessing_manager import PreprocessingManager
from base.split_data_dir import copy_from_restored_audio_database
from consts import error_code, model_consts
from machine_learning import MODEL_MAPPING


DEFAULT_DATA_PATH = "audio_data/train"
DEFAULT_TEST_DATA = "audio_data/test"
DEFAULT_MODEL_PATH = "models/"


def train(pre_labeled_dir,
          save_model_path=None,
          predict_dir=None,
          **kwargs):
    logger = LogManager("train")

    time_0 = time.time()

    save_config_path = kwargs.get("config_path", model_consts.CONFIG_PATH)
    config_path = model_consts.DEFAULT_DIR + save_config_path
    data_load_config = load_config(config_path=config_path, module_name="data_load")
    ret_code, ret = get_pre_labeled_audios(pre_labeled_dir, **data_load_config)
    if ret_code != error_code.OK:
        logger.error("failed to load audio samples")
        logger.shut_down()
        return json.dumps({"ret_code": ret_code,
                           "ret_msg": ret,
                           "result": ret})
    signals, file_names, fs, labels = ret
    logger.info("finish audio loading")

    preprocess_config = load_config(config_path=config_path, module_name="preprocess")
    x_train = preprocess_raw_signals(signals, fs, preprocess_config)
    y_train = labels
    logger.info("finish data preparing, data shape %s" % str(x_train.shape))

    kwargs["config_path"] = config_path
    model = init_model_from_config(**kwargs)
    if save_model_path and os.path.isfile(save_model_path):
        logger.info("model [%s] exists, keep training" % save_model_path)
        model.load_model(save_model_path)
    else:
        logger.info("init new model [%s]..." % save_model_path)
    model.fit(x_train, y_train)
    ret_msg = "finish training. time spent [%s] s" % (time.time() - time_0)
    logger.info(ret_msg)
    logger.shut_down()
    if predict_dir:
        ret_str = evaluate(predict_dir, model=model, verbose=2)
    else:
        ret_str = None

    if save_model_path:
        signal_length = len(signals[0])
        model.save_model(signal_length, save_model_path, save_config_path, ret_str, model_description="No description")
        ret_msg += ". model saved."

    return json.dumps({"ret_code": error_code.OK,
                       "ret_msg": ret_msg,
                       "result": ret_msg})


def evaluate(predict_dir,
             load_model_path=None,
             model=None,
             **kwargs):
    logger = LogManager("evaluate")

    ret_code, ret = get_pre_labeled_audios(predict_dir)
    if ret_code != error_code.OK:
        logger.error("failed to load audio samples")
        logger.shut_down()
        return json.dumps({"ret_code": ret_code,
                           "ret_msg": ret,
                           "result": ret})
    signals, file_names, fs, labels = ret

    save_config_path = kwargs.get("config_path", model_consts.CONFIG_PATH)
    config_path = model_consts.DEFAULT_DIR + save_config_path
    preprocess_config = load_config(config_path=config_path, module_name="preprocess")
    x_test = preprocess_raw_signals(signals, fs, preprocess_config)
    y_test = labels

    if load_model_path:
        kwargs["config_path"] = config_path
        model = init_model_from_config(**kwargs)
        model.load_model(load_model_path)
    if not model:
        logger.error("missing model")
        logger.shut_down()
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
    false_prediction = [file_names[i] for i in range(len_test) if y_test[i] != y_pred[i]]
    if verbose % 2:
        logger.info("number of test cases: %s" % len_test)
        logger.info(acc_info)
        logger.info(cm_info)
        logger.info("false prediction:\n%s" % false_prediction)
    if (verbose >> 1) % 2:
        print("number of test cases: %s" % len_test)
        print(acc_info)
        print(cm_info)
    if (verbose >> 2) % 2:
        print("false prediction:\n%s" % false_prediction)
    if (verbose >> 3) % 2:
        dm.display_pred_score(file_names, labels, pred_score)
    if (verbose >> 4) % 2:
        dm.display_pred_score(file_names, labels, pred_score, to_csv=True)

    model_detail = kwargs.get("model_detail", False)
    if model_detail:
        logger.info(model.model.summary())

    ret_str = json.dumps({"ret_code": error_code.OK,
                          "ret_msg": "finish evaluating",
                          "result": [acc_info, cm_info]})

    logger.shut_down()
    return ret_str


def predict(predict_dir,
            load_model_path=None,
            model=None,
            **kwargs):
    ret_code, ret = get_audio_files_and_labels(predict_dir)
    if ret_code != error_code.OK:
        return json.dumps({"ret_code": ret_code,
                           "ret_msg": ret,
                           "result": [[ret]]})
    signals, file_names, fs, _ = ret
    file_len = len(file_names)

    config_path = kwargs.get("config_path", model_consts.DEFAULT_DIR + model_consts.CONFIG_PATH)
    preprocess_config = load_config(config_path=config_path, module_name="preprocess")
    x_test = preprocess_raw_signals(signals, fs, preprocess_config)
    if load_model_path:
        model = init_model_from_config(**kwargs)
        model.load_model(load_model_path)
    if not model:
        return json.dumps({"ret_code": error_code.MISSING_MODEL,
                           "ret_msg": "missing model",
                           "result": [["missing model"]]})

    y_pred, pred_score = model.predict(x_test, acc_req=None, verbose=0)
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


def init_model_from_config(**kwargs):
    """
        Initialize the model based on configuration.

        Returns:
            Instantiate a model class based on the configuration.
    """
    config_path = kwargs.get("config_path", model_consts.DEFAULT_DIR + model_consts.CONFIG_PATH)
    model_config = load_config(config_path=config_path, module_name="model")
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


# parser = argparse.ArgumentParser(description='speaker anomaly detection')
# subparsers = parser.add_subparsers(help="sub-command help")
# parser.set_defaults(func="None")
#
# parser_train = subparsers.add_parser("train", help="train model")
# parser_train.add_argument("-d", "--data",
#                           required=True, help="training dataset path")
# parser_train.add_argument("-m", "--model",
#                           required=True, help="model save path")
# parser_train.add_argument("-t", "--test",
#                           help="validate dataset path")
# parser_train.set_defaults(func="train")
#
# parser_evaluate = subparsers.add_parser("evaluate", help="evaluate model")
# parser_evaluate.add_argument("-t", "--test",
#                              required=True, help="evaluate dataset path")
# parser_evaluate.add_argument("-m", "--model",
#                              required=True, help="saved model path")
# parser_evaluate.add_argument("-v", "--verbose",
#                              help="show detailed evaluate info, 0 ~ 3")
# parser_evaluate.set_defaults(func="evaluate")
#
# parser_predict = subparsers.add_parser("predict", help="predict samples")
# parser_predict.add_argument("-t", "--test",
#                             required=True, help="predict sample dir or file")
# parser_predict.add_argument("-m", "--model",
#                             required=True, help="saved model path")
# parser_predict.add_argument("-v", "--verbose",
#                             help="show detailed evaluate info, 0 ~ 3")
# parser_predict.set_defaults(func="predict")
#
# args = parser.parse_args()
#
# if __name__ == "__main__":
#     if args.func == "train":
#         train(args.data, save_model_path=args.model, predict_dir=args.test)
#     elif args.func == "evaluate":
#         verbose = int(args.verbose) if args.verbose else None
#         evaluate(args.test, load_model_path=args.model, verbose=verbose)
#     elif args.func == "predict":
#         predict(args.test, load_model_path=args.model)
#     else:
#         print("[%s] not support" % args.func)

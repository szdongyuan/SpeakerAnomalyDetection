import json

import numpy as np

from base.display import DisplayManager
from base.load_audio import get_pre_labeled_audios
from base.load_config import load_config
from base.log_manager import LogManager
from base.model_config import init_model_from_config, preprocess_raw_signals
from consts import error_code, model_consts


def evaluate(predict_dir, load_model_path=None, model=None, **kwargs):
    logger = LogManager.set_log_handler("evaluate")

    ret_code, ret = get_pre_labeled_audios(predict_dir)
    if ret_code != error_code.OK:
        logger.error("failed to load audio samples")
        return json.dumps({"ret_code": ret_code, "ret_msg": ret, "result": ret})
    signals, file_names, fs, labels = ret

    save_config_path = kwargs.get("config_path", model_consts.CONFIG_PATH)
    config_path = model_consts.DEFAULT_DIR + save_config_path
    preprocess_config = load_config(config_path=config_path, module_name="preprocess")

    if load_model_path:
        kwargs["config_path"] = config_path
        model = init_model_from_config(**kwargs)
        model.load_model(load_model_path)
    if not model:
        logger.error("missing model")
        return json.dumps({"ret_code": error_code.MISSING_MODEL, "ret_msg": "missing model", "result": "missing model"})

    acc_info, cm_info = evaluate_model(model, preprocess_config, signals, file_names, fs, labels, **kwargs)

    model_detail = kwargs.get("model_detail", False)
    if model_detail:
        logger.info(model.model.summary())

    ret_str = json.dumps({"ret_code": error_code.OK, "ret_msg": "finish evaluating", "result": [acc_info, cm_info]})
    return ret_str


def evaluate_model(model, preprocess_config, signals, file_names, fs, labels, **kwargs):
    logger = LogManager.set_log_handler("evaluate")

    x_test = preprocess_raw_signals(signals, fs, preprocess_config)
    y_test = labels
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

    return acc_info, cm_info

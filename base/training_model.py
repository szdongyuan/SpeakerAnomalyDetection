import json
import os
import time

from base.evaluate_model import evaluate
from base.load_audio import get_pre_labeled_audios
from base.load_config import load_config
from base.log_manager import LogManager
from base.model_config import init_model_from_config, preprocess_raw_signals
from consts import error_code, model_consts


def train(pre_labeled_dir, save_model_path=None, predict_dir=None, **kwargs):

    logger = LogManager.set_log_handler("train")

    save_config_path = kwargs.get("config_path", model_consts.CONFIG_PATH)
    config_path = model_consts.DEFAULT_DIR + save_config_path
    data_load_config = load_config(config_path=config_path, module_name="data_load")
    ret_code, ret = get_pre_labeled_audios(pre_labeled_dir, **data_load_config)

    if ret_code != error_code.OK:
        logger.error("failed to load audio samples")
        return json.dumps({"ret_code": ret_code, "ret_msg": ret, "result": ret})
    signals, file_names, fs, labels = ret
    logger.info("finish audio loading")

    train_kwargs = kwargs.copy()
    train_kwargs.pop("config_path", None)
    model, ret_msg = train_model(config_path, save_model_path, signals, fs, labels, **train_kwargs)
    if predict_dir:
        evaluate_kwargs = {"config_path": save_config_path, "verbose": 2}
        ret_str = evaluate(predict_dir, model=model, **evaluate_kwargs)
    else:
        ret_str = None

    if save_model_path:
        signal_length = len(signals[0])
        model.save_model(signal_length, save_model_path, save_config_path, ret_str, model_description="No description")
        ret_msg += ". model saved."

    return json.dumps({"ret_code": error_code.OK, "ret_msg": ret_msg, "result": ret_msg})


def train_model(config_path, save_model_path, signals, fs, labels, **kwargs):
    logger = LogManager.set_log_handler("train")

    time_0 = time.time()

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

    return model, ret_msg

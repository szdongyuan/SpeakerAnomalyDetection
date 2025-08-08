import json

from base.load_audio import get_audio_files_and_labels
from base.load_config import load_config
from base.model_config import init_model_from_config, preprocess_raw_signals
from consts import error_code, model_consts


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

    ret_str = predict_from_audio(signals, file_names, fs, load_model_path=None, model=None, **kwargs)

    return ret_str


def predict_from_audio(signals,
                       file_names,
                       fs,
                       load_model_path=None,
                       model=None,
                       **kwargs):
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
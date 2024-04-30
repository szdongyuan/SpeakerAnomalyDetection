import json

from base.display import DisplayManager

from base.load_audio import get_pre_labeled_audios, get_audio_files_and_labels, pre_process_data
from base.load_config import load_config
from base.pre_processing.sample_balance import balance_sample_number
from consts import error_code
from consts.model_consts import MODEL_LIST


DEFAULT_DATA_PATH = "audio_data/train"
DEFAULT_TEST_DATA = "audio_data/test"
DEFAULT_MODEL_PATH = "models/"


def train(pre_labeled_dir,
          save_model_path=None,
          predict_dir=None,
          **kwargs):
    ret_code, ret = get_pre_labeled_audios(pre_labeled_dir)
    if ret_code != error_code.OK:
        return json.dumps({"ret_code": ret_code,
                           "ret_msg": ret,
                           "result": ret})
    signals, file_names, fs, labels = ret
    print("finish audio loading")

    preprocess_config = load_config("preprocess")
    X_train = pre_process_data(signals, preprocess_config, fs=fs)
    y_train = labels
    print("finish data preparing")

    model = init_model_from_config()
    model.fit(X_train, y_train)
    ret_msg = "finish training"

    if save_model_path:
        model.save_model(save_model_path)
        ret_msg += ", model saved."
    if predict_dir:
        return evaluate(predict_dir, model=model, verbose=True)
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
    X_test = pre_process_data(signals, preprocess_config, fs=fs)
    y_test = labels

    if load_model_path:
        model = init_model_from_config()
        model.load_model(load_model_path)
    if not model:
        return json.dumps({"ret_code": error_code.MISSING_MODEL,
                           "ret_msg": "missing model",
                           "result": "missing model"})

    y_pred = model.predict(X_test)

    len_test = len(y_test)
    acc = sum((y_pred == y_test) * 1) / len_test
    acc_info = "accuracy: %s" % round(acc, 3)
    display_cm = DisplayManager().display_confusion_matrix(y_test, y_pred)
    cm_info = "Confusion Matrix: \n%s" % display_cm

    if kwargs.get("verbose"):
        print("number of test cases: %s" % len_test)
        print(acc_info)
        print(cm_info)
        false_prediction = [file_names[i] for i in range(len_test) if y_test[i] != y_pred[i]]
        print("false prediction %s" % false_prediction)

    return json.dumps({"ret_code": error_code.OK,
                       "ret_msg": "finish evaluating",
                       "result": [acc_info, cm_info]})


def predict(predict_dir, load_model_path=None, model=None):
    ret_code, ret = get_audio_files_and_labels(predict_dir)
    if ret_code != error_code.OK:
        return json.dumps({"ret_code": ret_code,
                           "ret_msg": ret,
                           "result": [[ret]]})
    signals, file_names, fs, _ = ret

    preprocess_config = load_config("preprocess")
    X_test = pre_process_data(signals, preprocess_config, fs=fs)
    if load_model_path:
        model = init_model_from_config()
        model.load_model(load_model_path)
    if not model:
        return json.dumps({"ret_code": error_code.MISSING_MODEL,
                           "ret_msg": "missing model",
                           "result": [["missing model"]]})

    y_pred = model.predict(X_test)
    result = [[file_names[i], "OK" if y_pred[i] else "NG"] for i in range(len(file_names))]
    return json.dumps({"ret_code": error_code.OK,
                       "ret_msg": "finish predicting",
                       "result": result})


def init_model_from_config():
    model_config = load_config("model")
    model_obj = MODEL_LIST.get(model_config.get("model_name"))
    model = model_obj(model_config)
    return model

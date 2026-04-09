import json
import os

from base.load_audio import get_audio_files_and_labels
from base.load_config import load_config
from base.model_config import init_model_from_config, preprocess_raw_signals
from base.onnx_audio_predictor import OnnxAudioPredictor
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
    ret_str, _ = predict_from_audio(signals, file_names, fs, load_model_path=load_model_path, model=model, **kwargs)

    return ret_str


def predict_from_audio(signals,
                       file_names,
                       fs,
                       load_model_path=None,
                       model=None,
                       **kwargs):
    backend = _get_inference_backend(load_model_path=load_model_path, model=model)
    if backend == "onnx":
        return _predict_with_onnx(
            signals=signals,
            file_names=file_names,
            fs=fs,
            load_model_path=load_model_path,
            **kwargs,
        )
    if backend == "pytorch":
        ret_str = json.dumps({
            "ret_code": error_code.INVALID_MODEL,
            "ret_msg": "PyTorch checkpoints are not supported directly. Please export the model to ONNX first.",
            "result": [["unsupported model type"]],
        })
        return ret_str, {"acc_req": 0.5}

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
    return ret_str, model.pred_config


def _get_inference_backend(load_model_path=None, model=None):
    if model is not None:
        return "keras"
    if not load_model_path:
        return "keras"

    model_ext = os.path.splitext(load_model_path)[1].lower()
    if model_ext == ".onnx":
        return "onnx"
    if model_ext in {".pth", ".pt"}:
        return "pytorch"
    return "keras"


def _predict_with_onnx(signals, file_names, fs, load_model_path=None, **kwargs):
    config_path = kwargs.get("config_path")
    if not config_path:
        ret_str = json.dumps({
            "ret_code": error_code.INVALID_CONFIG,
            "ret_msg": "missing config_path for ONNX inference",
            "result": [["missing config_path"]],
        })
        return ret_str, {"acc_req": 0.5}

    try:
        onnx_config = load_config(config_path=config_path)
        predictor = OnnxAudioPredictor(load_model_path, onnx_config)
        predictions = predictor.predict_arrays(signals, file_names, fs)
        result = [
            [item["file_name"], item["predicted_class"], str(round(item["probabilities"]["OK"], 3))]
            for item in predictions
        ]
        ret_str = json.dumps({
            "ret_code": error_code.OK,
            "ret_msg": "finish predicting",
            "result": result,
        })
        return ret_str, {"acc_req": predictor.acc_req}
    except Exception as e:
        ret_str = json.dumps({
            "ret_code": error_code.INVALID_MODEL,
            "ret_msg": str(e),
            "result": [[str(e)]],
        })
        return ret_str, {"acc_req": 0.5}

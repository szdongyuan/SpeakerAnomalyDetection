import json

import numpy as np

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
    ret_str = predict_from_audio(signals, file_names, fs, load_model_path=load_model_path, model=model, **kwargs)

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


# ========== 新增：多通道实时预测 ==========
def predict_multichannel_from_audio(multichannel_signal,
                                    file_name,
                                    sr,
                                    load_model_path=None,
                                    model=None,
                                    **kwargs):
    """
    多通道音频实时预测：对每个通道独立预测，然后融合结果。

    Args:
        multichannel_signal (np.ndarray): 多通道音频数据
            - 2D: (n_channels, n_samples)，通道数在前
            - 1D: (n_samples,)，自动转为单通道
        file_name (str): 文件名
        sr (int): 采样率
        load_model_path (str): 模型路径
        model: 已加载的模型实例
        **kwargs:
            - config_path: 配置文件路径

    Returns:
        str: JSON 字符串，格式与 predict_from_audio 一致
            {
                "ret_code": 错误码,
                "ret_msg": 消息,
                "result": [[file_name, "OK/NG", score, channel_details]]
            }
    """
    config_path = kwargs.get("config_path", model_consts.DEFAULT_DIR + model_consts.CONFIG_PATH)

    # 读取多通道配置
    data_load_config = load_config(config_path=config_path, module_name="data_load")
    multichannel_config = data_load_config.get("multichannel", {})
    n_channels_config = multichannel_config.get("n_channels", None)
    fusion_strategy = multichannel_config.get("fusion_strategy", "majority")
    channel_weights = multichannel_config.get("channel_weights", None)

    # 读取预处理配置
    preprocess_config = load_config(config_path=config_path, module_name="preprocess")

    # 加载模型
    if load_model_path and not model:
        model = init_model_from_config(**kwargs)
        model.load_model(load_model_path)

    if not model:
        return json.dumps({
            "ret_code": error_code.MISSING_MODEL,
            "ret_msg": "missing model",
            "result": [["missing model"]]
        })

    # 统一为 2D 格式 (n_channels, n_samples)
    if multichannel_signal.ndim == 1:
        multichannel_signal = multichannel_signal.reshape(1, -1)

    # 确定实际使用的通道数
    actual_channels = multichannel_signal.shape[0]
    if n_channels_config is not None:
        actual_channels = min(n_channels_config, actual_channels)

    # 每个通道独立预测
    channel_preds = []
    channel_scores = []

    for ch_idx in range(actual_channels):
        channel_signal = multichannel_signal[ch_idx]

        x_test = preprocess_raw_signals([channel_signal], [sr], preprocess_config)
        y_pred, pred_score = model.predict(x_test, acc_req=None, verbose=0)

        channel_preds.append(int(y_pred[0]))
        channel_scores.append(float(pred_score[0]))

    # 融合结果
    final_pred, final_score, channel_details = fuse_channel_results(
        channel_preds, channel_scores, strategy=fusion_strategy, weights=channel_weights
    )

    # 返回格式与 predict_from_audio 一致
    result = [[file_name, "OK" if final_pred else "NG", str(final_score), channel_details]]
    ret_str = json.dumps({
        "ret_code": error_code.OK,
        "ret_msg": "finish predicting",
        "result": result
    })
    return ret_str


def fuse_channel_results(channel_preds, channel_scores, strategy="majority", weights=None):
    """融合多通道预测结果"""
    n_channels = len(channel_preds)
    ok_count = sum(channel_preds)
    avg_score = float(np.mean(channel_scores))

    channel_details = "|".join([
        f"CH{i}:{'OK' if channel_preds[i] else 'NG'}({channel_scores[i]:.3f})"
        for i in range(n_channels)
    ])

    if strategy == "majority":
        final_pred = 1 if ok_count > n_channels / 2 else 0
        final_score = round(avg_score, 3)
    elif strategy == "any_ng":
        final_pred = 1 if ok_count == n_channels else 0
        final_score = round(min(channel_scores), 3)
    elif strategy == "weighted":
        if weights is None:
            final_score = round(avg_score, 3)
        else:
            final_score = float(np.average(np.array(channel_scores), weights=weights))
        final_pred = 1 if final_score >= 0.5 else 0
    elif strategy == "max_score":
        max_idx = int(np.argmax(channel_scores))
        final_pred = channel_preds[max_idx]
        final_score = round(channel_scores[max_idx], 3)
    else:
        final_pred = 1 if ok_count > n_channels / 2 else 0
        final_score = round(avg_score, 3)

    return final_pred, final_score, channel_details

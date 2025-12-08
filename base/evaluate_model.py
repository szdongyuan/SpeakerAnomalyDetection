import json

import numpy as np

from base.display import DisplayManager
from base.load_audio import get_pre_labeled_audios, get_pre_labeled_audios_from_dict
from base.load_config import load_config
from base.log_manager import LogManager
from base.model_config import init_model_from_config, preprocess_raw_signals
from base.predict_model import fuse_channel_results
from consts import error_code, model_consts


def evaluate(predict_dir, load_model_path=None, model=None, **kwargs):
    logger = LogManager.set_log_handler("evaluate")

    save_config_path = kwargs.get("config_path", model_consts.CONFIG_PATH)
    config_path = model_consts.DEFAULT_DIR + save_config_path

    # 读取 data_load 配置（包含多通道设置）
    data_load_config = load_config(config_path=config_path, module_name="data_load")

    multichannel_cfg = data_load_config.get("multichannel", {})
    multichannel_enabled = multichannel_cfg.get("enabled", False)
    logger.info(f"[评估] 多通道启用: {multichannel_enabled}")

    # 将配置传递给数据加载函数
    ret_code, ret = get_pre_labeled_audios(predict_dir, **data_load_config)
    if ret_code != error_code.OK:
        logger.error("failed to load audio samples")
        return json.dumps({"ret_code": ret_code, "ret_msg": ret, "result": ret})
    signals, file_names, fs, labels = ret

    preprocess_config = load_config(config_path=config_path, module_name="preprocess")

    if load_model_path:
        kwargs["config_path"] = config_path
        model = init_model_from_config(**kwargs)
        model.load_model(load_model_path)
    if not model:
        logger.error("missing model")
        return json.dumps({"ret_code": error_code.MISSING_MODEL, "ret_msg": "missing model", "result": "missing model"})

    if multichannel_enabled:
        acc_info, cm_info = evaluate_model_multichannel(
            model, preprocess_config, signals, file_names, fs, labels,
            multichannel_cfg, **kwargs
        )
    else:
        acc_info, cm_info = evaluate_model(model, preprocess_config, signals, file_names, fs, labels, **kwargs)

    model_detail = kwargs.get("model_detail", False)
    if model_detail:
        logger.info(model.model.summary())

    ret_str = json.dumps({"ret_code": error_code.OK, "ret_msg": "finish evaluating", "result": [acc_info, cm_info]})
    return ret_str


def evaluate_with_data(predict_dir, load_model_path=None, model=None, **kwargs):
    logger = LogManager.set_log_handler("evaluate")

    save_config_path = kwargs.get("config_path", model_consts.CONFIG_PATH)
    config_path = model_consts.DEFAULT_DIR + save_config_path
    data_load_config = load_config(config_path=config_path, module_name="data_load")

    ret_code, ret = get_pre_labeled_audios_from_dict(predict_dir, **data_load_config)
    if ret_code != error_code.OK:
        logger.error("failed to load audio samples")
        return json.dumps({"ret_code": ret_code, "ret_msg": ret, "result": ret})
    signals, file_names, fs, labels = ret

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


def evaluate_model_multichannel(model, preprocess_config, signals, file_names, fs, labels, multichannel_cfg, **kwargs):
    """
    多通道批量评估 - 与单通道保持一致的批量处理逻辑
    """
    logger = LogManager.set_log_handler("evaluate")
    # ========== 第一步：批量预处理和预测（与单通道一致） ==========
    x_test = preprocess_raw_signals(signals, fs, preprocess_config)
    y_pred_all, pred_score_all = model.predict(x_test)
    logger.info(f"[多通道评估] 批量预测完成，样本数: {len(y_pred_all)}")

    # ========== 第二步：按原始文件分组 ==========
    file_groups = {}

    for i, file_name in enumerate(file_names):
        # 直接分割，因为格式一定是 "原始文件名_ch通道号"
        parts = str(file_name).rsplit("_ch", 1)
        original_name = parts[0]
        ch_idx = int(parts[1])

        if original_name not in file_groups:
            file_groups[original_name] = []

        file_groups[original_name].append({
            "ch_idx": ch_idx,
            "pred": int(y_pred_all[i]),
            "score": float(pred_score_all[i]),
            "label": int(labels[i])
        })

    logger.info(f"[多通道评估] 文件分组完成，原始文件数: {len(file_groups)}")

    # ========== 第三步：按文件融合各通道结果 ==========
    fusion_strategy = multichannel_cfg.get("fusion_strategy", "majority")
    channel_weights = multichannel_cfg.get("channel_weights", None)

    y_pred_fused = []
    y_test_fused = []
    false_predictions = []

    for original_name, channels in file_groups.items():
        channels.sort(key=lambda x: x["ch_idx"])

        channel_preds = [ch["pred"] for ch in channels]
        channel_scores = [ch["score"] for ch in channels]
        true_label = channels[0]["label"]

        final_pred, final_score, channel_details = fuse_channel_results(
            channel_preds, channel_scores, strategy=fusion_strategy, weights=channel_weights
        )
        y_pred_fused.append(final_pred)
        y_test_fused.append(true_label)

        if final_pred != true_label:
            false_predictions.append(f"{original_name} ({channel_details})")
    # ========== 第四步：计算准确率（与单通道一致） ==========
    y_pred_fused = np.array(y_pred_fused)
    y_test_fused = np.array(y_test_fused)
    len_test = len(y_test_fused)
    acc = np.sum(y_pred_fused == y_test_fused) / len_test if len_test > 0 else 0
    acc_info = "accuracy (multichannel fusion): %s" % round(acc, 3)

    dm = DisplayManager()
    display_cm = dm.display_confusion_matrix(y_test_fused, y_pred_fused)
    cm_info = "Confusion Matrix (multichannel fusion): \n%s" % display_cm

    verbose = kwargs.get("verbose", 0)
    if verbose % 2:
        logger.info("number of test files: %s" % len_test)
        logger.info(acc_info)
        logger.info(cm_info)
        logger.info("false prediction:\n%s" % false_predictions)
    if (verbose >> 1) % 2:
        print("number of test files: %s" % len_test)
        print(acc_info)
        print(cm_info)
    if (verbose >> 2) % 2:
        print("false prediction:\n%s" % false_predictions)

    return acc_info, cm_info

import numpy as np

from base.load_config import load_config
from base.log_manager import LogManager
from base.pre_processing.preprocessing_manager import PreprocessingManager
from base.split_data_dir import copy_from_restored_audio_database
from consts import error_code, model_consts
from machine_learning import MODEL_MAPPING


default_logger = LogManager.set_log_handler("core")

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
        signal = np.asarray(raw_signals[i], dtype=np.float32)
        original_shape = tuple(signal.shape)
        squeezed_signal = np.squeeze(signal)
        if squeezed_signal.ndim == 0:
            signal = squeezed_signal.reshape(1)
        elif squeezed_signal.ndim == 1:
            signal = squeezed_signal
        if tuple(signal.shape) != original_shape:
            default_logger.info(
                "预处理输入归一化: index=%s, original_shape=%s, normalized_shape=%s",
                i,
                original_shape,
                tuple(signal.shape),
            )
        processed_data.append(pm.process(signal, fs[i], **preprocess_config))
    return np.array(processed_data)

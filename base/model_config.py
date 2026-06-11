import numpy as np

from base.load_config import load_config
from base.pre_processing.preprocessing_manager import PreprocessingManager
from base.split_data_dir import copy_from_restored_audio_database
from consts import error_code, model_consts
from machine_learning import get_model_class

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
    model_name = model_config.get("model_name")
    model_obj = get_model_class(model_name)
    if model_obj is None:
        raise ValueError(f"Unknown model_name: {model_name}")
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
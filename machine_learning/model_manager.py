import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import models

try:
    from keras.callbacks import EarlyStopping
except Exception as e:
    from keras.src.callbacks import EarlyStopping

from base.sample_balance import balance_sample_number
from base.training_model_management import TrainingModelManagement


class ModelManager(object):
    DEFAULT_CONFIG = {}

    def __init__(self, model_config):
        self.init_config = None
        self.fit_config = None
        self.pred_config = None
        self.model = None

        self._parse_config(model_config)

    def _parse_config(self, model_config):
        self.init_config = self.DEFAULT_CONFIG.get("model_init_config")
        for config_item in model_config.get("model_init_config", {}):
            self.init_config[config_item] = model_config["model_init_config"][config_item]

        self.fit_config = self.DEFAULT_CONFIG.get("model_fit_config")
        for config_item in model_config.get("model_fit_config", {}):
            self.fit_config[config_item] = model_config["model_fit_config"][config_item]

        self.pred_config = self.DEFAULT_CONFIG.get("model_predict_config")
        for config_item in model_config.get("model_predict_config", {}):
            self.pred_config[config_item] = model_config["model_predict_config"][config_item]

    @staticmethod
    def split_fit_valid(x, y, random_state=None):
        return train_test_split(x, y, test_size=0.2, random_state=random_state)

    def fit(self, x_train, y_train, validation_data=None):
        pass

    def predict(self, x_test):
        pass

    def save_model(self, signal_length, save_model_path, config_path, ret_str, model_description):
        pass

    def load_model(self, load_model_path):
        pass


class NeuralNetManager(ModelManager):

    def fit(self, x_train, y_train, validation_data=None):
        cycles = self.fit_config.get("cycles", 1)
        fit_kwargs = self.parse_fit_config()
        history = None
        for i in range(cycles):
            if self.fit_config.get("balance_sample_number"):
                x, y = balance_sample_number(x_train, y_train)
            else:
                x, y = x_train, y_train
            x_fit, x_valid, y_fit, y_valid = self.split_fit_valid(x, y)
            history = self.model.fit(x_fit, y_fit,
                                     validation_data=(x_valid, y_valid),
                                     **fit_kwargs)
            print("finish cycle %s" % i)
        return history

    def predict(self, x_test, acc_req=None, verbose=1):
        predictions = self.model.predict(x_test, verbose=verbose)
        if not acc_req:
            acc_req = self.pred_config.get("acc_req", 0.5)
        y_pred = [0 if i < acc_req else 1 for i in predictions]
        return np.array(y_pred), np.round(predictions.T[0], 3)

    def save_model(self, signal_length, save_model_path, config_path, ret_str, model_description):
        self.model.save(save_model_path)
        tmm = TrainingModelManagement()
        result = tmm.save_training_model_info_to_db(signal_length, save_model_path, config_path,
                                                    ret_str, model_description)

    def load_model(self, load_model_path):
        self.model = models.load_model(load_model_path)

    def parse_fit_config(self):
        fit_kwargs = {
            "epochs": self.fit_config.get("epochs", 10),
            "batch_size": self.fit_config.get("batch_size", 30)
        }
        if self.fit_config.get("class_weight"):
            fit_kwargs["class_weight"] = self.fit_config.get("class_weight")
        if self.fit_config.get("early_stop"):
            early_stop = EarlyStopping(monitor='val_loss', patience=3)
            fit_kwargs["callbacks"] = [early_stop]
        return fit_kwargs

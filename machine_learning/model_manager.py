import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import models

try:
    from keras.callbacks import EarlyStopping
except Exception as e:
    from keras.src.callbacks import EarlyStopping


class ModelManager(object):
    DEFAULT_CONFIG = {}

    def __init__(self, model_config):
        self.init_config = None
        self.fit_config = None
        self.pred_config = None

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
    def split_fit_valid(x, y):
        return train_test_split(x, y, test_size=0.2, random_state=42)

    def fit(self, x_train, y_train, validation_data=None):
        pass

    def predict(self, x_test):
        pass

    def save_model(self, save_model_path):
        pass

    def load_model(self, load_model_path):
        pass


class NeuralNetManager(ModelManager):

    def predict(self, x_test):
        predictions = self.model.predict(x_test)
        acc_req = self.pred_config.get("acc_req")
        if acc_req:
            y_pred = [0 if i[1] < acc_req else 1 for i in predictions]
            return np.array(y_pred), np.round(predictions[:, 1], 3)
        return np.argmax(predictions, axis=1), np.round(predictions[:, 1], 3)

    def save_model(self, save_model_path):
        self.model.save(save_model_path)

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

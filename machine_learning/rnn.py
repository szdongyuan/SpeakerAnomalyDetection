import numpy as np
from tensorflow.keras import layers, models, Input

from base.sample_balance import balance_sample_number
from machine_learning.model_manager import NeuralNetManager


class RNN(NeuralNetManager):
    DEFAULT_CONFIG = {
        "model_name": "RNN",
        "model_init_config": {
            "input_len_1": 64340,
            "input_len_2": 1,
            "hidden_units": 64,
            "num_classes": 1,
            "layers_param": [
                {"layer_name": "SimpleRNN",
                 "layer_kwargs": {"units": 64, "activation": "tanh", "return_sequences": True}},
                {"layer_name": "Flatten"},
                {"layer_name": "Dense",
                 "layer_kwargs": {"units": 64, "activation": "relu"}},
                {"layer_name": "Dense",
                 "layer_kwargs": {"units": 2, "activation": "softmax"}},
            ],
            "compile_param": {"optimizer": 'adam',
                              "loss": 'sparse_categorical_crossentropy',
                              "metrics": ["accuracy"]}
        },
        "model_fit_config":
            {
                "balance_sample_number": True,
                "cycles": 10,
                "epochs": 5,
                "batch_size": 30
            },
        "model_predict_config":
            {
                'acc_req': 0.8
            }
    }

    def __init__(self, model_config):
        super().__init__(model_config)
        self.model = None
        self._init_model()

    def _init_model(self):
        self.model = models.Sequential()
        self.model.add(
            Input(shape=(self.init_config.get("input_len_1"), self.init_config.get("input_len_2"))))
        for layer_param in self.init_config.get("layers_param", []):
            self._build_layer(layer_param)
        compile_kwargs = self.init_config.get("compile_param", {})
        self.model.compile(**compile_kwargs)

    def _build_layer(self, layer_param):
        layer = getattr(layers, layer_param.get("layer_name"))
        layer_kwargs = layer_param.get("layer_kwargs", {})
        self.model.add(layer(**layer_kwargs))

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

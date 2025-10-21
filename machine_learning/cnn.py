from tensorflow.keras import layers, models, Input

from base.sample_balance import balance_sample_number
from machine_learning.model_manager import NeuralNetManager


class CNN1d(NeuralNetManager):
    DEFAULT_CONFIG = {
        "model_name": "CNN1d",
        "model_init_config": {
            "input_len_1": 64340,
            "input_len_2": 1,
            "layers_param": [
                {"layer_name": "Conv1D",
                 "layer_kwargs": {"filters": 80, "kernel_size": 100, "activation": "relu", "strides": 10}},
                {"layer_name": "MaxPooling1D",
                 "layer_kwargs": {"pool_size": 4}},
                {"layer_name": "Conv1D",
                 "layer_kwargs": {"filters": 20, "kernel_size": 10, "activation": "relu", "strides": 2}},
                {"layer_name": "MaxPooling1D",
                 "layer_kwargs": {"pool_size": 4}},
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

        input_len_1 = self.init_config.get("input_len_1")
        input_len_2 = self.init_config.get("input_len_2")
        input_channels = self.init_config.get("input_channels", None)
        if input_channels is not None:
            input_shape = (input_len_1, input_len_2, input_channels)
        else:
            input_shape = (input_len_1, input_len_2)
        self.model.add(Input(shape=input_shape))

        for layer_param in self.init_config.get("layers_param", []):
            self._build_layer(layer_param)
        compile_kwargs = self.init_config.get("compile_param", {})
        self.model.compile(**compile_kwargs)

    def _build_layer(self, layer_param):
        layer = getattr(layers, layer_param.get("layer_name"))
        layer_kwargs = layer_param.get("layer_kwargs", {})
        self.model.add(layer(**layer_kwargs))

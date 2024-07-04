from tensorflow.keras import layers, models, Input

from base.sample_balance import balance_sample_number
from machine_learning.model_manager import NeuralNetManager


class Transformer(NeuralNetManager):
    DEFAULT_CONFIG = {
        "model_name": "Transformer",
        "model_init_config": {
            "input_len_1": 50,
            "input_len_2": 150,
            "layers_param": [
                {"layer_name": "TransformerEncoder",
                 "layer_kwargs": {"embed_dim": 150, "dense_dim": 200, "num_heads": 4}},
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
                "batch_size": 30,
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
        self.model.add(Input(shape=(self.init_config.get("input_len_1"),
                                    self.init_config.get("input_len_2"))))
        for layer_param in self.init_config.get("layers_param", []):
            self._build_layer(layer_param)
        compile_kwargs = self.init_config.get("compile_param", {})
        self.model.compile(**compile_kwargs)

    def _build_layer(self, layer_param):
        if layer_param.get("layer_name") == "TransformerEncoder":
            layer = TransformerEncoder
        else:
            layer = getattr(layers, layer_param.get("layer_name"))
        layer_kwargs = layer_param.get("layer_kwargs", {})
        self.model.add(layer(**layer_kwargs))


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        # Multi-Head Attention层
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        # Feed Forward层
        self.dense_proj = models.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim), ]
        )
        # Add&Norm层1
        self.layernorm_1 = layers.LayerNormalization()
        # Add&Norm层2
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs):
        # 首先经过Multi-Head Attention层
        attention_output = self.attention(inputs, inputs)
        # 经过Add&Norm层1
        proj_input = self.layernorm_1(inputs + attention_output)
        # 经过Feed Forward层
        proj_output = self.dense_proj(proj_input)
        # 经过Add&Norm层2
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config

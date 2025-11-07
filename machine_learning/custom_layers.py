from tensorflow.keras import layers, models, Input

CUSTOM_LAYER_REGISTRY = {}


def register_layer(layer_name: str):
    def decorator(cls):
        CUSTOM_LAYER_REGISTRY[layer_name] = cls
        return cls

    return decorator


@register_layer("TransformerEncoder")
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

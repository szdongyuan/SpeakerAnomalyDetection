from tensorflow.keras import layers, models, Input
import tensorflow as tf

CUSTOM_LAYER_REGISTRY = {}


def register_layer(layer_name: str):
    def decorator(cls):
        CUSTOM_LAYER_REGISTRY[layer_name] = cls
        return cls

    return decorator


@tf.keras.utils.register_keras_serializable()
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


@tf.keras.utils.register_keras_serializable()
@register_layer("SpatialPyramidPooling1D")
class SpatialPyramidPooling1D(layers.Layer):
    """Spatial Pyramid Pooling for 1D inputs.

    Splits the time axis into multiple scales (bin_sizes), pools each bin,
    and concatenates + flattens the result. Output size is fixed regardless
    of input length: sum(bin_sizes) * channels.

    Args:
        bin_sizes: List of int. Number of bins at each pyramid level.
        pool_mode: 'max' or 'avg'.
    """

    def __init__(self, bin_sizes=(1, 2, 4), pool_mode="max", **kwargs):
        super().__init__(**kwargs)
        self.bin_sizes = list(bin_sizes)
        self.pool_mode = pool_mode

    def _adaptive_pool_1d(self, x, bin_size):
        """Pool time axis into exactly bin_size bins using reshape + reduce."""
        shape = tf.shape(x)
        t = shape[1]
        channels = shape[2]
        # Pad time_steps to be divisible by bin_size
        remainder = t % bin_size
        pad_size = tf.where(remainder == 0, 0, bin_size - remainder)
        padded = tf.pad(x, [[0, 0], [0, pad_size], [0, 0]])
        seg_len = (t + pad_size) // bin_size
        # (batch, bin_size, seg_len, channels)
        reshaped = tf.reshape(padded, [-1, bin_size, seg_len, channels])
        if self.pool_mode == "max":
            return tf.reduce_max(reshaped, axis=2)
        else:
            return tf.reduce_mean(reshaped, axis=2)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        pooled_outputs = []
        for bin_size in self.bin_sizes:
            pooled = self._adaptive_pool_1d(inputs, bin_size)  # (batch, bin_size, channels)
            pooled_outputs.append(pooled)
        concatenated = tf.concat(pooled_outputs, axis=1)  # (batch, sum(bin_sizes), channels)
        return tf.reshape(concatenated, [input_shape[0], -1])

    def compute_output_shape(self, input_shape):
        channels = input_shape[-1]
        total_bins = sum(self.bin_sizes)
        return (input_shape[0], total_bins * channels)

    def get_config(self):
        config = super().get_config()
        config.update({
            "bin_sizes": self.bin_sizes,
            "pool_mode": self.pool_mode,
        })
        return config


@tf.keras.utils.register_keras_serializable()
@register_layer("SpatialPyramidPooling2D")
class SpatialPyramidPooling2D(layers.Layer):
    """Spatial Pyramid Pooling for 2D inputs.

    Splits H and W axes into bin_size x bin_size grids at each pyramid level,
    pools each cell, and concatenates + flattens the result.
    Output size: sum(b^2 for b in bin_sizes) * channels.

    Args:
        bin_sizes: List of int. Number of bins per spatial dimension at each level.
        pool_mode: 'max' or 'avg'.
    """

    def __init__(self, bin_sizes=(1, 2, 4), pool_mode="max", **kwargs):
        super().__init__(**kwargs)
        self.bin_sizes = list(bin_sizes)
        self.pool_mode = pool_mode

    def _adaptive_pool_2d(self, x, bin_size):
        """Pool H and W axes into bin_size x bin_size grid using reshape + reduce."""
        shape = tf.shape(x)
        h, w, channels = shape[1], shape[2], shape[3]

        # Pad H to be divisible by bin_size
        rem_h = h % bin_size
        pad_h = tf.where(rem_h == 0, 0, bin_size - rem_h)
        # Pad W to be divisible by bin_size
        rem_w = w % bin_size
        pad_w = tf.where(rem_w == 0, 0, bin_size - rem_w)

        padded = tf.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])

        seg_h = (h + pad_h) // bin_size
        seg_w = (w + pad_w) // bin_size

        # (batch, bin_size, seg_h, bin_size, seg_w, channels)
        reshaped = tf.reshape(padded, [-1, bin_size, seg_h, bin_size, seg_w, channels])
        # Transpose to (batch, bin_size, bin_size, seg_h, seg_w, channels)
        reshaped = tf.transpose(reshaped, [0, 1, 3, 2, 4, 5])

        if self.pool_mode == "max":
            pooled = tf.reduce_max(reshaped, axis=[3, 4])
        else:
            pooled = tf.reduce_mean(reshaped, axis=[3, 4])

        # (batch, bin_size*bin_size, channels)
        return tf.reshape(pooled, [-1, bin_size * bin_size, channels])

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        pooled_outputs = []
        for bin_size in self.bin_sizes:
            pooled = self._adaptive_pool_2d(inputs, bin_size)
            pooled_outputs.append(pooled)
        concatenated = tf.concat(pooled_outputs, axis=1)
        return tf.reshape(concatenated, [input_shape[0], -1])

    def compute_output_shape(self, input_shape):
        channels = input_shape[-1]
        total_bins = sum(b * b for b in self.bin_sizes)
        return (input_shape[0], total_bins * channels)

    def get_config(self):
        config = super().get_config()
        config.update({
            "bin_sizes": self.bin_sizes,
            "pool_mode": self.pool_mode,
        })
        return config

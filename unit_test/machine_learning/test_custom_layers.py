import numpy as np
import pytest
import tensorflow as tf


class TestSpatialPyramidPooling1D:
    """Tests for SpatialPyramidPooling1D layer."""

    def test_output_shape_default_bins(self):
        """bin_sizes=[1,2,4], 8 channels => output = (1+2+4)*8 = 56"""
        from machine_learning.custom_layers import CUSTOM_LAYER_REGISTRY
        SPP1D = CUSTOM_LAYER_REGISTRY["SpatialPyramidPooling1D"]
        layer = SPP1D(bin_sizes=[1, 2, 4], pool_mode="max")
        x = tf.random.normal((2, 100, 8))
        out = layer(x)
        assert out.shape == (2, 56)

    def test_output_shape_non_divisible(self):
        """time_steps=13 not divisible by bin_size=4, should still work."""
        from machine_learning.custom_layers import CUSTOM_LAYER_REGISTRY
        SPP1D = CUSTOM_LAYER_REGISTRY["SpatialPyramidPooling1D"]
        layer = SPP1D(bin_sizes=[1, 2, 4], pool_mode="max")
        x = tf.random.normal((2, 13, 8))
        out = layer(x)
        assert out.shape == (2, 56)

    def test_avg_pool_mode(self):
        """pool_mode='avg' should produce same shape, different values than max."""
        from machine_learning.custom_layers import CUSTOM_LAYER_REGISTRY
        SPP1D = CUSTOM_LAYER_REGISTRY["SpatialPyramidPooling1D"]
        layer_max = SPP1D(bin_sizes=[1, 2, 4], pool_mode="max")
        layer_avg = SPP1D(bin_sizes=[1, 2, 4], pool_mode="avg")
        x = tf.random.normal((2, 100, 8), seed=42)
        out_max = layer_max(x)
        out_avg = layer_avg(x)
        assert out_max.shape == out_avg.shape
        assert not np.allclose(out_max.numpy(), out_avg.numpy())

    def test_global_pool_bin_1(self):
        """bin_size=1 is equivalent to global pooling."""
        from machine_learning.custom_layers import CUSTOM_LAYER_REGISTRY
        SPP1D = CUSTOM_LAYER_REGISTRY["SpatialPyramidPooling1D"]
        layer = SPP1D(bin_sizes=[1], pool_mode="max")
        x = tf.constant([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # (1,3,2)
        out = layer(x)
        assert out.shape == (1, 2)
        np.testing.assert_array_almost_equal(out.numpy(), [[5.0, 6.0]])

    def test_get_config_roundtrip(self):
        """Layer should be serializable and deserializable."""
        from machine_learning.custom_layers import CUSTOM_LAYER_REGISTRY
        SPP1D = CUSTOM_LAYER_REGISTRY["SpatialPyramidPooling1D"]
        layer = SPP1D(bin_sizes=[1, 2, 4], pool_mode="avg")
        config = layer.get_config()
        assert config["bin_sizes"] == [1, 2, 4]
        assert config["pool_mode"] == "avg"
        restored = SPP1D.from_config(config)
        x = tf.random.normal((1, 50, 4))
        np.testing.assert_array_almost_equal(
            layer(x).numpy(), restored(x).numpy()
        )


class TestSpatialPyramidPooling2D:
    """Tests for SpatialPyramidPooling2D layer."""

    def test_output_shape_default_bins(self):
        """bin_sizes=[1,2,4], 8 channels => output = (1+4+16)*8 = 168"""
        from machine_learning.custom_layers import CUSTOM_LAYER_REGISTRY
        SPP2D = CUSTOM_LAYER_REGISTRY["SpatialPyramidPooling2D"]
        layer = SPP2D(bin_sizes=[1, 2, 4], pool_mode="max")
        x = tf.random.normal((2, 32, 32, 8))
        out = layer(x)
        assert out.shape == (2, 168)

    def test_output_shape_non_square(self):
        """Non-square input H=13, W=17 should still work."""
        from machine_learning.custom_layers import CUSTOM_LAYER_REGISTRY
        SPP2D = CUSTOM_LAYER_REGISTRY["SpatialPyramidPooling2D"]
        layer = SPP2D(bin_sizes=[1, 2, 4], pool_mode="max")
        x = tf.random.normal((2, 13, 17, 8))
        out = layer(x)
        assert out.shape == (2, 168)

    def test_avg_pool_mode(self):
        """pool_mode='avg' should produce same shape."""
        from machine_learning.custom_layers import CUSTOM_LAYER_REGISTRY
        SPP2D = CUSTOM_LAYER_REGISTRY["SpatialPyramidPooling2D"]
        layer_max = SPP2D(bin_sizes=[1, 2, 4], pool_mode="max")
        layer_avg = SPP2D(bin_sizes=[1, 2, 4], pool_mode="avg")
        x = tf.random.normal((2, 32, 32, 8), seed=42)
        out_max = layer_max(x)
        out_avg = layer_avg(x)
        assert out_max.shape == out_avg.shape
        assert not np.allclose(out_max.numpy(), out_avg.numpy())

    def test_global_pool_bin_1(self):
        """bin_size=1 is equivalent to global pooling."""
        from machine_learning.custom_layers import CUSTOM_LAYER_REGISTRY
        SPP2D = CUSTOM_LAYER_REGISTRY["SpatialPyramidPooling2D"]
        layer = SPP2D(bin_sizes=[1], pool_mode="max")
        x = tf.constant([[[[1.0], [2.0]], [[3.0], [4.0]]]])  # (1,2,2,1)
        out = layer(x)
        assert out.shape == (1, 1)
        np.testing.assert_array_almost_equal(out.numpy(), [[4.0]])

    def test_get_config_roundtrip(self):
        """Layer should be serializable and deserializable."""
        from machine_learning.custom_layers import CUSTOM_LAYER_REGISTRY
        SPP2D = CUSTOM_LAYER_REGISTRY["SpatialPyramidPooling2D"]
        layer = SPP2D(bin_sizes=[1, 2, 4], pool_mode="avg")
        config = layer.get_config()
        assert config["bin_sizes"] == [1, 2, 4]
        assert config["pool_mode"] == "avg"
        restored = SPP2D.from_config(config)
        x = tf.random.normal((1, 16, 16, 4))
        np.testing.assert_array_almost_equal(
            layer(x).numpy(), restored(x).numpy()
        )


class TestSPPIntegrationWithModelBuilder:
    """Verify SPP layers work with the project's model_builder pipeline."""

    def test_spp1d_in_functional_model(self):
        """Build a small model using SPP1D as the Flatten replacement."""
        from machine_learning.custom_layers import CUSTOM_LAYER_REGISTRY
        SPP1D = CUSTOM_LAYER_REGISTRY["SpatialPyramidPooling1D"]

        inp = tf.keras.Input(shape=(100, 1))
        x = tf.keras.layers.Conv1D(8, 10, activation="relu")(inp)
        x = SPP1D(bin_sizes=[1, 2, 4])(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inp, x)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        assert model.output_shape == (None, 1)

    def test_spp1d_layer_lookup_via_get_layer_class(self):
        """Verify get_layer_class finds SPP1D from the registry."""
        from machine_learning.model_builder import get_layer_class
        cls = get_layer_class("SpatialPyramidPooling1D")
        assert cls.__name__ == "SpatialPyramidPooling1D"

    def test_spp2d_layer_lookup_via_get_layer_class(self):
        """Verify get_layer_class finds SPP2D from the registry."""
        from machine_learning.model_builder import get_layer_class
        cls = get_layer_class("SpatialPyramidPooling2D")
        assert cls.__name__ == "SpatialPyramidPooling2D"

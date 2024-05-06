import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DataAlignment(object):

    @staticmethod
    def data_padding(raw_inputs, padding="post"):
        padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            raw_inputs, padding=padding
        )
        return padded_inputs

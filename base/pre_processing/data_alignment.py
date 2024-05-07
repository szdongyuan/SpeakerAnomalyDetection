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

    @staticmethod
    def chop_data(raw_inputs, chop_head=0, chop_tail=None):
        """
        :param raw_inputs: numpy ndarray
        :param chop_head: int, included in result
        :param chop_tail: int or None, not included in result
        :return: numpy ndarray. selected section of data
        """
        chopped_data = raw_inputs[:, chop_head:chop_tail]
        return chopped_data

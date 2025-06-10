import numpy as np


class Emphasis(object):

    @staticmethod
    def random_fluctuation(signal, sr, **kwargs):
        lower_bound = kwargs.get("lower_bound", 0.9)
        fluctuation_range = kwargs.get("upper_bound", 1.1) - lower_bound
        fluctuation_matrix = np.random.random(signal.shape[0])
        resize_array = np.ones(len(signal.shape))
        resize_array[0] = len(fluctuation_matrix)
        fluctuation_matrix.resize(resize_array.astype(int))
        fluctuation_matrix = fluctuation_matrix * fluctuation_range + lower_bound
        return signal * fluctuation_matrix

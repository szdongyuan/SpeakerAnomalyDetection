import numpy as np


def compare_dicts(dict1, dict2):
    if len(dict1) != len(dict2):
        return False
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        value1 = dict1[key]
        value2 = dict2[key]
        if isinstance(value1, (int, float, str, bool)) and isinstance(value2, (int, float, str, bool)):
            if value1 != value2:
                return False
        elif isinstance(value1, (list, tuple)) and isinstance(value2, (list, tuple)):
            if len(value1) != len(value2):
                return False
            for i, j in zip(value1, value2):
                if i != j:
                    return False
        elif isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
            if not np.array_equal(value1, value2):
                return False
        elif isinstance(value1, dict) and isinstance(value2, dict):
            if not compare_dicts(value1, value2):
                return False
        else:
            if value1 != value2:
                return False
    return True

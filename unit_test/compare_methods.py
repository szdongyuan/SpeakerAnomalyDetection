import numpy as np


def assert_equal(obj1, obj2):
    try:
        compare_method_dict = {
            dict: compare_dicts,
            np.ndarray: compare_numpy_array,
        }
        compare_method = compare_method_dict.get(type(obj1), compare_obj)
        if not compare_method(obj1, obj2):
            print(obj1)
            print(obj2)
        return compare_method(obj1, obj2)
    except Exception as e:
        print("Error, might be different input types")
        return False


def compare_dicts(dict1, dict2):
    if len(dict1) != len(dict2):
        return False
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        value1 = dict1[key]
        value2 = dict2[key]
        if not assert_equal(value1, value2):
            return False
    return True


def compare_numpy_array(array1, array2):
    return np.array_equal(array1, array2)


def compare_obj(obj1, obj2):
    return obj1 == obj2

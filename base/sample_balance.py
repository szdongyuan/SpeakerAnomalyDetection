import numpy as np


def balance_sample_number(x, y):
    y_classes = {}
    for i in y:
        y_classes[i] = y_classes.get(i, 0) + 1
    min_len = len(y)
    for i in y_classes:
        min_len = min(y_classes[i], min_len)

    x_list, y_list = [], []
    for single_class in y_classes:
        x_class = x[y == single_class]
        np.random.shuffle(x_class)
        x_list.append(x_class[:min_len])
        y_list += [single_class] * min_len
    x_balance = np.vstack(x_list)
    return x_balance, np.array(y_list)

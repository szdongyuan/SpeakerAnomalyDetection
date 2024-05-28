from sklearn.metrics import confusion_matrix


class DisplayManager(object):

    @staticmethod
    def display_confusion_matrix(y_true, y_pred, labels=None):
        if not labels:
            labels = [["true_NG", "true_OK"], ["predict_NG", "predict_OK"]]
        col_len_0 = max([len(label_str) for label_str in labels[0]]) + 1
        col_len_1 = max([len(label_str) for label_str in labels[1]])
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        result_str = ""
        for i in range(len(labels[1]) + 1):
            row_title = "" if i == 0 else str(labels[0][i - 1])
            row_entity = labels[1] if i == 0 else cm[i - 1]
            str_1 = row_title.rjust(col_len_0) + " | "
            str_2 = "-" * col_len_0 + "-+"
            for j in range(len(labels[0])):
                str_1 += str(row_entity[j]).rjust(col_len_1) + " | "
                str_2 += "-" * (col_len_1 + 2) + "+"
            result_str += "%s\n%s\n" % (str_1, str_2)
        return result_str

    @staticmethod
    def display_pred_score(file_names, labels, pred_score):
        for i in range(len(file_names)):
            print(file_names[i].ljust(25), labels[i], pred_score[i])

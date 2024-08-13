import csv

from sklearn.metrics import confusion_matrix

from consts.model_consts import LABEL_MAP


class DisplayManager(object):

    @staticmethod
    def display_confusion_matrix(y_true, y_pred, y_labels=None):

        """
            Calculates and displays the confusion matrix as a format string.

            Args:
            - y_true: list or array
                The true labels.
            - y_pred: list or array
                The Prediction labels as returned by a classifier.
            - y_labels: list
                Labels used to the confusion matrix display.

            Returns:
            - result_str: string
                The formatted string representation of the obfuscation matrix.
        """
        if not y_labels:
            y_labels = ["NG", "OK"]
        labels = [["true_" + i for i in y_labels], ["predict_" + i for i in y_labels]]
        col_len_0 = max([len(label_str) for label_str in labels[0]]) + 1
        col_len_1 = max([len(label_str) for label_str in labels[1]])
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[LABEL_MAP[i] for i in y_labels])
        result_str = ""
        for i in range(len(labels[1]) + 1):
            row_title = "" if i == 0 else str(labels[0][i - 1])
            row_entity = labels[1] if i == 0 else cm[i - 1]
            print(row_entity)
            str_1 = row_title.rjust(col_len_0) + " | "
            str_2 = "-" * col_len_0 + "-+"
            for j in range(len(labels[0])):
                str_1 += str(row_entity[j]).rjust(col_len_1) + " | "
                str_2 += "-" * (col_len_1 + 2) + "+"
            result_str += "%s\n%s\n" % (str_1, str_2)
        return result_str

    @staticmethod
    def display_pred_score(file_names, labels, pred_score, to_csv=False):
        """
            Display or save the predicted score for each audio file to be predicted.

            Args:
            - file_names: list
                List containing names of audio files.
            - labels: list
                True label for each audio file.
            - pred_score: list
                Predicted score for each audio file.
            - to_csv: bool
                Whether to save the output to a csv file or not (default is False).
        """
        if to_csv:
            pred_data = [[file_names[i], labels[i], pred_score[i]] for i in range(len(labels))]
            with open("pred_score.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(pred_data)
            print("finish writing csv.")
        else:
            for i in range(len(labels)):
                print(file_names[i].ljust(25), labels[i], pred_score[i])

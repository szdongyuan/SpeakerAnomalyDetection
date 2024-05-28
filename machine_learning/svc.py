import joblib
import numpy as np
from sklearn.svm import LinearSVC

from base.sample_balance import balance_sample_number
from machine_learning.model_manager import ModelManager


class EnsembleSVC(ModelManager):

    """
    config format:
---------------------------------------------------------------
- module_name: "model"

  module_config:

    model_name: "EnsembleSVC"

    model_init_config:
      n_models: # int, number of SVC models for ensemble
      param: # map, params to init single SVC model
        dual: "auto" or bool, prefer dual=False when n_samples > n_features.
        max_iter: int, maximum number of iterations to be run.
        ...

    model_fit_config:
      balance_sample_number: # bool, set True to choose identical size for each category

    model_predict_config:
      acc_req: # float [0-1],
---------------------------------------------------------------

    """

    DEFAULT_CONFIG = {'model_name': 'EnsembleSVC',
                      'model_init_config':
                          {
                              'n_models': 10,
                              'param': {'dual': False, 'max_iter': 5000}
                          },
                      'model_fit_config':
                          {
                              'balance_sample_number': True
                          },
                      'model_predict_config':
                          {
                              'acc_req': 0.8
                          }
                      }

    def __init__(self, model_config):
        super().__init__(model_config)
        self.model_list = None

        self._build_model()

    def _build_model(self):
        self.model_list = []
        for i in range(self.init_config.get("n_models")):
            svc_kwargs = self.init_config.get("param")
            self.model_list.append(LinearSVC(**svc_kwargs))

    def fit(self, x_train, y_train, **kwargs):
        for model in self.model_list:
            if self.fit_config.get("balance_sample_number"):
                x, y = balance_sample_number(x_train, y_train)
            else:
                x, y = x_train, y_train
            model.fit(x, y)
        return "finished training [%s] models" % len(self.model_list)

    def predict(self, x_test):
        y_pred_tot = []
        for model in self.model_list:
            y_pred_tot.append(model.predict(x_test))
        y_pred_tot = np.vstack(y_pred_tot)
        acc_req = self.pred_config.get("acc_req")
        y_pred = (np.sum(y_pred_tot, axis=0) >= len(self.model_list) * acc_req) * 1
        return y_pred, np.round(np.sum(y_pred_tot, axis=0) / len(self.model_list), 3)

    def save_model(self, save_model_path):
        joblib.dump(self.model_list, save_model_path)

    def load_model(self, load_model_path):
        self.model_list = joblib.load(load_model_path)


class SimpleSVC(ModelManager):

    DEFAULT_CONFIG = {'model_name': 'SimpleSVC',
                      'model_init_config':
                          {
                              'param': {'dual': False, 'max_iter': 5000}
                          },
                      'model_fit_config': {},
                      'model_predict_config': {}
                      }

    def __init__(self, model_config):
        super().__init__(model_config)
        self.model = None

        self._build_model()

    def _build_model(self):
        svc_kwargs = self.init_config.get("param")
        self.model = LinearSVC(**svc_kwargs)

    def fit(self, x_train, y_train, **kwargs):
        self.model.fit(x_train, x_train)
        return "finished training"

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        pred_score = self.model.decision_function(x_test)
        return y_pred, pred_score

    def save_model(self, save_model_path):
        joblib.dump(self.model, save_model_path)

    def load_model(self, load_model_path):
        self.model = joblib.load(load_model_path)

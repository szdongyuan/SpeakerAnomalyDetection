class ModelManager(object):

    DEFAULT_CONFIG = {}

    def __init__(self, model_config):
        self.init_config = None
        self.fit_config = None
        self.pred_config = None

        self._parse_config(model_config)

    def _parse_config(self, model_config):
        self.init_config = self.DEFAULT_CONFIG.get("model_init_config")
        for config_item in model_config.get("model_init_config", {}):
            self.init_config[config_item] = model_config["model_init_config"][config_item]
        self.fit_config = self.DEFAULT_CONFIG.get("model_fit_config")
        for config_item in model_config.get("model_fit_config", {}):
            self.fit_config[config_item] = model_config["model_fit_config"][config_item]
        self.pred_config = self.DEFAULT_CONFIG.get("model_predict_config")
        for config_item in model_config.get("model_predict_config", {}):
            self.pred_config[config_item] = model_config["model_predict_config"][config_item]

    def fit(self, x_train, y_train, validation_data=None):
        pass

    def predict(self, x_test):
        pass

    def save_model(self, save_model_path):
        pass

    def load_model(self, load_model_path):
        pass

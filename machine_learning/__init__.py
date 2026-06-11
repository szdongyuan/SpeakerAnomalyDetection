from importlib import import_module


MODEL_MAPPING = {
    "EnsembleSVC": ("machine_learning.svc", "EnsembleSVC"),
    "SimpleSVC": ("machine_learning.svc", "SimpleSVC"),
    "CNN1d": ("machine_learning.cnn", "CNN1d"),
    "RNN": ("machine_learning.rnn", "RNN"),
    "Transformer": ("machine_learning.transformer", "Transformer"),
}


def get_model_class(model_name):
    model_path = MODEL_MAPPING.get(model_name)
    if model_path is None:
        return None
    module_name, class_name = model_path
    module = import_module(module_name)
    return getattr(module, class_name)

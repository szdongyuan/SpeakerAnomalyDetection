import yaml

from consts.model_consts import CONFIG_PATH


def load_config(module_name=None):
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f.read())
        if module_name:
            for module_config in config:
                if module_config.get("module_name") == module_name:
                    result = module_config.get("module_config")
        else:
            result = config
    return result

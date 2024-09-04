import yaml

from consts.model_consts import CONFIG_PATH


def load_config(module_name=None):
    """
        Load configuration from a YAML file. Optionally, retrieve specific module configuration.

        Args:
        - module_name : string
            The name of the module whose configuration you want to retrieve.
            If None, the entire configuration is loaded.
        Returns:
        - result : dictionary
            The configuration dictionary that stores specific module configurations
            or entire configurations.
    """

    result = {}
    with open(CONFIG_PATH, encoding='utf-8') as f:
        config = yaml.safe_load(f.read())
        if module_name:
            for module_config in config:
                if module_config.get("module_name") == module_name:
                    result = module_config.get("module_config", {})
        else:
            result = config
    return result

import json
import os
import yaml

from consts import error_code
from consts.running_consts import DEFAULT_DIR


def load_config(config_path, module_name=None):
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
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f.read())
        if module_name:
            for module_config in config:
                if module_config.get("module_name") == module_name:
                    result = module_config.get("module_config", {})
        else:
            result = config
    return result


class LoadUiConfig(object):

    @staticmethod
    def load_sequence_config_from_json():
        """
        Loads analysis sequence configuration data from a specified JSON file.

        This function first checks if the JSON file exists. If not, it returns an error code and message.
        If the file exists, it attempts to read and parse the JSON file content, storing it in the class's
        `analysis_config` attribute. If any exception occurs during reading or parsing, it catches the
        exception and returns the corresponding error code and message.

        Returns:
            tuple: A tuple containing two elements:
                - The first element is an error code indicating the result status of the operation.
                - The second element is either an error message or the parsed JSON data.
        """
        json_file_path = DEFAULT_DIR + "ui/ui_config/analysis_temp_config.json"
        if not os.path.exists(json_file_path):
            return error_code.INVALID_DATA_LOADING, "This json file does not exist."
        try:
            with open(json_file_path, "r") as json_file:
                analysis_config = json.load(json_file)
                return error_code.OK, analysis_config
        except Exception as e:
            err_msg = "Failed to load analysis sequence data from json.%s" % (str(e)[:50])
            return error_code.INVALID_DATA_LOADING, err_msg

    @staticmethod
    def load_last_recorded_info(logger):
        """
        Load the recorded number from a text file.

        This method reads a recorded number and the last recorded date from a specified text file.
        If the file exists and the last recorded date matches the current date, it returns the recorded number;
        otherwise, it returns None.

        Returns:
            int or None: The recorded number if the file exists and the date matches; otherwise, None.
        """
        file_path = DEFAULT_DIR + "ui/ui_config/recorded_number.json"
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                return data
        except Exception as e:
            logger.error(f"Failed to read the info of recorded number: {e}")
            return None

import json
import os
import yaml

from datetime import datetime

from consts import error_code
from consts.running_consts import DEFAULT_DIR, SEQUENCE_CONFIG_REGISTRY_PATH
from base.log_manager import LogManager


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
    with open(config_path, encoding="utf-8") as f:
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
    def load_sequence_config_from_json(json_file_path):
        """
        Loads analysis sequence configuration data using the **new list-based format**.

        The new JSON layout is a list, whose first element is a dict with a single
        sequence key (e.g. "seq1").  Each sequence contains an "acq" section and an
        "analysis_list" section that keeps the previous flat configuration.  This
        function extracts and returns that inner ``analysis_list`` so that the rest
        of the code can keep working with the same dict structure as before.
        """
        if not json_file_path or not isinstance(json_file_path, (str, bytes, os.PathLike)):
            return error_code.INVALID_DATA_LOADING, "Invalid json file path."
        if not os.path.exists(json_file_path):
            return error_code.INVALID_DATA_LOADING, "This json file does not exist."
        try:
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                analysis_config = json.load(json_file)
                return error_code.OK, analysis_config
        except Exception as e:
            err_msg = "Failed to load analysis sequence data from json.%s" % (str(e)[:50])
            return error_code.INVALID_DATA_LOADING, err_msg

    @staticmethod
    def load_data_from_json(json_file_path):
        """
        Loads data from a specified JSON file and returns it with an error code.

        This method attempts to load JSON data from the provided file path. It first checks
        if the file exists, and if not, returns an error code with a descriptive message.

        Args:
            json_file_path (str): The path to the JSON file to be loaded.

        Returns:
            tuple: A tuple containing:
                - error_code (int): error_code.OK on success,
                  error_code.INVALID_DATA_LOADING on failure
                - data (dict/list) or error_message (str): Parsed JSON data on success,
                  error description on failure
        """
        if not os.path.exists(json_file_path):
            return error_code.INVALID_DATA_LOADING, "This json file does not exist."
        try:
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                analysis_config = json.load(json_file)
            return error_code.OK, analysis_config
        except Exception as e:
            err_msg = f"Failed to load analysis sequence data from json. {str(e)[:50]}"
            return error_code.INVALID_DATA_LOADING, err_msg

    @staticmethod
    def save_sequence_config_to_json(config_data, json_file_path):
        """Save ``config_data`` (the inner analysis_list dict) back to json file using the new format."""
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        try:
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=6, ensure_ascii=False)
            return True
        except Exception as e:
            return False

    @staticmethod
    def _load_sequence_config_registry(registry_path: str = None) -> dict:
        """
        Load the sequence config registry JSON.

        Returns an empty dict if:
        - file doesn't exist
        - file content is invalid / not a dict
        """
        registry_path = registry_path or SEQUENCE_CONFIG_REGISTRY_PATH
        try:
            if not os.path.exists(registry_path):
                return {}
            with open(registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            # Treat as empty registry on any parse/read error
            return {}

    @staticmethod
    def _save_sequence_config_registry(registry: dict, registry_path: str = None) -> bool:
        """Write registry JSON to disk (creates parent dir)."""
        registry_path = registry_path or SEQUENCE_CONFIG_REGISTRY_PATH
        try:
            os.makedirs(os.path.dirname(registry_path), exist_ok=True)
            with open(registry_path, "w", encoding="utf-8") as f:
                json.dump(registry or {}, f, indent=6, ensure_ascii=False)
            return True
        except Exception:
            return False

    @staticmethod
    def append_sequence_config_registry_entry(file_path: str, registry_path: str = None) -> bool:
        """
        Append/update one entry to registry using filename (without extension) as key,
        and full file path as value.
        """
        if not file_path:
            return False
        registry_path = registry_path or SEQUENCE_CONFIG_REGISTRY_PATH
        try:
            key = os.path.splitext(os.path.basename(file_path))[0]
            if not key:
                return False
            registry = LoadUiConfig._load_sequence_config_registry(registry_path)
            registry[key] = file_path
            return LoadUiConfig._save_sequence_config_registry(registry, registry_path)
        except Exception:
            return False

    @staticmethod
    def ensure_sequence_config_registry_field(field_key: str, field_value: str, registry_path: str = None) -> bool:
        """
        Ensure registry contains the given field_key.
        If missing, write field_key -> field_value.
        """
        if not field_key:
            return False
        registry_path = registry_path or SEQUENCE_CONFIG_REGISTRY_PATH
        try:
            registry = LoadUiConfig._load_sequence_config_registry(registry_path)
            if field_key in registry:
                return True
            registry[field_key] = field_value
            return LoadUiConfig._save_sequence_config_registry(registry, registry_path)
        except Exception:
            return False

    @staticmethod
    def update_using_config_path(using_config_path, registry_path: str = None) -> bool:
        """
        Update the using config path in the registry.
        """
        if not using_config_path:
            return False
        registry_path = registry_path or SEQUENCE_CONFIG_REGISTRY_PATH
        registry = LoadUiConfig._load_sequence_config_registry()
        registry["using_config_path"] = using_config_path
        return LoadUiConfig._save_sequence_config_registry(registry)

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
            logger.error(f"Failed to read the info of recorded number: {str(e)[:50]}")
            return None

    @staticmethod
    def load_recorded_num_from_json(logger):
        """
        Load the recorded number from a text file.

        This method reads a recorded number and the last recorded date from a specified text file.
        If the file exists and the last recorded date matches the current date, it returns the recorded number;
        otherwise, it returns None.

        Returns:
            int or None: The recorded number if the file exists and the date matches; otherwise, None.
        """
        result = LoadUiConfig().load_last_recorded_info(logger)
        if result:
            last_datetime = result.get("datetime")
            recorded_count = result.get("current_recorded_count")
            scanner_barcode = result.get("scanner_barcode")
            if last_datetime == datetime.now().strftime("%Y-%m-%d"):
                return recorded_count, scanner_barcode

        return None, None

    @staticmethod
    def get_rec_and_play_dict_base_sequence_dict(data_struct, total_time=None):
        """
        Generate dictionaries containing stimulus signal data and recording parameters.

        This function creates two dictionaries: one for the stimulus signal data and its related information,
        and another for the recording parameters. These dictionaries are used for subsequent signal processing and analysis.

        Args:
        - sample_rate (int): The sampling rate, indicating the number of samples collected per second.
        - total_time (int): The total recording time, indicates the duration of the recording.

        Returns:
        - stimulus_dict (dict): Dictionary containing the stimulus signal data and related information.
        - recorded_dict (dict): Dictionary containing the recording parameters.
        """
        # Define the prolongation time to calculate the extended frame count
        prolong = 0.5
        stimulus_dict = dict()
        if data_struct.stimulus_data is not None and len(data_struct.stimulus_data) > 0:
            stimulus_dict = {
                "data": data_struct.stimulus_data,
                "amplitude": data_struct.stimulus_info["amplitude"],
                "sr": data_struct.sample_rate,
            }
            num_frames = len(data_struct.stimulus_data) + int(prolong * data_struct.sample_rate)
            prolong_frames = int(prolong * data_struct.sample_rate)
        else:
            num_frames = int(total_time * data_struct.sample_rate)
            prolong_frames = 0
        recorded_dict = {
            "channels": 1,
            "sr": data_struct.sample_rate,
            "num_frames": num_frames,
            "prolong_frames": prolong_frames,
        }
        return stimulus_dict, recorded_dict

    @staticmethod
    def write_tcp_config(ip, port, logger):
        file_path = DEFAULT_DIR + "ui/ui_config/tcp_config.txt"

        try:
            with open(file_path, "w") as f:
                f.write(f"ip = {ip}\n")
                f.write(f"port = {port}\n")
            logger.info(f"write_tcp_config_success: {file_path}")
        except Exception as e:
            logger.error(f"write_tcp_config_error: {e}")

    @staticmethod
    def get_tcp_config():
        file_path = DEFAULT_DIR + "ui/ui_config/tcp_config.txt"
        with open(file_path, "r") as f:
            config_data = f.readlines()
            ip = config_data[0].split("=")[1].strip()
            port_text = config_data[1].split("=")[1].strip()
            port = int(port_text)
            return ip, port


class ConfigManager(object):
    """负责读写分析窗口各项配置的通用管理器，迁移自 ui.analysis_config_window"""

    def __init__(self, config_file):
        self.config_file = config_file
        self.default_logger = LogManager.set_log_handler("core")
        self.config = {}

    def save_config(self, type, config_data):
        if type in self.config:
            self.config[type].update(config_data)
        else:
            self.config[type] = config_data
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
                self.default_logger.info(f"The config info for {type} analysis has been saved to {self.config_file}.")
                return True
        except Exception as e:
            self.default_logger.error(f"The config info for {type} analysis save failed. {e}")
            return False

    def save_default_config(self, type, config_data):
        default_config_file = DEFAULT_DIR + "ui/ui_config/analysis_default_config.json"
        default_config = {}
        try:
            with open(default_config_file, "r", encoding="utf-8") as f:
                default_config = json.load(f)
                if type in default_config:
                    default_config[type].update(config_data)
                else:
                    default_config[type] = config_data
            with open(default_config_file, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
                self.default_logger.info(
                    f"The config info for {type} analysis has been saved to {default_config_file}."
                )
                return True
        except Exception as e:
            self.default_logger.error(f"Failed to load the default config file. {e}")
            return False

    def load_config(self):
        try:
            if self.config:
                return self.config
            with open(self.config_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            if isinstance(raw_data, list):
                self.config = LoadUiConfig._extract_analysis_list(raw_data)
            else:
                self.config = raw_data
            return self.config
        except Exception as e:
            self.default_logger.error(f"Failed to load the default or temp config file. {e}")
            return {}

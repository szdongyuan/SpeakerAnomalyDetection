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
    def _merge_dict_with_defaults(defaults, overrides):
        if not isinstance(defaults, dict):
            return overrides
        merged = dict(defaults)
        if not isinstance(overrides, dict):
            return merged
        for key, value in overrides.items():
            default_value = merged.get(key)
            if isinstance(default_value, dict) and isinstance(value, dict):
                merged[key] = LoadUiConfig._merge_dict_with_defaults(default_value, value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def get_default_serial_discrete_input_config():
        return {
            "_comment_global": "串口离散输入触发配置",
            "enabled": False,
            "device_model": "JY-DAM0404D",
            "serial_settings": {
                "port": "COM3",
                "baudrate": 9600,
                "bytesize": 8,
                "parity": "N",
                "stopbits": 1,
                "timeout": 0.1,
            },
            "polling_settings": {
                "interval_ms": 50,
                "query_command_hex": "FE 02 00 00 00 04 6D C6",
            },
            "trigger_settings": {
                "delay_seconds": 0.5,
                "direction_cycle_policy": {
                    "test_mode": "forward_then_reverse",
                },
            },
            "decoder": {
                "_comment": "mode 可以是 'full_frame' 或 'state_byte'",
                "mode": "full_frame",
                "state_byte_index": 3,
            },
            "state_maps": {
                "full_frame": {
                    "FE 02 01 01 50 5C": {
                        "action": "start_record",
                        "direction": "forward",
                        "description": "只按下绿色按钮 (正转)",
                    },
                    "FE 02 01 03 D1 9D": {
                        "action": "start_record",
                        "direction": "reverse",
                        "description": "绿色和红色按钮都按下 (反转)",
                    },
                    "FE 02 01 00 90 5C": {
                        "action": "idle",
                        "description": "全部松开 (空闲)",
                    },
                    "FE 02 01 02 91 9C": {
                        "action": "ignore",
                        "description": "只按下红色按钮 (忽略)",
                    },
                },
                "state_byte": {
                    "01": {
                        "action": "start_record",
                        "direction": "forward",
                        "description": "只按下绿色按钮 (正转)",
                    },
                    "03": {
                        "action": "start_record",
                        "direction": "reverse",
                        "description": "绿色和红色按钮都按下 (反转)",
                    },
                    "00": {
                        "action": "idle",
                        "description": "全部松开 (空闲)",
                    },
                    "02": {
                        "action": "ignore",
                        "description": "只按下红色按钮 (忽略)",
                    },
                },
            },
        }

    @staticmethod
    def normalize_serial_discrete_input_config(config_data):
        return LoadUiConfig._merge_dict_with_defaults(
            LoadUiConfig.get_default_serial_discrete_input_config(),
            config_data if isinstance(config_data, dict) else {},
        )

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
    def get_product_test_program_default_config_path():
        return os.path.join(
            DEFAULT_DIR,
            "ui",
            "ui_config",
            "product_test_programs",
            "default_config.json",
        ).replace("\\", "/")

    @staticmethod
    def load_product_test_program_condition_configs(config_path: str = None):
        """
        Load condition rows from product_test_programs/default_config.json.

        The UI only needs stable display names plus a lookup key. Keep the raw
        trigger_state/test_queue fields so later runtime wiring can route by
        either value without re-reading the file.
        """
        path = config_path or LoadUiConfig.get_product_test_program_default_config_path()
        load_code, data = LoadUiConfig.load_data_from_json(path)
        if load_code != error_code.OK or not isinstance(data, dict):
            return []

        sub_configs = data.get("sub_configs", [])
        if not isinstance(sub_configs, list):
            return []

        result = []
        used_keys = set()
        for index, item in enumerate(sub_configs):
            if not isinstance(item, dict):
                continue
            condition_name = str(item.get("condition_name") or "").strip()
            if not condition_name:
                continue
            trigger_state = str(item.get("trigger_state") or "").strip()
            test_queue = str(item.get("test_queue") or "").strip()
            base_key = trigger_state or test_queue or f"condition_{index + 1}"
            key = base_key
            if key in used_keys:
                key = f"{base_key}#{index + 1}"
            used_keys.add(key)
            result.append(
                {
                    "key": key,
                    "condition_name": condition_name,
                    "trigger_state": trigger_state,
                    "test_queue": test_queue,
                }
            )
        return result

    @staticmethod
    def load_product_test_program_pdf_report_config(config_path: str = None):
        path = config_path or LoadUiConfig.get_product_test_program_default_config_path()
        load_code, data = LoadUiConfig.load_data_from_json(path)
        if load_code != error_code.OK or not isinstance(data, dict):
            return {"enabled": False, "save_dir": ""}

        config = data.get("pdf_report")
        if not isinstance(config, dict):
            return {"enabled": False, "save_dir": ""}
        save_dir = config.get("save_dir", "")
        if not isinstance(save_dir, str):
            save_dir = ""
        return {
            "enabled": bool(config.get("enabled", False)),
            "save_dir": save_dir.strip(),
        }

    @staticmethod
    def load_product_test_program_close_trigger_state(config_path: str = None):
        path = config_path or LoadUiConfig.get_product_test_program_default_config_path()
        load_code, data = LoadUiConfig.load_data_from_json(path)
        if load_code != error_code.OK or not isinstance(data, dict):
            return ""

        close_trigger_state = data.get("close_trigger_state", "")
        if not isinstance(close_trigger_state, str):
            return ""
        return " ".join(close_trigger_state.strip().upper().split())

    @staticmethod
    def save_sequence_config_to_json(config_data, json_file_path):
        """Save ``config_data`` (the inner analysis_list dict) back to json file using the new format."""
        return LoadUiConfig.save_data_to_json(config_data, json_file_path, 6)

    @staticmethod
    def save_data_to_json(config_data, json_file_path, indent=2):
        """Atomically save JSON data using the project's UTF-8 format."""
        target_path = os.path.abspath(json_file_path)
        target_dir = os.path.dirname(target_path)
        temp_path = target_path + ".tmp"
        os.makedirs(target_dir, exist_ok=True)
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=indent, ensure_ascii=False)
            os.replace(temp_path, target_path)
            return True
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
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
        Load persisted sequence-page UI state from disk.

        The sequence page no longer stores a manual record count, but some legacy callers
        still expect this helper to return a `(count, barcode)` tuple. We keep that shape
        for compatibility and always return `None` for the removed count field.

        Returns:
            tuple: `(None, scanner_barcode)` for today, otherwise `(None, None)`.
        """
        result = LoadUiConfig().load_last_recorded_info(logger)
        if result:
            last_datetime = result.get("datetime")
            scanner_barcode = result.get("scanner_barcode")
            if last_datetime == datetime.now().strftime("%Y-%m-%d"):
                return None, scanner_barcode

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

    @staticmethod
    def get_default_scanner_config():
        """Canonical default for ``scanner_hid_config.json``.

        Shape matches the new layout: ``barcode_source`` at top level,
        ``scanner.hid`` / ``scanner.serial`` split by protocol, and
        ``sensor.enabled`` as an explicit on/off for the photoelectric
        hotkey path. Old flat ``scanner.vid/pid`` / ``sensor.vid/pid``
        files are upgraded by :meth:`normalize_scanner_config` and keep
        working without manual edits.
        """
        return {
            "barcode_source": "hid",
            "scanner": {
                "hid": {
                    "vid": "",
                    "pid": "",
                },
                "serial": {
                    "port": "",
                    "baudrate": 9600,
                    "bytesize": 8,
                    "parity": "N",
                    "stopbits": 1,
                    "timeout": 0.1,
                    "terminator": "\r\n",
                    "encoding": "utf-8",
                },
            },
            "sensor": {
                "enabled": True,
                "hid": {
                    "vid": "",
                    "pid": "",
                },
                "hotkey": "",
            },
        }

    @staticmethod
    def normalize_scanner_config(config_data):
        """Merge ``config_data`` onto the scanner defaults.

        Also upgrades the legacy flat structure (``scanner.vid/pid``,
        ``sensor.vid/pid``) into the nested ``scanner.hid`` /
        ``sensor.hid`` layout so downstream code only has to handle one
        shape. Any non-dict input degrades to pure defaults.
        """
        defaults = LoadUiConfig.get_default_scanner_config()
        if not isinstance(config_data, dict):
            return defaults

        upgraded = {k: v for k, v in config_data.items()}

        scanner = upgraded.get("scanner")
        if isinstance(scanner, dict):
            # Legacy flat: {"scanner": {"vid": "...", "pid": "..."}} -> nest under scanner.hid
            if "hid" not in scanner and ("vid" in scanner or "pid" in scanner):
                scanner = {
                    "hid": {
                        "vid": scanner.get("vid", ""),
                        "pid": scanner.get("pid", ""),
                    },
                }
                upgraded["scanner"] = scanner

        sensor = upgraded.get("sensor")
        if isinstance(sensor, dict):
            if "hid" not in sensor and ("vid" in sensor or "pid" in sensor):
                legacy_hotkey = sensor.get("hotkey", "")
                sensor = {
                    "hid": {
                        "vid": sensor.get("vid", ""),
                        "pid": sensor.get("pid", ""),
                    },
                    "hotkey": legacy_hotkey,
                }
                upgraded["sensor"] = sensor

        return LoadUiConfig._merge_dict_with_defaults(defaults, upgraded)

    @staticmethod
    def get_serial_discrete_input_config_path():
        return DEFAULT_DIR + "configs/scanner_barcode_config/serial_discrete_input.json"

    @staticmethod
    def load_serial_discrete_input_config():
        err_code, data = LoadUiConfig.load_data_from_json(LoadUiConfig.get_serial_discrete_input_config_path())
        if err_code != error_code.OK or not isinstance(data, dict):
            return err_code, data
        merged = LoadUiConfig.normalize_serial_discrete_input_config(data)
        return error_code.OK, merged

    @staticmethod
    def save_serial_discrete_input_config(config_data):
        file_path = LoadUiConfig.get_serial_discrete_input_config_path()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            base_config = LoadUiConfig.get_default_serial_discrete_input_config()
            existing_err_code, existing_data = LoadUiConfig.load_data_from_json(file_path)
            if existing_err_code == error_code.OK and isinstance(existing_data, dict):
                base_config = LoadUiConfig.normalize_serial_discrete_input_config(existing_data)
            merged_config = LoadUiConfig._merge_dict_with_defaults(
                base_config,
                config_data if isinstance(config_data, dict) else {},
            )
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(merged_config, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False


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

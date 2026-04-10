import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


DEFAULT_DIR = os.path.split(os.path.realpath(__file__))[0].replace("\\", "/") + "/../"
# DEFAULT_DIR = os.path.dirname(os.path.realpath(sys.argv[0])).replace("\\", "/") + "/"

MODEL_PATH = DEFAULT_DIR + "models/model_name.keras"
CONFIG_PATH = "configs/ai_model_config/config.yml"
TRAIN_PATH = DEFAULT_DIR + "audio_data/train"
TEST_PATH = DEFAULT_DIR + "audio_data/test"
TRAIN_OK_PATH = DEFAULT_DIR + "audio_data/train/OK"
TRAIN_NG_PATH = DEFAULT_DIR + "audio_data/train/NG"
TEST_OK_PATH = DEFAULT_DIR + "audio_data/test/OK"
TEST_NG_PATH = DEFAULT_DIR + "audio_data/test/NG"
STORED_SAMPLE_PATH = DEFAULT_DIR + "audio_data/stored_sample"
STORED_RECORDED_PATH = DEFAULT_DIR + "audio_data/stored_data"
STORED_RECORDED_UNLABELED_PATH = DEFAULT_DIR + "audio_data/stored_data/not_labeled"
STORED_RECORDED_OK_PATH = DEFAULT_DIR + "audio_data/stored_data/OK"
STORED_RECORDED_NG_PATH = DEFAULT_DIR + "audio_data/stored_data/NG"
STORED_PACKAGE_PATH = DEFAULT_DIR + "audio_data/stored_data/package"
STORED_STIMULUS_PATH = DEFAULT_DIR + "audio_data/stimulus"
JSON_DIR_PATH = DEFAULT_DIR + "ui/ui_config"

DATABASE_PATH = DEFAULT_DIR + "database/audio_data.db"

SAMPLE_RATE = 44100
POSITIVE_SAMPLE_LABEL = "OK"
NEGATIVE_SAMPLE_LABEL = "NG"
DB_AUDIO_COLUMNS = [
    "audio_data_id",
    "file_path",
    "product_model",
    "sample_rate",
    "record_date",
    "labels",
    "barcode",
    "stimulus_id",
]
DB_STIMULUS_COLUMNS = [
    "stimulus_id",
    "stimulus_method",
    "stimulus_type",
    "repeat_times",
    "start_freq",
    "stop_freq",
    "sample_rate",
    "total_time",
    "num_steps",
    "voltage_type",
    "voltage",
    "is_default",
    "stimulus_name",
]
DB_MODEL_COLUMNS = [
    "model_id",
    "model_name",
    "model_path",
    "config_path",
    "input_dim",
    "output_dim",
    "accuracy",
    "update_date",
    "model_description",
]
DB_USERS_COLUMNS = ["user_id", "user_name", "password", "access_level", "user_created_time", "user_updated_time"]
AUDIO_COLUMNS = [col for col in DB_AUDIO_COLUMNS if col != "audio_data_id"]
STIMULUS_COLUMNS = [col for col in DB_STIMULUS_COLUMNS if col not in ["stimulus_id", "stimulus_name"]]
STIMULUS_CONFIG_COLUMNS = [
    col for col in DB_STIMULUS_COLUMNS if col not in ["stimulus_id", "stimulus_name", "is_default"]
]
INERT_STIMULUS_CONFIG_COLUMNS = [col for col in DB_STIMULUS_COLUMNS if col not in ["stimulus_id"]]
MODEL_COLUMNS = [col for col in DB_MODEL_COLUMNS if col not in ["model_id", "update_date", "model_description"]]
INSERT_USERS_COLUMNS = [col for col in DB_USERS_COLUMNS if col != "user_id"]
USERS_COLUMNS = ["user_name", "password", "access_level"]
SELECT_COLUMNS = [
    "file_path",
    "product_model",
    "record_date",
    "stimulus_method",
    "stimulus_type",
    "total_time",
    "audio_data_table.stimulus_id",
    "labels",
]

LABEL_MAP = {
    "NG": 0,
    "OK": 1,
}

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


DEFAULT_DIR = os.path.split(os.path.realpath(__file__))[0].replace("\\", "/") + "/../"

CONFIG_PATH = DEFAULT_DIR + "config.yml"
TRAIN_PATH = DEFAULT_DIR + "audio_data/train"
TEST_PATH = DEFAULT_DIR + "audio_data/test"
TRAIN_OK_PATH = DEFAULT_DIR + "audio_data/train/OK"
TRAIN_NG_PATH = DEFAULT_DIR + "audio_data/train/NG"
TEST_OK_PATH = DEFAULT_DIR + "audio_data/test/OK"
TEST_NG_PATH = DEFAULT_DIR + "audio_data/test/NG"
STORED_SAMPLE_PATH = DEFAULT_DIR + "audio_data/stored_sample"

DATABASE_PATH = "database/audio_data.db"

SAMPLE_RATE = 44100
POSITIVE_SAMPLE_LABEL = "OK"
NEGATIVE_SAMPLE_LABEL = "NG"
DB_AUDIO_COLUMNS = ['audio_data_id', 'file_path', 'product_model', 'sample_rate', 'record_date', 'labels', 'stimulus_id']
DB_STIMULUS_COLUMNS = ['stimulus_id', 'sweep_method', 'sweep_type', 'repeats', 'start_feq', 'end_feq', 'sample_rate', 'sweep_duration']
AUDIO_DATA_TABLE_COLUMNS = ', '.join(DB_AUDIO_COLUMNS)
STIMULUS_SIGNAL_TABLE_COLUMNS = ', '.join(DB_STIMULUS_COLUMNS)
AUDIO_COLUMNS = DB_AUDIO_COLUMNS.remove("audio_data_id")
STIMULUS_COLUMNS = DB_STIMULUS_COLUMNS.remove("stimulus_id")
SELECT_COLUMNS = ', '.join(['file_path', 'product_model', 'record_date', 'sweep_method', 'sweep_type', 'sweep_duration', 'audio_data_table.stimulus_id', 'labels'])

LABEL_MAP = {
    "NG": 0,
    "OK": 1,
}

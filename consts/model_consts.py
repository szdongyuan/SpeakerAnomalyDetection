import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from machine_learning.cnn import CNN1d
from machine_learning.rnn import RNN
from machine_learning.svc import EnsembleSVC, SimpleSVC
from machine_learning.transformer import Transformer

DEFAULT_DIR = os.path.split(os.path.realpath(__file__))[0].replace("\\", "/") + "/../"

CONFIG_PATH = DEFAULT_DIR + "config.yml"
TRAIN_PATH = DEFAULT_DIR + "audio_data/train"
TEST_PATH = DEFAULT_DIR + "audio_data/test"
TRAIN_OK_PATH = DEFAULT_DIR + "audio_data/train/OK"
TRAIN_NG_PATH = DEFAULT_DIR + "audio_data/train/NG"
TEST_OK_PATH = DEFAULT_DIR + "audio_data/test/OK"
TEST_NG_PATH = DEFAULT_DIR + "audio_data/test/NG"
STORED_SAMPLE_PATH = DEFAULT_DIR + "audio_data/stored_sample"

SAMPLE_RATE = 44100
POSITIVE_SAMPLE_LABEL = "OK"
NEGATIVE_SAMPLE_LABEL = "NG"
AUDIO_DATA_TABLE_COLUMNS = ', '.join(['audio_data_id', 'file_path', 'product_model',
                                      'sample_rate', 'recode_date', 'labels'])
SWEEP_SIGNAL_TABLE_COLUMNS = ', '.join(['sweep_id', 'sweep_type', 'start_feq', 'end_feq',
                                        'sample_rate', 'sweep_duration'])


MODEL_MAPPING = {
    "EnsembleSVC": EnsembleSVC,
    "SimpleSVC": SimpleSVC,
    "CNN1d": CNN1d,
    "RNN": RNN,
    "Transformer": Transformer
}

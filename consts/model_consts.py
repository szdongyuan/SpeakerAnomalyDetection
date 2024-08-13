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

LABEL_MAP = {
    "NG": 0,
    "OK": 1,
}

MODEL_MAPPING = {
    "EnsembleSVC": EnsembleSVC,
    "SimpleSVC": SimpleSVC,
    "CNN1d": CNN1d,
    "RNN": RNN,
    "Transformer": Transformer
}

import numpy  as np


class DataDealStruct(object):
    _instance = None
    @classmethod
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DataDealStruct, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.fft_result = None
            self.stft_result = None

            self.store_wave_data = None
            self.stimulus_data = None
            self.stimulus_info = None

            self.hd_flag = 0
            self.fr_flag = 0
            # self.ai_flag = 0

            self._initialized = True

    def clear_data(self):
        self.store_wave_data = None
        # self.stimulus_data = None
        self.fft_result = None
        self.stft_result = None
        self.hd_flag = 0
        self.fr_flag = 0
        # self.ai_flag = 0

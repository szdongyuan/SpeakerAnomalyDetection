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
            self.split_repeat_data = None
            self.stimulus_data = None
            self.stimulus_info = None
            self.sample_rate = None
            self.num_channels = 1  # Track channel count (1-4 supported)

            self.stft_flag = 0
            self.fft_flag = 0
            
            # multiple PD instances result mapping: { analysis_key(str): [idx1, idx2, ...] }
            self.pd_peak_grid_points_map = {} 

            self._initialized = True

    def clear_data(self):
        self.store_wave_data = None
        self.split_repeat_data = None
        self.num_channels = 1  # Reset to default
        # self.stimulus_data = None
        self.fft_result = None
        self.stft_result = None

        # self.hd_flag = 0
        # self.fr_flag = 0
        # self.ai_flag = 0

    def add_stft_or_fft_count(self, text):
        if text in ["频响 (FR) ", "谐波失真 (HD) ", "FR", "HD"]:
            self.fft_flag += 1
        elif text in ["频谱分析 (Spec) ", "Spec"]:
            self.stft_flag += 1

    def minus_stft_or_fft_count(self, text):
        if text in ["频响 (FR) ", "谐波失真 (HD) ", "FR", "HD"]:
            self.fft_flag -= 1
        elif text in ["频谱分析 (Spec) ", "Spec"]:
            self.stft_flag -= 1

    def clear_fft_and_stft_flag(self):
        self.fft_flag = 0
        self.stft_flag = 0

    def update_channel_count(self):
        """Automatically update num_channels based on store_wave_data shape."""
        if self.store_wave_data is not None:
            if self.store_wave_data.ndim == 1:
                self.num_channels = 1
            elif self.store_wave_data.ndim == 2:
                self.num_channels = self.store_wave_data.shape[1]
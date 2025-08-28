import os
from datetime import datetime

from base.data_struct.data_deal_struct import DataDealStruct
from base.system_intervction.hardware_intervction import get_mac_address
from base.pre_processing.split_repeat_signal import SplitRepeatSignal
from base.recording_management import RecordingManager
from base.save_data import save_audio_simple
from base.soundcard_audio_processor import SoundcardAudioProcessor
from consts import error_code, model_consts

data_struct = DataDealStruct()


def record_without_play(recorded_dict, recorded_path, recorded_signal_info):
    """
    Implements the complete workflow for the record-only mode.
    It records audio based on the `sample_rate` specified in`data_struct`, then
    saves the result to a .wav file and the database (with `stimulus_id` set to
    0). All subsequent operations,such as FFT/STFT, automatic analysis, and
    button state updates, are the same as in the play-and-record mode.
    """

    sample_rate = data_struct.sample_rate
    record_code, recorded_signal = SoundcardAudioProcessor.sd_rec(recorded_dict)

    if record_code == error_code.OK:
        save_audio_simple(recorded_path, recorded_signal, sample_rate)

        data_struct.store_wave_data = recorded_signal

        recorded_signal_info["sample_rate"] = sample_rate
        RecordingManager().save_signal_info_to_db(recorded_signal_info, None)


def play_last_stimulus_wave(stimulus_dict, recorded_dict, recorded_path, recorded_signal_info):
    """
    Implements the complete workflow for the play and record mode.:
    1. Obtains the sample rate and generates dictionaries for the stimulus and recorded signals.
    2. Uses the soundcard audio processor to play the stimulus signal and record the response signal.
    3. If recording is successful, plots the recorded signal on the line graph and saves the signal information.
    4. Enables the data button and the replay button.
    """
    sample_rate = data_struct.sample_rate

    sap = SoundcardAudioProcessor()
    record_code, data_struct.store_wave_data = sap.sd_play_rec(recorded_dict, stimulus_dict, recorded_path)
    if record_code == error_code.OK:
        recorded_signal_info["sample_rate"] = sample_rate
        RecordingManager().save_signal_info_to_db(recorded_signal_info, data_struct.stimulus_info)

    repeat_times = data_struct.stimulus_info.get("repeat_times")
    if repeat_times > 1:
        kwargs = {"repeat_times": repeat_times}
        data_struct.split_repeat_data = SplitRepeatSignal().split_repeat_signal(
            data_struct.store_wave_data, sample_rate, **kwargs
        )


def get_recorded_info(product_model, product_number, label):
    """
        Generate recorded information.

        This function generates a unique recording file name based on the current date, MAC address, product model,
    and product number.
        It also constructs the path for the recording file. Additionally, it creates a dictionary containing the
    recording file path and product information.

        Returns:
            tuple: A tuple containing the recording file path and a dictionary with recording information.
    """
    recording_time = datetime.now().strftime("%Y-%m-%d")
    mac_address = get_mac_address()
    mac_address = mac_address.replace(":", "") if mac_address else None
    product_number = "{:03}".format(int(product_number))
    recorded_name = product_model + "_" + recording_time + "_" + mac_address + "_" + product_number
    recorded_name = recorded_name + ".wav"
    store_record_dir = model_consts.STORED_RECORDED_PATH + "/" + label
    if not os.path.exists(store_record_dir):
        os.makedirs(store_record_dir)
    recorded_path = store_record_dir + "/" + recorded_name
    recorded_signal_info = {
        "file_path": recorded_path,
        "product_model": product_model,
        "record_date": recording_time,
        "barcode": None,
        "labels": label,
    }

    return recorded_path, recorded_signal_info

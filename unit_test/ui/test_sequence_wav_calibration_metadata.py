

from base.data_struct.data_deal_struct import DataDealStruct


def test_data_struct_initializes_and_clears_wav_metadata_state():
    previous_instance = DataDealStruct._instance
    try:
        DataDealStruct._instance = None
        data_struct = DataDealStruct()

        assert data_struct.wav_calibration_metadata is None
        assert data_struct.wav_calibration_metadata_authoritative is False
        assert data_struct.wav_calibration_warning_shown is False

        data_struct.wav_calibration_metadata = {"recorded_channels": []}
        data_struct.wav_calibration_metadata_authoritative = True
        data_struct.wav_calibration_warning_shown = True

        data_struct.clear_data()

        assert data_struct.wav_calibration_metadata is None
        assert data_struct.wav_calibration_metadata_authoritative is False
        assert data_struct.wav_calibration_warning_shown is False
    finally:
        DataDealStruct._instance = previous_instance

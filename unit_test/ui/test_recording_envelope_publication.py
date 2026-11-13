"""Shared recorded-source state clears without the retired UI publication host."""
from types import SimpleNamespace
from unittest import mock


def test_data_reset_clears_saved_and_display_provenance():
    from base.data_struct.data_deal_struct import DataDealStruct
    data = SimpleNamespace(recording_source=object(), recording_sources={"old": object()},
                           clear_wav_calibration_context=mock.Mock())
    DataDealStruct.clear_data(data)
    assert data.recording_source is None
    assert data.recording_sources == {}

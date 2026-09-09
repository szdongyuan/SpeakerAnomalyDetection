"""PR1C extraction regression: frozen completion config reaches real analysis."""
import numpy as np
from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin
from unit_test.ui.test_sequence_channel_calibration import StreamingAnalysisHost


def test_recording_run_routes_frozen_configuration_through_real_analysis(monkeypatch):
    host = StreamingAnalysisHost()
    current = {"display_sequence": ["changed"], "changed": {"type": "UNKNOWN"}}
    frozen = {"display_sequence": ["captured"], "captured": {"type": "SPL", "analysis_channel": 0}}
    host.analysis_config = current
    host.data_struct.store_wave_data = np.array([.1, -.1], dtype=np.float32)
    host.data_struct.store_wave_data_multi = host.data_struct.store_wave_data[:, None]
    monkeypatch.setattr("ui.sequence.sequence_widget_analysis_ops.load_mic_channel_v2pa_factors", lambda device: {0: 1.0})
    SequenceWidgetAnalysisOpsMixin.run(host, show_windows=False,
        capture_product_report=False, analysis_config_override=frozen)
    assert [item._sequence_analysis_key for item in host.analysis_window] == ["captured"]
    assert "calculate" in host.events
    assert host.analysis_config is current

"""Current scanner preference restore and shared-file merge contracts."""
import json
from unittest.mock import Mock, call
from datetime import date

import pytest

from ui.sequence import workbench_input_ops as input_ops
from unit_test.ui.test_workbench_input_ops import host


@pytest.mark.parametrize('payload, enabled', [
    ({'scanner_barcode_check': True}, True),
    ({'scanner_barcode_check': 'true'}, True),
    ({'scanner_barcode_check': 'invalid'}, False),
    (None, False), ([1], False), ('invalid', False),
])
def test_restore_scanner_checkbox_without_persisting(host, monkeypatch, payload, enabled):
    saved = Mock()
    monkeypatch.setattr(input_ops.LoadUiConfig, 'load_last_recorded_info', lambda _: payload)
    monkeypatch.setattr(input_ops, 'save_recorded_data_to_json', saved)
    host.lineedit_s_or_n.setText('SN001')
    host.hw_manager.calls.clear()
    host.restore_scanner_checkbox_state()
    assert host.barcode_scanner_box.isChecked() is enabled
    assert host.lineedit_s_or_n.isEnabled() is enabled
    assert ('start' if enabled else 'stop') in host.hw_manager.calls
    if not enabled:
        assert host.lineedit_s_or_n.text() == ''
    saved.assert_not_called()


def test_clicked_scanner_persists_only_checkbox_state(host, monkeypatch):
    saved = Mock()
    monkeypatch.setattr(input_ops, 'save_recorded_data_to_json', saved)
    for enabled in (True, False):
        host.barcode_scanner_box.setChecked(enabled)
        host.clicked_scanner()
    assert saved.call_args_list == [
        call(scanner_barcode_check=True),
        call(scanner_barcode_check=False),
    ]


def test_save_recorded_data_to_json_merge_preserves_existing_fields(tmp_path, monkeypatch):
    from base import save_data
    monkeypatch.setattr(save_data, 'DEFAULT_DIR', str(tmp_path) + '/')
    path = tmp_path / 'ui/ui_config/recorded_number.json'
    path.parent.mkdir(parents=True)
    original = {'scanner_barcode_check': True, 'sequence_mode': 'mark',
                'sample_number': 'SAMPLE-001', 'current_test_round': 7, 'unknown': {'keep': 1}}
    path.write_text(json.dumps(original), encoding='utf-8')
    save_data.save_recorded_data_to_json(product_model='MODEL_A', scanner_barcode='SN1')
    assert json.loads(path.read_text()) == {**original, 'product_model': 'MODEL_A', 'scanner_barcode': 'SN1', 'datetime': date.today().isoformat()}

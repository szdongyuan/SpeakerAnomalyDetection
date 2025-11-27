import pytest
from ui.signal_analysis_window import get_class_mapping, RubAndBuzz


def test_class_mapping_includes_rb():
    """Verify class mapping includes RB -> RubAndBuzz"""
    mapping = get_class_mapping()

    assert "RB" in mapping
    assert mapping["RB"] == RubAndBuzz


def test_class_mapping_preserves_existing():
    """Verify adding RB doesn't break existing mappings"""
    mapping = get_class_mapping()

    # Original mappings should still exist
    expected_keys = {"SPL", "FR", "HD", "AI", "Spec", "LP", "PD", "PM", "ED", "RB"}
    assert set(mapping.keys()) == expected_keys

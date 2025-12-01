import pytest
from ui.signal_analysis_window import get_class_mapping, PerceptualRubAndBuzz


def test_class_mapping_includes_prb():
    """Verify class mapping includes PRB -> PerceptualRubAndBuzz"""
    mapping = get_class_mapping()

    assert "PRB" in mapping
    assert mapping["PRB"] == PerceptualRubAndBuzz


def test_class_mapping_preserves_existing():
    """Verify adding PRB doesn't break existing mappings"""
    mapping = get_class_mapping()

    # Original mappings should still exist
    expected_keys = {"SPL", "FR", "HD", "RB", "PRB", "AI", "Spec", "LP", "PD", "PM", "ED"}
    assert set(mapping.keys()) == expected_keys

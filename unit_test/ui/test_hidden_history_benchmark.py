import pytest

from unit_test.ui.benchmark_hidden_history import normalize, verify_equivalent


def test_equivalence_ignores_only_private_metadata_and_sequence_container_type():
    before = {"records": [{"result": "OK", "config": {"channels": (1, 2)}}]}
    after = {"records": [{"result": "OK", "config": {"channels": [1, 2]},
                          "_recent_group_metadata": {"time_text": "first"}}]}
    verify_equivalent(before, after)
    assert normalize(before) == normalize(after)


def test_equivalence_ignores_retired_automatic_pdf_fields():
    before = {"records": [{"result": "OK", "analysis_report_state": "not_required",
                           "analysis_report_items": []}]}
    after = {"records": [{"result": "OK"}]}
    verify_equivalent(before, after)


@pytest.mark.parametrize("field", ["result", "time_text", "config_snapshot", "analysis_result_dict", "segment_results"])
def test_equivalence_rejects_changed_business_fields(field):
    with pytest.raises(AssertionError, match="semantics differ"):
        verify_equivalent({field: "before"}, {field: "after"})

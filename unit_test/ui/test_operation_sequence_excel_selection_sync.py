from consts.excel_export_consts import SAVE_ITEM_OUTPUTS_KEY
from ui.operation_sequence import (
    _delete_excel_selection_item,
    _rename_excel_selection_item,
)


def _analysis_list(mapping=...):
    excel_config = {
        "type": "Excel",
        "save_items": ["SPLF 1", "FR 1"],
    }
    if mapping is not ...:
        excel_config[SAVE_ITEM_OUTPUTS_KEY] = mapping
    return {
        "Excel": excel_config,
        "SPLF 1": {"type": "SPLF"},
        "FR 1": {"type": "FR"},
    }


def test_excel_selection_rename_preserves_output_list_exactly():
    outputs = ["test_curve", "margin"]
    analysis_list = _analysis_list(
        {
            "SPLF 1": outputs,
            "FR 1": ["deviation"],
        }
    )

    _rename_excel_selection_item(analysis_list, "SPLF 1", "SPLF renamed")

    excel_config = analysis_list["Excel"]
    assert excel_config["save_items"] == ["SPLF renamed", "FR 1"]
    assert excel_config[SAVE_ITEM_OUTPUTS_KEY] == {
        "SPLF renamed": outputs,
        "FR 1": ["deviation"],
    }
    assert excel_config[SAVE_ITEM_OUTPUTS_KEY]["SPLF renamed"] is outputs


def test_excel_selection_delete_removes_both_compatibility_structures():
    analysis_list = _analysis_list(
        {
            "SPLF 1": ["test_curve", "margin"],
            "FR 1": ["deviation"],
        }
    )

    _delete_excel_selection_item(analysis_list, "FR 1")

    excel_config = analysis_list["Excel"]
    assert excel_config["save_items"] == ["SPLF 1"]
    assert excel_config[SAVE_ITEM_OUTPUTS_KEY] == {
        "SPLF 1": ["test_curve", "margin"]
    }


def test_excel_selection_sync_tolerates_absent_or_malformed_optional_mapping():
    absent_mapping = _analysis_list()
    malformed_mapping = _analysis_list(["not", "a", "mapping"])

    _rename_excel_selection_item(absent_mapping, "SPLF 1", "renamed")
    _delete_excel_selection_item(absent_mapping, "FR 1")
    _rename_excel_selection_item(malformed_mapping, "SPLF 1", "renamed")
    _delete_excel_selection_item(malformed_mapping, "FR 1")

    assert absent_mapping["Excel"]["save_items"] == ["renamed"]
    assert malformed_mapping["Excel"]["save_items"] == ["renamed"]
    assert SAVE_ITEM_OUTPUTS_KEY not in absent_mapping["Excel"]
    assert malformed_mapping["Excel"][SAVE_ITEM_OUTPUTS_KEY] == [
        "not",
        "a",
        "mapping",
    ]

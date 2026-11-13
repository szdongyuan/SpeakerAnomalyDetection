"""Shared backend judgement aggregation and audio storage labels."""

import pytest

from base.audio_label_policy import aggregate_judgements, audio_label_for_results


@pytest.mark.parametrize("values,expected", [
    ([], "--"),
    (["OK", "OK"], "OK"),
    (["OK", "NG", None], "NG"),
    ([None, "OK"], "OK"),
    ([None, "--", "failed"], "--"),
    ([None, 0, False, {}, [], "", "unknown"], "--"),
    ([" ok ", "\tOK\n"], "OK"),
    (["\tng\n", "ok"], "NG"),
    ([" ", "\t", "--"], "--"),
])
def test_aggregate_judgements_uses_ng_then_any_ok(values, expected):
    assert aggregate_judgements(iter(values)) == expected


@pytest.mark.parametrize("items,expected", [
    ([], "not_labeled"),
    ([{"judgement": "OK"}, {"judgement": "OK"}], "OK"),
    ([{"judgement": "OK"}, {"judgement": "NG"}], "NG"),
    ([{"judgement": "OK"}, {"judgement": None, "execution_status": "分析失败"}], "OK"),
    ([{"judgement": None, "execution_status": "分析失败"}], "not_labeled"),
    ([{}, {"judgement": "unknown"}], "not_labeled"),
    ([{"judgement": " ok ", "judgement_status": "unavailable",
       "execution_status": "分析失败", "error": {"stage": "csv_save"}}], "OK"),
    ([{"judgement": " nG\n"}, {"judgement": "ok"}], "NG"),
])
def test_audio_labels_use_only_judgements_and_map_unknown_at_storage_boundary(items, expected):
    assert audio_label_for_results(iter(items)) == expected

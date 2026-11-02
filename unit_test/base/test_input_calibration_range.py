"""The calibration display reads only the current queue's voltage range."""
import copy
from collections import UserDict
from types import MappingProxyType

import pytest

from base.input_calibration_range import resolve_test_full_scale_peak_volts


def _queue(detail):
    return [{"seq1": {"acq": {"detail": detail}}}]


@pytest.mark.parametrize("index,expected", enumerate((10., 5., 2.5, 1., .5, .1, .02)))
def test_supported_queue_ranges_are_read_without_mutation(index, expected):
    queue = _queue({
        "ve_range_index": index,
        "sample_rate": "broken",
        "range_min": "stale",
        "range_max": None,
    })
    before = copy.deepcopy(queue)

    assert resolve_test_full_scale_peak_volts(queue) == expected
    assert queue == before


def test_valid_empty_detail_uses_the_default_range_without_mutation():
    queue = _queue({})
    before = copy.deepcopy(queue)

    assert resolve_test_full_scale_peak_volts(queue) == 10.
    assert queue == before


@pytest.mark.parametrize("queue", [
    None,
    [],
    (),
    tuple(_queue({})),
    {"seq1": {"acq": {"detail": {}}}},
    "bad",
    [None],
    [[]],
    ["bad"],
    [{}],
    [{"seq1": None}],
    [{"seq1": []}],
    [{"seq1": "bad"}],
    [{"seq1": {}}],
    [{"seq1": {"acq": None}}],
    [{"seq1": {"acq": []}}],
    [{"seq1": {"acq": "bad"}}],
    [{"seq1": {"acq": {}}}],
    _queue(None),
    _queue([]),
    _queue("bad"),
])
def test_missing_or_malformed_queue_is_not_a_default_range(queue):
    before = copy.deepcopy(queue)

    assert resolve_test_full_scale_peak_volts(queue) is None
    assert queue == before


@pytest.mark.parametrize("index", [True, False, "2", 2.0, -1, 7, None])
def test_explicit_invalid_range_is_not_defaulted(index):
    queue = _queue({"ve_range_index": index})
    before = copy.deepcopy(queue)

    assert resolve_test_full_scale_peak_volts(queue) is None
    assert queue == before


@pytest.mark.parametrize("wrap", [dict, UserDict, MappingProxyType])
def test_nested_mapping_snapshots_are_supported(wrap):
    queue = [wrap({"seq1": wrap({"acq": wrap({
        "detail": wrap({"ve_range_index": 5}),
    })})})]

    assert resolve_test_full_scale_peak_volts(queue) == .1


def test_only_the_first_loaded_sequence_supplies_the_range():
    queue = _queue({"ve_range_index": 6}) + _queue({"ve_range_index": 0})
    before = copy.deepcopy(queue)

    assert resolve_test_full_scale_peak_volts(queue) == .02
    assert queue == before

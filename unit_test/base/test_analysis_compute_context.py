"""Task-local reuse contracts, independent of any numerical algorithm."""

import weakref

import pytest

from base.analysis_compute_context import ComputationFailure, TaskComputeContext


def test_reuses_result_and_clear_releases_only_context_reference():
    class Result:
        pass

    context = TaskComputeContext()
    result = context.get_or_compute(("frontend", 1), Result)
    reference = weakref.ref(result)
    assert context.get_or_compute(("frontend", 1), Result) is result
    assert (context.hits, context.misses, context.compute_count) == (1, 1, 1)
    context.clear()
    assert len(context) == 0 and reference() is result
    del result
    assert reference() is None
    assert context.lookup_seconds >= 0 and context.compute_seconds >= 0


def test_expected_failure_is_data_and_next_context_retries():
    calls = []

    def fail():
        calls.append(1)
        raise ValueError("invalid input")

    context = TaskComputeContext()
    for _ in range(2):
        failure = context.get_or_compute("bad", fail, failure_types=(ValueError,))
        assert failure == ComputationFailure("ValueError", "calculation", "invalid input")
        assert not isinstance(failure, BaseException)
    assert len(calls) == 1
    TaskComputeContext().get_or_compute("bad", fail, failure_types=(ValueError,))
    assert len(calls) == 2


@pytest.mark.parametrize("error", [KeyError("bug"), KeyboardInterrupt(), SystemExit()])
def test_unexpected_errors_and_termination_are_not_cached(error):
    def fail():
        raise error

    context = TaskComputeContext()
    with pytest.raises(type(error)):
        context.get_or_compute("bad", fail, failure_types=(ValueError,))
    assert len(context) == 0

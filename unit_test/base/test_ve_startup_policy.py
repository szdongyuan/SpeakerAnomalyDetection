"""Pure startup budget contracts; no SDK, thread, or wall-clock dependencies."""
import pickle

import pytest

from base.ve_startup_policy import VeStartupBudget


def test_default_budget_is_absolute_frozen_and_serializable():
    budget = VeStartupBudget.create(100.0)
    assert budget == VeStartupBudget(100.0, 103.5, 105.0, 110.0)
    assert pickle.loads(pickle.dumps(budget)) == budget
    with pytest.raises(AttributeError):
        budget.deadline = 120.0
    assert budget.remaining(104.0) == 6.0
    assert budget.remaining(105.0) == 5.0
    assert budget.first_attempt_remaining(102.0) == 1.5
    assert budget.cleanup_remaining(104.0) == 1.0


@pytest.mark.parametrize("total,first,cleanup,expected", [
    (2.0, 3.5, 1.5, (102.0, 102.0, 102.0)),
    (4.0, 3.5, 1.5, (103.5, 104.0, 104.0)),
    (100.0, 3.5, 1.5, (103.5, 105.0, 110.0)),
    (1.0, .2, .3, (100.2, 100.5, 101.0)),
    (0.0, 0.0, 0.0, (100.0, 100.0, 100.0)),
    (10.0, 20.0, 20.0, (110.0, 110.0, 110.0)),
])
def test_create_clips_stage_deadlines_to_shared_total(total, first, cleanup, expected):
    budget = VeStartupBudget.create(100.0, total, first, cleanup)
    assert (budget.first_attempt_deadline, budget.cleanup_deadline, budget.deadline) == expected


@pytest.mark.parametrize("method,deadline", [
    ("remaining", 110.0), ("first_attempt_remaining", 103.5),
    ("cleanup_remaining", 105.0),
])
def test_remaining_clamps_at_and_after_boundary(method, deadline):
    remaining = getattr(VeStartupBudget.create(100.0), method)
    assert remaining(deadline) == 0.0
    assert remaining(deadline + 100.0) == 0.0


@pytest.mark.parametrize("field", [
    "started_at", "total_timeout", "first_attempt_timeout", "cleanup_timeout",
])
@pytest.mark.parametrize("value", [-1, True, False, float("nan"), float("inf"),
                                  -float("inf"), "1", None])
def test_create_rejects_invalid_times(field, value):
    values = dict(started_at=100.0)
    values[field] = value
    with pytest.raises(ValueError, match=field):
        VeStartupBudget.create(**values)


@pytest.mark.parametrize("field", [
    "started_at", "first_attempt_deadline", "cleanup_deadline", "deadline",
])
@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf"), "1", None])
def test_direct_constructor_rejects_invalid_times(field, value):
    values = dict(started_at=100.0, first_attempt_deadline=103.5,
                  cleanup_deadline=105.0, deadline=110.0)
    values[field] = value
    with pytest.raises(ValueError, match=field):
        VeStartupBudget(**values)


@pytest.mark.parametrize("values", [
    (100.0, 99.0, 105.0, 110.0), (100.0, 106.0, 105.0, 110.0),
    (100.0, 103.5, 111.0, 110.0), (100.0, 103.5, 105.0, 110.01),
])
def test_direct_constructor_rejects_disorder_and_expanded_total(values):
    with pytest.raises(ValueError):
        VeStartupBudget(*values)


@pytest.mark.parametrize("method", ["remaining", "first_attempt_remaining", "cleanup_remaining"])
@pytest.mark.parametrize("now", [-1, True, float("nan"), float("inf"), "1", None])
def test_remaining_rejects_invalid_clock_values(method, now):
    with pytest.raises(ValueError, match="now"):
        getattr(VeStartupBudget.create(100.0), method)(now)

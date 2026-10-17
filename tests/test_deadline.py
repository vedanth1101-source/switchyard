import math

import pytest

from app.core.deadline import Deadline


class FakeClock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_remaining_decreases_with_fake_clock():
    clock = FakeClock()
    deadline = Deadline(50, clock=clock)

    assert deadline.remaining() == 50
    clock.advance(12.5)
    assert deadline.remaining() == 37.5


def test_not_expired_before_budget():
    clock = FakeClock()
    deadline = Deadline(50, clock=clock)

    clock.advance(49.5)

    assert not deadline.expired()
    assert deadline.remaining() == 0.5


def test_expired_after_budget():
    clock = FakeClock()
    deadline = Deadline(50, clock=clock)

    clock.advance(50)
    assert deadline.expired()
    assert deadline.remaining() == 0

    clock.advance(10)
    assert deadline.expired()
    assert deadline.remaining() == 0  # never negative


def test_budget_counts_from_construction_not_clock_zero():
    clock = FakeClock(now=5_000.0)
    clock.advance(100)

    deadline = Deadline(10, clock=clock)

    assert deadline.remaining() == 10


def test_default_clock_is_monotonic_time():
    deadline = Deadline(50)

    assert 0 < deadline.remaining() <= 50
    assert not deadline.expired()


@pytest.mark.parametrize("budget", [0, -1, math.nan, math.inf], ids=["zero", "negative", "nan", "inf"])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="budget"):
        Deadline(budget, clock=FakeClock())

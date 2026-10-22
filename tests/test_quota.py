import pytest

from src.api.quota import QuestionQuota, QuotaExceededError


class FakeClock:
    def __init__(self, start=1_800_000_000.0):  # a fixed UTC moment
        self.now = start

    def __call__(self):
        return self.now


def test_per_minute_limit_blocks_then_recovers():
    clock = FakeClock()
    quota = QuestionQuota(per_minute=2, per_day=0, clock=clock)
    quota.acquire()
    quota.acquire()
    with pytest.raises(QuotaExceededError) as raised:
        quota.acquire()
    assert raised.value.scope == "per minute"
    assert 1 <= raised.value.retry_after_seconds <= 60

    clock.now += 61
    quota.acquire()  # the window slid: allowed again


def test_per_day_limit_resets_the_next_utc_day():
    clock = FakeClock()
    quota = QuestionQuota(per_minute=0, per_day=3, clock=clock)
    for _ in range(3):
        quota.acquire()
        clock.now += 120
    with pytest.raises(QuotaExceededError) as raised:
        quota.acquire()
    assert raised.value.scope == "per day"

    clock.now += 86_400
    quota.acquire()


def test_rejected_questions_are_not_counted():
    clock = FakeClock()
    quota = QuestionQuota(per_minute=1, per_day=2, clock=clock)
    quota.acquire()
    for _ in range(5):
        with pytest.raises(QuotaExceededError):
            quota.acquire()
    clock.now += 61
    quota.acquire()  # still within the daily limit of 2


def test_zero_disables_limits():
    quota = QuestionQuota(per_minute=0, per_day=0, clock=FakeClock())
    for _ in range(1_000):
        quota.acquire()

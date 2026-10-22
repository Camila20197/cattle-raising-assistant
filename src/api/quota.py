"""Global limits on how many questions reach the LLM.

The public demo has no login, so a bot could send thousands of questions.
Every question that reaches the LLM costs Cloud Run time and Groq quota.
This quota caps the total, whoever sends it: once a limit is reached,
/ask answers 429 immediately, without calling the LLM.

Limits live in memory, per running instance. With ``--max-instances 1`` on
Cloud Run they are exact; with N instances the effective cap is up to N
times higher, and a restart resets the counters. That is acceptable for a
cost safeguard (the goal is a ceiling, not precise accounting).
"""

import threading
import time
from collections import deque
from datetime import datetime, timezone


class QuotaExceededError(Exception):
    def __init__(self, scope: str, retry_after_seconds: int):
        self.scope = scope
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Question limit reached ({scope}). Retry in {retry_after_seconds}s.")


class QuestionQuota:
    """Sliding one-minute window plus a per-UTC-day counter. 0 disables a limit."""

    def __init__(self, per_minute: int, per_day: int, clock=time.time):
        self.per_minute = per_minute
        self.per_day = per_day
        self._clock = clock
        self._lock = threading.Lock()
        self._recent: deque[float] = deque()
        self._day = self._today()
        self._today_count = 0

    def _today(self) -> str:
        return datetime.fromtimestamp(self._clock(), tz=timezone.utc).date().isoformat()

    def _seconds_until_tomorrow(self) -> int:
        now = datetime.fromtimestamp(self._clock(), tz=timezone.utc)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() + 86_400
        return max(1, int(midnight - now.timestamp()))

    def acquire(self) -> None:
        """Count one question, or raise QuotaExceededError without counting it."""
        with self._lock:
            now = self._clock()

            today = self._today()
            if today != self._day:
                self._day, self._today_count = today, 0

            while self._recent and now - self._recent[0] >= 60:
                self._recent.popleft()

            if self.per_day and self._today_count >= self.per_day:
                raise QuotaExceededError("per day", self._seconds_until_tomorrow())
            if self.per_minute and len(self._recent) >= self.per_minute:
                raise QuotaExceededError("per minute", max(1, int(60 - (now - self._recent[0]))))

            self._recent.append(now)
            self._today_count += 1

"""Request time budget (TRD §6).

A Deadline starts counting when it is created. The clock is injectable so tests can advance time
without sleeping; the default, time.monotonic, is unaffected by wall-clock changes.
"""

import math
import time
from typing import Callable


class Deadline:
    def __init__(self, budget: float, clock: Callable[[], float] = time.monotonic):
        """
        Args:
            budget: Seconds available from now; must be finite and greater than 0.
            clock: Returns the current time in seconds.
        """
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError(f"Deadline budget must be a finite number of seconds > 0, got {budget!r}")
        self._clock = clock
        self._expires_at = clock() + budget

    def remaining(self) -> float:
        """Seconds left; 0.0 once expired, never negative."""
        return max(0.0, self._expires_at - self._clock())

    def expired(self) -> bool:
        """True once the clock reaches the end of the budget."""
        return self._clock() >= self._expires_at

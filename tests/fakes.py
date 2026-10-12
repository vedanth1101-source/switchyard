"""Deterministic fakes for unit tests: no network, no real sleeping, no wall-clock time."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

from app.core.llm_client import LLMError, ModelResult


class FakeClock:
    """Manually advanced clock; pass the same instance to Deadline, the executor and the fakes."""

    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordingSleep:
    """Async sleep replacement that records each delay and advances the clock instead of waiting."""

    def __init__(self, clock: FakeClock):
        self.clock = clock
        self.delays: List[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)
        self.clock.advance(seconds)


@dataclass(frozen=True)
class LLMCall:
    model: str
    prompt: str
    max_tokens: int
    timeout: float


Outcome = Union[ModelResult, LLMError, Exception]
ScriptStep = Union[Outcome, Tuple[Outcome, float]]


class FakeLLM:
    """
    LLMClient that replays a script, one step per call, in order.

    A step is an outcome (a ModelResult to return or an exception to raise), optionally paired
    with the seconds the call takes, which advance the clock. Running past the end of the script
    fails the test rather than inventing a result.
    """

    def __init__(self, script: Sequence[ScriptStep], clock: FakeClock):
        self._script = list(script)
        self._clock = clock
        self.calls: List[LLMCall] = []

    async def complete(self, model: str, prompt: str, max_tokens: int, timeout: float) -> ModelResult:
        self.calls.append(LLMCall(model=model, prompt=prompt, max_tokens=max_tokens, timeout=timeout))
        if len(self.calls) > len(self._script):
            raise AssertionError(f"FakeLLM script exhausted: unexpected call #{len(self.calls)} to {model}")
        step = self._script[len(self.calls) - 1]
        outcome, seconds = step if isinstance(step, tuple) else (step, 0.0)
        self._clock.advance(seconds)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def models_called(self) -> List[str]:
        return [call.model for call in self.calls]


class FakeJev:
    """Jev evaluator returning a copy of a fixed response (or raising it), optionally taking clock time."""

    def __init__(self, outcome: Union[dict, Exception], clock: FakeClock | None = None, seconds: float = 0.0):
        self.outcome = outcome
        self.clock = clock
        self.seconds = seconds
        self.calls: List[Tuple[str, dict]] = []

    async def evaluate(self, state: str, questions: dict) -> dict:
        self.calls.append((state, questions))
        if self.clock is not None:
            self.clock.advance(self.seconds)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return copy.deepcopy(self.outcome)


class FakeBudget:
    """StrongBudget with fixed answers that counts calls."""

    def __init__(self, available: bool = True, reserve: bool = True):
        self._available = available
        self._reserve = reserve
        self.available_calls = 0
        self.reserve_calls = 0

    async def available(self) -> bool:
        self.available_calls += 1
        return self._available

    async def reserve(self) -> bool:
        self.reserve_calls += 1
        return self._reserve

"""FakeBackend: deterministic, dependency-free executor for control-flow tests.

Token for a request at completion position `p` is `(req_id * 1000 + p) % vocab`.
If a token lands in `stop_token_ids`, the request finishes with "stop"; requests
finish with "length" at `max_new_tokens`. Behavior is a pure function of the plan,
so tests can assert exact schedules.
"""

from __future__ import annotations

from ..scheduler.scheduler import StepPlan
from ..state.slots import StateStore
from .base import StepResult


class FakeBackend:
    def __init__(self, vocab: int = 100, stop_token_ids: frozenset[int] = frozenset(), delay: float = 0.0):
        self.vocab = vocab
        self.stop_token_ids = stop_token_ids
        self.delay = delay
        self.executed: list[StepPlan] = []

    def execute(self, plan: StepPlan, state: StateStore) -> list[StepResult]:
        return self.resolve(self.enqueue(plan, state))

    def enqueue(self, plan: StepPlan, state: StateStore):
        """Split step contract: the plan runs now, its results are resolved later.

        The fake has no device work and no host read, so the handle is simply the
        deferred result list. What the engine tests need from the split is that the
        engine is allowed to hold one handle while it enqueues the next step, and
        that is what this makes possible.
        """
        import time

        if self.delay:
            time.sleep(self.delay)
        self.executed.append(plan)
        return self._results(plan)

    def resolve(self, pending) -> list[StepResult]:
        return pending

    def _results(self, plan: StepPlan) -> list[StepResult]:
        results = []
        for row in plan.rows:
            if plan.op == "prefill":
                # first sampled token after consuming the prompt
                p = len(row.prompt_tokens)
                tokens = (self._token(row.req_id, p),)
            else:
                p = row.positions[0]
                tokens = (self._token(row.req_id, p),)
            finish = None
            if tokens[-1] in self.stop_token_ids:
                finish = "stop"
            results.append(StepResult(row.req_id, tokens, finish))
        return results

    def shutdown(self) -> None:
        pass

    def _token(self, req_id: int, position: int) -> int:
        return (req_id * 1000 + position) % self.vocab

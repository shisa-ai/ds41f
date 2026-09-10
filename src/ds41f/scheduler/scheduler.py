"""Scheduler: pure host metadata — cohort selection and step plans.

No torch, no model globals. The scheduler decides *what* to execute; the engine
owns *how* and *when*. Static cohorts only: rows are admitted together, stay
fixed until the cohort drains, and are never refilled mid-flight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

from ..state.slots import RowRef, StateStore

if TYPE_CHECKING:
    from ..engine.request import _Request, EngineConfig

@dataclass(frozen=True)
class PlanRow:
    req_id: int
    row: RowRef
    prompt_tokens: tuple[int, ...]
    positions: tuple[int, ...]  # absolute positions this step consumes
    max_new_tokens: int = 0  # prefill window sizing
    # VL inputs (image spans must lie inside the first prefill chunk; the reference
    # model takes images + token_types on the start_pos==0 forward only)
    token_types: tuple[int, ...] | None = None
    images: tuple | None = None


@dataclass(frozen=True)
class StepPlan:
    ordinal: int
    op: str  # "prefill" | "decode"
    rows: tuple[PlanRow, ...]


class StaticCohortScheduler:
    """Length-bucketed static cohorts.

    When no rows are active, admits up to `max_active_sequences` waiting requests
    whose prompt lengths are within `cohort_bucket_tolerance` of the head request's
    length (unequal prompts are expensive under the reference's shortest-first
    prefill). While rows are active, plans a decode step over exactly those rows.
    """

    def __init__(self, config: EngineConfig, state: StateStore):
        self.config = config
        self.state = state

    def next_plan(self, waiting: Sequence[_Request], active: dict[int, _Request]) -> Optional[StepPlan]:
        if active:
            if all(getattr(r, "done", False) for r in active.values()):
                # cohort drained: release every row (slot generations advance), admit next
                for slot, req in active.items():
                    self.state.release(slot, req.req_id, req.row_ref.generation)
                active.clear()
            else:
                # finished rows stay in the batch (fixed collectives) but stop emitting
                return self._decode_plan(active)
        if not waiting:
            return None
        return self._admit(waiting, active)

    # -- internals ---------------------------------------------------------

    def _admit(self, waiting, active) -> Optional[StepPlan]:
        """Pop admitted requests from `waiting`, publish their slots, return the plan."""
        if not waiting:
            return None
        head = waiting[0]
        tolerance = 1.0 + self.config.cohort_bucket_tolerance
        cohort: list[_Request] = []
        # scan the queue in order; pop candidates, keep the cohort length-similar
        candidates: list[_Request] = []
        while waiting and len(candidates) < self.config.max_active_sequences:
            candidates.append(waiting.popleft())
        stop = False
        requeue: list[_Request] = []
        for req in candidates:
            if stop or (
                cohort and len(req.prompt_tokens) > tolerance * len(head.prompt_tokens)
            ):
                stop = True  # too long for this cohort; it and everything after stay queued
                requeue.append(req)
                continue
            if not self.state.can_admit(len(req.prompt_tokens), req.params.max_new_tokens):
                requeue.append(req)  # over context budget: requeue, never admit partially
                continue
            cohort.append(req)
        if requeue:
            waiting.extendleft(reversed(requeue))  # preserve original queue order
        if not cohort:
            return None
        rows = []
        for req in cohort:
            row = self.state.alloc(req.req_id, len(req.prompt_tokens), req.params.max_new_tokens)
            req.row_ref = row  # type: ignore[attr-defined]
            active[row.slot] = req
            rows.append(
                PlanRow(
                    req.req_id,
                    row,
                    req.prompt_tokens,
                    tuple(range(len(req.prompt_tokens))),
                    max_new_tokens=req.params.max_new_tokens,
                )
            )
        return StepPlan(ordinal=0, op="prefill", rows=tuple(rows))

    def _decode_plan(self, active) -> StepPlan:
        rows = []
        for slot, req in sorted(active.items()):
            pos = len(req.prompt_tokens) + len(req.completion)
            rows.append(PlanRow(req.req_id, req.row_ref, (), (pos,)))
        return StepPlan(ordinal=0, op="decode", rows=tuple(rows))


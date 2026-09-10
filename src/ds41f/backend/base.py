"""Backend protocol and result types.

A backend executes ordered step plans against the model and private state rows.
Rank-0 sampling is the engine's job conceptually; the reference backend returns
sampled token ids so fake and real backends share one contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..scheduler.scheduler import StepPlan
    from ..state.slots import StateStore


@dataclass(frozen=True)
class StepResult:
    """Per-row outcome of one executed step."""

    req_id: int
    tokens: tuple[int, ...]  # sampled tokens this step (>=1 for prefill tail, 1 for decode)
    finish_reason: str | None = None  # "stop" | "length" | None


@runtime_checkable
class Backend(Protocol):
    def execute(self, plan: StepPlan, state: StateStore) -> list[StepResult]: ...
    def shutdown(self) -> None: ...

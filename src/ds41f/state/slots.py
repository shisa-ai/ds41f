"""Private dense state rows with generations.

One arena, private rows: requests never share mutable contents. A slot carries a
generation counter so a stale reference (e.g. a queued cancellation for a request that
already finished) can never act on the row's next occupant.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RowRef:
    """A reference to a slot at a specific generation."""

    slot: int
    generation: int


class StateStore:
    """Slot ownership and context-budget accounting. Host metadata only."""

    def __init__(self, max_rows: int, max_context_tokens: int):
        self.max_rows = max_rows
        self.max_context_tokens = max_context_tokens
        self._generations = [0] * max_rows
        self._owners: list[Optional[int]] = [None] * max_rows  # slot -> req_id

    def free_slots(self) -> int:
        return sum(1 for o in self._owners if o is None)

    def owned_by(self, req_id: int) -> Optional[int]:
        for slot, owner in enumerate(self._owners):
            if owner == req_id:
                return slot
        return None

    def can_admit(self, prompt_tokens: int, max_new_tokens: int) -> bool:
        return prompt_tokens + max_new_tokens <= self.max_context_tokens

    def alloc(self, req_id: int, prompt_tokens: int, max_new_tokens: int) -> RowRef:
        """Reserve a row for a request. Raises if admission limits are violated.

        Call only when admission has been decided; allocation publishes the slot."""
        if not self.can_admit(prompt_tokens, max_new_tokens):
            raise ValueError(
                f"request {req_id} exceeds context budget: "
                f"{prompt_tokens}+{max_new_tokens} > {self.max_context_tokens}"
            )
        for slot, owner in enumerate(self._owners):
            if owner is None:
                self._owners[slot] = req_id
                return RowRef(slot, self._generations[slot])
        raise RuntimeError(f"no free state rows (max_rows={self.max_rows})")

    def release(self, slot: int, req_id: int, generation: int) -> bool:
        """Release a row if it still belongs to `req_id` at `generation`.

        Returns False (and does nothing) on a stale reference."""
        if (
            0 <= slot < self.max_rows
            and self._owners[slot] == req_id
            and self._generations[slot] == generation
        ):
            self._owners[slot] = None
            self._generations[slot] += 1
            return True
        return False

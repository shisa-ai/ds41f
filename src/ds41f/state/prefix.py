"""Prefix snapshot store: exact-boundary prefix matching with a byte budget.

A snapshot represents the full model state for exactly tokens [0, P): it is
captured while a request's caches still hold that conversation, and restored
into a future request's row before its suffix is consumed. Entries are keyed by
the exact token sequence they represent; a lookup succeeds when a stored
sequence is a prefix of the new request's tokens (never longer, never partial).

The store holds *opaque payloads* (the cache view owns the actual tensors) and
only manages identity, bytes, ordering and invalidation. It is host metadata:
no torch, no model.

v1 semantics (deliberate):
- one boundary per snapshot, captured at an exact position (e.g. the end of a
  completed conversation turn); a ring captured at P+100 cannot be cropped to P
- single-entry match, restored requests run alone in their cohort (the restored
  row cannot share a start_pos=0 bulk prefill with fresh rows)
- payloads are immutable once published; restore copies, never aliases
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class PrefixEntry:
    tokens: tuple[int, ...]  # the exact sequence this state represents
    payload: Any  # opaque, owned by the cache view
    nbytes: int
    model_revision: str


@dataclass
class PrefixHit:
    entry: PrefixEntry
    shared_len: int  # == len(entry.tokens); the new request starts its suffix here


class PrefixSnapshotStore:
    """Byte-bounded LRU of exact-boundary snapshots.

    Thread safety: a lock guards the OrderedDict; payloads are treated as
    immutable. Generation counters invalidate every entry at once (model
    reload, engine restart) without walking the cache."""

    def __init__(self, byte_budget: int, model_revision: str = "v1"):
        if byte_budget <= 0:
            raise ValueError("byte_budget must be positive")
        self.byte_budget = byte_budget
        self.model_revision = model_revision
        self._entries: OrderedDict[tuple[int, ...], PrefixEntry] = OrderedDict()
        self._bytes = 0
        self._generation = 0
        self._lock = threading.Lock()

    # -- capacity ---------------------------------------------------------

    @property
    def bytes(self) -> int:
        with self._lock:
            return self._bytes

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    # -- publish / lookup ---------------------------------------------------

    def put(self, tokens: Sequence[int], payload: Any, nbytes: int) -> PrefixEntry:
        """Publish a snapshot for exactly `tokens`. Evicts LRU entries to fit
        the byte budget; a payload larger than the budget is rejected."""
        key = tuple(tokens)
        if nbytes > self.byte_budget:
            raise ValueError(f"snapshot {nbytes}B exceeds budget {self.byte_budget}B")
        with self._lock:
            if key in self._entries:
                old = self._entries.pop(key)
                self._bytes -= old.nbytes
            while self._bytes + nbytes > self.byte_budget and self._entries:
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= evicted.nbytes
            entry = PrefixEntry(key, payload, nbytes, self.model_revision)
            self._entries[key] = entry  # most recent
            self._bytes += nbytes
            return entry

    def lookup(self, tokens: Sequence[int], min_shared: int = 1) -> Optional[PrefixHit]:
        """Longest stored sequence that is a prefix of `tokens` (>= min_shared
        tokens). The suffix begins at len(entry.tokens)."""
        seq = tuple(tokens)
        with self._lock:
            best: Optional[PrefixEntry] = None
            for key, entry in self._entries.items():
                klen = len(key)
                if klen < min_shared or klen > len(seq):
                    continue
                if seq[:klen] == key:
                    if best is None or klen > len(best.tokens):
                        best = entry
            if best is None:
                return None
            self._entries.move_to_end(best.tokens)  # LRU touch
            return PrefixHit(best, len(best.tokens))

    # -- invalidation -------------------------------------------------------

    def invalidate(self) -> int:
        """Drop every entry (model reload / restart). Returns the freed bytes."""
        with self._lock:
            freed = self._bytes
            self._entries.clear()
            self._bytes = 0
            self._generation += 1
            return freed

"""DS41F engine: request lifecycle and public types.

The engine is the single execution owner (one dedicated thread talks to the backend);
API layers submit requests and consume events from per-request bounded queues.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from queue import Full, Queue
from typing import Iterable, Optional, Sequence


class OverloadedError(RuntimeError):
    """The bounded waiting queue is full; rejected before any state is allocated."""


class FinishReason(str, Enum):
    LENGTH = "length"
    STOP = "stop"
    CANCELLED = "cancelled"
    BACKPRESSURE = "backpressure"


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 128
    stop: tuple[str, ...] = ()


@dataclass
class TokenEvent:
    """One sampled token for one request. Emitted before the terminal event."""

    token: int
    position: int  # index within the completion


@dataclass
class TerminalEvent:
    """Exactly one per request, always the last event on its queue."""

    finish_reason: FinishReason
    prompt_tokens: int
    completion_tokens: int


@dataclass
class Completion:
    finish_reason: FinishReason
    token_ids: tuple[int, ...]
    prompt_tokens: int
    completion_tokens: int


class _Request:
    """Internal request record; the public handle is a view onto it."""

    _IDS = itertools.count(1)

    def __init__(self, prompt_tokens: Sequence[int], params: SamplingParams, exclusive: bool = False, meta=None):
        self.req_id = next(_Request._IDS)
        self.prompt_tokens: tuple[int, ...] = tuple(prompt_tokens)
        self.params = params
        # exclusive: schedule alone in its own cohort (e.g. prefix-cache restore rows
        # cannot share a start_pos=0 bulk prefill with fresh rows)
        self.exclusive = exclusive
        # meta: opaque per-request payload for backend adapters (prefix hits, VL inputs)
        self.meta = meta
        self.created = time.monotonic()
        self.completion: list[int] = []
        self.finish_reason: Optional[FinishReason] = None
        self.cancel_requested = threading.Event()
        self.terminal_sent = False
        self.mailbox: Queue = Queue(maxsize=4096)
        self.lock = threading.Lock()

    @property
    def done(self) -> bool:
        return self.finish_reason is not None


class RequestHandle:
    """Consumer-side view of a request: stream events, or wait for the completion."""

    def __init__(self, request: _Request):
        self._r = request

    @property
    def req_id(self) -> int:
        return self._r.req_id

    def cancel(self) -> None:
        """Request cancellation. Stops emission promptly; resources are reclaimed
        at the next safe execution boundary by the engine."""
        self._r.cancel_requested.set()

    def events(self) -> Iterable:
        """Yield TokenEvents until the TerminalEvent (inclusive). Never blocks after
        the terminal event has been consumed."""
        while True:
            ev = self._r.mailbox.get()
            yield ev
            if isinstance(ev, TerminalEvent):
                return

    def result(self, timeout: Optional[float] = None) -> Completion:
        """Block for the completion. Raises TimeoutError if `timeout` elapses."""
        tokens: list[int] = []
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(f"request {self._r.req_id} did not finish in time")
            try:
                ev = self._r.mailbox.get(timeout=max(remaining, 0.01) if remaining else None)
            except Exception:
                raise TimeoutError(f"request {self._r.req_id} did not finish in time")
            if isinstance(ev, TerminalEvent):
                return Completion(ev.finish_reason, tuple(tokens), ev.prompt_tokens, ev.completion_tokens)
            tokens.append(ev.token)

    def _emit(self, event) -> None:
        """Engine-side delivery. Bounded: a stalled reader cancels only this request."""
        r = self._r
        try:
            r.mailbox.put(event, timeout=5.0)
        except Full:
            self.cancel()
            r.finish_reason = FinishReason.BACKPRESSURE
            r.mailbox.put(TerminalEvent(FinishReason.BACKPRESSURE, len(r.prompt_tokens), len(r.completion)))


@dataclass
class EngineConfig:
    max_pending_requests: int = 64
    max_active_sequences: int = 8
    max_context_tokens: int = 8192  # per request: prompt + completion budget
    cohort_bucket_tolerance: float = 0.5  # max relative prompt-length spread in a cohort


@dataclass
class _EngineState:
    """Metadata shared with the scheduler; the engine thread is the only writer."""

    waiting: deque = field(default_factory=deque)
    active: dict[int, _Request] = field(default_factory=dict)  # slot -> request
    ordinal: int = 0
    stopping: bool = False

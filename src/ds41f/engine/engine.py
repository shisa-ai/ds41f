"""LLMEngine: single execution owner around a backend.

One dedicated thread drains control commands (cancellation, shutdown), asks the
scheduler for the next step plan, executes it on the backend, and delivers each
request's tokens through its bounded mailbox with exactly one terminal event.
API layers never touch the backend directly.
"""

from __future__ import annotations

import threading
from queue import Empty, Queue
from typing import Optional, Sequence

from ..backend.base import Backend
from ..scheduler.scheduler import StaticCohortScheduler
from ..state.slots import StateStore
from .request import (
    Completion,
    EngineConfig,
    FinishReason,
    OverloadedError,
    RequestHandle,
    SamplingParams,
    TerminalEvent,
    TokenEvent,
    _Request,
)

_CONTROL_STOP = "stop"
_CONTROL_CANCEL = "cancel"


class LLMEngine:
    def __init__(self, backend: Backend, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.state = StateStore(self.config.max_active_sequences, self.config.max_context_tokens)
        self.scheduler = StaticCohortScheduler(self.config, self.state)
        self.backend = backend
        from collections import deque

        self._waiting = deque()
        self._active: dict[int, _Request] = {}
        self._control: Queue = Queue()
        self._wakeup = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()

    # -- public API --------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return  # already started (idempotent; `with` may wrap an already-started engine)
        self._thread = threading.Thread(target=self._run, name="ds41f-engine", daemon=True)
        self._thread.start()

    def submit(self, prompt_tokens: Sequence[int], params: Optional[SamplingParams] = None) -> RequestHandle:
        params = params or SamplingParams()
        if len(self._waiting) >= self.config.max_pending_requests:
            raise OverloadedError("waiting queue is full")
        req = _Request(prompt_tokens, params)
        if not self.state.can_admit(len(req.prompt_tokens), params.max_new_tokens):
            raise ValueError(
                f"request exceeds context budget {self.state.max_context_tokens}: "
                f"{len(req.prompt_tokens)} prompt + {params.max_new_tokens} new tokens"
            )
        self._waiting.append(req)
        self._wakeup.set()
        return RequestHandle(req)

    def shutdown(self, timeout: float = 10.0) -> None:
        self._control.put(_CONTROL_STOP)
        self._wakeup.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self.backend.shutdown()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.shutdown()

    # -- engine thread ------------------------------------------------------

    def _run(self) -> None:
        while True:
            self._drain_control()
            if self._stopped.is_set():
                break
            plan = self.scheduler.next_plan(self._waiting, self._active)
            if plan is None:
                self._wakeup.clear()
                if self._waiting or self._active or not self._control.empty():
                    # waiting but not admissible yet (e.g. cohort constraints); yield briefly
                    self._wakeup.wait(timeout=0.005)
                    continue
                self._wakeup.wait(timeout=0.05)
                continue
            results = self.backend.execute(plan, self.state)
            self._deliver(results)
        self._fail_remaining()

    def _drain_control(self) -> None:
        while True:
            try:
                cmd = self._control.get_nowait()
            except Empty:
                return
            if cmd == _CONTROL_STOP:
                self._stopped.set()
                return

    def _deliver(self, results) -> None:
        for res in results:
            req = self._active.get(self._slot_of(res.req_id))
            if req is None or req.req_id != res.req_id:
                continue
            if req.done:
                continue  # already terminal (e.g. cancelled); row drains at next boundary
            finish = None
            if req.cancel_requested.is_set():
                finish = FinishReason.CANCELLED
            elif res.finish_reason == "stop":
                finish = FinishReason.STOP
            for tok in res.tokens:
                if req.finish_reason is None and not req.cancel_requested.is_set():
                    req.completion.append(tok)
                    handle = RequestHandle(req)
                    handle._emit(TokenEvent(tok, len(req.completion) - 1))
            if finish is None and len(req.completion) >= req.params.max_new_tokens:
                finish = FinishReason.LENGTH
            if finish is not None:
                self._finish(req, finish)

    def _finish(self, req: _Request, finish: FinishReason) -> None:
        assert not req.terminal_sent
        req.finish_reason = finish
        req.terminal_sent = True
        handle = RequestHandle(req)
        handle._emit(
            TerminalEvent(finish, len(req.prompt_tokens), len(req.completion))
        )
        # Stage A: the row stays in the fixed cohort (the reference batch and its
        # collectives cannot shrink mid-flight). The scheduler releases all rows
        # when the cohort drains; a cancelled row is simply a non-emitting row.

    def _slot_of(self, req_id: int) -> Optional[int]:
        return self.state.owned_by(req_id)

    def _fail_remaining(self) -> None:
        for req in list(self._active.values()) + list(self._waiting):
            if not req.done:
                self._finish(req, FinishReason.CANCELLED)

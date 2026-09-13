"""LLMEngine: single execution owner around a backend.

One dedicated thread drains control commands (cancellation, shutdown), asks the
scheduler for the next step plan, executes it on the backend, and delivers each
request's tokens through its bounded mailbox with exactly one terminal event.
API layers never touch the backend directly.
"""

from __future__ import annotations

import os
import threading
import time
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
        # Admission window. A cohort is whatever is in the wait queue when the engine
        # asks, so four simultaneous requests can be admitted as cohorts of 3 and 1 and
        # the straggler waits a whole generation (3.1-7.4 s TTFT at concurrency 4, with
        # no run admitting all four). Holding the cohort open for a bounded window after
        # the first arrival lets the rest land in the same batch. The wait happens only
        # at a cohort boundary and only while the cohort is under capacity, so it cannot
        # slow a running batch down; it trades a bounded start delay for a full
        # generation of a straggler's time. 0 keeps the old behaviour, and 0 is the
        # default because a lone request pays the whole window.
        #
        # Measured, 3 repeats of 4 x 128 tokens at concurrency 4 (docs/OPTIMIZE-RESULTS.md):
        #   0 ms  40.80 tok/s, worst TTFT 3702.8 ms, c=4 decode cohorts 1x248 3x254 4x115
        #   8 ms  52.42 tok/s, worst TTFT  226.3 ms, c=4 decode cohorts 1x341 4x358
        #  20 ms  51.52 tok/s, worst TTFT  227.0 ms
        #  60 ms  50.93 tok/s, worst TTFT  228.2 ms
        # 8 ms is the knee; past it the aggregate stops improving and the idle-time
        # delay grows linearly. Decode is untouched at every setting (c=1 ITL 28.67-28.78 ms).
        self._admit_window_s = float(os.environ.get("DS41F_ADMIT_WINDOW_MS", "0")) / 1000.0
        self._admit_deadline: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()
        # on_drain(cohort_requests) fires after a cohort fully drains and before the
        # next prefill executes -- the one window where the drained rows' caches are
        # still intact for snapshot capture
        self.on_drain = None
        # on_finish(req, row_index) fires when a request finishes, at the exact
        # boundary its caches represent -- the only correct snapshot point. The
        # engine must not block: slow callbacks delay the whole cohort.
        self.on_finish = None
        self._last_cohort: list = []
        self._dead = threading.Event()

    # -- public API --------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return  # already started (idempotent; `with` may wrap an already-started engine)
        self._thread = threading.Thread(target=self._run, name="ds41f-engine", daemon=True)
        self._thread.start()

    def submit(self, prompt_tokens: Sequence[int], params: Optional[SamplingParams] = None) -> RequestHandle:
        params = params or SamplingParams()
        if self._dead.is_set():
            raise RuntimeError("engine failed; restart required")
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
        try:
            self._run_inner()
        except BaseException:
            # the single execution owner died (backend/collective failure): close
            # admission and fail every outstanding handle exactly once. Do not let
            # new work enter a dead engine.
            self._dead.set()
            self._stopped.set()
            for req in list(self._active.values()) + list(self._waiting):
                if not req.done:
                    self._finish(req, FinishReason.CANCELLED)
            raise

    def _run_inner(self) -> None:
        while True:
            if self._dead.is_set():
                return
            self._drain_control()
            if self._stopped.is_set():
                break
            # Hold a fresh, under-filled cohort open for the admission window. Waiting
            # here rather than inside the scheduler keeps `_admit` free of policy and
            # keeps the wait before any row is allocated, so there is nothing to undo.
            if (
                self._admit_window_s > 0
                and not self._active
                and self._waiting
                and len(self._waiting) < self.config.max_active_sequences
            ):
                now = time.monotonic()
                if self._admit_deadline is None:
                    self._admit_deadline = now + self._admit_window_s
                remaining = self._admit_deadline - now
                if remaining > 0:
                    self._wakeup.clear()
                    self._wakeup.wait(timeout=remaining)
                    continue
            else:
                self._admit_deadline = None
            plan = self.scheduler.next_plan(self._waiting, self._active)
            # eager drain capture, two paths:
            # (a) the scheduler drained the cohort while admitting a new one (its
            #     rows are handed over via scheduler.drained; capture them now,
            #     before the admitted prefill overwrites their cache rows)
            # (b) the cohort drained with the engine otherwise idle (drain-check
            #     below fires within one loop tick of the last completion)
            drained = getattr(self.scheduler, "drained", None)
            if drained:
                self.scheduler.drained = []
                if self.on_drain is not None:
                    self.on_drain(drained)
            if self._last_cohort and not self._active and all(
                getattr(r, "done", False) for r in self._last_cohort
            ):
                if self.on_drain is not None:
                    self.on_drain(self._last_cohort)
                self._last_cohort = []
            if plan is None:
                self._wakeup.clear()
                if self._waiting or self._active or not self._control.empty():
                    # waiting but not admissible yet (e.g. cohort constraints); yield briefly
                    self._wakeup.wait(timeout=0.005)
                    continue
                self._wakeup.wait(timeout=0.05)
                continue
            staged = self._enqueue(plan)
            if plan.op == "prefill":
                self._last_cohort = sorted(self._active.values(), key=lambda r: r.row_ref.slot)
            self._deliver(self._resolve(staged))
        self._fail_remaining()

    def _enqueue(self, plan):
        """Start a step. A backend that only implements the `execute` protocol runs
        the step outright and hands back results that are already resolved."""
        if hasattr(self.backend, "enqueue"):
            return self.backend.enqueue(plan, self.state)
        return self.backend.execute(plan, self.state)

    def settings(self) -> dict:
        """The behaviour-affecting settings this engine resolved from the environment.

        Printed at server startup so an opt-in that silently did not take effect is
        visible in the log rather than inferred from a measurement that looks wrong.
        """
        return {
            "admit_window_ms": round(self._admit_window_s * 1000, 3),
            "split_backend": hasattr(self.backend, "enqueue"),
        }

    def _resolve(self, staged):
        if hasattr(self.backend, "resolve"):
            return self.backend.resolve(staged)
        return staged

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
        # Stage A: the row stays in the fixed cohort (the reference batch and its
        # collectives cannot shrink mid-flight). The scheduler releases all rows
        # when the cohort drains; a cancelled row is simply a non-emitting row.
        # When this was the cohort's last unfinished row, drain-capture fires
        # BEFORE the terminal event is queued (the client's next lookup must see it).
        if self.on_finish is not None:
            try:
                row_index = self._last_cohort.index(req) if req in self._last_cohort else None
            except ValueError:
                row_index = None
            if row_index is not None:
                self.on_finish(req, row_index)
        if self._last_cohort and all(
            getattr(r, "done", False) for r in self._last_cohort
        ):
            if self.on_drain is not None:
                self.on_drain(self._last_cohort)
            self._last_cohort = []
        # Per-request finish capture at the EXACT boundary the caches represent
        # (prompt + completion[:-1]: the final sampled token is not yet consumed by
        # the model). row_index = the request's batch position (its model cache row).
        # A finished row's caches keep advancing with cohort filler afterwards, so
        # this is the only correct snapshot point.

        handle = RequestHandle(req)
        handle._emit(
            TerminalEvent(finish, len(req.prompt_tokens), len(req.completion))
        )

    def _slot_of(self, req_id: int) -> Optional[int]:
        return self.state.owned_by(req_id)

    def _fail_remaining(self) -> None:
        for req in list(self._active.values()) + list(self._waiting):
            if not req.done:
                self._finish(req, FinishReason.CANCELLED)

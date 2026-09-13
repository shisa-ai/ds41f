"""End-to-end engine tests on the FakeBackend: sole driver, exactly one terminal,
cancellation, overload rejection, backpressure, isolation between requests."""

import threading
import time

import pytest

from ds41f.backend import FakeBackend
from ds41f.engine import EngineConfig, FinishReason, LLMEngine, OverloadedError, SamplingParams


def run_engine(config=None, backend=None):
    eng = LLMEngine(backend or FakeBackend(), config or EngineConfig(max_pending_requests=8, max_active_sequences=4))
    eng.start()
    return eng


def test_basic_generation_and_terminal_semantics():
    with run_engine() as eng:
        h = eng.submit([1, 2, 3, 4], SamplingParams(max_new_tokens=10))
        comp = h.result(timeout=10)
        assert comp.finish_reason == FinishReason.LENGTH
        assert comp.prompt_tokens == 4
        assert comp.completion_tokens == 10
        assert len(comp.token_ids) == 10
        # a second read on the stream must not produce another terminal
        # (events() already returned); terminal-sent flag guards re-emission


def test_stop_token_finishes_with_stop_reason():
    fake = FakeBackend(vocab=100, stop_token_ids={(1 * 1000 + 4) % 100})
    with run_engine(backend=fake) as eng:
        h = eng.submit([1, 2, 3], SamplingParams(max_new_tokens=10))
        comp = h.result(timeout=10)
        assert comp.finish_reason == FinishReason.STOP
        assert comp.completion_tokens == 2  # prefill token at p=3, then the stop token


def test_concurrent_clients_no_cross_talk():
    with run_engine(EngineConfig(max_pending_requests=16, max_active_sequences=8)) as eng:
        handles = [eng.submit([i] * (i + 1), SamplingParams(max_new_tokens=5)) for i in range(8)]
        comps = [h.result(timeout=30) for h in handles]
    for i, (h, comp) in enumerate(zip(handles, comps)):
        # FakeBackend token at completion position p is (req_id*1000+p) % vocab == p % vocab,
        # so each request must see exactly its own positions: prompt_len..prompt_len+4
        expected = tuple(p % 100 for p in range(i + 1, i + 1 + 5))
        assert comp.token_ids == expected, f"cross-talk in request {h.req_id}"


def test_overload_rejection_before_state():
    eng = run_engine(EngineConfig(max_pending_requests=2, max_active_sequences=1))
    try:
        eng.submit([1], SamplingParams(max_new_tokens=2))
        eng.submit([1], SamplingParams(max_new_tokens=2))
        with pytest.raises(OverloadedError):
            eng.submit([1], SamplingParams(max_new_tokens=2))
    finally:
        eng.shutdown()


def test_cancel_stops_emission_and_reclaims_slot():
    eng = run_engine(EngineConfig(max_active_sequences=1, max_context_tokens=200_000), backend=FakeBackend(delay=0.002))
    try:
        h = eng.submit([1, 2], SamplingParams(max_new_tokens=100000))
        time.sleep(0.2)  # let a few tokens flow
        h.cancel()
        comp = h.result(timeout=10)
        assert comp.finish_reason == FinishReason.CANCELLED
        assert comp.completion_tokens < 1000
        # slot was reclaimed: a new request runs fine
        h2 = eng.submit([1, 2], SamplingParams(max_new_tokens=3))
        assert h2.result(timeout=10).completion_tokens == 3
    finally:
        eng.shutdown()


def test_slow_reader_does_not_block_other_clients():
    # bounded mailboxes: a client that never drains must not stall the engine thread
    eng = run_engine(EngineConfig(max_active_sequences=4))
    try:
        slow = eng.submit([1], SamplingParams(max_new_tokens=5000))
        fast = eng.submit([2], SamplingParams(max_new_tokens=5))
        # never read from `slow`'s queue; `fast` must still complete promptly
        t0 = time.monotonic()
        comp = fast.result(timeout=30)
        assert time.monotonic() - t0 < 30
        assert comp.completion_tokens == 5
    finally:
        slow.cancel()
        eng.shutdown()


def test_shutdown_fails_inflight_with_cancelled():
    eng = run_engine(EngineConfig(max_context_tokens=200_000), backend=FakeBackend(delay=0.002))
    h = eng.submit([1], SamplingParams(max_new_tokens=100000))
    time.sleep(0.1)
    eng.shutdown(timeout=10)
    comp = h.result(timeout=5)
    assert comp.finish_reason == FinishReason.CANCELLED


def test_sole_driver_single_backend_thread():
    fake = FakeBackend(delay=0.001)
    eng = run_engine(backend=fake)
    try:
        hs = [eng.submit([1] * (i + 1), SamplingParams(max_new_tokens=3)) for i in range(6)]
        for h in hs:
            h.result(timeout=30)
        # every step happened on the engine thread
        recorded = []
        orig = fake.enqueue

        def recording(plan, state):
            recorded.append(threading.current_thread().name)
            return orig(plan, state)

        fake.enqueue = recording
        h = eng.submit([1], SamplingParams(max_new_tokens=2))
        h.result(timeout=10)
        assert recorded and all(n == "ds41f-engine" for n in recorded)
    finally:
        eng.shutdown()


# ------------------------------------------------ admission window (section 4)

class _RecordingBackend(FakeBackend):
    """FakeBackend that records the row count of every prefill plan it executes."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prefill_cohort_sizes = []

    def enqueue(self, plan, state):
        if plan.op == "prefill":
            self.prefill_cohort_sizes.append(len(plan.rows))
        return super().enqueue(plan, state)


def test_admission_window_off_admits_a_single_arrival():
    fake = _RecordingBackend()
    with run_engine(backend=fake) as eng:
        h = eng.submit([1, 2, 3], SamplingParams(max_new_tokens=2))
        h.result(timeout=10)
    assert fake.prefill_cohort_sizes[0] == 1


def test_admission_window_holds_a_cohort_open_for_late_arrivals(monkeypatch):
    """Four requests a millisecond apart must share one prefill cohort.

    Without the window the engine admits whatever the queue holds when it looks, so
    these arrive as a cohort of one or two and the rest wait a whole generation. That
    is what the concurrency-4 trace showed: cohorts of 3, 2, 2, 1 and 3, and a
    straggler at 3.1-7.4 s TTFT.
    """
    # Control arm first: the same submission pattern with the window off, so the
    # assertion below is read against a measured baseline rather than an assumption.
    fake_off = _RecordingBackend()
    with run_engine(backend=fake_off) as eng:
        off = []
        for _ in range(4):
            off.append(eng.submit([1, 2, 3, 4], SamplingParams(max_new_tokens=2)))
            time.sleep(0.001)
        for h in off:
            h.result(timeout=30)
    assert fake_off.prefill_cohort_sizes == [1, 1, 1, 1], fake_off.prefill_cohort_sizes

    monkeypatch.setenv("DS41F_ADMIT_WINDOW_MS", "60")
    fake = _RecordingBackend()
    # Equal-length prompts on purpose: length bucketing splits a cohort by prompt
    # length before the backend sees it, so mixed lengths would hide the window
    # behind that filter. Equal lengths are also the case the concurrency trace
    # measured -- four identical requests, cohorts of 3, 2, 2, 1 and 3.
    with run_engine(backend=fake) as eng:
        handles = []
        for _ in range(4):
            handles.append(eng.submit([1, 2, 3, 4], SamplingParams(max_new_tokens=2)))
            time.sleep(0.001)
        for h in handles:
            h.result(timeout=30)
    assert fake.prefill_cohort_sizes[0] == 4, fake.prefill_cohort_sizes


def test_admission_window_delays_a_lone_arrival_by_at_most_the_window(monkeypatch):
    monkeypatch.setenv("DS41F_ADMIT_WINDOW_MS", "60")
    fake = _RecordingBackend()
    with run_engine(backend=fake) as eng:
        t0 = time.perf_counter()
        h = eng.submit([1, 2, 3], SamplingParams(max_new_tokens=2))
        h.result(timeout=10)
        elapsed = time.perf_counter() - t0
    # it waited for the window rather than admitting at once ...
    assert fake.prefill_cohort_sizes[0] == 1
    assert elapsed >= 0.04, elapsed
    # ... and the wait is bounded, not open-ended
    assert elapsed < 2.0, elapsed


# ------------------------------------------------- split step contract

def test_the_engine_uses_the_split_step_contract_when_the_backend_has_one():
    """The engine prefers enqueue()/resolve() over execute() when both exist.

    This is the step contract the per-step-synchronization work needs: it is what
    would let the host read step i-1's tokens while step i is queued. It is *not*
    overlapped today -- see the note in ReferenceBackend.enqueue -- because the
    served topology blocks on a per-step broadcast inside the forward, so the token
    read is never on the critical path. The contract is kept and exercised so that
    removing that barrier is a change in one place.
    """
    calls = {"enqueue": 0, "resolve": 0, "execute": 0}
    fake = FakeBackend()

    orig_enqueue, orig_resolve, orig_execute = fake.enqueue, fake.resolve, fake.execute

    def enqueue(plan, state):
        calls["enqueue"] += 1
        return orig_enqueue(plan, state)

    def resolve(staged):
        calls["resolve"] += 1
        return orig_resolve(staged)

    def execute(plan, state):
        calls["execute"] += 1
        return orig_execute(plan, state)

    fake.enqueue, fake.resolve, fake.execute = enqueue, resolve, execute
    with run_engine(backend=fake) as eng:
        h = eng.submit([1, 2, 3], SamplingParams(max_new_tokens=3))
        assert h.result(timeout=10).completion_tokens == 3
    assert calls["enqueue"] > 0 and calls["enqueue"] == calls["resolve"]
    assert calls["execute"] == 0


def test_a_backend_with_only_execute_still_runs():
    """The Backend protocol is execute()/shutdown(); the split contract is optional."""
    inner = FakeBackend()

    class OnlyExecute:
        def execute(self, plan, state):
            return inner.execute(plan, state)

        def shutdown(self):
            pass

    assert not hasattr(OnlyExecute(), "enqueue")
    with run_engine(backend=OnlyExecute()) as eng:
        h = eng.submit([1, 2, 3], SamplingParams(max_new_tokens=3))
        assert h.result(timeout=10).completion_tokens == 3

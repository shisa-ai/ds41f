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


# ------------------------------------------------- pipelined delivery (section 3)

def _run_three(monkeypatch, pipeline: bool):
    """Two requests, one of which stops early, one of which runs to the cap.

    Returns the two completions and the plan schedule. The early stop is the
    interesting case: the finished row has to keep its place in the cohort and keep
    drawing for as long as the serial path drew for it, or the stream the other row
    sees shifts.
    """
    monkeypatch.setenv("DS41F_PIPELINE", "1" if pipeline else "0")
    fake = FakeBackend(vocab=1000, stop_token_ids=frozenset({1005}))
    with run_engine(backend=fake) as eng:
        a = eng.submit([1, 2, 3], SamplingParams(max_new_tokens=6))
        b = eng.submit([4, 5, 6, 7], SamplingParams(max_new_tokens=6))
        ha = a.result(timeout=10)
        hb = b.result(timeout=10)
    schedule = [(p.op, len(p.rows)) for p in fake.executed]
    return (ha.token_ids, ha.finish_reason), (hb.token_ids, hb.finish_reason), schedule


def test_the_pipeline_flag_actually_selects_the_pipelined_loop(monkeypatch):
    """Guards the two tests below: if the flag stopped selecting the loop they would
    compare the serial path against itself and pass."""
    from ds41f.backend import FakeBackend as FB
    from ds41f.engine import EngineConfig as EC
    from ds41f.engine import LLMEngine as LE

    for value, expected in (("0", False), ("1", True)):
        monkeypatch.setenv("DS41F_PIPELINE", value)
        assert LE(FB(), EC())._pipeline is expected
    # a backend with no split contract must refuse the flag rather than quietly
    # running the serial loop, which is how a first A/B compared serial to serial
    class OnlyExecute:
        def execute(self, plan, state):
            return []

    monkeypatch.setenv("DS41F_PIPELINE", "1")
    with pytest.raises(RuntimeError, match="enqueue"):
        LE(OnlyExecute(), EC())


def test_pipelined_delivery_is_token_identical_to_serial(monkeypatch):
    serial = _run_three(monkeypatch, pipeline=False)
    pipelined = _run_three(monkeypatch, pipeline=True)
    assert pipelined[0] == serial[0], (serial[0], pipelined[0])
    assert pipelined[1] == serial[1], (serial[1], pipelined[1])


def test_pipelined_delivery_keeps_the_batch_composition(monkeypatch):
    """Pipelining must not change which rows run together, and must cost at most one
    extra trailing decode step.

    The extra step is the price of building step i's plan before step i-1's tokens
    are read: the last rows to finish are not known to be finished yet, so the cohort
    gets one more step. Its own tokens are discarded (the rows are already terminal
    by the time it is delivered). On the real backend that step needs one position of
    slack in the cohort's token buffer.
    """
    serial = _run_three(monkeypatch, pipeline=False)[2]
    pipelined = _run_three(monkeypatch, pipeline=True)[2]
    assert len(pipelined) - len(serial) in (0, 1), (serial, pipelined)
    # same op sequence and same batch composition, up to that one trailing step
    assert pipelined[: len(serial)] == serial, (serial, pipelined)
    if len(pipelined) > len(serial):
        assert pipelined[-1][0] == "decode"


def test_pipelined_delivery_runs_a_batched_cohort(monkeypatch):
    """The one extra step a pipelined cohort can take must be legal and harmless."""
    monkeypatch.setenv("DS41F_PIPELINE", "1")
    fake = FakeBackend(vocab=1000)
    with run_engine(backend=fake) as eng:
        handles = [eng.submit([1, 2, 3], SamplingParams(max_new_tokens=4)) for _ in range(3)]
        for h in handles:
            assert h.result(timeout=10).completion_tokens == 4
    assert any(len(p.rows) == 3 for p in fake.executed), [len(p.rows) for p in fake.executed]

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
        # every execute() call happened on the engine thread
        thread_names = set()
        # (FakeBackend doesn't record threads; instrument here)
        recorded = []
        orig = fake.execute

        def recording(plan, state):
            recorded.append(threading.current_thread().name)
            return orig(plan, state)

        fake.execute = recording
        h = eng.submit([1], SamplingParams(max_new_tokens=2))
        h.result(timeout=10)
        assert recorded and all(n == "ds41f-engine" for n in recorded)
    finally:
        eng.shutdown()

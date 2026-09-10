"""Scheduler: static cohorts, length bucketing, admission limits."""

from ds41f.engine import EngineConfig, SamplingParams
from ds41f.engine.request import _Request
from ds41f.scheduler import StaticCohortScheduler
from collections import deque

from ds41f.state import StateStore


def make_requests(specs):
    return [_Request(tokens, SamplingParams(max_new_tokens=8)) for tokens in specs]


def test_admits_cohort_up_to_max_active():
    s = StateStore(max_rows=4, max_context_tokens=1000)
    sched = StaticCohortScheduler(EngineConfig(max_active_sequences=4), s)
    waiting, active = deque(make_requests([[1] * 10] * 6)), {}
    plan = sched.next_plan(waiting, active)
    assert plan.op == "prefill"
    assert len(plan.rows) == 4
    assert len(waiting) == 2  # two left queued
    assert set(active) == {r.row.slot for r in plan.rows}


def test_length_bucketing_keeps_cohort_similar():
    s = StateStore(max_rows=8, max_context_tokens=10_000)
    sched = StaticCohortScheduler(EngineConfig(max_active_sequences=8, cohort_bucket_tolerance=0.5), s)
    waiting, active = deque(make_requests([[1] * 100, [1] * 110, [1] * 400])), {}
    plan = sched.next_plan(waiting, active)
    admitted = {r.req_id for r in plan.rows}
    # head=100, 110 within tolerance, 400 is not; it stays queued
    assert len(admitted) == 2
    assert len(waiting) == 1 and len(waiting[0].prompt_tokens) == 400


def test_over_budget_request_is_requeued_not_admitted():
    s = StateStore(max_rows=4, max_context_tokens=20)
    sched = StaticCohortScheduler(EngineConfig(), s)
    req_ok = _Request([1] * 10, SamplingParams(max_new_tokens=5))
    req_big = _Request([1] * 19, SamplingParams(max_new_tokens=50))
    waiting, active = deque([req_big, req_ok]), {}
    plan = sched.next_plan(waiting, active)
    assert {r.req_id for r in plan.rows} == {req_ok.req_id}
    assert req_big in waiting


def test_decode_plan_covers_active_rows_with_positions():
    s = StateStore(max_rows=4, max_context_tokens=1000)
    sched = StaticCohortScheduler(EngineConfig(), s)
    waiting, active = deque(make_requests([[1] * 10, [2] * 12])), {}
    sched.next_plan(waiting, active)
    # simulate 3 completion tokens on the first request
    first = active[sorted(active)[0]]
    first.completion.extend([7, 8, 9])
    plan = sched.next_plan(waiting, active)
    assert plan.op == "decode"
    by_id = {r.req_id: r for r in plan.rows}
    assert by_id[first.req_id].positions == (13,)
    assert by_id[active[sorted(active)[1]].req_id].positions == (12,)

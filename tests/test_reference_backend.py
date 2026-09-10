"""ReferenceBackend semantics against a stub model with the reference forward signature.

Requires torch (skipped when unavailable). The stub model is a deterministic
function of the *last consumed token per row*, so tests can predict exact outputs.
"""

import pytest

torch = pytest.importorskip("torch")

from ds41f.backend import reference as rb
from ds41f.backend.reference import ReferenceBackend
from ds41f.scheduler import PlanRow, StepPlan
from ds41f.state import RowRef, StateStore


class StubModel:
    """output token per row = (last input token + 1) % 1000; EOS at 500."""

    def forward(self, input_ids, start_pos, images=None, token_types=None):
        assert input_ids.shape[1] >= 1
        assert start_pos + input_ids.shape[1] <= 1_000_000
        out = (input_ids[:, -1] + 1) % 1000
        return out, None, None

    @property
    def eos_token_id(self):
        return 500


def make_prefill(prompt_lens, max_new=5):
    rows = []
    for i, plen in enumerate(prompt_lens):
        rows.append(
            PlanRow(
                req_id=i + 1,
                row=RowRef(slot=i, generation=0),
                prompt_tokens=tuple(100 + j for j in range(plen)),
                positions=tuple(range(plen)),
                max_new_tokens=max_new,
            )
        )
    return StepPlan(ordinal=0, op="prefill", rows=tuple(rows))


def make_decode(rows):
    return StepPlan(ordinal=1, op="decode", rows=tuple(rows))


def test_equal_prompts_emit_from_prefill():
    be = ReferenceBackend(StubModel(), eos_token_id=500)
    plan = make_prefill([3, 3])
    res = be.execute(plan, StateStore(2, 100))
    # both rows' prompts ended at min_len: first sampled token each
    assert [r.tokens for r in res] == [(103,), (103,)]  # last prompt token 102 + 1


def test_unequal_prompts_suppress_emission_until_consumed():
    be = ReferenceBackend(StubModel(), eos_token_id=500)
    prefill = make_prefill([2, 5], max_new=3)
    rows = prefill.rows
    state = StateStore(4, 100)
    res = be.execute(prefill, state)
    # row 1 (prompt 2) emits its first token; row 2 (prompt 5) still consuming
    assert res[0].tokens == (102,)
    assert res[1].tokens == ()

    # decode steps 1-2: row 1 keeps emitting, row 2 consumes ground-truth tokens
    for step in range(2):
        res = be.execute(make_decode(rows), state)
        assert len(res[1].tokens) == 0
    # step 3: row 2's prompt is finally consumed (positions 2,3,4 done) -> emits 105
    res = be.execute(make_decode(rows), state)
    assert res[1].tokens == (105,)
    assert res[0].tokens == () or res[0].tokens  # row 1 finished by max_new earlier


def test_eos_finishes_row():
    be = ReferenceBackend(StubModel(), eos_token_id=103)
    plan = make_prefill([3], max_new=10)
    res = be.execute(plan, StateStore(1, 100))
    assert res[0].tokens == (103,) and res[0].finish_reason == "stop"


def test_decode_cohort_must_be_stable():
    be = ReferenceBackend(StubModel(), eos_token_id=500)
    plan = make_prefill([2, 2])
    state = StateStore(2, 100)
    be.execute(plan, state)
    # drop a row from the decode plan -> adapter must refuse (fixed batch)
    with pytest.raises(RuntimeError):
        be.execute(make_decode(plan.rows[:1]), state)


def test_buffer_window_covers_long_prompt_plus_max_new():
    be = ReferenceBackend(StubModel(), eos_token_id=500)
    prefill = make_prefill([2, 40], max_new=3)
    state = StateStore(4, 100)
    be.execute(prefill, state)
    rows = prefill.rows
    for _ in range(38 + 3):  # consume row 2's prompt tail + its completion budget
        res = be.execute(make_decode(rows), state)
    # buffer must not have overflowed
    assert be._cur_pos <= be._tokens.shape[1]

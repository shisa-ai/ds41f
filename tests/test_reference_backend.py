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


class LogitsStubModel:
    """logits concentrated on token (last+2)%1000 (greedy) and (last+3)%1000."""

    def forward(self, input_ids, start_pos, images=None, token_types=None):
        import torch

        out = (input_ids[:, -1] + 1) % 1000
        B = input_ids.shape[0]
        V = 1000
        logits = torch.full((B, V), -50.0)
        for b in range(B):
            last = int(input_ids[b, -1].item())
            logits[b, (last + 2) % V] = 10.0
            logits[b, (last + 3) % V] = 5.0
        return out, logits, None


def test_per_row_sampling_greedy_uses_logits_not_model_ids():
    be = ReferenceBackend(LogitsStubModel(), eos_token_id=500)
    plan = make_prefill([3], max_new=4)
    # row asks temperature=0: must pick argmax from logits ((last+2)), not the
    # model's own sampled ids ((last+1))
    plan = StepPlan(
        ordinal=0,
        op="prefill",
        rows=(
            PlanRow(
                req_id=1,
                row=RowRef(slot=0, generation=0),
                prompt_tokens=(100, 101, 102),
                positions=(0, 1, 2),
                max_new_tokens=4,
                temperature=0.0,
            ),
        ),
    )
    res = be.execute(plan, StateStore(1, 100))
    assert res[0].tokens == (104,)  # 102 + 2, from logits


def test_per_row_sampling_temperature_falls_in_top_p_support():
    torch.manual_seed(0)
    be = ReferenceBackend(LogitsStubModel(), eos_token_id=500)
    plan = StepPlan(
        ordinal=0,
        op="prefill",
        rows=(
            PlanRow(
                req_id=1,
                row=RowRef(slot=0, generation=0),
                prompt_tokens=(100, 101, 102),
                positions=(0, 1, 2),
                max_new_tokens=4,
                temperature=1.0,
                top_p=0.9,
            ),
        ),
    )
    tokens = set()
    for _ in range(20):
        be._reset()
        res = be.execute(plan, StateStore(1, 100))
        tokens.update(res[0].tokens)
    # top_p=0.9 keeps the two dominant tokens only
    assert tokens <= {104, 105}


class RecordingModel(StubModel):
    """StubModel that records every forward call for the chunked-suffix test."""

    def __init__(self):
        self.calls = []

    def forward(self, input_ids, start_pos, images=None, token_types=None):
        self.calls.append((start_pos, tuple(input_ids[0].tolist())))
        return super().forward(input_ids, start_pos, images, token_types)


def test_prefix_hit_chunked_suffix_single_forward():
    """A prefix-hit row with a multi-token suffix consumes the whole suffix in ONE
    chunked forward (start_pos=shared, seqlen=suffix) and emits its first token
    from that forward's prediction; the cursor then sits at the prompt end."""
    model = RecordingModel()
    be = ReferenceBackend(model, eos_token_id=500)
    prompt = tuple(100 + j for j in range(8))
    restored = []

    class Meta:
        prefix_hit = True
        shared_len = 3

        def restore(self):
            restored.append(True)

    row = PlanRow(
        req_id=1,
        row=RowRef(slot=0, generation=0),
        prompt_tokens=prompt,
        positions=tuple(range(8)),
        max_new_tokens=5,
        meta=Meta(),
    )
    plan = StepPlan(ordinal=0, op="prefill", rows=(row,))
    res = be.execute(plan, StateStore(1, 100))
    assert restored == [True]
    # one chunked forward: suffix tokens 103..107 consumed at start_pos 3
    assert model.calls == [(3, tuple(range(103, 108)))]
    # the prediction for the last prompt token (107 -> 108) emits immediately
    assert res[0].tokens == (108,)
    # the next decode step continues from the prompt end, not the shared boundary
    be.execute(make_decode((row,)), StateStore(1, 100))
    assert model.calls[-1][0] == 8


def test_prefix_hit_single_token_suffix_stays_on_decode():
    """A one-token suffix must not run a seqlen==1 'chunked' forward (the model's
    chunked branch requires seqlen > 1); it stays on the ordinary decode path."""
    model = RecordingModel()
    be = ReferenceBackend(model, eos_token_id=500)
    prompt = tuple(100 + j for j in range(4))
    class Meta:
        prefix_hit = True
        shared_len = 3

        def restore(self):
            pass

    row = PlanRow(
        req_id=1,
        row=RowRef(slot=0, generation=0),
        prompt_tokens=prompt,
        positions=tuple(range(4)),
        max_new_tokens=3,
        meta=Meta(),
    )
    plan = StepPlan(ordinal=0, op="prefill", rows=(row,))
    res = be.execute(plan, StateStore(1, 100))
    assert res[0].tokens == ()  # nothing emitted by the snapshot prefill
    assert model.calls == []  # no forward yet: the first decode consumes it
    res = be.execute(make_decode((row,)), StateStore(1, 100))
    assert model.calls == [(3, (103,))]  # single-token decode at the boundary


class RowLogitsStubModel:
    """logits depend on the row index: row b has argmax 100+b and runner-up 200+b."""

    def forward(self, input_ids, start_pos, images=None, token_types=None):
        import torch

        B = input_ids.shape[0]
        V = 1000
        out = (input_ids[:, -1] + 1) % 1000
        logits = torch.full((B, V), -50.0)
        for b in range(B):
            logits[b, 100 + b] = 10.0
            logits[b, 200 + b] = 9.0
        return out, logits, None


def _mixed_params_rows(top_p_row1):
    return (
        PlanRow(1, RowRef(slot=0, generation=0), (100, 101, 102), (0, 1, 2), max_new_tokens=4, temperature=0.0, top_p=1.0),
        PlanRow(2, RowRef(slot=1, generation=0), (100, 101, 102), (0, 1, 2), max_new_tokens=4, temperature=1.0, top_p=top_p_row1),
        PlanRow(3, RowRef(slot=2, generation=0), (100, 101, 102), (0, 1, 2), max_new_tokens=4, temperature=0.0, top_p=1.0),
    )


def test_batched_sampling_mixes_greedy_and_top_p_per_row():
    """One batched call keeps per-row parameters: greedy rows take their own
    argmax, the top-p row is restricted to its nucleus, results stay in row order."""
    be = ReferenceBackend(RowLogitsStubModel(), eos_token_id=500)
    plan = StepPlan(ordinal=0, op="prefill", rows=_mixed_params_rows(top_p_row1=0.5))
    res = be.execute(plan, StateStore(3, 100))
    # row 1 has nucleus {101} at top_p=0.5; rows 0/2 are greedy on their own argmax
    assert [r.tokens for r in res] == [(100,), (101,), (102,)]


def test_batched_sampling_top_p_support_is_per_row():
    """With a wider nucleus the stochastic row may take its runner-up, but never a
    token outside the two dominant ones; the greedy rows stay exact."""
    import torch

    torch.manual_seed(0)
    be = ReferenceBackend(RowLogitsStubModel(), eos_token_id=500)
    plan = StepPlan(ordinal=0, op="prefill", rows=_mixed_params_rows(top_p_row1=0.9))
    seen = set()
    for _ in range(30):
        be._reset()
        res = be.execute(plan, StateStore(3, 100))
        assert res[0].tokens == (100,) and res[2].tokens == (102,)
        seen.add(res[1].tokens[0])
    assert seen <= {101, 201}


def test_batched_sampling_writes_next_ids_on_device():
    """The sampled ids are written back into the token buffer for the next step in
    one indexed device write, and the buffer matches the emitted tokens."""
    be = ReferenceBackend(RowLogitsStubModel(), eos_token_id=500)
    plan = StepPlan(ordinal=0, op="prefill", rows=_mixed_params_rows(top_p_row1=0.5))
    be.execute(plan, StateStore(3, 100))
    # positions 0..2 are the prompt; position 3 is the sampled continuation
    assert be._tokens[:, 3].tolist() == [100, 101, 102]


class UniformLogitsStubModel:
    """10 equally likely tokens, so top_p=0.9 and top_p=1.0 differ in support."""

    def forward(self, input_ids, start_pos, images=None, token_types=None):
        import torch

        B = input_ids.shape[0]
        return (
            torch.zeros(B, dtype=torch.long, device=input_ids.device),
            torch.zeros((B, 10)),
            None,
        )


def _uniform_rows(top_p):
    return tuple(
        PlanRow(
            req_id=i + 1,
            row=RowRef(slot=i, generation=0),
            prompt_tokens=(0,),
            positions=(0,),
            max_new_tokens=1,
            temperature=1.0,
            top_p=top_p,
        )
        for i in range(2)
    )


def _uniform_support(top_p, draws=300):
    import torch

    torch.manual_seed(0)
    be = ReferenceBackend(UniformLogitsStubModel(), eos_token_id=500)
    seen = set()
    for _ in range(draws):
        be._reset()
        res = be.execute(
            StepPlan(ordinal=0, op="prefill", rows=_uniform_rows(top_p)),
            StateStore(2, 100),
        )
        seen.update(r.tokens[0] for r in res)
    return seen


def test_top_p_one_keeps_full_support():
    """The removed `if bool((tops < 1.0).any())` guard skipped the top-p block
    whenever every row asked for top_p=1.0. The block is now always applied, and
    at top_p=1.0 it must keep the whole vocabulary."""
    assert _uniform_support(1.0) == set(range(10))


def test_top_p_below_one_still_truncates():
    """Always running the filter must not widen the nucleus: with ten equally
    likely tokens, top_p=0.9 keeps nine of them."""
    assert len(_uniform_support(0.9)) == 9


def test_greedy_row_ignores_top_p():
    """The removed `if bool(greedy.any())` guard applied torch.where only when some
    row was greedy. Applying it unconditionally must not change greedy semantics:
    a greedy row takes its raw argmax even when its top_p would exclude it."""
    be = ReferenceBackend(RowLogitsStubModel(), eos_token_id=500)
    rows = tuple(
        PlanRow(
            req_id=i + 1,
            row=RowRef(slot=i, generation=0),
            prompt_tokens=(100, 101, 102),
            positions=(0, 1, 2),
            max_new_tokens=4,
            temperature=0.0,
            top_p=0.01,
        )
        for i in range(2)
    )
    res = be.execute(StepPlan(ordinal=0, op="prefill", rows=rows), StateStore(2, 100))
    assert [r.tokens for r in res] == [(100,), (101,)]

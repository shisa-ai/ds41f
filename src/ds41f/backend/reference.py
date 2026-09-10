"""ReferenceBackend: adapter over the DeepSeek-V4.1-Flash reference model.

Reproduces the reference generate.py batch semantics under the engine's plan
contract:

- Prefill plan: allocate one right-padded token buffer for the cohort and run the
  single start_pos=0 forward over the shortest prompt (image spans ride along via
  the plan rows' token_types/images).
- Decode plan: run one forward over the single new position per row; rows still
  inside their prompt are teacher-forced (their ground-truth token overrides the
  prediction) and emit nothing.

The engine only ever calls execute() from its single owner thread. Rank 0 owns
this object in the served topology; ranks 1..n run the same ordered sequence
inside ``model.forward`` and its collectives.

The model object must expose the reference signature::

    forward(input_ids: Tensor[b, s], start_pos: int, images=None,
            token_types=None) -> (output_ids[b], logits, main_hidden)
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..scheduler.scheduler import PlanRow, StepPlan
from ..state.slots import StateStore
from .base import StepResult

TEXT = -1  # token_type for text positions (matches image_processor.TEXT)


class ReferenceBackend:
    def __init__(self, model, eos_token_id: int, sampler=None):
        self.model = model
        self.eos_token_id = eos_token_id
        # sampler(logits_row, params) -> token_id; default greedy
        self.sampler = sampler
        self._reset()

    def _reset(self) -> None:
        self._tokens = None  # [B, T] buffer
        self._prompt_lens: list[int] = []
        self._req_ids: list[int] = []
        self._prev_pos = 0
        self._cur_pos = 0

    # -- Backend protocol ---------------------------------------------------

    def execute(self, plan: StepPlan, state: StateStore) -> list[StepResult]:
        if plan.op == "prefill":
            return self._prefill(plan)
        if plan.op == "decode":
            return self._decode(plan)
        raise ValueError(f"unknown plan op {plan.op!r}")

    def shutdown(self) -> None:
        self._reset()

    # -- prefill -------------------------------------------------------------

    def _prefill(self, plan: StepPlan) -> list[StepResult]:
        rows = sorted(plan.rows, key=lambda r: r.row.slot)
        prompt_lens = [len(r.prompt_tokens) for r in rows]
        min_len = min(prompt_lens)
        # window: the longest prompt still consumes (plen - min_len) one-token steps
        # before its completion begins, then every row needs its max_new_tokens
        window = max(
            plen - min_len + (r.max_new_tokens or 1)
            for r, plen in zip(rows, prompt_lens)
        )
        total = min_len + window
        self._tokens = _new_tokens_buffer(len(rows), total, rows, prompt_lens)
        self._prompt_lens = prompt_lens
        self._req_ids = [r.req_id for r in rows]

        out_ids, _, _ = self.model.forward(
            self._tokens[:, :min_len],
            0,
            images=[r.images for r in rows] if any(r.images for r in rows) else None,
            token_types=_stack_token_types(rows, min_len),
        )
        self._prev_pos = min_len
        self._cur_pos = min_len
        return self._collect(out_ids, min_len)

    # -- decode --------------------------------------------------------------

    def _decode(self, plan: StepPlan) -> list[StepResult]:
        # rows arrive sorted by slot; the cohort is fixed so order matches _req_ids
        by_req = {r.req_id: r for r in plan.rows}
        if [r.req_id for r in sorted(plan.rows, key=lambda r: r.row.slot)] != self._req_ids:
            raise RuntimeError("decode plan cohort changed mid-flight; not supported yet")
        cur = self._cur_pos
        total = self._tokens.shape[1]
        if cur >= total:
            raise RuntimeError("token buffer exhausted; cohort window was too small")
        out_ids, _, _ = self.model.forward(self._tokens[:, cur : cur + 1], cur)
        self._cur_pos = cur + 1
        return self._collect(out_ids, cur + 1)

    # -- shared --------------------------------------------------------------

    def _collect(self, out_ids, position: int) -> list[StepResult]:
        """Apply prompt override, decide emission and finishes at `position`."""
        results = []
        for i, req_id in enumerate(self._req_ids):
            prompt_len = self._prompt_lens[i]
            if position < prompt_len:
                # this row's prediction slot is still inside its prompt:
                # teacher-forced, nothing emitted. The ground-truth token already
                # sits in the buffer for the next step to consume.
                results.append(StepResult(req_id, (), None))
                continue
            token = int(out_ids[i].item() if hasattr(out_ids[i], "item") else out_ids[i][0])
            if position < self._tokens.shape[1]:
                self._tokens[i, position] = token
            finish = "stop" if token == self.eos_token_id else None
            results.append(StepResult(req_id, (token,), finish))
        return results


def _new_tokens_buffer(batch: int, total: int, rows, prompt_lens):
    """Build the [B, T] right-padded buffer; positions past each prompt are -1 padding
    but are filled with predictions as generation proceeds (reference convention:
    only already-filled positions are ever passed to the model)."""
    import torch

    tokens = torch.full((batch, total), 0, dtype=torch.long)
    for i, (r, plen) in enumerate(zip(rows, prompt_lens)):
        tokens[i, :plen] = torch.tensor(r.prompt_tokens, dtype=torch.long)
    return tokens


def _stack_token_types(rows, seqlen):
    """Stack per-row token_types for the prefill chunk; None when no row is VL."""
    if not any(r.token_types is not None for r in rows):
        return None
    import torch

    types = torch.full((len(rows), seqlen), TEXT, dtype=torch.long)
    for i, r in enumerate(rows):
        if r.token_types is not None:
            t = torch.tensor(r.token_types[:seqlen], dtype=torch.long)
            types[i, : t.numel()] = t
    return types

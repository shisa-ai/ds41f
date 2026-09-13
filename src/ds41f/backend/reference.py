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
DEVICE = "cuda"  # set by ReferenceBackend.__init__ (module-level default)


class ReferenceBackend:
    def __init__(
        self,
        model,
        eos_token_id: int,
        sampler=None,
        device: str = "cuda",
        apply_placement: bool = False,
    ):
        self.model = model
        self.eos_token_id = eos_token_id
        # explicit device: torch's default device is thread-local, and the engine
        # runs execute() on its own thread where no default was set
        self.device = device
        global DEVICE
        DEVICE = device
        # sampler(logits_row, params) -> token_id; default greedy
        self.sampler = sampler
        # Expert placement is a collective (one all_to_all per layer), so it can only
        # run here when every TP rank builds its own backend. Single-controller
        # deployments (rank 0 owns this object, the others are driven inside
        # model.forward) must instead call expert_placement.maybe_apply on every rank
        # where the model is loaded. Opt-in, because getting this wrong hangs.
        self.placement_applied = False
        if apply_placement:
            from . import expert_placement

            self.placement_applied = expert_placement.maybe_apply(model)
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
        self._rows = rows
        self._params = [(r.temperature, r.top_p) for r in rows]
        # prefix-cache restore: the row's state for tokens [0, shared) is supplied
        # by the snapshot; no start_pos=0 forward runs, and the suffix is consumed
        # by ordinary decode steps (teacher-forced until the prompt is exhausted).
        # Exclusive single-row cohorts only (the engine enforces this).
        if all(getattr(r, "meta", None) and getattr(r.meta, "prefix_hit", None) for r in rows) and len(rows) == 1:
            return self._prefill_from_snapshot(rows[0])
        if any(getattr(r, "meta", None) and getattr(r.meta, "prefix_hit", None) for r in rows):
            raise RuntimeError("prefix-hit rows must be scheduled in exclusive single-row cohorts")
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

        out_ids, logits, _ = self.model.forward(
            self._tokens[:, :min_len],
            0,
            images=[r.images for r in rows] if any(r.images for r in rows) else None,
            token_types=_stack_token_types(rows, min_len),
        )
        self._prev_pos = min_len
        self._cur_pos = min_len
        return self._collect(out_ids, logits, min_len)

    def _prefill_from_snapshot(self, row):
        """Restore row state from `row.meta` and position the cursor at the shared
        boundary. meta carries: prefix_hit (truthy), shared_len, restore(payload)
        callable, and optionally an engine-side hook for multi-rank restore.

        A multi-token suffix is consumed by ONE chunked forward (start_pos=shared,
        seqlen>1): measured ~5x faster than teacher-forced single-token steps for
        the 4600+19 benchmark scenario. A single remaining token stays on the
        decode path so the chunked branch never runs with seqlen == 1."""
        meta = row.meta
        self._tokens = _new_tokens_buffer(1, len(row.prompt_tokens) + row.max_new_tokens + 1,
                                           [row], [len(row.prompt_tokens)])
        self._prompt_lens = [len(row.prompt_tokens)]
        self._req_ids = [row.req_id]
        meta.restore()
        self._prev_pos = meta.shared_len
        self._cur_pos = meta.shared_len
        plen = len(row.prompt_tokens)
        suffix = plen - meta.shared_len
        if suffix > 1:
            out_ids, logits, _ = self.model.forward(self._tokens[:, meta.shared_len:plen], meta.shared_len)
            self._cur_pos = plen
            return self._collect(out_ids, logits, plen)
        # single-token suffix: no forward, no emission -- the first decode step
        # consumes prompt[shared]
        return [StepResult(row.req_id, (), None)]

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
        out_ids, logits, _ = self.model.forward(self._tokens[:, cur : cur + 1], cur)
        self._cur_pos = cur + 1
        return self._collect(out_ids, logits, cur + 1)

    # -- shared --------------------------------------------------------------

    def _collect(self, out_ids, logits, position: int) -> list[StepResult]:
        """Apply prompt override, sample per row, decide emission and finishes.

        Sampling is batched over the emitting rows: one device-side gather, one
        top-p pass, one sampling draw and a single host transfer, instead of a
        per-row `.item()`/softmax pair and a per-row Python->CUDA token write."""
        total = self._tokens.shape[1]
        emit = [i for i, plen in enumerate(self._prompt_lens) if position >= plen]
        sampled = self._sample_batch(emit, out_ids, logits) if emit else None
        if sampled is not None and position < total:
            # keep next-step ids on device: one indexed write, no Python round-trip
            import torch

            idx = torch.tensor(emit, dtype=torch.long, device=self._tokens.device)
            self._tokens[idx, position] = sampled.to(
                device=self._tokens.device, dtype=self._tokens.dtype
            )
        tokens = sampled.tolist() if sampled is not None else []
        by_i = dict(zip(emit, tokens))
        results = []
        for i, req_id in enumerate(self._req_ids):
            if i not in by_i:
                # this row's prediction slot is still inside its prompt:
                # teacher-forced, nothing emitted. The ground-truth token already
                # sits in the buffer for the next step to consume.
                results.append(StepResult(req_id, (), None))
                continue
            token = by_i[i]
            finish = "stop" if token == self.eos_token_id else None
            results.append(StepResult(req_id, (token,), finish))
        return results

    def _sample_batch(self, emit, out_ids, logits):
        """Batched per-row sampling over `emit` (indices into the cohort rows).
        temperature 0 = greedy; otherwise top-p then Gumbel-max, matching the
        per-row distribution. Returns a device tensor of token ids."""
        if logits is None:
            # stub models (or engines that sample in the model) provide ids directly
            import torch

            if torch.is_tensor(out_ids):
                return out_ids[emit]
            return torch.tensor(
                [
                    int(out_ids[i]) if not hasattr(out_ids[i], "__len__") else int(out_ids[i][0])
                    for i in emit
                ],
                dtype=torch.long,
                device=self._tokens.device,
            )
        import torch

        idx = torch.tensor(emit, dtype=torch.long, device=logits.device)
        rows = logits.index_select(0, idx).float()
        temps = torch.tensor(
            [self._params[i][0] for i in emit], dtype=torch.float32, device=rows.device
        )
        tops = torch.tensor(
            [self._params[i][1] for i in emit], dtype=torch.float32, device=rows.device
        )
        greedy = temps <= 0
        safe_temps = torch.where(greedy, torch.ones_like(temps), temps)
        scaled = rows / safe_temps.unsqueeze(1)  # argmax is scale-invariant
        # No host-side branch on the parameters: `bool(tops.any())` and `bool(greedy.any())`
        # each drain the pipeline once per step, and both guards are only shortcuts.
        # The top-p block is already a no-op for tops >= 1.0 (the `keep |` clause below
        # keeps every position), and torch.where is correct when no row is greedy.
        sorted_logits, order = scaled.sort(dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_logits, -1)
        cum = sorted_probs.cumsum(-1)
        # keep everything strictly before the mass crosses top_p
        keep = (cum - sorted_probs) < tops.unsqueeze(1)
        keep[:, 0] = True
        keep = keep | (tops.unsqueeze(1) >= 1.0)
        filtered = torch.full_like(scaled, float("-inf"))
        filtered.scatter_(-1, order, torch.where(keep, sorted_logits, float("-inf")))
        scaled = filtered
        probs = torch.softmax(scaled, -1)
        # Gumbel-max: argmax_i log p_i + Gumbel_i == argmax_i p_i / Exp_i
        u = torch.rand_like(probs).clamp_min(torch.finfo(probs.dtype).tiny)
        gumbel = -torch.log(-torch.log(u))
        sampled = (probs.log() + gumbel).argmax(-1)
        # greedy rows ignore top_p, so they take the raw argmax; torch.where is correct
        # when no row is greedy, so this needs no host-side branch.
        return torch.where(greedy, rows.argmax(-1), sampled)


def _new_tokens_buffer(batch: int, total: int, rows, prompt_lens):
    """Build the [B, T] right-padded buffer; positions past each prompt are -1 padding
    but are filled with predictions as generation proceeds (reference convention:
    only already-filled positions are ever passed to the model)."""
    import torch

    tokens = torch.full((batch, total), 0, dtype=torch.long, device=DEVICE)
    for i, (r, plen) in enumerate(zip(rows, prompt_lens)):
        tokens[i, :plen] = torch.tensor(r.prompt_tokens, dtype=torch.long, device=DEVICE)
    return tokens


def _stack_token_types(rows, seqlen):
    """Stack per-row token_types for the prefill chunk; None when no row is VL."""
    if not any(r.token_types is not None for r in rows):
        return None
    import torch

    types = torch.full((len(rows), seqlen), TEXT, dtype=torch.long, device=DEVICE)
    for i, r in enumerate(rows):
        if r.token_types is not None:
            t = torch.tensor(r.token_types[:seqlen], dtype=torch.long, device=DEVICE)
            types[i, : t.numel()] = t
    return types

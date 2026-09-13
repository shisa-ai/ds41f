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

from dataclasses import dataclass
from typing import Optional, Sequence

from ..scheduler.scheduler import PlanRow, StepPlan
from ..state.slots import StateStore
from .base import StepResult

@dataclass
class _Staged:
    """One step whose token read is in flight.

    `skeleton` holds (req_id, position in the read buffer or None) per cohort row in
    row order; None means the row emitted nothing this step. `pinned` is the host
    buffer being filled and `event` marks the compute-stream point at which it is
    valid; `event` is None when the read was already synchronous.
    """

    skeleton: list
    pinned: object
    event: object
    count: int


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
        # Pinned staging for the asynchronous token read, double-buffered because
        # the engine may hold one step's read in flight while the next is issued.
        self._pin = [None, None]
        self._pin_slot = 0

    # -- Backend protocol ---------------------------------------------------

    def execute(self, plan: StepPlan, state: StateStore) -> list[StepResult]:
        return self.resolve(self.enqueue(plan, state))

    def enqueue(self, plan: StepPlan, state: StateStore):
        """Run one step and start reading its sampled tokens, without waiting.

        Everything the step computes stays on the device: the forward, the
        sampling draw and the token write-back into `_tokens`. The only host
        interaction is a non-blocking copy of the sampled ids into a pinned
        buffer, issued on the compute stream at the point the sampling kernel
        retires -- which is *before* the next step is enqueued, so a later
        blocking read of that buffer does not wait for the next step.

        That ordering is the whole point. `sampled.tolist()` inside execute() is
        a stream-ordered blocking copy, so calling it after enqueueing step i+1
        would wait for step i+1: the host would still drain the pipeline, and the
        exposed drain is 3.12 ms/step (probe_decode_cpu.py, 27.99 ms against
        24.87 ms). Splitting enqueue from resolve is what lets the engine keep the
        queue non-empty while it reads the previous step's token.

        Returns an opaque handle for `resolve`.
        """
        if plan.op == "prefill":
            staged = self._prefill(plan)
        elif plan.op == "decode":
            staged = self._decode(plan)
        else:
            raise ValueError(f"unknown plan op {plan.op!r}")
        return staged

    def resolve(self, pending) -> list[StepResult]:
        """Wait for a step's token read and build its results."""
        return self._unstake(pending)

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
        # One position of slack. The serial engine runs exactly `window` decode steps
        # for a cohort; the pipelined loop builds step i's plan before step i-1's
        # tokens are read, so the rows that finish last are not yet known to be
        # finished and the cohort takes one more step. That step's own tokens are
        # discarded, but `_decode` refuses to run past the buffer, so the buffer has
        # to allow it.
        total = min_len + window + 1
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

    def _collect(self, out_ids, logits, position: int) -> "_Staged":
        """Apply prompt override, sample per row, and start reading the result.

        Sampling is batched over the emitting rows: one device-side gather, one
        top-p pass, one sampling draw and a single host transfer, instead of a
        per-row `.item()`/softmax pair and a per-row Python->CUDA token write.

        The emit set is a function of `position` against each row's prompt length
        and nothing else. It is deliberately *not* a function of finish state: a
        finished row stays in the fixed cohort and keeps drawing until the cohort
        drains, so the number of draws per step -- and therefore the global
        generator's stream -- is the same whether a row has finished or not. That
        property is what makes a pipelined delivery safe, because a pipelined step
        reads the previous step's tokens after the next step has been enqueued.
        """
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
        return self._stake(emit, sampled)

    def _stake(self, emit, sampled) -> "_Staged":
        """Start the host read of this step's sampled ids.

        The skeleton is everything about the step that does not depend on the token
        values: which rows emitted and in what order. Finish detection stays in
        `_unstake`, because it compares each token against the EOS id.
        """
        emit_at = {row: k for k, row in enumerate(emit)}
        skeleton = [(req_id, emit_at.get(i)) for i, req_id in enumerate(self._req_ids)]
        if sampled is None:
            return _Staged(skeleton, None, None, 0)
        import torch

        count = int(sampled.numel())
        if sampled.is_cuda:
            # Non-blocking into pinned memory. Issued here, at the end of this
            # step's work and before the next step is enqueued, so waiting on the
            # event later does not wait for the next step. Double-buffered: the
            # engine holds one step's read in flight while it issues the next.
            pin = self._pin[self._pin_slot]
            if pin is None or pin.numel() < count:
                # device="cpu" explicitly: the inference harnesses call
                # torch.set_default_device("cuda"), and a pinned tensor cannot be
                # allocated on the device ("Only dense CPU tensors can be pinned").
                pin = torch.empty(count, dtype=torch.long, pin_memory=True, device="cpu")
                self._pin[self._pin_slot] = pin
            self._pin_slot ^= 1
            pin[:count].copy_(sampled.reshape(-1), non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            return _Staged(skeleton, pin, event, count)
        # not a CUDA tensor (stub models, CPU tests): read it where it is
        return _Staged(skeleton, sampled.reshape(-1), None, count)

    def _unstake(self, staged) -> list[StepResult]:
        """Wait for a step's token read and build its results."""
        if not isinstance(staged, _Staged):
            return staged  # a literal result list (prefix-hit single-token suffix)
        if staged.event is not None:
            staged.event.synchronize()
        tokens = staged.pinned[: staged.count].tolist() if staged.pinned is not None else []
        results = []
        for req_id, at in staged.skeleton:
            if at is None:
                # this row's prediction slot is still inside its prompt:
                # teacher-forced, nothing emitted. The ground-truth token already
                # sits in the buffer for the next step to consume.
                results.append(StepResult(req_id, (), None))
                continue
            token = tokens[at]
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

        # Temperatures and top_p arrive from the plan as Python floats, so the branch
        # on them is a host-side branch that costs no device synchronization -- unlike
        # `bool(temps.any())`, which drains the pipeline once per step. It matters for
        # more than speed: a greedy request must not consume random numbers, or a
        # greedy workload advances the global generator and changes the stream seen by
        # every later stochastic request. The pre-batching path drew nothing for a
        # greedy row, so this restores that.
        params = [self._params[i] for i in emit]
        stoch = [k for k, (t, _) in enumerate(params) if t > 0]
        if not stoch:
            return rows.argmax(-1)
        needs_topp = any(p < 1.0 for _, p in params)

        s_idx = torch.tensor(stoch, dtype=torch.long, device=logits.device)
        s_rows = rows.index_select(0, s_idx)
        temps = torch.tensor([params[k][0] for k in stoch], dtype=torch.float32, device=rows.device)
        tops = torch.tensor([params[k][1] for k in stoch], dtype=torch.float32, device=rows.device)

        scaled = s_rows / temps.unsqueeze(1)
        if needs_topp:
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
        # Gumbel-max: argmax_i log p_i + Gumbel_i == argmax_i p_i / Exp_i. One draw per
        # stochastic row, in row order, so the generator advances exactly as it would
        # have under the per-row sampler.
        u = torch.rand_like(probs).clamp_min(torch.finfo(probs.dtype).tiny)
        gumbel = -torch.log(-torch.log(u))
        sampled = (probs.log() + gumbel).argmax(-1)

        if len(stoch) == rows.shape[0]:
            return sampled
        # greedy rows ignore top_p and the RNG entirely, so give them only their argmax
        greedy = [k for k, (t, _) in enumerate(params) if t <= 0]
        g_idx = torch.tensor(greedy, dtype=torch.long, device=logits.device)
        out = torch.empty(rows.shape[0], dtype=torch.long, device=rows.device)
        out[g_idx] = rows.index_select(0, g_idx).argmax(-1)
        out[s_idx] = sampled
        return out


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

# Optimization punchlist

Priority is expected end-to-end impact relative to effort and confidence: compare
proven kernels and transports before writing replacements. The order is provisional
until a fresh profile identifies current bottlenecks. There are no promised gains.

Target: **4 × H20-3e, TP=4**. H20 is Hopper: native FP8 tensor cores, **no native
FP4/MXFP4/NVFP4 tensor cores**. FP4 weights need a supported conversion/compute
path. Dense FP8 GEMV does not establish the best path for FP4 experts. Ampere and
Blackwell results do not transfer directly.

## Current checklist — September 14, 2026

The campaign is **paused at iteration 17, with ten accepted changes**. GPU work
is paused for other workloads. Both repositories were pushed after the stopping-point
review. The engine suite passed 87 tests with one skipped using a CPU override;
that check did not revalidate GPU execution.

| Area | Where we are | Remaining work |
| --- | --- | --- |
| Decode fusion | Accepted changes are enabled by default; the adjusted campaign estimate is 28.04 → 25.97 ms/step. | Rerun the headline and served benchmarks when GPUs are available; do not turn the adjusted estimate into a throughput claim. |
| Headline measurements | README uses the passing `trusted-rope-fused-4rep.json`: 35.2–36.4 tok/s across its context lengths. It predates later changes. | Measure current defaults under consistent conditions, with a new run manifest. |
| Batched decode | Graphs cover batches through 8; small batches use the decode expert kernels. Saved batch-2/4/8 rates are 39.7/65.2/97.2 aggregate tok/s. | Refresh these measurements after the latest fusions; qualify representative mixed workloads. |
| Serving and admission | Packed tensor broadcasts and header cleanup landed. An optional 8 ms admission window measured 40.80 → 52.42 aggregate tok/s at concurrency 4. | Continuous admission and per-row positions remain open; the admission window only helps form a new cohort. |
| Overlapped token delivery | Implemented, tested and removed: no served gain, with an extra-step RNG hazard. The enqueue/resolve interface remains. | Revisit only with evidence that a revised rank-coordination protocol removes the blocking dependency. |
| Custom allreduce | Integration works and remains opt-in; served latency measured about 29.0 → 27.6 ms. | Qualify model-level numerical/quality effects before making it the default; generated tokens differ. |
| Marlin | Isolated expert benchmark measured 2.8× faster execution; no model integration. | Match activation precision in an independent reference, test real layer inputs/routing, integrate optionally, then measure end to end. |
| Attention output projection | Tested FP8 implementation was slower than BF16. | No further work on that candidate unless new evidence changes the comparison. |
| Placement and provenance | Loader application, per-rank mapping audit, artifact hashing and manifests landed. | Held-out real-traffic calibration and qualification remain open. |
| Prefill correctness | Last-position checks pass in the headline artifact; whole-prompt equivalence is unresolved. | Independent state/routing comparisons and held-out quality evaluation. Keep timing separate from full-logit checks. |
| Small remaining operations | Sinkhorn costs about 0.31 ms/step; two mean reductions total about 0.63 ms in the recorded attribution. | Smaller-block Sinkhorn sweep is unfinished. Reduction-order compatibility remains unresolved. These costs are not promised savings. |
| DSpark | Source feasibility analysis complete; no custom-engine runtime benchmark. | Build the greedy single-request draft/verify/commit prototype in section 5. |
| Topology and broader scheduling | No completed EP/pipeline topology or continuous-admission implementation. | Defer topology changes until current profiles justify them. |

Next work order, when GPU access resumes:

1. Record a fresh baseline for current defaults, including served and batched
   execution, and preserve the configuration, placement hash and actual token counts.
2. Run a bounded Marlin qualification and integration experiment. The historical
   2.6 ms saving is an extrapolation, not a measured current-model gain.
3. Start the DSpark correctness prototype early; measure the six-position verifier
   and draft/commit cost before expanding its serving interface.
4. Continue independent prefill-state and quality qualification, including the
   optional custom-allreduce backend's numerical effects.
5. Revisit the small reduction kernels only as bounded experiments. Preserve the
   existing working implementations until correctness and an end-to-end gain pass.

The dated review below is retained as history. Its defect list describes the
original pass; this checklist is the current work order.

Status (September 13, 2026): the first custom-engine tuning pass is complete,
but this punchlist is **not fully implemented or qualified**. The review below
records implementation defects, evidence limits and the next work order. Keep the
dedicated [vLLM implementation](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
as the production comparison. Section 5 now includes our source-level DSpark
feasibility analysis; runtime integration and its performance gate remain open.

Status (September 14, 2026) -- the decode fusion campaign. Section 3's pass took
the decode step from **28.04 to 25.97 ms/step, -7.4%**, over ten kept changes,
every one behind a default-on flag with a bit-exactness gate of its own. It was
tracked on `bench_ab.py` (2K prompt, 30 decode steps, 9-15 interleaved repeats per
arm, step graph rebuilt per arm, identical tokens), and each change's raw samples
are in [`results/`](../results/). The last three are the third fusion pass
(+1.345 ms/step, +4.90%, measured as one change set rather than as the sum of its
four flags), the `act_quant` output-buffer cache (+0.078) and the gate's pre-top-k
chain (+0.072).

Two things to carry forward from it. The metric series is **drift-anchored, not
raw**: the machine was shared with another 131 GB job for the last iterations, so
each metric is that run's on-arm median corrected by the same run's off-arm offset
-- an estimate that remains sensitive to changes in contention between arms.
And the campaign's headline throughput table is a *different* harness, so it needs its own re-run
rather than being scaled from this series; the README says so where it quotes it.
What is left in section 3 is now tabulated with its measured size and its specific
blocker. Marlin and DSpark are broader remaining opportunities; the two
`aten::mean` sites are a smaller numerical-compatibility experiment.

**Decode, change by change.** Every row is that change's own interleaved A/B
(`bench_ab.py`, 2K prompt, 30 decode steps, B=1, TP4, worst rank), named by its
artifact in [`results/`](../results/), so each row's `off` arm is the configuration
as it stood when that change was measured. Two rows are change *sets* measured with
all their flags toggled together, which is why the four flags inside each do not
appear separately: adding a set's members up would double-count them.

| Change | Saved ms/step | Latency reduction | Artifact |
| --- | ---: | ---: | --- |
| First decode fusion pass (4 changes, one set) | +3.883 | +11.91% | `ab-combined-decode-fusion` |
| RoPE fused | +0.651 | +2.30% | `ab-rope-fused-b` |
| Engram gather LUT | +0.097 | +0.35% | `ab-engram-lut-b` |
| Gate weight cache | +0.455 | +1.65% | `ab-gate-weight-cache` |
| Shared-expert SwiGLU fused | +0.451 | +1.67% | `ab-expert-swiglu-fused` |
| MoE input row quantized once | +0.244 | +0.92% | `ab-moe-shared-quant` |
| Attention input quantized once | +0.115 | +0.44% | `ab-attn-shared-quant` |
| Third fusion pass (4 flags, one set) | +1.345 | +4.90% | `ab-third-fusion-pass-combined` |
| `act_quant` output-buffer cache | +0.078 | +0.30% | `ab-act-quant-cache-on-by-default` |
| Gate pre-top-k fused | +0.072 | +0.28% | `ab-gate-prep-fused` |

These per-change savings must not be added: baseline configurations, machine
conditions and overlapping changes differ. The chained estimate previously
reported as 19.8% and the reference-normalized 17.7–21.1% range depend on
assumptions about contention. Neither replaces a direct current-versus-baseline
measurement. The adjusted campaign series covers only its later portion.

The one-launch RMSNorm remains disabled because it changes numerical results.
Custom allreduce remains opt-in: its earlier `ab-custom-ar` null result did not
exercise the intended backend and is superseded by the served integration results
in section 2. The first activation-quantization cache experiment is superseded by
the later measurement and default-on implementation above.

The [README](../README.md#performance) reports the trusted, passing run
(`trusted-rope-fused-4rep.json`). The separate full-prompt diagnostic run in
[OPTIMIZE-RESULTS.md](OPTIMIZE-RESULTS.md#full-prompt-diagnostic-run) is marked
failed on correctness checks.
The review below retains the earlier tuning-pass numbers to explain its findings.

**The first pass's measurements are in [OPTIMIZE-RESULTS.md](OPTIMIZE-RESULTS.md).
The review below supersedes its incorrect parity interpretation and its claim
that alternative collective backends were benchmarked.**

**DSpark is a high-impact integration project, not a low-priority tuning idea.**
Our vLLM experience was roughly 110 → 350 tok/s for a single stream. The local
report distinguishes overall output from decode-only rates; match those metrics
before quoting a multiplier. DSpark has already been exercised locally in vLLM.
The source assessment in section 5 recommends an early custom-engine runtime
prototype alongside bounded kernel/transport comparisons.

## September 13 review and immediate next steps

Scope: inspected the engine, the external reference implementation under
`/root/glm-testing/ds41f/inference`, the serving loader, and the saved benchmark
JSON. Re-ran the engine suite: 37 tests passed at review time. Reproduced the
scheduler and greedy-RNG issues below independently. The full TP4 performance
campaign was **not** rerun by this review.

The saved optimized-arm averages support approximately 38% higher decode
throughput and prefill rates of 1,330 / 2,580 / 3,392 tok/s at 512 / 2048 / 8192
tokens. Small reporting correction: `baseline-gpu0123.json` averages **44.81**
ms/token at 2048, not 44.65; 44.65 was the earlier historical anchor. The shipped
artifact averages 32.46 ms/token. These are model-only, random-token measurements,
not a serving result or a matched vLLM speedup.

| Finding in the reviewed pass | Required closure evidence |
| --- | --- |
| Placement was applied by benchmarks/profilers, but not by `serve/server.py:load_model_rank` or `generate.py`. A file beside the module alone did not enable it in serving. | Apply after weight loading and before pointer-table construction/graph capture on every rank; verify actual serving dispatch and original-to-permuted expert identity. Pin/hash the calibration artifact and evaluate held-out real traffic. The report estimates about 15% lower 8K throughput without placement. |
| `prefill_top1_agreement` measured one last-position prediction at B=1, not 8192 prompt positions. `ParallelHead.forward` sliced `x[:, -1]`. | Correct the results writeup; compare selected/all prompt logits in an untimed correctness arm. A changed argmax and a small max-logit difference are compatible, not contradictory criteria. Keep `hc_mixes` disabled until it passes qualified gates. |
| Decode arms restored the same reference post-prefill snapshot; placement and several kernel defaults were shared between arms. | Independently produce and compare candidate/reference states and routing; run continuations from each arm's own state. Preserve separate same-state kernel tests. Add held-out natural-generation/task-quality evaluation; identical decode logits alone do not qualify prefill or shared changes. |
| `bench_collectives.py` timed NCCL algorithms/protocols. Its `--custom` branch only constructed a communicator and reported availability. | Actually invoke and time custom, FlashInfer and symmetric-memory backends with numerical checks, graph replay and all-rank traces. NCCL-only latency does not rule out a faster backend. |
| Batched sampling still used two CUDA-to-Python `bool(...any())` checks, rebuilt device parameter/index tensors, and ran stochastic work even for greedy rows. | Use persistent cohort metadata/buffers, eliminate device-dependent Python branches, add a greedy fast path, and trace the complete serving token-feedback path. Test RNG semantics: the reviewed greedy path advanced RNG state. |
| Scheduler stopped scanning after an oversized candidate, assuming the arrival queue was length-sorted. `[1000, 8192, 1050]` admitted only `[1000]`. | Admit compatible later candidates within a bounded scan, preserve fairness, and regression-test both mismatch directions plus queue wait and useful throughput. Symmetric spread alone did not finish this fix. |
| Step-graph dispatch required input shape `(1, 1)`; Engram remained outside replay and used CPU gather/dequantization. | Profile B=1/2/4/8 and exposed Engram/host coordination cost before choosing capture or lookup work. Do not extrapolate B=1 graph gains to concurrent serving. |
| Marlin/`wo_a`, DSpark integration, continuous admission and topology experiments remained undone. | Track these as open milestones. E2M1/E8M0 qualification is part of the Marlin comparison's scope, not evidence that the comparison can be skipped. |

The source tree is being updated concurrently. In particular, a new
`Transformer.forward(..., full_logits=True)` option appeared during the DSpark
analysis, and sampling/placement work was in progress. These are findings about
the reviewed pass, **not claims that every defect persists in the newest working
copy**. Close each row only with its regression/serving evidence. The original
`trusted-shipped.json` still measures last-position prefill logits; a new API
does not retroactively change that artifact.

Original September 13 work order (historical; superseded by the current checklist):

1. **Repair the evidence and serving defects above.** Pin a new source/config/
   placement manifest. Separate correctness runs from timing; full-vocabulary
   logits at every 8K position would add about 3.95 GiB and substantial head work
   at B=1, which must not contaminate a generation-prefill timing.
2. **Establish matched AR and current all-rank profiles.** Same checkpoint,
   traffic, sampling, context, cache/offload configuration and output lengths;
   include real HTTP serving and B=1/2/4/8. Measure object broadcasts and token
   copies as well as sampling: the current TP wrapper serializes tokens through
   CPU Python objects on every forward.
3. **Run the production-kernel and actual alternative-collective comparisons.**
   Include FP4 experts, dense projections and `wo_a`; use current profiles to
   prioritize Engram lookup and graph coverage alongside them.
4. **Execute section 5's B=1 DSpark correctness/performance prototype early.**
   Source feasibility is now assessed. The first decision is whether a correct
   six-position target verifier plus draft/commit overhead beats AR; do not wait
   for an EP rewrite, and do not enable speculation on unverified cache state.
5. **Expand only measured winners.** Add stochastic/block verification and
   adaptive policy, then concurrent state/serving support. Revisit admission and
   topology when cohort waste or rank traces justify the work.

## Correctness gates

1. Qualify full-model TP4 serving with Engram host offload: fidelity, memory,
   short/long context and concurrent requests.
2. Resolve prefill instability and repetitive-prompt quality with independent
   state/operation checks and held-out task evaluation. Decode equality from the
   same restored post-prefill state does not establish prefill equivalence.
3. Only then qualify prefix-hit admission and repeated HTTP profitability.
   Continuous admission, preemption and speculative decoding remain separate
   milestones with their own state and lifecycle checks.
4. For a smaller deployment, choose GPU/host memory, context, concurrency and
   latency budgets before choosing quantization. W3 weight bytes alone do not
   establish a 256 GB total-memory fit. This conditional work does not block
   optimization of the existing H20 system.

Every candidate needs independent baseline/candidate state, exact checks for
positions, masks, cache ownership and expert selection, plus documented numerical
and task-quality checks. Preserve activation quantization, route-weight placement,
clipping and rounding semantics unless a separately qualified change is intended.
Do not relax thresholds to promote a failing optimization.

## 0. Current profiling and matched vLLM baseline

Use the README's `trusted-rope-fused-4rep.json` as the latest saved passing
headline result, and collect a fresh current-default baseline before new tuning.
`trusted-shipped.json`'s 32.46 ms/token is the historical pre-fusion anchor;
`baseline-gpu0123.json` averages 44.81 ms/token at 2K.
Historical 99 ms expert and 138 ms NCCL timings predate major optimizations;
neither represents time available to save now. NCCL durations include rank
waiting: expert and collective savings must not be double-counted.

- Pin checkpoint, source revisions, kernel flags, GPU topology and runtime
  configuration. Warm exact shapes, JIT and graphs; restore independently verified
  state before each arm. Alternate arm order and repeat.
- Compare reference versus vLLM **autoregressive first**, then DSpark separately.
  Match prompts, context, sampling, output budgets, offload and cache state. Keep
  teacher-forced kernel timing separate from natural generation and HTTP timing.
- Measure B=1/2/4/8 and intended higher concurrency, short/long prompts, mixed
  prompt/output lengths, and cold/warm prefixes. Report TTFT, queue delay,
  p50/p95/p99 inter-token latency, useful aggregate output and peak memory.
  For speculation, include inter-burst delay as well as average tokens/second.
- Capture all ranks: component/kernel times, collective sizes and arrival skew,
  expert occupancy, achieved bandwidth, host synchronization and graph coverage.
  Count prompt-tail single-token steps and finished-row computation. Reuse the
  existing profilers; add `--profile`, `--kernel-profile` and `--kernel-trace FILE`
  only where instrumentation is missing.

The local September 11 evaluation reports 117.78 tok/s vLLM AR decode and
5,739 tok/s 8K cold prefill, versus this engine's 22.40 and 1,973. These are different
benchmark protocols, not a controlled speedup claim or broad quality qualification.
The evaluated vLLM startup selects Marlin MXFP4 experts and Marlin MXFP8 dense
kernels. Preserve run revisions and artifacts; recipe defaults alone do not
identify the executed kernels.

## 1. Marlin and existing-kernel performance shootout

Compare our FP4 expert GEMV/grouped GEMM and FP8 dense paths with the
Hopper-compatible Marlin implementations used by the evaluated vLLM. Include
other supported production backends that cover our formats and shapes.

- Sweep actual w1/w3/w2 and dense shapes, B=1 through batched decode and prefill,
  and measured tokens-per-expert distributions. Include routing, packing,
  activation quantization, combination and workspace costs in MoE results.
- Verify E2M1 weights, E8M0 scales, padding, accumulation, SwiGLU and route-weight
  semantics. Accepting “FP4” does not establish format or numerical compatibility.
- Include attention `wo_a`: our reference expands it to BF16 and uses grouped
  `einsum`; vLLM's FlashMLA path has an FP8 output-projection implementation.
- Select the fastest qualified path per shape regime when useful, with a fallback.
  Repack at load time and record startup, resident and peak memory. Promote only
  after full-layer and end-to-end validation.

**Priority rationale:** potentially large kernel gains with less new implementation
than designing a weight layout and GEMM from scratch. Matched gains are unmeasured.

**Status (September 13).** The MoE half is measured. Against vLLM's Marlin MXFP4
kernels, under graph replay, at the real per-rank decode mix (one token, top-6 of
384 global experts, 96 owned locally), Marlin is **2.80× faster** than the engine's
grouped GEMV, averaged over 14 cases (2.3× with no active local expert, 3.2× with
six). The earlier profile assigned 4.13 ms/step to the two expert projections.
Applying the ratio suggests about 2.6 ms/step saved, but the benchmark times a
whole expert block, so even that scope differs. This is a historical projection,
not a current end-to-end gain. Not integrated: the engine
quantizes activations to FP8 and Marlin uses bf16, so the paths are not
numerically interchangeable. The `wo_a` half is measured and rejected — the FP8
grouped GEMM is 1.4-7× slower than the bf16 `einsum` at every shape, and the bf16
op is only 1.9% of a decode step to begin with. See [Optimization
results](OPTIMIZE-RESULTS.md#fp4-expert-moe-against-vllms-marlin-mxfp4-kernels).

## 2. Benchmark TP collectives and select the fastest qualified path

Keep TP4 and compare NCCL with supported custom, FlashInfer and symmetric-memory
collectives on our actual four-GPU topology. The evaluated vLLM registers several
in dispatch order; registration does not prove which handles each call.

- Benchmark actual message sizes, dtypes and graph-replayed calls; check P2P and
  NVLink connectivity. Separate all-rank arrival skew from communication cost.
- Select the fastest correct backend per message/shape regime if there is no
  universal winner. Verify dispatch with traces and retain fallbacks.
- Confirm end-to-end gains under realistic rank skew and concurrency. Preserve
  required mathematical reductions. Communication-plus-norm fusion may reduce
  traffic and launches without eliminating a dependency.

**Priority rationale:** a bounded transport experiment before an EP/pipeline
rewrite. Re-profile after MoE changes because they alter collective waiting.

**Status (September 13).** Measured in [OPTIMIZE-RESULTS.md](OPTIMIZE-RESULTS.md#collectives),
and the decode backend is now switchable. Custom allreduce is
CUDA-graph-capturable under vLLM's supported capture procedure and is 1.3-1.6x
faster than graphed NCCL at decode sizes; routed through the engine behind
`DSV41F_CUSTOM_AR=1 DSV41F_EXPANDABLE_SEGMENTS=0` it takes the served decode step
from 29.0 to 27.6 ms (4.8%). Captured call sites differ from NCCL by at most
1.9e-06; eager BF16 call sites differ by up to 0.03125. These local checks do not
establish full-model equivalence.
It stays opt-in because it moves generated tokens on a near-tie-sensitive model.
FlashInfer is 1.2-1.3x faster than eager NCCL at decode sizes, no faster at
prefill sizes, and was not integrated. Both decline or are unsupported at prefill
sizes, so prefill keeps NCCL.

The two bugs that made an earlier pass call this a kernel-level failure were the
caching allocator (graph-buffer registration cannot export expandable-segment
memory) and a collective captured outside `CustomAllreduce.capture()`. Both are
isolated by `inference/check_custom_ar_ipc.py` without loading the model.

Still open under this heading: the gain has not been re-measured at concurrency
2/4, where the step is batched and the collective competes with more work.

## 3. Qualify existing fusion and close graph/host gaps

The first pass enabled fused `hc_pre`/`hc_post` after correcting association order.
Retain their exact operation checks and full-model gates. `hc_mixes` remains off;
its failing last-position argmax needs the corrected numerical/state evaluation
described above. Availability does not establish numerical safety.

Audit the serving path for device-resident positions, persistent buffers, captures
for relevant batch sizes, stable addresses and safe inactive-row handling. Trace
Engram, sampling and TP coordination across replay boundaries. One graph per GPU
does not prove that the whole serving step is captured.

Then fuse measured expensive sequences: RMSNorm, RoPE/quantization/cache writes,
SwiGLU/clipping/rounding and hyper-connections. Launch counts are diagnostics, not
a promised ~100-to-26 target. Fusion is one lever alongside better kernels,
allocation removal and host/device overlap.

**Status (September 13).** The audit is done for B=1 at 2K context and it changes
the shape of this section. It launches **6,232 kernels per step** with a median
duration of 1.82 µs, and 43% of the kernel time sits in 5,731 elementwise, copy,
reduce and activation quantization kernels; the five large groups (dense GEMV,
sparse attention, both MoE projections, all-reduce) are 51% of the time in 501
launches. The original note here read "the step is GPU-bound: the host enqueues a
step in 0.02 ms", and **that 0.02 ms figure is withdrawn** -- it timed the gap
between loop iterations and excluded the model call, so it measured nothing about
enqueue cost. Corrected, the host spends ~24.4 ms inside the call, and removing
~3.6 ms of it (`act_quant` dispatch caching) moves the step by only 0.71%, which
is measured evidence that host dispatch is largely overlapped rather than proof
from a launch count. See
[Decode step budget](OPTIMIZE-RESULTS.md#decode-step-budget). Separately, the
`torch.cuda.synchronize()` in the harness and the `sampled.tolist()` in
`ReferenceBackend._collect` cost 2.8-3.1 ms/step (9%) and are removable — pipelined
and synchronized decode produce identical tokens once prefill is deterministic —
but the delivery path must enqueue step *i+1* before reading step *i*'s token. See
[Decode step budget](OPTIMIZE-RESULTS.md#decode-step-budget).

**First fusion pass (September 13).** Three of the chains named above are now
fused: RMSNorm, the hyper-connection pre/post mixers, and the MoE SwiGLU+clamp+
route-weight. **6,232 → 4,558 launches/step (−27%), 11.9% lower decode latency**
(32.599 → 28.716 ms/step at B=1, 2K, worst rank, three interleaved repeats;
~13.5% higher tok/s), with **bit-identical decode tokens**. Each change is guarded by a bit-exactness check
against its reference expression, not by the benchmark harness's parity gate: that
gate compares its two arms, so a kernel present in both arms passes it.

Two findings from that pass constrain the rest of the section:

- **Triton's `tl.exp` and `tl.sqrt` are not `expf`/`sqrtf`.** The first single-kernel
  SwiGLU differed from `torch.exp` on 55,722 of 82,944 inputs; `tl.math.rsqrt` and
  `libdevice.exp` are bit-identical. Any further fusion has to check this.
- **`torch.mean`'s reduction order is not reproducible in a kernel** (three
  reduction shapes tried, none agree). The fused RMSNorm therefore keeps torch's
  `mean` and fuses only around it, which is why it is two kernels and not one. A
  one-kernel variant is 2.97% faster and is off by default at ~5 differing bf16
  elements per million.

Still unfused and named by this section: the KV-cache writes, and `hc_mixes` at
decode (its fused kernel is shaped one program per 64 tokens, so it needs a
decode-shaped variant). The remaining small-kernel groups are 2,774
elementwise/copy/reduce plus 410 activation-quantization launches per step; the
`act_quant` group is now the largest single fusion target at 1.05 ms/step. See
[Decode kernel fusion](OPTIMIZE-RESULTS.md#decode-kernel-fusion).

**Second fusion pass (September 14).** The rotary embedding is now fused too. It
was the largest single unfused item in the step's op attribution -- 0.519 ms/step
of `aten::copy_` plus 0.160 ms of `aten::mul` -- and a decode step calls it **198
times**, each call three launches over a small tensor. `rope_kernels.py` does the
cast, the complex multiply and the cast back in one launch, in place: **2.3% lower
decode latency** (28.365 → 27.714 ms/step, two interleaved A/B runs, 3 and 4
repeats) with **bit-identical decode logits and tokens**. See
[Fused rotary embedding](OPTIMIZE-RESULTS.md#fused-rotary-embedding).

It adds a third item to the list of things a fusion in this engine has to check,
and this one is the sharpest:

- **A fused kernel is not exact because the arithmetic looks the same.** torch's
  complex multiply contracts its products with FMA, and the four ways to write
  `(a+bi)(c+di)` in fp32 differ by one ulp on roughly one input in 300k. Here that
  is not negligible: the uncontracted form is 2.66% faster and **produces different
  tokens at step 8**, because the model turns a one-ulp logit difference into a
different argmax. The synthetic exactness check could not see it -- at its
  original 32 draws all four forms read zero differences, since random inputs
  rarely land on a bf16 rounding tie. What settled it was comparing both paths on
  every call the real model makes (`check_rope_inmodel.py`), over 3,192 calls and
  2,133,504 elements: the contracted form differs on 0, the other three on 23
  each. A draw count that reports zero is not evidence until it is large enough to
  have found something.

**Third fusion pass (September 14).** Four more changes, and this pass started by
fixing the thing that was supposed to be finding them. The per-line attribution tool
(`profile_decode_stacks.py`) printed `?` in its site column for every row and looked
like it worked: it read `cpu_op.args['source']`, and current kineto leaves that field
empty. The Python stacks are in the trace as their own `python_function` events with
nothing linking a `cpu_op` to them, but every kernel carries a `correlation` that both
`cuda_runtime` and `cuda_driver` record **with a host timestamp** -- the moment the
launch was issued -- and containment of that timestamp in the frames attributes
**100%** of kernels, including the Triton and TileLang ones the op-name join cannot
reach at all. `analyze_decode_stacks.py` is that analysis split out, so a saved trace
is re-analysed in seconds. The corrected per-function table is in
[Decode kernel fusion](OPTIMIZE-RESULTS.md#attributing-launches-to-the-function-that-issued-them).

| Change | ms/step | latency | tokens |
| --- | ---: | ---: | --- |
| Cache the gate's fp32 routing weight | +0.456 | +1.65% | identical |
| Fuse the shared expert's SwiGLU tail | +0.450 | +1.67% | identical |
| Quantize the MoE's input row once | +0.244 | +0.92% | identical |
| Quantize the attention block's input once | +0.136 | +0.52% | identical |

That is **1.345 ms/step, 4.90% lower decode latency** over the pass at identical
tokens, measured as one change set -- all four flags against none, five interleaved
repeats, `results/ab-third-fusion-pass-combined.json`. **Do not quote the sum of the
four rows above** (1.286 ms/step): each was measured against the baseline current at
the time, and the combined run is the clean basis. The same run is also the check that
the flags do not interact: prefill logits bit-equal over 66,191,360 values and no
differing token in 64 steps with all four toggled together.

One more change landed after that measurement and is therefore **not** in the 1.345:
the `act_quant` buffer cache, re-measured at **+0.078 ms/step (+0.30%)** on the
post-pass baseline (`results/ab-act-quant-cache-on-by-default.json`, 9 interleaved
repeats, identical tokens). It had been carried as the last "assumption, not a
guarantee" item; the assumption was instrumented and the flag is now on by default.
The mechanism and the two things it corrected are below.

A second follow-on is the gate's pre-top-k chain, **+0.072 ms/step (+0.28%)**
(`results/ab-gate-prep-fused.json`, 9 interleaved repeats, identical tokens,
`_GATE_PREP_FUSED`). `Gate.forward` spent four launches per layer -- softplus, sqrt,
the bias add, and a division by `gate_temp` that is a numeric no-op at the shipped
value of 1.0 -- on a `[n, 384]` fp32 tensor, so it was nearly all launch overhead.
Making it bit-exact needed two primitives that are not the ones that look right, and
the residue was localized rather than guessed at; see
[Verification](OPTIMIZE-RESULTS.md#verification).

The pass is **400 fewer launches per step** on the same analysis basis (3,896 →
3,496, `results/decode-stacks-attribution-moe-eager.txt`). That count is a different
series from the 6,232 → 4,558 above, which came from `analyze_decode_trace.py` on
graphed traces; do not splice them. Every change is behind a default-on flag with a
bit-exactness check of its own, listed in
[Verification](OPTIMIZE-RESULTS.md#verification).

Three findings from this pass, in the same spirit as the two above:

- **A replayed graph has no Python stack, and that hides whole functions.**
  `MoE.forward` keeps its own per-layer decode graph, so all 1,400 of its launches
  attributed to `MoE.forward` -- and the kernel-name view put the routing chain at
  0.67 ms/step when it is **1.46**. Re-profiling with that graph bypassed
  (`DSV41F_PROFILE_MOE_EAGER=1`, now a harness option) put `Gate.forward` on top and
  found the weight cast that the first change removes. An attribution that lands
  everything on one frame is a signal to open the graph, not a result.
- **An activation read by N quantized weights is quantized N times.** Every
  `Linear` quantizes its own input, so a row feeding three weights is quantized three
  times. `probe_quant_dupes.py` counts it exactly -- **410 calls in one eager decode
  step over 282 distinct inputs, so 128 redundant** -- and two of the four changes
  above remove 120 of them. It holds a reference to every input, because the caching
  allocator otherwise hands the same address to different activations and reports
  duplicates that are not there.
- **A cache of a derived tensor needs its invalidation checked against how the
  tensor actually changes.** The gate weight is cast to fp32 per call; caching it is
  free numerically, but `load_state_dict` bumps the parameter's version counter while
  expert placement reassigns `.data`, which moves the data pointer and does **not**.
  A version-keyed cache would have served the pre-placement weight for the life of the
  process. It also has to be a tuple rather than a bare tensor, since
  `nn.Module.__setattr__` registers any Tensor it sees as a buffer, which would put a
  cache in the checkpoint contract and make a strict load fail.

**What this pass leaves, with its measured size.** Non-collective kernel time is
16.89 ms/step in 3,404 launches, and the removable part of it is now small and
awkward:

| Item | ms/step | Why it is still there |
| --- | ---: | --- |
| `Gate.forward` elementwise tail and top-k | ~0.9, of which **the pre-top-k half is taken** | the pre-top-k chain is now fused (+0.072 ms/step, `_GATE_PREP_FUSED`). What remains is the post-top-k half -- `gather`, `sum`, `+1e-20`, `div`, `*route_scale` -- and it needs the 6-element sum order and the top-k tie-breaking bit-for-bit. The **sum order was probed and does not reproduce**: six candidate orders for `[1, 6]` fp32 all mismatch torch on about half of 2,000 draws, which is the same wall the two `mean`s hit |
| Two `aten::mean` sites (RMSNorm, `hc_mixes`) | 0.63 | torch's reduction order is not reproducible; the one-kernel RMSNorm is 2.97% faster and differs on ~5 bf16 elements per million |
| `hc_split_sinkhorn`, 2 calls per layer | **0.31**, and the stated reason was wrong | the attribution table already carried it at 0.33, so the 0.16 here was too low. Measured marginal graphed cost is 3.83 us/launch x 80 = **0.306 ms/step**, and it is the kernel's own serial work, not launch overhead: 1.41 us at `sinkhorn_iters=1` against 3.83 at 20, and flat in rows (n=1, 8 and 64 all ~4 us). The reason given here -- "the two calls use different scale/base tensors, so merging them changes its signature" -- is also wrong: the two calls per layer are *dependent*, because the FFN's `hc_mixes` reads the x that the attention sublayer just produced, so they cannot be merged at all. The lever is the kernel itself: 40 block-wide reductions over a 4x4 matrix with 64 threads |
| The two `torch.cat` in `Attention.forward` | 0.14 | removing them means giving `sparse_attn` two KV sources instead of one |
| `RowParallelLinear`'s fp32 cast pair | 0.14 | the fp32 all-reduce is deliberate; the cast into it could only go if the GEMV stored the bf16 rounding in fp32 |
| The Indexer's `torch.sort` of 512 indices | 0.21 | 22.8 µs for 512 int32, a fixed-cost radix kernel; a Triton bitonic sort measures 15.1 µs, so the prize is 0.06 ms/step |
| `kernel._ACT_QUANT_CACHE_ENABLED` | **taken** | was 0.27 and left off for an aliasing assumption; the assumption was tested instead of restated, and the flag is now on by default at `model._ACT_QUANT_CACHE`. Re-measured at **+0.078 ms/step (+0.30%)**, down from the +0.202 first measured, because the shared-quant changes above removed a third of the calls whose scratch it saves |

**The act_quant buffer cache, and what it turned out to be.** This was the last item
carried as "an assumption, not a guarantee", so the assumption was instrumented
rather than argued. `check_act_quant_alias.py` hands each produced pair back as a
fresh *view* of the shared buffer, tagged with the epoch it was written in -- views
share storage but are distinct objects, so the tag travels with the object the
consumer receives and a later produce cannot overwrite it. Every consumer is then
wrapped (the fp8/fp4 GEMV and GEMM entry points, and the MoE's Triton `_w13`/`_w2`)
and compares its tag against the cache's current epoch. Over one prefill and 32 decode
steps: **1,871 produces across 13 keys, 4,344 consumer reads, 0 stale reads.** The
detector is shown to have power rather than assumed to: `--inject` forces one real
stale read and is reported, naming the hand-out site and the stale consumer.

Two things about that result were not what the docstring said:

- The hazard it warned about -- "the MoE's graph build captures on a side stream" --
  does not exist in the shipped configuration. `MoE.forward` returns early when
  `_SG_BUILDING` or `torch.cuda.is_current_stream_capturing()`, so inside the
  whole-step capture the MoE is inlined into the outer graph and there is no second
  stream. The side-stream capture is the `_STEP_GRAPHS=0` path.
- The stated benefit was wrong for the default path. The docstring justified the cache
  with "dispatch is 16.8 us/call and this is called ~410x per step", but under step
  graphs the Python does not run at replay at all -- the layer bodies are captured, and
  both `_sg_build` segments cover `e0..e1-1` and `e1..end`. What the cache actually
  buys is **scratch footprint**: one buffer per key instead of one per call leaves
  ~3 MB less live scratch inside the captured graph. That is why an A/B that rebuilds
  the graph per arm (as `bench_ab.py` does) measures a real effect, and why the number
  shrank to +0.078 ms/step once the shared-quant changes removed a third of the calls.
  Both the docstring and the kernel's default now say this.

The flag is on at `model._ACT_QUANT_CACHE` rather than in `kernel`, so `act_quant`
keeps its safe default for any caller outside the verified set. Correctness:
`check_decode_parity.py --flag kernel._ACT_QUANT_CACHE_ENABLED` is bit-exact over
prefill (0/66191360) and 64 decode steps, and `check_act_quant_cache.py` still reports
0 differing bytes cached vs uncached. `check_decode_parity.py` gained dotted-name
`--flag` support to run that, since the flag lives in `kernel` and the harness
previously only did `setattr(model, flag)`.

The larger remaining lever is not in this table. The Marlin comparison is 2.8x faster
than the engine's grouped GEMV at the real per-rank mix, which on `_w13` + `_w2`
(4.44 ms/step) is worth several ms; it is W4A16 against the engine's W4A8, so it needs
its own correctness gate rather than this pass's bit-exactness one.

The Gate's tail is the item that looks most tractable and is not, so it is worth
saying why in full. Its chain is 400 launches at 0.465 ms/step around a `topk` that
costs 0.437, and the post-top-k half (`gather`, `sum`, `+1e-20`, `div`, `*route_scale`)
is five launches of exact fp32 arithmetic with no transcendental in it. What blocks it
is the `sum`: six candidate association orders for a `[1, 6]` fp32 reduction all
disagree with `torch.sum` on about half of 2,000 random draws, so the order is not one
of the plausible trees. That is the third reduction this engine has hit the same wall
on, after the RMSNorm `mean` (three shapes tried) and the `hc_mixes` `mean`. The
pre-top-k half has no reduction in it and could be fused, but the three launches it
replaces measure 0.161 ms/step, so the prize does not cover a new kernel plus its
exactness check.

## 4. Remove cohort waste and sampling synchronization

These serving-path issues are absent from the B=1 teacher-forced kernel benchmark.

- **Finish length bucketing.** The first pass fixed asymmetric spread, but its
  early stop still misses compatible requests in an arrival-ordered queue (see
  review). The backend bulk-prefills the shortest prompt and consumes longer
  tails one token per step. Tune spread and bounded scanning against useful
  throughput and queue wait. Even an allowed 4K/6K pair incurs 2K single-token
  prompt steps.
- **Finish batched sampling and token feedback.** The first pass batches row
  sampling and device writes, but retains synchronization and unnecessary greedy
  work (see review). Preserve per-request parameters, RNG semantics and terminal
  handling, and measure the TP broadcast wrapper as part of the complete path.
- Measure finished/cancelled-row waste and output-length imbalance. Safe masking
  may reduce work, but dummy rows still cost compute unless kernels skip it.
  Mid-flight refill remains a separate position/state-aware admission milestone.

Gate on mixed-length/sampling tests, request isolation, quality, queue latency
and useful output throughput. Move these fixes earlier if serving traces show
that they dominate; do not infer their gain from model-only timings.

## 5. DSpark integration: high upside, higher implementation cost

The September 13 source feasibility analysis below identifies draft loading,
reusable vLLM logic and missing state operations. Run its first runtime milestone
alongside the baseline/kernel evaluation. Promote integration ahead of remaining
tuning if a correct verifier demonstrates favorable round cost.
This checkpoint has three DSpark stages drafting five tokens, **not classic MTP**.
Loading those weights alone does not implement speculation.

Implement draft sampling, acceptance/rejection, bonus-token handling and exact state
commit/rollback across SWA, compressed/index caches and Engram history. Verify that
multiple candidate positions execute correctly; ordinary batched decode is not
sufficient by assumption. Include rejection at every draft position and partial
acceptance around cache/compressor boundaries in state tests.

Measure draft/verification cost, accepted tokens per verification, memory and net
throughput on representative coding, reasoning and language workloads. Tune the
supported policy by cohort size, including disabling speculation where it loses.
The local forced-length fixture's high acceptance is not representative evidence;
the former ~1.35× estimate is not a ceiling. Require distributional correctness
and held-out quality as well as state tests and speed measurements.

**Priority rationale:** potentially the largest single-stream gain. Its position
reflects verifier/state integration effort, not low expected impact. The roughly
threefold vLLM single-stream improvement warrants explicit evaluation, not dismissal
because some kernel optimizations are easier to implement.

### DSpark feasibility assessment — September 13, 2026

**Verdict: feasible to prototype using the existing reference drafter; not a
drop-in vLLM verifier import.** Weight availability is not the blocker. The hard
work is a qualified multi-position target forward, exact accepted-prefix state,
and an efficient execution path for small verification chunks. This assessment
is source inspection, checkpoint-header inspection and an isolated CPU state
probe; it does not claim measured DSpark speed or full-model correctness.

Evidence is pinned to the clean local vLLM checkout
`7d81d62702b41885e2ff3ebc7ad9dfb638cc429c` under
`/root/glm-testing/vllm-ds41f-src`. The reference is a changing external working
copy: inspected `model.py` SHA256
`b504179375ecef12c6a0cfe688f91274ced88fe2404c3462fc501c60d6cb77c6` and
`benchmark_ds41f.py` SHA256
`f4b2d9e0c0b6ac25e812300ccb202efce2fc444b1741c3d9a6800b6298f206f5`.
Re-pin before runtime work. The official
[model recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
also describes the three-stage, five-token DSpark head and workload-dependent
acceptance; it does not establish performance for our custom engine.

#### What already exists and what can be reused

| Component | Evidence and integration consequence |
| --- | --- |
| Draft weights and architecture | Reference `config.json` enables three `mtp` stages, block size 5, noise token 128799, target layers 37/38/39, Markov rank 256, and 3-of-128 routed experts per stage. `convert.py` handles the separate draft expert count. Rank 0's converted safetensors header contains 670 `mtp.*` tensors occupying 2.030 GiB on disk, including projection, Markov and confidence heads. This is stored payload, not measured incremental HBM. |
| Shared embedding/head | `Transformer.__init__` already constructs the enabled drafter and aliases the target embedding/head into its stages. Enabling speculation should not load another copy of those weights. Audit missing/unexpected keys and runtime memory, including dtype promotion and workspaces. |
| Target features | Eager target forward gathers mean-pooled hyper-connection inputs of layers 37/38/39, before their attention; segmented decode has auxiliary-feature storage too. Preserve those locations and return features for every verified input, not just the last. |
| Reference draft computation | `DSparkAttention`, `DSparkBlock.forward_embed`, `forward_head`, and `Transformer.forward_spec` already implement parallel non-causal draft features plus sequential Markov-biased sampling. `forward_head` returns raw confidence scores; an adaptive policy needs the appropriate sigmoid/calibration, not a direct score-as-probability assumption. |
| vLLM draft design | `vllm/models/deepseek_v4_1/nvidia/dspark.py` and `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` supply the matching feature/projection, context-KV insertion, anchor/noise layout, Markov sampling and logits-cache contracts. Reuse those contracts and tests; the reference already has its own kernels/weights. |
| vLLM acceptance code | `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` and `rejection_sampler_utils.py` contain greedy, tokenwise probabilistic and block verification. The wrapper depends on `InputBatch`, request/sampling state, positions and vocabulary-chunk workspaces. Low-level math/kernels are candidates for a small adapter, not an import-and-call integration. Preserve license notices for copied code and differential-test the adapter. |
| vLLM adaptive policy | `adaptive_verification.py` uses confidence survival products and measured draft/verify cost curves to allocate verification budget. Confidence schedules work; it does not replace target acceptance. Keep this off for the first fixed-five correctness prototype. |

DSpark's expensive draft backbone runs over five query positions together, but
the lightweight Markov head samples left-to-right: each proposal distribution
depends on the preceding sampled token. Retain the **actual conditional draft
distribution** used at each position for stochastic rejection. Confidence values
and pre-Markov logits are not substitutes for it.

#### The required verifier and state contract

Use an explicit consumed-token cursor, separate from emitted output length.
Suppose target state contains tokens `[0, P)`, and the already sampled anchor
`x_P` is pending consumption. The drafter uses committed target features/context
and this anchor to propose `d1..d5`. A target verification call consumes
`[x_P, d1, d2, d3, d4, d5]` at positions `P..P+5` and returns **six** logits rows:
five score the drafts and the sixth supplies the all-accepted bonus distribution.

If `a` drafts are accepted (0 through 5), commit target inputs
`[x_P, d1..da]`: the next consumed cursor is `P + a + 1`. Emit those `a` draft
tokens plus one replacement/bonus token; that final sampled token is the next
pending anchor and is not yet in target KV. Do not emit `x_P` twice. At EOS or the
output/context limit, truncate before publishing state and events to the actual
terminal boundary. Test the first round after prefill separately from steady
rounds; reference draft positions are offset from its latest target feature.

The current target's chunked continuation is a useful starting point, but it is
not a qualified verifier. The recently added `full_logits=True` exposes the head
at each position and bypasses step graphs. Qualify causal masks, compressed-key
visibility, index candidates, routing and per-position features against sequential
target execution. Avoid the ordinary forward's unused sampling/RNG side effects
when exposing verifier logits. A six-token eager forward need not be anywhere
near the cost of the optimized one-token graph.

| Mutable state | Why cursor rewind is insufficient / proposed handling |
| --- | --- |
| Main-model SWA rings (`window_kv_cache`) | Verification overwrites up to six modulo-128 slots per layer, possibly destroying live prefix rows. Journal overwritten slots or verify into staging storage; commit only the accepted input prefix. A ring snapshot from a later position cannot simply be cropped. |
| Compressor `kv_state` / `score_state` | Ratio-2 compression retains only the final partial group, losing intermediate accepted boundaries. Save per-input projected KV/score rows or replay the accepted prefix from a saved state. vLLM's `CompressorStateCache` uses a speculation-sized ring (at least 8 rows for five drafts), which is an applicable design reference. |
| `compress_kv_cache`, indexer `k_cache` | Preserve the previously live cache and publish only groups complete at the committed cursor. Stale speculative rows must be restored or made unreachable by exact length/mask logic. Test ratio-1 and ratio-2 source layers and every shared-cache consumer. |
| Engram token history | Verification writes candidate IDs into `NgramHashState.cache`; subsequent hashes must read only committed history plus the new pending input. Restore/invalidate the rejected suffix and qualify 2/3/4-gram history. vLLM reconstructs lookback from committed request tokens in `nvidia/model_state.py`; our cache needs an explicit equivalent invariant. |
| Draft context rings and target auxiliary features | Seed from prefill and append all newly committed target features after each round; rejected features must never enter draft context. `forward_spec` couples context ingestion to drafting, so expose a separate context-only update. Calling it on a multi-token continuation is not a clean substitute: attention seeds context while the enclosing block still executes other work. |
| Runtime pointers, position and RNG | Preserve graph buffer addresses; republish `shared_attn.pos` and source-cache pointers at the committed cursor. Use request-scoped, explicitly keyed sampling streams. All TP ranks must agree on proposals, acceptance length, commit and termination. |

An isolated CPU execution of the actual `Compressor.forward` method confirmed
the partial-state problem using identity projections and uniform pooling. With
an existing KV value 10 at position 0, verifying values 20..70 over six inputs
starting at position 1 leaves pending KV **70**. Executing only the first two
inputs, the accepted-prefix oracle, leaves pending KV **30**. Moving the cursor
back to 3 cannot recover that overwritten value. This is a state-bookkeeping
demonstration, not a numerical test of the trained model.

Existing prefix snapshot helpers in `serve/server.py` enumerate the relevant
cache buffer names (including draft rings through module traversal), but copy
entire rows to CPU. Use that inventory as a checklist, not that transport in
every speculative round. Initially, a GPU snapshot/restore plus accepted-prefix
replay can serve as the slow correctness oracle. The fast path needs bounded
GPU journals/staging and saved compressor intermediates so rejection does not
require replaying several full-model steps. Check all acceptance boundaries,
not only restoration to the start of a round.

#### Implementation options and recommendation

- **Keep vLLM serving DSpark:** already exercised locally and the shortest route
  to using its existing scheduler/cache/verifier combination. This supplies a
  comparison and oracle, but does not implement DSpark in this engine.
- **Integrate a B=1 reference prototype (recommended investigation):** retain
  our drafter and target, implement the contract above, and adapt acceptance
  logic behind a small tensor interface. Start text-only, greedy, fixed five,
  one exclusive request; use AR fallback. This isolates model-state correctness
  from concurrent admission and probabilistic sampling complexity.
- **Transplant the whole vLLM speculator/worker stack:** not a bounded shortcut.
  Its DFlash-derived proposer depends on paged cache groups, slot mappings,
  forward contexts, rejected-token metadata and graph management. A broad port
  would also replace much of the serving/state architecture. Prefer vLLM itself
  if reusing that entire stack is the objective.

The existing `StepResult.tokens` tuple can represent a burst, but the current
engine checks `max_new_tokens` after emitting the whole tuple. A speculative
backend must cap the burst and fix per-token terminal handling before integration;
the scalar backend cursor and prefix snapshots also need the consumed-boundary
contract. `BroadcastModel` currently exposes ordinary forward, not a round with
draft/verify/commit. Define that TP command explicitly. B>1 introduces differing
accepted cursors across rows and therefore requires per-row positions/masks or
an explicitly qualified fallback; it is a separate milestone.

#### Staged acceptance criteria and the performance decision

1. **Loading and drafting:** verify all draft keys/shapes, shared parameter
   identity, feature alignment and draft distributions. Test prefill context
   seeding, the first round, subsequent rounds, and the 128-token ring wrap.
   Measure draft-only latency, context insertion, resident/peak HBM and auxiliary
   feature storage. The drafter is already constructed in the enabled reference
   configuration; do not assume vLLM's AR-to-DSpark memory delta applies here.
2. **Target verifier and commit oracle:** compare six-position target logits,
   routes and state against independent one-token execution, at both compression
   parities and around positions 127/128/129. Force rejection at each draft
   position, all acceptance, EOS, output/context exhaustion and cancellation.
   Verify several subsequent AR tokens from every committed boundary; include
   long context, compressed/index visibility and Engram history. Record numerical
   tolerance separately from exact structural state/ownership checks.
3. **Fast B=1 greedy round:** replace full snapshots/replay with bounded state
   operations, then capture fixed-shape draft/verification work where safe.
   Time all ranks, including draft context update, six-row head, acceptance,
   commit and host/TP delivery. Require end-to-end improvement on held-out coding,
   reasoning and language traffic, and retain AR fallback when it loses.
4. **Stochastic correctness, then block/adaptive policy:** begin with standard
   tokenwise rejection (accept with `min(1, p(d)/q(d))`, resample from normalized
   positive `p-q`, and draw a target bonus when all drafts pass). Test small-vocab
   distributions, zero support, temperature/top-p and deterministic RNG handling.
   Then port/differential-test vLLM's distinct block verifier; it uses cumulative
   ratios and a different residual, so renaming tokenwise rejection is not enough.
   Finally calibrate confidence/cost curves for adaptive verification. Tests must
   establish the intended target distribution, not identical stochastic token
   sequences under an unrelated AR RNG schedule.
5. **Serving expansion:** qualify prefix restore with draft context, streaming
   burst timing, multi-request terminal handling and row-specific accepted
   lengths before claiming B=2/4/8 support. Re-profile acceptance and economics
   by concurrency; continuous admission remains separate.

For steady fixed-five rounds, let `A` be accepted drafts and `T_round` include
all draft, verification, context-update, acceptance, commit and delivery work.
The throughput estimate is `(1 + E[A]) / E[T_round]`; compare against a matched
AR step cost. At the reviewed model-only 32.46 ms anchor, mean acceptance 3
requires total round time below **129.84 ms** merely to win; accepting all five
allows **194.76 ms**. A 3x improvement with all five accepted would require at
most **64.92 ms/round**. These are arithmetic budgets, not predictions. Measure
short verification shapes explicitly; large-prefill throughput cannot predict
their latency. Any accepted-prefix replay belongs in the denominator.

**Decision:** proceed with milestones 1–3 as an early, bounded runtime experiment.
The source spike removes the presumed need to build a drafter from scratch, but
does not show a cheaply reusable verifier or justify promising vLLM-like speed.
Do not require DSpark to close the AR kernel/transport gap, and do not let that
gap indefinitely postpone measuring a correct speculative round.

## 6. Optimize Engram lookup and host placement

The reference CPU-gather fallback sends indices to the host, gathers/dequantizes
there and copies values back. The reviewed segmented decode wrapper executes
this path outside its graphs when Engram is offloaded. vLLM uses GPU lookup
through UVA views of pinned tables. Measure the reference path's exposed cost.

Compare current execution with GPU-driven lookup, appropriate NUMA placement and
GPU-resident tables where target context/concurrency leaves headroom. Explore
prefetch/overlap only where dependencies permit. Measure exposed step cost at B=1
and concurrent prefill/decode, plus pinned RAM and VRAM. The old ~2% offload penalty
at 270 ms/token is not a current estimate. Move this earlier if profiling exposes
substantial offload stalls.

**Status (September 14).** The exposed cost is measured, and it is 0.441 ms/step
(1.63%) for the two lookups -- not substantial enough to move this section earlier,
which answers the question this section poses. Splitting the call into its pieces
(`probe_engram_steps.py`) shows it is 33 us of D2H, 70 us of CPU gather and 62 us of
H2D over a 12 KiB payload: transfer latency and a scattered read of a 23 GiB pinned
table, not the dispatch count of the chain. The two fp8 decodes are now 256-entry
numpy tables, which is exact and worth 0.35-0.59% at identical tokens. The rest is
not taken: both lookups precede all dependent GPU work and the D2H is synchronous,
so reordering only moves the exposure; hiding it needs the gather on a worker thread
or on the GPU, and the GPU-resident tables are the reason the offload path exists
(45.8 GiB per rank against ~50 GiB of headroom). See
[Engram lookup cost](OPTIMIZE-RESULTS.md#engram-lookup-cost).

## 7. Improve remaining MoE dispatch/layout bottlenecks

Only after the production-kernel comparison, optimize remaining expensive work.
Our B=1 grouped path already avoids host counts and sorting. The first pass
replaced the multi-row `bincount` with fixed-range scatter accumulation and added
flat prefill tiles plus calibrated placement. Profile remaining prefix sums,
sorting, allocation and rank skew rather than attributing every cost to sorting.

Compare fixed-workspace device histogram/prefix/scatter or production dispatch.
Prefer fewer launches where beneficial, but do not require one launch if global
synchronization or occupancy makes it slower.

If weight access remains limiting, prototype a tiled FP4 layout such as
`[N/16][K/128][16 rows][64 B]` with compatible scales. Choose layout and launch
geometry from actual access patterns; do not assume the same layout suits FP4
experts, FP8 dense weights, GEMV and prefill GEMM. Measure bandwidth, conversion
cost, numerical behavior and memory before adopting it.

## 8. Workload-aware cohorts and continuous admission

After correcting length bucketing, test task/domain grouping only if routing
traces show meaningful expert overlap. Expert traffic depends on distinct experts,
tiling and reuse; task labels do not guarantee shared routing or speed. Measure
expert occupancy, rank balance, queue fairness and useful throughput. Batch
composition can change numerical results, so retain quality gates.

If cohort tail waste or queued arrivals dominate, prioritize continuous admission
and correct chunked prefill instead. This requires per-row positions and state
isolation throughout the model. Qualify prefix-hit profitability under concurrency:
an exclusive prefix hit may save prefill while reducing aggregate throughput.

## 9. EP or pipeline topology experiments

Revisit topology only if measurements justify it after kernel, collective and
scheduling improvements. Audit current expert placement first: the reference
already assigns local experts to ranks and combines outputs. Specify what a new
EP design changes in token dispatch, dense TP and combination.

Compare communication volume, load balance and useful compute with TP4. A layer
pipeline serializes a single request across GPUs and needs enough independent
work to fill stages; fewer reductions do not establish better B=1 latency.
Evaluate existing transports before a custom P2P protocol. Gate topology changes
on full parity, memory, all-rank timing and serving behavior.

## Conditional and out-of-scope work

- Warp count, tile size, FP4 unpacking and transfer overlap are tuning variables,
  not universally rejected ideas based on an A100 result. Revisit when the H20
  profile or production-kernel comparison identifies a reason.
- CPU expert offload/hot-expert caching and NVMe Engram storage are not priorities
  for this four-GPU, 2 TB host: current experts fit VRAM and Engram fits host RAM.
  Reassess if deployment memory budgets change.
- Native Blackwell FP4 tensor-core paths and topologies requiring more than four
  GPUs are outside this hardware target.

## Comparison references

- [vLLM model recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
- [vLLM MXFP4 backends](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/mxfp4/)
- [vLLM custom collectives](https://docs.vllm.ai/en/latest/api/vllm/distributed/device_communicators/custom_all_reduce/)
- [vLLM sampling implementation](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py)
- Local evaluation: `/root/glm-testing/ds41f/VLLM.md` and linked run artifacts;
  inspected vLLM checkout revision `7d81d62702b41885e2ff3ebc7ad9dfb638cc429c`.
  Local artifacts are not distributed with this repository. The numbers above
  are historical observations, not reproducible benchmarks from this repo.

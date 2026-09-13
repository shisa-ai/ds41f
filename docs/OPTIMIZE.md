# Optimization punchlist

Priority is expected end-to-end impact relative to effort and confidence: compare
proven kernels and transports before writing replacements. The order is provisional
until a fresh profile identifies current bottlenecks. There are no promised gains.

Target: **4 × H20-3e, TP=4**. H20 is Hopper: native FP8 tensor cores, **no native
FP4/MXFP4/NVFP4 tensor cores**. FP4 weights need a supported conversion/compute
path. Dense FP8 GEMV does not establish the best path for FP4 experts. Ampere and
Blackwell results do not transfer directly.

Status (September 13, 2026): the first custom-engine tuning pass is complete,
but this punchlist is **not fully implemented or qualified**. The review below
records implementation defects, evidence limits and the next work order. Keep the
dedicated [vLLM implementation](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
as the production comparison. Section 5 now includes our source-level DSpark
feasibility analysis; runtime integration and its performance gate remain open.

The [README](../README.md#performance) reports the trusted, passing run
(`trusted-shipped.json`). The newer full-prompt diagnostic run in
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

Next work order (priority within this document, not a promise of gains):

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

Use `trusted-shipped.json`'s 32.46 ms/token configuration as the reviewed
model-only anchor; preserve the older 44.65 ms/token historical result separately.
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

# Optimization punchlist

Priority is expected end-to-end impact relative to effort and confidence: compare
proven kernels and transports before writing replacements. The order is provisional
until a fresh profile identifies current bottlenecks. There are no promised gains.

Target: **4 × H20-3e, TP=4**. H20 is Hopper: native FP8 tensor cores, **no native
FP4/MXFP4/NVFP4 tensor cores**. FP4 weights need a supported conversion/compute
path. Dense FP8 GEMV does not establish the best path for FP4 experts. Ampere and
Blackwell results do not transfer directly.

Status: custom-engine optimization is paused while the dedicated
[vLLM DeepSeek-V4.1-Flash implementation](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
is evaluated. This is the implementation order if engine work resumes. Isolated
profiling and comparisons can inform that decision before feature qualification.

**What was implemented and measured from this list, with the parity and noise
measurements behind it, is in [OPTIMIZE-RESULTS.md](OPTIMIZE-RESULTS.md).**

**DSpark is a high-impact integration project, not a low-priority tuning idea.**
Our vLLM experience was roughly 110 → 350 tok/s for a single stream. The local
report distinguishes overall output from decode-only rates; match those metrics
before quoting a multiplier. Evaluate DSpark early in vLLM, then schedule its
custom-engine integration after bounded kernel/transport wins unless a feasibility
spike shows that a proven verifier can be reused cheaply.

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

Use the current 44.65 ms/token configuration as the reference-engine anchor.
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

First A/B the existing fused hyper-connection path, disabled in the measured
configuration. Availability does not make it numerically low-risk: verify Sinkhorn
and residual math before enabling it by default.

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

- **Correct length bucketing first.** `scheduler.py` checks only an upper length
  bound relative to the queue head: `[8192, 512]` prompts can share a cohort while
  the reverse order is split. The backend bulk-prefills the shortest prompt and
  consumes longer tails one token per step. Enforce symmetric length spread and
  tune it against useful throughput and bounded queue wait. Even an allowed
  4K/6K pair incurs 2K single-token prompt steps.
- **Batch sampling and token feedback.** `ReferenceBackend` samples per row,
  calls `.item()`, then writes each Python token back to CUDA. Top-p sorts the
  vocabulary and computes the same softmax twice. Compare optimized batched
  sampling, retain next-step IDs on-device, and consolidate host delivery.
  Preserve per-request parameters, RNG semantics and terminal handling.
- Measure finished/cancelled-row waste and output-length imbalance. Safe masking
  may reduce work, but dummy rows still cost compute unless kernels skip it.
  Mid-flight refill remains a separate position/state-aware admission milestone.

Gate on mixed-length/sampling tests, request isolation, quality, queue latency
and useful output throughput. Move these fixes earlier if serving traces show
that they dominate; do not infer their gain from model-only timings.

## 5. DSpark integration: high upside, higher implementation cost

Run an early feasibility spike alongside the baseline evaluation: identify draft
weight loading, reusable vLLM verification logic and missing model-state operations.
Promote integration ahead of the remaining tuning work if that reduces its cost.
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

## 6. Optimize Engram lookup and host placement

The reference CPU-gather fallback sends indices to the host, gathers/dequantizes
there and copies values back. vLLM uses GPU lookup through UVA views of pinned
tables. First identify which path the optimized graph configuration executes.

Compare current execution with GPU-driven lookup, appropriate NUMA placement and
GPU-resident tables where target context/concurrency leaves headroom. Explore
prefetch/overlap only where dependencies permit. Measure exposed step cost at B=1
and concurrent prefill/decode, plus pinned RAM and VRAM. The old ~2% offload penalty
at 270 ms/token is not a current estimate. Move this earlier if profiling exposes
substantial offload stalls.

## 7. Improve remaining MoE dispatch/layout bottlenecks

Only after the production-kernel comparison, optimize remaining expensive work.
Our B=1 grouped path already avoids host counts and sorting. The multi-row path
uses `bincount`, prefix sums and `argsort`; profile synchronization and allocation
rather than attributing every cost to sorting.

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

# Optimization results

A tuning pass over [OPTIMIZE.md](OPTIMIZE.md): what was implemented, what it
measured, and what is still open. It is **not** a complete implementation of that
list — see [Scope and status](#scope-and-status) before reading the numbers as a
progress figure.

The tuning-pass comparison comes from `benchmark_ds41f.py` on GPU0-3 (4 x H20-3e, TP=4) with
`DSV41F_ENGRAM_OFFLOAD=1`. That harness alternates the two arms, restores a
buffer snapshot before each arm, repeats, and reports the worst rank. The tables
below use its optimized arm. Prompt tokens are random ids.

These are single-host measurements, not a controlled speedup claim. They are
comparable to each other because the checkpoint, GPU set and harness are fixed;
they are not comparable to the vLLM figures in OPTIMIZE.md, which used a
different protocol.

## Scope and status

Done and measured: the decode kernel work (FP8 GEMV, warp counts), the prefill
MoE work (sync-free histogram, flat tile grid, load-balanced expert placement),
the fused hyper-connection kernels, and two serving-path fixes (symmetric length
bucketing, batched on-device sampling).

Remaining work and follow-up status:

| Item | State |
| --- | --- |
| Marlin / FP4-expert kernel comparison | Not attempted. The format qualification was part of the experiment, not a reason to call it complete. |
| Matched vLLM autoregressive baseline | Missing. The historical vLLM and current engine numbers use different protocols, so the gap is unquantified. |
| Graph coverage beyond B=1 | The segmented step-graph path requires input shape exactly `(1, 1)`. B=2/4/8 serving gains are unqualified. |
| Engram lookup cost | Measured at 0.45 ms/step, 1.4% of decode. The optimisation (moving the lookup into the graph or keeping a hot subset resident) is not attempted. See [Engram lookup cost](#engram-lookup-cost). |
| DSpark | Source-level feasibility analysis is complete in [OPTIMIZE.md](OPTIMIZE.md#dspark-feasibility-assessment--september-13-2026). A working verifier and runtime performance measurements remain open. |
| Custom / symmetric-memory collectives | The original pass measured NCCL only. Follow-up experiments are in progress; an end-to-end gain is not established. See [Collectives](#collectives). |
| Deterministic prefill | Not attempted. See [Noise floor](#noise-floor). |

## Latest saved model measurements

The newer `results/fulllogits-hcmixes-off.json` contains the following averages
of two optimized runs. It is marked **failed** on correctness checks; these
numbers describe execution time, not a validated release. The full-prompt score
comparison runs separately from the timed prompt-processing pass.

| Prompt/context length | Prompt processing | Time to process prompt | Decode throughput | Decode latency |
| --- | ---: | ---: | ---: | ---: |
| 2,048 tokens | 2,560 tok/s | 0.80 s | 30.62 tok/s | 32.66 ms/token |
| 8,192 tokens | 3,270 tok/s | 2.51 s | 29.29 tok/s | 34.20 ms/token |

These are model-only measurements on random-token prompts with predetermined
continuation tokens. Fresh end-to-end serving throughput and peak memory are not
recorded. The historical tuning comparison below uses an earlier run and must
not be described as the latest measurement.

## Tuning-pass comparison (earlier measurements)

| Workload | Before | After | Change |
| --- | --- | --- | --- |
| Decode, 2048 context | 44.81 ms/token | 32.46 ms/token | −27.6% |
| Decode throughput | 22.32 tok/s | 30.81 tok/s | +38.1% |
| Prefill 512 | 1071 tok/s | 1330 tok/s | +24.1% |
| Prefill 2048 | 1684 tok/s | 2580 tok/s | +53.2% |
| Prefill 8192 | 1948 tok/s | 3392 tok/s | +74.1% |

Before is `results/baseline-gpu0123.json`, the same harness on the same GPU set
before any of this work. After is `results/trusted-shipped.json`; both are the
mean of two repeats, and the two repeats are within 1.5% of each other.
The before decode figures have been corrected to the two-repeat 2048-token
average in that file. The previously quoted 44.65 ms/token and 22.40 tok/s were
an earlier comparison point, not that average. Percentages use unrounded values.

The decode row is the well-supported number: it is measured on a path whose two
arms are compared from the same restored state, so the comparison isolates the
decode kernels. The prefill rows are less well supported — see
[Noise floor](#noise-floor) and [What the harness does and does not
show](#what-the-harness-does-and-does-not-show).

In `trusted-shipped.json`, every recorded parity criterion passes on every row: `top1_agreement` 1.0,
`prefill_top1_agreement` 1.0, `max_logit_diff` 0.0, and
`prefill_max_logit_diff` 0.65-1.12 against its 1.25 gate.

Those prefill numbers are last-position only, and that is a weak signal — see
[Noise floor](#noise-floor). Measured over the whole prompt, the optimized and
naive arms differ by up to 16.3 with 0.76 top-1 agreement. Two runs of the *same*
optimized configuration differ by up to 12.3 with 0.76 agreement, while the naive
arm is bit-deterministic at 0.0000. That variation does not establish correctness
or rule out defects. Independent state checks and realistic quality tests remain
necessary. The saved `fulllogits-hcmixes-off.json` report is marked `failed`;
its measurements must not be presented as a passing full-prompt comparison.

## Decode changes

### M=1 dense linears use the FP8 GEMV kernel

`linear()` routed M=1 calls through the general FP8 GEMM above an 8 MiB
threshold, so every dense decode projection paid a GEMM built for large M. All
M=1 calls now use `fp8_gemv` (`DSV41F_FP8_GEMV_MIN`, default 0), guarded to
output widths divisible by 32.

Decode 44.65 → 40.9 ms/token.

### FP8 GEMV tiling

`kernel_gemv.py` used `BLOCK_N=32, num_warps=4`. A graph-captured microbenchmark
of the seven real M=1 shapes (in `bench_gemv.py`) put the total at 145.7 µs; the
same shapes at `BLOCK_N=8, num_warps=2` take 67.2 µs. Both are now defaults,
overridable with `DSV41F_GEMV_BLOCK_N` and `DSV41F_GEMV_WARPS`.

Decode 40.9 → 36.3 ms/token.

### Expert kernel warp counts

A single `num_warps` was shared by the grouped expert kernels. Profiling showed
`_w13` prefers 8 warps (4.12 → 2.18 ms/step) while `_w2` prefers 4 (1.64 → 3.41
ms/step when forced to 8). They are now separate
(`DSV41F_W13_WARPS`, `DSV41F_W2_WARPS`, `DSV41F_MOE_WARPS` as fallback).

Decode 36.3 → 32.7 ms/token.

## Prefill changes

### Synchronization-free MoE histogram

`GroupedMoE.routed_batch` called `torch.bincount` on CUDA, which calls `.item()`
internally to size its output and drains the pipeline once per layer. The valid
range is known, so a fixed-range `scatter_add_` replaces it. `DSV41F_BINCOUNT_SYNC=1`
restores the old path for A/B measurement.

### Fused hyper-connection kernels

`hc_pre` and `hc_post` materialized fp32 intermediates (`hc_post` builds a
`[b, s, hc, hc, d]` tensor, 2.7 GB at 8K). The fused Triton kernels in
`hc_kernels.py` compute the same expressions with bf16 loads, fp32 accumulation
and one bf16 store.

They were already implemented but disabled, because enabling them flipped the 8K
prefill top-1. The cause was an association difference: the fused `hc_post`
accumulated `post * x` first, while the reference sums the `comb`/`residual`
products first and adds `post * x` last. Adding it last makes both kernels
bit-identical to the reference at bf16 output on every shape tested
(`check_hc_exact.py`), so `hc_fused` is now the default.

Prefill 8192: 1982 → 2405 tok/s. Prefill 2048: 1718 → 2014 tok/s.

### Flat tile grid for the grouped prefill GEMMs

`_w13_m` and `_w2_m` launched a grid of `(expert, N-block)` and made each program
loop over its whole expert's token range. The largest expert set the tail, so
ranks reached the MoE `all_reduce` 10-16 ms apart. `_build_tiles` builds a
`(tile → expert, sorted-row start)` table on device with `cumsum` and
`searchsorted` (no host sync) and the grid becomes `(num_m_tiles, N/BN)` with one
equal `BM x BN` tile per program.

Prefill 8192: 2405 → 2967 tok/s. Prefill 2048: 2014 → 2363 tok/s. Kernel time
3.271 → 2.621 s, of which the MoE `all_reduce` wait fell from 0.751 s to 0.491 s.

### Load-balanced expert placement

Contiguous expert ownership (rank `r` owns experts `[96r, 96r+96)`) leaves the
busiest rank at 1.48x the mean routed-pair load (worst layer 2.17x), because the
router's expert popularity is skewed. `expert_placement.py` permutes the expert
axis of the gate and the expert weights together, per layer. The permutation is
mathematically a no-op — the same experts with the same weights, relabelled — so
only which rank computes which expert changes.

`plan_experts.py` calibrates per-layer per-expert routing counts from the gate
(the gate is replicated, so rank 0 sees the whole picture) and assigns experts
with length-capped longest-processing-time first. `apply()` permutes the gates
and redistributes experts with one `all_to_all` per layer. Per-rank routed-pair
spread falls from 2.99x to 1.05x; the busiest rank reaches 1.000x of ideal.

Prefill 8192: 2967 → 3400 tok/s. Prefill 2048: 2363 → 2589 tok/s. The MoE
`all_reduce` wait fell from 0.491 s to 0.100 s.

`maybe_apply` uses `DSV41F_EXPERT_PLACEMENT` when set, `expert_placement.pt` next
to the module when present, and `DSV41F_EXPERT_PLACEMENT=none` to disable.
Startup grows by 10-15 s for the redistribution.

It was initially applied only by the benchmark and the profilers, so `generate.py`
and the engine kept contiguous ownership and none of the 15% reached a served
request. `generate.py` now applies it after loading the model, and the engine
carries the module as `ds41f.backend.expert_placement` with an opt-in
`ReferenceBackend(apply_placement=True)`. It is a collective (one `all_to_all` per
layer), so every rank must call it together; the engine's single-controller
topology is why the backend's flag is opt-in rather than default.

### Fused `hc_mixes` (implemented, off by default)

`hc_mixes` computed `rsqrt(mean(x^2)+eps) * (x @ hc_fn)` in five steps over a
`[1, s, 20480]` fp32 tensor. At 8K those are 62.7 ms for the `x.float()` upcast,
25.0 ms for `square()`, 16.3 ms for `mean()`, 50.3 ms for the fp32 GEMM and a
scale — 154 ms, or 6.8% of the prefill. `hc_mixes_fused` does it in one pass over
the bf16 stream, with the normalization folded into the same loop.

The dot is the only part that costs precision. Measured against the fp32
reference on the 8192 x 20480 x 24 shape:

| Dot | Max abs error on mixes | Per 8K call |
| --- | --- | --- |
| bf16, weight split into hi + lo | 5.3e-3 | 0.25 ms |
| tf32, weight split into hi + lo | 3.0e-3 | 0.94 ms |
| tf32x3, fp32 weight | 4.4e-5 | 0.96 ms |
| reference: upcast, `pow`, `mean`, fp32 GEMM | — | 1.89 ms |

`input_precision="tf32x3"` splits both operands into a high and a low tf32 half
and keeps the three cross products, so the mantissa is effectively 21 bits. Its
error is a few 1e-5 across lengths, against 5.3e-3 for the bf16 split, and it is
close to the reference's own fp32 accumulation-order noise. It is the default
mode.

Splitting the weight into bf16 halves and using bf16 tensor cores is 3.9x faster
again but its 5.3e-3 error is in the same decade as the bf16 rounding of the
coefficients it produces. `DSV41F_HC_MIXES_MODE=split` selects it. Splitting into
*tf32* halves is not a middle ground: Triton lowers `input_precision="tf32"` such
that the residual is only partly recovered, giving 3.0e-3.

The kernel launches one program per 64 tokens, so it is latency-bound below about
4096 tokens — 0.75 ms flat from 512 to 4096, where the reference path is still
0.15-0.92 ms. The caller gates on `b * s >= 4096`.

It is off by default. `DSV41F_HC_MIXES_FUSED=1` enables it and takes 8K prefill
from 3400 to 3500 tok/s (+3%), with 512 and 2048 unchanged by the gate.

It is off because at 8192 it reported `prefill_top1_agreement` 0.0 where all three
runs without it reported 1.0. That criterion turned out to be a mean over a single
prompt position (see [Noise floor](#noise-floor)), so it was one 0/1 sample, and
the measurement that supported keeping it off was weaker than it looked. What the
full-prompt measurement shows is that the fused path's deviation is inside the
same-configuration noise floor at both lengths, which is the strongest evidence
available here. This does not establish correctness. Keep it disabled until
independent state and quality checks pass, as required by OPTIMIZE.md.

Note that this kernel sits in **both** harness arms, so the harness's parity gate
cannot see it either way — the same is true of every change in this document
except the two flags the arms toggle.

## Serving-path changes

**Expert placement was not wired into serving.** `expert_placement.maybe_apply`
was called by `benchmark_ds41f.py` and the profilers but not by
`serve/server.py:load_model_rank`, so a calibrated deployment served with
contiguous expert ownership and got none of the placement's ~15% on 8K prefill.
`generate.py` had been wired, but that is not the serving path. The loader now
calls it on every rank, after the weights load and before the first forward.

The engine module's default lookup is a file beside the installed package, which
does not exist for a model-repository calibration, so `maybe_apply` now takes an
explicit `path` and the loader resolves it: `DSV41F_EXPERT_PLACEMENT` if set, then
`<ckpt>/expert_placement.pt`, then the model repository's
`inference/expert_placement.pt`. It logs which file it used, so a missing
calibration is visible instead of silent. `apply` also now rejects a calibration
that is not a per-layer permutation of the model's expert count, which is what a
file from a different checkpoint looks like.

Verified on the real checkpoint: `torchrun --nproc-per-node 4` through
`serve/server.py`'s loader reports
`[placement] applied=True file=.../inference/expert_placement.pt`, with a
non-identity layer-0 permutation, on a run that exits cleanly.

**Length bucketing.** `scheduler.py` enforced only an upper length bound relative
to the queue head, so `[8192, 512]` could share a cohort while the reverse order
was split. `_admit` now enforces a symmetric spread
(`max_len <= (1 + tolerance) * min_len`) and skips an out-of-tolerance candidate
instead of stopping the scan. It used to stop when a candidate was *longer* than
the cohort's maximum, on the assumption that the queue was length-sorted; it is in
arrival order, so `[1000, 8192, 1050]` admitted only `[1000]` and left the
compatible 1050 waiting. Only an exclusive row stops the scan now.

**Batched sampling.** `ReferenceBackend` sampled per row, called `.item()`, and
wrote each token back to CUDA. Sampling is now batched over the emitting cohort:
one device gather, one top-p pass, one Gumbel-max draw, one indexed device write,
and a single host transfer. Per-row temperature and top-p are preserved.

The single-host-transfer claim was wrong as first written: `_sample_batch` still
had `if bool((tops < 1.0).any())` and `if bool(greedy.any())`, each of which drains
the pipeline, so a decode step did three host round-trips. Removing them fixed the
round-trips but not the deeper problem: the batched path drew a full `rand_like`
for every row even when the whole cohort was greedy, so a greedy workload advanced
the global generator and shifted the stream seen by later stochastic requests. The
per-row path drew nothing for a greedy row, so this was a regression in RNG
semantics, and the original requirement to preserve it was not met.

Temperatures and top_p reach `_sample_batch` as Python floats, so the branch is
host-side and costs no synchronization. A greedy-only cohort now returns argmax
without touching the RNG or computing softmax, the top-p sort is skipped when no
row asks for it, and the stochastic rows draw one `rand_like` each in row order,
so the generator advances exactly as it did before. Three tests pin this: a
greedy-only cohort leaves `torch.get_rng_state()` unchanged, a mixed cohort
advances it by exactly `stochastic_rows * vocab`, and `top_p=1.0` samples
identically to `top_p=0.999999`.

The current engine suite passes 79 tests (`python -m pytest -q`, September 13).
The scheduler and sampling changes are not visible in
`benchmark_ds41f.py`, which measures model-only prefill and decode.

## Collectives

`bench_collectives.py` times NCCL `all_reduce` at the engine's real shapes, eager
and under CUDA-graph replay (`results/collectives-*.json`). For the 168 MB
prefill reduction NCCL reaches 325 GB/s bus bandwidth, and the measured 9.4 ms per
call was rank waiting rather than transport, which the expert placement removed
(0.491 s -> 0.100 s). For the 20 KB decode reduction every NCCL
algorithm/protocol combination sits at ~31 us eager and ~21 us graphed, which is
launch latency rather than transport, so no NCCL algorithm choice helps.

**An alternative backend is now measured, and it does not win under graph
replay.** `--custom` attaches vLLM's `CustomAllreduce` with symmetric memory and
times it against NCCL per shape, with a numerical check against NCCL and per-rank
minima (`results/collectives-custom.json`). Two harness gaps had to be fixed
first. The communicator asserts unless it is attached to a non-NCCL group (vLLM
gives it a separate gloo group for the IPC handshake), and `custom_all_reduce`
returns `None` above its 8 MiB buffer, which is now recorded as
`unsupported_at_this_size` rather than an error.

| Shape | NCCL eager | NCCL graphed | Custom+symm | Custom vs eager |
| --- | ---: | ---: | ---: | ---: |
| decode MoE fp32, 20 KB | 27.3 us | 20.9 us | 20.9 us | 1.31x |
| decode MoE bf16, 10 KB | 26.3 us | 20.0 us | 20.9 us | 1.26x |
| decode attn fp32, 20 KB | 27.8 us | 21.0 us | 20.8 us | 1.34x |
| Engram bf16, 4.2 MB | 55.2 us | 47.9 us | 45.8 us | 1.21x |
| prefill / indexer, >= 16.8 MB | 110-773 us | 101-763 us | unsupported | — |

The decode-sized speedup is over *eager* NCCL only. Custom (~20.9 us) is within
noise of graph-replayed NCCL (~20.9 us), and the engine captures its decode
collectives, so switching backend would not move the graphed step. That supports
the earlier reading that the 20 KB cost is launch overhead: removing the launch
(custom, or a CUDA graph) recovers the same ~7 us, and doing both gains nothing.
For the prefill-sized reductions custom allreduce declines at its default buffer,
so NCCL remains the only path there.

Three limits are now recorded rather than assumed:

- Custom allreduce is **not CUDA-graph-capturable**: capturing
  `custom_all_reduce` aborts the process (SIGABRT, not a Python exception), so the
  graphed serving path cannot use it. The harness only attempts capture under
  `DSV41F_BENCH_CUSTOM_GRAPH=1`.
- Per-rank minima differ by under 4 us on every shape (`*_rank_spread_ms`), so
  this microbenchmark does not see the rank skew that dominated the real 8K
  prefill wait. It bounds rank timing spread under a barrier, not arrival skew.
- **FlashInfer was not measured.** Its import needs the `vllm-ds41f` env's
  `libstdc++` on `LD_LIBRARY_PATH` (the system one lacks `CXXABI_1.3.15`), and
  `FlashInferAllReduce` then asserts `distributed environment is not initialized`
  because it requires vLLM's own `parallel_state` init, which this standalone
  harness does not perform. That is an integration task, not a result.

## What the harness does and does not show

`benchmark_ds41f.py` compares two arms inside one process, and the decode portion
of each arm starts from a snapshot of the same reference-generated buffers
(`snapshot = [(b, b.clone()) for b in model.buffers()]`, restored before both
arms). That is deliberate: it isolates the decode kernels from prefill-state
differences. It also means the harness's exact decode parity says nothing about
whether the optimized prefill produced a correct state.

The arms toggle `_GROUPED_MOE` and `_STEP_GRAPHS`. Every other change in this
document — the fused hyper-connections, the fused `hc_mixes`, the FP8 GEMV
default, the warp counts, the histogram, the tile grid — is in **both** arms and
therefore invisible to this gate. For those, the evidence is:

- `check_hc_exact.py`: the fused `hc_pre`/`hc_post` are bit-identical to the
  reference expression at bf16 output.
- `check_placement.py`: placement moves logits by at most 0.71 against a
  same-configuration spread of 0.85, with an identical sampled token.
- `check_hc_mixes.py` and `check_prefill_parity.py`: the deviation of a candidate
  change against the same-configuration noise floor.

What none of that establishes: independent prefill-state equivalence (the whole
prompt, not one position), exact expert-routing agreement, or quality on
representative natural-generation prompts. Random-token prompts are not a quality
test, and no held-out task evaluation was run. Those are the missing evidence the
correctness gates in OPTIMIZE.md ask for.

## Engram lookup cost

With `DSV41F_ENGRAM_OFFLOAD=1` the two n-gram tables are CPU-resident, so every
Engram layer runs `_gather_cpu`:
indices go D2H, rows are gathered and dequantized on the CPU, and the result comes
back H2D. The D2H copy is synchronous, so that sequence is exposed in the step
rather than overlapped. `profile_engram.py` measures it directly
(`results/engram-cost.json`):

| | |
| --- | --- |
| Decode step, baseline | 32.33 ms |
| Engram calls per step | 2 |
| Engram per step | 0.452 ms |
| Engram per call | 0.226 ms |
| Share of the decode step | 1.40% |
| Instrumentation overhead | 0.052 ms (0.16%) |

The measurement adds synchronization and timing calls. The instrumented run was
0.052 ms slower per step; this difference is not a formal error bound. The
measured lookup accounts for about 1.4% of this B=1 decode workload. Eliminating
that measured cost alone would have a small effect; it does not explain the
reported gap to vLLM. This is not a comparison of offload enabled versus disabled,
and does not establish the cost for prefill or larger batches.

## Noise floor

The prefill output is not deterministic run to run. The grouped prefill's `_w2_m`
accumulates with `atomic_add`, so fp32 summation order varies; near-tie expert
selections then amplify the difference. `check_prefill_parity.py` measures this
over every prompt position at 2048, against the naive path as a control:

| Comparison | Max abs logit diff | Top-1 agreement |
| --- | --- | --- |
| naive vs itself | 0.0000 | 1.000 |
| optimized vs itself | 7.73 - 12.28 | 0.764 - 0.901 |
| optimized vs naive | 16.29 | 0.755 - 0.857 |

The naive path is bit-deterministic. The optimized path is not, and its own
run-to-run spread is the same order as its difference from the naive path. The
current measurements therefore do not establish full-prompt equivalence. Reduce
the nondeterminism and use independent state and quality checks; do not raise a
correctness threshold merely to accommodate the observed variation.

The current `benchmark_ds41f.py` gates on the last prompt position only, because
`Head.forward` slices `x[:, -1]` unless asked otherwise. In `trusted-shipped.json`,
that position passed (0.65-1.12, top-1 1.0), but it is a single sample:
`prefill_top1_agreement` is a mean over one element and is therefore exactly 0.0
or 1.0, not a rate over the prompt. The harness now also reports
`prefill_max_logit_diff_full` and `prefill_top1_agreement_full` over all prompt
positions, measured in a separate untimed pass, as diagnostics. The current
script does not gate those fields. The earlier saved full-prompt report is marked
failed, and one of its last-position differences also exceeds 1.25. A later
change to the script's checks does not make that saved report pass.

The practical consequence: agreement at one position is insufficient evidence
that two prefill paths agree. `check_placement.py` and `check_hc_mixes.py` also
compare repeated runs of the same configuration, but those controls do not
replace full-model correctness checks.

## Verification

`benchmark_ds41f.py`'s parity gate compares its two arms inside one process. That
catches kernel-level drift, but it cannot catch a change that both arms share, and
its prefill criterion is a single prompt position. Four checks cover those gaps:

- `check_placement.py` compares the same prompt before and after applying a
  placement. Placement moves logits by at most 0.71, while two identical runs
  already differ by at most 0.85. The sampled token is identical.
- `check_hc_exact.py` asserts the fused `hc_pre` and `hc_post` are bit-identical
  to the reference expression at bf16 output.
- `check_hc_mixes.py` measures a candidate change against the same-configuration
  noise floor at the same length, instead of against the fixed gate. It was used
  to choose between the `hc_mixes` dot modes and to gate the fused path.
- `check_prefill_parity.py` measures the whole-prompt divergence and the
  same-configuration noise floor side by side.

**Provenance pinning.** `ds41f.manifest` records a run's engine and model-repo
revisions, source-file hashes, config hash, checkpoint identity and the placement
calibration's sha256 plus permutation validity. `benchmark_ds41f.py` now embeds
that manifest in every report, and `expert_placement.apply` logs the resolved
file and its hash on rank 0 and enforces `DSV41F_EXPERT_PLACEMENT_SHA256` when it
is set, so a served request can name the exact calibration that relabelled its
experts. A recorded example is `results/manifest-shipped.json`
(placement sha256 `e8cab35b…`, 40 layers x 384 experts, valid permutation).

## Not done

- **Prefill determinism.** The grouped prefill's `_w2_m` uses `atomic_add`, which
  makes the whole prompt nondeterministic: two identical runs differ by up to 12.3
  on the logits with 0.76 top-1 agreement over 2048 positions. A deterministic
  reduction (a per-expert `index_add` into a zeroed buffer, or a fixed-order
  segmented reduction) would remove the run-to-run spread and make whole-prompt
  parity testable. Not attempted; it is a rewrite of the hot prefill kernel for a
  correctness and measurement benefit; its throughput effect is unmeasured.
- **Marlin kernel comparison (OPTIMIZE.md section 1).** Not attempted. It needs
  the vLLM Marlin kernels vendored and qualified against E2M1/E8M0 semantics.
  The FP8 GEMV/GEMM comparison in [Decode changes](#decode-changes) is a different
  experiment and does not stand in for it.
- **Engram lookup cost (section 6).** Measured, not optimised. It exposes 0.45 ms
  per decode step (1.4%). The graphed decode wrapper still executes Engram outside
  replay with a synchronous index transfer to CPU and a CPU gather/dequantize; the
  fix would be to move the lookup into the graph or keep a hot subset of rows
  resident on the GPU. The measurement caps the gain at ~1.4% of decode.
- **`_PREFILL_BF16` expert weight tables.** Implemented and left off. It is not
  memory-feasible at this scale: exact dequantization of the local experts to
  bf16 needs roughly 4x the fp4 storage, about 290 GB per rank, against 141 GB
  of HBM.
- **Gate in lower precision.** The gate projection (`aten::mm` of
  `[8192, 5120]` by `[5120, 384]`) runs in fp32 and costs 47 ms of an 8K prefill
  (2%). It selects experts, so changing its precision changes which experts run,
  not just the values. Not attempted; the same fused approach used for `hc_mixes`
  would apply, but the sensitivity is higher and the prize is smaller.
- **Custom or symmetric-memory collectives (section 2).** Measured: custom
  allreduce + symmetric memory is 1.2-1.34x faster than eager NCCL at
  decode-sized messages, ties graph-replayed NCCL, is not graph-capturable, and is
  unsupported above its 8 MiB buffer. FlashInfer was not measured. See
  [Collectives](#collectives).
- **DSpark integration (section 5), Engram GPU lookup (section 6), continuous
  admission (section 8), EP or pipeline topology (section 9).** Out of scope for
  this pass.
- **B=2/4/8 graph coverage.** The step-graph path requires `(1, 1)` inputs.
- **Independent prefill-state and routing validation, and natural-generation
  quality.** See [What the harness does and does not
  show](#what-the-harness-does-and-does-not-show).

## Reproducing

From `inference/`, on GPU0-3:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 DSV41F_ENGRAM_OFFLOAD=1 OMP_NUM_THREADS=1
PY=/root/miniforge3/envs/ds41f/bin/torchrun

# one-time calibration: the After numbers in this document assume it
$PY --nproc-per-node 4 plan_experts.py --out expert_placement.pt --samples 6 --length 8192

# baseline / candidate comparison
$PY --nproc-per-node 4 benchmark_ds41f.py --prompt-lens 512 2048 8192

# the fused hc_mixes, off by default
DSV41F_HC_MIXES_FUSED=1 $PY --nproc-per-node 4 benchmark_ds41f.py --prompt-lens 8192

# correctness checks
$PY --nproc-per-node 4 check_placement.py --placement expert_placement.pt
$PY --nproc-per-node 4 check_hc_mixes.py --length 2048
$PY --nproc-per-node 4 check_hc_mixes.py --length 8192
$PY --nproc-per-node 4 check_prefill_parity.py --length 2048
CUDA_VISIBLE_DEVICES=0 python check_hc_exact.py

# component profiles
$PY --nproc-per-node 4 profile_prefill.py
$PY --nproc-per-node 4 profile_decode.py
$PY --nproc-per-node 4 profile_prefill_ops.py
$PY --nproc-per-node 4 diag_balance.py
DSV41F_ENGRAM_OFFLOAD=1 $PY --nproc-per-node 4 profile_engram.py

# collectives; --custom measures vLLM's custom allreduce. Needs the vllm-ds41f env
# and its libstdc++ on LD_LIBRARY_PATH (the system one lacks CXXABI_1.3.15).
LD_LIBRARY_PATH=/root/miniforge3/envs/vllm-ds41f/lib:$LD_LIBRARY_PATH \
  /root/miniforge3/envs/vllm-ds41f/bin/torchrun --nproc-per-node 4 \
  bench_collectives.py --custom --output /root/ds41f/results/collectives-custom.json

# provenance: pin revisions, source/config hashes, checkpoint and placement
python -m ds41f.manifest --out results/manifest.json \
  --engine-repo /root/ds41f --model-repo /root/glm-testing \
  --config config.json --ckpt /data/ds41f/DSV41F-TP4 --placement expert_placement.pt
```

Set `DSV41F_EXPERT_PLACEMENT_SHA256` to the calibration's hash to make a run fail
loudly if a different placement file is used.

`expert_placement.pt` is a generated artifact and is not committed. `expert_placement.maybe_apply` picks it up when it sits next to the module, uses
`DSV41F_EXPERT_PLACEMENT` when that is set, and does nothing when it is `none`.
Without it, expert ownership is contiguous and 8K prefill is about 15% slower.

It is calibrated against a token distribution. It was calibrated here on random
ids; a deployment should recalibrate on representative traffic.

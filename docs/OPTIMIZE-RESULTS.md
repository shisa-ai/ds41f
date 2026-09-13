# Optimization results

What was implemented from [OPTIMIZE.md](OPTIMIZE.md), and what it measured.

All numbers come from `benchmark_ds41f.py` on GPU0-3 (4 x H20-3e, TP=4) with
`DSV41F_ENGRAM_OFFLOAD=1`. That harness alternates the two arms, restores a
buffer snapshot before each arm, repeats, and reports the worst rank. The tables
below use its optimized arm. Prompt tokens are random ids.

These are single-host measurements, not a controlled speedup claim. They are
comparable to each other because the checkpoint, GPU set and harness are fixed;
they are not comparable to the vLLM figures in OPTIMIZE.md, which used a
different protocol.

## Results

| Workload | Before | After | Change |
| --- | --- | --- | --- |
| Decode, 2048 context | 44.65 ms/token | 32.46 ms/token | −27.3% |
| Decode throughput | 22.40 tok/s | 30.81 tok/s | +37.6% |
| Prefill 512 | 1071 tok/s | 1330 tok/s | +24% |
| Prefill 2048 | 1684 tok/s | 2580 tok/s | +53% |
| Prefill 8192 | 1948 tok/s | 3392 tok/s | +74% |

Before is `results/baseline-gpu0123.json`, the same harness on the same GPU set
before any of this work. After is `results/trusted-shipped.json`; both are the
mean of two repeats, and the two repeats are within 1.5% of each other.

Every parity criterion passes on every row: `top1_agreement` 1.0,
`prefill_top1_agreement` 1.0, `max_logit_diff` 0.0, and
`prefill_max_logit_diff` 0.65-1.12 against its 1.25 gate.

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
from 3400 to 3500 tok/s (+3%), with 512 and 2048 unchanged by the gate. It is off
because it fails one of the harness's two prefill criteria: at 8192 it reports
`prefill_top1_agreement` 0.0, where all three runs of the configuration without
it reported 1.0. That criterion is a mean over the 8192 prompt positions, and
0.0 means every position's argmax moved while `prefill_max_logit_diff` stayed at
0.93-1.00, below the 1.25 gate — so the two criteria disagree, and the logits are
within a knife-edge of each other at this length. Both fused modes show it and
the unfused configuration does not, so it is attributable, and the gate is not
passed. Enabling it is a judgement call for a deployment, not a defect to fix.

## Serving-path changes

`ds41f/src/ds41f/scheduler/scheduler.py` enforced only an upper length bound
relative to the queue head, so `[8192, 512]` could share a cohort while the
reverse order was split. `_admit` now enforces a symmetric spread
(`max_len <= (1 + tolerance) * min_len`) and skips a too-short candidate instead
of stopping the scan.

`ReferenceBackend` sampled per row, called `.item()`, and wrote each token back to
CUDA. Sampling is now batched over the emitting cohort: one device gather, one
top-p pass, one Gumbel-max draw, one indexed device write, and a single host
transfer. Per-row temperature and top-p are preserved.

These are covered by 37 engine tests (`python -m pytest tests/` in the engine
repository), 5 of them new. The scheduler and sampling changes are not visible in
`benchmark_ds41f.py`, which measures model-only prefill and decode.

## Noise floor

The prefill output is not deterministic run to run. The grouped prefill's `_w2_m`
accumulates with `atomic_add`, so fp32 summation order varies; near-tie expert
selections then amplify the difference. `check_hc_mixes.py` measures this
directly by running the same configuration twice and comparing it against the
same run with `hc_mixes` fused:

| Length | Mode | Same setting twice | Fused off vs on |
| --- | --- | --- | --- |
| 2048 | tf32x3 | max 0.69, mean 0.078 | max 0.95, mean 0.146 |
| 2048 | split | max 0.83, mean 0.134 | max 1.17, mean 0.234 |
| 8192 | tf32x3 | max 2.83, mean 0.317 | max 1.45, mean 0.218 |
| 8192 | split | max 2.06, mean 0.293 | max 3.14, mean 0.299 |

Both fused modes are compared at 8192 in `results/trusted-hcmixes.json` (split)
and `results/trusted-final.json` (tf32x3).

At 8192 the same-configuration spread already exceeds the harness's 1.25 gate, so
`prefill_max_logit_diff` is not a usable signal at that length. The fused tf32x3
path stays at or below the same-configuration spread; the split path does not,
which is one reason the split path is not the default.

The noise is not only in the logit magnitudes. `prefill_top1_agreement` is a mean
over the prompt positions of argmax agreement between the two arms, and it is a
knife edge at 8192: runs whose `prefill_max_logit_diff` is 1.12 report 1.0, and
runs whose is 0.93 report 0.0. The two criteria therefore disagree at that
length, and a change that passes one can fail the other without either number
being informative. This is why the fused `hc_mixes` is off by default.

## Verification

`benchmark_ds41f.py`'s parity gate compares its two arms inside one process. That
catches kernel-level drift, but it cannot catch a change that both arms share.
Three checks cover that gap:

- `check_placement.py` compares the same prompt before and after applying a
  placement. Placement moves logits by at most 0.71, while two identical runs
  already differ by at most 0.85. The sampled token is identical.
- `check_hc_exact.py` asserts the fused `hc_pre` and `hc_post` are bit-identical
  to the reference expression at bf16 output.
- `check_hc_mixes.py` measures a candidate change against the same-configuration
  noise floor at the same length, instead of against the fixed gate. It was used
  to choose between the `hc_mixes` dot modes and to gate the fused path.

## Not done

- **Marlin kernel comparison (OPTIMIZE.md section 1).** Not attempted. It needs
  the vLLM Marlin kernels vendored and qualified against E2M1/E8M0 semantics.
- **`_PREFILL_BF16` expert weight tables.** Implemented and left off. It is not
  memory-feasible at this scale: exact dequantization of the local experts to
  bf16 needs roughly 4x the fp4 storage, about 290 GB per rank, against 141 GB
  of HBM.
- **Gate in lower precision.** The gate projection (`aten::mm` of
  `[8192, 5120]` by `[5120, 384]`) runs in fp32 and costs 47 ms of an 8K prefill
  (2%). It selects experts, so changing its precision changes which experts run,
  not just the values. Not attempted; the same fused approach used for `hc_mixes`
  would apply, but the sensitivity is higher and the prize is smaller.
- **Custom or symmetric-memory collectives (section 2).** Benchmarked
  (`bench_collectives.py`) but not adopted. For the 168 MB prefill reduction
  NCCL is at the transport limit; the measured 9.4 ms per call was rank waiting,
  which the placement change removed. For the 20 KB decode reduction the NCCL
  algorithms all sit at ~31 µs, which is launch latency rather than transport, so
  no algorithm change helps.
- **DSpark integration (section 5), Engram GPU lookup (section 6), continuous
  admission (section 8), EP or pipeline topology (section 9).** Out of scope for
  this pass.

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
CUDA_VISIBLE_DEVICES=0 python check_hc_exact.py

# component profiles
$PY --nproc-per-node 4 profile_prefill.py
$PY --nproc-per-node 4 profile_decode.py
$PY --nproc-per-node 4 profile_prefill_ops.py
$PY --nproc-per-node 4 diag_balance.py
```

`expert_placement.pt` is a generated artifact and is not committed.
`expert_placement.maybe_apply` picks it up when it sits next to the module, uses
`DSV41F_EXPERT_PLACEMENT` when that is set, and does nothing when it is `none`.
Without it, expert ownership is contiguous and 8K prefill is about 15% slower.

It is calibrated against a token distribution. It was calibrated here on random
ids; a deployment should recalibrate on representative traffic.

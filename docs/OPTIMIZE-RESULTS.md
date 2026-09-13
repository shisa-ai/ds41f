# Optimization results

A tuning pass over [OPTIMIZE.md](OPTIMIZE.md): what was implemented, what it
measured, and what is still open. It is **not** a complete implementation of that
list — see [Scope and status](#scope-and-status) before reading the numbers as a
progress figure.

The tuning-pass comparison comes from `benchmark_ds41f.py` on GPU0-3 (4 x H20-3e, TP=4) with
`DSV41F_ENGRAM_OFFLOAD=1`. That harness alternates the two arms, re-prefills for
each arm, repeats, and reports the worst rank. The tables
below use its optimized arm. Prompt tokens are random ids.

These are single-host measurements, not a controlled speedup claim. They are
comparable to each other because the checkpoint, GPU set and harness are fixed;
they are not comparable to the vLLM figures in OPTIMIZE.md, which used a
different protocol.

## Scope and status

Done and measured: the decode kernel work (FP8 GEMV, warp counts), the decode
fusion pass (RMSNorm, hyper-connections, MoE SwiGLU, rotary embedding; 6,232 to
4,558 launches/step and 11.9% lower decode latency for the first three, then a
further 2.3% for the rotary embedding, all bit-identical tokens), the prefill
MoE work (sync-free histogram, flat tile grid, load-balanced expert placement),
the fused hyper-connection kernels, two serving-path fixes (symmetric length
bucketing, batched on-device sampling), the packed token-broadcast protocol, a
decode-step budget with its kernel/launch breakdown, the Marlin MXFP4 MoE
comparison, batched decode (whole-step graphs and the MoE path both keyed by batch
size), and three measured-and-rejected candidates (`wo_a` in FP8, and the
object-plus-tensor broadcast protocols the packed one replaced).

Remaining work and follow-up status:

| Item | State |
| --- | --- |
| Marlin / FP4-expert kernel comparison | **Measured: 2.8× faster than the engine's grouped GEMV, graphed, at the real per-rank mix.** Not integrated; the two differ in activation precision (W4A16 vs the engine's W4A8), so a switch needs its own correctness gate. See [FP4 expert MoE against vLLM's Marlin MXFP4 kernels](#fp4-expert-moe-against-vllms-marlin-mxfp4-kernels). |
| Matched vLLM autoregressive baseline | Missing. The historical vLLM and current engine numbers use different protocols, so the gap is unquantified. |
| Graph coverage beyond B=1 | **Measured and fixed.** The step graphs and their buffers are keyed by batch size (`DSV41F_SG_BATCH_MAX`, default 8) and the batched MoE no longer takes the prefill tile path below 8 rows. Batches of 2, 4 and 8 cost 50.4, 61.3 and 82.3 ms/step against 158.5-159.8 before, and the graphed and eager paths are bit-identical at every size. See [Batched decode](#batched-decode). |
| Cohort admission | **Implemented and measured, off by default.** A cohort is whatever is in the wait queue when the engine asks, so a request that arrives a moment late waits for a full generation: the three concurrency-4 runs formed cohorts of 3, 2, 2, 1 and 3 rows and the stragglers saw 3.1-7.4 s TTFT. `DS41F_ADMIT_WINDOW_MS=8` holds an under-filled cohort open for 8 ms and fixes it: 40.80 -> 52.42 tok/s aggregate and 3702 -> 226 ms worst TTFT at concurrency 4, for a bounded delay paid only when the engine is idle. See [Concurrency](#concurrency-what-the-scheduler-actually-executed). |
| Engram lookup cost | **Measured and partly recovered.** 0.441 ms/step, 1.63% of decode, for two lookups. Splitting the call shows it is 33 us of D2H, 70 us of CPU gather and 62 us of H2D, so it is transfer latency and a scattered read of a 23 GiB table, not dispatch count. The fp8 decodes are now numpy tables, which is exact and worth 0.35-0.59%; the remaining ~0.28 ms/step needs a worker thread or a GPU-side gather, and is not taken. See [Engram lookup cost](#engram-lookup-cost). |
| Per-step synchronization | Measured at 3.12 ms/step (11%) post-fusion; 2.8-3.1 ms (9%) pre-fusion, in a single-process model loop. Pipelining the token read was implemented and measured: it recovers none of it, because the served path's per-step barrier is the blocking NCCL broadcast inside `BroadcastModel.forward`, not the read (median 28.80 ms in `enqueue`, 0.019 ms in `resolve`). See [Decode step budget](#decode-step-budget). |
| Kernel launch count | Measured and reduced: 6,232 to 4,558 launches/step (−27%) by fusing the decode RMSNorm, hyper-connection and SwiGLU chains, worth 11.9% lower decode latency (32.599 → 28.716 ms/step, ~13.5% higher tok/s) at identical tokens. See [Decode kernel fusion](#decode-kernel-fusion). |
| Decode rotary embedding | **Fused, 2.3% lower decode latency at identical tokens.** The largest single unfused item in the step's op attribution (0.68 ms/step of `copy_` and `mul`), called 198 times per step in three launches each. One launch per call now, bit-identical to the reference at the contracted product form. Decode only; prefill is untouched. See [Fused rotary embedding](#fused-rotary-embedding). |
| DSpark | Source-level feasibility analysis is complete in [OPTIMIZE.md](OPTIMIZE.md#dspark-feasibility-assessment--september-13-2026). A working verifier and runtime performance measurements remain open. |
| Custom / symmetric-memory collectives | **Integrated, opt-in, 4.8% on the served decode step.** Custom allreduce is capturable and 1.3-1.6x faster than graphed NCCL at decode sizes; routed through the engine it takes the served step from 29.0 to 27.6 ms. It requires `DSV41F_EXPANDABLE_SEGMENTS=0` (its graph-buffer registration cannot export expandable-segment memory) and stays off by default because it changes generated tokens on a near-tie-sensitive model. See [Collectives](#collectives). |
| Served latency (HTTP/SSE) | **Measured.** Decode 28.48 ms per step at batch 1 against the model loop's 26.96 ms at the same 61-token context, so the serving path adds ~1.5 ms/step. Prompt processing is 2,922 tok/s served against ~2,900 tok/s model-loop at 3,646 tokens. Concurrency is measured: see [Served latency](#served-latency). |
| Deterministic prefill | Measured, not enabled. A fixed-order accumulation makes the prefill bit-reproducible at ~2% prefill cost and no extra peak memory. See [Noise floor](#noise-floor). |

## Served latency

Every number elsewhere in this document is the model loop: `benchmark_ds41f.py`
has no HTTP, no scheduler and no token delivery, and the README says so.
`serve/bench_serving.py` talks to the running server over its OpenAI-compatible
SSE endpoint, so what it reports includes queueing, sampling, delivery and SSE
framing.

Three corrections to what this section previously claimed, all of them found by
instrumenting the server rather than the client:

- The overhead comparison was not matched. It put a 61-token prompt with at most
  128 generated tokens against the 8K row of the model loop, and explained the gap
  by saying the served context "grows through that range". It does not: 61 tokens
  in, at most 128 out, is at most 189 tokens of context.
- The client counted nonempty SSE content chunks as tokens. The server emits one
  chunk per decoded-text delta, and a delta can carry several tokens or none. The
  streaming path now returns a `usage` chunk when the request sets
  `stream_options.include_usage`, and `bench_serving.py` reads
  `completion_tokens` from it. For these runs the two counts happen to agree.
- "Per-request latency is unchanged at concurrency 2, so two rows cost the same
  per step as one" was an inference the measurement could not support. The step
trace below shows that at concurrency 2 the two requests ran as **two separate
single-row cohorts**, so nothing was batched and no batched step was timed.

### Matched measurement, one request

One request, four GPUs, greedy, 61-token prompt, `DS41F_STEP_TRACE` recording
every backend step:

| Measurement | Served | Model loop, same context | Overhead |
| --- | ---: | ---: | ---: |
| Model step, batch 1 | 28.48 ms | 26.96 ms | **+1.52 ms (+5.6%)** |
| Client inter-token latency | 28.78 ms | 26.96 ms | +1.82 ms (+6.8%) |
| Time to first token | 180.9 ms | — | — |

(`results/step-trace-final-all.jsonl`, `results/served-final-c1.json`,
`results/trusted-matched-64.json`. The model-loop column is the 64-token row of
the matched run, 26.90 and 27.02 ms over two repeats.)

The 1.52 ms between the two model steps is the broadcast from rank 0 to ranks 1-3
plus the backend's per-row top-p pass and token write-back; the further 0.30 ms to
the client is HTTP, SSE framing and detokenization. Both are real and bounded, and
the served step is still within 6% of a step that has no HTTP, no scheduler and no
delivery.

Prompt processing as a client sees it: **TTFT 1.25 s for a 3,646-token prompt**
(1.234 s, 1.248 s after a 1.551 s cold first call), which is 2,922 tok/s. The
model loop measures 2,561 tok/s at 2,048 tokens and 3,370 tok/s at 8,192, so
~3,650 tokens lands near 2,900 tok/s, or 1.26 s. The served path is at the
model-loop rate, not 1.5% above it.

### Concurrency: what the scheduler actually executed

`serve/server.py` writes one JSONL line per backend step when `DS41F_STEP_TRACE`
is set, recording the op, the number of rows, their request ids, the wall time and
how many of that step's forwards took the whole-step graph. A client cannot see
any of that: sequential single-row execution and batched execution produce the
same per-request latency when the batched step is slow, which is exactly the case
this section used to argue about.

| Concurrency | Aggregate | Client ITL | Worst TTFT | Cohorts observed |
| ---: | ---: | ---: | ---: | --- |
| 1 | 34.75 tok/s | 28.78 ms | 0.18 s | 3 of 3 runs one row |
| 2 | 33.04 tok/s | 28.60 ms | 3.12 s | 1 run of two rows, 2 runs of two single rows |
| 4 | 39.07 tok/s | 61.84 ms | 3.54 s | cohorts of 3, 2, 2, 1 and 3 rows |

Median per-step time by batch size, from the trace, every step replayed from the
whole-step graph. The model-loop column is `bench_batch_decode.py` at a 2K prompt,
which is the closest measured context; the served rows run a 61-token context:

| Batch | Served step | Model loop, 2K prompt | Steps observed |
| ---: | ---: | ---: | ---: |
| 1 | 28.48 ms | 24.72 ms | 960 |
| 2 | 55.26 ms | 50.39 ms | 376 |
| 3 | 61.79 ms | — | 235 |

Three findings, all of which the client-side numbers alone got wrong:

**Batching was not happening at concurrency 2.** In two of the three runs the two
requests were admitted as separate single-row cohorts, because the second request
had not arrived when the scheduler admitted the first. That is why per-request
latency matched concurrency 1 exactly: it was concurrency 1, twice. The 3.1-3.5 s
TTFT is the second request waiting for the first cohort's full 128-token
generation, which is `128 x 28.5 ms = 3.65 s`.

**The scheduler has no admission window, and adding one is measured.** A cohort is
whatever is in the wait queue at the instant the engine asks. Across the three
concurrency-4 runs the trace shows cohorts of 3, 2, 2, 1 and 3 rows: no run
admitted all four requests together, and the requests left out ran alone and
waited a full generation. Those stragglers are the 3.1-7.4 s TTFT. This is the
same mechanism as the concurrency-2 case and it is now measured rather than
inferred.

`DS41F_ADMIT_WINDOW_MS` holds a fresh, under-filled cohort open for a bounded
window after its first arrival, so requests sent together land in one batch. It
waits only when the engine has no active cohort and the queue is shorter than the
batch capacity, so a running batch is never slowed, and the wait happens before
any row is allocated. Three repeats of 4 x 128 tokens at concurrency 4:

| Window | Aggregate | Worst TTFT | c=1 TTFT | c=1 ITL | c=4 decode cohorts |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 ms (default) | 40.80 tok/s | 3702.8 ms | 181 ms | 28.78 ms | 1 row x248, 3 x254, 4 x115 |
| 8 ms | 52.42 tok/s | 226.3 ms | 196.8 ms | 28.77 ms | 1 row x341, 4 x358 |
| 20 ms | 51.52 tok/s | 227.0 ms | 207.9 ms | 28.67 ms | 1 row x376, 4 x377 |
| 60 ms | 50.93 tok/s | 228.2 ms | 246.9 ms | 28.73 ms | 1 row x360, 4 x381 |

8 ms is the knee: the cohorts fill, and past it the aggregate stops improving
while the idle-time delay grows linearly. The cost is bounded by the window and is
paid only when the engine would otherwise run an under-filled cohort; at c=1 that
is every request, so the measured TTFT moves 181 -> 197 ms, which is the window
plus the 179-201 ms spread the baseline itself shows across runs. Decode is
untouched at every setting (ITL 28.67-28.78 ms at c=1).

At concurrency 2 the window is throughput-neutral and a fairness win: aggregate
32.69 tok/s against 33.04, worst TTFT 202 ms against 3120 ms, but per-request ITL
55.52 ms against 28.60 ms. The old 28.60 ms was two serialized generations, so one
request waited the other out; the new number is both requests sharing one batched
step of the same total length. The default is 0, so every number elsewhere in this
document still describes the shipped configuration.

**The batched step was slow for two separate reasons, both now fixed.** See
[Batched decode](#batched-decode) for the measurements. Before the fixes a batch
of two cost 158.5 ms per step because it fell off the whole-step graph entirely,
and after that was fixed it still cost 71.5 ms because it took the grouped prefill
MoE path, whose tiles are 64 rows tall at any token count. At 71.5 ms a batch of
two was worse in aggregate than running the two rows one after the other, so
batching at concurrency 2 was a loss even once it worked.

With both fixes, a batch of two costs 55.26 ms against 2 x 28.48 = 56.96 ms of
sequential work, so batching is ahead at two and clearly ahead above it.
Concurrency 4 aggregate throughput went from 15.03 to 39.07 tok/s.

## Batched decode

Batches larger than one row were never on a fast path. Two separate causes, found
and fixed in that order.

### The whole-step graph was single-row only

`Transformer.forward` took the whole-step CUDA graph only when `input_ids.shape`
was exactly `(1, 1)`, and every buffer the graph captured was a single-row
constant. A cohort of two or four therefore ran eagerly and launched ~4,200 kernels
from the host per step. `bench_batch_decode.py` measures the result with the graphs
forced off, 2K prompt, 32 decode steps, worst rank:

| Batch | ms/step | Aggregate tok/s |
| ---: | ---: | ---: |
| 1 | 24.68 | 40.5 |
| 2 | 158.52 | 12.6 |
| 4 | 159.19 | 25.1 |
| 8 | 159.77 | 50.1 |

The cost is flat from batch 2 to batch 8, which is the signature of launch count
rather than rows. The graphs and their buffers are now keyed by batch size in
`Transformer._sg_states`, built on first use, with both position parities captured
per size. `DSV41F_SG_BATCH_MAX` (default 8) caps the sizes built; `1` restores the
single-row-only behaviour. `serve/server.py` warms every size the scheduler can
admit while loading, so the first cohort of a new size does not pay a multi-second
capture inside a request.

### The batched MoE took the prefill path

With the graph fixed, a batch of two still cost 67.7 ms. `MoE._forward` sent every
flattened token count above 1 to `GroupedMoE.routed_batch`, the grouped prefill
path, whose token tiles are `BM=64` rows tall and whose expert grid is sized for a
whole prompt. At two tokens that is the same work as at 64, and the cost was flat
from batch 2 to batch 4, so it was the tile machinery, not the rows.

`DSV41F_MOE_DECODE_ROWS` (default 8) now routes token counts up to 8 to the
single-row decode kernels, one call per row. The whole-step graph makes the extra
launches free. Measured worst rank, 2K prompt, 32 decode steps, 3 repeats:

| Batch | ms/step before | ms/step after | Aggregate tok/s before | after |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 24.78 | 24.72 | 40.4 | 40.5 |
| 2 | 67.70 | **50.39** | 29.5 | **39.7** |
| 4 | 77.57 | **61.32** | 51.6 | **65.2** |
| 8 | 93.29 | **82.34** | 85.8 | **97.2** |

(`results/batch-decode-sg.json` and `results/batch-decode-final.json`.)

This is the more accurate path, not a speed-for-accuracy trade.
`check_moe_decode_rows.py` compares both paths against the reference `Expert`
forwards on real checkpoint weights, with routing ids that repeat and that point
outside the rank's expert range:

| Batch | Per-row vs reference | `routed_batch` vs reference |
| ---: | ---: | ---: |
| 1 | 0.00000 | 0.04059 |
| 2 | 0.00000 | 0.04143 |
| 4 | 0.00000 | 0.04494 |
| 8 | 0.00000 | 0.04270 |
| 16 | 0.00001 | 0.04391 |

(Normalized RMS. `results/check-moe-decode-rows.json`.) The per-row path is
bit-identical to the reference at every batch size the threshold covers;
`routed_batch` is not. End to end, the greedy continuation from one restored state
is token-for-token identical with the threshold at 0 and at 8, at batches 2, 4 and
8 (`results/parity-moerows0.json`, `results/parity-moerows8.json`).

### Whole-step graph parity

Generalizing the graph to more rows does not change its results. Over 24 decode
steps from one restored state, the graphed arm and an eager arm agree exactly -- 0
differing token steps and `max_logit_diff` 0.0 at batches 1, 2, 4 and 8 -- and a
third arm that reruns the eager configuration from the same state also agrees at
0.0, so the comparison has no residual noise floor.

The parity harness needs `DSV41F_PF_FIXED_ORDER=1` to make that control arm
meaningful. Without it the same test reports 5.2, 11.9 and 28.3 at batches 2, 4 and
8, which is the pre-existing `atomic_add` nondeterminism of the grouped prefill
MoE that the batch path already used, not the graph. With the per-row MoE above,
batches up to 8 no longer reach `routed_batch` at all and the flag is not needed
for them.

### What is still open

A cohort is whatever is in the wait queue at the instant the engine asks for one.
`DS41F_ADMIT_WINDOW_MS` closes most of that: at 8 ms, three concurrency-4 repeats
admitted all four rows every time, aggregate went 40.80 -> 52.42 tok/s and worst
TTFT 3702 -> 226 ms, at the cost of the window itself when the engine is idle. It
defaults to 0, so the numbers in this document are the unwindowed ones; see
[Concurrency](#concurrency-what-the-scheduler-actually-executed) for the curve.
What it does not fix is a request that arrives *after* a cohort has started: that
one still waits for the cohort to drain, which is the mid-flight refill milestone
in `docs/OPTIMIZE.md` section 4.

Above the threshold `routed_batch` is still the right path, but where the
crossover sits is not settled. The isolated MoE benchmark in
`check_moe_decode_rows.py` puts it between batch 4 and batch 8, favouring
`routed_batch` from 8 upward -- yet end to end, batch 8 is 11 ms per step faster
with the per-row path. The isolated number cannot see that the whole-step graph
makes the extra launches free. Batch 16 and above was not measured end to end,
and the threshold is left at 8.

## Full-prompt diagnostic run

`results/fulllogits-hcmixes-off.json` is newer than the trusted run, but it is
marked **failed** on correctness checks. It contains the following averages of
two optimized runs; these numbers describe execution time, not a validated
release. The full-prompt score comparison runs separately from the timed
prompt-processing pass. The README reports the passing `trusted-shipped.json`
run instead.

| Prompt/context length | Prompt processing | Time to process prompt | Decode throughput | Decode latency |
| --- | ---: | ---: | ---: | ---: |
| 2,048 tokens | 2,560 tok/s | 0.80 s | 30.62 tok/s | 32.66 ms/token |
| 8,192 tokens | 3,270 tok/s | 2.51 s | 29.29 tok/s | 34.20 ms/token |

These are model-only measurements on random-token prompts with predetermined
continuation tokens. Fresh end-to-end serving throughput and peak memory are not
recorded. The tuning-pass comparison below is a separate, earlier run.

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
naive arms differ by up to 16.3 with 0.75–0.87 top-1 agreement. Two runs of the
*same* optimized configuration differ by up to 11.6 with 0.76 agreement, while the
naive arm is bit-deterministic at 0.0000. That variation does not establish
correctness or rule out defects. Independent state checks and realistic quality
tests remain necessary. The saved `fulllogits-hcmixes-off.json` report is marked
`failed`; its measurements must not be presented as a passing full-prompt
comparison.

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

### Attention output projection (`wo_a`) in FP8: measured, not adopted

`wo_a` is block-diagonal over `o_groups`, so the model runs a bf16 grouped
`einsum("bsgd,grd->bsgr")`. vLLM keeps this weight in FP8 and runs a grouped FP8
GEMM. `bench_wo_a.py` compares the two at the real per-rank shape (`o_groups`=2,
`o_lora_rank`=1024, input 4096) against a float64 reference, eager and under
graph replay. Weight quantization is an offline cost and is not timed; the
per-forward activation quantization is.

| Shape | bf16 einsum (graphed) | FP8 grouped GEMM (graphed) | FP8 speedup | bf16 max err | FP8 max err | ref absmax |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decode, 1 token | 15.4 µs | 112.5 µs | 0.14× | 0.031 | 0.461 | 14.0 |
| prefill, 512 | 84.6 µs | 177.0 µs | 0.48× | 0.031 | 0.626 | 15.3 |
| prefill, 2048 | 290.4 µs | 407.4 µs | 0.71× | 0.032 | 0.656 | 17.9 |
| prefill, 8192 | 1,007.9 µs | 1,448.7 µs | 0.70× | 0.058 | 0.643 | 17.3 |

FP8 is **slower at every shape**, by 7× at decode. The halved weight traffic does
not pay for the per-forward activation quantization plus the per-group
`_scaled_mm` launches; the bf16 einsum at this size is launch-bound, not
bandwidth-bound. The `einsum` also beats an explicit `bmm` reformulation
(36.8 µs vs 55.6 µs eager at decode), so the current formulation is already the
better bf16 one.

The realistic ceiling was small to begin with: graphed bf16 `wo_a` is 15.4 µs per
layer, or about 0.62 ms of a 32.4 ms decode step (1.9%) across 40 layers. vLLM's
version fuses the activation quantization into the preceding inverse-RoPE kernel,
which this isolated comparison does not; reproducing that is the real work, and
it would have to overcome the same launch-bound shape. Not adopted.

### FP4 expert MoE against vLLM's Marlin MXFP4 kernels

`bench_marlin_moe.py` loads the shipped rank-0 shard's 96 local experts, repacks
them with vLLM's own `prepare_moe_mxfp4_layer_for_marlin`, and times the engine's
`GroupedMoE.forward` (W4A8 grouped GEMV) against `fused_marlin_moe` (W4A16
Marlin) at the real per-rank decode shape: one token, top-6 of 384 global
experts, 96 owned locally. Graph-replayed, because that is what the decode path
pays:

| Active local experts | Marlin | engine grouped GEMV | speedup |
| ---: | ---: | ---: | ---: |
| 0 | 12.4 µs | 29.0 µs | 2.3× |
| 1 | 46.1 µs | 118.9 µs | 2.6× |
| 2 | 51.1 µs | 145.5 µs | 2.8× |
| 3 | 61.0 µs | 178.5 µs | 2.9× |
| 6 | 94.8 µs | 303.0 µs | 3.2× |
| mean, 14 cases with ≥1 local | 56.0 µs | 158.8 µs | **2.80×** |

Each number is the whole MoE block (both projections, SwiGLU clamp, both
activation quantizations), not one kernel. Eager, Marlin sits at a flat ~149 µs
for every row including the 0-local case, because `moe_align_block_size` and the
intermediate-cache allocations dominate; under replay that host cost disappears
and the device-side block count skips the empty experts. Only the graphed column
is comparable to the engine's decode step.

The engine's decode MoE is `_w13` + `_w2` = 4.13 ms/step, 12.7% of the step. At
the 2.8× ratio that would fall to about 1.5 ms/step, saving roughly 2.6 ms/step
(8% of decode). That projection assumes the ratio transfers from this isolated
harness to the in-graph path; it is not an end-to-end measurement.

**The two are not numerically interchangeable.** Against the eager per-expert
reference, the engine's grouped path is bit-identical (max abs 0.0, NRMSE 0.0) —
both quantize activations to FP8 (`linear()` calls `act_quant` for fp4 weights
too). Marlin differs from that reference by 0.03-0.22 max abs and **4.2-5.3%
NRMSE**. That gap is consistent with Marlin computing in bf16 activations while
the reference is FP8, so this does not show Marlin is less accurate — it shows the
engine's own activation quantization is the difference. Switching would still
need its own correctness gate, because it changes the numerics of every routed
expert.

## Decode step budget

A 20-step CUDA/CPU profile of the B=1 decode step at 2K context
(`profile_decode.py`, `results/profile-decode.log`) accounts for the 32.4 ms
published latency. The step is **GPU-bound**: the host enqueues a step in
0.02 ms, and the CUDA kernels plus their inter-kernel gaps fill the whole step.

> **Correction (September 13, later pass).** That 0.02 ms figure is wrong and has
> been withdrawn. `probe_decode_cpu.py` timed the gap *between* loop iterations,
> which is the loop's own bookkeeping plus the token write and excludes the model
> call entirely, so it could not measure enqueue cost at all. With the measurement
> moved inside the call, the host spends **~24.4 ms inside `model(...)`** on a
> ~24.5 ms GPU step (`results/probe-decode-cpu-pipe.json`).
>
> What that 24.4 ms consists of is **not** established. `time.process_time()`
> tracks wall time during a pure GPU wait (`probe_sync_spin.py`: 34.97 ms of process
> time against 35.10 ms of wall while blocking on one matmul), so it counts the
> sync spin and cannot separate host dispatch from blocking. The separate
> micro-benchmark of one call site is sound, because it loops without syncing:
> `act_quant` costs **16.8 µs of host dispatch per call** (wall and process time
> agree at 16.80/16.84 µs over 1000 iterations), and it is called ~410 times per
> step, so ~6.9 ms/step of host dispatch is real.
>
> Whether that host time is on the critical path was then tested directly rather
> than inferred: caching `act_quant`'s kernel lookup, allocations and views removes
> ~3.6 ms of that dispatch and moves the step by **+0.71%**
> (`results/ab-act-quant-cache.json`). A saving that large showing up that small
> means the host dispatch is largely overlapped with GPU execution, which is
> evidence *for* the GPU-bound reading -- but from a measurement, not from the
> withdrawn 0.02 ms.

Kernel time, grouped from the trace (6,232 launches per step, 25.77 ms of kernel
execution per step):

| Kernel group | ms/step | share of kernel time | launches/step |
| --- | ---: | ---: | ---: |
| elementwise, copy and reduce kernels | 9.97 | 38.7% | 4,938 |
| dense FP8 GEMV (`fp8_gemv_kernel`) | 3.18 | 12.3% | 290 |
| NCCL all-reduce (f32 + bf16) | 3.06 | 11.9% | 91 |
| sparse attention | 2.74 | 10.6% | 40 |
| MoE gate+up (`_w13`) | 2.36 | 9.2% | 40 |
| MoE down (`_w2`) | 1.77 | 6.9% | 40 |
| activation quantization (`act_quant_kernel`) | 1.05 | 4.1% | 410 |
| other GEMMs (`nvjet`, `cutlass`) | 0.43 | 1.7% | 60 |
| radix/bitonic sorts | 0.39 | 1.5% | 49 |
| remainder | 0.41 | 1.6% | 227 |

The median kernel in a decode step runs for **1.82 µs**; the 90th percentile is
6.37 µs. The five large groups (dense GEMV, sparse attention, MoE gate+up, MoE
down and NCCL all-reduce) are 51% of the kernel time in 501 of the 6,232
launches. The other 5,731 launches are elementwise, copy, reduce and activation
quantization kernels worth 43% of the time. **Decode is launch-count-bound, not
bandwidth-bound** is too strong a claim to make from these numbers: thousands of
small kernels make fusion promising, but a launch count does not by itself
establish the bottleneck, and under graph replay the launch count the host issues
is much smaller than the number of kernels the GPU runs. Summed kernel durations
from one run are also not an additive wall-time budget. The measured justification
for the fusion work is the end-to-end A/B, not the launch count.
does not.

### The per-step synchronization costs 2.8-3.1 ms

`benchmark_ds41f.py` calls `torch.cuda.synchronize()` inside the decode loop, and
`ReferenceBackend._collect` calls `sampled.tolist()` on every step to deliver the
token. Both drain the pipeline before the next step is enqueued.

`probe_decode_cpu.py --mode pipe|sync` measures the same 30 decode steps from
identical state with and without that drain, one mode per process:

| Mode | host time in `model()` | kernel span | wall |
| --- | ---: | ---: | ---: |
| pipelined (no per-step sync) | 24.669 ms | 24.76 ms | **24.87 ms** |
| synchronized each step | 13.214 ms | 27.83 ms | **27.99 ms** |

> **Correction.** This table previously reported an `enqueue` column of 0.018 ms
> (pipelined) and 0.021 ms (synchronized), and those numbers are withdrawn. They
> came from timing the gap between loop iterations, which is the loop's own
> bookkeeping plus the token write and excludes the model call, so they could not
> measure enqueue cost. The numbers above come from the timer moved *inside* the
> call (`results/probe-decode-cpu-{pipe,sync}.json`, B=1, 2K, all fusions on).
>
> Two things about the corrected column. First, it is host time spent inside the
> call, which is not the same as time spent issuing work: in pipelined mode the
> queue fills and the host blocks on backpressure, which is why it reads 24.7 ms
> there and 13.2 ms in synchronized mode, where every step starts from a drained
> queue. So this column does not show that dispatch is cheap, and it does not show
> that it is expensive either; the direct test of that is the `act_quant` cache
> below, which removes ~3.6 ms of dispatch for +0.71%. Second, in synchronized
> mode the `torch.cuda.synchronize()` sits *after* the timer, so that row's host
> time excludes the drain by construction.
>
> The wall-time comparison, which is what this experiment was for, is unaffected:
> **27.99 − 24.87 = 3.12 ms/step, about 11%** of the post-fusion step.

The drain costs **3.12 ms/step, about 11%** of the post-fusion step (it was
2.8-3.1 ms, 9%, against the pre-fusion 32.4 ms step), and it is not required for
correctness: with the deterministic prefill (`DSV41F_PF_FIXED_ORDER=1`) both modes
produce byte-identical token sequences from the same start state, including the
10-step warm-up before the measured window.

The comparison only holds under that flag. With the default nondeterministic
prefill, two runs of the *same* mode already differ from each other (the grouped
prefill's `atomic_add` reduction), so token equality across modes is not evidence
about pipelining until prefill is made deterministic. That confound produced a
wrong first reading of this measurement; it is recorded here so the check is not
repeated without the flag.

The step budget therefore reads: 24.8 ms of kernel execution, roughly 3.7 ms of
inter-kernel gaps inside the graph replays, and 3.12 ms of exposed
synchronization (post-fusion, B=1, 2K). The synchronization is the only part that
is pure host-side loss in a single-process model loop, and the obvious way to
recover it is to read a token from step *i* after enqueueing step *i+1*.

### That 3.12 ms was implemented and measured, and the served path cannot use it

`ReferenceBackend` now splits `execute` into `enqueue` (forward, sampling, device
write-back, then a non-blocking copy of the sampled ids into a pinned buffer and a
recorded event) and `resolve` (wait on the event, read the buffer). A pipelined
`LLMEngine` loop was written on top: build step *i*'s plan, enqueue it, and only
then read step *i-1*'s tokens. It produced token-identical greedy output on every
arm of `serve/check_pipeline_parity.py`, including a real prefix-cache hit
(shared=85 of 98 tokens) and a four-row stochastic cohort, and it **bought
nothing**:

| | serial | pipelined |
| --- | ---: | ---: |
| concurrency 1, inter-token | 28.77 ms | 28.91 ms |
| concurrency 4, inter-token | 68.23 ms | 68.31 ms |
| concurrency 1, TTFT | 196.3 ms | 233.0 ms |

Timed separately per step under the pipelined loop, the host spends a median
**28.80 ms inside `enqueue`** and **0.019 ms inside `resolve`**. The read is not on
the critical path at all. `enqueue` is where the whole step goes because the served
model is `BroadcastModel`, whose `forward` ends in a blocking NCCL broadcast that
every rank must arrive at: rank 0 cannot enqueue step *i+1*'s forward before step
*i* has finished on all four ranks, so the GPU queue can never hold two steps.

So the 3.12 ms is real and is not recoverable from the token read. The served path
does pay an equivalent per-step barrier, but the barrier is the broadcast: the fix
is to stop making rank 0 rendezvous with its peers every step (send the step spec
one step early, or let every rank drive its own engine), not to move the host read.
Until then the split contract stays as the engine's step contract, unused for
overlap, because it is the one place such a change would land.

Two things this run settled that were open questions above:

- **The RNG hazard is narrower than feared, and still real.** A finished row keeps
  its place in the fixed cohort and keeps drawing, so the *number* of draws per
  step does not depend on finishes -- the emit set is a function of position
  against prompt length only. What pipelining changes is the *number of steps*: the
  extra trailing decode step draws once more per cohort row from the global
  generator, which shifts the stream for every later stochastic request. Greedy
  output is unaffected. That is why the loop was removed rather than left behind a
  default-off flag: a flag that changes sampled tokens and buys nothing is a trap.
- **The capture boundary survives it.** `on_finish` keys a snapshot by
  `prompt + completion[:-1]`, and under pipelining the caches have advanced one
  position further when the hook fires. The key still names a prefix the caches
  cover, and the position past it is overwritten on restore from `shared_len`, so
  the round trip is exact: the prefix-hit arm of the parity harness produced the
  same continuation with and without pipelining, from a genuine hit.

The gate the previous text asked for -- a multi-request cohort with differing
finish positions and a nonzero temperature, compared against the serial path -- is
`serve/check_pipeline_parity.py`, and it is what caught the stochastic difference.

### What a pipelined step would have had to get right

The plan for step *i* is built from finish state that is one step old, because it
is built before step *i-1*'s tokens are read. Two consequences, both handled in the
implementation that was measured and then removed:

- A decode plan's row set does not depend on finishes, because a finished row keeps
  its place in the fixed cohort. The one case that does is a drained cohort, where
  the next plan would be a prefill that admits new rows and overwrites
  `_last_cohort` before the drained rows have been captured; the engine resolved the
  in-flight step before planning in that case.
- The last rows to finish are not known to be finished, so the cohort takes one
  more decode step than the serial path. That step needs one position of slack in
  the cohort's token buffer, and it is what shifts the generator stream.

## Decode kernel fusion

The [step budget](#decode-step-budget) says decode is launch-count-bound: 6,232
launches per step with a median kernel of 1.82 µs, and 37% of the kernel time in
the elementwise, copy and reduce group alone (41% including activation
quantization). That is where fusion pays, and this pass fused the three chains the
punchlist named
([OPTIMIZE.md](OPTIMIZE.md#3-qualify-existing-fusion-and-close-graphhost-gaps)):
RMSNorm, the hyper-connection pre/post mixers, and the MoE SwiGLU.

Each was accepted only after a **bit-exactness check against the reference
expression**, because decode is the one path in this engine that is currently
reproducible token-for-token, and that property is worth more than a few percent.
The checks are listed in [Verification](#verification).

| Change | Reference launches | Fused launches | Est. removed/step | ms/step | Decode | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Fused `hc_pre`/`hc_post` at decode | ~18/layer | 4/layer | ~560 | +1.755 | +5.76% | identical |
| Fused RMSNorm at decode | 8/norm | 2/norm | ~1,000 | +1.793 | +5.85% | identical |
| Fused SwiGLU+route in the MoE | 9/layer | 1/layer | ~320 | +0.569 | +1.93% | identical |
| Fused `hc_mixes` coefficient math | ~5/layer | 2/layer | ~120 | +0.422 | +1.47% | identical |
| **All three together** | | | **1,674** | **+3.883** | **11.9% lower** | **identical** |
| Fused rotary embedding | 3/call | 1/call | ~396 | +0.660 | +2.3% | identical |

The per-change rows are separate interleaved A/B runs (three repeats each, worst
rank) at B=1, 2K context, `DSV41F_ENGRAM_OFFLOAD=1`
(`results/ab-hc-fused-decode.json`, `results/ab-rmsnorm-fused.json`,
`results/ab-moe-swiglu-fused.json`). The combined row toggles all three flags at
once (`results/ab-combined-decode-fusion.json`); it is 11.9% rather than the
13.5% the rows sum to, which is the expected sub-additivity when several changes
all remove the same kind of launch. `+11.91%` in the saved JSON is a *latency*
reduction, not a throughput gain: 32.599 → 28.716 ms/step is 11.9% lower latency
and about 13.5% higher tok/s. The est. removed/step column is the launch
arithmetic, not a measurement; it sums to ~1,880 against a measured net of 1,674,
because the per-change counts come from the reference expressions while the
measured net comes from the two traces below.

The `hc_mixes` row is a later pass and is not in the combined row: the three
earlier fusions were already on when it was measured, so its `ms/step` is against
the fused baseline (28.753 → 28.332 ms/step, `results/ab-hc-mixes-decode.json`).
It fuses the cast-and-square and the add-rsqrt-multiply of the coefficient
calculation, keeping torch's mean and the cuBLAS GEMM; `tl.math.rsqrt` is
bit-identical to `torch.rsqrt`, so the whole chain is bit-exact -- 0 of 4,800 fp32
elements differ over 200 draws at four seeds (`check_hc_mixes_decode.py`).

The rotary-embedding row is a later pass again, measured against the same fused
baseline; see [Fused rotary embedding](#fused-rotary-embedding).

### Fused rotary embedding

The step budget's launch-count table groups kernels by name, which says *what* ran
but not *which line* emitted it. Joining the trace's kernel, runtime and CPU-op
levels by correlation id and source frame (`profile_decode_stacks.py`) ranks the
step by the line that emits each launch, and the largest single unfused item was
`apply_rotary_emb` (`model.py`): **0.519 ms/step of `aten::copy_` plus 0.160 ms of
`aten::mul`, 0.68 ms or 2.8% of the step**.

A decode step calls it **198 times** (`probe_rope_shapes.py`,
`results/probe-rope-shapes.txt`). Each call is three launches -- `float()`, the
complex multiply, `copy_` back -- over a small tensor, so the cost is launch count
rather than bytes. `rope_kernels.py` does the three in one launch, in place, which
is what the reference does anyway (`y.copy_(x)` with `y is x`).

The call sites pass slices `x[..., -rd:]`, so the last dimension is contiguous and
the one before it is strided. Over four decode steps the probe sees five
signatures, and every leading dimension except the last is 1:

| Site | Shape | Stride | Calls/step |
| --- | --- | --- | ---: |
| window KV latent | (1, 1, 64) | (512, 512, 1) | 63.2 |
| q | (1, 1, 16, 64) | (8192, 8192, 512, 1) | 59.5 |
| attention output (inverse) | (1, 1, 16, 64) | (8192, 8192, 512, 1) | 59.5 |
| indexer | (1, 1, 8, 64) | (1024, 1024, 128, 1) | 12.0 |
| main KV | (1, 1, 64) | (128, 128, 1) | 3.8 |

Those five per-site counts are averages over four steps and sum to exactly 198.0
per step; they are not integers because some call sites fire periodically rather
than every step. Because the rows are uniform, the kernel needs only a row count
and a row stride, and it handles the strided and contiguous cases and the
conjugate (inverse) rotation. Frequencies are `complex64` with a single row,
broadcast over heads, read from a float32 view of them with real and imaginary
interleaved.

**Exactness needed a measurement rather than an assumption, and the first
assumption was wrong.** torch's complex multiply contracts its products, and
*which* form it uses decides whether a one-ulp fp32 difference survives into the
bf16 result -- about once in 300k elements, which this model turns into a
different argmax. The synthetic check could not separate the forms: at 32 draws all
four read zero differences, because random inputs rarely land on a bf16 rounding
tie. At 512 draws it separates them, and `check_rope_inmodel.py`, which compares
both paths on every call the real model makes, settles it:

| Product form | Differing bf16 elements |
| --- | ---: |
| `re = fma(a,c,-(b*d))`, `im = fma(a,d,b*c)` | **0** |
| `re = fma(a,c,-(b*d))`, `im = fma(b,c,a*d)` | 23 |
| `re = a*c - b*d`, `im = fma(b,c,a*d)` | 23 |
| `re = a*c - b*d`, `im = a*d + b*c` | 23 |

over 3,192 calls and 2,133,504 elements in 24 decode steps. The first A/B with the
uncontracted form was **2.66% faster and produced different tokens at step 8**: a
one-ulp difference is not a rounding curiosity here, and the in-model comparison,
not the synthetic one, is what catches it. With the contracted form:

| Change | ms/step | Decode | Tokens |
| --- | ---: | ---: | --- |
| Fused rotary embedding at decode | +0.670, +0.651 | +2.36%, +2.30% | identical |

Two interleaved A/B runs of three and four repeats (`results/ab-rope-fused.json`,
`results/ab-rope-fused-b.json`): 28.400 → 27.729 and 28.365 → 27.714 ms/step. The
trace agrees independently: in the no-graph profiles the `aten::copy_` plus
`aten::mul` group falls from 0.930 to 0.250 ms/step, a 0.680 ms difference
(`results/decode-stacks-rope-off.txt`, `results/decode-stacks-rope-on.txt`). The
trusted benchmark (`results/trusted-rope-fused.json`) passes at 512, 2048 and 8192
with decode `max_logit_diff` 0.0, `mean_kl` 0.0 and `top1_agreement` 1.0: the fused
path is bit-identical there, not merely close.

Decode only. The reference indexes frequencies by sequence position and broadcasts
them over heads, so a multi-position tensor does not have uniform rows; the guard
requires a single position and leaves prefill on the reference expression. Prefill
is therefore unchanged, and its last-position gate values in the trusted run (0.49
to 1.25) sit inside the recorded spread of the pre-existing prefill
nondeterminism described in [Noise floor](#noise-floor).

The trusted re-run behind these numbers was made on a noisier machine than the
earlier runs, and that is worth recording rather than smoothing over. Its parity
result is clean -- 12 of 12 checks at decode `max_logit_diff` 0.0 and
`top1_agreement` 1.0, status `passed` -- but its latency is not: four repeats at
8,192 tokens give 27.47, 29.78, 27.49 and 29.01 ms/step
(`results/trusted-rope-fused-4rep.json`), while `results/trusted-sgbatch.json`,
`trusted-shipped.json` and `trusted-fused.json` each have repeats within 0.1 ms of
each other. The four-repeat medians are 27.45, 27.51 and 28.25 ms/step at 512,
2,048 and 8,192 against 27.94, 28.05 and 28.04 in the earlier run, so two lengths
read better and one reads worse by more than the effect being measured. The
interleaved A/B is the instrument that survives this: it alternates arms inside one
process and both runs agree to 0.02 ms. The 4.6% spread in this harness is not
attributable to a decode-only change of 0.66 ms, and no headline number is taken
from it.

### Host dispatch caching

`act_quant` costs **16.8 µs of host dispatch per call** and runs ~410 times per
step, so ~6.9 ms/step of the host's ~24.4 ms is that one function
(`probe_host_dispatch.py`; wall and process time agree, so this is dispatch and
not a sync spin). Caching its kernel lookup, its two allocations and its three
views removes ~3.6 ms of that, and moves the step by **+0.71%** (28.610 → 28.408
ms/step, `results/ab-act-quant-cache.json`), bit-identical outputs over 3 reps x 3
shapes (`check_act_quant_cache.py`).

That gap is the finding: a 3.6 ms saving showing up as 0.2 ms means host dispatch
is largely overlapped with GPU execution. It is left **off by default**
(`DSV41F_ACT_QUANT_CACHE=1` opts in) because the cached path returns *shared*
`y`/`s` buffers: a caller that holds them across another `act_quant` call of the
same shape sees them overwritten. That is safe on one stream, where the consumer
is enqueued immediately after the producer -- which is how the decode path uses
it -- but it is an assumption about the callers, and 0.71% does not buy the right
to make it silently.

### Where the launches went

Two 20-step traces at B=1, 2K context, analysed with `analyze_decode_trace.py`
(`results/decode-trace-fusion.txt`):

| Kernel group | launches/step, before | after | ms/step, before | after |
| --- | ---: | ---: | ---: | ---: |
| elementwise, copy and reduce | 4,980 | 2,774 | 9.48 | 5.21 |
| fused RMSNorm (`_square_cast` + `_rms_apply`) | — | 332 | — | 0.38 |
| fused hyper-connections | — | 161 | — | 0.24 |
| fused SwiGLU+route | — | 40 | — | 0.08 |
| dense FP8 GEMV | 290 | 290 | 3.18 | 3.18 |
| NCCL all-reduce | 92 | 92 | 3.07 | 4.86 |
| sparse attention | 40 | 40 | 2.74 | 2.75 |
| MoE gate+up (`_w13`) | 40 | 40 | 2.36 | 2.34 |
| MoE down (`_w2`) | 40 | 40 | 1.77 | 1.73 |
| activation quantization | 410 | 410 | 1.05 | 1.05 |
| radix/bitonic sorts | 98 | 98 | 0.79 | 0.79 |
| other GEMMs (nvjet/cutlass) | 60 | 60 | 0.62 | 0.62 |
| MoE combine | 40 | 40 | 0.10 | 0.10 |
| remainder | 140 | 140 | 0.61 | 0.61 |
| **total** | **6,232** | **4,558** | **25.77** | **23.92** |

**6,232 to 4,558 launches per step, a 27% reduction**, entirely in the small-kernel
groups. The fused kernels that replace them are 533 launches totalling 0.69 ms.

One row needs a caveat. The bf16 all-reduce is 11 launches in both traces but 0.46
ms before and 2.17 ms after. A single trace per configuration cannot separate
run-to-run variance in the collective from a real effect: with fewer kernels on
the GPU, a collective that previously overlapped other work is more exposed. The
end-to-end A/B (11.9% lower latency, three repeats, interleaved) is the
trustworthy number here; the per-group split is one trace each and should be read
as indicative.

### Why the reductions stayed in torch

The first attempt at each of these was a single kernel that computed the whole
chain, and for the RMSNorm and SwiGLU that kernel was *not* bit-exact. Two
specific causes, both worth recording because they are easy to hit again:

- **`tl.exp` and `tl.sqrt` are not `expf` and `sqrtf`.** Triton's `tl.exp`
lowers to a fast `exp2`-based path: `probe_swiglu_steps.py` finds it differing from
`torch.exp` on **55,722 of 82,944** fp32 inputs. `tl.sqrt` is `sqrt.approx.f32`, and
Triton's default fp32 division is not `div.rn`. The working formulations are
`triton.language.extra.libdevice.exp` and `tl.math.rsqrt`, which are bit-identical
to `torch.exp` and `torch.rsqrt` (4,096/4,096 in `probe_rsqrt.py`). The SwiGLU
kernel needed `tl.fdiv(..., ieee_rounding=True)` as well.
- **`torch.mean`'s reduction order is not reproducible in a kernel.** `probe_rsqrt.py`
tries a single masked block, 1024-wide and 128-wide chunked accumulation, and a
strictly sequential sum: none agree with `torch.mean` on more than part of the
dimensions (at 5120, four of eight rows differ for the best of them).

So the fused RMSNorm does **not** compute `var`. It fuses the cast and square into
one kernel, calls torch's own `mean`, and fuses `add + rsqrt + mul + mul + cast`
into a second. The reduction is torch's, so the result is exact by construction.
That is also why it is two kernels rather than one.

The same reasoning applied to the hyper-connection mixers, which already had
fused kernels validated for prefill: they were gated to `x.size(1) > 1`, and the
single-token path kept the reference expression on the assumption that its cost
was negligible. `check_hc_exact.py` now covers `s=1` and finds them bit-identical
there too, so the gate is relaxed behind its own `DSV41F_HC_FUSED_DECODE` flag.

### The one-launch RMSNorm, measured and not enabled

A single kernel that computes the sum of squares itself is faster still, and it is
kept as `DSV41F_RMSNORM_ONEPASS` (default off):

| Variant | Decode vs the two-kernel path | Exactness |
| --- | ---: | --- |
| Two kernels, torch reduction (default) | — | 0 / 1,408,000 bf16 elements differ |
| One kernel, own reduction | +2.97% | ~5 per 1,000,000 elements, 1 bf16 ulp |

The one-launch variant's decode tokens matched the reference over 64 eager steps
but diverged at step 19 in a 30-step graph-replayed A/B, which is what an error
rate of ~5e-6 over ~240 norm calls per step predicts. It is the same trade as the
fixed-order prefill path: measured, recorded, off by default.

### The end-to-end gate, and one field that is not about this change

`benchmark_ds41f.py` was run with all three fusions on and with all three off
(`results/decode-fusion-e2e.json`, `results/decode-fusion-e2e-off.json`), each
with `--no-parity-abort` so every length and repeat is recorded:

| Run | decode `top1_agreement` | decode `max_logit_diff` | prefill last-position `max_logit_diff` | gate |
| --- | ---: | ---: | --- | --- |
| all three fused | 1.0 (all 6 checks) | 0.0 (all 6) | 0.64, 0.70, 0.72, 0.66, **1.26**, 1.03 | failed on the 1.26 |
| all three off | 1.0 (all 6 checks) | 0.0 (all 6) | 0.52, 0.72, 0.92, 0.85, 1.18, **1.55** | failed on the 1.55 |

The decode fields are exact in both runs, which is the claim this section makes.
The gate still reports `failed`, and it does so **with the fusions off as well**,
worse: 1.55 against 1.26. The failing field is `prefill_max_logit_diff`, the
last-position distance between the optimized and naive *prefill* paths, which
`check_decode_parity.py` shows these changes do not touch (0 of 66,191,360 prefill
logits differ with them on versus off). It is the pre-existing `atomic_add`
nondeterminism described in [Noise floor](#noise-floor), whose own run-to-run
spread is 8-16 over the whole prompt, sitting against a 1.25 bound on a
single-position sample. Both runs are kept in `results/` rather than only the
passing one.

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

A file beside the module does not by itself enable placement in serving, and
reporting "applied" is not evidence that it was. `serve/verify_placement_dispatch.py`
loads the model through the real serving loader on all four ranks with
`audit=True`, which makes `apply()` check its own work on the model's real
tensors, and records the outcome (`results/placement-dispatch-verify.json`):

- all four ranks resolved the same file and the same sha256
  (`e8cab35b…`);
- the loader applied it on every rank;
- each gate is the pre-permutation gate reordered by the calibration's
  `new_to_old`;
- every local expert slot holds the bytes of the expert `exchange_plan` assigned
to it, checked against hashes gathered from the source rank;
- each rank's local expert ids match an independent recomputation from the
  placement file, and each layer's ids across the four ranks are a permutation of
  all 384 experts.

The calibration is pinned by hash: `DSV41F_EXPERT_PLACEMENT_SHA256` makes a run
fail loudly if a different file is used (`check_expected_hash`), and the manifest
records the sha256 of the calibration a run actually used.

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
algorithm/protocol combination sits at ~26-37 us eager and ~19-28 us graphed,
which is launch latency rather than transport, so no NCCL algorithm choice helps.

**vLLM's custom allreduce is capturable and 1.3-1.6x faster than NCCL under graph
replay.** `--custom` attaches `CustomAllreduce` with symmetric memory and times it
against NCCL per shape, eager and graphed, with a numerical check and per-rank
minima (`results/collectives-custom-graphed.json`).

| Shape | NCCL eager | NCCL graphed | Custom eager | Custom graphed | graphed custom vs graphed NCCL |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode MoE fp32, 20 KB | 28.1 us | 21.6 us | 21.2 us | 13.1 us | 1.65x |
| decode MoE bf16, 10 KB | 26.9 us | 20.4 us | 21.2 us | 13.0 us | 1.57x |
| decode attn fp32, 20 KB | 28.0 us | 21.4 us | 21.4 us | 13.1 us | 1.64x |
| Engram bf16, 4.2 MB | 55.8 us | 48.3 us | 46.1 us | 37.4 us | 1.29x |
| prefill / indexer, >= 16.8 MB | 110-774 us | 102-766 us | unsupported | unsupported | — |

An earlier pass reported that custom allreduce "cannot be captured" and does not
win under graph replay. Both were artifacts of the harness:

- Capture used plain `torch.cuda.graph()` without vLLM's
  `CustomAllreduce.capture()` context manager, which sets the capturing flag (so
  `custom_all_reduce` takes its registered-buffer branch) and registers the graph
  buffers on exit. Under the supported procedure capture succeeds and the replayed
  reduction matches the float64 reference, so graphed custom execution can be
  measured -- and it is 1.3-1.6x faster than graphed NCCL at these sizes.
- The correctness check ran `all_reduce` in place over the timing loop, so the
  input grew by `world` each iteration and overflowed to inf; `torch.allclose`
  accepts matching infinities, so the check passed vacuously. Timing now runs on a
  constant zero buffer (reduction latency is data-independent at these sizes), so
  nothing accumulates, and correctness is checked on fresh finite inputs against a
  float64 reduction with the summation-error bound `4 * world * eps * sum|x_i|`.
  That bound tolerates summation reordering (a bf16 difference is ~1 ulp) but not
  a wrong reduction.

`custom_all_reduce` returns `None` above its 8 MiB buffer, so the prefill-sized
reductions stay on NCCL; that is a coverage limit, not a failure. This is a
microbenchmark: it shows the decode collective can be made ~1.6x cheaper under
capture, not that the served step gets faster. Switching the engine's backend
needs an end-to-end measurement first.

**The end-to-end integration works, and the earlier "failed at the kernel level"
verdict was wrong about its cause.** All five decode-sized `dist.all_reduce` sites
(embedding, `RowParallelLinear`, Engram, indexer, MoE output) now go through one
helper, `model.all_reduce_`, behind `DSV41F_CUSTOM_AR`; it falls back to NCCL for
any shape `custom_all_reduce` declines, so prefill stays on NCCL without a separate
gate. Two independent bugs had to come out first, and
`inference/check_custom_ar_ipc.py` isolates both in about twenty seconds, without
loading the model:

1. `Failed: Cuda error .../csrc/custom_all_reduce.cuh:164 'invalid argument'` is
   `cudaIpcGetMemHandle`, not the reduction. Registering a captured reduction calls
   it on every buffer the graph touched (`get_graph_buffer_ipc_meta`), and it
   rejects the `cuMemCreate`/`cuMemMap` memory that `expandable_segments:True`
   hands out. `--stage probe` exports a plain allocation, a side-stream allocation
   and a graph-pool allocation: `cudaSuccess` for all three without expandable
   segments, `cudaErrorInvalidValue` for all three with them. The microbenchmark
   never set `expandable_segments`; every harness that loads the model did. That
   difference alone accounts for the failure, and it is an allocator policy, not a
   kernel or a CUDA version.
2. An illegal memory access on the first replay, in the per-layer MoE decode
   graphs. The C++ reserves a RankData slot for every collective it sees while the
   stream is capturing, and only `CustomAllreduce.capture()`'s exit fills that slot
   with peer pointers; those graphs were captured without `capture()`. `--stage
   graph_eager` captures one reduction both ways: without `capture()` the capture
   reports success and the first replay faults on all four ranks; with it, the
   replay is exact. A graph holding only a device-to-device copy also replays
   exactly, so the fault is the unregistered slot, not graph memcpy and not the
   allocator. Every capture that contains a collective now sits inside `capture()`.

The engine does not need expandable segments at the served configuration
(max-active 4, 8192 context, Engram offload on): peak is 93.8 GiB of 143.8 GiB per
GPU, and TTFT and ITL are unchanged. `inference/custom_ar.py` therefore requires
`DSV41F_EXPANDABLE_SEGMENTS=0` alongside `DSV41F_CUSTOM_AR=1` and refuses the
combination instead of letting it abort later.

With both fixed the served decode step is faster. Interleaved, same code and same
allocator, only the collective differs:

| Backend | ITL median ms/step | runs |
| --- | ---: | --- |
| NCCL | 29.09, 28.93 | 2 runs x 3 reps x 128 tokens, concurrency 1 |
| Custom allreduce | 27.62, 27.60 | same |

That is 1.4 ms/step, 4.8%, and the two backends' ranges do not overlap. The
microbenchmark's 13.1 vs 21.6 us understated it, because a barrier-synchronised
microbenchmark cannot see the rank skew a real step pays.

**Correctness is checked at the reduction, not at the tokens.** `DSV41F_AR_CHECK=1`
runs `dist.all_reduce` on a clone at every custom call site and accumulates the
difference, split by whether the call was being captured. Over a graphed and an
eager decode arm: captured call sites differ by at most **1.9e-06**, eager call
sites by at most **0.03125**, which is one bf16 ulp at magnitude 4 (the eager set
is all bf16). The fp32 reductions -- the row-parallel and MoE partials, which are
what the collective exists for -- are in the 1e-06 group. No call had a non-finite
input. The two artifacts that made an earlier version of this check useless are
recorded in the harness: reading the per-call difference tensor after the step
graphs were rebuilt reads freed pool memory and reports 1e38, and comparing during
`capture()`'s warm-up compares against `torch.empty_like`, which vLLM returns there
on purpose.

**Comparing generated tokens across backends does not isolate the collective**, and
should not be read as one. At B=1 the custom arm emits 2181 where NCCL emits 779,
but this model already produces logit differences of 5.22/11.92/28.34 from a
prefill fp32 reordering -- the `atomic_add` that `DSV41F_PF_FIXED_ORDER` exists to
remove -- so a one-ulp change in a collective is expected to move tokens too. That
is a statement about the model's near-tie sensitivity, not an equivalence proof, and
it is why `DSV41F_CUSTOM_AR` stays opt-in rather than becoming the default.

Two environment notes, since they cost most of the time: the `ds41f` env has no
`vllm`, so the harnesses need `PYTHONPATH=/root/glm-testing/vllm-ds41f-src` (its
import chain also needs `pyzmq`, `urllib3` and `requests`, installed). The
`vllm-ds41f` env has vLLM but cannot compile this model's TileLang kernels, so it
cannot run the end-to-end comparison.

Per-rank minima differ by under 2.1 us on every shape (`*_rank_spread_ms`), so
this microbenchmark does not see the rank skew that dominated the real 8K prefill
wait. It bounds rank timing spread under a barrier, not arrival skew.

**FlashInfer is measured, and wins only at decode sizes.** `--flashinfer`
constructs vLLM's `FlashInferAllReduce` on the same gloo group and times
`kAllReduce` against NCCL (`results/collectives-flashinfer.json`). It is timed in
its own run, because constructing it needs vLLM's own `parallel_state`
initialized, which is extra global process state.

| Shape | NCCL eager | FlashInfer | FlashInfer vs NCCL |
| --- | ---: | ---: | ---: |
| decode MoE fp32, 20 KB | 27.5 us | 21.9 us | 1.25x |
| decode MoE bf16, 10 KB | 26.9 us | 21.8 us | 1.24x |
| decode attn fp32, 20 KB | 28.4 us | 22.1 us | 1.28x |
| Engram bf16, 4.2 MB | 55.7 us | 42.8 us | 1.30x |
| prefill attn bf16, 84 MB | 408.8 us | 418.6 us | 0.98x |
| prefill mid fp32, 42 MB | 222.9 us | 220.0 us | 1.01x |
| indexer fp32, 16.8 MB | 110.2 us | 102.2 us | 1.08x |
| prefill attn fp32, 168 MB | 774.2 us | unsupported | — |

Two integration points were needed, both recorded because the earlier pass called
FlashInfer unmeasurable:

- Its import needs the `vllm-ds41f` env's `libstdc++` on `LD_LIBRARY_PATH`; the
  system one lacks `CXXABI_1.3.15`.
- Constructing `FlashInferAllReduce` asserts `distributed environment is not
  initialized` until vLLM's own `parallel_state` is initialized
  (`init_distributed_environment`), which `torch.distributed` alone does not do.

At prefill sizes FlashInfer is within noise of NCCL or slightly slower, and the
168 MB fp32 reduction exceeds its standalone workspace (179 MB available, 336 MB
needed). Like the custom result, this is a microbenchmark and not an end-to-end
gain.

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
- `verify_placement_dispatch.py`: the serving loader applies one pinned placement
  on all four ranks, and the resulting gate/expert mapping checks out on the
  model's real tensors (see [Load-balanced expert
  placement](#load-balanced-expert-placement)).
- `check_hc_mixes.py` and `check_prefill_parity.py`: the deviation of a candidate
  change against the same-configuration noise floor.
- `check_fixed_order_prefill.py`: the atomic prefill reduction is the sole source
  of the prefill nondeterminism, and the fixed-order path removes it
  bit-exactly at 2048 and 8192 (see [Noise floor](#noise-floor)).

What none of that establishes: independent prefill-state equivalence (the whole
prompt, not one position), exact expert-routing agreement **for the shipped
atomic path**, or quality on representative natural-generation prompts.
Random-token prompts are not a quality test, and no held-out task evaluation was
run. Those are the missing evidence the correctness gates in OPTIMIZE.md ask for.

The measurements added in this pass carry their own limits, stated where they
appear and repeated here so they are not read as end-to-end results:

- The [decode step budget](#decode-step-budget) and its pipelined-versus-
synchronized comparison are B=1 at 2K context on the model-only path. They exclude
HTTP, sampling delivery and any batch size above one. The pipelining result only
holds with `DSV41F_PF_FIXED_ORDER=1`; without it the prefill is nondeterministic
and no cross-run token comparison is meaningful.
- The [Marlin comparison](#fp4-expert-moe-against-vllms-marlin-mxfp4-kernels) is
an isolated one-layer harness on one GPU, not the engine's in-graph path, and the
projected 2.6 ms/step saving is an extrapolation from the measured kernel ratio.
Its accuracy column compares against the engine's own W4A8 reference, not against
a float64 ground truth, so it does not rank the two paths by accuracy.
- The [`wo_a` comparison](#attention-output-projection-wo_a-in-fp8-measured-not-adopted)
measures the op in isolation with an unfused activation quantization, which is
more pessimistic than vLLM's fused path. It bounds the shape's headroom, not
vLLM's implementation.

## Engram lookup cost

With `DSV41F_ENGRAM_OFFLOAD=1` the two n-gram tables are CPU-resident, so every
Engram layer runs `_gather_cpu`:
indices go D2H, rows are gathered and dequantized on the CPU, and the result comes
back H2D. The D2H copy is synchronous, so that sequence is exposed in the step
rather than overlapped. `profile_engram.py` measures it directly
(`results/engram-cost.json`, `results/probe-engram-torchops.txt`):

| | |
| --- | --- |
| Decode step, baseline | 27.10 ms |
| Engram calls per step | 2 |
| Engram per step | 0.441 ms |
| Engram per call | 0.220 ms |
| Share of the decode step | 1.63% |

The measurement adds synchronization and timing calls. The measured lookup accounts
for about 1.6% of this B=1 decode workload. Eliminating that measured cost alone
would have a small effect; it does not explain the reported gap to vLLM. This is
not a comparison of offload enabled versus disabled, and does not establish the
cost for prefill or larger batches.

### What the 0.22 ms/call is made of, and what was recovered

The first attempt at this cost attributed it to dispatch count -- the chain was
about nineteen tiny torch CPU ops -- and replaced the two fp8 decodes with
256-entry numpy tables. That is a real saving but a small one, and the probe that
split the call into its three pieces (`probe_engram_steps.py`,
`results/probe-engram-steps.txt`) shows why:

| Piece | per call |
| --- | ---: |
| D2H of the indices (192 bytes) | 29.9 us |
| CPU gather and dequantize (24 rows x 256 values) | 68.3 us |
| H2D of the gathered rows (12 KiB) | 62.0 us |
| **total** | **160 us** |

Both transfers are trivial in bytes, so the first two numbers are CUDA API and
stream-sync latency, and the third is the scattered read of 24 rows out of a 23 GiB
pinned table. Dispatch count was never the bulk of it. With the tables in place the
same probe measures 0.323 ms/step and 0.162 ms/call, against 0.441 and 0.220 with the
torch ops.

The table lookup is nonetheless kept, because it is exact and free:

| Change | ms/step | Decode | Tokens |
| --- | ---: | ---: | --- |
| fp8 and ue8m0 decodes as numpy tables | +0.165, +0.097 | +0.59%, +0.35% | identical |

Two interleaved A/B runs of three and four repeats (`results/ab-engram-lut.json`,
`results/ab-engram-lut-b.json`): 27.938 → 27.773 and 27.623 → 27.525 ms/step. It is
exact by construction, not by tolerance: torch's fp8 -> fp32 conversion is exact
(e4m3 carries three mantissa bits) and `2**(b - 127)` is an exact power of two for
every exponent byte, so the 256-entry tables reproduce `float()` and `exp2` bit for
bit over all 256 byte values, and the whole gather matches the expression it
replaced on 0 of 6,144 bf16 elements at three seeds. The A/B reports identical
tokens.

One bug worth recording, because it is the kind that only a real run finds: the
tables were built with `torch.arange(256, dtype=torch.uint8)`, which under the
harnesses' `torch.set_default_device("cuda")` lands on the GPU, and the `.numpy()`
that follows then raises. An isolated micro-benchmark did not call
`set_default_device` and passed.

**What is left, and why it was not taken.** Reordering cannot hide the remaining
~0.28 ms/step. The two lookups depend only on the token ids, so both could in
principle be issued at the top of the step, but the D2H is synchronous: issuing it
before the GPU work leaves the CPU chain exposed in front of an idle GPU, and
issuing it after means the D2H drains the segment that was just enqueued. Hiding it
needs the gather on a worker thread, or the gather on the GPU, and the GPU-resident
tables are the whole reason this path exists -- the two tables are 45.8 GiB per rank
against about 50 GiB of headroom. A worker thread would put a collective
(`ParallelEngramEmbedding` ends in an all-reduce) on a second thread for a 1.6%
ceiling, which is not a trade worth making silently.

## Noise floor

The prefill output is not deterministic run to run. The grouped prefill's `_w2_m`
accumulates with `atomic_add`, so fp32 summation order varies; near-tie expert
selections then amplify the difference. `check_prefill_parity.py` measures this
over every prompt position at 2048, with each arm run repeatedly
(`results/prefill-parity-repeats.log`):

| Comparison | Max abs logit diff | Top-1 agreement |
| --- | --- | --- |
| naive vs itself (two independent passes) | 0.0000 | 1.000 |
| optimized vs itself | 8.73 - 11.35 | 0.772 - 0.889 |
| optimized vs naive | 16.29 | 0.750 - 0.870 |

An earlier version of this table compared a single reference run with itself
(`stats(ref, ref)`), which reports 0.0000 by construction. The naive row now comes
from two independent reference passes, so it is a real measurement: the naive path
is bit-deterministic, the optimized path is not, and the optimized path's own
run-to-run spread is the same order as its distance from the naive path. These
measurements therefore do not establish full-prompt equivalence.

`check_fixed_order_prefill.py` isolates the cause and measures the cost of
removing it. It adds an opt-in deterministic accumulation path to the grouped
prefill (`DSV41F_PF_FIXED_ORDER=1`: per-expert `index_add_` into a
`[length, topk, dim]` buffer instead of `atomic_add` into `[length, dim]`) and
runs both paths twice at 2048 and 8192, fingerprinting every block's output and
tracing every MoE gate's routing (`results/fixed-order-prefill-*.json`).

| Length | Comparison | Max abs diff | Top-1 | First differing block | First differing MoE routing |
| --- | --- | ---: | ---: | ---: | ---: |
| 2048 | atomic vs atomic | 11.57 | 0.758 | 0 | 3 |
| 2048 | fixed vs fixed | 0.0000 | 1.000 | none | none |
| 8192 | atomic vs atomic | 6.17 | 0.883 | 0 | 2 |
| 8192 | fixed vs fixed | 0.0000 | 1.000 | none | none |

Every layer is a MoE layer, so block 0 is the first MoE block. The atomic path
diverges between two runs of the same input from that first block, and reaches
different expert routing a few layers later. The fixed-order path is bit-identical
across runs at both lengths, routing included. That identifies the atomic
down-projection accumulation as the sole source of the nondeterminism.

Cost of the deterministic path:

| Length | Atomic | Fixed order | Overhead | Peak memory delta |
| --- | ---: | ---: | ---: | ---: |
| 2048 | 784.6 ms | 795.8 ms | +11.3 ms (+1.4%) | 0.000 GiB |
| 8192 | 2415.2 ms | 2463.9 ms | +48.6 ms (+2.0%) | 0.000 GiB |

The temporary is `length * topk * dim * 4` bytes (0.23 GiB at 2048, 0.94 GiB at
8192), but the measured peak allocation is unchanged: the allocator reuses
headroom, so the deterministic path costs no extra peak memory at these lengths.
It is not wired into the engine; it is a measured option that makes the prefill
reproducible at ~2% prefill cost.

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

That bound is marginal on a single sample, and the decode-fusion pass measured it
both ways: with three decode fusions on, the worst last-position difference was
1.26; with all three off, 1.55. Both runs are marked failed
(`results/decode-fusion-e2e.json`, `results/decode-fusion-e2e-off.json`), and the
fusions do not touch prefill at all — 0 of 66,191,360 prefill logits differ with
them on versus off. The gate's prefill criterion should be read as a noisy
single-position sample, not as a pass/fail signal for a decode change. See
[Decode kernel fusion](#decode-kernel-fusion).

The practical consequence: agreement at one position is insufficient evidence
that two prefill paths agree. `check_placement.py` and `check_hc_mixes.py` also
compare repeated runs of the same configuration, but those controls do not
replace full-model correctness checks.

## Verification

`benchmark_ds41f.py`'s parity gate compares its two arms inside one process. That
catches kernel-level drift, but it cannot catch a change that both arms share, and
its prefill criterion is a single prompt position. Five checks cover those gaps:

- `check_placement.py` compares the same prompt before and after applying a
  placement. Placement moves logits by at most 0.71, while two identical runs
  already differ by at most 0.85. The sampled token is identical.
- `check_decode_parity.py` compares a flag's on and off arms directly, which is
  what "did this change the output" means: the benchmark harness compares its two
  arms, so a kernel that sits in *both* arms passes it. It reports prefill logits
  over every prompt position and per-step decode logits and tokens, and pins
  `DSV41F_PF_FIXED_ORDER=1` so the pre-existing prefill nondeterminism does not
  drown the comparison. Its `--same` control runs the flag off in both arms: on a
  clean harness that reports 0 differing logits and 0 differing tokens, and it is
  how the previous snapshot-restore harness was found to be comparing two
  different decode states rather than two flags.
- `check_rms_norm.py`, `check_moe_swiglu_exact.py` and `check_hc_exact.py` assert
  the fused kernels are bit-identical to their reference expressions at bf16
  output, over random draws plus inputs chosen to land on the clamp and rounding
  boundaries. `probe_rsqrt.py` and `probe_swiglu_steps.py` are the isolation
  probes that located the `tl.exp`/`tl.sqrt`/division differences behind the first
  non-exact attempts.
- `check_hc_exact.py` asserts the fused `hc_pre` and `hc_post` are bit-identical
  to their reference expressions; `check_hc_mixes_decode.py` does the same for the
  fused `hc_mixes` coefficient math at decode (0 of 4,800 fp32 elements differ over
  200 draws at four seeds), and `check_act_quant_cache.py` does the same for the cached
  `act_quant` path (0 differing bytes over 3 reps x 3 shapes).
- `probe_step_host_time.py` and `probe_sync_spin.py` are the control pair behind the
  withdrawn 0.02 ms enqueue figure: the first measures host time inside a decode
  step, the second shows `time.process_time()` tracks wall time during a pure GPU
  wait, which is why the first cannot separate dispatch from blocking.
  `probe_host_dispatch.py` measures one call site in isolation, which is why its
  16.8 µs/call figure survives.
  to the reference expression at bf16 output, including at `s=1`.
- `check_hc_mixes.py` measures a candidate change against the same-configuration
  noise floor at the same length, instead of against the fixed gate. It was used
  to choose between the `hc_mixes` dot modes and to gate the fused path.
- `check_rope_exact.py` and `check_rope_inmodel.py` gate the fused rotary
  embedding. The first compares the fused kernel against the reference expression
  on the five decode shapes; the second runs the real model and compares both
  paths on *every* call it makes, which is what caught a one-ulp product-contraction
  difference that the synthetic check read as zero at its original draw count. The
  in-model comparison is the gate: 0 of 2,133,504 bf16 elements over 3,192 calls.
  `probe_rope_shapes.py` records the call sites, shapes, strides and frequency
  layouts the kernel has to accept.
- `probe_engram_steps.py` splits the offloaded Engram gather into its D2H, CPU and
  H2D pieces, which is what showed that its 0.22 ms/call is transfer latency rather
  than the dispatch count the first attempt assumed. `probe_engram_chain.py` times
  the same pieces in isolation under four-rank host contention.
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

- **Prefill determinism.** Measured, not enabled. The grouped prefill's `_w2_m`
  uses `atomic_add`, which makes the whole prompt nondeterministic: two identical
  runs differ by up to 11.6 on the logits with 0.76 top-1 agreement over 2048
  positions. An opt-in fixed-order accumulation (per-expert `index_add` into a
  `[length, topk, dim]` buffer) is bit-identical across runs at 2048 and 8192, at
  +1.4-2.0% prefill time and no extra peak memory. It is not wired into the
  engine. See [Noise floor](#noise-floor).
- **Marlin kernel comparison (OPTIMIZE.md section 1).** Measured: 2.8× faster
  than the engine's grouped GEMV under graph replay at the real per-rank mix
  (one token, top-6 of 384 global experts, 96 owned locally), 2.3× at 0 active
  local experts up to 3.2× at 6. `bench_marlin_moe.py` loads the shipped shard's
  96 local experts and repacks them with vLLM's own
  `prepare_moe_mxfp4_layer_for_marlin`. Not integrated. The engine quantizes
  activations to FP8 and Marlin does not, so the two are not numerically
  interchangeable and a switch needs its own correctness gate. See [FP4 expert
  MoE against vLLM's Marlin MXFP4 kernels](#fp4-expert-moe-against-vllms-marlin-mxfp4-kernels).
- **Per-step synchronization (OPTIMIZE.md section 3).** Measured at 3.12 ms/step
  (11% of the post-fusion step; 2.8-3.1 ms, 9%, before fusion) in a single-process
  model loop, and shown to be removable there. Pipelining the token read was then
  implemented on real weights and recovered nothing on the served path: the host is
  blocked a median 28.80 ms inside `enqueue` (the blocking NCCL broadcast in
  `BroadcastModel.forward`) against 0.019 ms inside `resolve` (the read), and served
  inter-token latency was 28.77 ms serial against 28.91 ms pipelined. The barrier to
  remove is the per-step broadcast rendezvous, not the read. The split step contract
  (`enqueue`/`resolve`) and `serve/check_pipeline_parity.py` are kept for that work.
- **Kernel launch count (OPTIMIZE.md section 3).** Measured: 6,232 launches per
  decode step, median 1.82 µs, with 46% of kernel time in 5,772 small
  elementwise/copy/reduce kernels. No fusion implemented. This is the largest
  measured structural target in the decode step and it is not what the
  single-kernel replacements in items 2-4 address.
- **`wo_a` in FP8.** Measured and rejected: 1.4-7x slower than the bf16 grouped
  einsum at every decode and prefill shape. See [Decode changes](#decode-changes).
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
- **Custom or symmetric-memory collectives (section 2).** Integrated behind
  `DSV41F_CUSTOM_AR=1 DSV41F_EXPANDABLE_SEGMENTS=0`, measured at 4.8% on the served
  decode step (29.0 -> 27.6 ms), and numerically verified at the call sites to
  1.9e-06 on the fp32 reductions. Left opt-in: it changes generated tokens, and the
  served path is not deterministic enough to prove equivalence (see
  [Collectives](#collectives)). It is unsupported above its 8 MiB buffer, so the
  prefill-sized reductions stay on NCCL. FlashInfer is 1.2-1.3x faster than eager
  NCCL at decode sizes and no faster at prefill sizes; it was not integrated.
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
$PY --nproc-per-node 4 check_prefill_parity.py --length 2048 --reps 3
CUDA_VISIBLE_DEVICES=0 python check_hc_exact.py

# prefill determinism and the cost of removing it (writes results/fixed-order-prefill-*.json)
$PY --nproc-per-node 4 check_fixed_order_prefill.py --length 2048 --reps 2 --timed 5
$PY --nproc-per-node 4 check_fixed_order_prefill.py --length 8192 --reps 2 --timed 3

# serving dispatch of the pinned placement, on every rank
cd ../serve && $PY --nproc-per-node 4 verify_placement_dispatch.py --ckpt /data/ds41f/DSV41F-TP4 && cd ../inference

# decode fusion: interleaved A/B of one flag (or several, comma-separated) inside
# one process. Re-prefills per arm; pin the prefill or the arms differ for reasons
# unrelated to the flag.
DSV41F_ENGRAM_OFFLOAD=1 $PY --nproc-per-node 4 bench_ab.py --flag _RMSNORM_FUSED --repeats 3
DSV41F_ENGRAM_OFFLOAD=1 $PY --nproc-per-node 4 bench_ab.py \
  --flag _RMSNORM_FUSED,_HC_FUSED_DECODE,moe_kernels._MOE_SWIGLU_FUSED --repeats 3

# bitwise on/off comparison, and the control that validates the harness itself
DSV41F_ENGRAM_OFFLOAD=1 $PY --nproc-per-node 4 check_decode_parity.py --flag _RMSNORM_FUSED --steps 64
DSV41F_ENGRAM_OFFLOAD=1 $PY --nproc-per-node 4 check_decode_parity.py --flag _HC_FUSED_DECODE --steps 64 --same

# kernel-level exactness against the reference expression (one GPU)
CUDA_VISIBLE_DEVICES=0 python check_rms_norm.py 200 3
CUDA_VISIBLE_DEVICES=0 python check_moe_swiglu_exact.py
CUDA_VISIBLE_DEVICES=0 python check_hc_exact.py
CUDA_VISIBLE_DEVICES=0 python check_hc_mixes_decode.py
CUDA_VISIBLE_DEVICES=0 python check_act_quant_cache.py
CUDA_VISIBLE_DEVICES=0 python probe_host_dispatch.py
CUDA_VISIBLE_DEVICES=0 python probe_rsqrt.py
CUDA_VISIBLE_DEVICES=0 python probe_swiglu_steps.py

# component profiles
$PY --nproc-per-node 4 profile_prefill.py
$PY --nproc-per-node 4 profile_decode.py
python analyze_decode_trace.py /data/ds41f/decode_trace_rank0.json --steps 20 --top 20
$PY --nproc-per-node 4 profile_prefill_ops.py
$PY --nproc-per-node 4 diag_balance.py
DSV41F_ENGRAM_OFFLOAD=1 $PY --nproc-per-node 4 profile_engram.py

# decode step budget: pipelined vs per-step-synchronized decode, one mode per
# process so both start from identical state. Add DSV41F_PF_FIXED_ORDER=1 for a
# deterministic prefill; without it two runs of the *same* mode already differ,
# so token equality across modes says nothing about pipelining.
for m in pipe sync; do DSV41F_ENGRAM_OFFLOAD=1 $PY --nproc-per-node 4 \
  probe_decode_cpu.py --mode $m; done

# isolated wo_a comparison (bf16 grouped einsum vs FP8 grouped GEMM), one GPU
CUDA_VISIBLE_DEVICES=0 python bench_wo_a.py --iters 200 --output /root/ds41f/results/wo-a.json

# FP4 expert MoE against vLLM's Marlin MXFP4 kernels, one GPU. Needs the
# vllm-ds41f env (for vLLM) and its libstdc++ on LD_LIBRARY_PATH; the ds41f env
# lacks the vLLM dependencies. Loads the rank-0 shard's 96 local experts.
LD_LIBRARY_PATH=/root/miniforge3/envs/vllm-ds41f/lib:$LD_LIBRARY_PATH \
  CUDA_VISIBLE_DEVICES=0 /root/miniforge3/envs/vllm-ds41f/bin/python \
  bench_marlin_moe.py --iters 100 --output /root/ds41f/results/marlin-moe.json

# collectives; --custom measures vLLM's custom allreduce, --flashinfer measures
# FlashInfer. Both need the vllm-ds41f env and its libstdc++ on LD_LIBRARY_PATH
# (the system one lacks CXXABI_1.3.15). Run them separately: --flashinfer needs
# vLLM's parallel_state initialized, which is extra global process state.
LD_LIBRARY_PATH=/root/miniforge3/envs/vllm-ds41f/lib:$LD_LIBRARY_PATH \
  /root/miniforge3/envs/vllm-ds41f/bin/torchrun --nproc-per-node 4 \
  bench_collectives.py --custom --output /root/ds41f/results/collectives-custom-graphed.json

# graphed custom allreduce is measured in the same run with:
#   DSV41F_BENCH_CUSTOM_GRAPH=1

LD_LIBRARY_PATH=/root/miniforge3/envs/vllm-ds41f/lib:$LD_LIBRARY_PATH \
  /root/miniforge3/envs/vllm-ds41f/bin/torchrun --nproc-per-node 4 \
  bench_collectives.py --custom --flashinfer --output /root/ds41f/results/collectives-flashinfer.json

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

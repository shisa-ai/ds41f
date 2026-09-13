# DS41F: a serving engine for DeepSeek-V4.1-Flash

DS41F provides request scheduling, per-request state and fixed-batch execution
for the [reference inference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).

## Status

As of September 13, 2026:

- The reference backend and the external four-GPU HTTP/SSE server are implemented.
- The saved run in [`results/trusted-fused.json`](results/trusted-fused.json) passes the parity gate on all six rows (`top1_agreement` 1.0, `max_logit_diff` 0.0): 35.5 tok/s decode and 3,370 tok/s prefill at 8K context, on four H20-3e GPUs. The previous run, [`results/trusted-shipped.json`](results/trusted-shipped.json), is the pre-fusion baseline at 30.7 tok/s decode and 3,392 tok/s prefill.
- Whole-prompt prompt-processing parity is unresolved. Quality on realistic tasks is unvalidated, and concurrent requests are measured but not explained: throughput falls above concurrency 2.
- The engine is experimental. [Limits](#limits) lists what the measurements do not cover.

Measurements are in the [optimization results](docs/OPTIMIZE-RESULTS.md); the work plan is in [OPTIMIZE.md](docs/OPTIMIZE.md).

## Design

One thread owns the model and executes every step. Requests enter through `api`,
are grouped by `scheduler`, and run through `backend` against per-request state.

```text
HTTP clients ──▶ api (codec, wire shapes)
                    │ bounded request queue
                    ▼
              engine (single execution owner)
                    │ ordered step plans
                    ▼
              backend (reference adapter)  ◀── state (private dense rows)
                    │ per-request token + terminal mailboxes
                    ▼
              api ──▶ clients
```

| Module | Responsibility |
| --- | --- |
| `ds41f.api` | Transport-independent OpenAI-shaped request/response dataclasses |
| `ds41f.engine` | Execution ownership, bounded admission, lifecycle, delivery, cancellation and shutdown |
| `ds41f.scheduler` | Groups similar-length requests into fixed batches and schedules model calls |
| `ds41f.backend` | Reference model adapter, sampling and expert placement; deterministic fake backend for tests |
| `ds41f.state` | Per-request state, context limits and reusable prompt snapshots |
| `ds41f.manifest` | Records the code, configuration, model files and expert placement used for a run |

Behavior covered by tests:

- Each request produces exactly one terminal event, last on its queue.
- Admission rejects overload before allocating state. A stalled token-mailbox reader cancels only its own request.
- Slot generations stop a stale cancellation from affecting a reused slot.
- Rows stay in a batch until it drains. Finished rows still occupy compute, and unequal prompts consume their remaining prompt tails one token per step after the shortest prompt's prefill.
- API threads never call the model.
- Prefix snapshots save state at an exact token position and copy it on restore. Restored requests run alone; the throughput effect under concurrent traffic is unmeasured.

## Install and test

Python 3.12 or newer.

```bash
pip install -e ".[dev]"
python -m pytest -q tests/test_engine_fake.py tests/test_scheduler.py \
  tests/test_state_slots.py tests/test_prefix_store.py
```

- These tests need pytest but no PyTorch, GPU or model weights. The core package has no mandatory runtime dependencies.
- The full suite (`python -m pytest -q`) also covers sampling, placement and manifests, and needs PyTorch. Reference-backend tests use CUDA by default. Loader-wiring checks use a sibling model checkout when one is available.
- The tests cover engine behavior, not trained-model fidelity.

## Serving integration

- This repository is the engine package, not a model or server bundle.
- The optimized model, kernels, conversion and benchmark scripts, prompt encoding and FastAPI/SSE server are in the separate local checkout at `/root/glm-testing/ds41f/` (`inference/`, `encoding/`, `serve/server.py`). Those files and the model weights are not distributed here, so this repository alone cannot reproduce the GPU measurements or start that server.
- The server supports per-request temperature/top-p, streaming, fixed batches and prefix snapshots. It loads the converted TP4 checkpoint with Engram offload and applies calibrated expert placement on every rank before the first forward pass.
- Expert placement is a generated artifact. Set `DSV41F_EXPERT_PLACEMENT` to its path and optionally pin `DSV41F_EXPERT_PLACEMENT_SHA256`, or set `DSV41F_EXPERT_PLACEMENT=none` to disable it. All four workers must apply the same placement during loading.

## Measured configuration

TP=4 splits the model across four GPUs. B=1 means one request at a time.

| Component | Configuration |
| --- | --- |
| GPUs | 4 × NVIDIA H20-3e, TP=4; recorded tuning runs use GPU0–3 |
| Host RAM | 2 TB, with Engram tables in pinned host memory |
| Environment | PyTorch 2.14.0, Triton 3.8.0, TileLang 0.1.8 |
| Checkpoint | DeepSeek-V4.1-Flash converted to MP=4; MXFP4 routed experts and FP8 dense paths |
| Offload | `DSV41F_ENGRAM_OFFLOAD=1` |

Peak GPU and host memory are not recorded. Measure memory with the intended
context length and request count before sizing a deployment.

The [Engram cost measurement](results/engram-cost.json) records two lookups
totaling 0.452 ms per decode step, or 1.40% of a 32.33 ms B=1 step at 2K context.
It measures lookup time only; it does not compare offload enabled against
disabled. The path transfers indices to the CPU, gathers and dequantizes rows
there, and copies values back outside the step graphs. Larger batches and prefill
need their own measurements.

## Performance

One request, four GPUs, text only. Averages of two runs of the optimized
configuration in [`results/trusted-sgbatch.json`](results/trusted-sgbatch.json),
with the pre-fusion run in [`results/trusted-shipped.json`](results/trusted-shipped.json)
for comparison:

| Prompt/context length | Prompt processing | Time to process prompt | Decode throughput | Decode latency |
| --- | ---: | ---: | ---: | ---: |
| 512 tokens | 1,346 tok/s | 0.38 s | 35.79 tok/s | 27.94 ms/token |
| 2,048 tokens | 2,573 tok/s | 0.80 s | 35.65 tok/s | 28.05 ms/token |
| 8,192 tokens | 3,366 tok/s | 2.43 s | 35.67 tok/s | 28.04 ms/token |

The decode kernels are 13.8-14.3% faster than the pre-fusion baseline (32.40-32.56
ms/token to 27.94-28.05 ms/token, or +16.0% to +16.7% throughput). Prompt
processing is unchanged within noise (1,330 to 1,346, 2,580 to 2,573, 3,392 to
3,366 tok/s): the fused kernels are decode-only, and prefill keeps the reference
expressions. See [Decode kernel fusion](docs/OPTIMIZE-RESULTS.md#decode-kernel-fusion).

- Runs follow kernel compilation and warmup, use random-token prompts with predetermined continuation tokens, and measure 32 decode steps. Throughput is the slowest GPU worker.
- The run compares only the last prompt position; see [Limits](#limits).
- HTTP queueing, backend sampling and token delivery are excluded, so these are model-loop numbers. Served latency is measured in [Served latency](#served-latency).
- Results use optimized GPU kernels and an expert-placement file calibrated on random token IDs. The file is not committed, so a deployment needs its own calibration on realistic traffic. The fused `hc_mixes` coefficient projection is enabled at decode and disabled at prefill.
- Earlier runs and before/after comparisons are in the [detailed results](docs/OPTIMIZE-RESULTS.md).

Serving several rows at once is now a fast path rather than a fallback. Same
harness, 2K prompt, 32 decode steps, worst rank, in
[`results/batch-decode-final.json`](results/batch-decode-final.json):

| Batch | Decode latency | Aggregate throughput | Before |
| ---: | ---: | ---: | ---: |
| 1 | 24.72 ms/step | 40.5 tok/s | 24.68 ms/step |
| 2 | 50.39 ms/step | 39.7 tok/s | 158.52 ms/step |
| 4 | 61.32 ms/step | 65.2 tok/s | 159.19 ms/step |
| 8 | 82.34 ms/step | 97.2 tok/s | 159.77 ms/step |

The "before" column is the same harness with the whole-step graph restricted to a
single row, which is how the engine ran until now. Two sequential single-row steps
cost 49.4 ms, so a batch of two is within 2% of running the rows one after the
other and batches of four and eight win clearly. See
[Batched decode](docs/OPTIMIZE-RESULTS.md#batched-decode).

### Served latency

Through the HTTP/SSE server (`serve/bench_serving.py`), one request, greedy, 61-token prompt, with the server recording every backend step:

| Measurement | Served | Model loop, same context |
| --- | ---: | ---: |
| Time to first token | 181 ms | — |
| Model step, batch 1 | 28.48 ms | 26.96 ms |
| Client inter-token latency | 28.78 ms | 26.96 ms |

The serving path adds **1.5 ms per step** over the model loop at the same context: the rank-0-to-rank-3 broadcast plus the backend's per-row sampling and token write-back. A further 0.3 ms to the client is HTTP, SSE framing and detokenization.

Prompt processing as a client sees it is a **1.25 s TTFT for a 3,646-token prompt** (2,922 tok/s). The model loop measures 2,573 tok/s at 2,048 tokens and 3,366 tok/s at 8,192, so ~3,650 tokens lands near 2,900 tok/s. The served path runs at the model-loop rate.

Concurrency works, and it is worth reading the [step trace](docs/OPTIMIZE-RESULTS.md#concurrency-what-the-scheduler-actually-executed) rather than the client numbers:

| Concurrency | Aggregate | Client ITL | Worst TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 34.75 tok/s | 28.78 ms | 0.18 s |
| 2 | 33.04 tok/s | 28.60 ms | 3.12 s |
| 4 | 39.07 tok/s | 61.84 ms | 3.54 s |

At concurrency 2 the two requests ran as two separate single-row cohorts in two of three runs, so the unchanged latency is concurrency 1 twice, not a batched step. The three concurrency-4 runs formed cohorts of 3, 2, 2, 1 and 3 rows and never admitted all four together; the requests left out waited a full generation, which is the 3.1-7.4 s TTFT. A cohort is whatever is in the wait queue when the engine asks; there is no admission window.

The batched step itself was slow for two reasons, both fixed. A batch of two cost 158.5 ms/step because the whole-step graph only ran at input shape `(1, 1)`, then 67.7 ms because the batched MoE took the grouped prefill path, whose tiles are 64 rows tall at any token count. It now costs 50.4 ms, and batch 4 and 8 cost 61.3 and 82.3 ms. See [Batched decode](docs/OPTIMIZE-RESULTS.md#batched-decode).

### Limits

- **Whole-prompt parity is unresolved.** The trusted run compares only the last prompt position, where the two paths agree. Over the whole prompt they differ by 15.7–16.1 on the logits with 76.8–81.2% top-1 agreement, and repeated runs of the same configuration vary by a similar amount: the grouped prefill's `atomic_add` reduction is nondeterministic. Neither result establishes whole-prompt correctness.
- **Decode agreement holds only for a shared starting state.** It does not show that prompt processing builds equivalent state.
- **Concurrency is measured and works, with a cohorting gap.** Aggregate throughput at concurrency 4 went from 15.03 to 39.07 tok/s once batched decode stopped falling off the whole-step graph and off the prefill MoE path. What remains is scheduling, not compute: a cohort is whatever is in the wait queue when the engine asks, so a request that arrives a moment late waits a full generation (3.1-7.4 s TTFT at concurrency 4, where no run admitted all four requests together). See [Served latency](docs/OPTIMIZE-RESULTS.md#served-latency).
- **No matched vLLM comparison.** Local vLLM numbers use different workloads and methods, so the speedup is uncontrolled.
- **The custom-allreduce path runs, opt-in, and changes generated tokens.** It is 1.6x faster than graphed NCCL in isolation, and routed through the engine it takes the served decode step from 29.0 to 27.6 ms (4.8%). Enable it with `DSV41F_CUSTOM_AR=1 DSV41F_EXPANDABLE_SEGMENTS=0`; the second variable is required, because registering a captured reduction cannot export expandable-segment memory. It is not the default: it moves generated tokens at 2 of 3 tested batch sizes, and this model is near-tie sensitive enough that a one-ulp fp32 reordering shows up in the argmax. The reductions themselves agree with NCCL to 1.9e-06. See [Collectives](docs/OPTIMIZE-RESULTS.md#collectives).

## Next steps

The work order and acceptance criteria are in [OPTIMIZE.md](docs/OPTIMIZE.md).

1. Validate the state and expert choices produced during prompt processing, test quality on unseen realistic tasks, and measure complete served requests.
2. Compare ordinary token-by-token generation against vLLM on the same workload, measuring every GPU with 1, 2, 4 and 8 concurrent requests.
3. Compare production GPU kernels, including Marlin, and alternative GPU communication libraries on the current workload.
4. Run the proposed B=1 DSpark prototype. The [feasibility analysis](docs/OPTIMIZE.md#dspark-feasibility-assessment--september-13-2026) identifies the existing draft model and the work needed to verify its proposed tokens and retain correct state after accepting or rejecting them. DSpark is not integrated into this engine's generation loop.
5. Improve batching so new requests can join running batches when measurements justify the work, and verify that request state stays separate.

## License

Apache 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The model definition,
kernels and prompt encoding come from
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
(MIT). The request and step structure follows
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (MIT). Model weights
remain subject to their original license.

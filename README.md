# DS41F — a minimal serving engine for DeepSeek-V4.1-Flash

DS41F adapts the [reference inference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
into a serving engine with a small, testable core. It manages the model's
attention caches and Engram memory, keeps each request's state separate, and runs
requests in fixed batches.

**Status — September 13, 2026:** the reference backend and external four-GPU HTTP/SSE
server are implemented. The latest saved model measurements report **30.62 tok/s
decode at 2K context** and **3,270 tok/s prefill at 8K** on four H20-3e GPUs.
That run failed correctness checks; these are timings, not a validated release benchmark.
Correctness after prompt processing, quality on realistic tasks and performance
with concurrent requests still need validation. This is an experimental engine.

See [optimization results](docs/OPTIMIZE-RESULTS.md) for measurements and
[the reviewed optimization plan](docs/OPTIMIZE.md) for evidence limits and next steps.

## Design

Five small responsibilities, one thread owning the model:

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
| `ds41f.state` | Keeps request state separate, limits context use and stores reusable prompt snapshots |
| `ds41f.manifest` | Records the code, configuration, model files and expert placement used for a run |

Tested engine semantics include:

- **One terminal event** per request, always last on its queue.
- **Bounded queues:** admission rejects overload before state allocation;
  a stalled token-mailbox reader cancels its own request.
- **Slot generations** prevent stale cancellation from affecting a reused slot.
- **Fixed batches:** requests enter together and rows remain until the batch
  drains. Finished rows still occupy compute; unequal prompts consume their
  remaining prompt tails one token per step after the shortest prompt's prefill.
- **Single execution owner:** API threads never call the model directly.
- **Prefix snapshots:** saves state at an exact token position and copies it on
  restore. Restored requests run alone; the throughput benefit under concurrent
  traffic still needs measurement.

## Quick start and tests

Python 3.12 or newer is required. Install the engine and run its core tests:

```bash
pip install -e ".[dev]"
python -m pytest -q tests/test_engine_fake.py tests/test_scheduler.py \
  tests/test_state_slots.py tests/test_prefix_store.py
```

These tests need pytest but no PyTorch, GPU or model weights. The core package has
no mandatory runtime dependencies. The complete suite additionally exercises
sampling, placement and manifests and requires PyTorch; reference-backend tests
use CUDA by default. Loader-wiring checks use a sibling model checkout when it is
available. In the configured development environment, run:

```bash
python -m pytest -q
```

Passing unit tests establishes engine behavior, not trained-model fidelity.

## Serving integration

This repository contains the engine package, **not a standalone model/server
bundle**. The optimized model, kernels, conversion/benchmark scripts, prompt
encoding and FastAPI/SSE server live in the separate local checkout at
`/root/glm-testing/ds41f/` (`inference/`, `encoding/` and `serve/server.py`). Those
files and model weights are not distributed here; cloning this repository alone
is insufficient to reproduce the GPU measurements or start that server.

The external server supports per-request temperature/top-p, streaming,
fixed batches and prefix snapshots. It loads the converted TP4 checkpoint with
Engram offload and applies calibrated expert placement on every rank before the
first forward. The calibration is a generated artifact: configure
`DSV41F_EXPERT_PLACEMENT` explicitly and optionally pin
`DSV41F_EXPERT_PLACEMENT_SHA256`; `DSV41F_EXPERT_PLACEMENT=none` disables it.
All four GPU workers must apply the same placement together during loading.

## Measured configuration and memory

TP=4 means the model is split across four GPUs. B=1 means one request at a time.

| Component | Configuration |
| --- | --- |
| GPUs | 4 × NVIDIA H20-3e, TP=4; recorded tuning runs use GPU0–3 |
| Host RAM | 2 TB, with Engram tables in pinned host memory |
| Reference environment | PyTorch 2.14.0, Triton 3.8.0, TileLang 0.1.8 |
| Checkpoint | DeepSeek-V4.1-Flash converted to MP=4; MXFP4 routed experts and FP8 dense paths |
| Offload | `DSV41F_ENGRAM_OFFLOAD=1` |

A current peak GPU/host memory measurement is not recorded. Measure memory with
the intended context length and request count before sizing a deployment.

The current [Engram cost measurement](results/engram-cost.json) records two
lookups totaling **0.452 ms per decode step**, or **1.40%** of a 32.33 ms B=1 step
at 2K context. This measures lookup time; it does not compare offload enabled versus disabled.
The path still transfers indices to CPU, gathers/dequantizes rows there and copies
values back outside the step graphs. Larger batches and prefill need their own
measurements.

## Performance results (one request, four GPUs, text)

The latest saved [full-prompt diagnostic run](results/fulllogits-hcmixes-off.json)
contains these timings, averaged over two runs of the optimized configuration:

| Prompt/context length | Prompt processing | Time to process prompt | Decode throughput | Decode latency |
| --- | ---: | ---: | ---: | ---: |
| 2,048 tokens | **2,560 tok/s** | 0.80 s | **30.62 tok/s** | 32.66 ms/token |
| 8,192 tokens | **3,270 tok/s** | 2.51 s | **29.29 tok/s** | 34.20 ms/token |

The model runs after kernel compilation and warmup, with random-token prompts
and predetermined continuation tokens. Each comparison measures 32 decode steps
and reports the slowest GPU worker. Full-prompt prediction scores are compared
in a separate, untimed pass. HTTP queueing, backend sampling and serving token
delivery are excluded. **Current end-to-end serving throughput is not measured.**

These results use optimized GPU kernels and a generated expert-placement file
calibrated on random token IDs. The file is not committed; a deployment needs
calibration and validation on realistic traffic. Fused `hc_mixes` is disabled.
Earlier runs and before/after comparisons remain in the
[detailed results](docs/OPTIMIZE-RESULTS.md).

### Correctness and comparison limits

- **Decode matched from the same starting state:** both compared paths produced
  identical prediction scores in the saved run. This does not establish that
  they build equivalent state during prompt processing, or validate code changes
  used by both paths.
- **Prompt processing still has unresolved numerical differences.** The
  [run used above](results/fulllogits-hcmixes-off.json) is marked failed:
  it records maximum prediction-score differences of about 15.7–16.1, with the
  highest-scoring token matching at 76.8–81.2% of prompt positions across its
  2K/8K comparisons. Repeated runs of the same configuration also
  vary, as documented in the results writeup. That variation does not establish
  correctness or justify ignoring a failed check.
- **Multiple-request performance still needs testing.** The optimized decode
  graph handles one token from one request at a time; these gains do not establish
  performance for batches of 2, 4 or 8 requests.
- **No matched vLLM comparison yet.** Historical local vLLM results use different
  workloads and measurement methods. Compare ordinary generation first, then DSpark
  separately; do not quote a controlled speedup from those unmatched numbers.

## Next steps

The authoritative work order and acceptance criteria are in
[OPTIMIZE.md](docs/OPTIMIZE.md):

1. Validate the state and expert choices produced during prompt processing,
   test quality on unseen realistic tasks, and measure complete served requests.
2. Compare ordinary token-by-token generation against vLLM using the same
   workload; measure every GPU with 1, 2, 4 and 8 concurrent requests.
3. Compare production GPU kernels, including Marlin, and alternative GPU
   communication libraries using the current workload.
4. Run the proposed B=1 DSpark prototype. The
   [feasibility analysis](docs/OPTIMIZE.md#dspark-feasibility-assessment--september-13-2026)
   identifies the existing draft model and the work needed to verify its proposed
   tokens and retain the correct state after accepting or rejecting them.
   DSpark is not integrated into this engine's generation loop.
5. Improve batching and allow new requests to join running batches when
   measurements justify the work; verify that request state stays separate.

## License and acknowledgements

Apache 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The model definition,
kernels and prompt encoding this project serves originate from
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
(MIT); the engine's request/step structure is informed by
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (MIT). Model weights remain
subject to their original license.

# DS41F: a serving engine for DeepSeek-V4.1-Flash

DS41F provides request scheduling, per-request state and fixed-batch execution
for the [reference inference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).

## Status

As of September 13, 2026:

- The reference backend and the external four-GPU HTTP/SSE server are implemented.
- The saved run in [`results/trusted-shipped.json`](results/trusted-shipped.json) passes the parity gate: 30.81 tok/s decode at 2K context and 3,392 tok/s prefill at 8K, on four H20-3e GPUs.
- Whole-prompt prompt-processing parity is unresolved. Quality on realistic tasks and performance with concurrent requests are unvalidated.
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
configuration in [`results/trusted-shipped.json`](results/trusted-shipped.json):

| Prompt/context length | Prompt processing | Time to process prompt | Decode throughput | Decode latency |
| --- | ---: | ---: | ---: | ---: |
| 512 tokens | 1,330 tok/s | 0.38 s | 30.86 tok/s | 32.40 ms/token |
| 2,048 tokens | 2,580 tok/s | 0.79 s | 30.81 tok/s | 32.46 ms/token |
| 8,192 tokens | 3,392 tok/s | 2.42 s | 30.71 tok/s | 32.56 ms/token |

- Runs follow kernel compilation and warmup, use random-token prompts with predetermined continuation tokens, and measure 32 decode steps. Reported throughput is the slowest GPU worker.
- The run compares only the last prompt position. A separate full-prompt diagnostic is covered under [Limits](#limits).
- HTTP queueing, backend sampling and serving token delivery are excluded. End-to-end serving throughput is not measured.
- These results use optimized GPU kernels and a generated expert-placement file calibrated on random token IDs. The file is not committed, so a deployment needs its own calibration and validation on realistic traffic. Fused `hc_mixes` is disabled.
- Earlier runs and before/after comparisons are in the [detailed results](docs/OPTIMIZE-RESULTS.md).

### Limits

- Decode agreement holds only for the same starting state. Both compared paths produced identical prediction scores in the saved run, but that does not show that they build equivalent state during prompt processing, or validate code used by both paths.
- Whole-prompt prompt processing is not established. The trusted run compares only the last prompt position, where the paths agree (top-1 1.0, maximum score difference 0.68–1.00 against a 1.25 gate). The [full-prompt diagnostic run](results/fulllogits-hcmixes-off.json) recorded maximum score differences of about 15.7–16.1 with 76.8–81.2% top-1 agreement, and repeated runs of the same configuration vary by a similar amount because the grouped prefill's `atomic_add` reduction is nondeterministic. That run was marked failed by an earlier gate that included the whole-prompt numbers; the current gate checks the last position only. Neither result establishes whole-prompt correctness.
- Multiple-request performance is untested. The optimized decode graph handles one token from one request; the measurements do not cover batches of 2, 4 or 8.
- There is no matched vLLM comparison. The local vLLM results use different workloads and measurement methods, so they do not give a controlled speedup.

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

# DS41F — a minimal serving engine for DeepSeek-V4.1-Flash

DS41F turns the [reference inference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
released with DeepSeek-V4.1-Flash into a serving engine with a small, testable core. 

DS 4.1 Flash's  shared-latent KV, compressed sparse attention, SWA rings and Engram conditional memory don't fit conventional paged
KV abstractions, so the backend is a thin adapter over the reference model code, and the control plane is built for its actual constraints.

Status: engine skeleton with a deterministic fake backend and full control-flow test suite. The real backend adapter (torchrun TP=4) is in development; see roadmap below.

## Design

Five small responsibilities, one thread owning the model:

```text
HTTP clients ──▶ api (codec, wire shapes)
                    │ bounded request queue
                    ▼
              engine (single execution owner)
                    │ ordered step plans
                    ▼
              backend (DS41F reference adapter)  ◀── state (private dense rows)
                    │ per-request token + terminal mailboxes
                    ▼
              api ──▶ clients
```

| Module | Responsibility |
| --- | --- |
| `ds41f.api` | OpenAI-shaped request/response models; codec hooks for the V4.1 prompt format |
| `ds41f.engine` | Sole execution owner: bounded admission, lifecycle, delivery, cancellation, shutdown |
| `ds41f.scheduler` | Pure host metadata: static cohorts, length bucketing, step plans. No torch. |
| `ds41f.backend` | Ordered plan execution against the model; rank-0 sampling. `FakeBackend` for tests |
| `ds41f.state` | Private dense state rows with generation counters and context budgets |

Key semantics (each covered by tests):

- **Exactly one terminal event** per request, always last on its queue.
- **Bounded everything**: waiting queue rejects overload *before* allocating state;
  token mailboxes are bounded and a stalled reader cancels only itself.
- **Slot generations** prevent a stale cancellation from touching a slot's next occupant.
- **Static cohorts**: rows are admitted together and stay fixed until the cohort drains.
  Continuous mid-flight admission is a separate milestone (per-row positions across
  SWA rings, compressor tails, Engram history).
- **Single execution owner**: all backend calls happen on one engine thread; API
  layers never touch the model.

## Quick start

```bash
pip install -e ".[dev]"
pytest
```

The test suite is dependency-free (stdlib only) and runs the full engine against a
deterministic `FakeBackend` — no GPU or weights required.

## Hardware setup this project is built against

| Component | Value |
| --- | --- |
| GPUs | 4 × NVIDIA H20-3e (143.7 GB each), tensor-parallel 4 |
| Host RAM | 2 TB (Engram tables pinned in system memory) |
| Software | CUDA 13 stack, torch 2.14, tilelang 0.1.8, triton 3.8 |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash`, converted MP=4, experts FP4 |

## Memory budget (measured on the hardware above)

With the Engram hash tables offloaded to pinned host memory
(`DSV41F_ENGRAM_OFFLOAD=1` in the reference adapter):

| Quantity | Value |
| --- | --- |
| VRAM per GPU (inference) | **76.3 GiB** (79.7 GiB peak at 8K context) |
| Host RAM, pinned Engram tables | **202.8 GB** total (47.2 GiB per rank, both tables) |
| All-GPU alternative (engrams resident) | ~123.5 GiB peak per GPU — fits, but leaves little headroom |
| Decode cost of the offload | ~2% (270 vs 264 ms/token at B=1) |

The Engram is 196B parameters of n-gram-hash-addressed fp8 lookup tables; only tiny
per-token row gathers cross PCIe, which is why offloading it is essentially free.

## Decode profile (B=1, 2K prompt, greedy, TP=4)

Where a ~330 ms/token decode step goes (kernel time = 272 ms; the rest is host gaps):

| Component | ms/step | Notes |
| --- | ---: | --- |
| NCCL all_reduce | 138 | 91 reductions/step — real data dependencies (1 embed + 2 engram + 40 attn out-proj + 40 MoE combine + 8 indexer), not removable by fusion |
| FP4 expert GEMMs | 99 | ~185 GEMMs/step on rank 0; 536 µs each — small-M inefficiency, the main optimization target |
| FP8 dense GEMMs | 18 | reasonable |
| Sparse attention | 3 | fine |

Implications: the NCCL time partly reflects rank skew from the uneven expert
distribution; a grouped small-M MoE path is the highest-value optimization.

## Roadmap

1. **Stage A** — fixed cohorts served over an OpenAI-compatible API (this skeleton +
   the reference backend adapter + HTTP/SSE).
2. **Stage B** — position-aware dense slots for mid-flight admission (agent
   workloads).
3. **Backend optimizations** — grouped small-M MoE, collective/skew work, chunked
   prefill; prefix snapshots only after suffix prefill is efficient.

## License and acknowledgements

Apache 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The model definition,
kernels and prompt encoding this project serves originate from
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
(MIT); the engine's request/step structure is informed by
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (MIT). Model weights remain
subject to their original license.

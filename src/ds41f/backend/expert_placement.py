"""Load-balanced routed-expert placement across TP ranks.

Why: every MoE layer ends in a fp32 all_reduce, so the layer takes as long as the
slowest rank's expert work. With contiguous ownership (rank r owns experts
[96r, 96r+96)) the router's skewed expert popularity leaves the busiest rank at
1.48x the mean load (worst layer 2.17x). On an 8K prefill that is a 10-16 ms
all_reduce wait per layer.

How: permute the expert axis of the gate and the expert weights *together*. The
model computes the same function -- the chosen experts and their weights are just
relabelled -- so only which rank computes which expert changes. Greedy assignment
of the calibration loads to ranks brings the busiest rank to 1.000x of ideal and
the measured MoE all_reduce wait from 0.491 s to 0.100 s on an 8K prefill.

The permutation is per layer, because expert popularity differs per layer and the
all_reduce is per layer.

Placement file::

    torch.save({"new_to_old": int64[n_layers, n_experts], ...})

where ``new_to_old[L, j]`` is the original expert index that becomes global expert
j. Produce one with ``plan_experts.py`` in the model repository, which calibrates
per-layer per-expert routing counts from the gate.

Call :func:`maybe_apply` on **every** rank, once, after the model is built and
loaded and before the first forward. It contains a collective, so a rank that
skips it leaves the others blocked.
"""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import torch
import torch.distributed as dist

__all__ = [
    "greedy_placement",
    "new_to_old_from_assign",
    "permute_gate",
    "exchange_plan",
    "apply",
    "maybe_apply",
    "report",
    "fingerprint",
    "expected_sha256",
    "check_expected_hash",
]


def greedy_placement(loads: torch.Tensor, n_ranks: int, n_local: int | None = None) -> torch.Tensor:
    """Assign experts to ranks: exactly n_local each, greedily minimising the busiest.

    Longest-processing-time first with a cardinality cap. A plain min-load greedy
    would not give equal counts (a rank that drew the heaviest experts is avoided
    until the others catch up), and the kernels need a fixed per-rank expert count.

    loads: [n_experts] non-negative. Returns assign[e] = rank index.
    """
    n = loads.numel()
    n_local = n_local if n_local is not None else n // n_ranks
    assign = torch.empty(n, dtype=torch.int64, device=loads.device)
    bins = [0.0] * n_ranks
    counts = [0] * n_ranks
    for e in torch.argsort(loads, descending=True).tolist():
        cands = [i for i in range(n_ranks) if counts[i] < n_local]
        r = min(cands, key=lambda i: (bins[i], counts[i]))
        assign[e] = r
        bins[r] += float(loads[e])
        counts[r] += 1
    return assign


def new_to_old_from_assign(assign: torch.Tensor, n_ranks: int) -> torch.Tensor:
    """assign[e] = rank -> new_to_old[r*n_local + j] = e (heaviest expert first)."""
    n_local = assign.numel() // n_ranks
    n2o = torch.empty(assign.numel(), dtype=torch.int64, device=assign.device)
    for r in range(n_ranks):
        idx = (assign == r).nonzero().flatten()
        if idx.numel() != n_local:
            raise ValueError(f"rank {r} got {idx.numel()} experts, expected {n_local}")
        n2o[r * n_local : (r + 1) * n_local] = idx
    return n2o


def permute_gate(gate, n2o: torch.Tensor) -> None:
    """Reorder a gate's per-expert parameters in place so output index j matches the
    expert that now occupies slot j. The gate is replicated on every rank, so every
    rank applies the same row permutation."""
    gate.weight.data = gate.weight.data[n2o].contiguous()
    if getattr(gate, "bias", None) is not None:
        gate.bias.data = gate.bias.data[n2o].contiguous()
    if getattr(gate, "bias_vl", None) is not None:
        gate.bias_vl.data = gate.bias_vl.data[n2o].contiguous()


def _expert_tensors(ex):
    """Fixed-order tensors of one fp4 Expert."""
    return [ex.w1.weight, ex.w1.scale, ex.w2.weight, ex.w2.scale, ex.w3.weight, ex.w3.scale]


def _bytes(t: torch.Tensor) -> torch.Tensor:
    """Flat 1-byte view of an expert tensor.

    The packing below advances receive buffers by ``t.numel()``, which only equals a
    byte count when every tensor is one byte per element. The reference model stores
    fp4 weights as float4_e2m1fn_x2 (two values per byte) and their scales as
    float8_e8m0fnu, so this holds. Check it rather than corrupt the layout silently
    if that ever changes.
    """
    if t.element_size() != 1:
        raise TypeError(
            f"expert tensor dtype {t.dtype} is {t.element_size()} bytes per element; "
            "_pack/_unpack_into require 1-byte dtypes so a numel is a byte count"
        )
    return t.view(torch.uint8).reshape(-1)


def _pack(ex) -> torch.Tensor:
    return torch.cat([_bytes(t) for t in _expert_tensors(ex)])


def _expert_sha256(ex) -> str:
    """Content hash of one expert's packed bytes (audit only; forces a D2H copy)."""
    h = hashlib.sha256()
    for t in _expert_tensors(ex):
        h.update(_bytes(t).detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _unpack_into(buf: torch.Tensor, off: int, ex) -> int:
    for t in _expert_tensors(ex):
        b = _bytes(t)
        b.copy_(buf[off : off + b.numel()])
        off += b.numel()
    return off


@torch.no_grad()
def exchange_plan(n2o: torch.Tensor, rank: int, world: int):
    """Per-rank expert exchange indices, shared by every rank.

    Returns ``(mine, dest, want, src)``:

    - ``mine``: the global slots this rank holds before the permutation.
    - ``dest[i]``: the rank that wants the old expert currently in slot ``mine[i]``.
    - ``want[j]``: the old expert that will occupy this rank's local slot ``j``.
    - ``src[j]``: the rank that currently holds ``want[j]``.

    Split out from :func:`apply` because this arithmetic is the part that has to
    agree across ranks, and it is worth testing without a process group.

    The receive loop writes source ``s``'s payloads into local slots in increasing
    ``j`` order, and rank ``s`` sends them in increasing old-slot order, so the two
    orders line up only because ``new_to_old_from_assign`` emits each rank's range
    in ascending old-expert order. ``test_exchange_plan_places_each_expert_in_its_new_slot``
    pins that.
    """
    n_experts = n2o.numel()
    n_local = n_experts // world
    o2n = torch.empty_like(n2o)
    o2n[n2o] = torch.arange(n_experts, device=n2o.device)
    mine = torch.arange(rank * n_local, (rank + 1) * n_local, device=n2o.device)
    dest = o2n[mine] // n_local  # which rank wants each expert we hold
    want = n2o[rank * n_local : (rank + 1) * n_local]  # old ids we will hold
    src = want // n_local
    return mine, dest, want, src


def expected_sha256() -> str | None:
    """The placement hash a run is pinned to, from ``DSV41F_EXPERT_PLACEMENT_SHA256``.

    Unset means "accept whatever calibration is named". Set means the calibration is
    part of the run's identity: a different file must fail loudly rather than silently
    relabel experts and change which rank computes what.
    """
    value = os.environ.get("DSV41F_EXPERT_PLACEMENT_SHA256", "").strip().lower()
    return value or None


def fingerprint(path: str) -> dict:
    """Identity of a placement artifact: content hash, shape and permutation validity.

    Pure file inspection, no model and no process group, so a manifest can pin the
    calibration and a loader can log exactly which file it used.
    """
    raw = Path(path).read_bytes()
    info: dict = {
        "path": os.path.abspath(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "keys": [],
        "layers": None,
        "experts": None,
        "valid_permutation": False,
        "invalid_layers": [],
    }
    blob = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(blob, dict):
        return info
    info["keys"] = sorted(str(k) for k in blob.keys())
    n2o = blob.get("new_to_old")
    if n2o is None or not torch.is_tensor(n2o) or n2o.dim() != 2:
        return info
    n2o = n2o.long()
    info["layers"], info["experts"] = int(n2o.shape[0]), int(n2o.shape[1])
    # device=n2o.device, not the default device: the serving loader sets the default
    # device to CUDA, and this tensor is loaded to CPU, so a default-device arange
    # makes the comparison raise instead of validating.
    want = torch.arange(n2o.shape[1], device=n2o.device)
    invalid = [
        layer
        for layer in range(n2o.shape[0])
        if not torch.equal(n2o[layer].sort().values, want)
    ]
    info["invalid_layers"] = invalid
    info["valid_permutation"] = not invalid
    return info


def check_expected_hash(info: dict, expected: str | None) -> None:
    """Raise if ``info`` (from :func:`fingerprint`) does not match the pinned hash."""
    if expected is not None and info["sha256"] != expected:
        raise ValueError(
            f"{info['path']}: sha256 {info['sha256']} does not match "
            f"DSV41F_EXPERT_PLACEMENT_SHA256={expected}"
        )


@torch.no_grad()
def apply(
    model,
    placement_path: str,
    rank: int,
    world: int,
    verbose: bool = False,
    audit: bool = False,
) -> dict | None:
    """Permute gates and redistribute experts in place. Call before any forward.

    Collective: every rank in the TP group must call it with the same path.

    With ``audit=True`` it also checks its own work and returns a summary: for each
    MoE layer, whether the gate really is the pre-permutation gate reordered by
    ``new_to_old``, whether every local expert slot holds the bytes of the expert
    that ``exchange_plan`` assigned to it (verified against hashes gathered from
    the source rank), and the global expert ids this rank now holds. The check
    costs a D2H copy and a hash of every expert, so it is off by default.
    """
    info = fingerprint(placement_path)
    check_expected_hash(info, expected_sha256())
    if verbose and rank == 0:
        print(
            f"  [placement] file={info['path']} sha256={info['sha256']} "
            f"layers={info['layers']} experts={info['experts']}",
            flush=True,
        )
    blob = torch.load(placement_path, map_location="cpu", weights_only=True)
    n2o_all = blob["new_to_old"].long()
    dev = torch.cuda.current_device()
    audit_layers: list[dict] | None = [] if audit else None

    # A calibration is per-checkpoint. Applying one from a different checkpoint would
    # silently mislabel experts (or index out of range), so check the shape and that
    # each layer really is a permutation before touching the model.
    if n2o_all.dim() != 2:
        raise ValueError(f"{placement_path}: new_to_old must be 2-D, got {tuple(n2o_all.shape)}")

    for layer_id, block in enumerate(model.layers):
        moe = block.ffn
        if moe is None or not getattr(moe, "experts", None):
            continue
        n_experts = len(moe.experts)
        n_local = n_experts // world
        if layer_id >= n2o_all.shape[0]:
            raise ValueError(
                f"{placement_path}: has {n2o_all.shape[0]} layers but the model's MoE "
                f"layer {layer_id} needs one (calibration is from another checkpoint?)"
            )
        n2o = n2o_all[layer_id].to(dev)
        if n2o.numel() != n_experts or not torch.equal(
            n2o.sort().values, torch.arange(n_experts, device=dev)
        ):
            raise ValueError(
                f"{placement_path}: layer {layer_id} is not a permutation of "
                f"0..{n_experts - 1} (calibration is from another checkpoint?)"
            )

        mine, dest, want, src = exchange_plan(n2o, rank, world)
        if audit:
            gate_pre = moe.gate.weight.detach().clone()
            pre_hashes = {int(e): _expert_sha256(moe.experts[int(e)]) for e in mine.tolist()}
        permute_gate(moe.gate, n2o)
        if audit:
            gate_ok = bool(torch.equal(moe.gate.weight.detach(), gate_pre[n2o]))
        per_expert = _pack(moe.experts[int(mine[0])]).numel()
        send = []
        for d in range(world):
            sel = mine[dest == d]
            if sel.numel() == 0:
                send.append(torch.empty(0, dtype=torch.uint8, device=dev))
            else:
                send.append(torch.cat([_pack(moe.experts[int(e)]) for e in sel.tolist()]))
        recv = [
            torch.empty(int((src == s).sum()) * per_expert, dtype=torch.uint8, device=dev)
            for s in range(world)
        ]
        dist.all_to_all(recv, send)

        for s in range(world):
            buf = recv[s]
            off = 0
            for j in (src == s).nonzero().flatten().tolist():
                off = _unpack_into(buf, off, moe.experts[rank * n_local + j])
        if audit:
            # Every rank publishes the hashes of the experts it held before the
            # exchange; each local slot must now hold the bytes of the expert that
            # exchange_plan assigned to it. This checks the model's actual tensors,
            # not just that apply() was called.
            gathered = [None] * world
            dist.all_gather_object(gathered, pre_hashes)
            merged: dict[int, str] = {}
            for d in gathered:
                merged.update(d)
            exchange_ok = all(
                _expert_sha256(moe.experts[rank * n_local + j]) == merged[int(want[j])]
                for j in range(n_local)
            )
            audit_layers.append(
                {
                    "layer": layer_id,
                    "gate_permuted": gate_ok,
                    "exchange_ok": exchange_ok,
                    "local_experts": [int(x) for x in want.tolist()],
                }
            )
        if verbose and rank == 0:
            print(f"  [placement] layer {layer_id} redistributed")
    if verbose and rank == 0:
        print(f"  [placement] applied {placement_path}")
    if audit:
        return {
            "file": info["path"],
            "sha256": info["sha256"],
            "layers": audit_layers,
            "gate_permuted": all(layer["gate_permuted"] for layer in audit_layers),
            "exchange_ok": all(layer["exchange_ok"] for layer in audit_layers),
        }
    return None


def _rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def maybe_apply(
    model,
    rank: int | None = None,
    world: int | None = None,
    path: str | None = None,
    audit: bool = False,
) -> bool:
    """Apply a balanced placement if one is configured. Returns whether it ran.

    ``DSV41F_EXPERT_PLACEMENT`` names the file; "none" (or "0") disables it. When
    it is unset, a calibrated ``expert_placement.pt`` sitting next to this module
    is used, so a calibrated deployment gets the balanced placement by default and
    an uncalibrated one is unaffected.

    Pass ``path`` to name the file explicitly instead of using that lookup. Callers
    that own a checkpoint should do so: the calibration is derived from the
    checkpoint, so it belongs beside the checkpoint or the model repository, not
    inside the engine package.

    Must be called on every rank together, after the model is loaded and before the
    first forward. ``rank``/``world`` default to the active process group.

    With ``audit=True`` the returned placement summary is stored on
    ``model.placement_audit`` (see :func:`apply`); it is ``None`` when placement did
    not run.
    """
    if rank is None or world is None:
        r, w = _rank_world()
        rank = r if rank is None else rank
        world = w if world is None else world
    if world <= 1:
        model.placement_audit = None
        return False
    if path is None:
        path = os.environ.get("DSV41F_EXPERT_PLACEMENT")
        if path is None:
            default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expert_placement.pt")
            path = default if os.path.exists(default) else None
    if path is not None and path.lower() in ("", "none", "0"):
        path = None
    if not path:
        model.placement_audit = None
        return False
    model.placement_audit = apply(model, path, rank, world, verbose=(rank == 0), audit=audit)
    return True


def report(loads_by_layer: torch.Tensor, world: int) -> dict:
    """loads_by_layer: [n_layers, n_experts]. Compare current vs greedy balance."""
    ideal = loads_by_layer.sum(1).float() / world
    n_local = loads_by_layer.shape[1] // world
    cur = loads_by_layer.reshape(loads_by_layer.shape[0], world, n_local).sum(2).float()
    n2o = torch.stack(
        [
            new_to_old_from_assign(greedy_placement(loads_by_layer[l], world), world)
            for l in range(loads_by_layer.shape[0])
        ]
    )
    bal = (
        torch.gather(loads_by_layer, 1, n2o)
        .reshape(loads_by_layer.shape[0], world, n_local)
        .sum(2)
        .float()
    )
    return {
        "current_max_over_ideal": (cur.max(1).values / ideal),
        "balanced_max_over_ideal": (bal.max(1).values / ideal),
        "new_to_old": n2o,
    }

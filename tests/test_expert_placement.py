"""Expert placement: the assignment helpers, and that permuting is a no-op relabelling.

The permutation is only sound if the gate and the expert weights move together, so
the central test builds a stub MoE, applies a placement, and checks the layer
computes the same function as before. Everything runs on CPU with a gloo group so
the collective path is exercised without GPUs.
"""

import torch
import torch.distributed as dist
import pytest

from ds41f.backend import expert_placement as ep


# -- assignment helpers ------------------------------------------------------


def test_greedy_placement_gives_equal_counts_and_caps_the_busiest():
    loads = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0, 4.0])
    assign = ep.greedy_placement(loads, n_ranks=2)
    assert sorted(assign.tolist()) == [0, 0, 0, 1, 1, 1]
    per_rank = [loads[assign == r].sum().item() for r in range(2)]
    # {9,6,5} and {8,7,4}: the best achievable max for equal counts is 20
    assert per_rank == [20.0, 19.0]


def test_greedy_placement_beats_contiguous_on_skewed_loads():
    # two hot experts land on the same rank under contiguous ownership
    loads = torch.tensor([50.0, 50.0] + [1.0] * 6)
    assign = ep.greedy_placement(loads, n_ranks=4)
    contiguous = loads.reshape(4, 2).sum(1)
    placed = torch.tensor([loads[assign == r].sum().item() for r in range(4)])
    assert contiguous.max() == 100.0
    assert placed.max() == 51.0
    assert placed.max() < contiguous.max()
    assert sorted(assign.tolist()) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_new_to_old_is_a_permutation_grouped_by_rank():
    loads = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0, 4.0])
    n2o = ep.new_to_old_from_assign(ep.greedy_placement(loads, n_ranks=2), n_ranks=2)
    assert sorted(n2o.tolist()) == list(range(6))
    # rank 0 holds the heaviest of each pair it won: experts 0, 3 and 4
    assert set(n2o[:3].tolist()) == {0, 3, 4}
    assert set(n2o[3:].tolist()) == {1, 2, 5}


def test_new_to_old_rejects_an_uneven_assignment():
    assign = torch.tensor([0, 0, 0, 1])
    with pytest.raises(ValueError, match="expected 2"):
        ep.new_to_old_from_assign(assign, n_ranks=2)


def test_new_to_old_is_ascending_within_each_rank():
    """The exchange below relies on this: rank s sends its payloads in ascending
    old-expert order and the receiver fills local slots in ascending j order, which
    only matches if each rank's slice of n2o ascends."""
    loads = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0])
    n2o = ep.new_to_old_from_assign(ep.greedy_placement(loads, n_ranks=4), n_ranks=4)
    for r in range(4):
        chunk = n2o[r * 2 : (r + 1) * 2]
        assert chunk.tolist() == sorted(chunk.tolist())


def test_exchange_plan_places_each_expert_in_its_new_slot():
    """Simulate the whole exchange for every rank and check the invariant the layer
    depends on: after the redistribution, global slot j holds old expert n2o[j].
    Without a process group this is the only way to exercise the index arithmetic."""
    n_experts, world = 8, 4
    n_local = n_experts // world
    loads = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0])
    n2o = ep.new_to_old_from_assign(ep.greedy_placement(loads, n_ranks=world), n_ranks=world)

    plans = {r: ep.exchange_plan(n2o, r, world) for r in range(world)}

    # every rank sends exactly what its peer expects to receive, count-wise
    for r in range(world):
        mine, dest, want, src = plans[r]
        for d in range(world):
            sent = int((dest == d).sum())
            expected = int((plans[d][3] == r).sum())
            assert sent == expected, f"rank {r}->{d}: sends {sent}, peer expects {expected}"

    # replay the exchange and check the resulting slot -> expert mapping
    slots = {}  # (rank, local_slot) -> old expert id
    for r in range(world):
        mine, dest, want, src = plans[r]
        for i, e in enumerate(mine.tolist()):
            d = int(dest[i])
            # rank d will place this at the next free local slot sourced from r
            placed = [j for j in range(n_local) if int(plans[d][3][j]) == r]
            taken = [j for j in placed if (d, j) in slots]
            slots[(d, placed[len(taken)])] = e
    for r in range(world):
        for j in range(n_local):
            assert slots[(r, j)] == int(n2o[r * n_local + j])


def test_exchange_plan_identity_for_world_one():
    n2o = torch.arange(4)
    mine, dest, want, src = ep.exchange_plan(n2o, 0, 1)
    assert mine.tolist() == [0, 1, 2, 3]
    assert dest.tolist() == [0, 0, 0, 0]
    assert want.tolist() == [0, 1, 2, 3]
    assert src.tolist() == [0, 0, 0, 0]


def test_report_shows_the_balance_improvement():
    loads = torch.stack([torch.tensor([50.0, 50.0] + [1.0] * 6)] * 3)
    out = ep.report(loads, world=4)
    assert out["balanced_max_over_ideal"].max() < out["current_max_over_ideal"].max()
    assert out["new_to_old"].shape == (3, 8)


# -- apply() on a stub model -------------------------------------------------


class _Lin:
    def __init__(self, weight, scale):
        self.weight = weight
        self.scale = scale


class _Expert:
    """Same tensor layout expert_placement expects: w1/w2/w3, each weight+scale, all
    one byte per element (fp4 packed as float4_e2m1fn_x2, scales as float8_e8m0fnu)."""

    def __init__(self, seed, n=4):
        g = torch.Generator().manual_seed(seed)
        self.w1 = _Lin(torch.randint(0, 255, (n,), generator=g, dtype=torch.uint8), torch.ones(n, dtype=torch.uint8))
        self.w2 = _Lin(torch.randint(0, 255, (n,), generator=g, dtype=torch.uint8), torch.ones(n, dtype=torch.uint8))
        self.w3 = _Lin(torch.randint(0, 255, (n,), generator=g, dtype=torch.uint8), torch.ones(n, dtype=torch.uint8))


class _Gate:
    def __init__(self, n_experts, dim):
        self.weight = torch.nn.Parameter(torch.arange(n_experts * dim, dtype=torch.float32).reshape(n_experts, dim))
        self.bias = torch.nn.Parameter(torch.arange(n_experts, dtype=torch.float32))


class _MoE:
    def __init__(self, n_experts, dim):
        self.gate = _Gate(n_experts, dim)
        self.experts = [_Expert(seed=e) for e in range(n_experts)]


class _Block:
    def __init__(self, n_experts, dim):
        self.ffn = _MoE(n_experts, dim)


class _StubModel:
    def __init__(self, n_layers, n_experts, dim):
        self.layers = [_Block(n_experts, dim) for _ in range(n_layers)]

    def route(self, layer, x):
        """Stand-in for the real layer: gate rows pick experts, expert ids come from
        the permuted gate. Returns the (expert index, payload) pairs the layer would use."""
        gate = self.layers[layer].ffn.gate
        logits = gate.weight @ x + gate.bias
        order = torch.topk(logits, 2).indices
        return [(int(j), tuple(ep._pack(self.layers[layer].ffn.experts[int(j)]).tolist())) for j in order]


def _placement(n_layers, n_experts, n_ranks, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_layers):
        perm = torch.randperm(n_experts, generator=g)
        # group by rank so rank r owns slots [r*n_local, (r+1)*n_local)
        out.append(perm)
    return torch.stack(out)


@pytest.fixture(scope="module", autouse=True)
def _gloo():
    """Single-process gloo group so apply()'s all_to_all runs without GPUs."""
    import os

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
    yield


def test_gate_and_experts_move_together():
    """The whole point of the permutation: after permuting the gate rows and moving
    each expert to its new slot, the layer selects the same old expert with the same
    weights for a given input.

    Uses permute_gate plus the relocation the exchange-plan test proves, because
    apply() itself needs a process group.
    """
    n_layers, n_experts, world, dim = 2, 8, 4, 5
    model = _StubModel(n_layers, n_experts, dim)
    x = torch.arange(dim, dtype=torch.float32)
    before = [model.route(l, x) for l in range(n_layers)]

    loads = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0])
    n2o = ep.new_to_old_from_assign(ep.greedy_placement(loads, n_ranks=world), n_ranks=world)

    for l in range(n_layers):
        moe = model.layers[l].ffn
        old = [ep._pack(e).clone() for e in moe.experts]
        ep.permute_gate(moe.gate, n2o)
        for j in range(n_experts):
            ep._unpack_into(old[int(n2o[j])], 0, moe.experts[j])

    after = [model.route(l, x) for l in range(n_layers)]
    for l in range(n_layers):
        # before: slot j held old expert j, so (slot, payload) is (old expert, payload)
        assert {(int(n2o[j]), p) for j, p in after[l]} == set(before[l])


def test_apply_identity_placement_is_a_no_op(tmp_path, monkeypatch):
    """world=1 holds every expert on one rank, so the only valid placement is the
    identity; it must leave both the gate and the expert bytes untouched."""
    n_layers, n_experts, dim = 2, 4, 3
    model = _StubModel(n_layers, n_experts, dim)
    gate_before = [model.layers[l].ffn.gate.weight.detach().clone() for l in range(n_layers)]
    experts_before = [
        [ep._pack(e).clone() for e in model.layers[l].ffn.experts] for l in range(n_layers)
    ]

    n2o = torch.stack([torch.arange(n_experts) for _ in range(n_layers)])
    path = tmp_path / "identity.pt"
    torch.save({"new_to_old": n2o}, path)

    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    ep.apply(model, str(path), rank=0, world=1)

    for l in range(n_layers):
        assert torch.equal(model.layers[l].ffn.gate.weight.detach(), gate_before[l])
        for e, was in zip(model.layers[l].ffn.experts, experts_before[l]):
            assert torch.equal(ep._pack(e), was)


def test_maybe_apply_is_inert_without_a_placement(tmp_path, monkeypatch):
    model = _StubModel(1, 4, 3)
    monkeypatch.setenv("DSV41F_EXPERT_PLACEMENT", "none")
    assert ep.maybe_apply(model, rank=0, world=4) is False


def test_maybe_apply_is_inert_on_a_single_rank(tmp_path, monkeypatch):
    model = _StubModel(1, 4, 3)
    monkeypatch.setenv("DSV41F_EXPERT_PLACEMENT", str(tmp_path / "missing.pt"))
    assert ep.maybe_apply(model, rank=0, world=1) is False


def test_reference_backend_placement_is_opt_in():
    """Default construction must not touch the model: the placement is a collective
    and the single-controller topology would hang."""
    from ds41f.backend.reference import ReferenceBackend
    from tests.test_reference_backend import LogitsStubModel

    be = ReferenceBackend(LogitsStubModel(), eos_token_id=500)
    assert be.placement_applied is False


def test_pack_rejects_multi_byte_expert_tensors():
    """A 4-byte scale would make a numel stop being a byte count, desynchronizing the
    send and receive buffers. Fail loudly instead."""

    class _Bad(_Expert):
        def __init__(self):
            super().__init__(seed=0)
            self.w1 = _Lin(self.w1.weight, torch.ones(4, dtype=torch.float32))

    with pytest.raises(TypeError, match="bytes per element"):
        ep._pack(_Bad())


def test_apply_rejects_a_calibration_from_another_checkpoint(tmp_path, monkeypatch):
    """A calibration is per-checkpoint. A permutation of the wrong length would
    mislabel experts silently, so apply() must refuse it."""
    n_layers, n_experts, dim = 2, 4, 3
    model = _StubModel(n_layers, n_experts, dim)
    # layer 0 claims 8 experts where the model has 4
    bad = torch.stack([torch.randperm(8), torch.randperm(8)])
    path = tmp_path / "bad.pt"
    torch.save({"new_to_old": bad}, path)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    with pytest.raises(ValueError, match="not a permutation"):
        ep.apply(model, str(path), rank=0, world=1)


def test_apply_rejects_a_calibration_with_too_few_layers(tmp_path, monkeypatch):
    n_layers, n_experts, dim = 3, 4, 3
    model = _StubModel(n_layers, n_experts, dim)
    short = torch.stack([torch.arange(n_experts)])  # 1 layer for a 3-layer model
    path = tmp_path / "short.pt"
    torch.save({"new_to_old": short}, path)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    with pytest.raises(ValueError, match="needs one"):
        ep.apply(model, str(path), rank=0, world=1)


def test_apply_rejects_a_non_permutation(tmp_path, monkeypatch):
    """Repeated or out-of-range entries are not a relabelling at all."""
    n_layers, n_experts, dim = 2, 4, 3
    model = _StubModel(n_layers, n_experts, dim)
    dup = torch.stack([torch.tensor([0, 0, 1, 2]), torch.tensor([0, 1, 2, 3])])
    path = tmp_path / "dup.pt"
    torch.save({"new_to_old": dup}, path)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    with pytest.raises(ValueError, match="not a permutation"):
        ep.apply(model, str(path), rank=0, world=1)


def test_maybe_apply_honours_an_explicit_path(tmp_path, monkeypatch):
    """The engine default looks beside the installed package, which is wrong for a
    model-repository calibration. An explicit path must win, and must be usable
    without going through the environment variable.

    apply() is stubbed: a real one needs a multi-rank group, and the point here is
    which file gets selected.
    """
    monkeypatch.delenv("DSV41F_EXPERT_PLACEMENT", raising=False)
    seen = []

    def _stub_apply(m, p, r, w, verbose=False, audit=False):
        seen.append(p)
        return {"file": p, "layers": [], "gate_permuted": True, "exchange_ok": True} if audit else None

    monkeypatch.setattr(ep, "apply", _stub_apply)
    model = _StubModel(1, 4, 3)
    path = tmp_path / "explicit.pt"
    path.write_bytes(b"")  # never loaded; apply is stubbed

    # explicit path wins over the (absent) package-adjacent default
    assert ep.maybe_apply(model, rank=0, world=4, path=str(path)) is True
    assert seen == [str(path)]
    assert model.placement_audit is None  # audit off by default

    # audit=True records the placement summary on the model for a verifier to read
    assert ep.maybe_apply(model, rank=0, world=4, path=str(path), audit=True) is True
    assert model.placement_audit["gate_permuted"] is True

    # and "none" passed explicitly still disables it, clearing the audit
    assert ep.maybe_apply(model, rank=0, world=4, path="none") is False
    assert model.placement_audit is None
    assert seen == [str(path), str(path)]

    # the environment variable is still honoured when no path is given
    monkeypatch.setenv("DSV41F_EXPERT_PLACEMENT", str(path))
    assert ep.maybe_apply(model, rank=0, world=4) is True
    assert seen == [str(path), str(path), str(path)]

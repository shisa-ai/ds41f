"""StateStore: slot ownership, generations, budget."""

import pytest

from ds41f.state import StateStore


def test_alloc_release_and_generation():
    s = StateStore(max_rows=2, max_context_tokens=100)
    r = s.alloc(1, prompt_tokens=10, max_new_tokens=10)
    assert s.owned_by(1) == r.slot
    assert s.release(r.slot, 1, r.generation)
    assert s.owned_by(1) is None
    r2 = s.alloc(2, prompt_tokens=10, max_new_tokens=10)
    assert r2.slot == r.slot
    assert r2.generation == r.generation + 1


def test_stale_release_is_rejected():
    s = StateStore(max_rows=1, max_context_tokens=100)
    r = s.alloc(1, prompt_tokens=1, max_new_tokens=1)
    s.release(r.slot, 1, r.generation)
    s.alloc(2, prompt_tokens=1, max_new_tokens=1)  # new occupant
    # stale reference: old req_id at old generation must not release the new occupant
    assert not s.release(r.slot, 1, r.generation)
    assert s.owned_by(2) == r.slot


def test_context_budget_enforced():
    s = StateStore(max_rows=4, max_context_tokens=50)
    assert not s.can_admit(40, 20)
    with pytest.raises(ValueError):
        s.alloc(1, prompt_tokens=40, max_new_tokens=20)


def test_exhaustion():
    s = StateStore(max_rows=1, max_context_tokens=100)
    s.alloc(1, prompt_tokens=1, max_new_tokens=1)
    with pytest.raises(RuntimeError):
        s.alloc(2, prompt_tokens=1, max_new_tokens=1)

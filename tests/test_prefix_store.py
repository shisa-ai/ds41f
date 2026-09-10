"""PrefixSnapshotStore: exact-boundary matching, byte budget, LRU, invalidation."""

from ds41f.state import PrefixSnapshotStore


def test_exact_prefix_hit_and_miss():
    s = PrefixSnapshotStore(byte_budget=1000)
    s.put([1, 2, 3, 4], "payloadA", nbytes=100)
    hit = s.lookup([1, 2, 3, 4, 5, 6])  # stored seq is a prefix of the new tokens
    assert hit is not None and hit.shared_len == 4 and hit.entry.payload == "payloadA"
    assert s.lookup([1, 2, 3]) is None  # shorter than the stored sequence: no partial
    assert s.lookup([9, 9, 9, 4]) is None


def test_longest_match_wins():
    s = PrefixSnapshotStore(byte_budget=1000)
    s.put([1, 2], "short", 10)
    s.put([1, 2, 3, 4, 5], "long", 20)
    hit = s.lookup([1, 2, 3, 4, 5, 6, 7])
    assert hit.entry.payload == "long" and hit.shared_len == 5


def test_byte_budget_evicts_lru():
    s = PrefixSnapshotStore(byte_budget=250)
    s.put([1], "a", 100)
    s.put([2], "b", 100)
    s.lookup([1])  # touch a
    s.put([3], "c", 100)  # must evict b (LRU), not a
    assert s.lookup([1]) is not None
    assert s.lookup([2]) is None
    assert s.lookup([3]) is not None
    assert s.bytes == 200


def test_oversized_payload_rejected():
    s = PrefixSnapshotStore(byte_budget=50)
    try:
        s.put([1, 2], "big", 100)
        assert False
    except ValueError:
        pass


def test_invalidate_drops_all_and_bumps_generation():
    s = PrefixSnapshotStore(byte_budget=1000)
    s.put([1, 2, 3], "x", 10)
    assert s.size == 1
    freed = s.invalidate()
    assert freed == 10 and s.size == 0 and s.bytes == 0
    assert s.lookup([1, 2, 3, 4]) is None


def test_reput_replaces_entry():
    s = PrefixSnapshotStore(byte_budget=1000)
    s.put([1, 2], "old", 10)
    s.put([1, 2], "new", 30)
    assert s.size == 1 and s.bytes == 30
    assert s.lookup([1, 2, 3]).entry.payload == "new"


def test_min_shared_gate():
    s = PrefixSnapshotStore(byte_budget=1000)
    s.put([1], "tiny", 10)
    assert s.lookup([1, 2], min_shared=2) is None  # 1-token prefix not worth a restore
    assert s.lookup([1, 2], min_shared=1) is not None

"""Determinism and disjointness of the hash-based split."""

from src.data.splits import assign_split


def test_split_is_deterministic():
    assert assign_split("item_abc") == assign_split("item_abc")


def test_split_values_valid():
    for i in range(200):
        assert assign_split(f"item_{i}") in {"train", "val", "test"}


def test_split_partitions_roughly():
    counts = {"train": 0, "val": 0, "test": 0}
    for i in range(2000):
        counts[assign_split(f"item_{i}")] += 1
    # train should dominate; val/test should be non-empty and smaller.
    assert counts["train"] > counts["val"]
    assert counts["train"] > counts["test"]
    assert counts["val"] > 0 and counts["test"] > 0


def test_adding_items_does_not_move_existing():
    before = {f"item_{i}": assign_split(f"item_{i}") for i in range(50)}
    # Adding new ids cannot change existing assignments (hash is per-id).
    after = {f"item_{i}": assign_split(f"item_{i}") for i in range(100)}
    for k, v in before.items():
        assert after[k] == v

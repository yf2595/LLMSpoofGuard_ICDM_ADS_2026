"""
Train/test splitting helpers for the evaluation harness.

Splits are performed at the trajectory level (never at the point level)
so that points from the same flight never appear in both partitions.
A stratified split keeps the positive/negative ratio identical across
train and test.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def stratified_split(
    trajectories: Sequence,
    labels: Sequence[bool],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Return (train_indices, test_indices) lists with stratification."""
    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels, dtype=bool)
    pos_idx = np.flatnonzero(labels_arr)
    neg_idx = np.flatnonzero(~labels_arr)

    def _split(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if idx.size == 0:
            return idx, idx
        shuffled = rng.permutation(idx)
        n_test = max(1, int(round(idx.size * test_size))) if idx.size > 1 else 0
        return shuffled[n_test:], shuffled[:n_test]

    pos_train, pos_test = _split(pos_idx)
    neg_train, neg_test = _split(neg_idx)

    train = np.concatenate([pos_train, neg_train])
    test = np.concatenate([pos_test, neg_test])
    rng.shuffle(train)
    rng.shuffle(test)

    return train.tolist(), test.tolist()


def gather(items: Sequence, indices: Sequence[int]) -> list:
    """Select ``items[i] for i in indices``."""
    return [items[i] for i in indices]


def balanced_test_split(
    trajectories: Sequence,
    labels: Sequence[bool],
    n_per_class: int,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Reserve ``n_per_class`` positives + ``n_per_class`` negatives for the
    test set; everything else becomes the training pool.

    Useful when you want to evaluate every baseline (in particular the LLM)
    on an exactly class-balanced test set, regardless of the dataset's
    natural prior.

    Raises:
        ValueError: if the dataset does not contain enough positives or
        negatives to fill the requested test counts.
    """
    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels, dtype=bool)
    pos_idx = np.flatnonzero(labels_arr)
    neg_idx = np.flatnonzero(~labels_arr)

    if pos_idx.size < n_per_class:
        raise ValueError(
            f"Need {n_per_class} positive trajectories for balanced test set, "
            f"only {pos_idx.size} available."
        )
    if neg_idx.size < n_per_class:
        raise ValueError(
            f"Need {n_per_class} negative trajectories for balanced test set, "
            f"only {neg_idx.size} available."
        )

    pos_shuf = rng.permutation(pos_idx)
    neg_shuf = rng.permutation(neg_idx)

    pos_test, pos_train = pos_shuf[:n_per_class], pos_shuf[n_per_class:]
    neg_test, neg_train = neg_shuf[:n_per_class], neg_shuf[n_per_class:]

    train = np.concatenate([pos_train, neg_train])
    test = np.concatenate([pos_test, neg_test])
    rng.shuffle(train)
    rng.shuffle(test)

    return train.tolist(), test.tolist()

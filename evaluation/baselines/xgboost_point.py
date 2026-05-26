"""
XGBoost (point-level) baseline (paper Table II, S; P).

Supervised binary classifier on per-message kinematic deltas
(``src.features.point_deltas``). To produce a single trajectory-level
label, the per-point spoofing probabilities are aggregated via max-pool;
a trajectory is flagged spoofed if any of its points exceeds the
configured threshold.

For training, every point inherits its parent trajectory's label - this
matches the standard ADS-B supervised setup described in the paper.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from xgboost import XGBClassifier

from src.features import point_deltas

from .base import Baseline


class XGBoostPointBaseline(Baseline):
    name = "XGBoost-point"
    is_supervised = True
    granularity = "point"

    def __init__(
        self,
        threshold: float = 0.5,
        n_estimators: int = 400,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ):
        self.threshold = threshold
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
        )
        self._fitted = False

    def _featurise(
        self,
        trajs: Sequence[Sequence[Sequence]],
        labels: Sequence[bool] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, list[int]]:
        chunks: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        sizes: list[int] = []
        for i, t in enumerate(trajs):
            d = point_deltas(t)
            chunks.append(d)
            sizes.append(len(d))
            if labels is not None:
                ys.append(np.full(len(d), int(bool(labels[i])), dtype=np.int32))
        X = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 9), dtype=np.float32)
        y = np.concatenate(ys, axis=0) if ys else None
        return X, y, sizes

    def fit(
        self,
        train_trajectories: Sequence[Sequence[Sequence]],
        train_labels: Sequence[bool] | None = None,
    ) -> None:
        if train_labels is None:
            raise ValueError("XGBoost-point requires training labels.")
        X, y, _ = self._featurise(train_trajectories, train_labels)
        if X.size == 0 or y is None or len(set(y.tolist())) < 2:
            self._fitted = False
            return
        self.model.fit(X, y)
        self._fitted = True

    def predict(self, test_trajectories: Sequence[Sequence[Sequence]]) -> list[bool]:
        if not self._fitted:
            return [False] * len(test_trajectories)
        X, _, sizes = self._featurise(test_trajectories)
        if X.size == 0:
            return [False] * len(test_trajectories)
        probs = self.model.predict_proba(X)[:, 1]
        results: list[bool] = []
        cursor = 0
        for size in sizes:
            if size == 0:
                results.append(False)
                continue
            chunk = probs[cursor: cursor + size]
            results.append(bool(np.any(chunk >= self.threshold)))
            cursor += size
        return results

"""
Isolation Forest baseline (paper Table II, U; P).

Unsupervised, point-level. Trained on per-message kinematic deltas
(``src.features.point_deltas``). A trajectory is flagged spoofed iff at
least one of its points scores as an anomaly.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.ensemble import IsolationForest

from src.features import point_deltas

from .base import Baseline


class IsolationForestBaseline(Baseline):
    name = "IsolationForest"
    is_supervised = False
    granularity = "point"

    def __init__(
        self,
        contamination: float | str = "auto",
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    def _stack(self, trajs: Sequence[Sequence[Sequence]]) -> tuple[np.ndarray, list[int]]:
        chunks: list[np.ndarray] = []
        sizes: list[int] = []
        for t in trajs:
            d = point_deltas(t)
            chunks.append(d)
            sizes.append(len(d))
        if not chunks:
            return np.zeros((0, 9), dtype=np.float32), sizes
        return np.concatenate(chunks, axis=0), sizes

    def fit(
        self,
        train_trajectories: Sequence[Sequence[Sequence]],
        train_labels: Sequence[bool] | None = None,
    ) -> None:
        X, _ = self._stack(train_trajectories)
        if X.size == 0:
            self._fitted = False
            return
        self.model.fit(X)
        self._fitted = True

    def predict(self, test_trajectories: Sequence[Sequence[Sequence]]) -> list[bool]:
        if not self._fitted:
            return [False] * len(test_trajectories)
        X, sizes = self._stack(test_trajectories)
        preds = self.model.predict(X) if X.size else np.array([])
        results: list[bool] = []
        cursor = 0
        for size in sizes:
            if size == 0:
                results.append(False)
                continue
            chunk = preds[cursor: cursor + size]
            results.append(bool(np.any(chunk == -1)))
            cursor += size
        return results

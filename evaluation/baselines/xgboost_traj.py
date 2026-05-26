"""
XGBoost (trajectory-statistics) baseline.

Optional variant requested for the paper experiments: instead of
per-message deltas, each trajectory is summarised by a fixed-length
statistics vector (``src.features.trajectory_stats``) and the classifier
operates at the trajectory level.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from xgboost import XGBClassifier

from src.features import trajectory_stats

from .base import Baseline


class XGBoostTrajBaseline(Baseline):
    name = "XGBoost-Traj"
    is_supervised = True
    granularity = "trajectory"

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

    def _featurise(self, trajs: Sequence[Sequence[Sequence]]) -> np.ndarray:
        if not trajs:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([trajectory_stats(t) for t in trajs], axis=0)

    def fit(
        self,
        train_trajectories: Sequence[Sequence[Sequence]],
        train_labels: Sequence[bool] | None = None,
    ) -> None:
        if train_labels is None:
            raise ValueError("XGBoost-Traj requires training labels.")
        X = self._featurise(train_trajectories)
        y = np.asarray([int(bool(v)) for v in train_labels], dtype=np.int32)
        if X.size == 0 or len(set(y.tolist())) < 2:
            self._fitted = False
            return
        self.model.fit(X, y)
        self._fitted = True

    def predict(self, test_trajectories: Sequence[Sequence[Sequence]]) -> list[bool]:
        if not self._fitted:
            return [False] * len(test_trajectories)
        X = self._featurise(test_trajectories)
        if X.size == 0:
            return [False] * len(test_trajectories)
        probs = self.model.predict_proba(X)[:, 1]
        return [bool(p >= self.threshold) for p in probs]

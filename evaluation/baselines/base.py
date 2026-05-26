"""Common interface shared by every baseline detector."""

from __future__ import annotations

from typing import Sequence


class Baseline:
    """Minimal interface implemented by every baseline detector."""

    name: str = "baseline"
    is_supervised: bool = False
    granularity: str = "trajectory"  # "point" | "trajectory"

    def fit(
        self,
        train_trajectories: Sequence[Sequence[Sequence]],
        train_labels: Sequence[bool] | None = None,
    ) -> None:
        """Fit the detector. Unsupervised baselines may ignore the labels."""

    def predict(
        self, test_trajectories: Sequence[Sequence[Sequence]]
    ) -> list[bool]:
        """Return one boolean spoofing label per trajectory."""
        raise NotImplementedError

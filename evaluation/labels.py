"""
Ground-truth label generation for the evaluation harness.

The paper uses the Tier-1 Rule-Based Heuristics (RBH) as its annotation
source (paper Section V-B). This module is a thin wrapper around
``src.rules`` so that the labelling oracle and the RBH baseline share a
single source of truth - which makes the (acknowledged) circularity
explicit in the code.

Every baseline is evaluated against ``trajectory_labels``:
    True  = at least one RBH rule fires somewhere in the trajectory
    False = no RBH rule fires
"""

from __future__ import annotations

from typing import Sequence

from src.rules import (
    classify_trajectory,
    is_spoofed_trajectory,
    is_spoofed_trajectory_excluding,
)

# Categories omitted from evaluation ground truth (RBH may still fire on heading).
EXCLUDED_EVAL_CATEGORIES = frozenset({
    "Unrealistic heading change",
    "Unstable altitude",
})


def trajectory_labels(trajectories: Sequence[Sequence[Sequence]]) -> list[bool]:
    """Return one boolean label per trajectory using the RBH oracle."""
    return [is_spoofed_trajectory(traj) for traj in trajectories]


def trajectory_labels_excluding_heading(
    trajectories: Sequence[Sequence[Sequence]],
) -> list[bool]:
    """Positive iff RBH triggers on any rule except heading and unstable altitude.

    Unstable altitude is no longer implemented in ``src.rules``; heading is
    still detected by RBH but ignored for evaluation labels.
    """
    return [
        is_spoofed_trajectory_excluding(traj, EXCLUDED_EVAL_CATEGORIES)
        for traj in trajectories
    ]


def trajectory_categories(trajectories: Sequence[Sequence[Sequence]]) -> list[str | None]:
    """Return the most-frequent triggered category per trajectory (or None)."""
    return [classify_trajectory(traj) for traj in trajectories]

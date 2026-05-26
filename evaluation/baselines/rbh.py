"""
Rule-Based Heuristics baseline (paper Sec. V-B Tier 1).

Unsupervised, point-level. Implementation lives in ``src.rules``; this
file is a thin wrapper that exposes the standard ``Baseline`` interface.

Note: this baseline shares its rule set with the evaluation oracle in
``evaluation/labels.py``, so it will trivially agree with the oracle.
That is intentional and matches the paper's framing of RBH as both
labeller and baseline (Section V-B). The interesting comparison is
against the other methods.
"""

from __future__ import annotations

from typing import Sequence

from src.rules import is_spoofed_trajectory

from .base import Baseline


class RBHBaseline(Baseline):
    name = "RBH"
    is_supervised = False
    granularity = "point"

    def predict(self, test_trajectories: Sequence[Sequence[Sequence]]) -> list[bool]:
        return [is_spoofed_trajectory(t) for t in test_trajectories]

"""
Rule-Based Heuristics (RBH) for GPS spoofing detection.

This module implements the fixed-threshold rules drawn directly from
Section 6 ("Spoofing Category Bank") of the paper-only LLM prompt in
``prompts/gps_detection_prompt.py``. The rule module and the LLM prompt
therefore share a single category vocabulary.

Rules

    altitude_drop              decrease > 4000 m within 2 min
    altitude_increase          increase > 3000 m within 2 min (outside takeoff)
    timestamp_freeze           identical timestamp >= 3 consecutive points
    zero_velocity              velocity < 50 m/s while altitude > 5000 m
    sudden_positional_jump     great-circle displacement > 250 km in <= 120 s,
                               OR |Delta lat| > 1.8 deg OR |Delta lon| > 1.8 deg
    unrealistic_heading_change |Delta heading| > 120 deg between adjacent
    unrealistic_velocity_spike |Delta velocity| > 120 m/s in <= 2 min, OR
                               absolute velocity > 750 m/s

Time-gap semantics (Section 3 of the prompt)

    - Δt > 150 s   : reasoning boundary, no rule is evaluated.
    - 120 s < Δt <= 150 s : only absolute-threshold rules are evaluated
                            (timestamp_freeze, zero_velocity).
    - Δt <= 120 s : every rule is evaluated.

The module deliberately does NOT try to approximate the five legitimate
maneuver counterexamples (holding pattern, takeoff/landing, etc.) listed
in Section 5 of the prompt. This keeps RBH a deliberately-simple Tier-1
baseline (paper Sec. V-B). The LLM detector handles those edge cases via
the prompt's explicit counterexamples.

A trajectory is the tuple format produced by ``preprocessing``:
    (lat, lon, altitude, country, velocity, heading, model, timestamp)
Index 0=lat, 1=lon, 2=alt, 4=vel, 5=hdg, 7=ts.
"""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from typing import Sequence

import pandas as pd

# Indices into the ADS-B trajectory tuples
LAT, LON, ALT, COUNTRY, VEL, HDG, MODEL, TS = range(8)

# Thresholds (Section 6 of the paper-only prompt)
ALT_DROP_M = 4000.0
ALT_INCREASE_M = 3000.0
ZERO_VEL_THRESHOLD = 50.0
ZERO_VEL_ALT_GATE = 5000.0
POSITIONAL_JUMP_KM = 250.0
POSITIONAL_JUMP_DEG = 1.8
HEADING_CHANGE_DEG = 120.0
VEL_SPIKE_DELTA_MPS = 120.0
VEL_SPIKE_ABS_MPS = 750.0
TAKEOFF_ALT_GATE = 3000.0

# Time gates (Section 3 of the paper-only prompt)
DETECTION_GAP_S = 120.0
REASONING_BOUNDARY_S = 150.0
TWO_MIN_S = 120.0


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    r = 6_371_000.0
    p1, p2 = radians(lat1), radians(lat2)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(p1) * cos(p2) * sin(dlon / 2) ** 2
    return 2 * r * asin(sqrt(a))


def _heading_delta(h1: float, h2: float) -> float:
    """Smallest absolute difference between two compass headings (0..180)."""
    d = abs(h2 - h1) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _dt_seconds(prev_ts, curr_ts) -> float:
    return (pd.to_datetime(curr_ts) - pd.to_datetime(prev_ts)).total_seconds()


def evaluate_pair(prev_point: Sequence, curr_point: Sequence) -> str | None:
    """Return the triggered category for a single consecutive pair, or None.

    Only the kinematic-delta rules are evaluated here (altitude_drop,
    altitude_increase, sudden_positional_jump, unrealistic_heading_change,
    unrealistic_velocity_spike, zero_velocity). The window-based rule
    timestamp_freeze requires >= 3 points and is evaluated by
    ``flag_trajectory``.
    """
    dt = _dt_seconds(prev_point[TS], curr_point[TS])
    if dt > REASONING_BOUNDARY_S:
        return None
    if dt <= 0:
        return None

    if (
        curr_point[VEL] is not None
        and curr_point[ALT] is not None
        and curr_point[VEL] < ZERO_VEL_THRESHOLD
        and curr_point[ALT] > ZERO_VEL_ALT_GATE
    ):
        return "Zero velocity"

    if (
        curr_point[VEL] is not None
        and curr_point[VEL] > VEL_SPIKE_ABS_MPS
    ):
        return "Unrealistic velocity spike"

    if dt > DETECTION_GAP_S:
        return None

    if (
        prev_point[ALT] is not None
        and curr_point[ALT] is not None
    ):
        d_alt = curr_point[ALT] - prev_point[ALT]
        if dt <= TWO_MIN_S and d_alt < -ALT_DROP_M:
            return "Altitude drop"
        if (
            dt <= TWO_MIN_S
            and d_alt > ALT_INCREASE_M
            and prev_point[ALT] > TAKEOFF_ALT_GATE
        ):
            return "Altitude increase"

    if (
        prev_point[LAT] is not None
        and curr_point[LAT] is not None
        and prev_point[LON] is not None
        and curr_point[LON] is not None
    ):
        distance_km = _haversine_m(
            prev_point[LAT], prev_point[LON],
            curr_point[LAT], curr_point[LON],
        ) / 1000.0
        d_lat = abs(curr_point[LAT] - prev_point[LAT])
        d_lon = abs(curr_point[LON] - prev_point[LON])
        if dt <= TWO_MIN_S and (
            distance_km > POSITIONAL_JUMP_KM
            or d_lat > POSITIONAL_JUMP_DEG
            or d_lon > POSITIONAL_JUMP_DEG
        ):
            return "Sudden positional jump"

    if (
        prev_point[VEL] is not None
        and curr_point[VEL] is not None
        and dt <= TWO_MIN_S
        and abs(curr_point[VEL] - prev_point[VEL]) > VEL_SPIKE_DELTA_MPS
    ):
        return "Unrealistic velocity spike"

    if (
        prev_point[HDG] is not None
        and curr_point[HDG] is not None
        and _heading_delta(prev_point[HDG], curr_point[HDG]) > HEADING_CHANGE_DEG
    ):
        return "Unrealistic heading change"

    return None


def _flag_timestamp_freeze(traj: Sequence[Sequence]) -> list[str | None]:
    """Mark runs of >= 3 consecutive identical timestamps."""
    flags: list[str | None] = [None] * len(traj)
    n = len(traj)
    i = 0
    while i < n - 2:
        j = i + 1
        while j < n and traj[j][TS] == traj[i][TS]:
            j += 1
        run = j - i
        if run >= 3:
            for k in range(i, j):
                flags[k] = "Timestamp freeze"
            i = j
        else:
            i += 1
    return flags


def flag_trajectory(traj: Sequence[Sequence]) -> list[str | None]:
    """Return one category label per point (or None when no rule triggers).

    The pair-wise rule output is written to ``flags[i]`` for the curr point
    (i.e. the second member of the (prev, curr) pair). Timestamp freeze
    overlays on top when present.
    """
    n = len(traj)
    flags: list[str | None] = [None] * n

    for i in range(1, n):
        category = evaluate_pair(traj[i - 1], traj[i])
        if category is not None:
            flags[i] = category

    freeze = _flag_timestamp_freeze(traj)
    for i in range(n):
        if freeze[i] is not None:
            flags[i] = freeze[i]

    return flags


def is_spoofed_trajectory(traj: Sequence[Sequence]) -> bool:
    """True iff any rule triggers anywhere in the trajectory."""
    return any(c is not None for c in flag_trajectory(traj))


def is_spoofed_trajectory_excluding(
    traj: Sequence[Sequence],
    exclude_categories: frozenset[str] | set[str],
) -> bool:
    """True iff any rule triggers, ignoring points whose category is excluded."""
    return any(
        c is not None and c not in exclude_categories
        for c in flag_trajectory(traj)
    )


def classify_trajectory(traj: Sequence[Sequence]) -> str | None:
    """Return the most frequent triggered category, or None if clean."""
    flags = [c for c in flag_trajectory(traj) if c is not None]
    if not flags:
        return None
    counts: dict[str, int] = {}
    for f in flags:
        counts[f] = counts.get(f, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]

"""
Feature extraction utilities for the baseline detectors.

Three representations are exposed, matching paper Table II:

  - ``point_deltas``       : per-message kinematic deltas (Δlat, Δlon, Δalt,
                             Δvelocity, Δheading, Δt). Used by Isolation
                             Forest and XGBoost-point. Point-level samples.
  - ``trajectory_stats``   : per-trajectory aggregated statistics. Used by
                             the optional XGBoost-Traj variant. Trajectory-
                             level samples.
  - ``sequence_tensor``    : padded ``[T, F]`` normalised tensor. Used by
                             the LSTM baseline. Trajectory-level samples.
"""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from typing import Sequence

import numpy as np
import pandas as pd

from .rules import LAT, LON, ALT, VEL, HDG, TS

POINT_DELTA_COLS: tuple[str, ...] = (
    "dlat", "dlon", "dalt", "dv", "dheading", "dt_s",
    "implied_speed_mps", "abs_velocity", "abs_altitude",
)

TRAJ_STAT_COLS: tuple[str, ...] = (
    "n_points", "duration_s",
    "alt_min", "alt_max", "alt_range", "alt_std",
    "vel_min", "vel_max", "vel_mean", "vel_std",
    "hdg_std", "hdg_max_delta",
    "dlat_abs_max", "dlon_abs_max",
    "dv_abs_max", "dv_abs_mean",
    "implied_speed_max", "implied_speed_mean",
    "dt_max", "dt_min", "dt_std",
)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1, p2 = radians(lat1), radians(lat2)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(p1) * cos(p2) * sin(dlon / 2) ** 2
    return 2 * r * asin(sqrt(a))


def _heading_delta(h1: float, h2: float) -> float:
    d = abs(h2 - h1) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _dt_s(prev_ts, curr_ts) -> float:
    return max(1e-6, (pd.to_datetime(curr_ts) - pd.to_datetime(prev_ts)).total_seconds())


def point_deltas(traj: Sequence[Sequence]) -> np.ndarray:
    """Return an ``(N-1, len(POINT_DELTA_COLS))`` array of per-message deltas.

    The first row of the trajectory has no predecessor and is therefore not
    included. NaNs in the source data are passed through as 0.0 (matches
    the cleaning pipeline upstream).
    """
    if len(traj) < 2:
        return np.zeros((0, len(POINT_DELTA_COLS)), dtype=np.float32)

    rows: list[list[float]] = []
    for prev, curr in zip(traj[:-1], traj[1:]):
        dt = _dt_s(prev[TS], curr[TS])
        dlat = float(curr[LAT] - prev[LAT]) if curr[LAT] is not None else 0.0
        dlon = float(curr[LON] - prev[LON]) if curr[LON] is not None else 0.0
        dalt = float(curr[ALT] - prev[ALT]) if curr[ALT] is not None else 0.0
        dv = float(curr[VEL] - prev[VEL]) if curr[VEL] is not None else 0.0
        dheading = _heading_delta(prev[HDG], curr[HDG]) if curr[HDG] is not None else 0.0
        dist = _haversine_m(prev[LAT], prev[LON], curr[LAT], curr[LON])
        implied = dist / dt
        rows.append([
            dlat, dlon, dalt, dv, dheading, dt,
            implied, float(curr[VEL] or 0.0), float(curr[ALT] or 0.0),
        ])
    return np.asarray(rows, dtype=np.float32)


def trajectory_stats(traj: Sequence[Sequence]) -> np.ndarray:
    """Return a single ``(len(TRAJ_STAT_COLS),)`` feature vector."""
    n = len(traj)
    if n < 2:
        return np.zeros(len(TRAJ_STAT_COLS), dtype=np.float32)

    deltas = point_deltas(traj)
    alts = np.array([p[ALT] for p in traj if p[ALT] is not None], dtype=np.float32)
    vels = np.array([p[VEL] for p in traj if p[VEL] is not None], dtype=np.float32)
    hdgs = np.array([p[HDG] for p in traj if p[HDG] is not None], dtype=np.float32)

    duration_s = (pd.to_datetime(traj[-1][TS]) - pd.to_datetime(traj[0][TS])).total_seconds()

    def _safe(arr: np.ndarray, fn, default: float = 0.0) -> float:
        return float(fn(arr)) if arr.size else default

    hdg_max_delta = 0.0
    if hdgs.size >= 2:
        hdg_deltas = [_heading_delta(hdgs[i - 1], hdgs[i]) for i in range(1, len(hdgs))]
        hdg_max_delta = float(max(hdg_deltas))

    return np.asarray([
        n, duration_s,
        _safe(alts, np.min), _safe(alts, np.max),
        _safe(alts, np.max) - _safe(alts, np.min), _safe(alts, np.std),
        _safe(vels, np.min), _safe(vels, np.max), _safe(vels, np.mean), _safe(vels, np.std),
        _safe(hdgs, np.std), hdg_max_delta,
        float(np.max(np.abs(deltas[:, 0]))) if deltas.size else 0.0,
        float(np.max(np.abs(deltas[:, 1]))) if deltas.size else 0.0,
        float(np.max(np.abs(deltas[:, 3]))) if deltas.size else 0.0,
        float(np.mean(np.abs(deltas[:, 3]))) if deltas.size else 0.0,
        float(np.max(deltas[:, 6])) if deltas.size else 0.0,
        float(np.mean(deltas[:, 6])) if deltas.size else 0.0,
        float(np.max(deltas[:, 5])) if deltas.size else 0.0,
        float(np.min(deltas[:, 5])) if deltas.size else 0.0,
        float(np.std(deltas[:, 5])) if deltas.size else 0.0,
    ], dtype=np.float32)


SEQUENCE_FEATURE_COLS = ("lat", "lon", "alt", "velocity", "heading", "dt_s")


def sequence_tensor(
    traj: Sequence[Sequence],
    max_len: int,
    norm_stats: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a padded ``[max_len, len(SEQUENCE_FEATURE_COLS)]`` tensor.

    Returns the feature tensor and a ``[max_len]`` boolean mask
    (1 = real point, 0 = padding) for downstream masked LSTM training.
    """
    f = len(SEQUENCE_FEATURE_COLS)
    feats = np.zeros((max_len, f), dtype=np.float32)
    mask = np.zeros(max_len, dtype=np.float32)

    upto = min(len(traj), max_len)
    for i in range(upto):
        p = traj[i]
        dt = 0.0 if i == 0 else _dt_s(traj[i - 1][TS], p[TS])
        feats[i] = [
            float(p[LAT] or 0.0),
            float(p[LON] or 0.0),
            float(p[ALT] or 0.0),
            float(p[VEL] or 0.0),
            float(p[HDG] or 0.0),
            float(dt),
        ]
        mask[i] = 1.0

    if norm_stats is not None:
        feats = (feats - norm_stats["mean"]) / np.where(norm_stats["std"] > 1e-6, norm_stats["std"], 1.0)
        feats = feats * mask[:, None]

    return feats, mask


def compute_norm_stats(trajs: Sequence[Sequence[Sequence]], max_len: int) -> dict:
    """Compute per-feature mean/std over real (non-padded) points only."""
    feats_list: list[np.ndarray] = []
    for traj in trajs:
        f, m = sequence_tensor(traj, max_len)
        feats_list.append(f[m.astype(bool)])
    stacked = np.concatenate(feats_list, axis=0) if feats_list else np.zeros((1, len(SEQUENCE_FEATURE_COLS)))
    return {
        "mean": stacked.mean(axis=0).astype(np.float32),
        "std": stacked.std(axis=0).astype(np.float32),
    }

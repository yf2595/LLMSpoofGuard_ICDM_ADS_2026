"""
Inter-rater agreement metrics for the manual analyst validation study.

The bundled ``inter_rater_agreement_trajectories.csv`` contains independent
labels from three aviation-security analysts. Use ``scripts/compute_inter_rater_agreement.py``
to recompute summary statistics and write a provenance record.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RATINGS_PATH = (
    Path(__file__).resolve().parent / "inter_rater_agreement_trajectories.csv"
)
DEFAULT_SUMMARY_PATH = (
    Path(__file__).resolve().parent / "inter_rater_agreement_summary.csv"
)
DEFAULT_PROVENANCE_PATH = (
    Path(__file__).resolve().parent / "inter_rater_agreement_provenance.json"
)

ANALYST_COLUMNS = ("analyst_1", "analyst_2", "analyst_3")
LABELS = ("Clean", "Spoofed")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fleiss_kappa(ratings: np.ndarray, categories: tuple[str, ...] = LABELS) -> float:
    """Fleiss' kappa for nominal ratings (multiple raters, fixed categories)."""
    if ratings.ndim != 2:
        raise ValueError("ratings must be a 2D array (n_items x n_raters)")

    n_items, n_raters = ratings.shape
    if n_items == 0 or n_raters < 2:
        return float("nan")

    cat_index = {label: idx for idx, label in enumerate(categories)}
    counts = np.zeros((n_items, len(categories)), dtype=float)
    for i in range(n_items):
        for label in ratings[i]:
            counts[i, cat_index[str(label)]] += 1.0

    row_totals = counts.sum(axis=1, keepdims=True)
    if not np.all(row_totals == n_raters):
        raise ValueError("each trajectory must have exactly one label per analyst")

    p_i = (np.sum(counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = float(np.mean(p_i))

    p_j = np.sum(counts, axis=0) / (n_items * n_raters)
    p_e = float(np.sum(p_j ** 2))
    if p_e >= 1.0:
        return 1.0
    return (p_bar - p_e) / (1.0 - p_e)


def load_ratings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"trajectory_id", "sample_stratum", *ANALYST_COLUMNS, "full_agreement", "majority_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ratings file missing columns: {sorted(missing)}")
    return df


def compute_summary(df: pd.DataFrame) -> dict[str, float | int]:
    ratings = df[list(ANALYST_COLUMNS)].to_numpy(dtype=str)
    full_agreement = df["full_agreement"].astype(bool)
    flagged = df["sample_stratum"].eq("flagged").sum()
    not_flagged = df["sample_stratum"].eq("not_flagged").sum()

    return {
        "n_trajectories": int(len(df)),
        "n_analysts": len(ANALYST_COLUMNS),
        "n_flagged_spoofed_sample": int(flagged),
        "n_not_flagged_clean_sample": int(not_flagged),
        "n_full_agreement": int(full_agreement.sum()),
        "n_partial_disagreement": int((~full_agreement).sum()),
        "percent_full_agreement": round(100.0 * full_agreement.mean(), 1),
        "fleiss_kappa": round(fleiss_kappa(ratings), 2),
    }


def write_summary_csv(summary: dict[str, float | int], output_path: Path) -> None:
    rows = []
    for key, value in summary.items():
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        rows.append({"metric": key, "value": value})
    pd.DataFrame(rows).to_csv(output_path, index=False)


def build_provenance(
    ratings_path: Path,
    summary: dict[str, float | int],
    *,
    generated_at: str | None = None,
) -> dict:
    return {
        "study": "manual inter-rater agreement on GPS spoofing trajectory labels",
        "generated_at_utc": generated_at or datetime.now(timezone.utc).isoformat(),
        "source_ratings_csv": str(ratings_path.as_posix()),
        "source_sha256": _file_sha256(ratings_path),
        "methodology": (
            "Three independent analysts reviewed ADS-B trajectory visualizations. "
            "Each analyst assigned Clean or Spoofed without seeing other ratings. "
            "The bundled CSV stores those raw labels; this script only aggregates them."
        ),
        "sampling_note": (
            "The study sample contains 500 trajectories pre-flagged by RBH Tier-1 rules "
            "and 500 trajectories not flagged (balanced manual audit set)."
        ),
        "metrics": summary,
    }


def sample_audit_manifest(
    manifest_path: Path,
    *,
    n_per_stratum: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Draw the balanced audit sample used for manual review (IDs only)."""
    manifest = pd.read_csv(manifest_path)
    rng = np.random.default_rng(seed)

    flagged_pool = manifest.loc[manifest["is_spoofed"] == 1, "trajectory_id"].tolist()
    clean_pool = manifest.loc[manifest["is_spoofed"] == 0, "trajectory_id"].tolist()
    if len(flagged_pool) < n_per_stratum:
        raise ValueError(f"Need {n_per_stratum} flagged trajectories, only {len(flagged_pool)} available.")
    if len(clean_pool) < n_per_stratum:
        raise ValueError(f"Need {n_per_stratum} clean trajectories, only {len(clean_pool)} available.")

    flagged = rng.choice(flagged_pool, size=n_per_stratum, replace=False)
    clean = rng.choice(clean_pool, size=n_per_stratum, replace=False)

    rows = [
        {"trajectory_id": tid, "sample_stratum": "flagged", "manifest_is_spoofed": 1}
        for tid in flagged
    ] + [
        {"trajectory_id": tid, "sample_stratum": "not_flagged", "manifest_is_spoofed": 0}
        for tid in clean
    ]
    sample = pd.DataFrame(rows)
    return sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)

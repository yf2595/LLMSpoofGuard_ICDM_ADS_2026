"""
Category-stratified sampling for the prompt ablation study.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.run_evaluation import _load_and_concat, discover_dataset


def _trajectory_from_group(group: pd.DataFrame, min_samples: int) -> list[tuple] | None:
    group = group.sort_values("time_position")
    if len(group) < min_samples:
        return None
    country_val = group["country"].iloc[0] if "country" in group.columns else "Sea"
    model_col = "model" if "model" in group.columns else None
    traj: list[tuple] = []
    for row in group.itertuples(index=False):
        ts = pd.to_datetime(row.time_position, unit="s", utc=True)
        model = getattr(row, "model", "unknown") if model_col else "unknown"
        country = getattr(row, "country", country_val) if "country" in group.columns else country_val
        traj.append(
            (
                row.latitude,
                row.longitude,
                row.baro_altitude,
                country,
                row.velocity,
                row.true_track,
                model,
                ts,
            )
        )
    return traj


def sample_trajectory_ids_from_manifest(
    manifest_path: Path,
    *,
    n_per_class: int = 100,
    seed: int = 42,
) -> tuple[list[str], dict]:
    """Pick trajectory ids for a balanced ablation set before loading CSV rows."""
    manifest = pd.read_csv(manifest_path)
    rng = np.random.default_rng(seed)

    neg_pool = manifest.loc[manifest["is_spoofed"] == 0, "trajectory_id"].tolist()
    pos_df = manifest.loc[manifest["is_spoofed"] == 1]

    if len(neg_pool) < n_per_class:
        raise ValueError(f"Need {n_per_class} negatives, only {len(neg_pool)} available.")
    if len(pos_df) < n_per_class:
        raise ValueError(f"Need {n_per_class} positives, only {len(pos_df)} available.")

    neg_sample = rng.choice(neg_pool, size=n_per_class, replace=False).tolist()

    pos_by_cat: dict[str, list[str]] = {}
    for _, row in pos_df.iterrows():
        cat = row["spoof_category"] if pd.notna(row["spoof_category"]) else "Unknown"
        pos_by_cat.setdefault(str(cat), []).append(row["trajectory_id"])

    per_cat = max(1, n_per_class // len(pos_by_cat))
    pos_selected: list[str] = []
    cat_counts: Counter[str] = Counter()

    for cat in sorted(pos_by_cat):
        pool = list(pos_by_cat[cat])
        rng.shuffle(pool)
        chosen = pool[: min(per_cat, len(pool))]
        pos_selected.extend(chosen)
        cat_counts[cat] += len(chosen)

    remaining = n_per_class - len(pos_selected)
    if remaining > 0:
        leftover = [tid for tid in pos_df["trajectory_id"] if tid not in pos_selected]
        rng.shuffle(leftover)
        extra = leftover[:remaining]
        pos_selected.extend(extra)
        for tid in extra:
            cat = pos_df.loc[pos_df["trajectory_id"] == tid, "spoof_category"].iloc[0]
            cat_counts[str(cat) if pd.notna(cat) else "Unknown"] += 1

    pos_selected = pos_selected[:n_per_class]
    selected_ids = neg_sample + pos_selected
    rng.shuffle(selected_ids)

    meta = {
        "n_per_class": n_per_class,
        "seed": seed,
        "n_total": len(selected_ids),
        "positive_categories": dict(cat_counts),
        "trajectory_ids": selected_ids,
    }
    return selected_ids, meta


def load_trajectories_by_ids(
    dataset_dir: Path,
    trajectory_ids: list[str],
    *,
    min_samples: int = 5,
) -> tuple[list[list[tuple]], list[bool], list[str], list[str | None]]:
    """Load only the requested trajectories from dataset shards."""
    shards, manifest_path = discover_dataset(dataset_dir)
    manifest = pd.read_csv(manifest_path)
    label_by_id = dict(zip(manifest["trajectory_id"], manifest["is_spoofed"].astype(bool)))
    category_by_id = dict(zip(manifest["trajectory_id"], manifest["spoof_category"]))

    wanted = set(trajectory_ids)
    df = _load_and_concat(shards)
    df = df[df["trajectory_id"].isin(wanted)]

    by_id: dict[str, list[tuple]] = {}
    for tid, group in df.groupby("trajectory_id", sort=False):
        traj = _trajectory_from_group(group, min_samples)
        if traj is not None:
            by_id[tid] = traj

    trajectories: list[list[tuple]] = []
    labels: list[bool] = []
    ids: list[str] = []
    categories: list[str | None] = []

    missing: list[str] = []
    for tid in trajectory_ids:
        if tid not in by_id:
            missing.append(tid)
            continue
        trajectories.append(by_id[tid])
        labels.append(label_by_id.get(tid, False))
        ids.append(tid)
        cat = category_by_id.get(tid)
        categories.append(None if pd.isna(cat) else str(cat))

    if missing:
        raise ValueError(f"Could not load {len(missing)} sampled trajectories (min_samples={min_samples}).")

    return trajectories, labels, ids, categories


def load_all_manifest_trajectories(
    dataset_dir: Path,
    *,
    min_samples: int = 5,
) -> tuple[list[list[tuple]], list[bool], dict]:
    """Load every trajectory listed in the benchmark manifest."""
    _, manifest_path = discover_dataset(dataset_dir)
    manifest = pd.read_csv(manifest_path)
    selected_ids = manifest["trajectory_id"].tolist()
    trajectories, labels, ids, categories = load_trajectories_by_ids(
        dataset_dir,
        selected_ids,
        min_samples=min_samples,
    )
    pos_by_cat = Counter(
        cat for cat, label in zip(categories, labels) if label and cat is not None
    )
    info = {
        "dataset_dir": str(dataset_dir),
        "label_policy": "dataset manifest (is_spoofed)",
        "sampling": "full_benchmark",
        "n_total": len(ids),
        "n_positive": sum(labels),
        "n_negative": len(labels) - sum(labels),
        "positive_categories": dict(pos_by_cat),
        "trajectory_ids": ids,
        "categories": categories,
    }
    return trajectories, labels, info


def load_ablation_sample(
    dataset_dir: Path,
    *,
    n_per_class: int = 100,
    seed: int = 42,
    min_samples: int = 5,
    full_benchmark: bool = False,
) -> tuple[list[list[tuple]], list[bool], dict]:
    """Load trajectories for the prompt ablation study."""
    if full_benchmark:
        return load_all_manifest_trajectories(dataset_dir, min_samples=min_samples)

    _, manifest_path = discover_dataset(dataset_dir)
    selected_ids, meta = sample_trajectory_ids_from_manifest(
        Path(manifest_path),
        n_per_class=n_per_class,
        seed=seed,
    )
    trajectories, labels, ids, categories = load_trajectories_by_ids(
        dataset_dir,
        selected_ids,
        min_samples=min_samples,
    )
    info = {
        "dataset_dir": str(dataset_dir),
        "label_policy": "dataset manifest (is_spoofed)",
        "sampling": "balanced_category_stratified",
        **meta,
        "categories": categories,
    }
    return trajectories, labels, info

"""
Print summary statistics for the LLMSpoofGuard benchmark dataset.

Loads ``trajectory_manifest.csv`` and the monthly ``llmspoofguard_*.csv``
shards under ``data/dataset/`` and prints counts for reviewers (messages,
trajectories, per-country, per-day, spoof prevalence, per-category).

Usage:
    python scripts/dataset_stats.py
    python scripts/dataset_stats.py --dataset-dir data/dataset --top 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _find_shards(dataset_dir: Path) -> list[Path]:
    index_path = dataset_dir / "dataset_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = [dataset_dir / name for name in index.get("shards", [])]
        missing = [p for p in shards if not p.exists()]
        if not missing:
            return sorted(shards)
    return sorted(dataset_dir.glob("llmspoofguard_*.csv"))


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _print_series(label: str, series: pd.Series, top: int | None = None) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    if series.empty:
        print("  (none)")
        return
    total = series.sum()
    items = series.items() if top is None else series.head(top).items()
    for name, count in items:
        pct = 100.0 * count / total if total else 0.0
        print(f"  {name!s:<40} {count:>10,}  ({pct:5.1f}%)")
    if top is not None and len(series) > top:
        other = series.iloc[top:].sum()
        print(f"  {'(other)':<40} {other:>10,}  ({100.0 * other / total:5.1f}%)")


def load_points(dataset_dir: Path) -> pd.DataFrame:
    shards = _find_shards(dataset_dir)
    if not shards:
        raise FileNotFoundError(f"No llmspoofguard_*.csv shards in {dataset_dir}")

    usecols = [
        "time_position",
        "country",
        "trajectory_id",
        "is_spoofed",
        "spoof_category",
        "icao24",
    ]
    frames: list[pd.DataFrame] = []
    for path in shards:
        df = pd.read_csv(path, usecols=usecols)
        frames.append(df)
    points = pd.concat(frames, ignore_index=True)
    points["timestamp"] = pd.to_datetime(points["time_position"], unit="s", utc=True)
    points["date"] = points["timestamp"].dt.date
    return points


def load_manifest(dataset_dir: Path) -> pd.DataFrame:
    path = dataset_dir / "trajectory_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = pd.read_csv(path)
    manifest["start_time"] = pd.to_datetime(manifest["start_time"], utc=True)
    manifest["end_time"] = pd.to_datetime(manifest["end_time"], utc=True)
    manifest["start_date"] = manifest["start_time"].dt.date
    return manifest


def print_stats(dataset_dir: Path, top: int, full_days: bool) -> None:
    manifest = load_manifest(dataset_dir)
    points = load_points(dataset_dir)

    n_messages = len(points)
    n_trajectories = len(manifest)
    n_icao24 = manifest["icao24"].nunique()
    n_spoofed_traj = int(manifest["is_spoofed"].sum())
    n_spoofed_msg = int(points["is_spoofed"].sum())

    _section("LLMSpoofGuard benchmark dataset - overview")
    print(f"  Dataset directory     : {dataset_dir.resolve()}")
    print(f"  ADS-B messages (rows) : {n_messages:,}")
    print(f"  Trajectory segments   : {n_trajectories:,}")
    print(f"  Unique aircraft icao24: {n_icao24:,}")
    print(f"  Points per trajectory : "
          f"min={manifest['n_points'].min()}, "
          f"median={int(manifest['n_points'].median())}, "
          f"max={manifest['n_points'].max()}, "
          f"mean={manifest['n_points'].mean():.1f}")
    print(f"  Collection window     : "
          f"{points['timestamp'].min().date()} .. {points['timestamp'].max().date()}")
    print(f"  Spoofed trajectories  : {n_spoofed_traj:,} "
          f"({100.0 * n_spoofed_traj / n_trajectories:.2f}%)")
    print(f"  Spoofed messages      : {n_spoofed_msg:,} "
          f"({100.0 * n_spoofed_msg / n_messages:.2f}%)")

    _section("Spoofing labels (trajectory level)")
    clean = manifest[manifest["is_spoofed"] == 0]
    spoofed = manifest[manifest["is_spoofed"] == 1]
    print(f"  Clean trajectories  : {len(clean):,}")
    print(f"  Spoofed trajectories: {len(spoofed):,}")
    if len(spoofed):
        _print_series(
            "Spoofed trajectories by category",
            spoofed["spoof_category"].value_counts(),
        )

    points["month"] = points["timestamp"].dt.strftime("%Y-%m")
    manifest["start_month"] = manifest["start_time"].dt.strftime("%Y-%m")

    _section("ADS-B messages per month")
    _print_series("Messages", points.groupby("month", sort=True).size())

    _section("Trajectory segments per month (by segment start_time)")
    _print_series("Trajectories", manifest.groupby("start_month", sort=True).size())
    spoofed_month = manifest[manifest["is_spoofed"] == 1].groupby("start_month", sort=True).size()
    _print_series("Spoofed trajectories", spoofed_month)

    _section("ADS-B messages per day")
    msgs_by_day = points.groupby("date", sort=True).size()
    day_top = None if full_days else top
    _print_series("Messages", msgs_by_day, top=day_top)
    print(f"\n  Total distinct days : {msgs_by_day.index.nunique()}")

    _section("Trajectory segments per day (by segment start_time)")
    traj_by_day = manifest.groupby("start_date", sort=True).size()
    _print_series("Trajectories", traj_by_day, top=day_top)
    spoofed_by_day = manifest[manifest["is_spoofed"] == 1].groupby("start_date", sort=True).size()
    print(f"\n  Spoofed trajectories per day:")
    _print_series("Spoofed trajectories", spoofed_by_day, top=day_top)

    _section("ADS-B messages per country (overflown)")
    msgs_by_country = points["country"].fillna("Unknown").value_counts()
    _print_series("Messages", msgs_by_country, top=top)

    _section("Trajectory segments per country (majority country in segment)")
    # Majority country per trajectory from point counts
    pt_country = (
        points.groupby(["trajectory_id", "country"], observed=True)
        .size()
        .reset_index(name="n")
    )
    idx = pt_country.groupby("trajectory_id")["n"].idxmax()
    traj_country = pt_country.loc[idx].set_index("trajectory_id")["country"]
    manifest_country = manifest.set_index("trajectory_id").join(
        traj_country.rename("majority_country")
    )
    traj_by_country = manifest_country["majority_country"].fillna("Unknown").value_counts()
    _print_series("Trajectories", traj_by_country, top=top)

    spoofed_manifest = manifest_country[manifest_country["is_spoofed"] == 1]
    if len(spoofed_manifest):
        _section("Spoofed trajectories per country (majority country)")
        _print_series(
            "Spoofed trajectories",
            spoofed_manifest["majority_country"].fillna("Unknown").value_counts(),
            top=top,
        )

    _section("Cross-check (index file vs computed)")
    index_path = dataset_dir / "dataset_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        print(f"  dataset_index.json segments : {index.get('n_segments', '?'):,}")
        print(f"  dataset_index.json points   : {index.get('n_points', '?'):,}")
        print(f"  dataset_index.json spoofed  : {index.get('n_spoofed', '?'):,}")
        print(f"  dataset_index.json rate     : {index.get('spoof_rate', '?')}")
        if index.get("n_segments") != n_trajectories:
            print("  WARNING: segment count mismatch vs manifest")
        if index.get("n_points") != n_messages:
            print("  WARNING: point count mismatch vs shards")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description="Print statistics for the LLMSpoofGuard benchmark dataset.",
    )
    p.add_argument(
        "--dataset-dir",
        default=str(root / "data" / "dataset"),
        help="Directory containing trajectory_manifest.csv and shards.",
    )
    p.add_argument(
        "--top",
        type=int,
        default=25,
        help="Show top-N rows for long breakdown tables (rest summarized as 'other').",
    )
    p.add_argument(
        "--full-days",
        action="store_true",
        help="Print every day in the per-day tables (no top-N truncation).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")
    print_stats(dataset_dir, top=args.top, full_days=args.full_days)


if __name__ == "__main__":
    main()

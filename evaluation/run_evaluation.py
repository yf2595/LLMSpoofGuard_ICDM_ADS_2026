"""
End-to-end evaluation harness for the LLMSpoofGuard reproducibility package.

Pipeline (paper Table II):

  1. Load an ADS-B CSV file.
  2. Run the standard cleaning + trajectory-segmentation pipeline from
     ``src.preprocessing``.
  3. Auto-label every trajectory using the RBH oracle in
     ``evaluation/labels.py``.
  4. Split trajectories 80/20 stratified by label.
  5. Fit each supervised baseline (XGBoost-point, XGBoost-Traj, LSTM) on
     the train split.
  6. Score every baseline (including the unsupervised RBH, Isolation
     Forest, and the few-shot LLM) on the test split.
  7. Write metrics to ``evaluation/results/results.json`` and a flat CSV.

Usage:
    python scripts/run_benchmark.py

    # or explicitly:
    python -m evaluation.run_evaluation --dataset-dir data/dataset --skip-llm
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

# Allow running as a script: add the package root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from evaluation.baselines import (  # noqa: E402
    IsolationForestBaseline,
    LLMBaseline,
    LSTMBaseline,
    RBHBaseline,
    XGBoostPointBaseline,
    XGBoostTrajBaseline,
)
from evaluation.baselines.base import Baseline  # noqa: E402
from evaluation.labels import (  # noqa: E402
    trajectory_labels,
    trajectory_labels_excluding_heading,
)
from evaluation.metrics import compute_metrics  # noqa: E402
from evaluation.plot_results import METHOD_DISPLAY_NAMES, plot_all  # noqa: E402
from evaluation.splits import balanced_test_split, gather, stratified_split  # noqa: E402
from src.preprocessing import (  # noqa: E402
    clean_adsb_dataframe,
    dataframe_to_paths_dict,
    flatten_trajectories,
    split_trajectories,
)

logger = logging.getLogger(__name__)

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent / "data" / "dataset"


def discover_dataset(dataset_dir: Path) -> tuple[list[str], str]:
    """Return (shard_csv_paths, manifest_path) for the bundled benchmark."""
    dataset_dir = Path(dataset_dir)
    index_path = dataset_dir / "dataset_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = [str(dataset_dir / name) for name in index.get("shards", [])]
    else:
        shards = sorted(str(p) for p in dataset_dir.glob("llmspoofguard_*.csv"))

    manifest = str(dataset_dir / "trajectory_manifest.csv")
    if not Path(manifest).exists():
        raise FileNotFoundError(f"Missing trajectory manifest: {manifest}")
    if not shards:
        raise FileNotFoundError(f"No llmspoofguard_*.csv shards in {dataset_dir}")
    return shards, manifest


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _load_and_concat(csv_paths: Sequence[str | Path]) -> pd.DataFrame:
    """Read one or more CSVs, tag each with its source filename, and concat.

    Each input file represents a separate capture window (the filename
    encodes the start timestamp, e.g. ``2025_11_28_13_27_32.csv``).
    The same ``icao24`` may legitimately appear in more than one file
    because aircraft fly across capture windows; we keep all rows and
    rely on the downstream time-gap splitter
    (``split_trajectories``, default ``max_time_diff = 3 min``) to break
    the per-aircraft stream into chronological flight segments.
    """
    frames: list[pd.DataFrame] = []
    for p in csv_paths:
        p = Path(p)
        logger.info("Loading %s ...", p)
        df = pd.read_csv(p)
        df["source_file"] = p.name
        frames.append(df)
        logger.info("  -> %d raw rows", len(df))

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined %d files into %d raw ADS-B rows.", len(csv_paths), len(combined))
    if len(csv_paths) > 1:
        cross_date = (
            combined.groupby("icao24")["source_file"].nunique().gt(1).sum()
        )
        logger.info(
            "%d unique icao24 codes appear in more than one capture window "
            "- chronological segmentation will keep their flights separate.",
            int(cross_date),
        )
    return combined


def load_and_segment(
    csv_paths: Sequence[str | Path],
    focus_countries: Sequence[str] | None,
    max_trajectories: int | None,
    min_samples: int,
) -> list[list[tuple]]:
    """Multi-CSV -> cleaned, segmented trajectory list (one segment per item)."""
    df = _load_and_concat(csv_paths)

    df = clean_adsb_dataframe(df, focus_countries=focus_countries)
    logger.info("After cleaning: %d rows.", len(df))

    paths_dict = dataframe_to_paths_dict(df)
    logger.info("Grouped into %d aircraft.", len(paths_dict))

    segmented = split_trajectories(paths_dict, min_samples=min_samples)
    flat = flatten_trajectories(segmented)
    logger.info("Produced %d trajectory segments.", len(flat))

    if max_trajectories is not None and len(flat) > max_trajectories:
        flat = flat[:max_trajectories]
        logger.info("Trimmed to %d trajectory segments (--max-trajectories).", len(flat))

    return flat


def load_from_manifest(
    csv_paths: Sequence[str | Path],
    manifest_path: str | Path,
    max_trajectories: int | None,
    min_samples: int,
) -> tuple[list[list[tuple]], list[bool]]:
    """Load pre-segmented trajectories and manifest ground-truth labels.

    Expects row-level CSV shards from ``data/dataset/`` (columns include
    ``trajectory_id``, ``is_spoofed``, ``spoof_category``).
    """
    manifest = pd.read_csv(manifest_path)
    label_by_id = dict(zip(manifest["trajectory_id"], manifest["is_spoofed"].astype(bool)))

    df = _load_and_concat(csv_paths)
    required = {"trajectory_id", "latitude", "longitude", "baro_altitude", "true_track", "velocity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset CSV missing columns: {sorted(missing)}")

    trajectories: list[list[tuple]] = []
    labels: list[bool] = []

    for tid, group in df.groupby("trajectory_id", sort=False):
        group = group.sort_values("time_position")
        if len(group) < min_samples:
            continue
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
        trajectories.append(traj)
        labels.append(label_by_id.get(tid, False))

    logger.info(
        "Loaded %d trajectories from manifest %s (%d labelled positive).",
        len(trajectories),
        manifest_path,
        sum(labels),
    )

    if max_trajectories is not None and len(trajectories) > max_trajectories:
        trajectories = trajectories[:max_trajectories]
        labels = labels[:max_trajectories]
        logger.info("Trimmed to %d trajectories (--max-trajectories).", len(trajectories))

    return trajectories, labels


def build_baselines(args: argparse.Namespace) -> list[Baseline]:
    selected: list[Baseline] = []
    if not args.skip_rbh:
        selected.append(RBHBaseline())
    if not args.skip_iforest:
        selected.append(IsolationForestBaseline())
    if not args.skip_xgb_point:
        selected.append(XGBoostPointBaseline())
    if not args.skip_xgb_traj:
        selected.append(XGBoostTrajBaseline())
    if not args.skip_lstm:
        selected.append(LSTMBaseline(max_len=args.lstm_max_len, epochs=args.lstm_epochs))
    if not args.skip_llm:
        selected.append(LLMBaseline(
            model_name=args.llm_model,
            raw_output_path=args.llm_raw_output,
        ))
    return selected


def evaluate(
    trajectories: list[list[tuple]],
    baselines: list[Baseline],
    test_size: float,
    seed: int,
    balanced_test_per_class: int | None = None,
    exclude_heading_label: bool = False,
    manifest_labels: list[bool] | None = None,
) -> dict:
    if manifest_labels is not None:
        labels = manifest_labels
        label_policy = "dataset manifest (is_spoofed)"
        logger.info("Using manifest ground-truth labels ...")
    elif exclude_heading_label:
        logger.info(
            "Generating RBH ground-truth labels (excluding heading change "
            "and unstable altitude) ..."
        )
        labels = trajectory_labels_excluding_heading(trajectories)
        label_policy = "RBH excluding heading change and unstable altitude"
    else:
        logger.info("Generating RBH ground-truth labels ...")
        labels = trajectory_labels(trajectories)
        label_policy = "RBH all rules"
    pos = sum(labels)
    logger.info("Positives=%d / Total=%d (%.2f%%).", pos, len(labels), 100.0 * pos / max(1, len(labels)))

    if balanced_test_per_class is not None:
        logger.info(
            "Building class-balanced test set: %d positives + %d negatives.",
            balanced_test_per_class, balanced_test_per_class,
        )
        train_idx, test_idx = balanced_test_split(
            trajectories, labels, n_per_class=balanced_test_per_class, seed=seed,
        )
        split_strategy = f"balanced ({balanced_test_per_class}/class)"
    else:
        train_idx, test_idx = stratified_split(
            trajectories, labels, test_size=test_size, seed=seed,
        )
        split_strategy = f"stratified ({test_size:.0%} test)"

    train_trajs = gather(trajectories, train_idx)
    train_labels = gather(labels, train_idx)
    test_trajs = gather(trajectories, test_idx)
    test_labels = gather(labels, test_idx)
    test_pos = sum(test_labels)
    logger.info(
        "Train size=%d (pos=%d), Test size=%d (pos=%d, neg=%d) [%s].",
        len(train_trajs), sum(train_labels),
        len(test_trajs), test_pos, len(test_labels) - test_pos,
        split_strategy,
    )

    results: dict = {
        "dataset": {
            "n_trajectories": len(trajectories),
            "n_positive": pos,
            "n_negative": len(labels) - pos,
            "label_policy": label_policy,
            "benchmark": "conservative known-pattern proxy",
            "test_size": test_size,
            "seed": seed,
            "split_strategy": split_strategy,
            "n_test": len(test_trajs),
            "n_test_positive": test_pos,
            "n_test_negative": len(test_labels) - test_pos,
        },
        "methods": {},
    }

    for baseline in baselines:
        logger.info("=== %s ===", baseline.name)
        if baseline.is_supervised:
            baseline.fit(train_trajs, train_labels)
        else:
            baseline.fit(train_trajs)

        preds = baseline.predict(test_trajs)
        metrics = compute_metrics(test_labels, preds)
        metrics["is_supervised"] = baseline.is_supervised
        metrics["granularity"] = baseline.granularity
        logger.info(
            "%s -> acc=%.3f prec=%.3f rec=%.3f f1=%.3f",
            baseline.name, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"],
        )
        results["methods"][baseline.name] = metrics

    return results


def write_outputs(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote JSON results to %s", output_path)

    csv_path = output_path.with_suffix(".csv")
    rows = []
    for name, m in results["methods"].items():
        display = METHOD_DISPLAY_NAMES.get(name, name)
        rows.append({
            "method": display,
            "accuracy": round(m["accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "TP": m["TP"], "FP": m["FP"], "TN": m["TN"], "FN": m["FN"],
            "n": m["n"],
            "supervised": m["is_supervised"],
            "granularity": m["granularity"],
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("Wrote CSV summary to %s", csv_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate detection baselines on the LLMSpoofGuard benchmark.",
    )
    p.add_argument(
        "--dataset-dir",
        default=None,
        help="Load all monthly shards and trajectory_manifest.csv from this folder "
             f"(default when omitted: {DEFAULT_DATASET_DIR.relative_to(DEFAULT_DATASET_DIR.parent.parent)} "
             "if it exists).",
    )
    p.add_argument("--csv", nargs="+", default=None,
                   help="One or more ADS-B CSV paths (overrides --dataset-dir shard list).")
    p.add_argument("--manifest", default=None,
                   help="Path to trajectory_manifest.csv (default: <dataset-dir>/trajectory_manifest.csv).")
    p.add_argument(
        "--rbh-oracle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use RBH Tier-1 rules as ground truth (default: on). "
             "Disable to use manifest is_spoofed labels instead.",
    )
    p.add_argument("--output", default="evaluation/results/results.json")
    p.add_argument("--focus-countries", nargs="*", default=None,
                   help="If provided, keep only aircraft that crossed at least one country.")
    p.add_argument("--max-trajectories", type=int, default=None,
                   help="Truncate to the first N trajectory segments (useful for smoke tests).")
    p.add_argument("--min-samples", type=int, default=5,
                   help="Minimum points per trajectory segment.")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--balanced-test", type=int, default=None,
                   help="If set, build a class-balanced test set with N positives "
                        "+ N negatives (total 2N). Overrides --test-size. "
                        "Useful for evaluating the LLM on a balanced sample.")
    p.add_argument("--exclude-heading-label", action="store_true",
                   help="Ground-truth positives use RBH rules except "
                        "'Unrealistic heading change' and 'Unstable altitude' "
                        "(unstable altitude rule is not applied in RBH).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lstm-max-len", type=int, default=64)
    p.add_argument("--lstm-epochs", type=int, default=8)
    p.add_argument("--llm-model", default=None, help="OpenAI model name (defaults to env MODEL_NAME).")
    p.add_argument("--llm-raw-output", default=None,
                   help="If set, dump raw LLM responses to this JSONL path "
                        "(one record per trajectory, includes parsing status).")
    p.add_argument("--skip-rbh", action="store_true")
    p.add_argument("--skip-iforest", action="store_true")
    p.add_argument("--skip-xgb-point", action="store_true")
    p.add_argument("--skip-xgb-traj", action="store_true")
    p.add_argument("--skip-lstm", action="store_true")
    p.add_argument("--skip-llm", action="store_true",
                   help="Skip the LLM baseline (avoids OpenAI cost).")
    p.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write confusion matrices, metric bars, and table figure (default: on).",
    )
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    _setup_logging(args.verbose)

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else DEFAULT_DATASET_DIR
    if args.csv is None and dataset_dir.is_dir():
        args.csv, default_manifest = discover_dataset(dataset_dir)
        if args.manifest is None:
            args.manifest = default_manifest
        logger.info("Using benchmark in %s (%d shards).", dataset_dir, len(args.csv))

    if args.manifest:
        if not args.csv:
            raise SystemExit("Provide --csv paths or an existing --dataset-dir.")
        trajectories, manifest_labels = load_from_manifest(
            csv_paths=args.csv,
            manifest_path=args.manifest,
            max_trajectories=args.max_trajectories,
            min_samples=args.min_samples,
        )
        if args.rbh_oracle:
            label_kw = {"exclude_heading_label": args.exclude_heading_label}
        else:
            label_kw = {"manifest_labels": manifest_labels}
    else:
        if not args.csv:
            raise SystemExit("Provide --dataset-dir, --csv, or place data under data/dataset/.")
        trajectories = load_and_segment(
            csv_paths=args.csv,
            focus_countries=args.focus_countries,
            max_trajectories=args.max_trajectories,
            min_samples=args.min_samples,
        )
        label_kw = {"exclude_heading_label": args.exclude_heading_label}

    baselines = build_baselines(args)
    results = evaluate(
        trajectories=trajectories,
        baselines=baselines,
        test_size=args.test_size,
        seed=args.seed,
        balanced_test_per_class=args.balanced_test,
        **label_kw,
    )

    output_path = Path(args.output)
    write_outputs(results, output_path)

    if args.plot:
        logger.info("Rendering plots ...")
        plots_dir = output_path.parent / "plots"
        for plot_path in plot_all(results, plots_dir):
            logger.info("Wrote plot %s", plot_path)


if __name__ == "__main__":
    main()

"""
Recompute inter-rater agreement summary statistics from analyst ratings.

The bundled ``evaluation/inter_rater_agreement_trajectories.csv`` stores
independent Clean/Spoofed labels from three analysts. This script aggregates
those labels (Fleiss' kappa, full-agreement rate) and writes provenance metadata
showing the summary was derived from the ratings file.

Usage:
    python scripts/compute_inter_rater_agreement.py

    # Reproduce the balanced 500+500 audit sample list (trajectory IDs only)
    python scripts/compute_inter_rater_agreement.py --write-sample-manifest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.inter_rater_agreement import (  # noqa: E402
    DEFAULT_PROVENANCE_PATH,
    DEFAULT_RATINGS_PATH,
    DEFAULT_SUMMARY_PATH,
    build_provenance,
    compute_summary,
    load_ratings,
    sample_audit_manifest,
    write_summary_csv,
)
from evaluation.run_evaluation import DEFAULT_DATASET_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description="Aggregate manual analyst ratings into summary statistics.",
    )
    p.add_argument(
        "--ratings",
        default=str(DEFAULT_RATINGS_PATH),
        help="CSV with analyst_1..analyst_3 columns (default: bundled study file).",
    )
    p.add_argument(
        "--summary-out",
        default=str(DEFAULT_SUMMARY_PATH),
        help="Output path for metric/value summary CSV.",
    )
    p.add_argument(
        "--provenance-out",
        default=str(DEFAULT_PROVENANCE_PATH),
        help="Output path for provenance JSON (SHA-256 of ratings + metrics).",
    )
    p.add_argument(
        "--write-sample-manifest",
        action="store_true",
        help="Write the balanced 500+500 audit trajectory-id list from the manifest.",
    )
    p.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    p.add_argument("--n-per-stratum", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--sample-manifest-out",
        default=str(root / "evaluation" / "inter_rater_agreement_sample_manifest.csv"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ratings_path = Path(args.ratings)
    if not ratings_path.exists():
        raise SystemExit(f"Ratings file not found: {ratings_path}")

    df = load_ratings(ratings_path)
    summary = compute_summary(df)
    write_summary_csv(summary, Path(args.summary_out))

    provenance = build_provenance(ratings_path, summary)
    provenance_path = Path(args.provenance_out)
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"Wrote {args.summary_out}")
    print(f"Wrote {args.provenance_out}")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if args.write_sample_manifest:
        manifest_path = Path(args.dataset_dir) / "trajectory_manifest.csv"
        sample = sample_audit_manifest(
            manifest_path,
            n_per_stratum=args.n_per_stratum,
            seed=args.seed,
        )
        out = Path(args.sample_manifest_out)
        sample.to_csv(out, index=False)
        print(f"Wrote audit sample manifest ({len(sample)} trajectories) to {out}")


if __name__ == "__main__":
    main()

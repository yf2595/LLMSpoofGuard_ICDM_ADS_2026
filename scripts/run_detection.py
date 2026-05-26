"""
CLI entry point: full LLM detection + spoofing zone generation pipeline.

Replaces the production ``Server/Backend_Process.py`` entry point but
removes every Mongo coupling. The pipeline is:

    CSV  ->  clean + segment trajectories
         ->  LLM detection (paper-only prompt)
         ->  detections.jsonl + zones.json

Example:
    python scripts/run_detection.py --csv data/dataset/llmspoofguard_2025_01.csv \
        --output-dir evaluation/results/run1 \
        --focus-countries Israel Turkey Russia \
        --max-trajectories 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

from src.detection_llm import DEFAULT_MODEL, detect_batch  # noqa: E402
from src.preprocessing import (  # noqa: E402
    clean_adsb_dataframe,
    dataframe_to_paths_dict,
    flatten_trajectories,
    split_trajectories,
)
from src.zones import build_spoofing_polygons  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LLM detection + zone generation on an ADS-B CSV.")
    p.add_argument("--csv", required=True, help="Input ADS-B CSV path.")
    p.add_argument("--output-dir", default="evaluation/results/detection_run")
    p.add_argument("--focus-countries", nargs="*", default=None,
                   help="If provided, keep only aircraft that crossed at least one country.")
    p.add_argument("--max-trajectories", type=int, default=None)
    p.add_argument("--min-samples", type=int, default=5)
    p.add_argument("--model", default=None, help="OpenAI model name (defaults to env MODEL_NAME).")
    p.add_argument("--max-workers", type=int, default=10)
    p.add_argument("--eps-km", type=float, default=250.0, help="DBSCAN neighbourhood radius (km).")
    p.add_argument("--dbscan-min-samples", type=int, default=5,
                   help="DBSCAN minimum points per zone.")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    df = pd.read_csv(args.csv)
    logger.info("Loaded %d raw ADS-B rows from %s.", len(df), args.csv)

    df = clean_adsb_dataframe(df, focus_countries=args.focus_countries)
    paths = dataframe_to_paths_dict(df)
    segmented = split_trajectories(paths, min_samples=args.min_samples)
    flat = flatten_trajectories(segmented)
    if args.max_trajectories:
        flat = flat[: args.max_trajectories]
    logger.info("Prepared %d trajectory segments for LLM detection.", len(flat))

    if not flat:
        logger.warning("No trajectories to process. Exiting.")
        return

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model_name = args.model or os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    logger.info("Calling LLM (%s) with %d workers ...", model_name, args.max_workers)

    raw_responses = detect_batch(
        flat,
        client=client,
        model_name=model_name,
        max_workers=args.max_workers,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detections_path = output_dir / "detections.jsonl"
    with detections_path.open("w", encoding="utf-8") as f:
        for r in raw_responses:
            if r:
                f.write(r.strip() + "\n")
    logger.info("Wrote %d detection responses to %s.", len(raw_responses), detections_path)

    logger.info("Building spoofing zones (DBSCAN eps=%s km, min_samples=%d)...",
                args.eps_km, args.dbscan_min_samples)
    zones = build_spoofing_polygons(
        raw_responses,
        eps_km=args.eps_km,
        min_samples=args.dbscan_min_samples,
    )

    zones_path = output_dir / "zones.json"
    with zones_path.open("w", encoding="utf-8") as f:
        json.dump(zones, f, indent=2, default=str)
    logger.info("Wrote %d spoofing zones to %s.", len(zones), zones_path)


if __name__ == "__main__":
    main()

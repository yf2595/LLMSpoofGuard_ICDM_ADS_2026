"""
CLI entry point: collect live ADS-B data from the OpenSky Network.

Example:
    python scripts/collect_adsb.py --output data/live.csv --iterations 45 --interval 115
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from src.collector import track_flights  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect live ADS-B data from OpenSky.")
    p.add_argument("--output", required=True, help="Destination CSV path.")
    p.add_argument("--url", default=None,
                   help="OpenSky endpoint. Defaults to env OSN_URL or "
                        "https://opensky-network.org/api/states/all")
    p.add_argument("--interval", type=int, default=None,
                   help="Seconds between fetches. Defaults to env INTERVAL or 115.")
    p.add_argument("--iterations", type=int, default=None,
                   help="Number of successful fetches. Defaults to env ITERATION or 45.")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    url = args.url or os.environ.get("OSN_URL") or "https://opensky-network.org/api/states/all"
    interval = args.interval or int(os.environ.get("INTERVAL", "115"))
    iterations = args.iterations or int(os.environ.get("ITERATION", "45"))

    track_flights(url=url, output_path=args.output, interval=interval, iterations=iterations)


if __name__ == "__main__":
    main()

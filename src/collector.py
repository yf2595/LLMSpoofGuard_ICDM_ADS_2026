"""
OpenSky Network ADS-B collector for the LLMSpoofGuard reproducibility pipeline.

Periodically fetches live aircraft state vectors from the OpenSky Network
public REST API, accumulates them into a single CSV file, and writes the
result atomically. No remote upload, no production-side logic - this is the
minimal collector used by the paper experiments.

Reference: paper Section V-A "Data Collection".
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ADSB_COLUMNS = [
    "icao24", "callsign", "origin_country", "time_position", "last_contact",
    "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
    "true_track", "vertical_rate", "sensors", "geo_altitude", "squawk",
    "spi", "position_source",
]

logger = logging.getLogger(__name__)


def fetch_state_vectors(url: str, timeout: int = 30) -> pd.DataFrame | None:
    """Fetch a single batch of live ADS-B state vectors from OpenSky.

    Args:
        url: OpenSky ``/states/all`` endpoint.
        timeout: HTTP timeout in seconds.

    Returns:
        A DataFrame with the standard 17 ADS-B columns, or None on failure.
    """
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        logger.error("OpenSky request failed: %s", exc)
        return None

    if response.status_code != 200:
        logger.error("OpenSky returned HTTP %s", response.status_code)
        return None

    payload = response.json()
    states = payload.get("states") or []
    if not states:
        logger.warning("OpenSky returned an empty state vector list.")
        return pd.DataFrame(columns=ADSB_COLUMNS)

    return pd.DataFrame(states, columns=ADSB_COLUMNS)


def track_flights(
    url: str,
    output_path: str | Path,
    interval: int = 115,
    iterations: int = 45,
) -> Path:
    """Repeatedly fetch state vectors and write them atomically to a CSV.

    The collector matches the production cadence used by the paper: 45
    iterations spaced ~115 s apart yield ~90 minutes of coverage.

    Args:
        url: OpenSky ``/states/all`` endpoint.
        output_path: Final CSV path. The collector first writes to
            ``<path>.part`` and atomically renames on completion.
        interval: Seconds between consecutive fetches.
        iterations: Number of successful fetches required before writing.

    Returns:
        The final output path.
    """
    output_path = Path(output_path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    frames: list[pd.DataFrame] = []
    success_count = 0

    logger.info(
        "Starting OSN polling: %d iterations every %d s -> %s",
        iterations, interval, output_path,
    )

    while success_count < iterations:
        batch = fetch_state_vectors(url)
        now = datetime.utcnow().isoformat(timespec="seconds")
        if batch is None:
            logger.error("Fetch failed at %s; retrying after interval.", now)
        else:
            frames.append(batch)
            success_count += 1
            logger.info("Fetch %d/%d ok (%d rows).", success_count, iterations, len(batch))

        if success_count < iterations:
            time.sleep(interval)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ADSB_COLUMNS)
    combined.to_csv(tmp_path, index=False)
    os.replace(tmp_path, output_path)
    logger.info("Wrote %d rows to %s.", len(combined), output_path)
    return output_path

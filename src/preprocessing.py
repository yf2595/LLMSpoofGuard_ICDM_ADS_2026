"""
ADS-B preprocessing utilities for the LLMSpoofGuard reproducibility pipeline.

This module produces the same trajectory representation used by the
deployed system (paper Section IV-C):

  - Removes ground-state and missing-field rows.
  - Assigns an overflown country to every position via a spatial join
    against the Natural Earth ``ne_110m_admin_0_countries`` shapefile.
  - Enriches each aircraft with manufacturer/model from the OpenSky
    aircraft database.
  - Groups state vectors into per-aircraft trajectories.
  - Splits trajectories on time gaps > ``max_time_diff`` so that
    receiver-dropout gaps do not appear as kinematic jumps.

The resulting ``dict[icao24, list[list[tuple]]]`` structure is the
canonical input format for every detector in ``evaluation/baselines/``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

logger = logging.getLogger(__name__)

DEFAULT_COUNTRIES_SHP = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "countries"
    / "ne_110m_admin_0_countries.shp"
)
DEFAULT_AIRCRAFT_DB = (
    Path(__file__).resolve().parent.parent / "data" / "aircraftDatabase.csv"
)

_world_cache: gpd.GeoDataFrame | None = None


def _load_world(shapefile: str | Path | None = None) -> gpd.GeoDataFrame:
    """Lazily load the Natural Earth country borders shapefile."""
    global _world_cache
    if _world_cache is not None:
        return _world_cache

    path = Path(shapefile) if shapefile else DEFAULT_COUNTRIES_SHP
    if not path.exists():
        raise FileNotFoundError(
            f"Country borders shapefile not found at {path}. "
            "See ICDM2026/data/README.md for download instructions."
        )
    _world_cache = gpd.read_file(path)
    return _world_cache


def get_country(df: pd.DataFrame, shapefile: str | Path | None = None) -> pd.Series:
    """Assign each row a country name via spatial join. Maritime points -> 'Sea'."""
    world = _load_world(shapefile)
    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    geo_df = gpd.GeoDataFrame(df, geometry=geometry)
    geo_df.set_crs(epsg=4326, inplace=True)
    geo_df = gpd.sjoin(geo_df, world, how="left")
    geo_df["country"] = geo_df["SOVEREIGNT"].fillna("Sea")
    return geo_df["country"]


def enrich_with_model(
    df: pd.DataFrame, model_csv_path: str | Path | None = None
) -> pd.DataFrame:
    """Merge aircraft manufacturer/model strings from the OpenSky aircraft DB.

    If the aircraft DB is missing, every row is enriched with ``"unknown"``
    so that downstream code does not crash. This mirrors the paper's
    statement that metadata enrichment is a soft requirement.
    """
    path = Path(model_csv_path) if model_csv_path else DEFAULT_AIRCRAFT_DB
    df["icao24"] = df["icao24"].astype(str).str.lower()

    if not path.exists():
        logger.warning(
            "aircraftDatabase.csv not found at %s - enriching with 'unknown'.", path
        )
        df["model"] = "unknown"
        return df

    osn_db = pd.read_csv(
        path,
        usecols=["icao24", "manufacturericao", "model"],
        dtype={"icao24": str},
    )
    osn_db["icao24"] = osn_db["icao24"].str.lower()

    df = pd.merge(df, osn_db, how="left", on="icao24")
    df["manufacturericao"] = df["manufacturericao"].fillna("unknown")
    df["model"] = df["model"].fillna("unknown")

    df["model"] = df.apply(
        lambda row: row["manufacturericao"]
        if str(row["model"]).lower() == "unknown"
        else f"{row['manufacturericao']} - {row['model']}",
        axis=1,
    )
    return df.drop(columns=["manufacturericao"])


def clean_adsb_dataframe(
    df: pd.DataFrame, focus_countries: Iterable[str] | None = None
) -> pd.DataFrame:
    """Apply the standard cleaning pipeline used in offline evaluation.

    - Drop rows missing ``time_position``, position, altitude, or heading.
    - Convert Unix timestamps to datetimes.
    - Add a ``country`` column via spatial join.
    - Drop on-ground samples and verbose ADS-B fields not used downstream.
    - If ``focus_countries`` is supplied, keep only aircraft that crossed
      at least one of those countries.
    """
    df = df.dropna(subset=["time_position"]).copy()
    df["time_position"] = pd.to_datetime(
        df["time_position"], unit="s", errors="coerce"
    )
    df = df.sort_values(by=["icao24", "time_position"])
    df["country"] = get_country(df)

    drop_cols = [
        c for c in ("last_contact", "vertical_rate", "sensors", "squawk", "spi")
        if c in df.columns
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df[df["on_ground"] == False]  # noqa: E712
    df = df.dropna(subset=["latitude", "longitude", "baro_altitude", "true_track"])

    if focus_countries:
        focus = list(focus_countries)
        aircraft_over_focus = df[df["country"].isin(focus)]["icao24"].unique()
        df = df[df["icao24"].isin(aircraft_over_focus)]

    return df


def dataframe_to_paths_dict(df: pd.DataFrame) -> dict:
    """Group cleaned ADS-B rows into per-aircraft trajectories.

    Each value in the returned dict is a list of tuples shaped:
    ``(latitude, longitude, altitude, country, velocity, heading, model, timestamp)``.
    """
    df = enrich_with_model(df)
    df_sorted = df.sort_values(by=["icao24", "time_position"])
    return (
        df_sorted.groupby("icao24")
        .apply(
            lambda g: list(
                zip(
                    g["latitude"],
                    g["longitude"],
                    g["baro_altitude"],
                    g["country"],
                    g["velocity"],
                    g["true_track"],
                    g["model"],
                    g["time_position"],
                )
            )
        )
        .to_dict()
    )


def split_trajectories(
    trajectories: dict,
    max_time_diff: pd.Timedelta = pd.Timedelta(minutes=3),
    min_samples: int = 5,
) -> dict[str, list[list[tuple]]]:
    """Split each aircraft's stream on time gaps > ``max_time_diff``.

    The paper requires consistent two-minute sampling; gaps longer than
    that introduce artificial kinematic deltas that would be misread as
    spoofing. Segments shorter than ``min_samples`` are discarded.

    Returns:
        ``{icao24: [segment1, segment2, ...]}`` where each segment is a
        list of ADS-B tuples in chronological order.
    """
    final: dict[str, list[list[tuple]]] = {}

    for icao24, traj in trajectories.items():
        current: list[tuple] = []
        segments: list[list[tuple]] = []

        for point in traj:
            if not current:
                current.append(point)
                continue

            prev_time = pd.to_datetime(current[-1][-1])
            curr_time = pd.to_datetime(point[-1])
            delta = curr_time - prev_time

            if pd.Timedelta(0) < delta <= max_time_diff:
                current.append(point)
            else:
                if len(current) >= min_samples:
                    segments.append(current)
                current = [point]

        if len(current) >= min_samples:
            segments.append(current)

        if segments:
            final[icao24] = segments

    return final


def flatten_trajectories(trajectories: dict[str, list[list[tuple]]]) -> list[list[tuple]]:
    """Flatten the per-aircraft dict into a flat list of trajectory segments."""
    return [seg for segs in trajectories.values() for seg in segs]

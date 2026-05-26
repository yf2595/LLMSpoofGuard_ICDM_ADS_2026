"""
Spoofing zone construction (paper Section IV-E).

Aggregates per-trajectory spoofing detections into geographic spoofing
zones using DBSCAN with a Haversine metric on the spoofing entry/exit
coordinates. Each zone is summarised with a convex hull, centroid,
radius, aircraft involvement, and confidence.

Input is the list of raw JSON strings produced by
``src.detection_llm.detect_batch``; output is a list of zone dicts that
can be serialised to disk or visualised externally.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from typing import Any, Iterable

import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

EARTH_KM = 6371.0088


def haversine_km(coord1: tuple[float, float], coord2: tuple[float, float]) -> float:
    """Great-circle distance in kilometres."""
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return EARTH_KM * 2 * asin(sqrt(a))


def compute_centroid(points: Iterable[tuple[float, float]]) -> list[float]:
    pts = list(points)
    return [float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))]


def build_spoofing_polygons(
    detection_responses: Iterable[str],
    eps_km: float = 250.0,
    min_samples: int = 5,
) -> list[dict[str, Any]]:
    """Cluster spoofing entry/exit points into geographic zones.

    Args:
        detection_responses: Iterable of raw JSON strings produced by the
            LLM detector (each contains ``spoofing_detected`` plus
            ``spoofing_data`` per the prompt schema).
        eps_km: DBSCAN neighbourhood radius in kilometres.
        min_samples: Minimum number of points per zone.

    Returns:
        A list of zone dicts with polygon borders, centroid, radius,
        aircraft involvement and timestamps.
    """
    spoof_points: list[tuple[float, float]] = []
    point_to_entries: dict[tuple[float, float], list[dict]] = defaultdict(list)

    for raw in detection_responses:
        try:
            entry = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if not entry.get("spoofing_detected"):
            continue
        spoofing_data = entry.get("spoofing_data") or {}
        for key in ("spoofing_begin_point", "spoofing_end_point"):
            pt = spoofing_data.get(key)
            if pt:
                coord = (pt["latitude"], pt["longitude"])
                spoof_points.append(coord)
                point_to_entries[coord].append(entry)

    spoof_points = list({tuple(pt) for pt in spoof_points})
    if not spoof_points:
        return []

    coords_rad = np.radians(spoof_points)
    db = DBSCAN(
        eps=eps_km / EARTH_KM,
        min_samples=min_samples,
        algorithm="ball_tree",
        metric="haversine",
    )
    labels = db.fit_predict(coords_rad)

    clusters: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for idx, label in enumerate(labels):
        if label != -1:
            clusters[label].append(spoof_points[idx])

    zones: list[dict[str, Any]] = []
    for cluster_id, cluster_points in clusters.items():
        if len(cluster_points) < min_samples:
            continue

        if len(cluster_points) >= 3:
            hull = ConvexHull(cluster_points)
            borders = [cluster_points[v] for v in hull.vertices]
        else:
            borders = cluster_points

        centroid = compute_centroid(cluster_points)
        radius = max(haversine_km(centroid, pt) for pt in cluster_points)

        related_entry_ids: set[int] = set()
        aircraft_types: list[str] = []
        countries: set[str] = set()
        reasons: list[str] = []
        categories: list[str] = []
        timestamps: list[str] = []
        altitudes: list[float] = []
        confidence_sum = 0.0

        for point in cluster_points:
            for entry in point_to_entries[point]:
                if id(entry) in related_entry_ids:
                    continue
                related_entry_ids.add(id(entry))
                confidence_sum += float(entry.get("confidence") or 0.0)
                aircraft_types.append(entry.get("model") or "unknown")
                spoofing_data = entry.get("spoofing_data") or {}
                if "spoofing_reason" in spoofing_data:
                    reasons.append(spoofing_data["spoofing_reason"])
                if "spoofing_category" in spoofing_data:
                    categories.append(spoofing_data["spoofing_category"])

                for key in ("spoofing_begin_point", "spoofing_end_point"):
                    pt = spoofing_data.get(key)
                    if pt:
                        if pt.get("country"):
                            countries.add(pt["country"])
                        if pt.get("timestamp"):
                            timestamps.append(pt["timestamp"])
                        if pt.get("altitude") is not None:
                            altitudes.append(float(pt["altitude"]))
                for pt in spoofing_data.get("spoofing_locations") or []:
                    if pt.get("country"):
                        countries.add(pt["country"])
                    if pt.get("timestamp"):
                        timestamps.append(pt["timestamp"])
                    if pt.get("altitude") is not None:
                        altitudes.append(float(pt["altitude"]))

        zone: dict[str, Any] = {
            "Polygon id": int(cluster_id + 1),
            "Polygon borders": borders,
            "Center": centroid,
            "Radius (km)": round(radius, 2),
            "Aircraft count": len(related_entry_ids),
            "Aircraft types": sorted(set(aircraft_types)),
            "Countries involved": sorted(countries),
            "Spoofing reasons": reasons,
            "Spoofing categories": categories,
            "Confidence": confidence_sum / max(1, len(related_entry_ids)),
            "Min altitude (m)": round(min(altitudes), 1) if altitudes else None,
            "Max altitude (m)": round(max(altitudes), 1) if altitudes else None,
        }

        parsed_ts = []
        for ts in timestamps:
            try:
                parsed_ts.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
            except (TypeError, ValueError):
                continue
        if parsed_ts:
            zone["Time range"] = [
                min(parsed_ts).isoformat(sep=" "),
                max(parsed_ts).isoformat(sep=" "),
            ]
        else:
            zone["Time range"] = ["", ""]

        zones.append(zone)

    return zones

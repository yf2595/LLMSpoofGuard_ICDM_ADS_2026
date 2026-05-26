# Data assets

## Benchmark dataset (`dataset/`)

OpenSky Network ADS-B state vectors for **61,565** trajectory segments collected
over **November 2024 – June 2025** (~1.15M messages).

| File | Role |
|------|------|
| `trajectory_manifest.csv` | Segment metadata and labels (`is_spoofed`, `spoof_category`) |
| `llmspoofguard_YYYY_MM.csv` | Monthly point-level shards (OpenSky schema + labels) |
| `dataset_index.json` | Shard list and aggregate counts |

**Segmentation:** per-`icao24` streams split on gaps > 3 min; segments with < 5 points removed.

**Labels:** `is_spoofed` (binary) and `spoof_category` (closed set from the paper).

### Summary statistics

```bash
python scripts/dataset_stats.py
```

## Country borders (`countries/`)

Natural Earth Admin-0 shapefile (`ne_110m_admin_0_countries.*`) for overflown-country
assignment in `src/preprocessing.py`.

Source: [Natural Earth 110m Cultural Vectors](https://www.naturalearthdata.com/downloads/110m-cultural-vectors/)

## Optional: aircraft metadata

Download [OpenSky aircraftDatabase.csv](https://opensky-network.org/datasets/metadata/aircraft-database/)
to `data/aircraftDatabase.csv` for manufacturer enrichment. The pipeline defaults to
`model=unknown` if the file is absent.

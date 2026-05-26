# LLMSpoofGuard — ICDM 2026 Artifact

Reproducibility package for:

> **LLMSpoofGuard: Real-Time Detection and Situational Awareness of GPS Spoofing in Aviation via ADS-B Data**  
> IEEE International Conference on Data Mining (ICDM), 2026.

This repository provides the **benchmark dataset**, **detection pipeline**, **evaluation harness**, and scripts to **reproduce every quantitative result** reported in the paper. It is self-contained: no UI, database, or live-server deployment code.

**Important:** Metrics and figures under `evaluation/results/` are produced by running the scripts below. The tables in this README quote the paper; regenerate the JSON/CSV/plot artifacts locally to verify them.

---

## Contents

| Component | Location | Description |
| --- | --- | --- |
| Benchmark dataset | [`data/dataset/`](data/dataset/) | 61,565 ADS-B trajectory segments (Nov 2024 – Jun 2025) |
| LLM prompt | [`prompts/gps_detection_prompt.py`](prompts/gps_detection_prompt.py) | Few-shot GPS spoofing detector (Section 6 category bank) |
| Prompt ablation variants | [`prompts/prompt_variants.py`](prompts/prompt_variants.py) | Zero-shot vs few-shot, with/without category bank |
| Rule engine | [`src/rules.py`](src/rules.py) | Tier-1 RBH proxy labels |
| Preprocessing | [`src/preprocessing.py`](src/preprocessing.py) | Cleaning, country join, segmentation |
| Evaluation | [`evaluation/`](evaluation/) | Baselines, metrics, splits, plots |
| Generated outputs | [`evaluation/results/`](evaluation/results/) | Written by the scripts (see [`evaluation/results/README.md`](evaluation/results/README.md)) |

---

## Quick start

### 1. Environment

```bash
cd ICDM2026
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` when running **OpenAI API** experiments (`OPENAI_API_KEY`). For local open-weight models, also install `requirements-llm-local.txt`.

### 2. Dataset statistics

```bash
python scripts/dataset_stats.py
```

### 3. Reproduce Table II (baselines, no API cost)

```bash
python scripts/run_benchmark.py
```

Loads the full benchmark (61,565 segments), applies an **80/20 stratified train/test split** (seed 42), trains supervised models on the train fold, evaluates on the test fold (~12,313 segments), and writes:

- `evaluation/results/results.json`
- `evaluation/results/results.csv`
- `evaluation/results/plots/confusion_matrices.png`
- `evaluation/results/plots/metrics_bar.png`
- `evaluation/results/plots/paper_table.png`

**Ground truth:** RBH Tier-1 rules (conservative known-pattern proxy). RBH is the labelling oracle and should report **100%** accuracy on the test split by construction.

### 4. Full Table II (including GPT-4.1 mini)

```bash
python -m evaluation.run_evaluation --no-skip-llm
```

Requires a valid `OPENAI_API_KEY` in `.env`. Expect API cost and runtime proportional to the test-set size (~12k segments × one LLM call each).

---

## Prompt ablation (few-shot vs zero-shot, tab:prompt_ablation)

Compares four GPT-4.1 mini prompt configurations on manifest labels (`is_spoofed`):

| Variant | Description |
| --- | --- |
| Zero-shot, no category bank | Minimal prompt, binary output |
| Zero-shot + category bank | Closed-set categories in prompt |
| Few-shot, no category bank | Examples without categories |
| Few-shot + category bank + Unknown | Full deployed prompt (paper default) |

**Paper protocol** (balanced 200 trajectories: 100 spoofed + 100 clean, category-stratified):

```bash
python scripts/run_prompt_ablation.py
```

**Full benchmark** (all ~61k manifest segments — long-running, high API cost):

```bash
python scripts/run_prompt_ablation.py --full-benchmark --max-workers 10
```

Custom class count or subset of variants:

```bash
python scripts/run_prompt_ablation.py --n-per-class 100 --max-workers 10
python scripts/run_prompt_ablation.py --variants few_shot_category_bank_unknown zero_shot_no_bank
```

Outputs:

- `evaluation/results/prompt_ablation.json` / `.csv`
- `evaluation/results/plots/prompt_ablation_table.png`
- `evaluation/results/prompt_ablation_raw/<variant>.jsonl` — per-trajectory API responses

---

## LLM backbone comparison (tab:GPTs-comparison)

Evaluates multiple LLM backbones on the same 80/20 test fold as Table II.

**Full replication** (entire held-out test fold, ~12.3k trajectories):

```bash
python scripts/run_llm_comparison.py --group openai
python scripts/run_llm_comparison.py --group all   # includes local GPU models
```

**Smoke test** (100 randomly subsampled test trajectories, seeded):

```bash
python scripts/run_llm_comparison.py --group openai --max-test 100
```

Single model or local checkpoint:

```bash
python scripts/run_llm_comparison.py --models gpt-4.1-mini
python scripts/run_llm_comparison.py --models ministral-7b --max-test 200
```

Outputs:

- `evaluation/results/llm_comparison.json` / `.csv`
- `evaluation/results/plots/llm_comparison_table.png`
- `evaluation/results/llm_comparison_raw/<model>.jsonl`

The model registry (`evaluation/llm_comparison/registry.py`) stores **reference paper values** for side-by-side comparison after a run; measured metrics always come from live inference.

---

## Inter-rater agreement (manual analyst study)

Three independent analysts reviewed trajectory visualizations and assigned **Clean** or **Spoofed** labels. Raw ratings are bundled as:

- `evaluation/inter_rater_agreement_trajectories.csv` — 1,000 trajectories (500 RBH-flagged + 500 not-flagged)
- `evaluation/inter_rater_agreement_summary.csv` — aggregated metrics

Recompute the summary and write provenance metadata (SHA-256 of the ratings file):

```bash
python scripts/compute_inter_rater_agreement.py
```

This writes `evaluation/inter_rater_agreement_provenance.json`, documenting that the summary was **derived from** the bundled analyst ratings CSV, not hand-edited.

To reproduce the balanced audit **sample list** (trajectory IDs only):

```bash
python scripts/compute_inter_rater_agreement.py --write-sample-manifest
```

Analyst labels themselves are collected manually; the script aggregates them but does not synthesize ratings.

---

## Benchmark dataset

| Statistic | Value |
| --- | --- |
| Trajectory segments | 61,565 |
| ADS-B messages | 1,145,997 |
| Unique aircraft (`icao24`) | 27,105 |
| Collection window | 2024-11-01 – 2025-06-29 |
| Spoofed segments (manifest) | 6,156 (10.0%) |

### Spoofing category distribution (spoofed segments)

| Category | Share |
| --- | --- |
| Unrealistic heading change | 24.9% |
| Zero velocity | 13.6% |
| Altitude increase | 13.1% |
| Unknown (Later classified as Velocity Spikes) | 12.5% |
| Sudden positional jump | 12.5% |
| Timestamp freeze | 11.9% |
| Altitude drop | 11.5% |

See [`data/README.md`](data/README.md) for file layout and column definitions.

---

## Reference results (Table II)

Detection performance on the **conservative known-pattern proxy benchmark** (20% held-out test split). RBH defines the proxy labels and is shown as an oracle/reference, not as a competing detector.

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
| --- | --- | --- | --- | --- |
| LSTM (S; T) | 79.4 | 75.0 | 80.3 | 77.6 |
| XGBoost-point (S; P) | 64.3 | 55.5 | 96.9 | 71.3 |
| XGBoost-stat (S; T) | 85.4 | 80.4 | 88.6 | 84.3 |
| Isolation Forest (U; P) | 72.2 | 62.2 | 94.3 | 74.9 |
| **GPT-4.1 mini (F)** | **98.0** | **95.0** | **99.0** | **97.0** |
| *RBH (oracle)* | *100* | *100* | *100* | *100* |

S = supervised, U = unsupervised, F = few-shot. **T** = trajectory-level input, **P** = point-level input.

Run `python scripts/run_benchmark.py` (and `--no-skip-llm` for the LLM row) to regenerate figures under `evaluation/results/plots/`.

---

## Full replication checklist

| Step | Command | API / GPU |
| --- | --- | --- |
| Dataset stats | `python scripts/dataset_stats.py` | — |
| Table II baselines | `python scripts/run_benchmark.py` | CPU |
| Table II + GPT-4.1 mini | `python -m evaluation.run_evaluation --no-skip-llm` | OpenAI |
| Prompt ablation (paper 200) | `python scripts/run_prompt_ablation.py` | OpenAI |
| Prompt ablation (full) | `python scripts/run_prompt_ablation.py --full-benchmark` | OpenAI |
| LLM backbone comparison | `python scripts/run_llm_comparison.py --group all` | OpenAI + GPU |
| Inter-rater summary | `python scripts/compute_inter_rater_agreement.py` | — |

Use `--max-test N` or `--max-trajectories N` only for smoke tests; omit them for full paper replication.

---

## Evaluation protocol

1. **Load** all eight monthly shards plus `trajectory_manifest.csv` from `data/dataset/`.
2. **Label** trajectories with the RBH oracle (`--rbh-oracle`, default).
3. **Split** 80% train / 20% test, stratified by label (`seed=42`).
4. **Train** supervised baselines (LSTM, XGBoost-point, XGBoost-stat) on the train fold.
5. **Score** all methods on the test fold; write metrics and plots.

### Methods evaluated

| Key | Paper name | Type | Input |
| --- | --- | --- | --- |
| `LSTM` | LSTM (S; T) | Supervised | Sequence |
| `XGBoost-point` | XGBoost-point (S; P) | Supervised | Point deltas |
| `XGBoost-traj` | XGBoost-stat (S; T) | Supervised | Trajectory stats |
| `IsolationForest` | Isolation Forest (U; P) | Unsupervised | Point deltas |
| `LLM` | GPT-4.1 mini (F) | Few-shot | Full trajectory + prompt |
| `RBH` | RBH (oracle) | Rule-based | Tier-1 heuristics |

### Advanced CLI

```bash
python -m evaluation.run_evaluation \
  --dataset-dir data/dataset \
  --rbh-oracle \
  --test-size 0.2 \
  --seed 42 \
  --plot \
  --skip-llm \
  --output evaluation/results/results.json
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dataset-dir` | `data/dataset` | Auto-load all shards + manifest |
| `--rbh-oracle` / `--no-rbh-oracle` | on | RBH labels vs manifest `is_spoofed` |
| `--test-size` | `0.2` | Held-out test fraction |
| `--plot` / `--no-plot` | on | Write figures under `evaluation/results/plots/` |
| `--skip-llm` | off | Skip GPT-4.1 mini baseline |
| `--max-trajectories` | none | Smoke-test truncation only |

Regenerate plots from existing `results.json`:

```bash
python -m evaluation.plot_results --results evaluation/results/results.json
```

---

## Repository layout

```text
ICDM2026/
├── README.md
├── requirements.txt
├── requirements-llm-local.txt
├── .env.example
├── data/
│   ├── README.md
│   ├── dataset/                    # Benchmark CSV shards + manifest
│   └── countries/                  # Natural Earth shapefile
├── prompts/
│   ├── gps_detection_prompt.py
│   └── prompt_variants.py
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── rules.py
│   ├── detection_llm.py
│   ├── zones.py
│   └── collector.py
├── evaluation/
│   ├── run_evaluation.py
│   ├── inter_rater_agreement.py
│   ├── inter_rater_agreement_trajectories.csv
│   ├── inter_rater_agreement_summary.csv
│   ├── inter_rater_agreement_provenance.json
│   ├── prompt_ablation/
│   ├── llm_comparison/
│   ├── baselines/
│   └── results/                    # Generated by scripts (see README there)
└── scripts/
    ├── run_benchmark.py            # Table II baselines
    ├── train_baselines.py          # Alias for reviewers
    ├── run_prompt_ablation.py      # Few-shot vs zero-shot ablation
    ├── run_llm_comparison.py       # LLM backbone comparison
    ├── compute_inter_rater_agreement.py
    ├── dataset_stats.py
    ├── run_detection.py            # LLM detection + zone generation
    └── collect_adsb.py             # Optional live OpenSky capture
```

---

## Detection pipeline (optional)

Run the few-shot LLM detector and DBSCAN spoofing-zone extraction on a single shard:

```bash
python scripts/run_detection.py \
  --csv data/dataset/llmspoofguard_2025_01.csv \
  --output-dir evaluation/results/detection_run \
  --max-trajectories 50
```

---

## Methodological notes

**Proxy benchmark.** RBH implements the closed-set category bank from the paper prompt. It provides reproducible Tier-1 labels at scale. The LLM detector may disagree on legitimate-maneuver edge cases (counterexamples in the prompt); that is expected and discussed in Section V-B of the paper.

**RBH at 100%.** When `--rbh-oracle` is enabled, the RBH baseline is evaluated against the same rules used to define positives. Perfect scores confirm the harness is wired correctly; comparative insight comes from the other detectors.

**Determinism.** `seed=42` for splitting, XGBoost, Isolation Forest, and LSTM. LLM calls use `temperature=0`; minor API-side variance may still occur.

**No pre-filled metrics.** All evaluation JSON/CSV/plot artifacts are written at runtime. Bundled analyst-rating CSVs are primary data; their summary is recomputed via `compute_inter_rater_agreement.py`.

---

## Requirements

- Python 3.10+
- See [`requirements.txt`](requirements.txt) for pinned dependencies
- ~2 GB RAM for full evaluation; LLM experiments require network access (and a GPU for local models)

---

## Citation

If you use this artifact, please cite the ICDM 2026 LLMSpoofGuard paper.

---

## Scope

This artifact does **not** include the production web UI, MongoDB backend, or agentic chat stack in `Server/` and `ui/`. Those components support the operational deployment described in the paper but are outside the reproducibility bundle.

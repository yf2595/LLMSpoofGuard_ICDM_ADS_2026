# LLMSpoofGuard: Real-Time Detection and Situational Awareness of GPS Spoofing in Aviation via ADS-B Data

**IEEE International Conference on Data Mining (ICDM) 2026**

This repository accompanies the paper:

> **LLMSpoofGuard: Real-Time Detection and Situational Awareness of GPS Spoofing in Aviation via ADS-B Data**

LLMSpoofGuard is a deployed aviation-security framework that analyzes ADS-B trajectories with an LLM, identifies physically implausible flight behavior, assigns known or `Unknown` spoofing-effect categories, aggregates detected events into geographic spoofing zones, and supports analyst-facing situational awareness.

This public artifact contains the benchmark dataset, prompts, rule-based proxy-label implementation, detection pipeline, classical and learning-based baselines, prompt ablations, LLM-backbone comparison, analyst-agreement artifacts, and supporting evaluation code. Production web applications, databases, and deployment-only infrastructure are outside the public reproducibility package.

---

## Citation

```bibtex
@inproceedings{felendler2026llmspoofguard,
  author    = {Yuval Felendler and Yuval Elovici and Asaf Shabtai},
  title     = {{LLMSpoofGuard}: Real-Time Detection and Situational Awareness of {GPS} Spoofing in Aviation via {ADS-B} Data},
  booktitle = {2026 IEEE International Conference on Data Mining (ICDM)},
  year      = {2026}
}
```

---

## Repository contents

| Component | Location | Description |
| --- | --- | --- |
| Benchmark dataset | [`data/dataset/`](data/dataset/) | 61,565 ADS-B trajectory segments |
| Full few-shot prompt | [`prompts/gps_detection_prompt.py`](prompts/gps_detection_prompt.py) | Deployed trajectory-spoofing prompt |
| Prompt variants | [`prompts/prompt_variants.py`](prompts/prompt_variants.py) | Zero-shot / few-shot experimental variants |
| RBH rules | [`src/rules.py`](src/rules.py) | Six conservative Tier-1 proxy-label heuristics |
| Preprocessing | [`src/preprocessing.py`](src/preprocessing.py) | Cleaning, trajectory construction, geographic enrichment |
| LLM detector | [`src/detection_llm.py`](src/detection_llm.py) | Trajectory-level LLM detection |
| Zone generation | [`src/zones.py`](src/zones.py) | DBSCAN spoofing-zone aggregation |
| Evaluation harness | [`evaluation/`](evaluation/) | Baselines, metrics, splits, plots, model comparisons |
| Analyst ratings | [`evaluation/inter_rater_agreement_trajectories.csv`](evaluation/inter_rater_agreement_trajectories.csv) | 1,000-trajectory post-deployment validation sample |
| Generated outputs | [`evaluation/results/`](evaluation/results/) | JSON, CSV, raw model outputs, and plots |

---

## Quick start

### Environment

Python **3.10+** is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Install the standard dependencies:

```bash
pip install -r requirements.txt
```

Copy the environment template:

```bash
cp .env.example .env
```

OpenAI-backed experiments require:

```text
OPENAI_API_KEY=...
```

Local open-weight model experiments additionally use:

```bash
pip install -r requirements-llm-local.txt
```

---

## Benchmark dataset

The released benchmark contains:

| Statistic | Value |
| --- | ---: |
| Trajectory segments | 61,565 |
| ADS-B messages | 1,145,997 |
| Unique aircraft (`icao24`) | 27,105 |
| Collection window | 2024-11-01 – 2025-06-29 |
| Spoofed segments in manifest | 6,156 (10.0%) |

The data span more than 100 countries and include diverse operational conditions and spoofing prevalence.

Dataset layout and field definitions are documented in:

[`data/README.md`](data/README.md)

---

## Conservative proxy-label benchmark

Authoritative global ground truth for aviation GPS spoofing is rarely available at scale. The paper therefore uses a conservative known-pattern proxy benchmark.

The **RBH** procedure defines six physically grounded known-pattern heuristics:

| Rule | Condition |
| --- | --- |
| Altitude Drop | decrease > 4000 m within 2 min |
| Timestamp Freeze | same timestamp across ≥3 consecutive updates |
| Zero Velocity | ground speed < 50 m/s while altitude > 5000 m |
| Heading Change | absolute heading change > 120° between consecutive updates |
| Position Shift | >1.8° lat/lon or >250 km great-circle distance within 2 min |
| Altitude Increase | increase >3000 m within 2 min outside takeoff |

RBH is therefore an **oracle/reference for the proxy labels**, not an independent competing detector.

The LLM uses a known-effect category bank together with an explicit `Unknown` output for behavior that is not adequately captured by the predefined categories.

---

## Table II — spoofing-effect category distribution

| Category | Prevalence |
| --- | ---: |
| Heading change | 24.9% |
| Zero velocity | 13.6% |
| Altitude increase | 13.1% |
| Position shift | 12.5% |
| Unknown | 12.5% |
| Timestamp freeze | 11.9% |
| Altitude drop | 11.5% |

---

## Table III — detection performance

The learning-based baselines are evaluated over **five random seeds (42–46)**. Each seed uses an independent **80/20 stratified split**, and the paper reports the resulting aggregate performance.

GPT-4.1 mini is evaluated at temperature 0 with Wilson 95% confidence intervals over its per-trajectory predictions.

### Paper values

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| LSTM (S; T) | 79.4 | 75.0 | 80.3 | 77.6 |
| XGBoost-point (S; P) | 64.3 | 55.5 | 96.9 | 71.3 |
| XGBoost-Traj (S; T) | 85.4 | 80.4 | 88.6 | 84.3 |
| Isolation Forest (U; P) | 72.2 | 62.2 | 94.3 | 74.9 |
| **LLMSpoofGuard (GPT-4.1 mini, F)** | **98.0** | **95.0** | **99.0** | **97.0** |
| *RBH oracle/reference* | *100* | *100* | *100* | *100* |

`S` = supervised, `U` = unsupervised, `F` = few-shot.  
`T` = trajectory-level input, `P` = point-level input.

### Run the baseline benchmark

```bash
python scripts/run_benchmark.py
```

The lower-level evaluator can be used for individual seeded runs:

```bash
python -m evaluation.run_evaluation \
  --dataset-dir data/dataset \
  --rbh-oracle \
  --test-size 0.2 \
  --seed 42 \
  --plot \
  --skip-llm \
  --output evaluation/results/results_seed42.json
```

For the LLM row:

```bash
python -m evaluation.run_evaluation --no-skip-llm
```

Generated outputs are written under:

```text
evaluation/results/
```

---

## Table IV — prompt configuration ablation

The paper compares the deployed few-shot prompt with a zero-shot version using the same trajectory representation and output schema.

| Prompt setting | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| Zero-shot | 82.5 | 85.0 | 65.0 | 75.0 |
| Few-shot | 98.0 | 95.0 | 99.0 | 97.0 |

Run:

```bash
python scripts/run_prompt_ablation.py
```

Full-benchmark execution:

```bash
python scripts/run_prompt_ablation.py --full-benchmark --max-workers 10
```

Additional diagnostic variants are available through:

```bash
python scripts/run_prompt_ablation.py \
  --variants few_shot_category_bank_unknown zero_shot_no_bank
```

Outputs:

```text
evaluation/results/prompt_ablation.json
evaluation/results/prompt_ablation.csv
evaluation/results/plots/prompt_ablation_table.png
evaluation/results/prompt_ablation_raw/<variant>.jsonl
```

---

## Table V — LLM backbone comparison

The final paper evaluates multiple LLM backbones using the deployed few-shot prompt.

| Model | Accuracy (%) | Avg. inference time (s) | Input cost / 1M tokens ($) | Output cost / 1M tokens ($) |
| --- | ---: | ---: | ---: | ---: |
| GPT-4.1 | 98 | 0.70 | 2.00 | 8.00 |
| GPT-4o | 97 | 1.50 | 2.50 | 10.00 |
| GPT-4.1 mini | 98 | 0.49 | 0.40 | 1.60 |
| o1-mini | 99 | 2.60 | 1.10 | 4.40 |
| GPT-5.2 | 99 | 2.20 | 1.75 | 14.00 |
| Llama 3.1 8B | 96 | 0.80 | 0.27 | 0.27 |
| Mistral 7B | 97 | 0.30 | 0.10 | 0.10 |
| Qwen3-Max-Thinking | 99 | 3.50 | 0.20 | 0.20 |

Open-weight models were evaluated locally on a single A100 GPU. Cloud-model inference time excludes network latency.

### Full comparison

```bash
python scripts/run_llm_comparison.py --group openai
python scripts/run_llm_comparison.py --group all
```

### Smoke test

```bash
python scripts/run_llm_comparison.py --group openai --max-test 100
```

### Single-model execution

```bash
python scripts/run_llm_comparison.py --models gpt-4.1-mini
python scripts/run_llm_comparison.py --models ministral-7b --max-test 200
```

Outputs:

```text
evaluation/results/llm_comparison.json
evaluation/results/llm_comparison.csv
evaluation/results/plots/llm_comparison_table.png
evaluation/results/llm_comparison_raw/<model>.jsonl
```

---

## DBSCAN spoofing-zone generation

Detected trajectory-level events are aggregated into geographic spoofing zones using DBSCAN.

The deployment configuration reported in the paper uses:

```text
epsilon = 250 km
min_samples = 5
```

The zone-generation implementation is located at:

[`src/zones.py`](src/zones.py)

The detector + zone-generation path can be run on a benchmark shard with:

```bash
python scripts/run_detection.py \
  --csv data/dataset/llmspoofguard_2025_01.csv \
  --output-dir evaluation/results/detection_run \
  --max-trajectories 50
```

---

## Unknown-effect analyst review

The `Unknown` output is treated as an uncertainty / discovery channel rather than automatically as an error.

In the offline evaluation, **769 detected spoofing events** were assigned `Unknown` and reviewed by three aviation-security analysts.

| Outcome | Count | Share |
| --- | ---: | ---: |
| Outside known category bank | 513 | 66.7% |
| Variant of a known category, sub-threshold | 184 | 23.9% |
| Legitimate but unusual maneuver | 51 | 6.6% |
| Inconclusive / insufficient context | 21 | 2.8% |

The review distinguishes heuristic-bypassing behavior from ordinary false-positive noise.

---

## Post-deployment analyst validation

A separate post-deployment study uses a stratified sample of **1,000 trajectories**:

- 500 flagged,
- 500 non-flagged.

Three aviation analysts independently assigned the primary spoofing-relevant / clean label. Explanation/category consistency was evaluated in a separate pass.

Bundled files:

```text
evaluation/inter_rater_agreement_trajectories.csv
evaluation/inter_rater_agreement_summary.csv
```

Recompute the agreement summary:

```bash
python scripts/compute_inter_rater_agreement.py
```

The script writes:

```text
evaluation/inter_rater_agreement_provenance.json
```

containing provenance metadata and a SHA-256 hash of the ratings file.

Generate the sampled trajectory-ID manifest with:

```bash
python scripts/compute_inter_rater_agreement.py --write-sample-manifest
```

The analyst ratings are human annotations; the scripts aggregate the ratings but do not synthesize them.

### Paper summary

| Sample | Outcome | Rate | 95% CI |
| --- | --- | ---: | --- |
| Flagged | Spoofing-relevant | 99.0% | [97.7, 99.7] |
| Flagged | Disagreement | 1.0% | [0.3, 2.3] |
| Non-flagged | Clean | 98.0% | [96.4, 99.0] |
| Non-flagged | Disagreement | 2.0% | [1.0, 3.6] |
| Confirmed cases | Explanation/category consistent | 100.0% | — |

Analysts reached full agreement on 92.5% of the 1,000 trajectories, with the remaining cases resolved by majority vote.

---

## Evaluation protocol

### Learning-based benchmark

1. Load all benchmark shards and `trajectory_manifest.csv`.
2. Construct conservative Tier-1 proxy labels with the RBH rules.
3. Use independent 80/20 stratified splits for seeds **42–46**.
4. Train the supervised baselines on each corresponding training fold.
5. Evaluate all methods on the matching held-out fold.
6. Aggregate the learning-based baseline results across seeds.
7. Evaluate GPT-4.1 mini at temperature 0 using the deployed few-shot prompt.
8. Write generated metrics and plots under `evaluation/results/`.

### Methods

| Repository key | Paper name | Type | Input |
| --- | --- | --- | --- |
| `LSTM` | LSTM (S; T) | Supervised | Full sequence |
| `XGBoost-point` | XGBoost-point (S; P) | Supervised | Local point deltas |
| `XGBoost-traj` | XGBoost-Traj (S; T) | Supervised | Trajectory statistics |
| `IsolationForest` | Isolation Forest (U; P) | Unsupervised | Local point deltas |
| `LLM` | LLMSpoofGuard / GPT-4.1 mini (F) | Few-shot | Full trajectory + prompt |
| `RBH` | RBH oracle/reference | Rule-based | Six known-pattern heuristics |

---

## Reproduction commands

### Dataset statistics

```bash
python scripts/dataset_stats.py
```

### Detection benchmark

```bash
python scripts/run_benchmark.py
```

### Detection benchmark with GPT-4.1 mini

```bash
python -m evaluation.run_evaluation --no-skip-llm
```

### Prompt ablation

```bash
python scripts/run_prompt_ablation.py
```

### LLM backbone comparison

```bash
python scripts/run_llm_comparison.py --group all
```

### Inter-rater agreement

```bash
python scripts/compute_inter_rater_agreement.py
```

Use `--max-test N` or `--max-trajectories N` only for reduced smoke-test runs.

---

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── requirements-llm-local.txt
├── .env.example
├── data/
│   ├── README.md
│   ├── dataset/
│   └── countries/
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
│   └── results/
└── scripts/
    ├── run_benchmark.py
    ├── train_baselines.py
    ├── run_prompt_ablation.py
    ├── run_llm_comparison.py
    ├── compute_inter_rater_agreement.py
    ├── dataset_stats.py
    ├── run_detection.py
    └── collect_adsb.py
```

---

## Artifact scope

The repository is the public research artifact for the offline benchmark and released evaluation analyses.

It does not include the production web UI, MongoDB backend, live agentic chat stack, or other deployment infrastructure used by the operational system.

Production-only quantities in the paper, including sustained service measurements and analyst-workflow observations, are reported from the deployed environment rather than regenerated by the offline benchmark scripts.

---

## Requirements

- Python 3.10+
- Dependencies in [`requirements.txt`](requirements.txt)
- Additional local-model dependencies in `requirements-llm-local.txt`
- CPU for classical benchmark components
- OpenAI API access for GPT-family experiments
- GPU and model weights for local open-weight LLM comparisons

---

## Data and use conditions

ADS-B and other third-party data remain subject to the terms of their respective providers. The repository does not relicense provider-restricted raw data.

API credentials, `.env` files, private analyst information, internal production endpoints, and provider-restricted data are not part of the public artifact.

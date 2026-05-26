"""
Run the multi-LLM comparison experiment on the benchmark test split.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from evaluation.labels import trajectory_labels
from evaluation.llm_comparison.providers import InferenceResult, create_provider
from evaluation.llm_comparison.registry import LLMModelSpec, list_models
from evaluation.metrics import compute_metrics
from evaluation.run_evaluation import (
    DEFAULT_DATASET_DIR,
    discover_dataset,
    load_from_manifest,
)
from evaluation.splits import gather, stratified_split
logger = logging.getLogger(__name__)


def prepare_test_split(
    trajectories: list,
    labels: list[bool],
    test_size: float,
    seed: int,
    max_test: int | None,
) -> tuple[list, list[bool]]:
    train_idx, test_idx = stratified_split(
        trajectories, labels, test_size=test_size, seed=seed,
    )
    test_trajs = gather(trajectories, test_idx)
    test_labels = gather(labels, test_idx)
    if max_test is not None and len(test_trajs) > max_test:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(test_trajs), size=max_test, replace=False)
        test_trajs = [test_trajs[i] for i in chosen]
        test_labels = [test_labels[i] for i in chosen]
    return test_trajs, test_labels


def _run_openai_batch(
    provider,
    test_trajs: list,
    max_workers: int,
) -> list[InferenceResult]:
    results: list[InferenceResult | None] = [None] * len(test_trajs)

    def _one(i: int, traj) -> tuple[int, InferenceResult]:
        return i, provider.detect_trajectory(traj)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, i, t) for i, t in enumerate(test_trajs)]
        for fut in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=provider.spec.display_name,
        ):
            i, res = fut.result()
            results[i] = res

    return [r for r in results if r is not None]


def _run_local_sequential(provider, test_trajs: list) -> list[InferenceResult]:
    out: list[InferenceResult] = []
    for traj in tqdm(test_trajs, desc=provider.spec.display_name):
        out.append(provider.detect_trajectory(traj))
    return out


def _aggregate_model_run(
    spec: LLMModelSpec,
    infer_results: list[InferenceResult],
    y_true: list[bool],
) -> dict:
    preds = [r.spoofing_detected for r in infer_results]
    metrics = compute_metrics(y_true, preds)

    latencies = [r.latency_s for r in infer_results if r.error is None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    total_in = sum(r.input_tokens for r in infer_results)
    total_out = sum(r.output_tokens for r in infer_results)
    n = len(infer_results)
    est_cost = (
        (total_in / 1_000_000) * spec.input_cost_per_1m
        + (total_out / 1_000_000) * spec.output_cost_per_1m
    )

    return {
        "key": spec.key,
        "display_name": spec.display_name,
        "backend": spec.backend,
        "model_id": spec.model_id,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy_pct": round(metrics["accuracy"] * 100, 1),
        "avg_inference_time_s": round(avg_latency, 3),
        "input_cost_per_1m_usd": spec.input_cost_per_1m,
        "output_cost_per_1m_usd": spec.output_cost_per_1m,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "estimated_cost_usd": round(est_cost, 4),
        "n_test": n,
        "n_parsed_ok": sum(1 for r in infer_results if r.parsed_ok),
        "n_errors": sum(1 for r in infer_results if r.error),
        "reference_paper_accuracy_pct": spec.reference_paper_accuracy_pct,
        "reference_paper_latency_s": spec.reference_paper_latency_s,
        **{k: metrics[k] for k in ("TP", "FP", "TN", "FN")},
    }


def run_llm_comparison(
    test_trajs: list,
    test_labels: list[bool],
    models: list[LLMModelSpec],
    *,
    max_workers: int = 8,
    client: OpenAI | None = None,
    raw_dir: Path | None = None,
) -> dict:
    """Evaluate each LLM on the same test trajectories."""
    summary: dict = {
        "n_test": len(test_trajs),
        "models": {},
    }

    for spec in models:
        logger.info("=== %s (%s) ===", spec.display_name, spec.model_id)
        provider = create_provider(spec, client=client)

        try:
            if spec.backend == "openai":
                infer_results = _run_openai_batch(provider, test_trajs, max_workers)
            else:
                infer_results = _run_local_sequential(provider, test_trajs)
        finally:
            provider.unload()

        row = _aggregate_model_run(spec, infer_results, test_labels)
        summary["models"][spec.key] = row

        if raw_dir is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)
            path = raw_dir / f"{spec.key}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for i, (res, traj) in enumerate(zip(infer_results, test_trajs)):
                    f.write(json.dumps({
                        "index": i,
                        "latency_s": res.latency_s,
                        "input_tokens": res.input_tokens,
                        "output_tokens": res.output_tokens,
                        "spoofing_detected": res.spoofing_detected,
                        "parsed_ok": res.parsed_ok,
                        "error": res.error,
                        "trajectory_len": len(traj),
                    }, ensure_ascii=False) + "\n")
            logger.info("Wrote per-trajectory log to %s", path)

        logger.info(
            "%s -> acc=%.1f%% avg_time=%.3fs cost~$%.4f errors=%d",
            spec.display_name,
            row["accuracy_pct"],
            row["avg_inference_time_s"],
            row["estimated_cost_usd"],
            row["n_errors"],
        )

    return summary


def load_benchmark_test_split(
    dataset_dir: Path,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    max_trajectories: int | None = None,
    max_test: int | None = None,
    rbh_oracle: bool = True,
) -> tuple[list, list[bool], dict]:
    """Load dataset and return (test_trajectories, test_labels, dataset_info)."""
    shards, manifest = discover_dataset(dataset_dir)
    trajectories, manifest_labels = load_from_manifest(
        shards, manifest, max_trajectories=max_trajectories, min_samples=5,
    )

    if rbh_oracle:
        labels = trajectory_labels(trajectories)
        label_policy = "RBH conservative known-pattern proxy"
    else:
        labels = manifest_labels
        label_policy = "dataset manifest (is_spoofed)"

    test_trajs, test_labels = prepare_test_split(
        trajectories, labels, test_size=test_size, seed=seed, max_test=max_test,
    )

    info = {
        "n_trajectories": len(trajectories),
        "label_policy": label_policy,
        "test_size": test_size,
        "seed": seed,
        "n_test": len(test_trajs),
        "n_test_positive": sum(test_labels),
    }
    return test_trajs, test_labels, info


def write_comparison_outputs(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    rows = []
    for key in sorted(results["models"].keys(), key=lambda k: results["models"][k]["display_name"]):
        m = results["models"][key]
        rows.append({
            "llm": m["display_name"],
            "accuracy_pct": m["accuracy_pct"],
            "avg_inference_time_s": m["avg_inference_time_s"],
            "input_cost_per_1m_usd": m["input_cost_per_1m_usd"],
            "output_cost_per_1m_usd": m["output_cost_per_1m_usd"],
            "estimated_cost_usd": m["estimated_cost_usd"],
            "backend": m["backend"],
            "model_id": m["model_id"],
        })
    pd.DataFrame(rows).to_csv(output_path.with_suffix(".csv"), index=False)

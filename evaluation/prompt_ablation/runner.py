"""
Run GPT-4.1 mini across prompt ablation variants (tab:prompt_ablation).
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time
from pathlib import Path
from typing import Sequence

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from evaluation.llm_comparison.registry import LLM_MODEL_REGISTRY
from evaluation.metrics import compute_metrics
from evaluation.prompt_ablation.sampling import load_ablation_sample
from prompts.prompt_variants import (
    PROMPT_VARIANT_ORDER,
    PROMPT_VARIANTS,
    PromptVariantSpec,
    build_prompt_variant,
)
from src.detection_llm import DEFAULT_MODEL, DEFAULT_TEMPERATURE, format_trajectory_for_prompt, parse_detection_response

logger = logging.getLogger(__name__)


def _detect_one(
    traj: Sequence[Sequence],
    *,
    client: OpenAI,
    system_prompt: str,
    model_name: str,
) -> dict:
    payload = format_trajectory_for_prompt(traj)
    t0 = time.perf_counter()
    result = {
        "raw_text": "",
        "spoofing_detected": False,
        "parsed_ok": False,
        "latency_s": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "error": None,
    }
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload},
            ],
            temperature=DEFAULT_TEMPERATURE,
        )
        raw = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        result["raw_text"] = raw
        result["input_tokens"] = int(getattr(usage, "prompt_tokens", 0) or 0)
        result["output_tokens"] = int(getattr(usage, "completion_tokens", 0) or 0)
    except Exception as exc:
        result["error"] = str(exc)
        logger.error("LLM call failed: %s", exc)

    result["latency_s"] = time.perf_counter() - t0
    parsed = parse_detection_response(result["raw_text"])
    result["parsed_ok"] = parsed is not None
    result["spoofing_detected"] = bool(parsed.get("spoofing_detected")) if parsed else False
    return result


def _run_variant_batch(
    trajectories: list,
    *,
    client: OpenAI,
    spec: PromptVariantSpec,
    model_name: str,
    max_workers: int,
) -> list[dict]:
    prompt = build_prompt_variant(spec)
    results: list[dict | None] = [None] * len(trajectories)

    def _one(i: int, traj) -> tuple[int, dict]:
        return i, _detect_one(traj, client=client, system_prompt=prompt, model_name=model_name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, i, t) for i, t in enumerate(trajectories)]
        for fut in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=spec.display_name,
        ):
            i, res = fut.result()
            results[i] = res

    return [r for r in results if r is not None]


def _aggregate_variant(
    spec: PromptVariantSpec,
    infer_results: list[dict],
    y_true: list[bool],
    model_spec,
) -> dict:
    preds = [r["spoofing_detected"] for r in infer_results]
    metrics = compute_metrics(y_true, preds)
    latencies = [r["latency_s"] for r in infer_results if r["error"] is None]
    total_in = sum(r["input_tokens"] for r in infer_results)
    total_out = sum(r["output_tokens"] for r in infer_results)
    est_cost = (
        (total_in / 1_000_000) * model_spec.input_cost_per_1m
        + (total_out / 1_000_000) * model_spec.output_cost_per_1m
    )
    return {
        "key": spec.key,
        "display_name": spec.display_name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy_pct": round(metrics["accuracy"] * 100, 1),
        "precision_pct": round(metrics["precision"] * 100, 1),
        "recall_pct": round(metrics["recall"] * 100, 1),
        "f1_pct": round(metrics["f1"] * 100, 1),
        "avg_inference_time_s": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "estimated_cost_usd": round(est_cost, 4),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "n_test": len(infer_results),
        "n_parsed_ok": sum(1 for r in infer_results if r["parsed_ok"]),
        "n_errors": sum(1 for r in infer_results if r["error"]),
        **{k: metrics[k] for k in ("TP", "FP", "TN", "FN")},
    }


def run_prompt_ablation(
    trajectories: list,
    labels: list[bool],
    *,
    model_key: str = "gpt-4.1-mini",
    max_workers: int = 8,
    client: OpenAI | None = None,
    raw_dir: Path | None = None,
    variants: list[PromptVariantSpec] | None = None,
) -> dict:
    model_spec = LLM_MODEL_REGISTRY[model_key]
    model_name = model_spec.model_id
    client = client or OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    variant_specs = variants or [PROMPT_VARIANTS[k] for k in PROMPT_VARIANT_ORDER]

    summary: dict = {
        "model_key": model_key,
        "model_id": model_name,
        "n_test": len(trajectories),
        "variants": {},
    }

    for spec in variant_specs:
        logger.info("=== %s ===", spec.display_name)
        infer_results = _run_variant_batch(
            trajectories,
            client=client,
            spec=spec,
            model_name=model_name,
            max_workers=max_workers,
        )
        row = _aggregate_variant(spec, infer_results, labels, model_spec)
        summary["variants"][spec.key] = row

        if raw_dir is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)
            path = raw_dir / f"{spec.key}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for i, res in enumerate(infer_results):
                    f.write(json.dumps({"index": i, **res}, ensure_ascii=False) + "\n")
            logger.info("Wrote %s", path)

        logger.info(
            "%s -> acc=%.1f%% prec=%.1f%% rec=%.1f%% f1=%.1f%% cost~$%.4f",
            spec.display_name,
            row["accuracy_pct"],
            row["precision_pct"],
            row["recall_pct"],
            row["f1_pct"],
            row["estimated_cost_usd"],
        )

    return summary

def write_ablation_outputs(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    rows = []
    for key in PROMPT_VARIANT_ORDER:
        if key not in results.get("variants", {}):
            continue
        v = results["variants"][key]
        rows.append({
            "prompt_setting": v["display_name"],
            "accuracy_pct": v["accuracy_pct"],
            "precision_pct": v["precision_pct"],
            "recall_pct": v["recall_pct"],
            "f1_pct": v["f1_pct"],
            "estimated_cost_usd": v["estimated_cost_usd"],
            "n_parsed_ok": v["n_parsed_ok"],
            "n_errors": v["n_errors"],
        })
    pd.DataFrame(rows).to_csv(output_path.with_suffix(".csv"), index=False)


def default_output_path(root: Path | None = None) -> Path:
    root = root or Path(__file__).resolve().parent.parent.parent
    return root / "evaluation" / "results" / "prompt_ablation.json"

"""
LLM variant comparison experiment (paper Table: tab:GPTs-comparison).

Loads the benchmark, applies the same 80/20 stratified split as
``run_benchmark.py``, and evaluates each LLM on the held-out test fold.

OpenAI models (GPT-4.1, GPT-4o, GPT-4.1 mini, o1-mini, GPT-5.2):
    Set OPENAI_API_KEY in .env

Local open-weight models (Llama, Ministral, Qwen) on a CUDA GPU:
    pip install -r requirements-llm-local.txt
    huggingface-cli login   # if models are gated (e.g. Llama)

Usage:
    # Full paper replication on the entire test fold (~12.3k trajectories)
    python scripts/run_llm_comparison.py --group openai

    # Smoke test (100 stratified test trajectories)
    python scripts/run_llm_comparison.py --group openai --max-test 100

    # Single local model
    python scripts/run_llm_comparison.py --models ministral-7b --max-test 200

    # All registered models (OpenAI + local; long-running)
    python scripts/run_llm_comparison.py --group all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from evaluation.llm_comparison.plot_comparison import plot_from_json  # noqa: E402
from evaluation.llm_comparison.registry import LLM_MODEL_REGISTRY, list_models  # noqa: E402
from evaluation.llm_comparison.runner import (  # noqa: E402
    load_benchmark_test_split,
    run_llm_comparison,
    write_comparison_outputs,
)
from evaluation.run_evaluation import DEFAULT_DATASET_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Compare LLM variants on the benchmark test set.")
    p.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    p.add_argument("--output", default=str(root / "evaluation" / "results" / "llm_comparison.json"))
    p.add_argument(
        "--group",
        choices=("all", "openai", "local"),
        default="openai",
        help="Which model family to run (default: openai only).",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Explicit model keys (e.g. gpt-4.1-mini llama-3.3-7b). Overrides --group.",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-trajectories", type=int, default=None,
                   help="Cap trajectories loaded from disk.")
    p.add_argument("--max-test", type=int, default=None,
                   help="Randomly subsample the test fold to N trajectories "
                        "(seeded; omit for the full 20%% held-out set).")
    p.add_argument("--max-workers", type=int, default=8,
                   help="Parallel OpenAI requests.")
    p.add_argument("--no-rbh-oracle", action="store_true",
                   help="Use manifest is_spoofed labels instead of RBH.")
    p.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write evaluation/results/plots/llm_comparison_table.png",
    )
    p.add_argument("-v", action="count", default=1)
    return p.parse_args()


def _resolve_models(args: argparse.Namespace):
    if args.models:
        keys = args.models
    else:
        keys = [m.key for m in list_models(args.group)]
    missing = [k for k in keys if k not in LLM_MODEL_REGISTRY]
    if missing:
        raise SystemExit(f"Unknown model keys: {missing}. Choose from: {sorted(LLM_MODEL_REGISTRY)}")
    return [LLM_MODEL_REGISTRY[k] for k in keys]


def main() -> None:
    load_dotenv()
    args = parse_args()
    level = logging.WARNING if args.v == 0 else logging.INFO if args.v == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")

    models = _resolve_models(args)
    logging.info("Models to evaluate: %s", [m.display_name for m in models])

    test_trajs, test_labels, ds_info = load_benchmark_test_split(
        Path(args.dataset_dir),
        test_size=args.test_size,
        seed=args.seed,
        max_trajectories=args.max_trajectories,
        max_test=args.max_test,
        rbh_oracle=not args.no_rbh_oracle,
    )

    output_path = Path(args.output)
    raw_dir = output_path.parent / "llm_comparison_raw"

    results = run_llm_comparison(
        test_trajs,
        test_labels,
        models,
        max_workers=args.max_workers,
        raw_dir=raw_dir,
    )
    results["dataset"] = ds_info
    results["table"] = "tab:GPTs-comparison"

    write_comparison_outputs(results, output_path)

    if args.plot:
        plot_from_json(output_path, output_path.parent / "plots")

    print(f"Wrote {output_path}")
    print(f"Wrote {output_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()

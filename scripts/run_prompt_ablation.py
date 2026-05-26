"""
Prompt ablation study for GPT-4.1 mini (paper Table: tab:prompt_ablation).

Compares four prompt configurations on a balanced sample drawn from the
labeled benchmark with category coverage across spoof types.
pass
``--full-benchmark`` to evaluate every manifest segment (~61k trajectories).

Usage:
    python scripts/run_prompt_ablation.py
    python scripts/run_prompt_ablation.py --n-per-class 100 --max-workers 10
    python scripts/run_prompt_ablation.py --full-benchmark --max-workers 10
    python scripts/run_prompt_ablation.py --variants few_shot_category_bank_unknown
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from evaluation.prompt_ablation.plot_ablation import plot_from_json  # noqa: E402
from evaluation.prompt_ablation.runner import (  # noqa: E402
    default_output_path,
    load_ablation_sample,
    run_prompt_ablation,
    write_ablation_outputs,
)
from evaluation.run_evaluation import DEFAULT_DATASET_DIR  # noqa: E402
from prompts.prompt_variants import PROMPT_VARIANTS, PROMPT_VARIANT_ORDER  # noqa: E402


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Prompt ablation for GPT-4.1 mini.")
    p.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    p.add_argument("--output", default=str(default_output_path(root)))
    p.add_argument("--n-per-class", type=int, default=100,
                   help="Spoofed and clean trajectories per run when not using "
                        "--full-benchmark (default: 100 each, paper protocol).")
    p.add_argument(
        "--full-benchmark",
        action="store_true",
        help="Evaluate all manifest trajectories instead of a balanced subsample.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument(
        "--variants",
        nargs="+",
        default=None,
        choices=PROMPT_VARIANT_ORDER,
        help="Run only selected variants (default: all four).",
    )
    p.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write evaluation/results/plots/prompt_ablation_table.png",
    )
    p.add_argument(
        "--save-sample",
        default=None,
        help="Optional path to persist the sampled trajectory ids.",
    )
    p.add_argument("-v", action="count", default=1)
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    level = logging.WARNING if args.v == 0 else logging.INFO if args.v == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")

    test_trajs, test_labels, sample_info = load_ablation_sample(
        Path(args.dataset_dir),
        n_per_class=args.n_per_class,
        seed=args.seed,
        full_benchmark=args.full_benchmark,
    )
    logging.info(
        "Sample: %d trajectories (%d positive). Categories: %s",
        len(test_trajs),
        sum(test_labels),
        sample_info.get("positive_categories"),
    )

    if args.save_sample:
        Path(args.save_sample).write_text(json.dumps(sample_info, indent=2), encoding="utf-8")

    variant_specs = None
    if args.variants:
        variant_specs = [PROMPT_VARIANTS[k] for k in args.variants]

    output_path = Path(args.output)
    raw_dir = output_path.parent / "prompt_ablation_raw"

    results = run_prompt_ablation(
        test_trajs,
        test_labels,
        max_workers=args.max_workers,
        raw_dir=raw_dir,
        variants=variant_specs,
    )
    results["dataset"] = sample_info
    results["table"] = "tab:prompt_ablation"
    write_ablation_outputs(results, output_path)

    if args.plot:
        plot_from_json(output_path, output_path.parent / "plots")

    print(f"Wrote {output_path}")
    print(f"Wrote {output_path.with_suffix('.csv')}")
    for key in PROMPT_VARIANT_ORDER:
        if key not in results["variants"]:
            continue
        v = results["variants"][key]
        print(
            f"  {v['display_name']}: "
            f"Acc={v['accuracy_pct']:.1f}% Prec={v['precision_pct']:.1f}% "
            f"Rec={v['recall_pct']:.1f}% F1={v['f1_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()

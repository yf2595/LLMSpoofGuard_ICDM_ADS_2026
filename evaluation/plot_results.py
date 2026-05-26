"""
Plot helpers for the LLMSpoofGuard evaluation harness.

Reads ``results.json`` produced by ``evaluation.run_evaluation`` and writes:

    confusion_matrices.png   per-method 2x2 grids (including RBH oracle)
    metrics_bar.png          grouped bars for competing detectors (%)
    paper_table.png          tabular summary figure

Usage:
    python -m evaluation.plot_results --results evaluation/results/results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

METHOD_DISPLAY_NAMES: dict[str, str] = {
    "LSTM": "LSTM (S; T)",
    "XGBoost-point": "XGBoost-point (S; P)",
    "XGBoost-Traj": "XGBoost-Traj (S; T)",
    "IsolationForest": "Isolation Forest (U; P)",
    "LLM": "GPT-4.1 mini (F)",
    "GPT-4.1-mini": "GPT-4.1 mini (F)",
    "RBH": "RBH (oracle)",
}

METHOD_PLOT_ORDER: list[str] = [
    "LSTM",
    "XGBoost-point",
    "XGBoost-Traj",
    "IsolationForest",
    "LLM",
    "GPT-4.1-mini",
    "RBH",
]

COMPETING_METHODS = [m for m in METHOD_PLOT_ORDER if m != "RBH"]


def _display_name(method_key: str, methods: dict) -> str:
    return methods[method_key].get("display_name") or METHOD_DISPLAY_NAMES.get(
        method_key, method_key,
    )


def _ordered_methods(results: dict, *, include_oracle: bool = True) -> list[str]:
    available = set(results["methods"])
    order = [m for m in METHOD_PLOT_ORDER if m in available]
    if not order:
        order = list(results["methods"])
    if not include_oracle:
        order = [m for m in order if m != "RBH"]
    return order


def plot_confusion_matrices(results: dict, output_path: Path) -> Path:
    """Render a grid of confusion matrices, one panel per method."""
    methods_dict = results["methods"]
    method_keys = _ordered_methods(results, include_oracle=True)
    n = len(method_keys)
    cols = min(3, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.2 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")

    for ax, key in zip(axes.flat, method_keys):
        m = methods_dict[key]
        cm = np.array([
            [m["TN"], m["FP"]],
            [m["FN"], m["TP"]],
        ], dtype=int)
        label = _display_name(key, methods_dict)

        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(
            f"{label}\n"
            f"acc={m['accuracy'] * 100:.1f}%  f1={m['f1'] * 100:.1f}%",
            fontsize=10,
        )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred −", "Pred +"])
        ax.set_yticklabels(["True −", "True +"])

        v_max = cm.max() if cm.size else 1
        for i in range(2):
            for j in range(2):
                value = cm[i, j]
                colour = "white" if value > v_max / 2 else "black"
                ax.text(j, i, f"{value:,}", ha="center", va="center",
                        color=colour, fontsize=11, fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    benchmark = results.get("dataset", {}).get("benchmark", "proxy benchmark")
    fig.suptitle(
        f"Confusion matrices — {benchmark} (stratified test split)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)
    return output_path


def plot_metric_bars(results: dict, output_path: Path) -> Path:
    """Grouped bar chart for competing detectors (excludes RBH oracle)."""
    methods_dict = results["methods"]
    method_keys = _ordered_methods(results, include_oracle=False)
    if not method_keys:
        method_keys = [m for m in results["methods"] if m != "RBH"]

    labels = [_display_name(k, methods_dict) for k in method_keys]
    metric_names = ("accuracy", "precision", "recall", "f1")
    values = {
        metric: [methods_dict[k][metric] * 100.0 for k in method_keys]
        for metric in metric_names
    }

    x = np.arange(len(method_keys))
    width = 0.18
    colours = ("#4C72B0", "#55A868", "#C44E52", "#8172B2")

    fig, ax = plt.subplots(figsize=(1.35 * max(8, len(method_keys)), 5.5))
    for i, metric in enumerate(metric_names):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values[metric], width, label=metric.capitalize(),
                      color=colours[i])
        for bar, v in zip(bars, values[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f"{v:.1f}",
                ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Score (%)")
    ax.set_title(
        "Detection performance — conservative known-pattern proxy benchmark\n"
        "(RBH defines labels; oracle omitted from bars)",
        fontsize=11,
    )
    ax.legend(loc="lower right", framealpha=0.92, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)
    return output_path


def plot_paper_table(results: dict, output_path: Path) -> Path:
    """Render a matplotlib table from evaluation metrics."""
    methods_dict = results["methods"]
    rows: list[list[str]] = []
    for key in COMPETING_METHODS:
        if key not in methods_dict:
            continue
        m = methods_dict[key]
        rows.append([
            _display_name(key, methods_dict),
            f"{m['accuracy'] * 100:.1f}",
            f"{m['precision'] * 100:.1f}",
            f"{m['recall'] * 100:.1f}",
            f"{m['f1'] * 100:.1f}",
        ])

    fig, ax = plt.subplots(figsize=(9.5, 0.55 + 0.42 * max(len(rows), 1)))
    ax.axis("off")
    col_labels = ["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)"]
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 0:
            cell.set_facecolor("#E9EDF4")
        else:
            cell.set_facecolor("#FFFFFF")

    fig.suptitle(
        "Table II — Detection performance (conservative known-pattern proxy benchmark)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5, 0.02,
        "Oracle/reference: RBH obtains 100% by construction on the RBH proxy labels.",
        ha="center", fontsize=9, style="italic",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)
    return output_path


def plot_all(results: dict, plots_dir: Path) -> list[Path]:
    """Generate every figure and return their paths."""
    return [
        plot_confusion_matrices(results, plots_dir / "confusion_matrices.png"),
        plot_metric_bars(results, plots_dir / "metrics_bar.png"),
        plot_paper_table(results, plots_dir / "paper_table.png"),
    ]


def _load_results(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate plots from evaluation results.")
    p.add_argument("--results", required=True, help="Path to results.json")
    p.add_argument("--plots-dir", default=None,
                   help="Output directory (defaults to <results_dir>/plots)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    results_path = Path(args.results)
    plots_dir = Path(args.plots_dir) if args.plots_dir else results_path.parent / "plots"

    results = _load_results(results_path)
    paths = plot_all(results, plots_dir)
    for p in paths:
        logger.info("Saved %s", p)


if __name__ == "__main__":
    main()

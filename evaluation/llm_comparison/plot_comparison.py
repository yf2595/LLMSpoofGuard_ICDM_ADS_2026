"""
Plot the LLM comparison table (tab:GPTs-comparison).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)


def plot_llm_comparison_table(results: dict, output_path: Path) -> Path:
    models = results.get("models", {})
    order = sorted(models.values(), key=lambda m: m["display_name"])

    rows = [
        [
            m["display_name"],
            f"{m['accuracy_pct']:.0f}",
            f"~{m['avg_inference_time_s']:.2f}",
            f"{m['input_cost_per_1m_usd']:.2f}",
            f"{m['output_cost_per_1m_usd']:.2f}",
        ]
        for m in order
    ]

    fig, ax = plt.subplots(figsize=(11, 0.55 + 0.42 * max(len(rows), 1)))
    ax.axis("off")
    col_labels = [
        "LLM",
        "Accuracy (%)",
        "Avg. inference time (s)",
        "Input cost / 1M tokens ($)",
        "Output cost / 1M tokens ($)",
    ]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.55)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 0:
            cell.set_facecolor("#E9EDF4")

    fig.suptitle(
        "LLM variants — accuracy, inference time, and token cost",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Open-weight models evaluated locally on a single GPU; cloud models via OpenAI API.",
        ha="center",
        fontsize=8,
        style="italic",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)
    return output_path


def plot_from_json(results_path: Path, plots_dir: Path) -> Path:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    return plot_llm_comparison_table(results, plots_dir / "llm_comparison_table.png")

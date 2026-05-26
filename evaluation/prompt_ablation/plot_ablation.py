"""
Plot the prompt ablation table (tab:prompt_ablation).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from prompts.prompt_variants import PROMPT_VARIANT_ORDER

logger = logging.getLogger(__name__)


def plot_prompt_ablation_table(results: dict, output_path: Path) -> Path:
    variants = results.get("variants", {})
    rows = []
    for key in PROMPT_VARIANT_ORDER:
        if key not in variants:
            continue
        v = variants[key]
        rows.append([
            v["display_name"],
            f"{v['accuracy_pct']:.1f}",
            f"{v['precision_pct']:.1f}",
            f"{v['recall_pct']:.1f}",
            f"{v['f1_pct']:.1f}",
        ])

    fig, ax = plt.subplots(figsize=(10, 0.55 + 0.42 * max(len(rows), 1)))
    ax.axis("off")
    col_labels = ["Prompt setting", "Acc.", "Prec.", "Rec.", "F1"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 1.55)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 0:
            cell.set_facecolor("#E9EDF4")

    fig.suptitle(
        "Prompt configuration ablation for LLMSpoofGuard (GPT-4.1 mini)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)
    return output_path


def plot_from_json(results_path: Path, plots_dir: Path) -> Path:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    return plot_prompt_ablation_table(results, plots_dir / "prompt_ablation_table.png")

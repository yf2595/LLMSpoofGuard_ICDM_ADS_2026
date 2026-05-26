"""
One-command reproduction of Table II (baselines without LLM API cost).

Equivalent to:

    python -m evaluation.run_evaluation \\
        --dataset-dir data/dataset \\
        --rbh-oracle \\
        --plot \\
        --skip-llm

For the full pipeline including GPT-4.1 mini, omit --skip-llm and set
OPENAI_API_KEY in .env (see README).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.run_evaluation import main  # noqa: E402

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    defaults = [
        "--dataset-dir", str(root / "data" / "dataset"),
        "--rbh-oracle",
        "--plot",
        "--skip-llm",
    ]
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]
    main()

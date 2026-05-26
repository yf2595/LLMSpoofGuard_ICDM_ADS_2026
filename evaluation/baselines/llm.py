"""
LLM baseline (paper Table II, F).

Few-shot, trajectory-level. Uses the paper-only prompt in
``prompts/gps_detection_prompt.py`` via ``src.detection_llm``. The
detector is stateless: each trajectory is sent in an independent
chat completion call.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Sequence

from openai import OpenAI

from src.detection_llm import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    detect_batch,
    parse_detection_response,
)

from .base import Baseline

logger = logging.getLogger(__name__)


class LLMBaseline(Baseline):
    name = "LLM"
    is_supervised = False
    granularity = "trajectory"

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_workers: int = 10,
        client: OpenAI | None = None,
        raw_output_path: str | Path | None = None,
    ):
        self.model_name = model_name or os.environ.get("MODEL_NAME", DEFAULT_MODEL)
        self.temperature = temperature
        self.max_workers = max_workers
        self.client = client or OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.raw_output_path = Path(raw_output_path) if raw_output_path else None

    def predict(self, test_trajectories: Sequence[Sequence[Sequence]]) -> list[bool]:
        if not test_trajectories:
            return []

        logger.info(
            "Calling LLM '%s' on %d trajectories (max_workers=%d) ...",
            self.model_name, len(test_trajectories), self.max_workers,
        )
        raw = detect_batch(
            test_trajectories,
            client=self.client,
            model_name=self.model_name,
            temperature=self.temperature,
            max_workers=self.max_workers,
        )

        results: list[bool] = []
        parsed_ok = 0
        n_flagged = 0
        n_empty = 0
        raw_records: list[dict] = []

        for i, response_text in enumerate(raw):
            if not response_text:
                n_empty += 1
            parsed = parse_detection_response(response_text)
            if parsed is not None:
                parsed_ok += 1
            flagged = bool(parsed.get("spoofing_detected")) if parsed else False
            if flagged:
                n_flagged += 1
            results.append(flagged)

            if self.raw_output_path is not None:
                raw_records.append({
                    "index": i,
                    "raw": response_text,
                    "parsed_ok": parsed is not None,
                    "spoofing_detected": flagged,
                    "category": (parsed.get("spoofing_data") or {}).get("spoofing_category")
                    if parsed and parsed.get("spoofing_data") else None,
                    "confidence": parsed.get("confidence") if parsed else None,
                })

        logger.info(
            "LLM summary: %d/%d parsed OK, %d empty, %d flagged spoofed.",
            parsed_ok, len(raw), n_empty, n_flagged,
        )

        if self.raw_output_path is not None and raw_records:
            self.raw_output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.raw_output_path.open("w", encoding="utf-8") as f:
                for record in raw_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("Wrote raw LLM responses to %s", self.raw_output_path)

        return results

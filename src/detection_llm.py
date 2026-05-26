"""
LLM-based GPS spoofing detector used in the paper experiments.

Wraps a single OpenAI chat completion call per trajectory using the
paper-only prompt in ``prompts/gps_detection_prompt.py``. Each call is
stateless: history is created, the trajectory is appended, and the
response JSON is returned.

This module purposely avoids any of the production infrastructure
(MongoDB, agentic orchestrator, follow-up prompts, etc.) so it can be
used unchanged in both the live ``scripts/run_detection.py`` pipeline
and the ``evaluation/baselines/llm.py`` baseline.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
from typing import Sequence

from openai import OpenAI

from prompts.gps_detection_prompt import gps_detection_prompt

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_TEMPERATURE = 0.0


def format_trajectory_for_prompt(traj: Sequence[Sequence]) -> str:
    """Serialise a trajectory tuple-list into a compact CSV-ish payload.

    Mirrors the production format (one comma-separated row per point) so
    that the deployed prompt sees an identical input shape.
    """
    return "\n".join(",".join(str(x) for x in pt) for pt in traj)


def call_llm_for_trajectory(
    traj: Sequence[Sequence],
    client: OpenAI,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Send a single trajectory to the LLM and return the raw response string."""
    payload = format_trajectory_for_prompt(traj)
    messages = [
        {"role": "system", "content": gps_detection_prompt},
        {"role": "user", "content": payload},
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


def parse_detection_response(raw: str) -> dict | None:
    """Strip optional code fences and parse the LLM JSON. None on failure."""
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse LLM JSON: %s", exc)
        return None


def detect_batch(
    trajectories: Sequence[Sequence[Sequence]],
    client: OpenAI,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_workers: int = 10,
) -> list[str]:
    """Run the detector in parallel over a batch of trajectories.

    Returns the raw response strings in the same order as the input.
    Use :func:`parse_detection_response` per item to obtain dicts.
    """
    if not trajectories:
        return []

    def _runner(traj: Sequence[Sequence]) -> str:
        try:
            return call_llm_for_trajectory(traj, client, model_name, temperature)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_runner, trajectories))

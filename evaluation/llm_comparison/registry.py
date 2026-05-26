"""
Model registry for the LLM comparison experiment (tab:GPTs-comparison).

Pricing values follow the paper (API list prices as of February 2026).
Local-model cost columns are hosting-equivalent estimates from the paper;
override via environment variables if needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

Backend = Literal["openai", "local"]


@dataclass(frozen=True)
class LLMModelSpec:
    key: str
    display_name: str
    backend: Backend
    model_id: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    reasoning_model: bool = False
    # Published Table values for post-run comparison only — never used as outputs.
    reference_paper_accuracy_pct: float | None = None
    reference_paper_latency_s: float | None = None


def _env_model(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


# OpenAI API model IDs (override with OPENAI_MODEL_<KEY> if names change)
OPENAI_MODELS: list[LLMModelSpec] = [
    LLMModelSpec(
        key="gpt-4.1",
        display_name="GPT-4.1",
        backend="openai",
        model_id=_env_model("OPENAI_MODEL_GPT_4_1", "gpt-4.1"),
        input_cost_per_1m=2.00,
        output_cost_per_1m=8.00,
        reference_paper_accuracy_pct=98.0,
        reference_paper_latency_s=0.70,
    ),
    LLMModelSpec(
        key="gpt-4o",
        display_name="GPT-4o",
        backend="openai",
        model_id=_env_model("OPENAI_MODEL_GPT_4O", "gpt-4o"),
        input_cost_per_1m=2.50,
        output_cost_per_1m=10.00,
        reference_paper_accuracy_pct=97.0,
        reference_paper_latency_s=1.50,
    ),
    LLMModelSpec(
        key="gpt-4.1-mini",
        display_name="GPT-4.1 mini",
        backend="openai",
        model_id=_env_model("OPENAI_MODEL_GPT_4_1_MINI", "gpt-4.1-mini"),
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.60,
        reference_paper_accuracy_pct=98.0,
        reference_paper_latency_s=0.49,
    ),
    LLMModelSpec(
        key="o1-mini",
        display_name="GPT-o1-mini",
        backend="openai",
        model_id=_env_model("OPENAI_MODEL_O1_MINI", "o1-mini"),
        input_cost_per_1m=1.10,
        output_cost_per_1m=4.40,
        reasoning_model=True,
        reference_paper_accuracy_pct=99.0,
        reference_paper_latency_s=2.60,
    ),
    LLMModelSpec(
        key="gpt-5.2",
        display_name="GPT-5.2",
        backend="openai",
        model_id=_env_model("OPENAI_MODEL_GPT_5_2", "gpt-5.2"),
        input_cost_per_1m=4.00,
        output_cost_per_1m=12.00,
        reference_paper_accuracy_pct=99.0,
        reference_paper_latency_s=2.20,
    ),
]

# Hugging Face repos — set HF_* env vars to match your exact checkpoints
LOCAL_MODELS: list[LLMModelSpec] = [
    LLMModelSpec(
        key="llama-3.3-7b",
        display_name="Llama 3.3 7B",
        backend="local",
        model_id=_env_model("HF_LLAMA_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        input_cost_per_1m=0.27,
        output_cost_per_1m=0.27,
        reference_paper_accuracy_pct=96.0,
        reference_paper_latency_s=0.80,
    ),
    LLMModelSpec(
        key="ministral-7b",
        display_name="Ministral 7B",
        backend="local",
        model_id=_env_model("HF_MINISTRAL_MODEL", "mistralai/Ministral-8B-Instruct-2410"),
        input_cost_per_1m=0.10,
        output_cost_per_1m=0.10,
        reference_paper_accuracy_pct=97.0,
        reference_paper_latency_s=0.30,
    ),
    LLMModelSpec(
        key="qwen3-max-thinking",
        display_name="Qwen3-Max-Thinking",
        backend="local",
        model_id=_env_model("HF_QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        input_cost_per_1m=0.20,
        output_cost_per_1m=0.20,
        reference_paper_accuracy_pct=99.0,
        reference_paper_latency_s=3.50,
    ),
]

LLM_MODEL_REGISTRY: dict[str, LLMModelSpec] = {
    spec.key: spec for spec in (*OPENAI_MODELS, *LOCAL_MODELS)
}


def list_models(group: str | None = None) -> list[LLMModelSpec]:
    """Return registered models, optionally filtered by ``openai`` or ``local``."""
    specs = list(LLM_MODEL_REGISTRY.values())
    if group == "openai":
        return [s for s in specs if s.backend == "openai"]
    if group == "local":
        return [s for s in specs if s.backend == "local"]
    return specs

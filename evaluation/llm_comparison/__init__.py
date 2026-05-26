"""Multi-LLM comparison experiment (paper Table: GPTs-comparison)."""

from evaluation.llm_comparison.registry import LLM_MODEL_REGISTRY, list_models
from evaluation.llm_comparison.runner import run_llm_comparison

__all__ = ["LLM_MODEL_REGISTRY", "list_models", "run_llm_comparison"]

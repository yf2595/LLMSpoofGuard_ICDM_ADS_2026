"""
LLM inference backends for trajectory spoofing detection.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from openai import OpenAI

from evaluation.llm_comparison.registry import LLMModelSpec
from prompts.gps_detection_prompt import gps_detection_prompt
from src.detection_llm import format_trajectory_for_prompt, parse_detection_response

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    raw_text: str
    spoofing_detected: bool
    parsed_ok: bool
    latency_s: float
    input_tokens: int
    output_tokens: int
    error: str | None = None


class LLMProvider(ABC):
    def __init__(self, spec: LLMModelSpec):
        self.spec = spec

    @abstractmethod
    def detect_trajectory(self, traj: Sequence[Sequence]) -> InferenceResult:
        ...

    def unload(self) -> None:
        """Release GPU memory (local models only)."""


class OpenAIProvider(LLMProvider):
    def __init__(self, spec: LLMModelSpec, client: OpenAI | None = None):
        super().__init__(spec)
        import os

        self.client = client or OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def detect_trajectory(self, traj: Sequence[Sequence]) -> InferenceResult:
        user_content = format_trajectory_for_prompt(traj)
        t0 = time.perf_counter()

        try:
            if self.spec.reasoning_model:
                response = self.client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": f"{gps_detection_prompt}\n\n{user_content}",
                        },
                    ],
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=[
                        {"role": "system", "content": gps_detection_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.0,
                )
            raw = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
            err = None
        except Exception as exc:
            raw = ""
            in_tok = out_tok = 0
            err = str(exc)
            logger.error("[%s] API error: %s", self.spec.key, exc)

        latency = time.perf_counter() - t0
        parsed = parse_detection_response(raw)
        flagged = bool(parsed.get("spoofing_detected")) if parsed else False

        return InferenceResult(
            raw_text=raw,
            spoofing_detected=flagged,
            parsed_ok=parsed is not None,
            latency_s=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            error=err,
        )


class LocalHuggingFaceProvider(LLMProvider):
    """Run an instruction-tuned causal LM on a single CUDA device (e.g. A100)."""

    def __init__(
        self,
        spec: LLMModelSpec,
        device: str | None = None,
        max_new_tokens: int = 1024,
    ):
        super().__init__(spec)
        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.max_new_tokens = max_new_tokens
        self.device = device or os.environ.get("LLM_LOCAL_DEVICE", "cuda:0")

        logger.info(
            "Loading local model %s (%s) on %s ...",
            spec.display_name,
            spec.model_id,
            self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def detect_trajectory(self, traj: Sequence[Sequence]) -> InferenceResult:
        user_content = format_trajectory_for_prompt(traj)
        messages = [
            {"role": "system", "content": gps_detection_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt_text = f"{gps_detection_prompt}\n\n{user_content}\n"

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[-1]

        t0 = time.perf_counter()
        try:
            with self._torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = output_ids[0, input_len:]
            raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            in_tok = int(input_len)
            out_tok = int(new_tokens.shape[-1])
            err = None
        except Exception as exc:
            raw = ""
            in_tok = int(input_len)
            out_tok = 0
            err = str(exc)
            logger.error("[%s] local inference error: %s", self.spec.key, exc)

        latency = time.perf_counter() - t0
        parsed = parse_detection_response(raw)
        flagged = bool(parsed.get("spoofing_detected")) if parsed else False

        return InferenceResult(
            raw_text=raw,
            spoofing_detected=flagged,
            parsed_ok=parsed is not None,
            latency_s=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            error=err,
        )

    def unload(self) -> None:
        del self.model
        del self.tokenizer
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        logger.info("Unloaded local model %s.", self.spec.display_name)


def create_provider(spec: LLMModelSpec, client: OpenAI | None = None) -> LLMProvider:
    if spec.backend == "openai":
        return OpenAIProvider(spec, client=client)
    if spec.backend == "local":
        return LocalHuggingFaceProvider(spec)
    raise ValueError(f"Unknown backend: {spec.backend}")

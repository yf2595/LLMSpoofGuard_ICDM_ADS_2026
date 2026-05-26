"""
Prompt ablation variants for LLMSpoofGuard (paper Table: tab:prompt_ablation).

Builds four prompt configurations from the deployed prompt by toggling the
category bank and few-shot example blocks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from prompts.gps_detection_prompt import gps_detection_prompt

PromptVariantKey = Literal[
    "zero_shot_no_bank",
    "zero_shot_category_bank",
    "few_shot_no_bank",
    "few_shot_category_bank_unknown",
]

_SECTION_SPLIT = re.compile(
    r"(?=^={64}\n\d+\.)",
    re.MULTILINE,
)

_CATEGORY_LINE = re.compile(r'^\s*"spoofing_category":.*\n', re.MULTILINE)


@dataclass(frozen=True)
class PromptVariantSpec:
    key: PromptVariantKey
    display_name: str
    include_category_bank: bool
    include_few_shot: bool
    examples_include_category: bool
    output_include_category: bool


PROMPT_VARIANTS: dict[PromptVariantKey, PromptVariantSpec] = {
    "zero_shot_no_bank": PromptVariantSpec(
        key="zero_shot_no_bank",
        display_name="Zero-shot, no category bank",
        include_category_bank=False,
        include_few_shot=False,
        examples_include_category=False,
        output_include_category=False,
    ),
    "zero_shot_category_bank": PromptVariantSpec(
        key="zero_shot_category_bank",
        display_name="Zero-shot + category bank",
        include_category_bank=True,
        include_few_shot=False,
        examples_include_category=False,
        output_include_category=True,
    ),
    "few_shot_no_bank": PromptVariantSpec(
        key="few_shot_no_bank",
        display_name="Few-shot, no category bank",
        include_category_bank=False,
        include_few_shot=True,
        examples_include_category=False,
        output_include_category=False,
    ),
    "few_shot_category_bank_unknown": PromptVariantSpec(
        key="few_shot_category_bank_unknown",
        display_name="Few-shot + category bank + Unknown",
        include_category_bank=True,
        include_few_shot=True,
        examples_include_category=True,
        output_include_category=True,
    ),
}

PROMPT_VARIANT_ORDER: list[PromptVariantKey] = [
    "zero_shot_no_bank",
    "zero_shot_category_bank",
    "few_shot_no_bank",
    "few_shot_category_bank_unknown",
]


def _split_sections(prompt: str) -> dict[str, str]:
    parts = _SECTION_SPLIT.split(prompt.strip())
    sections: dict[str, str] = {}
    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = re.match(r"^=+\n(\d+\.\s.+?)\n=+", part, re.DOTALL)
        if match:
            sections[match.group(1)] = part
    return sections


def _build_output_section(include_category: bool) -> str:
    category_field = (
        '    "spoofing_category": "<one of the closed set in Section 5>",\n'
        if include_category
        else ""
    )
    return f"""\
================================================================
9. OUTPUT REQUIREMENTS
================================================================
Return a single valid JSON object - no markdown, no commentary, no code
fences. Use exactly this schema:

{{
  "spoofing_detected": <bool>,
  "confidence": <float 0.0-1.0>,
  "spoofing_data": {{
    "spoofing_begin_point": {{ "latitude": <float>, "longitude": <float>, "altitude": <float>, "country": "<string>", "velocity": <float>, "heading": <float>, "timestamp": "<string>" }},
    "spoofing_locations": [ {{ "latitude": <float>, "longitude": <float>, "altitude": <float>, "country": "<string>", "velocity": <float>, "heading": <float>, "timestamp": "<string>" }} ],
    "spoofing_end_point": {{ "latitude": <float>, "longitude": <float>, "altitude": <float>, "country": "<string>", "velocity": <float>, "heading": <float>, "timestamp": "<string>" }},
    "spoofing_reason": "<string>",
{category_field}    "spoofing_time_frame": {{ "begin_time": "<string>", "end_time": "<string>" }}
  }},
  "time_frame": {{ "begin_time": "<string>", "end_time": "<string>" }},
  "manufacturer": "<string>",
  "model": "<string>"
}}

If no spoofing is detected, set "spoofing_detected": false and
"spoofing_data": null. Always populate "time_frame", "manufacturer",
and "model"."""


def _strip_categories_from_examples(section: str) -> str:
    return _CATEGORY_LINE.sub("", section)


def build_prompt_variant(spec: PromptVariantSpec) -> str:
    """Assemble a prompt for one ablation configuration."""
    if (
        spec.key == "few_shot_category_bank_unknown"
        and spec.include_category_bank
        and spec.include_few_shot
        and spec.examples_include_category
        and spec.output_include_category
    ):
        return gps_detection_prompt

    sections = _split_sections(gps_detection_prompt)
    intro = gps_detection_prompt.split("================================================================\n1. BACKGROUND")[0].strip()
    ordered_keys = [
        "1. BACKGROUND: GPS SPOOFING",
        "2. INPUT FORMAT",
        "3. REASONING SCOPE",
        "4. SPOOFING DETECTION RULES (PHYSICAL CONSISTENCY)",
        "5. SPOOFING CATEGORY BANK (CLOSED SET)",
        "6. CONFIDENCE CALIBRATION",
        "7. AIRCRAFT METADATA NORMALIZATION",
        "8. FEW-SHOT EXAMPLES",
        "9. OUTPUT REQUIREMENTS",
    ]

    body_parts: list[str] = [intro]
    for key in ordered_keys:
        if key.startswith("5.") and not spec.include_category_bank:
            continue
        if key.startswith("8.") and not spec.include_few_shot:
            continue
        if key.startswith("9."):
            body_parts.append(_build_output_section(spec.output_include_category))
            continue
        section = sections.get(key)
        if section is None:
            continue
        if key.startswith("8.") and not spec.examples_include_category:
            section = _strip_categories_from_examples(section)
        body_parts.append(section)

    return "\n\n".join(part.strip() for part in body_parts if part.strip()) + "\n"


def get_prompt_variant(key: PromptVariantKey) -> str:
    return build_prompt_variant(PROMPT_VARIANTS[key])


def list_prompt_variants() -> list[PromptVariantSpec]:
    return [PROMPT_VARIANTS[k] for k in PROMPT_VARIANT_ORDER]

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class StructuredParseResult:
    data: dict[str, Any] | None
    warnings: list[str]


def _extract_json_candidate(raw: str) -> str | None:
    block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if block_match:
        return block_match.group(1)

    first = raw.find("{")
    last = raw.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    return raw[first : last + 1]


def parse_structured_output(raw: str, expected_fields: list[str]) -> StructuredParseResult:
    warnings: list[str] = []
    candidate = _extract_json_candidate(raw.strip())
    if candidate is None:
        return StructuredParseResult(
            data=None,
            warnings=["Could not find JSON object in model output."],
        )

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return StructuredParseResult(
            data=None,
            warnings=["Model returned malformed JSON."],
        )

    if not isinstance(parsed, dict):
        return StructuredParseResult(
            data=None,
            warnings=["Model returned JSON that is not an object."],
        )

    normalized: dict[str, Any] = {}
    for field in expected_fields:
        value = parsed.get(field)
        normalized[field] = value
        if value is None:
            warnings.append(f"Missing expected field: {field}")

    extra_fields = [key for key in parsed.keys() if key not in expected_fields]
    if extra_fields:
        warnings.append(f"Ignored extra fields: {', '.join(extra_fields)}")

    return StructuredParseResult(data=normalized, warnings=warnings)

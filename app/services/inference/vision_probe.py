from __future__ import annotations

import base64
import re

# Minimal 1×1 white PNG for a cheap multimodal capability probe.
_PROBE_IMAGE_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c\xf8\xff"
    b"\xff?\x00\x05\xfe\x02\xfe\xa7\xd6\xe9Q\x00\x00\x00\x00IEND\xaeB`\x82"
)
_PROBE_PROMPT = "Reply with the single character x."
_PROBE_MAX_TOKENS = 16

_VISION_NAME_HINT = re.compile(
    r"(?i)(vision|[-_/]vl[-_/]|[-_]vl$|^vl[-_/]|llava|glm-ocr|qwen[-_]?\d*[-_]?vl|"
    r"internvl|minicpm[-_]?v|pixtral|moondream|bakllava|cogvlm|idefics|fuyu|"
    r"llama[-_]?3\.2[-_]vision|gemma[-_]?3.*vision)"
)
_NON_VISION_NAME_HINT = re.compile(
    r"(?i)(embed|embedding|whisper|rerank|guard|moderation|text[-_]?only|"
    r"speech|tts|asr|bge-|e5-|nomic-embed)"
)

_VISION_PROBE_CACHE: dict[tuple[str, str], bool] = {}


def guess_vision_from_name(model: str) -> bool | None:
    """Return True/False when the model id strongly suggests vision or not."""
    name = model.strip()
    if not name:
        return None
    if _NON_VISION_NAME_HINT.search(name):
        return False
    if _VISION_NAME_HINT.search(name):
        return True
    return None


def probe_cache_key(*, base_url: str, model: str) -> tuple[str, str]:
    return (base_url.rstrip("/"), model.strip())


def get_cached_vision_probe(base_url: str, model: str) -> bool | None:
    return _VISION_PROBE_CACHE.get(probe_cache_key(base_url=base_url, model=model))


def set_cached_vision_probe(*, base_url: str, model: str, supports: bool) -> None:
    _VISION_PROBE_CACHE[probe_cache_key(base_url=base_url, model=model)] = supports


def probe_request_body(*, model: str) -> dict[str, object]:
    encoded = base64.b64encode(_PROBE_IMAGE_PNG).decode("ascii")
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": _PROBE_MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROBE_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ],
            }
        ],
    }


def response_indicates_no_vision(status_code: int, body_text: str) -> bool:
    if status_code not in {400, 404, 422}:
        return False
    lowered = body_text.lower()
    markers = (
        "multimodal",
        "image_url",
        "image url",
        "vision",
        "does not support",
        "not support",
        "unsupported",
        "invalid type",
        "content type",
    )
    return any(marker in lowered for marker in markers)

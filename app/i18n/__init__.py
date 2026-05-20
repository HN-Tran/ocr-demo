"""UI locale catalogs (en / de) for templates and static clients."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_LOCALES_DIR = Path(__file__).resolve().parent / "locales"
SUPPORTED_LOCALES = ("en", "de")
DEFAULT_LOCALE = "en"


def normalize_locale(value: str | None) -> str:
    raw = (value or "").strip().lower().replace("_", "-")
    if raw.startswith("de"):
        return "de"
    return "en"


@lru_cache(maxsize=len(SUPPORTED_LOCALES))
def load_messages(locale: str) -> dict[str, str]:
    code = normalize_locale(locale)
    path = _LOCALES_DIR / f"{code}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Locale file {path} must be a JSON object.")
    return {str(k): str(v) for k, v in data.items()}


def t(locale: str, key: str, **params: Any) -> str:
    messages = load_messages(locale)
    text = messages.get(key, key)
    for name, value in params.items():
        text = text.replace("{" + name + "}", str(value))
    return text


def resolve_locale(
    *,
    settings_locale: str,
    cookie: str | None = None,
    query: str | None = None,
) -> str:
    if query:
        return normalize_locale(query)
    if cookie:
        return normalize_locale(cookie)
    return normalize_locale(settings_locale)

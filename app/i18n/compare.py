"""Localized compare-engine labels for the UI."""

from __future__ import annotations

from app.i18n import normalize_locale, t
from app.services.compare_engines import available_engines as _available_engines

_ENGINE_KEYS = {
    "azure": "engine_azure",
    "local_models": "engine_local_models",
    "self_peer": "engine_self_peer",
    "google_vision": "engine_google_vision",
    "plain_text": "engine_plain_text",
}


def available_engines(locale: str | None = None) -> list[dict[str, str]]:
    code = normalize_locale(locale)
    out: list[dict[str, str]] = []
    for entry in _available_engines():
        key = _ENGINE_KEYS.get(entry["name"], entry["name"])
        out.append({"name": entry["name"], "label": t(code, key)})
    return out

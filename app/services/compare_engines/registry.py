"""Engine-Auswahl per Name aus den Form-Parametern von ``/api/compare``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .azure import AzureEngine
from .base import Engine, EngineResult
from .google_vision import GoogleVisionEngine
from .local_models import LocalModelsEngine
from .plain_text import PlainTextEngine
from .self_peer import SelfPeerEngine

if TYPE_CHECKING:
    from app.services.backend_router import OCRBackendRouter


def available_engines() -> list[dict[str, str]]:
    """Liste der unterstützten Engines fürs Frontend (Dropdown)."""
    return [
        {"name": AzureEngine.name, "label": AzureEngine.label},
        {"name": LocalModelsEngine.name, "label": LocalModelsEngine.label},
        {"name": SelfPeerEngine.name, "label": SelfPeerEngine.label},
        {"name": GoogleVisionEngine.name, "label": GoogleVisionEngine.label},
        {"name": PlainTextEngine.name, "label": PlainTextEngine.label},
    ]


def build_engine(
    name: str,
    config: dict[str, Any],
    *,
    verify_ssl: bool,
    pipeline: OCRBackendRouter | None = None,
) -> Engine:
    """Engine-Instanz aus Form-Konfig bauen.

    ``config`` ist ein flaches Dict aus den Form-Feldern (siehe Frontend
    bzw. ``/api/compare``-Signatur). Unbekannte Felder werden ignoriert.
    ``pipeline`` ist nur für ``local_models`` nötig (selbe Pipeline, anderes Modell).
    """
    normalized = (name or "azure").strip().lower()
    if normalized == AzureEngine.name:
        return AzureEngine(
            endpoint=str(config.get("azure_endpoint") or "").strip(),
            key=str(config.get("azure_key") or "").strip(),
            verify_ssl=verify_ssl,
        )
    if normalized == LocalModelsEngine.name:
        if pipeline is None:
            raise ValueError("local_models benötigt eine lokale Pipeline.")
        return LocalModelsEngine(
            pipeline=pipeline,
            model=str(config.get("their_model") or "").strip(),
            backend=(str(config.get("backend") or "").strip() or None),
        )
    if normalized == SelfPeerEngine.name:
        return SelfPeerEngine(
            base_url=str(config.get("peer_base_url") or "").strip(),
            backend=(str(config.get("peer_backend") or "").strip() or None),
            model=(str(config.get("peer_model") or "").strip() or None),
            verify_ssl=verify_ssl,
        )
    if normalized == GoogleVisionEngine.name:
        return GoogleVisionEngine(
            api_key=str(config.get("google_api_key") or "").strip(),
            verify_ssl=verify_ssl,
        )
    if normalized == PlainTextEngine.name:
        return PlainTextEngine(
            url=str(config.get("plain_text_url") or "").strip(),
            method=str(config.get("plain_text_method") or "POST").strip() or "POST",
            text_field=str(config.get("plain_text_field") or "text").strip() or "text",
            auth_header_name=str(config.get("plain_text_auth_header") or "").strip() or None,
            auth_header_value=str(config.get("plain_text_auth_value") or "").strip() or None,
            verify_ssl=verify_ssl,
        )
    raise ValueError(f"Unbekannte Vergleichs-Engine: {name!r}")


__all__ = [
    "Engine",
    "EngineResult",
    "available_engines",
    "build_engine",
]

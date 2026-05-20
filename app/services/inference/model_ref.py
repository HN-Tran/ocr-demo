from __future__ import annotations

SUPPORTED_INFERENCE_PROVIDERS = ("ollama", "openai_compatible")


def parse_model_ref(
    model: str | None,
    *,
    inference_provider: str | None,
    default_provider: str,
    known_providers: set[str],
) -> tuple[str, str | None]:
    """Resolve ``(provider_id, model_id)`` from request fields.

    Accepts qualified ids ``provider/model``. When ``model`` has no provider
    prefix, ``inference_provider`` or ``default_provider`` is used.
    """
    default = default_provider.strip().lower()
    explicit_provider = (inference_provider or "").strip().lower() or None
    raw_model = (model or "").strip() or None

    if raw_model and "/" in raw_model:
        prefix, _, remainder = raw_model.partition("/")
        provider_candidate = prefix.strip().lower()
        model_candidate = remainder.strip()
        if provider_candidate in known_providers and model_candidate:
            return provider_candidate, model_candidate
        if provider_candidate in known_providers and not model_candidate:
            raise ValueError(f"Modellname fehlt in qualifizierter Referenz {raw_model!r}.")

    if explicit_provider:
        if explicit_provider not in known_providers:
            raise ValueError(
                f"Unbekannter Inference-Provider '{explicit_provider}'. "
                f"Verfügbar: {', '.join(sorted(known_providers))}"
            )
        return explicit_provider, raw_model

    return default, raw_model


def format_model_ref(provider_id: str, model_id: str) -> str:
    return f"{provider_id}/{model_id}"

"""Backward-compatible re-exports; prefer ``app.services.inference``."""

from app.services.inference.ollama import OllamaClient, OllamaError

__all__ = ["OllamaClient", "OllamaError"]

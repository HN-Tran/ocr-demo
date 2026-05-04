"""Schnittstelle für externe OCR-Engines im /compare-Flow.

Jede Engine liefert ein einheitliches ``EngineResult`` zurück, sodass der
Rest der Compare-Pipeline (Diff, Metriken) engine-agnostisch bleibt.
Engines ohne Bounding-Box-Output liefern leere Polygone — der Diff
funktioniert dann textbasiert weiter, nur das Overlay bleibt leer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class EngineResult:
    text: str
    """Plain text wie es die Engine geliefert hat (komplettes Dokument)."""

    words_per_page: list[list[dict[str, Any]]] = field(default_factory=list)
    """Pro Seite eine Liste von ``{content, polygon, confidence}``-Wörtern.

    ``polygon`` ist ein flaches ``[x0,y0,x1,y1,x2,y2,x3,y3]`` in 0–1000-Skala
    (gleiche Konvention wie unser interner Detector). Leere Liste, wenn die
    Engine keine Boxen liefert — dann zeigt die Diff-Übersicht keinen
    Overlay, der Text-Diff funktioniert aber weiterhin.
    """

    warnings: list[str] = field(default_factory=list)
    """Engine-Warnungen, falls die Engine sie freiwillig liefert (z. B.
    fehlgeschlagene Seiten-OCR bei einem nachgeladenen lokalen Modell)."""

    raw: dict[str, Any] | None = None
    """Optional: rohe API-Antwort, nur fürs Debuggen."""


class Engine(Protocol):
    """Protokoll für eine austauschbare Vergleichs-Engine."""

    name: str
    label: str

    async def analyze(self, image_bytes: bytes, content_type: str) -> EngineResult: ...

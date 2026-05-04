"""Eingebetteten Textlayer aus einer PDF extrahieren.

Wird vom Frontend genutzt, um die Referenztext-Eingabe in der Compare-Ansicht
automatisch vorzubefüllen, falls die PDF einen brauchbaren Text-Layer mitbringt
(z. B. von Word/InDesign exportierte „digitale" PDFs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pypdfium2 as pdfium

# Codepoint-Bereiche, die als „kaputt" gelten (Private-Use Area + Steuerzeichen).
_PRIVATE_USE_RANGES = (
    (0xE000, 0xF8FF),
    (0xF0000, 0xFFFFD),
    (0x100000, 0x10FFFD),
)


@dataclass(frozen=True)
class PdfTextExtraction:
    text: str
    page_count: int
    has_text_layer: bool
    garbage_ratio: float  # 0.0 = sauber, 1.0 = komplett unleserlich


def _is_garbage_codepoint(cp: int) -> bool:
    if cp < 0x20 and cp not in (0x09, 0x0A, 0x0D):
        return True
    return any(lo <= cp <= hi for lo, hi in _PRIVATE_USE_RANGES)


def _garbage_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = sum(1 for ch in text if _is_garbage_codepoint(ord(ch)))
    return bad / len(text)


def extract(image_bytes: bytes) -> PdfTextExtraction:
    """Extrahiere den Text-Layer einer PDF.

    Gibt für nicht-PDFs oder PDFs ohne Text-Layer ein leeres Resultat zurück
    (``has_text_layer=False``). Pypdfium2 schluckt zwar binäre Eingaben, wir
    fangen aber alle Fehler ab und behandeln sie als „kein Text".
    """
    try:
        pdf = pdfium.PdfDocument(image_bytes)
    except Exception:  # noqa: BLE001
        return PdfTextExtraction("", 0, False, 0.0)

    try:
        page_count = len(pdf)
        chunks: list[str] = []
        for i in range(page_count):
            page = pdf[i]
            textpage = page.get_textpage()
            chunk = cast(str, textpage.get_text_range())
            if chunk:
                chunks.append(chunk)
            textpage.close()
            page.close()
        text = "\n\n".join(chunks).strip()
        return PdfTextExtraction(
            text=text,
            page_count=page_count,
            has_text_layer=bool(text),
            garbage_ratio=_garbage_ratio(text),
        )
    finally:
        pdf.close()

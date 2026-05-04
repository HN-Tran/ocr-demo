"""OCR-Vergleichsmetriken für die /api/compare-Antwort.

Drei Gruppen, die die UI als Tabs darstellt:

* Intrinsisch — beschreibende Statistiken pro Engine (Tokens, Zeichen,
  Konfidenz, Latenz). Keine Qualitätsaussage, nur Volumen/Performance.
* Vergleich   — paarweise Maße zwischen beiden Engines (normalisierte
  Levenshtein-Distanzen, Token-Jaccard, Token-Precision/Recall/F1).
  Bewusst NICHT als CER/WER bezeichnet, da diese Begriffe per Definition
  eine Referenz erfordern.
* Referenz    — echte CER/WER und Token-F1 gegen vom Nutzer gelieferte
  Ground-Truth, sofern vorhanden.
"""

from __future__ import annotations

import re
from typing import Any

from rapidfuzz.distance import Levenshtein

from eval.metrics import cer as _cer_against_reference
from eval.metrics import wer as _wer_against_reference

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def _safe_div(numer: float, denom: float) -> float:
    return numer / denom if denom else 0.0


def _norm_levenshtein(a: list[str] | str, b: list[str] | str) -> float:
    """Normalised symmetric edit distance in [0, 1].

    Defined as ``Lev(a,b) / max(len(a), len(b))``. Symmetric — neither
    side is treated as reference. Returned as a fraction (0 = identical,
    1 = completely different) to mirror CER/WER scaling, but explicitly
    NOT called CER/WER since neither side is ground truth.
    """
    if not a and not b:
        return 0.0
    distance = Levenshtein.distance(a, b)
    return distance / max(len(a), len(b))


def _intrinsic(
    *,
    text: str,
    words_per_page: list[list[dict[str, Any]]],
    latency_ms: int | None,
) -> dict[str, Any]:
    flat_words = [w for page in words_per_page for w in page]
    confidences: list[float] = []
    for w in flat_words:
        conf = w.get("confidence")
        if isinstance(conf, (int, float)) and conf > 0:
            confidences.append(float(conf))
    avg_conf = sum(confidences) / len(confidences) if confidences else None
    return {
        "tokens": len(_tokenize(text)),
        "chars": len(text),
        "avg_confidence": avg_conf,
        "latency_ms": latency_ms,
        "word_box_count": len(flat_words),
    }


def _comparison(our_text: str, their_text: str) -> dict[str, Any]:
    our_tokens = _tokenize(our_text)
    their_tokens = _tokenize(their_text)
    our_set = set(our_tokens)
    their_set = set(their_tokens)
    intersection = len(our_set & their_set)
    union = len(our_set | their_set)

    # Treat 'theirs' as the reference side for the asymmetric P/R/F1 view.
    # Precision = how many of our tokens are also in theirs (anti-hallucination).
    # Recall    = how many of their tokens we also produced (coverage).
    precision = _safe_div(intersection, len(our_set))
    recall = _safe_div(intersection, len(their_set))
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "delta_char": _norm_levenshtein(our_text, their_text),
        "delta_word": _norm_levenshtein(our_tokens, their_tokens),
        "token_jaccard": _safe_div(intersection, union),
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
        "reference_side": "theirs",
    }


def _reference_side(reference_text: str, hypothesis_text: str) -> dict[str, Any]:
    ref_tokens = set(_tokenize(reference_text))
    hyp_tokens = set(_tokenize(hypothesis_text))
    intersection = len(ref_tokens & hyp_tokens)
    precision = _safe_div(intersection, len(hyp_tokens))
    recall = _safe_div(intersection, len(ref_tokens))
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "cer": _cer_against_reference(reference_text, hypothesis_text),
        "wer": _wer_against_reference(reference_text, hypothesis_text),
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }


def compute(
    *,
    our_text: str,
    our_words_per_page: list[list[dict[str, Any]]],
    our_latency_ms: int | None,
    their_text: str,
    their_words_per_page: list[list[dict[str, Any]]],
    their_latency_ms: int | None,
    reference_text: str | None = None,
) -> dict[str, Any]:
    """Top-level builder consumed by /api/compare and the UI metrics panel."""
    intrinsic = {
        "ours": _intrinsic(
            text=our_text,
            words_per_page=our_words_per_page,
            latency_ms=our_latency_ms,
        ),
        "theirs": _intrinsic(
            text=their_text,
            words_per_page=their_words_per_page,
            latency_ms=their_latency_ms,
        ),
    }
    comparison = _comparison(our_text, their_text)
    reference: dict[str, Any] | None = None
    if reference_text and reference_text.strip():
        reference = {
            "ours": _reference_side(reference_text, our_text),
            "theirs": _reference_side(reference_text, their_text),
            "char_count": len(reference_text),
            "token_count": len(_tokenize(reference_text)),
        }
    return {
        "intrinsic": intrinsic,
        "comparison": comparison,
        "reference": reference,
    }

from __future__ import annotations


def _levenshtein(seq_a: list[str], seq_b: list[str]) -> int:
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)

    previous = list(range(len(seq_b) + 1))
    for i, token_a in enumerate(seq_a, start=1):
        current = [i]
        for j, token_b in enumerate(seq_b, start=1):
            cost = 0 if token_a == token_b else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref = list(reference)
    hyp = list(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def field_accuracy(reference: dict, prediction: dict) -> float:
    if not reference:
        return 1.0
    total = len(reference)
    correct = 0
    for key, ref_value in reference.items():
        pred_value = prediction.get(key) if prediction else None
        if _normalize(ref_value) == _normalize(pred_value):
            correct += 1
    return correct / total


def _normalize(value: object | None) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()

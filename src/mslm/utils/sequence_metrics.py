"""Strict token-sequence metrics for the A3/CIF Etapa 4 correction.

``legacy_prefix_exact`` is the historical metric: it only compares the first
``len(target)`` predicted tokens, so an over/under-predicted sequence whose
prefix happens to match still counts as exact. ``strict_exact`` additionally
requires the predicted token count to equal the target length, per the
Etapa 4 correction plan.
"""
from __future__ import annotations

from typing import Sequence


def levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, bj in enumerate(b, start=1):
            cost = 0 if ai == bj else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def token_error_rate(pred: Sequence[int], target: Sequence[int]) -> float:
    if not target:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, target) / len(target)


def token_edit_similarity(ter: float) -> float:
    return max(0.0, 1.0 - ter)


def legacy_prefix_exact(pred: Sequence[int], target: Sequence[int]) -> bool:
    """Historical metric: prefix of length len(target) must match. Ignores count."""
    return list(pred[: len(target)]) == list(target)


def strict_exact(pred: Sequence[int], target: Sequence[int]) -> bool:
    """pred_count == target_len AND every token matches."""
    return len(pred) == len(target) and list(pred) == list(target)

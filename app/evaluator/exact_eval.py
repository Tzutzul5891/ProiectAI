"""Exact / rule-based evaluation utilities (placeholder).

Some problems can be evaluated deterministically without any embeddings model
(e.g., coordinate checks, move validity, constraints satisfaction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationResult:
    """Standard evaluation result object."""

    score: float
    message: str
    details: dict[str, Any] | None = None


def evaluate_exact(user_text: str, expected_text: str) -> EvaluationResult:
    """Deterministic exact-match evaluator (minimal baseline)."""

    user_norm = (user_text or "").strip()
    expected_norm = (expected_text or "").strip()

    if not expected_norm:
        return EvaluationResult(score=0.0, message="Nu există un răspuns corect disponibil pentru comparație.")

    if user_norm == expected_norm:
        return EvaluationResult(score=100.0, message="✅ Răspuns exact corect.")

    return EvaluationResult(score=0.0, message="❌ Răspuns diferit de gold standard.")

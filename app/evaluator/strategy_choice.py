from __future__ import annotations

from typing import Any


_DIACRITICS_MAP = str.maketrans(
    {
        "ă": "a",
        "â": "a",
        "î": "i",
        "ș": "s",
        "ş": "s",
        "ț": "t",
        "ţ": "t",
        "Ă": "a",
        "Â": "a",
        "Î": "i",
        "Ș": "s",
        "Ş": "s",
        "Ț": "t",
        "Ţ": "t",
        "–": "-",
        "—": "-",
    }
)


def _norm(text: str) -> str:
    return str(text).translate(_DIACRITICS_MAP).lower()


def evaluate_strategy_choice(
    chosen_strategy: str | None,
    justification: str | None,
    *,
    correct_strategy: str,
    grading: dict[str, Any] | None = None,
) -> tuple[float, str, dict[str, Any]]:
    """Evaluate a strategy-choice answer (exact + heuristic partial credit)."""

    chosen = str(chosen_strategy or "").strip()
    if not chosen:
        return 0.0, "Necompletat: nu ai ales o strategie.", {"missing_choice": True}

    correct = str(correct_strategy or "").strip()
    grading = dict(grading or {})
    partial_scores_raw = grading.get("partial_scores") or {}
    partial_scores: dict[str, float] = {}
    if isinstance(partial_scores_raw, dict):
        for k, v in partial_scores_raw.items():
            try:
                partial_scores[str(k)] = float(v)
            except Exception:
                continue

    keywords_raw = grading.get("keywords") or []
    keywords: list[str] = [str(k).strip() for k in keywords_raw if str(k).strip()]

    if chosen == correct:
        score = 100.0
        message = "Corect: strategia aleasă este potrivită."
        status = "correct"
    elif chosen in partial_scores:
        score = float(partial_scores[chosen])
        message = "Parțial: strategia aleasă este apropiată, dar nu cea mai potrivită."
        status = "partial"
    else:
        score = 0.0
        message = "Incorect: strategia aleasă nu este potrivită pentru instanța dată."
        status = "wrong"

    justification_text = str(justification or "")
    justification_norm = _norm(justification_text)

    matched_keywords: list[str] = []
    missing_keywords: list[str] = []
    for kw in keywords:
        if _norm(kw) in justification_norm:
            matched_keywords.append(kw)
        else:
            missing_keywords.append(kw)

    details: dict[str, Any] = {
        "status": status,
        "chosen_strategy": chosen,
        "correct_strategy": correct,
        "keyword_total": len(keywords),
        "keyword_matches": matched_keywords,
        "keyword_missing": missing_keywords,
    }

    if keywords:
        message = f"{message} Cuvinte-cheie în justificare: {len(matched_keywords)}/{len(keywords)}."
        if not justification_text.strip():
            details["missing_justification"] = True

    return float(score), message, details


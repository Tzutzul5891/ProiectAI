from __future__ import annotations

import re
from typing import Any


_INT_RE = re.compile(r"^\s*-?\d+\s*$")


def _parse_int_field(text: str | None, *, field_name: str) -> tuple[int | None, str | None]:
    raw = str(text or "").strip()
    if raw == "":
        return None, None
    if not _INT_RE.match(raw):
        return None, f"Câmpul '{field_name}' trebuie să fie un întreg (ex: -2, 0, 7)."
    try:
        return int(raw), None
    except Exception:
        return None, f"Câmpul '{field_name}' nu a putut fi interpretat ca întreg."


def evaluate_alpha_beta_answer(
    root_value_text: str | None,
    visited_leaves_text: str | None,
    *,
    expected: dict[str, Any] | None,
) -> tuple[float, str, dict[str, Any]]:
    """Exact/partial evaluation for Cerința 4 (Alpha-Beta).

    Scoring:
      - 100% if both root value and visited leaves count are correct
      - 50% if exactly one of them is correct
      - 0% otherwise (or invalid/empty)
    """

    if not expected or not isinstance(expected, dict):
        return 0.0, "Nu există un răspuns gold disponibil pentru această întrebare.", {"missing_expected": True}

    expected_value = expected.get("value_at_root")
    expected_leaves = expected.get("visited_leaves_count")

    user_value, err_value = _parse_int_field(root_value_text, field_name="valoare_rădăcină")
    user_leaves, err_leaves = _parse_int_field(visited_leaves_text, field_name="frunze_vizitate")

    errors = [e for e in [err_value, err_leaves] if e]

    if user_value is None and user_leaves is None and not errors:
        return 0.0, "Necompletat.", {"missing": True, "expected_value": expected_value, "expected_leaves": expected_leaves}

    if errors:
        return 0.0, "Răspuns invalid (format).", {"errors": errors, "expected_value": expected_value, "expected_leaves": expected_leaves}

    value_ok = user_value == expected_value
    leaves_ok = user_leaves == expected_leaves

    if value_ok and leaves_ok:
        score = 100.0
        msg = "✅ Corect: valoarea din rădăcină și numărul de frunze vizitate."
        status = "correct"
    elif value_ok or leaves_ok:
        score = 50.0
        if value_ok:
            msg = "Parțial: valoarea din rădăcină este corectă, dar numărul de frunze vizitate este greșit."
        else:
            msg = "Parțial: numărul de frunze vizitate este corect, dar valoarea din rădăcină este greșită."
        status = "partial"
    else:
        score = 0.0
        msg = "❌ Incorect: atât valoarea din rădăcină cât și numărul de frunze vizitate sunt greșite."
        status = "wrong"

    details: dict[str, Any] = {
        "status": status,
        "got": {"value_at_root": user_value, "visited_leaves_count": user_leaves},
        "expected": {"value_at_root": expected_value, "visited_leaves_count": expected_leaves},
    }

    visited_order = expected.get("visited_leaves_order")
    if isinstance(visited_order, list):
        details["expected_visited_order"] = visited_order

    return score, msg, details


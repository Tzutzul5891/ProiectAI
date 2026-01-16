from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_PAIR_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*[:=\\-]\s*(.+?)\s*$")
_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")


@dataclass(frozen=True)
class ParsedCSPAssignment:
    assignment: dict[str, Any]
    errors: list[str]


def _coerce_token_to_domain(token: str, domain: list[Any]) -> Any | None:
    """Try to map a user token to an actual domain value."""

    raw = str(token).strip()
    if raw == "":
        return None

    # Fast path: exact match by string representation.
    for v in domain:
        if raw == str(v):
            return v

    # Numeric coercion (common for int domains).
    if _INT_RE.match(raw):
        try:
            cand_int = int(raw)
            if cand_int in domain:
                return cand_int
        except Exception:
            pass

    if _FLOAT_RE.match(raw):
        try:
            cand_float = float(raw)
            if cand_float in domain:
                return cand_float
        except Exception:
            pass

    # Case-insensitive for string domains.
    raw_upper = raw.upper()
    for v in domain:
        if isinstance(v, str) and v.upper() == raw_upper:
            return v

    return None


def parse_csp_assignment_text(
    text: str | None,
    *,
    variables: list[str],
    domains: dict[str, list[Any]],
) -> ParsedCSPAssignment:
    """Parse answers like: `X=1, Y=2` or multiline `X:1` etc."""

    assignment: dict[str, Any] = {}
    errors: list[str] = []

    if not text or not str(text).strip():
        return ParsedCSPAssignment(assignment={}, errors=[])

    var_set = {str(v) for v in variables}

    chunks = re.split(r"[,\n;]+", str(text))
    for raw in chunks:
        part = raw.strip()
        if not part:
            continue
        m = _PAIR_RE.match(part)
        if not m:
            errors.append(f"Format invalid: '{part}'. Folosește 'Variabila=valoare' (ex: X=1).")
            continue

        var = m.group(1).strip()
        val_token = m.group(2).strip()
        if var not in var_set:
            errors.append(f"Variabilă necunoscută: '{var}'. Variabile permise: {', '.join(variables)}.")
            continue
        if var not in domains:
            errors.append(f"Lipsește domeniul pentru variabila '{var}'.")
            continue

        coerced = _coerce_token_to_domain(val_token, list(domains[var] or []))
        if coerced is None:
            dom_preview = ", ".join(str(x) for x in (domains[var] or []))
            errors.append(f"Valoare invalidă pentru {var}: '{val_token}'. Domeniu: {{{dom_preview}}}.")
            continue

        assignment[var] = coerced

    return ParsedCSPAssignment(assignment=assignment, errors=errors)


def evaluate_csp_backtracking_answer(
    user_text: str | None,
    *,
    variables: list[str],
    domains: dict[str, list[Any]],
    partial_assignment: dict[str, Any] | None,
    expected_solution: dict[str, Any] | None,
) -> tuple[float, str, dict[str, Any]]:
    """Score a CSP completion answer (0–100) by exact match per remaining variable."""

    if not expected_solution:
        return 0.0, "Nu există o soluție gold disponibilă pentru această instanță.", {"missing_solution": True}

    partial_assignment = dict(partial_assignment or {})
    remaining = [v for v in variables if v not in partial_assignment]
    if not remaining:
        return 100.0, "Nu există variabile de completat (asignarea este deja completă).", {"remaining": []}

    parsed = parse_csp_assignment_text(user_text, variables=variables, domains=domains)
    user = parsed.assignment

    correct: dict[str, Any] = {}
    wrong: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    extra: list[str] = []

    for v in user.keys():
        if v not in variables:
            extra.append(v)
        elif v in partial_assignment:
            extra.append(v)

    correct_count = 0
    for v in remaining:
        if v not in user:
            missing.append(v)
            continue
        if user[v] == expected_solution.get(v):
            correct[v] = user[v]
            correct_count += 1
        else:
            wrong[v] = {"got": user[v], "expected": expected_solution.get(v)}

    score = 100.0 * (correct_count / max(1, len(remaining)))
    score = max(0.0, min(100.0, float(score)))

    if correct_count == len(remaining) and not parsed.errors:
        msg = f"✅ Corect: {correct_count}/{len(remaining)} variabile."
    else:
        msg = f"Scor: {correct_count}/{len(remaining)} variabile corecte."
        if parsed.errors:
            msg += f" • Probleme parsare: {len(parsed.errors)}"

    details: dict[str, Any] = {
        "remaining": remaining,
        "correct": correct,
        "wrong": wrong,
        "missing": missing,
        "extra": extra,
        "parse_errors": parsed.errors,
    }
    return score, msg, details


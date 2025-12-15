"""Small, dependency-light utilities used across the app."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any
import re


def set_global_seed(seed: int | None) -> None:
    """Set Python/NumPy RNG seeds for reproducible runs.

    The app does not require a fixed seed, but having a single helper makes it
    easy to turn determinism on/off via config or env vars.
    """

    if seed is None:
        return

    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        # NumPy may be optional in some environments.
        pass


def ensure_text(value: Any) -> str:
    """Convert arbitrary values to a safe string representation."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def normalize_problem_output(result: Any) -> dict[str, Any]:
    """Normalize multiple generator formats into a single dict contract.

    Convention used in this repo:
      - `data`: structured content used by UI/PDF (boards, matrices, etc.)
      - `prompt`: the statement shown to the student
      - `solution`: structured solution (if available)
      - `explanation`: gold-standard explanation (text)
      - `metadata`: optional extra fields (sizes, start positions, etc.)

    Supported inputs:
      - `ProblemInstance` (from `app.modules.base_problem`) -> dict
      - legacy tuple: `(data, explanation)` -> dict with empty prompt/solution
      - dict with the expected keys -> returned as-is (shallow-copied)
    """

    if hasattr(result, "__dataclass_fields__"):
        return asdict(result)

    if isinstance(result, tuple) and len(result) == 2:
        data, explanation = result
        return {
            "data": data,
            "prompt": "",
            "solution": None,
            "explanation": ensure_text(explanation),
            "metadata": {},
        }

    if isinstance(result, dict):
        # Avoid mutating the input.
        normalized = dict(result)
        normalized.setdefault("data", None)
        normalized.setdefault("prompt", "")
        normalized.setdefault("solution", None)
        normalized.setdefault("explanation", "")
        normalized.setdefault("metadata", {})
        return normalized

    raise TypeError(f"Unsupported problem output format: {type(result)!r}")


_NASH_COORD_RE = re.compile(r"(?i)\bL\s*(\d+)\s*[-–—/ ]\s*C\s*(\d+)\b")
_NASH_EXAMPLE_PREFIX_RE = re.compile(r"(?i)ex(?:emplu)?\s*[:\-]?\s*$")


def extract_nash_coordinates(text: Any) -> list[str]:
    """Extract Nash coordinates in canonical `Lx-Cy` form from arbitrary text."""

    raw = ensure_text(text)
    if not raw.strip():
        return []

    coords: list[str] = []
    seen: set[str] = set()

    for match in _NASH_COORD_RE.finditer(raw):
        before = raw[max(0, match.start() - 12) : match.start()]
        if _NASH_EXAMPLE_PREFIX_RE.search(before):
            continue

        coord = f"L{int(match.group(1))}-C{int(match.group(2))}"
        if coord in seen:
            continue
        seen.add(coord)
        coords.append(coord)

    return coords


_CSP_PAIR_SCAN_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*[:=\-]\s*([^\s,;\n\r]+)")
_CSP_IGNORED_VALUE_TOKENS = {"valoare", "value", "..."}


def extract_csp_var_value_pairs(
    text: Any,
    *,
    allowed_variables: list[str] | None = None,
) -> dict[str, str]:
    """Extract `Var=Val` pairs from arbitrary text (best-effort)."""

    raw = ensure_text(text)
    if not raw.strip():
        return {}

    allowed_map: dict[str, str] | None = None
    if allowed_variables:
        allowed_map = {str(v).strip().upper(): str(v).strip() for v in allowed_variables if str(v).strip()}

    pairs: dict[str, str] = {}
    for match in _CSP_PAIR_SCAN_RE.finditer(raw):
        var = match.group(1).strip()
        val = match.group(2).strip()
        if not var or not val:
            continue

        if val.lower() in _CSP_IGNORED_VALUE_TOKENS:
            continue

        if allowed_map is not None:
            normalized_var = allowed_map.get(var.upper())
            if not normalized_var:
                continue
            var = normalized_var

        pairs[var] = val

    return pairs


def format_csp_pairs(pairs: dict[str, Any]) -> str:
    if not pairs:
        return ""
    return ", ".join(f"{k}={pairs[k]}" for k in sorted(pairs, key=str))


_MINMAX_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bvalue\s*[:=]\s*(-?\d+)\b"),
    re.compile(r"(?i)\bvaloare(?:\s+(?:la|in|în)\s+(?:radacina|r[aă]d[aă]cin[aă]))?\s*[:=]\s*(-?\d+)\b"),
    re.compile(r"(?i)\bvalue_at_root\s*[:=]\s*(-?\d+)\b"),
)
_MINMAX_LEAVES_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bleaves\s*[:=]\s*(\d+)\b"),
    re.compile(r"(?i)\bfrunze(?:\s+(?:evaluate|vizitate))?\s*[:=]\s*(\d+)\b"),
    re.compile(r"(?i)\bvisited_leaves(?:_count)?\s*[:=]\s*(\d+)\b"),
)


def extract_minmax_value_and_leaves(text: Any) -> tuple[str | None, str | None]:
    """Extract `value=...` and `leaves=...` from arbitrary text."""

    raw = ensure_text(text)
    if not raw.strip():
        return None, None

    value: str | None = None
    leaves: str | None = None

    for pat in _MINMAX_VALUE_PATTERNS:
        m = pat.search(raw)
        if m:
            value = m.group(1).strip()
            break

    for pat in _MINMAX_LEAVES_PATTERNS:
        m = pat.search(raw)
        if m:
            leaves = m.group(1).strip()
            break

    return value, leaves


_GC_PAIR_SCAN_RE = re.compile(r"\b(\d+)\s*[:=\-]\s*([A-Za-z0-9]+)\b")


def extract_graph_coloring_mapping(
    text: Any,
    *,
    n: int | None = None,
    color_names: list[str] | None = None,
) -> dict[int, str]:
    """Extract a node->color mapping from arbitrary text (best-effort)."""

    raw = ensure_text(text)
    if not raw.strip():
        return {}

    allowed = [str(c).strip() for c in (color_names or []) if str(c).strip()]
    allowed_upper = {c.upper(): c for c in allowed}
    k = len(allowed)

    mapping: dict[int, str] = {}
    for match in _GC_PAIR_SCAN_RE.finditer(raw):
        try:
            node = int(match.group(1))
        except Exception:
            continue

        if n is not None and not (1 <= node <= int(n)):
            continue

        token = match.group(2).strip()
        if not token:
            continue

        normalized: str | None = None
        if token.isdigit() and k:
            idx = int(token)
            if 1 <= idx <= k:
                normalized = allowed[idx - 1]
        if normalized is None:
            normalized = allowed_upper.get(token.upper())

        if normalized is None and not allowed:
            # If we don't know allowed colors, keep the raw token.
            normalized = token

        if normalized is None:
            continue

        mapping[node] = normalized

    return mapping


def format_graph_coloring_mapping(mapping: dict[int, Any]) -> str:
    if not mapping:
        return ""
    return ", ".join(f"{int(node)}:{mapping[node]}" for node in sorted(mapping))

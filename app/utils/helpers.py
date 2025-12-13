"""Small, dependency-light utilities used across the app."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any


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

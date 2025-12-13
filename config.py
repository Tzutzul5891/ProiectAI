"""App-level configuration.

The project is designed to run locally and deterministically (no LLM API calls in
runtime). Some components (e.g., SBERT) can optionally load a local model from
disk; if unavailable, evaluators should gracefully fall back to simpler
deterministic scoring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """Configuration values used across the app."""

    # If set, evaluators should only try local model paths and never download.
    local_models_only: bool = os.getenv("SMARTEST_LOCAL_MODELS_ONLY", "0") == "1"

    # SBERT model name or local path. For offline runs, set this to a local dir.
    sbert_model_name_or_path: str = os.getenv("SMARTEST_SBERT_MODEL", "all-MiniLM-L6-v2")

    # Optional global seed for reproducible generation (leave empty for randomness).
    seed: int | None = int(os.getenv("SMARTEST_SEED")) if os.getenv("SMARTEST_SEED") else None


CONFIG = AppConfig()

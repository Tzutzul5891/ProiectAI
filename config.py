"""App-level configuration.

The project is designed to run locally and deterministically (no LLM API calls in
runtime). Some components (e.g., SBERT) can optionally load a local model from
disk; if unavailable, evaluators should gracefully fall back to simpler
deterministic scoring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_flag(name: str, default: str = "0") -> bool:
    raw = os.getenv(name, default).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class AppConfig:
    """Configuration values used across the app."""

    # If set, evaluators should never download models at runtime.
    # Backwards compatible with the older `SMARTEST_LOCAL_MODELS_ONLY`.
    offline_strict: bool = _env_flag("SMARTEST_OFFLINE_STRICT") or _env_flag("SMARTEST_LOCAL_MODELS_ONLY")

    # Backwards-compat alias used by existing modules.
    local_models_only: bool = offline_strict

    # Enable SBERT semantic similarity scoring where available.
    # If disabled, evaluators must use deterministic exact/regex/algorithmic scoring only.
    enable_sbert: bool = _env_flag("SMARTEST_ENABLE_SBERT", "1")

    # SBERT model name or local path. For offline runs, set this to a local dir.
    sbert_model_name_or_path: str = os.getenv("SMARTEST_SBERT_MODEL", "all-MiniLM-L6-v2")

    # Optional global seed for reproducible generation (leave empty for randomness).
    seed: int | None = int(os.getenv("SMARTEST_SEED")) if os.getenv("SMARTEST_SEED") else None


CONFIG = AppConfig()


def _apply_offline_env(config: AppConfig) -> None:
    if not config.offline_strict:
        return

    # Make HuggingFace/Transformers behave in offline mode (avoid remote checks/downloads).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


_apply_offline_env(CONFIG)

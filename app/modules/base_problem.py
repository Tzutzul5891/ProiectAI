"""Common problem-generator contract used by the app.

Why this exists:
  - makes it clear what every generator should return
  - keeps `main.py` UI simple (no special-casing per module)
  - provides a migration path from legacy `(data, explanation)` returns

Core convention for a generated problem:
  - `data`: structured content for UI/PDF (matrix, board, etc.)
  - `prompt`: the statement shown to the student
  - `solution`: structured solution (if available)
  - `explanation`: gold-standard explanation (text)
  - `metadata`: extra fields (sizes, start positions, optimal moves, etc.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProblemInstance:
    """A single generated problem instance."""

    data: Any
    prompt: str
    solution: Any | None
    explanation: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseProblem(ABC):
    """Base class for all problem generators."""

    problem_type: str = "unknown"

    @abstractmethod
    def generate(self) -> ProblemInstance:
        """Generate a new problem instance (preferred API)."""

    def generate_problem(self):
        """Legacy API used by the current Streamlit UI.

        Returns:
            (data, explanation)
        """

        instance = self.generate()
        return instance.data, instance.explanation

    def generate_question(
        self,
        *,
        ui_label: str | None = None,
        chapter: str | None = None,
        question_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ):
        """Generate a `Question` wrapper for building tests/answer keys.

        This keeps current generators compatible with the Streamlit UI while
        enabling a richer contract for multi-question tests.
        """

        instance = self.generate()
        question_type = getattr(self, "problem_type", self.__class__.__name__)
        derived_chapter = chapter or (question_type.split(":", 1)[0] if ":" in question_type else "unknown")

        metadata = dict(instance.metadata or {})
        if ui_label is not None:
            metadata.setdefault("ui_label", ui_label)
        if extra_metadata:
            metadata.update(extra_metadata)

        # Local import to avoid circular imports at module load time.
        from app.models.test_session import Question

        return Question.from_problem_instance(
            instance,
            question_type=question_type,
            chapter=derived_chapter,
            question_id=question_id,
            extra_metadata=metadata,
        )

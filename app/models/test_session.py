"""Common models for bundling questions into tests.

Goal:
  - represent every generated item as a `Question`
  - support building a `TestSession` (N questions + separate answer key)
  - keep everything local/deterministic (no runtime LLM API calls)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from app.modules.base_problem import ProblemInstance


def _jsonable(value: Any) -> Any:
    """Best-effort conversion to JSON-serializable Python primitives."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # NumPy scalars/arrays -> Python types/lists (optional dependency)
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}

    if isinstance(value, set):
        return [_jsonable(v) for v in value]

    return str(value)


@dataclass(frozen=True)
class Question:
    """A single exam-style question + its answer key."""

    id: str
    type: str
    chapter: str
    prompt_text: str
    data: Any
    correct_answer: Any | None
    correct_explanation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new_id(prefix: str = "q") -> str:
        return f"{prefix}_{uuid4().hex[:12]}"

    @classmethod
    def from_problem_instance(
        cls,
        instance: ProblemInstance,
        *,
        question_type: str,
        chapter: str | None = None,
        question_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> "Question":
        """Create a `Question` from the existing `ProblemInstance` contract."""

        derived_chapter = chapter
        if derived_chapter is None:
            derived_chapter = question_type.split(":", 1)[0] if ":" in question_type else "unknown"

        metadata = dict(instance.metadata or {})
        if extra_metadata:
            metadata.update(extra_metadata)

        return cls(
            id=question_id or cls.new_id(),
            type=question_type,
            chapter=derived_chapter,
            prompt_text=instance.prompt or "",
            data=instance.data,
            correct_answer=instance.solution,
            correct_explanation=instance.explanation or "",
            metadata=metadata,
        )

    def to_dict(self, *, include_answer_key: bool = True) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-friendly)."""

        payload: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "chapter": self.chapter,
            "prompt_text": self.prompt_text,
            "data": _jsonable(self.data),
            "metadata": _jsonable(self.metadata),
        }

        if include_answer_key:
            payload["correct_answer"] = _jsonable(self.correct_answer)
            payload["correct_explanation"] = _jsonable(self.correct_explanation)

        return payload

    def to_public_dict(self) -> dict[str, Any]:
        """Question-only payload (no answer key)."""

        return self.to_dict(include_answer_key=False)

    def to_answer_key_dict(self) -> dict[str, Any]:
        """Answer-key-only payload (no prompt/data duplication)."""

        return {
            "id": self.id,
            "type": self.type,
            "correct_answer": _jsonable(self.correct_answer),
            "correct_explanation": _jsonable(self.correct_explanation),
        }

    def pdf_kwargs(self, *, problem_type_label: str | None = None) -> dict[str, Any]:
        """Convenience mapping for `app.utils.pdf_generator.create_pdf(**kwargs)`."""

        label = problem_type_label or self.metadata.get("ui_label") or self.type
        kwargs: dict[str, Any] = {
            "problem_type": str(label),
            "requirement": self.prompt_text,
            "matrix_data": self.data,
        }
        if "initial_state" in self.metadata:
            kwargs["hanoi_state"] = self.metadata["initial_state"]
        return kwargs


@dataclass
class TestSession:
    """A test made of multiple questions, tracked in a single session."""

    questions: list[Question] = field(default_factory=list)
    current_index: int = 0
    user_answers: dict[str, Any] = field(default_factory=dict)  # keyed by question.id
    scores: dict[str, float] = field(default_factory=dict)  # keyed by question.id

    @property
    def current_question(self) -> Question | None:
        if 0 <= self.current_index < len(self.questions):
            return self.questions[self.current_index]
        return None

    def add_question(self, question: Question) -> None:
        self.questions.append(question)

    def record_answer(self, question_id: str, answer: Any, *, score: float | None = None) -> None:
        self.user_answers[question_id] = answer
        if score is not None:
            self.scores[question_id] = float(score)

    def export_test(self) -> list[dict[str, Any]]:
        """Export questions without answers (safe to share as a test)."""

        return [q.to_public_dict() for q in self.questions]

    def export_answer_key(self) -> list[dict[str, Any]]:
        """Export answer key only (keep separate)."""

        return [q.to_answer_key_dict() for q in self.questions]

    def to_dict(self, *, include_answer_key: bool = False) -> dict[str, Any]:
        return {
            "questions": [q.to_dict(include_answer_key=include_answer_key) for q in self.questions],
            "current_index": self.current_index,
            "user_answers": _jsonable(self.user_answers),
            "scores": _jsonable(self.scores),
        }


EXAMPLE_QUESTION_DICT: dict[str, Any] = {
    "id": "q_example_001",
    "type": "games:nash-2x2",
    "chapter": "games",
    "prompt_text": (
        "Se dă matricea de plăți 2x2 de mai jos. Identificați dacă există un Echilibru Nash pur și "
        "specificați coordonatele (ex: L1-C1)."
    ),
    "data": [["(3, 4)", "(1, 2)"], ["(5, 0)", "(2, 6)"]],
    "correct_answer": ["L2-C1"],
    "correct_explanation": "Există echilibru Nash la L2-C1.",
    "metadata": {"ui_label": "Jocuri (Echilibru Nash)"},
}


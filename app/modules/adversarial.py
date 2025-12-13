"""Adversarial search module (placeholder).

Intended for minimax/alpha-beta style problems (games with two players).
"""

from __future__ import annotations

from .base_problem import BaseProblem, ProblemInstance


class AdversarialProblem(BaseProblem):
    """Skeleton adversarial-search problem generator."""

    problem_type = "adversarial"

    def generate(self) -> ProblemInstance:
        raise NotImplementedError("Adversarial-search generators are not implemented yet.")

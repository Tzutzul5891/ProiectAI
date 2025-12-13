"""Constraint Satisfaction Problems (CSP) module (placeholder).

This is a stub to keep the project structure clear. Add concrete CSP generators
here (e.g., map coloring, Sudoku, scheduling) by subclassing `BaseProblem`.
"""

from __future__ import annotations

from .base_problem import BaseProblem, ProblemInstance


class CSPProblem(BaseProblem):
    """Skeleton CSP problem generator."""

    problem_type = "csp"

    def generate(self) -> ProblemInstance:
        raise NotImplementedError("CSP generators are not implemented yet.")

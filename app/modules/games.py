from __future__ import annotations

import numpy as np

from .base_problem import BaseProblem, ProblemInstance


class NashGame(BaseProblem):
    """2x2 normal-form game generator with pure Nash equilibrium check."""

    problem_type = "games:nash-2x2"

    def __init__(self):
        self.matrix = None
        self.solutions = []
        self.explanation = ""

    def generate(self) -> ProblemInstance:
        p1 = np.random.randint(1, 10, (2, 2))
        p2 = np.random.randint(1, 10, (2, 2))
        
        self.matrix = []
        for i in range(2):
            row = []
            for j in range(2):
                row.append(f"({p1[i][j]}, {p2[i][j]})")
            self.matrix.append(row)

        self.solutions = []
        for r in range(2):
            for c in range(2):
                val_p1 = p1[r][c]
                other_row_val = p1[1-r][c]
                best_p1 = val_p1 >= other_row_val

                val_p2 = p2[r][c]
                other_col_val = p2[r][1-c]
                best_p2 = val_p2 >= other_col_val

                if best_p1 and best_p2:
                    self.solutions.append(f"L{r+1}-C{c+1}")
                                                
        if self.solutions:
            self.explanation = f"Există echilibru Nash la {', '.join(self.solutions)}. În aceste puncte, niciun jucător nu vrea să schimbe strategia unilateral."
        else:
            self.explanation = "Nu există echilibru Nash pur în această configurație."
        
        prompt = (
            "Se dă o matrice de plăți 2x2. Identificați dacă există un Echilibru Nash pur și "
            "specificați coordonatele (ex: L1-C1)."
        )
        return ProblemInstance(
            data=self.matrix,
            prompt=prompt,
            solution=list(self.solutions),
            explanation=self.explanation,
            metadata={},
        )

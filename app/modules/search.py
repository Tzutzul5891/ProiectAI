import numpy as np

class NQueensProblem:
    def __init__(self):
        self.solution_board = None
        self.explanation = ""

    def solve_n_queens(self, n):
        col = set()
        pos_diag = set() # (r + c)
        neg_diag = set() # (r - c)
        board = [['.' for _ in range(n)] for _ in range(n)]
        result = []

        def backtrack(r):
            if r == n:
                copy = ["".join(row) for row in board]
                result.append(copy)
                return True

            for c in range(n):
                if c in col or (r + c) in pos_diag or (r - c) in neg_diag:
                    continue

                col.add(c)
                pos_diag.add(r + c)
                neg_diag.add(r - c)
                board[r][c] = 'Q'

                if backtrack(r + 1):
                    return True

                col.remove(c)
                pos_diag.remove(r + c)
                neg_diag.remove(r - c)
                board[r][c] = '.'
            return False

        backtrack(0)
        return result

    def generate_problem(self):
        self.n = 4
         
        solutions = self.solve_n_queens(self.n)
        
        if solutions:
            sol_matrix = solutions[0]
            positions = []
            for r in range(self.n):
                for c in range(self.n):
                    if sol_matrix[r][c] == 'Q':
                        positions.append(f"Rând {r+1}, Col {c+1}")
            
            self.explanation = f"O soluție validă pentru {self.n} regine este plasarea lor la coordonatele: {'; '.join(positions)}. Această configurare asigură că nicio regină nu se atacă reciproc pe linii, coloane sau diagonale."
            
            empty_board = [[" " for _ in range(self.n)] for _ in range(self.n)]
            return empty_board, self.explanation
        else:
            return [], "Eroare la generare."
from __future__ import annotations

import random

from .base_problem import BaseProblem, ProblemInstance


class KnightsTourProblem(BaseProblem):
    """Knight's Tour generator (small boards) using backtracking."""

    problem_type = "search:knights-tour"

    def __init__(self):
        self.solution_path = None
        self.explanation = ""
        self.n = None
        self.start_pos = None
    
    def is_valid_move(self, x, y, board, n):
        """Check if the knight can move to position (x, y)"""
        return 0 <= x < n and 0 <= y < n and board[x][y] == -1
    
    def solve_knights_tour(self, n, start_x=0, start_y=0):
        """Solve Knight's Tour using backtracking"""
        board = [[-1 for _ in range(n)] for _ in range(n)]
        
        # Knight's possible moves
        move_x = [2, 1, -1, -2, -2, -1, 1, 2]
        move_y = [1, 2, 2, 1, -1, -2, -2, -1]
        
        # Start position
        board[start_x][start_y] = 0
        pos = 1
        
        def backtrack(x, y, pos):
            # If all squares are visited
            if pos == n * n:
                return True
            
            # Try all 8 moves
            for i in range(8):
                next_x = x + move_x[i]
                next_y = y + move_y[i]
                
                if self.is_valid_move(next_x, next_y, board, n):
                    board[next_x][next_y] = pos
                    if backtrack(next_x, next_y, pos + 1):
                        return True
                    # Backtrack
                    board[next_x][next_y] = -1
            
            return False
        
        if backtrack(start_x, start_y, pos):
            return board
        return None
    
    def generate(self) -> ProblemInstance:
        """Generate a Knight's Tour problem."""
        # Use board size 5 or 6 (larger boards take too long)
        self.n = random.choice([5, 6])
        
        # Random starting position
        start_x = random.randint(0, self.n - 1)
        start_y = random.randint(0, self.n - 1)
        self.start_pos = (start_x, start_y)
        
        # Try to find a solution
        solution = self.solve_knights_tour(self.n, start_x, start_y)
        
        if solution:
            # Create explanation with the path
            path_description = f"Călătoria calului începe de la poziția Rând {start_x+1}, Col {start_y+1} și vizitează toate cele {self.n*self.n} căsuțe ale tablei de {self.n}x{self.n}. "
            path_description += "Calul se mișcă în forma literei 'L' (2 căsuțe într-o direcție și 1 căsuță perpendicular). "
            path_description += f"Secvența completă de mișcări formează un tur valid al calului."
            
            self.explanation = path_description
            self.solution_path = solution
            
            # Return empty board for user to fill
            empty_board = [[" " for _ in range(self.n)] for _ in range(self.n)]
            # Mark starting position
            empty_board[start_x][start_y] = "K"
            
            prompt = (
                f"Pe tabla de {self.n}x{self.n} de mai jos, creați un tur al calului care vizitează "
                "fiecare căsuță exact o singură dată. Calul se mișcă în formă de 'L'."
            )
            return ProblemInstance(
                data=empty_board,
                prompt=prompt,
                solution=solution,
                explanation=self.explanation,
                metadata={"n": self.n, "start_pos": self.start_pos},
            )
        else:
            # Fallback to smaller board or different start
            self.n = 5
            self.start_pos = (0, 0)
            solution = self.solve_knights_tour(5, 0, 0)
            if solution:
                self.explanation = f"Călătoria calului începe de la colțul tablei (Rând 1, Col 1) și vizitează toate cele 25 de căsuțe."
                self.solution_path = solution
                empty_board = [[" " for _ in range(5)] for _ in range(5)]
                empty_board[0][0] = "K"
                prompt = (
                    "Pe tabla de 5x5 de mai jos, creați un tur al calului care vizitează fiecare "
                    "căsuță exact o singură dată. Calul se mișcă în formă de 'L'."
                )
                return ProblemInstance(
                    data=empty_board,
                    prompt=prompt,
                    solution=solution,
                    explanation=self.explanation,
                    metadata={"n": self.n, "start_pos": self.start_pos},
                )
            
            return ProblemInstance(
                data=[],
                prompt="",
                solution=None,
                explanation="Eroare la generare.",
                metadata={},
            )


class TowerOfHanoiProblem(BaseProblem):
    """Tower of Hanoi generator (3 or 4 pegs) with optimal solution."""

    problem_type = "search:hanoi"

    def __init__(self):
        self.solution_moves = []
        self.explanation = ""
        self.num_disks = None
        self.num_pegs = None
        self.initial_state = None
        self.target_state = None
    
    def solve_hanoi_3pegs(self, n, source, target, auxiliary, moves):
        """Classic 3-peg Tower of Hanoi"""
        if n == 1:
            moves.append((source, target))
            return
        
        self.solve_hanoi_3pegs(n - 1, source, auxiliary, target, moves)
        moves.append((source, target))
        self.solve_hanoi_3pegs(n - 1, auxiliary, target, source, moves)
    
    def solve_hanoi_4pegs(self, n, source, target, aux1, aux2, moves):
        """Frame-Stewart algorithm for 4-peg Tower of Hanoi"""
        if n == 0:
            return
        if n == 1:
            moves.append((source, target))
            return
        
        # Optimal k for Frame-Stewart algorithm
        k = n - round((2 * n + 1) ** 0.5) + 1
        k = max(1, min(k, n - 1))
        
        # Move k disks to aux1 using all 4 pegs
        self.solve_hanoi_4pegs(k, source, aux1, aux2, target, moves)
        
        # Move remaining n-k disks to target using 3 pegs
        self.solve_hanoi_3pegs(n - k, source, target, aux2, moves)
        
        # Move k disks from aux1 to target using all 4 pegs
        self.solve_hanoi_4pegs(k, aux1, target, aux2, source, moves)
    
    def generate(self) -> ProblemInstance:
        """Generate a Tower of Hanoi problem."""
        # Random configuration
        self.num_disks = random.randint(3, 5)
        self.num_pegs = random.choice([3, 4])
        
        # Initial state: all disks on peg 0, other pegs empty
        self.initial_state = {i: [] for i in range(self.num_pegs)}
        self.initial_state[0] = list(range(self.num_disks, 0, -1))
        
        # Target state: all disks on last peg
        self.target_state = {i: [] for i in range(self.num_pegs)}
        self.target_state[self.num_pegs - 1] = list(range(self.num_disks, 0, -1))
        
        # Solve
        self.solution_moves = []
        if self.num_pegs == 3:
            self.solve_hanoi_3pegs(self.num_disks, 0, 2, 1, self.solution_moves)
            peg_names = ["A", "B", "C"]
        else:
            self.solve_hanoi_4pegs(self.num_disks, 0, 3, 1, 2, self.solution_moves)
            peg_names = ["A", "B", "C", "D"]
        
        # Create explanation
        move_descriptions = [f"{peg_names[src]} → {peg_names[dst]}" for src, dst in self.solution_moves[:10]]
        
        self.explanation = f"Problema Turnurilor din Hanoi cu {self.num_disks} discuri și {self.num_pegs} tije. "
        self.explanation += f"Trebuie să muți toate discurile de pe tija {peg_names[0]} pe tija {peg_names[-1]}. "
        self.explanation += "Reguli: (1) Doar un disc poate fi mutat odată. (2) Un disc poate fi plasat doar peste un disc mai mare. "
        self.explanation += f"Soluția optimă necesită {len(self.solution_moves)} mișcări. "
        self.explanation += f"Primele mișcări: {', '.join(move_descriptions[:5])}"
        if len(self.solution_moves) > 5:
            self.explanation += "..."
        
        # Return initial state as "matrix" for visualization
        empty_state = [[" " for _ in range(self.num_pegs)] for _ in range(self.num_disks)]
        peg_names = ["A", "B", "C", "D"][: self.num_pegs]
        prompt = (
            f"Turnurile din Hanoi: Mutați toate cele {self.num_disks} discuri de pe tija {peg_names[0]} "
            f"pe tija {peg_names[-1]}, respectând regulile (un disc mai mare nu poate fi plasat peste "
            "unul mai mic)."
        )
        return ProblemInstance(
            data=empty_state,
            prompt=prompt,
            solution=list(self.solution_moves),
            explanation=self.explanation,
            metadata={
                "num_disks": self.num_disks,
                "num_pegs": self.num_pegs,
                "initial_state": self.initial_state,
                "target_state": self.target_state,
            },
        )


class NQueensProblem(BaseProblem):
    """N-Queens generator with a single valid solution as gold standard."""

    problem_type = "search:n-queens"

    def __init__(self):
        self.solution_board = None
        self.explanation = ""
        self.n = None
        self.expected_queens = None

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

    def generate(self) -> ProblemInstance:
        # Randomly choose board size between 4 and 8
        self.n = random.randint(4, 8)
        self.expected_queens = self.n
         
        solutions = self.solve_n_queens(self.n)
        
        if solutions:
            sol_matrix = solutions[0]
            self.solution_board = sol_matrix
            positions = []
            for r in range(self.n):
                for c in range(self.n):
                    if sol_matrix[r][c] == 'Q':
                        positions.append(f"Rând {r+1}, Col {c+1}")
            
            self.explanation = f"O soluție validă pentru {self.n} regine este plasarea lor la coordonatele: {'; '.join(positions)}. Această configurare asigură că nicio regină nu se atacă reciproc pe linii, coloane sau diagonale."
            
            empty_board = [[" " for _ in range(self.n)] for _ in range(self.n)]
            prompt = (
                f"Pe tabla de {self.n}x{self.n} de mai jos, propuneți o configurare pentru regine astfel "
                "încât să nu se atace reciproc (pe linii, coloane sau diagonale)."
            )
            return ProblemInstance(
                data=empty_board,
                prompt=prompt,
                solution=sol_matrix,
                explanation=self.explanation,
                metadata={"n": self.n, "expected_queens": self.expected_queens},
            )
        else:
            self.solution_board = None
            return ProblemInstance(
                data=[],
                prompt="",
                solution=None,
                explanation="Eroare la generare.",
                metadata={},
            )

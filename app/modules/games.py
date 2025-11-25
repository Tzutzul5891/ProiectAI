import numpy as np

class NashGame:
    def __init__(self):
        self.matrix = None
        self.solutions = []
        self.explanation = ""

    def generate_problem(self):
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
        
        return self.matrix, self.explanation
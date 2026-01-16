from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .base_problem import BaseProblem, ProblemInstance


@dataclass(frozen=True)
class StrategyOption:
    id: str
    label: str


STRATEGY_OPTIONS: list[StrategyOption] = [
    StrategyOption(id="bt_simple", label="Backtracking (simplu / DFS)"),
    StrategyOption(
        id="bt_heuristics",
        label="Backtracking + heuristici (MRV/LCV) + Forward Checking",
    ),
    StrategyOption(id="csp_ac3", label="Constraint Propagation (AC-3) + Backtracking"),
    StrategyOption(id="warnsdorff", label="Heuristic Search / Warnsdorff (Turul Calului)"),
    StrategyOption(id="hanoi_rec_3", label="Recursie clasică Hanoi (3 tije)"),
    StrategyOption(id="hanoi_fs_k", label="Frame–Stewart (Hanoi generalizat, k tije)"),
]


def _labels(options: list[StrategyOption] | None = None) -> list[str]:
    return [opt.label for opt in (options or STRATEGY_OPTIONS)]


def _label_by_id(strategy_id: str) -> str:
    for opt in STRATEGY_OPTIONS:
        if opt.id == strategy_id:
            return opt.label
    raise KeyError(f"Unknown strategy id: {strategy_id}")


PROBLEM_KEYS: tuple[str, ...] = ("n-queens", "generalised-hanoi", "graph-coloring", "knights-tour")


class StrategyChoiceProblem(BaseProblem):
    """Theory-style question: pick the most suitable strategy + short justification.

    Constraints:
      - deterministic / no LLM
      - exact match on strategy label + heuristic partial credit
    """

    problem_type = "theory:strategy-choice"

    def __init__(self, *, allowed_problems: list[str] | None = None):
        self.allowed_problems = list(allowed_problems) if allowed_problems else list(PROBLEM_KEYS)

    @staticmethod
    def generate_examples(*, seed: int = 7) -> list[ProblemInstance]:
        """Return one example instance per supported problem (useful for docs/debug)."""

        rng = random.Random(seed)
        gen = StrategyChoiceProblem(allowed_problems=list(PROBLEM_KEYS))
        return [gen._build_instance(problem_key, rng) for problem_key in PROBLEM_KEYS]

    def generate(self) -> ProblemInstance:
        rng = random
        problem_key = rng.choice(self.allowed_problems or list(PROBLEM_KEYS))
        return self._build_instance(problem_key, rng)

    def _build_instance(self, problem_key: str, rng: random.Random) -> ProblemInstance:
        allowed = _labels()

        if problem_key == "n-queens":
            n = int(rng.choice([8, 10, 12]))
            correct = _label_by_id("bt_heuristics")
            partial_scores = {
                _label_by_id("bt_simple"): 70.0,
                _label_by_id("csp_ac3"): 85.0,
            }
            keywords = ["backtracking", "mrv", "lcv", "forward", "pruning", "csp", "constr"]
            reasons = [
                "Spațiul de căutare este combinatorial, iar backtracking-ul permite explorare sistematică cu pruning.",
                "Heuristici precum MRV/LCV reduc branching-ul; Forward Checking taie devreme ramurile imposibile.",
                "Modelarea ca CSP (variabile=rânduri, domenii=coloane) se potrivește natural cu aceste tehnici.",
            ]
            instance_rows = [
                ["Problemă", "N-Queens"],
                ["Instanță", f"n={n}"],
                ["Scop", "o configurație validă (nu optimizare)"],
            ]
            prompt_instance = (
                f"Problemă: N-Queens • Instanță: n={n} (plasează n regine pe o tablă n×n, fără atacuri)."
            )

        elif problem_key == "generalised-hanoi":
            num_pegs = 4
            num_disks = int(rng.choice([6, 7, 8]))
            correct = _label_by_id("hanoi_fs_k")
            partial_scores = {
                _label_by_id("hanoi_rec_3"): 70.0,
            }
            keywords = ["frame", "stewart", "recurs", "k", "tije", "minim", "optimal"]
            reasons = [
                "Este o problemă recursivă; varianta generalizată (k tije) cere alegerea unui k intermediar optim.",
                "Frame–Stewart este strategia standard din curs pentru a minimiza numărul de mutări la k>3 tije.",
                "Recursia clasică pe 3 tije funcționează, dar nu exploatează tija extra și nu e optimă pentru k=4.",
            ]
            instance_rows = [
                ["Problemă", "Hanoi generalizat"],
                ["Instanță", f"{num_disks} discuri, {num_pegs} tije"],
                ["Scop", "minimizarea numărului de mutări"],
            ]
            prompt_instance = (
                f"Problemă: Hanoi generalizat • Instanță: {num_disks} discuri, {num_pegs} tije (k>3)."
            )

        elif problem_key == "graph-coloring":
            n = int(rng.choice([10, 12, 14]))
            k = int(rng.choice([3, 4]))
            correct = _label_by_id("csp_ac3")
            partial_scores = {
                _label_by_id("bt_heuristics"): 85.0,
                _label_by_id("bt_simple"): 60.0,
            }
            keywords = ["csp", "ac-3", "ac3", "propag", "domen", "consisten", "mrv", "forward"]
            reasons = [
                "Colorarea unui graf este un CSP clasic (variabile=noduri, domenii=culori, constrângeri=adiacență).",
                "AC-3/propagarea constrângerilor reduce domeniile înainte și în timpul backtracking-ului (pruning).",
                "În combinație cu MRV/Forward Checking, se detectează rapid conflictele și se reduce branching-ul.",
            ]
            instance_rows = [
                ["Problemă", "Graph Coloring (k-coloring)"],
                ["Instanță", f"n={n}, k={k}"],
                ["Scop", "găsește o colorare validă (fără conflicte)"],
            ]
            prompt_instance = f"Problemă: Graph Coloring • Instanță: n={n}, k={k} (CSP cu constrângeri de adiacență)."

        elif problem_key == "knights-tour":
            board_size = 8
            start_r = int(rng.randint(1, board_size))
            start_c = int(rng.randint(1, board_size))
            correct = _label_by_id("warnsdorff")
            partial_scores = {
                _label_by_id("bt_simple"): 60.0,
            }
            keywords = ["warnsdorff", "heuristic", "greedy", "degree", "dead", "onward"]
            reasons = [
                "Turul calului are foarte multe ramuri; backtracking-ul simplu intră des în dead-end-uri.",
                "Regula lui Warnsdorff alege mutarea cu cele mai puține continuări (min-degree), reducând blocajele.",
                "În practică, heuristica produce rapid tururi complete pe table standard (ex: 8×8).",
            ]
            instance_rows = [
                ["Problemă", "Knight’s Tour"],
                ["Instanță", f"tablă {board_size}×{board_size}, start=({start_r},{start_c})"],
                ["Scop", "tur complet (vizitează fiecare celulă o singură dată)"],
            ]
            prompt_instance = (
                f"Problemă: Knight’s Tour • Instanță: tablă {board_size}×{board_size}, start=({start_r},{start_c})."
            )

        else:
            raise ValueError(f"Unknown problem_key: {problem_key}")

        strategies_text = "\n".join(f"- {label}" for label in allowed)
        prompt = (
            "Cerința 1 — Alegere strategie.\n\n"
            f"{prompt_instance}\n\n"
            "Alege strategia cea mai potrivită din lista de mai jos și justifică în 2–3 propoziții.\n\n"
            "Strategii permise:\n"
            f"{strategies_text}"
        )

        explanation = "Strategia corectă: " + correct + ".\n" + "\n".join(f"- {r}" for r in reasons)

        metadata: dict[str, Any] = {
            "strategy_choice": {
                "problem_key": problem_key,
                "allowed_strategies": allowed,
                "partial_scores": partial_scores,
                "keywords": keywords,
            }
        }

        return ProblemInstance(
            data=instance_rows,
            prompt=prompt,
            solution=correct,
            explanation=explanation,
            metadata=metadata,
        )


from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

from .base_problem import BaseProblem, ProblemInstance


DEFAULT_COLOR_NAMES: list[str] = ["R", "G", "B", "Y", "O", "P", "C", "M", "K", "W"]


def _build_adjacency(n: int, edges: list[tuple[int, int]]) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {i: set() for i in range(1, n + 1)}
    for u, v in edges:
        if u == v:
            continue
        if not (1 <= u <= n and 1 <= v <= n):
            continue
        adjacency[u].add(v)
        adjacency[v].add(u)
    return adjacency


def _dedupe_edges(edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    deduped: list[tuple[int, int]] = []
    for u, v in edges:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        deduped.append((a, b))
    return deduped


def generate_random_graph(
    n: int,
    density: float,
    *,
    rng: random.Random | None = None,
) -> list[tuple[int, int]]:
    """Generate an undirected simple graph as an edge list (1-indexed nodes)."""

    rng = rng or random
    edges: list[tuple[int, int]] = []
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            if rng.random() < density:
                edges.append((u, v))

    # Ensure the instance isn't completely empty (too trivial).
    if not edges and n >= 2:
        u = rng.randint(1, n)
        v = rng.randint(1, n - 1)
        if v >= u:
            v += 1
        a, b = (u, v) if u < v else (v, u)
        edges.append((a, b))

    return _dedupe_edges(edges)


def edges_to_adjacency_matrix(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for u, v in _dedupe_edges(edges):
        if 1 <= u <= n and 1 <= v <= n and u != v:
            matrix[u - 1][v - 1] = 1
            matrix[v - 1][u - 1] = 1
    return matrix


def solve_graph_coloring(
    n: int,
    edges: list[tuple[int, int]],
    k: int,
) -> dict[int, int] | None:
    """Backtracking + MRV + forward checking k-coloring solver.

    Returns:
        dict[node -> color_index] with node in 1..n and color_index in 0..k-1,
        or None if no solution exists.
    """

    if n <= 0 or k <= 0:
        return None

    edges = _dedupe_edges(edges)
    adjacency = _build_adjacency(n, edges)

    domains: dict[int, set[int]] = {i: set(range(k)) for i in range(1, n + 1)}
    assignment: dict[int, int] = {}

    def select_unassigned_variable() -> int:
        candidates = [v for v in range(1, n + 1) if v not in assignment]
        # MRV: smallest domain first; tie-breaker: highest degree
        return min(candidates, key=lambda v: (len(domains[v]), -len(adjacency[v]), v))

    def is_consistent(var: int, color: int) -> bool:
        return all(assignment.get(nei) != color for nei in adjacency[var])

    def order_colors(var: int) -> list[int]:
        # LCV-ish: prefer colors that constrain neighbors the least.
        def impact(color: int) -> int:
            return sum(1 for nei in adjacency[var] if nei not in assignment and color in domains[nei])

        return sorted(domains[var], key=lambda c: (impact(c), c))

    def forward_check(var: int, color: int) -> tuple[bool, list[tuple[int, int]]]:
        removed: list[tuple[int, int]] = []
        for nei in adjacency[var]:
            if nei in assignment:
                continue
            if color in domains[nei]:
                domains[nei].remove(color)
                removed.append((nei, color))
                if not domains[nei]:
                    return False, removed
        return True, removed

    def restore(removed: list[tuple[int, int]]) -> None:
        for node, color in removed:
            domains[node].add(color)

    def backtrack() -> bool:
        if len(assignment) == n:
            return True

        var = select_unassigned_variable()
        for color in order_colors(var):
            if not is_consistent(var, color):
                continue
            assignment[var] = color
            ok, removed = forward_check(var, color)
            if ok and backtrack():
                return True
            del assignment[var]
            restore(removed)
        return False

    if backtrack():
        return dict(assignment)
    return None


def format_coloring(assignment: dict[int, int], color_names: list[str]) -> dict[int, str]:
    names = color_names or DEFAULT_COLOR_NAMES
    formatted: dict[int, str] = {}
    for node, color_idx in assignment.items():
        if 0 <= int(color_idx) < len(names):
            formatted[int(node)] = str(names[int(color_idx)])
        else:
            formatted[int(node)] = f"C{int(color_idx) + 1}"
    return formatted


_PAIR_RE = re.compile(r"^\s*(\d+)\s*[:=\\-]\s*([A-Za-z0-9]+)\s*$")


@dataclass(frozen=True)
class ParsedColoring:
    assignment: dict[int, str]
    errors: list[str]


def parse_coloring_text(text: str, *, n: int, color_names: list[str]) -> ParsedColoring:
    """Parse text answers like: `1:R, 2:G, 3:B` or `1:1, 2:2`."""

    allowed = [c.strip() for c in (color_names or []) if str(c).strip()]
    allowed_upper = {c.upper(): c for c in allowed}
    k = len(allowed)

    assignment: dict[int, str] = {}
    errors: list[str] = []

    if not text or not str(text).strip():
        return ParsedColoring(assignment={}, errors=[])

    chunks = re.split(r"[,\n;]+", str(text))
    for raw in chunks:
        part = raw.strip()
        if not part:
            continue
        match = _PAIR_RE.match(part)
        if not match:
            errors.append(f"Format invalid: '{part}'. Folosește 'nod:culoare' (ex: 1:R).")
            continue

        node = int(match.group(1))
        if not (1 <= node <= n):
            errors.append(f"Nod invalid: {node} (așteptat 1..{n}).")
            continue

        token = match.group(2).strip()
        if not token:
            errors.append(f"Culoare lipsă pentru nodul {node}.")
            continue

        token_upper = token.upper()
        normalized: str | None = None
        if token.isdigit():
            idx = int(token)
            if 1 <= idx <= k:
                normalized = allowed[idx - 1]
        if normalized is None and token_upper in allowed_upper:
            normalized = allowed_upper[token_upper]

        if normalized is None:
            errors.append(
                f"Culoare invalidă pentru nodul {node}: '{token}'. Culori permise: {', '.join(allowed) or '—'}."
            )
            continue

        assignment[node] = normalized

    return ParsedColoring(assignment=assignment, errors=errors)


def evaluate_graph_coloring(
    assignment: dict[int, str] | None,
    *,
    n: int,
    edges: list[tuple[int, int]],
    color_names: list[str],
) -> tuple[float, str, dict[str, Any]]:
    """Score a (possibly partial) coloring attempt.

    Scoring:
      - completeness: fraction of colored nodes
      - conflicts: fraction of satisfied edges among colored endpoints
      - color limit: must use <= k colors (penalized if exceeded)
    """

    edges = _dedupe_edges(edges)
    k = len(color_names or [])
    allowed_upper = {c.upper() for c in (color_names or [])}

    raw = assignment or {}
    filtered: dict[int, str] = {}
    invalid_nodes: list[int] = []
    invalid_colors: list[tuple[int, str]] = []

    for node, color in raw.items():
        try:
            node_i = int(node)
        except Exception:
            continue
        if not (1 <= node_i <= n):
            invalid_nodes.append(node_i)
            continue
        color_s = str(color).strip()
        if not color_s:
            continue
        if allowed_upper and color_s.upper() not in allowed_upper:
            invalid_colors.append((node_i, color_s))
            continue
        filtered[node_i] = color_s.upper() if allowed_upper else color_s

    colored_nodes = sorted(filtered.keys())
    colored_count = len(colored_nodes)
    if colored_count == 0:
        return 0.0, "Necompletat.", {"n": n, "k": k, "colored": 0, "missing": list(range(1, n + 1))}

    missing_nodes = [i for i in range(1, n + 1) if i not in filtered]
    completeness = colored_count / max(1, n)

    considered_edges = 0
    conflicts: list[tuple[int, int]] = []
    for u, v in edges:
        cu = filtered.get(u)
        cv = filtered.get(v)
        if cu is None or cv is None:
            continue
        considered_edges += 1
        if cu == cv:
            conflicts.append((u, v))

    if considered_edges == 0:
        edge_quality = 1.0
    else:
        edge_quality = (considered_edges - len(conflicts)) / considered_edges

    colors_used = sorted({filtered[i] for i in colored_nodes})
    extra_colors = max(0, len(colors_used) - k) if k > 0 else 0
    color_penalty = 1.0
    if k > 0 and len(colors_used) > k:
        color_penalty = k / max(1, len(colors_used))

    score = float(100.0 * completeness * edge_quality * color_penalty)
    score = max(0.0, min(100.0, score))

    is_valid = (colored_count == n) and (len(conflicts) == 0) and (extra_colors == 0) and not invalid_colors

    if is_valid:
        message = f"✅ Colorare validă: {colored_count}/{n} noduri, 0 conflicte, {len(colors_used)}/{k} culori."
    else:
        message_parts = [
            f"Noduri colorate: {colored_count}/{n}",
            f"Conflicte: {len(conflicts)}" + (f"/{considered_edges}" if considered_edges else ""),
        ]
        if k > 0:
            message_parts.append(f"Culori folosite: {len(colors_used)}/{k}")
        if invalid_colors:
            message_parts.append(f"Culori invalide: {len(invalid_colors)}")
        message = " • ".join(message_parts)

    details: dict[str, Any] = {
        "n": n,
        "k": k,
        "colored": colored_count,
        "missing": missing_nodes,
        "colors_used": colors_used,
        "conflicts": conflicts,
        "considered_edges": considered_edges,
        "invalid_nodes": invalid_nodes,
        "invalid_colors": invalid_colors,
        "score_components": {
            "completeness": completeness,
            "edge_quality": edge_quality,
            "color_penalty": color_penalty,
        },
    }
    return score, message, details


class GraphColoringProblem(BaseProblem):
    """Graph coloring (k-coloring) CSP generator."""

    problem_type = "csp:graph-coloring"

    def __init__(
        self,
        *,
        n_range: tuple[int, int] = (6, 10),
        density_range: tuple[float, float] = (0.25, 0.55),
        k_choices: tuple[int, ...] = (3, 4),
        max_attempts: int = 60,
    ) -> None:
        self.n_range = n_range
        self.density_range = density_range
        self.k_choices = k_choices
        self.max_attempts = max_attempts

    def generate(self) -> ProblemInstance:
        rng = random
        last_error: str | None = None

        for _ in range(max(1, int(self.max_attempts))):
            n = rng.randint(int(self.n_range[0]), int(self.n_range[1]))
            n = max(2, min(n, 12))
            density = rng.uniform(float(self.density_range[0]), float(self.density_range[1]))
            density = max(0.0, min(1.0, density))

            edges = generate_random_graph(n, density, rng=rng)

            k = min(max(2, int(rng.choice(self.k_choices))), len(DEFAULT_COLOR_NAMES), n)
            solution = solve_graph_coloring(n, edges, k)
            if solution is None and k < min(4, n):
                k2 = min(4, n, len(DEFAULT_COLOR_NAMES))
                solution = solve_graph_coloring(n, edges, k2)
                if solution is not None:
                    k = k2

            if solution is None:
                last_error = "unsatisfiable instance"
                continue

            color_names = DEFAULT_COLOR_NAMES[:k]
            formatted_solution = format_coloring(solution, color_names)
            matrix = edges_to_adjacency_matrix(n, edges)

            prompt = (
                "Se dă un graf neorientat (noduri 1..n) reprezentat prin matricea de adiacență de mai jos. "
                f"Colorați nodurile folosind cel mult k={k} culori astfel încât două noduri adiacente "
                "să nu aibă aceeași culoare. "
                f"Culori permise: {', '.join(color_names)}."
            )
            explanation = "O colorare validă este: " + ", ".join(
                f"{node}:{formatted_solution[node]}" for node in sorted(formatted_solution)
            )

            return ProblemInstance(
                data=matrix,
                prompt=prompt,
                solution=formatted_solution,
                explanation=explanation,
                metadata={
                    "n": n,
                    "k": k,
                    "edges": edges,
                    "color_names": color_names,
                    "density": density,
                },
            )

        return ProblemInstance(
            data=[],
            prompt="",
            solution=None,
            explanation=f"Eroare la generare: {last_error or 'unknown'}",
            metadata={},
        )


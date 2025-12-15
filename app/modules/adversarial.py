"""Adversarial search: Minimax + Alpha-Beta pruning (Cerința 4).

This module provides:
  - a small JSON schema for game trees (MAX/MIN internal nodes + leaf utilities)
  - a deterministic Alpha-Beta implementation that counts evaluated leaves
  - a problem generator used by the Streamlit UI

Tree JSON schema (minimal):

{
  "id": "demo",
  "title": "optional",
  "traversal": "left-to-right",
  "root": {
    "type": "MAX",
    "id": "R",
    "children": [
      {"type": "MIN", "children": [{"id":"L1","value":3}, {"id":"L2","value":5}]},
      {"type": "MIN", "children": [{"id":"L3","value":2}, {"id":"L4","value":9}]}
    ]
  }
}
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from itertools import count
from math import inf
from pathlib import Path
from typing import Any, Literal

from .base_problem import BaseProblem, ProblemInstance


Player = Literal["MAX", "MIN"]


@dataclass(frozen=True)
class TreeNode:
    """Immutable parsed tree node."""

    role: Player | None  # None => leaf
    node_id: str
    children: tuple["TreeNode", ...] = ()
    value: int | float | None = None  # only for leaves

    @property
    def is_leaf(self) -> bool:
        return self.role is None


@dataclass(frozen=True)
class AlphaBetaStats:
    value_at_root: int | float
    visited_leaves_count: int
    visited_leaves_order: tuple[str, ...]


def _default_trees_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "adversarial_trees"


def list_adversarial_trees(trees_dir: str | Path | None = None) -> list[Path]:
    base = Path(trees_dir) if trees_dir is not None else _default_trees_dir()
    if not base.exists():
        return []
    return sorted([p for p in base.glob("*.json") if p.is_file()], key=lambda p: p.name)


def load_adversarial_tree_instance(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Tree instance JSON must be an object.")
    raw.setdefault("id", p.stem)
    raw.setdefault("title", raw.get("id"))
    raw.setdefault("traversal", "left-to-right")
    if "root" not in raw:
        raise ValueError("Tree instance must contain a `root` node.")
    return raw


def _norm_role(role: Any) -> Player:
    r = str(role or "").strip().upper()
    if r not in {"MAX", "MIN"}:
        raise ValueError(f"Invalid node type: {role!r}. Expected 'MAX' or 'MIN'.")
    return r  # type: ignore[return-value]


def _parse_tree_node(
    raw: Any,
    *,
    internal_counter: count,
    leaf_counter: count,
    path: str = "root",
) -> TreeNode:
    if not isinstance(raw, dict):
        raise ValueError(f"Node at {path} must be an object.")

    has_children = "children" in raw
    has_value = "value" in raw

    node_id = str(raw.get("id") or raw.get("label") or "").strip()

    if has_value and has_children:
        raise ValueError(f"Node at {path} cannot have both `value` and `children`.")

    if has_value:
        if not node_id:
            node_id = f"L{next(leaf_counter)}"
        val = raw.get("value")
        if not isinstance(val, (int, float)):
            raise ValueError(f"Leaf `{node_id}` at {path} must have numeric `value`.")
        return TreeNode(role=None, node_id=node_id, children=(), value=val)

    # Internal node
    role = _norm_role(raw.get("type") or raw.get("role"))
    if not node_id:
        node_id = f"N{next(internal_counter)}"

    children_raw = raw.get("children")
    if not isinstance(children_raw, list) or not children_raw:
        raise ValueError(f"Internal node `{node_id}` at {path} must have non-empty `children` list.")

    children: list[TreeNode] = []
    for idx, child in enumerate(children_raw):
        children.append(
            _parse_tree_node(
                child,
                internal_counter=internal_counter,
                leaf_counter=leaf_counter,
                path=f"{path}.children[{idx}]",
            )
        )
    return TreeNode(role=role, node_id=node_id, children=tuple(children), value=None)


def parse_tree_instance(raw: dict[str, Any]) -> TreeNode:
    """Parse a raw instance dict into an immutable TreeNode tree.

    IDs are optional; missing ones are auto-filled deterministically.
    """

    internal_counter = count(1)
    leaf_counter = count(1)
    return _parse_tree_node(raw.get("root"), internal_counter=internal_counter, leaf_counter=leaf_counter)


def tree_to_text_lines(root: TreeNode) -> list[str]:
    """Render a tree as indented text (left-to-right order)."""

    lines: list[str] = []

    def walk(node: TreeNode, depth: int) -> None:
        indent = "  " * depth
        if node.is_leaf:
            lines.append(f"{indent}{node.node_id} = {node.value}")
            return
        lines.append(f"{indent}{node.role} {node.node_id}")
        for ch in node.children:
            walk(ch, depth + 1)

    walk(root, 0)
    return lines


def count_leaves(root: TreeNode) -> int:
    if root.is_leaf:
        return 1
    return sum(count_leaves(ch) for ch in root.children)


def alphabeta_stats(root: TreeNode) -> AlphaBetaStats:
    """Alpha-Beta pruning, deterministic child order, counts evaluated leaves."""

    visited_order: list[str] = []
    visited_count = 0

    def eval_node(node: TreeNode, alpha: float, beta: float) -> float:
        nonlocal visited_count

        if node.is_leaf:
            visited_count += 1
            visited_order.append(node.node_id)
            # value is guaranteed numeric by parser
            return float(node.value)  # type: ignore[arg-type]

        if node.role == "MAX":
            value = -inf
            for ch in node.children:
                value = max(value, eval_node(ch, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        # MIN
        value = inf
        for ch in node.children:
            value = min(value, eval_node(ch, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    value = eval_node(root, -inf, inf)

    # Preserve int roots when possible (nicer UX: avoids "6.0").
    value_out: int | float
    if abs(value - int(value)) < 1e-9:
        value_out = int(value)
    else:
        value_out = float(value)

    return AlphaBetaStats(
        value_at_root=value_out,
        visited_leaves_count=int(visited_count),
        visited_leaves_order=tuple(visited_order),
    )


def alphabeta(root: TreeNode) -> tuple[int | float, int, list[str]]:
    """Convenience wrapper matching the course-style requirement."""

    stats = alphabeta_stats(root)
    return stats.value_at_root, stats.visited_leaves_count, list(stats.visited_leaves_order)


def _random_tree_instance(
    *,
    depth: int,
    branching: int,
    value_min: int,
    value_max: int,
    root_role: Player = "MAX",
    rng: random.Random = random,
) -> dict[str, Any]:
    """Create a random full tree instance dict.

    `depth` = number of edges from root to leaf (depth>=1).
    """

    if depth < 1:
        raise ValueError("depth must be >= 1")
    if branching < 2:
        raise ValueError("branching must be >= 2")
    if value_min > value_max:
        raise ValueError("value_min must be <= value_max")

    leaf_counter = count(1)
    internal_counter = count(1)

    def other(role: Player) -> Player:
        return "MIN" if role == "MAX" else "MAX"

    def build(level: int, role: Player) -> dict[str, Any]:
        if level == depth:
            leaf_id = f"L{next(leaf_counter)}"
            return {"id": leaf_id, "value": int(rng.randint(value_min, value_max))}

        node_id = "R" if level == 0 else f"N{next(internal_counter)}"
        return {
            "type": role,
            "id": node_id,
            "children": [build(level + 1, other(role)) for _ in range(branching)],
        }

    instance_id = f"random_d{depth}_b{branching}"
    return {
        "id": instance_id,
        "title": f"Random (d={depth}, b={branching})",
        "description": "Arbore generat aleator (determinist la același seed).",
        "traversal": "left-to-right",
        "root": build(0, root_role),
    }


class AlphaBetaTreeProblem(BaseProblem):
    """Cerința 4: compute root minimax value + visited leaves with Alpha-Beta."""

    problem_type = "adversarial:alphabeta"

    def __init__(
        self,
        *,
        instance_path: str | Path | None = None,
        depth: int = 3,
        branching: int = 2,
        value_min: int = -9,
        value_max: int = 9,
    ) -> None:
        self.instance_path = Path(instance_path) if instance_path is not None else None
        self.depth = int(depth)
        self.branching = int(branching)
        self.value_min = int(value_min)
        self.value_max = int(value_max)

    def generate(self) -> ProblemInstance:
        if self.instance_path is not None:
            raw = load_adversarial_tree_instance(self.instance_path)
            source = {"mode": "predefined", "file": self.instance_path.name}
        else:
            raw = _random_tree_instance(
                depth=self.depth,
                branching=self.branching,
                value_min=self.value_min,
                value_max=self.value_max,
            )
            source = {"mode": "random", "depth": self.depth, "branching": self.branching}

        root = parse_tree_instance(raw)
        stats = alphabeta_stats(root)

        lines = tree_to_text_lines(root)
        rows = [[line] for line in lines]

        total_leaves = count_leaves(root)

        prompt = (
            "Cerința 4 — MinMax + Alpha-Beta.\n\n"
            "Se dă un arbore de joc cu noduri MAX/MIN și utilități în frunze.\n"
            "Parcurgere: stânga → dreapta (ordinea copiilor din arbore).\n\n"
            "Cerință: calculează (1) valoarea din rădăcină și (2) câte frunze au fost evaluate "
            "(vizitate efectiv) când rulezi Minimax cu tăieri Alpha-Beta."
        )

        solution = {
            "value_at_root": stats.value_at_root,
            "visited_leaves_count": stats.visited_leaves_count,
            "visited_leaves_order": list(stats.visited_leaves_order),
        }

        explanation = (
            f"Valoare la rădăcină: {stats.value_at_root}. "
            f"Frunze evaluate: {stats.visited_leaves_count}/{total_leaves}. "
            f"Ordine frunze evaluate: {', '.join(stats.visited_leaves_order) if stats.visited_leaves_order else '—'}."
        )

        metadata: dict[str, Any] = {
            "adversarial": {
                "tree_id": raw.get("id"),
                "tree_title": raw.get("title"),
                "traversal": raw.get("traversal"),
                "source": source,
                "total_leaves": total_leaves,
            },
            "tree_raw": raw,
        }

        return ProblemInstance(
            data=rows,
            prompt=prompt,
            solution=solution,
            explanation=explanation,
            metadata=metadata,
        )


# Backwards-compatible name (older code may import this).
AdversarialProblem = AlphaBetaTreeProblem

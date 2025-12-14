"""Generic Constraint Satisfaction Problem (CSP) support + solver.

Implements Cerința 3:
  - CSP model: variables, domains, constraints (+ partial assignment)
  - Solver: Backtracking with optional MRV / Forward Checking / AC-3
  - JSON-driven instances (see `app/data/csp_instances/*.json`)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .base_problem import BaseProblem, ProblemInstance


JsonValue = Any
BinaryPredicate = Callable[[Any, Any], bool]


@dataclass(frozen=True)
class CSPSolverOptions:
    mrv: bool = False
    forward_checking: bool = False
    ac3_preprocess: bool = False
    ac3_interleaved: bool = False


def _safe_add(x: Any, y: Any) -> Any | None:
    try:
        return x + y
    except Exception:
        return None


def _safe_abs_diff(x: Any, y: Any) -> Any | None:
    try:
        return abs(x - y)
    except Exception:
        return None


def _safe_lt(x: Any, y: Any) -> bool:
    try:
        return x < y
    except Exception:
        return False


def _safe_gt(x: Any, y: Any) -> bool:
    try:
        return x > y
    except Exception:
        return False


def _norm_method_token(token: str) -> str:
    t = str(token or "").strip().upper().replace("_", "").replace(" ", "")
    t = t.replace("AC-3", "AC3")
    return t


def solver_options_from_instance(raw: dict[str, Any]) -> CSPSolverOptions:
    """Read solver options from a JSON instance dict.

    Supported:
      - `method`: string or list (tokens: MRV, FC, AC-3/AC3, MAC)
      - `ac3_mode`: "preprocess" | "interleaved" | "both"
      - `ac3`: {"preprocess": bool, "interleaved": bool}
    """

    method_raw = raw.get("method")
    tokens: list[str] = []
    if isinstance(method_raw, str):
        for chunk in method_raw.replace("+", "/").replace(",", "/").split("/"):
            if chunk.strip():
                tokens.append(chunk.strip())
    elif isinstance(method_raw, list):
        tokens = [str(x).strip() for x in method_raw if str(x).strip()]

    tokens_norm = {_norm_method_token(t) for t in tokens}

    mrv = "MRV" in tokens_norm
    fc = "FC" in tokens_norm or "FORWARDCHECKING" in tokens_norm
    ac3_in_method = "AC3" in tokens_norm
    mac_in_method = "MAC" in tokens_norm or "MAINTAINARC" in tokens_norm

    ac3_pre = bool(ac3_in_method or mac_in_method)
    ac3_int = bool(mac_in_method)

    ac3_mode = raw.get("ac3_mode")
    if isinstance(ac3_mode, str) and ac3_mode.strip():
        mode = ac3_mode.strip().lower()
        if mode in {"pre", "preprocess", "pre-processing"}:
            ac3_pre, ac3_int = True, False
        elif mode in {"interleaved", "mac", "maintain"}:
            ac3_pre, ac3_int = False, True
        elif mode in {"both", "pre+mac", "all"}:
            ac3_pre, ac3_int = True, True

    ac3_obj = raw.get("ac3")
    if isinstance(ac3_obj, dict):
        if "preprocess" in ac3_obj:
            ac3_pre = bool(ac3_obj.get("preprocess"))
        if "interleaved" in ac3_obj:
            ac3_int = bool(ac3_obj.get("interleaved"))

    return CSPSolverOptions(mrv=mrv, forward_checking=fc, ac3_preprocess=ac3_pre, ac3_interleaved=ac3_int)


@dataclass(frozen=True)
class CSPConstraint:
    """A constraint as defined by the JSON instance."""

    type: str
    vars: tuple[str, ...]
    params: dict[str, Any]

    def to_text(self) -> str:
        t = self.type
        vs = ", ".join(self.vars)
        if t == "all_different":
            return f"all-different({vs})"
        if t == "not_equal":
            return f"{self.vars[0]} != {self.vars[1]}"
        if t == "equal":
            return f"{self.vars[0]} == {self.vars[1]}"
        if t == "less_than":
            return f"{self.vars[0]} < {self.vars[1]}"
        if t == "greater_than":
            return f"{self.vars[0]} > {self.vars[1]}"
        if t == "sum_equals":
            s = self.params.get("sum")
            return f"{self.vars[0]} + {self.vars[1]} == {s}"
        if t == "sum_not_equals":
            s = self.params.get("sum")
            return f"{self.vars[0]} + {self.vars[1]} != {s}"
        if t == "abs_diff_equals":
            d = self.params.get("diff")
            return f"|{self.vars[0]} - {self.vars[1]}| == {d}"
        if t == "abs_diff_not_equals":
            d = self.params.get("diff")
            return f"|{self.vars[0]} - {self.vars[1]}| != {d}"
        if t == "allowed_pairs":
            return f"allowed-pairs({self.vars[0]}, {self.vars[1]})"
        if t == "forbidden_pairs":
            return f"forbidden-pairs({self.vars[0]}, {self.vars[1]})"
        return f"{t}({vs})"

    def is_satisfied(self, assignment: dict[str, Any]) -> bool:
        t = self.type
        vs = self.vars

        if t == "all_different":
            seen: set[Any] = set()
            for v in vs:
                if v not in assignment:
                    continue
                val = assignment[v]
                if val in seen:
                    return False
                seen.add(val)
            return True

        if len(vs) == 1:
            if vs[0] not in assignment:
                return True
            val = assignment[vs[0]]
            if t == "in_set":
                allowed = set(self.params.get("set") or [])
                return val in allowed
            return True

        if len(vs) != 2:
            # Unsupported n-ary constraint (besides all_different) – treat as not violated until complete.
            return True

        a, b = vs
        if a not in assignment or b not in assignment:
            return True
        x = assignment[a]
        y = assignment[b]

        if t == "not_equal":
            return x != y
        if t == "equal":
            return x == y
        if t == "less_than":
            try:
                return x < y
            except Exception:
                return False
        if t == "greater_than":
            try:
                return x > y
            except Exception:
                return False
        if t == "sum_equals":
            try:
                return (x + y) == self.params.get("sum")
            except Exception:
                return False
        if t == "sum_not_equals":
            try:
                return (x + y) != self.params.get("sum")
            except Exception:
                return False
        if t == "abs_diff_equals":
            try:
                return abs(x - y) == self.params.get("diff")
            except Exception:
                return False
        if t == "abs_diff_not_equals":
            try:
                return abs(x - y) != self.params.get("diff")
            except Exception:
                return False
        if t == "allowed_pairs":
            allowed = set(tuple(pair) for pair in (self.params.get("allowed") or []))
            return (x, y) in allowed
        if t == "forbidden_pairs":
            forbidden = set(tuple(pair) for pair in (self.params.get("forbidden") or []))
            return (x, y) not in forbidden

        # Unknown type: don't reject.
        return True

    def as_binary_relations(self) -> list[tuple[str, str, BinaryPredicate]]:
        """Return a list of directed binary relations induced by this constraint.

        AC-3 operates on binary relations, so `all_different` is decomposed into
        pairwise `!=` constraints.
        """

        t = self.type
        vs = self.vars

        if t == "all_different":
            rels: list[tuple[str, str, BinaryPredicate]] = []
            for i in range(len(vs)):
                for j in range(len(vs)):
                    if i == j:
                        continue
                    a = vs[i]
                    b = vs[j]

                    def _neq(x: Any, y: Any) -> bool:
                        return x != y

                    rels.append((a, b, _neq))
            return rels

        if len(vs) != 2:
            return []

        a, b = vs

        if t == "not_equal":
            return [
                (a, b, lambda x, y: x != y),
                (b, a, lambda y, x: y != x),
            ]
        if t == "equal":
            return [
                (a, b, lambda x, y: x == y),
                (b, a, lambda y, x: y == x),
            ]
        if t == "less_than":
            return [
                (a, b, lambda x, y: _safe_lt(x, y)),
                (b, a, lambda y, x: _safe_gt(y, x)),
            ]
        if t == "greater_than":
            return [
                (a, b, lambda x, y: _safe_gt(x, y)),
                (b, a, lambda y, x: _safe_lt(y, x)),
            ]
        if t == "sum_equals":
            s = self.params.get("sum")
            return [
                (a, b, lambda x, y, _s=s: _safe_add(x, y) == _s),
                (b, a, lambda y, x, _s=s: _safe_add(y, x) == _s),
            ]
        if t == "sum_not_equals":
            s = self.params.get("sum")
            return [
                (a, b, lambda x, y, _s=s: _safe_add(x, y) is not None and _safe_add(x, y) != _s),
                (b, a, lambda y, x, _s=s: _safe_add(y, x) is not None and _safe_add(y, x) != _s),
            ]
        if t == "abs_diff_equals":
            d = self.params.get("diff")
            return [
                (a, b, lambda x, y, _d=d: _safe_abs_diff(x, y) == _d),
                (b, a, lambda y, x, _d=d: _safe_abs_diff(y, x) == _d),
            ]
        if t == "abs_diff_not_equals":
            d = self.params.get("diff")
            return [
                (a, b, lambda x, y, _d=d: _safe_abs_diff(x, y) is not None and _safe_abs_diff(x, y) != _d),
                (b, a, lambda y, x, _d=d: _safe_abs_diff(y, x) is not None and _safe_abs_diff(y, x) != _d),
            ]
        if t == "allowed_pairs":
            allowed = set(tuple(pair) for pair in (self.params.get("allowed") or []))
            return [
                (a, b, lambda x, y, _a=allowed: (x, y) in _a),
                (b, a, lambda y, x, _a=allowed: (x, y) in _a),
            ]
        if t == "forbidden_pairs":
            forbidden = set(tuple(pair) for pair in (self.params.get("forbidden") or []))
            return [
                (a, b, lambda x, y, _f=forbidden: (x, y) not in _f),
                (b, a, lambda y, x, _f=forbidden: (x, y) not in _f),
            ]

        return []


class CSP:
    def __init__(
        self,
        *,
        variables: list[str],
        domains: dict[str, list[JsonValue]],
        constraints: list[CSPConstraint],
    ) -> None:
        if not variables:
            raise ValueError("CSP must have at least one variable.")

        self.variables = [str(v) for v in variables]
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("CSP variables must be unique.")

        self.domains: dict[str, list[JsonValue]] = {}
        for v in self.variables:
            if v not in domains:
                raise ValueError(f"Missing domain for variable '{v}'.")
            vals = list(domains[v])
            if not vals:
                raise ValueError(f"Empty domain for variable '{v}'.")
            self.domains[v] = vals

        self.constraints = list(constraints)

        self.binary_relations: dict[tuple[str, str], list[BinaryPredicate]] = {}
        self.neighbors: dict[str, set[str]] = {v: set() for v in self.variables}
        for c in self.constraints:
            for a, b, pred in c.as_binary_relations():
                if a not in self.variables or b not in self.variables:
                    continue
                key = (a, b)
                self.binary_relations.setdefault(key, []).append(pred)
                self.neighbors[a].add(b)

        self._variable_index: dict[str, int] = {v: i for i, v in enumerate(self.variables)}

    def constraints_for(self, var: str) -> list[CSPConstraint]:
        return [c for c in self.constraints if var in c.vars]

    def is_assignment_consistent(self, assignment: dict[str, Any]) -> bool:
        return all(c.is_satisfied(assignment) for c in self.constraints)


def _copy_domains(domains: dict[str, list[Any]]) -> dict[str, list[Any]]:
    return {v: list(vals) for v, vals in domains.items()}


def _select_unassigned_variable(
    csp: CSP,
    assignment: dict[str, Any],
    domains: dict[str, list[Any]],
    *,
    use_mrv: bool,
) -> str:
    unassigned = [v for v in csp.variables if v not in assignment]
    if not unassigned:
        raise ValueError("No unassigned variables.")
    if not use_mrv:
        return unassigned[0]
    # MRV with stable tie-break by variable order.
    return min(unassigned, key=lambda v: (len(domains.get(v, [])), csp._variable_index.get(v, 10**9)))


def _all_binary_satisfied(
    relations: list[BinaryPredicate] | None,
    a_val: Any,
    b_val: Any,
) -> bool:
    if not relations:
        return True
    return all(pred(a_val, b_val) for pred in relations)


def ac3(
    csp: CSP,
    domains: dict[str, list[Any]],
    *,
    queue: list[tuple[str, str]] | None = None,
) -> bool:
    """AC-3 arc consistency on current domains.

    Args:
        csp: CSP definition (binary relations are derived from constraints).
        domains: current domains (mutated in place).
        queue: optional initial queue of arcs (Xi, Xj). When None, uses all arcs.
    """

    def revise(xi: str, xj: str) -> bool:
        rels = csp.binary_relations.get((xi, xj)) or []
        if not rels:
            return False
        removed_any = False
        new_dom: list[Any] = []
        dom_i = domains.get(xi, [])
        dom_j = domains.get(xj, [])
        for x in dom_i:
            supported = any(_all_binary_satisfied(rels, x, y) for y in dom_j)
            if supported:
                new_dom.append(x)
            else:
                removed_any = True
        if removed_any:
            domains[xi] = new_dom
        return removed_any

    if queue is None:
        work: list[tuple[str, str]] = list(csp.binary_relations.keys())
    else:
        work = list(queue)

    while work:
        xi, xj = work.pop(0)
        if revise(xi, xj):
            if not domains.get(xi):
                return False
            for xk in csp.neighbors.get(xi, set()):
                if xk == xj:
                    continue
                work.append((xk, xi))

    return True


def forward_check(
    csp: CSP,
    domains: dict[str, list[Any]],
    assignment: dict[str, Any],
    *,
    just_assigned_var: str,
) -> bool:
    """Forward checking after assigning `just_assigned_var` (mutates domains)."""

    x = just_assigned_var
    if x not in assignment:
        raise ValueError("forward_check expects just_assigned_var to be assigned.")
    x_val = assignment[x]

    for y in csp.neighbors.get(x, set()):
        if y in assignment:
            continue
        rels = csp.binary_relations.get((y, x)) or []
        if not rels:
            continue
        new_dom: list[Any] = []
        for y_val in domains.get(y, []):
            if _all_binary_satisfied(rels, y_val, x_val):
                new_dom.append(y_val)
        domains[y] = new_dom
        if not new_dom:
            return False

    # All-different constraints benefit from pruning assigned value even if not explicitly in binary neighbors.
    for c in csp.constraints_for(x):
        if c.type != "all_different":
            continue
        for y in c.vars:
            if y == x or y in assignment:
                continue
            before = domains.get(y, [])
            after = [v for v in before if v != x_val]
            if len(after) != len(before):
                domains[y] = after
                if not after:
                    return False

    return True


def solve_csp(
    csp: CSP,
    *,
    partial_assignment: dict[str, Any] | None = None,
    options: CSPSolverOptions | None = None,
) -> dict[str, Any] | None:
    """Solve CSP by backtracking with optional MRV/FC/AC-3.

    Determinism:
      - variable order: as provided in `csp.variables` (MRV ties broken by that order)
      - value order: as provided in `csp.domains[var]`
    """

    options = options or CSPSolverOptions()
    assignment: dict[str, Any] = {}
    domains = _copy_domains(csp.domains)

    # Apply partial assignment
    for var, val in (partial_assignment or {}).items():
        v = str(var)
        if v not in csp.variables:
            raise ValueError(f"Unknown variable in partial_assignment: '{v}'.")
        if val not in csp.domains[v]:
            raise ValueError(f"Value {val!r} not in domain of '{v}'.")
        assignment[v] = val
        domains[v] = [val]

    if not csp.is_assignment_consistent(assignment):
        return None

    # Initial propagation
    if options.forward_checking and assignment:
        for v in csp.variables:
            if v in assignment:
                if not forward_check(csp, domains, assignment, just_assigned_var=v):
                    return None

    if options.ac3_preprocess:
        if not ac3(csp, domains):
            return None

    def backtrack(curr_assignment: dict[str, Any], curr_domains: dict[str, list[Any]]) -> dict[str, Any] | None:
        if len(curr_assignment) == len(csp.variables):
            return dict(curr_assignment)

        var = _select_unassigned_variable(
            csp,
            curr_assignment,
            curr_domains,
            use_mrv=bool(options.mrv),
        )

        for value in list(curr_domains.get(var, [])):
            next_assignment = dict(curr_assignment)
            next_assignment[var] = value
            if not csp.is_assignment_consistent(next_assignment):
                continue

            next_domains = _copy_domains(curr_domains)
            next_domains[var] = [value]

            if options.forward_checking:
                if not forward_check(csp, next_domains, next_assignment, just_assigned_var=var):
                    continue

            if options.ac3_interleaved:
                initial_arcs = [(nei, var) for nei in csp.neighbors.get(var, set()) if nei not in next_assignment]
                if not ac3(csp, next_domains, queue=initial_arcs):
                    continue

            result = backtrack(next_assignment, next_domains)
            if result is not None:
                return result

        return None

    return backtrack(assignment, domains)


def _default_instances_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "csp_instances"


def list_csp_instances(instances_dir: str | Path | None = None) -> list[Path]:
    base = Path(instances_dir) if instances_dir is not None else _default_instances_dir()
    if not base.exists():
        return []
    return sorted([p for p in base.glob("*.json") if p.is_file()], key=lambda p: p.name)


def load_csp_instance(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("CSP instance JSON must be an object.")
    raw.setdefault("id", p.stem)
    raw.setdefault("title", raw.get("id"))
    raw.setdefault("partial_assignment", {})
    return raw


def build_csp_from_instance(raw: dict[str, Any]) -> tuple[CSP, dict[str, Any]]:
    variables_raw = raw.get("variables")
    if not isinstance(variables_raw, list) or not variables_raw:
        raise ValueError("Instance must contain non-empty `variables` list.")
    variables = [str(v) for v in variables_raw]

    domains_raw = raw.get("domains")
    if not isinstance(domains_raw, dict) or not domains_raw:
        raise ValueError("Instance must contain `domains` object.")

    domains: dict[str, list[Any]] = {}
    for v in variables:
        if v not in domains_raw:
            raise ValueError(f"Missing domain for variable '{v}'.")
        dom = domains_raw[v]
        if not isinstance(dom, list) or not dom:
            raise ValueError(f"Domain for '{v}' must be a non-empty list.")
        domains[v] = dom

    constraints_raw = raw.get("constraints") or []
    if not isinstance(constraints_raw, list):
        raise ValueError("`constraints` must be a list.")

    constraints: list[CSPConstraint] = []
    normalized_specs: list[dict[str, Any]] = []
    for idx, c in enumerate(constraints_raw):
        if not isinstance(c, dict):
            raise ValueError(f"Constraint #{idx} must be an object.")
        c_type = str(c.get("type") or "").strip()
        if not c_type:
            raise ValueError(f"Constraint #{idx} missing `type`.")
        vars_raw = c.get("vars")
        if not isinstance(vars_raw, list) or not vars_raw:
            raise ValueError(f"Constraint #{idx} missing `vars` list.")
        c_vars = tuple(str(v) for v in vars_raw)
        params = {str(k): v for k, v in c.items() if k not in {"type", "vars"}}

        # Normalize common aliases
        alias_map = {
            "all-different": "all_different",
            "allDifferent": "all_different",
            "ac-3": "ac3",
            "!=": "not_equal",
            "==": "equal",
            "<": "less_than",
            ">": "greater_than",
        }
        c_type_norm = alias_map.get(c_type, c_type)

        constraints.append(CSPConstraint(type=c_type_norm, vars=c_vars, params=params))
        normalized_specs.append({"type": c_type_norm, "vars": list(c_vars), **params})

    csp = CSP(variables=variables, domains=domains, constraints=constraints)
    return csp, {"variables": variables, "domains": domains, "constraints": normalized_specs}


def csp_instance_to_rows(raw: dict[str, Any]) -> list[list[str]]:
    """Turn an instance dict into a 2-col table (for PDF + UI quick view)."""

    csp, normalized = build_csp_from_instance(raw)
    rows: list[list[str]] = []
    rows.append(["Instanță", str(raw.get("title") or raw.get("id") or "")])
    rows.append(["Variabile", ", ".join(csp.variables)])
    for v in csp.variables:
        dom = ", ".join(str(x) for x in csp.domains[v])
        rows.append([f"Dom({v})", "{" + dom + "}"])
    for c in csp.constraints:
        rows.append(["Constrângere", c.to_text()])
    pa = raw.get("partial_assignment") or {}
    if isinstance(pa, dict) and pa:
        pa_text = ", ".join(f"{k}={pa[k]}" for k in csp.variables if k in pa)
    else:
        pa_text = "—"
    rows.append(["Asignare parțială", pa_text])

    opts = solver_options_from_instance(raw)
    method_parts = ["BT"]
    if opts.mrv:
        method_parts.append("MRV")
    if opts.forward_checking:
        method_parts.append("FC")
    if opts.ac3_preprocess or opts.ac3_interleaved:
        if opts.ac3_preprocess and opts.ac3_interleaved:
            method_parts.append("AC-3(pre+mac)")
        elif opts.ac3_interleaved:
            method_parts.append("AC-3(mac)")
        else:
            method_parts.append("AC-3(pre)")
    rows.append(["Metodă", " + ".join(method_parts)])
    return rows


class CSPInstanceProblem(BaseProblem):
    """Cerința 3: solve a given CSP instance using a specified method."""

    problem_type = "csp:bt-fc-mrv-ac3"

    def __init__(self, *, instance_path: str | Path, instances_dir: str | Path | None = None) -> None:
        self.instances_dir = Path(instances_dir) if instances_dir is not None else _default_instances_dir()
        self.instance_path = Path(instance_path)

    def generate(self) -> ProblemInstance:
        raw = load_csp_instance(self.instance_path)
        csp, normalized = build_csp_from_instance(raw)
        options = solver_options_from_instance(raw)
        partial = raw.get("partial_assignment") or {}
        if not isinstance(partial, dict):
            raise ValueError("partial_assignment must be an object.")

        solution = solve_csp(csp, partial_assignment=partial, options=options)
        if solution is None:
            return ProblemInstance(
                data=[],
                prompt="",
                solution=None,
                explanation="Instanța este nesatisfiabilă (nu există soluție completă).",
                metadata={"csp": normalized, "partial_assignment": partial, "solver_options": options.__dict__},
            )

        remaining_vars = [v for v in csp.variables if v not in partial]
        remaining_text = ", ".join(remaining_vars) if remaining_vars else "—"

        opts_label = []
        if options.mrv:
            opts_label.append("MRV")
        if options.forward_checking:
            opts_label.append("FC")
        if options.ac3_preprocess or options.ac3_interleaved:
            opts_label.append("AC-3")

        prompt = (
            "Cerința 3 — CSP.\n\n"
            "Se dă un CSP (variabile, domenii, constrângeri) și o asignare parțială. "
            f"Folosind Backtracking{' cu ' + '/'.join(opts_label) if opts_label else ''}, "
            f"completează asignarea pentru variabilele rămase: {remaining_text}.\n"
            "Scrie răspunsul în format: X=valoare, Y=valoare, ...\n"
        )

        explanation = "Asignare finală (gold standard): " + ", ".join(f"{v}={solution[v]}" for v in csp.variables)

        rows = csp_instance_to_rows(raw)
        metadata = {
            "csp_instance_id": raw.get("id"),
            "csp_instance_title": raw.get("title"),
            "csp": normalized,
            "partial_assignment": partial,
            "solver_options": options.__dict__,
            "remaining_variables": remaining_vars,
        }

        return ProblemInstance(
            data=rows,
            prompt=prompt,
            solution=solution,
            explanation=explanation,
            metadata=metadata,
        )

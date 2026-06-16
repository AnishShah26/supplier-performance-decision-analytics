"""
src/solver.py
-------------
Shared PuLP solver selection with a runtime compatibility check.

PuLP may report bundled CBC as "available" on macOS arm64 even when the
binary is x86_64-only and fails at launch with "Bad CPU type in executable".
We probe candidate solvers with a tiny LP once, cache the first working
option, and reuse it throughout the project.
"""
from __future__ import annotations

from functools import lru_cache

import pulp


_SOLVER_CANDIDATES = (
    "HiGHS",
    "HiGHS_CMD",
    "COIN_CMD",
    "PULP_CBC_CMD",
)


def _solver_works(solver_name: str) -> bool:
    solver_cls = getattr(pulp, solver_name, None)
    if solver_cls is None:
        return False

    try:
        x = pulp.LpVariable("probe_x", lowBound=0)
        prob = pulp.LpProblem("solver_probe", pulp.LpMaximize)
        prob += x
        prob += x <= 1
        status = prob.solve(solver_cls(msg=False))
    except Exception:
        return False

    return status == pulp.LpStatusOptimal


@lru_cache(maxsize=1)
def _working_solver_name() -> str:
    for solver_name in _SOLVER_CANDIDATES:
        if _solver_works(solver_name):
            return solver_name

    raise RuntimeError(
        "No compatible PuLP solver is available. Install HiGHS or CBC and try again."
    )


def get_lp_solver(msg: bool = False) -> pulp.LpSolver:
    """Return a compatible PuLP solver instance for the current machine."""
    solver_cls = getattr(pulp, _working_solver_name())
    return solver_cls(msg=msg)

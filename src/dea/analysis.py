"""
src/dea/analysis.py
-------------------
Data Envelopment Analysis: input-oriented VRS DEA and CCR-based cross-efficiency.

Methodology references
----------------------
Emrouznejad, A., & Yang, G. L. (2018). A survey and analysis of the first
    40 years of scholarly literature in DEA: 1978–2016.
    Socio-Economic Planning Sciences, 61, 4–8.

Rashidi, K., & Cullinane, K. (2019). Evaluating the sustainability of
    national logistics performance using Data Envelopment Analysis.
    Transport Policy, 74, 35–46.

Wu, J., Sun, J., & Liang, L. (2016). Extended secondary goal models for
    weights selection in DEA cross-efficiency evaluation.
    Computers & Industrial Engineering, 93, 143–151.

Tavana, M., Khalili-Damghani, K., Santos-Arteaga, F. J., & Mahmoudi, R.
    (2021). Efficiency decomposition and measurement in two-stage fuzzy DEA
    models using a bargaining game approach.
    Computers & Industrial Engineering, 103, 107285.

Ang, S., & Chen, C. M. (2016). Pitfalls of decomposition weights in the
    additive multi-stage DEA model. Omega, 58, 139–153.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pulp

from src.data.loader import SUPPLIERS, N
from src.solver import get_lp_solver


# ── Internal LP helpers ──────────────────────────────────────────────────────

def _vrs_input_oriented(
    o: int,
    X: np.ndarray,   # shape (m_inputs, N)
    Y: np.ndarray,   # shape (s_outputs, N)
) -> Tuple[float, np.ndarray]:
    """
    Solve the input-oriented VRS (BCC) DEA model for DMU o.

    min  θ
    s.t. Xλ ≤ θ x_o      (input feasibility)
         Yλ ≥ y_o         (output feasibility)
         1ᵀλ = 1           (VRS convexity)
         λ ≥ 0

    Returns
    -------
    theta : float        Efficiency score ∈ (0, 1]
    lam   : np.ndarray   Lambda weights, shape (N,)
    """
    m, n = X.shape
    s    = Y.shape[0]

    prob  = pulp.LpProblem(f"VRS_IO_{SUPPLIERS[o]}", pulp.LpMinimize)
    theta = pulp.LpVariable("theta", lowBound=0)
    lam   = [pulp.LpVariable(f"lam_{j}", lowBound=0) for j in range(n)]

    prob += theta

    for i in range(m):
        prob += pulp.lpSum(lam[j] * X[i, j] for j in range(n)) <= theta * X[i, o]
    for r in range(s):
        prob += pulp.lpSum(lam[j] * Y[r, j] for j in range(n)) >= Y[r, o]

    prob += pulp.lpSum(lam) == 1.0   # VRS

    prob.solve(get_lp_solver(msg=False))

    theta_val = float(pulp.value(theta))
    lam_vals  = np.array([float(pulp.value(lam[j])) for j in range(n)])
    return theta_val, lam_vals


def _ccr_multipliers(
    d: int,
    X: np.ndarray,
    Y: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Solve the CCR input-oriented multiplier model for DMU d.

    max  uᵀ y_d
    s.t. vᵀ x_d = 1
         uᵀ y_j − vᵀ x_j ≤ 0   ∀j
         u, v ≥ ε

    Returns
    -------
    u : np.ndarray  Output multipliers, shape (s,)
    v : np.ndarray  Input  multipliers, shape (m,)
    """
    m, n = X.shape
    s    = Y.shape[0]
    eps  = 1e-6

    prob = pulp.LpProblem(f"CCR_Mult_{SUPPLIERS[d]}", pulp.LpMaximize)
    u = [pulp.LpVariable(f"u_{r}", lowBound=eps) for r in range(s)]
    v = [pulp.LpVariable(f"v_{i}", lowBound=eps) for i in range(m)]

    # Normalisation
    prob += pulp.lpSum(v[i] * X[i, d] for i in range(m)) == 1.0
    # Objective
    prob += pulp.lpSum(u[r] * Y[r, d] for r in range(s))
    # Constraints
    for j in range(n):
        prob += (
            pulp.lpSum(u[r] * Y[r, j] for r in range(s))
            - pulp.lpSum(v[i] * X[i, j] for i in range(m))
        ) <= 0

    status = prob.solve(get_lp_solver(msg=False))
    if status != 1:
        return None, None

    u_vals = np.array([float(pulp.value(u[r])) for r in range(s)])
    v_vals = np.array([float(pulp.value(v[i])) for i in range(m)])
    return u_vals, v_vals


# ── Result dataclasses ───────────────────────────────────────────────────────

@dataclass
class DEAResult:
    """Container for VRS efficiency and CCR cross-efficiency results."""

    suppliers:      List[str]
    vrs_efficiency: np.ndarray          # shape (N,)
    vrs_lambda:     np.ndarray          # shape (N, N) — lambda[o, j]
    ccr_self:       np.ndarray          # shape (N,)  — CCR self-efficiency
    cross_matrix:   np.ndarray          # shape (N, N) — cross_matrix[d, o]
    avg_cross:      np.ndarray          # shape (N,)

    @property
    def efficient_mask(self) -> np.ndarray:
        return self.vrs_efficiency >= 0.9999

    @property
    def inefficient_indices(self) -> List[int]:
        return [i for i in range(N) if not self.efficient_mask[i]]

    @property
    def vrs_summary(self) -> pd.DataFrame:
        rows = []
        for i, s in enumerate(self.suppliers):
            peers = {
                self.suppliers[j]: round(self.vrs_lambda[i, j], 4)
                for j in range(N) if self.vrs_lambda[i, j] > 1e-3
            }
            rows.append({
                "Supplier":   s,
                "VRS Score":  round(self.vrs_efficiency[i], 4),
                "Status":     "Efficient" if self.efficient_mask[i] else "Inefficient",
                "Benchmarks": peers,
            })
        return pd.DataFrame(rows)

    @property
    def cross_summary(self) -> pd.DataFrame:
        order = np.argsort(-self.avg_cross)
        return pd.DataFrame({
            "Rank":          np.arange(1, N + 1),
            "Supplier":      [self.suppliers[i] for i in order],
            "CCR Self":      self.ccr_self[order].round(4),
            "Avg Cross-Eff": self.avg_cross[order].round(4),
        })


# ── Main computation ─────────────────────────────────────────────────────────

def run(
    X: np.ndarray,
    Y: np.ndarray,
    suppliers: List[str] = SUPPLIERS,
) -> DEAResult:
    """
    Execute VRS DEA and CCR cross-efficiency for all DMUs.

    Parameters
    ----------
    X : np.ndarray, shape (m_inputs, N)
        Input matrix — price, late delivery, shipping errors, lead time.
    Y : np.ndarray, shape (s_outputs, N)
        Output matrix — total purchase, PQ index, CS index.
    suppliers : list[str]

    Returns
    -------
    DEAResult
    """
    n = X.shape[1]

    # ── Stage 1: VRS input-oriented DEA ─────────────────────────────────────
    vrs_eff = np.zeros(n)
    vrs_lam = np.zeros((n, n))

    for o in range(n):
        eff, lam = _vrs_input_oriented(o, X, Y)
        vrs_eff[o]    = eff
        vrs_lam[o, :] = lam

    # ── Stage 2: CCR multiplier model → cross-efficiency matrix ─────────────
    # Cross-efficiency follows the standard CCR framework of Sexton, Silkman
    # & Hogan (1986) and Wu, Sun & Liang (2016): DMU d's CCR-optimal multipliers
    # (u_d, v_d) are used to appraise ALL DMUs, yielding E_{d,o} = u_d'y_o / v_d'x_o.
    # Using CCR alongside VRS primary DEA is a deliberate two-stage procedure —
    # VRS classifies scale-efficient DMUs; CCR provides scale-invariant peer appraisal.
    cross_mat = np.zeros((n, n))
    ccr_self  = np.zeros(n)

    for d in range(n):
        u_d, v_d = _ccr_multipliers(d, X, Y)
        if u_d is None:
            continue
        for o in range(n):
            num = float(u_d @ Y[:, o])
            den = float(v_d @ X[:, o])
            cross_mat[d, o] = num / den if den > 1e-12 else 0.0
        ccr_self[d] = cross_mat[d, d]

    avg_cross = cross_mat.mean(axis=0)

    return DEAResult(
        suppliers=suppliers,
        vrs_efficiency=vrs_eff,
        vrs_lambda=vrs_lam,
        ccr_self=ccr_self,
        cross_matrix=cross_mat,
        avg_cross=avg_cross,
    )

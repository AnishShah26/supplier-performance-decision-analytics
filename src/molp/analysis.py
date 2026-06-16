"""
src/molp/analysis.py
--------------------
Multiple Objective Linear Programming via Weighted Goal Programming (WGP).

Ideal-point aspirations (best-in-class across all 12 suppliers) are set
independently of the DEA projections, ensuring the WGP optimisation is
genuinely non-trivial — the solver must trade off competing objectives
and finds frontier points that differ from the pure radial DEA projection.

Methodology references
----------------------
Nazari-Shirkouhi, S., Miri Nargesi, S., & Rajabi, M. (2020).
    Supplier selection and order allocation using a two-phase fuzzy
    multi-objective linear programming. Applied Mathematical Modelling,
    84, 496–521.

Dotoli, M., Epicoco, N., Falagario, M., & Sciancalepore, F. (2015).
    A cross-efficiency fuzzy data envelopment analysis technique for
    performance evaluation of decision making units under uncertainty.
    Computers & Industrial Engineering, 79, 103–114.

Zhu, J. (2020). DEA under big data: Data enabled analytics and network
    data envelopment analysis. Annals of Operations Research, 1–16.

Mahdiloo, M., Saen, R. F., & Lee, K. H. (2015). Technical, environmental
    and eco-efficiency measurement for supplier selection: an extension
    and application of data envelopment analysis.
    International Journal of Production Economics, 168, 279–289.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pulp

from src.data.loader import SUPPLIERS, N
from src.dea.analysis import DEAResult
from src.solver import get_lp_solver


# ── Aspiration levels (ideal point) ─────────────────────────────────────────

ASPIRATION_LABELS: List[str] = [
    "Price", "Late%", "Errors%", "Lead", "PQ", "CS",
]

# Normalisation ranges for dimensionless WGP objective
RANGES: Dict[str, float] = {
    "Price":   5.10 - 1.33,   # 3.77
    "Late%":   9.0  - 1.0,    # 8.0
    "Errors%": 8.0  - 3.0,    # 5.0
    "Lead":    7.0  - 2.0,    # 5.0
    "PQ":      1.0,
    "CS":      1.0,
}
RANGE_VEC = np.array([RANGES[k] for k in ASPIRATION_LABELS])


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class WGPSupplierResult:
    supplier:     str
    current:      np.ndarray    # [price, late, errors, lead, PQ, CS]
    target:       np.ndarray    # WGP-optimal frontier point
    deviations:   np.ndarray    # residual deviations from ideal (≥ 0)
    aspirations:  np.ndarray    # ideal aspiration values
    wgp_obj:      float         # WGP objective Z
    benchmarks:   Dict[str, float]

    @property
    def improvements(self) -> np.ndarray:
        """Signed improvement required: negative for cost criteria, positive for benefit."""
        imp = np.zeros(6)
        imp[:4] = self.current[:4] - self.target[:4]   # cost: reduction needed
        imp[4:] = self.target[4:]   - self.current[4:] # benefit: increase needed
        return imp

    def as_dataframe(self) -> pd.DataFrame:
        sign = np.array([-1,-1,-1,-1,+1,+1])
        return pd.DataFrame({
            "Criterion":   ASPIRATION_LABELS,
            "Current":     np.round(self.current, 4),
            "WGP Target":  np.round(self.target, 4),
            "Improvement": np.round(self.improvements, 4),
            "Residual Dev":np.round(self.deviations, 4),
            "Aspiration":  np.round(self.aspirations, 4),
        })


@dataclass
class MOLPResult:
    supplier_results: Dict[str, WGPSupplierResult]
    aspirations:      np.ndarray
    weights:          np.ndarray    # equal weights by default

    @property
    def suppliers(self) -> List[str]:
        return list(self.supplier_results.keys())

    def summary(self) -> pd.DataFrame:
        rows = []
        for s, r in self.supplier_results.items():
            row = {"Supplier": s, "WGP Objective Z": round(r.wgp_obj, 4)}
            for i, lbl in enumerate(ASPIRATION_LABELS):
                row[f"Target_{lbl}"] = round(r.target[i], 4)
                row[f"ResidDev_{lbl}"] = round(r.deviations[i], 4)
            rows.append(row)
        return pd.DataFrame(rows)


# ── Main computation ─────────────────────────────────────────────────────────

def run(
    dea_result: DEAResult,
    X: np.ndarray,         # shape (4, N) — price, late, errors, lead
    Y_pq: np.ndarray,      # shape (N,)   — PQ scores
    Y_cs: np.ndarray,      # shape (N,)   — CS scores
    Y_tp: np.ndarray,      # shape (N,)   — total purchase (DEA only)
    equal_weights: bool = True,
) -> MOLPResult:
    """
    Solve the WGP target-setting model for each VRS-inefficient supplier.

    For supplier o, the model minimises:
        Z = Σ_k  w_k · d_k / R_k
    subject to:
        DEA feasibility (VRS, λ ≥ 0, Σλ = 1, Xλ ≤ t_input, Yλ ≥ t_output)
        Improvement  (t_input ≤ x_o,  t_output ≥ y_o)
        Deviations   (t_input − d_input⁺ ≤ T_input, t_output + d_output⁻ ≥ T_output)

    Aspiration levels are set at the observed ideal (best-in-class) value
    independently of the DEA projection, ensuring non-trivial optimisation.
    Total purchase is excluded from WGP targets (not supplier-controllable).

    Parameters
    ----------
    dea_result   : DEAResult
    X            : np.ndarray (4, N)
    Y_pq, Y_cs   : np.ndarray (N,)
    Y_tp         : np.ndarray (N,)  — used only for DEA feasibility in WGP
    equal_weights: bool

    Returns
    -------
    MOLPResult
    """
    n  = X.shape[1]
    m  = 4    # inputs
    s3 = 3    # outputs in DEA (including TP)

    # Build full output matrix for DEA feasibility constraint
    Y_full = np.vstack([Y_tp, Y_pq, Y_cs])   # shape (3, N)

    # ── Ideal aspirations ────────────────────────────────────────────────────
    T_price  = float(X[0].min())    # 1.33
    T_late   = float(X[1].min())    # 1.0
    T_errors = float(X[2].min())    # 3.0
    T_lead   = float(X[3].min())    # 2.0
    T_PQ     = float(Y_pq.max())    # 0.7428
    T_CS     = float(Y_cs.max())    # 0.6590

    aspirations = np.array([T_price, T_late, T_errors, T_lead, T_PQ, T_CS])
    weights = np.ones(6) / 6.0 if equal_weights else np.ones(6) / 6.0

    supplier_results: Dict[str, WGPSupplierResult] = {}

    for o in dea_result.inefficient_indices:
        s_name = SUPPLIERS[o]
        current = np.array([
            X[0, o], X[1, o], X[2, o], X[3, o],
            Y_pq[o], Y_cs[o],
        ])

        prob = pulp.LpProblem(f"WGP_{s_name}", pulp.LpMinimize)

        # Lambda variables
        lam = [pulp.LpVariable(f"lam_{j}", lowBound=0) for j in range(n)]

        # Target variables (frontier point)
        t_price  = pulp.LpVariable("t_price",  lowBound=0)
        t_late   = pulp.LpVariable("t_late",   lowBound=0)
        t_errors = pulp.LpVariable("t_errors", lowBound=0)
        t_lead   = pulp.LpVariable("t_lead",   lowBound=0)
        t_PQ     = pulp.LpVariable("t_PQ",     lowBound=0)
        t_CS     = pulp.LpVariable("t_CS",     lowBound=0)

        # Deviation variables (from ideal, always ≥ 0)
        d_price  = pulp.LpVariable("d_price",  lowBound=0)
        d_late   = pulp.LpVariable("d_late",   lowBound=0)
        d_errors = pulp.LpVariable("d_errors", lowBound=0)
        d_lead   = pulp.LpVariable("d_lead",   lowBound=0)
        d_PQ     = pulp.LpVariable("d_PQ",     lowBound=0)
        d_CS     = pulp.LpVariable("d_CS",     lowBound=0)

        # Objective: normalised weighted deviations
        prob += (
            weights[0] * d_price  / RANGE_VEC[0] +
            weights[1] * d_late   / RANGE_VEC[1] +
            weights[2] * d_errors / RANGE_VEC[2] +
            weights[3] * d_lead   / RANGE_VEC[3] +
            weights[4] * d_PQ     / RANGE_VEC[4] +
            weights[5] * d_CS     / RANGE_VEC[5]
        )

        # ── DEA feasibility ───────────────────────────────────────────────
        # Input constraints: Σλ x_j ≤ t_input
        prob += pulp.lpSum(lam[j] * X[0, j] for j in range(n)) <= t_price
        prob += pulp.lpSum(lam[j] * X[1, j] for j in range(n)) <= t_late
        prob += pulp.lpSum(lam[j] * X[2, j] for j in range(n)) <= t_errors
        prob += pulp.lpSum(lam[j] * X[3, j] for j in range(n)) <= t_lead
        # Output constraints: Σλ y_j ≥ t_output (TP fixed to current)
        prob += pulp.lpSum(lam[j] * Y_tp[j]  for j in range(n)) >= Y_tp[o]
        prob += pulp.lpSum(lam[j] * Y_pq[j]  for j in range(n)) >= t_PQ
        prob += pulp.lpSum(lam[j] * Y_cs[j]  for j in range(n)) >= t_CS
        # VRS
        prob += pulp.lpSum(lam) == 1.0

        # ── Improvement constraints ───────────────────────────────────────
        prob += t_price  <= X[0, o]
        prob += t_late   <= X[1, o]
        prob += t_errors <= X[2, o]
        prob += t_lead   <= X[3, o]
        prob += t_PQ     >= Y_pq[o]
        prob += t_CS     >= Y_cs[o]

        # ── Deviation definitions ─────────────────────────────────────────
        # Cost criteria: t ≤ T  =>  deviation = max(0, t − T)
        prob += t_price  - d_price  <= T_price
        prob += t_late   - d_late   <= T_late
        prob += t_errors - d_errors <= T_errors
        prob += t_lead   - d_lead   <= T_lead
        # Benefit criteria: t ≥ T  =>  deviation = max(0, T − t)
        prob += t_PQ + d_PQ >= T_PQ
        prob += t_CS + d_CS >= T_CS

        prob.solve(get_lp_solver(msg=False))

        target = np.array([
            float(pulp.value(t_price)),
            float(pulp.value(t_late)),
            float(pulp.value(t_errors)),
            float(pulp.value(t_lead)),
            float(pulp.value(t_PQ)),
            float(pulp.value(t_CS)),
        ])
        devs = np.array([
            float(pulp.value(d_price)),
            float(pulp.value(d_late)),
            float(pulp.value(d_errors)),
            float(pulp.value(d_lead)),
            float(pulp.value(d_PQ)),
            float(pulp.value(d_CS)),
        ])
        benchmarks = {
            SUPPLIERS[j]: round(float(pulp.value(lam[j])), 4)
            for j in range(n) if float(pulp.value(lam[j])) > 1e-3
        }

        supplier_results[s_name] = WGPSupplierResult(
            supplier=s_name,
            current=current,
            target=target,
            deviations=devs,
            aspirations=aspirations,
            wgp_obj=float(pulp.value(prob.objective)),
            benchmarks=benchmarks,
        )

    return MOLPResult(
        supplier_results=supplier_results,
        aspirations=aspirations,
        weights=weights,
    )

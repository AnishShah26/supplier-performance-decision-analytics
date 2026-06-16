"""
src/mcda/analysis.py
--------------------
Weighted Additive Multi-Criteria Decision Analysis (MCDA).

Methodology references
----------------------
Zardari, N. H., Ahmed, K., Shirazi, S. M., & Yusop, Z. B. (2015).
    Weighting Methods and Their Effects on Multi-Criteria Decision Making
    Model Outcomes in Water Resources Management. Springer.

Mardani, A., Jusoh, A., Nor, K. M. D., Khalifah, Z., Zakwan, N., &
    Valipour, A. (2015). Multiple criteria decision-making techniques and
    their applications — a review of the literature from 2000 to 2014.
    Economic Research, 28(1), 516–571.

Emrouznejad, A., & Marra, M. (2017). The state of the art development of
    AHP (1979–2017): a literature review with a social network analysis.
    International Journal of Production Research, 55(22), 6653–6675.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.data.loader import SupplierData, SUPPLIERS, N, GRADE_UTILS


# ── Weight specifications ────────────────────────────────────────────────────

# Task 1 — Product Quality
# Durability, Defects, Innovation, Variety, Packaging ≈ equal
# Brand reputation slightly less important → ratio ~1.25
PQ_WEIGHTS: Dict[str, float] = {
    "Durability":       5 / 29,   # 0.17241
    "Defects":          5 / 29,
    "Innovation":       5 / 29,
    "Variety":          5 / 29,
    "Packaging":        5 / 29,
    "Brand Reputation": 4 / 29,   # 0.13793
}

# Task 2 — Customer Service
# After-sales ≥ 2× any other; Warranty = CommEase = Stock ≈ equal; MinOrder least
CS_RAW: Dict[str, float] = {
    "After-Sales":   2.0,
    "Warranty":      1.0,
    "Comm. Ease":    1.0,
    "Stock Avail.":  1.0,
    "Min. Order":    0.5,
}
_cs_total = sum(CS_RAW.values())
CS_WEIGHTS: Dict[str, float] = {k: v / _cs_total for k, v in CS_RAW.items()}


# ── Utility transformations ──────────────────────────────────────────────────

def _linear_benefit(x: np.ndarray) -> np.ndarray:
    """Higher-is-better normalisation → [0,1]."""
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.ones_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def _linear_cost(x: np.ndarray) -> np.ndarray:
    """Lower-is-better normalisation → [0,1]."""
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.ones_like(x, dtype=float)
    return (hi - x) / (hi - lo)


def _expected_utility(belief_matrix: np.ndarray) -> np.ndarray:
    """
    Convert qualitative belief distribution to expected utility.

    E[u_i] = Σ_n β_{in} · u(H_n)  where u(H_n) ∈ {0, 0.25, 0.50, 0.75, 1.0}.

    Parameters
    ----------
    belief_matrix : np.ndarray, shape (N, 5)

    Returns
    -------
    np.ndarray, shape (N,)
    """
    return belief_matrix @ GRADE_UTILS


# ── Main computation ─────────────────────────────────────────────────────────

@dataclass
class MCDAResult:
    pq_scores:          np.ndarray          # shape (N,)
    cs_scores:          np.ndarray
    pq_criteria_utils:  pd.DataFrame        # shape (N, #criteria)
    cs_criteria_utils:  pd.DataFrame
    pq_weights:         Dict[str, float]
    cs_weights:         Dict[str, float]
    suppliers:          List[str]

    @property
    def pq_ranking(self) -> pd.DataFrame:
        idx = np.argsort(-self.pq_scores)
        return pd.DataFrame({
            "Rank":     np.arange(1, N + 1),
            "Supplier": [self.suppliers[i] for i in idx],
            "PQ Index": self.pq_scores[idx].round(4),
        })

    @property
    def cs_ranking(self) -> pd.DataFrame:
        idx = np.argsort(-self.cs_scores)
        return pd.DataFrame({
            "Rank":     np.arange(1, N + 1),
            "Supplier": [self.suppliers[i] for i in idx],
            "CS Index": self.cs_scores[idx].round(4),
        })


def run(data: SupplierData) -> MCDAResult:
    """
    Execute full MCDA pipeline for Tasks 1 and 2.

    Parameters
    ----------
    data : SupplierData

    Returns
    -------
    MCDAResult
    """
    sup = data.suppliers

    # ── Task 1: Product Quality ──────────────────────────────────────────────
    u_dur  = _linear_benefit(data.durability)
    u_def  = _linear_cost(data.defects)
    u_inn  = _expected_utility(data.innovation)
    u_var  = _linear_benefit(data.variety)
    u_pkg  = _expected_utility(data.packaging)
    u_brd  = _expected_utility(data.brand)

    pq_utils_df = pd.DataFrame({
        "Durability":       u_dur,
        "Defects":          u_def,
        "Innovation":       u_inn,
        "Variety":          u_var,
        "Packaging":        u_pkg,
        "Brand Reputation": u_brd,
    }, index=sup)

    w_pq = np.array(list(PQ_WEIGHTS.values()))
    pq_scores = pq_utils_df.values @ w_pq

    # ── Task 2: Customer Service ─────────────────────────────────────────────
    # After-sales — Fine mapped to H3=0.50 (third grade on 5-point scale)
    u_as   = _expected_utility(data.aftersales)
    u_war  = data.warranty_bin.astype(float)
    u_comm = _expected_utility(data.communication)
    u_stk  = _expected_utility(data.stock)
    u_mo   = _linear_cost(data.min_order)

    cs_utils_df = pd.DataFrame({
        "After-Sales":  u_as,
        "Warranty":     u_war,
        "Comm. Ease":   u_comm,
        "Stock Avail.": u_stk,
        "Min. Order":   u_mo,
    }, index=sup)

    w_cs = np.array(list(CS_WEIGHTS.values()))
    cs_scores = cs_utils_df.values @ w_cs

    return MCDAResult(
        pq_scores=pq_scores,
        cs_scores=cs_scores,
        pq_criteria_utils=pq_utils_df,
        cs_criteria_utils=cs_utils_df,
        pq_weights=PQ_WEIGHTS,
        cs_weights=CS_WEIGHTS,
        suppliers=sup,
    )

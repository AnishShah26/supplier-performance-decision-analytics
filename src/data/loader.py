"""
src/data/loader.py
------------------
Central data store for the 12-supplier fashion jewellery retailer dataset.
All raw data is drawn directly from the BMAN60092 coursework brief appendices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List


SUPPLIERS: List[str] = list("ABCDEFGHIJKL")
N: int = len(SUPPLIERS)

# ── Quantitative data ────────────────────────────────────────────────────────

RAW_PRICE: List[float] = [1.34,1.44,2.13,2.01,3.44,2.98,1.34,5.10,1.40,1.50,1.33,1.94]
RAW_TOTAL_PURCHASE: List[float] = [
    5640.80, 38948.61, 2714.73, 13574.56, 3249.00, 2503.00,
    108124.50, 100241.16, 21985.05, 12091.10, 2245.00, 14723.00,
]
RAW_DURABILITY: List[float] = [10,10,5,5,5,5,3,10,5,4,5,5]
RAW_DEFECTS:    List[float] = [4,2,6,7,5,4,9,1,8,5,6,2]
RAW_VARIETY:    List[float] = [25,14,29,42,14,23,12,14,11,18,14,22]
RAW_LATE:       List[float] = [3,5,6,9,4,4,1,2,1,6,3,9]
RAW_ERRORS:     List[float] = [8,4,6,3,7,6,3,3,3,4,3,6]
RAW_LEAD:       List[float] = [5,4,4,5,2,4,2,3,4,7,7,7]
RAW_WARRANTY:   List[str]   = ["Y","Y","Y","N","N","N","N","Y","N","N","N","N"]
RAW_MIN_ORDER:  List[float] = [1000,1000,1000,1000,500,500,100,1000,500,500,1000,1000]

# ── Qualitative belief distributions ────────────────────────────────────────
# Rows = suppliers A-L, cols = H1..H5 (worst → best)

INNOVATION: np.ndarray = np.array([
    [0.00,0.00,0.50,0.50,0.00],
    [0.00,0.05,0.30,0.65,0.00],
    [0.00,0.00,0.70,0.30,0.00],
    [0.00,0.30,0.70,0.00,0.00],
    [0.00,0.70,0.20,0.10,0.00],
    [0.00,0.80,0.20,0.00,0.00],
    [0.50,0.50,0.00,0.00,0.00],
    [0.00,0.00,0.20,0.40,0.40],
    [0.00,0.20,0.40,0.40,0.00],
    [0.00,0.70,0.30,0.00,0.00],
    [0.20,0.50,0.30,0.00,0.00],
    [0.00,0.50,0.50,0.00,0.00],
])

BRAND: np.ndarray = np.array([
    [0.00,0.00,0.30,0.70,0.00],
    [0.00,0.10,0.40,0.50,0.00],
    [0.00,0.50,0.50,0.00,0.00],
    [0.10,0.50,0.40,0.00,0.00],
    [0.00,0.40,0.50,0.10,0.00],
    [0.30,0.40,0.30,0.00,0.00],
    [0.20,0.50,0.30,0.00,0.00],
    [0.00,0.00,0.10,0.60,0.30],
    [0.00,0.00,0.70,0.30,0.00],
    [0.00,0.30,0.40,0.30,0.00],
    [0.30,0.40,0.30,0.00,0.00],
    [0.00,0.30,0.70,0.00,0.00],
])

PACKAGING: np.ndarray = np.array([
    [0.00,0.00,0.40,0.20,0.40],
    [0.00,0.30,0.40,0.30,0.00],
    [0.00,0.40,0.50,0.10,0.00],
    [0.00,0.60,0.30,0.10,0.00],
    [0.00,0.50,0.50,0.00,0.00],
    [0.00,0.00,1.00,0.00,0.00],
    [0.10,0.50,0.40,0.00,0.00],
    [0.00,0.00,0.00,0.90,0.10],
    [0.00,0.20,0.20,0.60,0.00],
    [0.00,0.50,0.40,0.10,0.00],
    [0.00,0.70,0.30,0.00,0.00],
    [0.00,0.50,0.40,0.10,0.00],
])

AFTERSALES: np.ndarray = np.array([
    [0.00,0.00,0.40,0.60,0.00],
    [0.00,0.10,0.30,0.60,0.00],
    [0.00,0.00,0.70,0.30,0.00],
    [0.00,0.30,0.40,0.30,0.00],
    [0.10,0.40,0.50,0.00,0.00],
    [0.20,0.80,0.00,0.00,0.00],
    [0.30,0.30,0.40,0.00,0.00],
    [0.00,0.00,0.50,0.50,0.00],
    [0.00,0.10,0.60,0.30,0.00],
    [0.00,0.40,0.60,0.00,0.00],
    [0.00,0.30,0.60,0.10,0.00],
    [0.00,0.00,1.00,0.00,0.00],
])

COMMUNICATION: np.ndarray = np.array([
    [0.00,0.30,0.70,0.00,0.00],
    [0.00,0.60,0.40,0.00,0.00],
    [0.00,0.50,0.50,0.00,0.00],
    [0.00,0.30,0.70,0.00,0.00],
    [0.00,0.40,0.30,0.30,0.00],
    [0.00,0.70,0.30,0.00,0.00],
    [0.50,0.50,0.00,0.00,0.00],
    [0.00,0.00,1.00,0.00,0.00],
    [0.00,0.40,0.60,0.00,0.00],
    [0.20,0.70,0.10,0.00,0.00],
    [0.00,0.50,0.50,0.00,0.00],
    [0.00,0.20,0.50,0.30,0.00],
])

STOCK: np.ndarray = np.array([
    [0.00,0.00,0.90,0.10,0.00],
    [0.00,0.00,0.40,0.40,0.20],
    [0.00,0.00,0.80,0.20,0.00],
    [0.00,0.50,0.50,0.00,0.00],
    [0.00,0.40,0.60,0.00,0.00],
    [0.30,0.50,0.20,0.00,0.00],
    [0.00,0.70,0.30,0.00,0.00],
    [0.00,0.00,0.00,0.50,0.50],
    [0.00,0.60,0.40,0.00,0.00],
    [0.20,0.30,0.50,0.00,0.00],
    [0.00,0.40,0.50,0.10,0.00],
    [0.00,0.80,0.20,0.00,0.00],
])

# ── Grade utility scales  (H1=0, H2=0.25, H3=0.50, H4=0.75, H5=1.00) ────────
GRADE_UTILS: np.ndarray = np.array([0.00, 0.25, 0.50, 0.75, 1.00])


@dataclass
class SupplierData:
    """Validated, ready-to-use supplier dataset."""

    suppliers: List[str]

    # Quantitative raw vectors
    price:          np.ndarray
    total_purchase: np.ndarray
    durability:     np.ndarray
    defects:        np.ndarray
    variety:        np.ndarray
    late:           np.ndarray
    errors:         np.ndarray
    lead:           np.ndarray
    warranty_bin:   np.ndarray   # 1=Y, 0=N
    min_order:      np.ndarray

    # Qualitative belief matrices (12×5)
    innovation:    np.ndarray
    brand:         np.ndarray
    packaging:     np.ndarray
    aftersales:    np.ndarray
    communication: np.ndarray
    stock:         np.ndarray

    def __post_init__(self) -> None:
        assert all(len(getattr(self, f)) == N for f in [
            "price","total_purchase","durability","defects","variety",
            "late","errors","lead","warranty_bin","min_order",
        ]), "Quantitative arrays must all have length N=12"
        for mat_name in ["innovation","brand","packaging","aftersales","communication","stock"]:
            mat = getattr(self, mat_name)
            assert mat.shape == (N, 5), f"{mat_name} must be (12,5)"
            assert np.allclose(mat.sum(axis=1), 1.0, atol=1e-6), \
                f"{mat_name} rows must sum to 1"

    def as_input_matrix(self) -> np.ndarray:
        """Return DEA input matrix: shape (4, N) — price, late, errors, lead."""
        return np.vstack([self.price, self.late, self.errors, self.lead])

    def as_output_matrix(self, pq: np.ndarray, cs: np.ndarray) -> np.ndarray:
        """Return DEA output matrix: shape (3, N) — total_purchase in $000, pq, cs.

        Total purchase is scaled to thousands for numerical stability and to match
        the DEA matrix reported in the coursework appendix. Scaling a whole output
        column by a constant does not change VRS DEA classification, but it makes
        multiplier-based outputs easier to inspect.
        """
        return np.vstack([self.total_purchase / 1000.0, pq, cs])


def load() -> SupplierData:
    """Factory function — returns the validated SupplierData object."""
    warranty_bin = np.array([1 if w == "Y" else 0 for w in RAW_WARRANTY], dtype=float)
    return SupplierData(
        suppliers=SUPPLIERS,
        price=np.array(RAW_PRICE),
        total_purchase=np.array(RAW_TOTAL_PURCHASE),
        durability=np.array(RAW_DURABILITY),
        defects=np.array(RAW_DEFECTS),
        variety=np.array(RAW_VARIETY),
        late=np.array(RAW_LATE),
        errors=np.array(RAW_ERRORS),
        lead=np.array(RAW_LEAD),
        warranty_bin=warranty_bin,
        min_order=np.array(RAW_MIN_ORDER),
        innovation=INNOVATION,
        brand=BRAND,
        packaging=PACKAGING,
        aftersales=AFTERSALES,
        communication=COMMUNICATION,
        stock=STOCK,
    )

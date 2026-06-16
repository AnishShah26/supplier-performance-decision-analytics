"""
main.py
-------
Orchestrates the full supplier analysis pipeline:
    1. Load and validate data
    2. Run MCDA  (Tasks 1 & 2)
    3. Run DEA   (VRS + CCR cross-efficiency)
    4. Run MOLP  (WGP target-setting)
    5. Generate selected charts and print result summaries
"""
from __future__ import annotations

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

import src.data          as data_mod
import src.mcda          as mcda_mod
import src.dea           as dea_mod
import src.molp          as molp_mod
import src.visualization as viz_mod


def main() -> None:
    print("=" * 60)
    print("  Supplier Performance Decision Analytics")
    print("  MCDA · DEA · Cross-Efficiency · MOLP/WGP")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading and validating dataset …")
    d = data_mod.load()
    print(f"  {len(d.suppliers)} suppliers loaded: {', '.join(d.suppliers)}")

    # ── 2. MCDA ──────────────────────────────────────────────────────────────
    print("\n[2/5] Running MCDA …")
    mcda = mcda_mod.run(d)
    print("\n  Product Quality Ranking:")
    print(mcda.pq_ranking.to_string(index=False))
    print("\n  Customer Service Ranking:")
    print(mcda.cs_ranking.to_string(index=False))

    # ── 3. DEA ───────────────────────────────────────────────────────────────
    print("\n[3/5] Running DEA (VRS + CCR cross-efficiency) …")
    X = d.as_input_matrix()                          # shape (4, 12)
    Y = d.as_output_matrix(mcda.pq_scores,           # shape (3, 12)
                            mcda.cs_scores)
    dea = dea_mod.run(X, Y)
    print("\n  VRS DEA Summary:")
    print(dea.vrs_summary.drop(columns="Benchmarks").to_string(index=False))
    print("\n  Cross-Efficiency Ranking:")
    print(dea.cross_summary.to_string(index=False))

    # ── 4. MOLP / WGP ────────────────────────────────────────────────────────
    print("\n[4/5] Running MOLP / WGP target-setting …")
    molp = molp_mod.run(
        dea_result=dea,
        X=X,
        Y_pq=mcda.pq_scores,
        Y_cs=mcda.cs_scores,
        Y_tp=d.total_purchase,
    )
    print("\n  WGP Results Summary:")
    print(molp.summary().to_string(index=False))
    for s, r in molp.supplier_results.items():
        print(f"\n  Supplier {s}  (Z = {r.wgp_obj:.4f}  |  benchmarks: {r.benchmarks})")
        print(r.as_dataframe().to_string(index=False))

    # ── 5. Charts ────────────────────────────────────────────────────────────
    print("\n[5/5] Generating charts …")
    chart_paths = viz_mod.generate_all(mcda, dea, molp)

    # ── Done ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Analysis complete.")
    print("  Charts → outputs/charts/")
    print("=" * 60)

    return mcda, dea, molp, chart_paths


if __name__ == "__main__":
    main()

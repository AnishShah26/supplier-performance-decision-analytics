"""
src/visualization/charts.py
---------------------------
Publication-quality charts for MCDA, DEA and MOLP/WGP results.
All figures use a consistent colour palette and are saved as 300 dpi PNG.
"""
from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

from src.mcda.analysis  import MCDAResult
from src.dea.analysis   import DEAResult
from src.molp.analysis  import MOLPResult, ASPIRATION_LABELS

# ── Global style ─────────────────────────────────────────────────────────────

PALETTE = {
    "blue_dark":  "#1F4E79",
    "blue_mid":   "#2E75B6",
    "blue_light": "#BDD7EE",
    "teal":       "#00B0A0",
    "orange":     "#E07B39",
    "red":        "#C0392B",
    "green":      "#1A7A4A",
    "gold":       "#D4AC0D",
    "grey":       "#7F8C8D",
    "bg":         "#FAFBFC",
    "line":       "#DEE2E6",
}

INEEFF_COLOR = "#C0392B"
EFF_COLOR    = "#1F4E79"
TARGET_COLOR = "#1A7A4A"
CURRENT_COLOR = "#E07B39"
IDEAL_COLOR  = "#8E44AD"

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.facecolor":     PALETTE["bg"],
    "figure.facecolor":   "white",
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.framealpha":  0.85,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
})

OUTDIR = Path("outputs/charts")
OUTDIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> Path:
    path = OUTDIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  Saved {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  MCDA CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def fig01_pq_ranking(res: MCDAResult) -> Path:
    """Horizontal bar chart — Product Quality ranking."""
    df  = res.pq_ranking
    sup = df["Supplier"].tolist()
    sc  = df["PQ Index"].tolist()
    rk  = df["Rank"].tolist()

    colors = [
        PALETTE["blue_dark"]  if r <= 3
        else PALETTE["blue_mid"] if r <= 6
        else PALETTE["blue_light"] if r <= 9
        else "#D6DCE4"
        for r in rk
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.barh(range(len(sup)), sc, color=colors, edgecolor="white",
                   linewidth=0.6, height=0.7)

    for bar, s, score, rank in zip(bars, sup, sc, rk):
        ax.text(score + 0.005, bar.get_y() + bar.get_height()/2,
                f"{score:.4f}", va="center", ha="left", fontsize=8.5,
                color=PALETTE["blue_dark"], fontweight="bold" if rank <= 3 else "normal")

    ax.set_yticks(range(len(sup)))
    ax.set_yticklabels([f"#{r}  {s}" for r, s in zip(rk, sup)], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Product Quality Index (MCDA Weighted Score)", labelpad=6)
    ax.set_title("Figure 1 — Product Quality MCDA Ranking (12 Suppliers)", pad=10)
    ax.set_xlim(0, max(sc) * 1.15)
    ax.axvline(x=0, color=PALETTE["line"], lw=0.8)

    legend_handles = [
        mpatches.Patch(color=PALETTE["blue_dark"],  label="Top 3"),
        mpatches.Patch(color=PALETTE["blue_mid"],   label="Ranks 4–6"),
        mpatches.Patch(color=PALETTE["blue_light"], label="Ranks 7–9"),
        mpatches.Patch(color="#D6DCE4",             label="Bottom 3"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)
    ax.grid(axis="x", color=PALETTE["line"], linewidth=0.5, linestyle="--")
    fig.tight_layout()
    return _save(fig, "fig01_pq_ranking")


def fig02_cs_ranking(res: MCDAResult) -> Path:
    """Horizontal bar chart — Customer Service ranking."""
    df  = res.cs_ranking
    sup = df["Supplier"].tolist()
    sc  = df["CS Index"].tolist()
    rk  = df["Rank"].tolist()

    colors = [
        PALETTE["teal"]       if r <= 3
        else "#48C9B0"        if r <= 6
        else "#A3E4D7"        if r <= 9
        else "#D1F2EB"
        for r in rk
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.barh(range(len(sup)), sc, color=colors, edgecolor="white",
                   linewidth=0.6, height=0.7)

    for bar, s, score, rank in zip(bars, sup, sc, rk):
        ax.text(score + 0.005, bar.get_y() + bar.get_height()/2,
                f"{score:.4f}", va="center", ha="left", fontsize=8.5,
                color=PALETTE["blue_dark"], fontweight="bold" if rank <= 3 else "normal")

    ax.set_yticks(range(len(sup)))
    ax.set_yticklabels([f"#{r}  {s}" for r, s in zip(rk, sup)], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Customer Service Index (MCDA Weighted Score)", labelpad=6)
    ax.set_title("Figure 2 — Customer Service MCDA Ranking (12 Suppliers)", pad=10)
    ax.set_xlim(0, max(sc) * 1.15)
    ax.axvline(x=0, color=PALETTE["line"], lw=0.8)

    legend_handles = [
        mpatches.Patch(color=PALETTE["teal"], label="Top 3"),
        mpatches.Patch(color="#48C9B0",       label="Ranks 4–6"),
        mpatches.Patch(color="#A3E4D7",       label="Ranks 7–9"),
        mpatches.Patch(color="#D1F2EB",       label="Bottom 3"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)
    ax.grid(axis="x", color=PALETTE["line"], linewidth=0.5, linestyle="--")
    fig.tight_layout()
    return _save(fig, "fig02_cs_ranking")


def fig03_pq_cs_scatter(res: MCDAResult) -> Path:
    """PQ vs CS scatter — quadrant analysis."""
    pq  = res.pq_scores
    cs  = res.cs_scores
    sup = res.suppliers

    pq_mid = float(np.median(pq))
    cs_mid = float(np.median(cs))

    fig, ax = plt.subplots(figsize=(7.5, 6))

    scatter = ax.scatter(pq, cs, s=110, zorder=5,
                         c=[PALETTE["blue_dark"] if p >= pq_mid and c >= cs_mid
                            else PALETTE["orange"] if p < pq_mid and c < cs_mid
                            else PALETTE["teal"] if p >= pq_mid
                            else PALETTE["grey"]
                            for p, c in zip(pq, cs)],
                         edgecolors="white", linewidths=0.8)

    for i, s in enumerate(sup):
        ax.annotate(s, (pq[i], cs[i]), xytext=(5, 4),
                    textcoords="offset points", fontsize=9, fontweight="bold",
                    color=PALETTE["blue_dark"])

    ax.axvline(pq_mid, color=PALETTE["line"], lw=1, linestyle="--")
    ax.axhline(cs_mid, color=PALETTE["line"], lw=1, linestyle="--")

    ax.text(pq_mid + 0.005, ax.get_ylim()[1] * 0.97,
            "Median PQ", fontsize=7.5, color=PALETTE["grey"])
    ax.text(ax.get_xlim()[0] + 0.005, cs_mid + 0.008,
            "Median CS", fontsize=7.5, color=PALETTE["grey"])

    # Quadrant labels
    for txt, x, y in [
        ("High PQ\nHigh CS", max(pq)*0.85, max(cs)*0.93),
        ("Low PQ\nLow CS",  min(pq)*1.05,  min(cs)*1.25),
    ]:
        ax.text(x, y, txt, fontsize=7.5, ha="center",
                color=PALETTE["grey"], style="italic")

    ax.set_xlabel("Product Quality Index (PQ)", labelpad=6)
    ax.set_ylabel("Customer Service Index (CS)", labelpad=6)
    ax.set_title("Figure 3 — PQ vs CS Quadrant Analysis", pad=10)

    legend_handles = [
        mpatches.Patch(color=PALETTE["blue_dark"], label="High PQ & High CS"),
        mpatches.Patch(color=PALETTE["teal"],      label="High PQ, Low CS"),
        mpatches.Patch(color=PALETTE["grey"],      label="Low PQ, High CS"),
        mpatches.Patch(color=PALETTE["orange"],    label="Low PQ & Low CS"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=True)
    ax.grid(color=PALETTE["line"], linewidth=0.4, linestyle=":")
    fig.tight_layout()
    return _save(fig, "fig03_pq_cs_scatter")


def fig04_criteria_heatmap(res: MCDAResult) -> Path:
    """Heatmap of normalised utility scores for all criteria × suppliers."""
    pq_df = res.pq_criteria_utils.copy()
    cs_df = res.cs_criteria_utils.copy()

    # Combine and sort by PQ score
    combined = pd.concat([pq_df, cs_df], axis=1)
    order = np.argsort(-res.pq_scores)
    combined = combined.iloc[order]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(
        combined,
        ax=ax,
        cmap=cmap,
        vmin=0, vmax=1,
        annot=True, fmt=".2f",
        annot_kws={"size": 7.5},
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Utility [0,1]", "shrink": 0.7},
    )

    # Separate PQ and CS columns visually
    pq_cols = len(pq_df.columns)
    ax.axvline(pq_cols, color=PALETTE["orange"], lw=2)

    ax.set_title("Figure 4 — Normalised Utility Heatmap: All Criteria × All Suppliers\n"
                 "(Suppliers sorted by descending PQ; orange line separates PQ | CS criteria)",
                 pad=10, fontsize=11)
    ax.set_ylabel("Supplier (PQ-ranked)", labelpad=6)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8.5)
    fig.tight_layout()
    return _save(fig, "fig04_criteria_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
#  DEA CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def fig05_vrs_efficiency(res: DEAResult) -> Path:
    """Bar chart — VRS DEA efficiency scores."""
    order = np.argsort(-res.vrs_efficiency)
    sup   = [res.suppliers[i] for i in order]
    eff   = res.vrs_efficiency[order]
    is_eff = eff >= 0.9999

    colors = [EFF_COLOR if e else INEEFF_COLOR for e in is_eff]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(sup, eff, color=colors, edgecolor="white", linewidth=0.6, width=0.65)

    for bar, score, eff_flag in zip(bars, eff, is_eff):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.012,
                f"{score:.4f}", ha="center", va="bottom", fontsize=7.5,
                color=EFF_COLOR if eff_flag else INEEFF_COLOR,
                fontweight="bold" if eff_flag else "normal")

    ax.axhline(1.0, color=PALETTE["gold"], lw=1.5, linestyle="--",
               label="Efficiency frontier (θ = 1)")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("VRS Efficiency Score (θ)", labelpad=6)
    ax.set_title("Figure 5 — Input-Oriented VRS DEA Efficiency Scores (12 Suppliers)", pad=10)

    legend_handles = [
        mpatches.Patch(color=EFF_COLOR,    label="Efficient (θ = 1.000)"),
        mpatches.Patch(color=INEEFF_COLOR, label="Inefficient (θ < 1.000)"),
        Line2D([0],[0], color=PALETTE["gold"], lw=1.5, linestyle="--",
               label="Frontier (θ = 1)"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=True)
    ax.grid(axis="y", color=PALETTE["line"], linewidth=0.4, linestyle=":")
    fig.tight_layout()
    return _save(fig, "fig05_vrs_efficiency")


def fig06_cross_efficiency(res: DEAResult) -> Path:
    """Grouped bar — CCR self-efficiency and average cross-efficiency."""
    df    = res.cross_summary
    sup   = df["Supplier"].tolist()
    ccr   = df["CCR Self"].tolist()
    cross = df["Avg Cross-Eff"].tolist()

    x = np.arange(len(sup))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.8))
    b1 = ax.bar(x - w/2, ccr,   width=w, label="CCR Self-Efficiency",
                color=PALETTE["blue_mid"], edgecolor="white", lw=0.5)
    b2 = ax.bar(x + w/2, cross, width=w, label="Avg Cross-Efficiency",
                color=PALETTE["teal"],     edgecolor="white", lw=0.5)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.012,
                f"{h:.3f}", ha="center", va="bottom", fontsize=6.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"#{i+1} {s}" for i, s in enumerate(sup)],
                       rotation=30, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Efficiency Score", labelpad=6)
    ax.set_title("Figure 6 — CCR Self-Efficiency vs Average Cross-Efficiency\n"
                 "(Suppliers ranked by cross-efficiency)", pad=10)
    ax.axhline(1.0, color=PALETTE["gold"], lw=1.2, linestyle="--", alpha=0.7)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", color=PALETTE["line"], linewidth=0.4, linestyle=":")
    fig.tight_layout()
    return _save(fig, "fig06_cross_efficiency")


def fig07_vrs_vs_cross_scatter(res: DEAResult) -> Path:
    """Scatter — VRS efficiency vs avg cross-efficiency (quadrant view)."""
    vrs   = res.vrs_efficiency
    cross = res.avg_cross
    sup   = res.suppliers

    vrs_threshold   = 1.0
    cross_threshold = float(np.median(cross))

    def _color(v, c):
        if v >= 0.9999 and c >= cross_threshold: return PALETTE["blue_dark"]
        if v >= 0.9999 and c <  cross_threshold: return PALETTE["gold"]
        return INEEFF_COLOR

    colors = [_color(v, c) for v, c in zip(vrs, cross)]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(vrs, cross, s=120, c=colors, edgecolors="white",
               linewidths=0.8, zorder=5)

    for i, s in enumerate(sup):
        ax.annotate(s, (vrs[i], cross[i]), xytext=(5, 4),
                    textcoords="offset points", fontsize=9,
                    fontweight="bold", color=PALETTE["blue_dark"])

    ax.axvline(0.9999, color=PALETTE["line"], lw=1.2, linestyle="--",
               label="VRS frontier (θ=1)")
    ax.axhline(cross_threshold, color=PALETTE["line"], lw=1.2, linestyle=":",
               label=f"Median cross-eff ({cross_threshold:.3f})")

    ax.set_xlabel("VRS Efficiency Score (θ)", labelpad=6)
    ax.set_ylabel("Average Cross-Efficiency", labelpad=6)
    ax.set_title("Figure 7 — VRS Efficiency vs Average Cross-Efficiency\n"
                 "(Quadrant view: broadly robust vs. locally efficient)", pad=10)

    legend_handles = [
        mpatches.Patch(color=PALETTE["blue_dark"], label="VRS-eff & above median cross-eff"),
        mpatches.Patch(color=PALETTE["gold"],       label="VRS-eff but below median cross-eff"),
        mpatches.Patch(color=INEEFF_COLOR,          label="VRS-inefficient"),
        Line2D([0],[0], color=PALETTE["line"], lw=1.2, linestyle="--", label="VRS frontier"),
        Line2D([0],[0], color=PALETTE["line"], lw=1.2, linestyle=":",  label="Median cross-eff"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=True, fontsize=8)
    ax.grid(color=PALETTE["line"], linewidth=0.4, linestyle=":")
    fig.tight_layout()
    return _save(fig, "fig07_vrs_vs_cross_scatter")


# ─────────────────────────────────────────────────────────────────────────────
#  MOLP / WGP CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def fig08_wgp_gap_analysis(res: MOLPResult) -> Path:
    """
    Grouped bar chart — current vs WGP target vs ideal aspiration
    for each criterion, one sub-panel per inefficient supplier.
    """
    suppliers = list(res.supplier_results.keys())
    ns = len(suppliers)
    labels_short = ["Price", "Late%", "Err%", "Lead", "PQ", "CS"]
    n_crit = 6

    fig, axes = plt.subplots(1, ns, figsize=(14, 5.2), sharey=False)
    if ns == 1:
        axes = [axes]

    x = np.arange(n_crit)
    w = 0.26

    for ax, s_name in zip(axes, suppliers):
        r = res.supplier_results[s_name]
        cur  = r.current
        tgt  = r.target
        asp  = r.aspirations

        b1 = ax.bar(x - w, cur, width=w, color=CURRENT_COLOR, label="Current",
                    edgecolor="white", lw=0.5, alpha=0.9)
        b2 = ax.bar(x,     tgt, width=w, color=TARGET_COLOR,  label="WGP Target",
                    edgecolor="white", lw=0.5, alpha=0.9)
        b3 = ax.bar(x + w, asp, width=w, color=IDEAL_COLOR,   label="Ideal Aspiration",
                    edgecolor="white", lw=0.5, alpha=0.6, linestyle="--")

        ax.set_xticks(x)
        ax.set_xticklabels(labels_short, fontsize=8)
        ax.set_title(f"Supplier {s_name}\nZ = {r.wgp_obj:.4f}", fontsize=10,
                     fontweight="bold", color=INEEFF_COLOR)
        ax.grid(axis="y", color=PALETTE["line"], linewidth=0.4, linestyle=":")
        ax.set_ylabel("Value", fontsize=8, labelpad=4)

    legend_handles = [
        mpatches.Patch(color=CURRENT_COLOR, label="Current"),
        mpatches.Patch(color=TARGET_COLOR,  label="WGP Target"),
        mpatches.Patch(color=IDEAL_COLOR,   label="Ideal Aspiration"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Figure 8 — WGP Gap Analysis: Current / Target / Ideal per Criterion",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "fig08_wgp_gap_analysis")


def fig09_residual_deviation_heatmap(res: MOLPResult) -> Path:
    """Heatmap — residual deviations from ideal after WGP optimisation."""
    suppliers = list(res.supplier_results.keys())
    devs = np.array([res.supplier_results[s].deviations for s in suppliers])
    # Normalise by range for comparability
    ranges = np.array([3.77, 8.0, 5.0, 5.0, 1.0, 1.0])
    devs_norm = devs / ranges

    df = pd.DataFrame(devs_norm, index=suppliers, columns=ASPIRATION_LABELS)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(
        df, ax=ax,
        cmap=cmap, vmin=0, vmax=0.55,
        annot=True, fmt=".3f",
        annot_kws={"size": 9.5, "weight": "bold"},
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Normalised Residual Deviation", "shrink": 0.8},
    )
    ax.set_title("Figure 9 — Normalised Residual Deviations from Ideal (WGP Solution)\n"
                 "Darker = further from ideal after optimal improvement",
                 pad=10, fontsize=11)
    ax.set_ylabel("Supplier", labelpad=6)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    fig.tight_layout()
    return _save(fig, "fig09_residual_deviation_heatmap")


def fig10_radar_current_vs_target(res: MOLPResult) -> Path:
    """
    Radar (spider) chart — current vs WGP target for each inefficient supplier.
    Scores are normalised to [0,1] via ideal aspiration so all axes share a scale.
    """
    suppliers = list(res.supplier_results.keys())
    ns = len(suppliers)
    cats = ASPIRATION_LABELS
    n_cats = len(cats)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, axes = plt.subplots(
        1, ns, figsize=(14, 5.5),
        subplot_kw=dict(projection="polar"),
    )
    if ns == 1:
        axes = [axes]

    for ax, s_name in zip(axes, suppliers):
        r   = res.supplier_results[s_name]
        asp = r.aspirations

        # Normalise: for cost criteria, score = 1 − (val/ideal)·(range_norm)
        # Use "distance from worst" normalisation so higher = better on radar
        worst = np.array([
            5.10, 9.0, 8.0, 7.0, 0.0, 0.0,
        ])
        def norm(v, w, a):
            """Map v ∈ [worst, best=aspiration] → [0,1]."""
            span = abs(a - w)
            if span < 1e-9:
                return 1.0
            return float(abs(v - w) / span)

        cur_norm = [norm(r.current[i], worst[i], asp[i]) for i in range(6)]
        tgt_norm = [norm(r.target[i],  worst[i], asp[i]) for i in range(6)]

        cur_norm += cur_norm[:1]
        tgt_norm += tgt_norm[:1]

        ax.plot(angles, cur_norm, "o-", lw=2,    color=CURRENT_COLOR,
                label="Current", markersize=4)
        ax.fill(angles, cur_norm, alpha=0.15, color=CURRENT_COLOR)
        ax.plot(angles, tgt_norm, "s-", lw=2,    color=TARGET_COLOR,
                label="WGP Target", markersize=4)
        ax.fill(angles, tgt_norm, alpha=0.15, color=TARGET_COLOR)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cats, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.25, 0.50, 0.75, 1.00])
        ax.set_yticklabels(["0.25","0.50","0.75","1.00"], fontsize=6, color="grey")
        ax.set_title(f"Supplier {s_name}", fontsize=11, fontweight="bold",
                     color=INEEFF_COLOR, pad=14)
        ax.tick_params(pad=6)
        ax.grid(color=PALETTE["line"], linewidth=0.6)

    handles = [
        Line2D([0],[0], color=CURRENT_COLOR, lw=2, label="Current"),
        Line2D([0],[0], color=TARGET_COLOR,  lw=2, label="WGP Target"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Figure 10 — Radar Charts: Current vs WGP Target (Normalised [0=Worst, 1=Ideal])",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "fig10_radar_current_vs_target")


# ─────────────────────────────────────────────────────────────────────────────
#  MASTER RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def generate_all(
    mcda: MCDAResult,
    dea:  DEAResult,
    molp: MOLPResult,
) -> Dict[str, Path]:
    """Generate all 10 figures and return a dict {fig_id: path}."""
    print("\n── Generating charts ──────────────────────────────────────────")
    paths = {
        "fig01": fig01_pq_ranking(mcda),
        "fig02": fig02_cs_ranking(mcda),
        "fig03": fig03_pq_cs_scatter(mcda),
        "fig04": fig04_criteria_heatmap(mcda),
        "fig05": fig05_vrs_efficiency(dea),
        "fig06": fig06_cross_efficiency(dea),
        "fig07": fig07_vrs_vs_cross_scatter(dea),
        "fig08": fig08_wgp_gap_analysis(molp),
        "fig09": fig09_residual_deviation_heatmap(molp),
        "fig10": fig10_radar_current_vs_target(molp),
    }
    print(f"\n  All {len(paths)} figures saved to {OUTDIR}/\n")
    return paths

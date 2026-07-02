"""Docking-score analysis — reproduces Section 3.2 / Figure 2.

For each of the 6 pest-relevant proteins, compare the docking-score distribution of active
(pesticides) vs inactive compounds with a Mann-Whitney U test + Benjamini-Hochberg correction.
Five targets show the expected trend (actives bind more strongly, Δ<0); OR28 shows the opposite
(Δ>0), consistent with its chemosensory (repellent) rather than lethal role.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import false_discovery_control, mannwhitneyu

from .dataset import DOCK_COLS, PROTEINS, Dataset


def active_vs_inactive(ds: Dataset) -> dict:
    active = ds.dock[ds.active_mask]
    inactive = ds.dock[ds.inactive_mask]
    metabolites = ds.dock[ds.metabolite_mask]

    rows, pvals = [], []
    for j, col in enumerate(DOCK_COLS):
        a = active[:, j][~np.isnan(active[:, j])]
        i = inactive[:, j][~np.isnan(inactive[:, j])]
        delta = float(np.median(a) - np.median(i))          # <0 => actives bind stronger
        u, p = mannwhitneyu(a, i, alternative="two-sided")
        rows.append({
            "protein": PROTEINS[col]["gene"], "pdb": PROTEINS[col]["pdb"],
            "role": PROTEINS[col]["role"],
            "median_active": round(float(np.median(a)), 3),
            "median_inactive": round(float(np.median(i)), 3),
            "delta": round(delta, 3),
            "median_metabolite": round(float(np.nanmedian(metabolites[:, j])), 3),
            "expected_trend": delta < 0,                     # actives stronger
            "p_value": float(p),
        })
        pvals.append(p)
    p_adj = false_discovery_control(pvals, method="bh")      # Benjamini-Hochberg
    for r, pa in zip(rows, p_adj):
        r["p_adjusted_bh"] = float(pa)
        r["significant"] = bool(pa < 0.001)
    return {
        "per_protein": rows,
        "anomalous_protein": next((r["protein"] for r in rows if not r["expected_trend"]), None),
        "n_active": int(ds.active_mask.sum()),
        "n_inactive": int(ds.inactive_mask.sum()),
        "n_metabolites": int(ds.metabolite_mask.sum()),
        "metabolite_median_range": [
            round(float(np.nanmin([r["median_metabolite"] for r in rows])), 2),
            round(float(np.nanmax([r["median_metabolite"] for r in rows])), 2),
        ],
    }

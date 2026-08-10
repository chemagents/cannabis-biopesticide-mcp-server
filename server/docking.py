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


def metabolite_pesticide_overlap(ds: Dataset, n_bits: int = 2048,
                                 max_metabolites: int = 800) -> dict:
    """Measure how far the C. sativa metabolites sit from the labelled pesticide space.

    ``chemical_space`` used to *assert* that the two spaces overlap, with no number behind it and
    the assertion still returned when the t-SNE figure failed. This computes the claim instead:
    for each metabolite, the ECFP4 Tanimoto nearest neighbour among the labelled compounds and
    whether that neighbour is an active pesticide.
    """
    from rdkit import DataStructs
    from rdkit.Chem import rdMolDescriptors

    rng = np.random.default_rng(0)
    meta_idx = np.flatnonzero(ds.metabolite_mask)
    if meta_idx.size > max_metabolites:
        meta_idx = np.sort(rng.choice(meta_idx, size=max_metabolites, replace=False))
    lab_idx = np.flatnonzero(ds.labelled_mask)
    if lab_idx.size == 0 or meta_idx.size == 0:
        return {}

    def fp(i):
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(ds.mols[i], 2, nBits=n_bits)

    lab_fps = [fp(i) for i in lab_idx]
    lab_active = ds.active_mask[lab_idx]
    nn_sim, nn_is_active = [], []
    for i in meta_idx:
        sims = DataStructs.BulkTanimotoSimilarity(fp(i), lab_fps)
        j = int(np.argmax(sims))
        nn_sim.append(float(sims[j]))
        nn_is_active.append(bool(lab_active[j]))
    nn_sim_a = np.asarray(nn_sim)
    return {
        "n_metabolites": int(meta_idx.size),
        "n_labelled_reference": int(lab_idx.size),
        "frac_nn_active": round(float(np.mean(nn_is_active)), 3),
        "median_nn_similarity": round(float(np.median(nn_sim_a)), 3),
        "frac_nn_similarity_above_0_4": round(float(np.mean(nn_sim_a > 0.4)), 3),
    }


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

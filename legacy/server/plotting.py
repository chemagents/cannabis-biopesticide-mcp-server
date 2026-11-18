"""Figures for the biopesticide reproduction (docking bars, chemical-space t-SNE)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import artifacts, chemistry  # noqa: E402
from .config import get_settings  # noqa: E402
from .dataset import Dataset  # noqa: E402


def plot_docking(per_protein: list[dict]) -> dict:
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r["protein"] for r in per_protein]
    deltas = [r["delta"] for r in per_protein]
    colors = ["#d62728" if d > 0 else "#1f77b4" for d in deltas]
    ax.bar(names, deltas, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Δ median docking (active − inactive), kcal/mol")
    ax.set_title("Docking: actives bind stronger (Δ<0); OR28 opposite (Δ>0)")
    plt.xticks(rotation=20)
    return artifacts.save_figure(fig, "fig2_docking")


def plot_chemical_space(ds: Dataset, sample: int = 1500) -> dict:
    from sklearn.manifold import TSNE

    s = get_settings()
    rng = np.random.default_rng(s.random_state)
    idx = np.sort(rng.choice(ds.n, size=min(sample, ds.n), replace=False))
    X = np.vstack([chemistry.differential_fp(ds.mols[i], s.morgan_radius, s.morgan_nbits) for i in idx])
    emb = TSNE(n_components=2, init="pca", random_state=s.random_state,
              perplexity=min(s.tsne_perplexity, 30)).fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    groups = [("C. sativa metabolite", ds.metabolite_mask, "#2ca02c", 0.5),
              ("pesticide (active)", ds.active_mask, "#d62728", 0.7),
              ("inactive", ds.inactive_mask, "#7f7f7f", 0.4)]
    for label, mask, color, alpha in groups:
        sel = [k for k, i in enumerate(idx) if mask[i]]
        ax.scatter(emb[sel, 0], emb[sel, 1], s=12, c=color, alpha=alpha, label=label, edgecolors="none")
    ax.set_title("Chemical space (differential fingerprint t-SNE): metabolites overlap pesticides")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.legend(fontsize=8)
    return artifacts.save_figure(fig, "figS2_chemical_space")

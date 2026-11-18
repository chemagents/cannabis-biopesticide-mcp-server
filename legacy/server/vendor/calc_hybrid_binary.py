#!/usr/bin/env python3
"""MI-based hybrid ranking and inner CV for binary classification (canpest).

Methods:
  mi        — ranking by mutual_info_classif (no RMT prior)
  hybrid_mi — ranking by MI_i * s_i^alpha (combines MI + RMT prior)

Inner CV m_opt selection: LogisticRegression on top-m sum → ROC AUC.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from calc_hybrid import load_residue_matrix, normalize_ligand_id, rmt_prior_from_train


def mi_by_feature(X: np.ndarray, y: np.ndarray, seed: int = 42) -> np.ndarray:
    """Mutual information between each residue column and binary y."""
    return mutual_info_classif(X, y, random_state=seed, n_neighbors=3)


def build_hybrid_rank_binary(
    X: np.ndarray, y: np.ndarray, alpha: float = 1.0, method: str = "mi", seed: int = 42
) -> dict:
    """Return ranking indices and scores for MI or hybrid_MI method.

    Returns dict with:
      ranking  — argsort descending by score
      scores   — the ranking score per residue
      mi       — raw MI per residue
      s_i      — RMT prior (or None for mi method)
      n_signal — number of signal residues (>lambda_plus)
      lambda_plus — RMT threshold
    """
    n_res = X.shape[1]
    mi = mi_by_feature(X, y, seed)

    if method == "mi":
        scores = mi
        s_i = None
        n_signal = None
        lambda_plus = None
    elif method == "hybrid_mi":
        s_i, lambda_plus, n_signal_full, _ = rmt_prior_from_train(X)
        scores = mi * (s_i ** alpha)
    else:
        raise ValueError(f"Unknown method: {method}")

    ranking = np.argsort(scores)[::-1]
    return {
        "ranking": ranking,
        "scores": scores,
        "mi": mi,
        "s_i": s_i,
        "n_signal": n_signal_full if method == "hybrid_mi" else None,
        "lambda_plus": lambda_plus,
    }


def select_m_inner_cv_binary(
    X: np.ndarray,
    y: np.ndarray,
    ranking: np.ndarray,
    n_residues: int,
    repeats: int = 3,
    inner_train_frac: float = 0.7,
    seed: int = 42,
    max_m: int | None = None,
) -> tuple[int, pd.DataFrame]:
    """Inner CV: LogisticRegression on top-m sum → ROC AUC, pick m with max mean.

    Returns (m_opt, cv_df) where cv_df has columns m, roc_auc (per fold).
    """
    if max_m is None:
        max_m = n_residues
    else:
        max_m = min(max_m, n_residues)

    # Precompute cumulative sums for all m
    sort_idx = ranking[:max_m]
    X_sorted = X[:, sort_idx]
    cumsum = np.cumsum(X_sorted, axis=1)

    skf = StratifiedShuffleSplit(
        n_splits=repeats, train_size=inner_train_frac, random_state=seed
    )

    records = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        for m in range(1, max_m + 1):
            top_m_sum = cumsum[:, m - 1].reshape(-1, 1)
            lr = LogisticRegression(max_iter=1000, solver="lbfgs")
            lr.fit(top_m_sum[tr], y[tr])
            y_prob = lr.predict_proba(top_m_sum[va])[:, 1]
            roc = roc_auc_score(y[va], y_prob)
            records.append({"m": m, "fold": fold, "roc_auc": roc})

    cv_df = pd.DataFrame(records)
    mean_roc = cv_df.groupby("m")["roc_auc"].mean()
    m_opt = int(mean_roc.idxmax())
    return m_opt, cv_df


def fit_protein_hybrid_transform_binary(
    protein: str,
    matrix_df: pd.DataFrame,
    y_train: np.ndarray,
    train_ligands: set,
    alpha: float = 1.0,
    max_m_requested: int | None = None,
    repeats: int = 3,
    inner_train_frac: float = 0.7,
    seed: int = 42,
    method: str = "mi",
) -> tuple[dict, pd.DataFrame, dict]:
    """Full per-protein fit: ranking + inner CV m selection + transform spec.

    Returns (spec_dict, scan_df, extra) where:
      spec_dict — transform_spec.json-compatible dict
      scan_df   — m-scan trace (all folds)
    """
    residue_cols = [c for c in matrix_df.columns if c != "ligand_id" and c != "ligand_id_norm"]
    n_res = len(residue_cols)
    X = matrix_df[residue_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = y_train[: len(matrix_df)]

    # Check data quality
    n_signal_present = max(1, int(n_res * 0.1))
    mask = np.isfinite(X).all(axis=1)
    X, y = X[mask], y[mask]

    rank_result = build_hybrid_rank_binary(X, y, alpha, method, seed)

    if max_m_requested is not None:
        max_m = min(max_m_requested, n_res)
    else:
        max_m = n_res

    m_opt, scan_df = select_m_inner_cv_binary(
        X, y, rank_result["ranking"], n_res,
        repeats=repeats, inner_train_frac=inner_train_frac,
        seed=seed, max_m=max_m,
    )

    # Mean ROC AUC at m_opt
    mean_roc = scan_df.groupby("m")["roc_auc"].mean()
    cv_roc_at_opt = float(mean_roc.loc[m_opt]) if m_opt in mean_roc.index else None

    spec = {
        "protein": protein,
        "method": method,
        "alpha": alpha,
        "m_opt": int(m_opt),
        "n_residue_cols": n_res,
        "n_signal_full_train": int(rank_result["n_signal"]) if rank_result["n_signal"] is not None else None,
        "lambda_plus_full_train": float(rank_result["lambda_plus"]) if rank_result["lambda_plus"] is not None else None,
        "cv_mean_roc_auc_at_m_opt": cv_roc_at_opt,
        "top_residue_indices": rank_result["ranking"][:m_opt].tolist(),
        "n_top_residues": int(m_opt),
    }

    extra = {
        "scores": rank_result["scores"],
        "mi": rank_result["mi"],
        "s_i": rank_result["s_i"],
    }

    return spec, scan_df, extra

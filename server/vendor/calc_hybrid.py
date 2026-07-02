#!/usr/bin/env python3
"""RMT (Random Matrix Theory) math for per-protein hybrid features.

Ported from tox/calc_hybrid.py. See tox/AGENTS.md § Methods for theory.

Functions:
  normalize_ligand_id  — normalise residue-matrix IDs to zfilled 5-digit str
  rmt_prior_from_train — RMT signal s_i per residue (unsupervised)
  abs_spearman_by_feature — |Spearman| per residue (supervised)
  build_hybrid_rank   — hybrid rank = |ρ_i| · s_i^α
  select_m_inner_cv   — choose top-m residues via inner CV
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def normalize_ligand_id(series: pd.Series) -> pd.Series:
    """Convert ligand_00001_tau_1 → 00001 (zfilled 5-digit str)."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace("ligand_", "", regex=False)
        .str.replace("_tau_1", "", regex=False)
        .str.replace("_tau_2", "", regex=False)
        .str.replace("_tau_3", "", regex=False)
    )


def zscore_train(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    sd = xc.std(axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    xz = xc / sd
    return xz, mu, sd


def rmt_prior_from_train(
    x_train: np.ndarray,
) -> tuple[np.ndarray, float, int, float]:
    """RMT signal s_i for each residue.

    s_i = Σ_j λ_j · v_ij²  over signal components λ_j > λ_₊.

    Returns: (s_i array, lambda_plus, n_signal, q)
    """
    xz, _, _ = zscore_train(x_train)
    n, p = xz.shape
    corr = (xz.T @ xz) / (n - 1)

    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    q = p / n
    lambda_plus = float((1.0 + np.sqrt(q)) ** 2)
    signal_mask = eigvals > lambda_plus
    n_signal = int(signal_mask.sum())

    if n_signal == 0:
        signal_mask = np.zeros_like(eigvals, dtype=bool)
        signal_mask[0] = True
        n_signal = 1

    lambdas = eigvals[signal_mask]
    vecs = eigvecs[:, signal_mask]
    s = (vecs * vecs) @ lambdas
    return s, lambda_plus, n_signal, float(q)


def abs_spearman_by_feature(
    x_train: np.ndarray, y_train: np.ndarray
) -> np.ndarray:
    out = np.zeros(x_train.shape[1], dtype=float)
    for i in range(x_train.shape[1]):
        xi = x_train[:, i]
        if np.std(xi) == 0:
            out[i] = 0.0
            continue
        rho = spearmanr(xi, y_train, nan_policy="omit").statistic
        out[i] = 0.0 if np.isnan(rho) else abs(float(rho))
    return out


def build_hybrid_rank(
    x_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0
) -> dict:
    """Hybrid ranking = |ρ_i| · s_i^α.

    Returns dict with keys: ranking, rank_score, s_score, abs_rho, ...
    """
    s, lambda_plus, n_signal, q = rmt_prior_from_train(x_train)
    abs_rho = abs_spearman_by_feature(x_train, y_train)
    s_norm = (s - s.min()) / (s.max() - s.min() + 1e-12)
    rank_score = abs_rho * np.power(s_norm, alpha)
    ranking = np.argsort(rank_score)[::-1]
    return {
        "ranking": ranking,
        "rank_score": rank_score,
        "s_score": s,
        "abs_rho": abs_rho,
        "lambda_plus": lambda_plus,
        "n_signal": n_signal,
        "q": q,
    }


def select_m_inner_cv(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0,
    max_m_requested: int | None = None,
    repeats: int = 3,
    inner_train_frac: float = 0.7,
    seed: int = 42,
) -> tuple[int, pd.DataFrame, int]:
    """Select optimal m via inner CV on train split.

    Returns: (m_opt, scan_df, max_m_eff)
    """
    n, p = x_train.shape
    max_m_eff = p if max_m_requested is None else min(max_m_requested, p)
    if max_m_eff < 1:
        raise ValueError("max_m must be at least 1")

    n_inner = int(n * inner_train_frac)
    if n_inner <= 1 or n_inner >= n:
        raise ValueError("inner_train_frac gives invalid split size")

    rng = np.random.default_rng(seed)
    abs_rho_sum = np.zeros(max_m_eff, dtype=float)
    rho_sum = np.zeros(max_m_eff, dtype=float)
    rho_sq_sum = np.zeros(max_m_eff, dtype=float)
    valid_n = np.zeros(max_m_eff, dtype=int)

    for _ in range(repeats):
        perm = rng.permutation(n)
        tr = perm[:n_inner]
        va = perm[n_inner:]

        x_tr = x_train[tr]
        y_tr_val = y_train[tr]
        x_va = x_train[va]
        y_va = y_train[va]

        rank = build_hybrid_rank(x_tr, y_tr_val, alpha)["ranking"]
        top = rank[:max_m_eff]

        score_mat = np.cumsum(x_va[:, top], axis=1)
        for m_idx in range(max_m_eff):
            score = score_mat[:, m_idx]
            if np.std(score) == 0:
                continue
            rho = spearmanr(score, y_va, nan_policy="omit").statistic
            if np.isnan(rho):
                continue
            abs_rho_sum[m_idx] += abs(float(rho))
            rho_sum[m_idx] += float(rho)
            rho_sq_sum[m_idx] += float(rho) * float(rho)
            valid_n[m_idx] += 1

    mean_abs = np.divide(
        abs_rho_sum, valid_n, out=np.full(max_m_eff, np.nan), where=valid_n > 0
    )
    mean_rho = np.divide(
        rho_sum, valid_n, out=np.full(max_m_eff, np.nan), where=valid_n > 0
    )
    mean_sq = np.divide(
        rho_sq_sum, valid_n, out=np.full(max_m_eff, np.nan), where=valid_n > 0
    )
    std_rho = np.sqrt(np.clip(mean_sq - mean_rho * mean_rho, a_min=0.0, a_max=None))

    if np.all(np.isnan(mean_abs)):
        m_opt = 1
    else:
        m_opt = int(np.nanargmax(mean_abs) + 1)

    scan_df = pd.DataFrame(
        {
            "m": np.arange(1, max_m_eff + 1),
            "mean_abs_spearman": mean_abs,
            "mean_spearman": mean_rho,
            "std_spearman": std_rho,
            "n_valid_splits": valid_n,
        }
    )
    return m_opt, scan_df, max_m_eff


def fit_protein_hybrid_transform(
    protein: str,
    matrix_df: pd.DataFrame,
    y_train: np.ndarray,
    train_ligands: set[str],
    alpha: float = 1.0,
    max_m_requested: int | None = None,
    repeats: int = 3,
    inner_train_frac: float = 0.7,
    seed: int = 42,
    method: str = "hybrid",
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """Fit per-protein hybrid or s_only transform on train ligands.

    method='hybrid' uses |ρ_i| · s_i^α ranking.
    method='s_only' uses RMT s_i ranking only (no supervised term).

    Returns: (spec_dict, scan_df, train_scores_array)
    """
    residue_cols = [c for c in matrix_df.columns if c != "ligand_id"]
    if not residue_cols:
        raise ValueError(f"No residue columns for protein {protein}")

    work = matrix_df.copy()
    work["ligand_id_norm"] = normalize_ligand_id(work["ligand_id"])
    work = work[work["ligand_id_norm"].isin(train_ligands)]

    if len(work) < 10:
        raise ValueError(f"Too few train rows for protein {protein}: {len(work)}")

    work = work.sort_values("ligand_id_norm").reset_index(drop=True)
    x_arr = (
        work[residue_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    y_arr = y_train.copy()

    if method == "hybrid":
        m_opt, scan_df, max_m_eff = select_m_inner_cv(
            x_train=x_arr,
            y_train=y_arr,
            alpha=alpha,
            max_m_requested=max_m_requested,
            repeats=repeats,
            inner_train_frac=inner_train_frac,
            seed=seed,
        )
        final_rank = build_hybrid_rank(x_arr, y_arr, alpha)
        top_idx = final_rank["ranking"][:m_opt]
        ranking_desc = "|rho_i| * s_i^alpha"
    else:
        s_i, lambda_plus, n_signal, q = rmt_prior_from_train(x_arr)
        ranking = np.argsort(s_i)[::-1]
        m_opt, scan_df, max_m_eff = select_m_inner_cv(
            x_train=x_arr,
            y_train=y_arr,
            alpha=0.0,
            max_m_requested=max_m_requested,
            repeats=repeats,
            inner_train_frac=inner_train_frac,
            seed=seed,
        )
        top_idx = ranking[:m_opt]
        final_rank = {
            "ranking": ranking,
            "s_score": s_i,
            "lambda_plus": lambda_plus,
            "n_signal": n_signal,
            "q": q,
        }
        ranking_desc = "s_i (RMT prior only)"

    top_cols = [residue_cols[i] for i in top_idx]
    train_score = x_arr[:, top_idx].sum(axis=1)

    best_row = scan_df.loc[scan_df["m"] == m_opt]
    cv_val = (
        None
        if best_row.empty or pd.isna(best_row["mean_abs_spearman"].iloc[0])
        else float(best_row["mean_abs_spearman"].iloc[0])
    )

    spec = {
        "protein": protein,
        "method": method,
        "alpha": alpha,
        "n_train_rows": int(len(work)),
        "n_residue_cols": int(len(residue_cols)),
        "max_m_requested": None if max_m_requested is None else int(max_m_requested),
        "max_m_used": int(max_m_eff),
        "m_opt": int(m_opt),
        "cv_mean_abs_spearman_at_m_opt": cv_val,
        "lambda_plus_full_train": float(final_rank["lambda_plus"]),
        "n_signal_full_train": int(final_rank["n_signal"]),
        "q_full_train": float(final_rank["q"]),
        "top_residue_cols": top_cols,
        "ranking_description": ranking_desc,
    }

    return spec, scan_df, train_score


def load_residue_matrix(protein_dir: Path, protein_key: str) -> pd.DataFrame:
    """Load the residue matrix CSV for a given protein key.

    Scans protein_dir for files matching the protein key.
    """
    for f in protein_dir.glob("*_residue_matrix.csv"):
        stem = f.stem.replace("_residue_matrix", "")
        if protein_key.lower() in stem.lower():
            return pd.read_csv(f)
    available = [f.stem.replace("_residue_matrix", "") for f in protein_dir.glob("*_residue_matrix.csv")]
    raise FileNotFoundError(
        f"No residue matrix for '{protein_key}' in {protein_dir}. "
        f"Available: {available}"
    )

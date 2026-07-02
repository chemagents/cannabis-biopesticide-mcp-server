"""Random Matrix Theory feature selection — reproduces Section 3.3 (rmt_filter.py port).

Given the 390 residue-term-energy (RTE) features on the training split:

  1. train-only median-impute + z-score;
  2. eigendecompose the feature correlation matrix;
  3. Marchenko-Pastur threshold  lambda_+ = (1 + sqrt(q))^2 ,  q = n_features / n_rows ,
     keeps only signal eigenvalues (lambda_i > lambda_+);
  4. RMT prior  s_i = sum_j  lambda_j * v_ij^2  over signal eigenvectors;
  5. hybrid ranking  rank_i = s_i * |Spearman(feature_i, activity)| ;
  6. m_opt chosen by inner cross-validation (ROC-AUC of a logistic model on the top-m columns);
  7. produce `rmt_rte_sel` (top-m raw columns) and `rmt_rte_rec` (signal-subspace reconstruction).

The deterministic RMT numbers (lambda_+, q, n_signal) reproduce the paper's values exactly;
m_opt is inner-CV-dependent and reproduced within tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from .config import get_settings


@dataclass
class RMTResult:
    m_opt: int
    lambda_plus: float
    q: float
    n_signal: int
    order: np.ndarray             # feature indices, best-first
    selected_idx: np.ndarray      # top-m_opt indices
    inner_mean_auc: float
    inner_std_auc: float
    rank_scores: np.ndarray


def _zscore_fit(X: np.ndarray):
    mean = np.nanmean(X, axis=0)
    filled = np.where(np.isnan(X), mean, X)
    scale = filled.std(axis=0, ddof=0)
    scale[scale < 1e-12] = 1.0
    return mean, scale


def _zscore_apply(X: np.ndarray, mean, scale) -> np.ndarray:
    return (np.where(np.isnan(X), mean, X) - mean) / scale


def rmt_prior(Xz: np.ndarray):
    """RMT signal separation on a z-scored (n_rows x n_cols) matrix."""
    n_rows, n_cols = Xz.shape
    corr = (Xz.T @ Xz) / (n_rows - 1)
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    q = n_cols / n_rows
    lambda_plus = (1.0 + np.sqrt(q)) ** 2
    signal = eigvals > lambda_plus
    if not signal.any():
        signal[0] = True
    s_i = (eigvecs[:, signal] ** 2) @ eigvals[signal]      # RMT prior per feature
    return s_i, float(lambda_plus), float(q), int(signal.sum()), eigvecs, eigvals, signal


def hybrid_ranking(s_i: np.ndarray, Xz: np.ndarray, y: np.ndarray) -> np.ndarray:
    rho = np.array([abs(spearmanr(Xz[:, j], y).statistic) for j in range(Xz.shape[1])])
    rho = np.nan_to_num(rho)
    return s_i * rho                                       # rank_by = "hybrid"


def _select_m(Xz: np.ndarray, y: np.ndarray, order: np.ndarray, settings) -> tuple[int, float, float]:
    """Inner CV: pick m maximising ROC-AUC of a logistic model on the top-m columns."""
    n_feat = Xz.shape[1]
    grid = sorted(set(int(m) for m in np.unique(
        np.clip(np.round(np.linspace(2, n_feat, 60)).astype(int), 2, n_feat))))
    sss = StratifiedShuffleSplit(n_splits=settings.rmt_inner_splits,
                                 train_size=settings.rmt_inner_train_frac,
                                 random_state=settings.random_state)
    scores = {m: [] for m in grid}
    for tr, va in sss.split(Xz, y):
        for m in grid:
            cols = order[:m]
            clf = LogisticRegression(max_iter=300, C=1.0)
            clf.fit(Xz[np.ix_(tr, cols)], y[tr])
            p = clf.predict_proba(Xz[np.ix_(va, cols)])[:, 1]
            scores[m].append(roc_auc_score(y[va], p))
    means = {m: float(np.mean(v)) for m, v in scores.items()}
    m_opt = max(means, key=means.get)
    return m_opt, means[m_opt], float(np.std(scores[m_opt]))


def rmt_filter(X_train: np.ndarray, y_train: np.ndarray) -> RMTResult:
    settings = get_settings()
    mean, scale = _zscore_fit(X_train)
    Xz = _zscore_apply(X_train, mean, scale)
    s_i, lam, q, n_signal, _, _, _ = rmt_prior(Xz)
    rank = hybrid_ranking(s_i, Xz, y_train)
    order = np.argsort(rank)[::-1]
    m_opt, auc_mean, auc_std = _select_m(Xz, y_train, order, settings)
    return RMTResult(m_opt=m_opt, lambda_plus=lam, q=q, n_signal=n_signal, order=order,
                     selected_idx=order[:m_opt], inner_mean_auc=auc_mean,
                     inner_std_auc=auc_std, rank_scores=rank)


def reconstruct_signal(X: np.ndarray, mean, scale, eigvecs, signal_mask) -> np.ndarray:
    """Project onto the signal eigen-subspace and invert the z-score (rmt_rte_rec)."""
    Xz = _zscore_apply(X, mean, scale)
    v = eigvecs[:, signal_mask]
    return (Xz @ v) @ v.T * scale + mean

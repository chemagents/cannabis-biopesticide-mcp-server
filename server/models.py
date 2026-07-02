"""QSAR models — reproduces Sections 3.3-3.4.

The paper's main model is a DMPNN (graph neural net) + GBM blend (ROC-AUC ~0.93). The open,
torch-free analogue here is **HistGradientBoosting on RDKit-2D descriptors** (+ optional
docking / RMT-RTE features), which the authors' own ablation shows reaches ROC-AUC ~0.93
(rdkit2d 0.9343, rdkit2d_rmt_rec 0.9415). CB-SD = gradient boosting on the 6 docking scores
(the docking-only baseline, ROC-AUC ~0.68-0.80). An applicability-domain score (kNN k=5 +
Gaussian) gives the per-compound "Probability" used to pick the >0.7 candidate set.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit.Chem import Descriptors
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from . import rmt
from .config import get_settings
from .dataset import DOCK_COLS, Dataset, load_rte, load_split

_DESC = [(n, f) for n, f in Descriptors._descList]      # ~210 RDKit 2D descriptors


@lru_cache(maxsize=1)
def rdkit2d_bundled() -> np.ndarray:
    """The authors' exact 217 RDKit2D descriptors (aligned to the dataset rows), NaN-imputed.

    Falls back to on-the-fly RDKit descriptors if the bundled matrix is absent or misaligned.
    """
    ds = __import__("server.dataset", fromlist=["load_dataset"]).load_dataset()
    path = Path(get_settings().fp_rdkit2d_path)
    if path.exists():
        X = np.load(path).astype(float)
        if X.shape[0] != ds.n:
            X = rdkit2d_matrix(ds.mols)
    else:
        X = rdkit2d_matrix(ds.mols)
    col = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
    col = np.where(np.isfinite(col), col, 0.0)
    return np.where(np.isfinite(X), X, col)


def _iid(x) -> int:
    """Normalize a ligand_id to int (data.csv uses '01349', splits use '1349')."""
    return int(str(x).strip())


def rdkit2d_matrix(mols) -> np.ndarray:
    X = np.array([[f(m) for _, f in _DESC] for m in mols], dtype=float)
    X[~np.isfinite(X)] = np.nan
    col_mean = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    return np.where(np.isfinite(X), X, col_mean)


def _hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, validation_fraction=0.1,
        early_stopping=True, random_state=get_settings().random_state)


def _metrics(y, p) -> dict:
    return {"roc_auc": round(float(roc_auc_score(y, p)), 4),
            "pr_auc": round(float(average_precision_score(y, p)), 4)}


def _fit_predict(Xtr, ytr, Xte) -> np.ndarray:
    clf = _hgb()
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


# --------------------------------------------------------------------------- #
# Ablation over feature sets on one split (Section 3.3 / README ablation).
# --------------------------------------------------------------------------- #
def _aligned(ds: Dataset, ids: list[str]):
    idx = {lid: i for i, lid in enumerate(ds.ligand_ids)}
    rows = [idx[i] for i in ids if i in idx]
    return rows


def qsar_ablation(scaffold: bool = False, split_id: str | None = None) -> dict:
    ds_mod = __import__("server.dataset", fromlist=["load_dataset"])
    ds = ds_mod.load_dataset()
    split = load_split(scaffold=scaffold)
    split_id = split_id or ("scaffold_00" if scaffold else "split_00")
    s = split[split.split_id == split_id]
    tr_ids = s[s.set == "train"]["ligand_id"].tolist()
    te_ids = s[s.set == "test"]["ligand_id"].tolist()

    rte_df, feat_cols = load_rte()
    rte_idx = {_iid(lid): i for i, lid in enumerate(rte_df["ligand_id"].tolist())}
    rte_arr = rte_df[feat_cols].to_numpy(float)
    dmap = {_iid(lid): i for i, lid in enumerate(ds.ligand_ids)}
    yq = {_iid(lid): float(a) for lid, a in zip(s.ligand_id, s.activity)}

    tr = [_iid(i) for i in tr_ids if _iid(i) in rte_idx and _iid(i) in dmap]
    te = [_iid(i) for i in te_ids if _iid(i) in rte_idx and _iid(i) in dmap]
    y_tr = np.array([yq[i] for i in tr]); y_te = np.array([yq[i] for i in te])

    # RDKit2D (authors' exact 217-descriptor matrix, aligned to dataset rows)
    X2d_all = rdkit2d_bundled()
    X2d_tr = X2d_all[[dmap[i] for i in tr]]; X2d_te = X2d_all[[dmap[i] for i in te]]
    dock_tr = ds.dock[[dmap[i] for i in tr]]; dock_te = ds.dock[[dmap[i] for i in te]]

    # RTE + RMT transforms
    rt_tr = np.array([rte_arr[rte_idx[i]] for i in tr])
    rt_te = np.array([rte_arr[rte_idx[i]] for i in te])
    res = rmt.rmt_filter(rt_tr, y_tr)
    sel = res.selected_idx
    mean, scale = rmt._zscore_fit(rt_tr)
    _, _, _, _, eigvecs, _, signal = rmt.rmt_prior(rmt._zscore_apply(rt_tr, mean, scale))
    rec_tr = rmt.reconstruct_signal(rt_tr, mean, scale, eigvecs, signal)
    rec_te = rmt.reconstruct_signal(rt_te, mean, scale, eigvecs, signal)

    d2 = [DOCK_COLS.index("d8v7j0"), DOCK_COLS.index("3rif")]   # dock2 = ACHE, GLC1
    variants = {
        "rdkit2d": (X2d_tr, X2d_te),
        "rdkit2d_dock2": (np.hstack([X2d_tr, dock_tr[:, d2]]), np.hstack([X2d_te, dock_te[:, d2]])),
        "rdkit2d_raw_rte": (np.hstack([X2d_tr, rt_tr]), np.hstack([X2d_te, rt_te])),
        "rdkit2d_rmt_sel": (np.hstack([X2d_tr, rt_tr[:, sel]]), np.hstack([X2d_te, rt_te[:, sel]])),
        "rdkit2d_rmt_rec": (np.hstack([X2d_tr, rec_tr[:, sel]]), np.hstack([X2d_te, rec_te[:, sel]])),
    }
    out = {}
    for name, (Xtr, Xte) in variants.items():
        p = _fit_predict(Xtr, y_tr, Xte)
        out[name] = _metrics(y_te, p)
    return {"split_id": split_id, "m_opt": res.m_opt, "n_signal": res.n_signal,
            "lambda_plus": round(res.lambda_plus, 4), "results": out,
            "n_train": len(tr), "n_test": len(te)}


def rmt_selection(scaffold: bool = False, split_id: str | None = None) -> dict:
    """Run the RMT filter on a split's training RTE features (Section 3.3)."""
    ds = __import__("server.dataset", fromlist=["load_dataset"]).load_dataset()
    split = load_split(scaffold=scaffold)
    split_id = split_id or ("scaffold_00" if scaffold else "split_00")
    s = split[split.split_id == split_id]
    rte_df, feat_cols = load_rte()
    rte_idx = {_iid(lid): i for i, lid in enumerate(rte_df["ligand_id"].tolist())}
    rte_arr = rte_df[feat_cols].to_numpy(float)
    yq = {_iid(lid): float(a) for lid, a in zip(s.ligand_id, s.activity)}
    tr = [_iid(i) for i in s[s.set == "train"].ligand_id if _iid(i) in rte_idx]
    X = np.array([rte_arr[rte_idx[i]] for i in tr]); y = np.array([yq[i] for i in tr])
    res = rmt.rmt_filter(X, y)
    top = [{"feature": feat_cols[j], "rank_score": round(float(res.rank_scores[j]), 4)}
           for j in res.order[:15]]
    return {"split_id": split_id, "lambda_plus": round(res.lambda_plus, 4), "q": round(res.q, 4),
            "n_signal": res.n_signal, "m_opt": res.m_opt,
            "inner_auc": round(res.inner_mean_auc, 4), "inner_std": round(res.inner_std_auc, 4),
            "n_features_total": len(feat_cols), "top_features": top}


def cb_sd_docking(scaffold: bool = False) -> dict:
    """Docking-only baseline (CB-SD analogue): gradient boosting on the 6 docking scores."""
    ds = __import__("server.dataset", fromlist=["load_dataset"]).load_dataset()
    split = load_split(scaffold=scaffold)
    sid = "scaffold_00" if scaffold else "split_00"
    s = split[split.split_id == sid]
    dmap = {_iid(lid): i for i, lid in enumerate(ds.ligand_ids)}
    tr = [_iid(i) for i in s[s.set == "train"].ligand_id if _iid(i) in dmap]
    te = [_iid(i) for i in s[s.set == "test"].ligand_id if _iid(i) in dmap]
    yq = {_iid(lid): float(a) for lid, a in zip(s.ligand_id, s.activity)}
    y_tr = np.array([yq[i] for i in tr]); y_te = np.array([yq[i] for i in te])
    Xtr = ds.dock[[dmap[i] for i in tr]]; Xte = ds.dock[[dmap[i] for i in te]]
    p = _fit_predict(np.nan_to_num(Xtr), y_tr, np.nan_to_num(Xte))
    return {"all6_docking": _metrics(y_te, p), "n_train": len(tr), "n_test": len(te)}


def _build_split(scaffold: bool, split_id: str | None):
    """Aligned train/test arrays for a split: RDKit2D, RTE, RMT sel/rec, and labels."""
    ds = __import__("server.dataset", fromlist=["load_dataset"]).load_dataset()
    split = load_split(scaffold=scaffold)
    split_id = split_id or ("scaffold_00" if scaffold else "split_00")
    s = split[split.split_id == split_id]
    rte_df, feat_cols = load_rte()
    rte_idx = {_iid(l): i for i, l in enumerate(rte_df["ligand_id"].tolist())}
    rte_arr = rte_df[feat_cols].to_numpy(float)
    dmap = {_iid(l): i for i, l in enumerate(ds.ligand_ids)}
    yq = {_iid(l): float(a) for l, a in zip(s.ligand_id, s.activity)}
    tr = [_iid(i) for i in s[s.set == "train"].ligand_id if _iid(i) in rte_idx and _iid(i) in dmap]
    te = [_iid(i) for i in s[s.set == "test"].ligand_id if _iid(i) in rte_idx and _iid(i) in dmap]
    y_tr = np.array([yq[i] for i in tr]); y_te = np.array([yq[i] for i in te])
    X2d = rdkit2d_bundled()
    rt_tr = np.array([rte_arr[rte_idx[i]] for i in tr]); rt_te = np.array([rte_arr[rte_idx[i]] for i in te])
    res = rmt.rmt_filter(rt_tr, y_tr)
    mean, scale = rmt._zscore_fit(rt_tr)
    _, _, _, _, eigvecs, _, signal = rmt.rmt_prior(rmt._zscore_apply(rt_tr, mean, scale))
    rec_tr = rmt.reconstruct_signal(rt_tr, mean, scale, eigvecs, signal)[:, res.selected_idx]
    rec_te = rmt.reconstruct_signal(rt_te, mean, scale, eigvecs, signal)[:, res.selected_idx]
    return {"X2d_tr": X2d[[dmap[i] for i in tr]], "X2d_te": X2d[[dmap[i] for i in te]],
            "rt_tr": rt_tr, "rt_te": rt_te, "rec_tr": rec_tr, "rec_te": rec_te,
            "y_tr": y_tr, "y_te": y_te, "m_opt": res.m_opt}


def cb_sd_rte(scaffold: bool = False) -> dict:
    """CB-SD analogue: gradient boosting on the 390 residue-term energies (paper ALL6 ~0.802)."""
    d = _build_split(scaffold, None)
    p = _fit_predict(np.nan_to_num(d["rt_tr"]), d["y_tr"], np.nan_to_num(d["rt_te"]))
    return {"all6_rte": _metrics(d["y_te"], p), "n_train": len(d["y_tr"]), "n_test": len(d["y_te"])}


def docking_veto(scaffold: bool = False, threshold: float = 0.5) -> dict:
    """Docking-consistency veto (Section 3.3): p_final = p_QSAR × p_RMT-RTE reduces false positives.

    The structure model (RDKit2D HGB, DMPNN-SD analogue) gives p_qsar; an independent
    RMT-RTE model gives p_rmt; their product can only *lower* the probability (a soft physical
    consistency filter), cutting the false-positive rate at the 0.5 operating point.
    """
    d = _build_split(scaffold, None)
    p_qsar = _fit_predict(d["X2d_tr"], d["y_tr"], d["X2d_te"])
    p_rmt = _fit_predict(np.nan_to_num(d["rec_tr"]), d["y_tr"], np.nan_to_num(d["rec_te"]))
    p_final = p_qsar * p_rmt
    y = d["y_te"]; neg = y == 0; pos = y == 1

    def fpr(p):
        return float(((p[neg] >= threshold).sum()) / max(neg.sum(), 1))

    def recall(p):
        return float(((p[pos] >= threshold).sum()) / max(pos.sum(), 1))
    return {
        "roc_auc_qsar": round(float(roc_auc_score(y, p_qsar)), 4),
        "roc_auc_after_veto": round(float(roc_auc_score(y, p_final)), 4),
        "fpr_before": round(fpr(p_qsar), 4), "fpr_after_veto": round(fpr(p_final), 4),
        "fpr_reduction_pct": round(100 * (fpr(p_qsar) - fpr(p_final)) / max(fpr(p_qsar), 1e-9), 1),
        "recall_before": round(recall(p_qsar), 4), "recall_after_veto": round(recall(p_final), 4),
    }


# --------------------------------------------------------------------------- #
# Applicability domain + biopesticide prediction (Section 3.4).
# --------------------------------------------------------------------------- #
def _tanimoto_ad(train_fp: np.ndarray, query_fp: np.ndarray, k: int, thr: float) -> np.ndarray:
    q_cnt = query_fp.sum(1); t_cnt = train_fp.sum(1)
    inter = query_fp @ train_fp.T
    union = q_cnt[:, None] + t_cnt[None, :] - inter
    sim = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    d = 1.0 - sim
    kk = min(k, d.shape[1])
    nn = np.partition(d, kk - 1, axis=1)[:, :kk].mean(1)
    return np.clip(np.exp(-((nn / max(thr, 1e-9)) ** 2)), 0, 1)


def predict_biopesticides() -> dict:
    """Train on all labelled compounds, predict the unlabelled C. sativa metabolites."""
    from . import chemistry
    ds = __import__("server.dataset", fromlist=["load_dataset"]).load_dataset()
    settings = get_settings()
    lab = ds.labelled_mask; meta = ds.metabolite_mask
    X = rdkit2d_bundled()
    clf = _hgb(); clf.fit(X[lab], ds.activity[lab])
    proba = clf.predict_proba(X[meta])[:, 1]

    fp = lambda i: chemistry.morgan_fp(ds.mols[i], settings.morgan_radius, settings.morgan_nbits)
    train_fp = np.vstack([fp(i) for i in np.where(lab)[0]]).astype(np.uint8)
    meta_fp = np.vstack([fp(i) for i in np.where(meta)[0]]).astype(np.uint8)
    thr = _ad_threshold(train_fp, settings)
    ad = _tanimoto_ad(train_fp, meta_fp, settings.ad_k_neighbors, thr)

    cut = settings.candidate_probability
    in_domain = ad >= 0.5
    candidates = int(((proba >= cut) & in_domain).sum())
    return {
        "n_metabolites": int(meta.sum()),
        "candidates_prob_gt_0.7": candidates,
        "candidate_fraction": round(candidates / int(meta.sum()), 4),
        "n_outside_ad": int((~in_domain).sum()),
        "outside_ad_fraction": round(int((~in_domain).sum()) / int(meta.sum()), 4),
        "probability_cutoff": cut,
    }


def _ad_threshold(train: np.ndarray, settings, sample: int = 500) -> float:
    rng = np.random.default_rng(0)
    idx = rng.choice(train.shape[0], size=min(sample, train.shape[0]), replace=False)
    sub = train[idx]
    inter = sub @ train.T
    union = sub.sum(1)[:, None] + train.sum(1)[None, :] - inter
    sim = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    d = 1.0 - sim
    nn = np.partition(d, settings.ad_k_neighbors, axis=1)[:, 1:settings.ad_k_neighbors + 1].mean(1)
    return float(nn.mean() + settings.ad_threshold_sigma * nn.std())

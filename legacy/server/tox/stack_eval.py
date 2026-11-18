#!/usr/bin/env python
"""Stacking eval for the tox endpoints: blend CatBoost-FP + XGBoost-frag, exactly like the
cannabis DMPNN+HGB soft-voting stack. Collect out-of-fold (OOF) predictions from both models on
identical folds, then blend p = w*p_cat + (1-w)*p_xgb. Report the honest no-tuning blend@0.5 AND
the OOF-optimal weight (mildly optimistic, same convention as the cannabis blend_w).
"""
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error
import tox_train as T


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def metric(y, p, task):
    return float(roc_auc_score(y, p)) if task == "classification" else _rmse(y, p)


def oof(csv, task):
    mols, y, n_bad, n_total = T.load_dataset(csv)
    y = np.asarray(y, dtype=float)
    Xfp, Xfr = T.featurize_ecfp(mols), T.featurize_fragments(mols)
    n = len(mols)
    p_cat, p_xgb = np.zeros(n), np.zeros(n)
    if task == "classification":
        sp = StratifiedKFold(T.N_SPLITS, shuffle=True, random_state=T.SEED).split(Xfp, y)
    else:
        sp = KFold(T.N_SPLITS, shuffle=True, random_state=T.SEED).split(Xfp)
    for tr, te in sp:
        cb = T.make_catboost(task); cb.fit(Xfp[tr], y[tr])
        xb = T.make_xgboost(task); xb.fit(Xfr[tr], y[tr])
        if task == "classification":
            p_cat[te] = cb.predict_proba(Xfp[te])[:, 1]
            p_xgb[te] = xb.predict_proba(Xfr[te])[:, 1]
        else:
            p_cat[te] = cb.predict(Xfp[te])
            p_xgb[te] = xb.predict(Xfr[te])
    return y, p_cat, p_xgb, n


def run(csv, task, name):
    y, pc, px, n = oof(csv, task)
    mc, mx = metric(y, pc, task), metric(y, px, task)
    m_half = metric(y, 0.5 * pc + 0.5 * px, task)
    grid = [(w, metric(y, w * pc + (1 - w) * px, task)) for w in np.linspace(0, 1, 41)]
    pick = max if task == "classification" else min          # ROC-AUC up, RMSE down
    w_opt, m_opt = pick(grid, key=lambda t: t[1])
    best_single = (max if task == "classification" else min)(mc, mx)
    lift = (m_opt - best_single) if task == "classification" else (best_single - m_opt)
    up = "↑" if task == "classification" else "↓"
    print(f"{name:13s} [{task[:4]} {up}] n={n:5d}  CatBoost={mc:.4f}  XGBoost={mx:.4f}  "
          f"blend@0.5={m_half:.4f}  blend*={m_opt:.4f}(w_cat={w_opt:.2f})  "
          f"lift_vs_best={lift:+.4f}")
    return dict(name=name, task=task, n=n, catboost=mc, xgboost=mx,
                blend_half=m_half, blend_opt=m_opt, w_cat=w_opt, lift=lift)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])

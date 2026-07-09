#!/usr/bin/env python
"""'Our best models': a richer ensemble than the paper's 2-model recipe. Diverse base learners
(CatBoost / XGBoost / LightGBM / ExtraTrees) over three feature views (ECFP, fragment panel, full
RDKit descriptors), combined by a CROSS-VALIDATED meta-learner (leak-free 2-level stack). Reports
each base, the best single, a simple mean blend, and the meta-stack.
"""
import sys
import warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from rdkit.Chem import Descriptors
import tox_train as T

warnings.filterwarnings("ignore")
SEED, K = 42, 5
_DESCS = [(n, f) for n, f in Descriptors._descList]


def feat_desc(mols):
    X = np.zeros((len(mols), len(_DESCS)), dtype=np.float32)
    for i, m in enumerate(mols):
        for j, (_, f) in enumerate(_DESCS):
            try:
                X[i, j] = f(m)
            except Exception:
                X[i, j] = 0.0
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def score(y, p, task):
    return float(roc_auc_score(y, p)) if task == "classification" else rmse(y, p)


def bases(task):
    C = task == "classification"
    def cat(**k): return (CatBoostClassifier if C else CatBoostRegressor)(random_seed=SEED, verbose=0, thread_count=-1, **k)
    def xgb(**k): return (XGBClassifier if C else XGBRegressor)(random_state=SEED, n_jobs=-1, tree_method="hist", **k)
    def lgbm(**k): return (LGBMClassifier if C else LGBMRegressor)(random_state=SEED, n_jobs=-1, verbose=-1, **k)
    def xt(**k): return (ExtraTreesClassifier if C else ExtraTreesRegressor)(random_state=SEED, n_jobs=-1, **k)
    return [
        ("cat_ecfp", "ecfp", lambda: cat(iterations=1200, depth=6, learning_rate=0.05)),
        ("xgb_frag", "frag", lambda: xgb(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)),
        ("lgbm_ecfp", "ecfp", lambda: lgbm(n_estimators=900, num_leaves=64, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8)),
        ("cat_desc", "desc", lambda: cat(iterations=1000, depth=6, learning_rate=0.05)),
        ("xt_desc", "desc", lambda: xt(n_estimators=800)),
    ]


def run(csv, task, name):
    mols, y, nb, nt = T.load_dataset(csv)
    y = np.asarray(y, dtype=float)
    n = len(mols)
    F = {"ecfp": T.featurize_ecfp(mols), "frag": T.featurize_fragments(mols), "desc": feat_desc(mols)}
    if task == "classification":
        folds = list(StratifiedKFold(K, shuffle=True, random_state=SEED).split(F["ecfp"], y))
    else:
        folds = list(KFold(K, shuffle=True, random_state=SEED).split(F["ecfp"]))
    B = bases(task)
    oof = {}
    for nm, fk, fac in B:
        X = F[fk]; p = np.zeros(n)
        for tr, te in folds:
            m = fac(); m.fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1] if task == "classification" else m.predict(X[te])
        oof[nm] = p
    bs = {nm: score(y, oof[nm], task) for nm in oof}
    best = (min if task == "regression" else max)(bs, key=bs.get)
    P = np.column_stack([oof[nm] for nm, _, _ in B])
    mean_blend = score(y, P.mean(1), task)
    if task == "classification":
        meta = cross_val_predict(LogisticRegression(max_iter=1000), P, y, cv=K, method="predict_proba")[:, 1]
    else:
        meta = cross_val_predict(Ridge(alpha=1.0), P, y, cv=K)
    meta_s = score(y, meta, task)
    up = "↑" if task == "classification" else "↓"
    print(f"=== {name} [{task} {up}] n={n} ===")
    for nm in oof:
        print(f"   {nm:10s} {bs[nm]:.4f}")
    print(f"   -> best_single={bs[best]:.4f} ({best})  mean_blend={mean_blend:.4f}  META_STACK={meta_s:.4f}")
    return meta_s


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])

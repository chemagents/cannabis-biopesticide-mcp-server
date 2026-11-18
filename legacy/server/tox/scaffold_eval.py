#!/usr/bin/env python
"""Scaffold-split evaluation of the v2 ensemble — the honest generalization number for deploying on
chemically-distinct molecules (Cannabis metabolites). Bemis-Murcko scaffolds are the CV groups
(StratifiedGroupKFold for classification, GroupKFold for regression), so no scaffold spans
train/test. Compare against the random-split numbers.
"""
import sys
import warnings
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression, Ridge
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
import tox_train as T
from stack_v2 import feat_desc, bases, score, SEED, K

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


def scaffolds(mols):
    g = []
    for m in mols:
        sc = ""
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:
            sc = ""
        g.append(sc if sc else Chem.MolToSmiles(m))   # acyclic -> its own group
    return np.array(g)


def run(csv, task, name):
    mols, y, _, _ = T.load_dataset(csv)
    y = np.asarray(y, dtype=float)
    n = len(mols)
    groups = scaffolds(mols)
    F = {"ecfp": T.featurize_ecfp(mols), "frag": T.featurize_fragments(mols), "desc": feat_desc(mols)}
    if task == "classification":
        folds = list(StratifiedGroupKFold(K, shuffle=True, random_state=SEED).split(F["ecfp"], y, groups))
    else:
        folds = list(GroupKFold(K).split(F["ecfp"], y, groups))
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
    print(f"=== {name} [{task} {up}] n={n}  scaffolds={len(set(groups))} ===")
    for nm in oof:
        print(f"   {nm:10s} {bs[nm]:.4f}")
    print(f"   -> best={bs[best]:.4f} ({best})  mean_blend={mean_blend:.4f}  META={meta_s:.4f}")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])

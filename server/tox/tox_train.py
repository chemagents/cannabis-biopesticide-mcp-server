#!/usr/bin/env python
"""
Reusable trainer reproducing the Syntelly open toxicity-model recipe
(Sosnin/Shkil et al., Molecules 2024, 29, 1826, section 4.3).

Per endpoint, train TWO gradient-boosting models:
  (a) CatBoost on molecular fingerprints (Morgan/ECFP, 2048 bits)
  (b) XGBoost on molecular fragment descriptors
      (RDKit fr_* counts + TPSA, MolLogP, LabuteASA, Kappa1/2/3,
       SlogP_VSA*, SMR_VSA*, EState_VSA*)

Evaluation: 5-fold CV.
  classification -> StratifiedKFold, mean ROC-AUC
  regression     -> KFold, mean RMSE

CLI:
  python tox_train.py <csv> <classification|regression> [--smiles-col smiles] [--y-col y]

The CSV must have a SMILES column and a target column (defaults: 'smiles', 'y').
"""
from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit import DataStructs

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error

from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 5
ECFP_BITS = 2048
ECFP_RADIUS = 2  # radius 2 == ECFP4 == Morgan diameter 4, 2048 bits


# --------------------------------------------------------------------------- #
# Featurization
# --------------------------------------------------------------------------- #
def parse_smiles(smiles_list):
    """Return (mols, valid_mask). Invalid/unparseable SMILES -> None / False."""
    mols, valid = [], []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        mols.append(m)
        valid.append(m is not None)
    return mols, np.asarray(valid, dtype=bool)


_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=ECFP_RADIUS, fpSize=ECFP_BITS
)


def featurize_ecfp(mols, n_bits=ECFP_BITS):
    """Morgan/ECFP bit vectors -> (n, n_bits) int8 array. Feed only valid mols."""
    X = np.zeros((len(mols), n_bits), dtype=np.int8)
    for i, m in enumerate(mols):
        fp = _MORGAN_GEN.GetFingerprint(m)
        row = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, row)
        X[i] = row
    return X


def _build_fragment_descriptor_list():
    """(name, fn) list: all fr_* plus the explicit descriptor families in the recipe."""
    by_name = dict(Descriptors._descList)  # name -> fn
    selected, seen = [], set()

    def add(name):
        if name in by_name and name not in seen:
            selected.append((name, by_name[name]))
            seen.add(name)

    # fr_* fragment counts
    for name, _ in Descriptors._descList:
        if name.startswith("fr_"):
            add(name)
    # explicit physchem / shape descriptors
    for name in ("TPSA", "MolLogP", "LabuteASA", "Kappa1", "Kappa2", "Kappa3"):
        add(name)
    # VSA descriptor families
    for name, _ in Descriptors._descList:
        if name.startswith(("SlogP_VSA", "SMR_VSA", "EState_VSA")):
            add(name)
    return selected


_FRAG_DESCS = _build_fragment_descriptor_list()
FRAG_FEATURE_NAMES = [n for n, _ in _FRAG_DESCS]


def featurize_fragments(mols):
    """Fragment + descriptor matrix -> (n, n_feat) float32 array. Feed only valid mols."""
    n_feat = len(_FRAG_DESCS)
    X = np.zeros((len(mols), n_feat), dtype=np.float32)
    for i, m in enumerate(mols):
        for j, (_, fn) in enumerate(_FRAG_DESCS):
            try:
                X[i, j] = fn(m)
            except Exception:
                X[i, j] = np.nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# --------------------------------------------------------------------------- #
# Model factories
# --------------------------------------------------------------------------- #
def make_catboost(task):
    common = dict(iterations=1000, depth=6, learning_rate=0.05,
                  random_seed=SEED, verbose=0, thread_count=-1)
    if task == "classification":
        return CatBoostClassifier(loss_function="Logloss", **common)
    return CatBoostRegressor(loss_function="RMSE", **common)


def make_xgboost(task):
    common = dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                  random_state=SEED, n_jobs=-1, tree_method="hist")
    if task == "classification":
        return XGBClassifier(eval_metric="logloss", **common)
    return XGBRegressor(eval_metric="rmse", **common)


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def cv_score(X, y, model_factory, task, n_splits=N_SPLITS, seed=SEED):
    """Run k-fold CV; return (mean, std, per_fold_scores). Metric: ROC-AUC / RMSE."""
    X = np.asarray(X)
    y = np.asarray(y)
    if task == "classification":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=seed).split(X, y)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True,
                         random_state=seed).split(X)

    scores = []
    for tr, te in splitter:
        model = model_factory()
        model.fit(X[tr], y[tr])
        if task == "classification":
            p = model.predict_proba(X[te])[:, 1]
            scores.append(roc_auc_score(y[te], p))
        else:
            scores.append(_rmse(y[te], model.predict(X[te])))
    return float(np.mean(scores)), float(np.std(scores)), scores


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def load_dataset(csv_path, smiles_col="smiles", y_col="y"):
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns or y_col not in df.columns:
        raise SystemExit(
            f"CSV must contain columns '{smiles_col}' and '{y_col}'; "
            f"found {list(df.columns)}"
        )
    smiles = df[smiles_col].tolist()
    mols, valid = parse_smiles(smiles)
    n_bad = int((~valid).sum())
    mols_ok = [m for m, v in zip(mols, valid) if v]
    y = df[y_col].to_numpy()[valid]
    return mols_ok, y, n_bad, len(df)


def run(csv_path, task, smiles_col="smiles", y_col="y", verbose=True):
    """Reproduce the recipe on one endpoint. Returns a results dict."""
    assert task in ("classification", "regression")
    mols, y, n_bad, n_total = load_dataset(csv_path, smiles_col, y_col)

    metric = "ROC-AUC" if task == "classification" else "RMSE"
    if verbose:
        print(f"[data] {csv_path}: {n_total} rows, {len(mols)} valid mols, "
              f"{n_bad} dropped; task={task}")
        if task == "classification":
            vals, cnts = np.unique(y, return_counts=True)
            print(f"[data] class balance: {dict(zip(vals.tolist(), cnts.tolist()))}")

    # (a) CatBoost on ECFP fingerprints
    X_fp = featurize_ecfp(mols)
    cat_mean, cat_std, cat_folds = cv_score(X_fp, y, lambda: make_catboost(task), task)
    if verbose:
        print(f"[CatBoost-FP  ] {X_fp.shape[1]} bits  -> mean {metric} = "
              f"{cat_mean:.4f} +/- {cat_std:.4f}  folds={[round(s,4) for s in cat_folds]}")

    # (b) XGBoost on fragment descriptors
    X_frag = featurize_fragments(mols)
    xgb_mean, xgb_std, xgb_folds = cv_score(X_frag, y, lambda: make_xgboost(task), task)
    if verbose:
        print(f"[XGBoost-frag ] {X_frag.shape[1]} feats -> mean {metric} = "
              f"{xgb_mean:.4f} +/- {xgb_std:.4f}  folds={[round(s,4) for s in xgb_folds]}")

    return {
        "task": task,
        "metric": metric,
        "n_total": n_total,
        "n_valid": len(mols),
        "n_dropped": n_bad,
        "catboost_fp": {"mean": cat_mean, "std": cat_std, "folds": cat_folds,
                        "n_features": int(X_fp.shape[1])},
        "xgboost_frag": {"mean": xgb_mean, "std": xgb_std, "folds": xgb_folds,
                         "n_features": int(X_frag.shape[1])},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="dataset CSV with SMILES + target columns")
    ap.add_argument("task", choices=["classification", "regression"])
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--y-col", default="y")
    args = ap.parse_args(argv)

    res = run(args.csv, args.task, args.smiles_col, args.y_col)

    print("\n=== SUMMARY ===")
    print(f"task            : {res['task']}  ({res['metric']})")
    print(f"CatBoost-FP     : {res['catboost_fp']['mean']:.4f}")
    print(f"XGBoost-frag    : {res['xgboost_frag']['mean']:.4f}")
    return res


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Permutation importance for saved DMPNN+HGB RMT-RTE model.

Importance is measured as the drop in test ROC AUC after shuffling one scaled
global feature column. The molecular graph is left unchanged, so the result is
importance of global features only: RMT-RTE, docking engineered, and RDKit2D.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit.Chem import Descriptors
from sklearn.metrics import roc_auc_score


BASE = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_dmpnn_rmt_rte import (  # noqa: E402
    DATA_CSV,
    DOCK_COLS,
    DMPNNClassifier,
    batch_tensors,
    docking_engineered,
    load_rdkit2d,
)


# The saved scaler was pickled while train_dmpnn_rmt_rte.py was executed as
# __main__, so joblib expects a __main__.FeatureScaler class at load time.
class FeatureScaler:
    def fit(self, x):
        self.median_ = np.nanmedian(x, axis=0)
        xf = np.where(np.isnan(x), self.median_, x)
        self.mean_ = xf.mean(axis=0)
        self.std_ = xf.std(axis=0)
        self.keep_ = self.std_ > 1e-6
        self.std_[~self.keep_] = 1.0
        return self

    def transform(self, x):
        xf = np.where(np.isnan(x), self.median_, x)
        xz = np.clip((xf - self.mean_) / self.std_, -5.0, 5.0)
        return xz[:, self.keep_].astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Permutation importance for DMPNN+HGB RMT-RTE model")
    p.add_argument(
        "--model-dir",
        default="pipeline_runs/dmpnn_rmt_rte_split00_test/rmt_rte_rec/split_00",
        help="Directory with dmpnn_model.pt, hgb_model.pkl, feature_scaler.pkl, metrics.json",
    )
    p.add_argument(
        "--rmt-dir",
        default="pipeline_runs/rmt_rte_filter_split00_test",
        help="Output directory from rmt_filter.py",
    )
    p.add_argument("--mode", default="rmt_rte_rec")
    p.add_argument("--split-id", default="split_00")
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-n", type=int, default=30)
    return p.parse_args()


def predict_dmpnn(model, smiles, xg, device, batch_size=256):
    model.eval()
    out = np.zeros(len(smiles), dtype=np.float32)
    idx = np.arange(len(smiles))
    with torch.no_grad():
        for k in range(0, len(idx), batch_size):
            sub = idx[k:k + batch_size]
            batch = batch_tensors([smiles[i] for i in sub], device)
            g = torch.from_numpy(xg[sub]).to(device)
            out[sub] = torch.sigmoid(model(batch, g)).cpu().numpy()
    return out


def global_feature_names(rmt_cols: list[str], scaler: FeatureScaler) -> list[str]:
    docking_names = (
        DOCK_COLS
        + ["dock_min", "dock_max", "dock_mean", "dock_std", "dock_range", "dock_best2"]
        + [f"dock_rank_{col}" for col in DOCK_COLS]
    )
    rdkit_names = [name for name, _ in Descriptors.descList]
    names = list(rmt_cols) + docking_names + rdkit_names
    if len(names) != len(scaler.keep_):
        raise ValueError(f"Feature name count {len(names)} != scaler.keep_ length {len(scaler.keep_)}")
    return [name for name, keep in zip(names, scaler.keep_) if keep]


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = BASE / model_dir
    rmt_dir = Path(args.rmt_dir)
    if not rmt_dir.is_absolute():
        rmt_dir = BASE / rmt_dir

    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    weight = float(metrics["blend_weight_dmpnn"])
    n_global = int(metrics["n_global_after_filter"])

    scaler: FeatureScaler = joblib.load(model_dir / "feature_scaler.pkl")
    hgb = joblib.load(model_dir / "hgb_model.pkl")
    rmt_cols = np.load(model_dir / "selected_rmt_feature_names.npy", allow_pickle=True).tolist()

    data = pd.read_csv(DATA_CSV)
    data["ligand_id"] = data["ligand_id"].astype(int)
    id_to_full = {int(lid): i for i, lid in enumerate(data["ligand_id"].values)}

    test_rmt = pd.read_csv(rmt_dir / args.split_id / f"{args.mode}_test.csv")
    test_rmt["ligand_id"] = test_rmt["ligand_id"].astype(int)
    test_ids = test_rmt["ligand_id"].to_numpy(dtype=int)
    full_idx = np.array([id_to_full[int(lid)] for lid in test_ids], dtype=int)

    smiles = data.loc[full_idx, "SMILES"].to_numpy()
    y = data.loc[full_idx, "activity"].to_numpy(dtype=int)
    rmt_x = test_rmt[rmt_cols].to_numpy(dtype=np.float64)
    dock_eng = docking_engineered(data.loc[full_idx, DOCK_COLS].to_numpy(dtype=np.float64))
    rdkit2d = load_rdkit2d(data["SMILES"].to_numpy())[full_idx]
    x_global = np.concatenate([rmt_x, dock_eng, rdkit2d], axis=1).astype(np.float64)
    x_scaled = scaler.transform(x_global)
    if x_scaled.shape[1] != n_global:
        raise ValueError(f"Scaled feature count {x_scaled.shape[1]} != metrics n_global {n_global}")
    kept_names = global_feature_names(rmt_cols, scaler)

    run_config_path = model_dir.parents[1] / "run_config.json"
    config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DMPNNClassifier(
        n_global_features=n_global,
        hidden_size=int(config.get("hidden_size", 300)),
        depth=int(config.get("depth", 3)),
        dropout=float(config.get("dropout", 0.15)),
        ffn_hidden=int(config.get("ffn_hidden", 300)),
        ffn_num_layers=int(config.get("ffn_layers", 2)),
    ).to(device)
    state = torch.load(model_dir / "dmpnn_model.pt", map_location=device)
    model.load_state_dict(state)

    dmpnn_base = predict_dmpnn(model, smiles, x_scaled, device)
    hgb_base = hgb.predict_proba(x_scaled)[:, 1]
    blend_base = weight * dmpnn_base + (1.0 - weight) * hgb_base
    base_auc = float(roc_auc_score(y, blend_base))

    rng = np.random.RandomState(args.seed)
    rows = []
    for j, name in enumerate(kept_names):
        drops = []
        aucs = []
        for rep in range(args.n_repeats):
            xp = x_scaled.copy()
            xp[:, j] = xp[rng.permutation(xp.shape[0]), j]
            dmpnn_p = predict_dmpnn(model, smiles, xp, device)
            hgb_p = hgb.predict_proba(xp)[:, 1]
            blend_p = weight * dmpnn_p + (1.0 - weight) * hgb_p
            auc = float(roc_auc_score(y, blend_p))
            aucs.append(auc)
            drops.append(base_auc - auc)
        rows.append(
            {
                "feature": name,
                "importance_mean_drop_roc_auc": float(np.mean(drops)),
                "importance_std_drop_roc_auc": float(np.std(drops, ddof=1)) if args.n_repeats > 1 else 0.0,
                "permuted_auc_mean": float(np.mean(aucs)),
                "permuted_auc_std": float(np.std(aucs, ddof=1)) if args.n_repeats > 1 else 0.0,
                "baseline_auc": base_auc,
                "n_repeats": int(args.n_repeats),
            }
        )
        if (j + 1) % 50 == 0 or j + 1 == len(kept_names):
            print(f"  permuted {j + 1}/{len(kept_names)}", flush=True)

    imp = pd.DataFrame(rows).sort_values("importance_mean_drop_roc_auc", ascending=False).reset_index(drop=True)
    csv_path = model_dir / "permutation_importance_blend.csv"
    imp.to_csv(csv_path, index=False)

    top = imp.head(args.top_n).iloc[::-1]
    fig_h = max(7.0, 0.26 * len(top) + 2.0)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.barh(top["feature"], top["importance_mean_drop_roc_auc"], color="#2f5f8f")
    ax.set_xlabel("Drop in test ROC AUC after permutation")
    ax.set_title(f"Permutation importance: {args.mode} {args.split_id} blend\nBaseline ROC AUC = {base_auc:.4f}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    png_path = model_dir / "permutation_importance_blend_top30.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    print(f"Baseline blend ROC AUC: {base_auc:.6f}")
    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()

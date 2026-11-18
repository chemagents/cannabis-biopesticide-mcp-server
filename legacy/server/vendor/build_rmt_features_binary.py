#!/usr/bin/env python3
"""Compute MI-based per-protein RMT features (mi_only and hybrid_mi) for canpest.

Usage:
    python scripts/build_rmt_features_binary.py
    python scripts/build_rmt_features_binary.py --repeats 5 --max-m 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from calc_hybrid_binary import (
    fit_protein_hybrid_transform_binary,
    load_residue_matrix,
    normalize_ligand_id,
)

BASE = Path(__file__).resolve().parent.parent
RESIDUE_DIR = BASE / "data" / "residue_matrices"
FEATURES_ROOT = BASE / "features"

PROTEIN_MAP = {
    "2imi": "gste2",
    "d8v7j0": "ache",
    "3rif": "glc1",
    "8sfy": "ugt202a2",
    "8udb": "gstm12",
    "8v3d": "agamor28",
}

METHODS = ["mi", "hybrid_mi"]
ALPHA = 1.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build MI-based RMT features for canpest")
    p.add_argument("--repeats", type=int, default=3, help="Inner CV repeats")
    p.add_argument("--max-m", type=int, default=None, help="Max residues to test")
    p.add_argument("--inner-train-frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    return p


def main() -> None:
    args = build_parser().parse_args()

    activity = pd.read_csv("data/data.csv", dtype={"ligand_id": str})
    activity = activity.dropna(subset=["activity"])
    activity["activity"] = activity["activity"].astype(int)

    train_df, test_df = train_test_split(
        activity, test_size=args.test_size, stratify=activity["activity"],
        random_state=args.seed,
    )
    train_ids = set(train_df["ligand_id"])
    y_train = train_df.set_index("ligand_id")["activity"]
    y_test = test_df.set_index("ligand_id")["activity"]

    train_scores: dict[str, dict[str, np.ndarray]] = {"mi": {}, "hybrid_mi": {}}
    test_scores: dict[str, dict[str, np.ndarray]] = {"mi": {}, "hybrid_mi": {}}
    specs: list[dict] = []
    m_summaries = []

    for dock_col, gene in PROTEIN_MAP.items():
        print(f"\n--- {gene} ({dock_col}) ---")

        try:
            mat = load_residue_matrix(RESIDUE_DIR, dock_col)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        mat = mat[mat["ligand_id"].str.contains("_tau_1", regex=False)].copy()
        mat["ligand_id_norm"] = normalize_ligand_id(mat["ligand_id"])

        mat_train = mat[mat["ligand_id_norm"].isin(train_ids)].copy()
        mat_test = mat[mat["ligand_id_norm"].isin(test_df["ligand_id"])].copy()

        y_train_arr = y_train.reindex(mat_train["ligand_id_norm"]).to_numpy(dtype=float)
        y_test_arr = y_test.reindex(mat_test["ligand_id_norm"]).to_numpy(dtype=float)

        if len(mat_train) < 10:
            print(f"  SKIP: too few train rows ({len(mat_train)})")
            continue

        for method in METHODS:
            t0 = time.time()

            spec, scan_df, _ = fit_protein_hybrid_transform_binary(
                protein=gene,
                matrix_df=mat_train,
                y_train=y_train_arr,
                train_ligands=train_ids,
                alpha=ALPHA,
                max_m_requested=args.max_m,
                repeats=args.repeats,
                inner_train_frac=args.inner_train_frac,
                seed=args.seed,
                method=method,
            )

            m_opt = spec["m_opt"]
            residue_cols = [c for c in mat.columns if c != "ligand_id" and c != "ligand_id_norm"]

            all_mat = pd.concat([mat_train, mat_test], ignore_index=True)
            x_all = all_mat[residue_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            top_idx = np.array(spec["top_residue_indices"])
            all_scores = x_all[:, top_idx].sum(axis=1)
            all_ids = all_mat["ligand_id_norm"].to_numpy()

            n_train = len(mat_train)
            train_scores[method][gene] = all_scores[:n_train]
            test_scores[method][gene] = all_scores[n_train:]

            out_dir = FEATURES_ROOT / "protein-related" / gene / "activity" / f"split_{args.test_size}"
            (out_dir / method).mkdir(parents=True, exist_ok=True)

            score_df = pd.DataFrame({
                "ligand_id": all_ids,
                f"{gene}_{method}_score": all_scores,
            })
            score_df.to_csv(out_dir / method / "score.csv", index=False)

            with open(out_dir / method / "transform_spec.json", "w") as f:
                spec_json = {k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in spec.items()}
                json.dump(spec_json, f, indent=2, default=str)

            scan_df.to_csv(out_dir / method / "m_scan.csv", index=False)

            m_summaries.append({
                "protein": gene,
                "method": method,
                "m_opt": m_opt,
                "n_residues": spec["n_residue_cols"],
                "n_signal": spec["n_signal_full_train"],
                "lambda_plus": spec["lambda_plus_full_train"],
                "cv_roc_auc": round(roc, 4) if (roc := spec["cv_mean_roc_auc_at_m_opt"]) is not None else None,
            })

            print(f"    m_opt={m_opt}, n_signal={spec['n_signal_full_train']}, "
                  f"cv_roc_auc={spec['cv_mean_roc_auc_at_m_opt']:.4f}, "
                  f"elapsed={time.time()-t0:.1f}s")

    protein_list = [v for k, v in PROTEIN_MAP.items() if v in train_scores["mi"]]
    if not protein_list:
        print("\nNo proteins processed.")
        return

    rmt_root = FEATURES_ROOT / "protein-related"
    rmt_root.mkdir(parents=True, exist_ok=True)

    for method in METHODS:
        all_ids = np.concatenate([
            train_df["ligand_id"].to_numpy(),
            test_df["ligand_id"].to_numpy(),
        ])
        score_dict = {"ligand_id": all_ids}
        for gene in protein_list:
            if gene in train_scores[method] and gene in test_scores[method]:
                score_dict[gene] = np.concatenate([
                    train_scores[method][gene],
                    test_scores[method][gene],
                ])
        panel_df = pd.DataFrame(score_dict)
        panel_path = rmt_root / f"{method}_6.csv"
        panel_df.to_csv(panel_path, index=False)
        print(f"\nPanel-wide {method}_6: {panel_path} ({panel_df.shape})")

    plain_path = FEATURES_ROOT / "structural" / "plain.csv"
    if plain_path.exists():
        plain = pd.read_csv(plain_path, dtype={"ligand_id": str})
        for method in METHODS:
            panel_path = rmt_root / f"{method}_6.csv"
            if not panel_path.exists():
                continue
            rmt_df = pd.read_csv(panel_path, dtype={"ligand_id": str})
            merged = plain.merge(rmt_df, on="ligand_id")
            (FEATURES_ROOT / "composite").mkdir(parents=True, exist_ok=True)
            comp_path = FEATURES_ROOT / "composite" / f"plain_{method}_6.csv"
            merged.to_csv(comp_path, index=False)
            print(f"Composite plain_{method}_6: {comp_path} ({merged.shape})")

    summary_df = pd.DataFrame(m_summaries).sort_values(["method", "protein"]).reset_index(drop=True)
    summary_path = rmt_root / "mi_m_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

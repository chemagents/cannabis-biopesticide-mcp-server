#!/usr/bin/env python3
"""Compute RMT-based per-protein features (s_only and hybrid) for canpest.

For each of the 6 proteins:
  1. Load residue matrix and merge with activity labels
  2. Stratified 80/20 split (train/test)
  3. Fit per-protein hybrid/s_only transform on train split only
  4. Score all molecules (train + test) using the fitted transform
  5. Save scores + transform spec + m-scan trace

Then build panel-wide matrices:
  - features/protein-related/s_only_6.csv   — 6 per-protein s_only scores
  - features/protein-related/hybrid_6.csv   — 6 per-protein hybrid scores

And composite matrices:
  - features/composite/plain_s_only_6.csv   — plain + s_only_6
  - features/composite/plain_hybrid_6.csv   — plain + hybrid_6

Usage:
    python scripts/build_rmt_features.py
    python scripts/build_rmt_features.py --repeats 5 --max-m 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from calc_hybrid import (
    build_hybrid_rank,
    fit_protein_hybrid_transform,
    load_residue_matrix,
    normalize_ligand_id,
    select_m_inner_cv,
)

BASE = Path(__file__).resolve().parent.parent
RESIDUE_DIR = BASE / "data" / "residue_matrices"
FEATURES_ROOT = BASE / "features"
DOCK_COLS = ["2imi", "d8v7j0", "3rif", "8sfy", "8udb", "8v3d"]

PROTEIN_MAP = {
    "2imi": "gste2",
    "d8v7j0": "ache",
    "3rif": "glc1",
    "8sfy": "ugt202a2",
    "8udb": "gstm12",
    "8v3d": "agamor28",
}

METHODS = ["s_only", "hybrid"]
ALPHA = 1.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build RMT features for canpest")
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
    print(f"Labeled: {len(activity)} ({activity['activity'].value_counts().get(1, 0)} active, {activity['activity'].value_counts().get(0, 0)} inactive)")

    train_df, test_df = train_test_split(
        activity, test_size=args.test_size, stratify=activity["activity"],
        random_state=args.seed,
    )
    train_ids = set(train_df["ligand_id"])
    y_train = train_df.set_index("ligand_id")["activity"]
    y_test = test_df.set_index("ligand_id")["activity"]
    print(f"Train: {len(train_ids)}, Test: {len(test_df)}")

    train_scores: dict[str, dict[str, np.ndarray]] = {"s_only": {}, "hybrid": {}}
    test_scores: dict[str, dict[str, np.ndarray]] = {"s_only": {}, "hybrid": {}}
    specs: list[dict] = []
    m_summaries = []

    for dock_col, gene in PROTEIN_MAP.items():
        print(f"\n--- {gene} ({dock_col}) ---")

        try:
            mat = load_residue_matrix(RESIDUE_DIR, dock_col)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        # Keep only tau=1 rows
        mat = mat[mat["ligand_id"].str.contains("_tau_1", regex=False)].copy()
        mat["ligand_id_norm"] = normalize_ligand_id(mat["ligand_id"])

        # Split matrix rows into train/test
        mat_train = mat[mat["ligand_id_norm"].isin(train_ids)].copy()
        mat_test = mat[mat["ligand_id_norm"].isin(test_df["ligand_id"])].copy()

        y_train_arr = y_train.reindex(mat_train["ligand_id_norm"]).to_numpy(dtype=float)
        y_test_arr = y_test.reindex(mat_test["ligand_id_norm"]).to_numpy(dtype=float)

        print(f"  Train rows: {len(mat_train)}, Test rows: {len(mat_test)}")

        if len(mat_train) < 10:
            print(f"  SKIP: too few train rows ({len(mat_train)})")
            continue

        for method in METHODS:
            print(f"  Method: {method}")
            t0 = time.time()

            spec, scan_df, _ = fit_protein_hybrid_transform(
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
            top_idx_list = []
            residue_cols = [c for c in mat.columns if c != "ligand_id" and c != "ligand_id_norm"]

            if method == "hybrid":
                final_rank = build_hybrid_rank(
                    mat_train[residue_cols].to_numpy(dtype=float), y_train_arr, ALPHA
                )
                top_idx = final_rank["ranking"][:m_opt]
            else:
                from calc_hybrid import rmt_prior_from_train
                s_i, _, _, _ = rmt_prior_from_train(
                    mat_train[residue_cols].to_numpy(dtype=float)
                )
                top_idx = np.argsort(s_i)[::-1][:m_opt]

            # Score all molecules (train + test) with the same transform
            all_mat = pd.concat([mat_train, mat_test], ignore_index=True)
            x_all = all_mat[residue_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            all_scores = x_all[:, top_idx].sum(axis=1)
            all_ids = all_mat["ligand_id_norm"].to_numpy()

            # Split back
            n_train = len(mat_train)
            train_scores[method][gene] = all_scores[:n_train]
            test_scores[method][gene] = all_scores[n_train:]

            # Save per-protein artifacts
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
                "lambda_plus": round(spec["lambda_plus_full_train"], 4),
                "cv_abs_spearman": round(cv, 4) if (cv := spec["cv_mean_abs_spearman_at_m_opt"]) is not None else None,
            })
            specs.append(spec)

            print(f"    m_opt={m_opt}, n_signal={spec['n_signal_full_train']}, "
                  f"cv_abs_rho={spec['cv_mean_abs_spearman_at_m_opt']}, "
                  f"elapsed={time.time()-t0:.1f}s")

    # --- Build panel-wide matrices ---
    protein_list = [v for k, v in PROTEIN_MAP.items() if v in train_scores["s_only"]]

    if not protein_list:
        print("\nNo proteins processed. Nothing to save.")
        return

    rmt_root = FEATURES_ROOT / "protein-related"
    rmt_root.mkdir(parents=True, exist_ok=True)

    for method in METHODS:
        # Build full score matrix (train + test)
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

    # --- Build composites with plain ---
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

    # --- Summary ---
    summary_df = pd.DataFrame(m_summaries).sort_values(["method", "protein"]).reset_index(drop=True)
    summary_path = rmt_root / "rmt_m_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

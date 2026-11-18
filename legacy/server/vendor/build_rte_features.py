"""Build raw residue-term energy (RTE/VinaTerms) feature CSV.

This is a thin reproducibility wrapper around historical He-Lee artifacts:

  data/he_lee/X_hl_data.npy
  data/he_lee/lig_ids_hl.npy
  data/he_lee/hl_feat_names.json

It writes a project-facing feature table:

  features/protein-related/rte.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent.parent
HL_DIR = BASE / "data" / "he_lee"
OUT_DIR = BASE / "features" / "protein-related"
OUT_CSV = OUT_DIR / "rte.csv"
DATA_CSV = BASE / "data" / "data.csv"


def load_inputs() -> tuple[np.ndarray, np.ndarray, list[str]]:
    x = np.load(HL_DIR / "X_hl_data.npy")
    ligand_ids = np.load(HL_DIR / "lig_ids_hl.npy").astype(int)
    names = json.loads((HL_DIR / "hl_feat_names.json").read_text(encoding="utf-8"))
    if x.shape[1] != len(names):
        raise ValueError(f"Feature name count mismatch: X={x.shape}, names={len(names)}")
    if x.shape[0] != len(ligand_ids):
        raise ValueError(f"Ligand id count mismatch: X={x.shape}, ids={len(ligand_ids)}")
    return x, ligand_ids, names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RTE/VinaTerms feature CSV")
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Keep all 5920 molecules. Default keeps only molecules with activity 0/1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x, ligand_ids, names = load_inputs()
    df = pd.DataFrame(x, columns=names)
    df.insert(0, "ligand_id", ligand_ids)

    data = pd.read_csv(DATA_CSV)
    data["ligand_id"] = data["ligand_id"].astype(int)
    labeled_ids = set(data.loc[data["activity"].notna(), "ligand_id"].astype(int))
    if not args.include_unlabeled:
        df = df[df["ligand_id"].astype(int).isin(labeled_ids)].copy()
        df = df.reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} shape={df.shape}")


if __name__ == "__main__":
    main()
